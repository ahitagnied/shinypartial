import cv2 as cv
import os, subprocess, shutil
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

def sample_frames(video, fps):
    frames = []
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    
    fps_og = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps_og / fps))
    
    for i in range(0, total_frames, frame_interval):
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret: 
            frames.append(frame)
            
    return frames


def rm_blurry_frames(frames, n_frames=None):

    def _get_blur_score(frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return cv.Laplacian(gray, cv.CV_64F).var()
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        clean_frames = list(executor.map(lambda frame: (frame, _get_blur_score(frame)), frames))
    
    clean_frames.sort(key=lambda x: x[1], reverse=True)
    clean_frames = clean_frames[:n_frames]

    return [frame for frame, _ in clean_frames]


def save_frames(frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv.imwrite(os.path.join(output_dir, f"frame_{i:06d}.png"), frame)

def _run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_colmap(frames_dir, use_gpu=True):
    work = Path(frames_dir)
    images = work/"images"
    db = work/"colmap.db"
    sparse = work/"sparse"
    und = work/"undistorted"
    
    work.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(exist_ok=True)
    und.mkdir(exist_ok=True)
    
    if db.exists():
        db.unlink()
    
    _run([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
    ])

    _run([
        "colmap", "sequential_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        "--SequentialMatching.overlap", "10",
        "--SequentialMatching.quadratic_overlap", "1",
    ])

    _run([
        "colmap", "mapper",
        "--database_path", str(db),
        "--image_path", str(images),
        "--output_path", str(sparse),
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "1",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.tri_ignore_two_view_tracks", "1",
    ])

    models = sorted([p for p in sparse.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not models:
        raise RuntimeError("COLMAP produced no sparse models.")
    model = models[0]

    _run([
        "colmap", "image_undistorter",
        "--image_path", str(images),
        "--input_path", str(model),
        "--output_path", str(und),
        "--output_type", "COLMAP",
    ])

    return und


if __name__ == "__main__":    
    video_file = sys.argv[1]
    
    if not os.path.exists(video_file):
        print(f"error: video file '{video_file}' not found")
        sys.exit(1)
    
    print(f"processing: {video_file}")
    
    frames = sample_frames(video_file, fps=20)
    print(f"extracted {len(frames)} frames")
    
    images_dir = "frames/images"
    save_frames(frames, images_dir)
    print(f"saved frames to {images_dir}/")
    
    print("running COLMAP...")
    run_colmap("frames", use_gpu=False)
    print("done!")
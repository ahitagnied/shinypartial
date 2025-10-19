import cv2 as cv
import os, subprocess
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

def filter_sharp_frames(frames, keep_n=350):
    def blur_score(frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return cv.Laplacian(gray, cv.CV_64F).var()
    with ThreadPoolExecutor() as executor:
        scored = list(executor.map(lambda f: (f, blur_score(f)), frames))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scored[:keep_n]]

def preprocess_frame(frame, target_size):
    h, w = frame.shape[:2]
    crop_size = min(w, h)
    offset_x = (w - crop_size) // 2
    offset_y = (h - crop_size) // 2
    cropped = frame[offset_y:offset_y + crop_size, offset_x:offset_x + crop_size]
    resized = cv.resize(cropped, (target_size, target_size), interpolation=cv.INTER_AREA)
    return resized

def save_frames(frames, output_dir, target_size=1200):
    os.makedirs(output_dir, exist_ok=True)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    print(f"preprocessing: {w}x{h} → center crop → {target_size}x{target_size}")
    print(f"saving {len(frames)} frames...")
    for i, frame in enumerate(frames):
        processed = preprocess_frame(frame, target_size)
        cv.imwrite(os.path.join(output_dir, f"frame_{i:06d}.png"), processed)

def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def get_model_size(model_path):
    try:
        import struct
        with open(model_path / "images.bin", "rb") as f:
            return struct.unpack('Q', f.read(8))[0]
    except:
        return 0

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
    
    print("extracting features...")
    _run([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
        "--SiftExtraction.max_num_features", "4096",
    ])
    
    print("matching features...")
    _run([
        "colmap", "sequential_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        "--SequentialMatching.overlap", "15",
        "--SequentialMatching.quadratic_overlap", "1",
    ])
    
    print("running mapper...")
    _run([
        "colmap", "mapper",
        "--database_path", str(db),
        "--image_path", str(images),
        "--output_path", str(sparse),
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "1",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.num_threads", "16",
    ])
    
    models = [p for p in sparse.iterdir() if p.is_dir()]
    if not models:
        raise RuntimeError("COLMAP produced no sparse models.")
    model = max(models, key=get_model_size)
    num_images = get_model_size(model)
    print(f"selected model {model.name} with {num_images} images")
    
    if model.name != "0":
        print(f"renaming {model.name} to 0 and removing other models...")
        import shutil
        
        for other_model in models:
            if other_model != model:
                shutil.rmtree(other_model)
                print(f"removed {other_model.name}")
        
        model_0 = sparse / "0"
        if model_0.exists():
            shutil.rmtree(model_0)
        model.rename(model_0)
        model = model_0
    
    print("undistorting images...")
    _run([
        "colmap", "image_undistorter",
        "--image_path", str(images),
        "--input_path", str(model),
        "--output_path", str(und),
        "--output_type", "COLMAP",
    ])
    
    sparse_out = und / "sparse"
    sparse_0 = sparse_out / "0"
    if sparse_out.exists() and not sparse_0.exists():
        sparse_0.mkdir(parents=True, exist_ok=True)
        for f in sparse_out.iterdir():
            if f.is_file():
                f.rename(sparse_0 / f.name)
    
    return und

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(description="convert video to colmap dataset")
    parser.add_argument("video_path", help="path to input video file")
    parser.add_argument("--output_dir", "-o", default="frames", help="output directory")
    parser.add_argument("--fps", type=float, default=12.0, help="frames per second")
    parser.add_argument("--max_frames", type=int, default=350, help="max frames to keep (default: 350, keeps sharpest)")
    parser.add_argument("--target_size", type=int, default=1200, help="target image size after crop/resize (default: 1200)")
    parser.add_argument("--use_gpu", action="store_true", help="use GPU for COLMAP")
    args = parser.parse_args()
    if not os.path.exists(args.video_path):
        print(f"error: video file '{args.video_path}' not found")
        sys.exit(1)
    print(f"processing: {args.video_path}")
    frames = sample_frames(args.video_path, fps=args.fps)
    print(f"sampled {len(frames)} frames")
    
    if len(frames) > args.max_frames:
        print(f"filtering to {args.max_frames} sharpest frames...")
        frames = filter_sharp_frames(frames, keep_n=args.max_frames)
    
    images_dir = os.path.join(args.output_dir, "images")
    save_frames(frames, images_dir, target_size=args.target_size)
    run_colmap(args.output_dir, use_gpu=args.use_gpu)
    print(f"done! sparse model: {args.output_dir}/undistorted/sparse/0/")

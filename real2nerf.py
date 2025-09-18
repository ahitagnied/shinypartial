"""
convert video to nerf dataset using colmap sfm
"""

import os
import sys
import json
import cv2
import numpy as np
import subprocess
import argparse
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_frames(video_path: str, output_dir: str, fps: float = 2.0, max_frames: Optional[int] = None, image_scale: float = 1.0) -> List[str]:
    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
        
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(original_fps / fps))
    
    logger.info(f"video: {original_fps:.1f} fps, {total_frames} total frames")
    logger.info(f"extracting at {fps} fps, interval: {frame_interval}")
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened() and saved_count < (max_frames or float('inf')):
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            if image_scale != 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * image_scale), int(w * image_scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            frame_filename = frames_dir / f"frame_{saved_count:06d}.png"
            cv2.imwrite(str(frame_filename), frame)
            frame_paths.append(str(frame_filename))
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    logger.info(f"extracted {len(frame_paths)} frames")
    return frame_paths

def check_colmap():
    try:
        subprocess.run(['colmap', '--help'], capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        raise RuntimeError("colmap not found. install from https://colmap.github.io/")

def run_colmap(frames_dir: str, output_dir: str, min_matches: int = 15) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    database_path = output_path / "database.db"
    sparse_dir = output_path / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting COLMAP reconstruction for {frames_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Try automatic reconstruction first (often better for video sequences)
    try:
        logger.info("Trying automatic reconstruction...")
        auto_cmd = [
            "colmap", "automatic_reconstructor",
            "--workspace_path", str(output_path),
            "--image_path", frames_dir,
            "--data_type", "video",
            "--quality", "medium",
            "--camera_model", "SIMPLE_RADIAL",
            "--single_camera", "1"
        ]
        logger.info(f"Running: {' '.join(auto_cmd)}")
        result = subprocess.run(auto_cmd, check=True, capture_output=True, text=True)
        logger.info("Automatic reconstruction completed successfully")
        
        # Check if reconstruction was successful
        if (sparse_dir / "0").exists():
            logger.info("Automatic reconstruction successful, using results")
            return {
                "database": str(database_path),
                "sparse": str(sparse_dir / "0"),
                "frames_dir": frames_dir
            }
        else:
            logger.warning("Automatic reconstruction completed but no sparse model found")
    except subprocess.CalledProcessError as e:
        logger.info(f"Automatic reconstruction failed: {e}")
        logger.info("Falling back to manual pipeline")
    
    # Fallback to manual pipeline with relaxed parameters
    logger.info("Starting manual COLMAP pipeline...")
    
    # feature extraction
    logger.info("Step 1: Feature extraction...")
    feature_cmd = [
        "colmap", "feature_extractor", 
        "--database_path", str(database_path), 
        "--image_path", frames_dir, 
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", "8192"
    ]
    logger.info(f"Running: {' '.join(feature_cmd)}")
    result = subprocess.run(feature_cmd, check=True, capture_output=True, text=True)
    logger.info("Feature extraction completed")
    
    # feature matching - try multiple approaches
    logger.info("Step 2: Feature matching...")
    
    # Try exhaustive matcher first (more thorough)
    try:
        logger.info("Trying exhaustive matcher...")
        matcher_cmd = [
            "colmap", "exhaustive_matcher", 
            "--database_path", str(database_path)
        ]
        logger.info(f"Running: {' '.join(matcher_cmd)}")
        result = subprocess.run(matcher_cmd, check=True, capture_output=True, text=True)
        logger.info("Exhaustive matching completed")
    except subprocess.CalledProcessError:
        logger.info("Exhaustive matching failed, trying sequential matcher...")
        matcher_cmd = [
            "colmap", "sequential_matcher", 
            "--database_path", str(database_path),
            "--SequentialMatching.overlap", "3",
            "--SequentialMatching.quadratic_overlap", "1"
        ]
        logger.info(f"Running: {' '.join(matcher_cmd)}")
        result = subprocess.run(matcher_cmd, check=True, capture_output=True, text=True)
        logger.info("Sequential matching completed")
    
    # Try hierarchical mapper (often better for video sequences)
    logger.info("Step 3: Sparse reconstruction with hierarchical mapper...")
    try:
        mapper_cmd = [
            "colmap", "hierarchical_mapper", 
            "--database_path", str(database_path), 
            "--image_path", frames_dir, 
            "--output_path", str(sparse_dir),
            "--Mapper.min_num_matches", "5",
            "--Mapper.init_min_num_inliers", "30",
            "--Mapper.init_max_error", "12.0",
            "--Mapper.init_max_forward_motion", "0.99",
            "--Mapper.init_min_tri_angle", "5.0"
        ]
        logger.info(f"Running: {' '.join(mapper_cmd)}")
        result = subprocess.run(mapper_cmd, check=True, capture_output=True, text=True)
        logger.info("Hierarchical reconstruction completed")
    except subprocess.CalledProcessError:
        logger.info("Hierarchical mapper failed, trying standard mapper...")
        # Fallback to standard mapper with very relaxed parameters
        mapper_cmd = [
            "colmap", "mapper", 
            "--database_path", str(database_path), 
            "--image_path", frames_dir, 
            "--output_path", str(sparse_dir), 
            "--Mapper.min_num_matches", "5",
            "--Mapper.init_min_num_inliers", "30",
            "--Mapper.init_max_error", "12.0",
            "--Mapper.init_max_forward_motion", "0.99",
            "--Mapper.init_min_tri_angle", "5.0",
            "--Mapper.multiple_models", "0",
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0"
        ]
        logger.info(f"Running: {' '.join(mapper_cmd)}")
        result = subprocess.run(mapper_cmd, check=True, capture_output=True, text=True)
        logger.info("Standard reconstruction completed")
    
    if not (sparse_dir / "0").exists():
        logger.error("Sparse reconstruction failed - no model found")
        raise RuntimeError("sfm reconstruction failed")
    
    logger.info("COLMAP reconstruction completed successfully")
    return {
        "database": str(database_path),
        "sparse": str(sparse_dir / "0"),
        "frames_dir": frames_dir
    }

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        try:
            num_cameras = read_next_bytes(fid, 8, "Q")[0]
            logger.info(f"found {num_cameras} cameras")
            for _ in range(num_cameras):
                try:
                    camera_properties = read_next_bytes(fid, num_bytes=20, format_char_sequence="iiiii")
                    if len(camera_properties) < 5:
                        logger.warning(f"camera data incomplete: {len(camera_properties)} values")
                        continue
                    camera_id = camera_properties[0]
                    model_id = camera_properties[1]
                    width = camera_properties[2]
                    height = camera_properties[3]
                    num_params = camera_properties[4]
                    params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
                    cameras[camera_id] = {
                        "id": camera_id,
                        "model": model_id,
                        "width": width,
                        "height": height,
                        "params": params
                    }
                except Exception as e:
                    logger.warning(f"skipping camera: {e}")
                    continue
        except Exception as e:
            logger.error(f"failed to read cameras file: {e}")
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            try:
                binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
                image_id = binary_image_properties[0]
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_name = ""
                current_char = read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(fid, 1, "c")[0]
                num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
                x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                images[image_id] = {
                    "id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": image_name,
                    "xys": xys,
                    "point3D_ids": point3D_ids
                }
            except Exception as e:
                logger.warning(f"skipping image: {e}")
                continue
    return images

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def create_nerf_dataset(sfm_results: Dict[str, str], output_dir: str, train_ratio: float = 0.8) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    cameras_file = Path(sfm_results["sparse"]) / "cameras.bin"
    images_file = Path(sfm_results["sparse"]) / "images.bin"
    
    if not cameras_file.exists() or not images_file.exists():
        raise FileNotFoundError("colmap output files not found")
        
    cameras = read_cameras_binary(str(cameras_file))
    images = read_images_binary(str(images_file))
    
    if not cameras or not images:
        raise ValueError("no cameras or images found")
        
    camera = list(cameras.values())[0]
    fx, fy, cx, cy = camera["params"][:4]
    w, h = camera["width"], camera["height"]
    camera_angle_x = 2 * np.arctan(w / (2 * fx))
    
    image_items = list(images.items())
    np.random.seed(42)
    np.random.shuffle(image_items)
    
    num_train = int(len(image_items) * train_ratio)
    train_items = image_items[:num_train]
    test_items = image_items[num_train:]
    
    train_transforms = create_transforms(train_items, cameras, camera_angle_x, str(train_dir), "train")
    test_transforms = create_transforms(test_items, cameras, camera_angle_x, str(test_dir), "test")
    
    train_json_path = output_path / "transforms_train.json"
    test_json_path = output_path / "transforms_test.json"
    
    with open(train_json_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_json_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)
        
    copy_images(train_items, sfm_results["frames_dir"], str(train_dir))
    copy_images(test_items, sfm_results["frames_dir"], str(test_dir))
    
    return {
        "output_dir": str(output_path),
        "train_transforms": str(train_json_path),
        "test_transforms": str(test_json_path),
        "num_train": len(train_items),
        "num_test": len(test_items)
    }

def create_transforms(image_items: List, cameras: Dict, camera_angle_x: float, output_dir: str, split: str) -> Dict:
    frames = []
    
    for idx, (image_id, image) in enumerate(image_items):
        camera = cameras[image["camera_id"]]
        fx, fy, cx, cy = camera["params"][:4]
        
        R = qvec2rotmat(image["qvec"])
        t = image["tvec"]
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = t
        transform_matrix[1:3, :] *= -1  # convert to nerf coordinate system
        
        frame = {
            "file_path": f"{split}/frame_{idx:06d}.png",
            "transform_matrix": transform_matrix.tolist()
        }
        frames.append(frame)
    
    return {
        "camera_angle_x": camera_angle_x,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": cameras[list(cameras.keys())[0]]["width"],
        "h": cameras[list(cameras.keys())[0]]["height"],
        "frames": frames
    }

def copy_images(image_items: List, source_dir: str, target_dir: str):
    for idx, (image_id, image) in enumerate(image_items):
        source_path = Path(source_dir) / image["name"]
        target_path = Path(target_dir) / f"frame_{idx:06d}.png"
        
        if source_path.exists():
            shutil.copy2(source_path, target_path)

def generate_output_name(video_path: str, fps: float, max_frames: Optional[int] = None) -> str:
    video_name = Path(video_path).stem
    fps_str = f"fps{fps:.1f}".replace(".", "")
    
    if max_frames:
        frames_str = f"max{max_frames}"
        return f"{video_name}_{fps_str}_{frames_str}_nerf"
    else:
        return f"{video_name}_{fps_str}_nerf"

def main():
    parser = argparse.ArgumentParser(description="convert video to nerf dataset")
    parser.add_argument("video_path", help="path to input video file")
    parser.add_argument("--output_dir", "-o", default=None, help="output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="frames per second")
    parser.add_argument("--max_frames", type=int, default=None, help="maximum frames")
    parser.add_argument("--image_scale", type=float, default=1.0, help="image scale")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train ratio")
    parser.add_argument("--min_matches", type=int, default=15, help="min matches")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.video_path):
        logger.error(f"video file not found: {args.video_path}")
        sys.exit(1)
    
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        logger.error("train ratio must be between 0 and 1")
        sys.exit(1)
    
    if args.output_dir is None:
        output_name = generate_output_name(args.video_path, args.fps, args.max_frames)
        args.output_dir = f"./{output_name}"
    
    try:
        check_colmap()
        
        # extract frames
        frame_paths = extract_frames(args.video_path, args.output_dir, args.fps, args.max_frames, args.image_scale)
        
        if len(frame_paths) < 10:
            raise ValueError("too few frames extracted")
        
        # run sfm
        sfm_output_dir = Path(args.output_dir) / "sfm_output"
        sfm_results = run_colmap(str(Path(args.output_dir) / "frames"), str(sfm_output_dir), args.min_matches)
        
        # create nerf dataset
        nerf_output_dir = Path(args.output_dir) / "nerf_dataset"
        results = create_nerf_dataset(sfm_results, str(nerf_output_dir), args.train_ratio)
        
        print(f"\n✓ nerf dataset created successfully!")
        print(f"[DIR] output directory: {results['output_dir']}")
        print(f"[INFO] train images: {results['num_train']}")
        print(f"[INFO] test images: {results['num_test']}")
        print(f"[FILE] train transforms: {results['train_transforms']}")
        print(f"[FILE] test transforms: {results['test_transforms']}")
        
    except Exception as e:
        logger.error(f"pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
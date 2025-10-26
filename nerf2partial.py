import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import shutil
import random

def load_cameras(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cameras = []
    for frame in data['frames']:
        T = np.array(frame['transform_matrix'])
        pos = T[:3, 3]
        dir = -T[:3, 2]
        cameras.append((pos, dir, frame))
    
    return cameras

def filter_cameras_by_frustum(cameras, center_dir, vertical_span, horizontal_span):
    center_dir = np.array(center_dir) / np.linalg.norm(center_dir)
    v_half = np.radians(vertical_span / 2)
    h_half = np.radians(horizontal_span / 2)
    
    filtered = []
    for pos, dir, frame in cameras:
        pos_norm = pos / np.linalg.norm(pos)
        angle = np.arccos(np.clip(np.dot(center_dir, pos_norm), -1, 1))
        
        if angle <= min(v_half, h_half):
            filtered.append((pos, dir, frame))
    
    return filtered

def create_3d_plot(train_cameras, test_cameras, output_path='cameras.png', 
                   all_train=None, all_test=None):
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
    
    if all_train and all_test:
        for pos, _, _ in all_train:
            ax.scatter(*pos, c='lightblue', s=5, alpha=0.3)
        for pos, _, _ in all_test:
            ax.scatter(*pos, c='lightcoral', s=5, alpha=0.3)
    
    for i, (pos, dir, _) in enumerate(train_cameras):
        ax.scatter(*pos, c='blue', s=20, label='Training' if i == 0 else "")
        ax.quiver(*pos, *dir, length=0.3, color='blue')
    
    for i, (pos, dir, _) in enumerate(test_cameras):
        ax.scatter(*pos, c='red', s=20, label='Test' if i == 0 else "")
        ax.quiver(*pos, *dir, length=0.3, color='red')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Camera Positions and Viewing Directions')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")


def create_subset_dataset(input_folder, output_path, name, center_dir, v_span, h_span):
    with open(f'{input_folder}/transforms_train.json', 'r') as f:
        train_data = json.load(f)
    camera_angle_x = train_data['camera_angle_x']
    
    all_train = load_cameras(f'{input_folder}/transforms_train.json')
    all_test = load_cameras(f'{input_folder}/transforms_test.json')
    
    filtered_train = filter_cameras_by_frustum(all_train, center_dir, v_span, h_span)
    filtered_test = filter_cameras_by_frustum(all_test, center_dir, v_span, h_span)
    
    all_filtered = filtered_train + filtered_test
    random.shuffle(all_filtered)
    split_idx = int(len(all_filtered) * 0.4)
    new_train = all_filtered[:split_idx]
    new_test = all_filtered[split_idx:]
    
    output_folder = f'{output_path}/{name}'
    os.makedirs(f'{output_folder}/train', exist_ok=True)
    os.makedirs(f'{output_folder}/test', exist_ok=True)
    
    for i, (_, _, frame) in enumerate(new_train):
        src_path = f"{input_folder}/{frame['file_path']}.png"
        dst_path = f"{output_folder}/train/r_{i}.png"
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    for i, (_, _, frame) in enumerate(new_test):
        src_path = f"{input_folder}/{frame['file_path']}.png"
        dst_path = f"{output_folder}/test/r_{i}.png"
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    train_json = {
        "camera_angle_x": camera_angle_x,
        "frames": [
            {
                "file_path": f"./train/r_{i}",
                "rotation": frame["rotation"],
                "transform_matrix": frame["transform_matrix"]
            }
            for i, (_, _, frame) in enumerate(new_train)
        ]
    }
    
    test_json = {
        "camera_angle_x": camera_angle_x,
        "frames": [
            {
                "file_path": f"./test/r_{i}",
                "rotation": frame["rotation"],
                "transform_matrix": frame["transform_matrix"]
            }
            for i, (_, _, frame) in enumerate(new_test)
        ]
    }
    
    with open(f'{output_folder}/transforms_train.json', 'w') as f:
        json.dump(train_json, f, indent=2)
    with open(f'{output_folder}/transforms_test.json', 'w') as f:
        json.dump(test_json, f, indent=2)
    
    print(f"Created subset dataset: {len(new_train)} train, {len(new_test)} test images")
    
    create_3d_plot(new_train, new_test, f'{output_folder}/cameras.png', all_train, all_test)

if len(sys.argv) >= 7:
    folder = sys.argv[1]
    center_dir = [float(x) for x in sys.argv[2].split(',')]
    v_span, h_span = float(sys.argv[3]), float(sys.argv[4])
    output_path = sys.argv[5]
    name = sys.argv[6]
    
    create_subset_dataset(folder, output_path, name, center_dir, v_span, h_span)
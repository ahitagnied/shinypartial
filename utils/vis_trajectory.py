import numpy as np
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.edgecolor'] = 'black'

def qvec2rotmat(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def read_images_binary(path):
    images = []
    with open(path, 'rb') as f:
        num_images = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('i', f.read(4))[0]
            qvec = struct.unpack('d' * 4, f.read(8 * 4))
            tvec = struct.unpack('d' * 3, f.read(8 * 3))
            camera_id = struct.unpack('i', f.read(4))[0]
            
            name_chars = []
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name_chars.append(c)
            name = b''.join(name_chars).decode('utf-8')
            
            num_points2D = struct.unpack('Q', f.read(8))[0]
            f.read(24 * num_points2D)
            
            images.append({
                'qvec': np.array(qvec),
                'tvec': np.array(tvec),
                'name': name
            })
    return images

def read_points3D_binary(path):
    points = []
    with open(path, 'rb') as f:
        num_points = struct.unpack('Q', f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack('Q', f.read(8))[0]
            xyz = struct.unpack('d' * 3, f.read(8 * 3))
            rgb = struct.unpack('B' * 3, f.read(3))
            error = struct.unpack('d', f.read(8))[0]
            
            track_length = struct.unpack('Q', f.read(8))[0]
            f.read(8 * track_length)
            
            points.append({
                'xyz': np.array(xyz),
                'rgb': np.array(rgb) / 255.0
            })
    
    return points

def compute_camera_positions(images, flip_yz=True):
    positions = []
    orientations = []
    
    for img in images:
        R_w2c = qvec2rotmat(img['qvec'])
        t_w2c = img['tvec']
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        
        if flip_yz:
            t_c2w_flipped = t_c2w.copy()
            t_c2w_flipped[1] *= -1
            t_c2w_flipped[2] *= -1
            
            R_c2w_flipped = R_c2w.copy()
            R_c2w_flipped[1, :] *= -1
            R_c2w_flipped[2, :] *= -1
            
            positions.append(t_c2w_flipped)
            orientations.append(R_c2w_flipped)
        else:
            positions.append(t_c2w)
            orientations.append(R_c2w)
    
    return np.array(positions), orientations

def plot_point_cloud(sparse_dir, save_path=None, max_points=50000, point_size=2, 
                     title=None, show_axes=True, view_angle=(30, 45), show_cameras=True):
    sparse_path = Path(sparse_dir)
    
    print(f"Loading reconstruction from {sparse_path}")
    
    try:
        images = read_images_binary(sparse_path / "images.bin")
        positions, orientations = compute_camera_positions(images)
        print(f"Loaded {len(images)} camera poses")
    except Exception as e:
        print(f"Error loading cameras: {e}")
        positions, orientations = None, None
    
    try:
        points = read_points3D_binary(sparse_path / "points3D.bin")
        print(f"Loaded {len(points)} 3D points")
    except Exception as e:
        print(f"Error loading points: {e}")
        return
    
    if len(points) == 0:
        print("Error: No 3D points in reconstruction!")
        return
    
    if len(points) > max_points:
        print(f"Sampling {max_points} points for visualization...")
        indices = np.random.choice(len(points), max_points, replace=False)
        points = [points[i] for i in indices]
    
    pts = np.array([p['xyz'] for p in points])
    pts[:, 1] *= -1
    pts[:, 2] *= -1
    colors = np.array([p['rgb'] for p in points])
    
    all_pts = pts.copy()
    if positions is not None:
        all_pts = np.vstack([all_pts, positions])
    
    center = all_pts.mean(axis=0)
    extent = all_pts.max(axis=0) - all_pts.min(axis=0)
    max_extent = extent.max()
    
    print(f"\nReconstruction Statistics:")
    if positions is not None:
        print(f"  Number of cameras: {len(positions)}")
    print(f"  Number of 3D points: {len(points):,}")
    print(f"  Scene center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"  Scene extent: X={extent[0]:.2f}, Y={extent[1]:.2f}, Z={extent[2]:.2f}")
    print(f"  Max extent: {max_extent:.2f}")
    
    adaptive_point_size = point_size
    if len(points) < 1000 and max_extent > 10:
        adaptive_point_size = max(point_size, max_extent * 0.5)
        print(f"  Auto-scaled point size: {point_size} → {adaptive_point_size:.1f} (sparse cloud)")
    
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=colors,
                        s=adaptive_point_size,
                        alpha=0.9,
                        edgecolors='black' if len(points) < 1000 else 'none',
                        linewidths=0.5 if len(points) < 1000 else 0,
                        depthshade=True,
                        label='Point cloud')
    
    if show_cameras and positions is not None:
        frustum_scale = max_extent * 0.08
        aspect = 0.6
        
        for i, (pos, R) in enumerate(zip(positions, orientations)):
            corners_cam = np.array([
                [-aspect * frustum_scale, -aspect * frustum_scale, frustum_scale],
                [ aspect * frustum_scale, -aspect * frustum_scale, frustum_scale],
                [ aspect * frustum_scale,  aspect * frustum_scale, frustum_scale],
                [-aspect * frustum_scale,  aspect * frustum_scale, frustum_scale]
            ])
            
            corners_world = pos + (R @ corners_cam.T).T
            
            for j in range(4):
                ax.plot([pos[0], corners_world[j, 0]], 
                       [pos[1], corners_world[j, 1]], 
                       [pos[2], corners_world[j, 2]], 
                       'k-', linewidth=0.8, alpha=0.5)
            
            for j in range(4):
                next_j = (j + 1) % 4
                ax.plot([corners_world[j, 0], corners_world[next_j, 0]], 
                       [corners_world[j, 1], corners_world[next_j, 1]], 
                       [corners_world[j, 2], corners_world[next_j, 2]], 
                       'k-', linewidth=1.2, alpha=0.7,
                       label='Cameras' if i == 0 and j == 0 else '')
        
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='red', s=50, marker='o', edgecolors='black', 
                  linewidths=1, alpha=0.9, label='Camera positions')
    
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    mid = center
    ax.set_xlim(mid[0] - max_extent/2, mid[0] + max_extent/2)
    ax.set_ylim(mid[1] - max_extent/2, mid[1] + max_extent/2)
    ax.set_zlim(mid[2] - max_extent/2, mid[2] + max_extent/2)
    
    if show_axes:
        ax.set_xlabel('X (units)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (units)', fontsize=14, fontweight='bold')
        ax.set_zlabel('Z (units)', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\nSaved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("sparse_dir")
    parser.add_argument("--save", "-s")
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--title")
    parser.add_argument("--no-axes", action="store_true")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--elevation", type=float, default=30)
    parser.add_argument("--azimuth", type=float, default=45)
    
    args = parser.parse_args()
    
    plot_point_cloud(
        args.sparse_dir,
        save_path=args.save,
        max_points=args.max_points,
        point_size=args.point_size,
        title=args.title,
        show_axes=not args.no_axes,
        view_angle=(args.elevation, args.azimuth),
        show_cameras=not args.no_cameras
    )

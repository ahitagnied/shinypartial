import numpy as np
import os

MIRROR_WIDTH = 1024
MIRROR_HEIGHT = 512
MIRROR_SIZE_X = 300.0
MIRROR_SIZE_Y = 150.0
BORDER_WIDTH = 20.0
DEFORM_AMPLITUDE = 8.0
OUTPUT_FILENAME = "funny_mirror.obj"

def create_funny_mirror_obj(filename=OUTPUT_FILENAME, width=MIRROR_WIDTH, height=MIRROR_HEIGHT):
    os.makedirs("files", exist_ok=True)
    filepath = os.path.join("files", filename)
    
    x = np.linspace(-MIRROR_SIZE_X, MIRROR_SIZE_X, width)
    y = np.linspace(-MIRROR_SIZE_Y, MIRROR_SIZE_Y, height)
    X, Y = np.meshgrid(x, y)
    
    wave_x = np.sin(X * 0.03) * np.cos(Y * 0.05)
    wave_y = np.sin(Y * 0.04) * np.cos(X * 0.02)
    ripple = np.sin(np.sqrt(X**2 + Y**2) * 0.02)
    
    Z = DEFORM_AMPLITUDE * (wave_x + wave_y + ripple * 0.5)
    
    border_x_min = -MIRROR_SIZE_X + BORDER_WIDTH
    border_x_max = MIRROR_SIZE_X - BORDER_WIDTH
    border_y_min = -MIRROR_SIZE_Y + BORDER_WIDTH
    border_y_max = MIRROR_SIZE_Y - BORDER_WIDTH
    
    border_mask = (X < border_x_min) | (X > border_x_max) | (Y < border_y_min) | (Y > border_y_max)
    
    vertices = []
    normals = []
    colors = []
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    for i in range(height):
        for j in range(width):
            vertices.append([X[i,j], Y[i,j], Z[i,j]])
            
            if i > 0 and i < height-1 and j > 0 and j < width-1:
                dz_dx = (Z[i, j+1] - Z[i, j-1]) / (2 * dx)
                dz_dy = (Z[i+1, j] - Z[i-1, j]) / (2 * dy)
                
                normal = np.array([-dz_dx, -dz_dy, 1.0])
                normal = normal / np.linalg.norm(normal)
                normals.append(normal)
            else:
                normals.append([0, 0, 1])
            
            if border_mask[i, j]:
                colors.append([0.0, 0.0, 0.0])
            else:
                colors.append([0.9, 0.9, 0.95])
    
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            v1 = i * width + j + 1
            v2 = v1 + 1
            v3 = (i + 1) * width + j + 1
            v4 = v3 + 1
            
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    with open(filepath, 'w') as f:
        f.write("mtllib funny_mirror.mtl\n")
        f.write("usemtl mirror_material\n\n")
        
        for i, v in enumerate(vertices):
            c = colors[i]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
        
        f.write("\n")
        
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")
    
    mtl_filepath = os.path.join("files", "funny_mirror.mtl")
    with open(mtl_filepath, 'w') as f:
        f.write("newmtl mirror_material\n")
        f.write("Ka 0.1 0.1 0.1\n")
        f.write("Kd 0.1 0.1 0.1\n")
        f.write("Ks 1.0 1.0 1.0\n")
        f.write("Ns 10000.0\n")
        f.write("Ni 1.52\n")
        f.write("d 1.0\n")
        f.write("illum 3\n")
    
    print(f"funny mirror exported to '{filepath}'")
    print(f"vertices: {len(vertices)}, faces: {len(faces)}")
    print(f"resolution: {width}x{height}")

if __name__ == "__main__":
    create_funny_mirror_obj()


import numpy as np
from scipy.ndimage import gaussian_filter
import os

NOISE_WIDTH = 512
NOISE_HEIGHT = 512
NOISE_SCALE = 20.0
NOISE_OCTAVES = 4

MIRROR_RESOLUTION = 512
MIRROR_SIZE = 220.0  # -mirror_size to +mirror_size
DEFORM_SCALE = 2.0
OUTPUT_FILENAME = "deformed_mirror.obj"

SMOOTH_SCALE = 30.0
SMOOTH_OCTAVES = 2

def generate_perlin_noise(width=NOISE_WIDTH, height=NOISE_HEIGHT, scale=NOISE_SCALE, octaves=NOISE_OCTAVES):
    def interpolate(a, b, x):
        return a + (b - a) * (3 * x**2 - 2 * x**3)
    
    total = np.zeros((height, width))
    frequency = 1.0
    amplitude = 1.0
    
    for _ in range(octaves):
        n = max(4, round(frequency * width / 6))
        noise = np.random.rand(n, n)
        
        for i in range(height):
            for j in range(width):
                x = j * frequency / width
                y = i * frequency / height
                
                x_int = int(x)
                y_int = int(y)
                x_frac = x - x_int
                y_frac = y - y_int
                
                x0y0 = noise[y_int % n][x_int % n]
                x1y0 = noise[y_int % n][(x_int + 1) % n]
                x0y1 = noise[(y_int + 1) % n][x_int % n]
                x1y1 = noise[(y_int + 1) % n][(x_int + 1) % n]
                
                y0 = interpolate(x0y0, x1y0, x_frac)
                y1 = interpolate(x0y1, x1y1, x_frac)
                
                total[i][j] += amplitude * interpolate(y0, y1, y_frac)
        
        amplitude *= 0.8
        frequency *= 2.0
    
    noise = gaussian_filter(total, sigma=scale/12)
    noise = gaussian_filter(noise, sigma=1.5)
    
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

def create_smooth_mirror_obj(filename=OUTPUT_FILENAME, resolution=MIRROR_RESOLUTION, deform_scale=DEFORM_SCALE):
    os.makedirs("files", exist_ok=True)
    filepath = os.path.join("files", filename)
    
    noise = generate_perlin_noise(resolution, resolution, scale=SMOOTH_SCALE, octaves=SMOOTH_OCTAVES)
    
    x = np.linspace(-MIRROR_SIZE, MIRROR_SIZE, resolution)
    y = np.linspace(-MIRROR_SIZE, MIRROR_SIZE, resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = deform_scale * noise
    
    vertices = []
    normals = []
    
    for i in range(resolution):
        for j in range(resolution):
            vertices.append([X[i,j], Y[i,j], Z[i,j]])
            
            if i > 0 and i < resolution-1 and j > 0 and j < resolution-1:
                dz_dx = (Z[i, j+1] - Z[i, j-1]) / (2 * (x[1] - x[0]))
                dz_dy = (Z[i+1, j] - Z[i-1, j]) / (2 * (y[1] - y[0]))
                
                normal = np.array([-dz_dx, -dz_dy, 1.0])
                normal = normal / np.linalg.norm(normal)
                normals.append(normal)
            else:
                normals.append([0, 0, 1])
    
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v1 = i * resolution + j + 1
            v2 = v1 + 1
            v3 = (i + 1) * resolution + j + 1
            v4 = v3 + 1
            
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    with open(filepath, 'w') as f:
        f.write("# Ultra-smooth mirror surface\n")
        f.write("# Generated for Blender mirror rendering\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")
    
    print(f"deformed surface exported to '{filepath}'")
    print(f"vertices: {len(vertices)}, faces: {len(faces)}")

if __name__ == "__main__":
    create_smooth_mirror_obj()
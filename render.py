import bpy
import os
import sys
import numpy as np
from mathutils import Vector

sys.path.append(os.path.dirname(__file__))
from trajectory import get_trajectory
from setup import *

def render_scene(config, train_ratio=0.33, mode="auto"):
    if mode == "auto":
        mode = "multiple" if "items" in config["blender_obj"] else "single"
    
    folder_name = generate_folder_name(config, mode)
    output_dir = config["output"]["directory"]
    object_dir = os.path.join(output_dir, folder_name)
    train_dir = os.path.join(object_dir, "train")
    test_dir = os.path.join(object_dir, "test")
    
    os.makedirs(object_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    setup_scene(config)
    
    if mode == "single":
        objects = [import_single_object(config["blender_obj"])]
        target = objects[0].location
    else:
        objects = import_multiple_objects(config)
        target = get_centroid(objects)
    
    setup_environment(config)
    
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    
    camera_data = camera.data
    camera_angle_x = camera_data.angle_x
    
    trajectory = get_trajectory(config)
    num_images = config["camera"]["num_images"]
    
    num_train = round(num_images * train_ratio)
    stride = round(num_images / num_train)
    
    ntrain = ntest = 0
    train_camera_params = []
    test_camera_params = []
    
    for i in range(num_images):
        if trajectory["mode"] == "spherical":
            theta = trajectory["thetas"][i]
            phi = trajectory["phis"][i]
            r = config["camera"]["distance"]
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z_cam = r * np.cos(theta)
            
            camera.location = (x, y, z_cam)
            
            if trajectory["look_at"] == "center":
                direction = target - camera.location
                rot_quat = direction.to_track_quat('-Z', 'Y')
                camera.rotation_euler = rot_quat.to_euler()
        
        elif trajectory["mode"] == "planar":
            pos = trajectory["positions"][i]
            plane_normal = Vector(trajectory["plane_normal"]).normalized()
            
            right = Vector((1, 0, 0)) if abs(plane_normal.dot(Vector((1, 0, 0)))) < 0.9 else Vector((0, 1, 0))
            up = plane_normal.cross(right).normalized()
            right = up.cross(plane_normal).normalized()
            
            camera_pos = Vector(target) - plane_normal * pos[2] + right * pos[0] + up * pos[1]
            camera.location = camera_pos
            
            if trajectory["look_at"] == "perpendicular":
                rot_quat = plane_normal.to_track_quat('-Z', 'Y')
                camera.rotation_euler = rot_quat.to_euler()
            elif trajectory["look_at"] == "center":
                direction = target - camera.location
                rot_quat = direction.to_track_quat('-Z', 'Y')
                camera.rotation_euler = rot_quat.to_euler()
        
        bpy.context.view_layer.update()
        
        frame = {
            "file_path": f"train/r_{ntrain}" if i % stride == 0 else f"test/r_{ntest}",
            "rotation": trajectory["rotation_step"],
            "transform_matrix": [list(row) for row in camera.matrix_world],
            "camera_angle_x": camera_angle_x
        }
        
        if i % stride == 0:
            train_camera_params.append(frame)
            ntrain += 1
            out_dir = train_dir
            fname = f"r_{ntrain-1}"
        else:
            test_camera_params.append(frame)
            ntest += 1
            out_dir = test_dir
            fname = f"r_{ntest-1}"
        
        bpy.context.scene.render.filepath = os.path.join(out_dir, f"{fname}.png")
        bpy.ops.render.render(write_still=True)
    
    create_transform_json(config, train_camera_params, "transforms_train.json", folder_name)
    create_transform_json(config, test_camera_params, "transforms_test.json", folder_name)
    
    print(f"finished rendering {num_images} images ({ntrain} train, {ntest} test)")

if __name__ == "__main__":
    import sys
    
    script_args = []
    try:
        separator_index = sys.argv.index('--')
        script_args = sys.argv[separator_index + 1:]
    except ValueError:
        for arg in sys.argv[1:]:
            if not arg.startswith('--') and not arg.endswith('.py'):
                script_args.append(arg)
    
    if script_args:
        config_name = script_args[0]
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
        config_path = os.path.join(os.path.dirname(__file__), "configs", config_name)
    else:
        config_path = os.path.join(os.path.dirname(__file__), "configs", "single_obj.yaml")
    
    config = load_config(config_path)
    render_scene(config, train_ratio=0.33)
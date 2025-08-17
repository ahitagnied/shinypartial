import bpy
import math
import os
import json
from mathutils import Vector
import yaml
import inspect
import numpy as np
import bpy
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.preferences.addon_enable(module="io_scene_fbx")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_scene(config):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    
    bpy.context.scene.render.engine = 'CYCLES'
    resolution = config["output"]["resolution"]
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.image_settings.file_format = config["output"]["format"]
    bpy.context.scene.cycles.samples = config["output"]["samples"]
    bpy.context.scene.render.film_transparent = True

def import_object(config):
    obj_cfg = config["blender_obj"]
    path = obj_cfg["path"]
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise Exception(f"Unsupported: {ext}")

    imported = bpy.context.selected_objects
    if not imported:
        raise Exception("no objects were imported")
    
    obj = imported[0]
    obj.scale = tuple(obj_cfg["scale"])
    obj.location = tuple(obj_cfg["location"])

    rot_deg = obj_cfg.get("rotation", [0, 0, 0])
    obj.rotation_euler = tuple(math.radians(r) for r in rot_deg)
    
    apply_bevel(obj, obj_cfg["bevel"])
    
    if not obj.data.materials or obj_cfg.get("override_materials", False):
        apply_shiny_material(obj, obj_cfg["material"])
    
    return [obj]

def apply_bevel(obj, bevel_cfg):
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Bevel", 'BEVEL')
    mod.width, mod.segments, mod.profile = (
      bevel_cfg["width"], bevel_cfg["segments"], bevel_cfg["profile"]
    )
    bpy.ops.object.modifier_apply(modifier=mod.name)

def apply_shiny_material(obj, mat_cfg):
    mat = bpy.data.materials.new("Shiny")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    for n in list(nodes): 
        nodes.remove(n)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out  = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    bsdf.inputs["Metallic"].default_value  = mat_cfg["metallic"]
    bsdf.inputs["Roughness"].default_value = mat_cfg["roughness"]
    spec = bsdf.inputs.get("Specular")
    if spec:
        spec.default_value = mat_cfg["specular"]
    else:
        bsdf.inputs["IOR"].default_value = mat_cfg["specular"]

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def setup_environment(config):
    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world
        
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links

    for node in list(world_nodes):
        world_nodes.remove(node)

    env_tex = world_nodes.new(type='ShaderNodeTexEnvironment')
    env_tex.image = bpy.data.images.load(config["environment"]["hdri_path"])
    env_tex.location = (-300, 0)

    background = world_nodes.new(type='ShaderNodeBackground')
    background.location = (0, 0)

    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    world_output.location = (300, 0)

    world_links.new(env_tex.outputs['Color'], background.inputs['Color'])
    world_links.new(background.outputs['Background'], world_output.inputs['Surface'])

def render_obj(config, train_ratio):
    folder_name = os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
    
    output_dir = config["output"]["directory"]
    object_dir = os.path.join(output_dir, folder_name)
    train_dir = os.path.join(object_dir, "train")
    test_dir = os.path.join(object_dir, "test")
    
    os.makedirs(object_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    setup_scene(config)
    blender_obj_list = import_object(config)
    setup_environment(config)
    
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object  
    bpy.context.scene.camera = camera  

    obj = blender_obj_list[0]  
    obj.location = (0, 0, 0)

    bpy.context.view_layer.update()
    
    camera_data = camera.data
    camera_angle_x = camera_data.angle_x

    obj = blender_obj_list[0]

    num_images = config["camera"]["num_images"]
    phi_g = np.pi*(3 - np.sqrt(5))
    theta_min = np.deg2rad(config["camera"]["theta_min_deg"])
    theta_max = np.deg2rad(config["camera"]["theta_max_deg"])
    
    phi_start_deg = config["camera"].get("phi_start_deg", 0)
    phi_end_deg = config["camera"].get("phi_end_deg", 360)
    phi_start_rad = np.deg2rad(phi_start_deg)
    phi_end_rad = np.deg2rad(phi_end_deg)
    
    phi_range_rad = phi_end_rad - phi_start_rad
    if phi_range_rad < 2*np.pi:
        phi_g = phi_range_rad / num_images
    else:
        phi_g = np.pi*(3 - np.sqrt(5))

    distance = config["camera"]["distance"]
    i = np.arange(num_images)

    z = np.cos(theta_min) + i/(num_images-1)*(np.cos(theta_max) - np.cos(theta_min))
    thetas = np.arccos(z)
    
    if phi_range_rad < 2*np.pi:
        phis = phi_start_rad + i * phi_g
    else:
        phis = (i * phi_g) % (2*np.pi)  
    
    rotation_step = phi_g

    num_train = round(num_images * train_ratio)
    stride    = round(num_images / num_train)

    ntrain = ntest = 0
    train_camera_params = []
    test_camera_params  = []

    for i in range(num_images):
        theta = thetas[i]
        phi = phis[i]

        r = distance  
        x = r * np.sin(theta) * np.cos(phi)  
        y = r * np.sin(theta) * np.sin(phi)  
        z_cam = r * np.cos(theta)

        camera.location = (x, y, z_cam)

        direction = Vector((0, 0, 0)) - camera.location  
        rot_quat = direction.to_track_quat('-Z', 'Y')  
        camera.rotation_euler = rot_quat.to_euler()  

        bpy.context.view_layer.update()  

        frame = {  
            "file_path": f"train/r_{ntrain}" if i % stride == 0 else f"test/r_{ntest}",  
            "rotation": rotation_step,  
            "transform_matrix": [list(row) for row in camera.matrix_world],  
            "camera_angle_x": camera_angle_x  
        }  

        if i % stride == 0:
            train_camera_params.append(frame);  ntrain += 1
            out_dir = train_dir;  fname = f"r_{ntrain-1:04d}"
        else:
            test_camera_params.append(frame);   ntest  += 1
            out_dir = test_dir;   fname = f"r_{ntest-1:04d}"

        bpy.context.scene.render.filepath = os.path.join(out_dir, f"{fname}.png")
        bpy.ops.render.render(write_still=True)
    
    create_transform_json(config, train_camera_params, "transforms_train.json", folder_name)
    create_transform_json(config, test_camera_params, "transforms_test.json", folder_name)
    
    print(f"finished rendering {num_images} images ({ntrain} train, {ntest} test)")

def create_transform_json(config, camera_params, output_filename, folder_name):
    if not camera_params:
        print(f"warning: no camera parameters for {output_filename}")
        return
        
    output_dir = config["output"]["directory"]
    object_dir = os.path.join(output_dir, folder_name)
    
    transform_data = {
        "camera_angle_x": camera_params[0]["camera_angle_x"],
        "frames": []
    }
    
    for params in camera_params:
        frame_data = {
            "file_path": params["file_path"],
            "rotation": params["rotation"],
            "transform_matrix": params["transform_matrix"]
        }
        transform_data["frames"].append(frame_data)
    
    transform_json_path = os.path.join(object_dir, output_filename)
    with open(transform_json_path, "w") as f:
        json.dump(transform_data, f, indent=4)
    
    print(f"saved {output_filename} to {transform_json_path}")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "blender_obj.yaml")
    config = load_config(config_path)
    render_obj(config, train_ratio=0.33)
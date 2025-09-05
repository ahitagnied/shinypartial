import bpy
import os
import json
import math
import numpy as np
from mathutils import Vector

def load_config(config_path):
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_scene(config):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="io_scene_fbx")
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    
    bpy.context.scene.render.engine = 'CYCLES'
    resolution = config["output"]["resolution"]
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.image_settings.file_format = config["output"]["format"]
    bpy.context.scene.cycles.samples = config["output"]["samples"]
    bpy.context.scene.render.film_transparent = True

def import_single_object(obj_cfg):
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
    
    return obj

def import_multiple_objects(config):
    objs = []
    spacing = config["blender_obj"]["spacing"]
    for idx, obj_cfg in enumerate(config["blender_obj"]["items"]):
        obj = import_single_object(obj_cfg)
        x0, y0, z0 = obj_cfg.get("base_location", [0, 0, 0])
        obj.location = (x0 + idx * spacing, y0, z0)
        objs.append(obj)
    return objs

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

def get_centroid(objects):
    if not objects:
        return Vector((0, 0, 0))
    
    total = Vector((0, 0, 0))
    for obj in objects:
        total += obj.location
    return total / len(objects)

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

def generate_folder_name(config, mode="single"):
    if mode == "single":
        obj_path = config["blender_obj"]["path"]
        obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    else:
        obj_names = []
        for item in config['blender_obj']['items']:
            name = os.path.splitext(os.path.basename(item['path']))[0]
            obj_names.append(name)
        obj_name = "_".join(obj_names)
    
    theta_min = config["camera"]["theta_min_deg"]
    theta_max = config["camera"]["theta_max_deg"]
    phi_min = config["camera"].get("phi_start_deg", 0)
    phi_max = config["camera"].get("phi_end_deg", 360)
    trajectory_type = config["camera"].get("trajectory_type", "golden_spiral")
    look_at = config["camera"].get("look_at", "perpendicular")
    
    folder_name = f"{obj_name}_t{theta_min}-{theta_max}_p{phi_min}-{phi_max}_{trajectory_type}_{look_at}"
    
    if trajectory_type != "golden_spiral":
        trajectory_offset = config["camera"].get("trajectory_offset", [0, 0, 0])
        if trajectory_offset != [0, 0, 0]:
            offset_str = f"_offset{trajectory_offset[0]}{trajectory_offset[1]}{trajectory_offset[2]}"
            folder_name += offset_str
    
    return folder_name
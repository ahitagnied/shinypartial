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
    """load configuration from yaml file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_scene(config):
    """setup the basic blender scene"""
    # clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)
    
    # set render settings
    bpy.context.scene.render.engine = 'CYCLES'
    resolution = config["output"]["resolution"]
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.image_settings.file_format = config["output"]["format"]
    bpy.context.scene.cycles.samples = config["output"]["samples"]
    bpy.context.scene.render.film_transparent = True # <-- make background transparent like NeRO

#-----------------------------------------------------------------------------------------------
def import_objects(config):
    objs = []
    spacing = config["blender_obj"]["spacing"]
    for idx, obj_cfg in enumerate(config["blender_obj"]["items"]):
        path = obj_cfg["path"]
        ext  = os.path.splitext(path)[1].lower()
        if ext == ".obj":
            # Simple import - materials are imported by default if .mtl exists
            bpy.ops.wm.obj_import(filepath=path)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path)
        else:
            raise Exception(f"Unsupported: {ext}")

        imported = bpy.context.selected_objects
        obj = imported[0]
        obj.scale = tuple(obj_cfg["scale"])
        x0, y0, z0 = obj_cfg["base_location"]
        obj.location = (x0 + idx * spacing, y0, z0)
        rot_deg = obj_cfg.get("rotation", [0, 0, 0])
        obj.rotation_euler = tuple(math.radians(r) for r in rot_deg)
        apply_bevel(obj, obj_cfg["bevel"])
        
        # Only apply procedural material if no materials were imported
        if not obj.data.materials or obj_cfg.get("override_materials", False):
            apply_shiny_material(obj, obj_cfg["material"])
        
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

    # --- assign the material correctly:
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
#-----------------------------------------------------------------------------------------------

def setup_environment(config):
    """set up the hdri environment"""
    # add hdri environment map
    if bpy.context.scene.world is None:
        # create a new world if it doesn't exist
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world
        
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links

    # clear default nodes
    for node in list(world_nodes):
        world_nodes.remove(node)

    # add environment texture node
    env_tex = world_nodes.new(type='ShaderNodeTexEnvironment')
    env_tex.image = bpy.data.images.load(config["environment"]["hdri_path"])
    env_tex.location = (-300, 0)

    # add background node
    background = world_nodes.new(type='ShaderNodeBackground')
    background.location = (0, 0)

    # add output node
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    world_output.location = (300, 0)

    # link nodes
    world_links.new(env_tex.outputs['Color'], background.inputs['Color'])
    world_links.new(background.outputs['Background'], world_output.inputs['Surface'])

def render_obj(config, train_ratio):
    """render blender_obj and save to train and test folders based on script name"""
    # get script name for folder name
    folder_name = os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0]
    
    # setup directories
    output_dir = config["output"]["directory"]
    object_dir = os.path.join(output_dir, folder_name)
    train_dir = os.path.join(object_dir, "train")
    test_dir = os.path.join(object_dir, "test")
    
    os.makedirs(object_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # setup scene and camera
    setup_scene(config)
    all_objs = import_objects(config)
    setup_environment(config)

    # after import_objects()
    locs = [obj.location for obj in all_objs]
    centroid = sum(locs, Vector()) / len(locs)
    target = centroid
    
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.location = (5, 0, 0)
    camera.rotation_euler = (math.pi/2, 0, math.pi/2)
    bpy.context.scene.camera = camera
    
    # calculate camera parameters
    camera_data = camera.data
    camera_angle_x = camera_data.angle_x
    target = centroid

    # render settings
    num_images = config["camera"]["num_images"]
    phi_g = np.pi*(3 - np.sqrt(5))                  # golden angle ≃ 2.39996 rad
    theta_max = np.deg2rad(config["camera"]["theta_max_deg"])  # e.g. 82.9

    distance = config["camera"]["distance"]
    i = np.arange(num_images)

    # uniformly in [cos(theta_max), 1]
    z = 1 - i/(num_images-1)*(1 - np.cos(theta_max))
    thetas = np.arccos(z)                           # elevation array of length N
    phis = (i * phi_g) % (2*np.pi)                  # azimuth array of length N
    rotation_step = phi_g                           # store the same for every frame

    num_train = round(num_images * train_ratio)
    stride    = round(num_images / num_train)       # e.g. if train_ratio=1/3 on N=300 → stride=3

    ntrain = ntest = 0
    train_camera_params = []
    test_camera_params  = []

    for i in range(num_images):
        # pick out this frame's elevation and azimuth
        theta = thetas[i]
        phi = phis[i]

        # spherical --> cartesian
        r = distance
        z_cam = r * np.cos(theta)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)

        camera.location = (x, y, z_cam)

        # point at origin
        direction = target - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()

        # build frame dict & decide train vs test by i % stride
        frame = {
            "file_path": f"train/r_{ntrain}" if i % stride == 0 else f"test/r_{ntest}",
            "rotation": rotation_step,
            "transform_matrix": [list(row) for row in camera.matrix_world],
            "camera_angle_x": camera_angle_x
        }

        if i % stride == 0:
            train_camera_params.append(frame);  ntrain += 1
            out_dir = train_dir;  fname = f"r_{ntrain-1}"
        else:
            test_camera_params.append(frame);   ntest  += 1
            out_dir = test_dir;   fname = f"r_{ntest-1}"

        # render & save
        bpy.context.scene.render.filepath = os.path.join(out_dir, f"{fname}.png")
        bpy.ops.render.render(write_still=True)
    
    # create transform json files
    create_transform_json(config, train_camera_params, "transforms_train.json", folder_name)
    create_transform_json(config, test_camera_params, "transforms_test.json", folder_name)
    
    print(f"finished rendering {num_images} images ({ntrain} train, {ntest} test)")

def create_transform_json(config, camera_params, output_filename, folder_name):
    """create transform.json file in nero dataset format"""
    if not camera_params:
        print(f"warning: no camera parameters for {output_filename}")
        return
        
    # setup output directory
    output_dir = config["output"]["directory"]
    object_dir = os.path.join(output_dir, folder_name)
    
    # create transform data
    transform_data = {
        "camera_angle_x": camera_params[0]["camera_angle_x"],
        "frames": []
    }
    
    # add frames data
    for params in camera_params:
        frame_data = {
            "file_path": params["file_path"],
            "rotation": params["rotation"],
            "transform_matrix": params["transform_matrix"]
        }
        transform_data["frames"].append(frame_data)
    
    # save json file
    transform_json_path = os.path.join(object_dir, output_filename)
    with open(transform_json_path, "w") as f:
        json.dump(transform_data, f, indent=4)
    
    print(f"saved {output_filename} to {transform_json_path}")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "blender_obj.yaml")
    config = load_config(config_path)
    render_obj(config, train_ratio=0.33)
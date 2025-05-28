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
bpy.ops.preferences.addon_enable(module="io_import_scene_obj")
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

def import_object(config):
    # import
    obj_path = config["blender_obj"]["path"]
    ext = os.path.splitext(obj_path)[1].lower()
    
    if ext == ".obj":
        # try different obj import operators
        bpy.ops.wm.obj_import(filepath=obj_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=obj_path)
    else:
        raise Exception(f"Unsupported file format: {ext}")
    
    objs = bpy.context.selected_objects
    
    if not objs:
        raise Exception("No objects were imported")
    
    # assume first imported object
    obj = objs[0]
    obj.scale    = tuple(config["blender_obj"]["scale"])
    obj.location = tuple(config["blender_obj"]["location"])

    # bevel
    bpy.context.view_layer.objects.active = obj
    bevel_mod = obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel_mod.width    = config["blender_obj"]["bevel"]["width"]
    bevel_mod.segments = config["blender_obj"]["bevel"]["segments"]
    bevel_mod.profile  = config["blender_obj"]["bevel"]["profile"]
    bpy.ops.object.modifier_apply(modifier=bevel_mod.name)

    # create a shiny material for the object
    material = bpy.data.materials.new("ShinyMaterial")

    # ─── insert material node setup here ─────────────────────────────────────
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    # clear out the default nodes
    for n in list(nodes):
        nodes.remove(n)
    # add principled BSDF and output
    bsdf   = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    # set your parameters
    # after you've created bsdf = nodes.new(…) …
    # set metallic & roughness as before
    bsdf.inputs["Metallic"].default_value  = config["blender_obj"]["material"]["metallic"]
    bsdf.inputs["Roughness"].default_value = config["blender_obj"]["material"]["roughness"]

    # look up the specular socket by name
    spec_sock = next((sock for sock in bsdf.inputs if sock.name == "Specular"), None)
    if spec_sock is not None:
        spec_sock.default_value = config["blender_obj"]["material"]["specular"]
    else:
        # fallback: maybe they intended the IOR socket?
        ior_sock = next((sock for sock in bsdf.inputs if sock.name == "IOR"), None)
        if ior_sock is not None:
            ior_sock.default_value = config["blender_obj"]["material"]["specular"]
        else:
            print("warning: no Specular or IOR input found on Principled BSDF")

    # ───────────────────────────────────────────────────────────────────────────

    # assign material to the object
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

    return obj

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
    blender_obj = import_object(config)
    setup_environment(config)
    
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.location = (5, 0, 0)
    camera.rotation_euler = (math.pi/2, 0, math.pi/2)
    bpy.context.scene.camera = camera
    
    # calculate camera parameters
    camera_data = camera.data
    camera_angle_x = camera_data.angle_x
    target = Vector(tuple(config["blender_obj"]["location"]))

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
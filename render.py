import bpy
import os
import sys
import numpy as np
from mathutils import Vector

# Blender uses its own Python; install deps into blender_libs so it can find them.
# Run once: python3.11 -m pip install --target /path/to/gs-dataset/blender_libs pyyaml
_script_dir = os.path.dirname(os.path.abspath(__file__))
_blender_libs = os.path.join(_script_dir, "blender_libs")
if os.path.isdir(_blender_libs) and _blender_libs not in sys.path:
    sys.path.insert(0, _blender_libs)
sys.path.append(_script_dir)

from trajectory import get_trajectory
from setup import (
    load_config,
    generate_folder_name,
    setup_scene,
    setup_environment,
    create_transform_json,
    import_single_object,
    import_multiple_objects,
    get_centroid,
)


def setup_normal_pass(config):
    """Enable Normal pass and compositor (main output stays Image for PNG)."""
    scene = bpy.context.scene
    scene.render.use_compositing = True
    bpy.context.view_layer.use_pass_normal = True

    tree = bpy.data.node_groups.new("Compositor", "CompositorNodeTree")
    scene.compositing_node_group = tree
    nodes, links = tree.nodes, tree.links

    rl = nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    group_out = nodes.new("NodeGroupOutput")
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    links.new(rl.outputs["Image"], group_out.inputs["Image"])


def _get_compositor_nodes(comp):
    """Return (RenderLayers node, GroupOutput node) or (None, None)."""
    rl = group_out = None
    for n in comp.nodes:
        if n.type == "R_LAYERS":
            rl = n
        elif n.type == "GROUP_OUTPUT":
            group_out = n
    return rl, group_out


def render_and_save_normal_npy(out_dir, fname):
    """Re-render with Normal pass, save world-space normals as fname_normal.npy (EXR temp to avoid clamping)."""
    scene = bpy.context.scene
    comp = scene.compositing_node_group
    rl, group_out = _get_compositor_nodes(comp) if comp else (None, None)
    if not rl or not group_out:
        return

    out_socket = group_out.inputs["Image"]
    for link in list(out_socket.links):
        comp.links.remove(link)
    comp.links.new(rl.outputs["Normal"], out_socket)

    out_dir_abs = os.path.abspath(out_dir)
    temp_exr = os.path.join(out_dir_abs, f"{fname}_normal_temp.exr")
    orig_format = scene.render.image_settings.file_format
    orig_samples = scene.cycles.samples
    view_layer = bpy.context.view_layer
    orig_alpha = view_layer.pass_alpha_threshold

    try:
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.filepath = temp_exr
        scene.cycles.samples = min(orig_samples, 8)
        view_layer.pass_alpha_threshold = 0.0
        bpy.ops.render.render(write_still=True)
    finally:
        for link in list(out_socket.links):
            comp.links.remove(link)
        comp.links.new(rl.outputs["Image"], out_socket)
        scene.render.image_settings.file_format = orig_format
        scene.cycles.samples = orig_samples
        view_layer.pass_alpha_threshold = orig_alpha

    try:
        loaded = bpy.data.images.load(temp_exr, check_existing=False)
        h, w = loaded.size[1], loaded.size[0]
        pixels = np.array(loaded.pixels, dtype=np.float32).reshape(h, w, -1)
        normals = np.ascontiguousarray(pixels[::-1, :, :3])
        np.save(os.path.join(out_dir, f"{fname}_normal.npy"), normals)
        bpy.data.images.remove(loaded)
        print(f"  Saved normals: {out_dir}/{fname}_normal.npy")
    except Exception as e:
        print(f"  Warning: could not save normals: {e}")
    finally:
        if os.path.exists(temp_exr):
            try:
                os.remove(temp_exr)
            except OSError:
                pass


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
    setup_normal_pass(config)

    if mode == "single":
        objects = [import_single_object(config["blender_obj"])]
        obj = objects[0]
        bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        target = sum(bbox_corners, Vector((0, 0, 0))) / len(bbox_corners)
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
        render_and_save_normal_npy(out_dir, fname)

    create_transform_json(config, train_camera_params, "transforms_train.json", folder_name)
    create_transform_json(config, test_camera_params, "transforms_test.json", folder_name)
    
    print(f"finished rendering {num_images} images ({ntrain} train, {ntest} test)")

if __name__ == "__main__":
    try:
        args = sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        args = [a for a in sys.argv[1:] if not a.startswith("--") and not a.endswith(".py")]
    config_name = (args[0] if args else "single_obj")
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    config_path = os.path.join(_script_dir, "configs", config_name)
    config = load_config(config_path)
    render_scene(config, train_ratio=0.33)

import bpy, sys, os, math
from mathutils import Vector, Matrix
import bpy
from PIL import Image
import numpy as np
import cv2

def getImage(name):
    """
        Processes an image so it fits the model by cutting out the background
    :param name: String name of the image.
    :return: Image array in HSV space.
    """
    # Load image to RGBA
    img = np.array(Image.open(name).convert("RGBA"))
    # Trim image so it perfectly fits the model
    cutD1 = ~(np.sum(img[:, :, 3] != 0, axis = 1) == 0)
    cutD2 = ~(np.sum(img[:, :, 3] != 0, axis = 0) == 0)
    img = img[cutD1,:,:][:,cutD2,:]
    # Convert to HSV
    img = np.array(Image.fromarray(img).convert("HSV"))

    # cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
    # cv2.waitKey(0)

    return img
def orthographicProjections(name):
    def getOrthographicProjection(name, axis, resolution, margin):
        """
            Stores a PNG image of the model (in glb format) in the requested orthographic projection view.
        :param name: String name of the model.
        :param axis: String name and direction of the projection view. <+,-><x,y,z>
        :param resolution: image resolution.
        :param margin: margin.
        :return: Name of the stored image.
        """
        glbPath = rf"Objects\{name}\object\{name}.glb"
        outPng = f"SavedStates\\{name}\\{name}{axis}.png"

        print(glbPath)
        print(outPng)

        # Headless: no UI is launched when using bpy from Python
        # Clean scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        # Ensure the glTF importer is enabled
        try:
            bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
        except Exception:
            pass  # usually already enabled

        # Import GLB
        res = bpy.ops.import_scene.gltf(filepath=os.path.abspath(glbPath))
        print("Import result:", res)

        # List imported meshes
        scene = bpy.context.scene
        meshes = [o for o in scene.objects if o.type == 'MESH']
        if not meshes:
            raise RuntimeError("No mesh objects found after import.")

        # Render settings (Cycles + diffuse color pass, transparent, neutral CM)
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 16
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        # Transparent background
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode  = 'RGBA'
        scene.render.image_settings.color_depth = '8'


        vs = scene.view_settings
        vs.view_transform = 'Standard'
        vs.look = 'None'
        vs.exposure = 0.0
        vs.gamma = 1.0

        layer = scene.view_layers["ViewLayer"]
        layer.use_pass_diffuse_color = True  # gives pure base-color

        scene.use_nodes = True
        scene.render.use_compositing = True
        nt = scene.node_tree
        nt.nodes.clear()
        n_rl = nt.nodes.new("CompositorNodeRLayers")
        n_comp = nt.nodes.new("CompositorNodeComposite")
        n_seta = nt.nodes.new("CompositorNodeSetAlpha")

        # handle Blender label differences: "DiffCol" vs "Diffuse Color"
        diff_out = n_rl.outputs.get("DiffCol") or n_rl.outputs.get("Diffuse Color")
        alpha_out = n_rl.outputs.get("Alpha")
        if diff_out is None:
            raise RuntimeError("Diffuse Color pass not available — check use_pass_diffuse_color.")
        nt.links.new(diff_out, n_seta.inputs["Image"])
        nt.links.new(alpha_out, n_seta.inputs["Alpha"])
        nt.links.new(n_seta.outputs["Image"], n_comp.inputs["Image"])

        # Camera (orthographic) + framing
        cam_data = bpy.data.cameras.new("OrthoCam")
        cam_data.type = 'ORTHO'
        cam = bpy.data.objects.new("OrthoCam", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam


        # world-space bounds
        def world_bounds(objs):
            mins = Vector((1e9, 1e9, 1e9))
            maxs = Vector((-1e9, -1e9, -1e9))
            for o in objs:
                mw = o.matrix_world
                for c in o.bound_box:
                    w = mw @ Vector(c)
                    mins.x = min(mins.x, w.x)
                    mins.y = min(mins.y, w.y)
                    mins.z = min(mins.z, w.z)
                    maxs.x = max(maxs.x, w.x)
                    maxs.y = max(maxs.y, w.y)
                    maxs.z = max(maxs.z, w.z)
            return mins, maxs


        bmin, bmax = world_bounds(meshes)
        center = (bmin + bmax) * 0.5
        extent = bmax - bmin
        diag = max(extent.x, extent.y, extent.z)



        ax = axis.lower()
        if ax in ['+x', '-x']:
            plane_w, plane_h = extent.y, extent.z
            view_dir = Vector((1, 0, 0)) if ax == '+x' else Vector((-1, 0, 0))
            up = Vector((0, 0, 1))
        elif ax in ['+y', '-y']:
            plane_w, plane_h = extent.x, extent.z
            view_dir = Vector((0, 1, 0)) if ax == '+y' else Vector((0, -1, 0))
            up = Vector((0, 0, 1))
        else:
            plane_w, plane_h = extent.x, extent.y
            view_dir = Vector((0, 0, 1)) if ax == '+z' else Vector((0, 0, -1))
            up = Vector((0, 1, 0))

        cam.data.ortho_scale = float(max(plane_w, plane_h) * margin)
        dist = diag * 1.5 + 0.1
        cam.location = center + view_dir * dist
        cam.data.clip_start = 0.01
        cam.data.clip_end = dist * 4.0

        # orient camera to look at center with chosen up
        forward = (center - cam.location).normalized()
        right = forward.cross(up).normalized()
        true_up = right.cross(forward).normalized()
        rot = Matrix((
            (right.x, true_up.x, -forward.x, 0.0),
            (right.y, true_up.y, -forward.y, 0.0),
            (right.z, true_up.z, -forward.z, 0.0),
            (0.0, 0.0, 0.0, 1.0)
        ))
        cam.matrix_world = Matrix.Translation(cam.location) @ rot

        # --- Render and save exactly to out_png
        scene.render.image_settings.file_format = 'PNG'
        # Render to the Composite output (our Diffuse Color)
        bpy.ops.render.render(write_still=False)
        bpy.data.images['Render Result'].save_render(filepath=os.path.abspath(outPng))

        return outPng

    directions = ["+x", "-x", "+y", "-y", "+z", "-z"]
    resolution = 1024
    margin = 1.02

    orthoImages = {}
    imgNames =[]# ['..\\SavedStates\\Bug\\Bug+x.png', '..\\SavedStates\\Bug\\Bug-x.png', '..\\SavedStates\\Bug\\Bug+y.png', '..\\SavedStates\\Bug\\Bug-y.png', '..\\SavedStates\\Bug\\Bug+z.png', '..\\SavedStates\\Bug\\Bug-z.png']
    for direction in directions:
        imgName = getOrthographicProjection(name, direction, resolution, margin)
        imgNames.append(imgName)
        orthoImages[direction] = getImage(imgName)

    for direction,imgName in zip(directions, imgNames):
        orthoImages[direction] = getImage(imgName)

    # for direction in directions:
    #     print(f"Direction: {direction}, dim: {orthoImages[direction].shape}")
    #     cv2.imshow("Image", cv2.cvtColor(orthoImages[direction], cv2.COLOR_HSV2BGR))
    #     cv2.waitKey(0)

    return orthoImages








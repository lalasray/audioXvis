import bpy
import os
import sys

argv = sys.argv[sys.argv.index('--') + 1:]
fbx_path = argv[0]
out_dir = argv[1]
start_frame = int(argv[2])
end_frame = int(argv[3])
object_name = argv[4] if len(argv) > 4 and argv[4] else None

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_path)

scene = bpy.context.scene
mesh_objs = [o for o in scene.objects if o.type == 'MESH']
if object_name:
    selected_objs = [o for o in mesh_objs if o.name == object_name]
    if not selected_objs:
        raise RuntimeError(f'FBX object not found: {object_name}')
else:
    selected_objs = mesh_objs
if not selected_objs:
    raise RuntimeError('No mesh objects found to export from FBX')

for frame in range(start_frame, end_frame + 1):
    scene.frame_set(frame)
    bpy.ops.object.select_all(action='DESELECT')
    for o in selected_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = selected_objs[0]
    out_file = os.path.join(out_dir, f'fbx_frame_{frame:04d}.obj')
    try:
        bpy.ops.wm.obj_export(filepath=out_file, export_selected_objects=True, export_materials=True)
    except Exception:
        bpy.ops.export_scene.obj(filepath=out_file, use_selection=True, use_materials=True)

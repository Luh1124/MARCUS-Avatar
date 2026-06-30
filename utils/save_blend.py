import os
import subprocess
import tempfile
import shutil
import json
import numpy as np
from PIL import Image, ImageOps


BLENDER_EXECUTABLE = os.environ.get("BLENDER_PATH", "blender")

def save_texture(img_or_path, name, texture_size=1024):
    """Save textures to a temporary directory"""
    cache_dir = os.path.abspath("./tmp_cache/blend_export")
    os.makedirs(cache_dir, exist_ok=True)
    
    if img_or_path is None:
        img = Image.new('RGB', (texture_size, texture_size), color=(128, 128, 128))
    elif isinstance(img_or_path, Image.Image):
        img = img_or_path
    else:
        if not os.path.exists(img_or_path):
            print(f"  Warning: texture file does not exist; using placeholder: {img_or_path}")
            img = Image.new('RGB', (texture_size, texture_size), color=(128, 128, 128))
        else:
            img = Image.open(img_or_path)
    
    if img.mode != "RGB" and img.mode != "RGBA":
        img = img.convert("RGB")
    
    img = img.resize((texture_size, texture_size))
    # Flip Y axis for Blender (Blender's UV coordinate system is inverted)
    img = ImageOps.flip(img)
    path = os.path.join(cache_dir, f"{name}.png")
    img.save(path)
    return path

def _save_obj_with_uvs(obj_path, vertices, faces, uvs, face_uvs):
    """Save an OBJ file with UVs"""
    with open(obj_path, 'w') as f:
        f.write("# Exported for Blender\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")
        for i in range(len(faces)):
            f_v = faces[i] + 1
            f_vt = face_uvs[i] + 1
            f.write(f"f {f_v[0]}/{f_vt[0]} {f_v[1]}/{f_vt[1]} {f_v[2]}/{f_vt[2]}\n")

def _create_blender_script(
    script_path,
    obj_file,
    albedo_file, normal_file, roughness_file, specular_file, displacement_file,
    output_blend_path
):
    """Generate the Blender script (fix texture packing)"""
    
    params = {
        "OBJ_FILE": os.path.abspath(obj_file),
        "ALBEDO_FILE": os.path.abspath(albedo_file),
        "NORMAL_FILE": os.path.abspath(normal_file),
        "ROUGHNESS_FILE": os.path.abspath(roughness_file),
        "SPECULAR_FILE": os.path.abspath(specular_file),
        "DISPLACEMENT_FILE": os.path.abspath(displacement_file),
        "OUTPUT_BLEND": os.path.abspath(output_blend_path)
    }
    
    params_code = "\n".join([f'{k} = {json.dumps(v)}' for k, v in params.items()])

    script_content = f"""
import bpy
import os
import sys
import traceback

# ================= Configuration =================
{params_code}
# ===========================================

def load_texture(mat, filepath, name, color_space='sRGB'):
    '''Helper: load a texture node and force packing'''
    if not os.path.exists(filepath):
        print(f"Warning: texture file not found {{filepath}}")
        return None
        
    tex_node = mat.node_tree.nodes.new(type='ShaderNodeTexImage')
    try:
        # 1. Load image
        img = bpy.data.images.load(filepath)
        
        # 2. Core fix: call pack() directly on the image object
        # This is much more reliable than bpy.ops.file.pack_all() in background mode
        if img:
            img.colorspace_settings.name = color_space
            print(f"Packing texture: {{name}}...")
            try:
                img.pack()
                if img.packed_file:
                    print(f"  ✓ Packed successfully (size: {{img.packed_file.size}})")
                else:
                    print(f"  ⚠️ Packing may have failed (no packed_file data)")
            except Exception as pack_err:
                print(f"  ⚠️ Packing failed: {{pack_err}}")
        
        # 3. Assign to node
        tex_node.image = img
        tex_node.name = name
        
    except Exception as e:
        print(f"Failed to load texture {{filepath}}: {{e}}")
        return None
    return tex_node

def main():
    print("=" * 60)
    print("Blender script started...")
    
    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Import OBJ
    print(f"Importing: {{OBJ_FILE}}")
    bpy.ops.wm.obj_import(filepath=OBJ_FILE)
    
    mesh_obj = bpy.context.selected_objects[0]
    mesh_obj.name = "Face_Mesh"
    bpy.context.view_layer.objects.active = mesh_obj
    
    # Create material
    mat = bpy.data.materials.new(name="Face_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    
    bsdf.location = (0, 0)
    output_node = nodes.get("Material Output")
    output_node.location = (300, 0)

    # Load textures (packing logic is included internally)
    albedo_node = load_texture(mat, ALBEDO_FILE, "Albedo", 'sRGB')
    if albedo_node:
        albedo_node.location = (-600, 300)
        links.new(albedo_node.outputs["Color"], bsdf.inputs["Base Color"])
        
    normal_node = load_texture(mat, NORMAL_FILE, "Normal", 'Non-Color')
    if normal_node:
        normal_node.location = (-600, 0)
        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (-300, 0)
        links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    rough_node = load_texture(mat, ROUGHNESS_FILE, "Roughness", 'Non-Color')
    if rough_node:
        rough_node.location = (-600, -300)
        links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])

    spec_node = load_texture(mat, SPECULAR_FILE, "Specular", 'Non-Color')
    if spec_node:
        spec_node.location = (-600, -600)
        target_input = "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
        links.new(spec_node.outputs["Color"], bsdf.inputs[target_input])

    disp_node = load_texture(mat, DISPLACEMENT_FILE, "Displacement", 'Non-Color')
    if disp_node:
        disp_node.location = (-600, -900)
        disp_shader = nodes.new(type='ShaderNodeDisplacement')
        disp_shader.location = (0, -500)
        disp_shader.inputs["Scale"].default_value = 0.01
        
        links.new(disp_node.outputs["Color"], disp_shader.inputs["Height"])
        links.new(disp_shader.outputs["Displacement"], output_node.inputs["Displacement"])

    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = mat
    else:
        mesh_obj.data.materials.append(mat)
        
    # Extra safety: iterate over all images and try packing again
    print("Checking packed status for all images...")
    for img in bpy.data.images:
        if not img.packed_file:
            print(f"  Packing missed image: {{img.name}}")
            try:
                img.pack()
            except:
                pass

    # Save file
    print(f"Saving to: {{OUTPUT_BLEND}}")
    output_dir = os.path.dirname(OUTPUT_BLEND)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # compress=True can reduce size, but the key here is to embed the images
    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND, compress=True)
    print("Saved successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Blender script failed:")
        traceback.print_exc()
        sys.exit(1)
"""
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

def save_blend_file(
    vertices, faces, uvs, face_uvs,
    albedo_path, normal_path, roughness_path, specular_path, displacement_path,
    output_blend_path,
    texture_size=1024,
):
    print(f"Start exporting Blender file: {output_blend_path}")
    
    # 1. Prepare directories
    temp_dir = os.path.abspath("./tmp_cache/blend_export")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 2. Process textures
        tex_files = {}
        for key, path in [
            ("albedo", albedo_path), ("normal", normal_path),
            ("roughness", roughness_path), ("specular", specular_path),
            ("displacement", displacement_path)
        ]:
            tex_files[key] = save_texture(path, key, texture_size)
            
        # 3. Save OBJ
        obj_file = os.path.join(temp_dir, "mesh.obj")
        _save_obj_with_uvs(obj_file, vertices, faces, uvs, face_uvs)
        
        # 4. Generate script
        blend_script = os.path.join(temp_dir, "create_blend.py")
        _create_blender_script(
            blend_script, obj_file, 
            tex_files["albedo"], tex_files["normal"], 
            tex_files["roughness"], tex_files["specular"], 
            tex_files["displacement"], output_blend_path
        )
        
        # 5. Invoke Blender
        blender_candidates = [
            os.environ.get("BLENDER_PATH"),
            BLENDER_EXECUTABLE,
            "blender",
            "/usr/bin/blender",
            "/Applications/Blender.app/Contents/MacOS/Blender"
        ]
        
        blender_cmd = None
        for cmd in blender_candidates:
            if cmd and (shutil.which(cmd) or os.path.exists(cmd)):
                blender_cmd = cmd
                break
                
        if not blender_cmd:
            raise RuntimeError("Blender executable was not found; set the BLENDER_PATH environment variable")
            
        print(f"Using Blender: {blender_cmd}")
        
        cmd_args = [
            blender_cmd,
            "--background",
            "--python", blend_script
        ]
        
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            cwd=temp_dir
        )
        
        if result.returncode != 0:
            print("Blender output (STDOUT):")
            print(result.stdout)
            print("Blender error (STDERR):")
            print(result.stderr)
            raise RuntimeError(f"Blender failed with return code: {result.returncode}")
            
        if not os.path.exists(output_blend_path):
            raise FileNotFoundError(f"Blender finished but did not create the output file: {output_blend_path}")
            
        print(f"✓ Generated Blender file successfully: {output_blend_path} ({os.path.getsize(output_blend_path)/1024/1024:.2f} MB)")
        
    finally:
        pass

# ================= Test section =================

def _parse_obj_file(obj_path):
    """Simple OBJ parser"""
    vertices, uvs, faces, face_uvs = [], [], [], []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith('vt '):
                uvs.append(list(map(float, line.split()[1:3])))
            elif line.startswith('f '):
                f_v, f_vt = [], []
                for part in line.split()[1:]:
                    vals = part.split('/')
                    f_v.append(int(vals[0])-1)
                    if len(vals) > 1 and vals[1]: f_vt.append(int(vals[1])-1)
                    else: f_vt.append(0)
                faces.append(f_v)
                face_uvs.append(f_vt)
    return np.array(vertices), np.array(faces), np.array(uvs), np.array(face_uvs)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    
    if not os.path.exists(outputs_dir):
        print(f"outputs directory not found: {outputs_dir}")
        return

    import glob
    res_dirs = glob.glob(os.path.join(outputs_dir, "res_*"))
    if not res_dirs:
        print("No result directory found")
        return
        
    while True:
        latest_dir = max(res_dirs, key=os.path.getmtime)
        print(f"Latest directory: {latest_dir}")
        if os.path.isdir(latest_dir):
            break
        else:
            res_dirs.remove(latest_dir)
    
    obj_path = os.path.join(latest_dir, "mesh_exp.obj")
    if not os.path.exists(obj_path):
        print("mesh_exp.obj not found")
        return
        
    v, f, uv, f_uv = _parse_obj_file(obj_path)
    
    save_blend_file(
        vertices=v, faces=f, uvs=uv, face_uvs=f_uv,
        albedo_path=os.path.join(latest_dir, "align_res_albedo.png"),
        normal_path=os.path.join(latest_dir, "align_res_normal.png"),
        roughness_path=os.path.join(latest_dir, "align_res_roughness.png"),
        specular_path=os.path.join(latest_dir, "align_res_specular.png"),
        displacement_path=os.path.join(latest_dir, "align_res_displacement.png"),
        output_blend_path=os.path.join(latest_dir, "test_output.blend")
    )

if __name__ == "__main__":
    main()

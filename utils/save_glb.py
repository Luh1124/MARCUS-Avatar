import io
import os
import numpy as np
from PIL import Image

from pygltflib import (
    GLTF2, Asset, Scene, Node, Mesh, Primitive, Attributes,
    Buffer, BufferView, Accessor,
    Material, PbrMetallicRoughness, TextureInfo, NormalMaterialTexture,
    Image as GLTFImage, Texture, Sampler
)

def _png_bytes(img: Image.Image) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()

def _pack_metallic_roughness(roughness_img, metallic_value=0.0):
    # Is this MR or RM?
    # glTF metallicRoughnessTexture: G=roughness, B=metallic, R=unused(usually set to 0)
    # "mr" means G=roughness, B=metallic. glTF uses the same layout.
    r = Image.new("L", roughness_img.size, color=0)
    g = roughness_img.convert("L")
    m = int(np.clip(metallic_value, 0.0, 1.0) * 255)
    b = Image.new("L", roughness_img.size, color=m)
    return Image.merge("RGB", (r, g, b))

def _pack_specular_alpha(specular_gray_img):
    # KHR_materials_specular.specularTexture using the alpha channel is most compatible
    a = specular_gray_img.convert("L")
    one = Image.new("L", specular_gray_img.size, color=255)
    return Image.merge("RGBA", (one, one, one, a))

def _float_minmax(arr):
    arr = arr.astype(np.float32)
    return arr.min(axis=0).tolist(), arr.max(axis=0).tolist()

def compute_vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices, dtype=np.float32)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    fn = np.cross(e1, e2)

    fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_len = np.where(fn_len == 0, 1.0, fn_len)
    fn = fn / fn_len

    for k in range(3):
        np.add.at(normals, faces[:, k], fn)

    n_len = np.linalg.norm(normals, axis=1, keepdims=True)
    n_len = np.where(n_len == 0, 1.0, n_len)
    normals = normals / n_len
    return normals.astype(np.float32)

def save_glb_with_specular(
    vertices, faces, uvs, face_uvs,
    albedo_path, normal_path, roughness_path, specular_path,
    output_glb_path,
    texture_size=1024,
    normal_scale=1.0
):
    """
    Flattened (F*3) export: stable and compatible, following the successful path while truly embedding textures in the GLB.
    """
    print("Start exporting GLB...")

    # ---------- 1) Compute smooth vertex normals ----------
    print("  Computing vertex normals...")
    faces = faces.astype(np.int64)
    face_uvs = face_uvs.astype(np.int64)
    vertices = vertices.astype(np.float32)
    uvs = uvs.astype(np.float32)

    vertex_normals = compute_vertex_normals(vertices, faces)

    # ---------- 2) Flatten geometry: F*3 vertices ----------
    idx_flat_pos = faces.reshape(-1)       # (F*3,)
    idx_flat_uv = face_uvs.reshape(-1)     # (F*3,)

    positions = vertices[idx_flat_pos].astype(np.float32)       # (F*3,3)
    normals = vertex_normals[idx_flat_pos].astype(np.float32)   # (F*3,3)
    texcoords = uvs[idx_flat_uv].astype(np.float32)             # (F*3,2)

    # glTF UV: lower-left origin, flip V
    texcoords[:, 1] = 1.0 - texcoords[:, 1]

    indices = np.arange(positions.shape[0], dtype=np.uint32)    # (F*3,)

    print(f"  Vertices: {positions.shape[0]}, Faces: {faces.shape[0]}")
    print(f"  UV range: U=[{texcoords[:,0].min():.3f}, {texcoords[:,0].max():.3f}], "
          f"V=[{texcoords[:,1].min():.3f}, {texcoords[:,1].max():.3f}]")

    # ---------- 3) Load and pack textures ----------
    print("  Loading and processing textures...")
    albedo_img = Image.open(albedo_path).convert("RGB").resize((texture_size, texture_size)) if not isinstance(albedo_path, Image.Image) else albedo_path
    normal_img = Image.open(normal_path).convert("RGB").resize((texture_size, texture_size)) if not isinstance(normal_path, Image.Image) else normal_path
    rough_img = Image.open(roughness_path).convert("L").resize((texture_size, texture_size)) if not isinstance(roughness_path, Image.Image) else roughness_path
    spec_img = Image.open(specular_path).convert("L").resize((texture_size, texture_size)) if not isinstance(specular_path, Image.Image) else specular_path

    mr_img = _pack_metallic_roughness(rough_img, metallic_value=0.0)  # RGB
    spec_rgba = _pack_specular_alpha(spec_img)                         # RGBA(A=specular)

    albedo_bytes = _png_bytes(albedo_img)
    normal_bytes = _png_bytes(normal_img)
    mr_bytes = _png_bytes(mr_img)
    spec_bytes = _png_bytes(spec_rgba)

    # ---------- 4) Assemble GLB binary buffer (write geometry and PNGs into the buffer) ----------
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0", generator="FFHQ Face Exporter")
    gltf.extensionsUsed = ["KHR_materials_specular"]
    gltf.extensionsRequired = ["KHR_materials_specular"]

    blob = bytearray()
    bufferViews = []
    accessors = []

    def align4():
        pad = (-len(blob)) % 4
        if pad:
            blob.extend(b"\x00" * pad)

    def add_view(data: bytes, target=None) -> int:
        align4()
        off = len(blob)
        blob.extend(data)
        bufferViews.append(BufferView(buffer=0, byteOffset=off, byteLength=len(data), target=target))
        return len(bufferViews) - 1

    def add_accessor(bv, componentType, count, type_str, minv=None, maxv=None) -> int:
        acc = Accessor(bufferView=bv, componentType=componentType, count=count, type=type_str)
        if minv is not None: acc.min = minv
        if maxv is not None: acc.max = maxv
        accessors.append(acc)
        return len(accessors) - 1

    # --- Geometry ---
    bv_pos = add_view(positions.tobytes(), target=34962)
    pos_min, pos_max = _float_minmax(positions)
    acc_pos = add_accessor(bv_pos, 5126, positions.shape[0], "VEC3", pos_min, pos_max)

    bv_nrm = add_view(normals.tobytes(), target=34962)
    nrm_min, nrm_max = _float_minmax(normals)
    acc_nrm = add_accessor(bv_nrm, 5126, normals.shape[0], "VEC3", nrm_min, nrm_max)

    bv_uv = add_view(texcoords.tobytes(), target=34962)
    acc_uv = add_accessor(bv_uv, 5126, texcoords.shape[0], "VEC2")

    bv_idx = add_view(indices.tobytes(), target=34963)
    acc_idx = add_accessor(
        bv_idx, 5125, indices.shape[0], "SCALAR",
        minv=[int(indices.min())], maxv=[int(indices.max())]
    )

    # --- Textures (as image bufferViews) ---
    # Note: image bufferView does not need target
    bv_alb = add_view(albedo_bytes, target=None)
    bv_nmp = add_view(normal_bytes, target=None)
    bv_mr  = add_view(mr_bytes, target=None)
    bv_spc = add_view(spec_bytes, target=None)

    gltf.images = [
        GLTFImage(bufferView=bv_alb, mimeType="image/png"),
        GLTFImage(bufferView=bv_nmp, mimeType="image/png"),
        GLTFImage(bufferView=bv_mr,  mimeType="image/png"),
        GLTFImage(bufferView=bv_spc, mimeType="image/png"),
    ]

    gltf.samplers = [Sampler(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)]
    gltf.textures = [
        Texture(source=0, sampler=0),  # albedo
        Texture(source=1, sampler=0),  # normal
        Texture(source=2, sampler=0),  # metallicRoughness
        Texture(source=3, sampler=0),  # specular(A)
    ]

    # ---------- 5) Material ----------
    pbr = PbrMetallicRoughness(
        baseColorTexture=TextureInfo(index=0),
        metallicRoughnessTexture=TextureInfo(index=2),
        metallicFactor=0.0,
        roughnessFactor=1.0
    )

    # Create Material and use NormalMaterialTexture to support the scale parameter
    material = Material(
        name="HiFi3D_Face_Material",
        pbrMetallicRoughness=pbr,
        normalTexture=NormalMaterialTexture(index=1, scale=float(normal_scale)),
        extensions={
            "KHR_materials_specular": {
                "specularFactor": 1.0,
                "specularTexture": {"index": 3}
            }
        }
    )

    # ---------- 6) Mesh / Scene ----------
    primitive = Primitive(
        attributes=Attributes(POSITION=acc_pos, NORMAL=acc_nrm, TEXCOORD_0=acc_uv),
        indices=acc_idx,
        material=0
    )

    mesh = Mesh(primitives=[primitive], name="HiFi3D_Face")
    node = Node(mesh=0, name="Face")
    scene = Scene(nodes=[0], name="Scene")

    gltf.materials = [material]
    gltf.meshes = [mesh]
    gltf.nodes = [node]
    gltf.scenes = [scene]
    gltf.scene = 0

    gltf.bufferViews = bufferViews
    gltf.accessors = accessors
    gltf.buffers = [Buffer(byteLength=len(blob))]
    gltf.set_binary_blob(bytes(blob))

    # ---------- 7) Save ----------
    os.makedirs(os.path.dirname(output_glb_path) or ".", exist_ok=True)
    gltf.save_binary(output_glb_path)

    print(f"\n{'='*60}")
    print(f"✓ GLB Saved successfully: {output_glb_path}")
    print(f"{'='*60}\n")


def save_glb_white_model(
    vertices, faces, uvs, face_uvs,
    normal_path, roughness_path, specular_path,
    output_glb_path,
    texture_size=1024,
    normal_scale=1.0,
    base_color=(1.0, 1.0, 1.0, 1.0)
):
    """
    Export white-model GLB: basecolor is a solid color while normal, roughness, specular, and other textures are retained
    """
    print("Start exporting white-model GLB...")

    # ---------- 1) Compute smooth vertex normals ----------
    print("  Computing vertex normals...")
    faces = faces.astype(np.int64)
    face_uvs = face_uvs.astype(np.int64)
    vertices = vertices.astype(np.float32)
    uvs = uvs.astype(np.float32)

    vertex_normals = compute_vertex_normals(vertices, faces)

    # ---------- 2) Flatten geometry: F*3 vertices ----------
    idx_flat_pos = faces.reshape(-1)       # (F*3,)
    idx_flat_uv = face_uvs.reshape(-1)     # (F*3,)

    positions = vertices[idx_flat_pos].astype(np.float32)       # (F*3,3)
    normals = vertex_normals[idx_flat_pos].astype(np.float32)   # (F*3,3)
    texcoords = uvs[idx_flat_uv].astype(np.float32)             # (F*3,2)

    # glTF UV: lower-left origin, flip V
    texcoords[:, 1] = 1.0 - texcoords[:, 1]

    indices = np.arange(positions.shape[0], dtype=np.uint32)    # (F*3,)

    print(f"  Vertices: {positions.shape[0]}, Faces: {faces.shape[0]}")

    # ---------- 3) Load and pack textures (do not load albedo; use a solid color) ----------
    print("  Loading and processing textures (white model: basecolor is a solid color)...")
    normal_img = Image.open(normal_path).convert("RGB").resize((texture_size, texture_size)) if not isinstance(normal_path, Image.Image) else normal_path
    rough_img = Image.open(roughness_path).convert("L").resize((texture_size, texture_size)) if not isinstance(roughness_path, Image.Image) else roughness_path
    spec_img = Image.open(specular_path).convert("L").resize((texture_size, texture_size)) if not isinstance(specular_path, Image.Image) else specular_path

    mr_img = _pack_metallic_roughness(rough_img, metallic_value=0.0)  # RGB
    spec_rgba = _pack_specular_alpha(spec_img)                         # RGBA(A=specular)

    normal_bytes = _png_bytes(normal_img)
    mr_bytes = _png_bytes(mr_img)
    spec_bytes = _png_bytes(spec_rgba)

    # ---------- 4) Assemble GLB binary buffer ----------
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0", generator="FFHQ Face Exporter (White Model)")
    gltf.extensionsUsed = ["KHR_materials_specular"]
    gltf.extensionsRequired = ["KHR_materials_specular"]

    blob = bytearray()
    bufferViews = []
    accessors = []

    def align4():
        pad = (-len(blob)) % 4
        if pad:
            blob.extend(b"\x00" * pad)

    def add_view(data: bytes, target=None) -> int:
        align4()
        off = len(blob)
        blob.extend(data)
        bufferViews.append(BufferView(buffer=0, byteOffset=off, byteLength=len(data), target=target))
        return len(bufferViews) - 1

    def add_accessor(bv, componentType, count, type_str, minv=None, maxv=None) -> int:
        acc = Accessor(bufferView=bv, componentType=componentType, count=count, type=type_str)
        if minv is not None: acc.min = minv
        if maxv is not None: acc.max = maxv
        accessors.append(acc)
        return len(accessors) - 1

    # --- Geometry ---
    bv_pos = add_view(positions.tobytes(), target=34962)
    pos_min, pos_max = _float_minmax(positions)
    acc_pos = add_accessor(bv_pos, 5126, positions.shape[0], "VEC3", pos_min, pos_max)

    bv_nrm = add_view(normals.tobytes(), target=34962)
    nrm_min, nrm_max = _float_minmax(normals)
    acc_nrm = add_accessor(bv_nrm, 5126, normals.shape[0], "VEC3", nrm_min, nrm_max)

    bv_uv = add_view(texcoords.tobytes(), target=34962)
    acc_uv = add_accessor(bv_uv, 5126, texcoords.shape[0], "VEC2")

    bv_idx = add_view(indices.tobytes(), target=34963)
    acc_idx = add_accessor(
        bv_idx, 5125, indices.shape[0], "SCALAR",
        minv=[int(indices.min())], maxv=[int(indices.max())]
    )

    # --- Textures (excluding albedo) ---
    bv_nmp = add_view(normal_bytes, target=None)
    bv_mr  = add_view(mr_bytes, target=None)
    bv_spc = add_view(spec_bytes, target=None)

    gltf.images = [
        GLTFImage(bufferView=bv_nmp, mimeType="image/png"),
        GLTFImage(bufferView=bv_mr,  mimeType="image/png"),
        GLTFImage(bufferView=bv_spc, mimeType="image/png"),
    ]

    gltf.samplers = [Sampler(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)]
    gltf.textures = [
        Texture(source=0, sampler=0),  # normal
        Texture(source=1, sampler=0),  # metallicRoughness
        Texture(source=2, sampler=0),  # specular(A)
    ]

    # ---------- 5) Material (basecolor is a solid color without texture) ----------
    pbr = PbrMetallicRoughness(
        baseColorFactor=list(base_color),  # use solid color without texture
        metallicRoughnessTexture=TextureInfo(index=1),
        metallicFactor=0.0,
        roughnessFactor=1.0
    )

    material = Material(
        name="HiFi3D_Face_Material_White",
        pbrMetallicRoughness=pbr,
        normalTexture=NormalMaterialTexture(index=0, scale=float(normal_scale)),
        extensions={
            "KHR_materials_specular": {
                "specularFactor": 1.0,
                "specularTexture": {"index": 2}
            }
        }
    )

    # ---------- 6) Mesh / Scene ----------
    primitive = Primitive(
        attributes=Attributes(POSITION=acc_pos, NORMAL=acc_nrm, TEXCOORD_0=acc_uv),
        indices=acc_idx,
        material=0
    )

    mesh = Mesh(primitives=[primitive], name="HiFi3D_Face_White")
    node = Node(mesh=0, name="Face_White")
    scene = Scene(nodes=[0], name="Scene")

    gltf.materials = [material]
    gltf.meshes = [mesh]
    gltf.nodes = [node]
    gltf.scenes = [scene]
    gltf.scene = 0

    gltf.bufferViews = bufferViews
    gltf.accessors = accessors
    gltf.buffers = [Buffer(byteLength=len(blob))]
    gltf.set_binary_blob(bytes(blob))

    # ---------- 7) Save ----------
    os.makedirs(os.path.dirname(output_glb_path) or ".", exist_ok=True)
    gltf.save_binary(output_glb_path)

    print(f"\n{'='*60}")
    print(f"✓ White-model GLB saved successfully: {output_glb_path}")
    print(f"{'='*60}\n")

"""
Batch inference script
Supports single-image or folder batch processing for the full texture generation pipeline.

Usage:
    # Process a single image
    python batch_infer.py /path/to/image.jpg -o /path/to/output
    
    # Process all images in a folder
    python batch_infer.py /path/to/images/ -o /path/to/output
    
    # Use custom parameters
    python batch_infer.py /path/to/images/ -o /path/to/output \\
        --seed 42 \\
        --lora-scale 0.8 \\
        --align-steps 50 \\
        --use-seq-separate  # Enable Step 3.3 sequence decomposition
    
    # Skip Step 3.3 and do not save GLB or Blender files
    python batch_infer.py /path/to/images/ -o /path/to/output \\
        --no-glb --no-blend

Output structure:
    output/
    ├── image1/
    │   ├── input.jpg
    │   ├── caption.txt
    │   ├── mesh_exp.obj
    │   ├── mesh_neutral.obj
    │   ├── uv_unwrap_raw.png
    │   ├── uv_input_masked.png
    │   ├── seq_step1_inpaint.png
    │   ├── seq_step2_delight.png
    │   ├── align_res_albedo.png
    │   ├── align_res_normal.png
    │   ├── align_res_roughness.png
    │   ├── align_res_specular.png
    │   ├── align_res_displacement.png
    │   ├── model.glb (optional)
    │   └── model.blend (optional)
    └── image2/
        └── ...
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

# Import export helpers
from utils import save_glb_with_specular, save_blend_file

# Import project libraries
sys.path.append(os.getcwd()) 
from utils import read_img, np2tensor, np2pillow
from longcat_image.preprocess import Preprocess_API
from longcat_image.face3d_recon import Face3d_Recon_API
from longcat_image.tex import Tex_API
from diffusers import LongCatImageEditPipeline
from diffusers.models.transformers import LongCatImageTransformer2DModel

from peft import LoraConfig, PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration
from longcat_image.models.pipeline_intrinsix_image_edit import IntrinsiXEditPipeline
from longcat_image.models.batch_lora import inject_trainable_batched_lora
from safetensors.torch import load_file
from longcat_image.models.cross_intrinsic_attention import CrossIntrinsicAttnProcessor2_0
from runtime_paths import (
    BASE_MODEL_PATH,
    JOY_CAPTION_MODEL,
    ckpt_path,
    is_local_model_path,
    topo_path,
)

# ================= Configuration section =================
ALIGN_LORA_PATH = ckpt_path("lora_align/33000.safetensors")
FRONT_FACE_MASK_PATH = topo_path("front_face_mask.png")
ERODE_MASK_PATH = topo_path("minor_valid_front_mask_v3.png")

Front_Face_Mask = Image.open(FRONT_FACE_MASK_PATH).convert("L")

LORA_CONFIGS = {
    "inpainting": ckpt_path("lora_inpainting", "transformer_caption"),
    "delight": ckpt_path("lora_delight", "transformer"),
    "albedo": ckpt_path("lora_albedo", "transformer"),
    "rsd": ckpt_path("lora_rsd", "transformer"),
    "normal": ckpt_path("lora_normal", "transformer")
}

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def validate_runtime_files():
    required_paths = {
        "front face mask": FRONT_FACE_MASK_PATH,
        "erosion mask": ERODE_MASK_PATH,
        "3D landmarks": topo_path("similarity_Lm3D_all.mat"),
        "HiFi3D model info": topo_path("hifi3dpp_model_info.mat"),
        "unwrap info": topo_path("unwrap_1024_info.mat"),
        "unwrap mask": topo_path("unwrap_1024_info_mask.png"),
        "68 landmark detector": ckpt_path("lm_model/68lm_detector.pb"),
        "FaceBox landmark model": ckpt_path("face_box/large_base_net.pth"),
        "FaceBox RetinaFace model": ckpt_path("face_box/retinaface_resnet50_2020-07-20_old_torch.pth"),
        "face parsing model": ckpt_path("parsing_model/dml_csr_celebA.pth"),
        "3D recon model": ckpt_path("deep3d_model/epoch_latest.pth"),
        "super-resolution model": ckpt_path("sr_model/RealESRGAN_x4plus.pth"),
        "align LoRA": ALIGN_LORA_PATH,
    }
    for name, path in LORA_CONFIGS.items():
        required_paths[f"{name} LoRA"] = path
    if is_local_model_path(BASE_MODEL_PATH):
        required_paths["base diffusion model"] = BASE_MODEL_PATH
    if is_local_model_path(JOY_CAPTION_MODEL):
        required_paths["JoyCaption model"] = JOY_CAPTION_MODEL

    missing = [f"{name}: {path}" for name, path in required_paths.items() if not Path(path).exists()]
    if missing:
        message = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "MARCUS-Avatar runtime files are missing:\n"
            f"{message}\n\n"
            "Run `python download_weights.py` to restore ckpts/ and assets/topo/. "
            "Set BASE_MODEL_PATH and JOY_CAPTION_MODEL if those models live elsewhere."
        )


validate_runtime_files()

# Common typo: DIVICES has no effect and still uses physical GPU 0
_wrong_cvd = os.environ.get("CUDA_VISIBLE_DIVICES")
if _wrong_cvd:
    print(
        f"[WARN] Detected incorrect variable CUDA_VISIBLE_DIVICES={_wrong_cvd!r}, "
        "The correct name is CUDA_VISIBLE_DEVICES; GPU masking will not work as expected."
    )
if torch.cuda.is_available():
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    print(
        f"[INFO] CUDA_VISIBLE_DEVICES={_cvd}  ->  "
        f"torch.cuda.device_count()={torch.cuda.device_count()}, DEVICE={DEVICE}"
    )

# ================= Global model initialization =================
print("[INFO] Initializing Global Models...")
# ... (This initialization code remains unchanged) ...
preprocess_model = Preprocess_API(
    lm_detector_path=ckpt_path("lm_model/68lm_detector.pb"),
    mtcnn_path=ckpt_path("face_box/large_base_net.pth"),
    lm68_3d_path=topo_path("similarity_Lm3D_all.mat"),
    parsing_pth=ckpt_path("parsing_model/dml_csr_celebA.pth"),
    target_size=224, rescale_factor=102.0, device=DEVICE,
)

face3d_model = Face3d_Recon_API(
    pfm_model_path=topo_path("hifi3dpp_model_info.mat"),
    recon_model_path=ckpt_path("deep3d_model/epoch_latest.pth"),
    image_super_net_path=ckpt_path("sr_model/RealESRGAN_x4plus.pth"),
    focal=1015.0, camera_distance=10.0, device=DEVICE,
)

tex_model = Tex_API(
    unwrap_info_path=topo_path('unwrap_1024_info.mat'),
    unwrap_info_mask_path=topo_path('unwrap_1024_info_mask.png'),
    unwrap_size=1024,
)

# Load the base pipeline without device argument; move it manually later
pipe = LongCatImageEditPipeline.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
)

if len(LORA_CONFIGS) > 0:
    first_lora_name = list(LORA_CONFIGS.keys())[0]
    first_lora_path = LORA_CONFIGS[first_lora_name]
    print(f"[INFO] Initializing PeftModel with LoRA: {first_lora_name}")
    
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer, 
        first_lora_path, 
        adapter_name=first_lora_name
    )

    # 2. Load additional LoRAs
    for name, path in LORA_CONFIGS.items():
        if name == first_lora_name:
            continue
        print(f"[INFO] Loading additional LoRA: {name}")
        pipe.transformer.load_adapter(path, adapter_name=name)
else:
    print("[WARN] No LoRA configs provided, using base model.")

# Important: ensure the entire pipeline is on the correct device
pipe = pipe.to(DEVICE)
print(f"[INFO] Pipeline moved to {DEVICE}")

# Prepare the transformer for the align pipeline
features = ["albedo", "material", "normal"] # Make sure this matches training config
lora_configs_align = []
for feature in features:
        lora_configs_align.append({"r": 32, "dropout_p": 0.0, "scale": 1.0}) # Use config from training

# Extract the base transformer from pipe.transformer (unload PeftModel)
intrinsix_transformer = LongCatImageTransformer2DModel.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

# Inject the batched LoRA structure
inject_trainable_batched_lora(
        model=intrinsix_transformer,
        target_modules={"to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2"},
        lora_configs=lora_configs_align,
        verbose=False
    )

# Load Align LoRA weights
align_state_dict = load_file(ALIGN_LORA_PATH)
intrinsix_transformer.load_state_dict(align_state_dict, strict=False)

# Important: ensure the transformer is on the correct device
intrinsix_transformer = intrinsix_transformer.to(DEVICE)

# Create the IntrinsiX pipeline
intrinsix_pipeline = IntrinsiXEditPipeline(
    scheduler=pipe.scheduler,
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    text_processor=pipe.text_processor,
    transformer=intrinsix_transformer,
)
crossattn_processor = CrossIntrinsicAttnProcessor2_0()
intrinsix_pipeline.set_attn_processor(crossattn_processor)

# Important: ensure the entire pipeline is on the correct device
intrinsix_pipeline = intrinsix_pipeline.to(DEVICE)
print(f"[INFO] Intrinsix pipeline moved to {DEVICE}")

def apply_mask(image, mask):
    return Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)

class JoyCaptioner:
    # ... (JoyCaptioner code remains unchanged) ...
    def __init__(self, model_name, device):
        print(f"[INFO] Loading JoyCaption: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.device = device
        
        self.uv_prompt_with_lighting = """
        Analyze this facial image focusing on physical attributes and lighting. 
        Ignore the background.
        Provide a concise, comma-separated description covering:
        1. Demographics: Age, Gender, Ethnicity.
        2. Skin Details: Describe texture (pores, wrinkles, moles, scars), skin tone.
        3. Lighting: Direction, Color Tint (e.g. blue/red light), Contrast, Shadows.
        
        Do NOT describe the image composition (e.g. "close-up", "portrait"), just the face itself.
        """
        
        self.uv_prompt_without_lighting = """
        Analyze this facial image focusing on physical attributes only. 
        Ignore the background and lighting conditions.
        Provide a concise, comma-separated description covering:
        1. Demographics: Age, Gender, Ethnicity.
        2. Skin Details: Describe texture (pores, wrinkles, moles, scars, and facial hair), skin tone, facial features.
        
        Do NOT describe:
        - Lighting conditions (direction, color, brightness, shadows, contrast)
        - Image composition (e.g. "close-up", "portrait")
        - Camera angles or exposure settings
        
        Focus only on the inherent physical characteristics of the face.
        """

    def caption(self, image, include_lighting=True):
        if image.mode != "RGB": image = image.convert("RGB")
        prompt = self.uv_prompt_with_lighting if include_lighting else self.uv_prompt_without_lighting
        convo = [{"role": "system", "content": "You are a helpful image captioner."},
                 {"role": "user", "content": prompt}]
        convo_str = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[convo_str], images=[image], return_tensors="pt").to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.6)[0]
        gen_ids = gen_ids[inputs['input_ids'].shape[1]:]
        return self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

joy_captioner = JoyCaptioner(JOY_CAPTION_MODEL, DEVICE)


def save_obj(path, vertices, faces, uvs, face_uvs):
    with open(path, 'w') as f:
        f.write(f"# Exported by LongCat\n")
        for v in vertices: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs: f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")
        for i in range(len(faces)):
            f_v, f_vt = faces[i] + 1, face_uvs[i] + 1
            f.write(f"f {f_v[0]}/{f_vt[0]} {f_v[1]}/{f_vt[1]} {f_v[2]}/{f_vt[2]}\n")

def sanitize_caption(raw_caption):
    blacklist = [
        "mask-like", "black rectangular", "irregular edges", 
        "missing", "cropped", "black background", "no visible texture",
        "artificial", "stylized", "close-up", "portrait of"
    ]
    clean_caption = raw_caption
    for word in blacklist:
        clean_caption = clean_caption.replace(word, "")
    return clean_caption.strip(", ")


def process_single_image(
    input_image_path: str,
    output_dir: str,
    seed: int = 43,
    use_erosion: bool = True,
    erosion_iterations: int = 1,
    erosion_kernel_size: int = 12,
    use_sr: bool = False,
    lora_scale: float = 1.0,
    use_seq_separate: bool = False,
    inpaint_steps: int = 30,
    inpaint_guidance: float = 1.0,
    delight_steps: int = 30,
    delight_guidance: float = 2.0,
    separate_steps: int = 30,
    separate_guidance: float = 2.0,
    align_steps: int = 40,
    save_glb: bool = True,
    save_blend: bool = True,
    save_blend_masked: bool = True,
):
    """
    Full pipeline for processing a single image
    
    Args:
        input_image_path: Input image path
        output_dir: Output directory
        seed: Random seed
        use_erosion: Whether to use erosion
        erosion_iterations: Erosion iteration count
        erosion_kernel_size: Erode kernel size
        use_sr: Whether to use super-resolution (default False)
        lora_scale: LoRA strength
        use_seq_separate: Whether to run Step 3.3 sequence decomposition (default False, use align results directly)
        inpaint_steps: Inpaint sampling steps
        inpaint_guidance: Inpaint guidance scale
        delight_steps: Delight sampling steps
        delight_guidance: Delight guidance scale
        separate_steps: Step 3.3 Separate Channels sampling steps
        separate_guidance: Step 3.3 Separate Channels guidance scale
        align_steps: Align sampling steps
        save_glb: Whether to save GLB files
        save_blend: Whether to save Blender files
        save_blend_masked: Whether to save masked Blender files
    """
    print(f"\n{'='*60}")
    print(f"[INFO] Processing: {input_image_path}")
    print(f"{'='*60}")
    
    # Create output directory
    image_name = Path(input_image_path).stem
    save_dir = os.path.join(output_dir, image_name)
    
    # Skip if the output directory exists and contains key files
    key_file = os.path.join(save_dir, "align_res_albedo.png")
    if os.path.exists(key_file):
        print(f"[SKIP] Already processed: {image_name}")
        print(f"  Output directory: {save_dir}")
        print(f"{'='*60}\n")
        return True
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # ================= Step 1: Preprocess =================
        print("\n[Step 1] 3D Reconstruction & Texture Unwrap...")
        input_img = Image.open(input_image_path).convert("RGB")
        input_path = os.path.join(save_dir, "input.jpg")
        input_img.save(input_path)
        
        # Preprocess
        align_img, hr_img, trans_params, lm68_2d, seg_mask, skin_mask = preprocess_model(np.array(input_img))
        coeffs = face3d_model.pred_coeffs(np2tensor(align_img, device=DEVICE))
        
        hr_img_pil = np2pillow(hr_img)
        hr_img_pil.save(os.path.join(save_dir, "hr_align_img.png"))

        # Geometry
        face_shape, neutral_shape = face3d_model.facemodel.compute_shape(coeffs['id'], coeffs['exp'])
        rotation = face3d_model.facemodel.compute_rotation(coeffs['angle'])
        face_shape_t = face3d_model.facemodel.transform(face_shape, rotation, coeffs['trans'])
        
        # Extract Mesh Data
        mesh_data = {
            "vertices": face_shape_t[0].cpu().numpy(),
            "neutral_vertices": neutral_shape[0].cpu().numpy(),
            "faces": face3d_model.facemodel.head_buf.cpu().numpy(),
            "uvs": face3d_model.facemodel.vt_list.cpu().numpy(),
            "face_uvs": face3d_model.facemodel.head_tri_vt.cpu().numpy()
        }
        
        # Save OBJs
        save_obj(os.path.join(save_dir, "mesh_exp.obj"), mesh_data["vertices"], mesh_data["faces"], mesh_data["uvs"], mesh_data["face_uvs"])
        save_obj(os.path.join(save_dir, "mesh_neutral.obj"), mesh_data["neutral_vertices"], mesh_data["faces"], mesh_data["uvs"], mesh_data["face_uvs"])
        
        # Unwrap
        projXY, norm = face3d_model.compute_224projXY_norm_by_pin_hole(coeffs)
        projXY, norm = projXY[0].cpu().numpy(), norm[0].cpu().numpy()
        projXY = preprocess_model.trans_projXY_back_to_ori_coord(projXY, trans_params)
        
        unwrap_tex, _ = tex_model(np.array(input_img), seg_mask, projXY, norm)
        unwrap_pil = np2pillow(unwrap_tex)
        unwrap_pil.save(os.path.join(save_dir, "uv_unwrap_raw.png"))
        
        # Masking
        ref_cv = cv2.cvtColor(np.array(unwrap_pil), cv2.COLOR_RGB2BGR)
        _, mask = cv2.threshold(cv2.cvtColor(ref_cv, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

        if use_erosion:
            Erosion_Mask_np = cv2.imread(ERODE_MASK_PATH, cv2.IMREAD_GRAYSCALE)

            if Erosion_Mask_np.shape != mask.shape:
                Erosion_Mask_np = cv2.resize(Erosion_Mask_np, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.where((mask == 255) | (Erosion_Mask_np == 255), 255, 0).astype(np.uint8)

            kernel_size = int(erosion_kernel_size)
            iterations = int(erosion_iterations)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
            ref_eroded = np.zeros_like(ref_cv)
            ref_eroded[eroded_mask == 255] = ref_cv[eroded_mask == 255]
            ref_rgb = cv2.cvtColor(ref_eroded, cv2.COLOR_BGR2RGB)
        else:
            ref_eroded = np.zeros_like(ref_cv)
            ref_eroded[mask == 255] = ref_cv[mask == 255]
            ref_rgb = cv2.cvtColor(ref_eroded, cv2.COLOR_BGR2RGB)
        
        if ref_rgb.dtype != np.uint8:
            ref_rgb = np.clip(ref_rgb, 0, 255).astype(np.uint8)
        ref_pil_masked = Image.fromarray(ref_rgb)  # Eroded image; if erosion is disabled, this is the original unwrap
        
        # Save the image before super-resolution
        ref_pil_masked.save(os.path.join(save_dir, "uv_input_masked_before_sr.png"))
        
        # Super-resolution processing (optional)
        if use_sr:
            print("[Step 1] Using super-resolution...")
            try:
                ref_pil_sr = face3d_model.sr_model(ref_pil_masked)
                # Save the original-size image after super-resolution
                ref_pil_sr.save(os.path.join(save_dir, "uv_input_masked_after_sr.png"))
                # Resize to 1024x1024
                ref_pil = ref_pil_sr.resize((1024, 1024))
            except Exception as e:
                print(f"[WARNING] Super-resolution failed: {e}. Falling back to resize.")
                # If super-resolution fails, resize directly
                ref_pil = ref_pil_masked.resize((1024, 1024))
        else:
            print("[Step 1] Skipping super-resolution, using direct resize.")
            # Resize directly to 1024x1024
            ref_pil = ref_pil_masked.resize((1024, 1024))
            # Save a file marked as post-SR for consistency, using 4x resize from the original masked image
            ref_pil_4x = ref_pil_masked.resize((ref_pil_masked.width * 4, ref_pil_masked.height * 4))
            ref_pil_4x.save(os.path.join(save_dir, "uv_input_masked_after_sr.png"))
        
        # Save the final image used
        ref_pil.save(os.path.join(save_dir, "uv_input_masked.png"))
        
        print("[Step 1] Done.")
        
        # ================= Step 2: Caption & Prompt Construction =================
        print("\n[Step 2] Auto Captioning & Prompt Construction...")
        raw_caption_with_lighting = joy_captioner.caption(input_img, include_lighting=True)
        caption_with_lighting = sanitize_caption(raw_caption_with_lighting)
        
        raw_caption_without_lighting = joy_captioner.caption(input_img, include_lighting=False)
        caption_without_lighting = sanitize_caption(raw_caption_without_lighting)
        
        # Build prompts
        semantic_header_inpaint = f"An unfolded UV texture map of {caption_with_lighting}"
        semantic_header_others = f"An unfolded UV texture map of {caption_without_lighting}"
        
        prompt_inpaint = f"{semantic_header_inpaint}, seamless, continuous texture, high fidelity, 8k resolution. Synthesize high-frequency micro-details, pixel-perfect continuity."
        
        base_delight_def = "Normalized lighting texture map, calibrated neutral illumination. Uniform pixel intensity distribution across the entire facial surface. Eliminate all lighting gradients, environmental bias, and directional shading. Perfectly balanced exposure, strictly retaining high-frequency micro-details, razor-sharp pores, and authentic skin grain. High-fidelity raw texture quality. Raw, 8k, highly detailed, macro photography, hard focus. Remove all lighting, shadows, and shading. Generate flat, unlit base color texture."
        prompt_delight = f"{semantic_header_others}. {base_delight_def}"
        
        base_albedo_def = "Ultra-High Definition 8K Diffuse Albedo map, flat lighting, unlit base color, pixel-perfect clarity. The texture reveals natural melanin pigmentation, hemoglobin redness, and distinct subsurface scattering warmth zones. The chart is completely void of baked shadows, ambient occlusion, or specular highlights, representing pure biological skin color values in UV space."
        prompt_align_albedo = f"{semantic_header_others}. {base_albedo_def}"
        
        base_normal_def = "Ultra-High Definition 8K Surface Normal map in UV space, displaying sharp high-frequency relief with pixel-perfect clarity. The texture emphasizes intricate pore structures, fine wrinkles, and precise skin micro-geometry orientation relative to UV coordinates."
        prompt_align_normal = f"{semantic_header_others}. {base_normal_def}"
        
        base_rsd_def = "Ultra-High Definition 8K packed RSD technical texture map in UV space, consisting of three specific PBR channels. Red: Roughness/Oiliness; Green: Specular Intensity; Blue: Displacement Depth. The map captures purely physical skin surface properties."
        prompt_align_rsd = f"{semantic_header_others}. {base_rsd_def}"
        
        # Save caption
        with open(os.path.join(save_dir, "caption.txt"), "w") as f:
            f.write(f"Caption (with lighting): {caption_with_lighting}\n")
            f.write(f"Caption (without lighting): {caption_without_lighting}\n")
        
        print("[Step 2] Done.")
        
        # ================= Step 3.1: Inpainting =================
        print("\n[Step 3.1] Inpainting...")
        # Use global pipe (already loaded with all LoRAs in app.py)
        
        # Verify and set adapter
        if "inpainting" not in pipe.transformer.peft_config:
            raise RuntimeError(f"LoRA adapter 'inpainting' not loaded. Available adapters: {list(pipe.transformer.peft_config.keys())}")
        pipe.transformer.set_adapter("inpainting")
        print(f"[INFO] Using LoRA adapter: inpainting (scale={lora_scale})")
        
        # Set LoRA scale by modifying peft_config
        if lora_scale != 1.0 and "inpainting" in pipe.transformer.peft_config:
            original_alpha = pipe.transformer.peft_config["inpainting"].lora_alpha
            new_alpha = int(original_alpha * lora_scale)
            pipe.transformer.peft_config["inpainting"].lora_alpha = new_alpha
            print(f"  Adjusted lora_alpha: {original_alpha} -> {new_alpha}")
        
        negative_prompt_inpaint = "blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts"
        generator = torch.Generator(DEVICE).manual_seed(seed)
        
        inpaint_res = pipe(
            ref_pil, prompt_inpaint, negative_prompt=negative_prompt_inpaint,
            num_inference_steps=int(inpaint_steps), generator=generator,
            guidance_scale=inpaint_guidance
        ).images[0]
        inpaint_res.save(os.path.join(save_dir, "seq_step1_inpaint.png"))
        print("[Step 3.1] Done.")
        
        # ================= Step 3.2: Delight =================
        print("\n[Step 3.2] Delight...")
        
        # Verify and set adapter
        if "delight" not in pipe.transformer.peft_config:
            raise RuntimeError(f"LoRA adapter 'delight' not loaded. Available adapters: {list(pipe.transformer.peft_config.keys())}")
        pipe.transformer.set_adapter("delight")
        print(f"[INFO] Using LoRA adapter: delight (scale={lora_scale})")
        
        # Set LoRA scale by modifying peft_config
        if lora_scale != 1.0 and "delight" in pipe.transformer.peft_config:
            original_alpha = pipe.transformer.peft_config["delight"].lora_alpha
            new_alpha = int(original_alpha * lora_scale)
            pipe.transformer.peft_config["delight"].lora_alpha = new_alpha
            print(f"  Adjusted lora_alpha: {original_alpha} -> {new_alpha}")
        
        negative_prompt_delight = "blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts"
        generator = torch.Generator(DEVICE).manual_seed(seed)
        
        delight_res = pipe(
            inpaint_res, prompt_delight, negative_prompt=negative_prompt_delight,
            guidance_scale=delight_guidance, num_inference_steps=int(delight_steps), generator=generator
        ).images[0]
        delight_res.save(os.path.join(save_dir, "seq_step2_delight.png"))
        print("[Step 3.2] Done.")
        
        # ================= Step 3.3: Separate Channels (Optional) =================
        if use_seq_separate:
            print("\n[Step 3.3] Separate Channels...")
            negative_prompt_sep = "blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts"
            
            for mode in ["albedo", "normal", "rsd"]:
                if mode == "albedo":
                    prompt_sep = "Ultra-High Definition 8K Diffuse Albedo map, flat lighting, unlit base color."
                elif mode == "normal":
                    prompt_sep = "Ultra-High Definition 8K Surface Normal map, displaying sharp high-frequency relief."
                else:  # rsd
                    prompt_sep = "Ultra-High Definition 8K packed RSD technical texture map (Roughness, Specular, Displacement)."
                
                # Verify and set adapter
                if mode not in pipe.transformer.peft_config:
                    raise RuntimeError(f"LoRA adapter '{mode}' not loaded. Available adapters: {list(pipe.transformer.peft_config.keys())}")
                pipe.transformer.set_adapter(mode)
                print(f"[INFO] Using LoRA adapter: {mode} (scale={lora_scale})")
                
                # Set LoRA scale by modifying peft_config
                if lora_scale != 1.0 and mode in pipe.transformer.peft_config:
                    original_alpha = pipe.transformer.peft_config[mode].lora_alpha
                    new_alpha = int(original_alpha * lora_scale)
                    pipe.transformer.peft_config[mode].lora_alpha = new_alpha
                    print(f"  Adjusted lora_alpha: {original_alpha} -> {new_alpha}")
                
                generator = torch.Generator(DEVICE).manual_seed(seed)
                
                res = pipe(
                    delight_res, prompt_sep, negative_prompt=negative_prompt_sep,
                    guidance_scale=separate_guidance, num_inference_steps=int(separate_steps), generator=generator
                ).images[0]
                res.save(os.path.join(save_dir, f"seq_res_{mode}.png"))
                print(f"  - {mode} saved")
            print("[Step 3.3] Done.")
        else:
            print("\n[Step 3.3] Skipped (using align results directly)")
        
        # ================= Step 4: Align / IntrinsiX Generation =================
        print("\n[Step 4] IntrinsiX Align Generation...")
        
        # Use the global intrinsix_pipeline initialized in app.py
        negative_prompt_align = "blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts, low resolution, baked lighting removal, cartoonish, loss of detail, interpolated pixels"
        generator = torch.Generator(DEVICE).manual_seed(seed)
        
        res_list = intrinsix_pipeline(
            delight_res,
            prompt_albedo=prompt_align_albedo,
            prompt_normal=prompt_align_normal,
            prompt_rsd=prompt_align_rsd,
            negative_prompt_albedo=negative_prompt_align,
            negative_prompt_normal=negative_prompt_align,
            negative_prompt_rsd=negative_prompt_align,
            guidance_scale=2.0,
            num_inference_steps=int(align_steps),
            generator=generator
        ).images
        
        align_albedo, align_normal, align_rsd = res_list
        align_albedo.save(os.path.join(save_dir, "align_res_albedo.png"))
        align_normal.save(os.path.join(save_dir, "align_res_normal.png"))
        
        align_roughness = align_rsd.getchannel(0)
        align_specular = align_rsd.getchannel(1)
        align_displacement = align_rsd.getchannel(2)
        align_roughness.save(os.path.join(save_dir, "align_res_roughness.png"))
        align_specular.save(os.path.join(save_dir, "align_res_specular.png"))
        align_displacement.save(os.path.join(save_dir, "align_res_displacement.png"))
        
        align_albedo_masked = apply_mask(align_albedo, Front_Face_Mask)
        align_normal_masked = apply_mask(align_normal, Front_Face_Mask)
        align_roughness_masked = apply_mask(align_roughness, Front_Face_Mask)
        align_specular_masked = apply_mask(align_specular, Front_Face_Mask)
        align_displacement_masked = apply_mask(align_displacement, Front_Face_Mask)
        
        align_albedo_masked.save(os.path.join(save_dir, "align_res_albedo_masked.png"))
        align_normal_masked.save(os.path.join(save_dir, "align_res_normal_masked.png"))
        align_roughness_masked.save(os.path.join(save_dir, "align_res_roughness_masked.png"))
        align_specular_masked.save(os.path.join(save_dir, "align_res_specular_masked.png"))
        align_displacement_masked.save(os.path.join(save_dir, "align_res_displacement_masked.png"))
        
        print("[Step 4] Done.")
        
        # ================= Save 3D Files =================
        if save_glb:
            print("\n[Saving] GLB file...")
            glb_path = os.path.join(save_dir, "model.glb")
            try:
                save_glb_with_specular(
                    vertices=mesh_data["vertices"],
                    faces=mesh_data["faces"],
                    uvs=mesh_data["uvs"],
                    face_uvs=mesh_data["face_uvs"],
                    albedo_path=align_albedo,
                    normal_path=align_normal,
                    roughness_path=align_roughness,
                    specular_path=align_specular,
                    output_glb_path=glb_path
                )
                print(f"  - GLB saved: {glb_path}")
            except Exception as e:
                print(f"  - GLB Error: {e}")
        
        if save_blend:
            print("\n[Saving] Blender file...")
            blend_path = os.path.join(save_dir, "model.blend")
            try:
                save_blend_file(
                    vertices=mesh_data["vertices"],
                    faces=mesh_data["faces"],
                    uvs=mesh_data["uvs"],
                    face_uvs=mesh_data["face_uvs"],
                    albedo_path=align_albedo,
                    normal_path=align_normal,
                    roughness_path=align_roughness,
                    specular_path=align_specular,
                    displacement_path=align_displacement,
                    output_blend_path=blend_path
                )
                print(f"  - Blender file saved: {blend_path}")
            except Exception as e:
                print(f"  - Blender Error: {e}")
        
        if save_blend_masked:
            print("\n[Saving] Masked Blender file...")
            blend_path_masked = os.path.join(save_dir, "model_masked.blend")
            try:
                save_blend_file(
                    vertices=mesh_data["vertices"],
                    faces=mesh_data["faces"],
                    uvs=mesh_data["uvs"],
                    face_uvs=mesh_data["face_uvs"],
                    albedo_path=align_albedo_masked,
                    normal_path=align_normal_masked,
                    roughness_path=align_roughness_masked,
                    specular_path=align_specular_masked,
                    displacement_path=align_displacement_masked,
                    output_blend_path=blend_path_masked
                )
                print(f"  - Masked Blender file saved: {blend_path_masked}")
            except Exception as e:
                print(f"  - Masked Blender Error: {e}")
        
        print(f"\n{'='*60}")
        print(f"[SUCCESS] Completed: {image_name}")
        print(f"  Output directory: {save_dir}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing {input_image_path}: {str(e)}\n{traceback.format_exc()}"
        print(f"\n{'='*60}")
        print(f"[ERROR] Failed: {image_name}")
        print(f"{'='*60}")
        print(error_msg)
        
        # Save error log
        error_log_path = os.path.join(save_dir, "error.log")
        with open(error_log_path, "w") as f:
            f.write(error_msg)
        
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch texture generation inference script")
    parser.add_argument(
        "input",
        type=str,
        help="Input image path or folder path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output folder path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed (default: 43)"
    )
    parser.add_argument(
        "--use-erosion",
        action="store_true",
        default=True,
        help="Use erosion (default: True)"
    )
    parser.add_argument(
        "--no-erosion",
        action="store_false",
        dest="use_erosion",
        help="Do not use erosion"
    )
    parser.add_argument(
        "--erosion-iterations",
        type=int,
        default=1,
        help="Erosion iteration count (default: 1)"
    )
    parser.add_argument(
        "--erosion-kernel-size",
        type=int,
        default=12,
        help="Erode kernel size (default: 12)"
    )
    parser.add_argument(
        "--use-sr",
        action="store_true",
        help="Use super-resolution (default: False)"
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="LoRA strength (default: 1.0)"
    )
    parser.add_argument(
        "--use-seq-separate",
        action="store_true",
        help="Run Step 3.3 sequence decomposition; skipped by default, using align results directly"
    )
    parser.add_argument(
        "--inpaint-steps",
        type=int,
        default=30,
        help="Inpaint sampling steps (default: 30)"
    )
    parser.add_argument(
        "--inpaint-guidance",
        type=float,
        default=1.0,
        help="Inpaint guidance scale (default: 1.0)"
    )
    parser.add_argument(
        "--delight-steps",
        type=int,
        default=30,
        help="Delight sampling steps (default: 30)"
    )
    parser.add_argument(
        "--delight-guidance",
        type=float,
        default=2.0,
        help="Delight guidance scale (default: 2.0)"
    )
    parser.add_argument(
        "--separate-steps",
        type=int,
        default=30,
        help="Step 3.3 Separate Channels sampling steps (default: 30)"
    )
    parser.add_argument(
        "--separate-guidance",
        type=float,
        default=2.0,
        help="Step 3.3 Separate Channels guidance scale (default: 2.0)"
    )
    parser.add_argument(
        "--align-steps",
        type=int,
        default=40,
        help="Align sampling steps (default: 40)"
    )
    parser.add_argument(
        "--no-glb",
        action="store_true",
        help="Do not save GLB files"
    )
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="Do not save Blender files"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="Supported image extensions (default: .jpg .jpeg .png .bmp)"
    )
    
    args = parser.parse_args()
    
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Collect input images
    input_paths = []
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        if input_path.suffix.lower() in [ext.lower() for ext in args.extensions]:
            input_paths.append(str(input_path))
        else:
            print(f"[ERROR] Unsupported file format: {input_path.suffix}")
            sys.exit(1)
    elif input_path.is_dir():
        # Folder
        # Only check jpg, webp, and png suffixes
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        input_paths = []
        for file in input_path.iterdir():
            if file.is_file() and file.suffix.lower() in valid_exts:
                input_paths.append(str(file))
        input_paths = sorted(input_paths)
    else:
        print(f"[ERROR] Input path does not exist: {args.input}")
        sys.exit(1)
    
    if len(input_paths) == 0:
        print(f"[ERROR] No images found in: {args.input}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"[INFO] Batch Inference")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Total images: {len(input_paths)}")
    print(f"Use seq separate: {args.use_seq_separate}")
    print(f"{'='*60}\n")
    
    # Process each image
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    for idx, img_path in enumerate(input_paths, 1):
        print(f"\n[{idx}/{len(input_paths)}] Processing...")
        
        success = process_single_image(
            input_image_path=img_path,
            output_dir=args.output,
            seed=args.seed,
            use_erosion=args.use_erosion,
            erosion_iterations=args.erosion_iterations,
            erosion_kernel_size=args.erosion_kernel_size,
            use_sr=args.use_sr,
            lora_scale=args.lora_scale,
            use_seq_separate=args.use_seq_separate,
            inpaint_steps=args.inpaint_steps,
            inpaint_guidance=args.inpaint_guidance,
            delight_steps=args.delight_steps,
            delight_guidance=args.delight_guidance,
            separate_steps=args.separate_steps,
            separate_guidance=args.separate_guidance,
            align_steps=args.align_steps,
            save_glb=not args.no_glb,
            save_blend=not args.no_blend,
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[SUMMARY]")
    print(f"{'='*60}")
    print(f"Total: {len(input_paths)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Time: {elapsed_time:.2f}s ({elapsed_time/len(input_paths):.2f}s per image)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

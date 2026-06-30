import os
import sys

# Set this before importing torch or any library that initializes CUDA; otherwise it has no effect.
# If CUDA_VISIBLE_DEVICES was already exported before launch, do not override it here.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "cache", "matplotlib")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import time
import numpy as np
import torch
import cv2
import gradio as gr
from styles import FACE_STUDIO_CSS, face_studio_theme
import zipfile
from PIL import Image
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from utils import save_glb_with_specular, save_glb_white_model, save_blend_file

sys.path.append(os.getcwd()) 
from utils import read_img, np2tensor, np2pillow
from longcat_image.preprocess import Preprocess_API
from longcat_image.face3d_recon import Face3d_Recon_API
from longcat_image.tex import Tex_API
from longcat_image.models.pipeline_longcat_inpainting import LongCatImageInpaintingPipeline
from longcat_image.models import LongCatImageTransformer2DModel

from peft import LoraConfig, PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration
from longcat_image.models.pipeline_intrinsix_image_edit import IntrinsiXEditPipeline
from longcat_image.models.batch_lora import inject_trainable_batched_lora
from safetensors.torch import load_file
from longcat_image.models.cross_intrinsic_attention import CrossIntrinsicAttnProcessor2_0
from runtime_paths import (
    BASE_MODEL_PATH,
    JOY_CAPTION_MODEL,
    TOPO_DIR,
    ckpt_path,
    is_local_model_path,
    topo_path,
)

# ================= config section =================
ALIGN_LORA_PATH = ckpt_path("lora_align/27000_0120.safetensors")

DALIGN_LORA_PATH = ckpt_path("lora_dalign/33000.safetensors")

FRONT_FACE_MASK_PATH = topo_path("front_face_mask.png")
ERODE_MASK_PATH = topo_path("minor_valid_front_mask_v3.png")

LORA_CONFIGS = {
    "inpainting_ori": ckpt_path("lora_inpainting", "transformer_caption_ori"),
    "inpainting": ckpt_path("lora_inpainting", "transformer_caption"),
    "delight": ckpt_path("lora_delight", "transformer"),
    "albedo": ckpt_path("lora_albedo", "transformer"),
    "rsd": ckpt_path("lora_rsd", "transformer"),
    "normal": ckpt_path("lora_normal", "transformer")
}

# ================= static file service config section =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = "outputs"
USE_STATIC_SERVER = True
HTML_HEIGHT = 650
HTML_WIDTH = 790


def validate_runtime_files():
    required_paths = {
        "front face mask": FRONT_FACE_MASK_PATH,
        "erosion mask": ERODE_MASK_PATH,
        "3D landmarks": topo_path("similarity_Lm3D_all.mat"),
        "HiFi3D model info": topo_path("hifi3dpp_model_info.mat"),
        "unwrap info": topo_path("unwrap_1024_info.mat"),
        "unwrap mask": topo_path("unwrap_1024_info_mask.png"),
        "modelviewer template": topo_path("modelviewer-template.html"),
        "modelviewer textured template": topo_path("modelviewer-textured-template.html"),
        "environment map gradient": topo_path("env_maps/gradient.jpg"),
        "environment map white": topo_path("env_maps/white.jpg"),
        "68 landmark detector": ckpt_path("lm_model/68lm_detector.pb"),
        "FaceBox landmark model": ckpt_path("face_box/large_base_net.pth"),
        "FaceBox RetinaFace model": ckpt_path("face_box/retinaface_resnet50_2020-07-20_old_torch.pth"),
        "face parsing model": ckpt_path("parsing_model/dml_csr_celebA.pth"),
        "3D recon model": ckpt_path("deep3d_model/epoch_latest.pth"),
        "super-resolution model": ckpt_path("sr_model/RealESRGAN_x4plus.pth"),
        "align LoRA": ALIGN_LORA_PATH,
        "DAlign LoRA": DALIGN_LORA_PATH,
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
Front_Face_Mask = Image.open(FRONT_FACE_MASK_PATH).convert("L")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ================= global model initialization section =================
print("[INFO] Initializing Global Models...")
preprocess_model = Preprocess_API(
    lm_detector_path=ckpt_path("lm_model/68lm_detector.pb"),
    mtcnn_path=ckpt_path("face_box/large_base_net.pth"),
    lm68_3d_path=topo_path("similarity_Lm3D_all.mat"),
    parsing_pth=ckpt_path("parsing_model/dml_csr_celebA.pth"),
    target_size=224, rescale_factor=102.0, device=DEVICE,
)

face3d_model = Face3d_Recon_API(
    pfm_model_path=topo_path("hifi3dpp_model_info.mat"),
    # recon_model_path=ckpt_path("deep3d_merge_model/epoch_latest.pth"),
    recon_model_path=ckpt_path("deep3d_model/epoch_latest.pth"),
    image_super_net_path=ckpt_path("sr_model/RealESRGAN_x4plus.pth"),
    focal=1015.0, camera_distance=10.0, device=DEVICE, use_merge_model=False,
)

tex_model = Tex_API(
    unwrap_info_path=topo_path('unwrap_1024_info.mat'),
    unwrap_info_mask_path=topo_path('unwrap_1024_info_mask.png'),
    unwrap_size=1024,
)

# load base pipeline
pipe = LongCatImageInpaintingPipeline.from_pretrained(
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

# ensure the pipeline is on the correct device
pipe = pipe.to(DEVICE)
print(f"[INFO] Pipeline moved to {DEVICE}")

# prepare align pipeline transformer
features = ["albedo", "material", "normal"] # Make sure this matches training config
lora_configs_align = []
for feature in features:
        lora_configs_align.append({"r": 32, "dropout_p": 0.0, "scale": 1.0}) # Use config from training

# extract base transformer from pipe.transformer (unload PeftModel)
intrinsix_transformer = LongCatImageTransformer2DModel.from_pretrained(
    BASE_MODEL_PATH,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

# inject Batched LoRA structure
inject_trainable_batched_lora(
        model=intrinsix_transformer,
        target_modules={"to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2"},
        lora_configs=lora_configs_align,
        verbose=False
    )

# load Align LoRA weights
align_state_dict = load_file(ALIGN_LORA_PATH)

dalign_state_dict = load_file(DALIGN_LORA_PATH)

intrinsix_transformer.load_state_dict(align_state_dict, strict=False)

# ensure the transformer is on the correct device
intrinsix_transformer = intrinsix_transformer.to(DEVICE)

# create IntrinsiX pipeline
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

# ensure the pipeline is on the correct device
intrinsix_pipeline = intrinsix_pipeline.to(DEVICE)
print(f"[INFO] Intrinsix pipeline moved to {DEVICE}")


def apply_mask(image, mask):
    return Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)

class JoyCaptioner:
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
        2. Skin Details: Describe texture (pores, wrinkles, moles, scars, facial hairs), skin tone.
        4. Lighting: Direction, Color Tint (e.g. blue/red light), Contrast, Shadows.
        
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


# ================= Step Functions (new set_adapter_safe) =================

def step1_preprocess(input_img, seed, use_erosion=True, erosion_iterations=1, erosion_kernel_size=12, use_sr=True, use_glasses=False):
    try:
        if input_img is None: 
            return None, None, None, None, None, "Please upload an image."
        
        timestamp = int(time.time())
        unique_id = f"res_{timestamp}"
        save_dir = os.path.join("outputs", unique_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save input
        input_path = os.path.join(save_dir, "input.jpg")
        input_img.save(input_path)

        # Preprocess
        require_part = ['face', 'l_eye', 'r_eye', 'mouth']
        if use_glasses:
            require_part.append('eye_g')
        align_img, hr_img, trans_params, lm68_2d, seg_mask, skin_mask = preprocess_model(np.array(input_img), require_part=require_part)
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

        # Masking for Input
        ref_cv = cv2.cvtColor(np.array(unwrap_pil), cv2.COLOR_RGB2BGR)
        _, mask = cv2.threshold(cv2.cvtColor(ref_cv, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

        if use_erosion:
            Erosion_Mask_np = cv2.imread(ERODE_MASK_PATH, cv2.IMREAD_GRAYSCALE)

            if Erosion_Mask_np.shape != mask.shape:
                Erosion_Mask_np = cv2.resize(Erosion_Mask_np, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.where((mask == 255) | (Erosion_Mask_np == 255), 255, 0).astype(np.uint8)

            # use configurable kernel size and iterations
            kernel_size = int(erosion_kernel_size)
            iterations = int(erosion_iterations)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
            ref_eroded = np.zeros_like(ref_cv)
            ref_eroded[eroded_mask == 255] = ref_cv[eroded_mask == 255]
            ref_rgb = cv2.cvtColor(ref_eroded, cv2.COLOR_BGR2RGB)
            print(f"[INFO] Using erosion for mask refinement (kernel={kernel_size}, iterations={iterations})")
        else:
            ref_eroded = np.zeros_like(ref_cv)
            ref_eroded[mask == 255] = ref_cv[mask == 255]
            ref_rgb = cv2.cvtColor(ref_eroded, cv2.COLOR_BGR2RGB)
            print("[INFO] Skipping erosion, using original mask")
        
        if ref_rgb.dtype != np.uint8:
            ref_rgb = np.clip(ref_rgb, 0, 255).astype(np.uint8)
        ref_pil_masked = Image.fromarray(ref_rgb)  # eroded image (if erosion is not used, it is the original unwrap)
        
        # save the image before super-resolution
        ref_pil_masked.save(os.path.join(save_dir, "uv_input_masked_before_sr.png"))

        # super-resolution (optional)
        if use_sr:
            print("[INFO] Using super-resolution...")
            try:
                ref_pil_sr = face3d_model.sr_model(ref_pil_masked)
                # save the original-size image after super-resolution
                ref_pil_sr.save(os.path.join(save_dir, "uv_input_masked_after_sr.png"))
                # resize to 1024x1024 (the final used image)
                ref_pil_final = ref_pil_sr.resize((1024, 1024))
            except Exception as e:
                print(f"[WARNING] Super-resolution failed: {e}. Falling back to resize.")
                # if super-resolution fails, resize directly
                ref_pil_final = ref_pil_masked.resize((1024, 1024))
        else:
            print("[INFO] Skipping super-resolution, using direct resize.")
            # resize directly to 1024x1024
            ref_pil_final = ref_pil_masked.resize((1024, 1024))
            # save a file marked as super-resolution (for consistency, based on the original mask image for 4x resize)
            ref_pil_4x = ref_pil_masked.resize((ref_pil_masked.width * 4, ref_pil_masked.height * 4))
            ref_pil_4x.save(os.path.join(save_dir, "uv_input_masked_after_sr.png"))
        
        # save the final used image
        ref_pil_final.save(os.path.join(save_dir, "uv_input_masked.png"))

        # State update
        state = {
            "save_dir": save_dir,
            "seed": seed,
            "mesh_data": mesh_data,
            "ref_image": ref_pil_final,  
            "input_image": hr_img_pil, # for caption
        }
        
        # return: original unwrap, eroded (if erosion is not used, it is the original unwrap), super-resolution resized back
        paint_editor_update = gr.update(value=ref_pil_final, interactive=True)
        return state, hr_img_pil, unwrap_pil, ref_pil_masked, ref_pil_final, paint_editor_update, "Step 1 Done: 3D Recon & Unwrap complete."
    except Exception as e:
        import traceback
        error_msg = f"Step 1 Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, None, None, gr.update(value=None, interactive=True), error_msg


def apply_manual_paint(state, painted_image):
    """Apply black brush strokes to the ref image for inpainting."""
    try:
        if not state or "ref_image" not in state:
            return None, None, None, "Please run Step 1 first."

        if painted_image is None:
            return state, state["ref_image"], state["ref_image"], "No manual paint provided."

        base_image = state["ref_image"].convert("RGB")
        save_dir = state["save_dir"]

        # Gradio ImageEditor/Sketchpad may return a dict with composite/mask
        if isinstance(painted_image, dict):
            paint_layer = (
                painted_image.get("image")
                or painted_image.get("composite")
                or painted_image.get("background")
            )
            paint_mask = painted_image.get("mask")
        else:
            paint_layer = painted_image
            paint_mask = None

        if paint_mask is not None:
            if isinstance(paint_mask, np.ndarray):
                paint_mask = Image.fromarray(paint_mask)
            paint_mask = paint_mask.convert("L")
            black = Image.new("RGB", base_image.size, (0, 0, 0))
            merged = Image.composite(black, base_image, paint_mask)
        else:
            if paint_layer is None:
                return state, base_image, base_image, "No manual paint provided."
            if isinstance(paint_layer, np.ndarray):
                paint_layer = Image.fromarray(paint_layer)
            merged = paint_layer.convert("RGB")

        merged.save(os.path.join(save_dir, "uv_input_masked_manual.png"))
        state["ref_image"] = merged
        return state, merged, merged, "Manual paint applied."
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, None, None, str(e)

def step2_caption(state):
    try:
        if not state or "input_image" not in state: 
            return [None] * 6 + ["Please run Step 1 first."] # Return Nones for textboxes (caption, inpaint, delight, align_albedo, align_normal, align_rsd)
        
        print("[INFO] Generating Caption from original input image...")
        input_image = state["input_image"]
        
        # 1. call JoyCaptioner twice, generate two versions of Caption
        # - first time: include lighting description (for Inpaint)
        # - second time: no lighting description (for other LoRAs)
        print("[INFO] Generating caption with lighting (for Inpaint)...")
        raw_caption_with_lighting = joy_captioner.caption(input_image, include_lighting=True)
        caption_with_lighting = sanitize_caption(raw_caption_with_lighting)
        
        print("[INFO] Generating caption without lighting (for other LoRAs)...")
        raw_caption_without_lighting = joy_captioner.caption(input_image, include_lighting=False)
        caption_without_lighting = sanitize_caption(raw_caption_without_lighting)
        
        # save the complete version to state (for display)
        state["caption"] = caption_with_lighting
        
        # 2. build semantic header (Semantic Header)
        semantic_header_inpaint = f"An unfolded UV texture map of {caption_with_lighting}"
        semantic_header_others = f"An unfolded UV texture map of {caption_without_lighting}"
        
        # 5. automatically assemble prompts for each stage
        
        # For Step 3.1: Inpainting (use complete caption, include lighting information)
        prompt_inpaint = f"{semantic_header_inpaint}, seamless, continuous texture, high fidelity, 8k resolution. Synthesize high-frequency micro-details, pixel-perfect continuity."
        
        # For Step 3.2: Delight (use caption without lighting)
        base_delight_def = "Normalized lighting texture map, calibrated neutral illumination. Uniform pixel intensity distribution across the entire facial surface. Eliminate all lighting gradients, environmental bias, and directional shading. Perfectly balanced exposure, strictly retaining high-frequency micro-details, razor-sharp pores, and authentic skin grain. High-fidelity raw texture quality. Raw, 8k, highly detailed, macro photography, hard focus. Remove all lighting, shadows, and shading. Generate flat, unlit base color texture."
        prompt_delight = f"{semantic_header_others} {base_delight_def}"
        
        # For Step 4: Align (use caption without lighting)
        # we put the semantic at the front, and the technical parameters at the back
        
        base_albedo_def = "Ultra-High Definition 8K Diffuse Albedo map, flat lighting, unlit base color, pixel-perfect clarity. The texture reveals natural melanin pigmentation, hemoglobin redness, and distinct subsurface scattering warmth zones. The chart is completely void of baked shadows, ambient occlusion, or specular highlights, representing pure biological skin color values in UV space."
        prompt_albedo = f"{semantic_header_others} {base_albedo_def}"
        prompt_align_albedo = f"{semantic_header_others} {base_albedo_def}"
        
        base_normal_def = "Ultra-High Definition 8K Surface Normal map in UV space, displaying sharp high-frequency relief with pixel-perfect clarity. The texture emphasizes intricate pore structures, fine wrinkles, and precise skin micro-geometry orientation relative to UV coordinates."
        prompt_normal = f"{semantic_header_others} {base_normal_def}"
        prompt_align_normal = f"{semantic_header_others} {base_normal_def}"
        
        base_rsd_def = "Ultra-High Definition 8K packed RSD technical texture map in UV space, consisting of three specific PBR channels. Red: Roughness/Oiliness; Green: Specular Intensity; Blue: Displacement Depth. The map captures purely physical skin surface properties."
        prompt_rsd = f"{semantic_header_others} {base_rsd_def}"
        prompt_align_rsd = f"{semantic_header_others} {base_rsd_def}"
        
        status_msg = f"Step 2 Done: Caption generated and prompts constructed based on image context."
        
        # return to the frontend Textbox (in the order defined by the UI)
        return (
            caption_with_lighting,  # caption_box (display the complete version, include lighting)
            prompt_inpaint,         # prompt_inpaint
            prompt_delight,         # prompt_delight
            prompt_albedo,          # prompt_albedo
            prompt_normal,          # prompt_normal
            prompt_rsd,             # prompt_rsd
            prompt_align_albedo,    # prompt_align_albedo
            prompt_align_normal,     # prompt_align_normal
            prompt_align_rsd,       # prompt_align_rsd
            status_msg              # status
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Step 2 Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # return 7 values: caption, prompt_inpaint, prompt_delight, prompt_align_albedo, prompt_align_normal, prompt_align_rsd, status
        return None, None, None, None, None, None, error_msg

def step3_inpaint(state, prompt, negative_prompt, steps=31, guidance_scale=1.0, use_input_image_for_inpainting=False):
    """Step 3.1: Inpainting"""
    try:
        if not state or "ref_image" not in state: return None, None, "Please run Step 1 first."
        save_dir = state["save_dir"]
        seed = int(state["seed"])
        ref_image = state["ref_image"]
        
        if not prompt or prompt.strip() == "":
            prompt = "Seamlessly restore and complete. Synthesize high-frequency details."

        if use_input_image_for_inpainting:
            input_image = state["input_image"]
            pipe.transformer.set_adapter("inpainting_ori")
        else:
            input_image = None
            pipe.transformer.set_adapter("inpainting")
        
        generator = torch.Generator(DEVICE).manual_seed(seed)
        
        print(f"[INFO] Inpainting Prompt: {prompt[:100]}...")

        inpaint_res = pipe(
            image=ref_image, ref_image=input_image, prompt=prompt, negative_prompt=negative_prompt, 
            num_inference_steps=int(steps), generator=generator,
            guidance_scale=guidance_scale
        ).images[0]
        inpaint_res.save(os.path.join(save_dir, "seq_step1_inpaint.png"))
        
        state["inpaint_res"] = inpaint_res
        return state, inpaint_res, f"Step 3.1 Done."
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, None, str(e)

def step3_delight(state, prompt, negative_prompt, steps=31, guidance_scale=2.0):
    """Step 3.2: Delight"""
    try:
        if not state or "inpaint_res" not in state: return None, None, "Please run Step 3.1 first."
        save_dir = state["save_dir"]
        seed = int(state["seed"])
        inpaint_res = state["inpaint_res"]
        
        pipe.transformer.set_adapter("delight")
        generator = torch.Generator(DEVICE).manual_seed(seed)
        
        delight_res = pipe(
            inpaint_res, prompt=prompt, negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale, num_inference_steps=int(steps), generator=generator
        ).images[0]
        delight_res.save(os.path.join(save_dir, "seq_step2_delight.png"))
        
        state["delight_res"] = delight_res
        return state, delight_res, f"Step 3.2 Done."
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, None, str(e)

def step3_separate(state, mode, prompt, negative_prompt, steps=31, guidance_scale=2.0):
    """Step 3.3: Separate Channels"""
    try:
        if not state or "delight_res" not in state: return None, None, "Please run Step 3.2 first."
        save_dir = state["save_dir"]
        seed = int(state["seed"])
        delight_res = state["delight_res"]
        
        # verify and set adapter
        if mode not in pipe.transformer.peft_config:
            raise RuntimeError(f"LoRA adapter '{mode}' not loaded. Available adapters: {list(pipe.transformer.peft_config.keys())}")
        pipe.transformer.set_adapter(mode)
        
        generator = torch.Generator(DEVICE).manual_seed(seed)
        
        res = pipe(
            delight_res, prompt=prompt, negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale, num_inference_steps=int(steps), generator=generator
        ).images[0]

        res_flipped = res.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        res.save(os.path.join(save_dir, f"seq_res_{mode}_original.png"))
        res_flipped.save(os.path.join(save_dir, f"seq_res_{mode}_flipped.png"))

        if mode == "rsd":
            res_roughness = res_flipped.getchannel(0)
            res_specular = res_flipped.getchannel(1)
            res_displacement = res_flipped.getchannel(2)
            res_roughness.save(os.path.join(save_dir, f"seq_res_roughness_flipped.png"))
            res_specular.save(os.path.join(save_dir, f"seq_res_specular_flipped.png"))
            res_displacement.save(os.path.join(save_dir, f"seq_res_displacement_flipped.png"))
        
        if "seq_results" not in state: state["seq_results"] = {}
        state["seq_results"][mode] = res

        if mode == "rsd":
            state["seq_results"]["roughness"] = res_roughness
            state["seq_results"]["specular"] = res_specular
            state["seq_results"]["displacement"] = res_displacement
        return state, res, f"Step 3.3 {mode} Done."
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, None, str(e)

def step4_align(state, align_steps=40, guidance_scale=2.0,
                prompt_albedo="", prompt_normal="", prompt_rsd="",
                negative_prompt_albedo="", negative_prompt_normal="", negative_prompt_rsd="",
                use_dalign=False):
    """Step 4: Align / IntrinsiX Generation"""
    try:
        if not state or "delight_res" not in state or "mesh_data" not in state: 
            return None, None, None, "Please run Step 1 and Step 3.2 first."
        save_dir = state["save_dir"]
        seed = int(state["seed"])
        delight_res = state["delight_res"]
        inpaint_res = state["inpaint_res"]
        mesh_data = state["mesh_data"]
        
        # Default prompts (if Step 2 did not generate, use default values)
        if not prompt_albedo or prompt_albedo.strip() == "":
            prompt_albedo = "Ultra-High Definition 8K Diffuse Albedo map of a human face texture, flat lighting, unlit base color, pixel-perfect clarity. The texture reveals natural melanin pigmentation, hemoglobin redness, and distinct subsurface scattering warmth zones. It features high-frequency skin details including specific facial moles, freckles, and capillary variations. The chart is completely void of baked shadows, ambient occlusion, or specular highlights, representing pure biological skin color values in UV space."
        if not prompt_normal or prompt_normal.strip() == "":
            prompt_normal = "Ultra-High Definition 8K Surface Normal map of a human face texture in UV space, displaying sharp high-frequency relief with pixel-perfect clarity. The texture emphasizes intricate pore structures, fine wrinkles, and precise skin micro-geometry orientation relative to UV coordinates, where RGB vectors accurately represent surface angles and bumps without any albedo color or lighting information, achieving absolute biological structural realism."
        if not prompt_rsd or prompt_rsd.strip() == "":
            prompt_rsd = "Ultra-High Definition 8K packed RSD technical texture map of a human face in UV space, consisting of three specific PBR channels. The Red channel encodes detailed micro-surface roughness and oiliness zones; the Green channel defines high-fidelity specular reflection intensity; and the Blue channel represents displacement depth for palpable pores and wrinkles. The map captures purely physical skin surface properties distinct from diffuse color or tangent normals."
        
        default_neg = "blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts, low resolution, baked lighting removal, cartoonish, loss of detail, interpolated pixels"
        if not negative_prompt_albedo or negative_prompt_albedo.strip() == "":
            negative_prompt_albedo = default_neg
        if not negative_prompt_normal or negative_prompt_normal.strip() == "":
            negative_prompt_normal = default_neg
        if not negative_prompt_rsd or negative_prompt_rsd.strip() == "":
            negative_prompt_rsd = default_neg
                
        generator = torch.Generator(DEVICE).manual_seed(seed)
        print(f"[INFO] Align Prompts - Albedo: {prompt_albedo[:50]}...")

        if use_dalign: # replace delight_res with inpaint results, use dalign_lora
            intrinsix_pipeline.transformer.load_state_dict(dalign_state_dict, strict=False)
            res_list = intrinsix_pipeline(
                inpaint_res, 
                prompt_albedo=prompt_albedo, prompt_normal=prompt_normal, prompt_rsd=prompt_rsd,
                negative_prompt_albedo=negative_prompt_albedo, negative_prompt_normal=negative_prompt_normal, negative_prompt_rsd=negative_prompt_rsd,
                guidance_scale=guidance_scale, num_inference_steps=int(align_steps), generator=generator
            ).images
        else:
            intrinsix_pipeline.transformer.load_state_dict(align_state_dict, strict=False)
            res_list = intrinsix_pipeline(
                delight_res, 
                prompt_albedo=prompt_albedo, prompt_normal=prompt_normal, prompt_rsd=prompt_rsd,
                negative_prompt_albedo=negative_prompt_albedo, negative_prompt_normal=negative_prompt_normal, negative_prompt_rsd=negative_prompt_rsd,
                guidance_scale=guidance_scale, num_inference_steps=int(align_steps), generator=generator
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

        # save apply mask to align_albedo, align_normal, align_roughness, align_specular, align_displacement
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
    
        # GLB Export
        glb_path_textured = os.path.join(save_dir, "textured_mesh.glb")
        glb_path_white = os.path.join(save_dir, "white_mesh.glb")

        try:
            save_glb_with_specular(
                vertices=mesh_data["vertices"], faces=mesh_data["faces"], uvs=mesh_data["uvs"], face_uvs=mesh_data["face_uvs"],
                albedo_path=align_albedo, normal_path=align_normal, roughness_path=align_roughness, specular_path=align_specular,
                output_glb_path=glb_path_textured
            )
            save_glb_white_model(
                vertices=mesh_data["vertices"], faces=mesh_data["faces"], uvs=mesh_data["uvs"], face_uvs=mesh_data["face_uvs"],
                normal_path=align_normal, roughness_path=align_roughness, specular_path=align_specular,
                output_glb_path=glb_path_white,
                # common suggestions for base_color: white(1.0,1.0,1.0,1.0), gray(0.8,0.8,0.8,1.0), or slightly beige(0.95,0.95,0.88,1.0)
                base_color=(0.65, 0.65, 0.65, 1.0)
            )
            # generate HTML viewer
            model_viewer_html_textured = build_model_viewer_html(save_dir, height=HTML_HEIGHT, width=HTML_WIDTH, textured=True)
            # model_viewer_html_white = build_model_viewer_html(save_dir, height=HTML_HEIGHT, width=HTML_WIDTH, textured=False)
            # convert to absolute path
            glb_path_textured = os.path.abspath(glb_path_textured)
            glb_path_white = os.path.abspath(glb_path_white)
        except Exception as e:
            print(f"GLB Error: {e}")
            import traceback
            traceback.print_exc()
            glb_path_textured = None
            glb_path_white = None
            model_viewer_html_textured = None
            # model_viewer_html_white = None
        
        state["align_textures"] = {
            "albedo": align_albedo,
            "normal": align_normal,
            "roughness": align_roughness,
            "specular": align_specular,
            "displacement": align_displacement,
            "albedo_masked": align_albedo_masked,
            "normal_masked": align_normal_masked,
            "roughness_masked": align_roughness_masked,
            "specular_masked": align_specular_masked,
            "displacement_masked": align_displacement_masked
        }
        state["glb_path_textured"] = glb_path_textured
        state["glb_path_white"] = glb_path_white
        state["model_viewer_html_textured"] = model_viewer_html_textured
        # state["model_viewer_html_white"] = model_viewer_html_white
        
        return state, [align_albedo, align_normal, align_roughness, align_specular, align_displacement], model_viewer_html_textured, "Step 4 Done."
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, None, None, None, str(e)

def save_glb(state):
    """Save GLB (.glb) file (both textured and white model)"""
    try:
        if not state or "save_dir" not in state or "align_textures" not in state:
            return None, None, "Run Step 4 first."
        # Return both textured and white model versions if available
        glb_path_textured = state.get("glb_path_textured", None)
        glb_path_white = state.get("glb_path_white", None)
        if glb_path_textured and glb_path_white:
            return glb_path_textured, glb_path_white, "Saved GLB (Textured & White Model)."
        elif glb_path_textured:
            return glb_path_textured, None, "Saved GLB (Textured only)."
        elif glb_path_white:
            return None, glb_path_white, "Saved GLB (White Model only)."
        else:
            return None, None, "GLB not generated yet."
    except Exception as e:
        return None, None, str(e)

def save_blend(state):
    """Save Blender (.blend) file"""
    try:
        if not state or "save_dir" not in state or "align_textures" not in state: return None, "Run Step 4 first."
        save_dir = state["save_dir"]
        mesh_data = state["mesh_data"]
        textures = state["align_textures"]
        blend_path = os.path.join(save_dir, "model.blend")
        blend_path_abs = os.path.abspath(blend_path)
        blend_path_abs_masked = os.path.join(save_dir, "model_masked.blend")
        blend_path_abs_masked = os.path.abspath(blend_path_abs_masked)

        os.makedirs(os.path.dirname(blend_path_abs), exist_ok=True)
        
        save_blend_file(
            vertices=mesh_data["vertices"], faces=mesh_data["faces"], uvs=mesh_data["uvs"], face_uvs=mesh_data["face_uvs"],
            albedo_path=textures["albedo"], normal_path=textures["normal"], roughness_path=textures["roughness"], specular_path=textures["specular"], displacement_path=textures["displacement"],
            output_blend_path=blend_path_abs  # use absolute path
        )
        
        # save masked blend
        save_blend_file(
            vertices=mesh_data["vertices"], faces=mesh_data["faces"], uvs=mesh_data["uvs"], face_uvs=mesh_data["face_uvs"],
            albedo_path=textures["albedo_masked"], normal_path=textures["normal"], roughness_path=textures["roughness"], specular_path=textures["specular_masked"], displacement_path=textures["displacement"],
            output_blend_path=blend_path_abs_masked  # use absolute path
        )
        try:
            sep_textures = state["seq_results"]
            blend_path_abs_sep = os.path.join(save_dir, "model_sep.blend")
            blend_path_abs_sep = os.path.abspath(blend_path_abs_sep)
            save_blend_file(
                vertices=mesh_data["vertices"], faces=mesh_data["faces"], uvs=mesh_data["uvs"], face_uvs=mesh_data["face_uvs"],
                albedo_path=sep_textures["albedo"], normal_path=sep_textures["normal"], roughness_path=sep_textures["roughness"], specular_path=sep_textures["specular"], displacement_path=sep_textures["displacement"],
                output_blend_path=blend_path_abs_sep  # use absolute path
            )
        except Exception as e:
            print(f"Save Sep Blend Error: {e}")
            import traceback
            traceback.print_exc()
            blend_path_abs_sep = None
            return blend_path_abs, blend_path_abs_masked, None, "Saved Blender."
        
        # verify if the file exists
        if not os.path.exists(blend_path_abs):
            return None, f"Blender file not found at {blend_path_abs}"
        return blend_path_abs, blend_path_abs_masked, blend_path_abs_sep, "Saved Blender."
    except Exception as e: 
        import traceback
        error_msg = f"Save Blender Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, error_msg

def step5_pack(state):
    try:
        if not state or "save_dir" not in state: return None, "No state."
        save_dir = state["save_dir"]
        zip_path = save_dir + ".zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(save_dir):
                for file in files: zipf.write(os.path.join(root, file), arcname=file)
        # return absolute path, ensure Gradio can access it correctly
        zip_path_abs = os.path.abspath(zip_path)
        return zip_path_abs, "Zip ready."
    except Exception as e: return None, str(e)


def build_model_viewer_html(save_folder, height=650, width=790, textured=False):
    """
    build custom 3D model viewer HTML
    """
    import html
    import urllib.parse
    
    if textured:
        glb_filename = "textured_mesh.glb"
        template_name = os.path.join(CURRENT_DIR, TOPO_DIR, "modelviewer-textured-template.html")
        output_html_path = os.path.join(save_folder, "textured_mesh.html")
    else:
        glb_filename = "white_mesh.glb"
        template_name = os.path.join(CURRENT_DIR, TOPO_DIR, "modelviewer-template.html")
        output_html_path = os.path.join(save_folder, "white_mesh.html")
    
    # absolute path of the GLB file
    glb_path = os.path.abspath(os.path.join(save_folder, glb_filename))
    if USE_STATIC_SERVER:
        # static URLs under /outputs and /static
        rel_glb_path = os.path.relpath(glb_path, SAVE_DIR).replace(os.sep, "/")
        glb_file_url = f"/outputs/{rel_glb_path}"
        env_map_gradient_url = "/custom_assets/env_maps/gradient.jpg"
        env_map_white_url = "/custom_assets/env_maps/white.jpg"
    else:
        # Gradio /file= path does not need URL encoding (encoded path will be treated as a literal path, causing 404)
        glb_file_url = f"/gradio_api/file={glb_path}"
        # environment map path (located in assets/env_maps)
        env_map_gradient_path = os.path.abspath(os.path.join(TOPO_DIR, "env_maps", "gradient.jpg"))
        env_map_white_path = os.path.abspath(os.path.join(TOPO_DIR, "env_maps", "white.jpg"))
        env_map_gradient_url = f"/gradio_api/file={env_map_gradient_path}"
        env_map_white_url = f"/gradio_api/file={env_map_white_path}"
    
    offset = 50 if textured else 10
    with open(template_name, 'r', encoding='utf-8') as f:
        template_html = f.read()

    template_html = template_html.replace('#height#', f'{height - offset}')
    template_html = template_html.replace('#width#', f'{width}')
    # use Gradio file interface path, not relative path
    template_html = template_html.replace('#src#', glb_file_url)
    # replace environment map paths to target URLs
    template_html = template_html.replace("./env_maps/gradient.jpg", env_map_gradient_url)
    template_html = template_html.replace("./env_maps/white.jpg", env_map_white_url)
    template_html = template_html.replace("/static/env_maps/gradient.jpg", env_map_gradient_url)
    template_html = template_html.replace("/static/env_maps/white.jpg", env_map_white_url)

    # export the html with replaced content, for external access
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(template_html)

    if USE_STATIC_SERVER:
        rel_html_path = os.path.relpath(output_html_path, SAVE_DIR).replace(os.sep, "/")
        html_url = f"/outputs/{rel_html_path}"
        iframe_tag = (
            f'<iframe src="{html_url}" height="{height}" width="100%" '
            f'frameborder="0"></iframe>'
        )
    else:
        # use srcdoc to directly embed HTML, avoid /file= HTML resources 404
        srcdoc = html.escape(template_html, quote=True)
        iframe_tag = (
            f'<iframe srcdoc="{srcdoc}" height="{height}" width="100%" '
            f'frameborder="0" sandbox="allow-scripts allow-same-origin"></iframe>'
        )

    return f"""
        <div style='height: {height}px; width: 100%;'>
        {iframe_tag}
        </div>
    """

# ================= UI Layout section =================
with gr.Blocks(
    title="FaceTex Studio Texturing Workflow",
    theme=face_studio_theme,
    css=FACE_STUDIO_CSS,
) as demo:
    state = gr.State({})
    
    title_html = """
    <div style="text-align:center; margin-bottom: 16px;">
        <div style="font-size: 2.2em; font-weight: 800; letter-spacing: 0.5px; color: #FFFFFF;">
            🎨 FaceTex Studio
        </div>
        <div style="margin-top: 6px; font-size: 1.02em; color: #9aa0aa;">
            Monocular Avatar Reconstruction via Cascaded Diffusion Priors & UV-Space Differentiable Shading
        </div>
    </div>
    """
    gr.HTML(title_html)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_img = gr.Image(type="pil", label="Input")

            examples_dir = "examples"
            if os.path.exists(examples_dir):
                example_images = [os.path.join(examples_dir, f) for f in os.listdir(examples_dir) if f.lower().endswith(('jpg','png','jpeg','webp'))]
                if example_images:
                    gr.Examples(examples=example_images, inputs=[input_img], label="Examples", examples_per_page=5)

            seed_input = gr.Number(value=43, label="Seed", precision=0)
            status = gr.Textbox(label="Status", interactive=False, lines=3)
            
        with gr.Column(scale=3):
            # --- Step 1 ---
            with gr.Accordion("Step 1: Recon & Unwrap", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        use_erosion = gr.Checkbox(value=True, label="Erosion")
                        use_sr = gr.Checkbox(value=False, label="SR")
                        use_glasses = gr.Checkbox(value=False, label="Glasses", visible=False)
                    with gr.Column(scale=2):
                        with gr.Row():
                            erosion_iterations = gr.Number(value=3, label="Iterations", precision=0, minimum=1, maximum=10)
                            erosion_kernel_size = gr.Number(value=12, label="Kernel", precision=0, minimum=3, maximum=50)
                with gr.Row():
                    out_align_img = gr.Image(label="Align", type="pil", height=200)
                    out_unwrap = gr.Image(label="Unwrap", type="pil", height=200)
                    out_masked = gr.Image(label="Erosion", type="pil", height=200)
                    out_sr_final = gr.Image(label="SR Final", type="pil", height=200)
                btn_step1 = gr.Button("Run Step 1: Preprocess", elem_classes="step-btn")

                with gr.Accordion("Manual Paint", open=False):
                    with gr.Row():
                        manual_paint_img = gr.ImageEditor(
                            label="Paint UV",
                            type="pil",
                            interactive=True,
                            height=400,
                        )
                        manual_paint_preview = gr.Image(
                            label="Preview",
                            type="pil",
                            height=400,
                        )
                    btn_apply_paint = gr.Button("Apply Paint", variant="secondary")
            
            # --- Step 2 ---
            with gr.Accordion("Step 2: Caption", open=True):
                caption_box = gr.Textbox(label="Caption", lines=1)
                btn_step2 = gr.Button("Run Step 2: Caption", elem_classes="step-btn")
            

            # --- Step 3.1: Inpaint  &  Step 3.2: Delight ---
            with gr.Accordion("Step 3: Material Refinement", open=True):
                with gr.Row():
                    # Left side: Inpaint
                    with gr.Column(scale=1):
                        with gr.Accordion("Prompt & Settings (Inpaint)", open=False):
                            prompt_inpaint = gr.Textbox(label="Prompt", lines=2, placeholder="Waiting for Step 2...")
                            neg_prompt_inpaint = gr.Textbox(label="Negative Prompt", lines=1, value="blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts, overexposure, overexposed, blown highlights, clipped whites, excessive brightness, loss of detail in highlights")
                        with gr.Row():
                            inpaint_guidance = gr.Slider(minimum=0.0, maximum=10.0, value=1.0, step=0.1, label="Guidance")
                            inpaint_steps = gr.Number(value=31, label="Steps", precision=0)
                        use_input_image_for_inpainting_checkbox = gr.Checkbox(value=False, label="Use Input Image")
                        out_inpaint = gr.Image(label="Inpaint Result", type="pil", height=360)
                        btn_step3_1 = gr.Button("Run Step 3.1: Inpainting", elem_classes="step-btn")                    
                    # Right side: Delight
                    with gr.Column(scale=1):
                        with gr.Accordion("Prompt & Settings (Delight)", open=False):
                            prompt_delight = gr.Textbox(label="Prompt", lines=2, placeholder="Waiting for Step 2...")
                            neg_prompt_delight = gr.Textbox(label="Negative Prompt", lines=1, value="blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts")
                        with gr.Row():
                            delight_guidance = gr.Slider(minimum=0.0, maximum=10.0, value=2.0, step=0.1, label="Guidance")
                            delight_steps = gr.Number(value=31, label="Steps", precision=0)
                        out_delight = gr.Image(label="Delight Result", type="pil", height=360)
                        btn_step3_2 = gr.Button("Run Step 3.2: Delighting", elem_classes="step-btn")
            
            # --- Step 3.3: Separate Material Groups ---
            with gr.Accordion("Step 3.3: Separate Material Groups", open=False):
                with gr.Tabs():
                    with gr.Tab("Albedo"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                with gr.Accordion("Prompt & Settings (Albedo)", open=False):
                                    prompt_albedo = gr.Textbox(label="Prompt", lines=2, value="Ultra-High Definition 8K Diffuse Albedo map, flat lighting, unlit base color.")
                                    neg_prompt_albedo = gr.Textbox(label="Negative", lines=1, value="blur, smoothing, flat shading, mismatched lighting")
                                with gr.Row():
                                    albedo_steps = gr.Number(value=31, label="Steps", precision=0)
                                    albedo_guidance = gr.Slider(minimum=0.0, maximum=10.0, value=2.0, step=0.1, label="Guidance")
                            with gr.Column(scale=1):
                                out_albedo = gr.Image(label="Result", type="pil", height=400)
                        btn_step3_3_albedo = gr.Button("Run Step 3.3.1: Albedo", elem_classes="step-btn")

                    with gr.Tab("Normal"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                with gr.Accordion("Prompt & Settings (Normal)", open=False):
                                    prompt_normal = gr.Textbox(label="Prompt", lines=2, value="Ultra-High Definition 8K Surface Normal map, displaying sharp high-frequency relief.")
                                    neg_prompt_normal = gr.Textbox(label="Negative", lines=1, value="blur, smoothing, flat shading, mismatched lighting")
                                with gr.Row():
                                    normal_steps = gr.Number(value=31, label="Steps", precision=0)
                                    normal_guidance = gr.Slider(minimum=0.0, maximum=10.0, value=2.0, step=0.1, label="Guidance")
                            with gr.Column(scale=1):
                                out_normal = gr.Image(label="Result", type="pil", height=400)
                        btn_step3_3_normal = gr.Button("Run Step 3.3.2: Normal", elem_classes="step-btn")
                            
                    with gr.Tab("RSD"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                with gr.Accordion("Prompt & Settings (RSD)", open=False):
                                    prompt_rsd = gr.Textbox(label="Prompt", lines=2, value="Ultra-High Definition 8K packed RSD technical texture map.")
                                    neg_prompt_rsd = gr.Textbox(label="Negative", lines=1, value="blur, smoothing, flat shading, mismatched lighting")
                                with gr.Row():
                                    rsd_steps = gr.Number(value=31, label="Steps", precision=0)
                                    rsd_guidance = gr.Slider(minimum=0.0, maximum=10.0, value=2.0, step=0.1, label="Guidance")
                            with gr.Column(scale=1):
                                out_rsd = gr.Image(label="Result", type="pil", height=400)
                        btn_step3_3_rsd = gr.Button("Run Step 3.3.3: RSD", elem_classes="step-btn")
                        

            # --- Step 4 ---
            with gr.Accordion("Step 4: Align & Export", open=True):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("Albedo"):
                                with gr.Accordion("Prompt & Settings (Albedo)", open=False):
                                    prompt_align_albedo = gr.Textbox(label="Prompt", lines=2, placeholder="Waiting for Step 2...")
                                    neg_prompt_align_albedo = gr.Textbox(label="Neg", lines=1, value="blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts, low resolution, baked lighting removal, cartoonish, loss of detail, interpolated pixels")
                            with gr.Tab("Normal"):
                                with gr.Accordion("Prompt & Settings (Normal)", open=False):
                                    prompt_align_normal = gr.Textbox(label="Prompt", lines=2, placeholder="Waiting for Step 2...")
                                    neg_prompt_align_normal = gr.Textbox(label="Neg", lines=1, value="blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts, low resolution, baked lighting removal, cartoonish, loss of detail, interpolated pixels")
                            with gr.Tab("RSD"):
                                with gr.Accordion("Prompt & Settings (RSD)", open=False):
                                    prompt_align_rsd = gr.Textbox(label="Prompt", lines=2, placeholder="Waiting for Step 2...")
                                    neg_prompt_align_rsd = gr.Textbox(label="Neg", lines=1, value="blur, smoothing, flat shading, mismatched lighting, visible seams, artifacts, low resolution, baked lighting removal, cartoonish, loss of detail, interpolated pixels")
                    with gr.Column(scale=1):
                        align_steps = gr.Number(value=40, label="Steps", precision=0)
                        align_guidance_slider = gr.Slider(minimum=0.0, maximum=10.0, value=2.0, step=0.1, label="Guidance")
                        use_dalign_checkbox = gr.Checkbox(value=False, label="Use DAlign")
                btn_step4 = gr.Button("Run Step 4: Material Alignment", elem_classes="step-btn")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        gallery_align = gr.Gallery(label="Align Results", columns=2, height=600)
                # with gr.Row():
                    with gr.Column(scale=4):
                        out_glb_textured = gr.HTML(label="3D Preview (Textured)", value="<div style='height: 650px; width: 100%; display: flex; justify-content: center; align-items: center;'><p>Waiting for model generation...</p></div>")
                    # with gr.Column(scale=1):
                        # out_glb_white = gr.HTML(label="3D Preview (White Model)", value="<div style='height: 650px; width: 100%; display: flex; justify-content: center; align-items: center;'><p>Waiting for model generation...</p></div>")
                with gr.Row():
                    btn_save_glb = gr.Button("Save GLB: Textured Model", variant="secondary")
                    btn_save_blend = gr.Button("Save Blend: Textured Model", variant="secondary")
                    btn_save_blend_sep = gr.Button("Save Blend: Separated Model", variant="secondary")
                with gr.Row():
                    out_glb_file = gr.File(label="Textured Model GLB File", interactive=False, file_types=[".glb"], elem_classes="file-output", height=80)
                    out_glb_white_file = gr.File(label="White Model GLB File", interactive=False, file_types=[".glb"], elem_classes="file-output", height=80)
                    out_blend = gr.File(label="Blend File", interactive=False, file_types=[".blend"], elem_classes="file-output", height=80)
                    out_blend_masked = gr.File(label="Masked Blend File", interactive=False, file_types=[".blend"], elem_classes="file-output", height=80)
                    out_blend_sep = gr.File(label="Separated Blend File", interactive=False, file_types=[".blend"], elem_classes="file-output", height=80)
            btn_pack = gr.Button("Step 5: Pack & Download", variant="primary")
            dl_file = gr.File(label="Zip File")

    # Events Binding
    btn_step1.click(
        step1_preprocess,
        [input_img, seed_input, use_erosion, erosion_iterations, erosion_kernel_size, use_sr, use_glasses],
        [state, out_align_img, out_unwrap, out_masked, out_sr_final, manual_paint_img, status],
    )
    btn_apply_paint.click(apply_manual_paint, [state, manual_paint_img], [state, out_sr_final, manual_paint_preview, status])

    btn_step2.click(step2_caption, [state], [caption_box, prompt_inpaint, prompt_delight, prompt_albedo, prompt_normal, prompt_rsd, prompt_align_albedo, prompt_align_normal, prompt_align_rsd, status])
    btn_step3_1.click(step3_inpaint, [state, prompt_inpaint, neg_prompt_inpaint, inpaint_steps, inpaint_guidance, use_input_image_for_inpainting_checkbox], [state, out_inpaint, status])
    btn_step3_2.click(step3_delight, [state, prompt_delight, neg_prompt_delight, delight_steps, delight_guidance], [state, out_delight, status])
    btn_step3_3_albedo.click(lambda s,p,n,st,g: step3_separate(s,"albedo",p,n,st,g), [state, prompt_albedo, neg_prompt_albedo, albedo_steps, albedo_guidance], [state, out_albedo, status])
    btn_step3_3_normal.click(lambda s,p,n,st,g: step3_separate(s,"normal",p,n,st,g), [state, prompt_normal, neg_prompt_normal, normal_steps, normal_guidance], [state, out_normal, status])
    btn_step3_3_rsd.click(lambda s,p,n,st,g: step3_separate(s,"rsd",p,n,st,g), [state, prompt_rsd, neg_prompt_rsd, rsd_steps, rsd_guidance], [state, out_rsd, status])
    btn_step4.click(step4_align, [state, align_steps, align_guidance_slider, prompt_align_albedo, prompt_align_normal, prompt_align_rsd, neg_prompt_align_albedo, neg_prompt_align_normal, neg_prompt_align_rsd, use_dalign_checkbox], [state, gallery_align, out_glb_textured, status])
    btn_save_glb.click(save_glb, [state], [out_glb_file, out_glb_white_file ,status])
    btn_save_blend.click(save_blend, [state], [out_blend, out_blend_masked, out_blend_sep, status])
    btn_pack.click(step5_pack, [state], [dl_file, status])


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    abs_save_dir = os.path.abspath(SAVE_DIR) 
    abs_assets_dir = os.path.abspath(os.path.join(CURRENT_DIR, TOPO_DIR))

    if USE_STATIC_SERVER:
        app = FastAPI()
        # avoid clobbering Gradio's /assets (CSS/JS/theme)
        app.mount("/custom_assets", StaticFiles(directory=abs_assets_dir), name="custom_assets")
        app.mount("/outputs", StaticFiles(directory=abs_save_dir), name="outputs")
        app = gr.mount_gradio_app(
            app, demo, path="/",
        )
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=7895,
            timeout_keep_alive=5, 
            timeout_graceful_shutdown=5 
        )
    else:
        demo.queue().launch(
            server_name="0.0.0.0",
            server_port=7891,
            allowed_paths=[abs_save_dir, abs_assets_dir],
            favicon_path=None,             
            show_error=True                
        )

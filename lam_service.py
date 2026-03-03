import os
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import uvicorn

from omegaconf import OmegaConf
from safetensors.torch import load_file

from lam.models import ModelLAM
from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
from tools.flame_tracking_single_image import FlameTrackingSingleImage
from tools.generateARKITGLBWithBlender import generate_glb


class AvatarReconstructor:
    """Handles avatar reconstruction logic"""

    def __init__(self, cfg, lam_model, flame_tracker):
        self.cfg = cfg
        self.lam = lam_model
        self.flametracking = flame_tracker

    def process_image(self, image_path: str, working_dir: str) -> str:
        """
        Process input image and generate avatar ZIP file

        Args:
            image_path: Path to input image
            working_dir: Temporary working directory

        Returns:
            Path to output ZIP file
        """
        image_raw = os.path.join(working_dir, "raw.png")
        with Image.open(image_path).convert('RGB') as img:
            img.save(image_raw)

        video_params = "./assets/sample_motion/export/Taylor_Swift/Taylor_Swift.mp4"
        base_vid = os.path.basename(video_params).split(".")[0]
        flame_params_dir = os.path.join("./assets/sample_motion/export", base_vid, "flame_param")
        base_iid = os.path.basename(image_path).split('.')[0]

        dump_video_path = os.path.join(working_dir, "output.mp4")
        dump_image_path = os.path.join(working_dir, "output.png")

        # Prepare dump paths
        omit_prefix = os.path.dirname(image_raw)
        image_name = os.path.basename(image_raw)
        uid = image_name.split(".")[0]

        motion_seqs_dir = flame_params_dir
        dump_image_dir = os.path.dirname(dump_image_path)
        os.makedirs(dump_image_dir, exist_ok=True)
        dump_tmp_dir = dump_image_dir

        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)
        vis_motion = self.cfg.get("vis_motion", False)

        # Preprocess input image: segmentation, flame params estimation
        return_code = self.flametracking.preprocess(image_raw)
        if return_code != 0:
            raise RuntimeError("flametracking preprocess failed!")

        return_code = self.flametracking.optimize()
        if return_code != 0:
            raise RuntimeError("flametracking optimize failed!")

        return_code, output_dir = self.flametracking.export()
        if return_code != 0:
            raise RuntimeError("flametracking export failed!")

        image_path = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")

        aspect_standard = 1.0 / 1.0
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_fps = 30

        # Prepare reference image
        image, _, _, shape_param = preprocess_image(
            image_path, mask_path=mask_path, intr=None, pad_ratio=0,
            bg_color=1., max_tgt_size=None, aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0], render_tgt_size=source_size,
            multiply=14, need_mask=True, get_shape_param=True
        )

        # Save masked image for vis
        save_ref_img_path = os.path.join(dump_tmp_dir, "output.png")
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype('uint8')
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # Prepare motion seq
        src = image_path.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]

        motion_seq = prepare_motion_seqs(
            motion_seqs_dir, None, save_root=dump_tmp_dir, fps=render_fps,
            bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
            render_image_res=render_size, multiply=16,
            need_mask=motion_img_need_mask, vis_motion=vis_motion,
            shape_param=shape_param, test_sample=False, cross_id=False,
            src_driven=src_driven
        )

        # Start inference
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        device, dtype = "cuda", torch.float32

        print("Starting inference...")
        with torch.no_grad():
            res = self.lam.infer_single_view(
                image.unsqueeze(0).to(device, dtype), None, None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()},
                #image_name=image_name,
                #output_dir="output"
            )

        # Generate OAC export
        project_root = os.path.abspath(os.path.dirname(__file__))
        oac_dir = os.path.join(project_root, "output", "open_avatar_chat", base_iid)
        os.makedirs(oac_dir, exist_ok=True)

        saved_head_path = self.lam.renderer.flame_model.save_shaped_mesh(
            shape_param.unsqueeze(0).cuda(), fd=oac_dir
        )
        res['cano_gs_lst'][0].save_ply(
            os.path.join(oac_dir, "offset.ply"), rgb2sh=False, offset2xyz=True
        )

        generate_glb(
            input_mesh=Path(saved_head_path),
            template_fbx=Path("./assets/sample_oac/template_file.fbx"),
            output_glb=Path(os.path.join(oac_dir, "skin.glb")),
            blender_exec=Path(self.cfg.blender_path)
        )

        shutil.copy(
            src='./assets/sample_oac/animation.glb',
            dst=os.path.join(oac_dir, 'animation.glb')
        )
        os.remove(saved_head_path)

        # Create ZIP file
        output_zip_path = os.path.join(project_root, "output", "open_avatar_chat", base_iid + ".zip")
        if os.path.exists(output_zip_path):
            os.remove(output_zip_path)

        zip_base = os.path.splitext(output_zip_path)[0]
        shutil.make_archive(
            base_name=zip_base,
            format="zip",
            root_dir=os.path.dirname(oac_dir),
            base_dir=os.path.basename(oac_dir)
        )

        # Cleanup
        shutil.rmtree(oac_dir)

        return output_zip_path


def build_model(cfg):
    """Build and load LAM model"""
    model = ModelLAM(**cfg.model)
    resume = os.path.join(cfg.model_name, "model.safetensors")

    print("=" * 100)
    print("Loading pretrained weight from:", resume)

    if resume.endswith('safetensors'):
        ckpt = load_file(resume, device='cpu')
    else:
        ckpt = torch.load(resume, map_location='cpu')

    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f"WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.")
        else:
            print(f"WARN] unexpected param {k}: {v.shape}")

    print("Finish loading pretrained weight from:", resume)
    print("=" * 100)
    return model


def parse_configs(blender_path: Optional[str] = None):
    """Parse configuration from environment and files"""
    cfg = OmegaConf.create()

    # Load from environment variables
    model_name = os.environ.get("APP_MODEL_NAME", './model_zoo/lam_models/releases/lam/lam-20k/step_045500/')
    infer_config = os.environ.get("APP_INFER", './configs/inference/lam-20k-8gpu.yaml')

    # Blender path priority: argument > env variable > default
    if blender_path is None:
        blender_path = os.environ.get("BLENDER_PATH", "blender")

    cfg.model_name = model_name
    cfg.blender_path = blender_path

    # Load inference config
    cfg_infer = OmegaConf.load(infer_config)
    cfg.merge_with(cfg_infer)

    # Load training config for sizes
    if os.path.exists(infer_config):
        cfg_train = OmegaConf.load(infer_config)
        cfg.source_size = cfg_train.get('dataset', {}).get('source_image_res', 512)
        cfg.src_head_size = cfg_train.get('dataset', {}).get('src_head_size', 112)
        cfg.render_size = cfg_train.get('dataset', {}).get('render_image', {}).get('high', 512)

    cfg.motion_video_read_fps = 30
    cfg.setdefault("logger", "INFO")

    return cfg


# Initialize FastAPI app
app = FastAPI(title="Avatar Reconstruction API", version="1.0.0")

# Global variables for models
reconstructor: Optional[AvatarReconstructor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global reconstructor

    # Set environment variables
    os.environ.update({
        'APP_ENABLED': '1',
        'APP_MODEL_NAME': './model_zoo/lam_models/releases/lam/lam-20k/step_045500/',
        'APP_INFER': './configs/inference/lam-20k-8gpu.yaml',
        'APP_TYPE': 'infer.lam',
        'NUMBA_THREADING_LAYER': 'omp',
    })

    print("Loading models...")

    # Get blender path from args if provided
    blender_path = getattr(app.state, 'blender_path', None)
    cfg = parse_configs(blender_path=blender_path)

    print(f"Using Blender path: {cfg.blender_path}")

    # Build LAM model
    lam = build_model(cfg)
    lam.to('cuda')
    lam.eval()

    # Initialize FLAME tracker
    flametracking = FlameTrackingSingleImage(
        output_dir='output/tracking',
        alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
        vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
        human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
        facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
        detect_iris_landmarks=False
    )

    reconstructor = AvatarReconstructor(cfg, lam, flametracking)
    print("Models loaded successfully!")


@app.post("/reconstruct-avatar")
async def reconstruct_avatar(file: UploadFile = File(...)):
    """
    Reconstruct avatar from input image

    Args:
        file: Input image file (JPG, PNG)

    Returns:
        ZIP file containing reconstructed avatar files
    """
    if reconstructor is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as working_dir:
        # Save uploaded file
        input_path = os.path.join(working_dir, "input.png")
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            # Process image and generate ZIP
            output_zip_path = reconstructor.process_image(input_path, working_dir)

            # Return ZIP file
            return FileResponse(
                output_zip_path,
                media_type="application/zip",
                filename=f"{Path(file.filename).stem}_avatar.zip"
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    blender_path = "unknown"
    blender_exists = False

    if reconstructor is not None:
        blender_path = reconstructor.cfg.blender_path
        blender_exists = os.path.exists(blender_path) if blender_path != "blender" else None

    return {
        "status": "healthy",
        "models_loaded": reconstructor is not None,
        "blender_path": blender_path,
        "blender_exists": blender_exists
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Avatar Reconstruction FastAPI Service")
    parser.add_argument(
        "--blender_path",
        type=str,
        default=None,
        help="Path to Blender executable (e.g., ~/software/blender-4.0.2-linux-x64/blender)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    # Store blender_path in app state for access in startup event
    if args.blender_path:
        app.state.blender_path = args.blender_path
        print(f"Using Blender path from command line: {args.blender_path}")

    uvicorn.run(app, host=args.host, port=args.port)

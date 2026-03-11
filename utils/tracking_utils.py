"""
Shared utilities for object tracking scripts.

Contains SAM3 loading/inference, mesh loading, FoundationPose estimator
construction, camera intrinsics helpers, and visualization.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBJECT_DIR = PROJECT_ROOT / "object"
FP_DIR = PROJECT_ROOT / "FoundationPose"
sys.path.insert(0, str(FP_DIR))

import nvdiffrast.torch as dr
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed


def load_sam3(confidence: float = 0.5):
    """
    Load SAM3 model and processor for text-prompted segmentation.

    Args:
        confidence (float): Detection confidence threshold.

    Returns:
        tuple: (model, processor) ready for inference.
    """
    # Reason: the project root contains a sam3/ directory (the git repo) that
    # Python finds as a namespace package, shadowing the real editable-installed
    # sam3 package. Temporarily strip conflicting paths and clear any cached
    # namespace entry so the real package in site-packages is resolved.
    _sam3_shadow = str(PROJECT_ROOT / "sam3")
    _conflicting = []
    for p in list(sys.path):
        resolved = str(Path(p).resolve()) if p else str(PROJECT_ROOT)
        if Path(resolved) == PROJECT_ROOT or Path(resolved) == Path(_sam3_shadow):
            _conflicting.append(p)
            sys.path.remove(p)

    for key in [k for k in sys.modules if k == "sam3" or k.startswith("sam3.")]:
        del sys.modules[key]

    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    for p in _conflicting:
        if p not in sys.path:
            sys.path.append(p)

    logging.info("Loading SAM3 model (this may take a moment on first run)...")
    model = build_sam3_image_model(
        device="cuda",
        eval_mode=True,
        enable_segmentation=True,
    )
    processor = Sam3Processor(model, confidence_threshold=confidence)
    logging.info("SAM3 model loaded")
    return model, processor


def get_sam3_mask(
    processor,
    color_rgb: np.ndarray,
    object_name: str,
) -> Optional[np.ndarray]:
    """
    Use SAM3 text-prompted segmentation to find and segment the object.

    Args:
        processor: Sam3Processor instance.
        color_rgb (np.ndarray): RGB image (H, W, 3), uint8.
        object_name (str): Text prompt for the object to detect.

    Returns:
        Optional[np.ndarray]: Binary mask (H, W) as uint8 (0 or 1),
                              or None if no detection.
    """
    from PIL import Image

    pil_image = Image.fromarray(color_rgb)
    inference_state = processor.set_image(pil_image)
    output = processor.set_text_prompt(state=inference_state, prompt=object_name)

    masks = output["masks"]
    scores = output["scores"]

    if masks is None or len(masks) == 0:
        return None

    # Reason: pick highest-confidence detection if multiple instances found
    best_idx = torch.argmax(scores).item()
    mask_np = masks[best_idx, 0].cpu().numpy().astype(np.uint8)

    if mask_np.sum() == 0:
        return None

    # Reason: CUDA kernels may still be in-flight; synchronize before
    # returning to caller so cv2.imshow doesn't race with the GPU.
    torch.cuda.synchronize()

    logging.info(
        f"SAM3 detected '{object_name}' with score {scores[best_idx]:.3f}, "
        f"mask pixels: {mask_np.sum()}"
    )
    return mask_np


def load_mesh(object_name: str) -> Tuple[str, Path]:
    """
    Locate the .obj mesh file for the given object.

    Args:
        object_name (str): Name matching a folder in object/.

    Returns:
        tuple: (mesh_path_str, mesh_dir) for the object.

    Raises:
        SystemExit: If object directory or mesh file not found.
    """
    mesh_dir = OBJECT_DIR / object_name
    if not mesh_dir.exists():
        logging.error(f"Object directory not found: {mesh_dir}")
        available = [
            d.name for d in OBJECT_DIR.iterdir() if d.is_dir()
        ]
        logging.info(f"Available objects: {available}")
        sys.exit(1)

    mesh_files = list(mesh_dir.glob("*.obj"))
    if not mesh_files:
        logging.error(f"No .obj file found in {mesh_dir}")
        sys.exit(1)

    mesh_path = mesh_files[0]
    logging.info(f"Using mesh: {mesh_path}")
    return str(mesh_path), mesh_dir


def build_estimator(
    mesh_path: str,
    debug_dir: str = "/tmp/fp_debug",
    est_refine_iter: int = 2,
    track_refine_iter: int = 2,
    debug: int = 0,
) -> Tuple[FoundationPose, trimesh.Trimesh, np.ndarray, np.ndarray]:
    """
    Build the FoundationPose estimator from a mesh file.

    Args:
        mesh_path (str): Path to the .obj mesh file.
        debug_dir (str): Directory for debug output.
        est_refine_iter (int): Refinement iterations for registration.
        track_refine_iter (int): Refinement iterations for tracking.
        debug (int): Debug level (0=off, 1=basic, 2=detailed).

    Returns:
        tuple: (estimator, mesh, to_origin, bbox).
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.vertices = mesh.vertices.astype(np.float32)
    mesh.vertex_normals = mesh.vertex_normals.astype(np.float32)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3).astype(np.float32)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    est.est_refine_iter = est_refine_iter
    est.track_refine_iter = track_refine_iter

    logging.info("FoundationPose estimator ready")
    return est, mesh, to_origin, bbox


def load_camera_serial(name: Optional[str] = None) -> Optional[str]:
    """
    Look up a RealSense serial number from camera_config.yaml.

    Args:
        name (Optional[str]): Camera name (e.g. ``"robotcam"``).
            If None, returns None immediately (use any available camera).

    Returns:
        Optional[str]: Serial number string, or None to use any camera.
    """
    if not name:
        return None

    config_path = PROJECT_ROOT / "camera_config.yaml"
    if not config_path.exists():
        logging.warning(
            f"camera_config.yaml not found — cannot look up '{name}'. "
            "Using any available camera."
        )
        return None

    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cameras = cfg.get("cameras", {})
    if name not in cameras:
        available = list(cameras.keys())
        logging.error(f"Camera '{name}' not in config. Available: {available}")
        return None

    serial = cameras[name]["serial"]
    logging.info(f"Using camera '{name}' (serial {serial})")
    return serial


def intrinsics_to_K(intr) -> np.ndarray:
    """
    Convert RealSense intrinsics to a 3x3 camera matrix.

    Args:
        intr: pyrealsense2 intrinsics object.

    Returns:
        np.ndarray: 3x3 camera intrinsic matrix.
    """
    return np.array(
        [
            [float(intr.fx), 0.0, float(intr.ppx)],
            [0.0, float(intr.fy), float(intr.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def print_pose(pose: np.ndarray, object_name: str) -> None:
    """
    Print pose to console.

    Args:
        pose (np.ndarray): 4x4 pose matrix.
        object_name (str): Name of the tracked object.
    """
    from scipy.spatial.transform import Rotation as R

    t = pose[:3, 3]
    quat = R.from_matrix(pose[:3, :3]).as_quat()
    logging.info(
        f"[{object_name}] pos=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) "
        f"quat=({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})"
    )


def draw_tracking_vis(
    color_bgr: np.ndarray,
    pose: Optional[np.ndarray],
    to_origin: np.ndarray,
    bbox: np.ndarray,
    K: np.ndarray,
    initialized: bool,
    fps_val: float,
    object_name: str,
) -> np.ndarray:
    """
    Render the tracking overlay on a BGR image.

    Args:
        color_bgr (np.ndarray): BGR camera frame.
        pose (Optional[np.ndarray]): Current 4x4 pose, or None.
        to_origin (np.ndarray): Mesh-to-origin transform.
        bbox (np.ndarray): Bounding box corners (2, 3).
        K (np.ndarray): Camera intrinsics (3, 3).
        initialized (bool): Whether tracking is active.
        fps_val (float): Current FPS for display.
        object_name (str): Object name for HUD.

    Returns:
        np.ndarray: BGR image with overlay drawn.
    """
    vis_bgr = color_bgr.copy()
    if initialized and pose is not None:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        vis_rgb = draw_posed_3d_box(K, img=vis_rgb, ob_in_cam=center_pose, bbox=bbox)
        vis_rgb = draw_xyz_axis(
            vis_rgb,
            ob_in_cam=center_pose,
            scale=0.1,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    status = "TRACKING" if initialized else "DETECTING..."
    color_status = (0, 255, 0) if initialized else (0, 0, 255)
    cv2.putText(
        vis_bgr,
        f"FPS: {fps_val:.1f} | {status} | {object_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color_status,
        2,
    )
    return vis_bgr



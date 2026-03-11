#!/usr/bin/env python3
"""
Interactive SAM3 viewer with live RealSense feed.

Opens a window showing the camera feed. Type a text prompt in the terminal
to see SAM3 segment that object in real time. Change prompts on the fly
without restarting.

Usage:
    python scripts/sam3_view.py
    python scripts/sam3_view.py --confidence 0.3
"""

import argparse
import logging
import select
import sys
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image as PILImage, ImageTk

COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
]


def segment_all_prompts(processor, color_rgb: np.ndarray, prompts: List[str]) -> List[dict]:
    """
    Run SAM3 text-prompted segmentation for each prompt.

    Args:
        processor: Sam3Processor instance.
        color_rgb (np.ndarray): RGB image (H, W, 3), uint8.
        prompts (list[str]): Text prompts to segment.

    Returns:
        list[dict]: Per-prompt results with keys: prompt, mask, score, bbox.
    """
    import torch

    pil_image = PILImage.fromarray(color_rgb)
    results = []

    for prompt in prompts:
        inference_state = processor.set_image(pil_image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output["masks"]
        scores = output["scores"]
        boxes = output["boxes"]

        if masks is None or len(masks) == 0:
            results.append({"prompt": prompt, "mask": None, "score": 0.0, "bbox": None})
            continue

        best_idx = torch.argmax(scores).item()
        mask_np = masks[best_idx, 0].cpu().numpy().astype(np.uint8)
        score = float(scores[best_idx])
        bbox = boxes[best_idx].cpu().numpy().astype(int) if boxes is not None else None

        results.append({
            "prompt": prompt,
            "mask": mask_np if mask_np.sum() > 0 else None,
            "score": score,
            "bbox": bbox,
        })

    torch.cuda.synchronize()
    return results


def draw_segmentation_overlay(
    color_bgr: np.ndarray,
    results: List[dict],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Draw colored mask overlays, bounding boxes, and labels on the image.

    Args:
        color_bgr (np.ndarray): BGR camera frame.
        results (list[dict]): Segmentation results from segment_all_prompts.
        alpha (float): Overlay transparency (0=invisible, 1=opaque).

    Returns:
        np.ndarray: BGR image with overlays.
    """
    vis = color_bgr.copy()
    overlay = vis.copy()

    for i, res in enumerate(results):
        color = COLORS[i % len(COLORS)]
        prompt = res["prompt"]
        mask = res["mask"]
        score = res["score"]
        bbox = res["bbox"]

        if mask is None:
            continue

        overlay[mask > 0] = color

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)

        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label_y = max(y1 - 8, 20)
        else:
            ys, xs = np.where(mask > 0)
            label_y = max(int(ys.min()) - 8, 20)
            x1 = int(xs.min())

        label = f"{prompt} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 4), color, -1)
        cv2.putText(vis, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return vis


def check_stdin() -> str:
    """
    Non-blocking read from stdin (Unix only).

    Returns:
        str: The line read, or empty string if nothing available.
    """
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return ""


def main():
    """Interactive SAM3 viewer: live camera feed + text-prompted segmentation."""
    parser = argparse.ArgumentParser(
        description="Interactive SAM3 viewer with live RealSense camera"
    )
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="SAM3 detection confidence threshold (lower = more detections)",
    )
    parser.add_argument(
        "--camera", type=str, default=None,
        help="Camera name from camera_config.yaml (e.g. robotcam, tablecam). "
             "Default uses the 'default' key in the config.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # --- Tkinter window (no GPU conflict unlike OpenCV Qt5 backend) ---
    root = tk.Tk()
    root.title("SAM3 Viewer")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    canvas_label = tk.Label(root)
    canvas_label.pack()

    # Deferred imports: keep CUDA init after GUI init
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    logging.info("Loading SAM3 model...")
    model = build_sam3_image_model(
        device="cuda",
        eval_mode=True,
        enable_segmentation=True,
    )
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    logging.info("SAM3 ready")

    import pyrealsense2 as rs

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.tracking_utils import load_camera_serial

    pipeline = rs.pipeline()
    config = rs.config()
    serial = load_camera_serial(args.camera)
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    logging.info(f"RealSense started: {args.width}x{args.height}@{args.fps}fps")

    print("\n" + "=" * 50)
    print("  SAM3 Interactive Viewer")
    print("=" * 50)
    print("Type a prompt and press Enter to segment that object.")
    print("Comma-separate for multiple: cup, bottle, phone")
    print("Type 'clear' to remove all prompts.")
    print("Type 'quit' to exit. (or close the window)")
    print("=" * 50)
    print("prompt> ", end="", flush=True)

    current_prompts: List[str] = []
    fps_hist: deque = deque(maxlen=30)
    last_results: List[dict] = []
    tk_image = None  # prevent garbage collection

    def update_frame():
        """Grab a frame, optionally segment, and update the Tkinter display."""
        nonlocal current_prompts, last_results, tk_image

        # Non-blocking stdin check
        user_input = check_stdin()
        if user_input:
            if user_input.lower() == "quit":
                root.destroy()
                return
            elif user_input.lower() == "clear":
                current_prompts = []
                last_results = []
                print("[cleared all prompts]")
            else:
                current_prompts = [p.strip() for p in user_input.split(",") if p.strip()]
                print(f"[segmenting: {current_prompts}]")
            print("prompt> ", end="", flush=True)

        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError:
            root.after(33, update_frame)
            return

        color_frame = frames.get_color_frame()
        if not color_frame:
            root.after(33, update_frame)
            return

        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.time()

        if current_prompts:
            last_results = segment_all_prompts(processor, color_rgb, current_prompts)
            vis_bgr = draw_segmentation_overlay(color_bgr, last_results)
        else:
            vis_bgr = color_bgr.copy()
            last_results = []

        dt = time.time() - t0
        fps_hist.append(1.0 / dt if dt > 1e-4 else 0.0)
        fps_val = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0

        prompt_text = ", ".join(current_prompts) if current_prompts else "(no prompt)"
        cv2.putText(vis_bgr, f"FPS: {fps_val:.1f} | {prompt_text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not current_prompts:
            cv2.putText(vis_bgr, "Type a prompt in the terminal to start",
                        (10, args.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 255), 2, cv2.LINE_AA)

        y_offset = args.height - 50
        for i, res in enumerate(last_results):
            color = COLORS[i % len(COLORS)]
            status = f"{res['score']:.2f}" if res["mask"] is not None else "not found"
            cv2.putText(vis_bgr, f"{res['prompt']}: {status}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset -= 22

        # Convert BGR -> RGB for Tkinter display
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(vis_rgb)
        tk_image = ImageTk.PhotoImage(pil_img)
        canvas_label.configure(image=tk_image)

        root.after(1, update_frame)

    root.after(1, update_frame)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        logging.info("Shutdown complete")


if __name__ == "__main__":
    main()

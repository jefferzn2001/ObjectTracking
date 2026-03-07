#!/usr/bin/env python3
"""
Standalone 6-DoF object pose tracking (no ROS required).

Uses SAM3 for automatic object detection (text-prompted segmentation)
and FoundationPose for 6-DoF pose estimation and frame-to-frame tracking.
Prints pose (position + quaternion) to the console each frame.

Usage:
    python scripts/track_object.py --object cup
    python scripts/track_object.py --object can --confidence 0.4
"""

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# Reason: append (not insert) so site-packages sam3 is found before
# the project's sam3/ directory which would shadow it as a namespace package.
sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    """Main tracking loop: SAM3 detection -> FoundationPose tracking -> console output."""
    parser = argparse.ArgumentParser(
        description="6-DoF object tracking with SAM3 + FoundationPose (standalone)"
    )
    parser.add_argument(
        "--object", type=str, required=True,
        help="Object name (must match folder in object/)",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument(
        "--est_refine_iter", type=int, default=2,
        help="Refinement iterations for initial registration",
    )
    parser.add_argument(
        "--track_refine_iter", type=int, default=2,
        help="Refinement iterations for frame-to-frame tracking",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="SAM3 detection confidence threshold",
    )
    parser.add_argument("--debug", type=int, default=1, help="Debug level (0=off, 1=vis, 2=save)")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization window")
    args = parser.parse_args()

    display = None
    if not args.no_vis:
        from utils.display import TkDisplay
        display = TkDisplay(title="ObjectTracker")

    # Deferred imports: these trigger CUDA init via FoundationPose/PyTorch
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError(
            "pyrealsense2 not found. Install: pip install pyrealsense2"
        ) from exc

    from utils.tracking_utils import (
        build_estimator,
        draw_tracking_vis,
        get_sam3_mask,
        intrinsics_to_K,
        load_mesh,
        load_sam3,
        print_pose,
        set_logging_format,
        set_seed,
    )

    set_logging_format()
    set_seed(0)

    mesh_path, _ = load_mesh(args.object)

    debug_dir = f"/tmp/fp_debug/{args.object}"
    os.makedirs(debug_dir, exist_ok=True)
    est, mesh, to_origin, bbox = build_estimator(
        mesh_path=mesh_path,
        debug_dir=debug_dir,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        debug=args.debug,
    )

    _, sam_processor = load_sam3(confidence=args.confidence)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    K = intrinsics_to_K(intr)
    logging.info(
        f"RealSense started: {args.width}x{args.height}@{args.fps}fps, "
        f"depth_scale={depth_scale}"
    )

    pose = None
    initialized = False
    fps_hist: deque = deque(maxlen=30)

    # Reason: SAM3 detection is expensive (~200ms). Run it once to get the
    # initial mask, then hand off to FoundationPose for real-time tracking.
    logging.info(f"Tracking '{args.object}' — running SAM3 for initial detection...")

    # --- Phase 1: one-shot SAM3 detection for initial mask ---
    while not initialized:
        if display and display.closed:
            break

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

        mask = get_sam3_mask(sam_processor, color_rgb, args.object)
        if mask is not None and mask.sum() > 100:
            try:
                pose = est.register(
                    K=K,
                    rgb=color_rgb,
                    depth=depth,
                    ob_mask=mask.astype(bool),
                    iteration=args.est_refine_iter,
                )
                initialized = True
                logging.info("Initial pose registered — switching to FP tracking")
            except Exception as e:
                logging.warning(f"Registration failed: {e}")

        if display:
            vis_bgr = draw_tracking_vis(
                color_bgr, None, to_origin, bbox, K,
                False, 0.0, args.object,
            )
            key = display.show(vis_bgr)
            if key in ("q", "Escape"):
                return

    # --- Phase 2: real-time FoundationPose tracking (no SAM3) ---
    logging.info("Entering real-time tracking loop (SAM3 is no longer running)")

    try:
        while True:
            if display and display.closed:
                break

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            t0 = time.time()

            try:
                pose = est.track_one(
                    rgb=color_rgb,
                    depth=depth,
                    K=K,
                    iteration=args.track_refine_iter,
                )
            except Exception as e:
                logging.warning(f"Tracking failed: {e}")
                initialized = False
                pose = None
                continue

            print_pose(pose, args.object)

            dt = time.time() - t0
            fps_hist.append(1.0 / dt if dt > 1e-4 else 0.0)
            fps_val = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0

            if display:
                vis_bgr = draw_tracking_vis(
                    color_bgr, pose, to_origin, bbox, K,
                    initialized, fps_val, args.object,
                )
                key = display.show(vis_bgr)
                if key in ("q", "Escape"):
                    break
                if key == "r":
                    initialized = False
                    pose = None
                    logging.info("Tracking reset — waiting for SAM3 re-detection")

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        pipeline.stop()
        if display:
            display.destroy()
        logging.info("Shutdown complete")


if __name__ == "__main__":
    main()

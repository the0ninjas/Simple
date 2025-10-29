#!/usr/bin/env python3
"""
AU Heatmap Visualizer for OpenFace-3.0
Real-time visualization of Action Unit intensities as heatmap overlays on facial regions.

This script combines face detection, landmark detection, and multitask prediction
to create anatomically meaningful heatmap visualizations of facial muscle activity.
"""

import sys
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import torch


# Configuration
OPENFACE_ROOT = Path(__file__).resolve().parent / "OpenFace-3.0"
WEIGHTS_DIR = OPENFACE_ROOT / "weights"

# AU mapping configuration
DEFAULT_AU_INDEX_TO_NUM: List[int] = [1, 2, 4, 6, 9, 12, 25, 26]

# 68-point landmark regions (0-based indices)
LM68_REGIONS = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_lip": list(range(48, 60)),
    "inner_lip": list(range(60, 68)),
}

# AU-to-landmark mapping for 68-point layout
LM68_AU_TO_IDXS: Dict[int, List[int]] = {
    1: LM68_REGIONS["right_eyebrow"][2:] + LM68_REGIONS["left_eyebrow"][:3],           # AU01 Inner brow raiser
    2: LM68_REGIONS["right_eyebrow"][:2] + LM68_REGIONS["left_eyebrow"][3:],           # AU02 Outer brow raiser
    4: LM68_REGIONS["right_eyebrow"] + LM68_REGIONS["left_eyebrow"],                   # AU04 Brow lowerer
    5: LM68_REGIONS["right_eye"] + LM68_REGIONS["left_eye"],                            # AU05 Upper lid raiser
    6: LM68_REGIONS["right_eye"] + LM68_REGIONS["left_eye"] + LM68_REGIONS["outer_lip"][:3] + LM68_REGIONS["outer_lip"][-3:],  # AU06 Cheek raiser
    7: LM68_REGIONS["right_eye"] + LM68_REGIONS["left_eye"],                            # AU07 Lid tightener
    9: LM68_REGIONS["nose"],                                                              # AU09 Nose wrinkler
    10: LM68_REGIONS["outer_lip"][48-48:54-48+1] + LM68_REGIONS["inner_lip"][61-60:64-60],  # AU10 Upper lip raiser
    12: [48, 54],                                                                          # AU12 Lip corner puller
    14: [48, 54],                                                                          # AU14 Dimpler
    15: [48, 54],                                                                          # AU15 Lip corner depressor
    17: LM68_REGIONS["inner_lip"][62-60:66-60+1],                                         # AU17 Chin raiser
    20: [48, 54] + LM68_REGIONS["outer_lip"][49-48:53-48+1],                              # AU20 Lip stretcher
    23: LM68_REGIONS["outer_lip"],                                                        # AU23 Lip tightener
    24: LM68_REGIONS["outer_lip"],                                                        # AU24 Lip pressor
    25: LM68_REGIONS["inner_lip"],                                                        # AU25 Lips part
    26: LM68_REGIONS["inner_lip"] + LM68_REGIONS["jaw"][6:11],                           # AU26 Jaw drop
    28: LM68_REGIONS["inner_lip"],                                                        # AU28 Lip suck
    45: LM68_REGIONS["right_eye"] + LM68_REGIONS["left_eye"],                            # AU45 Blink
}

# AU names for display
AU_NAMES = {
    1: "Inner Brow Raiser",
    2: "Outer Brow Raiser", 
    4: "Brow Lowerer",
    5: "Upper Lid Raiser",
    6: "Cheek Raiser",
    7: "Lid Tightener",
    8: "Lips Toward Each Other",
    9: "Nose Wrinkler",
    10: "Upper Lip Raiser",
    12: "Lip Corner Puller (Smile)",
    14: "Dimpler",
    15: "Lip Corner Depressor",
    17: "Chin Raiser",
    20: "Lip Stretcher",
    23: "Lip Tightener",
    24: "Lip Pressor",
    25: "Lips Part",
    26: "Jaw Drop",
    28: "Lip Suck",
    45: "Blink (Eye Closure)"
}


def ensure_openface_import(openface_root: Path) -> None:
    """Ensure OpenFace-3.0 modules are importable."""
    try:
        import openface  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    if str(openface_root) not in sys.path:
        sys.path.insert(0, str(openface_root))
    import openface  # type: ignore  # noqa: F401


def select_device() -> str:
    """Select best available device."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    return "cpu"


def resolve_weights(weights_dir: Path) -> Tuple[Path, Path, Path]:
    """Resolve required weight paths with fallbacks."""
    face_primary = weights_dir / "Alignment_RetinaFace.pth"
    mtl_primary = weights_dir / "MTL_backbone.pth"
    lm_primary = weights_dir / "Landmark_98.pkl"

    face_path: Optional[Path] = face_primary if face_primary.exists() else None
    mtl_path: Optional[Path] = mtl_primary if mtl_primary.exists() else None
    lm_path: Optional[Path] = lm_primary if lm_primary.exists() else None

    # Face detector fallback
    if face_path is None:
        alt_face = weights_dir / "mobilenet0.25_Final.pth"
        if alt_face.exists():
            face_path = alt_face
            print(f"[INFO] Using fallback face weights: {alt_face.name}")

    # Multitask model fallback
    if mtl_path is None:
        stage_candidates = sorted(weights_dir.glob("stage*_loss_*_acc_*.pth"))
        if stage_candidates:
            best = None
            best_loss: Optional[float] = None
            for p in stage_candidates:
                parts = p.stem.split('_')
                try:
                    if 'loss' in parts:
                        idx = parts.index('loss')
                        loss_val = float(parts[idx + 1])
                        if best_loss is None or loss_val < best_loss:
                            best_loss = loss_val
                            best = p
                except Exception:
                    continue
            if best is None:
                best = stage_candidates[0]
            mtl_path = best
            print(f"[INFO] Using fallback multitask weights: {best.name}")

    # Landmark model fallback
    if lm_path is None:
        alt_lm = weights_dir / "Landmark_68.pkl"
        if alt_lm.exists():
            lm_path = alt_lm
            print(f"[INFO] Using fallback landmark weights: {alt_lm.name}")

    missing: List[Path] = []
    if face_path is None:
        missing.append(face_primary)
    if mtl_path is None:
        missing.append(mtl_primary)
    if lm_path is None:
        missing.append(lm_primary)
    
    if missing:
        details = "\n".join(str(m) for m in missing)
        available = "\n".join(str(p.name) for p in weights_dir.glob("*.pth")) or "<none>"
        raise FileNotFoundError(
            f"Missing required weight file(s) and no suitable fallbacks found:\n{details}\n"
            f"Searched directory: {weights_dir}\nAvailable .pth files:\n{available}\n"
            f"You can: (1) place correct files, (2) rename existing compatible weights, or (3) update script."
        )

    return face_path, mtl_path, lm_path


def au_to_intensity(au_values: np.ndarray, clamp: bool = True) -> np.ndarray:
    """Convert AU values to intensity range [0, 1]."""
    vals = au_values.astype(np.float32)
    if clamp:
        return np.clip(vals, 0.0, 1.0)
    else:
        # Rescale based on percentiles
        vmin = np.percentile(vals, 5)
        vmax = np.percentile(vals, 95)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        return np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)


def gaussian_heatmap_overlay(image: np.ndarray, points: np.ndarray, 
                           weights: Optional[np.ndarray] = None, 
                           sigma: int = 8, alpha: float = 0.65) -> np.ndarray:
    """Create gaussian heatmap overlay on image."""
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    
    if weights is None:
        weights = np.ones((points.shape[0],), dtype=np.float32)
    
    # Draw weighted circles at landmark points
    for (x, y), wgt in zip(points.astype(int), weights.astype(np.float32)):
        if 0 <= x < w and 0 <= y < h and wgt > 0:
            cv2.circle(heat, (int(x), int(y)), max(1, sigma // 2), float(wgt), -1)
    
    # Apply Gaussian blur for smooth heatmap
    ksize = max(3, sigma * 4 + 1)
    heat = cv2.GaussianBlur(heat, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    heat = np.clip(heat / (heat.max() + 1e-6), 0.0, 1.0)
    
    # Convert to color map and overlay
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1.0, heat_color, alpha, 0)


def compute_landmark_weights(landmarks: np.ndarray, au_intensity: np.ndarray, 
                           au_index_to_num: List[int], selected_au_num: Optional[int]) -> np.ndarray:
    """Map AU intensities to landmark weights."""
    num_points = landmarks.shape[0]
    weights = np.zeros((num_points,), dtype=np.float32)
    
    if au_intensity.size == 0:
        return weights
    
    # Check if using 68-point layout
    is_68 = (num_points == 68)
    
    if selected_au_num is None:
        # Blend all AUs using their intensities
        if is_68:
            for au_idx, au_num in enumerate(au_index_to_num[:au_intensity.size]):
                v = float(au_intensity[au_idx])
                idxs = LM68_AU_TO_IDXS.get(au_num)
                if idxs:
                    weights[idxs] += v
            if weights.max() > 0:
                weights /= (weights.max() + 1e-6)
        else:
            weights[:] = float(np.mean(au_intensity))
    else:
        # Single AU focus
        try:
            if is_68:
                idxs = LM68_AU_TO_IDXS.get(selected_au_num)
                if idxs:
                    if selected_au_num in au_index_to_num:
                        au_idx = au_index_to_num.index(selected_au_num)
                        v = float(au_intensity[au_idx]) if au_idx < au_intensity.size else 0.0
                    else:
                        v = 0.0
                    weights[idxs] = v
                else:
                    weights[:] = float(np.mean(au_intensity))
            else:
                weights[:] = float(np.mean(au_intensity))
        except Exception:
            weights[:] = float(np.mean(au_intensity))
        
        if weights.max() > 0:
            weights /= (weights.max() + 1e-6)
    
    return weights


def draw_info_overlay(image: np.ndarray, au_intensity: np.ndarray, 
                     selected_au_num: Optional[int], clamp: bool, 
                     sigma: int, inference_time: float) -> None:
    """Draw information overlay on image."""
    # AU focus info
    if selected_au_num is None:
        au_focus = "ALL AUs"
        au_name = ""
    else:
        au_focus = f"AU{selected_au_num:02d}"
        au_name = AU_NAMES.get(selected_au_num, "")
    
    # Draw text overlays
    y_pos = 20
    cv2.putText(image, f"Focus: {au_focus} {au_name}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y_pos += 20
    
    cv2.putText(image, f"Mode: {'CLAMP' if clamp else 'RESCALE'}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y_pos += 20
    
    cv2.putText(image, f"Sigma: {sigma}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y_pos += 20
    
    cv2.putText(image, f"Infer: {inference_time:.1f}ms", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    y_pos += 20
    
    if au_intensity.size > 0:
        mean_intensity = float(np.mean(au_intensity))
        cv2.putText(image, f"Mean AU: {mean_intensity:.2f}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def print_help():
    """Print keyboard controls help."""
    print("\n" + "="*60)
    print("AU HEATMAP VISUALIZER - KEYBOARD CONTROLS")
    print("="*60)
    print("Numbers 1-9: Focus on specific AU (1=AU01, 2=AU02, etc.)")
    print("0: Show all AUs blended together")
    print("m: Toggle clamp/rescale normalization mode")
    print("p: Toggle landmark points overlay")
    print("+/=: Increase heatmap blur (sigma)")
    print("-/_: Decrease heatmap blur (sigma)")
    print("h: Show this help")
    print("q/Esc: Exit")
    print("="*60)


def main() -> None:
    """Main function."""
    ensure_openface_import(OPENFACE_ROOT)
    if not OPENFACE_ROOT.exists():
        raise RuntimeError(f"OPENFACE_ROOT not found: {OPENFACE_ROOT}")

    from openface.face_detection import FaceDetector  # type: ignore
    from openface.multitask_model import MultitaskPredictor  # type: ignore
    from openface.landmark_detection import LandmarkDetector  # type: ignore

    device = select_device()
    face_w, mtl_w, lm_w = resolve_weights(WEIGHTS_DIR)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera index 0")

    # Initialize models
    old_cwd = os.getcwd()
    os.chdir(str(OPENFACE_ROOT))
    try:
        face_detector = FaceDetector(model_path=str(face_w), device=device)
        mtl = MultitaskPredictor(model_path=str(mtl_w), device=device)
        try:
            landmark_detector = LandmarkDetector(model_path=str(lm_w), device=device)
        except ValueError:
            landmark_detector = LandmarkDetector(model_path=str(lm_w), device="cpu")
    finally:
        os.chdir(old_cwd)

    # Visualization parameters
    clamp = True
    show_points = False
    selected_au_num: Optional[int] = None
    sigma = 8
    
    print_help()
    print(f"\n[INFO] Starting AU heatmap visualization on device: {device}")
    print("[INFO] Press 'h' for help, 'q' to exit")
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Face detection
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            try:
                cv2.imwrite(tmp_path, frame)
                start_time = time.perf_counter()
                cropped_face, dets = face_detector.get_face(tmp_path)
                inference_time = (time.perf_counter() - start_time) * 1000.0
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            vis = frame.copy()
            if dets is not None and len(dets) > 0 and cropped_face is not None:
                # Get AU predictions
                with torch.no_grad():
                    emotion_logits, gaze_output, au_output = mtl.predict(cropped_face)

                # Convert AU outputs to intensities
                try:
                    au_vals = au_output.detach().cpu().numpy().reshape(-1)
                except Exception:
                    au_vals = np.asarray(au_output).reshape(-1)
                
                au_intensity = au_to_intensity(au_vals, clamp)

                # Get landmarks
                try:
                    lm_list = landmark_detector.detect_landmarks(frame, dets)
                except Exception:
                    lm_list = None

                if lm_list:
                    lm = lm_list[0]
                    # Compute landmark-localized weights from AU intensities
                    au_index_to_num = DEFAULT_AU_INDEX_TO_NUM[:au_intensity.size]
                    weights = compute_landmark_weights(lm, au_intensity, au_index_to_num, selected_au_num)
                    
                    # Create heatmap overlay
                    vis = gaussian_heatmap_overlay(vis, lm, weights=weights, sigma=sigma, alpha=0.6)
                    
                    # Draw landmark points if enabled
                    if show_points:
                        for (x, y) in lm.astype(int):
                            cv2.circle(vis, (int(x), int(y)), 1, (255, 255, 255), -1)

                # Draw face bounding box
                try:
                    x1, y1, x2, y2 = dets[0][:4].astype(int)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                except Exception:
                    pass

                # Draw information overlay
                draw_info_overlay(vis, au_intensity, selected_au_num, clamp, sigma, inference_time)

            cv2.imshow("AU Heatmap Visualizer", vis)
            key = cv2.waitKey(1) & 0xFF
            
            # Handle keyboard input
            if key in (ord('q'), 27):  # Quit
                break
            elif key in (ord('h'), ord('H')):  # Help
                print_help()
            elif key in (ord('m'), ord('M')):  # Toggle clamp/rescale
                clamp = not clamp
                print(f"[INFO] Mode: {'CLAMP' if clamp else 'RESCALE'}")
            elif key in (ord('p'), ord('P')):  # Toggle points
                show_points = not show_points
                print(f"[INFO] Landmark points: {'ON' if show_points else 'OFF'}")
            elif key == ord('0'):  # Show all AUs
                selected_au_num = None
                print("[INFO] Showing all AUs blended")
            elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), 
                        ord('6'), ord('7'), ord('8'), ord('9')):  # Select specific AU
                idx = int(chr(key)) - 1
                if 0 <= idx < len(DEFAULT_AU_INDEX_TO_NUM):
                    selected_au_num = DEFAULT_AU_INDEX_TO_NUM[idx]
                    au_name = AU_NAMES.get(selected_au_num, "")
                    print(f"[INFO] Focusing on AU{selected_au_num:02d}: {au_name}")
            elif key in (ord('+'), ord('=')):  # Increase sigma
                sigma = min(31, sigma + 1)
                print(f"[INFO] Sigma: {sigma}")
            elif key in (ord('-'), ord('_')):  # Decrease sigma
                sigma = max(1, sigma - 1)
                print(f"[INFO] Sigma: {sigma}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] AU Heatmap Visualizer stopped")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Real-time FAU Gaussian heatmap overlay using OpenFace-3.0.

This script:
- Captures frames from the default webcam
- Uses OpenFace-3.0 for face detection, landmark detection (98 pts), and AU prediction
- Maps AU intensities to facial regions using AU landmark maps and overlays weighted Gaussian heatmaps

Keyboard:
- q / Esc: quit
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


# Paths
ROOT_DIR = Path(__file__).resolve().parent
OPENFACE_ROOT = ROOT_DIR / "OpenFace-3.0"
OF_WEIGHTS = OPENFACE_ROOT / "weights"
ROOT_WEIGHTS = ROOT_DIR / "weights"

# Names for common AUs used by the multitask model
AU_NAMES: Dict[int, str] = {
    1: "Inner Brow Raiser",
    2: "Outer Brow Raiser",
    4: "Brow Lowerer",
    6: "Cheek Raiser",
    9: "Nose Wrinkler",
    12: "Lip Corner Puller",
    25: "Lips Part",
    26: "Jaw Drop",
}


def ensure_openface_import(openface_root: Path) -> None:
    try:
        import openface  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    if str(openface_root) not in sys.path:
        sys.path.insert(0, str(openface_root))
    import openface  # type: ignore  # noqa: F401


def select_device() -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    return "cpu"


def resolve_weight(candidates: List[str], search_dirs: List[Path]) -> Optional[Path]:
    for d in search_dirs:
        for name in candidates:
            p = d / name
            if p.exists():
                return p
    return None


def resolve_model_paths() -> Tuple[Path, Path, Path]:
    """Resolve face detector, multitask AU, and 68-point landmark weights.

    Preference order:
    - Face: Alignment_RetinaFace.pth → mobilenet0.25_Final.pth
    - AU: MTL_backbone.pth → best stage*_loss_*.pth (lowest loss)
    - Landmarks: Landmark_68.pkl → Landmark_98.pkl
    """
    # Face
    face_path = resolve_weight(
        ["Alignment_RetinaFace.pth", "mobilenet0.25_Final.pth"],
        [OF_WEIGHTS, ROOT_WEIGHTS],
    )

    # Multitask AU
    mtl_path = resolve_weight(["MTL_backbone.pth"], [ROOT_WEIGHTS, OF_WEIGHTS])
    if mtl_path is None:
        # Fallback to best stage checkpoint in OpenFace weights
        stage_candidates = sorted(OF_WEIGHTS.glob("stage*_loss_*_acc_*.pth"))
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

    # Landmarks (prefer 68 to match AU_LANDMARK_MAP indexing)
    lm_path = resolve_weight(["Landmark_68.pkl"], [OF_WEIGHTS, ROOT_WEIGHTS])
    if lm_path is None:
        lm_path = resolve_weight(["Landmark_98.pkl"], [OF_WEIGHTS, ROOT_WEIGHTS])

    missing: List[str] = []
    if face_path is None:
        missing.append("face weights (Alignment_RetinaFace.pth or mobilenet0.25_Final.pth)")
    if mtl_path is None:
        missing.append("multitask AU weights (MTL_backbone.pth or stage*_loss_*.pth)")
    if lm_path is None:
        missing.append("landmark weights (Landmark_68.pkl or Landmark_98.pkl)")
    if missing:
        raise FileNotFoundError("Missing required weights: " + ", ".join(missing))

    return face_path, mtl_path, lm_path


def add_gaussian(heatmap: np.ndarray, center: Tuple[float, float], amplitude: float, sigma: float) -> None:
    """Add a 2D Gaussian to heatmap at given center.

    The Gaussian is applied in a local window of radius ~3*sigma for efficiency.
    """
    h, w = heatmap.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return

    radius = int(max(3, 3.0 * sigma))
    x0 = max(0, int(cx) - radius)
    x1 = min(w, int(cx) + radius + 1)
    y0 = max(0, int(cy) - radius)
    y1 = min(h, int(cy) + radius + 1)

    if x1 <= x0 or y1 <= y0:
        return

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    # 2D Gaussian
    inv_two_sigma2 = 1.0 / (2.0 * (sigma ** 2) + 1e-6)
    g = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) * inv_two_sigma2)

    # Accumulate
    heatmap[y0:y1, x0:x1] += g.astype(np.float32)


def normalize_to_uint8(heatmap: np.ndarray) -> np.ndarray:
    hm = np.maximum(heatmap, 0.0)
    maxv = float(hm.max())
    if maxv <= 1e-8:
        return np.zeros_like(hm, dtype=np.uint8)
    norm = (hm / maxv) * 255.0
    return norm.clip(0, 255).astype(np.uint8)


def compute_region_centers(landmarks: np.ndarray, indices: List[int]) -> Optional[Tuple[float, float]]:
    pts = []
    h, w = None, None  # unused here, kept for clarity
    for idx in indices:
        if 0 <= idx < landmarks.shape[0]:
            pts.append(landmarks[idx])
    if not pts:
        return None
    pts_arr = np.asarray(pts, dtype=np.float32)
    mean = pts_arr.mean(axis=0)
    return float(mean[0]), float(mean[1])


def draw_au_bars(
    image: np.ndarray,
    au_values: np.ndarray,
    au_index_to_num: List[int],
    start_xy: Tuple[int, int] = (10, 60),
    bar_size: Tuple[int, int] = (360, 22),
    gap: int = 10,
) -> None:
    """Draw horizontal bars for AU intensities (assumes values in [0,1])."""
    x0, y0 = start_xy
    bar_w, bar_h = bar_size
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, au_num in enumerate(au_index_to_num):
        if idx >= au_values.size:
            break
        val = float(max(0.0, min(1.0, au_values[idx])))
        y = y0 + idx * (bar_h + gap)

        # Background rectangle (border)
        cv2.rectangle(image, (x0, y), (x0 + bar_w, y + bar_h), (60, 60, 60), 2)
        # Filled portion
        fill_w = int(bar_w * val)
        if fill_w > 0:
            cv2.rectangle(image, (x0 + 2, y + 2), (x0 + 2 + fill_w, y + bar_h - 2), (0, 220, 0), -1)

        # Text label to the right
        label_x = x0 + bar_w + 20
        label = f"AU{au_num:02d} {AU_NAMES.get(au_num, '')}: {val:.2f}"
        cv2.putText(image, label, (label_x, y + bar_h - 4), font, 0.6, (230, 230, 230), 1, cv2.LINE_AA)


def main() -> None:
    ensure_openface_import(OPENFACE_ROOT)

    # Import OpenFace modules after sys.path adjustment
    from openface.face_detection import FaceDetector  # type: ignore
    from openface.landmark_detection import LandmarkDetector  # type: ignore
    from openface.multitask_model import MultitaskPredictor  # type: ignore

    # Import AU landmark map from project root (use 98-point map)
    try:
        from au_landmark_map import AU_LANDMARK_MAP_98 as AU_LANDMARK_MAP  # type: ignore
    except Exception:
        # Best-effort import via sys.path tweak
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))
        from au_landmark_map import AU_LANDMARK_MAP_98 as AU_LANDMARK_MAP  # type: ignore

    device = select_device()
    face_w, mtl_w, _ = resolve_model_paths()
    # Use the user-provided 98-point landmark weights
    lm_w = Path(r"C:\Users\ARCLP\Documents\Code\openface_demo\weights\Landmark_98.pkl")
    if not lm_w.exists():
        raise FileNotFoundError(f"Landmark weights not found at: {lm_w}")
    print(f"[INFO] Using landmark weights: {lm_w}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera index 0")

    # Initialize models (many OpenFace modules expect cwd at repo root)
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

    # AU setup: mapping from model output indices to AU numbers
    # This mirrors au_heatmap_visualizer.DEFAULT_AU_INDEX_TO_NUM
    AU_INDEX_TO_NUM: List[int] = [1, 2, 4, 6, 9, 12, 25, 26]
    AU_THRESHOLD = 0
    BASE_SIGMA = 10.0  # pixels, scaled by intensity
    MIN_SIGMA = 6.0

    print(f"[INFO] Starting FAU heatmap on device: {device}")
    print("[INFO] Press 'q' or Esc to quit; 'p' toggles landmark points")

    last_face_ms = 0.0
    show_points = False
    last_landmarks: Optional[np.ndarray] = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            vis = frame.copy()
            h, w = frame.shape[:2]
            cumulative = np.zeros((h, w), dtype=np.float32)
            # Reset per-frame predictors for HUD
            au_vals = None

            # Face detection (FaceDetector expects path input)
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            dets = None
            cropped_face = None
            try:
                cv2.imwrite(tmp_path, frame)
                t0 = time.perf_counter()
                cropped_face, dets = face_detector.get_face(tmp_path)
                last_face_ms = (time.perf_counter() - t0) * 1000.0
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if dets is not None and len(dets) > 0 and cropped_face is not None:
                # AU prediction
                with torch.no_grad():
                    _, _, au_output = mtl.predict(cropped_face)
                try:
                    au_vals = au_output.detach().cpu().numpy().reshape(-1)
                except Exception:
                    au_vals = np.asarray(au_output).reshape(-1)

                # Landmarks
                try:
                    lm_list = landmark_detector.detect_landmarks(frame, dets)
                except Exception:
                    lm_list = None

                if lm_list:
                    lm = lm_list[0]
                    last_landmarks = lm
                    num_points = lm.shape[0]

                    # Using 98-point landmarks with AU_LANDMARK_MAP (98); bounds are still checked

                    # Iterate AUs
                    for idx, au_num in enumerate(AU_INDEX_TO_NUM):
                        if idx >= au_vals.size:
                            break
                        intensity = float(au_vals[idx])
                        if intensity < AU_THRESHOLD:
                            continue

                        # Find mapping for this AU
                        au_key = f"AU{au_num}"
                        region_dict: Optional[Dict[str, List[int]]] = AU_LANDMARK_MAP.get(au_key)  # type: ignore
                        if not region_dict:
                            continue

                        # Amplitude and sigma scaled by intensity
                        amplitude = max(0.0, min(1.0, intensity))
                        sigma = MIN_SIGMA + (BASE_SIGMA - MIN_SIGMA) * amplitude

                        for side in ("left", "right", "centre"):
                            idxs = region_dict.get(side, [])
                            if not idxs:
                                continue
                            center = compute_region_centers(lm, idxs)
                            if center is None:
                                continue
                            add_gaussian(cumulative, center, amplitude=amplitude, sigma=sigma)

                    # Optional: draw face bbox
                    try:
                        x1, y1, x2, y2 = dets[0][:4].astype(int)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    except Exception:
                        pass

            # Normalize and overlay
            heat_uint8 = normalize_to_uint8(cumulative)
            heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(vis, 1.0, heat_color, 0.55, 0)

            # Optional: draw points overlay
            if show_points and last_landmarks is not None:
                for (x, y) in last_landmarks.astype(int):
                    cv2.circle(overlay, (int(x), int(y)), 1, (255, 255, 255), -1)

            # HUD (infer time, AU bars, help)
            cv2.putText(overlay, f"Infer: {last_face_ms:.1f} ms", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"AU thr: {AU_THRESHOLD:.2f}    Toggle points: p", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1, cv2.LINE_AA)

            # If we have current AU values and emotion logits, draw bars and emotion label
            current_au_vals = au_vals
            if current_au_vals is not None:
                draw_au_bars(overlay, current_au_vals, AU_INDEX_TO_NUM, start_xy=(10, 70), bar_size=(360, 22), gap=10)

            cv2.imshow("FAU Heatmap (AU_LANDMARK_MAP)", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key in (ord('p'), ord('P')):
                show_points = not show_points
                print(f"[INFO] Landmark points: {'ON' if show_points else 'OFF'}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



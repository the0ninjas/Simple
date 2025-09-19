import sys
import time
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from contextlib import contextmanager

# ---------------- Configuration (edit as needed, no CLI) ----------------
OPENFACE_ROOT = Path(__file__).resolve().parent / "OpenFace-3.0"
WEIGHTS_DIR = OPENFACE_ROOT / "weights"
DEVICE = "cpu"          # set to "cuda" if you have a GPU + torch with CUDA
CAMERA_INDEX = 0          # default webcam index
FRAME_SCALE = 1.0         # resize factor applied to incoming frames
DRAW_OVERLAYS = True      # draw visualization (bbox, AUs, emotion, gaze, landmarks)
MAX_AU_VIS = 25           # max AU bars to show
LOAD_LANDMARKS = True     # attempt to load & draw facial landmarks if model present
WINDOW_NAME = "OpenFace-3.0 AU Demo"
DRAW_GAZE = False         # whether to draw gaze overlay text
EXCLUDED_AUS = {3}        # AU numbers to hide from display

# ---------------- Utility ----------------

def ensure_openface_import(openface_root_hint: Optional[Path]) -> None:
    """Ensure OpenFace-3.0 modules are importable."""
    try:
        import openface  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    candidate_dirs: List[Path] = []
    if openface_root_hint is not None:
        candidate_dirs.append(openface_root_hint)
    candidate_dirs.append(Path(__file__).resolve().parent / "OpenFace-3.0")
    for candidate in candidate_dirs:
        if candidate.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            try:
                import openface  # type: ignore  # noqa: F401
                return
            except Exception:
                continue

@contextmanager
def temp_cwd(path: Path):
    old = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)

EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

# AU number to name mapping
AU_NAMES = {
    1: "Inner Brow Raiser",
    2: "Outer Brow Raiser", 
    4: "Brow Lowerer",
    5: "Upper Lid Raiser",
    6: "Cheek Raiser",
    7: "Lid Tightener",
    8: "Lips Toward Each Other",  # AU08
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

# Mapping from AU output index (0-based) to FACS AU number as used in demo2.py
# Model outputs 8 AU channels ordered as: [AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26]
AU_INDEX_TO_NUM = [1, 2, 4, 6, 9, 12, 25, 26]

# Display all available AUs (will be determined dynamically)
DISPLAY_ALL_AUS = True

# Fallback list for when DISPLAY_ALL_AUS is False
DISPLAY_AUS = [2, 5, 6]

# ---------------- Visualization ----------------

def draw_au_bars(image: np.ndarray, au_values: np.ndarray, origin_xy: Tuple[int, int] = (10, 30), max_count: int = 12) -> None:
    max_width = 200
    bar_height = 16
    gap = 6
    x0, y0 = origin_xy
    
    if DISPLAY_ALL_AUS:
        # Display AUs following the fixed mapping from output index -> AU number
        displayed_count = 0
        for idx, au_num in enumerate(AU_INDEX_TO_NUM):
            if idx >= len(au_values):
                break
            if au_num in EXCLUDED_AUS:
                continue
            if displayed_count >= max_count:
                break

            value = au_values[idx]
            clamped = float(np.clip(value, 0.0, 1.0))
            w = int(clamped * max_width)
            y = y0 + displayed_count * (bar_height + gap)

            # Draw the bar background and fill
            cv2.rectangle(image, (x0, y), (x0 + max_width, y + bar_height), (50, 50, 50), 1)
            cv2.rectangle(image, (x0, y), (x0 + w, y + bar_height), (0, 200, 0), -1)

            # Display AU number, name, and value
            mapped = AU_NAMES.get(au_num)
            label = f"AU{au_num:02d} {mapped}" if mapped else f"AU{au_num:02d}"
            text = f"{label}: {clamped:.2f}"
            cv2.putText(image, text, (x0 + max_width + 8, y + bar_height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            displayed_count += 1
    else:
        # Original behavior: display only specific AUs
        filtered = [n for n in DISPLAY_AUS if n not in EXCLUDED_AUS]
        for vis_idx, au_num in enumerate(filtered):
            if au_num in AU_INDEX_TO_NUM:
                src_idx = AU_INDEX_TO_NUM.index(au_num)
                if src_idx < len(au_values):
                    value = au_values[src_idx]
                    clamped = float(np.clip(value, 0.0, 1.0))
                    w = int(clamped * max_width)
                    y = y0 + vis_idx * (bar_height + gap)

                    cv2.rectangle(image, (x0, y), (x0 + max_width, y + bar_height), (50, 50, 50), 1)
                    cv2.rectangle(image, (x0, y), (x0 + w, y + bar_height), (0, 200, 0), -1)

                    mapped = AU_NAMES.get(au_num)
                    label = f"AU{au_num:02d} {mapped}" if mapped else f"AU{au_num:02d}"
                    text = f"{label}: {clamped:.2f}"
                    cv2.putText(image, text, (x0 + max_width + 8, y + bar_height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

# ---------------- Main ----------------

def resolve_required_weights() -> Tuple[Path, Path]:
    """Resolve required weight paths with fallbacks.

    Primary expected filenames:
      - Alignment_RetinaFace.pth (face detector)
      - MTL_backbone.pth (multitask predictor)

    Fallbacks if missing:
      - Face: use mobilenet0.25_Final.pth if present (common RetinaFace export)
      - Multitask: choose best stage*_loss_*.pth (pick lowest loss in filename) if present
    """
    face_primary = WEIGHTS_DIR / "Alignment_RetinaFace.pth"
    mtl_primary = WEIGHTS_DIR / "MTL_backbone.pth"

    face_path = face_primary if face_primary.exists() else None
    mtl_path = mtl_primary if mtl_primary.exists() else None

    # Face fallback
    if face_path is None:
        alt_face = WEIGHTS_DIR / "mobilenet0.25_Final.pth"
        if alt_face.exists():
            face_path = alt_face
            print(f"[INFO] Using fallback face detector weights: {alt_face.name}")

    # Multitask fallback: look for stage*.pth pattern and choose lowest loss
    if mtl_path is None:
        stage_candidates = sorted(WEIGHTS_DIR.glob("stage*_loss_*_acc_*.pth"))
        if stage_candidates:
            # Parse loss from filename pattern: stageX_epoch_Y_loss_<loss>_acc_<acc>.pth
            best = None
            best_loss = None
            for p in stage_candidates:
                parts = p.stem.split('_')
                # Attempt to find 'loss' token and parse next item as float
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

    missing: List[Path] = []
    if face_path is None:
        missing.append(face_primary)
    if mtl_path is None:
        missing.append(mtl_primary)

    if missing:
        details = "\n".join(str(m) for m in missing)
        available = "\n".join(str(p.name) for p in WEIGHTS_DIR.glob("*.pth")) or "<none>"
        raise FileNotFoundError(
            f"Missing required weight file(s) and no suitable fallbacks found:\n{details}\n"
            f"Searched directory: {WEIGHTS_DIR}\nAvailable .pth files:\n{available}\n"
            f"You can: (1) place correct files, (2) rename existing compatible weights, or (3) update script." )

    # Final paths (type ignore since None handled)
    return face_path, mtl_path  # type: ignore

def main() -> None:
    ensure_openface_import(OPENFACE_ROOT)
    if not OPENFACE_ROOT.exists():
        raise RuntimeError(f"OPENFACE_ROOT not found: {OPENFACE_ROOT}")
    if not WEIGHTS_DIR.exists():
        print(f"[WARN] Weights directory not found: {WEIGHTS_DIR}")

    from openface.face_detection import FaceDetector  # type: ignore
    from openface.multitask_model import MultitaskPredictor  # type: ignore

    landmark_detector = None
    if LOAD_LANDMARKS:
        lm_path = WEIGHTS_DIR / "Landmark_98.pkl"
        if lm_path.exists():
            try:
                from openface.landmark_detection import LandmarkDetector  # type: ignore
                landmark_detector = LandmarkDetector(model_path=str(lm_path), device=DEVICE)
            except Exception as e:
                print(f"[WARN] Could not init LandmarkDetector: {e}")
        else:
            print(f"[INFO] Landmark model not found at {lm_path} (skipping landmarks).")

    face_model_path, mtl_model_path = resolve_required_weights()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {CAMERA_INDEX}")

    # Some internal RetinaFace code expects current working dir to contain ./weights/...
    with temp_cwd(OPENFACE_ROOT):
        face_detector = FaceDetector(model_path=str(face_model_path), device=DEVICE)
        multitask_model = MultitaskPredictor(model_path=str(mtl_model_path), device=DEVICE)

    last_infer_ms = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] Camera frame not received, exiting.")
                break
            if FRAME_SCALE != 1.0:
                frame = cv2.resize(frame, None, fx=FRAME_SCALE, fy=FRAME_SCALE)

            # Temporary file path (current model API expects file path input)
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            try:
                cv2.imwrite(tmp_path, frame)
                start = time.perf_counter()
                cropped_face, dets = face_detector.get_face(tmp_path)
                last_infer_ms = (time.perf_counter() - start) * 1000.0
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            au_values = None
            emotion_label = None
            gaze_yaw = None
            gaze_pitch = None
            bbox = None
            landmarks_first = None

            if dets is not None and len(dets) > 0 and cropped_face is not None:
                with torch.no_grad():
                    emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)
                # AUs
                try:
                    if isinstance(au_output, torch.Tensor):
                        au_values = au_output.squeeze().detach().cpu().numpy()
                    else:
                        au_values = np.array(au_output).astype(np.float32)
                except Exception:
                    au_values = None
                # Emotion
                try:
                    if isinstance(emotion_logits, torch.Tensor):
                        idx = int(torch.argmax(emotion_logits if emotion_logits.ndim == 2 else emotion_logits, dim=-1).item())
                        if 0 <= idx < len(EMOTION_LABELS):
                            emotion_label = EMOTION_LABELS[idx]
                except Exception:
                    pass
                # Gaze
                try:
                    if isinstance(gaze_output, torch.Tensor):
                        g = gaze_output.squeeze().detach().cpu().numpy()
                        if g.size >= 2:
                            gaze_yaw, gaze_pitch = float(g[0]), float(g[1])
                except Exception:
                    pass
                # BBox
                try:
                    bbox = dets[0][:4].astype(int)
                except Exception:
                    bbox = None
                # Landmarks
                if landmark_detector is not None:
                    try:
                        lm_list = landmark_detector.detect_landmarks(frame, dets)
                        if lm_list:
                            landmarks_first = lm_list[0]
                    except Exception:
                        pass

            if DRAW_OVERLAYS:
                vis = frame.copy()
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                y_text = 20
                cv2.putText(vis, f"Infer: {last_infer_ms:.1f} ms", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                y_text += 20
                if emotion_label is not None:
                    cv2.putText(vis, f"Emotion: {emotion_label}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
                    y_text += 20
                if DRAW_GAZE and (gaze_yaw is not None and gaze_pitch is not None):
                    cv2.putText(vis, f"Gaze Y/P: {gaze_yaw:.1f}/{gaze_pitch:.1f}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1, cv2.LINE_AA)
                    y_text += 20
                if au_values is not None:
                    # Draw AU bars just below the emotion (or infer) text block
                    draw_au_bars(vis, au_values, origin_xy=(10, y_text + 10), max_count=MAX_AU_VIS)
                if landmarks_first is not None:
                    for (lx, ly) in landmarks_first.astype(int):
                        cv2.circle(vis, (int(lx), int(ly)), 1, (0, 0, 255), -1)
                cv2.imshow(WINDOW_NAME, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
            else:
                # If not drawing, still allow quit with 'q'
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
    finally:
        cap.release()
        if DRAW_OVERLAYS:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

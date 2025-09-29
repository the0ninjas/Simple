import csv
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch


EMOTION_LABELS = [
    "Neutral",
    "Happy",
    "Sad",
    "Surprise",
    "Fear",
    "Disgust",
    "Anger",
    "Contempt",
]

def ensure_openface_import() -> None:
    """Best-effort import of the packaged openface API."""
    try:
        import openface  # noqa: F401
        return
    except Exception as e:
        raise RuntimeError(
            "Failed to import 'openface'. Ensure OpenFace-3.0 is installed as a package."
        ) from e


def write_header(writer: csv.writer, au_dim: int, delimiter: str) -> None:
    columns = [
        "timestamp_unix",
        "timestamp_iso",
        "frame_index",
        "emotion_index",
        "emotion_label",
        "gaze_yaw",
        "gaze_pitch",
    ]
    columns.extend([f"au_{i+1:02d}" for i in range(au_dim)])
    writer.writerow(columns)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def resolve_weights(weights_dir: Path) -> Tuple[Path, Path]:
    """Resolve face and multitask weights with sensible fallbacks.

    Primary expected filenames:
      - Alignment_RetinaFace.pth (face detector)
      - MTL_backbone.pth (multitask predictor)

    Fallbacks if missing:
      - Face: mobilenet0.25_Final.pth
      - Multitask: choose stage*_loss_*_acc_*.pth with lowest loss in name
    """
    face_primary = weights_dir / "Alignment_RetinaFace.pth"
    mtl_primary = weights_dir / "MTL_backbone.pth"

    face_path: Optional[Path] = face_primary if face_primary.exists() else None
    mtl_path: Optional[Path] = mtl_primary if mtl_primary.exists() else None

    if face_path is None:
        alt_face = weights_dir / "mobilenet0.25_Final.pth"
        if alt_face.exists():
            face_path = alt_face
            print(f"[INFO] Using fallback face weights: {alt_face.name}")

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

    missing: List[Path] = []
    if face_path is None:
        missing.append(face_primary)
    if mtl_path is None:
        missing.append(mtl_primary)
    if missing:
        details = "\n".join(str(m) for m in missing)
        available = "\n".join(str(p.name) for p in weights_dir.glob("*.pth")) or "<none>"
        raise FileNotFoundError(
            f"Missing required weight file(s) and no suitable fallbacks found:\n{details}\n"
            f"Searched directory: {weights_dir}\nAvailable .pth files:\n{available}\n"
            f"You can: (1) place correct files, (2) rename existing compatible weights, or (3) update script." )

    return face_path, mtl_path


def main() -> None:
    ensure_openface_import()

    from openface.face_detection import FaceDetector  # type: ignore
    from openface.multitask_model import MultitaskPredictor  # type: ignore

    # Defaults: autodetect device, camera index 0, output in CWD, TAB delimiter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    camera_index = 0
    weights_dir = Path(__file__).resolve().parent / "OpenFace-3.0" / "weights"
    out_path = Path.cwd() / f"mtl_stream_{int(time.time())}.tsv"
    delimiter = "\t"
    max_frames = 0

    face_weights, mtl_weights = resolve_weights(weights_dir)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    # Initialize models
    face_detector = FaceDetector(model_path=str(face_weights), device=device)
    mtl = MultitaskPredictor(model_path=str(mtl_weights), device=device)

    # Prepare output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" avoids blank lines on Windows when using csv.writer
    out_file = open(out_path, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(out_file, delimiter=delimiter)

    frame_index = 0
    header_written = False
    start_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # FaceDetector API expects an image path; write temp file
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            try:
                cv2.imwrite(tmp_path, frame)
                cropped_face, dets = face_detector.get_face(tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if dets is None or len(dets) == 0 or cropped_face is None:
                frame_index += 1
                if max_frames and frame_index >= max_frames:
                    break
                continue

            with torch.no_grad():
                emotion_logits, gaze_output, au_output = mtl.predict(cropped_face)

            # Convert outputs
            emotion_logits_np = to_numpy(emotion_logits)
            gaze_np = to_numpy(gaze_output).reshape(-1)
            au_np = to_numpy(au_output).reshape(-1)

            # Emotion index and label
            if emotion_logits_np.ndim == 2:
                emotion_idx = int(np.argmax(emotion_logits_np, axis=1).item())
            else:
                emotion_idx = int(np.argmax(emotion_logits_np))
            emotion_label = EMOTION_LABELS[emotion_idx] if 0 <= emotion_idx < len(EMOTION_LABELS) else str(emotion_idx)

            # Gaze yaw/pitch
            gaze_yaw = float(gaze_np[0]) if gaze_np.size >= 1 else np.nan
            gaze_pitch = float(gaze_np[1]) if gaze_np.size >= 2 else np.nan

            # Write header lazily when AU dimension known
            if not header_written:
                write_header(writer, au_dim=int(au_np.size), delimiter=delimiter)
                header_written = True

            # Time fields
            ts = time.time()
            ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

            row = [
                f"{ts:.6f}",
                ts_iso,
                str(frame_index),
                str(emotion_idx),
                emotion_label,
                f"{gaze_yaw:.6f}",
                f"{gaze_pitch:.6f}",
            ]
            row.extend([f"{float(v):.6f}" for v in au_np.tolist()])
            writer.writerow(row)

            frame_index += 1
            if max_frames and frame_index >= max_frames:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        out_file.close()

    elapsed = time.perf_counter() - start_time
    print(f"Saved {frame_index} frames to {out_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()



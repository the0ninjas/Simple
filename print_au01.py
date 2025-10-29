import sys
import os
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


OPENFACE_ROOT = Path(__file__).resolve().parent / "OpenFace-3.0"
WEIGHTS_DIR = OPENFACE_ROOT / "weights"


def ensure_openface_import(openface_root: Path) -> None:
    try:
        import openface  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    if str(openface_root) not in sys.path:
        sys.path.insert(0, str(openface_root))
    import openface  # type: ignore  # noqa: F401


def resolve_weights(weights_dir: Path) -> Tuple[Path, Path]:
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

    missing = []
    if face_path is None:
        missing.append(str(face_primary))
    if mtl_path is None:
        missing.append(str(mtl_primary))
    if missing:
        available = "\n".join(str(p.name) for p in weights_dir.glob("*.pth")) or "<none>"
        raise FileNotFoundError(
            "Missing required weights:\n" + "\n".join(missing) +
            f"\nSearched: {weights_dir}\nAvailable .pth files:\n{available}"
        )

    return face_path, mtl_path


def main() -> None:
    ensure_openface_import(OPENFACE_ROOT)
    if not OPENFACE_ROOT.exists():
        raise RuntimeError(f"OPENFACE_ROOT not found: {OPENFACE_ROOT}")

    from openface.face_detection import FaceDetector  # type: ignore
    from openface.multitask_model import MultitaskPredictor  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = 0.2  # intensity threshold in [0,1]

    face_weights, mtl_weights = resolve_weights(WEIGHTS_DIR)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera index 0")

    # Some internal RetinaFace code expects CWD containing ./weights
    old_cwd = os.getcwd()
    os.chdir(str(OPENFACE_ROOT))
    try:
        face_detector = FaceDetector(model_path=str(face_weights), device=device)
        mtl = MultitaskPredictor(model_path=str(mtl_weights), device=device)
    finally:
        os.chdir(old_cwd)

    frame_index = 0
    print("[INFO] Streaming AU01 intensity; press Ctrl+C to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # FaceDetector expects an image path; write a temp file
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
                continue

            with torch.no_grad():
                _, _, au_output = mtl.predict(cropped_face)

            # Get AU01 (index 0). Model outputs raw scores in [-1,1]; clamp to [0,1] for intensity.
            try:
                au_values = au_output.detach().cpu().numpy().reshape(-1)
            except Exception:
                au_values = np.asarray(au_output).reshape(-1)

            if au_values.size == 0:
                frame_index += 1
                continue

            au01_raw = float(au_values[0])
            au01_intensity = float(np.clip(au01_raw, 0.0, 1.0))

            if au01_intensity > threshold:
                ts = time.time()
                ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                print(f"{ts_iso} frame={frame_index} AU01_intensity={au01_intensity:.3f}")

            frame_index += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()



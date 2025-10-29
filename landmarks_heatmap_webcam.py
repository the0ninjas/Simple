import sys
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


OPENFACE_ROOT = Path(__file__).resolve().parent / "OpenFace-3.0"


def ensure_openface_import(openface_root: Path) -> None:
    try:
        import openface  # type: ignore  # noqa: F401
        return
    except Exception:
        pass
    if str(openface_root) not in sys.path:
        sys.path.insert(0, str(openface_root))
    import openface  # type: ignore  # noqa: F401


def resolve_weight(path_primary: Path, search_dirs: list[Path], candidates: list[str]) -> Optional[Path]:
    if path_primary.exists():
        return path_primary
    for d in search_dirs:
        for name in candidates:
            p = d / name
            if p.exists():
                return p
    return None


def select_device() -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    return "cpu"


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray) -> None:
    for (x, y) in landmarks.astype(int):
        cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)


def draw_landmarks_heatmap(image: np.ndarray, landmarks: np.ndarray, sigma: int = 6, alpha: float = 0.6) -> np.ndarray:
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for (x, y) in landmarks.astype(int):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(heat, (int(x), int(y)), max(1, sigma // 2), 1.0, -1)
    ksize = max(3, sigma * 4 + 1)
    heat = cv2.GaussianBlur(heat, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    heat = np.clip(heat / (heat.max() + 1e-6), 0.0, 1.0)
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1.0, heat_color, alpha, 0)
    return overlay


def main() -> None:
    ensure_openface_import(OPENFACE_ROOT)

    from openface.face_detection import FaceDetector  # type: ignore
    from openface.landmark_detection import LandmarkDetector  # type: ignore

    device = select_device()
    of_weights = OPENFACE_ROOT / "weights"
    root_weights = Path(__file__).resolve().parent / "weights"

    face_weights = resolve_weight(
        of_weights / "Alignment_RetinaFace.pth",
        [of_weights, root_weights],
        ["Alignment_RetinaFace.pth", "mobilenet0.25_Final.pth"],
    )
    if face_weights is None:
        raise FileNotFoundError("Face detector weights not found. Expected Alignment_RetinaFace.pth or mobilenet0.25_Final.pth")

    landmark_weights = resolve_weight(
        of_weights / "Landmark_98.pkl",
        [of_weights, root_weights],
        ["Landmark_98.pkl", "Landmark_68.pkl"],
    )
    if landmark_weights is None:
        raise FileNotFoundError("Landmark model not found. Expected Landmark_98.pkl or Landmark_68.pkl")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera index 0")

    old_cwd = os.getcwd()
    os.chdir(str(OPENFACE_ROOT))
    try:
        face_detector = FaceDetector(model_path=str(face_weights), device=device)
        try:
            landmark_detector = LandmarkDetector(model_path=str(landmark_weights), device=device)
        except ValueError:
            # Fallback for packages requiring explicit device IDs or CPU only
            landmark_detector = LandmarkDetector(model_path=str(landmark_weights), device="cpu")
    finally:
        os.chdir(old_cwd)

    last_infer_ms = 0.0
    print("[INFO] Running real-time landmark detection. Press 'q'/Esc to exit, 'h' for heatmap.")
    use_heatmap = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # FaceDetector expects an image path
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

            vis = frame.copy()
            if dets is not None and len(dets) > 0:
                try:
                    landmarks_list = landmark_detector.detect_landmarks(frame, dets)
                except Exception:
                    landmarks_list = None

                if landmarks_list:
                    for i, lm in enumerate(landmarks_list):
                        if use_heatmap:
                            vis = draw_landmarks_heatmap(vis, lm)
                        else:
                            draw_landmarks(vis, lm)
                    # draw first bbox
                    try:
                        x1, y1, x2, y2 = dets[0][:4].astype(int)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    except Exception:
                        pass

            cv2.putText(vis, f"Infer: {last_infer_ms:.1f} ms", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, f"Mode: {'HEATMAP' if use_heatmap else 'POINTS'} (toggle: h)", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1, cv2.LINE_AA)
            cv2.imshow("OpenFace-3.0 Landmarks", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key in (ord('h'), ord('H')):
                use_heatmap = not use_heatmap

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



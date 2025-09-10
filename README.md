# Real-Time Facial AU / Emotion / Gaze Demo (OpenFace-3.0)

`au_realtime.py` is a zero‑argument, self‑contained webcam demo that uses OpenFace-3.0 components to:
- Detect a face (RetinaFace)
- Predict Action Units (AUs)
- Predict basic emotion class
- Predict coarse gaze (yaw / pitch)
- Optionally detect and draw facial landmarks (if the landmark model is present)

All configuration is done by editing a few constants at the top of the script (no CLI parsing).

## Quick Start
1. Ensure Python 3.10+ (tested with 3.11). Use a clean Conda env if possible.
2. Install dependencies (example CPU environment):
   ```bash
   conda create -n openface-rt python=3.11 numpy opencv pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
   conda activate openface-rt
   ```
   Or install via pip inside an existing env:
   ```bash
   pip install numpy opencv-python torch torchvision
   ```
3. Place (or clone) the OpenFace-3.0 repository as a folder named `OpenFace-3.0` in the same directory as `au_realtime.py`.
4. Put required weight files into `OpenFace-3.0/weights/` (see list below).
5. Run:
   ```bash
   python au_realtime.py
   ```
6. Press `q` or `Esc` to exit the window.

## Directory Layout (expected minimal)
```
YourFolder/
  au_realtime.py
  OpenFace-3.0/
    weights/
      Alignment_RetinaFace.pth         (required)
      MTL_backbone.pth                 (required)
      mobilenetV1X0.25_pretrain.tar    (RetinaFace auxiliary – required)
      Landmark_98.pkl                  (optional for landmarks)
      mobilenet0.25_Final.pth          (often present; kept if repo provides it)
```

## Configuration Constants (edit inside `au_realtime.py`)
| Constant | Purpose |
|----------|---------|
| `OPENFACE_ROOT` | Path to the local OpenFace-3.0 repo (auto set relative to script). |
| `WEIGHTS_DIR` | Location of model weights (defaults to `OPENFACE_ROOT / "weights"`). |
| `DEVICE` | `"cpu"` or `"cuda"` (set to `"cuda"` if you have a GPU + CUDA torch). |
| `CAMERA_INDEX` | OpenCV camera index (0 is default webcam). |
| `FRAME_SCALE` | Uniform resize factor (<1.0 for speed). |
| `DRAW_OVERLAYS` | Toggle visualization drawing. |
| `MAX_AU_VIS` | Number of AU bars to render (first N). |
| `LOAD_LANDMARKS` | Attempt to load and draw landmarks if model is present. |
| `WINDOW_NAME` | Title of the OpenCV window. |

Modify these directly in the script—no command line flags are used.

## Required / Optional Weight Files
| File | Required | Description |
|------|----------|-------------|
| `Alignment_RetinaFace.pth` | Yes | RetinaFace alignment / detection weights used by `FaceDetector`. |
| `MTL_backbone.pth` | Yes | Multitask model (emotion, gaze, AU). |
| `mobilenetV1X0.25_pretrain.tar` | Yes* | RetinaFace internal auxiliary weight file; the detector code expects it at `./weights/` relative to current working directory. The script temporarily `chdir`s into `OPENFACE_ROOT` so this relative path resolves. |
| `Landmark_98.pkl` | Optional | Loads if present for drawing 98 facial landmarks. |
| `mobilenet0.25_Final.pth` | Optional | May be part of the upstream repo; not directly loaded by the simplified script. |

*If `mobilenetV1X0.25_pretrain.tar` is missing you may get a `FileNotFoundError` during face detector init.

## How the Script Handles Paths
1. Adds `OPENFACE_ROOT` to `sys.path` if `openface` module import fails.
2. Validates presence of `Alignment_RetinaFace.pth` and `MTL_backbone.pth` inside `WEIGHTS_DIR` at startup. Missing files raise a clear error.
3. Temporarily changes the current working directory to `OPENFACE_ROOT` while constructing `FaceDetector` and `MultitaskPredictor` so any internal relative `./weights/...` references succeed.

## Runtime Behavior
- Captures frames from the configured camera.
- Writes each frame to a temporary JPEG file (current detector API expects a file path) and deletes it after detection.
- Runs multitask inference (emotion logits, gaze vector, AU outputs) for the first detected face.
- Overlays (if enabled):
  - Face bounding box (yellow)
  - AU bars (green) with intensity values 0–1 (first `MAX_AU_VIS` AUs)
  - Inference time in ms
  - Emotion label (from predefined list)
  - Gaze yaw / pitch (degrees or model units—depends on model calibration)
  - 98 landmarks (red points) if the landmark model is available

## Controls
- `q` or `Esc` closes the window and stops capture.

## Emotion Labels
Defined in the script as:
```
["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
```
The index selected is `argmax` over emotion logits.

## Gaze Output
A 2‑value vector (yaw, pitch). No further smoothing or calibration is applied in this demo.

## Performance Tips
- Reduce `FRAME_SCALE` (e.g., `0.75` or `0.5`).
- Switch `DEVICE = "cuda"` if a compatible GPU build of PyTorch is installed.
- Close other applications using the camera.

## Troubleshooting
| Symptom | Cause / Fix |
|---------|-------------|
| `FileNotFoundError` listing weight paths | Place missing files into `OpenFace-3.0/weights/` with exact names. |
| `ModuleNotFoundError: openface` | Ensure `OpenFace-3.0` folder is adjacent to the script (or install the package version). |
| Camera not opening | Use a different `CAMERA_INDEX` (1,2,...). Check privacy settings or close other apps. |
| Extremely slow | Lower `FRAME_SCALE`; confirm you are not running in a throttled VM; consider GPU. |
| No AU bars / empty values | Ensure `MTL_backbone.pth` matches the expected model architecture (version mismatch). |
| Landmarks absent | `Landmark_98.pkl` not present or failed to load (warning printed). |
| Crash in RetinaFace re: `mobilenetV1X0.25_pretrain.tar` | Ensure that file exists under `OpenFace-3.0/weights/`; script relies on temporary `chdir` so relative reference resolves. |

## Extending / Modifying
- Replace the temporary file I/O with direct ndarray handling when the detector supports in‑memory images.
- Log AU / emotion / gaze values by appending writes where they are computed in the loop.
- Integrate smoothing filters (EMA) for more stable visualizations.

## Windows Notes
- If you encounter Unicode encoding issues in the terminal, set environment variable:
  PowerShell: `$env:PYTHONIOENCODING = "utf-8"`
  CMD: `set PYTHONIOENCODING=utf-8`

## License
`au_realtime.py` is provided under the same licensing terms that apply to your local OpenFace-3.0 usage (OpenFace is separately licensed—respect its terms). Provide attribution where required.

---
Updated for the argument‑free simplified script.

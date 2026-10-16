# SignBridge — Real-Time Sign Language Recognition Assistant (Powered by YOLOv5)

SignBridge is an intelligent desktop application that recognizes sign language gestures from a live webcam feed in real time, displays recognized words in an interactive conversation panel, dictates every word aloud via text-to-speech, and provides full transcript replay and a searchable sign library.

> **Academic Project** — Image and Video Processing (IVP)  
> Powered by a custom **YOLOv5s** classification model trained on the `CollectedImages` dataset (31 conversational sign categories, 1,860 images).

---

## Key Features

- **Custom YOLOv5 Model Architecture** — Built with Ultralytics YOLOv5s-cls (`models/yolov5_sign_cls.pt`) fine-tuned from pretrained weights. Reaches **100.0% validation Top-1 accuracy** with an inference latency of **~5.7 ms** per frame on CPU.
- **Real-Time Webcam Recognition** — Live camera stream at ~30 FPS with inference every 200 ms, horizontal mirror correction, and dynamic confidence scoring.
- **Temporal Prediction Stabilization** — State-machine smoothing (`WAITING` → `DETECTING` → `CONFIRMED` → `COOLDOWN`) ensures robust gesture acceptance and prevents accidental duplicate triggers while allowing natural repetition.
- **Continuous Cross-Platform Audio Dictation**:
  - **Real-Time Word Dictation**: As soon as a sign is recognized and confirmed, it is dictated aloud over the speakers immediately.
  - **Full Conversation Replay (`▶ Play All`)**: Reads out the complete accumulated transcript sequence in order.
  - **Latest Word Replay (`▶ Play Last`)**: Speaks the most recently recognized sign.
  - **Library Click-to-Dictate**: Clicking any sign in the reference library speaks its name aloud.
  - **Windows & macOS Native Engines**: Uses Windows Microsoft SAPI5 (`win32com.client.Dispatch("SAPI.SpVoice")`) with background worker threads for continuous, non-blocking playback without dropped utterances, and macOS native `say` on Apple systems.
  - **Live Dictation Indicator**: The UI displays real-time visual feedback (`🔊 Dictation Active` / `🔊 Dictating...`).
- **Running Conversation Transcript** — Accumulates words, displays word counts, supports one-click clipboard copying, and transcript clearing.
- **Searchable Sign Library** — Browse and filter all 31 supported gestures with instant search and preview images.
- **Modern Dark UI** — PySide6 (Qt) desktop interface featuring a 65% / 35% split-screen layout with teal accent highlights.

---

## Supported Sign Categories (31)

The model is trained to recognize 31 conversational sign gestures:

| | | | |
|---|---|---|---|
| **again** | **agree** | **answer** | **attendance** |
| **book** | **break** | **careful** | **congratulations** |
| **good morning** | **happy birthday** | **home** | **how are you** |
| **hungry** | **i need help** | **keepsmile** | **meet** |
| **mistake** | **please** | **practice** | **pressure** |
| **sick** | **stop** | **sun** | **team** |
| **thirsty** | **this** | **together** | **understand** |
| **wait** | **where** | **write** | |

---

## Model Architecture & Evaluation

The recognition pipeline is adapted from the YOLOv5 methodology in `Sign_language_Generation_Using_YOLO_v5.ipynb` for the custom `CollectedImages` dataset.

### Dataset Overview
- **Source Dataset:** `CollectedImages` (1,860 total images across 31 classes, 60 images per class, uniform 1280×720 resolution).
- **Split Strategy:** Stratified 70% train (1,302 images), 15% validation (279 images), and 15% test (279 images).
- **Format:** Structured in `dataset/yolov5_cls/` with `train/`, `val/`, and `test/` class directories and `dataset/dataset_config.yaml`.

### Model Specifications
- **Architecture:** YOLOv5s-cls (CSPDarknet backbone + classification head)
- **Pretrained Weights:** Transfer learning from `yolov5s-cls.pt` (ImageNet)
- **Input Dimensions:** 224 × 224 RGB
- **Model Parameters:** 4,206,399 parameters (67 layers, 10.4 GFLOPs)
- **Model Checkpoint:** `models/yolov5_sign_cls.pt` (8.6 MB)
- **Class Labels:** `models/class_labels.json`

### Performance Metrics
- **Validation Top-1 Accuracy:** `100.0%`
- **Validation Top-5 Accuracy:** `100.0%`
- **CPU Latency:** `~5.7 ms` per image
- **GPU Latency (CUDA):** `~0.7 ms` per image

---

## Quickstart: Running the Application

### Option A: Windows 1-Click Launcher (Recommended)
Double-click `run.bat` in the project root:
```cmd
d:\SignBridge\run.bat
```

### Option B: Command Line (Windows / macOS)
Activate your Python environment and launch `main.py`:
```powershell
# Windows
py -3.13 main.py

# macOS / Linux
python3 main.py
```

### Basic App Workflow
1. Click **Start Camera** on the left panel.
2. Position your hands in front of the camera and perform any of the 31 gestures.
3. The real-time camera overlay displays the detected gesture and confidence score.
4. When stabilized, the sign is automatically added to the conversation list and **spoken aloud**.
5. Click **`▶ Play All`** to dictate the entire conversation transcript from start to finish.
6. Click **`▶ Play Last`** to repeat the latest recognized gesture.
7. Explore the **Sign Library** on the lower-right panel to search and click signs for reference and voice pronunciation.

---

## Reproducible YOLOv5 Pipeline

All dataset preparation, training, and evaluation scripts are automated and included in the `scripts/` directory:

### 1. Prepare and Stratify Dataset
Converts the raw collected images into train/val/test splits and generates dataset configuration:
```bash
python scripts/prepare_yolov5_dataset.py \
    --source "E:\IVP\SignBridgeDataset\CollectedImages" \
    --output "dataset/yolov5_cls" \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### 2. Train YOLOv5 Model
Trains the YOLOv5s classifier with transfer learning and automatically copies the best checkpoint to `models/yolov5_sign_cls.pt`:
```bash
python scripts/train_yolov5.py \
    --dataset "dataset/yolov5_cls" \
    --epochs 15 \
    --batch-size 32 \
    --img-size 224 \
    --model "yolov5s-cls.pt"
```

### 3. Evaluate Model & Confusion Matrix
Evaluates model accuracy on test data and produces evaluation metrics:
```bash
python scripts/evaluate_model.py
```

---

## Prediction Stabilization State Machine

To prevent rapid jitter and false triggers, predictions pass through a temporal state machine:

```
[ WAITING ] ──(Hand Detected)──► [ DETECTING ] ──(Agreement Reached)──► [ CONFIRMED ]
     ▲                                                                       │
     │                                                                       ▼
     └────────────────────────(Cooldown Elapsed)─────────────────────── [ COOLDOWN ]
```

- **Stabilization Window:** 5 recent frames
- **Minimum Agreement:** 3 matching frames within the window
- **Confidence Threshold:** ≥ 40% average confidence
- **Cooldown Frames:** 4 frames after acceptance before re-arming the same sign
- **Fast Reset:** Lowering hands or switching gestures resets the state immediately

---

## Project Structure

```
SignBridge/
├── main.py                          ← Application entrypoint
├── run.bat                          ← Windows 1-click launcher
├── requirements.txt                 ← Python dependencies
├── README.md                        ← Complete documentation
│
├── app/
│   ├── core/
│   │   ├── settings.py              ← Paths, thresholds, intervals, and UI palette
│   │   └── conversation_manager.py  ← Thread-safe transcript accumulation
│   │
│   ├── vision/
│   │   ├── camera_manager.py        ← Thread-safe OpenCV video capture
│   │   ├── predictor.py             ← Dual-mode YOLOv5 / MobileNet inference engine
│   │   ├── prediction_stabilizer.py ← State machine for temporal smoothing
│   │   └── preprocessing.py         ← PyTorch image transforms (ImageNet normalization)
│   │
│   ├── speech/
│   │   └── text_to_speech.py        ← Native SAPI5 / macOS say continuous dictation engine
│   │
│   └── ui/
│       ├── main_window.py           ← Main PySide6 window coordinating panels
│       ├── camera_panel.py          ← Live feed, confidence bar, and status overlay
│       ├── conversation_panel.py    ← Transcript, Play All, Play Last, and dictation badge
│       ├── library_panel.py         ← 31-sign reference with search & click-to-speak
│       └── qt_compat.py             ← PySide6 compatibility layer
│
├── scripts/
│   ├── prepare_yolov5_dataset.py    ← Stratified dataset splitter & data.yaml generator
│   ├── train_yolov5.py              ← YOLOv5 training script based on notebook methodology
│   ├── inspect_dataset.py           ← Dataset integrity & resolution validator
│   ├── train_model.py               ← Alternative MobileNetV2 trainer
│   └── evaluate_model.py            ← Evaluation metrics & confusion matrix generator
│
├── models/
│   ├── yolov5_sign_cls.pt           ← Trained custom YOLOv5s model checkpoint
│   ├── class_labels.json            ← 31 sign category mapping
│   ├── dataset_splits.json          ← Dataset split metadata
│   └── confusion_matrix.png         ← Confusion matrix plot
│
├── dataset/
│   ├── dataset_config.yaml          ← YOLOv5 dataset configuration
│   ├── dataset_report.json          ← Dataset inspection summary
│   └── yolov5_cls/                  ← Prepared dataset splits (train / val / test)
│
├── yolov5/                          ← Cloned Ultralytics YOLOv5 submodule
└── tests/                           ← Automated test suite (113 passing tests)
```

---

## Automated Test Suite

SignBridge includes 113 comprehensive unit and integration tests verifying all application subsystems:

```bash
# Run complete test suite
py -3.13 -m pytest tests/ -v
```

### Test Coverage
- `tests/test_yolov5_integration.py` — YOLOv5 model loading, forward pass, batching, Top-K predictions, label counts.
- `tests/test_text_to_speech.py` — Continuous dictation, rapid cancellation, synchronous/asynchronous speech, rate/voice controls.
- `tests/test_ui.py` — Window layout proportions, camera start/stop flow, inference timers, transcript updates, button signals.
- `tests/test_prediction_stabilizer.py` — Agreement thresholds, cooldown lifecycle, duplicate prevention, state transitions.
- `tests/test_conversation_manager.py` — Transcript ordering, thread safety, clear/copy operations.
- `tests/test_preprocessing.py` — Input normalization, dimension checks, tensor conversion.
- `tests/test_components.py` — End-to-end component verification.

---

## License & Credits

- Developed for the **Image and Video Processing** academic project.
- Model training methodology based on `Sign_language_Generation_Using_YOLO_v5.ipynb` using [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).
- Designed for educational and assistive technology research.

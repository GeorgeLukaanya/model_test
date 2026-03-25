# Bee Video Tracking Pipeline

This project runs bee detection on recorded videos using your trained YOLO model and writes:

- an annotated output video with bee boxes and IDs
- a JSON summary with per-frame counts and per-track details

The project includes:

- [bee_video_pipeline.py](/home/ltgwgeorge/Desktop/Business/IoT-RA/model_test/bee_video_pipeline.py): command-line pipeline
- [bee_video_ui.py](/home/ltgwgeorge/Desktop/Business/IoT-RA/model_test/bee_video_ui.py): simple desktop UI
- [best.onnx](/home/ltgwgeorge/Desktop/Business/IoT-RA/model_test/best.onnx): lightweight export supported directly
- [best.pt](/home/ltgwgeorge/Desktop/Business/IoT-RA/model_test/best.pt): Ultralytics weights also supported

## How It Works

The pipeline does this for each video:

1. reads the video frame by frame
2. runs detection with either `best.onnx` or `best.pt`
3. applies non-maximum suppression to remove duplicate boxes
4. assigns persistent IDs with a lightweight IoU-based tracker
5. draws:
   - bounding boxes
   - bee IDs
   - confidence scores
   - counts on the frame
6. writes the annotated video
7. writes a JSON summary

The counts shown are:

- `Active bees`: number of tracked bees visible in the current frame
- `Unique bees confirmed`: tracks that survived at least the configured minimum number of frames
- `Unique bee IDs seen`: all IDs created so far, including very short-lived tracks

## Files

- [bee_video_pipeline.py](/home/ltgwgeorge/Desktop/Business/IoT-RA/model_test/bee_video_pipeline.py): detection, tracking, video export, JSON export
- [bee_video_ui.py](/home/ltgwgeorge/Desktop/Business/IoT-RA/model_test/bee_video_ui.py): desktop app using `tkinter`
- [requirements.txt](/home/ltgwgeorge/Desktop/Business/IoT-RA/model_test/requirements.txt): minimal Python dependencies

## Requirements

- Python 3
- a model file in `.onnx` or `.pt` format
- for `.pt` models: `ultralytics` must also be installed

## Setup

From the project directory:

 ```bash
 cd /home/ltgwgeorge/Desktop/Business/IoT-RA/model_test
 python -m venv .venv
 .venv/bin/python -m pip install -r requirements.txt
 ```

If you want to use `best.pt` directly, install Ultralytics too:

```bash
.venv/bin/python -m pip install ultralytics
```

## Run The UI

Start the desktop app with:

```bash
cd /home/ltgwgeorge/Desktop/Business/IoT-RA/model_test
.venv/bin/python bee_video_ui.py
```

Then in the window:

1. choose the model, usually `best.onnx` or `best.pt`
2. choose the input video
3. choose the output folder
4. adjust settings if needed
5. click `Run pipeline`

When processing finishes, the UI shows:

- the annotated video path
- the summary JSON path
- the total counts for that video

## Run From The Command Line

You can also run the pipeline directly:

```bash
cd /home/ltgwgeorge/Desktop/Business/IoT-RA/model_test
.venv/bin/python bee_video_pipeline.py --input /path/to/video.mp4 --model best.onnx --output-dir outputs
```

Example with custom settings:

```bash
.venv/bin/python bee_video_pipeline.py \
  --input /path/to/video.mp4 \
  --model best.onnx \
  --output-dir outputs \
  --conf 0.25 \
  --nms-iou 0.45 \
  --imgsz 640 \
  --track-iou 0.30 \
  --max-track-age 20 \
  --min-track-frames 3 \
  --line-thickness 2
```

Example using `.pt`:

```bash
.venv/bin/python -m pip install ultralytics
.venv/bin/python bee_video_pipeline.py --input /path/to/video.mp4 --model best.pt --output-dir outputs
```

## Output Files

For an input video named `my_video.mp4`, the pipeline writes:

- `outputs/my_video_annotated.mp4`
- `outputs/my_video_summary.json`

The JSON summary includes:

- input video path
- model path
- output video path
- total frames processed
- maximum active bees in one frame
- total unique IDs seen
- total confirmed unique bees
- per-frame counts
- per-track history

## Settings Guide

- `Confidence`: lower values detect more bees but can add false positives
- `NMS IoU`: controls how aggressively overlapping boxes are merged
- `Image size`: larger values may improve detection but slow processing
- `Track IoU`: higher values make ID matching stricter
- `Max track age`: how long to keep a bee track alive after it disappears
- `Min track frames`: filters out short-lived noisy tracks from the confirmed count
- `Line thickness`: box thickness in the annotated video

## Important Limitation

The current tracker is intentionally simple. It works by matching boxes between nearby frames using IoU.

That means:

- IDs are intended to stay consistent within a single video
- IDs may switch if bees overlap heavily or disappear and re-enter
- `Unique bees confirmed` is a practical estimate, not a guaranteed biological count

If you later need stronger ID consistency, the next step is improving the tracker rather than changing the UI.

## Troubleshooting

If the UI does not start:

- make sure the virtual environment exists
- make sure dependencies are installed
- run from the project directory

If the model fails to load:

- confirm the selected model is a `.onnx` or `.pt` file
- confirm the file path is correct
- if you selected `.pt`, make sure `ultralytics` is installed

If the output video is empty or counts look wrong:

- lower the confidence threshold
- increase `max track age`
- reduce `min track frames`
- test on a shorter video first

## Quick Start

```bash
cd /home/ltgwgeorge/Desktop/Business/IoT-RA/model_test
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python bee_video_ui.py
```

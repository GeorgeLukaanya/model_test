#!/usr/bin/env python3
"""Offline bee detection and tracking pipeline for saved videos."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
SUPPORTED_MODEL_EXTENSIONS = {".onnx", ".pt"}

# Smoothing factor for the velocity exponential moving average (0 = no update, 1 = instant).
_VELOCITY_ALPHA = 0.5


@dataclass
class TrackStats:
    first_frame: int
    last_frame: int
    frames_seen: int = 1
    max_confidence: float = 0.0
    last_bbox: list[int] = field(default_factory=list)

    def update(self, frame_index: int, confidence: float, bbox: list[int]) -> None:
        self.last_frame = frame_index
        self.frames_seen += 1
        self.max_confidence = max(self.max_confidence, confidence)
        self.last_bbox = bbox


@dataclass
class Detection:
    bbox: list[int]
    confidence: float


@dataclass
class Track:
    track_id: int
    bbox: list[int]
    confidence: float
    first_frame: int
    last_frame: int
    hits: int = 1
    age: int = 0
    vx: float = 0.0  # centroid velocity x (pixels/frame, EMA-smoothed)
    vy: float = 0.0  # centroid velocity y

    def update(self, bbox: list[int], confidence: float, frame_index: int) -> None:
        # Update velocity before changing bbox so we measure displacement from the old centre.
        old_cx = (self.bbox[0] + self.bbox[2]) / 2.0
        old_cy = (self.bbox[1] + self.bbox[3]) / 2.0
        new_cx = (bbox[0] + bbox[2]) / 2.0
        new_cy = (bbox[1] + bbox[3]) / 2.0
        self.vx = _VELOCITY_ALPHA * (new_cx - old_cx) + (1 - _VELOCITY_ALPHA) * self.vx
        self.vy = _VELOCITY_ALPHA * (new_cy - old_cy) + (1 - _VELOCITY_ALPHA) * self.vy
        self.bbox = bbox
        self.confidence = confidence
        self.last_frame = frame_index
        self.hits += 1
        self.age = 0

    def predicted_bbox(self) -> list[int]:
        """Return the expected bbox one frame ahead based on current velocity."""
        return [
            int(self.bbox[0] + self.vx),
            int(self.bbox[1] + self.vy),
            int(self.bbox[2] + self.vx),
            int(self.bbox[3] + self.vy),
        ]


def _combined_score(
    predicted_bbox: list[int],
    det_bbox: list[int],
    iou_threshold: float,
    max_dist_px: float,
) -> float:
    """Return a match score in (0, 1].

    Uses IoU against the track's *predicted* next position so that fast-moving
    bees are matched before they drift out of the last known box.
    Falls back to a small proximity score (0, 0.05] when IoU is zero but the
    centroid is within max_dist_px, so bees keep their ID even between
    non-overlapping frames.
    Returns 0 when neither criterion is met.
    """
    iou = bbox_iou(predicted_bbox, det_bbox)
    if iou >= iou_threshold:
        return iou

    cx_a = (predicted_bbox[0] + predicted_bbox[2]) / 2.0
    cy_a = (predicted_bbox[1] + predicted_bbox[3]) / 2.0
    cx_b = (det_bbox[0] + det_bbox[2]) / 2.0
    cy_b = (det_bbox[1] + det_bbox[3]) / 2.0
    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    if dist < max_dist_px:
        return (1.0 - dist / max_dist_px) * 0.05
    return 0.0


class BeeTracker:
    def __init__(
        self,
        iou_threshold: float = 0.1,
        max_age: int = 20,
        max_dist_px: float = 150.0,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.max_dist_px = max_dist_px
        self.next_track_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], frame_index: int) -> list[Track]:
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_id, det_idx in matches:
            detection = detections[det_idx]
            self.tracks[track_id].update(detection.bbox, detection.confidence, frame_index)

        for track_id in unmatched_tracks:
            self.tracks[track_id].age += 1

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.age > self.max_age
        ]
        for track_id in expired:
            del self.tracks[track_id]

        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            self.tracks[self.next_track_id] = Track(
                track_id=self.next_track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                first_frame=frame_index,
                last_frame=frame_index,
            )
            self.next_track_id += 1

        return sorted(
            (track for track in self.tracks.values() if track.age == 0),
            key=lambda track: track.track_id,
        )

    def _match(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self.tracks or not detections:
            return [], list(self.tracks.keys()), list(range(len(detections)))

        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        # Build score matrix [n_tracks × n_dets] using velocity-predicted positions.
        score_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            predicted = self.tracks[tid].predicted_bbox()
            for j, det in enumerate(detections):
                score_matrix[i, j] = _combined_score(
                    predicted, det.bbox, self.iou_threshold, self.max_dist_px
                )

        # Hungarian optimal assignment (minimises cost = 1 − score).
        row_ind, col_ind = linear_sum_assignment(1.0 - score_matrix)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: list[tuple[int, int]] = []

        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if score_matrix[r, c] > 0.0:
                tid = track_ids[r]
                matched_tracks.add(tid)
                matched_detections.add(c)
                matches.append((tid, c))

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        unmatched_detections = [j for j in range(n_dets) if j not in matched_detections]
        return matches, unmatched_tracks, unmatched_detections


class BaseDetector:
    def detect(self, frame: np.ndarray, args: argparse.Namespace) -> list[Detection]:
        raise NotImplementedError


class OnnxDetector(BaseDetector):
    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "Using a .onnx model requires onnxruntime. Install it with "
                "'pip install onnxruntime' (or onnxruntime-gpu for CUDA)."
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        logger.info("Loading ONNX model: %s (providers: %s)", model_path, providers)
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: np.ndarray, args: argparse.Namespace) -> list[Detection]:
        resized, scale, pad_x, pad_y = letterbox(frame, args.imgsz)
        # BGR → RGB, normalise to [0, 1], layout [1, 3, H, W]
        blob = resized[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        output = self.session.run(None, {self.input_name: blob})[0]
        return postprocess_detections(
            output=output,
            frame_shape=frame.shape[:2],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            conf_threshold=args.conf,
            nms_iou=args.nms_iou,
        )


class PtDetector(BaseDetector):
    def __init__(self, model_path: Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Using a .pt model requires Ultralytics. Install it with "
                "'pip install ultralytics'."
            ) from exc

        logger.info("Loading PyTorch model: %s", model_path)
        self.model = YOLO(str(model_path))

    def detect(self, frame: np.ndarray, args: argparse.Namespace) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=args.conf,
            iou=args.nms_iou,
            imgsz=args.imgsz,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        return [
            Detection(bbox=bbox, confidence=float(confidence))
            for bbox, confidence in zip(xyxy, confidences)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bee detection + tracking on a video and export annotations."
    )
    parser.add_argument("--input", required=True, help="Path to the input video.")
    parser.add_argument(
        "--model",
        default="best.onnx",
        help="Path to the trained model (.onnx or .pt).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for the annotated video and summary files.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--track-iou",
        type=float,
        default=0.1,
        help="Minimum IoU to keep a detection on the same track ID.",
    )
    parser.add_argument(
        "--max-track-age",
        type=int,
        default=20,
        help="Consecutive frames a missing track is kept alive.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=3,
        help="Frames a track must survive before counting toward unique bees.",
    )
    parser.add_argument("--line-thickness", type=int, default=2, help="Bounding box line thickness.")
    parser.add_argument(
        "--max-dist-px",
        type=float,
        default=150.0,
        help="Max centroid distance (px) for fallback distance-based matching.",
    )
    parser.add_argument(
        "--full-frame-log",
        action="store_true",
        default=False,
        help="Write one JSON record per frame instead of per-second aggregates.",
    )
    return parser.parse_args()


def build_args(
    input_path: str,
    model_path: str = "best.onnx",
    output_dir: str = "outputs",
    conf: float = 0.25,
    nms_iou: float = 0.45,
    imgsz: int = 640,
    track_iou: float = 0.1,
    max_track_age: int = 20,
    min_track_frames: int = 3,
    line_thickness: int = 2,
    max_dist_px: float = 150.0,
    full_frame_log: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        input=input_path,
        model=model_path,
        output_dir=output_dir,
        conf=conf,
        nms_iou=nms_iou,
        imgsz=imgsz,
        track_iou=track_iou,
        max_track_age=max_track_age,
        min_track_frames=min_track_frames,
        line_thickness=line_thickness,
        max_dist_px=max_dist_px,
        full_frame_log=full_frame_log,
    )


def ensure_video(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{input_path.suffix}'. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )


def ensure_model(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if model_path.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
        raise ValueError(
            f"Unsupported model format '{model_path.suffix}'. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_MODEL_EXTENSIONS))}"
        )


def create_detector(model_path: Path) -> BaseDetector:
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return OnnxDetector(model_path)
    if suffix == ".pt":
        return PtDetector(model_path)
    raise ValueError(f"Unsupported model format: {model_path.suffix}")


def make_output_paths(output_dir: Path, input_video: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_video.stem
    return output_dir / f"{stem}_annotated.mp4", output_dir / f"{stem}_summary.json"


def choose_writer_fps(capture: cv2.VideoCapture) -> float:
    fps = capture.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else 30.0


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return (
        (37 * track_id) % 255,
        (17 * track_id + 91) % 255,
        (29 * track_id + 53) % 255,
    )


def bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


def postprocess_detections(
    output: np.ndarray,
    frame_shape: tuple[int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
    conf_threshold: float,
    nms_iou: float,
) -> list[Detection]:
    # YOLOv8 ONNX output: [1, features, anchors] or [1, anchors, features].
    if output.ndim == 3:
        predictions = output[0]
    elif output.ndim == 2:
        predictions = output
    else:
        raise ValueError(f"Unexpected ONNX output shape: {output.shape}")
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected predictions shape after batch strip: {predictions.shape}")
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    frame_h, frame_w = frame_shape
    boxes: list[list[int]] = []
    confidences: list[float] = []

    for prediction in predictions:
        confidence = float(prediction[4])
        if confidence < conf_threshold:
            continue

        center_x, center_y, width, height = prediction[:4]
        x1 = int(round((center_x - width / 2 - pad_x) / scale))
        y1 = int(round((center_y - height / 2 - pad_y) / scale))
        x2 = int(round((center_x + width / 2 - pad_x) / scale))
        y2 = int(round((center_y + height / 2 - pad_y) / scale))

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(confidence)

    if not boxes:
        return []

    selected = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_iou)
    if len(selected) == 0:
        return []

    detections: list[Detection] = []
    for idx in np.array(selected).flatten():
        x, y, w, h = boxes[idx]
        detections.append(Detection(bbox=[x, y, x + w, y + h], confidence=float(confidences[idx])))
    return detections


def draw_overlay(
    frame: Any,
    frame_index: int,
    active_count: int,
    confirmed_unique_count: int,
    provisional_unique_count: int,
) -> None:
    lines = [
        f"Frame: {frame_index}",
        f"Active bees: {active_count}",
        f"Unique bees confirmed: {confirmed_unique_count}",
        f"Unique bee IDs seen: {provisional_unique_count}",
    ]
    x, y = 15, 30
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def _aggregate_frame_records(
    frame_records: list[dict[str, Any]], fps: float
) -> list[dict[str, Any]]:
    """Collapse per-frame records into per-second aggregates to keep JSON small."""
    if not frame_records:
        return []
    fps = fps if fps > 0 else 30.0
    by_second: dict[int, dict[str, Any]] = {}
    for rec in frame_records:
        sec = int(rec["frame_index"] / fps)
        if sec not in by_second:
            by_second[sec] = {
                "second": sec,
                "active_bees_peak": 0,
                "detections_total": 0,
                "unique_bees_confirmed": rec["unique_bees_confirmed_so_far"],
                "unique_bee_ids_seen": rec["unique_bee_ids_seen_so_far"],
            }
        entry = by_second[sec]
        entry["active_bees_peak"] = max(entry["active_bees_peak"], rec["active_bees"])
        entry["detections_total"] += rec["detections"]
        # Keep the latest (highest-frame) value for cumulative counters.
        entry["unique_bees_confirmed"] = rec["unique_bees_confirmed_so_far"]
        entry["unique_bee_ids_seen"] = rec["unique_bee_ids_seen_so_far"]
    return list(by_second.values())


def build_summary(
    input_video: Path,
    model_path: Path,
    annotated_video: Path,
    frame_records: list[dict[str, Any]],
    track_history: dict[int, TrackStats],
    confirmed_ids: set[int],
    fps: float = 30.0,
    full_frame_log: bool = False,
) -> dict[str, Any]:
    max_active = max((r["active_bees"] for r in frame_records), default=0)
    summary_tracks = {
        str(track_id): {
            "first_frame": stats.first_frame,
            "last_frame": stats.last_frame,
            "frames_seen": stats.frames_seen,
            "max_confidence": round(stats.max_confidence, 4),
            "last_bbox": stats.last_bbox,
            "confirmed": track_id in confirmed_ids,
        }
        for track_id, stats in sorted(track_history.items())
    }

    if full_frame_log:
        timeline_key = "frame_counts"
        timeline = frame_records
    else:
        timeline_key = "second_counts"
        timeline = _aggregate_frame_records(frame_records, fps)

    return {
        "input_video": str(input_video),
        "model_path": str(model_path),
        "annotated_video": str(annotated_video),
        "total_frames_processed": len(frame_records),
        "fps": round(fps, 3),
        "max_active_bees_in_frame": max_active,
        "unique_bee_ids_seen": len(track_history),
        "unique_bees_confirmed": len(confirmed_ids),
        timeline_key: timeline,
        "tracks": summary_tracks,
    }


def process_video(
    args: argparse.Namespace,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[Path, Path]:
    model_path = Path(args.model).resolve()
    input_video = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    ensure_video(input_video)
    ensure_model(model_path)

    annotated_video, summary_json = make_output_paths(output_dir, input_video)

    logger.info("Loading detector from %s", model_path)
    detector = create_detector(model_path)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = choose_writer_fps(capture)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    logger.info(
        "Video: %dx%d @ %.2f fps, ~%s frames",
        width, height, fps, total_frames if total_frames else "unknown",
    )

    writer = cv2.VideoWriter(
        str(annotated_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {annotated_video}")

    frame_index = 0
    frame_records: list[dict[str, Any]] = []
    track_history: dict[int, TrackStats] = {}
    confirmed_ids: set[int] = set()
    tracker = BeeTracker(
        iou_threshold=args.track_iou,
        max_age=args.max_track_age,
        max_dist_px=args.max_dist_px,
    )

    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Processing cancelled at frame %d.", frame_index)
                break

            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.detect(frame, args)
            active_tracks = tracker.update(detections, frame_index)

            active_ids: set[int] = set()
            for track in active_tracks:
                track_id = track.track_id
                bbox = track.bbox
                confidence = track.confidence
                active_ids.add(track_id)
                if track_id in track_history:
                    track_history[track_id].update(frame_index, confidence, bbox)
                else:
                    track_history[track_id] = TrackStats(
                        first_frame=frame_index,
                        last_frame=frame_index,
                        frames_seen=1,
                        max_confidence=confidence,
                        last_bbox=bbox,
                    )

                if track_history[track_id].frames_seen >= args.min_track_frames:
                    confirmed_ids.add(track_id)

                x1, y1, x2, y2 = bbox
                color = color_for_track(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.line_thickness)
                cv2.putText(
                    frame,
                    f"Bee {track_id} | {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            draw_overlay(
                frame=frame,
                frame_index=frame_index,
                active_count=len(active_ids),
                confirmed_unique_count=len(confirmed_ids),
                provisional_unique_count=len(track_history),
            )
            writer.write(frame)
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "active_bees": len(active_ids),
                    "detections": len(detections),
                    "unique_bees_confirmed_so_far": len(confirmed_ids),
                    "unique_bee_ids_seen_so_far": len(track_history),
                }
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "frame_index": frame_index,
                        "total_frames": total_frames,
                        "active_bees": len(active_ids),
                        "unique_bees_confirmed": len(confirmed_ids),
                    }
                )
            frame_index += 1

            if frame_index % 500 == 0:
                logger.debug(
                    "Frame %d — active tracks: %d, confirmed IDs: %d",
                    frame_index, len(active_ids), len(confirmed_ids),
                )
    finally:
        capture.release()
        writer.release()

    logger.info(
        "Done. %d frames processed, %d unique bees confirmed.",
        frame_index, len(confirmed_ids),
    )

    summary = build_summary(
        input_video=input_video,
        model_path=model_path,
        annotated_video=annotated_video,
        frame_records=frame_records,
        track_history=track_history,
        confirmed_ids=confirmed_ids,
        fps=fps,
        full_frame_log=getattr(args, "full_frame_log", False),
    )
    summary_json.write_text(json.dumps(summary, indent=2))
    return annotated_video, summary_json


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    annotated_video, summary_json = process_video(args)
    logger.info("Annotated video: %s", annotated_video)
    logger.info("Summary: %s", summary_json)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline bee detection and tracking pipeline for saved videos."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
SUPPORTED_MODEL_EXTENSIONS = {".onnx", ".pt"}


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

    def update(self, bbox: list[int], confidence: float, frame_index: int) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.last_frame = frame_index
        self.hits += 1
        self.age = 0


class BeeTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 20) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
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

        candidate_pairs: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for det_idx, detection in enumerate(detections):
                score = bbox_iou(track.bbox, detection.bbox)
                if score >= self.iou_threshold:
                    candidate_pairs.append((score, track_id, det_idx))

        candidate_pairs.sort(reverse=True)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: list[tuple[int, int]] = []

        for _, track_id, det_idx in candidate_pairs:
            if track_id in matched_tracks or det_idx in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)
            matches.append((track_id, det_idx))

        unmatched_tracks = [
            track_id for track_id in self.tracks if track_id not in matched_tracks
        ]
        unmatched_detections = [
            det_idx for det_idx in range(len(detections)) if det_idx not in matched_detections
        ]
        return matches, unmatched_tracks, unmatched_detections


class BaseDetector:
    def detect(self, frame: np.ndarray, args: argparse.Namespace) -> list[Detection]:
        raise NotImplementedError


class OnnxDetector(BaseDetector):
    def __init__(self, model_path: Path) -> None:
        self.net = cv2.dnn.readNetFromONNX(str(model_path))

    def detect(self, frame: np.ndarray, args: argparse.Namespace) -> list[Detection]:
        resized, scale, pad_x, pad_y = letterbox(frame, args.imgsz)
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1 / 255.0,
            size=(args.imgsz, args.imgsz),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]
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
                "'.venv/bin/python -m pip install ultralytics'."
            ) from exc

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
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--model",
        default="best.onnx",
        help="Path to the trained model. Supports .onnx and .pt. Defaults to best.onnx.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where the annotated video and summary files will be saved.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.45,
        help="IoU threshold used during non-maximum suppression.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--track-iou",
        type=float,
        default=0.3,
        help="Minimum IoU required to keep a detection on the same track ID.",
    )
    parser.add_argument(
        "--max-track-age",
        type=int,
        default=20,
        help="How many consecutive frames a missing track is kept alive.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=3,
        help="Minimum number of frames a track must survive before it counts toward unique bees.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Bounding box line thickness for annotations.",
    )
    return parser.parse_args()


def build_args(
    input_path: str,
    model_path: str = "best.onnx",
    output_dir: str = "outputs",
    conf: float = 0.25,
    nms_iou: float = 0.45,
    imgsz: int = 640,
    track_iou: float = 0.3,
    max_track_age: int = 20,
    min_track_frames: int = 3,
    line_thickness: int = 2,
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
    annotated_video = output_dir / f"{stem}_annotated.mp4"
    summary_json = output_dir / f"{stem}_summary.json"
    return annotated_video, summary_json


def choose_writer_fps(capture: cv2.VideoCapture) -> float:
    fps = capture.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else 30.0


def color_for_track(track_id: int) -> tuple[int, int, int]:
    # Stable color per track ID so the same bee tends to look consistent.
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
    predictions = output[0]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.transpose(1, 0)

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
        detections.append(
            Detection(
                bbox=[x, y, x + w, y + h],
                confidence=float(confidences[idx]),
            )
        )
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
    x = 15
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28


def build_summary(
    input_video: Path,
    model_path: Path,
    annotated_video: Path,
    frame_records: list[dict[str, Any]],
    track_history: dict[int, TrackStats],
    confirmed_ids: set[int],
) -> dict[str, Any]:
    max_active = max((record["active_bees"] for record in frame_records), default=0)
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
    return {
        "input_video": str(input_video),
        "model_path": str(model_path),
        "annotated_video": str(annotated_video),
        "total_frames_processed": len(frame_records),
        "max_active_bees_in_frame": max_active,
        "unique_bee_ids_seen": len(track_history),
        "unique_bees_confirmed": len(confirmed_ids),
        "frame_counts": frame_records,
        "tracks": summary_tracks,
    }


def process_video(
    args: argparse.Namespace,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, Path]:
    model_path = Path(args.model).resolve()
    input_video = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    ensure_video(input_video)
    ensure_model(model_path)

    annotated_video, summary_json = make_output_paths(output_dir, input_video)

    detector = create_detector(model_path)
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = choose_writer_fps(capture)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
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
    tracker = BeeTracker(iou_threshold=args.track_iou, max_age=args.max_track_age)

    try:
        while True:
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
                label = f"Bee ID {track_id} | {confidence:.2f}"
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    args.line_thickness,
                )
                cv2.putText(
                    frame,
                    label,
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
    finally:
        capture.release()
        writer.release()

    summary = build_summary(
        input_video=input_video,
        model_path=model_path,
        annotated_video=annotated_video,
        frame_records=frame_records,
        track_history=track_history,
        confirmed_ids=confirmed_ids,
    )
    summary_json.write_text(json.dumps(summary, indent=2))
    return annotated_video, summary_json


def main() -> None:
    args = parse_args()
    annotated_video, summary_json = process_video(args)
    print(f"Annotated video written to: {annotated_video}")
    print(f"Summary written to: {summary_json}")


if __name__ == "__main__":
    main()

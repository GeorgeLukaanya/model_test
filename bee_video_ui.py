#!/usr/bin/env python3
"""Simple desktop UI for the bee video pipeline."""

from __future__ import annotations

import datetime
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from bee_video_pipeline import build_args, process_video


# ---------------------------------------------------------------------------
# Log handler that forwards pipeline log records into the UI event queue.
# ---------------------------------------------------------------------------

class _QueueLogHandler(logging.Handler):
    def __init__(self, event_queue: queue.Queue) -> None:
        super().__init__()
        self._queue = event_queue

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put(("log", self.format(record)))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class BeeVideoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Bee Video Tracker")
        self.root.geometry("900x780")
        self.root.minsize(780, 660)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._log_handler: _QueueLogHandler | None = None
        self._last_output_dir: Path | None = None

        default_model = Path("best.onnx")
        if not default_model.exists():
            default_model = Path("best.pt")
        self.model_var = tk.StringVar(value=str(default_model.resolve()))
        self.input_var = tk.StringVar()
        self.video_info_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value=str(Path("outputs").resolve()))
        self.conf_var = tk.DoubleVar(value=0.25)
        self.nms_iou_var = tk.DoubleVar(value=0.45)
        self.imgsz_var = tk.IntVar(value=640)
        self.track_iou_var = tk.DoubleVar(value=0.10)
        self.max_track_age_var = tk.IntVar(value=20)
        self.min_track_frames_var = tk.IntVar(value=3)
        self.line_thickness_var = tk.IntVar(value=2)
        self.max_dist_px_var = tk.DoubleVar(value=150.0)
        self.status_var = tk.StringVar(value="Choose a video and click Run pipeline.")

        self._build_layout()
        self.root.after(150, self._poll_events)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        container = ttk.Frame(self.root, padding=16)
        container.grid(sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=0)
        container.rowconfigure(6, weight=1)  # status log expands

        # Title
        ttk.Label(
            container,
            text="Bee Detection and Tracking",
            font=("TkDefaultFont", 16, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            container,
            text="Select a model and a video, then export an annotated video plus a summary JSON.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        # --- Inputs ---
        form = ttk.LabelFrame(container, text="Inputs", padding=12)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        self._add_path_row(form, 0, "Model (.onnx or .pt)", self.model_var, self._browse_model)
        self._add_path_row(form, 1, "Input video", self.input_var, self._browse_video)

        # Video info strip (shown after a video is picked)
        self._video_info_label = ttk.Label(
            form, textvariable=self.video_info_var, foreground="gray"
        )
        self._video_info_label.grid(row=2, column=1, sticky="w", padx=8, pady=(0, 4))

        self._add_path_row(form, 3, "Output folder", self.output_var, self._browse_output_dir)

        # --- Tracking settings ---
        tuning = ttk.LabelFrame(container, text="Tracking Settings", padding=12)
        tuning.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        for col in range(4):
            tuning.columnconfigure(col, weight=1)

        self._add_spinbox(tuning, 0, 0, "Confidence", self.conf_var, 0.05, 1.0, 0.01)
        self._add_spinbox(tuning, 0, 1, "NMS IoU", self.nms_iou_var, 0.05, 0.95, 0.01)
        self._add_spinbox(tuning, 0, 2, "Image size", self.imgsz_var, 320, 1280, 32)
        self._add_spinbox(tuning, 0, 3, "Track IoU", self.track_iou_var, 0.01, 0.95, 0.01)
        self._add_spinbox(tuning, 1, 0, "Max track age", self.max_track_age_var, 1, 200, 1)
        self._add_spinbox(tuning, 1, 1, "Min track frames", self.min_track_frames_var, 1, 50, 1)
        self._add_spinbox(tuning, 1, 2, "Line thickness", self.line_thickness_var, 1, 8, 1)
        self._add_spinbox(tuning, 1, 3, "Max dist (px)", self.max_dist_px_var, 10, 500, 10)

        # --- Actions row ---
        actions = ttk.Frame(container, padding=(0, 12, 0, 0))
        actions.grid(row=4, column=0, sticky="ew")
        actions.columnconfigure(2, weight=1)

        self.run_button = ttk.Button(actions, text="Run pipeline", command=self._start_run)
        self.run_button.grid(row=0, column=0, sticky="w", padx=(0, 6))

        self.cancel_button = ttk.Button(
            actions, text="Cancel", command=self._cancel_run, state="disabled"
        )
        self.cancel_button.grid(row=0, column=1, sticky="w", padx=(0, 12))

        self._open_folder_button = ttk.Button(
            actions, text="Open output folder", command=self._open_output_folder, state="disabled"
        )
        self._open_folder_button.grid(row=0, column=2, sticky="w")

        progress_label = ttk.Label(actions, textvariable=self.status_var, wraplength=400)
        progress_label.grid(row=0, column=3, sticky="e")

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            container, orient="horizontal", mode="determinate", maximum=100
        )
        self.progress_bar.grid(row=5, column=0, sticky="ew", pady=(6, 0))

        # --- Status log ---
        status_frame = ttk.LabelFrame(container, text="Log", padding=12)
        status_frame.grid(row=6, column=0, sticky="nsew", pady=(12, 0))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        self.output_text = tk.Text(status_frame, height=14, wrap="word")
        scrollbar = ttk.Scrollbar(status_frame, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(state="disabled")

    def _add_path_row(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: object,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="ew")

    def _add_spinbox(
        self,
        parent: ttk.Widget,
        row: int,
        col: int,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
        increment: float,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, sticky="ew", padx=4, pady=6)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            frame, textvariable=variable, from_=from_, to=to, increment=increment
        ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

    # ------------------------------------------------------------------
    # Browse callbacks
    # ------------------------------------------------------------------

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose model",
            filetypes=[("Model files", "*.onnx *.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose input video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_var.set(path)
            self._update_video_info(Path(path))

    def _update_video_info(self, video_path: Path) -> None:
        """Read and display resolution, fps and duration for the chosen video."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.video_info_var.set("Could not read video metadata.")
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration_s = n_frames / fps if fps > 0 else 0
            mins, secs = divmod(int(duration_s), 60)
            self.video_info_var.set(
                f"{width}×{height}  {fps:.2f} fps  {mins}m {secs:02d}s  ({n_frames} frames)"
            )
        except Exception:
            self.video_info_var.set("")

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_var.set(path)

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        input_str = self.input_var.get().strip()
        model_str = self.model_var.get().strip()
        output_dir = Path(self.output_var.get().strip())

        if not input_str:
            messagebox.showerror("Missing input", "Choose an input video first.")
            return
        if not model_str:
            messagebox.showerror("Missing model", "Choose a model first.")
            return

        input_path = Path(input_str)
        model_path = Path(model_str)

        if model_path.suffix.lower() not in {".onnx", ".pt"}:
            messagebox.showerror("Wrong model type", "Choose a .onnx or .pt model.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        self._last_output_dir = output_dir

        args = build_args(
            input_path=str(input_path),
            model_path=str(model_path),
            output_dir=str(output_dir),
            conf=float(self.conf_var.get()),
            nms_iou=float(self.nms_iou_var.get()),
            imgsz=int(self.imgsz_var.get()),
            track_iou=float(self.track_iou_var.get()),
            max_track_age=int(self.max_track_age_var.get()),
            min_track_frames=int(self.min_track_frames_var.get()),
            line_thickness=int(self.line_thickness_var.get()),
            max_dist_px=float(self.max_dist_px_var.get()),
        )

        # Attach log handler so pipeline logger feeds into the status box.
        self._log_handler = _QueueLogHandler(self.events)
        self._log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        pipeline_logger = logging.getLogger("bee_video_pipeline")
        pipeline_logger.addHandler(self._log_handler)
        pipeline_logger.setLevel(logging.DEBUG)

        self._cancel_event.clear()
        self._clear_log()
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self._open_folder_button.configure(state="disabled")
        self.status_var.set("Processing…")
        self.progress_bar["value"] = 0

        self.worker = threading.Thread(
            target=self._run_pipeline,
            args=(args,),
            daemon=True,
        )
        self.worker.start()

    def _cancel_run(self) -> None:
        self._cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status_var.set("Cancelling…")

    def _run_pipeline(self, args) -> None:
        def on_progress(payload: dict[str, object]) -> None:
            self.events.put(("progress", payload))

        try:
            annotated_video, summary_json = process_video(
                args,
                progress_callback=on_progress,
                cancel_event=self._cancel_event,
            )
            summary = json.loads(summary_json.read_text())
            self.events.put(
                (
                    "done",
                    {
                        "annotated_video": str(annotated_video),
                        "summary_json": str(summary_json),
                        "summary": summary,
                        "cancelled": self._cancel_event.is_set(),
                    },
                )
            )
        except Exception as exc:
            self.events.put(("error", str(exc)))

    # ------------------------------------------------------------------
    # Event polling
    # ------------------------------------------------------------------

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "progress":
                    self._handle_progress(payload)
                elif event == "done":
                    self._handle_done(payload)
                elif event == "error":
                    self._handle_error(payload)
                elif event == "log":
                    self._append_log(str(payload))
        except queue.Empty:
            pass
        finally:
            self.root.after(150, self._poll_events)

    def _handle_progress(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        frame_index = int(data.get("frame_index", 0)) + 1
        total_frames = data.get("total_frames")
        active_bees = data.get("active_bees", 0)
        confirmed = data.get("unique_bees_confirmed", 0)

        if total_frames:
            total = int(total_frames)
            pct = min(100, int(frame_index / total * 100))
            self.progress_bar["value"] = pct
            self.status_var.set(
                f"Frame {frame_index}/{total} ({pct}%) | Active: {active_bees} | Confirmed: {confirmed}"
            )
        else:
            self.status_var.set(
                f"Frame {frame_index} | Active: {active_bees} | Confirmed: {confirmed}"
            )

    def _handle_done(self, payload: object) -> None:
        self._detach_log_handler()
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.progress_bar["value"] = 100

        data = payload if isinstance(payload, dict) else {}
        cancelled = data.get("cancelled", False)
        summary = data.get("summary", {})

        if cancelled:
            self.status_var.set("Cancelled. Partial output was saved.")
        else:
            self.status_var.set("Finished. Annotated video and summary JSON written.")

        if self._last_output_dir is not None:
            self._open_folder_button.configure(state="normal")

        message = (
            f"Annotated video : {data.get('annotated_video', '')}\n"
            f"Summary JSON    : {data.get('summary_json', '')}\n\n"
            f"Confirmed unique bees   : {summary.get('unique_bees_confirmed', 0)}\n"
            f"Unique bee IDs seen     : {summary.get('unique_bee_ids_seen', 0)}\n"
            f"Max active bees / frame : {summary.get('max_active_bees_in_frame', 0)}\n"
            f"Frames processed        : {summary.get('total_frames_processed', 0)}"
        )
        self._append_log(message)

    def _handle_error(self, payload: object) -> None:
        self._detach_log_handler()
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.progress_bar["value"] = 0
        message = str(payload)
        self.status_var.set("Pipeline failed. See log for details.")
        self._append_log(f"ERROR: {message}")
        messagebox.showerror("Pipeline failed", message)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.output_text.configure(state="normal")
        self.output_text.insert(tk.END, f"[{now}] {text}\n")
        self.output_text.see(tk.END)
        self.output_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state="disabled")

    def _detach_log_handler(self) -> None:
        if self._log_handler is not None:
            logging.getLogger("bee_video_pipeline").removeHandler(self._log_handler)
            self._log_handler = None

    # ------------------------------------------------------------------
    # Open output folder
    # ------------------------------------------------------------------

    def _open_output_folder(self) -> None:
        if self._last_output_dir is None:
            return
        folder = str(self._last_output_dir)
        try:
            # os.startfile exists on Windows; fall back to xdg-open / open elsewhere.
            if hasattr(os, "startfile"):
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                opener = shutil.which("xdg-open") or shutil.which("open")
                if opener:
                    subprocess.Popen([opener, folder])
                else:
                    messagebox.showinfo("Output folder", folder)
        except Exception as exc:
            messagebox.showerror("Could not open folder", str(exc))


def main() -> None:
    root = tk.Tk()
    app = BeeVideoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

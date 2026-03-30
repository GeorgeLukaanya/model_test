#!/usr/bin/env python3
"""Simple desktop UI for the bee video pipeline."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from bee_video_pipeline import build_args, process_video


class BeeVideoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Bee Video Tracker")
        self.root.geometry("860x720")
        self.root.minsize(760, 620)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None

        default_model = Path("best.onnx")
        if not default_model.exists():
            default_model = Path("best.pt")
        self.model_var = tk.StringVar(value=str(default_model.resolve()))
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path("outputs").resolve()))
        self.conf_var = tk.DoubleVar(value=0.25)
        self.nms_iou_var = tk.DoubleVar(value=0.45)
        self.imgsz_var = tk.IntVar(value=640)
        self.track_iou_var = tk.DoubleVar(value=0.30)
        self.max_track_age_var = tk.IntVar(value=20)
        self.min_track_frames_var = tk.IntVar(value=3)
        self.line_thickness_var = tk.IntVar(value=2)
        self.max_dist_px_var = tk.DoubleVar(value=150.0)
        self.status_var = tk.StringVar(value="Choose a video and click Run pipeline.")
        self.progress_var = tk.StringVar(value="Idle")

        self._build_layout()
        self.root.after(150, self._poll_events)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        container = ttk.Frame(self.root, padding=16)
        container.grid(sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)
        container.rowconfigure(5, weight=1)

        title = ttk.Label(
            container,
            text="Bee Detection and Tracking",
            font=("TkDefaultFont", 16, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            container,
            text="Select a model and a video, then export an annotated video plus a summary JSON.",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 12))

        form = ttk.LabelFrame(container, text="Inputs", padding=12)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        self._add_path_row(form, 0, "Model (.onnx or .pt)", self.model_var, self._browse_model)
        self._add_path_row(form, 1, "Input video", self.input_var, self._browse_video)
        self._add_path_row(form, 2, "Output folder", self.output_var, self._browse_output_dir)

        tuning = ttk.LabelFrame(container, text="Tracking Settings", padding=12)
        tuning.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        for col in range(4):
            tuning.columnconfigure(col, weight=1)

        self._add_spinbox(tuning, 0, 0, "Confidence", self.conf_var, 0.05, 1.0, 0.01)
        self._add_spinbox(tuning, 0, 1, "NMS IoU", self.nms_iou_var, 0.05, 0.95, 0.01)
        self._add_spinbox(tuning, 0, 2, "Image size", self.imgsz_var, 320, 1280, 32)
        self._add_spinbox(tuning, 0, 3, "Track IoU", self.track_iou_var, 0.05, 0.95, 0.01)
        self._add_spinbox(tuning, 1, 0, "Max track age", self.max_track_age_var, 1, 200, 1)
        self._add_spinbox(tuning, 1, 1, "Min track frames", self.min_track_frames_var, 1, 50, 1)
        self._add_spinbox(tuning, 1, 2, "Line thickness", self.line_thickness_var, 1, 8, 1)
        self._add_spinbox(tuning, 1, 3, "Max dist (px)", self.max_dist_px_var, 10, 500, 10)

        actions = ttk.Frame(container, padding=(0, 12, 0, 0))
        actions.grid(row=4, column=0, sticky="ew")
        actions.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(actions, text="Run pipeline", command=self._start_run)
        self.run_button.grid(row=0, column=0, sticky="w")

        progress = ttk.Label(actions, textvariable=self.progress_var)
        progress.grid(row=0, column=1, sticky="e")

        status_frame = ttk.LabelFrame(container, text="Status", padding=12)
        status_frame.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(1, weight=1)

        status_label = ttk.Label(status_frame, textvariable=self.status_var, wraplength=760)
        status_label.grid(row=0, column=0, sticky="ew")

        self.output_text = tk.Text(status_frame, height=14, wrap="word")
        self.output_text.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
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
            frame,
            textvariable=variable,
            from_=from_,
            to=to,
            increment=increment,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

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

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_var.set(path)

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

        self._set_output_text("")
        self.run_button.configure(state="disabled")
        self.status_var.set("Processing video. This can take a while on longer files.")
        self.progress_var.set("Starting...")

        self.worker = threading.Thread(
            target=self._run_pipeline,
            args=(args,),
            daemon=True,
        )
        self.worker.start()

    def _run_pipeline(self, args) -> None:
        def on_progress(payload: dict[str, object]) -> None:
            self.events.put(("progress", payload))

        try:
            annotated_video, summary_json = process_video(args, progress_callback=on_progress)
            summary = json.loads(summary_json.read_text())
            self.events.put(
                (
                    "done",
                    {
                        "annotated_video": str(annotated_video),
                        "summary_json": str(summary_json),
                        "summary": summary,
                    },
                )
            )
        except Exception as exc:
            self.events.put(("error", str(exc)))

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
            self.progress_var.set(
                f"Frame {frame_index}/{int(total_frames)} | Active: {active_bees} | Confirmed: {confirmed}"
            )
        else:
            self.progress_var.set(
                f"Frame {frame_index} | Active: {active_bees} | Confirmed: {confirmed}"
            )

    def _handle_done(self, payload: object) -> None:
        self.run_button.configure(state="normal")
        self.progress_var.set("Completed")
        data = payload if isinstance(payload, dict) else {}
        summary = data.get("summary", {})
        self.status_var.set("Finished. The annotated video and summary JSON were written successfully.")
        message = (
            f"Annotated video: {data.get('annotated_video', '')}\n"
            f"Summary JSON: {data.get('summary_json', '')}\n\n"
            f"Confirmed unique bees: {summary.get('unique_bees_confirmed', 0)}\n"
            f"Unique bee IDs seen: {summary.get('unique_bee_ids_seen', 0)}\n"
            f"Max active bees in frame: {summary.get('max_active_bees_in_frame', 0)}\n"
            f"Frames processed: {summary.get('total_frames_processed', 0)}"
        )
        self._set_output_text(message)

    def _handle_error(self, payload: object) -> None:
        self.run_button.configure(state="normal")
        self.progress_var.set("Failed")
        message = str(payload)
        self.status_var.set("The pipeline failed. Review the error below.")
        self._set_output_text(message)
        messagebox.showerror("Pipeline failed", message)

    def _set_output_text(self, text: str) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    app = BeeVideoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

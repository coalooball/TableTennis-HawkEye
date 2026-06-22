from __future__ import annotations

import base64
import csv
from threading import RLock
from pathlib import Path

import cv2
from pydantic import BaseModel, ConfigDict


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(word.capitalize() for word in rest)


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

from ball_detector import DEFAULT_BALL_MODEL, YoloBallDetector
from ball_tracker import BallTracker
from inference import DEFAULT_MODEL, PoseActionEngine
from landing_detector import LandingDetector
from speed_estimator import BallSpeedEstimator
from table_calibration import TableCalibrator


class AppState(ApiModel):
    source: str | None
    has_capture: bool
    has_frame: bool
    playing: bool
    frame_index: int
    fps: float
    video_name: str | None
    video_file_size_bytes: int | None
    video_width: int | None
    video_height: int | None
    video_frame_count: int | None
    video_duration_seconds: float | None
    pose_model_path: str
    ball_model_path: str
    action: str
    detail: str
    status: str


class SourceRequest(ApiModel):
    path: str | None = None


class ConfidenceRequest(ApiModel):
    confidence: float


class ModelPathRequest(ApiModel):
    path: str


class SaveFrameRequest(ApiModel):
    path: str = "outputs/frame.jpg"


class TablePointRequest(ApiModel):
    x: int
    y: int


class AnalyzeResult(ApiModel):
    video_path: str
    csv_path: str


class FrameResponse(ApiModel):
    state: AppState
    frame: str | None = None


class TableTennisBackend:
    def __init__(self) -> None:
        self.engine: PoseActionEngine | None = None
        self.ball_detector: YoloBallDetector | None = None
        self.capture: cv2.VideoCapture | None = None
        self.raw_frame = None
        self.current_frame = None
        self.current_source: str | int | None = None
        self.video_name: str | None = None
        self.video_file_size_bytes: int | None = None
        self.video_width: int | None = None
        self.video_height: int | None = None
        self.video_frame_count: int | None = None
        self.video_duration_seconds: float | None = None
        self.playing = False
        self.frame_index = 0
        self.source_fps = 30.0
        self.confidence = 0.35
        self.model_path = DEFAULT_MODEL
        self.ball_model_path = DEFAULT_BALL_MODEL
        self.action_text = "Action: -"
        self.detail_text = f"Model: {self.model_path}"
        self.status_text = "Select a video or camera"
        self.lock = RLock()

        self.ball_tracker = BallTracker()
        self.speed_estimator = BallSpeedEstimator()
        self.table_calibrator = TableCalibrator()
        self.landing_detector = LandingDetector()

    def state(self) -> AppState:
        return AppState(
            source=None if self.current_source is None else str(self.current_source),
            has_capture=self.capture is not None,
            has_frame=self.raw_frame is not None,
            playing=self.playing,
            frame_index=self.frame_index,
            fps=self.source_fps,
            video_name=self.video_name,
            video_file_size_bytes=self.video_file_size_bytes,
            video_width=self.video_width,
            video_height=self.video_height,
            video_frame_count=self.video_frame_count,
            video_duration_seconds=self.video_duration_seconds,
            pose_model_path=str(self.model_path),
            ball_model_path=str(self.ball_model_path),
            action=self.action_text,
            detail=self.detail_text,
            status=self.status_text,
        )

    def close_capture(self) -> None:
        self.playing = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.video_name = None
        self.video_file_size_bytes = None
        self.video_width = None
        self.video_height = None
        self.video_frame_count = None
        self.video_duration_seconds = None

    def clear_source(self) -> FrameResponse:
        self.close_capture()
        self.current_source = None
        self.raw_frame = None
        self.current_frame = None
        self.frame_index = 0
        self.source_fps = 30.0
        self.action_text = "Action: -"
        self.detail_text = f"Model: {self.model_path}"
        self.clear_tracking()
        self.status_text = "Select a video or camera"
        return self.frame_response()

    def open_video(self, path: str) -> FrameResponse:
        if not path:
            self.status_text = "No video selected"
            return self.frame_response()
        self.close_capture()
        return self.open_capture(path)

    def open_camera(self) -> FrameResponse:
        self.close_capture()
        return self.open_capture(0)

    def open_capture(self, source: str | int) -> FrameResponse:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            self.status_text = f"Unable to open source: {source}"
            return self.frame_response()

        self.capture = capture
        self.current_source = source
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps if fps and fps > 0 else 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = width if width > 0 else None
        self.video_height = height if height > 0 else None
        self.video_frame_count = frame_count if frame_count > 0 else None
        self.video_duration_seconds = frame_count / self.source_fps if frame_count > 0 and self.source_fps > 0 else None
        if isinstance(source, str):
            source_path = Path(source)
            self.video_name = source_path.name
            self.video_file_size_bytes = source_path.stat().st_size if source_path.exists() else None
        else:
            self.video_name = None
            self.video_file_size_bytes = None
        self.frame_index = 0
        self.clear_tracking()

        ok, frame = capture.read()
        if ok:
            self.process_frame(frame, run_pose=False, update_tracking=False)
            self.frame_index = 1
        self.status_text = str(source)
        return self.frame_response()

    def start(self) -> AppState:
        if self.capture is not None:
            self.playing = True
            self.status_text = "Playing"
        return self.state()

    def stop(self) -> AppState:
        self.playing = False
        self.status_text = "Stopped"
        return self.state()

    def next_frame(self) -> FrameResponse:
        if self.capture is None:
            return self.frame_response()
        if not self.playing:
            return self.frame_response()

        ok, frame = self.capture.read()
        if not ok:
            self.playing = False
            self.status_text = "End of source"
            return self.frame_response()

        run_pose = self.ensure_engine()
        if not self.ensure_ball_detector():
            return self.frame_response()
        self.process_frame(frame, run_pose=run_pose, update_tracking=True)
        self.frame_index += 1
        return self.frame_response()

    def infer_current_frame(self) -> FrameResponse:
        if self.raw_frame is None:
            return self.frame_response()
        if self.ensure_engine():
            self.process_frame(self.raw_frame, run_pose=True, update_tracking=False)
        return self.frame_response()

    def set_confidence(self, confidence: float) -> AppState:
        self.confidence = max(0.1, min(0.9, confidence))
        if self.engine is not None:
            self.engine.conf = self.confidence
        self.status_text = f"Confidence: {self.confidence:.2f}"
        return self.state()

    def set_model_path(self, path: str) -> AppState:
        model_path = Path(path).expanduser()
        if not model_path.exists():
            self.status_text = f"Pose model not found: {model_path}"
            return self.state()
        if model_path.suffix.lower() not in {".pt", ".onnx"}:
            self.status_text = f"Unsupported pose model type: {model_path.suffix}"
            return self.state()
        self.model_path = model_path
        self.engine = None
        self.action_text = "Action: -"
        self.detail_text = f"Model: {self.model_path}"
        self.status_text = f"Pose model path set: {self.model_path}"
        return self.state()

    def set_ball_model_path(self, path: str) -> AppState:
        model_path = Path(path).expanduser()
        if not model_path.exists():
            self.status_text = f"Ball model not found: {model_path}"
            return self.state()
        if model_path.suffix.lower() not in {".pt", ".onnx"}:
            self.status_text = f"Unsupported ball model type: {model_path.suffix}"
            return self.state()
        self.ball_model_path = model_path
        self.ball_detector = None
        self.clear_tracking()
        self.status_text = f"Ball model path set: {self.ball_model_path}"
        return self.state()

    def save_current_frame(self, path: str) -> AppState:
        if self.current_frame is None:
            self.status_text = "No frame to save"
            return self.state()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), self.current_frame)
        self.status_text = f"Saved {output_path}"
        return self.state()

    def calibrate_table(self) -> FrameResponse:
        self.table_calibrator.reset()
        self.clear_tracking()
        self.status_text = self.table_calibrator.next_prompt()
        if self.raw_frame is not None:
            self.process_frame(self.raw_frame, run_pose=False, update_tracking=False)
        return self.frame_response()

    def add_table_point(self, x: int, y: int) -> FrameResponse:
        if self.raw_frame is None:
            self.status_text = "No frame to calibrate"
            return self.frame_response()
        self.table_calibrator.add_point((x, y))
        self.status_text = self.table_calibrator.next_prompt()
        self.process_frame(self.raw_frame, run_pose=False, update_tracking=False)
        return self.frame_response()

    def clear_tracking(self) -> AppState:
        self.ball_tracker.clear()
        self.speed_estimator.clear()
        self.landing_detector.clear()
        return self.state()

    def analyze_current_video(self) -> AnalyzeResult:
        if not isinstance(self.current_source, str):
            raise ValueError("Open a video file before running offline analysis.")
        if not self.ensure_ball_detector():
            raise ValueError(self.status_text)

        source_path = Path(self.current_source)
        if not source_path.exists():
            raise ValueError(f"Video not found: {source_path}")

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {source_path}")

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"annotated_{source_path.stem}.mp4"
        csv_path = output_dir / f"trajectory_{source_path.stem}.csv"

        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"Unable to create output video: {video_path}")

        old_tracker = self.ball_tracker
        old_speed = self.speed_estimator
        old_landing = self.landing_detector
        self.ball_tracker = BallTracker()
        self.speed_estimator = BallSpeedEstimator()
        self.landing_detector = LandingDetector()

        try:
            with csv_path.open("w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    [
                        "frame_index",
                        "time_seconds",
                        "x",
                        "y",
                        "confidence",
                        "accepted",
                        "speed_px_s",
                        "speed_m_s",
                        "speed_km_h",
                    ]
                )

                frame_index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    annotated = self.annotate_frame(frame, run_pose=False, update_tracking=True, frame_index=frame_index, fps=fps)
                    writer.write(annotated)
                    detection = self.ball_tracker.last_detection
                    point = self.ball_tracker.trajectory[-1] if self.ball_tracker.trajectory else None
                    speed_px = self.speed_estimator.last_speed_px_s
                    speed_m = self.speed_estimator.last_speed_m_s
                    csv_writer.writerow(
                        [
                            frame_index,
                            f"{frame_index / fps:.3f}",
                            "" if point is None else point[0],
                            "" if point is None else point[1],
                            "" if detection is None else f"{detection.confidence:.4f}",
                            bool(detection and detection.accepted),
                            "" if speed_px is None else f"{speed_px:.3f}",
                            "" if speed_m is None else f"{speed_m:.5f}",
                            "" if speed_m is None else f"{speed_m * 3.6:.3f}",
                        ]
                    )
                    frame_index += 1
        finally:
            capture.release()
            writer.release()
            self.ball_tracker = old_tracker
            self.speed_estimator = old_speed
            self.landing_detector = old_landing

        self.status_text = f"Saved {video_path} and {csv_path}"
        return AnalyzeResult(video_path=str(video_path), csv_path=str(csv_path))

    def ensure_engine(self) -> bool:
        if self.engine is None:
            try:
                self.engine = PoseActionEngine(model_path=self.model_path, conf=self.confidence)
                self.status_text = "Model loaded"
            except Exception as exc:
                self.status_text = f"Model error: {exc}"
                return False
        self.engine.conf = self.confidence
        return True

    def ensure_ball_detector(self) -> bool:
        if self.ball_detector is not None:
            return True
        try:
            self.ball_detector = YoloBallDetector(self.ball_model_path)
            self.status_text = f"Ball model loaded: {self.ball_model_path}"
            return True
        except Exception as exc:
            self.status_text = f"Ball model error: {exc}"
            return False

    def process_frame(self, frame, run_pose: bool, update_tracking: bool) -> None:
        self.raw_frame = frame.copy()
        display = self.annotate_frame(frame, run_pose, update_tracking, self.frame_index, self.source_fps)
        self.current_frame = display

    def annotate_frame(self, frame, run_pose: bool, update_tracking: bool, frame_index: int, fps: float):
        display = frame.copy()
        if run_pose:
            assert self.engine is not None
            display, action = self.engine.infer_frame(display)
            self.action_text = f"Action: {action.label} ({action.confidence:.2f})"
            self.detail_text = f"{action.reason}\nModel: {self.engine.model_path.name}"

        if update_tracking:
            if not self.ensure_ball_detector():
                return display
            detection = self.ball_detector.detect(frame, predicted=self.ball_tracker.predict_next())
            tracked_point = self.ball_tracker.update(detection)
            self.speed_estimator.update(frame_index, tracked_point, fps, self.table_calibrator)

        self.ball_tracker.draw(display, show_trajectory=True)
        self.speed_estimator.draw(display)
        self.landing_detector.update(list(self.ball_tracker.trajectory), self.table_calibrator)
        self.landing_detector.draw(display)

        if self.table_calibrator.points:
            self.table_calibrator.draw(display)

        return display

    def frame_response(self) -> FrameResponse:
        frame = None
        if self.current_frame is not None:
            ok, encoded = cv2.imencode(".jpg", self.current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if ok:
                frame = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        return FrameResponse(state=self.state(), frame=frame)

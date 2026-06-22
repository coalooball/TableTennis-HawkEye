from __future__ import annotations

import csv
from pathlib import Path

import cv2
from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QMouseEvent, QPixmap
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QSizePolicy, QStatusBar, QWidget

from ball_detector import DEFAULT_BALL_MODEL, YoloBallDetector
from ball_tracker import BallTracker
from inference import DEFAULT_MODEL, PoseActionEngine, bgr_to_qimage
from landing_detector import LandingDetector
from speed_estimator import BallSpeedEstimator
from table_calibration import TableCalibrator
from ui.sidebar import Sidebar


APP_ICON_PATH = Path(__file__).resolve().with_name("app.webp")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TableTennis-HawkEye")
        self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1180, 760)

        self.engine: PoseActionEngine | None = None
        self.ball_detector: YoloBallDetector | None = None
        self.capture: cv2.VideoCapture | None = None
        self.raw_frame = None
        self.current_frame = None
        self.current_source: str | int | None = None
        self.playing = False
        self.frame_index = 0
        self.source_fps = 30.0
        self.displayed_frame_shape: tuple[int, int] | None = None
        self.displayed_pixmap_size = QSize()
        self.calibrating_table = False

        self.ball_tracker = BallTracker()
        self.speed_estimator = BallSpeedEstimator()
        self.table_calibrator = TableCalibrator()
        self.landing_detector = LandingDetector()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(860, 560)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label.setStyleSheet("background: #111; color: #bbb; border: 1px solid #333;")
        self.video_label.setText("Select an image, video, or camera")
        self.video_label.setScaledContents(False)
        self.video_label.mousePressEvent = self.handle_video_click

        self.sidebar = Sidebar()
        self.sidebar.open_image_requested.connect(self.open_image)
        self.sidebar.open_video_requested.connect(self.open_video)
        self.sidebar.camera_requested.connect(self.open_camera)
        self.sidebar.start_requested.connect(self.start)
        self.sidebar.stop_requested.connect(self.stop)
        self.sidebar.infer_requested.connect(self.infer_current_frame)
        self.sidebar.save_requested.connect(self.save_current_frame)
        self.sidebar.confidence_changed.connect(self.update_confidence)
        self.sidebar.calibrate_table_requested.connect(self.start_table_calibration)
        self.sidebar.clear_trajectory_requested.connect(lambda: self.clear_trajectory())
        self.sidebar.analyze_video_requested.connect(self.analyze_current_video)

        root = QHBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addWidget(self.video_label, 1)
        root.addWidget(self.sidebar, 0)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar())
        self.build_menu()
        self.update_buttons()

    def build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_model = QAction("Select Model", self)
        open_model.triggered.connect(self.select_model)
        file_menu.addAction(open_model)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def select_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Pose Model", "models", "Model (*.pt *.onnx)")
        if not path:
            return
        self.load_engine(Path(path))

    def load_engine(self, model_path: Path = DEFAULT_MODEL) -> bool:
        try:
            self.engine = PoseActionEngine(model_path=model_path, conf=self.sidebar.confidence())
            self.sidebar.set_model_text(f"Model: {model_path}")
            self.statusBar().showMessage("Model loaded", 3000)
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Model Error", str(exc))
            return False

    def ensure_engine(self) -> bool:
        if self.engine is None:
            return self.load_engine(DEFAULT_MODEL)
        self.engine.conf = self.sidebar.confidence()
        return True

    def ensure_ball_detector(self) -> bool:
        if self.ball_detector is not None:
            return True
        try:
            self.ball_detector = YoloBallDetector(DEFAULT_BALL_MODEL)
            self.statusBar().showMessage(f"Ball model loaded: {DEFAULT_BALL_MODEL}", 3000)
            return True
        except Exception as exc:
            self.stop()
            QMessageBox.critical(self, "Ball Model Error", str(exc))
            return False

    def open_image(self) -> None:
        self.close_capture()
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Open Image", "Unable to read image.")
            return
        self.current_source = path
        self.frame_index = 0
        self.source_fps = 30.0
        self.ball_tracker.clear()
        self.speed_estimator.clear()
        self.landing_detector.clear()
        self.process_frame(frame, run_pose=False, update_tracking=False)
        self.statusBar().showMessage(path)
        self.update_buttons()

    def open_video(self) -> None:
        self.close_capture()
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.mov *.avi *.mkv *.webm)")
        if not path:
            return
        self.open_capture(path)

    def open_camera(self) -> None:
        self.close_capture()
        self.open_capture(0)

    def open_capture(self, source: str | int) -> None:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            QMessageBox.warning(self, "Open Source", f"Unable to open source: {source}")
            return
        self.capture = capture
        self.current_source = source
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps if fps and fps > 0 else 30.0
        self.frame_index = 0
        self.ball_tracker.clear()
        self.speed_estimator.clear()
        self.landing_detector.clear()
        ok, frame = capture.read()
        if ok:
            self.process_frame(frame, run_pose=False, update_tracking=False)
            self.frame_index = 1
        self.statusBar().showMessage(str(source))
        self.update_buttons()

    def start(self) -> None:
        if self.capture is None:
            return
        self.playing = True
        self.timer.start(30)
        self.update_buttons()

    def stop(self) -> None:
        self.playing = False
        self.timer.stop()
        self.update_buttons()

    def close_capture(self) -> None:
        self.playing = False
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.update_buttons()

    def process_next_frame(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.stop()
            return
        run_pose = self.sidebar.realtime_enabled() and self.ensure_engine()
        if not self.ensure_ball_detector():
            return
        self.process_frame(frame, run_pose=run_pose, update_tracking=True)
        self.frame_index += 1

    def infer_current_frame(self) -> None:
        if self.raw_frame is None:
            return
        if self.ensure_engine():
            self.process_frame(self.raw_frame, run_pose=True, update_tracking=False)

    def process_frame(self, frame, run_pose: bool, update_tracking: bool) -> None:
        self.raw_frame = frame.copy()
        display = self.annotate_frame(frame, run_pose, update_tracking, self.frame_index, self.source_fps)
        self.current_frame = display
        self.show_frame(display)

    def annotate_frame(self, frame, run_pose: bool, update_tracking: bool, frame_index: int, fps: float):
        display = frame.copy()
        if run_pose:
            assert self.engine is not None
            display, action = self.engine.infer_frame(display)
            self.sidebar.set_action_text(f"Action: {action.label} ({action.confidence:.2f})")
            self.sidebar.set_model_text(f"{action.reason}\nModel: {self.engine.model_path.name}")

        if update_tracking:
            if not self.ensure_ball_detector():
                return display
            detection = self.ball_detector.detect(frame, predicted=self.ball_tracker.predict_next())
            tracked_point = self.ball_tracker.update(detection)
            self.speed_estimator.update(frame_index, tracked_point, fps, self.table_calibrator)

        self.ball_tracker.draw(display, show_trajectory=self.sidebar.ball_trajectory_enabled())
        self.speed_estimator.draw(display)

        if self.sidebar.landing_points_enabled():
            self.landing_detector.update(list(self.ball_tracker.trajectory), self.table_calibrator)
            self.landing_detector.draw(display)

        if self.table_calibrator.points:
            self.table_calibrator.draw(display)

        return display

    def show_frame(self, frame) -> None:
        image = bgr_to_qimage(frame)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.displayed_frame_shape = (frame.shape[0], frame.shape[1])
        self.displayed_pixmap_size = scaled.size()
        self.video_label.setPixmap(scaled)

    def save_current_frame(self) -> None:
        if self.current_frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Frame", "outputs/frame.jpg", "Images (*.jpg *.png)")
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, self.current_frame)
        self.statusBar().showMessage(f"Saved {path}", 3000)

    def update_confidence(self, confidence: float) -> None:
        if self.engine is not None:
            self.engine.conf = confidence

    def start_table_calibration(self) -> None:
        if self.raw_frame is None:
            return
        self.stop()
        self.calibrating_table = True
        self.table_calibrator.reset()
        self.clear_trajectory(redraw=False)
        self.statusBar().showMessage(self.table_calibrator.next_prompt())
        self.process_frame(self.raw_frame, run_pose=False, update_tracking=False)

    def clear_trajectory(self, redraw: bool = True) -> None:
        self.ball_tracker.clear()
        self.speed_estimator.clear()
        self.landing_detector.clear()
        if redraw and self.raw_frame is not None:
            self.process_frame(self.raw_frame, run_pose=False, update_tracking=False)

    def analyze_current_video(self) -> None:
        if not isinstance(self.current_source, str):
            QMessageBox.warning(self, "Analyze Video", "Open a video file before running offline analysis.")
            return
        if not self.ensure_ball_detector():
            return

        source_path = Path(self.current_source)
        if not source_path.exists():
            QMessageBox.warning(self, "Analyze Video", f"Video not found: {source_path}")
            return

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            QMessageBox.warning(self, "Analyze Video", f"Unable to open video: {source_path}")
            return

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"annotated_{source_path.stem}.mp4"
        csv_path = output_dir / f"trajectory_{source_path.stem}.csv"

        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            QMessageBox.warning(self, "Analyze Video", f"Unable to create output video: {video_path}")
            return

        self.stop()
        old_tracker = self.ball_tracker
        old_speed = self.speed_estimator
        old_landing = self.landing_detector
        self.ball_tracker = BallTracker()
        self.speed_estimator = BallSpeedEstimator()
        self.landing_detector = LandingDetector()
        self.sidebar.analyze_video_button.setEnabled(False)
        self.statusBar().showMessage("Analyzing video...")

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
                    if total > 0 and frame_index % 30 == 0:
                        self.statusBar().showMessage(f"Analyzing video... {frame_index}/{total}")
        finally:
            capture.release()
            writer.release()
            self.ball_tracker = old_tracker
            self.speed_estimator = old_speed
            self.landing_detector = old_landing
            self.sidebar.analyze_video_button.setEnabled(True)
            self.update_buttons()

        self.statusBar().showMessage(f"Saved {video_path} and {csv_path}", 5000)
        QMessageBox.information(self, "Analyze Video", f"Saved:\n{video_path}\n{csv_path}")

    def handle_video_click(self, event: QMouseEvent) -> None:
        if not self.calibrating_table:
            return
        frame_point = self.map_label_point_to_frame(event.position().toPoint())
        if frame_point is None:
            return

        self.table_calibrator.add_point(frame_point)
        if self.table_calibrator.is_complete:
            self.calibrating_table = False
            self.statusBar().showMessage("Table calibrated", 3000)
        else:
            self.statusBar().showMessage(self.table_calibrator.next_prompt())

        if self.raw_frame is not None:
            self.process_frame(self.raw_frame, run_pose=False, update_tracking=False)

    def map_label_point_to_frame(self, point: QPoint) -> tuple[int, int] | None:
        if self.displayed_frame_shape is None or self.displayed_pixmap_size.isEmpty():
            return None

        label_width = self.video_label.width()
        label_height = self.video_label.height()
        pixmap_width = self.displayed_pixmap_size.width()
        pixmap_height = self.displayed_pixmap_size.height()
        offset_x = (label_width - pixmap_width) / 2
        offset_y = (label_height - pixmap_height) / 2

        x = point.x() - offset_x
        y = point.y() - offset_y
        if x < 0 or y < 0 or x > pixmap_width or y > pixmap_height:
            return None

        frame_height, frame_width = self.displayed_frame_shape
        frame_x = int(round(x * frame_width / pixmap_width))
        frame_y = int(round(y * frame_height / pixmap_height))
        return (
            max(0, min(frame_width - 1, frame_x)),
            max(0, min(frame_height - 1, frame_y)),
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.show_frame(self.current_frame)

    def closeEvent(self, event) -> None:
        self.close_capture()
        event.accept()

    def update_buttons(self) -> None:
        self.sidebar.update_buttons(
            has_capture=self.capture is not None,
            has_frame=self.raw_frame is not None,
            playing=self.playing,
        )

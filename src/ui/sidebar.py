from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget


SIDEBAR_WIDTH = 280


class Sidebar(QWidget):
    open_image_requested = Signal()
    open_video_requested = Signal()
    camera_requested = Signal()
    start_requested = Signal()
    stop_requested = Signal()
    infer_requested = Signal()
    save_requested = Signal()
    confidence_changed = Signal(float)
    calibrate_table_requested = Signal()
    clear_trajectory_requested = Signal()
    analyze_video_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self.action_label = QLabel("Action: -")
        self.action_label.setStyleSheet("font-size: 22px; font-weight: 600;")
        self.action_label.setFixedWidth(SIDEBAR_WIDTH - 24)
        self.detail_label = QLabel("Model: models/yolo26n-pose.pt")
        self.detail_label.setWordWrap(True)
        self.detail_label.setFixedWidth(SIDEBAR_WIDTH - 24)
        self.detail_label.setMinimumHeight(72)
        self.detail_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        self.open_image_button = QPushButton("Open Image")
        self.open_video_button = QPushButton("Open Video")
        self.camera_button = QPushButton("Camera")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.infer_button = QPushButton("Infer Current")
        self.save_button = QPushButton("Save Frame")
        self.calibrate_table_button = QPushButton("Calibrate Table")
        self.clear_trajectory_button = QPushButton("Clear Trajectory")
        self.analyze_video_button = QPushButton("Analyze Video")

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(35)
        self.conf_label = QLabel("Confidence: 0.35")

        self.realtime_check = QCheckBox("Realtime video inference")
        self.realtime_check.setChecked(True)
        self.ball_trajectory_check = QCheckBox("Show ball trajectory")
        self.ball_trajectory_check.setChecked(True)
        self.landing_points_check = QCheckBox("Show landing points")
        self.landing_points_check.setChecked(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.action_label)
        layout.addWidget(self.detail_label)
        layout.addSpacing(10)
        layout.addWidget(self.open_image_button)
        layout.addWidget(self.open_video_button)
        layout.addWidget(self.camera_button)
        layout.addSpacing(10)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.infer_button)
        layout.addWidget(self.save_button)
        layout.addSpacing(10)
        layout.addWidget(self.calibrate_table_button)
        layout.addWidget(self.clear_trajectory_button)
        layout.addWidget(self.analyze_video_button)
        layout.addSpacing(10)
        layout.addWidget(self.conf_label)
        layout.addWidget(self.conf_slider)
        layout.addWidget(self.realtime_check)
        layout.addWidget(self.ball_trajectory_check)
        layout.addWidget(self.landing_points_check)
        layout.addStretch()

        self.open_image_button.clicked.connect(self.open_image_requested)
        self.open_video_button.clicked.connect(self.open_video_requested)
        self.camera_button.clicked.connect(self.camera_requested)
        self.start_button.clicked.connect(self.start_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.infer_button.clicked.connect(self.infer_requested)
        self.save_button.clicked.connect(self.save_requested)
        self.calibrate_table_button.clicked.connect(self.calibrate_table_requested)
        self.clear_trajectory_button.clicked.connect(self.clear_trajectory_requested)
        self.analyze_video_button.clicked.connect(self.analyze_video_requested)
        self.conf_slider.valueChanged.connect(self._emit_confidence)

    def confidence(self) -> float:
        return self.conf_slider.value() / 100

    def realtime_enabled(self) -> bool:
        return self.realtime_check.isChecked()

    def ball_trajectory_enabled(self) -> bool:
        return self.ball_trajectory_check.isChecked()

    def landing_points_enabled(self) -> bool:
        return self.landing_points_check.isChecked()

    def set_model_text(self, text: str) -> None:
        self.detail_label.setText(text)

    def set_action_text(self, text: str) -> None:
        self.action_label.setText(text)

    def update_buttons(self, has_capture: bool, has_frame: bool, playing: bool) -> None:
        self.start_button.setEnabled(has_capture and not playing)
        self.stop_button.setEnabled(has_capture or playing)
        self.infer_button.setEnabled(has_frame)
        self.save_button.setEnabled(has_frame)
        self.calibrate_table_button.setEnabled(has_frame)
        self.clear_trajectory_button.setEnabled(has_frame)
        self.analyze_video_button.setEnabled(has_frame)

    def _emit_confidence(self) -> None:
        confidence = self.confidence()
        self.conf_label.setText(f"Confidence: {confidence:.2f}")
        self.confidence_changed.emit(confidence)

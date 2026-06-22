from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from action import ActionResult, SimpleActionRecognizer


DEFAULT_MODEL = Path("models/yolo26n-pose.pt")


class PoseActionEngine:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, conf: float = 0.35):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.recognizer = SimpleActionRecognizer()
        self.conf = conf

    def reset(self) -> None:
        self.recognizer = SimpleActionRecognizer()

    def infer_frame(self, frame: np.ndarray) -> tuple[np.ndarray, ActionResult]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        result = results[0]
        annotated = result.plot()
        keypoints = first_person_keypoints(result)
        action = self.recognizer.update(keypoints)
        draw_action(annotated, action)
        return annotated, action


def first_person_keypoints(result):
    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return None

    xy = result.keypoints.xy.cpu().numpy()
    conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
    if xy.shape[0] == 0:
        return None

    index = largest_box_index(result)
    person_xy = xy[index]
    if conf is None:
        return person_xy
    return np.concatenate([person_xy, conf[index][:, None]], axis=1)


def largest_box_index(result) -> int:
    if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
        return 0
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def draw_action(frame: np.ndarray, action: ActionResult) -> None:
    text = f"{action.label} {action.confidence:.2f}"
    cv2.rectangle(frame, (12, 12), (430, 58), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (24, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2, cv2.LINE_AA)


def bgr_to_qimage(frame: np.ndarray):
    from PySide6.QtGui import QImage

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()

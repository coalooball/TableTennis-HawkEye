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
        self.recognizers: dict[int, SimpleActionRecognizer] = {}
        self.conf = conf

    def reset(self) -> None:
        self.recognizers = {}

    def infer_frame(
        self,
        frame: np.ndarray,
        show_person_boxes: bool = True,
        show_skeleton: bool = True,
        show_pose_labels: bool = True,
    ) -> tuple[np.ndarray, ActionResult]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        result = results[0]
        annotated = result.plot(boxes=show_person_boxes, labels=show_person_boxes, kpt_line=show_skeleton, kpt_radius=5 if show_skeleton else 0)
        people = person_keypoints_and_boxes(result)
        if not people:
            return annotated, SimpleActionRecognizer().update(None)

        action = ActionResult("no_person", 0.0, "no valid person skeleton")
        primary_index = min(largest_box_index(result), len(people) - 1)
        active_indices = set(range(len(people)))
        self.recognizers = {index: recognizer for index, recognizer in self.recognizers.items() if index in active_indices}
        for index, (keypoints, box) in enumerate(people):
            recognizer = self.recognizers.setdefault(index, SimpleActionRecognizer())
            person_action = recognizer.update(keypoints)
            if show_pose_labels:
                draw_action_label(annotated, person_action, box)
            if index == primary_index:
                action = person_action
        return annotated, action


def person_keypoints_and_boxes(result) -> list[tuple[np.ndarray, np.ndarray]]:
    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return []

    xy = result.keypoints.xy.cpu().numpy()
    conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
    if xy.shape[0] == 0:
        return []

    if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
        boxes = np.array([keypoint_box(points) for points in xy])
    else:
        boxes = result.boxes.xyxy.cpu().numpy()

    people = []
    count = min(len(xy), len(boxes))
    for index in range(count):
        person_xy = xy[index]
        if conf is None:
            keypoints = person_xy
        else:
            keypoints = np.concatenate([person_xy, conf[index][:, None]], axis=1)
        people.append((keypoints, boxes[index]))
    return people


def first_person_keypoints(result):
    people = person_keypoints_and_boxes(result)
    if not people:
        return None
    index = min(largest_box_index(result), len(people) - 1)
    return people[index][0]


def keypoint_box(keypoints: np.ndarray) -> np.ndarray:
    if keypoints.size == 0:
        return np.array([0, 0, 0, 0], dtype=float)
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)


def draw_action_label(frame: np.ndarray, action: ActionResult, box: np.ndarray) -> None:
    text = f"{action.label} {action.confidence:.2f}"
    x1, y1, x2, _ = box.astype(int)
    image_height, image_width = frame.shape[:2]
    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    label_x = max(0, min(x1, image_width - text_width - 8))
    label_bottom = max(text_height + baseline + 6, y1 - 6)
    label_top = label_bottom - text_height - baseline - 8
    label_right = min(image_width - 1, label_x + text_width + 8)

    cv2.rectangle(frame, (label_x, label_top), (label_right, label_bottom), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        text,
        (label_x + 4, label_bottom - baseline - 4),
        font,
        font_scale,
        (0, 255, 128),
        thickness,
        cv2.LINE_AA,
    )


def largest_box_index(result) -> int:
    if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
        return 0
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))

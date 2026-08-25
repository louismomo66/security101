"""YOLO11n-pose skeleton detection via ONNX Runtime (ultralytics export format)."""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

# COCO 17-keypoint skeleton connections (parent → child)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # head
    (5, 6),                                 # shoulders
    (5, 7), (7, 9),                         # left arm
    (6, 8), (8, 10),                        # right arm
    (5, 11), (6, 12),                       # torso
    (11, 12),                               # hips
    (11, 13), (13, 15),                     # left leg
    (12, 14), (14, 16),                     # right leg
]

COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Per-limb colours for drawing
_LIMB_COLORS = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
    (255, 51, 255),
    (255, 128, 0), (255, 153, 51),
    (51, 153, 255), (51, 178, 255),
    (51, 255, 51), (51, 255, 153),
    (0, 255, 0),
    (51, 255, 51), (51, 255, 153),
    (0, 153, 255), (0, 178, 255),
]


class YOLOPoseDetector:
    """YOLO11n-pose via ONNX Runtime (CPU).  Ultralytics export format.

    Output tensor shape from ONNX: [1, 56, 8400]
      56 = 4 (bbox cx,cy,w,h) + 1 (person conf) + 51 (17 keypoints × 3)
    """

    def __init__(self, model_path: str, conf: float = 0.5,
                 iou: float = 0.45, img_size: int = 640):
        self.session = ort.InferenceSession(model_path)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size

    # ── preprocess ────────────────────────────────────────────────────────

    def _preprocess(self, img_rgb: np.ndarray):
        h, w = img_rgb.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img_rgb, (nw, nh))
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        top, left = (self.img_size - nh) // 2, (self.img_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW
        return blob, scale, top, left

    # ── detect ────────────────────────────────────────────────────────────

    def detect(self, img_rgb: np.ndarray, conf: float | None = None,
               iou: float | None = None):
        """Run pose detection on an RGB numpy array.

        Returns
        -------
        boxes : np.ndarray[N, 4]   – x1 y1 x2 y2 in original coords
        scores : np.ndarray[N]     – person confidence
        keypoints : np.ndarray[N, 17, 3]  – (x, y, kpt_conf) per joint
        """
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou

        blob, scale, pad_top, pad_left = self._preprocess(img_rgb)
        raw = self.session.run(None, {"images": blob})[0]  # [1, 56, 8400]
        preds = raw[0].T  # [8400, 56]

        # Filter by person confidence (column 4)
        person_scores = preds[:, 4]
        keep = person_scores > conf
        preds = preds[keep]
        person_scores = person_scores[keep]

        # Bounding boxes: cx,cy,w,h → x1,y1,x2,y2 → undo letterbox
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - bw / 2 - pad_left) / scale
        y1 = (cy - bh / 2 - pad_top) / scale
        x2 = (cx + bw / 2 - pad_left) / scale
        y2 = (cy + bh / 2 - pad_top) / scale
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(int)

        # Keypoints: [N, 51] → [N, 17, 3]  then undo letterbox
        kpts = preds[:, 5:].reshape(-1, 17, 3).copy()
        kpts[:, :, 0] = (kpts[:, :, 0] - pad_left) / scale
        kpts[:, :, 1] = (kpts[:, :, 1] - pad_top) / scale

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), person_scores.tolist(), conf, iou,
        )
        if len(indices) == 0:
            return (np.empty((0, 4), int), np.array([]),
                    np.empty((0, 17, 3), np.float32))
        indices = np.array(indices).flatten()
        return boxes[indices], person_scores[indices], kpts[indices]

    # ── drawing ───────────────────────────────────────────────────────────

    @staticmethod
    def draw(img_rgb: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
             keypoints: np.ndarray, kpt_thresh: float = 0.5,
             draw_boxes: bool = True) -> np.ndarray:
        """Draw skeletons, and optionally bounding boxes, on a copy of the image.

        `draw_boxes=False` when the object detector has already boxed the same
        people. Detection and pose are two independent networks with their own
        NMS, so their person boxes land a few pixels apart and stacking both
        overlays renders every person twice.
        """
        vis = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        font_scale = min(h, w) * 0.0006
        thickness = max(1, int(min(h, w) * 0.001))

        for i in range(len(boxes)):
            kpts = keypoints[i]  # [17, 3]
            box = boxes[i]

            # Bounding box
            if draw_boxes:
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]),
                              (0, 255, 128), 2)
                label = f"person {int(scores[i] * 100)}%"
                (tw, th_), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(vis, (box[0], box[1] - int(th_ * 1.4)),
                              (box[0] + tw, box[1]), (0, 255, 128), -1)
                cv2.putText(vis, label, (box[0], box[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 0, 0), thickness, cv2.LINE_AA)

            # Skeleton edges
            for si, (a, b) in enumerate(COCO_SKELETON):
                if kpts[a][2] > kpt_thresh and kpts[b][2] > kpt_thresh:
                    pt1 = (int(kpts[a][0]), int(kpts[a][1]))
                    pt2 = (int(kpts[b][0]), int(kpts[b][1]))
                    cv2.line(vis, pt1, pt2,
                             _LIMB_COLORS[si % len(_LIMB_COLORS)], 2)

            # Keypoint dots
            for kx, ky, kc in kpts:
                if kc > kpt_thresh:
                    cv2.circle(vis, (int(kx), int(ky)), 4, (0, 255, 255), -1)

        return vis

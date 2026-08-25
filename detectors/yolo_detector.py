"""YOLO11n object detection via ONNX Runtime (ultralytics export format)."""

import cv2
import numpy as np
import onnxruntime as ort

from .coco_classes import COCO_CLASSES, COCO_COLORS


class YOLODetector:
    """YOLO11n via ONNX Runtime (CPU).  Ultralytics export format."""

    def __init__(self, model_path, conf=0.5, iou=0.45, img_size=640, nc=None):
        self.session = ort.InferenceSession(model_path)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        # Plain detection exports carry only box+class columns, so every
        # column past the first 4 is a class score. Segmentation exports
        # (e.g. yolov8s-seg) append 32 mask-coefficient columns after the
        # class scores — those are not classes and must not be argmax'd
        # over, or mask coefficients get silently misread as detections of
        # phantom extra classes. Pass nc explicitly for segmentation-format
        # models; leave it None for plain detection models (unchanged
        # behaviour, e.g. the COCO 80-class detector used elsewhere).
        self.nc = nc

    def _preprocess(self, img_rgb):
        """Letterbox-resize to img_size x img_size, normalise to 0-1."""
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

    def detect(self, img_rgb, conf=None, iou=None):
        """Run detection on an RGB numpy array.
        Returns (boxes, scores, class_ids) in original-image coordinates."""
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou

        blob, scale, pad_top, pad_left = self._preprocess(img_rgb)
        raw = self.session.run(None, {'images': blob})[0]  # [1, 84, 8400]
        preds = raw[0].T  # [8400, 84]

        # columns: cx, cy, w, h, cls0..clsN[, mask_coeff0..31]
        cls_scores = preds[:, 4:4 + self.nc] if self.nc is not None else preds[:, 4:]
        max_scores = cls_scores.max(axis=1)
        keep = max_scores > conf
        preds = preds[keep]
        max_scores = max_scores[keep]
        class_ids = cls_scores[keep].argmax(axis=1)

        # Convert cx,cy,w,h -> x1,y1,x2,y2 in letterboxed space
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Undo letterbox -> original image coords
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(int)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), max_scores.tolist(), conf, iou,
        )
        if len(indices) == 0:
            return np.empty((0, 4), int), np.array([]), np.array([], int)
        indices = np.array(indices).flatten()
        return boxes[indices], max_scores[indices], class_ids[indices]

    @staticmethod
    def draw(img_rgb, boxes, scores, class_ids, alpha=0.3):
        """Draw boxes + labels on a copy of the image. Returns RGB."""
        det = img_rgb.copy()
        mask = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        font_scale = min(h, w) * 0.0006
        thickness = max(1, int(min(h, w) * 0.001))

        for cid, box, score in zip(class_ids, boxes, scores):
            color = COCO_COLORS[cid].tolist()
            x1, y1, x2, y2 = box
            cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(det, (x1, y1), (x2, y2), color, 2)
            label = f'{COCO_CLASSES[cid]} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness)
            cv2.rectangle(det, (x1, y1 - int(th * 1.4)), (x1 + tw, y1), color, -1)
            cv2.putText(det, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask, alpha, det, 1 - alpha, 0)

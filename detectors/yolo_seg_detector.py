"""YOLO11n-seg instance segmentation via ONNX Runtime (ultralytics export format)."""

import cv2
import numpy as np
import onnxruntime as ort

from .coco_classes import COCO_CLASSES, COCO_COLORS


class YOLOSegDetector:
    """YOLO11n-seg via ONNX Runtime (CPU).  Ultralytics export format.

    Outputs:
      output0 [1, 116, 8400] -- 4 box + 80 classes + 32 mask coefficients
      output1 [1, 32, 160, 160] -- 32 mask prototypes
    """

    def __init__(self, model_path, conf=0.5, iou=0.45, img_size=640):
        self.session = ort.InferenceSession(model_path)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.mask_h = 160
        self.mask_w = 160

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
        """Run segmentation on an RGB numpy array.
        Returns (boxes, scores, class_ids, masks) where masks is a list of
        boolean masks in original-image coordinates."""
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou
        h_orig, w_orig = img_rgb.shape[:2]

        blob, scale, pad_top, pad_left = self._preprocess(img_rgb)
        outputs = self.session.run(None, {'images': blob})
        raw_det = outputs[0][0]    # [116, 8400]
        protos = outputs[1][0]     # [32, 160, 160]

        preds = raw_det.T  # [8400, 116]

        # columns: cx, cy, w, h, cls0..cls79, mask0..mask31
        cls_scores = preds[:, 4:84]
        max_scores = cls_scores.max(axis=1)
        keep = max_scores > conf
        preds = preds[keep]
        max_scores = max_scores[keep]
        class_ids = cls_scores[keep].argmax(axis=1)

        if len(preds) == 0:
            return np.empty((0, 4), int), np.array([]), np.array([], int), []

        # mask coefficients
        mask_coeffs = preds[:, 84:]  # [N, 32]

        # Convert cx,cy,w,h -> x1,y1,x2,y2 in letterboxed space
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Undo letterbox -> original image coords
        x1_orig = (x1 - pad_left) / scale
        y1_orig = (y1 - pad_top) / scale
        x2_orig = (x2 - pad_left) / scale
        y2_orig = (y2 - pad_top) / scale

        boxes = np.stack([x1_orig, y1_orig, x2_orig, y2_orig], axis=1).astype(int)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), max_scores.tolist(), conf, iou,
        )
        if len(indices) == 0:
            return np.empty((0, 4), int), np.array([]), np.array([], int), []
        indices = np.array(indices).flatten()
        boxes = boxes[indices]
        max_scores = max_scores[indices]
        class_ids = class_ids[indices]
        mask_coeffs = mask_coeffs[indices]

        # Compute per-instance masks: coefficients @ prototypes
        # mask_coeffs [N, 32] @ protos [32, 160*160] -> [N, 160*160]
        raw_masks = mask_coeffs @ protos.reshape(32, -1)  # [N, 25600]
        raw_masks = raw_masks.reshape(-1, self.mask_h, self.mask_w)  # [N, 160, 160]

        # Sigmoid
        raw_masks = 1.0 / (1.0 + np.exp(-raw_masks))

        # Map mask coordinates back from letterboxed 640 -> original image
        # The mask is at 160x160 for 640x640 input (4x downsampled)
        mask_scale = self.mask_h / self.img_size  # 0.25
        pad_top_m = int(pad_top * mask_scale)
        pad_left_m = int(pad_left * mask_scale)
        nh_m = int(int(h_orig * scale) * mask_scale)
        nw_m = int(int(w_orig * scale) * mask_scale)

        # Crop to the non-padded region, then resize to original image size
        masks = []
        for i in range(len(boxes)):
            m = raw_masks[i]
            # Crop out padding
            m_crop = m[pad_top_m:pad_top_m + nh_m, pad_left_m:pad_left_m + nw_m]
            # Resize to original image size
            m_resized = cv2.resize(m_crop, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            # Clip to bounding box for cleaner masks
            bx1, by1, bx2, by2 = boxes[i]
            bx1 = max(0, bx1); by1 = max(0, by1)
            bx2 = min(w_orig, bx2); by2 = min(h_orig, by2)
            box_mask = np.zeros_like(m_resized)
            box_mask[by1:by2, bx1:bx2] = 1.0
            m_resized = m_resized * box_mask
            masks.append(m_resized > 0.5)

        return boxes, max_scores, class_ids, masks

    @staticmethod
    def draw(img_rgb, boxes, scores, class_ids, masks, alpha=0.5):
        """Draw masks + boxes + labels. Returns RGB."""
        out = img_rgb.copy()
        overlay = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        font_scale = min(h, w) * 0.0006
        thickness = max(1, int(min(h, w) * 0.001))

        for i, (cid, box, score) in enumerate(zip(class_ids, boxes, scores)):
            color = COCO_COLORS[cid].tolist()
            mask = masks[i] if i < len(masks) else None

            # Fill mask region with color
            if mask is not None:
                overlay[mask] = color

            # Draw bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f'{COCO_CLASSES[cid]} {int(score * 100)}%'
            (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale, thickness)
            cv2.rectangle(out, (x1, y1 - int(th_t * 1.4)), (x1 + tw, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

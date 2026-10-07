"""YOLO11n object detection via ONNX Runtime (ultralytics export format)."""

import os

import cv2
import numpy as np
import onnxruntime as ort

from .coco_classes import COCO_CLASSES, COCO_COLORS


def _cudnn_lib_dir():
    """Where pip's nvidia-cudnn wheel put its shared libraries, if anywhere.

    ONNX Runtime's CUDA provider dlopen()s libcudnn at session creation and
    does not search Python's site-packages, so a pip-installed cuDNN is
    invisible to it unless the directory is on the loader path. Adding it
    here means CUDA works without the caller having to remember
    LD_LIBRARY_PATH.
    """
    try:
        import nvidia.cudnn  # noqa: F401  (pip wheel, not always present)
    except ImportError:
        return None
    import nvidia.cudnn
    d = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib")
    return d if os.path.isdir(d) else None


def _make_session(model_path, providers=None):
    """Build an inference session: CUDA, CoreML or CPU.

    The weapon detector runs at 960px and dominates frame time. Measured on
    yolov8s-seg at 960 — the same architecture and input size as
    weapons_v2 — with each provider in its own process:

        Tesla V100, CUDA           30.4 ms     <- SENTINEL_ORT_PROVIDER=cuda
        Apple Neural Engine        64   ms     <- SENTINEL_ORT_PROVIDER=coreml
        Xeon, 80 cores            290   ms
        Apple M-series CPU        304   ms

    Two traps are worth knowing, because both fail *quietly*:

    1. A CUDA provider that cannot load its libraries does not raise. ONNX
       Runtime logs a line and silently runs the graph on CPU, so the only
       symptom is that the GPU made no difference. The measurement above was
       nearly missed for exactly this reason: CUDA read 275ms and CPU 272ms,
       because both were the CPU.

    2. Version coupling is strict. onnxruntime-gpu 1.19 wants cuDNN 9; 1.18
       wants cuDNN 8. Installing the wrong pair produces trap 1.

    So after creating a CUDA session we verify the provider actually took,
    and say so loudly if it did not.

    Default stays CPU: every threshold in training/weapons/README.md was
    tuned there. SENTINEL_ORT_PROVIDER picks cuda | coreml | cpu.
    """
    want = providers
    verify_cuda = False
    if want is None:
        choice = os.environ.get("SENTINEL_ORT_PROVIDER", "cpu").lower()
        avail = ort.get_available_providers()
        if choice == "cuda" and "CUDAExecutionProvider" in avail:
            want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            verify_cuda = True
            # Put pip's cuDNN on the loader path before the session is built;
            # ORT resolves the library at construction, so doing it after is
            # too late.
            d = _cudnn_lib_dir()
            if d and d not in os.environ.get("LD_LIBRARY_PATH", ""):
                os.environ["LD_LIBRARY_PATH"] = (
                    d + ":" + os.environ.get("LD_LIBRARY_PATH", "")).rstrip(":")
        elif choice == "coreml" and "CoreMLExecutionProvider" in avail:
            want = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            want = ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(model_path, providers=want)
    except Exception as exc:
        if want == ["CPUExecutionProvider"]:
            raise
        print(f"[yolo] {want[0]} failed ({type(exc).__name__}: {exc}); "
              f"falling back to CPU", flush=True)
        return ort.InferenceSession(model_path,
                                    providers=["CPUExecutionProvider"])

    if verify_cuda and "CUDAExecutionProvider" not in sess.get_providers():
        # Asked for the GPU, got the CPU. Worth shouting about: the run will
        # otherwise look fine and be ten times slower than it should be.
        print("[yolo] CUDA was requested but the session is running on "
              f"{sess.get_providers()}. The provider library did not load — "
              "usually an onnxruntime-gpu / cuDNN version mismatch "
              "(1.19 needs cuDNN 9, 1.18 needs cuDNN 8).", flush=True)
    return sess


class YOLODetector:
    """YOLO11n via ONNX Runtime (CPU, CUDA or CoreML).  Ultralytics export."""

    def __init__(self, model_path, conf=0.5, iou=0.45, img_size=None, nc=None,
                 providers=None):
        self.session = _make_session(model_path, providers)
        self.providers = self.session.get_providers()
        self.conf = conf
        self.iou = iou

        # Read the input size off the model rather than assuming 640. An
        # export at any other imgsz fails outright with "Got invalid
        # dimensions for input: images", which is a confusing way to learn
        # that a perfectly good checkpoint was trained at 960.
        shape = self.session.get_inputs()[0].shape
        if img_size is None:
            img_size = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 640
        self.img_size = img_size
        # Plain detection exports carry only box+class columns, so every
        # column past the first 4 is a class score. Segmentation exports
        # (e.g. yolov8s-seg) append 32 mask-coefficient columns after the
        # class scores — those are not classes and must not be argmax'd
        # over, or mask coefficients get silently misread as detections of
        # phantom extra classes. Pass nc explicitly for segmentation-format
        # models; leave it None for plain detection models (unchanged
        # behaviour, e.g. the COCO 80-class detector used elsewhere).
        #
        # Ultralytics stamps the class names into the ONNX metadata, so when
        # nc is not given it can be read rather than guessed — which stops a
        # segmentation export from silently reporting 34 classes (2 real ones
        # plus 32 mask coefficients) to a caller that never thought to pass it.
        if nc is None:
            meta = (self.session.get_modelmeta().custom_metadata_map or {})
            raw = meta.get("names")
            if raw:
                try:
                    import ast
                    nc = len(ast.literal_eval(raw))
                except (ValueError, SyntaxError):
                    pass
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

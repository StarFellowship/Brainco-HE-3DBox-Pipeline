"""파이프라인 1~3단계 모델 래퍼 (GPU 필요).
각 모델은 독립적으로 lazy-load 되며, 실패해도 나머지 단계는 대체 경로로 동작한다.
  1단계 GroundingDINO : 텍스트 -> 2D box
  2단계 SAM 2.1       : box -> mask (+비디오 추적)
  3단계 UniDepthV2    : RGB -> metric depth,  (WildCamera는 쓰지 않는다 — UniDepthV2가 intrinsics를 함께 출력)
"""
import os, sys, warnings
import numpy as np
import torch

ROOT = os.path.expanduser("~/task3")
warnings.filterwarnings("ignore")


def vram(dev=0):
    """현재 GPU 사용량(GB)"""
    try:
        return torch.cuda.memory_allocated(dev) / 1e9
    except Exception:
        return 0.0


# ------------------------------------------------------------------ 1단계
class Detector:
    """GroundingDINO — open-vocabulary 2D 검출.
    로컬 레포(CUDA 확장 빌드 필요)가 실패하면 transformers 구현으로 자동 폴백한다.
    두 경로 모두 동일 가중치 계열이라 결과는 동등하다."""

    def __init__(self, device="cuda:0", box_thr=0.35, text_thr=0.25):
        self.device, self.box_thr, self.text_thr = device, box_thr, text_thr
        self.backend = None
        # 1순위: 로컬 GroundingDINO
        try:
            sys.path.insert(0, f"{ROOT}/models/GroundingDINO")
            from groundingdino.util.inference import load_model
            cfg = f"{ROOT}/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            ckpt = f"{ROOT}/models/GroundingDINO/weights/groundingdino_swint_ogc.pth"
            self.model = load_model(cfg, ckpt).to(device).eval()
            self.backend = "local"
        except Exception as e:
            print(f"[Detector] 로컬 GroundingDINO 실패 -> transformers 폴백: {str(e)[:100]}")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            mid = "IDEA-Research/grounding-dino-base"
            self.proc = AutoProcessor.from_pretrained(mid)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device).eval()
            self.backend = "hf"

    def __call__(self, image_bgr, prompt):
        """반환: boxes (N,4) xyxy 픽셀, phrases, scores"""
        H, W = image_bgr.shape[:2]
        from PIL import Image
        pil = Image.fromarray(image_bgr[:, :, ::-1])

        if self.backend == "local":
            import groundingdino.datasets.transforms as T
            from groundingdino.util.inference import predict
            tf = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            img_t, _ = tf(pil, None)
            boxes, logits, phrases = predict(self.model, img_t, prompt.lower().strip(),
                                             self.box_thr, self.text_thr, device=self.device)
            if len(boxes) == 0:
                return np.empty((0, 4)), [], np.empty(0)
            b = boxes.cpu().numpy()
            xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H,
                             (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H], axis=1)
            return xyxy, list(phrases), logits.cpu().numpy()

        # transformers 경로
        inputs = self.proc(images=pil, text=prompt.lower().strip(), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_thr,
            text_threshold=self.text_thr, target_sizes=[(H, W)])[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res.get("text_labels", res.get("labels"))
        phrases = [str(x) for x in labels]
        return boxes, phrases, scores


# ------------------------------------------------------------------ 2단계
class Segmenter:
    """SAM 2.1 — 박스 프롬프트 -> 마스크.
    로컬 sam2 로드 실패 시 HF SAM2 -> SAM1 순으로 폴백한다."""

    def __init__(self, device="cuda:0", size="large"):
        self.device = device
        self.backend = None
        try:                                   # 1순위: 로컬 sam2
            sys.path.insert(0, f"{ROOT}/models/sam2")
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            ckpt = f"{ROOT}/models/sam2/checkpoints/sam2.1_hiera_large.pt"
            self.model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt, device=device)
            self.predictor = SAM2ImagePredictor(self.model)
            self.backend = "sam2_local"
        except Exception as e1:
            print(f"[Segmenter] 로컬 sam2 실패: {str(e1)[:90]}")
            try:                               # 2순위: HF SAM2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
                self.backend = "sam2_hf"
            except Exception as e2:
                print(f"[Segmenter] HF sam2 실패 -> SAM1 폴백: {str(e2)[:90]}")
                from transformers import SamModel, SamProcessor
                self.proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
                self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()
                self.backend = "sam1"

    def __call__(self, image_bgr, boxes_xyxy):
        """반환: masks (N,H,W) bool"""
        if len(boxes_xyxy) == 0:
            return np.empty((0,) + image_bgr.shape[:2], dtype=bool)
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])   # negative stride 방지
        if self.backend in ("sam2_local", "sam2_hf"):
            self.predictor.set_image(rgb)
            masks, _, _ = self.predictor.predict(box=boxes_xyxy, multimask_output=False)
            m = masks if masks.ndim == 3 else masks[:, 0]
            return m.astype(bool)
        # SAM1
        from PIL import Image
        pil = Image.fromarray(rgb)
        inputs = self.proc(pil, input_boxes=[[list(map(float, b)) for b in boxes_xyxy]],
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=False)
        masks = self.proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0]
        return masks.squeeze(1).numpy().astype(bool)


# ------------------------------------------------------------------ 3단계
class DepthEstimator:
    """UniDepthV2 — metric depth + intrinsics.
    로컬/HF 로드가 모두 실패하면 Depth-Anything-V2-Metric으로 폴백(그 경우 K는 None)."""

    def __init__(self, device="cuda:0", model_name="lpiccinelli/unidepth-v2-vitl14"):
        self.device = device
        self.backend = None
        try:
            sys.path.insert(0, f"{ROOT}/models/UniDepth")
            from unidepth.models import UniDepthV2
            self.model = UniDepthV2.from_pretrained(model_name).to(device).eval()
            self.backend = "unidepth"
        except Exception as e:
            print(f"[Depth] UniDepthV2 실패 -> Depth-Anything-V2 폴백: {str(e)[:100]}")
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            mid = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
            self.proc = AutoImageProcessor.from_pretrained(mid)
            self.model = AutoModelForDepthEstimation.from_pretrained(mid).to(device).eval()
            self.backend = "dav2"

    @torch.no_grad()
    def __call__(self, image_bgr):
        """반환: depth (H,W) meters, K (3,3) 또는 None"""
        if self.backend == "unidepth":
            rgb = torch.from_numpy(image_bgr[:, :, ::-1].copy()).permute(2, 0, 1).to(self.device)
            pred = self.model.infer(rgb)
            depth = pred["depth"].squeeze().cpu().numpy()
            K = pred.get("intrinsics")
            K = K.squeeze().cpu().numpy() if K is not None else None
            return depth, K
        from PIL import Image
        H, W = image_bgr.shape[:2]
        pil = Image.fromarray(image_bgr[:, :, ::-1])
        inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        d = torch.nn.functional.interpolate(out.predicted_depth.unsqueeze(1), size=(H, W),
                                            mode="bicubic", align_corners=False)
        return d.squeeze().cpu().numpy(), None



import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

TREE_MODEL_PATH = "models/tree_model.pt"
STICK_MODEL_PATH = "models/stick_model.pt"
CLASSIFIER_PATH = "models/classifier.pth"
CLASS_NAMES_RU = ["берёза", "дуб", "ель", "сосна", "тополь"]

REAL_STICK_LENGTH_M = 1.0  # длина палки в метрах

app = FastAPI(title="ArborScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # потом можно ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return None
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    if mask.max() == 0:
        return mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def measure_tree(mask: np.ndarray, meters_per_px: float):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or meters_per_px is None:
        return None, None, None
    y_min, y_max = ys.min(), ys.max()
    height_px = y_max - y_min
    height_m = height_px * meters_per_px

    crown_top = int(y_min)
    crown_bot = int(y_min + 0.7 * height_px)
    crown_w = 0
    for y in range(crown_top, crown_bot):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0:
            crown_w = max(crown_w, row.max() - row.min())
    crown_m = crown_w * meters_per_px

    trunk_top = int(y_max - 0.2 * height_px)
    trunk_w = []
    for y in range(trunk_top, y_max):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0:
            width = row.max() - row.min()
            if width > 10:
                trunk_w.append(width)

    trunk_m = (np.mean(trunk_w) * meters_per_px) if trunk_w else None

    height_m = round(height_m, 2)
    crown_m = round(crown_m, 2)
    trunk_m = round(trunk_m, 2) if trunk_m else None

    return height_m, crown_m, trunk_m


def load_classifier(model_path: str, class_names_ru):
    if not os.path.exists(model_path):
        print(f"[!] Классификатор не найден: {model_path}")
        return None, None
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names_ru))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, tfm


def classify_tree(model, tfm, img_bgr, bbox):
    if model is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    crop = Image.fromarray(crop)
    tens = tfm(crop).unsqueeze(0)
    with torch.no_grad():
        logits = model(tens)
        cls_id = int(torch.argmax(logits, dim=1).item())
    return CLASS_NAMES_RU[cls_id]


print("Загрузка моделей...")
yolo_tree = YOLO(TREE_MODEL_PATH)
yolo_stick = YOLO(STICK_MODEL_PATH)
clf, clf_tfm = load_classifier(CLASSIFIER_PATH, CLASS_NAMES_RU)
print("Модели загружены.")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Не удалось прочитать изображение"}
            )

        H, W = img_bgr.shape[:2]

        res_tree = yolo_tree(img_bgr)[0]
        if res_tree.masks is None:
            return {
                "species": None,
                "height_m": None,
                "crown_width_m": None,
                "trunk_diameter_m": None,
                "scale_m_per_px": None,
                "message": "Дерево не найдено"
            }

        areas, valid_masks, valid_boxes = [], [], []
        for i, mask_data in enumerate(res_tree.masks.data):
            mask = (mask_data.cpu().numpy() > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = postprocess_mask(mask)
            if mask is not None and mask.max() > 0:
                area = cv2.countNonZero(mask)
                if area < 500:
                    continue
                areas.append(area)
                valid_masks.append(mask)
                valid_boxes.append(res_tree.boxes.xyxy[i].cpu().numpy().astype(int))

        if not valid_masks:
            return {
                "species": None,
                "height_m": None,
                "crown_width_m": None,
                "trunk_diameter_m": None,
                "scale_m_per_px": None,
                "message": "Дерево не найдено (маски слишком маленькие)"
            }

        idx = int(np.argmax(areas))
        mask = valid_masks[idx]
        xyxy = valid_boxes[idx]

        scale = None
        stick_res = yolo_stick(img_bgr, conf=0.3)[0]

        if len(stick_res.boxes) > 0:
            best_box = max(stick_res.boxes,
                           key=lambda b: (b.xyxy[0][3] - b.xyxy[0][1]))
            x1s, y1s, x2s, y2s = best_box.xyxy[0].cpu().numpy().astype(int)
            stick_h = y2s - y1s
            if stick_h > 20:
                scale_tmp = REAL_STICK_LENGTH_M / stick_h
                if 0.001 < scale_tmp < 0.05:
                    scale = scale_tmp

        if mask is not None and mask.max() > 0:
            meters_per_px = scale if scale else 1.0
            h_m, cw_m, dbh_m = measure_tree(mask, meters_per_px)
        else:
            h_m = cw_m = dbh_m = None

        species = classify_tree(clf, clf_tfm, img_bgr, xyxy) or None

        return {
            "species": species,
            "height_m": h_m if scale else None,
            "crown_width_m": cw_m if scale else None,
            "trunk_diameter_m": dbh_m if (scale and dbh_m) else None,
            "height_px": h_m if not scale else None,
            "crown_width_px": cw_m if not scale else None,
            "trunk_diameter_px": dbh_m if (not scale and dbh_m) else None,
            "scale_m_per_px": scale,
            "message": "ok" if species else "вид не определён"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Внутренняя ошибка: {str(e)}"}
        )

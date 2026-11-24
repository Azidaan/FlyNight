from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


folder = Path(__file__).resolve().parent
result = json.loads((folder / "earlier_result.json").read_bytes())
pixels = folder / "earlier_pixels.npz"
assert hashlib.sha256(pixels.read_bytes()).hexdigest() == result["pixels_sha256"]
frame = folder / "frames/00001.jpg"
image_bytes = frame.read_bytes()
inputs = json.loads((folder / "earlier_inputs.json").read_bytes())["sha256"]
assert hashlib.sha256(image_bytes).hexdigest() == inputs[
    f"data/bdd100k/images/track/val/{result['video']}/{result['video']}-0000172.jpg"]
rgb = np.asarray(Image.open(frame).convert("RGB"))
with np.load(pixels) as archive:
    panels = [archive["control"][:, -1], archive["earlier"][:, -1],
              archive["current"][result["correct_candidates"]]]

colors = [(35, 205, 243), (244, 137, 61), (234, 219, 91)]
labels = [("Память с кадра 171", "Машина Б сохранила номер, машина А ждёт"),
          ("Память с кадра 170", "Обе машины сохранили свои номера"),
          ("Текущие маски SAM 2", "Получены из двух рамок YOLO на кадре 172")]
font_folder = Path("C:/Windows/Fonts")
title = ImageFont.truetype(str(font_folder / "arialbd.ttf"), 22)
caption = ImageFont.truetype(str(font_folder / "arial.ttf"), 17)
overall = ImageFont.truetype(str(font_folder / "arialbd.ttf"), 27)

panel_width = 428
crop = (78, 341, 364, 493)
panel_height = round(panel_width * (crop[3] - crop[1]) / (crop[2] - crop[0]))
gap = 12
board = Image.new("RGB", (3 * panel_width + 4 * gap, panel_height + 155), "#172129")
draw = ImageDraw.Draw(board)
draw.text((gap, 8), "Кадр 172. Сравнение двух способов начать сопровождение", fill="#f3f6f8", font=overall)
for index, ((left, right), (name, note)) in enumerate(zip(panels, labels)):
    ownership = left.astype(np.uint8) + 2 * right.astype(np.uint8)
    canvas = rgb.copy()
    for label, color in enumerate(colors, 1):
        region = ownership == label
        canvas[region] = np.rint(0.55 * canvas[region] + 0.45 * np.asarray(color)).astype(np.uint8)
        outline, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, outline, -1, color, 1, lineType=cv2.LINE_AA)
    zoom = Image.fromarray(canvas).crop(crop).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    x = gap + index * (panel_width + gap)
    draw.text((x, 51), name, font=title, fill="#f1f5f7")
    draw.text((x, 80), note, font=caption, fill="#d0d8dd")
    board.paste(zoom, (x, 108))
for x, name, color in zip((gap, 160, 310), ("Машина А", "Машина Б", "Пересечение масок"), colors):
    y = panel_height + 123
    draw.rounded_rectangle((x, y, x+17, y+17), 3, fill=color)
    draw.text((x+25, y-1), name, font=caption, fill="#f1f5f7")
board.save(folder / "compare_172.jpg", quality=95, subsampling=0)

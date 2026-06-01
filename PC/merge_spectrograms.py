"""
把 dataset/ 下各手势的幅度包络图和微多普勒谱图分别拼成大图。
用法: python merge_spectrograms.py                  # 全部手势, 两种图各一张
      python merge_spectrograms.py Push             # 只拼一个手势
      python merge_spectrograms.py Push Pull        # 拼多个手势
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = r"C:\Users\86132\Documents\手势毕设"
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
OUT_DIR     = os.path.join(PROJECT_DIR, "merged")

COLS    = 5
THUMB_W = 560
THUMB_H = 140

IMG_TYPES = {
    "幅度包络图":   "幅度包络图",
    "微多普勒谱图": "微多普勒谱图",
}

GESTURES = ["Push", "Pull", "Sweep", "Slide", "Fist_bump", "Grab", "Tap"]


def merge_gesture(gesture):
    """为指定手势拼接两种图"""
    for type_key, type_dir in IMG_TYPES.items():
        img_dir = os.path.join(DATASET_DIR, gesture, type_dir)
        if not os.path.isdir(img_dir):
            print(f"  [{gesture}] 无 {type_key} 目录，跳过")
            continue

        pngs = sorted([
            f for f in os.listdir(img_dir) if f.endswith(".png")
        ])
        if not pngs:
            print(f"  [{gesture}] {type_key} 无图片，跳过")
            continue

        png_paths = [os.path.join(img_dir, p) for p in pngs]
        labels    = [os.path.splitext(p)[0] for p in pngs]
        images    = [Image.open(pp).resize((THUMB_W, THUMB_H)) for pp in png_paths]

        n = len(images)
        cols = min(COLS, n)
        rows = (n + cols - 1) // cols

        label_h = 24
        tile_w, tile_h = THUMB_W, THUMB_H + label_h

        canvas = Image.new("RGB", (tile_w * cols, tile_h * rows), color=(30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("simhei.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        for i, (im, label) in enumerate(zip(images, labels)):
            r, c = i // cols, i % cols
            x, y = c * tile_w, r * tile_h
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x + (tile_w - tw) // 2, y + 4), label, fill=(220, 220, 220), font=font)
            canvas.paste(im, (x, y + label_h))

        os.makedirs(OUT_DIR, exist_ok=True)
        out_file = os.path.join(OUT_DIR, f"merged_{type_key}_{gesture}.png")
        canvas.save(out_file, quality=90)
        w, h = canvas.size
        print(f"  已保存: {os.path.basename(out_file)}  ({rows}行×{cols}列, {n}张, {os.path.getsize(out_file)/1024:.0f}KB)")


def main():
    if len(sys.argv) > 1:
        gestures = [g for g in sys.argv[1:] if g in GESTURES]
        if not gestures:
            print(f"未知手势，可选: {GESTURES}")
            return
    else:
        gestures = GESTURES

    print(f"拼接 {len(gestures)} 个手势的对比图\n")
    for g in gestures:
        print(f"[{g}]")
        merge_gesture(g)

    print(f"\n输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()

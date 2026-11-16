"""生成「现场标定测量示意图」：在真实监控画面上标注 5 个控制点与测量方式。

控制点像素坐标是在 data/outputs/calib_frames/f002.png（recorded 08:01:12）上
逐点用亮度剖面核对过的，改动机位后必须重新标注。

用法：
    python scripts/make_calibration_figure.py \
        --frame data/outputs/calib_frames/f002.png \
        --output docs/calibration_guide.png
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

# 5 个控制点（像素坐标）
POINTS = {
    "P1": (496, 676),    # 减速带最左端（黄色端头尖端）
    "P2": (1712, 845),   # 减速带最右端（灰黑段与黄色铺装交界）
    "P3": (435, 646),    # 路缘石下沿接缝 A
    "P4": (626, 568),    # 路缘石下沿接缝 B
    "P5": (764, 511),    # 路缘石下沿接缝 C
}

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# 每个点标签单独给偏移，避免互相压盖
LABEL_OFFSET = {
    "P1": (24, 8),
    "P2": (-96, -52),
    "P3": (-92, 16),
    "P4": (22, -56),
    "P5": (22, -52),
}


def marker(draw, xy, label, font):
    x, y = xy
    r = 17
    draw.ellipse([x - r, y - r, x + r, y + r], outline=(200, 30, 30, 255), width=4)
    draw.line([x - r - 12, y, x + r + 12, y], fill=(200, 30, 30, 255), width=3)
    draw.line([x, y - r - 12, x, y + r + 12], fill=(200, 30, 30, 255), width=3)
    dx, dy = LABEL_OFFSET.get(label, (22, -50))
    draw.text((x + dx, y + dy), label, fill=(200, 30, 30, 255), font=font,
              stroke_width=3, stroke_fill=(255, 255, 255, 235))


def main():
    parser = argparse.ArgumentParser(description="Draw the on-site calibration measurement guide")
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    img = Image.open(args.frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    f_label = load_font(40)
    f_body = load_font(27)
    f_title = load_font(34)

    # 减速带轴线（X 轴方向）
    draw.line([POINTS["P1"], POINTS["P2"]], fill=(255, 110, 0, 210), width=4)
    # 路缘石轴线（Y 轴方向）
    draw.line([POINTS["P3"], POINTS["P5"]], fill=(0, 190, 110, 210), width=4)

    for name, xy in POINTS.items():
        marker(draw, xy, name, f_label)

    # 图例
    box = [36, 30, 1330, 372]
    draw.rectangle(box, fill=(255, 255, 255, 232), outline=(110, 110, 110, 255), width=2)
    draw.text((56, 44), "行政车道进 · 现场标定测量示意（5 个控制点）", fill=(15, 15, 15), font=f_title)

    lines = [
        ("P1", "减速带最左端的黄色端头尖端"),
        ("P2", "减速带最右端（灰黑防滑段与黄色铺装交界处）"),
        ("P3/P4/P5", "白色路缘石下沿（靠路面那条棱）上的三条接缝"),
        ("", ""),
        ("①", "W  = 沿减速带量 P1 → P2 的长度（整条减速带）"),
        ("②", "g  = 从 P1 向路缘石作垂线，量到路缘石的距离，垂足记作 P0"),
        ("③④⑤", "d3 / d4 / d5 = 从 P0 沿路缘石向远处，量到 P3 / P4 / P5 的距离"),
    ]
    y = 96
    for tag, text in lines:
        if not tag:
            y += 14
            continue
        draw.text((56, y), tag, fill=(200, 30, 30), font=f_body)
        draw.text((176, y), text, fill=(25, 25, 25), font=f_body)
        y += 37

    draw.text((56, 330), "橙色线 = 减速带轴线（X）　　绿线 = 路缘石轴线（Y）", fill=(0, 110, 170), font=f_body)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    img.save(args.output)
    print("saved", args.output, img.size)


if __name__ == "__main__":
    main()

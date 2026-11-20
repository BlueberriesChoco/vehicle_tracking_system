#!/usr/bin/env python
"""生成 A4 可打印的现场标定记录单（图 + 填写表合成一页）。

用法：python scripts/make_calibration_fieldsheet.py
输出：docs/calibration_fieldsheet.png（A4 200dpi，直接打印）
"""

import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE = os.path.join(ROOT, "docs", "calibration_guide.png")
OUTPUT = os.path.join(ROOT, "docs", "calibration_fieldsheet.png")

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
DPI = 200
A4 = (int(8.27 * DPI), int(11.69 * DPI))  # 1654 x 2339

MARGIN = 90
INK = (30, 30, 30)
GREY = (110, 110, 110)
RULE = (180, 180, 180)

ROWS = [
    ("W", "沿减速带，P1 → P2（整条减速带长度）"),
    ("g", "从 P1 向路缘石作垂线，量到路缘石（垂足记 P0）"),
    ("d3", "从 P0 沿路缘石向远处，量到 P3"),
    ("d4", "从 P0 沿路缘石向远处，量到 P4"),
    ("d5", "从 P0 沿路缘石向远处，量到 P5"),
    ("check_width", "（自检）P4 处横跨车道再量一次路面宽度"),
    ("check_tape", "（自检）P0 → P5 复核一次，应与 d5 一致"),
]

META = ["测量日期：", "测量人：", "天气：", "摄像头是否移动过：", "备注："]


def font(size, bold=False):
    # msyh.ttc 的第 0 个字面是常规体，第 1 个是粗体
    index = 1 if bold else 0
    try:
        return ImageFont.truetype(FONT_PATH, size, index=index)
    except OSError:
        return ImageFont.truetype(FONT_PATH, size)


def draw_text(draw, xy, text, fnt, fill=INK):
    draw.text(xy, text, font=fnt, fill=fill)


def main():
    canvas = Image.new("RGB", A4, "white")
    draw = ImageDraw.Draw(canvas)

    y = MARGIN
    draw_text(draw, (MARGIN, y), "行政车道进 · 现场标定记录表", font(46, bold=True))
    y += 62
    draw_text(draw, (MARGIN, y), "只填「实测值」列，单位米、保留两位小数（例：8.40）", font(26), GREY)
    y += 52

    # 图
    figure = Image.open(FIGURE).convert("RGB")
    width = A4[0] - MARGIN * 2
    height = int(figure.height * width / figure.width)
    figure = figure.resize((width, height), Image.LANCZOS)
    canvas.paste(figure, (MARGIN, y))
    y += height + 40

    # 表头
    col_item = MARGIN
    col_desc = MARGIN + 260
    col_value = MARGIN + 1130
    header_font = font(27, bold=True)
    value_font = font(27)
    desc_font = font(24)

    draw_text(draw, (col_item, y), "项", header_font)
    draw_text(draw, (col_desc, y), "怎么量", header_font)
    draw_text(draw, (col_value, y), "实测值（米）", header_font)
    y += 44
    draw.line([(MARGIN, y), (A4[0] - MARGIN, y)], fill=RULE, width=2)
    y += 14

    for key, desc in ROWS:
        row_height = 78 if len(desc) < 26 else 104
        draw_text(draw, (col_item, y), key, value_font)
        # 描述过长时折行
        if len(desc) <= 26:
            draw_text(draw, (col_desc, y), desc, desc_font)
        else:
            draw_text(draw, (col_desc, y), desc[:26], desc_font)
            draw_text(draw, (col_desc, y + 38), desc[26:], desc_font)
        draw_text(draw, (col_value, y), "________", value_font, GREY)
        y += row_height
        draw.line([(MARGIN, y - 12), (A4[0] - MARGIN, y - 12)], fill=(230, 230, 230), width=1)

    y += 18
    draw_text(draw, (MARGIN, y), "控制点像素坐标（已核对，不用改）", font(28, bold=True))
    y += 46
    pixels = [
        ("P1", 496, 676, "减速带最左端黄色端头尖端"),
        ("P2", 1712, 845, "减速带最右端（灰黑与黄色铺装交界）"),
        ("P3", 435, 646, "路缘石下沿接缝 A"),
        ("P4", 626, 568, "路缘石下沿接缝 B"),
        ("P5", 764, 511, "路缘石下沿接缝 C"),
    ]
    for name, px, py, desc in pixels:
        draw_text(draw, (MARGIN, y), f"{name}  ({px}, {py})", font(25))
        draw_text(draw, (MARGIN + 300, y), desc, font(24), GREY)
        y += 38

    y += 26
    for label in META:
        draw_text(draw, (MARGIN, y), label, font(26))
        draw.line([(MARGIN + 260, y + 34), (A4[0] - MARGIN, y + 34)], fill=RULE, width=1)
        y += 62

    canvas.save(OUTPUT, dpi=(DPI, DPI))
    print("saved", OUTPUT, canvas.size)


if __name__ == "__main__":
    main()

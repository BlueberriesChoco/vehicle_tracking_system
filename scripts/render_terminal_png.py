#!/usr/bin/env python
"""把终端输出渲染成 PNG（深色主题），便于贴进文档。

不是真机截图：内容来自实际运行输出，只是换了呈现方式。
ASCII 用 Consolas、中文用黑体，按终端列宽（中文占 2 列）排版，因此原有的
空格对齐不会跑位。

用法:
    python scripts/render_terminal_png.py --input out.txt --output shot.png
"""

import argparse
import os
import unicodedata

from PIL import Image, ImageDraw, ImageFont

BG = (30, 30, 30)
FG = (212, 212, 212)
RED = (241, 105, 105)
ORANGE = (229, 181, 103)

ASCII_FONT = "C:/Windows/Fonts/consola.ttf"
CJK_FONT = "C:/Windows/Fonts/simhei.ttf"

# (关键词, 颜色)：整行高亮，用于标出"错误"所在
RULES = [
    ("凭空造出来", RED),
    ("完全不匹配", RED),
    ("结论", ORANGE),
    ("旧写法多报", ORANGE),
    ("[整列为空", ORANGE),
    ("[整列为零·机位未标定", ORANGE),
]


def is_wide(ch):
    return unicodedata.east_asian_width(ch) in "WF"


def color_for(line):
    for key, color in RULES:
        if key in line:
            return color
    return FG


def render(text, out, size=19):
    ascii_font = ImageFont.truetype(ASCII_FONT, size)
    cjk_font = ImageFont.truetype(CJK_FONT, int(size * 1.02))
    cell = ascii_font.getlength("M")
    ascent = ascii_font.getmetrics()[0]
    line_h = int(size * 1.6)
    pad = 24

    lines = text.replace("\r\n", "\n").rstrip("\n").split("\n")
    cols = max(sum(2 if is_wide(ch) else 1 for ch in line) for line in lines)
    width = int(cols * cell) + pad * 2
    height = line_h * len(lines) + pad * 2

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    y = pad
    for line in lines:
        color = color_for(line)
        x = pad
        for ch in line:
            if is_wide(ch):
                glyph_w = cjk_font.getlength(ch)
                draw.text((x + cell - glyph_w / 2, y + ascent), ch, font=cjk_font, fill=color, anchor="ls")
                x += 2 * cell
            else:
                draw.text((x, y + ascent), ch, font=ascii_font, fill=color, anchor="ls")
                x += cell
        y += line_h

    image.save(out)
    return out


def main():
    parser = argparse.ArgumentParser(description="终端输出转 PNG")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=19)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    print(render(text, args.output, args.size))


if __name__ == "__main__":
    main()

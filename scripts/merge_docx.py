#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把续写稿的正文合并进原件：原件已有的段落用原文字节，新增段落用续写稿，其余部件全部保留原件。

为什么不用 Office 库重存：原件里有 customXml/*、docProps/thumbnail.wmf、
word/fonts/font1.odttf（方正仿宋_GB2312 内嵌字体）、word/_rels/fontTable.xml.rels。
任何"打开→保存"都会重写整包，这些部件极易丢失，字体一丢排版就变。

为什么不是整段替换 document.xml：编辑器 SDK 保存时会给原有每个 run 补一个
<w:rPr><w:rFonts w:hint="eastAsia"/></w:rPr>，也会给原来的 heading 样式重新起名
（00001a/00001b/00001d）。前者虽然不改排版，但原件段落就不再是原字节了；
后者必须映射回原件的 1/3/4，否则样式断链。所以这里按"最长文本公共前缀"切一刀：
前缀用原文字节，新增部分用续写稿并做 ID 映射。

图片关系：副本里是 rId7 -> media/image1.png，原件是 rId6。图片落在前缀段里，
用的是原件字节，本身已经是 rId6；全新段落若有引用则一并映射。

用法:
    python scripts/merge_docx.py <原件.docx> <续写稿.docx> <输出.docx>
"""
import re
import sys
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
PARA_RE = re.compile(r"<w:p[ >].*?</w:p>", re.S)
STYLE_MAP = {"00001a": "1", "00001b": "3", "00001d": "4"}


def split_body(doc):
    """切成 (body 开始到第一个 <w:p> 之前的片段, 段落列表, 段落之后的片段)。"""
    matches = list(PARA_RE.finditer(doc))
    if not matches:
        raise SystemExit("没找到段落")
    head = doc[: matches[0].start()]
    paras = [m.group(0) for m in matches]
    tail = doc[matches[-1].end():]
    return head, paras, tail


def para_text(p):
    return "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))


def remap(xml):
    """样式 ID 与图片关系映射到原件可识别的值。"""
    n_style = 0
    for old, new in STYLE_MAP.items():
        xml, n = re.subn(r'(<w:pStyle w:val=")' + old + r'(")', r"\g<1>" + new + r"\g<2>", xml)
        n_style += n
    xml, n_img = re.subn(r'r:embed="rId7"', 'r:embed="rId6"', xml)
    return xml, n_style, n_img


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    base_path, cont_path, out_path = sys.argv[1:4]

    with zipfile.ZipFile(base_path) as zb:
        base_names = {n for n in zb.namelist() if not n.endswith("/")}
        base_doc = zb.read("word/document.xml").decode("utf-8")
    with zipfile.ZipFile(cont_path) as zc:
        cont_doc = zc.read("word/document.xml").decode("utf-8")

    b_head, b_paras, b_tail = split_body(base_doc)
    c_head, c_paras, c_tail = split_body(cont_doc)

    # 最长公共前缀：文本完全相同才算同一段
    k = 0
    while k < min(len(b_paras), len(c_paras)) and para_text(b_paras[k]) == para_text(c_paras[k]):
        k += 1
    print(f"原件段落 {len(b_paras)}，续写稿段落 {len(c_paras)}，公共前缀 {k} 段")
    print(f"  -> 采用原件字节 {k} 段 + 续写稿新增 {len(c_paras) - k} 段 = {len(c_paras)} 段")

    kept, n_style, n_img = remap("".join(c_paras[k:]))
    print(f"  新增段落内改写: 样式 {n_style} 处, 图片关系 {n_img} 处")
    new_doc = c_head + "".join(b_paras[:k]) + kept + c_tail

    # 结构自检：必须能解析，且不能残留未映射的 ID
    ET.fromstring(new_doc.encode("utf-8"))
    left = set(re.findall(r'w:val="(0000[0-9a-f]{2})"', new_doc))
    if left:
        raise SystemExit(f"仍有未映射的样式 ID: {left}")
    left_rid = set(re.findall(r'r:(?:id|embed)="(rId\d+)"', new_doc))
    if left_rid - {"rId6"}:
        raise SystemExit(f"仍有未映射的关系 ID: {left_rid}")
    if len(PARA_RE.findall(new_doc)) != len(c_paras):
        raise SystemExit("段落数对不上")

    data = new_doc.encode("utf-8")
    with zipfile.ZipFile(base_path) as zb, zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zw:
        for info in zb.infolist():
            if info.filename.endswith("/"):
                continue
            payload = data if info.filename == "word/document.xml" else zb.read(info.filename)
            zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            zi.compress_type = info.compress_type
            zi.external_attr = info.external_attr
            zw.writestr(zi, payload)

    with zipfile.ZipFile(out_path) as zo:
        out_names = set(zo.namelist())
    lost = base_names - out_names
    if lost:
        raise SystemExit(f"丢了部件: {lost}")
    print(f"完成: {out_path} | 部件 {len(out_names)} 个 | 正文 {len(data)} 字节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

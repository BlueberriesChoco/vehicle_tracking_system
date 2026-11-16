#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用新的正文段落替换指定区间，格式从模板段落照抄。

用途：某一节的文字要整段重写（比如改口径、改语气），但排版不能动。
做法与 insert_docx_paras.py 一致：pPr 与首个 run 的 rPr 从模板段原样拷贝，
换掉 paraId 与文字；区间之外的段落逐字节保留。

用法:
    python scripts/replace_docx_paras.py <输入.docx> <输出.docx> <起始段> <结束段(不含)> <模板段> <文本文件>
"""
import hashlib
import re
import sys
import zipfile
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

PARA_RE = re.compile(r"<w:p[ >].*?</w:p>", re.S)


def main():
    if len(sys.argv) != 7:
        print(__doc__)
        return 1
    src, dst, start_s, end_s, tpl_s, txt_path = sys.argv[1:7]
    start, end, tpl = int(start_s), int(end_s), int(tpl_s)

    with zipfile.ZipFile(src) as z:
        doc = z.read("word/document.xml").decode("utf-8")
        items = [(i.filename, z.read(i.filename), i) for i in z.infolist() if not i.filename.endswith("/")]

    matches = list(PARA_RE.finditer(doc))
    paras = [m.group(0) for m in matches]
    if not (0 <= start < end <= len(paras)):
        raise SystemExit(f"区间越界：共 {len(paras)} 段，请求 {start}..{end}")

    template = paras[tpl]
    ppr = re.search(r"<w:pPr>.*?</w:pPr>", template, re.S).group(0)
    rpr = re.search(r"<w:r>\s*(<w:rPr>.*?</w:rPr>)", template, re.S).group(1)

    with open(txt_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise SystemExit("文本文件是空的")

    used = set(re.findall(r'w14:paraId="([0-9A-Fa-f]{8})"', doc))
    new_paras = []
    for k, text in enumerate(lines):
        pid = hashlib.md5(f"{start}-{k}-{text}".encode("utf-8")).hexdigest()[:8].upper()
        while pid in used:
            pid = hashlib.md5((pid + "salt").encode()).hexdigest()[:8].upper()
        used.add(pid)
        new_paras.append(f'<w:p w14:paraId="{pid}">{ppr}<w:r>{rpr}<w:t>{escape(text)}</w:t></w:r></w:p>')

    new_doc = (
        doc[: matches[start].start()]
        + "".join(new_paras)
        + doc[matches[end - 1].end():]
    )

    ET.fromstring(new_doc.encode("utf-8"))
    expect = len(paras) - (end - start) + len(lines)
    if len(PARA_RE.findall(new_doc)) != expect:
        raise SystemExit(f"段落数对不上，期望 {expect}")

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zw:
        for name, payload, info in items:
            if name == "word/document.xml":
                payload = new_doc.encode("utf-8")
            zi = zipfile.ZipInfo(name, date_time=info.date_time)
            zi.compress_type = info.compress_type
            zi.external_attr = info.external_attr
            zw.writestr(zi, payload)

    print(f"替换第 {start}..{end - 1} 段（共 {end - start} 段）为 {len(lines)} 段，格式取自第 {tpl} 段")
    print(f"完成: {dst} | 总段数 {expect}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

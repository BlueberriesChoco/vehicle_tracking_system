#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在指定段落之后插入若干正文段落，格式从模板段落照抄。

为什么不用编辑器 SDK：它保存时会给每个原有 run 补 <w:rPr><w:rFonts w:hint="eastAsia"/></w:rPr>
并重写整包，内嵌字体（word/fonts/font1.odttf）会掉，方正仿宋_GB2312 就回落到系统字体。
这里直接拼 XML：pPr 与首个 run 的 rPr 从模板段落原样拷贝，只换 paraId 与文字。

好处是新增段落的行距、缩进、字体、字号、对齐与模板完全同构，不存在"看起来一样但
样式表不同源"的情况。

用法:
    python scripts/insert_docx_paras.py <输入.docx> <输出.docx> <锚点段号> <模板段号> <文本文件>
    文本文件每行一段，空行忽略。
"""
import hashlib
import re
import sys
import zipfile
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

PARA_RE = re.compile(r"<w:p[ >].*?</w:p>", re.S)


def main():
    if len(sys.argv) != 6:
        print(__doc__)
        return 1
    src, dst, anchor_s, tpl_s, txt_path = sys.argv[1:6]
    anchor, tpl = int(anchor_s), int(tpl_s)

    with zipfile.ZipFile(src) as z:
        doc = z.read("word/document.xml").decode("utf-8")
        items = [(i.filename, z.read(i.filename), i) for i in z.infolist() if not i.filename.endswith("/")]

    matches = list(PARA_RE.finditer(doc))
    paras = [m.group(0) for m in matches]
    template = paras[tpl]

    m_ppr = re.search(r"<w:pPr>.*?</w:pPr>", template, re.S)
    if not m_ppr:
        raise SystemExit("模板段落没有 pPr")
    ppr = m_ppr.group(0)
    m_rpr = re.search(r"<w:r>\s*<w:rPr>.*?</w:rPr>", template, re.S)
    if not m_rpr:
        raise SystemExit("模板段落首个 run 没有 rPr")
    rpr = re.search(r"<w:r>\s*(<w:rPr>.*?</w:rPr>)", template, re.S).group(1)

    with open(txt_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise SystemExit("文本文件是空的")

    used = set(re.findall(r'w14:paraId="([0-9A-Fa-f]{8})"', doc))
    new_paras = []
    for k, text in enumerate(lines):
        pid = hashlib.md5(f"{anchor}-{k}-{text}".encode("utf-8")).hexdigest()[:8].upper()
        while pid in used:
            pid = hashlib.md5((pid + "salt").encode()).hexdigest()[:8].upper()
        used.add(pid)
        new_paras.append(
            f'<w:p w14:paraId="{pid}">{ppr}<w:r>{rpr}<w:t>{escape(text)}</w:t></w:r></w:p>'
        )

    at = matches[anchor].end()
    new_doc = doc[:at] + "".join(new_paras) + doc[at:]

    ET.fromstring(new_doc.encode("utf-8"))
    if len(PARA_RE.findall(new_doc)) != len(paras) + len(lines):
        raise SystemExit("段落数对不上")

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zw:
        for name, payload, info in items:
            if name == "word/document.xml":
                payload = new_doc.encode("utf-8")
            zi = zipfile.ZipInfo(name, date_time=info.date_time)
            zi.compress_type = info.compress_type
            zi.external_attr = info.external_attr
            zw.writestr(zi, payload)

    print(f"在第 {anchor} 段后插入 {len(lines)} 段（格式取自第 {tpl} 段）")
    for i, ln in enumerate(lines):
        print(f"  +{i + 1} {ln[:40]}...")
    print(f"完成: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

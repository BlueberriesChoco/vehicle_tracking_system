#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验定稿 docx 的结构与格式是否与原件等价。

检查项:
  1. 原件独有部件（customXml / 缩略图 / 内嵌字体 / fontTable 关系）是否齐全
  2. 图片是否与原件字节一致
  3. 每个段落的"生效格式"（沿 basedOn 继承链解析后的字体/字号/缩进/对齐/行距）
  4. 是否存在空标题、未被使用的残留样式 ID

用法: python scripts/check_docx.py <定稿.docx> <原件.docx>
"""
import hashlib
import re
import sys
import zipfile
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def load_styles(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/styles.xml").decode("utf-8"))
    styles = {}
    for s in root.findall(W + "style"):
        sid = s.get(W + "styleId")
        based = s.find(W + "basedOn")
        name = s.find(W + "name")
        styles[sid] = {
            "name": name.get(W + "val") if name is not None else "",
            "basedOn": based.get(W + "val") if based is not None else None,
            "pPr": s.find(W + "pPr"),
            "rPr": s.find(W + "rPr"),
            "type": s.get(W + "type"),
            "default": s.get(W + "default"),
        }
    return styles


def resolve(styles, sid):
    """沿 basedOn 链由近到远收集 pPr/rPr, 近的优先。"""
    chain, cur, seen = [], sid, set()
    while cur and cur in styles and cur not in seen:
        seen.add(cur)
        chain.append(styles[cur])
        cur = styles[cur]["basedOn"]
    return chain


def eff_prop(chain, part, tagname):
    for st in chain:
        node = st[part]
        if node is None:
            continue
        el = node.find(W + tagname)
        if el is not None:
            return el
    return None


def para_fmt(p, styles, default_sid):
    ppr = p.find(W + "pPr")
    sid = None
    if ppr is not None:
        s = ppr.find(W + "pStyle")
        if s is not None:
            sid = s.get(W + "val")
    sid = sid or default_sid
    chain = resolve(styles, sid)

    # 直接格式优先于样式
    def pick(part, tag):
        if ppr is not None:
            d = ppr.find(W + tag)
            if d is not None:
                return d
        return eff_prop(chain, part, tag)

    # rFonts / sz 必须按属性逐个继承：只写了 w:hint 的 rFonts 不覆盖任何字体，
    # 之前把它当成"字体已指定"直接取用，会把标题判成"无中文字体"——检查器自己的错。
    def attr_of(tag, attr):
        holders = []
        if ppr is not None:
            mk = ppr.find(W + "rPr")
            if mk is not None:
                holders.append(mk)
        for r in p.findall(W + "r"):
            rpr = r.find(W + "rPr")
            if rpr is not None:
                holders.append(rpr)
        for h in holders:
            el = h.find(W + tag)
            if el is not None and el.get(W + attr):
                return el.get(W + attr)
        for st in chain:
            rpr = st["rPr"]
            if rpr is None:
                continue
            el = rpr.find(W + tag)
            if el is not None and el.get(W + attr):
                return el.get(W + attr)
        return None

    ind = pick("pPr", "ind")
    jc = pick("pPr", "jc")
    spacing = pick("pPr", "spacing")
    olvl = pick("pPr", "outlineLvl")
    return {
        "style_id": sid,
        "style_name": styles.get(sid, {}).get("name", "?"),
        "eastAsia": attr_of("rFonts", "eastAsia"),
        "ascii": attr_of("rFonts", "ascii"),
        "sz": attr_of("sz", "val"),
        "firstLineChars": (ind.get(W + "firstLineChars") if ind is not None else None),
        "firstLine": (ind.get(W + "firstLine") if ind is not None else None),
        "jc": (jc.get(W + "val") if jc is not None else None),
        "line": (spacing.get(W + "line") if spacing is not None else None),
        "lineRule": (spacing.get(W + "lineRule") if spacing is not None else None),
        "outlineLvl": (olvl.get(W + "val") if olvl is not None else None),
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    out_path, base_path = sys.argv[1], sys.argv[2]

    with zipfile.ZipFile(base_path) as zb:
        base_names = {n for n in zb.namelist() if not n.endswith("/")}
        base_img = zb.read("word/media/image1.png")
        base_font = zb.read("word/fonts/font1.odttf") if "word/fonts/font1.odttf" in zb.namelist() else None
        base_doc = ET.fromstring(zb.read("word/document.xml").decode("utf-8"))
        base_paras = base_doc.find(W + "body").findall(W + "p")

    with zipfile.ZipFile(out_path) as zo:
        out_names = {n for n in zo.namelist() if not n.endswith("/")}
        out_img = zo.read("word/media/image1.png")
        out_font = zo.read("word/fonts/font1.odttf") if "word/fonts/font1.odttf" in out_names else None
        out_doc = ET.fromstring(zo.read("word/document.xml").decode("utf-8"))
        out_paras = out_doc.find(W + "body").findall(W + "p")
        rels = zo.read("word/_rels/document.xml.rels").decode("utf-8")
        docxml = zo.read("word/document.xml").decode("utf-8")
        styles_xml = zo.read("word/styles.xml").decode("utf-8")

    print("=" * 64)
    print("1. 部件完整性")
    missing = base_names - out_names
    print("   原件独有部件丢失:", missing or "无")
    print("   customXml:", any(n.startswith("customXml/") for n in out_names))
    print("   缩略图:", "docProps/thumbnail.wmf" in out_names)
    print("   fontTable 关系:", "word/_rels/fontTable.xml.rels" in out_names)
    print("   内嵌字体 font1.odttf:", out_font is not None,
          "字节一致" if out_font == base_font else "不一致!")

    print("2. 图片")
    print("   image1.png 字节一致:", hashlib.md5(out_img).hexdigest() == hashlib.md5(base_img).hexdigest(),
          f"({len(out_img)} 字节)")

    # 图片关系
    rid = re.search(r'<Relationship Id="(rId\d+)"[^>]*media/image1\.png', rels)
    embed = re.search(r'r:embed="(rId\d+)"', docxml)
    print("   关系表图片 ID:", rid.group(1) if rid else None,
          "| 正文引用:", embed.group(1) if embed else None,
          "->", "匹配" if rid and embed and rid.group(1) == embed.group(1) else "断链!")

    print("3. 残留样式 ID")
    print("   正文里的 0000xx 样式:", set(re.findall(r'w:val="(0000[0-9a-f]{2})"', docxml)) or "无")

    styles = load_styles(out_path)
    default_sid = next((k for k, v in styles.items() if v["default"] == "1" and v["type"] == "paragraph"), None)
    print("   默认段落样式:", default_sid, styles.get(default_sid, {}).get("name"))

    print("4. 段落格式（生效值）")
    print(f"   总段数 {len(out_paras)}，原件前 {len(base_paras)} 段应逐段等价")
    hdr = f"{'#':>3} {'样式':<10}{'中文字体':<18}{'字号':<5}{'首行':<8}{'对齐':<6}{'行距':<7}{'大纲':<4} 文本"
    print("   " + hdr)
    problems = []
    for i, p in enumerate(out_paras):
        fm = para_fmt(p, styles, default_sid)
        txt = "".join(t.text or "" for t in p.iter(W + "t"))
        sz = fm["sz"]
        szpt = f"{int(sz)/2:g}" if sz else "-"
        ind = fm["firstLineChars"] or fm["firstLine"] or "-"
        print(f"   {i:>3} {fm['style_name']:<10}{str(fm['eastAsia']):<18}{szpt:<5}{str(ind):<8}"
              f"{str(fm['jc']):<6}{str(fm['line']):<7}{str(fm['outlineLvl']):<4} {txt[:26]}")
        if fm["style_name"].startswith("heading") and not txt.strip():
            problems.append(f"第{i}段是空标题")
        # 只校验新增段落：原件段落沿用原有排版（含图注 12pt），不套用新规则
        if i >= len(base_paras) and txt.strip() and fm["sz"] != "32":
            problems.append(f"新增第{i}段字号 {fm['sz']} 不是 16pt")
        if i >= len(base_paras) and fm["style_name"].startswith("heading") and fm["eastAsia"] != "方正黑体_GBK":
            problems.append(f"新增第{i}段标题字体 {fm['eastAsia']}")
        if (
            i >= len(base_paras)
            and not fm["style_name"].startswith("heading")
            and txt.strip()
            and fm["eastAsia"] != "方正仿宋_GB2312"
        ):
            problems.append(f"新增第{i}段正文字体 {fm['eastAsia']}")
        if (
            i >= len(base_paras)
            and not fm["style_name"].startswith("heading")
            and txt.strip()
            and fm["firstLineChars"] != "200"
        ):
            problems.append(f"新增第{i}段首行缩进 {fm['firstLineChars']}")

    print("5. 前 N 段与原件的一致性")
    diff = 0
    for i, (a, b) in enumerate(zip(base_paras, out_paras)):
        fa, fb = para_fmt(a, styles, default_sid), para_fmt(b, styles, default_sid)
        ta = "".join(t.text or "" for t in a.iter(W + "t"))
        tb = "".join(t.text or "" for t in b.iter(W + "t"))
        keys = ("style_name", "eastAsia", "sz", "firstLineChars", "firstLine", "jc", "line")
        bad = [k for k in keys if fa[k] != fb[k]]
        if bad or ta != tb:
            diff += 1
            print(f"   第{i}段差异: {bad} | 文本同: {ta == tb}")
            for k in bad:
                print(f"      {k}: 原件={fa[k]!r} 定稿={fb[k]!r}")
    print("   差异段数:", diff)

    print("6. 结论")
    if problems:
        print("   问题:", problems)
    else:
        print("   未发现空标题/字号异常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

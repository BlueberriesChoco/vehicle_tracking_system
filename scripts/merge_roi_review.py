#!/usr/bin/env python
"""合并两轮 ROI 复核结果，重建最终 Word 复核单。

场景：第一轮全量推导后有 98 台退回全画幅；第二轮只对这 98 台换时段重试。
本脚本把第二轮里成功的记录替换进第一轮，再按原顺序重建复核单。

用法：
    python scripts/merge_roi_review.py \
        --base data/outputs/roi_review/summary.json \
        --retry data/outputs/roi_review_retry/summary.json \
        --docx docs/roi_review.docx
"""

from __future__ import annotations

import argparse
import json
import os


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--retry", required=True)
    parser.add_argument("--docx", default="docs/roi_review.docx")
    args = parser.parse_args()

    from docx import Document
    from docx.shared import Cm

    base = load(args.base)
    retry = load(args.retry)
    base_records = {record["camera"]: record for record in base["records"]}
    order = [record["camera"] for record in base["records"]]

    improved = 0
    for record in retry["records"]:
        camera = record.get("camera")
        if camera not in base_records:
            continue
        old = base_records[camera]
        # 只有重试拿到运动 ROI 才替换；仍然是全画幅就保留第一轮的信息
        if record.get("mode") == "motion" and old.get("mode") != "motion":
            base_records[camera] = record
            improved += 1

    document = Document()
    document.add_heading("机位 ROI 复核单", level=0)
    document.add_paragraph(
        "每页一台机位。黄线为推导出的 ROI，绿点为参与推导的运动目标，紫点为被判为静止、"
        "已剔除的停放车辆。只替换了 ROI，已有的人工进出线与现场标定均保留。"
        "标注「需要人工圈定」的机位在采样时段内没有检测到运动车辆。"
    )

    motion = 0
    for index, camera in enumerate(order, start=1):
        record = base_records[camera]
        document.add_heading(f"{index}. {camera}", level=2)
        if not record.get("ok"):
            document.add_paragraph(f"⚠ 推导失败：{record.get('error')}")
            continue

        detail = (f"来源：{record.get('source', '')}　分辨率：{record['width']}x{record['height']}　"
                  f"采样 {record.get('frames')} 帧，检出点 {record.get('points')}，"
                  f"保留运动点 {record.get('kept')}，剔除静止网格 {record.get('static_cells')}")
        if record.get("attempts"):
            detail += f"　尝试时段 {record['attempts']} 个"
        document.add_paragraph(detail)

        if record.get("mode") == "motion":
            motion += 1
        elif record.get("kept_original_roi"):
            document.add_paragraph("⚠ 采样时段内没有运动车辆：未覆盖原 ROI，请人工确认原 ROI 是否仍正确。")
        else:
            document.add_paragraph("⚠ 采样时段内没有运动车辆，已退回全画幅，需要人工圈定 ROI。")

        overlay = record.get("overlay")
        if overlay and os.path.isfile(overlay):
            document.add_picture(overlay, width=Cm(16))

    os.makedirs(os.path.dirname(os.path.abspath(args.docx)) or ".", exist_ok=True)
    document.save(args.docx)

    summary = {
        "cameras": len(order),
        "motion_roi": motion,
        "fallback": len(order) - motion,
        "improved_by_retry": improved,
        "docx": args.docx,
    }
    print(json.dumps(summary, ensure_ascii=False))
    with open(os.path.join(os.path.dirname(args.base), "summary_merged.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": [base_records[c] for c in order]},
                  handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

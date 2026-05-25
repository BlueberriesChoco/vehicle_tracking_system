#!/usr/bin/env python
"""摄像头场景标定辅助工具。

通过鼠标交互在视频首帧上标注：
  1. ROI 多边形顶点
  2. 通道入口线
  3. 通道出口线
  4. 参照物标定线段（用于像素→米换算）

操作:
  - 左键点击：添加顶点
  - 右键点击：完成当前对象，切换到下一个
  - 按 'u'：撤销上一个顶点
  - 按 's'：保存配置到 YAML 文件
  - 按 'q'：退出

用法:
    python scripts/calibrate_camera.py \\
        --image data/test_videos/frame_0001.jpg \\
        --camera cam01
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import yaml


class CalibrationTool:
    """交互式场景标定工具。"""

    COLORS = {
        "roi": (0, 255, 0),
        "entry": (0, 255, 255),
        "exit": (0, 0, 255),
        "reference": (255, 255, 0),
    }

    def __init__(self, image_path: str, camera_id: str = "cam01"):
        self.image_path = image_path
        self.camera_id = camera_id
        self.frame = cv2.imread(image_path)
        if self.frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        self.window_name = f"Camera Calibration - {camera_id}"
        self.stages = ["roi", "entry", "exit", "reference"]
        self.current_stage_idx = 0
        self.points: dict = {s: [] for s in self.stages}
        self.display_frame = self.frame.copy()

    def run(self):
        """启动交互式标定。"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("=" * 60)
        print(f"Camera Calibration Tool - {self.camera_id}")
        print("=" * 60)
        print("Controls:")
        print("  Left click  : Add point")
        print("  Right click : Next stage")
        print("  'u'         : Undo last point")
        print("  's'         : Save & exit")
        print("  'q'         : Quit without saving")
        print("-" * 60)

        self._update_display()

        while True:
            cv2.imshow(self.window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nQuit without saving.")
                break
            elif key == ord("s"):
                self._save_config()
                break
            elif key == ord("u"):
                self._undo_point()

        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._next_stage()

    def _add_point(self, x: int, y: int):
        stage = self.stages[self.current_stage_idx]
        self.points[stage].append([x, y])
        self._update_display()
        print(f"[{stage}] Added point: ({x}, {y})")

    def _next_stage(self):
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            stage = self.stages[self.current_stage_idx]
            print(f"\n>>> Now annotating: {stage}")
            self._update_display()
        else:
            print("\nAll stages completed. Press 's' to save.")

    def _undo_point(self):
        stage = self.stages[self.current_stage_idx]
        if self.points[stage]:
            removed = self.points[stage].pop()
            print(f"[{stage}] Removed point: ({removed[0]}, {removed[1]})")
            self._update_display()

    def _update_display(self):
        self.display_frame = self.frame.copy()

        # 绘制已标注的点
        for stage in self.stages:
            color = self.COLORS[stage]
            pts = self.points[stage]
            for i, pt in enumerate(pts):
                cv2.circle(self.display_frame, tuple(pt), 5, color, -1)
                cv2.putText(
                    self.display_frame, str(i),
                    (pt[0] + 8, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )

            # 绘制连线
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    cv2.line(self.display_frame, tuple(pts[i]), tuple(pts[i + 1]), color, 1)
                # ROI 闭合
                if stage == "roi" and len(pts) >= 3:
                    cv2.line(self.display_frame, tuple(pts[-1]), tuple(pts[0]), color, 1)

        # 当前阶段提示
        current_stage = self.stages[self.current_stage_idx]
        info_text = f"Stage: {current_stage} | Points: {len(self.points[current_stage])} | Right-click: next | 's': save | 'q': quit"
        cv2.putText(
            self.display_frame, info_text,
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )

    def _save_config(self):
        """将标注结果保存为摄像头配置文件。"""
        config = {
            "camera_id": self.camera_id,
            "camera_name": f"Camera {self.camera_id}",
            "roi": {
                "polygon": self.points["roi"],
            },
            "entry_line": {
                "p1": self.points["entry"][0] if len(self.points["entry"]) >= 1 else [0, 0],
                "p2": self.points["entry"][1] if len(self.points["entry"]) >= 2 else [0, 0],
            },
            "exit_line": {
                "p1": self.points["exit"][0] if len(self.points["exit"]) >= 1 else [0, 0],
                "p2": self.points["exit"][1] if len(self.points["exit"]) >= 2 else [0, 0],
            },
            "calibration": {
                "ref_point_1": self.points["reference"][0] if len(self.points["reference"]) >= 1 else [0, 0],
                "ref_point_2": self.points["reference"][1] if len(self.points["reference"]) >= 2 else [0, 0],
                "ref_length_m": 6.0,
                "px_per_meter": self._compute_px_per_meter(),
            },
        }

        output_path = f"config/camera/{self.camera_id}.yaml"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"\nConfig saved to: {output_path}")
        print(json.dumps(config, indent=2, ensure_ascii=False))

    def _compute_px_per_meter(self) -> float:
        """根据参照物线段计算像素/米比例。"""
        pts = self.points["reference"]
        if len(pts) < 2:
            return 0.0
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        px_length = (dx * dx + dy * dy) ** 0.5
        ref_m = 6.0  # 默认6米（可手动修改）
        return round(px_length / ref_m, 2)


def main():
    parser = argparse.ArgumentParser(description="Camera scene calibration tool")
    parser.add_argument("--image", required=True, help="Path to a sample frame image")
    parser.add_argument("--camera", default="cam01", help="Camera ID")
    args = parser.parse_args()

    tool = CalibrationTool(args.image, args.camera)
    tool.run()


if __name__ == "__main__":
    main()

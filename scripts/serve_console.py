#!/usr/bin/env python
"""启动只读排查控制台（标准库 HTTP 服务，无第三方依赖）。

    python scripts/serve_console.py                 # 只读，浏览已有 run
    python scripts/serve_console.py --allow-runs    # 额外允许 POST /api/runs 触发新任务

浏览器打开输出的地址即可。默认只监听 127.0.0.1，不对外暴露。
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.web.server import ROOT, serve


def main():
    parser = argparse.ArgumentParser(description="Serve the read-only inspection console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", default=str(ROOT / "data" / "outputs"), help="Directory containing run folders")
    parser.add_argument("--allow-runs", action="store_true", help="Allow POST /api/runs to launch pipelines")
    args = parser.parse_args()

    server = serve(args.host, args.port, Path(args.root), args.allow_runs)
    host, port = server.server_address[:2]
    print(f"Console: http://{host}:{port}")
    print(f"Runs root: {args.root}")
    print(f"Allow new runs: {args.allow_runs}")
    print("Ctrl+C 退出")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()

"""生成《Agent 在科研与算法工作中的使用心得》Word 文档。"""

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm
from docx.oxml import OxmlElement

BODY_FONT = "微软雅黑"
OUT = "Agent使用心得-科研与算法实践.docx"


def set_cjk(style_or_run, name=BODY_FONT):
    """中文字体必须显式设置 eastAsia，否则只设 name 不生效。"""
    style_or_run.font.name = name
    element = style_or_run.element
    rpr = element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)


def style_document(document):
    normal = document.styles["Normal"]
    set_cjk(normal)
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.45
    normal.paragraph_format.space_after = Pt(4)

    for level, size in ((1, 15), (2, 12.5), (3, 11)):
        style = document.styles[f"Heading {level}"]
        set_cjk(style)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(4)


def para(document, text, bold_prefix=None, indent=0):
    p = document.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Cm(0.6 * indent)
    if bold_prefix:
        run = p.add_run(bold_prefix)
        run.bold = True
        set_cjk(run)
    # 支持行内 **加粗**（python-docx 不解析 Markdown，必须自己拆 run）
    if "**" in text:
        for index, chunk in enumerate(text.split("**")):
            if not chunk:
                continue
            run = p.add_run(chunk)
            run.bold = (index % 2 == 1)
            set_cjk(run)
        return p
    run = p.add_run(text)
    set_cjk(run)
    return p


def bullet(document, text, bold_prefix=None):
    p = document.add_paragraph(style="List Bullet")
    if bold_prefix:
        run = p.add_run(bold_prefix)
        run.bold = True
        set_cjk(run)
    run = p.add_run(text)
    set_cjk(run)
    return p


def code_block(document, text):
    p = document.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), "F2F4F7")
    p._p.get_or_add_pPr().append(shading)
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    return p


def table(document, header, rows, widths=None):
    t = document.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 必须关掉自动布局并显式写 tblLayout，否则 Word/LO 会按内容重新分配列宽
    t.autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(layout)
    hdr = t.rows[0].cells
    for i, text in enumerate(header):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(9.5)
        set_cjk(run)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(text)
            run.font.size = Pt(9.5)
            set_cjk(run)
    if widths:
        for i, width in enumerate(widths):
            t.columns[i].width = Cm(width)
        for row in t.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Cm(width)
    document.add_paragraph()
    return t


def build():
    doc = Document()
    style_document(doc)
    section = doc.sections[0]
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

    title = doc.add_heading("Agent 在科研与算法工作中的使用心得", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("——以综保区行政通道智慧监管原型（车路侧四层算法链路）为例")
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x55, 0x5F, 0x6D)
    set_cjk(run)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run("2026 年 11 月　｜　文中所有数字均可在仓库内复现")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x77, 0x7F, 0x8A)
    set_cjk(run)

    # ---------- 一 ----------
    doc.add_heading("一、为什么写这份心得", level=1)
    para(doc, "过去几个月，我用 agent 从零搭建了一套仅依赖监控视频的车辆异常排查系统："
              "从 66 分钟真实录像出发，做检测、跟踪、行为向量化、无监督异常检测、车辆关系图谱、"
              "风险融合与可解释报告，最终形成一套**可在开发机上端到端跑通的算法原型**"
              "（命令行形态，一次运行约 45 秒产出向量/告警/图谱/风险/报告五类产物）、"
              "14 份文档与 255 个通过的单元测试。")
    para(doc, "这里要区分「原型」与「产品」：前者指算法链路能跑通、结果可复现、有门禁把关；"
              "后者还要求服务化、容器化、边缘验证与安装封装。本项目只做到前者，"
              "具体边界见 6.5 节——这个区分在下面会反复用到。")
    para(doc, "过程中最值得记下来的不是代码本身，而是**哪些事该交给 agent、哪些事交出去会出事**。"
              "这份心得按「角色—方法—翻车—边界」四部分组织，尽量用本项目的具体例子说明，"
              "而不是停留在泛泛的体会。")

    # ---------- 二 ----------
    doc.add_heading("二、Agent 在算法工作中的四种角色", level=1)
    para(doc, "把「agent 帮写代码」当成一件事，是低估也是误判。实际用下来，它至少扮演四种角色，"
              "而这四种角色的可替代性差别很大。")
    table(doc,
          ["角色", "具体做什么", "本项目的例子", "可替代性"],
          [
              ["代码考古", "通读模块、理清数据契约与调用链", "73 个 src 模块的数据流梳理成 CODE_STRUCTURE.md", "高"],
              ["实现者", "写模块、补测试、接 CLI", "Isolation Forest 纯 NumPy 实现、Louvain、证据链", "高（但需验收）"],
              ["实验员", "设计对照实验、跑基准、量化指标", "解码预取 +28%、skip=2 +47%、多进程 0.82×", "中"],
              ["审计者", "找矛盾、拆数字、自我证伪", "41 告警容斥闭合、T 维度双阻塞、样本内问题", "低（最难替代）"],
          ],
          widths=[2.2, 4.6, 6.8, 2.4])
    para(doc, "一个直接的经验：让 agent 写代码，收益是线性的；让它审计自己的产出，收益往往是非线性的。"
              "因为写出来的东西如果不审计，错误会被后续所有工作继承。")

    # ---------- 三 ----------
    doc.add_heading("三、Agent 真正不可替代的三件事", level=1)

    doc.add_heading("3.1 把模糊说法变成可测数字", level=2)
    para(doc, "研究早期到处都是「效果一般」「误差偏大」这类描述。agent 的价值在于把它们压成数字，"
              "并且是**带条件的数字**。三个例子：")
    bullet(doc, "「误报率偏高」→ 实测 9.23%（41/444），再拆成 contamination 曲线："
                "0.01→2.25%、0.05→9.23%、0.08→14.41%、0.10→17.79%。结论从「偏高」变成"
                "「这是参数选择，调 0.08 仍达标」。", bold_prefix="① ")
    bullet(doc, "「预测不准」→ 分三档视野测：0.2 秒 0.21 米、1 秒 1.22 米（81.6% 样本 ≤2 米）、"
                "3 秒无法评估。", bold_prefix="② ")
    bullet(doc, "「3 秒做不到」→ 继续查原因：该车道轨迹中位数只有 1.84 秒，"
                "够长到能测 3 秒的仅 13/444（2.9%）。于是结论从「模型不行」变成"
                "「机位看不到 3 秒，需要换镜头」。", bold_prefix="③ ")
    para(doc, "这类工作的共同点：agent 不只是算一个数，而是把「结论成立的条件」一并算出来。"
              "条件往往比数字本身更有用。")

    doc.add_heading("3.2 数字溯源：让每个数都能当众算平", level=2)
    para(doc, "验收时最容易被问住的是「这个数怎么来的」。本项目的关键数字最后都做到了可拆解：")
    table(doc,
          ["数字", "拆解方式", "结论"],
          [
              ["444 次通行", "432 分析目标 + 12 排除（8 卡车 + 4 摩托车）", "类型过滤没有误杀"],
              ["41 辆告警", "22(IF) + 22(Transformer) + 4(规则) − 4 − 3 − 2 + 2（容斥）", "没有第四种隐藏来源"],
              ["1276 条证据", "432 + 432 + 408 + 4（按信号来源）", "12 辆非目标车不产证据"],
              ["105 个社区", "按规模加权 = 432 节点", "无孤立节点，最大 10 辆"],
          ],
          widths=[3.0, 9.0, 4.0])
    para(doc, "「能算平」这件事本身是一种质量信号：算不平就意味着有未被理解的部分，"
              "通常是 bug 或口径混乱。41 告警那次容斥闭合，直接证明了告警系统没有隐藏逻辑。")

    doc.add_heading("3.3 主动找反例：让 agent 证伪自己", level=2)
    para(doc, "这是我认为 agent 在科研场景里最被低估的用法。项目里有四个例子，都是「结论被自己的实验推翻」：")
    table(doc,
          ["原本的判断", "做的对照实验", "实测结果"],
          [
              ["「9.23% 是模型的固有误报率」", "改 contamination 扫描", "它是分位数构造出来的，是旋钮不是属性"],
              ["「A 对规则命中的异常无感」", "蛇形车 vs 其余车 Mann-Whitney", "A 显著更高（p=0.002），原判断错"],
              ["「模型在训练数据上表现良好」", "同一天时间留出（259 训练 / 173 测试）", "IF 5.02%→6.36%，Transformer 5.02%→4.05%"],
              ["「多开进程可撑 20 路」", "1 进程 vs 2 进程并发基准", "聚合吞吐 0.82×，反而不如单进程"],
          ],
          widths=[5.4, 5.6, 5.0])
    para(doc, "这些实验都不是我一开始就想做的，而是「既然这个数字要写进验收报告，那就先试着推翻它」。"
              "四个里面有三个被直接推翻，还有一个被打折——这是好事，写进报告的都是活下来的结论。")

    # ---------- 四 ----------
    doc.add_heading("四、四类翻车与对应教训", level=1)
    para(doc, "同样的，失败模式比成功经验更值得记。以下四类在本项目里都真实发生过。")

    doc.add_heading("4.1 过度概括", level=2)
    para(doc, "发现「合成注入的样本对无监督模型不可见（AUC 0.39）」后，我写下"
              "「本项目的检出主要靠规则，无监督模型基本没起作用」。下一轮做对照实验发现："
              "被规则判为蛇形的 4 辆车里，3 辆同时触发 IF，且 A 显著更高（0.686 vs 0.451，p=0.002）。")
    para(doc, "教训：**一个负面结果只对当时那组实验成立**。推广到系统层面之前，必须换一个角度再验证一次。")

    doc.add_heading("4.2 自己写的检查器自己错", level=2)
    para(doc, "两次典型：交付检查脚本把文档里的占位符 scripts/xxx.py 当成真实引用，导致检查失败；"
              "并发基准脚本的「跳过」分支被误挂到 elif 下，内存够也会被跳过。")
    para(doc, "教训：检查工具也需要被检查。最简单的办法是让它先报一次「已知应该通过」的用例——"
              "如果它报错了，多半是检查器的问题而不是项目的问题。")

    doc.add_heading("4.3 静默丢弃数据", level=2)
    para(doc, "标注评估脚本按 passage_id 关联标注。当标注文件里部分 id 写错时，脚本会静默跳过，"
              "只评估对上的那部分——用户标了 100 条，可能实际只算到 70 条，而且不会知道。")
    para(doc, "修复后输出：")
    code_block(doc, "[WARNING] 20 / 60 labels did not match any vector passage_id (33.3%)\n"
                    '{"labels_provided": 60, "labels_unmatched": 20, "labeled_vehicles": 40}')
    para(doc, "教训：**凡是有 join 的地方，都要报「没 join 上多少」**。静默丢弃比报错危险得多。")

    doc.add_heading("4.4 归因不彻底", level=2)
    para(doc, "T 维度（时间频率）恒为 0。第一版归因是「缺车牌识别」；补查后发现还缺第二个条件："
              "样本只有 66 分钟，每辆车只通行 1 次，所以就算车牌完美，日内高频也无从观察。")
    para(doc, "教训：第一个找到的原因往往只是**充分条件之一**。对「为什么是 0」这类问题，"
              "要追问到「把所有必要条件都列出来」。")

    # ---------- 五 ----------
    doc.add_heading("五、可复用的五种工作方法", level=1)
    table(doc,
          ["方法", "做法", "本项目的体现"],
          [
              ["先量化再断言", "任何「好/差/高/低」都必须附数字与条件", "所有指标表都带实测值与复现命令"],
              ["数字必须能拆平", "关键计数要能用另一种方式加总回来", "444 / 41 / 1276 / 105 四项全部闭合"],
              ["把检查自动化", "把「人工会忘的事」写成脚本并入 CI", "CI 6 步 + 交付门禁 5 项（文件/配置/引用/目录）"],
              ["报告要人读", "格式正确不代表语义正确，必须通读", "低风险车写「行为模式不符」、把检测器分数写成「规则命中」"],
              ["负面结果也交叉验证", "推翻结论前先换角度复查", "A 对蛇形「无感」→ 实测 p=0.002"],
          ],
          widths=[3.4, 6.2, 6.4])

    para(doc, "其中「报告要人读」这条最容易被忽视。本项目的自动化测试全部通过，但通读报告时发现："
              "低风险车辆的证据区仍写着「行为模式与常态不符」，与结论「风险较低」自相矛盾；"
              "还有把 iforest_score=0.71 这种检测器分数写成「命中预设监管规则」的假证据。"
              "这些问题字段正确、格式正确、可追溯性 100%，只有人读才能发现。")

    # ---------- 六 ----------
    doc.add_heading("六、Agent 做不到的事（边界）", level=1)
    para(doc, "把边界写清楚，比列举能力更有用。以下四类，agent 在本项目里确实无能为力：")

    doc.add_heading("6.1 没有真值就无法判断", level=2)
    para(doc, "23 项技术指标里只有 6 项能在当前数据下实测达标。其余 17 项不是「没做」，而是缺数据："
              "违规标注、双机位同车真值、现场标定控制点、整天以上录像、CUDA 13 环境。")
    para(doc, "agent 可以在拿到这些数据后立刻算出召回率、精确率、AUC——回填链路已经用合成标签验证过，"
              "但它不能替你把数据变出来。这一点必须诚实说明，否则会变成「看起来完成了」。")

    doc.add_heading("6.2 业务定义权", level=2)
    para(doc, "什么算「蚂蚁搬家」？多少个节点、多长时间窗？高风险阈值 80 分该不该下调？"
              "融合口径选「群体优先」还是「个体优先」？这些问题没有技术上的最优解，只有业务上的取舍。")
    para(doc, "agent 能做的是把每种选择的后果算清楚（例如：默认 8 辆中风险 / 开启印证 13 辆），"
              "然后把决定权交回给人。")

    doc.add_heading("6.3 现场与硬件", level=2)
    para(doc, "标定需要人去车道上量五个距离；Jetson 需要一台真机；Docker 镜像需要 daemon 跑起来。"
              "我可以把图和填写表准备好（已生成标定示意图与量测表），但量尺子这件事替代不了。")

    doc.add_heading("6.4 判断「够不够」", level=2)
    para(doc, "agent 倾向于把能测的都测了，但「这就够交付了吗」是人的判断。"
              "本项目当前状态是：代码侧闭环、指标侧受外部输入限制——这个结论该由项目负责人认，"
              "而不是由 agent 宣布完成。")

    doc.add_heading("6.5 补充：「原型」与「产品」的界线", level=2)
    para(doc, "「可运行的原型」容易被读成「可以上线了」。更准确的分档是这样：")
    table(doc,
          ["能力", "状态", "说明"],
          [
              ["全链路端到端跑通", "✅ 已验证", "视频 → 向量 → 告警 → 图谱 → 风险 → 报告，单次约 45 秒"],
              ["结果可复现", "✅ 已验证", "同一输入两次独立运行，444 车 / 41 告警 / 8 中风险完全一致"],
              ["有自动化门禁", "✅ 已验证", "CI 6 步（含端到端冒烟）+ 交付清单 5 项"],
              ["服务化（API/常驻/界面）", "❌ 没有", "目前只有命令行脚本，按文件批处理"],
              ["多路视频接入", "❌ 没有", "一次处理一批文件，无流式接入与调度"],
              ["容器镜像", "🟡 未验证", "Dockerfile 已写，但开发机 daemon 未启动，从未真实构建"],
              ["边缘设备（ARM64）", "❌ 未验证", "代码无 x86 专属依赖，但构建链需换 JetPack 版 wheel"],
              ["安装封装", "❌ 没有", "需要手工建 venv 并装依赖"],
              ["性能达标", "🟠 未达标", "单进程 8.4 路，目标 20 路"],
          ],
          widths=[5.6, 2.6, 8.2])
    para(doc, "所以对外应表述为「可在开发机端到端跑通的算法原型」，"
              "而不是「可运行的系统」。这个差别不是文字游戏——它决定了对方能拿它做什么："
              "能用来验证算法与指标，不能直接部署到卡口。")

    # ---------- 七 ----------
    doc.add_heading("七、本项目的量化账", level=1)
    table(doc,
          ["项目", "数字", "说明"],
          [
              ["单元测试", "255 个通过", "含边界、可复现性、CLI 错误路径、依赖声明"],
              ["CI 步骤", "6 步全绿", "单测 / 冒烟 / 规则审计 / 指标 / 演示包 / 交付门禁"],
              ["变更记录", "153 条", "每条含原因、改动、验证；其中 20 条为缺陷修复"],
              ["代码规模", "73 个 src 模块 / 37 个脚本 / 39 个测试文件", "—"],
              ["文档", "14 份", "含验收报告、指标溯源、运维、性能、效益、标定指南"],
              ["指标达成", "6 / 23 实测达标", "10 项待外部输入、2 项未达标、5 项无真值"],
          ],
          widths=[3.2, 6.8, 6.0])

    # ---------- 八 ----------
    doc.add_heading("八、给使用者的建议", level=1)
    bullet(doc, "先让 agent 读代码、复述数据契约，再让它动手写。本项目最早的一次线上事故就是"
                "「没看清调用约定，改了函数签名导致整条链路产出 0 辆车」。", bold_prefix="① ")
    bullet(doc, "要命令，不要结论。让它给「跑什么命令、看什么产物」，比让它说「我验证过了」可靠。",
           bold_prefix="② ")
    bullet(doc, "定期问一句「哪些数字你还没有验证过」。这一问往往能挖出若干「看起来有、实际没测」的指标。",
           bold_prefix="③ ")
    bullet(doc, "保留它的负面结果。本项目的负面结论（多进程 0.82×、T 恒为 0、80 分档不可达）"
                "比正面结果更能防止后续误判。", bold_prefix="④ ")
    bullet(doc, "人负责定义与拍板。指标口径、业务阈值、验收标准，这些不该由 agent 决定。", bold_prefix="⑤ ")

    # ---------- 九 ----------
    doc.add_heading("九、结语", level=1)
    para(doc, "这套系统最终交付的是一个「可运行、可验证、但指标只完成了一部分」的原型。"
              "如果只用一句话概括 agent 在其中的作用：")
    quote = doc.add_paragraph()
    quote.paragraph_format.left_indent = Cm(1.0)
    run = quote.add_run("它把「做研究时大量存在但没人愿意做的核对工作」变成了可以持续进行的事——"
                        "拆数字、跑对照、读报告、查依赖、复现每一步；但它不替你做判断，也不替你要数据。")
    run.italic = True
    set_cjk(run)
    para(doc, "这也是我认为 agent 目前最适合科研与算法工作的定位：不是替代研究者，"
              "而是把研究过程中那些「应该做、但时间不够所以常常跳过」的验证环节真正做掉。")

    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the agent-experience Word document")
    parser.add_argument("--output", default=OUT, help="Output .docx path")
    args = parser.parse_args()
    OUT = args.output
    build()

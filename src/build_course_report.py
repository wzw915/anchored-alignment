# -*- coding: utf-8 -*-
"""人工智能基础大作业报告生成器.

基于 deeplogit 锚定对齐项目的真实实验结果, 按课程模板格式规范生成 docx:
- 章标题 小三号黑体加粗(15pt), 段前段后0.5行; 2级 四号黑体加粗(14pt); 3级 小四黑体不加粗(12pt)
- 正文 小四宋体(12pt), 西文 Times New Roman, 1.5倍行距, 首行缩进2字符
- 三线表, 图表题注 (图3-1 / 表3-1)

用法: python src/build_course_report.py <output.docx>
"""
import os
import sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(PROJ, "figures")

HEI, SONG = "黑体", "宋体"
WEST = "Times New Roman"


def set_font(run, east=SONG, west=WEST, size=12, bold=False, color=None):
    run.font.name = west
    run.font.size = Pt(size)
    run.font.bold = bold
    r = run._element.rPr
    rf = r.rFonts if r.rFonts is not None else r.get_or_add_rFonts()
    rf.set(qn("w:eastAsia"), east)
    if color:
        run.font.color.rgb = color


def para(doc, text, size=12, east=SONG, bold=False, align=None,
         indent=True, line15=True, before=0, after=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    if line15:
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        pf.alignment = align
    r = p.add_run(text)
    set_font(r, east=east, size=size, bold=bold)
    return p


def h1(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(13)
    pf.space_after = Pt(13)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    r = p.add_run(text)
    set_font(r, east=HEI, size=15, bold=True)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    r = p.add_run(text)
    set_font(r, east=HEI, size=14, bold=True)
    return p


def h3(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(3)
    pf.space_after = Pt(3)
    r = p.add_run(text)
    set_font(r, east=HEI, size=12, bold=False)
    return p


def mixed(doc, segments):
    """正文段落, segments = [(text, east, bold), ...]; 宋体小四1.5倍距缩进."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.first_line_indent = Pt(24)
    for text, east, bold in segments:
        r = p.add_run(text)
        set_font(r, east=east, size=12, bold=bold)
    return p


def caption(doc, text, before=4, after=10):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    r = p.add_run(text)
    set_font(r, east=HEI, size=10.5, bold=False)
    return p


def picture(doc, fname, width_cm, cap):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run()
    r.add_picture(os.path.join(FIG, fname), width=Cm(width_cm))
    caption(doc, cap)


def _set_border(el, edge, sz, val="single"):
    e = OxmlElement(f"w:{edge}")
    e.set(qn("w:val"), val)
    e.set(qn("w:sz"), str(sz))
    e.set(qn("w:space"), "0")
    e.set(qn("w:color"), "000000")
    el.append(e)


def three_line_table(doc, header, rows, widths=None, cap=None, note=None):
    """三线表: 顶线/底线 1.5pt, 栏目线 0.75pt, 无竖线."""
    if cap:
        caption(doc, cap, before=8, after=4)
    t = doc.add_table(rows=len(rows) + 1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    _set_border(borders, "top", 12)
    _set_border(borders, "bottom", 12)
    _set_border(borders, "left", 0, "none")
    _set_border(borders, "right", 0, "none")
    _set_border(borders, "insideH", 0, "none")
    _set_border(borders, "insideV", 0, "none")
    tblPr.append(borders)
    # 栏目线: 表头行下边框 0.75pt
    for cell in t.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcB = OxmlElement("w:tcBorders")
        _set_border(tcB, "bottom", 6)
        tcPr.append(tcB)
    data = [header] + rows
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.paragraphs[0].text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            set_font(r, east=HEI if ri == 0 else SONG, size=10.5,
                     bold=(ri == 0))
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(8)
        r = p.add_run(note)
        set_font(r, east=SONG, size=9)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def formula(doc, text, number):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    r = p.add_run(f"{text}        ({number})")
    set_font(r, east=SONG, west=WEST, size=12)
    return p


def cover(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("南京理工大学紫金学院")
    set_font(r, east=HEI, size=26, bold=True)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(30)
    r = p.add_run("人工智能基础大作业报告")
    set_font(r, east=HEI, size=22, bold=True)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(24)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("基于多模态嵌入锚定对齐的零样本商品替代关系估计")
    set_font(r, east=HEI, size=15, bold=True)

    info = [("课程编号：", "033370021"), ("课程名称：", "人工智能基础"),
            ("学    院：", "计算机与人工智能"), ("专    业：", "计算机科学与技术"),
            ("起止时间：", "2026.9 - 2026.12"), ("指导教师：", "杜兰")]
    t = doc.add_table(rows=len(info), cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, (k, v) in enumerate(info):
        for ci, val in enumerate((k, v)):
            cell = t.cell(ri, ci)
            cell.paragraphs[0].text = ""
            pp = cell.paragraphs[0]
            pp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = pp.add_run(val)
            set_font(rr, east=SONG, size=14, bold=(ci == 0))

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("小组成员与分工")
    set_font(r, east=HEI, size=12, bold=True)

    members = [
        ("学号", "姓名", "主要工作", "贡献值"),
        ("2315302414", "王政文", "锚定对齐模型设计与训练框架实现、投影梯度优化与 δ_max 选取、"
                                 "主实验与消融实验、数据溯源审计与可复现代码整理", "40%"),
        ("2315302401", "岑琼", "数据集构建与外部验证（问卷数据处理、Amazon 共购/共评论图构建）",
         "30%"),
        ("2315302407", "林佳琪", "实验设计、下游定价与并购仿真、图表绘制与报告撰写", "30%"),
    ]
    t = doc.add_table(rows=4, cols=4)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _w = [Cm(2.5), Cm(2.0), Cm(8.7), Cm(2.0)]
    for _row in t.rows:
        for _ci, _wd in enumerate(_w):
            _row.cells[_ci].width = _wd
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        _set_border(borders, edge, 4)
    tblPr.append(borders)
    for ri, row in enumerate(members):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.paragraphs[0].text = ""
            pp = cell.paragraphs[0]
            pp.paragraph_format.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if (ci == 2 and ri > 0)
                else WD_ALIGN_PARAGRAPH.CENTER)
            rr = pp.add_run(val)
            set_font(rr, east=HEI if ri == 0 else SONG, size=10.5,
                     bold=(ri == 0))

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(60)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("2026年12月")
    set_font(r, east=SONG, size=15)
    doc.add_page_break()


def toc(doc):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("目  录")
    set_font(r, east=HEI, size=15, bold=True)
    entries = [
        ("1  概述", 0), ("1.1  项目背景", 1), ("1.2  文献综述", 1),
        ("1.3  主要工作", 1),
        ("2  理论基础及模型构建", 0), ("2.1  替代结构的经验分解算法", 1),
        ("2.2  基于多模态嵌入的对比表示学习", 1),
        ("2.3  范数有界投影梯度优化算法", 1),
        ("2.4  三者整合：锚定对齐模型", 1),
        ("3  实验分析", 0), ("3.1  实验设计", 1),
        ("3.2  实验结果分析", 1),
        ("4  结论", 0), ("5  总结", 0),
        ("数据可用性声明（Data Availability）", 0), ("6  参考文献", 0),
    ]
    for text, lv in entries:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.left_indent = Pt(18 * lv)
        r = p.add_run(text)
        set_font(r, east=HEI if lv == 0 else SONG, size=12, bold=(lv == 0))
    doc.add_page_break()


def chapter1(doc):
    h1(doc, "1  概述")
    h2(doc, "1.1  项目背景")
    para(doc, "电子商务平台上每年有大量新商品上市。对平台和卖家而言，准估计这些新商品的替代关系，"
              "即当某件商品缺货或涨价时，需求会转移到哪些其他商品，是定价、选品和并购筛查等决策的基础。"
              "传统需求计量模型需要每个商品积累充分的历史选择数据才能估计替代结构，而新上市商品恰恰没有"
              "这类数据，这构成了典型的冷启动问题。")
    para(doc, "近年来，多模态深度学习的发展使得商品的自然语言描述与商品图片可以被预训练模型编码为稠密"
              "向量（嵌入），文本相似度被用来代理替代关系，从而绕开对历史选择数据的依赖。然而一个自然的"
              "做法是用平台积累的个体级选择数据（谁买了什么、接着又买了什么）对预训练嵌入进行微调，希望"
              "微调后的嵌入更贴近真实的替代结构。本项目的研究起点正是检验这一做法：个体级微调在冷启动的"
              "零样本替代关系估计任务上是否有效。实验给出了否定的答案，并进一步提出了修复方案。")
    h2(doc, "1.2  文献综述")
    para(doc, "将文本与图像信息引入需求估计是近年计量经济学与营销科学交叉领域的热点方向。Compiani 等"
              "（2026）系统提出了用文本与图像嵌入估计需求的工作框架，证明语义相似度在陈述性选择数据上与"
              "真实替代份额显著相关，但其行为数据实验仍依赖商品级汇总信息。在表示学习方面，对比学习"
              "（如 InfoNCE 损失）与预训练语言模型微调已成为表征适配的标准范式；然而大量实证研究也注意到"
              "微调过程可能破坏预训练空间的整体几何结构，导致下游零样本任务性能退化，这一现象与灾难性"
              "遗忘在机理上同源。针对微调退化，参数高效微调与约束微调（如对更新量施加范数约束或投影）"
              "被证明能在保留预训练知识的同时吸收任务信号。")
    para(doc, "综合已有工作可以发现两点不足：其一，个体级选择数据与商品级替代结构之间的聚合层级差异"
              "（谁为什么-switch 与哪个商品替代哪个商品是两种不同的统计对象）尚缺少正式刻画，多数工作"
              "默认个体级数据是更高保真的监督信号；其二，面向冷启动零样本场景的约束微调方案缺少在真实"
              "平台数据与下游商业决策任务上的系统验证。本项目围绕这两点展开。")
    h2(doc, "1.3  主要工作")
    para(doc, "本项目在 Compiani 等（2026）的框架上完成了一项完整的修正性研究，主要工作包括三个方面："
              "（1）通过少样本反学习曲线实验与嵌入几何分析，正式提出并验证了聚合层级错配假说，说明个体级"
              "微调失败的原因是监督层级与任务层级不匹配，而非数据量不足；（2）提出锚定对齐模型，将预训练"
              "嵌入冻结为锚点，仅训练范数有界的残差，并以商品级替代份额为监督信号，从机制上禁止微调破坏"
              "预训练几何；（3）在问卷数据与两个真实 Amazon 行为数据集上完成留一商品交叉验证、消融实验与"
              "外部验证，并将估计结果接入 Bertrand 定价仿真与并购筛查仿真，量化错误替代矩阵的商业代价。"
              "全部实验基于真实数据，代码与数据处理脚本可完整复现。")


def chapter2(doc):
    h1(doc, "2  理论基础及模型构建")
    h2(doc, "2.1  替代结构的经验分解算法")
    para(doc, "替代结构被表示为替代份额矩阵（diversion matrix）S，其中元素 S_ik 表示当商品 i 不可得时，"
              "其需求转移到商品 k 的条件概率。该矩阵由原始选择数据经条件分解得到：对于问卷数据，受访者"
              "首先选出首选商品 i，再在其余商品中选出第二选择 k，大量受访者的二阶选择按列归一化后即得到"
              "经验转移概率：")
    formula(doc, "S_ik = P( 第二选择 = k | 第一选择 = i )", "2-1")
    para(doc, "对于行为数据，替代代理从共购买（co-purchase）或共评论（co-review）图中构造：若某用户"
              "同批次购买（或先后评论）了商品 i 与 k，则记一条共现边；将商品 i 的共现邻居集合按共现权重"
              "行归一化，得到行为侧的替代代理矩阵。问卷分解给出的是显式陈述的替代结构，行为图给出的是"
              "显示性偏好代理，二者共同构成本项目的监督与评估基准。")
    h2(doc, "2.2  基于多模态嵌入的对比表示学习")
    para(doc, "商品的文本标题、描述与图片经预训练多模态编码器映射为 384 维嵌入向量 c_j，经行归一化后"
              "位于单位超球面上，其余弦相似度 c_i·c_j 语义相似度的度量。基线方法将全部嵌入作为可训练"
              "参数，以个体级选择对为正样本、其余商品为负样本，采用 InfoNCE 对比损失进行个体级微调。"
              "本项目实验证明该做法在零样本替代估计任务上失效：微调后同类目商品的平均余弦相似度从"
              " 0.185 跌至 0.040，预训练空间的类别簇结构被打散。")
    para(doc, "锚定对齐（Anchored Alignment）是本项目的核心模型。它将预训练嵌入冻结为锚点 c_j，为每个"
              "商品引入可训练残差向量，最终表示为二者之和：")
    formula(doc, "e_j = c_j + \u0394_j ,  约束  \u2016\u0394_j\u2016\u2082 \u2264 \u03b4_max", "2-2")
    para(doc, "监督信号不再是个体选择对，而是 2.1 节分解得到的商品级替代份额：以温度化 softmax 将"
              " e_i 与全部 e_k 的内积归一化为预测切换分布，与真实份额做交叉熵：")
    formula(doc, "L = \u2212 \u03a3_i \u03a3_{k\u2260i} S_ik log softmax( e_i\u1d40 e_k / \u03c4_train )_k", "2-3")
    para(doc, "其中训练温度 \u03c4_train 固定，评估温度 \u03c4_eval = 0.3 为独立的预测期参数，二者不联合调参。"
              "由于监督目标本身是商品级的，损失函数与任务层级一致，从数据流上消除了聚合层级错配。")
    picture(doc, "figS1_anchor_geometry.png", 11.5,
            "图2-1  锚定对齐几何示意：表示停留在锚点邻域内，个体级微调则可能漂移出预训练流形")
    h2(doc, "2.3  范数有界投影梯度优化算法")
    para(doc, "模型采用投影梯度下降训练：每步梯度更新后，将所有残差投影回半径 \u03b4_max 的球面，"
              "即 \u0394_j \u2190 \u0394_j \u00b7 min(1, \u03b4_max/\u2016\u0394_j\u2016\u2082)。该约束使任意锚点与最终表示的余弦相似度有下界"
              " \u221a(1\u2212\u03b4_max\u00b2)，从机制上禁止训练把表示推出预训练几何的邻域。需要强调的是，范数"
              "约束在本模型中的角色是鲁棒性保险而非精度来源：精度增益来自商品级监督层级的校正，消融"
              "实验（3.2 节）对这一成分分离给出了直接验证。部署时 \u03b4_max 仅用训练行以小网格"
              " {0.1, 0.3, 0.5, 0.7} 选取，留出商品的列对选择步骤不可见，因此不存在测试集信息泄露。")
    h2(doc, "2.4  三者整合：锚定对齐模型")
    para(doc, "三个组件构成完整的零样本替代关系估计流水线：经验分解算法（2.1）把原始选择数据转化为"
              "商品级监督矩阵；对比表示学习框架（2.2）提供预训练语义嵌入与残差适配结构；投影梯度优化"
              "（2.3）在有界约束下训练残差。整体流程为：预训练嵌入冻结并行归一化 \u2192 构造替代份额"
              "矩阵 \u2192 留一商品划分训练/评估折 \u2192 残差初始化为零并在有界球内以交叉熵训练 \u2192 用"
              "余弦相似度矩阵输出完整替代结构。对无任何选择数据的新商品，其残差恒为零，模型退化为"
              "冻结嵌入相似度，服务时零额外开销。总体技术路线如图 2-2 所示。")
    picture(doc, "fig_course_framework.png", 13.0,
            "图2-2  锚定对齐零样本替代关系估计总体技术路线")


def chapter3(doc):
    h1(doc, "3  实验分析")
    h2(doc, "3.1  实验设计")
    h3(doc, "3.1.1  实验环境")
    para(doc, "本项目实验环境配置如表 3-1 所示。模型基于 PyTorch 框架实现，全部训练与评估在 CPU 上"
              "完成，单次完整实验（三个数据集留一交叉验证）约 3 小时。")
    three_line_table(
        doc,
        ["类别", "配置"],
        [["操作系统", "Windows 11 x64"],
         ["深度学习框架", "PyTorch 2.14.1 (CPU)"],
         ["开发语言", "Python 3.11"],
         ["主要依赖库", "scikit-learn, pandas, NumPy, matplotlib"],
         ["硬件环境", "Intel Core 处理器 / 32 GB 内存"]],
        cap="表3-1  实验环境配置")
    h3(doc, "3.1.2  实验数据集")
    para(doc, "实验使用三个真实数据集，统计信息如表 3-2 所示。数据集 1 为图书品类消费者问卷：受访者"
              "对真实在售图书完成首选与第二选择，共得到 9,265 个有效二阶选择对，商品的文本与图片嵌入"
              "取自公开多模态编码器。数据集 2 与 3 来自 Amazon Reviews 2023 公开语料：图书共购子集"
              "覆盖 249 万名购买 3 本以上图书的用户，视频游戏共评论子集覆盖 275 万名评论者；共现图"
              "行归一化后作为行为侧替代代理。")
    three_line_table(
        doc,
        ["数据集", "商品数", "监督来源", "规模"],
        [["图书问卷", "10", "二阶选择（显式）", "9,265 个选择对"],
         ["Amazon 图书共购", "175", "共购图（行为代理）", "6,906 条边 / 249 万用户"],
         ["Amazon 游戏共评论", "200", "共评论图（行为代理）", "10,051 条边 / 275 万评论者"]],
        cap="表3-2  实验数据集统计信息")
    h3(doc, "3.1.3  评价指标")
    para(doc, "采用均方根误差（RMSE）衡量替代矩阵逐元素的点预测精度，Kendall \u03c4 秩相关系数衡量"
              "替代排序质量（对每列预测份额与真实份额的秩一致性），Top-3 命中率衡量最重要三个替代品"
              "的识别准确率。显著性检验采用十折上的精确符号翻转置换检验（枚举全部 1,024 种符号分配），"
              "多重比较按 Benjamini-Hochberg 程序在错误发现率 q = 0.05 下校正。")
    h3(doc, "3.1.4  模型参数设置与训练")
    para(doc, "关键超参数设置如下：评估温度 \u03c4_eval = 0.3；残差界 \u03b4_max = 0.5（由训练行网格 "
              "{0.1, 0.3, 0.5, 0.7} 选取，0.3\u20130.7 区间为稳定平台）；训练温度 \u03c4_train 固定；"
              "优化器为 Adam，外部验证集训练 800 轮；评估协议为留一商品 10 折交叉验证，即每折遮蔽"
              "一个商品的替代行列，用其余商品训练、被遮蔽商品评估。对比方法包括：冻结嵌入基线、"
              "个体级 InfoNCE 微调、无界残差（相同商品级监督但去掉范数约束）与本文锚定对齐。")
    h3(doc, "3.1.5  对比实验设计")
    para(doc, "为验证模型有效性与机理，实验设计如下：（1）主对比实验，四种方法在问卷数据集上的替代"
              "矩阵恢复精度与排序质量比较；（2）消融实验，少样本反学习曲线（训练书数 2\u21929）、半径"
              "扫描（\u03b4_max 从 0.1 到 0.7）与主/居中两种预处理协议下的成分分离；（3）外部验证，"
              "模型不经修改直接迁移到两个 Amazon 行为数据集；（4）下游任务仿真，以各方法估计的替代"
              "矩阵驱动 Bertrand 定价优化与并购筛查，用真实矩阵评估决策损失。")
    h2(doc, "3.2  实验结果分析")
    para(doc, "主对比实验结果如表 3-3 所示。个体级微调使 RMSE 从 0.081 恶化到 0.123（p = 0.006），"
              "甚至差于均匀分配需求；在均值居中的预处理协议下其排序质量同步崩塌（\u03c4 从 0.478 跌至"
              " 0.139，p = 0.004）。锚定对齐将 RMSE 恢复到 0.077、\u03c4 提升到 0.656（对比冻结基线 "
              "p = 0.004），两项指标均为最优，验证了商品级监督与有界约束组合的有效性。")
    three_line_table(
        doc,
        ["方法", "监督层级", "RMSE", "Kendall \u03c4"],
        [["冻结嵌入（基线）", "—", "0.0808", "0.472"],
         ["个体级 InfoNCE 微调", "个体级（9,265 对）", "0.1225", "0.433"],
         ["无界残差", "商品级", "0.0788", "0.656"],
         ["锚定对齐（本文）", "商品级", "0.0774", "0.656"]],
        cap="表3-3  主对比实验结果（留一商品 10 折，均值）",
        note="注：个体级微调 RMSE 劣于冻结基线，p = 0.006；锚定对齐 \u03c4 优于冻结基线，p = 0.004（置换检验，FDR 校正后仍显著）。")
    para(doc, "少样本实验给出了失败机理的直接证据：个体级微调的训练书数从 2 本增加到 9 本时，RMSE"
              " 从 0.115 单调恶化到 0.296，学习曲线反向，说明失败源于聚合层级错配而非数据量不足。"
              "消融实验进一步完成成分分离：监督层级携带几乎全部精度增益，范数约束则贡献跨预处理协议"
              "的稳定性，有界训练在两种协议下 RMSE 波动不超过 0.010，而无界残差波动达 0.061 且可能"
              "劣于冻结基线。图 3-1 的 t-SNE 可视化直观展示了三种策略下的嵌入几何：冻结嵌入保留"
              "类别簇（同类目平均余弦 0.185），个体级微调将同类目打散（0.040），锚定对齐在吸收任务"
              "信号的同时基本保留簇结构（0.547）。")
    picture(doc, "fig10b_tsne_triptych.png", 15.5,
            "图3-1  三种表示策略下的 t-SNE 嵌入空间对比：(A)冻结 (B)个体级微调 (C)锚定对齐")
    para(doc, "图 3-2 给出残差界 \u03b4_max 的敏感性分析。主协议下 RMSE 与 \u03c4 在 \u03b4_max \u2208 "
              "[0.3, 0.7] 区间同时形成平台：两项指标均优于完全微调且不低于冻结基线，说明该方法"
              "不需要精细调参；居中协议下曲线的单调性则从另一侧面印证了界对精度-稳定性权衡的控制"
              "作用。")
    picture(doc, "figS2_delta_robustness.png", 15.5,
            "图3-2  残差界 \u03b4_max 敏感性：(A)RMSE 与 (B)Kendall \u03c4 随界半径变化（阴影为稳定平台区）")
    para(doc, "外部验证结果如表 3-4 与图 3-3 所示。在不修改任何模型结构与超参数的前提下，锚定对齐"
              "在图书共购数据上将 \u03c4 从 0.160 提升到 0.208（+31%），在游戏共评论数据上从 0.199 提升"
              "到 0.257（+29%），置换检验 p 均小于 0.0001，且不以 Top-3 命中率为代价。无界残差虽能"
              "进一步提高 \u03c4，但在图书数据上将 Top-3 命中率减半（p < 0.0001），说明去掉约束后在"
              "稠密行为图上过拟合排序而牺牲头部识别。")
    three_line_table(
        doc,
        ["数据集", "方法", "Kendall \u03c4", "相对冻结提升"],
        [["图书共购", "冻结 / 锚定对齐", "0.160 / 0.208", "+31%"],
         ["图书共购", "无界残差", "0.258", "+61%（Top-3 减半）"],
         ["游戏共评论", "冻结 / 锚定对齐", "0.199 / 0.257", "+29%"],
         ["游戏共评论", "无界残差", "0.276", "+39%"]],
        cap="表3-4  外部验证结果（行为替代代理，留一商品协议）")
    picture(doc, "fig12_external.png", 15.5,
            "图3-3  跨数据集外部验证：(A)语义-替代相关系数及 95% 置信区间 (B)排序性能对比")
    para(doc, "下游决策仿真量化了错误替代矩阵的商业代价。定价仿真中，以各方法估计的替代矩阵驱动"
              " Bertrand-Nash 价格优化、用真实矩阵评估利润：个体微调矩阵导致单品级利润损失 25.45%，"
              "而锚定对齐与冻结基线的损失均低于 0.25%（仿真参数：毛利率 40%，价格敏感度 \u03b2 = 0.5，"
              "外部选项权重 0.3，见图 3-4）。并购筛查仿真覆盖全部 45 个商品对：真实涨价压力最高达 "
              "36.9%，而个体微调矩阵的预测最高仅 3.6%，在任何常规阈值下都不会触发任何筛查警报，"
              "等价于无条件放行；锚定矩阵可恢复检出能力。需要说明的是，该仿真的九对放行场景无法"
              "评估假警报率，外部效度有限，结论应理解为筛查风险的方向性证据。")
    picture(doc, "fig11_profit_loss.png", 12.5,
            "图3-4  定价仿真：不同替代矩阵驱动定价的利润损失（真实矩阵评估）")


def chapter4(doc):
    h1(doc, "4  结论")
    para(doc, "本项目围绕多模态嵌入在冷启动零样本需求估计中的应用完成了建模、实验与下游验证，"
              "得到四点结论。第一，个体级微调在冷启动替代估计任务中稳定有害：其失败源于个体选择"
              "信号与商品级替代结构之间的聚合层级错配，且随个体数据增多而加剧。第二，锚定对齐模型"
              "通过冻结锚点、有界残差与商品级监督三者的组合，将替代矩阵恢复精度恢复并超过冻结基线"
              "（RMSE 0.077，\u03c4 0.656），且几何可视化证实其保留了预训练空间的类别拓扑。第三，"
              "增益来源可以明确归因：商品级监督层级携带精度增益，范数约束携带跨协议稳定性，二者"
              "缺一不可。第四，错误替代矩阵的决策代价是实在的，单品定价损失可达 25.45%，并购筛查"
              "会退化为无条件放行，而锚定对齐在两类任务中均将损失控制在与冻结基线相当的水平。")
    para(doc, "本研究的实践含义是：在缺乏商品级选择数据的冷启动场景，不应直接用个体级数据微调"
              "预训练嵌入；若能获得哪怕少量的商品级二阶选择信息（如小型上市调研），锚定对齐即可"
              "安全地吸收任务信号。语义嵌入相似度在行为数据上的代理能力有限（相关系数约 0.23），"
              "上线前应针对目标品类用真实替代数据校准。")


def chapter5(doc):
    h1(doc, "5  总结")
    para(doc, "本项目从选题、数据构建、模型设计到实验分析历时一学期完成。工作中遇到并解决的典型"
              "问题包括：（1）全量共购图的对账审计在大内存字典方案下内存溢出，改用整数位图索引后"
              "流式扫描 2,900 万行原始数据完成逐边核对；（2）共现边构建时发现权重被静默截断，"
              "经无截断重跑验证所有结论在完整数据下不变，并在报告中如实披露截断口径；（3）一次"
              "交叉验证折数统计错误使样本量被低估一半（4,632 对 9,265），经唯一键排查修正后重跑"
              "全部主实验；（4）t-SNE 可视化的逐点文本标注相互重叠，改为三面板联合最大化间距的"
              "少量代表样本标注方案。这些问题使我们体会到，实验可信度不仅取决于模型本身，更取决"
              "于数据管道的逐字节可核查与评估协议的严格区分。")
    para(doc, "自我评价方面，本项目完成了从数据到决策的完整闭环，全部结果基于真实数据可复现，"
              "但仍有明确局限：主问卷数据仅覆盖 10 种商品，统计功效有限；行为侧共购与共评论只是"
              "替代关系的代理而非真值；并购仿真仅能评估漏检方向而无法评估假警报。后续工作可在"
              "三个方向推进：扩充问卷商品规模以 tighten 置信区间；引入更细粒度的用户异质性建模；"
              "将锚定残差结构推广到价格弹性等更丰富的需求参数。")


def declaration(doc):
    """数据可用性与代码可用性声明 (仿 DSS 期刊 Declarations 惯例, 独立成节)."""
    h1(doc, "数据可用性声明（Data Availability）")
    para(doc, "数据集 2 与 3 来自公开的 Amazon Reviews 2023 语料 [2]，可自由获取；本项目的下载、"
              "清洗与共现图构建脚本随代码一并提供。数据集 1 问卷原始数据包含受访者个体信息，不对外"
              "公开；经匿名化与聚合处理后得到的商品级替代份额矩阵随代码提供。")
    para(doc, "代码可用性：全部代码（模型训练、数据构建、审计脚本与端到端复现指令）公开托管于项目"
              "代码仓库，并附公开数据源的 SHA-256 数据集指纹，保证所有图表结果可逐字节复现。")


def chapter6(doc):
    h1(doc, "6  参考文献")
    refs = [
        "[1] Compiani G, Morozov I, Seiler S. Demand estimation with text and image data[J]. "
        "Decision Support Systems, 2026, 202: 114599.",
        "[2] Ni J, Li J, McAuley J. Justifying recommendations using distantly-labeled reviews "
        "and fine-grained aspects[C]. Proceedings of EMNLP-IJCNLP, 2019: 188-197.",
        "[3] van den Oord A, Li Y, Vinyals O. Representation learning with contrastive "
        "predictive coding[EB/OL]. arXiv preprint arXiv:1807.03748, 2018.",
        "[4] Kendall M G. A new measure of rank correlation[J]. Biometrika, 1938, 30(1/2): 81-93.",
        "[5] Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and "
        "powerful approach to multiple testing[J]. Journal of the Royal Statistical Society: "
        "Series B, 1995, 57(1): 289-300.",
        "[6] Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural Computation, "
        "1997, 9(8): 1735-1780.",
        "[7] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]. Advances in "
        "Neural Information Processing Systems, 2017: 5998-6008.",
        "[8] Goodfellow I, Bengio Y, Courville A. Deep Learning[M]. Cambridge: MIT Press, 2016.",
        "[9] 周志华. 机器学习[M]. 北京: 清华大学出版社, 2016.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.space_after = Pt(2)
        r = p.add_run(ref)
        set_font(r, east=SONG, size=12)


def main(out):
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin, sec.bottom_margin = Cm(2.54), Cm(2.54)
    sec.left_margin, sec.right_margin = Cm(3.0), Cm(2.6)
    cover(doc)
    toc(doc)
    chapter1(doc)
    chapter2(doc)
    chapter3(doc)
    chapter4(doc)
    chapter5(doc)
    declaration(doc)
    chapter6(doc)
    doc.save(out)
    print("saved:", out)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(PROJ), "人工智能基础大作业报告_锚定对齐.docx")
    main(out)

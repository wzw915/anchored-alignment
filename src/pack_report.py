"""把报告草稿打包为自包含目录：图片归入 images/、Markdown 图片路径改写为相对路径。

产物：deliverable/报告正文.md + deliverable/images/*.png
"""
import os
import re
import shutil

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_MD = os.path.join(PROJ, "report", "大作业报告_草稿.md")
FIG = os.path.join(PROJ, "figures")
OUT = os.path.join(PROJ, "deliverable")
IMG = os.path.join(OUT, "images")

IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def main():
    if not os.path.exists(SRC_MD):
        raise SystemExit(f"缺少报告草稿：{SRC_MD}")
    os.makedirs(IMG, exist_ok=True)
    text = open(SRC_MD, encoding="utf-8").read()

    used, missing = [], []

    def repl(m):
        alt, path = m.group(1), m.group(2)
        base = os.path.basename(path.replace("\\", "/"))
        src = os.path.join(FIG, base)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(IMG, base))
            if base not in used:
                used.append(base)
            return f"![{alt}](images/{base})"
        missing.append(base)
        return m.group(0)

    text = IMG_RE.sub(repl, text)
    dst = os.path.join(OUT, "报告正文.md")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[pack] 报告 -> {dst}")
    print(f"[pack] 图片 {len(used)} 张 -> {IMG}")
    for b in used:
        print(f"        - {b}")
    if missing:
        print(f"[pack] !! 以下图片引用了但未生成：{missing}")


if __name__ == "__main__":
    main()

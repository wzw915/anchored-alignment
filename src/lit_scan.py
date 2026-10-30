# -*- coding: utf-8 -*-
"""批量扫描文献 PDF：抽取每篇首页+第2页文本，保存 JSON 索引供分类筛选。"""
import json
import re
from pathlib import Path

from pypdf import PdfReader

PROJ = Path(__file__).resolve().parent.parent
LIT_DIR = Path(r"E:\人工智能\Demand Estimation with Text and Image Data")
OUT = PROJ / "results" / "lit_scan.json"

def clean(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def main():
    recs = []
    for pdf in sorted(LIT_DIR.glob("*.pdf")):
        rec = {"file": pdf.name, "chars": pdf.stat().st_size}
        try:
            r = PdfReader(str(pdf))
            rec["pages"] = len(r.pages)
            txt = []
            for i in range(min(2, len(r.pages))):
                txt.append(r.pages[i].extract_text() or "")
            rec["head"] = clean(" ".join(txt))[:2200]
        except Exception as e:
            rec["error"] = str(e)
        recs.append(rec)
        print(f"[ok] {pdf.name} pages={rec.get('pages')} err={rec.get('error','')}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {OUT}  ({len(recs)} files)")

if __name__ == "__main__":
    main()

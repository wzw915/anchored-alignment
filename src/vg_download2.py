# -*- coding: utf-8 -*-
"""VG 下载 v2: 僵死检测 + 断点续传,直到两个文件字节级完整。"""
import os, time, subprocess, urllib.request

BASE = "https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/"
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
FILES = [("meta_Video_Games.jsonl", "meta_categories/meta_Video_Games.jsonl"),
         ("Video_Games.jsonl", "review_categories/Video_Games.jsonl")]
NPART = 12
CURL = ["curl", "-sL", "--retry", "5", "--speed-time", "30",
        "--speed-limit", "20000", "-m", "900"]


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length", 0)) or \
               int(r.headers.get("X-Linked-Size", 0))


def fill(url, out, size):
    part = size // NPART
    bounds = [(i * part, size - 1 if i == NPART - 1 else (i + 1) * part - 1)
              for i in range(NPART)]
    for rnd in range(60):
        todo = []
        for i, (a, b) in enumerate(bounds):
            pf = f"{out}.p{i}"
            want = b - a + 1
            cur = os.path.getsize(pf) if os.path.exists(pf) else 0
            if cur < want:
                todo.append((i, a, b, want, cur))
        if not todo:
            return True
        print(f"  round {rnd}: {len(todo)} parts behind "
              f"({sum(c for *_, c in todo)/1e6:.0f}MB to go)", flush=True)
        procs = []
        for i, a, b, want, cur in todo:
            pf = f"{out}.p{i}"
            tmp = f"{out}.p{i}.new"
            procs.append(subprocess.Popen(
                CURL + ["-r", f"{a + cur}-{b}", "-o", tmp, url],
                stdout=subprocess.DEVNULL))
        for (i, a, b, want, cur), p in zip(todo, procs):
            p.wait()
            pf = f"{out}.p{i}"
            tmp = f"{out}.p{i}.new"
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                with open(pf, "ab") as fdst, open(tmp, "rb") as fsrc:
                    while True:
                        c = fsrc.read(1 << 22)
                        if not c:
                            break
                        fdst.write(c)
                os.remove(tmp)
    return False


ok = True
for name, path in FILES:
    url = BASE + path
    out = os.path.join(AMZ, "vg_" + name)
    size = head_size(url)
    have = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"{name}: size={size/1e6:.1f}MB have={have/1e6:.1f}MB", flush=True)
    if have == size:
        print("  complete", flush=True)
        continue
    done = fill(url, out, size)
    if done:
        with open(out, "wb") as fo:
            for i in range(NPART):
                pf = f"{out}.p{i}"
                with open(pf, "rb") as fi:
                    while True:
                        c = fi.read(1 << 22)
                        if not c:
                            break
                        fo.write(c)
        got = os.path.getsize(out)
        print(f"  assembled {got} (want {size}) "
              f"{'OK' if got == size else 'BAD'}", flush=True)
        if got == size:
            for i in range(NPART):
                os.remove(f"{out}.p{i}")
            continue
    ok = False
    print("  INCOMPLETE — will resume on next run", flush=True)
print("ALL_OK" if ok else "INCOMPLETE")

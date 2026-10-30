# -*- coding: utf-8 -*-
"""8 路 Range 并行下载 Digital_Music meta+reviews，断点续传。"""
import os, sys, time, urllib.request

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMZ = os.path.join(PROJ, "data", "amazon")
BASE = "https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/"
FILES = [("meta_Digital_Music.jsonl", "meta_categories/meta_Digital_Music.jsonl"),
         ("Digital_Music.jsonl", "review_categories/Digital_Music.jsonl")]
NPART = 8


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length", 0)) or \
               int(r.headers.get("X-Linked-Size", 0))


def fetch(url, out, size):
    part = size // NPART
    bounds = [(i * part, (size - 1) if i == NPART - 1 else (i + 1) * part - 1)
              for i in range(NPART)]
    procs = []
    for i, (a, b) in enumerate(bounds):
        pf = f"{out}.p{i}"
        want = b - a + 1
        have = os.path.getsize(pf) if os.path.exists(pf) else 0
        if have >= want:
            continue
        # 断点续传: 从已有位置继续
        import subprocess
        hdr = ["-r", f"{a + have}-{b}"] if have else ["-r", f"{a}-{b}"]
        p = subprocess.Popen(
            ["curl", "-sL", "--retry", "5"] + hdr +
            ["-o", pf, url], stdout=subprocess.DEVNULL)
        procs.append((p, pf, want))
    # 等待 + 补漏循环
    for rnd in range(20):
        alive = [p for p, _, _ in procs if p.poll() is None]
        if not alive:
            break
        time.sleep(10)
    # 检查每片并重试缺的
    for rnd in range(10):
        missing = []
        for i, (a, b) in enumerate(bounds):
            pf = f"{out}.p{i}"
            want = b - a + 1
            have = os.path.getsize(pf) if os.path.exists(pf) else 0
            if have < want:
                missing.append(i)
        if not missing:
            break
        print(f"round {rnd}: retry parts {missing}", flush=True)
        for i in missing:
            a, b = bounds[i]
            pf = f"{out}.p{i}"
            have = os.path.getsize(pf) if os.path.exists(pf) else 0
            subprocess.run(["curl", "-sL", "--retry", "5",
                            "-r", f"{a + have}-{b}", "-o", pf, url],
                           stdout=subprocess.DEVNULL)
    # 拼接
    with open(out, "wb") as fo:
        for i in range(NPART):
            pf = f"{out}.p{i}"
            with open(pf, "rb") as fi:
                while True:
                    chunk = fi.read(1 << 22)
                    if not chunk:
                        break
                    fo.write(chunk)
    got = os.path.getsize(out)
    print(f"assembled {out}: {got} bytes (want {size})", flush=True)
    if got != size:
        # 清掉分片让下次重来
        for i in range(NPART):
            os.remove(f"{out}.p{i}")
        return False
    for i in range(NPART):
        os.remove(f"{out}.p{i}")
    return True


ok = True
for name, path in FILES:
    url = BASE + path
    out = os.path.join(AMZ, "dm_" + name)
    size = head_size(url)
    have = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"{name}: size={size/1e6:.1f}MB have={have/1e6:.1f}MB", flush=True)
    if have == size:
        print("  complete, skip", flush=True)
        continue
    if os.path.exists(out):
        os.remove(out)
    ok &= fetch(url, out, size)
print("ALL_OK" if ok else "INCOMPLETE")

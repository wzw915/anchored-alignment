"""用 curl 从 hf-mirror 下载 HF 模型到本地目录（绕过 huggingface_hub 空文件问题）。

用法: python hf_fetch.py <repo_id> <local_dir>
示例: python hf_fetch.py sentence-transformers/all-MiniLM-L6-v2 models/st_minilm
"""
import json
import os
import subprocess
import sys

MIRROR = "https://hf-mirror.com"


def _curl(url: str, out: str = None) -> str:
    cmd = ["curl", "-sL", "--max-time", "600"]
    if out:
        cmd += ["-o", out, "-w", "%{http_code}"]
    r = subprocess.run(cmd + [url], capture_output=True, text=True)
    return r.stdout.strip()


def list_repo_files(repo_id: str) -> list:
    raw = _curl(f"{MIRROR}/api/models/{repo_id}/tree/main?recursive=true")
    tree = json.loads(raw)
    return [it["path"] for it in tree if it["type"] == "file"]


def main(repo_id: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    files = list_repo_files(repo_id)
    print(f"[hf_fetch] {repo_id}: {len(files)} 个文件")
    ok, fail = [], []
    for f in files:
        dst = os.path.join(out_dir, f.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # 跳过无关大文件
        if f.endswith((".bin", ".h5", ".msgpack", ".onnx", ".tflite", ".ot")) and \
           not f.endswith(("pytorch_model.bin",)):
            print(f"  跳过 {f}")
            continue
        if f == "rust_model.ot":
            continue
        url = f"{MIRROR}/{repo_id}/resolve/main/{f}"
        code = _curl(url, dst)
        size = os.path.getsize(dst) if os.path.exists(dst) else 0
        if code == "200" and size > 0:
            ok.append(f)
            print(f"  OK   {f} ({size/1e6:.1f} MB)")
        else:
            fail.append(f)
            print(f"  FAIL {f} (http {code}, {size} bytes)")
    if fail:
        print("[hf_fetch] 失败文件:", fail)
        sys.exit(1)
    print("[hf_fetch] 全部完成 ->", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

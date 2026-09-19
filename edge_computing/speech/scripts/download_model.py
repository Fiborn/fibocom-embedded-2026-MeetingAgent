#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载实时会议所需模型：ASR + 说话人嵌入 + VAD
用法: python3 scripts/download_model.py
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ASR_DIR = MODELS / "asr"

# 默认 2023 版（约 223MB，板端稳定）
ASR_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2"
)
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)
VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.onnx"
)


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("[跳过] {}".format(dest))
        return
    print("下载: {}".format(url))

    def prog(block, size, total):
        if total > 0:
            sys.stdout.write("\r进度: {}%".format(min(100, block * size * 100 // total)))
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=prog)
    print("\n完成: {}".format(dest.name))


def extract_asr(archive):
    model = ASR_DIR / "model.onnx"
    tokens = ASR_DIR / "tokens.txt"
    if model.exists() and tokens.exists():
        print("[跳过] ASR 已解压")
        return
    print("解压 ASR 模型…")
    ASR_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive), "r:bz2") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name == "tokens.txt":
                member.name = name
                tar.extract(member, str(ASR_DIR))
            elif name in ("model.onnx", "model.int8.onnx"):
                member.name = "model.onnx"
                tar.extract(member, str(ASR_DIR))
    if not model.exists() or not tokens.exists():
        missing = []
        if not model.exists():
            missing.append(str(model))
        if not tokens.exists():
            missing.append(str(tokens))
        raise RuntimeError("ASR 解压不完整，缺少: {}".format(", ".join(missing)))
    print("ASR -> {}/".format(ASR_DIR))


def main():
    print("=" * 50)
    print("下载实时会议模型（ASR + 说话人 + VAD）")
    print("=" * 50)
    archive = MODELS / "asr.tar.bz2"
    asr_model = ASR_DIR / "model.onnx"
    asr_tokens = ASR_DIR / "tokens.txt"
    try:
        if not (asr_model.exists() and asr_tokens.exists()):
            if asr_tokens.exists() and not asr_model.exists():
                print("检测到 tokens.txt 但缺少 model.onnx，重新下载 ASR…")
            download(ASR_ARCHIVE_URL, archive)
            extract_asr(archive)
            if archive.exists():
                archive.unlink()
        download(EMBEDDING_URL, MODELS / "embedding.onnx")
        download(VAD_URL, MODELS / "silero_vad.onnx")
    except Exception as e:
        print("\n失败: {}".format(e))
        print("请检查网络，或手动从 sherpa-onnx releases 下载到 models/")
        sys.exit(1)
    print("\n下一步: python3 src/check_env.py")
    print("启动实时: python3 src/meeting_realtime.py")


if __name__ == "__main__":
    main()

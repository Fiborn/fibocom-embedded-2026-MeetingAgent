# -*- coding: utf-8 -*-
"""环境检查"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

OK, FAIL = "[OK]", "[FAIL]"


def main():
    print("=" * 50)
    print("实时会议识别 — 环境检查")
    print("=" * 50)

    ok = True
    print("\n【Python 依赖】")
    for m in ["numpy", "librosa", "pyaudio", "onnxruntime", "sherpa_onnx"]:
        try:
            __import__(m)
            print("  {} {}".format(OK, m))
        except ImportError:
            print("  {} {}".format(FAIL, m))
            ok = False

    print("\n【模型文件】")
    checks = [
        ("ASR", config.ASR_MODEL),
        ("ASR tokens", config.ASR_TOKENS),
        ("说话人嵌入", config.EMBEDDING_MODEL),
        ("VAD", config.VAD_MODEL),
    ]
    for name, p in checks:
        if p.exists():
            print("  {} {} ({:.1f} MB)".format(OK, name, p.stat().st_size / 1024 / 1024))
        else:
            print("  {} {} 缺失".format(FAIL, name))
            ok = False

    print("\n【麦克风】")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        idx = config.INPUT_DEVICE_INDEX
        if idx is None:
            idx = p.get_default_input_device_info()["index"]
        info = p.get_device_info_by_index(idx)
        print("  {} [{}] {}".format(OK, idx, info["name"]))
        p.terminate()
    except Exception as e:
        print("  {} {}".format(FAIL, e))
        ok = False

    print("=" * 50)
    print("Python: {}.{}.{}".format(*sys.version_info[:3]))
    if ok:
        print("就绪。运行: python3 src/meeting_realtime.py")
    else:
        print("请先: bash scripts/install_env.sh && python3 scripts/download_model.py")
        sys.exit(1)


if __name__ == "__main__":
    main()

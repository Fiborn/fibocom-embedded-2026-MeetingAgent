# -*- coding: utf-8 -*-
"""HTTP 服务：实时结果推送可在此基础上扩展"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request

from analyze_file import analyze_wav

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": "realtime_transcript"})


@app.route("/meeting/analyze", methods=["POST"])
def analyze():
    """离线 wav → [{"speaker":"说话人1","text":"..."}, ...]"""
    wav = request.get_json(force=True).get("wav_path")
    if not wav or not Path(wav).exists():
        return jsonify({"error": "wav_path 无效"}), 400
    lines = analyze_wav(wav)
    result = []
    for line in lines:
        if ": " in line:
            sp, txt = line.split(": ", 1)
            result.append({"speaker": sp, "text": txt})
    return jsonify({"utterances": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

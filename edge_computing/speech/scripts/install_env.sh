#!/bin/bash
# SC171 板端安装依赖（直接使用系统 Python 3.8.10，不创建虚拟环境）
# 用法: cd speech && bash scripts/install_env.sh

set -e

echo "========== 1/3 检查 Python 版本 =========="
python3 --version

echo ""
echo "========== 2/3 安装系统依赖 =========="
sudo apt-get update
sudo apt-get install -y \
    python3-dev python3-pip \
    portaudio19-dev libsndfile1 ffmpeg wget

echo ""
echo "========== 3/3 安装 Python 库（系统 Python）=========="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python3 -m pip install -U pip setuptools wheel --user
python3 -m pip install --user -r "$PROJECT_DIR/requirements-py38.txt" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "=========================================="
echo "安装完成！直接使用 python3 运行，无需 activate"
echo ""
echo "下一步:"
echo "  cd $PROJECT_DIR"
echo "  python3 scripts/download_model.py"
echo "  python3 src/list_devices.py"
echo "  python3 src/check_env.py"
echo "  python3 src/meeting_realtime.py"
echo "=========================================="

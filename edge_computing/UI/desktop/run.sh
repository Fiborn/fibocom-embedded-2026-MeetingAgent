#!/usr/bin/env bash
# 在板端实体屏（DSI 1920x1080 / DISPLAY=:0）启动 Meeting Agent 桌面 UI
# 注意：必须能访问 /dev/snd（不要在隔离沙箱里跑，否则无实时 ASR）
set -euo pipefail
cd "$(dirname "$0")/.."

export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR/pulse" 2>/dev/null || true

if lsusb 2>/dev/null | grep -qiE 'cafe:4110|Timesintelli' \
  && ! grep -qiE 'Timesintelli|USB-Audio|\[Device' /proc/asound/cards 2>/dev/null; then
  echo "[错误] USB 麦已插入，但 ALSA 未注册声卡（设备挂死）。" >&2
  echo "       请拔掉 Timesintelli USB 麦，等待 5 秒再插上，然后重新运行本脚本。" >&2
  exit 1
fi

# Cursor Agent 沙箱会隐藏 /dev/snd/pcmC*：/proc 有卡，但节点不存在 → arecord Invalid value for card
usb_card=""
while read -r num _rest; do
  case "$num" in
    ''|*[!0-9]*) continue ;;
  esac
  if grep -A1 -E "^[[:space:]]*${num}[[:space:]]" /proc/asound/cards 2>/dev/null \
    | grep -qiE 'Timesintelli|USB-Audio|\[Device'; then
    usb_card="$num"
    break
  fi
done < <(awk '/^[[:space:]]*[0-9]+[[:space:]]+\[/{print $1}' /proc/asound/cards 2>/dev/null)

if [ -n "$usb_card" ] && [ ! -e "/dev/snd/pcmC${usb_card}D0c" ]; then
  echo "[错误] 声卡 card ${usb_card} 在 /proc 可见，但 /dev/snd/pcmC${usb_card}D0c 不存在。" >&2
  echo "       这通常是 Cursor/沙箱隔离了音频设备，不是麦克风坏了。" >&2
  echo "       请打开板端本机终端（SSH/本地 tty，非 Agent 沙箱）执行：" >&2
  echo "       cd /home/fibo/MeetingAgent/UI && bash desktop/run.sh" >&2
  exit 1
fi

if [ ! -d /dev/snd ] || ! grep -q 'USB\|DOV\|Timesintelli\|\[Device' /proc/asound/cards 2>/dev/null; then
  echo "[警告] 未检测到可用麦克风声卡。" >&2
  echo "       请在板端本机终端启动（勿在 Cursor 沙箱中），并确认 USB 麦已连接：" >&2
  echo "       cd /home/fibo/MeetingAgent/UI && bash desktop/run.sh" >&2
fi
echo "[音频] /proc/asound/cards:" >&2
cat /proc/asound/cards 2>/dev/null | sed 's/^/  /' >&2 || true
if [ -n "$usb_card" ]; then
  echo "[音频] 设备节点: /dev/snd/pcmC${usb_card}D0c OK" >&2
fi

exec python3 -m desktop.app "$@"

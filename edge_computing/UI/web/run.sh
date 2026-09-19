#!/usr/bin/env bash
# Meeting Agent 动态 Web UI（Flask + SSE）
# 注意：必须能访问 /dev/snd；勿在 Cursor Agent 沙箱中启动。
set -euo pipefail
cd "$(dirname "$0")/.."

export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export MEETING_WEB_PORT="${MEETING_WEB_PORT:-8787}"
mkdir -p "$XDG_RUNTIME_DIR/pulse" 2>/dev/null || true

if lsusb 2>/dev/null | grep -qiE 'cafe:4110|Timesintelli|ff00:0002|CF-IC' \
  && ! grep -qiE 'Timesintelli|USB-Audio|\[Device|\bDOV\b' /proc/asound/cards 2>/dev/null; then
  echo "[错误] USB 麦已插入但 ALSA 未注册声卡，请拔插后重试。" >&2
  exit 1
fi

usb_card=""
while read -r num _rest; do
  case "$num" in ''|*[!0-9]*) continue ;; esac
  if grep -A1 -E "^[[:space:]]*${num}[[:space:]]" /proc/asound/cards 2>/dev/null \
    | grep -qiE 'Timesintelli|USB-Audio|\[Device|\bDOV\b'; then
    usb_card="$num"
    break
  fi
done < <(awk '/^[[:space:]]*[0-9]+[[:space:]]+\[/{print $1}' /proc/asound/cards 2>/dev/null)

if [ -n "$usb_card" ] && [ ! -e "/dev/snd/pcmC${usb_card}D0c" ]; then
  echo "[错误] 声卡可见但 /dev/snd/pcmC${usb_card}D0c 不存在（沙箱隔离）。" >&2
  echo "       请在板端本机终端运行本脚本。" >&2
  exit 1
fi

# 若旧桌面 Tk UI 占用麦克风，提示用户先退出（勿在本脚本里 pkill 自身命令行）
if pgrep -f '[p]ython3 -m desktop.app' >/dev/null 2>&1; then
  echo "[警告] 检测到 desktop.app 仍在运行，可能占用麦克风。请先关闭 Tk 桌面 UI。" >&2
fi

# 按实际控件拉满 Capture（兼容 DOV 单路音量 / Timesintelli 四路）
if [ -n "$usb_card" ]; then
  python3 - "$usb_card" <<'PY' 2>/dev/null || true
import re, subprocess, sys
card = sys.argv[1]
try:
    contents = subprocess.check_output(["amixer", "-c", card, "contents"], text=True, stderr=subprocess.STDOUT, timeout=3)
except Exception as exc:
    print(f"[音频] 读取 mixer 失败: {exc}", file=sys.stderr)
    raise SystemExit(0)
for block in re.split(r"\n(?=numid=)", contents):
    nm = re.search(r"numid=(\d+)", block)
    name_m = re.search(r"name='([^']+)'", block)
    if not nm or not name_m:
        continue
    numid, name = nm.group(1), name_m.group(1)
    type_m = re.search(r"type=([A-Z]+)", block)
    values_m = re.search(r"values=(\d+)", block)
    max_m = re.search(r"max=(\d+)", block)
    ctype = (type_m.group(1) if type_m else "").upper()
    nvals = int(values_m.group(1)) if values_m else 1
    vmax = int(max_m.group(1)) if max_m else None
    try:
        if name == "Keep Interface" and ctype == "BOOLEAN":
            subprocess.run(["amixer", "-c", card, "-q", "cset", f"numid={numid}", "on"], capture_output=True, timeout=2)
            print(f"[音频] card {card} Keep Interface=on", file=sys.stderr)
        elif "Capture Switch" in name and ctype == "BOOLEAN":
            payload = ",".join(["on"] * max(1, nvals))
            subprocess.run(["amixer", "-c", card, "-q", "cset", f"numid={numid}", payload], capture_output=True, timeout=2)
            print(f"[音频] card {card} {name}={payload}", file=sys.stderr)
        elif "Capture Volume" in name and ctype == "INTEGER" and vmax is not None:
            payload = ",".join([str(vmax)] * max(1, nvals))
            subprocess.run(["amixer", "-c", card, "-q", "cset", f"numid={numid}", payload], capture_output=True, timeout=2)
            subprocess.run(["amixer", "-c", card, "-q", "cset", f"numid={numid}", payload], capture_output=True, timeout=2)
            print(f"[音频] card {card} {name}={payload} (max)", file=sys.stderr)
    except Exception:
        pass
PY
fi

echo "[web] http://127.0.0.1:${MEETING_WEB_PORT}" >&2
echo "[音频] /proc/asound/cards:" >&2
cat /proc/asound/cards 2>/dev/null | sed 's/^/  /' >&2 || true

# 可选：自动打开板端浏览器（kiosk 全屏）
if [ "${MEETING_WEB_OPEN:-1}" = "1" ]; then
  (
    sleep 2
    url="http://127.0.0.1:${MEETING_WEB_PORT}/"
    # 日志文件可能被其他用户创建导致 Permission denied；用可写路径并忽略失败
    blog="${TMPDIR:-/tmp}/meeting-web-browser-$(id -u).log"
    : >"$blog" 2>/dev/null || blog="/dev/null"
    # root 开板端桌面时需要 DISPLAY；保留调用方已有值
    export DISPLAY="${DISPLAY:-:0}"
    if command -v firefox >/dev/null 2>&1; then
      firefox --kiosk "$url" >"$blog" 2>&1 &
      echo "[web] 已尝试打开 Firefox kiosk → $url  (log: $blog)" >&2
    elif command -v chromium-browser >/dev/null 2>&1; then
      chromium-browser --kiosk --app="$url" >"$blog" 2>&1 &
      echo "[web] 已尝试打开 Chromium kiosk → $url  (log: $blog)" >&2
    else
      echo "[web] 未找到 firefox/chromium，请手动打开 $url" >&2
    fi
  ) &
fi

exec python3 -m web.app "$@"

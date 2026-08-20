#!/usr/bin/env bash
# =============================================================================
# GAPAGO 실행 스크립트
#
#   배치 분석 : ./run_agent.sh [입력JSON] [출력JSON]
#               인자 생략 시 data/input_sample.json → results/output.json
#   웹 서버   : ./run_agent.sh serve [포트]     (기본 8000)
#
# 필요한 API 키는 환경변수로 주입한다. .env.example 참고.
# =============================================================================
set -euo pipefail

ORIG_PWD="$PWD"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 호출자가 넘긴 상대경로는 호출 시점의 디렉터리 기준으로 확정한다.
# (아래에서 APP_DIR 로 이동하므로, 여기서 절대경로로 바꿔두지 않으면 깨진다)
abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s\n' "$ORIG_PWD/$1" ;;
    esac
}

cd "$APP_DIR"

# ── 웹 서버 모드 ─────────────────────────────────────────────────────────────
if [ "${1:-}" = "serve" ]; then
    PORT="${2:-8000}"
    echo "[GAPAGO] 웹 서버 시작 — http://0.0.0.0:${PORT}"
    exec python -m uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
fi

# ── 배치 분석 모드 ───────────────────────────────────────────────────────────
INPUT_FILE="${1:-}"
OUTPUT_FILE="${2:-}"

if [ -n "$INPUT_FILE" ]; then
    INPUT_FILE="$(abspath "$INPUT_FILE")"
else
    INPUT_FILE="$APP_DIR/data/input_sample.json"
fi

if [ -n "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="$(abspath "$OUTPUT_FILE")"
else
    OUTPUT_FILE="$APP_DIR/results/output.json"
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "[ERROR] 입력 파일을 찾을 수 없습니다: $INPUT_FILE" >&2
    echo "        사용법: ./run_agent.sh <입력JSON> <출력JSON>" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "[GAPAGO] 배치 분석 시작"
echo "  Input  : $INPUT_FILE"
echo "  Output : $OUTPUT_FILE"

exec python main.py --input "$INPUT_FILE" --output "$OUTPUT_FILE"

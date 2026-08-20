#!/usr/bin/env bash
# =============================================================================
# GAPAGO 실행 스크립트
#
#   ./run_agent.sh setup                 환경 준비 (venv · 의존성 · 랜딩 빌드 · .env 점검)
#   ./run_agent.sh serve                 웹 서버
#   ./run_agent.sh analyze "연구 주제"     1회 분석
#   ./run_agent.sh --help                전체 사용법
#
# setup 을 제외한 모든 인자는 main.py 로 그대로 전달된다.
# 즉 실행 옵션의 정의는 main.py 의 argparse 한 곳에만 존재한다.
# =============================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# 가상환경 위치. 홈 디렉터리 쿼터가 빡빡한 HPC 등에서는
#   GAPAGO_VENV=/scratch/$USER/gapago-venv ./run_agent.sh setup
# 처럼 다른 파일시스템을 가리키게 할 수 있다. (의존성은 2GB 안팎)
VENV="${GAPAGO_VENV:-$APP_DIR/.venv}"
MIN_PY="3.10"

# 컨테이너 등 venv 가 필요 없는 환경에서는 시스템 python 을 그대로 쓴다
pick_python() {
    if [ -x "$VENV/bin/python" ]; then
        echo "$VENV/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python3"
    else
        echo "python"
    fi
}

check_python_version() {
    local py="$1"
    "$py" - "$MIN_PY" <<'EOF'
import sys
need = tuple(int(x) for x in sys.argv[1].split("."))
if sys.version_info[:2] < need:
    sys.exit(f"Python {sys.argv[1]} 이상이 필요합니다 (현재 {sys.version.split()[0]})")
EOF
}

# ── setup ────────────────────────────────────────────────────────────────────
if [ "${1:-}" = "setup" ]; then
    echo "[1/4] Python 확인"
    BOOTSTRAP_PY="$(command -v python3 || command -v python)"
    check_python_version "$BOOTSTRAP_PY"
    echo "      $("$BOOTSTRAP_PY" -V)"

    echo "[2/4] 가상환경 · 의존성"
    echo "      위치: $VENV"
    avail_kb=$(df -Pk "$(dirname "$VENV")" 2>/dev/null | awk 'NR==2{print $4}')
    if [ -n "$avail_kb" ] && [ "$avail_kb" -lt 3000000 ]; then
        echo "      ⚠ 여유 공간이 $((avail_kb/1024))MB 입니다. 설치에 약 2GB 가 필요합니다."
        echo "        공간이 넉넉한 곳을 쓰려면:"
        echo "          GAPAGO_VENV=/path/to/venv ./run_agent.sh setup"
    fi
    if [ ! -d "$VENV" ]; then
        "$BOOTSTRAP_PY" -m venv "$VENV"
        echo "      생성 완료"
    fi
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install -r "$APP_DIR/requirements.txt"

    echo "[3/4] .env 점검"
    if [ ! -f "$APP_DIR/.env" ]; then
        cp "$APP_DIR/.env.example" "$APP_DIR/.env"
        echo "      .env 를 생성했습니다. 키를 채운 뒤 다시 실행하세요."
        echo "      최소 요구: TAVILY_API_KEY + LLM provider 1개"
        exit 1
    fi
    missing=""
    grep -qE '^TAVILY_API_KEY=.+' "$APP_DIR/.env" || missing="$missing TAVILY_API_KEY"
    grep -qE '^(AZURE_OPENAI_API_KEY|AWS_ACCESS_KEY_ID|GROQ_API_KEY|GOOGLE_API_KEY)=.+' "$APP_DIR/.env" \
        || missing="$missing LLM_PROVIDER_KEY"
    if [ -n "$missing" ]; then
        echo "      ⚠ 값이 비어 있습니다:$missing"
        echo "        자세한 설명은 docs/CONFIGURATION.md 참고"
    else
        echo "      필수 키 확인 완료"
    fi

    echo "[4/4] 랜딩 페이지"
    if command -v npm >/dev/null 2>&1; then
        (cd "$APP_DIR/web/landing" && npm install --silent && npm run build --silent)
        echo "      빌드 완료"
    else
        echo "      npm 이 없어 건너뜁니다. / 는 분석 앱으로 대체됩니다."
    fi

    echo
    echo "준비 완료. 다음을 실행하세요:"
    if [ "$VENV" != "$APP_DIR/.venv" ]; then
        echo "  GAPAGO_VENV=$VENV ./run_agent.sh serve"
    else
        echo "  ./run_agent.sh serve"
    fi
    exit 0
fi

# ── 그 외 전부 main.py 로 전달 ────────────────────────────────────────────────
PY="$(pick_python)"
exec "$PY" "$APP_DIR/main.py" "$@"

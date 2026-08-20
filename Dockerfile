# =============================================================================
# GAPAGO — 연구 GAP 분석 멀티 에이전트
#
#   빌드 : docker build -t gapago .
#   배치 : docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
#            --env-file .env gapago /app/data/input_sample.json /app/results/output.json
#   서버 : docker run --rm -p 8000:8000 --env-file .env gapago serve
#
# API 키는 이미지에 굽지 않는다. 반드시 실행 시 환경변수로 주입한다.
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 일부 휠이 소스 빌드를 요구할 때를 대비
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성을 먼저 설치한다. 소스만 바뀌면 이 레이어는 캐시에서 재사용된다.
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

COPY . .
RUN chmod +x run_agent.sh

# 키는 값 없이 선언만 한다 (docker run -e / --env-file 로 주입).
# 빈 문자열이 아니라 미설정 상태로 두어야 코드의 기본값 로직이 동작한다.
ENV LLM_PROVIDER=azure \
    RERANK_MODELS=light \
    ORT_DISABLE_GPU_DEVICE_ENUMERATION=1

EXPOSE 8000

ENTRYPOINT ["./run_agent.sh"]

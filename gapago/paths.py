"""
프로젝트 경로의 단일 기준점.

의존성이 없는 최하위 모듈이므로 어디서 import 해도 순환이 생기지 않는다.
각 모듈이 __file__ 로 상대 경로를 세는 방식은 디렉터리를 옮길 때마다
조용히 어긋나므로, 경로는 전부 여기서만 정의한다.
"""

from pathlib import Path

# gapago/paths.py → gapago/ → 리포 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"

# 웹 자산
WEB_DIR = PROJECT_ROOT / "web"
FRONTEND_DIR = WEB_DIR / "app"
LANDING_DIR = WEB_DIR / "landing" / "dist"

import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 리포 루트
ENV_PATH = os.path.join(BASE_DIR, ".env")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

load_dotenv(ENV_PATH, override=False)


def langsmith(project_name=None):
    enabled = os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"

    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        print("LangSmith 추적 OFF")
        return

    project_name = project_name or os.getenv("LANGSMITH_PROJECT", "default-project")

    langchain_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    langsmith_key = os.getenv("LANGSMITH_API_KEY", "").strip()

    # 더 긴 키 선택
    key = max(langchain_key, langsmith_key, key=len)

    if not key:
        print("LangSmith API Key 없음")
        return

    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project_name
    os.environ["LANGSMITH_API_KEY"] = key  

    print(f"LangSmith 추적 ON | project={project_name}")


def env_variable(key, value):
    os.environ[key] = value
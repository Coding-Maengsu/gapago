import os
from functools import lru_cache
from dotenv import load_dotenv
import tempfile

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI 
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrockConverse
from langchain_groq import ChatGroq

load_dotenv()

creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if creds_json:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name

# ── Provider 목록 (사용자 선택용) ──────────────────────────────────
# groq, qwq는 gap_agent 추론 단계 전용 → GAP_REASONING_PROVIDER 환경변수로 설정
# 여기서는 전체 파이프라인에 사용할 기본 LLM만 선택
AVAILABLE_PROVIDERS = {
    "1": ("azure",   "Azure OpenAI (GPT)"),
    "2": ("claude",  "Claude (AWS Bedrock)"),
    "3": ("gemini",  "Google Gemini"),
    "4": ("exaone",  "LG EXAONE (Local GPU)"),
}


@lru_cache(maxsize=8)
def get_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    provider = (provider or os.getenv("LLM_PROVIDER", "azure")).lower()
    model = model or os.getenv("LLM_MODEL", "")

    # ── Azure OpenAI ──
    if provider == "azure":
        deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1-chat")
        return AzureChatOpenAI(
            openai_api_version=os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            ),
            azure_deployment=deployment,
        )

    # ── Claude via AWS Bedrock ──
    if provider in ("claude", "anthropic"):
        from botocore.config import Config as BotoConfig
        bedrock_model = model or os.getenv(
            "BEDROCK_CLAUDE_MODEL",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
        )
        return ChatBedrockConverse(
            model=bedrock_model,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            config=BotoConfig(read_timeout=300, connect_timeout=10, retries={"max_attempts": 2}),
        )

    # ── Google Gemini ──
    if provider in ("gemini", "google"):
        return ChatVertexAI(
            model_name=model or "gemini-3.1-flash-lite-preview",
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "coding-beast"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    # ── LG EXAONE (로컬 GPU, transformers) ──
    if provider == "exaone":
        return _build_exaone_llm(model)

    # ── QwQ-32B / Groq: 사용자 선택 목록에는 없음 ──────────────────────
    # gap_agent가 GAP_REASONING_PROVIDER 환경변수를 읽어
    # get_llm(provider="qwq") 또는 get_llm(provider="groq")를 내부적으로 호출.
    # 사용자가 main.py 실행 시 직접 선택하는 용도가 아닌, 내부 라우팅 전용.
    if provider == "qwq":
        return _build_qwq_llm(model)

    if provider == "groq":
        return _build_groq_llm(model)

    raise ValueError(f"Unsupported provider: {provider}")


def _build_groq_llm(model: str | None = None) -> BaseChatModel:
    """
    Groq API를 통한 Qwen3-32B 호출.
    - ~535 tokens/sec의 빠른 추론 속도
    - Thinking Mode: reasoning_effort="default" (gap_agent 추론 단계)
    - Non-thinking Mode: reasoning_effort="none" (빠른 응답)
    - 가격: $0.29/M input, $0.59/M output tokens
    """
    model_name = model or os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
    reasoning_effort = os.getenv("GROQ_REASONING_EFFORT", "default")  # "default" | "none"

    print(f"  [groq] model={model_name}, reasoning_effort={reasoning_effort}")
    return ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.6,               # Thinking Mode 권장값
        max_tokens=8192,               # CoT 출력 넉넉하게
        reasoning_effort=reasoning_effort,  # model_kwargs가 아닌 직접 파라미터
    )


def _build_qwq_llm(model: str | None = None) -> BaseChatModel:
    """
    QwQ-32B 추론 특화 모델을 HuggingFace pipeline으로 로드.
    GAP 분석의 핵심 추론 단계(장벽 분석, 창의적 방향 제안)에 사용.
    A100 80GB 기준 float16으로 약 64GB 사용.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.chat_models.huggingface import ChatHuggingFace

    model_name = model or os.getenv("QWQ_MODEL_PATH", "Qwen/QwQ-32B")
    print(f"  [qwq] Loading {model_name} ... (첫 호출 시 수 분 소요, A100 권장)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",          # A100 단일 GPU에 자동 배치
        trust_remote_code=True,
    )

    pipe = pipeline(
        "text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        max_new_tokens=4096,        # 추론 모델은 CoT가 길어서 넉넉하게
        do_sample=False,            # greedy decoding (추론 일관성)
        temperature=None,           # do_sample=False 시 불필요
        top_p=None,
        repetition_penalty=1.1,     # 반복 억제
    )
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=hf_llm)


def _build_exaone_llm(model: str | None = None) -> BaseChatModel:
    """EXAONE 모델을 HuggingFace pipeline으로 로드하여 LangChain LLM으로 반환."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline
    from langchain_core.language_models import BaseChatModel
    from langchain_community.chat_models.huggingface import ChatHuggingFace

    model_name = model or os.getenv("EXAONE_MODEL_PATH", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    print(f"  [exaone] Loading {model_name} ... (첫 호출 시 수 분 소요)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    pipe = pipeline(
        "text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        do_sample=False,
    )
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=hf_llm)


def select_provider_interactive() -> str:
    """사용자에게 LLM provider를 선택하게 하고, 선택된 provider 키를 반환."""
    print("\n=== LLM Provider 선택 (전체 파이프라인 기본 LLM) ===")
    for key, (_, desc) in AVAILABLE_PROVIDERS.items():
        print(f"  {key}) {desc}")

    reasoning = os.getenv("GAP_REASONING_PROVIDER", "")
    if reasoning:
        print(f"\n  ※ GAP 추론 단계는 별도로 [{reasoning}] 사용 (GAP_REASONING_PROVIDER 설정됨)")

    current = os.getenv("LLM_PROVIDER", "azure")
    choice = input(f"\n선택 (기본값: {current}) > ").strip()

    if choice in AVAILABLE_PROVIDERS:
        selected = AVAILABLE_PROVIDERS[choice][0]
    elif choice == "":
        selected = current
    else:
        # 직접 provider 이름 입력도 허용
        selected = choice.lower()

    print(f"  → {selected} 선택됨")
    return selected

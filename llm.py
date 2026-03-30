import os
from functools import lru_cache
from dotenv import load_dotenv

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrockConverse

load_dotenv()

# ── Provider 목록 (사용자 선택용) ──────────────────────────────────
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
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
        )

    # ── LG EXAONE (로컬 GPU, transformers) ──
    if provider == "exaone":
        return _build_exaone_llm(model)

    raise ValueError(f"Unsupported provider: {provider}")


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
    print("\n=== LLM Provider 선택 ===")
    for key, (_, desc) in AVAILABLE_PROVIDERS.items():
        print(f"  {key}) {desc}")

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

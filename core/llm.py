import os
from functools import lru_cache
from dotenv import load_dotenv

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from google import genai
from google.genai.types import HttpOptions
import json

# ── Render 환경 Google 인증 처리 (load_dotenv 이전에 실행!) ──────
creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if creds_json:
    creds_path = "/tmp/google_credentials.json"
    with open(creds_path, 'w') as f:
        f.write(creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

load_dotenv(override=False)  # 기존 환경변수 덮어쓰지 않음

# ── Gemini 래퍼 클래스 (gcloud 없이 직접 인증) ──────────────────────
class GeminiVertexChat(BaseChatModel):
    model_name: str = "gemini-3.1-flash-lite-preview"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        prompt = "\n".join([m.content for m in messages])
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response.text))])

    def with_structured_output(self, schema, **kwargs):
        """structured output 지원 - JSON 파싱 후 Pydantic 모델로 변환"""
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        import re

        llm = self

        class StructuredWrapper(Runnable):
            def invoke(self, messages, config=None, **kw):
                # 스키마 정보를 프롬프트에 추가
                schema_str = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
                
                # 마지막 메시지에 JSON 형식 요청 추가
                from langchain_core.messages import SystemMessage
                json_instruction = SystemMessage(
                    content=f"You MUST respond ONLY with a valid JSON object matching this schema. No explanation, no markdown, just raw JSON:\n{schema_str}"
                )
                full_messages = [json_instruction] + list(messages)
                
                result = llm._generate(full_messages)
                text = result.generations[0].message.content
                
                # JSON 추출
                text = text.strip()
                # 마크다운 코드블록 제거
                text = re.sub(r'^```json\s*', '', text)
                text = re.sub(r'^```\s*', '', text)
                text = re.sub(r'\s*```$', '', text)
                text = text.strip()
                
                parsed = json.loads(text)
                return schema(**parsed)

        return StructuredWrapper()

    @property
    def _llm_type(self) -> str:
        return "gemini-vertex"

# ── Provider 목록 (사용자 선택용) ──────────────────────────────────
AVAILABLE_PROVIDERS = {
    "1": ("azure",   "Azure OpenAI (GPT)"),
    "2": ("claude",  "Claude (AWS Bedrock)"),
    "3": ("exaone",  "LG EXAONE (Local GPU)"),
}


@lru_cache(maxsize=8)
def get_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    provider = (provider or os.getenv("LLM_PROVIDER", "azure")).lower()
    model = model or os.getenv("LLM_MODEL", "")

    # ── Azure OpenAI ──
    if provider == "azure":
        from langchain_openai import AzureChatOpenAI
        deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1-chat")
        return AzureChatOpenAI(
            openai_api_version=os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            ),
            azure_deployment=deployment,
        )

    # ── Claude via AWS Bedrock ──
    if provider in ("claude", "anthropic"):
        from langchain_aws import ChatBedrockConverse
        from botocore.config import Config as BotoConfig
        bedrock_model = model or os.getenv(
            "BEDROCK_CLAUDE_MODEL",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
        )
        return ChatBedrockConverse(
            model=bedrock_model,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            temperature=0.2,
            config=BotoConfig(read_timeout=300, connect_timeout=10, retries={"max_attempts": 2}),
        )

    # ── Google Gemini ──
    if provider in ("gemini", "google"):
        return GeminiVertexChat(
            model_name=model or "gemini-3.1-flash-lite-preview"
        )

    # ── LG EXAONE (로컬 GPU, transformers) ──
    if provider == "exaone":
        return _build_exaone_llm(model)

    if provider == "qwq":
        return _build_qwq_llm(model)

    if provider == "groq":
        return _build_groq_llm(model)

    raise ValueError(f"Unsupported provider: {provider}")


def _build_groq_llm(model: str | None = None) -> BaseChatModel:
    model_name = model or os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
    reasoning_effort = os.getenv("GROQ_REASONING_EFFORT", "default")

    from langchain_groq import ChatGroq
    print(f"  [groq] model={model_name}, reasoning_effort={reasoning_effort}")
    return ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.4,
        max_tokens=8192,
        reasoning_effort=reasoning_effort,
    )


def _build_qwq_llm(model: str | None = None) -> BaseChatModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline

    model_name = model or os.getenv("QWQ_MODEL_PATH", "Qwen/QwQ-32B")
    print(f"  [qwq] Loading {model_name} ...")

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
        max_new_tokens=4096,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=pipe)


def _build_exaone_llm(model: str | None = None) -> BaseChatModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline

    model_name = model or os.getenv("EXAONE_MODEL_PATH", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    print(f"  [exaone] Loading {model_name} ...")

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
        max_new_tokens=6000,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


def get_llm_for_agent(state: dict, agent_name: str):
    """model_routing이 있으면 라우터 사용, 없으면 기존 llm_provider fallback"""
    routing = state.get("model_routing") if state else None
    if routing:
        from core.model_router import ModelRouter
        return ModelRouter.from_dict(routing).get_llm(agent_name)
    return get_llm(provider=state.get("llm_provider") if state else None)

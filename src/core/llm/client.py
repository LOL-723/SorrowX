from openai import OpenAI

from set.config import require_llm_settings, settings
from trace.recorder import current_run_id, get_trace_recorder


class LLMClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        timeout: float = 30.0,
        temperature: float = 0.1,
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.client: OpenAI | None = None

    def chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
    ) -> str:
        if not user_message or not user_message.strip():
            raise ValueError("message cannot be empty")

        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        run_id = current_run_id()
        recorder = get_trace_recorder()
        call_id = recorder.record_core_to_llm(
            run_id,
            model=self._model(),
            message_count=len(messages),
            tool_count=0,
        )
        try:
            response = self._openai_client().chat.completions.create(
                model=self._model(),
                messages=messages,
                temperature=self.temperature,
            )
        except Exception as exc:
            recorder.record_llm_to_core(run_id, call_id=call_id, error=str(exc))
            raise
        recorder.record_llm_to_core(run_id, call_id=call_id, usage=getattr(response, "usage", None))
        return response.choices[0].message.content or ""

    def stream_chat(
        self,
        user_message: str,
        system_prompt: str | None = None,
    ):
        if not user_message or not user_message.strip():
            raise ValueError("message cannot be empty")

        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        run_id = current_run_id()
        recorder = get_trace_recorder()
        call_id = recorder.record_core_to_llm(
            run_id,
            model=self._model(),
            message_count=len(messages),
            tool_count=0,
        )
        try:
            stream = self._openai_client().chat.completions.create(
                model=self._model(),
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
        except Exception as exc:
            recorder.record_llm_to_core(run_id, call_id=call_id, error=str(exc))
            raise

        for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if content:
                yield content
        recorder.record_llm_to_core(run_id, call_id=call_id, usage=None)

    def _openai_client(self) -> OpenAI:
        if self.client is None:
            configured = require_llm_settings()
            self.client = OpenAI(
                api_key=self.api_key or configured.DEEPSEEK_API_KEY,
                base_url=self.base_url or configured.DEEPSEEK_BASE_URL,
                timeout=self.timeout,
            )
        return self.client

    def _model(self) -> str:
        if self.model:
            return self.model
        return require_llm_settings().LLM_MODEL or ""

llm_client = LLMClient(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    model=settings.LLM_MODEL,
    timeout=settings.LLM_TIMEOUT,
    temperature=settings.LLM_TEMPERATURE,
)

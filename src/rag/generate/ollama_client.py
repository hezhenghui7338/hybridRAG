"""Ollama chat client.

Uses the official `ollama` Python SDK to talk to a local Ollama daemon. If you don't
have Ollama running, the generation step will fail with a clear error — but the
retrieval pipeline itself is unaffected.
"""

from __future__ import annotations

from collections.abc import Iterator

import ollama


class OllamaGenerator:
    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:7b",
        temperature: float = 0.2,
        num_ctx: int = 4096,
        timeout: int = 120,
        system_prompt: str = "",
    ) -> None:
        self.host = host
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.system_prompt = system_prompt.strip() or (
            "You are a careful assistant. Answer based only on the provided context. "
            "If the context is insufficient, say so."
        )
        # Construct client lazily so missing Ollama only breaks at generate time.
        self._client = ollama.Client(host=self.host, timeout=self.timeout)

    def _format_messages(self, question: str, contexts: list[str]) -> list[dict]:
        ctx_block = "\n\n".join(
            f"[{i + 1}] {c.strip()}" for i, c in enumerate(contexts) if c.strip()
        )
        user_prompt = (
            f"【参考资料】\n{ctx_block}\n\n"
            f"【用户问题】\n{question}\n\n"
            "请根据以上参考资料给出准确、简洁的回答,并在引用处标注编号。"
        )
        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    def generate(self, question: str, contexts: list[str]) -> str:
        msgs = self._format_messages(question, contexts)
        resp = self._client.chat(
            model=self.model,
            messages=msgs,
            options={
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
            # Disable "thinking" so we get the final answer directly.
            # qwen3 / qwen3.5 default to thinking=True which costs seconds and
            # sometimes returns an empty content while filling thinking only.
            think=False,
            stream=False,
        )
        # ollama 0.3+ returns a pydantic ChatResponse; older versions returned dict.
        msg = getattr(resp, "message", None)
        if msg is None and isinstance(resp, dict):
            msg = resp.get("message", {})
        content = getattr(msg, "content", None) if msg is not None else None
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")
        return content or ""

    def stream(self, question: str, contexts: list[str]) -> Iterator[str]:
        msgs = self._format_messages(question, contexts)
        stream = self._client.chat(
            model=self.model,
            messages=msgs,
            options={
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
            think=False,
            stream=True,
        )
        for chunk in stream:
            msg = getattr(chunk, "message", None)
            if msg is None and isinstance(chunk, dict):
                msg = chunk.get("message", {})
            piece = getattr(msg, "content", None) if msg is not None else None
            if piece is None and isinstance(msg, dict):
                piece = msg.get("content")
            if piece:
                yield piece

    def is_available(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False

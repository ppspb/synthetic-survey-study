from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class GenerationTrace:
    parsed: dict[str, Any]
    raw_text: str
    parser_used: str
    attempts: list[dict[str, Any]]
    used_repair: bool = False
    chosen_model: str | None = None
    used_structured_output: bool = False


class GenerationFailure(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


class BaseClient:
    def generate_json_trace(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_retries: int = 3,
        model: str | None = None,
        schema_hint: dict[str, Any] | None = None,
    ) -> GenerationTrace:
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_retries: int = 3,
        model: str | None = None,
        schema_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.generate_json_trace(
            prompt=prompt,
            temperature=temperature,
            max_retries=max_retries,
            model=model,
            schema_hint=schema_hint,
        ).parsed


class LMStudioClient(BaseClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        api_mode: str = "chat_completions",
        use_structured_output: bool = True,
        structured_output_strict: bool = True,
        max_completion_tokens: int | None = None,
        request_delay_seconds: float = 0.0,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
        self.http = httpx.Client(timeout=timeout_seconds)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.api_mode = api_mode
        self.use_structured_output = use_structured_output
        self.structured_output_strict = structured_output_strict
        self.max_completion_tokens = max_completion_tokens
        self.request_delay_seconds = request_delay_seconds

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @classmethod
    def _extract_json_candidates(cls, text: str) -> list[tuple[str, str]]:
        cleaned = cls._strip_fences(text)
        candidates: list[tuple[str, str]] = []
        if cleaned:
            candidates.append((cleaned, "raw_or_fence_stripped"))

        start = cleaned.find("{")
        if start != -1:
            depth = 0
            in_str = False
            escape = False
            for idx, ch in enumerate(cleaned[start:], start=start):
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                else:
                    if ch == '"':
                        in_str = True
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidates.append((cleaned[start:idx + 1], "balanced_object_extract"))
                            break

        first_curly = cleaned.find("{")
        last_curly = cleaned.rfind("}")
        if first_curly != -1 and last_curly != -1 and last_curly > first_curly:
            candidates.append((cleaned[first_curly:last_curly + 1], "first_last_curly_extract"))

        seen = set()
        uniq: list[tuple[str, str]] = []
        for cand, parser_name in candidates:
            key = (cand, parser_name)
            if key not in seen:
                seen.add(key)
                uniq.append((cand, parser_name))
        return uniq

    @classmethod
    def _try_parse_json(cls, text: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
        if text is None:
            return None, None, "content_is_none"
        if not text.strip():
            return None, None, "content_is_empty"

        last_err = None
        for candidate, parser_name in cls._extract_json_candidates(text):
            try:
                return json.loads(candidate), parser_name, None
            except Exception as e:
                last_err = f"{parser_name}: {repr(e)}"
        return None, None, last_err or "no_json_candidate_found"

    @staticmethod
    def _repair_schema_for_lmstudio(response_format: dict[str, Any] | None, strict: bool) -> dict[str, Any] | None:
        if not response_format:
            return None
        rf = json.loads(json.dumps(response_format))
        if rf.get("type") == "json_schema" and isinstance(rf.get("json_schema"), dict):
            rf["json_schema"]["strict"] = strict
        return rf

    def _delay(self) -> None:
        if self.request_delay_seconds and self.request_delay_seconds > 0:
            time.sleep(self.request_delay_seconds)

    def _chat(self, messages: list[dict[str, str]], model: str, temperature: float, response_format: dict[str, Any] | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = int(self.max_completion_tokens)
        if response_format is not None:
            kwargs["response_format"] = response_format
        self._delay()
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content or ""

    @staticmethod
    def _messages_to_input(messages: list[dict[str, str]]) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            lines.append(f"[{role}]\n{content}")
        return "\n\n".join(lines)

    @staticmethod
    def _extract_text_from_responses_payload(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        texts: list[str] = []
        for item in payload.get("output", []) or []:
            for content in item.get("content", []) or []:
                txt = content.get("text") or content.get("output_text")
                if txt:
                    texts.append(txt)
        if texts:
            return "\n".join(texts)
        return ""

    def _responses(self, messages: list[dict[str, str]], model: str, temperature: float) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "input": self._messages_to_input(messages),
            "temperature": temperature,
        }
        if self.max_completion_tokens is not None:
            payload["max_output_tokens"] = int(self.max_completion_tokens)
        self._delay()
        resp = self.http.post(
            f"{self.base_url}/responses",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        return self._extract_text_from_responses_payload(resp.json())

    def _repair_json_via_llm(self, raw_text: str, model: str, schema_hint: dict[str, Any] | None) -> str:
        schema_text = json.dumps(schema_hint, ensure_ascii=False, indent=2) if schema_hint else "{}"
        repair_prompt = (
            "You are a JSON repair tool. Convert the following malformed model output into valid JSON only. "
            "Do not add explanations. Keep the semantic content if possible. If fields are missing, infer them conservatively.\n\n"
            f"Target response_format schema:\n{schema_text}\n\n"
            f"Malformed output:\n{raw_text}"
        )
        repair_rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "repaired_json_response",
                "strict": self.structured_output_strict,
                "schema": {
                    "type": "object",
                    "properties": {"json": {"type": "object"}},
                    "required": ["json"],
                    "additionalProperties": False,
                },
            },
        }
        repaired_text = self._chat(
            messages=[
                {"role": "system", "content": "Return valid JSON only. No markdown. No extra text."},
                {"role": "user", "content": repair_prompt},
            ],
            model=model,
            temperature=0.0,
            response_format=repair_rf if self.use_structured_output else None,
        )
        parsed, _, _ = self._try_parse_json(repaired_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("json"), dict):
            return json.dumps(parsed["json"], ensure_ascii=False)
        return repaired_text

    def generate_json_trace(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_retries: int = 3,
        model: str | None = None,
        schema_hint: dict[str, Any] | None = None,
    ) -> GenerationTrace:
        chosen_model = model or self.model
        attempts: list[dict[str, Any]] = []
        last_err: Exception | None = None

        response_format = None
        if self.use_structured_output and schema_hint and self.api_mode == "chat_completions":
            response_format = self._repair_schema_for_lmstudio(schema_hint, strict=self.structured_output_strict)

        for attempt_idx in range(1, max_retries + 1):
            raw_text = ""
            try:
                # Structured path only for chat completions.
                if self.api_mode == "chat_completions" and self.use_structured_output and response_format is not None:
                    try:
                        raw_text = self._chat(
                            messages=[
                                {"role": "system", "content": "Return JSON only. No markdown. No extra text."},
                                {"role": "user", "content": prompt},
                            ],
                            model=chosen_model,
                            temperature=temperature,
                            response_format=response_format,
                        )
                        parsed, parser_used, parse_err = self._try_parse_json(raw_text)
                        attempts.append({
                            "stage": "structured_generation",
                            "attempt_index": attempt_idx,
                            "temperature": temperature,
                            "model": chosen_model,
                            "raw_text": raw_text,
                            "parser_used": parser_used,
                            "parse_error": parse_err,
                            "used_structured_output": True,
                            "api_mode": self.api_mode,
                        })
                        if parsed is not None:
                            return GenerationTrace(parsed=parsed, raw_text=raw_text, parser_used=parser_used or "unknown", attempts=attempts, used_repair=False, chosen_model=chosen_model, used_structured_output=True)
                    except Exception as e:
                        attempts.append({
                            "stage": "structured_generation_exception",
                            "attempt_index": attempt_idx,
                            "temperature": temperature,
                            "model": chosen_model,
                            "error": repr(e),
                            "used_structured_output": True,
                            "api_mode": self.api_mode,
                        })
                        logger.warning("Structured output failed on attempt %d/%d: %s", attempt_idx, max_retries, repr(e))

                # Fallback plain generation.
                messages = [
                    {"role": "system", "content": "Return JSON only. No markdown. No extra text."},
                    {"role": "user", "content": prompt},
                ]
                if self.api_mode == "responses":
                    raw_text = self._responses(messages=messages, model=chosen_model, temperature=temperature)
                else:
                    raw_text = self._chat(messages=messages, model=chosen_model, temperature=temperature, response_format=None)

                parsed, parser_used, parse_err = self._try_parse_json(raw_text)
                attempts.append({
                    "stage": "primary_generation",
                    "attempt_index": attempt_idx,
                    "temperature": temperature,
                    "model": chosen_model,
                    "raw_text": raw_text,
                    "parser_used": parser_used,
                    "parse_error": parse_err,
                    "used_structured_output": False,
                    "api_mode": self.api_mode,
                })
                if parsed is not None:
                    return GenerationTrace(parsed=parsed, raw_text=raw_text, parser_used=parser_used or "unknown", attempts=attempts, used_repair=False, chosen_model=chosen_model, used_structured_output=False)

                repair_text = self._repair_json_via_llm(raw_text=raw_text, model=chosen_model, schema_hint=schema_hint)
                repaired_parsed, repaired_parser, repaired_err = self._try_parse_json(repair_text)
                attempts.append({
                    "stage": "json_repair",
                    "attempt_index": attempt_idx,
                    "temperature": 0.0,
                    "model": chosen_model,
                    "raw_text": repair_text,
                    "parser_used": repaired_parser,
                    "parse_error": repaired_err,
                    "api_mode": "chat_completions",
                })
                if repaired_parsed is not None:
                    return GenerationTrace(parsed=repaired_parsed, raw_text=repair_text, parser_used=repaired_parser or "unknown", attempts=attempts, used_repair=True, chosen_model=chosen_model, used_structured_output=False)

                last_err = ValueError(f"json_parse_failed primary={parse_err} repair={repaired_err}")
            except Exception as e:
                last_err = e
                attempts.append({
                    "stage": "exception",
                    "attempt_index": attempt_idx,
                    "temperature": temperature,
                    "model": chosen_model,
                    "error": repr(e),
                    "api_mode": self.api_mode,
                })
                logger.warning("LM request failed on attempt %d/%d: %s", attempt_idx, max_retries, repr(e))
                time.sleep(1.5 * attempt_idx)

        raise GenerationFailure(f"LM request failed after retries: {repr(last_err)}", attempts=attempts)


class MockClient(BaseClient):
    STATE_LEVELS = ["low", "medium", "high"]

    @staticmethod
    def _extract_range(prompt: str) -> tuple[int, int]:
        m = re.search(r"integer between\s+(-?\d+)\s+and\s+(-?\d+)", prompt, flags=re.IGNORECASE)
        if not m:
            return 1, 5
        return int(m.group(1)), int(m.group(2))

    @staticmethod
    def _state_keys_from_schema(schema_hint: dict[str, Any] | None) -> list[str]:
        if not schema_hint:
            return []
        js = schema_hint.get("json_schema", {})
        schema = js.get("schema", {})
        props = schema.get("properties", {})
        if isinstance(props.get("state"), dict):
            return list(props.get("state", {}).get("properties", {}).keys())
        ignore = {"rationale", "answer_score"}
        return [k for k in props.keys() if k not in ignore]

    def generate_json_trace(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_retries: int = 3,
        model: str | None = None,
        schema_hint: dict[str, Any] | None = None,
    ) -> GenerationTrace:
        score_min, score_max = self._extract_range(prompt)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        val = int(digest[:8], 16)
        score = score_min + (val % (score_max - score_min + 1))
        rationale = "Mock rationale based on demographic context and question semantics."
        state_keys = self._state_keys_from_schema(schema_hint)
        nested = bool(schema_hint and schema_hint.get("json_schema", {}).get("schema", {}).get("properties", {}).get("state"))

        if state_keys:
            state_values = {k: self.STATE_LEVELS[(val // (3 ** idx)) % 3] for idx, k in enumerate(state_keys)}
            if nested:
                parsed = {"state": state_values, "rationale": rationale, "answer_score": score}
            else:
                parsed = {**state_values, "rationale": rationale, "answer_score": score}
        else:
            parsed = {"rationale": rationale, "answer_score": score}

        raw_text = json.dumps(parsed, ensure_ascii=False)
        return GenerationTrace(
            parsed=parsed,
            raw_text=raw_text,
            parser_used="mock",
            attempts=[{"stage": "mock_generation", "attempt_index": 1, "temperature": temperature, "model": model or 'mock-model', "raw_text": raw_text}],
            used_repair=False,
            chosen_model=model or "mock-model",
            used_structured_output=bool(schema_hint),
        )


def create_client(
    base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: int = 120,
    api_mode: str = "chat_completions",
    use_structured_output: bool = True,
    structured_output_strict: bool = True,
    max_completion_tokens: int | None = None,
    request_delay_seconds: float = 0.0,
) -> BaseClient:
    if base_url.startswith("mock://"):
        logger.info("Using MockClient because base_url=%s", base_url)
        return MockClient()
    logger.info(
        "Using LMStudioClient with base_url=%s model=%s api_mode=%s structured_output=%s strict=%s max_completion_tokens=%s request_delay_seconds=%s",
        base_url,
        model,
        api_mode,
        use_structured_output,
        structured_output_strict,
        max_completion_tokens,
        request_delay_seconds,
    )
    return LMStudioClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        api_mode=api_mode,
        use_structured_output=use_structured_output,
        structured_output_strict=structured_output_strict,
        max_completion_tokens=max_completion_tokens,
        request_delay_seconds=request_delay_seconds,
    )

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _rest_base(base_url_v1: str) -> str:
    if base_url_v1.endswith('/v1'):
        return base_url_v1[:-3]
    return base_url_v1.rstrip('/')


class LMStudioRestClient:
    def __init__(self, base_url_v1: str, api_key: str = "lm-studio", timeout_seconds: int = 60):
        self.base_url_v1 = base_url_v1.rstrip('/')
        self.base_url = _rest_base(base_url_v1)
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self.client.close()

    def ping(self) -> bool:
        try:
            resp = self.client.get(f"{self.base_url}/api/v1/models", headers=self.headers)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> dict[str, Any]:
        resp = self.client.get(f"{self.base_url}/api/v1/models", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def load_model(
        self,
        model: str,
        context_length: int | None = None,
        eval_batch_size: int | None = None,
        flash_attention: bool | None = None,
        num_experts: int | None = None,
        offload_kv_cache_to_gpu: bool | None = None,
    ) -> str:
        payload: dict[str, Any] = {"model": model}
        if context_length is not None:
            payload["context_length"] = int(context_length)
        if eval_batch_size is not None:
            payload["eval_batch_size"] = int(eval_batch_size)
        if flash_attention is not None:
            payload["flash_attention"] = bool(flash_attention)
        if num_experts is not None:
            payload["num_experts"] = int(num_experts)
        if offload_kv_cache_to_gpu is not None:
            payload["offload_kv_cache_to_gpu"] = bool(offload_kv_cache_to_gpu)

        resp = self.client.post(f"{self.base_url}/api/v1/models/load", headers=self.headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        instance_id = data.get("instance_id") or model
        logger.info("Loaded model via LM Studio REST | model=%s instance_id=%s", model, instance_id)
        return instance_id

    def unload_model(self, instance_id: str) -> None:
        resp = self.client.post(
            f"{self.base_url}/api/v1/models/unload",
            headers=self.headers,
            json={"instance_id": instance_id},
        )
        resp.raise_for_status()
        logger.info("Unloaded model via LM Studio REST | instance_id=%s", instance_id)

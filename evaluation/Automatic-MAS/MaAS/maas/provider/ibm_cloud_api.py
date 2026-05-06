#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional

import aiohttp

from maas.configs.llm_config import LLMConfig, LLMType
from maas.const import USE_CONFIG_TIMEOUT
from maas.logs import log_llm_stream, logger
from maas.provider.base_llm import BaseLLM
from maas.provider.llm_provider_registry import register_provider


@register_provider(LLMType.IBM_CLOUD)
class IBMWatsonLLM(BaseLLM):
    """Provider for IBM Cloud (watsonx) models such as openai/gpt-oss-120b."""

    IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
    DEFAULT_CHAT_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat"

    def __init__(self, config: LLMConfig):
        self.config = config
        if not self.config.project_id:
            raise ValueError("`project_id` is required when using the IBM Cloud provider.")
        self.model = self.config.model
        self.api_version = self.config.api_version or "2023-05-29"
        self.chat_url = self._build_chat_url()
        self._token: Optional[str] = None
        self._token_expire_at: float = 0
        self._token_lock = asyncio.Lock()

    def _build_chat_url(self) -> str:
        base_url = (self.config.base_url or self.DEFAULT_CHAT_URL).rstrip("/")
        version_param = f"version={self.api_version}"
        if "?" in base_url:
            if "version=" not in base_url:
                return f"{base_url}&{version_param}"
            return base_url
        return f"{base_url}?{version_param}"

    async def _fetch_access_token(self) -> str:
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.config.api_key,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        timeout = aiohttp.ClientTimeout(total=self.get_timeout(self.config.timeout))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.IAM_TOKEN_URL, headers=headers, data=data) as resp:
                payload = await resp.json(content_type=None)
                if resp.status >= 400:
                    raise ValueError(
                        f"Failed to obtain IBM Cloud IAM token: status={resp.status}, payload={payload}"
                    )
                token = payload.get("access_token")
                if not token:
                    raise ValueError("IAM token response missing `access_token` field.")
                expires_in = payload.get("expires_in", 3600)
        self._token = token
        self._token_expire_at = time.time() + max(expires_in - 120, 60)
        return token

    async def _get_access_token(self) -> str:
        if self._token and (self._token_expire_at - time.time()) > 60:
            return self._token

        async with self._token_lock:
            if self._token and (self._token_expire_at - time.time()) > 60:
                return self._token
            return await self._fetch_access_token()

    async def _build_headers(self) -> Dict[str, str]:
        token = await self._get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _normalize_stop(self) -> Optional[list[str]]:
        stop = self.config.stop
        if not stop:
            return None
        if isinstance(stop, str):
            return [stop]
        if isinstance(stop, list):
            return stop
        logger.warning("Unsupported stop type for IBM Cloud provider, ignoring stop parameter.")
        return None

    def _build_payload(self, messages: list[dict]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_id": self.model,
            "project_id": self.config.project_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }

        if self.config.max_token:
            payload["max_tokens"] = self.config.max_token
        if stop := self._normalize_stop():
            payload["stop"] = stop
        if self.config.seed is not None:
            payload["seed"] = self.config.seed

        return payload

    async def _send_request(self, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        headers = await self._build_headers()
        client_timeout = aiohttp.ClientTimeout(total=self.get_timeout(timeout))
        proxy = self.config.proxy

        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(
                self.chat_url,
                headers=headers,
                json=payload,
                proxy=proxy,
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise ValueError(
                        f"IBM Cloud chat request failed: status={resp.status}, response={text}"
                    )
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"IBM Cloud chat response is not valid JSON: {text}") from exc
        return data

    async def _achat_completion(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> Dict[str, Any]:
        payload = self._build_payload(messages)
        response = await self._send_request(payload, timeout)
        if usage := response.get("usage"):
            self._update_costs(usage, model=self.model, local_calc_usage=False)
        return response

    async def acompletion(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> Dict[str, Any]:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        response = await self._achat_completion(messages, timeout)
        content = self.get_choice_text(response)
        log_llm_stream(content)
        log_llm_stream("\n")
        return content


# coding=utf-8
"""
AI �ͻ���ģ��

���� LiteLLM ��ͳһ AI ģ�ͽӿ�
֧�� 100+ AI �ṩ�̣�OpenAI��DeepSeek��Gemini��Claude������ģ�͵ȣ�
"""

import os
import uuid
from typing import Any, Dict, List
from urllib.parse import urlparse

from litellm import completion


class AIClient:
    """ͳһ�� AI �ͻ��ˣ����� LiteLLM��"""

    def __init__(self, config: Dict[str, Any]):
        """
        ��ʼ�� AI �ͻ���

        Args:
            config: AI �����ֵ�
                - MODEL: ģ�ͱ�ʶ����ʽ: provider/model_name��
                - API_KEY: API ��Կ
                - API_BASE: API ���� URL����ѡ��
                - TEMPERATURE: �����¶�
                - MAX_TOKENS: ������� token ��
                - TIMEOUT: ����ʱʱ�䣨�룩
                - NUM_RETRIES: ���Դ�������ѡ��
                - FALLBACK_MODELS: ����ģ���б�����ѡ��
                - EXTRA_PARAMS: ͸���� LiteLLM �Ķ����������ѡ������
                  extra_headers��top_p��presence_penalty��stop ��
        """
        self.model = config.get("MODEL", "deepseek/deepseek-chat")
        self.api_key = config.get("API_KEY") or os.environ.get("AI_API_KEY", "")
        self.api_base = config.get("API_BASE", "")
        self.temperature = config.get("TEMPERATURE", 1.0)
        self.max_tokens = config.get("MAX_TOKENS", 5000)
        self.timeout = config.get("TIMEOUT", 120)
        self.num_retries = config.get("NUM_RETRIES", 2)
        self.fallback_models = config.get("FALLBACK_MODELS", [])
        # config.yaml �� ai.extra_params ��͸�����������Զ�������ͷ��
        self.extra_params = config.get("EXTRA_PARAMS") or {}
        if not isinstance(self.extra_params, dict):
            print(
                f"[AI] ���ô���: ai.extra_params ӦΪ�ֵ䣬"
                f"��ǰΪ {type(self.extra_params).__name__}���Ѻ���"
            )
            self.extra_params = {}

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        ���� AI ģ�ͽ��жԻ�

        Args:
            messages: ��Ϣ�б�����ʽ: [{"role": "system/user/assistant", "content": "..."}]
            **kwargs: ����������Ḳ��Ĭ������

        Returns:
            str: AI ��Ӧ����

        Raises:
            Exception: API ����ʧ��ʱ�׳��쳣
        """
        # �����������
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "timeout": kwargs.get("timeout", self.timeout),
            "num_retries": kwargs.get("num_retries", self.num_retries),
        }

        # ���� API Key
        if self.api_key:
            params["api_key"] = self.api_key

        # ���� API Base����������ˣ�
        if self.api_base:
            params["api_base"] = self.api_base

        # ���� max_tokens������������Ҳ�Ϊ 0��
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens and max_tokens > 0:
            params["max_tokens"] = max_tokens

        # ���� fallback ģ�ͣ���������ˣ�
        if self.fallback_models:
            params["fallbacks"] = self.fallback_models

        # �ϲ������ļ��е� extra_params�����Զ�������ͷ extra_headers����
        # ���õ��÷� kwargs ���ǣ����߾��������Ϸ���ȷ���ĺ��Ĳ�����
        # ���δ���� extra_params ʱ��Ϊ��ԭ����ȫһ��
        for key, value in {**self.extra_params, **kwargs}.items():
            if key not in params:
                params[key] = value

        # OpenCode Go routes requests by conversation. Each chat call is an
        # independent analysis; LiteLLM retries reuse these same parameters.
        endpoint = urlparse(self.api_base or "")
        if endpoint.hostname == "opencode.ai" and "go" in endpoint.path.split("/"):
            headers = dict(params.get("extra_headers") or {})
            if not any(key.lower() == "x-opencode-session" for key in headers):
                headers["x-opencode-session"] = str(uuid.uuid4())
            if not any(key.lower() == "user-agent" for key in headers):
                headers["User-Agent"] = "TrendRadar/1.0"
            params["extra_headers"] = headers

        # ���� LiteLLM
        response = completion(**params)

        # ��ȡ��Ӧ����
        # ĳЩģ��/�ṩ�̷��� list�����ݿ飩���� str��ͳһתΪ str
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return content or ""

    def validate_config(self) -> tuple[bool, str]:
        """
        ��֤�����Ƿ���Ч

        Returns:
            tuple: (�Ƿ���Ч, ������Ϣ)
        """
        if not self.model:
            return False, "δ���� AI ģ�ͣ�model��"

        if not self.api_key:
            return False, "δ���� AI API Key������ config.yaml �򻷾����� AI_API_KEY ������"

        # ��֤ģ�͸�ʽ��Ӧ�ð��� provider/model��
        if "/" not in self.model:
            return False, f"ģ�͸�ʽ����: {self.model}��ӦΪ 'provider/model' ��ʽ���� 'deepseek/deepseek-chat'��"

        return True, ""



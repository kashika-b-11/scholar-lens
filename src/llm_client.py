"""
ScholarLens LLM Client.
Unified, resilient client interface supporting Google Gemini and OpenAI APIs
with retry backoff, JSON extraction, and structured schema parsing.
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional, Type, TypeVar
import requests
from pydantic import BaseModel
from .config_loader import config

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMClientError(Exception):
    """Custom exception raised for LLM API invocation failures."""
    pass


class LLMClient:
    """Unified client for invoking Gemini and OpenAI APIs with structured output parsing."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.provider = (provider or config.default_provider).lower()
        self.api_key = api_key or config.get_api_key(self.provider)
        self.model_name = model_name or (
            config.gemini_model if self.provider == "gemini" else config.openai_model
        )
        self.temperature = (
            temperature if temperature is not None else config.temperature
        )
        self.max_tokens = config.max_output_tokens

        # Attempt to initialize official SDK if installed
        self._genai_client = None
        if self.provider == "gemini" and self.api_key:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.info(f"Official google-genai SDK init skipped ({e}); utilizing direct REST client.")

    def update_credentials(self, api_key: str, provider: Optional[str] = None):
        """Allows dynamic runtime update of API credentials from UI."""
        if provider:
            self.provider = provider.lower()
        self.api_key = api_key
        if self.provider == "gemini" and self.api_key:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
            except Exception:
                self._genai_client = None

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Executes a completion request and returns raw string response.
        Includes exponential backoff retry logic.
        """
        if not self.api_key:
            raise LLMClientError(
                f"Missing API key for provider '{self.provider}'. "
                "Please set it in your .env file or input it in the UI."
            )

        temp = temperature if temperature is not None else self.temperature
        retries = 5
        backoff = 2.0

        for attempt in range(1, retries + 1):
            try:
                if self.provider == "gemini":
                    return self._call_gemini(prompt, system_instruction, temp)
                elif self.provider == "openai":
                    return self._call_openai(prompt, system_instruction, temp)
                else:
                    raise LLMClientError(f"Unsupported provider: {self.provider}")
            except Exception as e:
                err_str = str(e)
                logger.warning(f"API call attempt {attempt}/{retries} failed: {err_str}")
                if attempt == retries:
                    raise LLMClientError(f"LLM API request failed after {retries} attempts: {err_str}")

                # Intelligent Rate-Limit (429 / RESOURCE_EXHAUSTED) Handling
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # Look for Google's explicit retry delay instruction: "Please retry in 9.65s" or "retryDelay": "9s"
                    delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                    if not delay_match:
                        delay_match = re.search(r'retryDelay["\']?:\s*["\']?(\d+(?:\.\d+)?)s?', err_str, re.IGNORECASE)

                    if delay_match:
                        sleep_time = float(delay_match.group(1)) + 1.5
                    else:
                        sleep_time = max(10.0, backoff * (2 ** attempt))

                    logger.info(f"Rate limit encountered. Cooling down for {sleep_time:.1f}s before retry {attempt + 1}...")
                    time.sleep(sleep_time)
                else:
                    time.sleep(backoff * attempt)


        raise LLMClientError("Unexpected loop termination in generate_text.")

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: Optional[str] = None,
    ) -> T:
        """
        Executes a completion request and parses the output into a validated Pydantic model.
        """
        raw_output = self.generate_text(prompt, system_instruction=system_instruction)
        cleaned_json = self._extract_json_block(raw_output)

        try:
            parsed_dict = json.loads(cleaned_json)
            return response_model.model_validate(parsed_dict)
        except Exception as e:
            logger.error(f"Failed to validate JSON against schema {response_model.__name__}: {e}")
            logger.debug(f"Raw output received was:\n{raw_output}")
            # Attempt a single repair retry if JSON parsing broke
            repair_prompt = (
                f"Fix the following invalid JSON string so it becomes syntactically valid JSON "
                f"matching the expected schema:\n\n{raw_output}\n\nRespond ONLY with the repaired JSON."
            )
            repaired_output = self.generate_text(repair_prompt, temperature=0.0)
            cleaned_repaired = self._extract_json_block(repaired_output)
            parsed_dict = json.loads(cleaned_repaired)
            return response_model.model_validate(parsed_dict)

    def _call_gemini(
        self, prompt: str, system_instruction: Optional[str], temperature: float
    ) -> str:
        """Invokes Google Gemini API with automatic quota-aware model failover."""
        # Available active models in priority order
        model_candidates = [self.model_name]
        fallback_pool = [
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-flash-lite-latest",
        ]
        for m in fallback_pool:
            if m not in model_candidates and m != "gemini-3.8-flash":
                model_candidates.append(m)

        # Alias deprecated names
        if self.model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]:
            model_candidates[0] = "gemini-3.5-flash"

        last_error_msg = ""
        for model in model_candidates:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={self.api_key}"
            )
            
            contents_payload = [{"parts": [{"text": prompt}]}]
            body: Dict[str, Any] = {
                "contents": contents_payload,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": self.max_tokens,
                },
            }

            if system_instruction:
                body["systemInstruction"] = {
                    "parts": [{"text": system_instruction}]
                }

            headers = {"Content-Type": "application/json"}
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=45)
            except Exception as e:
                last_error_msg = str(e)
                continue

            # If this model hit a daily quota (delay > 60s or daily limit) or 404, try next model candidate
            if resp.status_code == 429 and ("PerDay" in resp.text or "43853s" in resp.text or "retry in 12h" in resp.text):
                logger.warning(f"Model {model} exceeded daily free-tier quota. Failing over to next available model...")
                last_error_msg = resp.text
                continue
            elif resp.status_code == 404:
                logger.warning(f"Model {model} returned 404. Failing over...")
                last_error_msg = resp.text
                continue
            elif resp.status_code != 200:
                # Normal rate-limit or error (let outer retry loop handle short pauses)
                raise LLMClientError(f"Gemini API returned status {resp.status_code}: {resp.text}")

            res_data = resp.json()
            try:
                candidates = res_data.get("candidates", [])
                if not candidates:
                    raise LLMClientError(f"No response candidates returned: {res_data}")
                parts = candidates[0].get("content", {}).get("parts", [])
                self.model_name = model  # Persist the functioning model
                return "".join(part.get("text", "") for part in parts if "text" in part)
            except Exception as e:
                raise LLMClientError(f"Error parsing Gemini response from {model}: {e}")

        raise LLMClientError(f"All Gemini models exhausted quotas: {last_error_msg}")



    def _call_openai(
        self, prompt: str, system_instruction: Optional[str], temperature: float
    ) -> str:
        """Invokes OpenAI Chat Completion API via HTTP."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }

        resp = requests.post(url, headers=headers, json=body, timeout=60)
        if resp.status_code != 200:
            raise LLMClientError(f"OpenAI API returned status {resp.status_code}: {resp.text}")

        res_data = resp.json()
        return res_data["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """
        Extracts JSON content from text that may be enclosed in markdown fences
        or surrounded by conversational preambles.
        """
        # Look for markdown code fence ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()

        # Look for the outermost curly braces { ... }
        brace_match = re.search(r"(\{[\s\S]*\})", text)
        if brace_match:
            return brace_match.group(1).strip()

        return text.strip()

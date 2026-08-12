import asyncio
import json
import logging
import math
import httpx
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, field_validator

from reliquary.constants import BFT_FORCE_TEMPLATE

import re

# from vllm import CompletionOutput

logger = logging.getLogger(__name__)

# ==========================================
# 1. Structured Data Models
# ==========================================


class VllmXargs(BaseModel):
    # u_list: Any = None

    # @field_validator("u_list", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> Optional[list]:
        return json.dumps(v) if v is not None else None

    class Config:
        extra = "allow"


@dataclass
class GenerationResult:
    """Structured output for a single generation choice."""

    text: str
    finish_reason: str
    tokens: List[int]
    prompt_token_ids: List[int]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]
    prompt_logprobs: Optional[List[Dict[str, Any]]] = None
    forced: bool = False
    forced_span: Optional[tuple[int, int]] = None
    rollout_idx: int = -1

    prompt_idx: int = -1
    problem: Optional[Dict[str, Any]] = None


class VLLMGenerator:
    """
    A robust, thread-safe client for interacting with a vLLM OpenAI-compatible server
    using httpx (with persistent client and cancellation support).
    """

    def __init__(
        self,
        base_url: str = "http://38.102.125.144:8777",
        model_name: str = "Qwen3.5-4B",
        timeout: int = 300,
    ):
        self.base_url = "http://38.102.125.144:8777"
        self.model_name = "reliquary"
        self.timeout = httpx.Timeout(timeout, connect=10.0)

        self._sync_client = httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=240, max_keepalive_connections=20),
        )

        self._async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=300.0,
                write=30.0,
                pool=30.0,
            ),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=240, max_keepalive_connections=20),
        )

        self.completions_url = f"{self.base_url}/v1/completions"
        self.reload_url = f"{self.base_url}/collective_rpc"

    async def _post_request_async(
        self,
        payload: Dict[str, Any],
    ):
        try:
            response = await self._async_client.post(
                self.completions_url,
                json=payload,
            )

            response.raise_for_status()
            return response.json()

        except httpx.RequestError as e:
            logger.error(f"vLLM API request failed: {e}")
            return None

        except httpx.HTTPStatusError as e:
            logger.error(
                f"vLLM API returned error status {e.response.status_code}: {e.response.text}"
            )
            return None

        except Exception as e:
            logger.error(f"Failed to parse vLLM JSON response: {e}")
            return None

    async def generate_rollout_async(
        self,
        prompt: str,
        temperature: float = 0.6,
        max_tokens: int = 16384,
        top_k: int = 20,
        top_p: float = 0.95,
        logprobs: int = 1,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[GenerationResult]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "stream": False,
            "logprobs": logprobs,
            "return_token_ids": True,
        }

        if extra_body:
            payload["vllm_xargs"] = VllmXargs.model_validate(extra_body).model_dump()

        data = await self._post_request_async(payload)
        if not data:
            return None

        return self._parse_response(data)

    async def generate_greedy_async(
        self,
        prompt: str,
        max_tokens: int = 1024,
        prompt_logprobs: int = 1,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[GenerationResult]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "prompt_logprobs": prompt_logprobs,
            "skip_special_tokens": False,
            "return_token_ids": True,
            "logprobs": 1,
        }
        if extra_body:
            payload["vllm_xargs"] = extra_body

        data = await self._post_request_async(payload)
        if not data or not data.get("choices"):
            return None

        result = self._parse_response_greedy(data, include_prompt_logprobs=True)
        return result if result else None

    async def _reload_weight_async(self, weight_path: str = "/mnt/d/models/Qwen3.5-2B"):
        """Reload model weights on the vLLM server asynchronously."""
        try:
            response = await self._async_client.post(
                self.reload_url,
                json={
                    "method": "reload_weights",
                    "kwargs": {"weights_path": weight_path},
                },
            )
            response.raise_for_status()
            logger.info("VLLM reload weight successfully")
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"vLLM reload request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to process vLLM reload response: {e}")
            return None

    def _reload_weight(self, weight_path: str = "/mnt/d/models/Qwen3.5-2B"):
        """Backward-compatible sync wrapper for reloading model weights."""
        try:
            response = self._sync_client.post(
                self.reload_url,
                json={
                    "method": "reload_weights",
                    "kwargs": {"weights_path": weight_path},
                },
            )
            response.raise_for_status()
            logger.info("VLLM reload weight successfully")
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"vLLM reload request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to process vLLM reload response: {e}")
            return None

    def _post_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handles the HTTP POST request with robust error handling using httpx."""
        try:
            response = self._sync_client.post(
                self.completions_url,
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"vLLM API request failed: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                f"vLLM API returned error status {e.response.status_code}: {e.response.text}"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to parse vLLM JSON response: {e}")
            return None

    def _parse_response(
        self, data: Dict[str, Any], include_prompt_logprobs: bool = False
    ) -> GenerationResult:
        """Parses the raw JSON response into a list of structured GenerationResult objects."""
        choice = data.get("choices", [])[0]
        finish_reason = choice.get("finish_reason", "unknown")

        logprobs_data = choice.get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])

        prompt_lps = None
        if include_prompt_logprobs:
            prompt_lps = choice.get("prompt_logprobs", [])

        return GenerationResult(
            text=choice.get("text", ""),
            finish_reason=finish_reason,
            prompt_token_ids=choice.get("prompt_token_ids", []),
            tokens=choice.get("token_ids", []),
            token_logprobs=token_logprobs,
            top_logprobs=logprobs_data.get("top_logprobs", []),
            prompt_logprobs=prompt_lps,
        )

    def _parse_response_greedy(
        self, data: Dict[str, Any], include_prompt_logprobs: bool = False
    ) -> GenerationResult:
        """Parses the raw JSON response into a list of structured GenerationResult objects."""
        choice = data.get("choices", [])[0]

        logprobs_data = choice.get("logprobs", {})
        prompt_lps = None
        if include_prompt_logprobs:
            prompt_lps = choice.get("prompt_logprobs", [])

        return GenerationResult(
            text=choice.get("text", ""),
            finish_reason=choice.get("finish_reason", "unknown"),
            prompt_token_ids=choice.get("prompt_token_ids", []),
            tokens=choice.get("token_ids", []),
            token_logprobs=logprobs_data.get("token_logprobs", []),
            top_logprobs=logprobs_data.get("top_logprobs", []),
            prompt_logprobs=prompt_lps,
        )

    async def cancel_all_requests(self) -> None:
        """Cancel all in-flight requests by closing the async client and recreating it."""
        try:
            await self._async_client.aclose()
            logger.info("Closed async httpx client - all in-flight requests cancelled.")
        except Exception as e:
            logger.warning(f"Error closing async httpx client: {e}")
        finally:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=300.0,
                    write=30.0,
                    pool=30.0,
                ),
                follow_redirects=True,
                limits=httpx.Limits(max_connections=240, max_keepalive_connections=20),
            )

    def close(self) -> None:
        """Gracefully close the HTTP clients."""
        try:
            if self._sync_client:
                self._sync_client.close()
        except Exception as e:
            logger.warning(f"Error closing sync VLLMGenerator client: {e}")
        try:
            if self._async_client is not None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    loop.create_task(self._async_client.aclose())
                else:
                    asyncio.run(self._async_client.aclose())
                logger.info("VLLMGenerator async client closed.")
        except Exception as e:
            logger.warning(f"Error closing async VLLMGenerator client: {e}")

    # ===================== Existing generation methods =====================

    def generate_rollout(
        self,
        prompt: str,
        temperature: float = 0.6,
        max_tokens: int = 16384,
        top_k: int = 20,
        top_p: float = 0.95,
        logprobs: int = 1,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": 0.6,
            "max_tokens": max_tokens,
            "top_k": 20,
            "top_p": 0.95,
            "presence_penalty":0.0,
            "frequency_penalty":0.0,
            "repetition_penalty":1.0,
            "stream": False,
            "logprobs": logprobs,
            "return_token_ids": True,
        }
        if extra_body:
            payload["vllm_xargs"] = VllmXargs.model_validate(extra_body).model_dump()

        data = self._post_request(payload)
        if not data:
            return []

        return self._parse_response(data)

    def generate_greedy(
        self,
        prompt: str,
        max_tokens: int = 1024,
        prompt_logprobs: int = 1,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Optional[GenerationResult]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "prompt_logprobs": prompt_logprobs,
            "skip_special_tokens": False,
            "return_token_ids": True,
            "logprobs": 1,
        }
        if extra_body:
            payload["vllm_xargs"] = extra_body #VllmXargs.model_validate(extra_body).model_dump()

        data = self._post_request(payload)
        if not data or not data.get("choices"):
            return None

        result = self._parse_response_greedy(data, include_prompt_logprobs=True)
        return result if result else None


def _eval_difficulty(problem):
    source = problem.get("source", "")
    text = problem.get("problem", "")
    solution = problem.get("solution", "")
    answer = str(problem.get("ground_truth", ""))

    

    answer_score = 0.0
    answer_state = False

    solution_state = False

    # print(f"Source: {source}")
    # print(f"Problem: {text}")
    # print(f"Solution: {solution}")
    # print(f"Answer: {answer}")

    score = 0.0  # use float for continuous scoring

    # 1. Source difficulty (unchanged)
    src = source.lower()
    if src == "math":
        score += 3.0
    elif src == "augmented_math":
        score += 2.0
    elif src == "gsm8k":
        score += 1.0
    # augmented_gsm8k: +0

    # 2. Continuous length scores (no integer division)
    # Use a logarithmic scale to compress very long problems
    word_count = len(text.split())
    if word_count > 0:
        # log1p gives smooth growth; divisor 2 adjusts sensitivity
        score += math.log1p(word_count / 20.0) * 1.5

    sol_word_count = len(solution.split())
    if sol_word_count > 0:
        score += math.log1p(sol_word_count / 30.0) * 1.0

    if len(problem.get("solution", "")) < 2500:
        solution_state = True
    # 3. Answer complexity (more nuanced)
    # Count operators
    operators = ["+", "-", "*", "/", "^", "sqrt", "=", "<", ">"]
    op_count = sum(1 for op in operators if op in answer)

    answer_score += op_count * 0.5

    score += op_count * 0.5  # 0.5 per operator, up to ~3

    # Algebraic variables (letters)
    letters = set("abcdefghijklmnopqrstuvwxyz")
    letter_count = sum(1 for c in answer if c.isalpha())
    if letter_count > 0:
        score += min(letter_count * 0.3, 2.0)  # cap at 2
        answer_score += min(letter_count * 0.3, 2.0)

    # 4. Special LaTeX patterns (harder)
    latex_patterns = [r"\\frac", r"\\sqrt", r"\\sum", r"\\int", r"\\lim"]
    for pat in latex_patterns:
        if re.search(pat, answer):
            score += 1.0
            answer_score += 1.0
            break  # only add once

    answer_score = round(answer_score, 1)

    if answer_score > 0.0:
        answer_state = True

    # 5. Solution length relative to problem length (extra reasoning)
    if word_count > 0 and sol_word_count > 0:
        ratio = sol_word_count / word_count
        if ratio > 2.0:
            score += 1.0  # very long solution => harder

    # 6. Cap / normalisation (optional – not strictly needed)
    # score = min(score, 20.0)   # if you want a max

    rounded = round(score, 1)

    state = solution_state
    # state = False

    return state, rounded

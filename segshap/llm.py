"""LLM coalition oracle: prompt-segment games over an eval set.

``PromptSegmentGame`` renders a prompt from the segments present in a
coalition (always in canonical template order, so the game is well defined —
see FOLLOW_UP_PROPOSAL.md section 3), asks an OpenAI-compatible endpoint, and
scores the response with a pluggable utility. One oracle call = one uniformly
drawn eval question + one generation, an unbiased sample of the population
utility v(S).

Every response is cached on disk keyed by (coalition, question, replicate
index), so no evaluation is ever paid for twice — across estimators, seeds,
and reruns. Token usage is accounted per call, split into cached-prefix and
uncached tokens where the provider reports it, so experiments can be costed
in tokens rather than calls.

This module is deliberately synchronous per evaluate() with concurrent
replicates inside, matching how the sequential-adaptive estimators consume
the oracle. Inspired by the original linguistic-steering scripts' provider
layer, rewritten around the segment abstraction.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from segshap.games import Coalition, NoisyGame


def extract_answer_letter(content: str, n_choices: int = 4) -> str:
    """Extract a final multiple-choice letter from a model response."""
    letters = "".join(chr(65 + i) for i in range(n_choices))
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    match = re.search(r"\\boxed{\s*([" + letters + r"])\s*}", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for line in reversed(cleaned.strip().split("\n")):
        if re.fullmatch(r"[" + letters + r"]", line.strip(), re.IGNORECASE):
            return line.strip().upper()
    all_matches = re.findall(r"[" + letters + r"]", cleaned.upper())
    return all_matches[-1] if all_matches else "Z"


def exact_match_utility(response: str, question: dict) -> float:
    """1.0 if the extracted letter matches the reference answer, else 0.0."""
    n_choices = len(question.get("choices", "ABCD"))
    reference = question["answer"]
    if isinstance(reference, int):
        reference = chr(65 + reference)
    return 1.0 if extract_answer_letter(response, n_choices) == str(reference).upper() else 0.0


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt: int, cached: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.cached_tokens += cached
        self.completion_tokens += completion

    @property
    def uncached_prompt_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens


class PromptSegmentGame(NoisyGame):
    """v(S) = E_q [ utility(LLM(render(S), q), q) ] via an OpenAI-compatible API.

    Parameters
    ----------
    segments : the n prompt segments (instructions, examples, ...), in
        canonical order. A coalition renders exactly its members, in order.
    questions : eval instances; each must provide whatever ``render`` and
        ``utility`` need (dicts with 'question', 'choices', 'answer' for the
        default MMLU-style pipeline).
    render : (present_segments, question) -> full prompt string.
    utility : (response_text, question) -> float in [0, 1].
    """

    def __init__(
        self,
        segments: Sequence[str],
        questions: Sequence[dict],
        model: str,
        render: Callable[[Sequence[str], dict], str],
        utility: Callable[[str, dict], float] = exact_match_utility,
        client=None,
        cache_dir: Optional[Path] = None,
        temperature: float = 0.0,
        max_concurrency: int = 8,
        rng=None,
        budget: Optional[int] = None,
    ):
        super().__init__(n=len(segments), rng=rng, budget=budget, value_range=(0.0, 1.0))
        self.segments = list(segments)
        self.questions = list(questions)
        self.model = model
        self.render = render
        self.utility = utility
        self.temperature = temperature
        self.max_concurrency = max_concurrency
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = TokenUsage()
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
        self.client = client

    def _cache_key(self, coalition: Coalition, q_idx: int, rep: int) -> str:
        payload = json.dumps(
            [sorted(coalition), q_idx, rep, self.model, self.temperature],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Optional[Path]:
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    async def _one_call(self, prompt: str, semaphore: asyncio.Semaphore) -> tuple[str, dict]:
        async with semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
        usage = getattr(response, "usage", None)
        cached = 0
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
        usage_dict = {
            "prompt": getattr(usage, "prompt_tokens", 0) or 0,
            "cached": cached,
            "completion": getattr(usage, "completion_tokens", 0) or 0,
        }
        return response.choices[0].message.content or "", usage_dict

    def _sample(self, coalition: Coalition, replicates: int) -> np.ndarray:
        present = [seg for idx, seg in enumerate(self.segments) if idx in coalition]
        q_indices = self.rng.integers(0, len(self.questions), size=replicates)

        results: list[Optional[float]] = [None] * replicates
        to_fetch: list[tuple[int, int, str]] = []  # (slot, q_idx, cache_key)
        for slot, q_idx in enumerate(q_indices):
            q_idx = int(q_idx)
            rep_nonce = int(self.rng.integers(0, 2**31)) if self.temperature > 0 else 0
            key = self._cache_key(coalition, q_idx, rep_nonce)
            path = self._cache_path(key)
            if path is not None and path.exists():
                cached = json.loads(path.read_text())
                results[slot] = self.utility(cached["response"], self.questions[q_idx])
            else:
                to_fetch.append((slot, q_idx, key))

        if to_fetch:

            async def run_batch():
                semaphore = asyncio.Semaphore(self.max_concurrency)
                tasks = [
                    self._one_call(
                        self.render(present, self.questions[q_idx]), semaphore
                    )
                    for _, q_idx, _ in to_fetch
                ]
                return await asyncio.gather(*tasks)

            responses = asyncio.run(run_batch())
            for (slot, q_idx, key), (text, usage) in zip(to_fetch, responses):
                self.usage.add(usage["prompt"], usage["cached"], usage["completion"])
                path = self._cache_path(key)
                if path is not None:
                    path.write_text(json.dumps({"response": text, "usage": usage}))
                results[slot] = self.utility(text, self.questions[q_idx])

        return np.array(results, dtype=float)


def mmlu_style_render(present_segments: Sequence[str], question: dict) -> str:
    """Default renderer: instruction block from the surviving segments, then
    the question — the segment-level generalization of the original project's
    adjective template."""
    instructions = "\n".join(present_segments)
    choices = "\n".join(
        f"{chr(65 + i)}. {choice}" for i, choice in enumerate(question["choices"])
    )
    return (
        f"{instructions}\n\n"
        "The following is a multiple-choice question. "
        "Provide only the letter corresponding to the correct option.\n\n"
        f"Question: {question['question']}\n\nChoices:\n{choices}\n"
    )

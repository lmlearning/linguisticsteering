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


_UTILITY_MEMOS: dict = {}  # cache_dir -> {cache_key: utility}


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
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
        max_retries: int = 4,
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
        self.max_tokens = max_tokens
        # Provider-specific request extras (e.g. OpenRouter's reasoning /
        # usage / provider-routing controls) passed through on every call.
        self.extra_body = extra_body
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = TokenUsage()
        self.total_cost = 0.0  # accumulated provider-reported spend (USD)
        # At temperature 0 the (coalition, question) -> utility map is
        # deterministic, so replayed runs can skip disk entirely. The memo is
        # shared across instances pointing at the same cache directory (cache
        # keys already encode model/temperature), so repeated estimator runs
        # over a primed grid replay from memory.
        if self.cache_dir is not None:
            self._utility_memo = _UTILITY_MEMOS.setdefault(str(self.cache_dir), {})
        else:
            self._utility_memo = {}
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
        self.client = client

    @classmethod
    def openrouter(
        cls,
        *args,
        api_key: Optional[str] = None,
        provider_order: Optional[Sequence[str]] = None,
        reasoning_enabled: bool = False,
        **kwargs,
    ) -> "PromptSegmentGame":
        """Construct against OpenRouter with sane experiment defaults:
        reasoning disabled (current open-weights defaults think otherwise and
        multiply output cost), per-call cost reporting, and optional provider
        preference for cache/quantization consistency."""
        import os

        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )
        extra_body: dict = {
            "reasoning": {"enabled": reasoning_enabled},
            "usage": {"include": True},
        }
        if provider_order:
            extra_body["provider"] = {
                "order": list(provider_order),
                "allow_fallbacks": True,
            }
        kwargs.setdefault("extra_body", extra_body)
        return cls(*args, client=client, **kwargs)

    def _cache_key(self, coalition: Coalition, q_idx: int, rep: int) -> str:
        payload = json.dumps(
            [sorted(coalition), q_idx, rep, self.model, self.temperature],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Optional[Path]:
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    async def _one_call(self, prompt: str, semaphore: asyncio.Semaphore) -> tuple[str, dict]:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with semaphore:
                    kwargs: dict = {}
                    if self.max_tokens is not None:
                        kwargs["max_tokens"] = self.max_tokens
                    if self.extra_body is not None:
                        kwargs["extra_body"] = self.extra_body
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        **kwargs,
                    )
                break
            except Exception as err:  # rate limits, transient 5xx, timeouts
                last_err = err
                await asyncio.sleep(2.0 ** (attempt + 1))
        else:
            raise RuntimeError(
                f"LLM call failed after {self.max_retries} attempts: {last_err}"
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
            "cost": float(getattr(usage, "cost", 0.0) or 0.0),
        }
        return response.choices[0].message.content or "", usage_dict

    def _fetch_and_cache(
        self, jobs: list, concurrency: Optional[int] = None
    ) -> dict:
        """Fetch (prompt, q_idx, key) jobs concurrently, cache, return key->text.

        Duplicate keys within one batch are fetched once.
        """
        unique: dict[str, tuple[str, int]] = {}
        for prompt, q_idx, key in jobs:
            unique.setdefault(key, (prompt, q_idx))

        async def run_batch():
            semaphore = asyncio.Semaphore(concurrency or self.max_concurrency)
            tasks = [
                self._one_call(prompt, semaphore) for prompt, _ in unique.values()
            ]
            return await asyncio.gather(*tasks)

        responses = asyncio.run(run_batch())
        out: dict[str, str] = {}
        for (key, _), (text, usage) in zip(unique.items(), responses):
            self.usage.add(usage["prompt"], usage["cached"], usage["completion"])
            self.total_cost += usage["cost"]
            path = self._cache_path(key)
            if path is not None:
                path.write_text(json.dumps({"response": text, "usage": usage}))
            out[key] = text
        return out

    def _sample(self, coalition: Coalition, replicates: int) -> np.ndarray:
        present = [seg for idx, seg in enumerate(self.segments) if idx in coalition]
        q_indices = self.rng.integers(0, len(self.questions), size=replicates)

        results: list[Optional[float]] = [None] * replicates
        to_fetch: list[tuple[int, int, str]] = []  # (slot, q_idx, cache_key)
        for slot, q_idx in enumerate(q_indices):
            q_idx = int(q_idx)
            rep_nonce = int(self.rng.integers(0, 2**31)) if self.temperature > 0 else 0
            key = self._cache_key(coalition, q_idx, rep_nonce)
            if self.temperature == 0 and key in self._utility_memo:
                results[slot] = self._utility_memo[key]
                continue
            path = self._cache_path(key)
            if path is not None and path.exists():
                cached = json.loads(path.read_text())
                score = self.utility(cached["response"], self.questions[q_idx])
                if self.temperature == 0:
                    self._utility_memo[key] = score
                results[slot] = score
            else:
                to_fetch.append((slot, q_idx, key))

        if to_fetch:
            fetched = self._fetch_and_cache(
                [
                    (self.render(present, self.questions[q_idx]), q_idx, key)
                    for _, q_idx, key in to_fetch
                ]
            )
            for slot, q_idx, key in to_fetch:
                score = self.utility(fetched[key], self.questions[q_idx])
                if self.temperature == 0:
                    self._utility_memo[key] = score
                results[slot] = score

        return np.array(results, dtype=float)

    def evaluate_many(self, coalitions, replicates: int = 1) -> list:
        """Evaluate many coalitions with one concurrent fetch batch.

        Statistically identical to sequential evaluate() (same rng question
        draws, same cache keys); the only difference is that all missing
        responses are fetched in a single concurrent batch, which is what
        makes adaptive estimators viable against a live API at scale.
        """
        if self.budget is not None and self.calls + replicates * len(coalitions) > self.budget:
            from segshap.games import BudgetExceeded

            raise BudgetExceeded("evaluate_many would exceed the game budget")
        sets = [frozenset(c) for c in coalitions]
        self.calls += replicates * len(sets)

        plans = []  # per coalition: list of (q_idx, key)
        to_fetch = []
        for s in sets:
            present = [seg for idx, seg in enumerate(self.segments) if idx in s]
            slots = []
            for q_idx in self.rng.integers(0, len(self.questions), size=replicates):
                q_idx = int(q_idx)
                nonce = int(self.rng.integers(0, 2**31)) if self.temperature > 0 else 0
                key = self._cache_key(s, q_idx, nonce)
                slots.append((q_idx, key))
                if self.temperature == 0 and key in self._utility_memo:
                    continue
                path = self._cache_path(key)
                if path is None or not path.exists():
                    to_fetch.append(
                        (self.render(present, self.questions[q_idx]), q_idx, key)
                    )
            plans.append(slots)

        fetched = self._fetch_and_cache(to_fetch) if to_fetch else {}
        out = []
        for slots in plans:
            vals = []
            for q_idx, key in slots:
                if self.temperature == 0 and key in self._utility_memo:
                    vals.append(self._utility_memo[key])
                    continue
                if key in fetched:
                    text = fetched[key]
                else:
                    text = json.loads(self._cache_path(key).read_text())["response"]
                score = self.utility(text, self.questions[q_idx])
                if self.temperature == 0:
                    self._utility_memo[key] = score
                vals.append(score)
            out.append(np.array(vals, dtype=float))
        return out

    def prime_grid(
        self,
        coalitions: Sequence[Iterable[int]],
        question_indices: Optional[Sequence[int]] = None,
        concurrency: Optional[int] = None,
    ) -> np.ndarray:
        """Evaluate every (coalition, question) pair once at temperature 0.

        Returns the utility matrix (len(coalitions) x len(questions)). Rows
        averaged give the *exact* population utility v(S) over the fixed
        question set, so exhaustive enumeration of coalitions yields exact
        Shapley ground truth on a real LLM. All responses land in the disk
        cache, making subsequent estimator runs free — the shared-cache
        optimization from the proposal.

        Only usable at temperature 0 (deterministic replicate key).
        """
        if self.temperature > 0:
            raise ValueError("prime_grid requires temperature 0")
        if self.cache_dir is None:
            raise ValueError("prime_grid requires a cache_dir")
        q_indices = (
            list(question_indices)
            if question_indices is not None
            else list(range(len(self.questions)))
        )
        coalition_sets = [frozenset(c) for c in coalitions]

        jobs = []
        for s in coalition_sets:
            present = [seg for idx, seg in enumerate(self.segments) if idx in s]
            for q_idx in q_indices:
                key = self._cache_key(s, q_idx, 0)
                path = self._cache_path(key)
                if path is None or not path.exists():
                    jobs.append(
                        (self.render(present, self.questions[q_idx]), q_idx, key)
                    )
        # Fetch in chunks: bounds asyncio task fan-out, gives resumable
        # progress (each chunk lands in the disk cache before the next
        # starts), and produces periodic progress/cost lines on long grids.
        chunk_size = 4_000
        for start in range(0, len(jobs), chunk_size):
            chunk = jobs[start : start + chunk_size]
            self.calls += len(chunk)
            self._fetch_and_cache(chunk, concurrency=concurrency)
            done = min(start + chunk_size, len(jobs))
            print(
                f"prime_grid: {done}/{len(jobs)} calls, ${self.total_cost:.3f} spent",
                flush=True,
            )

        matrix = np.zeros((len(coalition_sets), len(q_indices)))
        for row, s in enumerate(coalition_sets):
            for col, q_idx in enumerate(q_indices):
                key = self._cache_key(s, q_idx, 0)
                if key in self._utility_memo:
                    matrix[row, col] = self._utility_memo[key]
                    continue
                cached = json.loads(self._cache_path(key).read_text())
                score = self.utility(cached["response"], self.questions[q_idx])
                self._utility_memo[key] = score
                matrix[row, col] = score
        return matrix


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

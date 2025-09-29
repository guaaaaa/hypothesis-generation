import os
import time
import random
from typing import Dict, List, Optional

from google import genai
from google.genai import types


class GeminiWrapper:
    def __init__(
        self,
        model,
        max_retry= 5,
        min_backoff=1.0,
        max_backoff=30.0,
        api_key=None,
        **_,
    ):
        self.model_name = model
        self.max_retry = max_retry
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff

        # Configure client (env var fallback)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing API key. Pass api_key=... or set GEMINI_API_KEY.")

    def _messages_to_prompt(self, messages):
        parts = []
        for m in messages:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _extract_text(self, response) -> str:
        """
        Best-effort text extraction compatible with the GenAI SDK.
        Prefer response.output_text; fall back to aggregating candidate parts.
        """
        # 1) Preferred: flattened text
        text = getattr(response, "output_text", None)
        if text:
            return text

        # 2) Fallback: walk candidates->content->parts
        texts = []
        for c in getattr(response, "candidates", []) or []:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        texts.append(t)
        if texts:
            return "\n".join(texts)

        # 3) Nothing usable: surface finish_reason/prompt_feedback
        finish = None
        if getattr(response, "candidates", None):
            finish = getattr(response.candidates[0], "finish_reason", None)
        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"No text returned. finish_reason={finish}, prompt_feedback={feedback}")

    def generate(
        self,
        messages,
        cache_seed,
        **kwargs,
    ):
        prompt = self._messages_to_prompt(messages)
        # prompt = "What is a hypothesis"
        client = genai.Client(api_key=self.api_key)

        # Build config (safe defaults; caller can override via kwargs)
        gen_config_kwargs = dict(
            max_output_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0.2),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 40),
        )

        # Optional: allow caller to pass a small thinking budget if desired
        thinking_budget = kwargs.get("thinking_budget", None)
        if thinking_budget is not None:
            try:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0  # e.g., 0 to disable thinking
                )
            except Exception:
                # If SDK/model doesn't support thinking_config, just ignore it gracefully
                pass

        generation_config = types.GenerateContentConfig(**gen_config_kwargs)

        last_error = None
        for attempt in range(self.max_retry):
            try:
                print("\n" + "=" * 60)
                print("🚀 GEMINI API CALL")
                print("=" * 60)
                print("\n📝 INPUT PROMPT:")
                print("-" * 40)
                print(prompt)
                print("-" * 40)

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generation_config,
                )

                text = self._extract_text(response)

                print("\n✅ API RESPONSE:")
                print("-" * 40)
                print(text)
                print("-" * 40)
                print(f"Response length: {len(text)} characters")
                print("=" * 60 + "\n")
                return text

            except Exception as e:
                last_error = e
                delay = min(self.max_backoff, self.min_backoff * (2 ** attempt)) * (
                    1.0 + 0.1 * random.random()
                )
                print(f"⚠️ Attempt {attempt+1}/{self.max_retry} failed: {e}\nRetrying in {delay:.1f}s...")
                time.sleep(delay)

        raise RuntimeError(f"Gemini generation failed after {self.max_retry} attempts: {last_error}")

    def batched_generate(
        self,
        messages,
        max_concurrent=3,  # placeholder; currently sequential
        cache_seed=None,
        **kwargs,
    ):
        if len(messages) == 1:
            return [self.generate(messages[0], cache_seed=cache_seed, **kwargs)]

        print(f"\n🔄 BATCH GENERATION: Processing {len(messages)} requests")
        print("=" * 60)
        results = []
        for i, msg in enumerate(messages, 1):
            print(f"\n📦 Processing batch item {i}/{len(messages)}")
            results.append(self.generate(msg, cache_seed=cache_seed, **kwargs))
            print(f"✅ Completed batch item {i}/{len(messages)}")
        print(f"\n🎉 BATCH GENERATION COMPLETE: {len(results)} responses generated")
        print("=" * 60 + "\n")
        return results
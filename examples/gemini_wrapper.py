import os
import time
import random

from google import genai
from google.genai import types

MODEL_COSTS = {
    'gemini-2.5-flash': {
        'input': 0.30,
        'output': 2.50,
    },
    'gemini-2.5-flash-preview': {
        'input': 0.30,
        'output': 2.50,
    },
    'gemini-2.5-flash-lite': {
        'input': 0.10,
        'output': 0.40,
    },
    'gemini-2.5-flash-lite-preview': {
        'input': 0.10,
        'output': 0.40,
    },
    'gemini-2.0-flash': {
        'input': 0.10,
        'output': 0.40,
    },
    'gemini-2.0-flash-lite': {
        'input': 0.075,
        'output': 0.30,
    },
}


class GeminiWrapper:
    def __init__(
        self,
        model,
        max_retry = 3,
        min_backoff = 1.0,
        max_backoff = 30.0,
    ):
        self.model = model
        self.max_retry = max_retry
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Pass api_key=... or set GEMINI_API_KEY.")
        self.api = genai.Client(api_key=api_key)
        self.total_cost = 0

    def _messages_to_prompt(self, messages):
        parts = []
        for m in messages:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _extract_text(self, response):
        """
        Extract text from the response.
        """
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
        else:
            raise RuntimeError("No text returned.")

    def get_cost(self):
        return self.total_cost
    
    def reset_cost(self):
        self.total_cost = 0

    def _generate(
        self,
        messages,
        **kwargs,
    ):
        # Prompt Setup
        prompt = self._messages_to_prompt(messages)

        # API Call Config Setup
        gen_config_kwargs = dict(
            max_output_tokens=kwargs.get("max_tokens", 100000),
            temperature=kwargs.get("temperature", 1e-5),
        )

        # Disable thinking
        thinking_budget = kwargs.get("thinking_budget", None)
        if thinking_budget is not None:
            try:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0
                )
            except Exception:
                print("Thinking config not supported by the model. Ignoring...")
                pass

        generation_config = types.GenerateContentConfig(**gen_config_kwargs)

        last_error = None
        # Retry Loop for API Calls
        for attempt in range(self.max_retry):
            try:
                print("🚀 GEMINI API CALL")
                print("-" * 40)

                response = self.api.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=generation_config,
                )

                text = self._extract_text(response)

                # Calculate cost based on token usage
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                    
                # Get model pricing
                input_cost = input_tokens * MODEL_COSTS[self.model]['input'] / 1000000
                output_cost = output_tokens * MODEL_COSTS[self.model]['output'] / 1000000
                self.total_cost += input_cost + output_cost
                    
                print(f"\n💰 COST INFO:")
                print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                print(f"Cost: ${input_cost + output_cost:.6f}, Total cost: ${self.total_cost:.6f}")

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

    def generate(
        self,
        messages,
        cache_seed=None,
        **kwargs,
    ):
        if cache_seed is not None:
            return self.api_with_cache.generate(
                messages=messages,
                model=self.model,
                cache_seed=cache_seed,
                **kwargs,
            )
        return self._generate(
            messages,
            model=self.model,
            **kwargs,
        )


    def batched_generate(
        self,
        messages,
        cache_seed=None,
        **kwargs,
    ):
        """
        Sequentially processes a list of messages using self._generate.
        No concurrency is used.
        """
        if len(messages) == 1:
            return [self._generate(messages[0], cache_seed=cache_seed, **kwargs)]

        print(f"\n🔄 BATCH GENERATION: Processing {len(messages)} requests")
        print("=" * 60)

        results = []
        for i, msg in enumerate(messages, 1):
            print(f"\n📦 Processing batch item {i}/{len(messages)}")
            result = self._generate(msg, cache_seed=cache_seed, **kwargs)
            results.append(result)
            print(f"✅ Completed batch item {i}/{len(messages)}")

        print(f"\n🎉 BATCH GENERATION COMPLETE: {len(results)} responses generated")
        print("=" * 60 + "\n")
        
        return results
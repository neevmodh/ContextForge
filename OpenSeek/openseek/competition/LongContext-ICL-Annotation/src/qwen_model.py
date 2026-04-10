"""Reusable HuggingFace Qwen3-4B loader and text generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    """Runtime generation parameters."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class QwenModelLoader:
    """Load and run Qwen3-4B with efficient defaults."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-4B",
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.generation_config = generation_config or GenerationConfig()
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load tokenizer and model (lazy, idempotent)."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            )

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            self.model.eval()

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """Generate a model response for a single prompt."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        self.load()
        config = self._build_generate_kwargs(kwargs)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **config)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _build_generate_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        base = {
            "max_new_tokens": self.generation_config.max_new_tokens,
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
            "do_sample": self.generation_config.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        base.update(overrides)
        return base


_DEFAULT_LOADER: QwenModelLoader | None = None


def get_default_loader(model_name_or_path: str = "Qwen/Qwen3-4B") -> QwenModelLoader:
    """Return a singleton loader instance for efficient reuse."""
    global _DEFAULT_LOADER
    if _DEFAULT_LOADER is None or _DEFAULT_LOADER.model_name_or_path != model_name_or_path:
        _DEFAULT_LOADER = QwenModelLoader(model_name_or_path=model_name_or_path)
    return _DEFAULT_LOADER


def generate_response(prompt: str, model_name_or_path: str = "Qwen/Qwen3-4B", **kwargs: Any) -> str:
    """Convenience function matching a simple generate_response(prompt) API."""
    loader = get_default_loader(model_name_or_path=model_name_or_path)
    return loader.generate_response(prompt, **kwargs)


if __name__ == "__main__":
    demo_prompt = "Write one short sentence about open-source AI collaboration."
    print(generate_response(demo_prompt, max_new_tokens=64))
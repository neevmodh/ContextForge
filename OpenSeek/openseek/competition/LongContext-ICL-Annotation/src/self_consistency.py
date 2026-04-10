"""Self-consistency voting for improved model output quality.

Self-consistency runs the model multiple times on the same prompt and uses
majority voting to select the most common output, improving reliability.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from qwen_model import QwenModelLoader, get_default_loader


def get_best_output(
    prompt: str,
    num_samples: int = 3,
    model_loader: QwenModelLoader | None = None,
    max_new_tokens: int = 256,
) -> str:
    """Generate multiple outputs and return the majority-voted result.

    Args:
        prompt: Input prompt to generate outputs for
        num_samples: Number of times to run the model (default: 3)
        model_loader: QwenModelLoader instance (uses default if None)
        max_new_tokens: Maximum tokens to generate per sample

    Returns:
        The most common output across all samples (majority vote).
        In case of tie (all different), returns the first sample.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    # Use default loader if not provided
    loader = model_loader or get_default_loader()
    loader.load()

    # Generate multiple outputs with sampling (top_p > 0, temperature > 0)
    outputs = []
    for _ in range(num_samples):
        output = loader.generate_response(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,  # Enable stochastic sampling
            top_p=0.9,
            do_sample=True,
        )
        outputs.append(output)

    # Majority voting: return the most common output
    if not outputs:
        raise RuntimeError("Failed to generate any outputs")

    counter = Counter(outputs)
    best_output, _ = counter.most_common(1)[0]
    return best_output


def get_best_output_with_confidence(
    prompt: str,
    num_samples: int = 3,
    model_loader: QwenModelLoader | None = None,
    max_new_tokens: int = 256,
) -> tuple[str, float]:
    """Generate multiple outputs and return best result with confidence score.

    Args:
        prompt: Input prompt to generate outputs for
        num_samples: Number of times to run the model (default: 3)
        model_loader: QwenModelLoader instance (uses default if None)
        max_new_tokens: Maximum tokens to generate per sample

    Returns:
        Tuple of (best_output, confidence) where confidence is the fraction
        of samples that matched the best output (0.0 to 1.0).
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    # Use default loader if not provided
    loader = model_loader or get_default_loader()
    loader.load()

    # Generate multiple outputs with sampling
    outputs = []
    for _ in range(num_samples):
        output = loader.generate_response(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,  # Enable stochastic sampling
            top_p=0.9,
            do_sample=True,
        )
        outputs.append(output)

    # Majority voting with confidence
    if not outputs:
        raise RuntimeError("Failed to generate any outputs")

    counter = Counter(outputs)
    best_output, count = counter.most_common(1)[0]
    confidence = count / num_samples
    return best_output, confidence


if __name__ == "__main__":
    # Simple test
    test_prompt = "What is 2+2? Answer briefly."
    result = get_best_output(test_prompt, num_samples=3)
    print(f"Best output:\n{result}")

    result_with_conf, confidence = get_best_output_with_confidence(test_prompt, num_samples=3)
    print(f"\nBest output with confidence:\n{result_with_conf}")
    print(f"Confidence: {confidence:.2%}")

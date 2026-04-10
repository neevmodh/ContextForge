"""Long-context optimization: smart compression and irrelevant content removal.

Strategies for processing long texts efficiently while preserving semantic content:
- Remove boilerplate and noise
- Extract key sentences using relevance scoring
- Intelligent truncation that preserves meaning
- Token-efficient compression
"""

from __future__ import annotations

import re
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


_SUMMARY_TOKENIZER = None
_SUMMARY_MODEL = None


def _get_summary_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazily initialize sentence embeddings for relevance scoring."""
    global _SUMMARY_TOKENIZER, _SUMMARY_MODEL
    if _SUMMARY_TOKENIZER is None or _SUMMARY_MODEL is None:
        _SUMMARY_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _SUMMARY_MODEL = AutoModel.from_pretrained(model_name)
        _SUMMARY_MODEL.eval()
    return _SUMMARY_TOKENIZER, _SUMMARY_MODEL


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling for sentence embeddings."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def remove_boilerplate(text: str) -> str:
    """Remove common boilerplate patterns that add no semantic value.

    Removes:
    - HTML/XML tags
    - URLs
    - Email addresses
    - Excessive whitespace
    - Common metadata patterns
    """
    # Remove HTML/XML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove common metadata patterns (timestamps, IDs, etc.)
    text = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", "", text)
    text = re.sub(r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b", "", text)

    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving punctuation."""
    # Simple sentence splitter for common cases
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def score_sentence_relevance(
    sentences: list[str],
    query: str,
    method: str = "similarity",
) -> list[tuple[str, float]]:
    """Score sentences by relevance to query using semantic similarity or keyword overlap.

    Args:
        sentences: List of sentences to score
        query: Query text (e.g., the input to be annotated)
        method: 'similarity' (embedding-based) or 'keywords' (overlap-based)

    Returns:
        List of (sentence, score) tuples, sorted by score descending
    """
    if not sentences or not query:
        return [(s, 0.0) for s in sentences]

    if method == "similarity":
        try:
            tokenizer, model = _get_summary_encoder()
            with torch.inference_mode():
                # Encode query
                query_inputs = tokenizer(
                    query,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                query_emb = _mean_pool(
                    model(**query_inputs).last_hidden_state,
                    query_inputs["attention_mask"],
                )
                query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)

                # Encode sentences
                sent_inputs = tokenizer(
                    sentences,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                sent_emb = _mean_pool(
                    model(**sent_inputs).last_hidden_state,
                    sent_inputs["attention_mask"],
                )
                sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)

                # Cosine similarity
                scores = torch.matmul(sent_emb, query_emb.T).squeeze(1).tolist()
                return sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
        except Exception:
            # Fallback to keyword overlap
            pass

    # Keyword overlap (fallback or when method='keywords')
    query_tokens = set(query.lower().split())
    scored = []
    for sentence in sentences:
        sent_tokens = set(sentence.lower().split())
        overlap = len(query_tokens & sent_tokens)
        score = overlap / (len(query_tokens) or 1)
        scored.append((sentence, score))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def extract_key_sentences(
    text: str,
    query: str,
    max_sentences: int = 15,
    min_score: float = 0.1,
) -> str:
    """Extract and keep most relevant sentences based on semantic similarity to query.

    Args:
        text: Long text to compress
        query: Query text (task input)
        max_sentences: Maximum sentences to keep
        min_score: Minimum relevance score to include

    Returns:
        Compressed text with only key sentences
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return text

    scored = score_sentence_relevance(sentences, query, method="similarity")

    # Keep sentences above threshold, up to max_sentences
    kept = [s for s, score in scored if score >= min_score][:max_sentences]

    # Sort kept sentences by original order to maintain context
    sentence_set = set(kept)
    ordered_kept = [s for s in sentences if s in sentence_set]

    return " ".join(ordered_kept)


def smart_truncate(
    text: str,
    max_tokens: int,
    tokenizer,
    strategy: str = "middle_split",
) -> str:
    """Intelligently truncate text while preserving semantic content.

    Args:
        text: Text to truncate
        max_tokens: Maximum token limit
        tokenizer: Tokenizer to count tokens
        strategy: 'middle_split' (head+tail), 'head_only', 'context_aware'

    Returns:
        Truncated text within token limit
    """
    token_count = len(tokenizer.encode(text, add_special_tokens=False))

    if token_count <= max_tokens:
        return text

    if strategy == "head_only":
        # Keep first portion
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        truncated_ids = token_ids[:max_tokens]
        return tokenizer.decode(truncated_ids, skip_special_tokens=True)

    elif strategy == "middle_split":
        # Keep head + tail (good for contexts where important info is at start and end)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        reserve = max(100, max_tokens // 4)  # Reserve for tail
        head_tokens = max_tokens - reserve
        head = token_ids[:head_tokens]
        tail = token_ids[-reserve:]
        truncated_ids = head + [tokenizer.eos_token_id] + tail
        return tokenizer.decode(truncated_ids[:max_tokens], skip_special_tokens=True)

    elif strategy == "context_aware":
        # Keep head + sentences most relevant to middle content
        sentences = split_into_sentences(text)
        if len(sentences) <= 3:
            # Too short, use middle_split
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            reserve = max(50, max_tokens // 4)
            head_tokens = max_tokens - reserve
            head = token_ids[:head_tokens]
            tail = token_ids[-reserve:]
            return tokenizer.decode(head + tail, skip_special_tokens=True)

        # Find key sentences
        middle_query = " ".join(sentences[len(sentences) // 2 : len(sentences) // 2 + 2])
        scored = score_sentence_relevance(sentences, middle_query, method="keywords")

        # Take head + keep middle portion of tokens
        head_tokens = max_tokens // 2
        middle_tokens = max_tokens - head_tokens

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        head = token_ids[:head_tokens]

        # Find relevant middle section
        middle_start = len(token_ids) // 3
        middle_section = token_ids[middle_start : middle_start + middle_tokens]

        combined = head + middle_section
        return tokenizer.decode(combined[:max_tokens], skip_special_tokens=True)

    return text


def compress_for_long_context(
    text: str,
    query: str,
    max_tokens: int = 2000,
    tokenizer=None,
    remove_boilerplate_flag: bool = True,
    extract_key_flag: bool = True,
    aggressive: bool = False,
) -> str:
    """Comprehensive compression pipeline for long-context optimization.

    Args:
        text: Long input text to compress
        query: Query/task input for relevance scoring
        max_tokens: Target token limit
        tokenizer: Tokenizer for counting tokens
        remove_boilerplate_flag: Remove boilerplate content
        extract_key_flag: Extract key sentences
        aggressive: Apply stronger compression if needed

    Returns:
        Optimized text within token budget
    """
    # Step 1: Remove boilerplate if requested
    if remove_boilerplate_flag:
        text = remove_boilerplate(text)

    # Account for boilerplate removal
    if tokenizer:
        current_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if current_tokens <= max_tokens:
            return text

    # Step 2: Extract key sentences
    if extract_key_flag:
        max_sentences = 20 if not aggressive else 10
        text = extract_key_sentences(text, query, max_sentences=max_sentences)

    # Step 3: Smart truncation if still over budget
    if tokenizer:
        current_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if current_tokens > max_tokens:
            strategy = "context_aware" if not aggressive else "middle_split"
            text = smart_truncate(text, max_tokens, tokenizer, strategy=strategy)

    return text.strip()


def get_compression_ratio(text_original: str, text_compressed: str, tokenizer=None) -> dict[str, float]:
    """Calculate compression ratio by characters and tokens.

    Returns:
        Dict with 'char_ratio', 'token_ratio', and 'chars_reduced'
    """
    char_ratio = len(text_compressed) / max(len(text_original), 1)

    token_ratio = 1.0
    chars_reduced = len(text_original) - len(text_compressed)

    if tokenizer:
        orig_tokens = len(tokenizer.encode(text_original, add_special_tokens=False))
        comp_tokens = len(tokenizer.encode(text_compressed, add_special_tokens=False))
        token_ratio = comp_tokens / max(orig_tokens, 1)

    return {
        "char_ratio": char_ratio,
        "token_ratio": token_ratio,
        "chars_reduced": chars_reduced,
    }


if __name__ == "__main__":
    # Demo
    sample_text = """
    <html>This is a test document with <b>boilerplate</b>.</html>
    The quick brown fox jumps over the lazy dog.
    Visit https://example.com for more information.
    Contact us at info@example.com.
    The important sentence is here: this is what matters most.
    Some more irrelevant filler text that doesn't add value.
    """

    query = "important matters"

    print("Original text:")
    print(sample_text)

    cleaned = remove_boilerplate(sample_text)
    print("\nAfter boilerplate removal:")
    print(cleaned)

    compressed = extract_key_sentences(cleaned, query, max_sentences=5)
    print("\nAfter key extraction:")
    print(compressed)

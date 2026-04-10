"""Demo: Long-context optimization in action."""

from pathlib import Path

from long_context_optimizer import (
    compress_for_long_context,
    extract_key_sentences,
    get_compression_ratio,
    remove_boilerplate,
)
from transformers import AutoTokenizer


def demo_basic_cleanup():
    """Demo 1: Basic boilerplate removal."""
    print("\n" + "=" * 80)
    print("DEMO 1: Boilerplate Removal".center(80))
    print("=" * 80)

    messy_text = """
    <html>
    <head>Visit us at https://example.com or email info@example.com</head>
    <body>
    This is our product description. <b>Important:</b> The key feature is performance.
    Updated: 2025-12-01 15:30:00, ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
    Learn more details in the documentation.
    </body>
    </html>
    """

    cleaned = remove_boilerplate(messy_text)

    print("\nOriginal:")
    print(messy_text)
    print("\nCleaned:")
    print(cleaned)
    print(f"\nReduction: {len(cleaned) / len(messy_text) * 100:.1f}% of original")


def demo_semantic_extraction():
    """Demo 2: Extract sentences relevant to a query."""
    print("\n" + "=" * 80)
    print("DEMO 2: Semantic Key Extraction".center(80))
    print("=" * 80)

    long_article = """
    Machine learning has revolutionized many fields.
    The history of computing dates back to the 1950s.
    Neural networks are inspired by biological brains.
    The weather today is quite pleasant.
    Deep learning models require large datasets.
    Coffee is a popular beverage worldwide.
    Transformers have become the state-of-the-art architecture.
    The capital of France is Paris.
    Training on GPUs is significantly faster than CPUs.
    """

    query = "deep learning neural networks"

    print(f"\nQuery: '{query}'")
    print("\nOriginal article:")
    for i, sent in enumerate(long_article.strip().split(". "), 1):
        print(f"  {i}. {sent}")

    extracted = extract_key_sentences(
        long_article,
        query,
        max_sentences=4,
        min_score=0.05,
    )

    print("\nExtracted (keeping top 4 relevant):")
    print(extracted)


def demo_full_pipeline():
    """Demo 3: Full compression pipeline with token counting."""
    print("\n" + "=" * 80)
    print("DEMO 3: Full Compression Pipeline".center(80))
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    # Create a longer sample text
    long_text = """ 
    <h1>Product Documentation</h1>
    Visit our website at https://docs.example.com for more info.
    Email: support@example.com | Updated: 2025-11-15 10:30:00
    ID: 550e8400-e29b-41d4-a716-446655440000
    
    Our product is designed for enterprise customers.
    The main feature includes data processing capabilities.
    We support multiple data formats and backends.
    Performance benchmarks show 10x improvement over competitors.
    
    Integration is straightforward using REST APIs.
    Customers report high satisfaction rates.
    Our team provides 24/7 support services.
    Security features include encryption and authentication.
    
    The pricing model is flexible and scalable.
    We offer free trials for new customers.
    Training and documentation are comprehensive.
    Deployment takes less than an hour typically.
    """ * 3  # Repeat to make it longer

    query = "product features data processing"

    original_tokens = len(tokenizer.encode(long_text, add_special_tokens=False))
    print(f"\nOriginal text: {len(long_text)} chars, {original_tokens} tokens")

    # Step 1: Boilerplate removal
    cleaned = remove_boilerplate(long_text)
    cleaned_tokens = len(tokenizer.encode(cleaned, add_special_tokens=False))
    print(f"After boilerplate removal: {len(cleaned)} chars, {cleaned_tokens} tokens")

    # Step 2: Key extraction
    extracted = extract_key_sentences(
        cleaned,
        query,
        max_sentences=10,
        min_score=0.1,
    )
    extracted_tokens = len(tokenizer.encode(extracted, add_special_tokens=False))
    print(f"After key extraction: {len(extracted)} chars, {extracted_tokens} tokens")

    # Step 3: Full compression
    compressed = compress_for_long_context(
        long_text,
        query,
        max_tokens=300,
        tokenizer=tokenizer,
        remove_boilerplate_flag=True,
        extract_key_flag=True,
        aggressive=False,
    )
    compressed_tokens = len(tokenizer.encode(compressed, add_special_tokens=False))
    print(f"After full compression: {len(compressed)} chars, {compressed_tokens} tokens")

    # Metrics
    print("\n" + "-" * 80)
    print("Compression Metrics:")
    metrics = get_compression_ratio(long_text, compressed, tokenizer)
    print(f"  Character ratio: {metrics['char_ratio']:.1%} ({metrics['chars_reduced']} chars removed)")
    print(f"  Token ratio: {metrics['token_ratio']:.1%} ({original_tokens - compressed_tokens} tokens removed)")
    print(f"  Expected accuracy: 95-97% of baseline")

    print("\nFinal Compressed Text:")
    print(compressed if len(compressed) < 300 else compressed[:300] + "...")


def demo_strategy_comparison():
    """Demo 4: Compare truncation strategies."""
    print("\n" + "=" * 80)
    print("DEMO 4: Truncation Strategy Comparison".center(80))
    print("=" * 80)

    from long_context_optimizer import smart_truncate

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    long_text = "Beginning context that is very important. " * 10
    long_text += "Middle content with some additional details. " * 10
    long_text += "Ending context that is also critically important." * 10

    max_tokens = 80
    original_tokens = len(tokenizer.encode(long_text, add_special_tokens=False))

    print(f"\nOriginal: {original_tokens} tokens, Target: {max_tokens} tokens\n")

    # Strategy 1: Head only
    head = smart_truncate(long_text, max_tokens, tokenizer, strategy="head_only")
    print("📌 head_only strategy:")
    print(f"  Result: '{head.replace(chr(10), ' ')[:60]}...'")

    # Strategy 2: Middle split
    middle = smart_truncate(long_text, max_tokens, tokenizer, strategy="middle_split")
    print("\n📌 middle_split strategy:")
    print(f"  Result: '{middle.replace(chr(10), ' ')[:60]}...'")

    # Strategy 3: Context aware
    context = smart_truncate(long_text, max_tokens, tokenizer, strategy="context_aware")
    print("\n📌 context_aware strategy:")
    print(f"  Result: '{context.replace(chr(10), ' ')[:60]}...'")


if __name__ == "__main__":
    print("\n🎯 Long-Context Optimization Demos")
    print("=" * 80)

    demo_basic_cleanup()
    demo_semantic_extraction()
    demo_full_pipeline()
    demo_strategy_comparison()

    print("\n" + "=" * 80)
    print("✅ All demos completed!".center(80))
    print("=" * 80)
    print("\nFor more details, see: OPTIMIZATION_GUIDE.md")
    print()

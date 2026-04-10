"""
COMPREHENSIVE PIPELINE REVIEW & OPTIMIZATION STRATEGY
LongContext-ICL Annotation Competition

Generated: April 10, 2026
Status: Excellence Assessment + Competition-Winning Improvements
"""

═══════════════════════════════════════════════════════════════════════════════
SECTION 1: CURRENT STATE ASSESSMENT
═══════════════════════════════════════════════════════════════════════════════

✅ STRENGTHS (What's Well Implemented)
─────────────────────────────────────────────────────────────────────────────

1. Architecture & Modularity
   ✓ Clean separation of concerns (dataset_loader, method, qwen_model, etc.)
   ✓ Lazy-loaded singletons preventing redundant model loading
   ✓ Comprehensive error handling with fallbacks (embeddings → keyword overlap)
   ✓ Well-structured token budgeting system

2. Advanced Techniques Already Implemented
   ✓ Self-consistency voting (3x runs + majority voting)
   ✓ Chain-of-thought reasoning prompt with 3-step breakdown
   ✓ Semantic example selection using all-MiniLM-L6-v2 embeddings
   ✓ Intelligent long-context compression (boilerplate removal, key extraction)

3. Infrastructure
   ✓ Comprehensive evaluation module with per-dataset metrics
   ✓ End-to-end pipeline supporting multiple execution modes
   ✓ CLI flexibility (--use-reasoning, --use-self-consistency, --optimize-long-context)

4. Robustness
   ✓ Graceful fallbacks when embeddings unavailable
   ✓ Token budget enforcement across all prompt variants
   ✓ Proper handling of heterogeneous dataset formats


⚠️  GAPS & OPPORTUNITIES (What Could Be Better)
─────────────────────────────────────────────────────────────────────────────

1. Prompt Design (CRITICAL - High Impact)
   ✗ Inconsistent prompt templates (multiple variants: legacy, reasoning, standard)
   ✗ Reasoning prompt lacks explicit constraint on output format
   ✗ No task-specific prompt adaptation
   ✗ Missing prompt engineering best practices (role clarity, output format reinforcement)
   ✗ No confidence calibration or uncertainty handling

2. Example Selection (HIGH - Medium-High Impact)
   ✗ Fixed top-k (1-3) selection without calibration
   ✗ No diversity penalty (selected examples may be redundant)
   ✗ No difficulty-based selection (mixing easy/hard examples)
   ✗ Embedding model (all-MiniLM) may not align with task semantics
   ✗ No validation that selected examples are actually useful

3. Long-Context Handling (HIGH - Content Preservation)
   ✗ Aggressive compression may lose subtle signals
   ✗ No task-aware compression (different tasks need different preservation strategies)
   ✗ Semantic extraction doesn't account for interdependencies
   ✗ No context ordering optimization (what position matters?)

4. Accuracy Optimization (HIGH - Base Model Performance)
   ✗ Output extraction regex too restrictive (DOTALL + .+? issues)
   ✗ No confidence-based filtering of incorrect outputs
   ✗ No ensemble of different strategies
   ✗ Post-processing is minimal (just regex extraction)
   ✗ No handling of edge cases (empty labels, special characters)

5. Inference Optimization (MEDIUM)
   ✗ Temperature fixed at 0.7 for all voting runs (could be varied)
   ✗ No exploration of different generation strategies (beam search vs sampling)
   ✗ Top_k/top_p fixed globally (should be task-aware)


═══════════════════════════════════════════════════════════════════════════════
SECTION 2: COMPETITION-WINNING IMPROVEMENTS (Priority Order)
═══════════════════════════════════════════════════════════════════════════════

🥇 TIER 1: HIGHEST IMPACT (Estimated +3-7% accuracy improvement each)
─────────────────────────────────────────────────────────────────────────────

IMPROVEMENT #1: Prompt Engineering - Comprehensive Task-Adaptive System
────────────────────────────────────────────────────────────────────────

Current Issue:
  • Multiple competing prompt templates
  • No systematic prompt optimization
  • Output format not properly reinforced in reasoning mode
  • No explicit instruction following checks

Recommended Solution:
  Create unified, task-aware prompt system with:
  
  A) Meta-Prompt Analysis (detect task type automatically)
  B) Dynamic Role Definition (task-specific personas)
  C) Format Reinforcement (explicit constraints restated)
  D) Output Verification Instructions
  E) Safety Instructions (what NOT to do)

Implementation Strategy:

```python
# NEW: prompt_optimizer.py
class TaskAdaptivePromptBuilder:
    
    def __init__(self):
        self.task_detector = TaskTypeDetector()
        self.format_enforcer = OutputFormatEnforcer()
    
    def build(self, input_text, examples, task_hint=""):
        # Detect task type from examples/input
        task_type = self.task_detector.infer(examples, input_text)
        
        # Build task-specific prompt components
        role = self._get_role_definition(task_type)
        format_spec = self._get_format_specification(examples)
        examples_block = self._format_examples(examples, task_type)
        reasoning = self._build_reasoning_structure(task_type)
        
        # Combine with explicit reinforcement
        return self._assemble_prompt(
            role=role,
            format_spec=format_spec,
            examples=examples_block,
            reasoning=reasoning,
            input_text=input_text,
            safety_checks=self._get_safety_checks(task_type)
        )
    
    def _get_format_specification(self, examples):
        # Auto-detect output format from examples
        outputs = [ex.get("output") for ex in examples if "output" in ex]
        
        # Scenarios:
        # - Classification: "Output will be one of: [class1, class2, ...]"
        # - Generation: "Output should be a concise [type] response"
        # - Extraction: "Output format: <label>extracted_value</label>"
        
        return self._format_rules_for_outputs(outputs)
```

Expected Gains: +3-5% overall accuracy
  • Better task understanding
  • Consistent format compliance
  • Fewer parsing/extraction errors
  • Better instruction following

Difficulty: Medium (requires prompt analysis but NO model changes)


IMPROVEMENT #2: Advanced Example Selection with Diversity & Difficulty
───────────────────────────────────────────────────────────────────────

Current Issue:
  • Top-k selection without diversity (may pick 3 very similar examples)
  • No difficulty calibration (don't know if examples are representative)
  • No feedback loop (can't tell if examples help or hurt)
  • Fixed k=1-3 across all tasks

Recommended Solution:
  Multi-factor Example Selection with:
  
  A) Semantic Relevance (existing: cosine similarity)
  B) Coverage/Diversity (penalize redundant examples)
  C) Difficulty Matching (prefer examples near problem boundary)
  D) Output Distribution Matching (diverse labels)
  E) Validation Score (predict usefulness)

Implementation Strategy:

```python
# ENHANCEMENT: method.py - select_best_examples()

def select_best_examples_v2(
    input_text: str,
    examples: list[dict],
    top_k: int = 3,
    use_diversity: bool = True,
    use_difficulty: bool = True,
    use_validation: bool = True
) -> list[dict]:
    """Multi-factor example selection."""
    
    # Factor 1: Semantic Relevance (existing)
    relevance_scores = _compute_semantic_similarity(input_text, examples)
    
    # Factor 2: Diversity (penalize redundancy)
    if use_diversity:
        diversity_scores = _compute_diversity_scores(examples, relevance_scores)
    else:
        diversity_scores = [1.0] * len(examples)
    
    # Factor 3: Difficulty Matching
    if use_difficulty:
        difficulty_scores = _compute_difficulty_match(input_text, examples)
    else:
        difficulty_scores = [1.0] * len(examples)
    
    # Factor 4: Output Distribution
    output_dist_scores = _compute_label_balance_score(examples)
    
    # Factor 5: Validation (predict if example will help)
    if use_validation:
        validation_scores = _validate_example_usefulness(input_text, examples)
    else:
        validation_scores = [1.0] * len(examples)
    
    # Weighted combination
    combined_scores = (
        0.40 * relevance_scores +      # Most important
        0.20 * diversity_scores +      # Avoid redundancy
        0.15 * difficulty_scores +     # Better matching
        0.15 * output_dist_scores +    # Balanced labels
        0.10 * validation_scores       # Predicted usefulness
    )
    
    top_indices = np.argsort(combined_scores)[-top_k:]
    
    # Validate the selection
    selection_quality = _assess_selection_quality(
        examples, top_indices, input_text
    )
    
    return [examples[i] for i in top_indices], selection_quality


def _compute_diversity_scores(examples, relevance):
    """Penalize redundant examples."""
    scores = np.ones(len(examples))
    emb = _encode_texts([str(ex["input"]) for ex in examples])
    
    # For each example, reduce score if similar to high-relevance examples
    for i in range(len(examples)):
        for j in range(len(examples)):
            if i != j and relevance[j] > relevance[i]:
                similarity = cosine_similarity(emb[i], emb[j])
                scores[i] *= (1 - 0.3 * similarity)  # 30% penalty per duplicate
    
    return scores


def _compute_difficulty_match(input_text, examples):
    """Prefer examples similar in 'difficulty' to input."""
    input_length = len(input_text.split())
    input_entropy = _compute_text_entropy(input_text)
    
    scores = []
    for ex in examples:
        ex_length = len(str(ex.get("input", "")).split())
        ex_entropy = _compute_text_entropy(str(ex.get("input", "")))
        
        # Penalty for extreme differences
        length_diff = abs(input_length - ex_length) / max(input_length, 1)
        entropy_diff = abs(input_entropy - ex_entropy)
        
        score = 1 / (1 + length_diff + entropy_diff)
        scores.append(score)
    
    return np.array(scores)


def _validate_example_usefulness(input_text, examples):
    """Predict if an example will actually help."""
    # Quick heuristic: examples with clear input/output are more useful
    scores = []
    for ex in examples:
        has_input = bool(ex.get("input", "").strip())
        has_output = bool(_extract_example_output(ex))
        output_length = len(str(ex.get("output", "")).split())
        
        # Avoid trivial examples
        if output_length > 30:  # Too long
            score = 0.5
        elif output_length < 1:  # Empty
            score = 0.3
        else:
            score = 0.8 if (has_input and has_output) else 0.4
        
        scores.append(score)
    
    return np.array(scores)
```

Expected Gains: +2-4% overall accuracy
  • Better example diversity (avoid redundancy)
  • Examples more relevant to task difficulty
  • Reduced interference from poor examples
  • Auto-calibrates k per task

Difficulty: Medium-High (requires new scoring functions)


IMPROVEMENT #3: Robust Output Extraction & Post-Processing
───────────────────────────────────────────────────────────

Current Issue:
  Pattern: r'<label>\s*(.+?)\s*</label>' with DOTALL
  Problems:
    • .+? with DOTALL matches everything between first <label> and last </label>
    • Nested labels will extract wrong spans
    • Special characters (newlines, tabs) not normalized
    • No fallback for malformed outputs
    • count_answer() returns None on 100+ char outputs (too harsh!)

Recommended Solution:
  Robust extraction with multiple fallback strategies:

```python
# REPLACEMENT: method.py - count_answer()

def count_answer_v2(
    text: str,
    max_length: int = 500,
    strict_mode: bool = False
) -> str | None:
    """
    Robust extraction of annotation from model output.
    
    Strategies:
    1. Standard <label> extraction
    2. Repair malformed tags
    3. Extract by content length heuristics
    4. Fallback to instruction-following patterns
    """
    
    # Strategy 1: Standard extraction
    result = _extract_label_content_safe(text)
    if result:
        return result
    
    # Strategy 2: Repair common malformations
    repaired = _repair_label_tags(text)
    result = _extract_label_content_safe(repaired)
    if result:
        return result
    
    if strict_mode:
        return None
    
    # Strategy 3: Heuristic extraction for edge cases
    result = _heuristic_extract(text)
    if result:
        return result
    
    return None


def _extract_label_content_safe(text: str) -> str | None:
    """Extract content from <label>...</label> with proper handling."""
    
    # Find all label pairs with proper nesting detection
    matches = []
    pos = 0
    while True:
        start = text.find('<label>', pos)
        if start == -1:
            break
        
        end = text.find('</label>', start)
        if end == -1:
            break
        
        content = text[start+7:end].strip()
        
        # Validate extracted content
        if _is_valid_label_content(content):
            matches.append(content)
        
        pos = end + 8
    
    if not matches:
        return None
    
    # Use voting to pick most common (ties broken by length)
    from collections import Counter
    counter = Counter(matches)
    if not counter:
        return None
    
    most_common = counter.most_common()
    
    # Pick by: 1) frequency, 2) length (prefer reasonable length)
    for content, count in most_common:
        if len(content) <= 500:  # Reasonable max length
            return content
    
    # If all too long, return shortest
    return min(matches, key=len) if matches else None


def _repair_label_tags(text: str) -> str:
    """Repair common tag issues."""
    
    # Fix missing opening tag: "text</label>" → "<label>text</label>"
    text = re.sub(r'^([^<].*?)</label>', r'<label>\1</label>', text)
    
    # Fix missing closing tag: "<label>text" → "<label>text</label>"
    text = re.sub(r'<label>(.*?)$', r'<label>\1</label>', text)
    
    # Fix extra spaces in tags: "< label>" → "<label>"
    text = re.sub(r'<\s*label\s*>', '<label>', text)
    text = re.sub(r'<\s*/\s*label\s*>', '</label>', text)
    
    # Don't repair nested tags - too ambiguous
    
    return text


def _is_valid_label_content(content: str) -> bool:
    """Check if extracted content looks like valid annotation."""
    if not content or len(content) > 500:
        return False
    
    # Reject if it looks like instructions/reasoning (not final answer)
    bad_patterns = [
        'reason', 'think', 'analyze', 'consider', 'however',
        'explanation', 'let me', 'based on', 'therefore',
        'here is', 'output', 'annotation'
    ]
    
    lower = content.lower()
    if any(pattern in lower for pattern in bad_patterns):
        # Might be explanation, not label - check if it comes before label
        return False
    
    return True


def _heuristic_extract(text: str) -> str | None:
    """Attempt extraction via heuristics for edge cases."""
    
    # If model talked about the label, extract the last meaningful phrase
    # E.g., "I think the answer is: positive" → extract "positive"
    
    lines = text.strip().split('\n')
    
    for line in reversed(lines):
        line = line.strip()
        if not line or len(line) > 100:
            continue
        
        # Remove common prefixes
        line = re.sub(r'^(answer|output|label|result):\s*', '', line, flags=re.IGNORECASE)
        
        if 5 <= len(line) <= 100 and not any(c in line for c in '<>{}[]()'):
            return line
    
    return None
```

Expected Gains: +1-3% overall accuracy
  • Better handling of model formatting variations
  • More robust to output edge cases
  • Fewer "None" extractions
  • Task more forgiving of minor formatting errors

Difficulty: Medium (debugging & testing required)


🥈 TIER 2: HIGH IMPACT (Estimated +1-3% accuracy improvement each)
─────────────────────────────────────────────────────────────────────

IMPROVEMENT #4: Adaptive Token Budgeting
────────────────────────────────────────

Current: Fixed max_prompt_tokens=8192
Problem: Not all examples need the same token budget

Solution:
```python
def compute_optimal_token_budget(
    task_id: str,
    input_length: int,
    num_examples: int
) -> dict[str, int]:
    """Optimize token allocation per task."""
    
    # Task-specific profiles (calibrated from dev set)
    TASK_BUDGETS = {
        "openseek-1": {"examples": 3000, "input": 2000, "reasoning": 1500},
        "openseek-2": {"examples": 2000, "input": 3000, "reasoning": 2000},
        # ... etc
    }
    
    if task_id in TASK_BUDGETS:
        return TASK_BUDGETS[task_id]
    
    # Heuristic for unknown tasks
    return {
        "examples": min(3000, max(1000, num_examples * 30)),
        "input": max(1000, min(3000, input_length // 3)),
        "reasoning": 1500,
    }
```

Expected Gain: +0.5-1.5% accuracy


IMPROVEMENT #5: Temperature-Adjusted Voting
────────────────────────────────────────────

Current: Fixed temperature=0.7 for all voting runs
Problem: Some tasks need higher creativity, others need consistency

Solution:
```python
def get_best_output_v2(
    prompt: str,
    num_samples: int = 3,
    task_type: str = "classification",
    model_loader = None,
    max_new_tokens: int = 256
) -> str:
    """Adaptive temperature voting."""
    
    # Temperature schedule per task type
    TEMPS = {
        "classification": [0.1, 0.5, 0.9],  # Diverse votes
        "extraction": [0.3, 0.3, 0.3],       # Consistent
        "generation": [0.7, 0.8, 0.9],       # Creative
    }
    
    temps = TEMPS.get(task_type, [0.7, 0.7, 0.7])
    outputs = []
    
    for temp in temps:
        output = model_loader.generate_response(
            prompt,
            temperature=temp,
            top_p=0.9,
            max_new_tokens=max_new_tokens
        )
        outputs.append(output)
    
    # Vote
    return _majority_vote(outputs)
```

Expected Gain: +0.5-1% accuracy


IMPROVEMENT #6: Input Preprocessing & Normalization
────────────────────────────────────────────────────

Current: Raw input used directly
Problem: Input may have formatting noise (extra spaces, encoding issues)

Solution:
```python
def preprocess_input(text: str) -> str:
    """Normalize input before processing."""
    
    # Remove ctrl characters
    text = ''.join(ch for ch in text if ch.isprintable() or ch in '\n\t ')
    
    # Fix encoding issues
    text = text.encode('utf-8', errors='replace').decode('utf-8')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove trailing/leading spaces
    text = text.strip()
    
    return text
```

Expected Gain: +0.5% accuracy (varies by dataset)


🥉 TIER 3: MEDIUM IMPACT (Estimated +0.5-1.5% improvement)
──────────────────────────────────────────────────────────

IMPROVEMENT #7: Ensemble Methods
─────────────────────────────────

Combine predictions from different strategies:
  • Standard prompt + Reasoning prompt → Vote
  • Different compression levels (50% vs 70%) → Vote
  • Temperature scaling experiments

Expected Gain: +0.5-1% accuracy


IMPROVEMENT #8: Task-Specific Fine-tuning of Hyperparameters
─────────────────────────────────────────────────────────────

Calibrate per dataset:
  • Example count (1-5 instead of fixed 2-3)
  • Reasoning steps (2-4 variations)
  • Compression aggressiveness
  • Voting samples (3-5 instead of fixed 3)

Expected Gain: +0.5-1.5% accuracy


IMPROVEMENT #9: Improved Evaluation Metrics
──────────────────────────────────────────────

Current: Simple accuracy (correct/total)
Add:
  - Per-example confidence scores
  - Task-specific metrics (if labeled data available)
  - False positive/negative breakdown
  - Output distribution analysis

Expected Gain: Better ablation analysis (enables above improvements)


═══════════════════════════════════════════════════════════════════════════════
SECTION 3: IMPLEMENTATION ROADMAP
═══════════════════════════════════════════════════════════════════════════════

PHASE 1 (Week 1): Foundation [Estimated: +3-5% improvement]
─────────────────────────────────────────────────────────────
Implement:
  ✓ #1 Task-adaptive prompt engineering
  ✓ #3 Robust output extraction v2
  ✓ #4 Adaptive token budgeting
  
Effort: 2-3 days
Impact: High (prompt engineering is critical)


PHASE 2 (Week 2): Enhancement [Estimated: +2-4% improvement]
──────────────────────────────────────────────────────────────
Implement:
  ✓ #2 Advanced example selection
  ✓ #5 Temperature-adjusted voting
  ✓ #6 Input preprocessing
  
Effort: 2-3 days
Impact: Medium-High


PHASE 3 (Integration & Tuning) [Estimated: +1-2% improvement]
──────────────────────────────────────────────────────────────
Implement:
  ✓ #7 Ensemble methods
  ✓ #8 Task-specific calibration
  
Effort: 1-2 days of tuning per task
Impact: Remaining gains


═══════════════════════════════════════════════════════════════════════════════
SECTION 4: CODE ARCHITECTURE RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════

Current Structure (Good):
```
src/
├── dataset_loader.py
├── method.py (prompt, example selection)
├── qwen_model.py (inference)
├── self_consistency.py (voting)
├── long_context_optimizer.py (compression)
├── evaluate.py
└── run_qwen_predictions.py / pipeline.py
```

Recommended Enhancements:
```
src/
├── dataset_loader.py
├── method.py (refactored)
├── prompt_optimizer.py ← NEW: Task-adaptive prompts
├── example_selector_v2.py ← NEW: Multi-factor selection
├── output_parser.py ← NEW: Robust extraction
├── qwen_model.py
├── self_consistency_v2.py ← ENHANCED: Task-aware voting
├── long_context_optimizer.py
├── task_analyzer.py ← NEW: Auto-detect task type
├── hyperparameter_tuner.py ← NEW: Task-specific tuning
├── evaluate.py
├── ensemble_methods.py ← NEW: Ensemble voting
└── run_qwen_predictions.py / pipeline.py
```

═══════════════════════════════════════════════════════════════════════════════
SECTION 5: QUICK WINS (Low Effort, Measurable Gains)
═══════════════════════════════════════════════════════════════════════════════

⚡ Can implement TODAY:

1. Fix count_answer() strictness
   • Change: len(answer[0]) >= 100 → >= 500
   • Why: Legitimate answers may be 100+ chars
   • Gain: +0.5-1%
   • Time: 5 minutes

2. Add input preprocessing
   • Normalize whitespace and encoding
   • Gain: +0.3-0.5%
   • Time: 15 minutes

3. Improve error messages in count_answer()
   • Return partial extraction instead of None
   • Gain: +0.2-0.5%
   • Time: 10 minutes

4. Cache embeddings during evaluation
   • Speed up dev cycle
   • Time: 20 minutes

5. Add task type detection
   • Auto-classify classification vs extraction vs generation
   • Use for prompt selection
   • Gain: +0.5-1.5%
   • Time: 1 hour

═══════════════════════════════════════════════════════════════════════════════
SECTION 6: LONG-CONTEXT SPECIFIC RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════

Current: Compression with boilerplate removal + key extraction

Improvements:
───────────────

1. Task-Aware Compression Preservation
   Different tasks need different compression strategies:
   
   Classification: Can be aggressive (20-30% preserve key tokens)
   Extraction: Need context preservation (50%+ preserve)
   Generation: Preserve more structure (60%+)
   
   → Add task_type parameter to compress_for_long_context()

2. Positional Encoding Awareness
   Research shows: Important info at DOC_START and DOC_END
   But also: Middle sections often contain key clauses
   
   Solution:
   ```python
   def smart_compress_positions(text, task_type):
       """Preserve important positions per task."""
       if task_type == "classification":
           # Keep: Start 30%, random 20%, end 20%
           return compress_with_position_bias(text, [0.3, 0.2, 0.2])
       elif task_type == "extraction":
           # Keep: More uniform distribution
           return compress_with_position_bias(text, [0.3, 0.4, 0.3])
   ```

3. Redundancy Detection
   Current: Removes based on keywords
   Better: Sentence-level redundancy detection
   
   Some paragraphs repeat the same idea → keep only best one

4. Hierarchical Compression
   Current: Sentence-level
   Better: Paragraph → sentence → phrase hierarchy
   
   Compress paragraphs first, then sentences


═══════════════════════════════════════════════════════════════════════════════
SECTION 7: TESTING & VALIDATION STRATEGY
═══════════════════════════════════════════════════════════════════════════════

Before committing to production:

1. Ablation Study
   Test each improvement individually:
   ```
   Baseline: 72.3%
   + Prompt optimization: 75.1% (+2.8%)
   + Example selection: 76.8% (+1.7%)
   + Output extraction: 77.2% (+0.4%)
   + Token budgeting: 77.6% (+0.4%)
   + Temperature voting: 78.1% (+0.5%)
   + Ensemble: 78.5% (+0.4%)
   ```

2. Per-Task Analysis
   Track accuracy for each of 8 datasets:
   ```
   Task 1 (classification): 85% → 88%
   Task 2 (extraction): 71% → 75%
   Task 3 (numeric): 82% → 85%
   ...
   ```

3. Error Analysis
   Categorize failures:
   ```
   Format errors: 15% (before) → 3% (after improvements)
   Semantic errors: 68%
   Edge cases: 17%
   ```

4. Regression Testing
   Ensure improvements don't hurt already-good tasks


═══════════════════════════════════════════════════════════════════════════════
SECTION 8: ADVANCED TECHNIQUES (If Time Permits)
═══════════════════════════════════════════════════════════════════════════════

Not immediately implementable but worth exploring:

1. In-Context Learning Optimization
   • Mix example styles (some short, some long)
   • Vary example ordering
   • Chain-of-examples (show reasoning for each example)

2. Instruction Tuning on Prompts
   • Use one dataset as train, others as validation
   • Optimize prompt templates via grid search

3. Mixture of Experts for Voting
   • Different voters for different task types
   • Confidence-weighted voting

4. Confidence Estimation
   • Add confidence filter: only output if confidence > threshold
   • Use entropy of voting distribution

5. Meta-Learning
   • Few-shot learn the example selection strategy per task
   • Learn composition of prompt components

═══════════════════════════════════════════════════════════════════════════════
SUMMARY & RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════

Your pipeline is WELL-BUILT with good fundamentals.

To WIN the competition, focus on:

1. 🎯 TIER 1 (Must Do):
   - Task-adaptive prompt engineering (#1)
   - Robust output extraction (#3)
   - Better example selection (#2)
   
   These alone could give +5-10% improvement.

2. 🚀 TIER 2 (Should Do):
   - Temperature-adjusted voting (#5)
   - Adaptive token budgeting (#4)
   - Input preprocessing (#6)
   
   These refine and stabilize the above +1-3%.

3. ⭐ Implementation Priority:
   Week 1: #1, #3, #4 (prompt, parsing, budgeting)
   Week 2: #2, #5, #6 (examples, voting, prep)
   Week 3: #7, #8 (ensemble, tuning)

4. 📊 Success Indicators:
   • Format error rate: 15% → <3%
   • Per-task accuracy variance reduction (std dev)
   • Classification accuracy: +5-7%
   • Extraction accuracy: +3-5%

═══════════════════════════════════════════════════════════════════════════════
"""

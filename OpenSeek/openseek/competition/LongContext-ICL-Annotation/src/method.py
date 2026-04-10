
import re
from collections import Counter
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


""" Here is an example of implementation of Long-Context Data Annotation. """


_SIM_TOKENIZER = None
_SIM_MODEL = None
_QWEN_TOKENIZER = None


def _get_similarity_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazily initialize sentence-embedding encoder."""
    global _SIM_TOKENIZER, _SIM_MODEL
    if _SIM_TOKENIZER is None or _SIM_MODEL is None:
        _SIM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _SIM_MODEL = AutoModel.from_pretrained(model_name)
        _SIM_MODEL.eval()
    return _SIM_TOKENIZER, _SIM_MODEL


def _get_qwen_tokenizer(model_name_or_path: str = "Qwen/Qwen3-4B"):
    """Lazily initialize Qwen tokenizer for token-budget checks."""
    global _QWEN_TOKENIZER
    if _QWEN_TOKENIZER is None:
        _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    return _QWEN_TOKENIZER


def _compact_long_text(text: str, max_chars: int = 2400) -> str:
    """Keep long text compact for efficient embedding while preserving key context."""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return head + "\n...\n" + tail


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _encode_texts(texts: list[str], batch_size: int = 32) -> torch.Tensor:
    """Encode texts into normalized sentence embeddings."""
    tokenizer, model = _get_similarity_encoder()
    all_embeddings = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            emb = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb)

    return torch.cat(all_embeddings, dim=0)


def select_best_examples(input_text: str, dataset_examples: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    """Select top-k most similar examples using cosine similarity on sentence embeddings.

    This is efficient for long text by compacting content before encoding and
    encoding examples in batches.
    """
    if not dataset_examples:
        return []

    valid_examples = [ex for ex in dataset_examples if isinstance(ex, dict) and "input" in ex]
    if not valid_examples:
        return []

    top_k = max(1, min(top_k, len(valid_examples)))
    query_text = _compact_long_text(str(input_text))
    example_texts = [_compact_long_text(str(ex.get("input", ""))) for ex in valid_examples]

    try:
        query_emb = _encode_texts([query_text])
        example_emb = _encode_texts(example_texts)
        scores = torch.matmul(example_emb, query_emb.T).squeeze(1)
        top_indices = torch.topk(scores, k=top_k).indices.tolist()
        return [valid_examples[i] for i in top_indices]
    except Exception:
        # Fallback: lexical overlap score if embedding model is unavailable.
        query_tokens = set(query_text.lower().split())
        scored = []
        for idx, ex_text in enumerate(example_texts):
            ex_tokens = set(ex_text.lower().split())
            union = len(query_tokens | ex_tokens) or 1
            score = len(query_tokens & ex_tokens) / union
            scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [valid_examples[idx] for _, idx in scored[:top_k]]

def build_prompt____(task_description: str, text2annotate: str) -> str:
    """
    Build a high-precision English prompt for long-context data annotation (optimized for Qwen3-4B).
    Core requirement: Final answer MUST be wrapped in <label> tags (no extra content outside tags).
    """
    prompt = (
        "### Role Definition\n"
        "You are a professional data annotation expert specializing in long-context text labeling. "
        "Your work must strictly comply with the following rules, with the highest priority given to output format accuracy.\n\n"
        
        "### Core Annotation Task\n"
        f"{task_description}\n\n"
        
        "### Non-Negotiable Annotation Rules (Highest Priority)\n"
        "1. **Final Output Mandate**: Your annotation result MUST be wrapped in <label> tags — NO text, symbols, spaces, or explanations are allowed outside the tags.\n"
        "2. **Internal Reasoning Permission**: You may perform logical reasoning, text analysis, or context comprehension internally (in your thought process), but NONE of these thoughts may appear in the final output.\n"
        "3. **Label Format Strictness**: <label> is the opening tag and </label> is the closing tag — they must appear in pairs, with NO extra spaces or characters inside the tags (e.g., <label>  Good Review  </label> is invalid).\n"
        "4. **Prohibited Outputs**: \n"
        "   - ❌ Prohibited: 'After analysis, this is a positive review: <label>Good Review</label>' (extra text outside tags)\n"
        "   - ❌ Prohibited: 'Bad Review' (missing <label> tags entirely)\n"
        "   - ❌ Prohibited: '<label>Bad Review' (unpaired/closing tag missing)\n\n"
        
        "### Correct vs. Incorrect Examples\n"
        "✅ Correct Example 1: <label>answer</label>\n"
        "✅ Correct Example 2: <label>Bad Review</label>\n"
        "❌ Incorrect Example 1: I think this review is negative → <label>Bad Review</label>\n"
        "❌ Incorrect Example 2: <label>  Neutral Review  </label> (extra spaces inside tags)\n"
        "❌ Incorrect Example 3: Neutral Review (no label tags)\n\n"
        
        "### Reference Annotation Examples\n"
        "{EXAMPLES}\n\n"
        
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        
        "### Final Output Command (Re-emphasized)\n"
        "You may complete any internal reasoning process, but your FINAL OUTPUT MUST consist solely of the annotation result wrapped in <label> tags (no other content whatsoever).\n"
        "Annotation Result: "
    )
    return prompt

def build_prompt_legacy(task_description: str, text2annotate: str) -> str:
    """
    Construct a high-precision prompt for long-context data annotation (optimized for Qwen3-4B).
    task_description: Clear description of the annotation task (e.g., "Classify English product reviews as Good Review/Bad Review").
    text2annotate: The text to be annotated (single text or batch texts).
    """
    prompt = (
        "### Role Definition\n"
        "You are a professional data annotation expert specialized in long-context text labeling. "
        "Your work must strictly follow the task rules, fully learn from the provided examples, and ensure the final annotation result is 100% enclosed in <label> tags.\n\n"
        
        "### Core Task\n"
        f"{task_description}\n\n"
        
        "### Critical Annotation Guidelines\n"
        "1. **Example Learning Requirement**: Thoroughly analyze and fully learn from the annotation logic, format, and criteria in the Examples section. "
        "Your annotation must align with the style, judgment standards, and tag usage shown in the examples.\n"
        "2. **Thinking Process**: You may (and are encouraged to) explain your annotation reasoning step by step (e.g., key information extraction, judgment basis, rule matching).\n"
        "3. **Mandatory Output Rule**: Regardless of any thinking process you provide, your final annotation result MUST be enclosed in <label> tags (this is non-negotiable).\n"
        "   - Correct example: \n"
        "     Reasoning: This review mentions 'excellent quality' and 'very satisfied', which meets the criteria for a Good Review.\n"
        "     <label>Good Review</label>\n"
        "   - Wrong example 1 (missing tags): This review is negative.\n"
        "   - Wrong example 2 (incomplete tags): Bad Review</label>\n"
        "4. **Length Adaptation**: For long texts, maintain complete thinking process and ensure the final <label> tags contain the accurate annotation result (no truncation).\n\n"
        
        "### Examples (Must Be Fully Followed)\n"
        "[[EXAMPLES]]\n\n"
        
        "### Text to Annotate\n"
        f"{text2annotate}\n\n"
        
        "### Final Requirement Summary\n"
        "1. You can (and should) provide clear thinking process for your annotation.\n"
        "2. The final annotation result MUST be wrapped in <label> tags (no exceptions).\n"
        "3. All annotation logic must strictly follow the examples provided above.\n"
    )
    return prompt


def _extract_example_output(example: dict) -> str:
    """Normalize output fields across dataset variants."""
    for key in ("label", "labels", "target", "targets", "output", "outputs", "answer"):
        if key in example:
            value = example[key]
            if isinstance(value, list):
                if not value:
                    return ""
                return str(value[0])
            return str(value)
    return ""


def _select_icl_examples(examples: list[dict], min_count: int = 2, max_count: int = 3) -> list[dict]:
    """Pick 2-3 examples for ICL; fallback to whatever is available."""
    valid_examples = [ex for ex in examples if isinstance(ex, dict) and "input" in ex]
    if not valid_examples:
        return []

    if len(valid_examples) < min_count:
        return valid_examples
    return valid_examples[:max_count]


def build_prompt(input_text: str, examples: list[dict] | str) -> str:
    """Build a structured ICL prompt.

    Primary mode:
        build_prompt(input_text, examples)
        - input_text: text to annotate
        - examples: dataset examples list, each containing input/output-like fields

    Backward-compatible mode:
        build_prompt(task_description, text2annotate)
        - If `examples` is passed as a string, this function routes to the
          legacy task-description prompt builder.
    """
    if isinstance(examples, str):
        return build_prompt_legacy(task_description=input_text, text2annotate=examples)

    chosen_examples = _select_icl_examples(examples)
    examples_block_lines = []
    for idx, example in enumerate(chosen_examples, start=1):
        ex_input = str(example.get("input", "")).strip()
        ex_output = _extract_example_output(example).strip()
        examples_block_lines.append(
            f"Example {idx}:\n"
            f"Input: {ex_input}\n"
            f"Output: {ex_output}"
        )

    examples_block = "\n\n".join(examples_block_lines) if examples_block_lines else "(No examples provided)"

    prompt = (
        "### Instruction\n"
        "You are a data annotation assistant. Learn from the provided in-context examples and "
        "produce the best output for the given input using the same style and format.\n\n"
        "### Examples\n"
        f"{examples_block}\n\n"
        "### Input\n"
        f"{input_text}\n\n"
        "### Output\n"
    )
    return prompt


def build_prompt_with_reasoning(
    input_text: str,
    examples: list[dict[str, Any]] | None = None,
    task_instruction: str = "",
) -> str:
    """Build a prompt that encourages multi-step chain-of-thought reasoning.

    Structure:
    1. Task instruction
    2. In-context examples (if provided)
    3. Explicit reasoning steps:
       - Step 1: Extract key information
       - Step 2: Analyze context
       - Step 3: Generate final annotation

    This format helps LLMs reason through complex tasks more systematically.
    """
    examples = examples or []
    chosen_examples = _select_icl_examples(examples, min_count=1, max_count=2)

    examples_block_lines = []
    for idx, example in enumerate(chosen_examples, start=1):
        ex_input = str(example.get("input", "")).strip()
        ex_output = _extract_example_output(example).strip()
        ex_reasoning = example.get("reasoning", "")
        
        examples_block_lines.append(
            f"Example {idx}:\n"
            f"Input: {ex_input}\n"
        )
        if ex_reasoning:
            examples_block_lines[-1] += f"Reasoning Process: {ex_reasoning}\n"
        examples_block_lines[-1] += f"Output: {ex_output}"

    examples_block = "\n\n".join(examples_block_lines) if examples_block_lines else ""

    prompt = (
        "### Task Instruction\n"
    )
    if task_instruction:
        prompt += f"{task_instruction}\n\n"
    else:
        prompt += (
            "You are a data annotation expert. Your task is to carefully analyze the given input "
            "and produce a precise annotation following the provided examples.\n\n"
        )

    if examples_block:
        prompt += (
            "### In-Context Examples (Learn from these)\n"
            f"{examples_block}\n\n"
        )

    prompt += (
        "### Text to Annotate\n"
        f"{input_text}\n\n"
        "### Reasoning and Annotation Process\n"
        "Please work through the following steps to produce your annotation:\n\n"
        "**Step 1: Extract Key Information**\n"
        "- Identify the core content, entities, sentiment, or relevant attributes from the input.\n"
        "- Note any important patterns or keywords that relate to the annotation task.\n\n"
        "**Step 2: Analyze Context**\n"
        "- Compare the extracted information against the patterns shown in the examples above.\n"
        "- Determine which example(s) are most similar and why.\n"
        "- Consider any edge cases or special context that might affect the annotation.\n\n"
        "**Step 3: Generate Final Annotation**\n"
        "- Based on your analysis, produce the final annotation in the same format as the examples.\n"
        "- Ensure your output follows the structure and style of the in-context examples.\n"
        "- Wrap your final answer in <label> tags: <label>your annotation</label>\n\n"
        "### Your Response\n"
        "Please proceed with the three-step reasoning process above, then provide your final annotation:\n"
    )
    return prompt

def build_prompt_backup(task_description:str, text2annotate:str)->str:
    """
        Construct the prompt for annotation based on the task description.
        task_description: 
            The description of the annotation task. 
            For example, ``Given an English language product review, 
            determine if it is a Good Review or a Bad Review.`` 
        text2annotate:
            The text that needs to be annotated.
            For example, ``My son received this book as a gift. I was extremely disappointed.``
    """
    prompt = (
        "You are a data annotation assistant. "
        "Your task is to label the given texts according to the task description "
        "and annotation guidelines provided below.\n\n"
        f"[Task Description]\n {task_description}\n\n"
        "[Examples]\n {EXAMPLES}\n\n"
        "Please follow these instructions when labeling:\n"
        "1. **Output Format**: Annotate the text directly by wrapping each labeled "
        "span with <label> tags in the following format: <label> annotation result </label>.\n"
        # "2. Do not add any extra text, explanations, or commentary in the labeled spans.\n\n"
        f"[Task Description (repeat)] \n {task_description}\n\n"
        f"[Input Texts]\n {text2annotate}\n\n"
        "Please output the annotation results: "
    )
    return prompt

def select_examples_backup(all_examples:list[dict], task_description:str, text2annotate:str)->str:
    """
        Select examples from all_examples to fit into the target context length.
        all_examples:
            A list of examples, where each example is a dict with keys 'input', 'output', and 'length'.
            For example, ``{"input": "The material is good and looks great.", "output": "Good Review", "length": 79``},
        task_description:
            The description of the annotation task which may be used for example evaluation. 
            For example, ``Given an English language product review, 
            determine if it is a Good Review or a Bad Review.`` 
        text2annotate:
            The text that needs to be annotated  which may be used for example retrieval.
            For example, ``My son received this book as a gift. I was extremely disappointed.``
        
    """
    # Notice that the maximum context length is restricted.
    target_length = 10_000
    
    input_list = [example['input'] for example in all_examples]
    output_list = [example['output'][0] for example in all_examples]
    length_list = [example['length'] for example in all_examples]
    
    # <label> have 2 tokens; </label> have 3 tokens; \n have 1 token; # have 1 token.
    examples_str, token_num = "", 0
    for i, (input_text, output_text, length) in enumerate(zip(input_list, output_list, length_list)):
        if length + token_num <= target_length:
            token_num += (length + 2 + 3 + 1 + 1)
            example_str = f"# {input_text} <label> {output_text} </label>\n"
            examples_str += example_str
        else:
            return examples_str, i
    return examples_str

def select_examples(all_examples: list[dict], task_description: str, text2annotate: str) -> str:
    """
        Select examples from all_examples to fit into the target context length (适配Qwen3-4B的token计算).
        all_examples:
            A list of examples, where each example is a dict with keys 'input' and 'output' (no 'length' needed).
            For example, ``{"input": "The material is good and looks great.", "output": "Good Review"}``,
        task_description:
            The description of the annotation task which may be used for example evaluation. 
        text2annotate:
            The text that needs to be annotated  which may be used for example retrieval.
    """
    tokenizer = _get_qwen_tokenizer()
    target_length = 8192

    # Use semantic retrieval to pick the most relevant 2-3 examples.
    selected = select_best_examples(text2annotate, all_examples, top_k=3)
    if len(selected) == 1 and len(all_examples) > 1:
        selected = select_best_examples(text2annotate, all_examples, top_k=2)

    examples_str = ""
    token_num = 0
    for i, example in enumerate(selected):
        try:
            input_text = str(example["input"])
            output_text = _extract_example_output(example)

            input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            length = input_tokens + output_tokens

            if length + token_num > target_length:
                break

            token_num += (length + 7)
            examples_str += f"# {input_text} <label> {output_text} </label>\n"
        except KeyError as e:
            print(f"Warning: example {i} missing key {e}, skipped")
            continue

    return examples_str




def count_answer(text: str) -> tuple[list, dict]:
    """
    提取字符串中<label>标签内的所有内容（字符串形式），统计出现次数最多的内容
    :param text: 包含<label>标签的原始字符串
    :return: 出现次数最多的内容列表、所有内容的频次统计字典
    """
    pattern = r'<label>\s*(.+?)\s*</label>'
    content_matches = re.findall(pattern, text, re.DOTALL) 
    
    content_counter = Counter(content_matches)
    if not content_counter:
        return None
    
    max_count = max(content_counter.values())
    answer = [content for content, count in content_counter.items() if count == max_count]
    
    if (len(answer[0]) >= 100):
        return None
    return answer[0]


def annotate_nvidia(input_prompt:str)->list[str]:
    """
        Annotate the unlabeled data using an LLM API (nvidia GPU).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
    """
    import requests
    URL="http://0.0.0.0:2026/v1/completions"
    
    data = {
        "model": "../Qwen3-4B",
        "prompt": input_prompt,
        "max_tokens": 10_000, # max_token = 10k
    }

    try:
        resp = requests.post(URL, json=data)
        whole_result = resp.json()["choices"][0]["text"]
    except Exception as e:
        whole_result = "None"


    prediction = count_answer(whole_result)
    return prediction

def annotate_ascend(input_prompt:str)->list[str]:
    """
        Annotate the unlabeled data using an LLM API (Huawei Ascend).
        prompts:
            A prompt constructed for annotation.
            For example, ``["You are a data annotation assistant. Your task is to label ..."]``
    """
    import openai
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:9010/v1/"
    model = "Qwen3-4B-ascend-flagos"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        top_p=0.95,
        max_tokens=10_000,
        stream=False,
    )
    whole_result = response.choices[0].message.content
    prediction = count_answer(whole_result)
    return prediction

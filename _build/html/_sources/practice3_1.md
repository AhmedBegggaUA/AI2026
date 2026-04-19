# Intelligent Agents

## Practical Session 3 — Part 1: Large Language Models, Hugging Face, and Prompt Engineering

### Introduction

In this first part of Practical Session 3, we will explore the foundations of **Large Language Models (LLMs)** and how to interact with them programmatically using the **Hugging Face** ecosystem. LLMs are deep learning models trained on massive text corpora that can generate coherent text, answer questions, summarize documents, translate languages, and perform many other natural language tasks. Their emergence has fundamentally changed the landscape of Artificial Intelligence, enabling the construction of sophisticated **intelligent agents** that reason, plan, and communicate in natural language.

Rather than training an LLM from scratch — which requires enormous computational resources — we will focus on **loading pre-trained models**, **running inference**, and mastering **prompt engineering**: the art of crafting effective instructions that guide a model toward the desired output. These skills form the essential building blocks for the RAG systems and multi-agent architectures that we will build in Parts 2 and 3.

Throughout this session we will work with **`Qwen/Qwen2.5-1.5B-Instruct`**, a small instruction-tuned model with 1.5 billion parameters that runs comfortably on a CPU with approximately **3–6 GB of RAM**.

### Prerequisites

Before starting, install the required libraries:

```{code-block} bash
pip install torch transformers accelerate sentencepiece protobuf matplotlib
```

```{note}
- We will load the model in **`bfloat16`** precision, which halves the memory footprint compared to full precision (`float32`) and is well supported on modern CPUs. If your CPU does not support `bfloat16`, the code includes a fallback to `float32`.
- We recommend using **Python 3.10+** and a Jupyter Notebook environment (Jupyter Lab, Google Colab, or VS Code Notebooks).
- All exercises can be completed on a **CPU-only** machine. Inference will take a few seconds per response, which is acceptable for the tasks in this session.
```

### Background: What Are Large Language Models?

A Large Language Model is, at its core, a **next-token predictor**. Given a sequence of tokens $x_1, x_2, \ldots, x_{t-1}$, the model estimates the probability distribution over the next token:

$$
P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)
$$

where $\theta$ represents the model's learned parameters. Modern LLMs are based on the **Transformer** architecture (Vaswani et al., 2017), which uses **self-attention** mechanisms to capture long-range dependencies in text. The key innovation is the attention function:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

where $Q$, $K$, and $V$ are the query, key, and value matrices derived from the input, and $d_k$ is the dimension of the keys. This mechanism allows each token to "attend" to every other token in the sequence, weighting their contributions by relevance.

During text generation, the model produces tokens **autoregressively**: at each step it samples (or selects) a token from the predicted distribution and appends it to the context, repeating the process until a stopping criterion is met (e.g., an end-of-sequence token or a maximum length).


### The Hugging Face Ecosystem

**Hugging Face** is the de facto open-source platform for working with pre-trained models. Its core components include:

- **`transformers`**: A Python library providing a unified API (`AutoModelForCausalLM`, `AutoTokenizer`, `pipeline`) to load and run thousands of pre-trained models.
- **Hugging Face Hub**: A model repository hosting over 500,000 models, datasets, and spaces.
- **`accelerate`**: A library that simplifies model loading and device placement across CPUs and GPUs.

The typical workflow is:

1. Choose a model from the Hub (e.g., `Qwen/Qwen2.5-1.5B-Instruct`)
2. Load the tokenizer and model with `AutoTokenizer` and `AutoModelForCausalLM`
3. Tokenize the input, run inference, and decode the output

### Model Precision and Memory

Full-precision LLMs store each parameter as a 32-bit floating-point number (FP32). A model with 1.5 billion parameters therefore requires approximately **6 GB** of RAM in FP32. By using **`bfloat16`** (Brain Floating Point 16), we store each parameter in 16 bits instead, cutting the memory requirement roughly in half to about **3 GB**. Unlike standard `float16`, `bfloat16` preserves the same exponent range as `float32`, which avoids numerical overflow issues that can cause `nan` values during inference.

| Precision | Bits per Parameter | ~Size for 1.5B params |
|-----------|-------------------|-----------------------|
| FP32      | 32                | ~6.0 GB               |
| BF16      | 16                | ~3.0 GB               |

```{important}
**Do not use `float16` on CPU.** Standard half-precision (`float16`) has a limited numerical range and is poorly supported for CPU computation — it frequently produces `nan` values in the model's logits, causing garbled outputs. Always use `bfloat16` or `float32` when running on CPU.
```

---

## Part 1: Loading and Running a Small LLM

### Choosing the Right Model

For this practical session we will use **`Qwen/Qwen2.5-1.5B-Instruct`**, a 1.5-billion-parameter instruction-tuned language model from the Qwen family. This model provides a good balance between capability and resource requirements:

- **BF16**: ~3 GB RAM
- **FP32**: ~6 GB RAM
- **Instruction-tuned**: Already fine-tuned to follow instructions in a chat-like format

### Loading the Model

```{code-block} python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Determine the best dtype for the current hardware
# bfloat16 is preferred (half memory, stable on CPU)
# float32 is the safe fallback
if torch.cuda.is_available():
    dtype = torch.bfloat16
    device_map = "auto"
elif hasattr(torch, "bfloat16"):
    dtype = torch.bfloat16
    device_map = "cpu"
else:
    dtype = torch.float32
    device_map = "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype = dtype,
    device_map = device_map,
)

print(f"Model loaded on {model.device} with dtype {dtype}")
print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
```

```{note}
On CPU, inference will be slower (a few seconds per response). For the exercises in this session, this is perfectly acceptable.
```

### Verifying the Model Works

After loading, it is good practice to verify that the model produces meaningful output before proceeding:

```{code-block} python
# Quick sanity check
test_input = tokenizer("The capital of France is", return_tensors = "pt").to(model.device)

with torch.no_grad():
    logits = model(**test_input).logits
    last_logits = logits[0, -1, :]
    print(f"Logits — min: {last_logits.min().item():.2f}, max: {last_logits.max().item():.2f}")
    print(f"Contains NaN: {torch.isnan(last_logits).any().item()}")
    
    # Generate a few tokens
    out = model.generate(**test_input, max_new_tokens = 10, do_sample = False,
                         pad_token_id = tokenizer.eos_token_id)
    print(f"Output: {tokenizer.decode(out[0], skip_special_tokens = True)}")
```

If you see `Contains NaN: True`, your dtype is not compatible with your hardware. Switch to `torch.float32`.

### Understanding Tokenization

Before feeding text to the model, it must be converted into **tokens** — integer indices that the model understands. The tokenizer handles this conversion and also applies any special formatting required by the model (e.g., chat templates).

```{code-block} python
# Basic tokenization
text = "Hello, how are you doing today?"
tokens = tokenizer(text, return_tensors = "pt")

print(f"Input text: {text}")
print(f"Token IDs: {tokens['input_ids']}")
print(f"Number of tokens: {tokens['input_ids'].shape[1]}")
print(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
```

### Generating Text

The simplest way to generate text is using the `model.generate()` method:

```{code-block} python
def generate_response(model, tokenizer, prompt, max_new_tokens = 256):
    """
    Generate a response from the model given a raw text prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text string
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        str: The generated response
    """
    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = True,
            temperature = 0.7,
            top_p = 0.9,
            top_k = 50,
            pad_token_id = tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (exclude the prompt)
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens = True
    )
    return response


# Test
response = generate_response(model, tokenizer, "The capital of France is")
print(response)
```

### Using Chat Templates

Instruction-tuned models expect input formatted according to a specific **chat template**. The Qwen2.5 Instruct model uses a structured format with system, user, and assistant roles. The `tokenizer.apply_chat_template()` method handles this automatically:

```{code-block} python
def chat(model, tokenizer, user_message, system_message = "You are a helpful assistant.",
         max_new_tokens = 256, temperature = 0.7, top_p = 0.9, top_k = 50):
    """
    Send a message using the model's chat template.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        user_message: The user's message
        system_message: The system prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (use 0.0 for greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (limits vocabulary at each step)
    
    Returns:
        str: The assistant's response
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = True
    )
    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
    
    gen_kwargs = dict(
        max_new_tokens = max_new_tokens,
        pad_token_id = tokenizer.eos_token_id,
    )
    
    if temperature < 0.01:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens = True
    )
    return response


# Test
response = chat(model, tokenizer, "Explain what a neural network is in 2 sentences.")
print(response)
```

### Generation Parameters

The behavior of text generation can be controlled through several important parameters:

**Temperature ($\tau$):** Controls the randomness of the output. Given logits $z_i$ for each token $i$, the probability is:

$$
P(x_t = i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

- $\tau \to 0$: The distribution becomes peaked — the model always picks the most likely token (deterministic, **greedy** decoding).
- $\tau = 1$: Standard sampling from the learned distribution.
- $\tau > 1$: Flatter distribution — more random and creative outputs.

**Top-p (Nucleus Sampling):** Instead of considering all tokens, only consider the smallest set of tokens whose cumulative probability exceeds $p$. For example, `top_p = 0.9` means the model samples from tokens that together account for 90% of the probability mass.

**Top-k:** Only consider the $k$ tokens with the highest probability. This acts as a hard cutoff that prevents sampling from very unlikely tokens (which can cause numerical issues with small models).

**Max New Tokens:** The maximum number of tokens the model will generate before stopping.

```{code-block} python
# Compare different temperature values
prompt = "Write a creative name for a coffee shop."

print("=== Temperature 0.3 (near deterministic) ===")
for i in range(3):
    r = chat(model, tokenizer, prompt, max_new_tokens = 30, temperature = 0.3)
    print(f"  Run {i + 1}: {r.strip()}")

print("\n=== Temperature 0.7 (balanced) ===")
for i in range(3):
    r = chat(model, tokenizer, prompt, max_new_tokens = 30, temperature = 0.7)
    print(f"  Run {i + 1}: {r.strip()}")

print("\n=== Temperature 1.2 (creative) ===")
for i in range(3):
    r = chat(model, tokenizer, prompt, max_new_tokens = 30, temperature = 1.2)
    print(f"  Run {i + 1}: {r.strip()}")
```

---

## Part 2: Prompt Engineering Fundamentals

### What Is Prompt Engineering?

**Prompt engineering** is the practice of designing and refining the textual input (prompt) given to a language model in order to elicit the most accurate, relevant, and useful response. Since LLMs are general-purpose models, the quality of their output depends critically on *how* we ask them to perform a task.

A well-crafted prompt can dramatically improve performance without changing the model or its weights — it is a form of **in-context learning**, where the model adapts its behavior based solely on the information provided in the input.

### The Anatomy of a Good Prompt

An effective prompt typically contains some or all of the following components:

1. **Role / Persona**: Define who the model should act as (e.g., "You are an expert Python programmer").
2. **Task Description**: Clearly state what you want the model to do.
3. **Context / Background**: Provide any relevant information the model needs.
4. **Input Data**: The specific data to process.
5. **Output Format**: Specify how the response should be structured (e.g., JSON, bullet points, a table).
6. **Constraints**: Any limitations or rules (e.g., "Answer in 3 sentences or fewer").
7. **Examples**: One or more input-output examples to guide the model (few-shot learning).

### Zero-Shot Prompting

In **zero-shot** prompting, we give the model a task description without any examples. The model must rely entirely on its pre-trained knowledge.

```{code-block} python
# Zero-shot sentiment analysis
zero_shot_prompt = """Classify the sentiment of the following review as POSITIVE, NEGATIVE, or NEUTRAL.

Review: "The food was absolutely delicious and the service was impeccable. 
Best restaurant experience I've had in years!"

Sentiment:"""

response = chat(model, tokenizer, zero_shot_prompt, max_new_tokens = 10, temperature = 0.0)
print(f"Sentiment: {response.strip()}")
```

### Few-Shot Prompting

In **few-shot** prompting, we provide one or more examples (demonstrations) before the actual query. This helps the model understand the expected format and behavior.

```{code-block} python
# Few-shot sentiment analysis
few_shot_prompt = """Classify the sentiment of each review as POSITIVE, NEGATIVE, or NEUTRAL.

Review: "Great product, works exactly as described!"
Sentiment: POSITIVE

Review: "Terrible quality. Broke after one day."
Sentiment: NEGATIVE

Review: "It's okay, nothing special but gets the job done."
Sentiment: NEUTRAL

Review: "I was hesitant at first, but this turned out to be one of the best purchases I've ever made. Highly recommend!"
Sentiment:"""

response = chat(model, tokenizer, few_shot_prompt, max_new_tokens = 10, temperature = 0.0)
print(f"Sentiment: {response.strip()}")
```

### Chain-of-Thought (CoT) Prompting

**Chain-of-Thought** prompting encourages the model to reason step by step before arriving at a final answer. This technique significantly improves performance on tasks that require logical reasoning, arithmetic, or multi-step problem solving.

The key idea is simple: instead of asking for a direct answer, we ask the model to "think step by step" or provide explicit reasoning examples.

```{code-block} python
# Without Chain-of-Thought
direct_prompt = """If a store sells apples for 2 euros each and oranges for 3 euros each,
and Maria buys 4 apples and 5 oranges, how much does she pay in total?

Answer:"""

response_direct = chat(model, tokenizer, direct_prompt, max_new_tokens = 50, temperature = 0.0)
print(f"Direct answer: {response_direct.strip()}")

# With Chain-of-Thought
cot_prompt = """If a store sells apples for 2 euros each and oranges for 3 euros each,
and Maria buys 4 apples and 5 oranges, how much does she pay in total?

Let's solve this step by step:"""

response_cot = chat(model, tokenizer, cot_prompt, max_new_tokens = 150, temperature = 0.0)
print(f"\nChain-of-Thought answer:\n{response_cot.strip()}")
```

### Structured Output Prompting

For integration into software systems, we often need the model to return output in a specific structured format such as **JSON**. This is particularly important when building agents that must parse model outputs programmatically.

```{code-block} python
import json

structured_prompt = """Extract the following information from the text and return it as a JSON object
with the keys: "name", "age", "city", and "occupation".

Text: "My name is Carlos Garcia. I am 34 years old and I work as a software engineer in Barcelona."

Return ONLY the JSON object, nothing else:"""

response = chat(model, tokenizer, structured_prompt, max_new_tokens = 100, temperature = 0.0)
print(f"Raw response: {response.strip()}")

# Try to parse the JSON
try:
    data = json.loads(response.strip())
    print(f"\nParsed JSON:")
    for key, value in data.items():
        print(f"  {key}: {value}")
except json.JSONDecodeError:
    print("\nFailed to parse as JSON. The model did not follow the format exactly.")
```

### Role Prompting

By assigning a specific **role** or **persona** to the model through the system message, we can influence its tone, expertise level, and style of response.

```{code-block} python
# Same question, different roles
question = "Explain what an API is."

# Role 1: Expert for developers
response_expert = chat(
    model, tokenizer, question,
    system_message = "You are a senior software architect. Give precise, technical explanations.",
    max_new_tokens = 150, temperature = 0.3
)
print("=== Expert Explanation ===")
print(response_expert.strip())

# Role 2: Teacher for beginners
response_teacher = chat(
    model, tokenizer, question,
    system_message = "You are a patient teacher explaining concepts to a 10-year-old. Use simple words and analogies.",
    max_new_tokens = 150, temperature = 0.3
)
print("\n=== Beginner Explanation ===")
print(response_teacher.strip())
```

---

## Part 3: Prompt Chaining

### Breaking Complex Tasks into Steps

**Prompt chaining** involves breaking a complex task into a sequence of simpler sub-tasks, where the output of one prompt becomes the input to the next. This is a precursor to the agent architectures we will build in Part 3 of this practical session.

The advantages of prompt chaining over a single monolithic prompt are:

- Each step is simpler and more reliable.
- Intermediate results can be inspected and validated.
- Different steps can use different prompting strategies.
- Error propagation can be detected and handled.

```{code-block} python
def prompt_chain_analysis(model, tokenizer, text):
    """
    Analyze a text through a chain of prompts:
    1. Summarize the text
    2. Extract key entities
    3. Determine the overall sentiment
    4. Generate a final report
    """
    # Step 1: Summarize
    summary = chat(
        model, tokenizer,
        f"Summarize the following text in 2 sentences:\n\n{text}",
        max_new_tokens = 100, temperature = 0.3
    )
    print(f"Step 1 — Summary:\n{summary.strip()}\n")
    
    # Step 2: Extract entities
    entities = chat(
        model, tokenizer,
        f"From the following summary, list the key entities (people, organizations, locations) as a comma-separated list:\n\n{summary}",
        max_new_tokens = 100, temperature = 0.0
    )
    print(f"Step 2 — Entities:\n{entities.strip()}\n")
    
    # Step 3: Sentiment
    sentiment = chat(
        model, tokenizer,
        f"What is the overall sentiment of this text? Answer with one word (POSITIVE, NEGATIVE, or NEUTRAL):\n\n{summary}",
        max_new_tokens = 10, temperature = 0.0
    )
    print(f"Step 3 — Sentiment:\n{sentiment.strip()}\n")
    
    # Step 4: Final report
    report = chat(
        model, tokenizer,
        f"""Generate a brief analytical report given the following information:
Summary: {summary}
Key Entities: {entities}
Sentiment: {sentiment}

Write a 3-sentence report:""",
        max_new_tokens = 150, temperature = 0.3
    )
    print(f"Step 4 — Final Report:\n{report.strip()}")
    
    return {
        "summary": summary.strip(),
        "entities": entities.strip(),
        "sentiment": sentiment.strip(),
        "report": report.strip()
    }


# Test with a sample text
sample_text = """
The European Commission announced today a landmark agreement on artificial intelligence regulation.
The AI Act, which has been under negotiation for over three years, establishes a risk-based framework
for AI systems deployed across the European Union. Commissioner Thierry Breton stated that this
regulation will serve as a global benchmark. Technology companies including Google, Microsoft, and
Meta have expressed mixed reactions, with some praising the clarity it brings while others worry
about potential impacts on innovation. The regulation is expected to take full effect by 2026.
"""

result = prompt_chain_analysis(model, tokenizer, sample_text)
```

---

## Part 4: Evaluating Prompt Strategies

### Building an Evaluation Framework

To compare different prompting strategies systematically, we need a small evaluation benchmark. We will create a set of tasks with known correct answers and measure the accuracy of different prompting approaches.

```{code-block} python
# Define a small evaluation dataset
eval_dataset = [
    {
        "question": "What is the capital of Japan?",
        "expected": "Tokyo",
        "category": "factual"
    },
    {
        "question": "If a shirt costs 25 euros and is on sale for 20% off, what is the sale price?",
        "expected": "20",
        "category": "math"
    },
    {
        "question": "Sort these numbers from smallest to largest: 7, 2, 9, 1, 5",
        "expected": "1, 2, 5, 7, 9",
        "category": "reasoning"
    },
    {
        "question": "What is the sentiment of this sentence: 'I absolutely hated every minute of that terrible movie.'",
        "expected": "NEGATIVE",
        "category": "classification"
    },
    {
        "question": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        "expected": "9",
        "category": "reasoning"
    },
    {
        "question": "Translate to French: 'The weather is beautiful today.'",
        "expected": "Il fait beau aujourd'hui",
        "category": "translation"
    },
]

def evaluate_strategy(model, tokenizer, dataset, prompt_fn, strategy_name):
    """
    Evaluate a prompting strategy on the dataset.
    """
    results = []
    
    for item in dataset:
        prompt = prompt_fn(item["question"])
        response = chat(model, tokenizer, prompt, max_new_tokens = 50, temperature = 0.0)
        response_clean = response.strip().lower()
        expected_clean = item["expected"].lower()
        
        correct = expected_clean in response_clean
        
        results.append({
            "question": item["question"],
            "expected": item["expected"],
            "response": response.strip(),
            "correct": correct,
            "category": item["category"]
        })
    
    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\n{'=' * 60}")
    print(f"Strategy: {strategy_name}")
    print(f"Overall Accuracy: {accuracy:.1%}")
    print(f"{'=' * 60}")
    
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  [{status}] {r['category']:15s} | Expected: {r['expected']:20s} | Got: {r['response'][:40]}")
    
    return {"strategy": strategy_name, "accuracy": accuracy, "details": results}
```

### Comparing Strategies

```{code-block} python
# Strategy 1: Direct (zero-shot, no special instructions)
def direct_prompt(question):
    return f"{question}\n\nAnswer concisely:"

# Strategy 2: Instructed (with clear formatting instructions)
def instructed_prompt(question):
    return f"""Answer the following question. Be concise and give ONLY the answer, no explanation.

Question: {question}

Answer:"""

# Strategy 3: Chain-of-Thought
def cot_prompt(question):
    return f"""Answer the following question. Think step by step, then provide your final answer 
on the last line prefixed with "ANSWER:".

Question: {question}

Step-by-step reasoning:"""


# Run evaluation
results_direct = evaluate_strategy(model, tokenizer, eval_dataset, direct_prompt, "Direct")
results_instructed = evaluate_strategy(model, tokenizer, eval_dataset, instructed_prompt, "Instructed")
results_cot = evaluate_strategy(model, tokenizer, eval_dataset, cot_prompt, "Chain-of-Thought")
```

### Visualizing Results

```{code-block} python
import matplotlib.pyplot as plt
import numpy as np

strategies = [results_direct, results_instructed, results_cot]
strategy_names = [r["strategy"] for r in strategies]
accuracies = [r["accuracy"] for r in strategies]

fig, axes = plt.subplots(1, 2, figsize = (14, 5))

# Plot 1: Overall accuracy
bars = axes[0].bar(strategy_names, accuracies, color = ['#e41a1c', '#377eb8', '#4daf4a'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Overall Accuracy by Prompting Strategy')
axes[0].set_ylim(0, 1.0)
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                 f'{acc:.0%}', ha = 'center', fontweight = 'bold')

# Plot 2: Per-category accuracy
categories = list(set(item["category"] for item in eval_dataset))
x = np.arange(len(categories))
width = 0.25

for i, result in enumerate(strategies):
    cat_acc = {}
    for detail in result["details"]:
        cat = detail["category"]
        if cat not in cat_acc:
            cat_acc[cat] = []
        cat_acc[cat].append(detail["correct"])
    
    cat_accuracies = [np.mean(cat_acc.get(cat, [0])) for cat in categories]
    axes[1].bar(x + i * width, cat_accuracies, width, label = result["strategy"])

axes[1].set_xlabel('Category')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy by Category and Strategy')
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(categories, rotation = 45, ha = 'right')
axes[1].legend()
axes[1].set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig("prompt_strategy_comparison.png", dpi = 150)
plt.show()
```
---

## Exercises

### Exercise 1: Model Exploration

#### Objectives

The objective of this exercise is to familiarize yourself with the Hugging Face `transformers` library, understand tokenization, and explore how generation parameters affect model output.

#### Task Description

1. **Load the model**: Load `Qwen/Qwen2.5-1.5B-Instruct` using `bfloat16` precision. Print the model's memory footprint and the number of parameters. Verify that the logits do not contain `nan` values.

2. **Tokenization analysis**: Choose 5 sentences of varying complexity (short, long, with numbers, with special characters, in different languages). For each sentence:
   - Tokenize it and report the number of tokens.
   - Print the individual tokens to understand how the tokenizer splits text.
   - Discuss any surprising tokenization behavior you observe.

3. **Temperature experiment**: Choose a creative prompt (e.g., "Write a one-sentence story about a robot"). Generate 10 responses at each of the following temperatures: 0.3, 0.5, 0.7, 1.0, 1.2. For each temperature:
   - Report the diversity of responses (how many unique responses out of 10).
   - Compute the average response length.
   - Provide a qualitative assessment of the output quality.

4. **Top-p vs Top-k**: Using the same prompt, compare `top_p = [0.5, 0.9, 1.0]` and `top_k = [10, 50, 200]`. Generate 5 responses for each setting and describe the differences.

#### Deliverables

- Complete code in a Jupyter Notebook
- A table summarizing the diversity and quality of responses at each temperature
- A brief discussion (2–3 paragraphs) on how generation parameters affect output quality and when you would choose each setting

---

### Exercise 2: Prompt Engineering Challenge

#### Objectives

The objective of this exercise is to practice and compare different prompt engineering techniques on a variety of NLP tasks.

#### Task Description

You must implement and evaluate **four** prompting strategies on the following **five** tasks. For each task, design the best prompt you can for each strategy.

**Prompting Strategies:**

1. **Zero-shot**: No examples, just a task description.
2. **Few-shot** (3 examples): Provide 3 input-output examples before the query.
3. **Chain-of-Thought**: Ask the model to reason step by step.
4. **Role + Structured Output**: Assign a persona and request output in JSON format.

**Tasks:**

- **Task A — Named Entity Recognition**: Given a sentence, extract all person names, locations, and organizations. Use at least 5 test sentences.
- **Task B — Text Classification**: Classify news headlines into categories (SPORTS, POLITICS, TECHNOLOGY, ENTERTAINMENT, SCIENCE). Use at least 10 test headlines.
- **Task C — Mathematical Word Problems**: Solve arithmetic word problems. Use at least 5 problems of increasing difficulty.
- **Task D — Code Generation**: Given a natural language description, generate a Python function. Use at least 3 different descriptions (e.g., "Write a function that checks if a number is prime").
- **Task E — Summarization**: Summarize a paragraph into a single sentence. Use at least 3 different paragraphs from different domains.

#### Deliverables

- A Jupyter Notebook with all implementations
- A summary table showing accuracy/quality for each strategy × task combination (20 cells)
- A discussion (3–4 paragraphs) analyzing:
  - Which strategies worked best for which tasks and why
  - The limitations you encountered with a small model
  - How you iterated on your prompts to improve results

#### Evaluation Criteria

- Correctness and completeness of the implementations
- Quality and creativity of the prompts designed
- Rigor of the evaluation (meaningful test cases, fair comparisons)
- Depth of analysis in the discussion

---

### Exercise 3: Building a Prompt-Based Application

#### Objectives

The objective of this exercise is to build a complete, functional application powered by an LLM using only prompt engineering techniques. This exercise bridges the gap between isolated prompting experiments and the agent-based systems you will build in Parts 2 and 3.

#### Task Description

Design and implement a **multi-step text analysis pipeline** that processes a given document through the following stages:

1. **Language Detection**: Determine the language of the input text.
2. **Summary Generation**: Produce a concise summary (2–3 sentences).
3. **Keyword Extraction**: Extract the 5 most important keywords.
4. **Sentiment Analysis**: Classify the overall sentiment with a confidence score.
5. **Question Generation**: Generate 3 questions that could be answered by the text.

Your pipeline must:

- Chain the prompts together, using output from earlier stages as context for later ones.
- Handle errors gracefully (e.g., if JSON parsing fails, retry with a more explicit prompt).
- Return a structured report (as a Python dictionary) with all results.
- Include timing information for each stage.

#### Test Data

Run your pipeline on **at least 3 documents** from different domains (e.g., a news article, a scientific abstract, a product review). You can copy-paste real text or create realistic synthetic examples.

#### Deliverables

- Complete, documented Python code in a Jupyter Notebook
- The structured output (dictionary/JSON) for each test document
- A timing breakdown showing how long each stage takes
- A brief report (2–3 paragraphs) discussing:
  - The challenges of chaining prompts together
  - How output quality from early stages affects later stages
  - Potential improvements you would make if using a larger model

#### Evaluation Criteria

- Robustness of the pipeline (error handling, edge cases)
- Quality of the prompt design at each stage
- Code organization and documentation
- Thoughtfulness of the analysis

---

## References

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
2. Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS.
3. Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS.
4. Wang, X., et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR.
5. Qwen Team (2024). *Qwen2.5 Technical Report*. arXiv.
6. Hugging Face Documentation. *Transformers Library*. https://huggingface.co/docs/transformers

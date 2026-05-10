# Intelligent Agents

## Practical Session 3 — Part 3: Multi-Agent Systems with LangChain

### Introduction

In Parts 1 and 2, we built LLM interactions and RAG pipelines by writing everything from scratch. While this is valuable for understanding the fundamentals, real-world projects use **frameworks** that provide standardized abstractions for models, tools, chains, and agents. **LangChain** is the most widely adopted open-source framework for building LLM-powered applications.

In this final part, we will learn how to:

1. Wrap our local Hugging Face model as a LangChain-compatible LLM.
2. Define **tools** — Python functions that an agent can invoke.
3. Build **chains** — sequential compositions of prompts and models.
4. Construct **agents** — LLM-powered systems that reason, decide which tools to use, and act iteratively.
5. Orchestrate **multi-agent pipelines** where specialized agents collaborate on a task.

### Prerequisites

```{code-block} bash
pip install torch transformers accelerate sentencepiece protobuf
pip install langchain langchain-huggingface langchain-community
```

```{note}
- We reuse the same **`Qwen/Qwen2.5-1.5B-Instruct`** model in `bfloat16` from Parts 1 and 2.
- **`langchain-huggingface`** is the official partner package that integrates Hugging Face models into LangChain.
- No external APIs, GPU, or paid services are required.
- Small models (1.5B) have **limited ability to follow the ReAct agent format** reliably. This is expected and is itself an important learning outcome — understanding where small models fall short motivates the use of larger models or fine-tuned agents in production.
```

---

## Part 1: LangChain Fundamentals

### Wrapping a Local Model

LangChain provides `HuggingFacePipeline` to wrap any local Hugging Face model as a LangChain-compatible LLM, and `ChatHuggingFace` to add chat capabilities on top.

```{code-block} python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if torch.cuda.is_available():
    dtype = torch.bfloat16
    device = 0
else:
    dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32
    device = -1  # CPU

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype = dtype)

# Create a transformers pipeline
pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 256,
    do_sample = False,
    return_full_text = False,
)

# Wrap as LangChain LLM
llm = HuggingFacePipeline(pipeline = pipe)

# Wrap as Chat Model (supports system/user/assistant messages)
chat_model = ChatHuggingFace(llm = llm)

# Test
response = chat_model.invoke("What is the capital of France?")
print(response.content)
```

### Prompt Templates

LangChain's `PromptTemplate` and `ChatPromptTemplate` provide a structured way to build prompts with variables.

```{code-block} python
from langchain_core.prompts import ChatPromptTemplate

# Define a reusable prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions concisely."),
    ("human", "{question}"),
])

# Create a chain: prompt → model
chain = prompt | chat_model

# Invoke
response = chain.invoke({"question": "What is machine learning?"})
print(response.content)
```

### Chains: Composing Steps

The **LCEL (LangChain Expression Language)** uses the pipe operator (`|`) to compose steps into a chain. Each step transforms the data and passes it to the next.

```{code-block} python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Chain: prompt → model → parse output to string
summarize_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize the following text in one sentence."),
        ("human", "{text}"),
    ])
    | chat_model
    | StrOutputParser()
)

text = """
Artificial intelligence has transformed numerous industries over the past decade.
Healthcare systems now use AI for early disease detection. Financial institutions
employ machine learning for fraud detection. The transportation sector is being
reshaped by autonomous vehicles.
"""

summary = summarize_chain.invoke({"text": text})
print(f"Summary: {summary}")
```

### Sequential Chains

We can compose multiple chains into a pipeline where the output of one feeds into the next:

```{code-block} python
# Chain 1: Summarize
summarize = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize this text in 2 sentences."),
        ("human", "{text}"),
    ])
    | chat_model
    | StrOutputParser()
)

# Chain 2: Extract keywords from a summary
extract_keywords = (
    ChatPromptTemplate.from_messages([
        ("system", "Extract 5 keywords from this text. Return only the keywords, comma-separated."),
        ("human", "{summary}"),
    ])
    | chat_model
    | StrOutputParser()
)

# Run sequentially
summary = summarize.invoke({"text": text})
keywords = extract_keywords.invoke({"summary": summary})

print(f"Summary: {summary}")
print(f"Keywords: {keywords}")
```

---

## Part 2: Tools and Agents

### Defining Custom Tools

A **tool** is a function that an agent can call. In LangChain, tools are defined using the `@tool` decorator.

```{code-block} python
from langchain_core.tools import tool
from datetime import datetime
import math

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result. Example input: '2 + 3 * 4'"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters"
        return str(round(eval(expression), 4))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_date(dummy: str = "") -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def company_lookup(query: str) -> str:
    """Search the company database for information about employees, revenue, or products."""
    db = {
        "revenue_2024": "58.7 million euros",
        "employees": "342 employees",
        "ceo": "Elena Vidal",
        "product": "CloudSync — enterprise cloud management platform",
        "founded": "2018 in Barcelona",
    }
    query_lower = query.lower()
    results = [f"{k}: {v}" for k, v in db.items() if query_lower in k or query_lower in v.lower()]
    return "\n".join(results) if results else "No matching information found."


# Test tools directly
print(calculator.invoke("15 * 3.5"))
print(get_current_date.invoke(""))
print(company_lookup.invoke("revenue"))
```

### Building a ReAct Agent

The **ReAct** (Reasoning + Acting) pattern is the standard approach for LLM agents. The model iterates through a loop of: **Thought** → **Action** (tool call) → **Observation** (tool result) → **Thought** → ... until it has enough information to produce a final answer.

```{code-block} python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Load the standard ReAct prompt template
react_prompt = hub.pull("hwchase17/react")

# Define available tools
tools = [calculator, get_current_date, company_lookup]

# Create the agent
agent = create_react_agent(chat_model, tools, react_prompt)

# Create an executor (handles the agent loop)
agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True,          # Print the reasoning trace
    max_iterations = 5,      # Limit iterations to prevent infinite loops
    handle_parsing_errors = True,  # Gracefully handle malformed output
)

# Test
result = agent_executor.invoke({"input": "What is the company's revenue for 2024?"})
print(f"\nFinal Answer: {result['output']}")
```

```{important}
**Small models and ReAct.** A 1.5B parameter model will frequently fail to follow the ReAct format correctly — it may skip the `Action:` line, hallucinate tool names, or enter infinite loops. This is expected. The `handle_parsing_errors = True` flag helps recover from some of these issues. In production, ReAct agents require models with 7B+ parameters or models specifically fine-tuned for tool calling. Understanding this limitation is an important learning objective.
```

### A Simpler Agent Pattern for Small Models

When the ReAct format is too demanding for the model, we can implement a **manual agent loop** that is more forgiving:

```{code-block} python
class SimpleAgent:
    """
    A lightweight agent that works reliably with small models.
    Instead of requiring strict ReAct formatting, it uses explicit
    prompting to decide tool usage step by step.
    """
    
    def __init__(self, chat_model, tools, system_prompt):
        self.chat_model = chat_model
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
    
    def _decide_tool(self, question):
        """Ask the model which tool to use."""
        tool_descriptions = "\n".join(
            [f"- {name}: {t.description}" for name, t in self.tools.items()]
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", f"""Available tools:
{tool_descriptions}
- none: Answer directly without tools.

Question: {question}

Which tool should I use? Respond with ONLY the tool name."""),
        ])
        chain = prompt | self.chat_model | StrOutputParser()
        return chain.invoke({"question": question}).strip().lower()
    
    def _call_tool(self, tool_name, question):
        """Call a tool with the question as input."""
        if tool_name in self.tools:
            return self.tools[tool_name].invoke(question)
        return None
    
    def _generate_answer(self, question, tool_result = None):
        """Generate a final answer, optionally using tool output."""
        if tool_result:
            context = f"Tool result: {tool_result}\n\nUsing this information, answer: {question}"
        else:
            context = question
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", context),
        ])
        chain = prompt | self.chat_model | StrOutputParser()
        return chain.invoke({})
    
    def run(self, question, verbose = True):
        """Run the agent on a question."""
        # Step 1: Decide tool
        tool_choice = self._decide_tool(question)
        if verbose:
            print(f"  Tool decision: {tool_choice}")
        
        # Step 2: Use tool if needed
        tool_result = None
        if tool_choice != "none" and tool_choice in self.tools:
            tool_result = self._call_tool(tool_choice, question)
            if verbose:
                print(f"  Tool result: {tool_result}")
        
        # Step 3: Generate answer
        answer = self._generate_answer(question, tool_result)
        if verbose:
            print(f"  Answer: {answer}")
        
        return answer


# Create and test the simple agent
agent = SimpleAgent(
    chat_model = chat_model,
    tools = [calculator, get_current_date, company_lookup],
    system_prompt = "You are a helpful business assistant. Use tools when needed.",
)

for q in ["What is the company's revenue?", "What is 125 * 48?", "What day is today?"]:
    print(f"\nQ: {q}")
    agent.run(q)
```

---

## Part 3: Multi-Agent Orchestration

### Router Pattern

A **router** uses one agent to classify the incoming request and dispatch it to the appropriate specialist agent.

```{code-block} python
class MultiAgentRouter:
    """Route queries to specialized agents based on a coordinator's decision."""
    
    def __init__(self, chat_model, specialists):
        """
        Args:
            chat_model: The LangChain chat model
            specialists: Dict of {name: {"prompt": str, "tools": list}}
        """
        self.chat_model = chat_model
        self.specialists = specialists
    
    def route(self, query):
        """Decide which specialist handles this query."""
        names = ", ".join(self.specialists.keys())
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a router. Choose the best specialist from: {names}. Respond with ONLY the name."),
            ("human", query),
        ])
        chain = prompt | self.chat_model | StrOutputParser()
        decision = chain.invoke({}).strip()
        
        # Match to a specialist
        for name in self.specialists:
            if name.lower() in decision.lower():
                return name
        return list(self.specialists.keys())[0]  # fallback
    
    def run(self, query, verbose = True):
        """Route and execute."""
        chosen = self.route(query)
        spec = self.specialists[chosen]
        
        if verbose:
            print(f"  Routed to: {chosen}")
        
        agent = SimpleAgent(
            chat_model = self.chat_model,
            tools = spec.get("tools", []),
            system_prompt = spec["prompt"],
        )
        return agent.run(query, verbose = verbose)
```

### Sequential Pipeline

A **pipeline** passes output from one agent to the next, where each agent has a different role.

```{code-block} python
class AgentPipeline:
    """Execute agents in sequence, passing each output to the next."""
    
    def __init__(self, stages):
        """
        Args:
            stages: List of dicts with "name", "prompt", and optionally "tools"
        """
        self.stages = stages
    
    def run(self, initial_input, chat_model, verbose = True):
        current = initial_input
        results = {}
        
        for stage in self.stages:
            if verbose:
                print(f"\n{'─'*40}")
                print(f"▶ {stage['name']}")
            
            agent = SimpleAgent(
                chat_model = chat_model,
                tools = stage.get("tools", []),
                system_prompt = stage["prompt"],
            )
            output = agent.run(current, verbose = verbose)
            results[stage["name"]] = output
            
            current = f"Previous step ({stage['name']}) output:\n{output}\n\nContinue with the next step."
        
        return results
```

### Example: Automated Analysis Report

```{code-block} python
pipeline = AgentPipeline([
    {
        "name": "Researcher",
        "prompt": "You are a data researcher. Gather relevant facts using the tools available. List key findings.",
        "tools": [company_lookup, calculator],
    },
    {
        "name": "Writer",
        "prompt": "You are a report writer. Given research findings, write a professional 3-paragraph summary.",
    },
    {
        "name": "Reviewer",
        "prompt": "You are an editor. Review the report for clarity and completeness. Suggest improvements if needed.",
    },
])

results = pipeline.run(
    "Write a brief report about the company's profile and financials.",
    chat_model = chat_model,
)
```

---

## Exercises

### Exercise 1: LangChain Agent with Custom Tools

#### Objectives

Build an agent using LangChain that can answer questions about a fictional scenario by combining reasoning with tool calls.

#### Task Description

1. **Define at least 4 custom tools** using the `@tool` decorator for a domain of your choice (e.g., a restaurant booking system, a university helpdesk, a travel planner). Examples:
   - `check_availability(date)`: Check if a date has openings
   - `get_menu(category)`: Return menu items for a category
   - `calculate_total(items)`: Calculate the bill
   - `make_reservation(details)`: Confirm a booking

2. **Build two versions of the agent**:
   - A LangChain `create_react_agent` with `AgentExecutor` (may fail with the small model — document the failures)
   - A `SimpleAgent` using the manual pattern shown in the tutorial

3. **Test with at least 6 queries**, including queries that require tool use and queries that do not. Compare the two agent implementations.

#### Deliverables

- A Jupyter Notebook with both implementations
- A comparison table showing success/failure for each agent × query
- A discussion (2–3 paragraphs) analyzing when ReAct works vs. when the simple agent is more reliable, and what model size you think would be needed for reliable ReAct behavior

---
#### Deliverables

- A Jupyter Notebook with the complete implementation
- Results for all test queries/topics
- A timing breakdown for each agent's execution
- A discussion (2–3 paragraphs) on orchestration challenges and how output quality degrades through the pipeline

#### Evaluation Criteria

- Correctness and robustness of the implementations
- Quality and diversity of tools and test cases
- Honest analysis of where the small model succeeds and fails
- Code organization and documentation

---

## References

1. Yao, S., et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR.
2. Schick, T., et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS.
3. Chase, H. (2022). *LangChain Documentation*. https://docs.langchain.com
4. LangChain Hugging Face Integration. https://huggingface.co/blog/langchain
5. Wu, Q., et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. arXiv.
6. Wang, L., et al. (2024). *A Survey on Large Language Model-Based Autonomous Agents*. Frontiers of Computer Science.
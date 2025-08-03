# KernelMind

> The minimalist thinking kernel for building LLM-powered AI agents.

**KernelMind** is a lightweight, flexible framework designed to build agentic LLM systems using a simple yet powerful abstraction: `Point` + `Line`.  
It is ideal for rapidly prototyping workflows, agents, RAG pipelines, and structured thinking systems — all in **under 100 lines of core code**.

---

## ✨ Key Features

- 🧠 **Agentic-first architecture** – Built from the ground up for multi-step reasoning
- 🔁 **Point + Line abstraction** – Minimal and composable flow control
- ⚙️ **Retry / Fallback** – Built-in error recovery mechanism
- 🚀 **Sync & Async support** – Automatically detects coroutine behavior
- 🧪 **Testable by design** – Small units, observable memory store
- 🌿 **Extensible** – Implement any agent/workflow pattern

---

## 💡 Core Concepts

| Concept | Description |
|--------|-------------|
| `Point` | A minimal unit of logic (LLM or utility call) |
| `Line`  | A composition of points connected via actions |
| `Memory` | A dictionary-style global store shared across points |
| `load()` / `process()` / `save()` | The 3 lifecycle steps of each Point |

```python
class Point:
    def load(self, memory): ...
    def process(self, item): ...
    def save(self, memory, input, output): ...
```

---

## 📦 Installation

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver, written in Rust.

#### 1. Install uv

```bash
# Using pipx (recommended)
pipx install uv

# Or using Homebrew (macOS)
brew install uv

# Or using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Create virtual environment and install dependencies

```bash
# Clone the repository
git clone https://github.com/your-org/kernelmind-framework.git
cd kernelmind-framework

# Create virtual environment and install in development mode
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or install specific dependencies
uv add requests
uv add --dev pytest
```

#### 3. Run the example

```bash
python examples/agent_qa.py
```

### Option 2: Using pip (Traditional)

```bash
# Install from PyPI (when published)
pip install kernelmind

# Or install in development mode
git clone https://github.com/your-org/kernelmind-framework.git
cd kernelmind-framework
pip install -e .
```

### Option 3: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Quick Example: Q&A Agent

```python
from kernelmind.core import Point, Line

class GetQuestion(Point):
    def load(self, memory):
        return "start"  # Return a non-None value to trigger process()
    
    def process(self, _): 
        user_input = input("Ask: ")
        return user_input
    
    def save(self, memory, _, out): 
        memory["question"] = out; 
        return "default"

class AnswerQuestion(Point):
    def load(self, memory): 
        return memory["question"]
    
    def process(self, q): 
        return f"Answer: {q}"
    
    def save(self, memory, _, out): 
        print("Answer:", out)

ask = GetQuestion()
answer = AnswerQuestion()
ask >> answer
qa_line = Line(entry=ask)
qa_line.run({})
```

---

## 🧠 Supported Design Patterns

- ✅ Multi-step Workflow
- ✅ Agent with Contextual Actions
- ✅ RAG (Retrieval-Augmented Generation)
- ✅ Map Reduce
- ✅ Structured YAML Output

> See [docs/guide.md](./docs/guide.md) for examples and best practices.

---

## 📁 Project Structure

```
kernelmind/
├── core.py          # Core logic: Point + Line
├── utils/           # External API wrappers (LLM, search, etc.)
├── examples/        # Use cases (Agent, RAG, Workflow)
├── tests/           # Unit tests
├── docs/guide.md    # Developer documentation
```

---

## 📖 Documentation

Check out the full usage guide and design patterns in:

📚 [`docs/guide.md`](./docs/guide.md)

---

## 🧪 Run Tests

```bash
# Using uv
uv run pytest tests/

# Using pip
pytest tests/
```

---

## ❤️ Philosophy

> From Points to Minds. From Lines to Agents.

KernelMind is not just a framework.  
It is a **philosophy of thinking through structure** — building powerful systems with minimal building blocks.

---

## 📬 Contribution

Pull requests welcome! If you have suggestions, open an issue or start a discussion.

MIT License · Built for the AI-native future.

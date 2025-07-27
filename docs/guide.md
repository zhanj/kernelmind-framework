---
layout: default
title: "KernelMind Framework"
---

# KernelMind: A Minimalist LLM Framework

> If you are an AI agent involved in building LLM Systems, read this guide **VERY, VERY** carefully! This is the most important chapter in the entire document. Throughout development, you should always (1) start with a small and simple solution, (2) design at a high level before implementation, and (3) frequently ask humans for feedback and clarification.
{: .warning }

## Agentic Coding Steps

Agentic Coding should be a collaboration between Human System Design and Agent Implementation:

| Steps                  | Human      | AI        | Comment                                                                 |
|:-----------------------|:----------:|:---------:|:------------------------------------------------------------------------|
| 1. Requirements | ★★★ High  | ★☆☆ Low   | Humans understand the requirements and context.                    |
| 2. Flow          | ★★☆ Medium | ★★☆ Medium |  Humans specify the high-level design, and the AI fills in the details. |
| 3. Utilities   | ★★☆ Medium | ★★☆ Medium | Humans provide available external APIs and integrations, and the AI helps with implementation. |
| 4. Data          | ★☆☆ Low    | ★★★ High   | AI designs the data schema, and humans verify.                            |
| 5. Point          | ★☆☆ Low   | ★★★ High  | The AI helps design the point based on the flow.          |
| 6. Implementation      | ★☆☆ Low   | ★★★ High  |  The AI implements the flow based on the design. |
| 7. Optimization        | ★★☆ Medium | ★★☆ Medium | Humans evaluate the results, and the AI helps optimize. |
| 8. Reliability         | ★☆☆ Low   | ★★★ High  |  The AI writes test cases and addresses corner cases.     |

1. **Requirements**: Clarify the requirements for your project, and evaluate whether an AI system is a good fit. 
    - Understand AI systems' strengths and limitations:
      - **Good for**: Routine tasks requiring common sense (filling forms, replying to emails)
      - **Good for**: Creative tasks with well-defined inputs (building slides, writing SQL)
      - **Not good for**: Ambiguous problems requiring complex decision-making (business strategy, startup planning)
    - **Keep It User-Centric:** Explain the "problem" from the user's perspective rather than just listing features.
    - **Balance complexity vs. impact**: Aim to deliver the highest value features with minimal complexity early.

2. **Flow Design**: Outline at a high level, describe how your AI system orchestrates points.
    - Identify applicable design patterns (e.g., Map Reduce, Agent, RAG).
      - For each point in the flow, start with a high-level one-line description of what it does.
      - If using **Map Reduce**, specify how to map (what to split) and how to reduce (how to combine).
      - If using **Agent**, specify what are the inputs (context) and what are the possible actions.
      - If using **RAG**, specify what to embed, noting that there's usually both offline (indexing) and online (retrieval) workflows.
    - Outline the flow and draw it in a mermaid diagram. For example:
      ```mermaid
      flowchart LR
          start[Start] --> batch[Batch]
          batch --> check[Check]
          check -->|OK| process
          check -->|Error| fix[Fix]
          fix --> check
          
          subgraph process[Process]
            step1[Step 1] --> step2[Step 2]
          end
          
          process --> endNode[End]
      ```
    - > **If Humans can't specify the flow, AI Agents can't automate it!** Before building an LLM system, thoroughly understand the problem and potential solution by manually solving example inputs to develop intuition.  
      {: .best-practice }

3. **Utilities**: Based on the Flow Design, identify and implement necessary utility functions.
    - Think of your AI system as the brain. It needs a body—these *external utility functions*—to interact with the real world:
        <div align="center"><img src="https://github.com/the-pocket/.github/raw/main/assets/utility.png?raw=true" width="400"/></div>

        - Reading inputs (e.g., retrieving Slack messages, reading emails)
        - Writing outputs (e.g., generating reports, sending emails)
        - Using external tools (e.g., calling LLMs, searching the web)
        - **NOTE**: *LLM-based tasks* (e.g., summarizing text, analyzing sentiment) are **NOT** utility functions; rather, they are *core functions* internal in the AI system.
    - For each utility function, implement it and write a simple test.
    - Document their input/output, as well as why they are necessary. For example:
      - `name`: `get_embedding` (`utils/get_embedding.py`)
      - `input`: `str`
      - `output`: a vector of 3072 floats
      - `necessity`: Used by the second point to embed text
    - Example utility implementation:
      ```python
      # utils/call_llm.py
      from openai import OpenAI

      def call_llm(prompt):    
          client = OpenAI(api_key="YOUR_API_KEY_HERE")
          r = client.chat.completions.create(
              model="gpt-4o",
              messages=[{"role": "user", "content": prompt}]
          )
          return r.choices[0].message.content
          
      if __name__ == "__main__":
          prompt = "What is the meaning of life?"
          print(call_llm(prompt))
      ```
    - > **Sometimes, design Utilities before Flow:**  For example, for an LLM project to automate a legacy system, the bottleneck will likely be the available interface to that system. Start by designing the hardest utilities for interfacing, and then build the flow around them.
      {: .best-practice }
    - > **Avoid Exception Handling in Utilities**: If a utility function is called from a Point's `process()` method, avoid using `try...except` blocks within the utility. Let the Point's built-in retry mechanism handle failures.
      {: .warning }

4. **Data Design**: Design the memory store that points will use to communicate.
   - One core design principle for KernelMind is to use a well-designed memory store—a data contract that all points agree upon to retrieve and store data.
      - For simple systems, use an in-memory dictionary.
      - For more complex systems or when persistence is required, use a database.
      - **Don't Repeat Yourself**: Use in-memory references or foreign keys.
      - Example memory store design:
        ```python
        memory = {
            "user": {
                "id": "user123",
                "context": {                # Another nested dict
                    "weather": {"temp": 72, "condition": "sunny"},
                    "location": "San Francisco"
                }
            },
            "results": {}                   # Empty dict to store outputs
        }
        ```

5. **Point Design**: Plan how each point will read and write data, and use utility functions.
   - For each Point, describe its type, how it reads and writes data, and which utility function it uses. Keep it specific but high-level without codes. For example:
     - `type`: Regular (or Batch, or Async)
     - `load`: Read "text" from the memory store
     - `process`: Call the embedding utility function. **Avoid exception handling here**; let the Point's retry mechanism manage failures.
     - `save`: Write "embedding" to the memory store

6. **Implementation**: Implement the initial points and flows based on the design.
   - 🎉 If you've reached this step, humans have finished the design. Now *Agentic Coding* begins!
   - **"Keep it simple, stupid!"** Avoid complex features and full-scale type checking.
   - **FAIL FAST**! Leverage the built-in Point retry and fallback mechanisms to handle failures gracefully. This helps you quickly identify weak points in the system.
   - Add logging throughout the code to facilitate debugging.

7. **Optimization**:
   - **Use Intuition**: For a quick initial evaluation, human intuition is often a good start.
   - **Redesign Flow (Back to Step 3)**: Consider breaking down tasks further, introducing agentic decisions, or better managing input contexts.
   - If your flow design is already solid, move on to micro-optimizations:
     - **Prompt Engineering**: Use clear, specific instructions with examples to reduce ambiguity.
     - **In-Context Learning**: Provide robust examples for tasks that are difficult to specify with instructions alone.

   - > **You'll likely iterate a lot!** Expect to repeat Steps 3–6 hundreds of times.
     >
     > <div align="center"><img src="https://github.com/the-pocket/.github/raw/main/assets/success.png?raw=true" width="400"/></div>
     {: .best-practice }

8. **Reliability**  
   - **Point Retries**: Add checks in the point `process` to ensure outputs meet requirements, and consider increasing `max_retries` and `wait` times.
   - **Logging and Visualization**: Maintain logs of all attempts and visualize point results for easier debugging.
   - **Self-Evaluation**: Add a separate point (powered by an LLM) to review outputs when results are uncertain.

## Core Abstraction

We model the LLM workflow as a **Graph + Memory Store**:

- **Point** handles simple (LLM) tasks with enhanced type safety and async support.
- **Line** connects points through **Actions** (labeled edges) with simplified logic.
- **Memory Store** enables communication between points within flows.
- **Batch Processing** is integrated into the main Point class without separate classes.
- **Async Support** is built-in with automatic detection and proper handling.

<div align="center">
  <img src="https://github.com/the-pocket/.github/raw/main/assets/abstraction.png" width="500"/>
</div>

## Core Components

### Point Class

The `Point` class provides a unified interface for both sync and async operations:

```python
class Point:
    def __init__(self, retries: int = 1, wait: float = 0, parallel: bool = False, is_async: bool = False):
        self.next_points: Dict[str, 'Point'] = {}
        self.retries, self.wait, self.parallel, self.is_async = retries, wait, parallel, is_async
        if is_async and not inspect.iscoroutinefunction(self.process):
            warnings.warn("is_async=True but process method is not async. This may cause runtime errors.")
    
    def load(self, memory: Dict[str, Any]) -> Any: pass
    def process(self, item: Any) -> Any: pass
    def save(self, memory: Dict[str, Any], prep_result: Any, exec_result: Any) -> Optional[str]: pass
    def on_fail(self, item: Any, exc: Exception) -> Any: 
        raise exc
```

**Key Features:**
- **Type Safety**: Full type hints for better development experience
- **Async Detection**: Automatic detection of async methods to prevent runtime errors
- **Unified Interface**: Single class handles both sync and async operations
- **Enhanced Retry Logic**: More sophisticated retry mechanism with proper async support

### Line Class

The `Line` class provides orchestration:

```python
class Line(Point):
    def __init__(self, entry: Optional[Point] = None, **kwargs):
        super().__init__(**kwargs)
        self.entry = entry
    
    def run(self, memory: Dict[str, Any]) -> Optional[str]:
        current_point = copy.copy(self.entry) if self.entry else None
        last_step = None       
        while current_point:
            last_step = current_point.run(memory)
            next_node = current_point.next_points.get(last_step or "default")
            if not next_node: 
                print(f"Line ends: no successor for action '{last_step}' in {list(current_point.next_points.keys())}")
                break
            current_point = copy.copy(next_node) if next_node else None
        return last_step
```

**Features:**
- **Simplified Logic**: Cleaner flow execution without complex nested conditions
- **Better Error Messages**: Informative messages when flow ends
- **Early Termination**: Proper break when no successor is found

## Design Pattern

From there, it's easy to implement popular design patterns:

- **Agent** autonomously makes decisions.
- **Workflow** chains multiple tasks into pipelines.
- **RAG** integrates data retrieval with generation.
- **Map Reduce** splits data tasks into Map and Reduce steps.
- **Structured Output** formats outputs consistently.

<div align="center">
  <img src="https://github.com/the-pocket/.github/raw/main/assets/design.png" width="500"/>
</div>

---

## Agent

Agent is a powerful design pattern in which points can take dynamic actions based on the context.

<div align="center">
  <img src="https://github.com/the-pocket/.github/raw/main/assets/agent.png?raw=true" width="350"/>
</div>

### Implement Agent with Graph

1. **Context and Action:** Implement points that supply context and perform actions.  
2. **Branching:** Use branching to connect each action point to an agent point. Use action to allow the agent to direct the flow between points—and potentially loop back for multi-step.
3. **Agent Point:** Provide a prompt to decide action—for example:

```python
f"""
### CONTEXT
Task: {task_description}
Previous Actions: {previous_actions}
Current State: {current_state}

### ACTION SPACE
[1] search
  Description: Use web search to get results
  Parameters:
    - query (str): What to search for

[2] answer
  Description: Conclude based on the results
  Parameters:
    - result (str): Final answer to provide

### NEXT ACTION
Decide the next action based on the current context and available action space.
Return your response in the following format:

```yaml
thinking: |
    <your step-by-step reasoning process>
action: <action_name>
parameters:
    <parameter_name>: <parameter_value>
```"""
```

The core of building **high-performance** and **reliable** agents boils down to:

1. **Context Management:** Provide *relevant, minimal context.* For example, rather than including an entire chat history, retrieve the most relevant via RAG. Even with larger context windows, LLMs still fall victim to "lost in the middle", overlooking mid-prompt content.

2. **Action Space:** Provide *a well-structured and unambiguous* set of actions—avoiding overlap like separate `read_databases` or `read_csvs`. Instead, import CSVs into the database.

### Example Good Action Design

- **Incremental:** Feed content in manageable chunks (500 lines or 1 page) instead of all at once.

- **Overview-zoom-in:** First provide high-level structure (table of contents, summary), then allow drilling into details (raw texts).

- **Parameterized/Programmable:** Instead of fixed actions, enable parameterized (columns to select) or programmable (SQL queries) actions, for example, to read CSV files.

- **Backtracking:** Let the agent undo the last step instead of restarting entirely, preserving progress when encountering errors or dead ends.

### Example: Search Agent

This agent:
1. Decides whether to search or answer
2. If searches, loops back to decide if more search needed
3. Answers when enough context gathered

```python
class DecideAction(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        context = memory.get("context", "No previous search")
        query = memory["query"]
        return query, context
        
    def process(self, inputs: Any) -> Any:
        query, context = inputs
        prompt = f"""
Given input: {query}
Previous search results: {context}
Should I: 1) Search web for more info 2) Answer with current knowledge
Output in yaml:
```yaml
action: search/answer
reason: why this action
search_term: search phrase if action is search
```"""
        resp = call_llm(prompt)
        yaml_str = resp.split("```yaml")[1].split("```")[0].strip()
        result = yaml.safe_load(yaml_str)
        
        assert isinstance(result, dict)
        assert "action" in result
        assert "reason" in result
        assert result["action"] in ["search", "answer"]
        if result["action"] == "search":
            assert "search_term" in result
        
        return result

    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        if exec_res["action"] == "search":
            memory["search_term"] = exec_res["search_term"]
        return exec_res["action"]

class SearchWeb(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        return memory["search_term"]
        
    def process(self, search_term: Any) -> Any:
        return search_web(search_term)
    
    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        prev_searches = memory.get("context", [])
        memory["context"] = prev_searches + [
            {"term": memory["search_term"], "result": exec_res}
        ]
        return "decide"
        
class DirectAnswer(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        return memory["query"], memory.get("context", "")
        
    def process(self, inputs: Any) -> Any:
        query, context = inputs
        return call_llm(f"Context: {context}\nAnswer: {query}")

    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
       print(f"Answer: {exec_res}")
       memory["answer"] = exec_res

# Connect points
decide = DecideAction()
search = SearchWeb()
answer = DirectAnswer()

decide - "search" >> search
decide - "answer" >> answer
search - "decide" >> decide  # Loop back

line = Line(entry=decide)
line.run({"query": "Who won the Nobel Prize in Physics 2024?"})
```

---

## Workflow

Many real-world tasks are too complex for one LLM call. The solution is to **Task Decomposition**: decompose them into a chain of multiple Points.

<div align="center">
  <img src="https://github.com/the-pocket/.github/raw/main/assets/workflow.png?raw=true" width="400"/>
</div>

> - You don't want to make each task **too coarse**, because it may be *too complex for one LLM call*.
> - You don't want to make each task **too granular**, because then *the LLM call doesn't have enough context* and results are *not consistent across points*.
> 
> You usually need multiple *iterations* to find the *sweet spot*. If the task has too many *edge cases*, consider using Agents.
{: .best-practice }

### Example: Article Writing

```python
class GenerateOutline(Point):
    def load(self, memory: Dict[str, Any]) -> Any: 
        return memory["topic"]
    def process(self, topic: Any) -> Any: 
        return call_llm(f"Create a detailed outline for an article about {topic}")
    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]: 
        memory["outline"] = exec_res
        return "default"

class WriteSection(Point):
    def load(self, memory: Dict[str, Any]) -> Any: 
        return memory["outline"]
    def process(self, outline: Any) -> Any: 
        return call_llm(f"Write content based on this outline: {outline}")
    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]: 
        memory["draft"] = exec_res
        return "default"

class ReviewAndRefine(Point):
    def load(self, memory: Dict[str, Any]) -> Any: 
        return memory["draft"]
    def process(self, draft: Any) -> Any: 
        return call_llm(f"Review and improve this draft: {draft}")
    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]: 
        memory["final_article"] = exec_res
        return "default"

# Connect points
outline = GenerateOutline()
write = WriteSection()
review = ReviewAndRefine()

outline >> write >> review

# Create and run line
writing_line = Line(entry=outline)
memory = {"topic": "AI Safety"}
writing_line.run(memory)
```

For *dynamic cases*, consider using Agents.

---

## RAG (Retrieval Augmented Generation)

For certain LLM tasks like answering questions, providing relevant context is essential. One common architecture is a **two-stage** RAG pipeline:

<div align="center">
  <img src="https://github.com/the-pocket/.github/raw/main/assets/rag.png?raw=true" width="400"/>
</div>

1. **Offline stage**: Preprocess and index documents ("building the index").
2. **Online stage**: Given a question, generate answers by retrieving the most relevant context.

### Stage 1: Offline Indexing

We create three Points:
1. `ChunkDocs` – chunks raw text.
2. `EmbedDocs` – embeds each chunk.
3. `StoreIndex` – stores embeddings into a vector database.

```python
class ChunkDocs(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        # A list of file paths in memory["files"]. We process each file.
        return memory["files"]

    def process(self, filepath: Any) -> Any:
        # read file content. In real usage, do error handling.
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        # chunk by 100 chars each
        chunks = []
        size = 100
        for i in range(0, len(text), size):
            chunks.append(text[i : i + size])
        return chunks
    
    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        # exec_res is a list of chunk-lists, one per file.
        # flatten them all into a single list of chunks.
        all_chunks = []
        for chunk_list in exec_res:
            all_chunks.extend(chunk_list)
        memory["all_chunks"] = all_chunks
        return "default"

class EmbedDocs(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        return memory["all_chunks"]

    def process(self, chunk: Any) -> Any:
        return get_embedding(chunk)

    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        # Store the list of embeddings.
        memory["all_embeds"] = exec_res
        print(f"Total embeddings: {len(exec_res)}")
        return "default"

class StoreIndex(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        # We'll read all embeds from memory.
        return memory["all_embeds"]

    def process(self, all_embeds: Any) -> Any:
        # Create a vector index (faiss or other DB in real usage).
        index = create_index(all_embeds)
        return index

    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        memory["index"] = exec_res
        return "default"

# Wire them in sequence
chunk_point = ChunkDocs()
embed_point = EmbedDocs()
store_point = StoreIndex()

chunk_point >> embed_point >> store_point

OfflineLine = Line(entry=chunk_point)
```

Usage example:

```python
memory = {
    "files": ["doc1.txt", "doc2.txt"],  # any text files
}
OfflineLine.run(memory)
```

### Stage 2: Online Query & Answer

We have 3 points:
1. `EmbedQuery` – embeds the user's question.
2. `RetrieveDocs` – retrieves top chunk from the index.
3. `GenerateAnswer` – calls the LLM with the question + chunk to produce the final answer.

```python
class EmbedQuery(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        return memory["question"]

    def process(self, question: Any) -> Any:
        return get_embedding(question)

    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        memory["q_emb"] = exec_res
        return "default"

class RetrieveDocs(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        # We'll need the query embedding, plus the offline index/chunks
        return memory["q_emb"], memory["index"], memory["all_chunks"]

    def process(self, inputs: Any) -> Any:
        q_emb, index, chunks = inputs
        I, D = search_index(index, q_emb, top_k=1)
        best_id = I[0][0]
        relevant_chunk = chunks[best_id]
        return relevant_chunk

    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        memory["retrieved_chunk"] = exec_res
        print("Retrieved chunk:", exec_res[:60], "...")
        return "default"

class GenerateAnswer(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        return memory["question"], memory["retrieved_chunk"]

    def process(self, inputs: Any) -> Any:
        question, chunk = inputs
        prompt = f"Question: {question}\nContext: {chunk}\nAnswer:"
        return call_llm(prompt)

    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        memory["answer"] = exec_res
        print("Answer:", exec_res)
        return "default"

embed_qpoint = EmbedQuery()
retrieve_point = RetrieveDocs()
generate_point = GenerateAnswer()

embed_qpoint >> retrieve_point >> generate_point
OnlineLine = Line(entry=embed_qpoint)
```

Usage example:

```python
# Suppose we already ran OfflineLine and have:
# memory["all_chunks"], memory["index"], etc.
memory["question"] = "Why do people like cats?"

OnlineLine.run(memory)
# final answer in memory["answer"]
```

---

## Map Reduce

MapReduce is a design pattern suitable when you have either:
- Large input data (e.g., multiple files to process), or
- Large output data (e.g., multiple forms to fill)

and there is a logical way to break the task into smaller, ideally independent parts. 

<div align="center">
  <img src="https://github.com/the-pocket/.github/raw/main/assets/mapreduce.png?raw=true" width="400"/>
</div>

You first break down the task using batch processing in the map phase, followed by aggregation in the reduce phase.

### Example: Document Summarization

```python
class SummarizeAllFiles(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        files_dict = memory["files"]  # e.g. 10 files
        return list(files_dict.items())  # [("file1.txt", "aaa..."), ("file2.txt", "bbb..."), ...]

    def process(self, one_file: Any) -> Any:
        filename, file_content = one_file
        summary_text = call_llm(f"Summarize the following file:\n{file_content}")
        return (filename, summary_text)

    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        memory["file_summaries"] = dict(exec_res)
        return "default"

class CombineSummaries(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        return memory["file_summaries"]

    def process(self, file_summaries: Any) -> Any:
        # format as: "File1: summary\nFile2: summary...\n"
        text_list = []
        for fname, summ in file_summaries.items():
            text_list.append(f"{fname} summary:\n{summ}\n")
        big_text = "\n---\n".join(text_list)

        return call_llm(f"Combine these file summaries into one final summary:\n{big_text}")

    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        memory["all_files_summary"] = exec_res
        return "default"

batch_point = SummarizeAllFiles()
combine_point = CombineSummaries()
batch_point >> combine_point

line = Line(entry=batch_point)

memory = {
    "files": {
        "file1.txt": "Alice was beginning to get very tired of sitting by her sister...",
        "file2.txt": "Some other interesting text ...",
        # ...
    }
}
line.run(memory)
print("Individual Summaries:", memory["file_summaries"])
print("\nFinal Summary:\n", memory["all_files_summary"])
```

---

## Structured Output

In many use cases, you may want the LLM to output a specific structure, such as a list or a dictionary with predefined keys.

There are several approaches to achieve a structured output:
- **Prompting** the LLM to strictly return a defined structure.
- Using LLMs that natively support **schema enforcement**.
- **Post-processing** the LLM's response to extract structured content.

In practice, **Prompting** is simple and reliable for modern LLMs.

### Example Use Cases

- Extracting Key Information 

```yaml
product:
  name: Widget Pro
  price: 199.99
  description: |
    A high-quality widget designed for professionals.
    Recommended for advanced users.
```

- Summarizing Documents into Bullet Points

```yaml
summary:
  - This product is easy to use.
  - It is cost-effective.
  - Suitable for all skill levels.
```

- Generating Configuration Files

```yaml
server:
  host: 127.0.0.1
  port: 8080
  ssl: true
```

### Prompt Engineering

When prompting the LLM to produce **structured** output:
1. **Wrap** the structure in code fences (e.g., `yaml`).
2. **Validate** that all required fields exist (and let `Point` handles retry).

### Example Text Summarization

```python
class SummarizePoint(Point):
    def process(self, prep_res: Any) -> Any:
        # Suppose `prep_res` is the text to summarize.
        prompt = f"""
Please summarize the following text as YAML, with exactly 3 bullet points

{prep_res}

Now, output:
```yaml
summary:
  - bullet 1
  - bullet 2
  - bullet 3
```"""
        response = call_llm(prompt)
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()

        import yaml
        structured_result = yaml.safe_load(yaml_str)

        assert "summary" in structured_result
        assert isinstance(structured_result["summary"], list)

        return structured_result
```

> Besides using `assert` statements, another popular way to validate schemas is Pydantic
{: .note }

### Why YAML instead of JSON?

Current LLMs struggle with escaping. YAML is easier with strings since they don't always need quotes.

**In JSON**  

```json
{
  "dialogue": "Alice said: \"Hello Bob.\\nHow are you?\\nI am good.\""
}
```

- Every double quote inside the string must be escaped with `\"`.
- Each newline in the dialogue must be represented as `\n`.

**In YAML**  

```yaml
dialogue: |
  Alice said: "Hello Bob.
  How are you?
  I am good."
```

- No need to escape interior quotes—just place the entire text under a block literal (`|`).
- Newlines are naturally preserved without needing `\n`.

---

## Implementation Details

### 1. Type Safety

KernelMind uses comprehensive type hints:

```python
from typing import Any, Dict, Optional

def load(self, memory: Dict[str, Any]) -> Any: pass
def process(self, item: Any) -> Any: pass
def save(self, memory: Dict[str, Any], prep_result: Any, exec_result: Any) -> Optional[str]: pass
```

**Benefits:**
- **IDE Support**: Better autocomplete and error detection
- **Documentation**: Types serve as inline documentation
- **Maintainability**: Easier to understand and modify code

### 2. Async Support

Enhanced async handling with automatic detection:

```python
async def _async_single(self, item: Any) -> Any:
    for i in range(self.retries):
        try:
            if inspect.iscoroutinefunction(self.process):
                return await self.process(item)
            else:
                result = self.process(item)
                return await result if inspect.iscoroutine(result) else result
        except Exception as e:
            if i == self.retries-1:
                fallback_result = self.on_fail(item, e)
                return await fallback_result if inspect.iscoroutine(fallback_result) else fallback_result
            if self.wait > 0: await asyncio.sleep(self.wait)
```

**Features:**
- **Automatic Detection**: Uses `inspect.iscoroutinefunction()` to detect async methods
- **Mixed Support**: Handles both sync and async methods in the same class
- **Proper Awaiting**: Correctly awaits coroutines and handles mixed return types

### 3. Batch Processing

Integrated batch processing without separate classes:

```python
def _sync_batch(self, items: list) -> list:
    return [self._sync_single(item) for item in items]

async def _async_batch(self, items: list) -> list:
    if self.parallel:
        return await asyncio.gather(*[self._async_single(item) for item in items])
    else:
        return [await self._async_single(item) for item in items]
```

**Advantages:**
- **Unified Interface**: No need for separate `BatchPoint` classes
- **Parallel Support**: Built-in parallel processing for async operations
- **Flexible**: Can handle both sequential and parallel execution

### 4. Error Handling

Enhanced error handling with proper fallback mechanisms:

```python
def _sync_single(self, item: Any) -> Any:
    for i in range(self.retries):
        try: 
            return self.process(item)
        except Exception as e:
            if i == self.retries-1: 
                return self.on_fail(item, e)
            if self.wait > 0: 
                time.sleep(self.wait)
```

**Improvements:**
- **Graceful Degradation**: Proper fallback handling
- **Configurable Retries**: Adjustable retry count and wait times
- **Async Compatible**: Works with both sync and async fallbacks

## Communication

Points and Lines **communicate** in 2 ways:

1. **Memory Store (for almost all the cases)** 
   - A global data structure (often an in-mem dict) that all points can read (`load()`) and write (`save()`).  
   - Great for data results, large content, or anything multiple points need.
   - You shall design the data structure and populate it ahead.

2. **Params (only for Batch operations)** 
   - Each point has a local, ephemeral `params` dict passed in by the **parent Line**, used as an identifier for tasks.
   - Good for identifiers like filenames or numeric IDs, in Batch mode.

### Example Memory Store Usage

```python
class LoadData(Point):
    def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
        # We write data to memory store
        memory["data"] = "Some text content"
        return None

class Summarize(Point):
    def load(self, memory: Dict[str, Any]) -> Any:
        # We read data from memory store
        return memory["data"]

    def process(self, prep_res: Any) -> Any:
        # Call LLM to summarize
        prompt = f"Summarize: {prep_res}"
        summary = call_llm(prompt)
        return summary

    def save(self, memory: Dict[str, Any], prep_result: Any, exec_result: Any) -> Optional[str]:
        # We write summary to memory store
        memory["summary"] = exec_result
        return "default"

load_data = LoadData()
summarize = Summarize()
load_data >> summarize
line = Line(entry=load_data)

memory = {}
line.run(memory)
```

## Example LLM Project File Structure

```
my_project/
├── main.py
├── points.py
├── line.py
├── utils/
│   ├── __init__.py
│   ├── call_llm.py
│   └── search_web.py
├── requirements.txt
└── docs/
    └── design.md
```

- **`docs/design.md`**: Contains project documentation for each step above. This should be *high-level* and *no-code*.
- **`utils/`**: Contains all utility functions.
  - It's recommended to dedicate one Python file to each API call, for example `call_llm.py` or `search_web.py`.
  - Each file should also include a `main()` function to try that API call
- **`points.py`**: Contains all the point definitions.
  ```python
  # points.py
  from kernelmind_core import Point
  from utils.call_llm import call_llm

  class GetQuestionPoint(Point):
      def process(self, _: Any) -> Any:
          # Get question directly from user input
          user_question = input("Enter your question: ")
          return user_question
      
      def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
          # Store the user's question
          memory["question"] = exec_res
          return "default"  # Go to the next point

  class AnswerPoint(Point):
      def load(self, memory: Dict[str, Any]) -> Any:
          # Read question from memory
          return memory["question"]
      
      def process(self, question: Any) -> Any:
          # Call LLM to get the answer
          return call_llm(question)
      
      def save(self, memory: Dict[str, Any], prep_res: Any, exec_res: Any) -> Optional[str]:
          # Store the answer in memory
          memory["answer"] = exec_res
          return "default"
  ```
- **`line.py`**: Implements functions that create lines by importing point definitions and connecting them.
  ```python
  # line.py
  from kernelmind_core import Line
  from points import GetQuestionPoint, AnswerPoint

  def create_qa_line():
      """Create and return a question-answering line."""
      # Create points
      get_question_point = GetQuestionPoint()
      answer_point = AnswerPoint()
      
      # Connect points in sequence
      get_question_point >> answer_point
      
      # Create line starting with input point
      return Line(entry=get_question_point)
  ```
- **`main.py`**: Serves as the project's entry point.
  ```python
  # main.py
  from line import create_qa_line

  # Example main function
  # Please replace this with your own main function
  def main():
      memory = {
          "question": None,  # Will be populated by GetQuestionPoint from user input
          "answer": None     # Will be populated by AnswerPoint
      }

      # Create the line and run it
      qa_line = create_qa_line()
      qa_line.run(memory)
      print(f"Question: {memory['question']}")
      print(f"Answer: {memory['answer']}")

  if __name__ == "__main__":
      main()
  ```

## Best Practices

### 1. Type Safety
- Always use type hints for better code quality
- Leverage IDE features for error detection
- Document expected input/output types

### 2. Async Handling
- Use `is_async=True` for async points
- Let the framework handle async detection
- Avoid mixing sync and async operations in the same method

### 3. Error Handling
- Implement proper `on_fail` methods
- Use appropriate retry counts and wait times
- Handle both sync and async exceptions

### 4. Flow Design
- Keep flows simple and linear when possible
- Use meaningful action names
- Properly handle terminal points

### 5. Utility Functions
- Keep utilities focused and single-purpose
- Avoid exception handling in utilities
- Let Point retry mechanisms handle failures

## Utility Function

We **do not** provide built-in utilities. Instead, we offer *examples*—please *implement your own*:

- **LLM Wrapper**: Call various LLM providers (OpenAI, Claude, etc.)
- **Web Search**: Search the web for information
- **Chunking**: Split large texts into manageable chunks
- **Embedding**: Convert text to vector representations
- **Vector Databases**: Store and retrieve embeddings
- **Text-to-Speech**: Convert text to speech

**Why not built-in?**: I believe it's a *bad practice* for vendor-specific APIs in a general framework:
- *API Volatility*: Frequent changes lead to heavy maintenance for hardcoded APIs.
- *Flexibility*: You may want to switch vendors, use fine-tuned models, or run them locally.
- *Optimizations*: Prompt caching, batching, and streaming are easier without vendor lock-in.

## Conclusion

KernelMind provides a robust foundation for building LLM applications with:

- **Better Developer Experience**: Type hints and IDE support
- **Enhanced Reliability**: Improved error handling and async support
- **Simplified Architecture**: Unified interface for all point types
- **Future-Proof Design**: Extensible and maintainable codebase

## Ready to build your Apps? 

Check out this guide, the fastest way to develop LLM projects with KernelMind! 
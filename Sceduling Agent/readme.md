Got it — thanks for the clarification. Here's a corrected and precise version of the `README.md` for your **AI Meeting Scheduler Agent**, without any mention of voice functionality:

---

# 🗓️ AI-Powered Meeting Scheduler Agent

This project is an **AI-driven meeting scheduler** built using **LangGraph**, **LangChain**, **Groq's LLaMA3-70B**, and a **custom MCP tool**. The system reads a natural language meeting request, extracts all necessary information in structured form, and invokes a scheduling tool through an external process.

---

## 🚀 Features

* 🤖 **Autonomous agent** using [LangGraph](https://www.langchain.com/langgraph) to plan and route tasks
* 🧠 **LLM reasoning with LLaMA3-70B** via [Groq API](https://groq.com/)
* 📤 **Structured data extraction** using `PydanticOutputParser`
* 🛠️ **MCP Tool Integration** to perform scheduling logic via a subprocess tool
* 🔄 **Dynamic branching**: decides whether to call a tool or terminate

---

## 🔧 Tech Stack

| Layer         | Tool / Library                                   |
| ------------- | ------------------------------------------------ |
| LLM           | [Groq's LLaMA3-70B](https://groq.com/)           |
| Orchestration | [LangGraph](https://www.langchain.com/langgraph) |
| Parsing       | Pydantic + LangChain Output Parser               |
| Tools Runtime | MCP Adapter (`langchain_mcp_adapters`)           |
| Prompts       | LangChain PromptTemplate                         |
| Runtime Env   | Python + AsyncIO                                 |

---

## 📂 File Overview

```bash
📦ai-meeting-scheduler/
 ┣ 📜client.py           # Main AI agent logic
 ┣ 📜scedule_tool.py     # Tool invoked for scheduling via MCP
 ┣ 📜.env                # Groq API keys (not committed)
 ┗ 📜README.md           # Project documentation
```

---

## 🧠 How It Works

### 1. **Prompt + Parsing**

A structured prompt gathers meeting details from the input message using `PydanticOutputParser`. It extracts:

* `name` – Client's name
* `email` – Client's email
* `date` – Date of the meeting
* `meeting_start` – Start time
* `meeting_end` – End time

---

### 2. **LangGraph Workflow**

```text
START
  ↓
llm_call (LLaMA3-70B processes the request)
  ↓
Check if tool_call needed?
  ├─ Yes → tools (MCP scheduling tool)
  └─ No  → END
  ↓
END
```

---

### 3. **Tool Execution**

Tool metadata:

```python
{
  "AI Scedule Tool": {
    "command": "python",
    "args": ["scedule_tool.py"],
    "transport": "stdio"
  }
}
```

The tool is invoked only if the LLM message contains `tool_calls`.

---

### ✅ Sample Run

#### 📨 Input

```text
"Schedule a meeting for a client named Timmy, with email: 0T9QF@example.com,
 starting at 10:00 and ending at 11:00, on 2025-07-18"
```

#### 📋 Output

```text
Message 1: AIMessage
   Content: Structured meeting parsed
   Tool calls made: 1
      Tool: AI Scedule Tool
      Args: {name: Timmy, ...}

Message 2: ToolMessage
   Content: Meeting booked successfully.
```



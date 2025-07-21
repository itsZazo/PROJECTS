Here’s your fancy, **visually styled and formatted** version — great for publishing on GitHub, Notion, a blog, or as a professional project showcase:

---

# 🧠 **Voice-Enabled AI Agent (with Tool Integration using LangGraph)**

A **Flask-based web application** that connects a **LangGraph-based AI Agent** with an interactive frontend using **SocketIO**.
It leverages **Groq's LLaMA 3** model, supports tool usage (e.g., image generation), and integrates **OpenAI Whisper** for **speech-to-text interaction**.

---

## 🚀 **Features**

* 🔁 **Real-time interaction** with LLM agent via **WebSockets** (`Flask-SocketIO`)
* 🧩 **Reasoning loop** via **LangGraph**, enabling conditional tool usage
* 🖼️ **Multi-modal outputs** – including **image generation** via tools like Gemini or Stable Diffusion
* 🔌 **MCP Client integration** to plug in multiple tools seamlessly
* ⚙️ **Modular & production-ready** with async support and extensibility

---

## 🛠️ **Technologies Used**

* `Flask` + `Flask-SocketIO` for backend and real-time updates
* `LangGraph` for stateful multi-turn agent workflows
* `Groq's LLaMA 3` via `langchain_groq`
* `langchain-mcp-adapters` for tool orchestration
* `dotenv` for API/config management
* `OpenAI Whisper` (optional) for **voice-based interaction**

---

## 📂 **Folder Structure**

```
.
├── app.py                        # Main Flask app
├── templates/
│   └── index.html               # Frontend template
├── static/                      # For images, CSS, JS
├── .env                         # API keys & config
├── gemini_image_generator.py 
|__ stable_diffusion_server.py   # (Tool server) Image generator tool
├── requirements.txt
└── README.md
```

---

## ⚙️ **Setup Instructions**

### 1. 🔑 Clone the repository and install dependencies

```bash
git clone https://github.com/yourname/ai-agent-app.git
cd ai-agent-app
pip install -r requirements.txt
```

---

### 2. 🔐 Setup `.env` file

Create a `.env` file with the following content:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_key_if_using_whisper_or_other_tools
```

> ✅ You can get a **free Groq API key** from [Groq Cloud](https://console.groq.com/).

---

### ▶️ Run the Agent

```bash
python app.py
```

🔗 Your server will be live at: [http://localhost:5000](http://localhost:5000)

---

## 🧰 **Switching from Gemini to Stable Diffusion**

The app uses **Gemini** by default for image generation (`gemini_image_generator.py`).

To switch to **Stable Diffusion**, modify the **MCP client config** in `app.py`.

### 🔄 Current Configuration:

```python
client = MultiServerMCPClient(
    {
        "image_generation": {
            "command": "python",
            "args": ["gemini_image_generator.py"],
            "transport": "stdio"
        }
    }
)
```

### ✅ Update To:

```python
client = MultiServerMCPClient(
    {
        "image_generation": {
            "command": "python",
            "args": ["stable_diffusion_server.py"],
            "transport": "stdio"
        }
    }
)
```

> 📌 Make sure `stable_diffusion_server.py` is implemented and follows **MCP-style messaging**.

---

## 🎤 **Voice Integration (Optional)**

To enable **voice interaction**:

* Use OpenAI Whisper to transcribe user speech.
* Send transcribed text to the chat endpoint (`/message`) via frontend.
* Respond to voice inputs like normal messages, with optional visual or tool responses.



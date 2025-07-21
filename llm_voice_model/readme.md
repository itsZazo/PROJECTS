Here’s a **fancified version** of your project description with **highlighted headings, emojis, and formatting** — perfect for your portfolio, GitHub, or resume:

---

## 🧠🎙️ Voice AI Assistant (Multilingual, VAD-Based)

An intelligent **voice-enabled AI assistant** that seamlessly converts speech to text and back to speech — enabling **fluid, multilingual conversations**! Built using **LangChain**, **LLMs**, **Silero VAD**, **Vosk ASR**, and **Edge-TTS**, this assistant runs locally via Flask and supports English and Hindi.

---

### 🔧 Key Features

* 🧠 **LLM Chat Integration** (LLaMA3-70B or ChatGPT via LangChain)
* 🎤 **Voice Input with Voice Activity Detection** (Silero VAD)
* 🗣️ **Speech Recognition** via Vosk (Supports English 🇺🇸 & Hindi 🇮🇳)
* 🔊 **Text-to-Speech** using Microsoft Edge-TTS (Multilingual support)
* 🌐 **REST API** powered by Flask
* 🧩 LangChain RunnableChain for smooth LLM orchestration

---

### 🚀 How It Works

1. **🎙️ You speak** — VAD detects speech and starts/stops recording automatically.
2. **🧾 Audio is transcribed** using Vosk (offline ASR model).
3. **💬 LangChain builds a prompt** and sends it to the LLM.
4. **🗣️ The assistant speaks back** using Edge-TTS in the selected language.

Supports **seamless switching** between English and Hindi, with more languages easily integrable.

---

### 🛠️ Tech Stack

* **Flask** – Python web server
* **LangChain** – LLM routing & orchestration
* **Silero VAD** – Lightweight voice activity detection
* **Vosk ASR** – Offline speech recognition (English & Hindi models)
* **Edge-TTS** – Text-to-speech synthesis using Microsoft Edge voices
* **PyAudio / PyGame** – Audio input/output and playback

---

### 📁 Folder Structure

```
├── app.py                        # Main Flask app
├── templates/
│   └── index.html               # UI template
├── vosk models/
│   ├── vosk-model-small-en-us-0.15
│   └── vosk-model-small-hi-0.22
├── requirements.txt             # Python dependencies
└── .env                         # OpenAI API key
```

---

### 📦 Setup & Requirements

* Python 3.8+

* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

* Add `.env` file:

  ```ini
  OPENAI_API_KEY=your_openai_key_here
  ```

* Download Vosk models from [Vosk GitHub](https://github.com/alphacep/vosk-api)

---

### 🔄 API Endpoints

| Endpoint        | Method | Description                       |
| --------------- | ------ | --------------------------------- |
| `/`             | GET    | Load the UI                       |
| `/record`       | POST   | Start VAD recording + AI response |
| `/set_language` | POST   | Set current language (e.g., `en`) |
| `/get_language` | GET    | Return current language setting   |

---

### 🌍 Supported Languages

* ✅ English (`en`)
* ✅ Hindi (`hi`)
* 🧪 Edge-TTS also supports: Urdu (`ur`), Spanish (`es`), German (`de`), Japanese (`ja`), Korean (`ko`) — and more.

---

### 🧠 Example Use Case

> **Speak** ➝ **Transcribe with Vosk** ➝ **Generate with LLM** ➝ **Speak response using Edge-TTS**

---

### 💡 Future Enhancements

* 🎙️ Add UI microphone button for recording
* 🧠 Add Whisper ASR for multilingual and high-accuracy transcription
* 🌐 WebSocket support for **real-time streaming**
* 🌎 Dynamic language expansion from Edge-TTS

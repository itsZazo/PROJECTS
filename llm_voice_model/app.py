from flask import Flask, render_template, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import pyaudio
import vosk
import json
import wave
import io
import os
import edge_tts
import pygame
import tempfile
import os
import asyncio
from langdetect import detect
import tempfile
import os
import time
import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps, VADIterator

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Global variables for models
vosk_model_en = None
vosk_model_hi = None
selected_language = "en"  # Default language

#loading vosk model
print("Loading Vosk English model...")
try:
    vosk_model_en = vosk.Model(r"D:\AI LABRATORY\llm voice using vad\vosk models\vosk-model-small-en-us-0.15")
    print("✅ Vosk English model loaded successfully")
except Exception as e:
    print(f"❌ Error loading English Vosk model: {e}")

#loading vosk model
print("Loading Vosk Hindi model...")
try:
    vosk_model_hi = vosk.Model(r"D:\AI LABRATORY\llm voice using vad\vosk models\vosk-model-small-hi-0.22")
    print("✅ Vosk Hindi model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Hindi Vosk model: {e}")

# Initialize VAD model
vad_model = load_silero_vad()

# Initialize the chat model
llm = ChatOpenAI(
    model="llama3-70b-8192", 
    temperature=0.5
)

# Prompt template
prompt = PromptTemplate(
    template="""
You are a Very Helpful Assisstant
Answer the following question: 
{history}

IMPORTANT:
- Answer in the Language the Question is asked
- If it does not make sense, please ask to the user to rephrase the question
- Answer the Questions to your full knowledge 
- Be Concise
- Don't over explain
- 
""",
    input_variables=["history"]
)

def record_audio_function(input_data):
    print("Audio recording with VAD...")
    
    format = pyaudio.paInt16
    channels = 1
    rate = 16000
    chunk_size = 1024  # Smaller chunks for better responsiveness
    
    # VAD parameters
    speech_threshold = 0.5  # Probability threshold for speech
    silence_duration = 1.5  # Seconds of silence before stopping
    max_recording_time = 20  # Maximum recording time in seconds
    min_recording_time = 1   # Minimum recording time in seconds
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)
    
    all_audio = []  # Store all audio for processing
    speech_detected = False
    last_speech_time = None
    recording_start_time = time.time()
    
    print(f"Listening for speech in {selected_language.upper()}... (speak now)")
    
    try:
        while True:
            # Read audio chunk
            data = stream.read(chunk_size)
            all_audio.append(data)
            
            # Convert to numpy array for VAD processing
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Process with VAD every few chunks to reduce computation
            if len(all_audio) % 3 == 0:  # Process every 3rd chunk
                try:
                    # Get larger window for VAD (combine last few chunks)
                    window_data = b''.join(all_audio[-6:]) if len(all_audio) >= 6 else b''.join(all_audio)
                    window_np = np.frombuffer(window_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Ensure we have enough samples (VAD needs minimum length)
                    if len(window_np) >= 512:
                        # Use get_speech_timestamps for more reliable detection
                        speech_timestamps = get_speech_timestamps(torch.from_numpy(window_np), vad_model, sampling_rate=rate)
                        
                        if speech_timestamps:
                            if not speech_detected:
                                print("🎤 Speech detected! Recording...")
                                speech_detected = True
                            last_speech_time = time.time()
                        
                except Exception as e:
                    print(f"VAD processing error: {e}")
                    # Continue without VAD if there's an error
                    pass
            
            current_time = time.time()
            recording_duration = current_time - recording_start_time
            
            # Stop conditions
            if speech_detected:
                # Stop if we haven't detected speech for silence_duration seconds
                if last_speech_time and (current_time - last_speech_time) > silence_duration:
                    if recording_duration >= min_recording_time:
                        print("🔇 Silence detected, stopping recording...")
                        break
                
                # Stop if maximum recording time reached
                if recording_duration >= max_recording_time:
                    print("⏰ Maximum recording time reached...")
                    break
            else:
                # If no speech detected after 10 seconds, stop listening
                if recording_duration > 10:
                    print("⚠️ No speech detected, stopping...")
                    break
    
    except Exception as e:
        print(f"Recording error: {e}")
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    # Convert all audio to WAV format
    if not all_audio:
        print("❌ No audio recorded")
        return {"audio_bytes": b'', "language": selected_language}  # Include language in return
    
    # Combine all audio chunks
    audio_data = b''.join(all_audio)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(audio_data)
    
    wav_bytes = buffer.getvalue()
    duration = len(audio_data) / (rate * channels * 2)  # 2 bytes per sample
    
    print(f"✅ Audio recorded: {duration:.2f} seconds, {len(wav_bytes)} bytes")
    
    if speech_detected:
        print("🎯 Speech was detected during recording")
    else:
        print("⚠️ No speech detected - transcription may be empty")
    
    return {"audio_bytes": wav_bytes, "language": selected_language}

chat_history = []

def transcribe_audio_function(input):
    global chat_history
    print("🎯 Starting Vosk audio transcription...")
    audio_bytes = input.get("audio_bytes")
    language = input.get("language", "en")
    
    if not audio_bytes:
        print("❌ No audio bytes received")
        return {"history": chat_history}
    
    # Select the appropriate model based on language
    if language == "hi":
        vosk_model = vosk_model_hi
        print("Using Hindi Vosk model")
    else:
        vosk_model = vosk_model_en
        print("Using English Vosk model")
    
    if not vosk_model:
        print(f"❌ Vosk model for {language} not loaded")
        return {"history": chat_history}
    
    print(f"📊 Audio data size: {len(audio_bytes)} bytes")
    
    # Save bytes to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_file_path = temp_file.name
    
    try:
        print(f"🤖 Running Vosk transcription in {language.upper()}...")
        
        # Open the WAV file
        wf = wave.open(temp_file_path, 'rb')
        
        # Check if audio format is compatible
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("❌ Audio file must be WAV format mono PCM.")
            transcription_text = "Error: Invalid audio format"
        else:
            # Initialize recognizer with the sample rate from the audio file
            recognizer = vosk.KaldiRecognizer(vosk_model, wf.getframerate())
            
            # Process audio in chunks
            transcription_text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if result.get('text'):
                        transcription_text += result['text'] + " "
            
            # Get final result
            final_result = json.loads(recognizer.FinalResult())
            if final_result.get('text'):
                transcription_text += final_result['text']
            
            transcription_text = transcription_text.strip()
        
        wf.close()
        
        print(f"📝 Raw transcription ({language}): '{transcription_text}'")
        print(f"📊 Transcription length: {len(transcription_text)} characters")
        
        if not transcription_text:
            print("⚠️ Empty transcription - audio might be silent or unclear")
            transcription_text = "No speech detected in audio"
        
    except Exception as e:
        print(f"❌ Vosk transcription error: {str(e)}")
        transcription_text = f"Error: {str(e)}"
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass
    
    print(f"✅ Final transcription: '{transcription_text}'")
    print("📤 Sending to LLM...")
    
    chat_history.append(HumanMessage(transcription_text))
    return {"history": chat_history}

# Language to voice mapping for Edge-TTS
VOICE_MAPPING = {
    'en': 'en-US-AriaNeural',
    'ur': 'ur-PK-AsadNeural',
    'ar': 'ar-SA-HamedNeural',
    'hi': 'hi-IN-MadhurNeural',
    'zh': 'zh-CN-XiaoxiaoNeural',
    'ja': 'ja-JP-NanamiNeural',
    'ko': 'ko-KR-SunHiNeural',
    'es': 'es-ES-ElviraNeural',
    'fr': 'fr-FR-DeniseNeural',
    'de': 'de-DE-KatjaNeural',
    'it': 'it-IT-ElsaNeural',
    'pt': 'pt-BR-FranciscaNeural',
    'ru': 'ru-RU-SvetlanaNeural',
    'tr': 'tr-TR-EmelNeural',
}

async def generate_multilang_speech(text, voice):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(temp_file.name)
    return temp_file.name

def speak_text_function(input):
    print("speaking with Edge-TTS multi-language...")
    global chat_history
    
    if hasattr(input, 'content'):
        response_text = input.content
        chat_history.append(AIMessage(response_text))
    else:
        response_text = str(input)
    
    if response_text:
        try:
            # Use selected language or detect language
            if selected_language in VOICE_MAPPING:
                voice = VOICE_MAPPING[selected_language]
                print(f"Using selected language voice: {voice}")
            else:
                # Fallback to language detection
                detected_lang = detect(response_text)
                voice = VOICE_MAPPING.get(detected_lang, VOICE_MAPPING['en'])
                print(f"Detected language: {detected_lang}, using voice: {voice}")
            
            # Generate speech
            audio_file = asyncio.run(generate_multilang_speech(response_text, voice))
            
            # Play audio
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            print(f"Multi-language speech completed!")
            
        except Exception as e:
            print(f"Edge TTS Error: {e}")
            return {"status": "error", "message": str(e)}
        
        finally:
            try:
                pygame.mixer.quit()
                if 'audio_file' in locals() and os.path.exists(audio_file):
                    os.unlink(audio_file)
            except:
                pass
    
    return {"status": "spoken", "text": response_text}

# Create runnables
runnable_audio_generator = RunnableLambda(record_audio_function)
runnable_transcription = RunnableLambda(transcribe_audio_function)
runnable_speak = RunnableLambda(speak_text_function)

# Create the chain
chain = runnable_audio_generator | runnable_transcription | prompt | llm | runnable_speak

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_language', methods=['POST'])
def set_language():
    global selected_language
    try:
        data = request.get_json()
        language = data.get('language', 'en')
        
        # Validate language
        if language in ['en', 'hi']:
            selected_language = language
            print(f"Language set to: {selected_language}")
            return jsonify({
                'success': True,
                'message': f'Language set to {language.upper()}',
                'language': selected_language
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid language. Supported: en, hi'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get_language', methods=['GET'])
def get_language():
    return jsonify({
        'language': selected_language
    })

@app.route('/record', methods=['POST'])
def record():
    try:
        print(f"Starting VAD-based recording in {selected_language.upper()}...")
        
        # Execute the chain
        result = chain.invoke({})
        
        return jsonify({
            'success': True,
            'message': f'Recording completed and processed with VAD in {selected_language.upper()}!',
            'response': result.get('text', 'Processing completed'),
            'language': selected_language
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
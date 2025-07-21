from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import asyncio
import os
import uuid
import base64
import threading
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import re
import json
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for the agent
agent_app = None
client = None
agent_initialized = False

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def initialize_agent():
    global agent_app, client, agent_initialized
    
    print("🔄 Starting agent initialization...")
    
    try:
        # Initialize the event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def async_init():
            global agent_app, client, agent_initialized
            
            print("📡 Connecting to MCP client...")
            client = MultiServerMCPClient(
                {
                    "image_generation": {
                        "command": "python",
                        "args": ["gemini_image_generator.py"],
                        "transport": "stdio"
                    }
                }
            )

            print("🔧 Getting tools from MCP client...")
            tools = await client.get_tools()
            print(f"✅ Found {len(tools)} tools: {[tool.name for tool in tools]}")

            print("🤖 Initializing LLM...")
            llm = ChatGroq(
                model="llama3-70b-8192",
                temperature=0
            )

            llm = llm.bind_tools(tools)
            print("✅ LLM initialized and tools bound")

            async def model_call(state: AgentState) -> AgentState:
                print(f"🧠 Model call with {len(state['messages'])} messages")
                system_prompt = SystemMessage(
                    content="""You are my AI Assistant. Please answer my query to the best of your ability.
                    - You are an expert in Image Generation
                    - Use the tools to generate images if the user requests them
                    - Be helpful, friendly, and conversational
                    - When generating images, provide a brief description of what you're creating
                    - Keep responses concise but informative
                    """
                )

                print("📤 Sending request to LLM...")
                response = await llm.ainvoke([system_prompt] + state["messages"])
                print(f"📥 Got response from LLM: {response.content[:100]}...")
                return {"messages": [response]}

            def should_continue(state: AgentState) -> str:
                messages = state["messages"]
                last_message = messages[-1]
                if not last_message.tool_calls:
                    print("🔚 No tool calls - ending conversation")
                    return "end_edge"
                else:
                    print(f"🔧 Tool calls found: {len(last_message.tool_calls)}")
                    return "continue_edge"

            print("🏗️ Building agent graph...")
            graph = StateGraph(AgentState)
            graph.add_node("our_agent", model_call)
            
            tool_node = ToolNode(tools=tools)
            graph.add_node("tools", tool_node)
            
            graph.set_entry_point("our_agent")
            
            graph.add_conditional_edges(
                "our_agent", 
                should_continue,
                {
                    "continue_edge": "tools",
                    "end_edge": END
                }
            )
            
            graph.add_edge("tools", "our_agent")
            
            agent_app = graph.compile()
            agent_initialized = True
            print("✅ Agent initialization complete!")
        
        loop.run_until_complete(async_init())
        
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        import traceback
        traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated images"""
    try:
        print(f"📥 Download requested for: {filename}")
        return send_file(filename, as_attachment=True)
    except Exception as e:
        print(f"❌ Download error: {e}")
        return jsonify({"error": str(e)}), 404

@socketio.on('connect')
def handle_connect():
    print('🔗 Client connected')
    emit('status', {'msg': 'Connected to AI Agent'})

@socketio.on('disconnect')
def handle_disconnect():
    print('🔌 Client disconnected')

@socketio.on('message')
def handle_message(data):
    user_message = data['message']
    print(f"📨 Received message: {user_message}")
    
    # Run the agent in a separate thread
    thread = threading.Thread(target=process_message_sync, args=(user_message,))
    thread.daemon = True
    thread.start()

def process_message_sync(user_message):
    """Synchronous wrapper for async message processing"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(process_message(user_message))
    except Exception as e:
        print(f"❌ Error in message processing: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'error': str(e)})

async def process_message(user_message):
    global agent_app, agent_initialized
    
    print(f"🔄 Processing message: {user_message}")
    
    if not agent_initialized:
        print("⚠️ Agent not initialized, initializing now...")
        socketio.emit('status', {'msg': 'Initializing AI Agent...'})
        initialize_agent()
        if not agent_initialized:
            socketio.emit('error', {'error': 'Failed to initialize agent'})
            return
    
    try:
        print("🚀 Starting agent processing...")
        # Create input for the agent
        inputs = {"messages": [HumanMessage(content=user_message)]}
        
        # Process the message through the agent
        response_content = ""
        generated_images = []
        
        print("🔄 Starting agent stream...")
        async for s in agent_app.astream(inputs, stream_mode="values"):
            message = s["messages"][-1]
            print(f"📋 Processing message type: {type(message).__name__}")
            
            if isinstance(message, ToolMessage):
                print(f"🔧 Tool response: {message.content}")
                # Handle tool responses (image generation)
                tool_response = message.content
                
                # Extract image filename if present
                match = re.search(r"saved as ([\w_\-\.]+)", tool_response)
                if match:
                    image_filename = match.group(1)
                    print(f"🖼️ Image generated: {image_filename}")
                    if os.path.exists(image_filename):
                        # Convert image to base64 for display
                        with open(image_filename, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode()
                        
                        generated_images.append({
                            'filename': image_filename,
                            'data': img_data,
                            'timestamp': datetime.now().isoformat()
                        })
                        print(f"✅ Image processed for display")
                    else:
                        print(f"❌ Image file not found: {image_filename}")
            
            elif isinstance(message, AIMessage):
                print(f"🤖 AI response: {message.content}")
                response_content = message.content
        
        print("📤 Sending response to client...")
        # Send the final response
        socketio.emit('ai_response', {
            'content': response_content,
            'images': generated_images,
            'timestamp': datetime.now().isoformat()
        })
        print("✅ Response sent successfully")
    
    except Exception as e:
        print(f"❌ Error in agent processing: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Flask application...")
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Created templates directory")
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
        print("📁 Created static directory")
    
    # Initialize the agent in the background
    print("🔄 Starting background agent initialization...")
    init_thread = threading.Thread(target=initialize_agent)
    init_thread.daemon = True
    init_thread.start()
    
    print("🌐 Starting Flask server on http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
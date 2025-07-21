# client.py
from typing import Annotated, Sequence, TypedDict
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import asyncio

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class StructureOutput(BaseModel):
    name: str = Field(..., description="The name of the Client")
    email: str = Field(..., description="The email of the Client")
    date: str = Field(..., description="The date of the meeting")
    meeting_start: str = Field(..., description="The start time of the meeting")
    meeting_end: str = Field(..., description="The end time of the meeting")

async def main():
    """Main async function containing all the scheduling logic"""
    
    print("🚀 Starting Meeting Scheduler...")
    
    # Setup Pydantic parser
    struct_output = PydanticOutputParser(pydantic_object=StructureOutput)
    
    # Setup prompt template
    prompt = PromptTemplate(
        template="""
        -You are a meeting planner. You will be given a meeting request from a client.
        -You will take all the relative information from the request and plan a meeting.
        -the output should always be as defined below 
        The Structure:
        {format_instructions}

        The Meeting Request:
        {meeting_request}
        """,
        partial_variables={"format_instructions": struct_output.get_format_instructions()},
        input_variables=["meeting_request"]
    )
    
    # Create MCP client
    client = MultiServerMCPClient({
        "AI Scedule Tool": {
            "command": "python",
            "args": ["scedule_tool.py"],
            "transport": "stdio"
        }
    })
    
    try:
        # Start MCP client and get tools
        print("🔧 Setting up MCP client and tools...")
        tools = await client.get_tools()
        
        # Setup LLM with tools
        llm = ChatGroq(model="llama3-70b-8192", temperature=0)
        llm_with_tools = llm.bind_tools(tools)
        
        # Define LLM call function
        def llm_call(state: AgentState) -> AgentState:
            prompt1 = prompt.invoke({"meeting_request": state["messages"][-1].content})
            response = llm_with_tools.invoke(prompt1)
            return {"messages": [response]}
        
        # Define conditional logic
        def should_continue(state: AgentState) -> str:
            messages = state["messages"]
            last_message = messages[-1]
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return "end_edge"
            else:
                return "continue_edge"
        
        # Build the graph
        print("🏗️ Building LangGraph workflow...")
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("llm_call", llm_call)
        tool_node = ToolNode(tools=tools)
        graph.add_node("tools", tool_node)
        
        # Add edges
        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges(
            "llm_call", 
            should_continue,
            {
                "continue_edge": "tools",
                "end_edge": END
            }
        )
        graph.add_edge("tools", END)
        
        # Compile the graph
        app = graph.compile()
        
        # Test the scheduler
        print("🧪 Testing the scheduler...")
        
        input_msg = {
            "messages": [HumanMessage(
                content="Schedule a meeting for a client named Timmy, with email: 0T9QF@example.com, starting at 10:00 and ending at 11:00, on 2025-07-18"
            )]
        }
        
        print("📅 Scheduling meeting...")
        result = await app.ainvoke(input_msg)
        
        print("\n📋 Results:")
        print("-" * 50)
        
        for i, message in enumerate(result["messages"]):
            print(f"Message {i+1}: {type(message).__name__}")
            
            if hasattr(message, 'content') and message.content:
                print(f"   Content: {message.content}")
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"   Tool calls made: {len(message.tool_calls)}")
                for j, tool_call in enumerate(message.tool_calls):
                    print(f"      Tool {j+1}: {tool_call.get('name', 'Unknown')}")
                    print(f"      Args: {tool_call.get('args', {})}")
            
            print()
        
        print("✅ Scheduler test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    

if __name__ == "__main__":
    asyncio.run(main())

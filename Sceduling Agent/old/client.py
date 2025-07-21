# Langchain Imports
from typing import Annotated, Sequence, TypedDict
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
# langgraph Imports
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import asyncio


load_dotenv()


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

class StructureOutput(BaseModel):
    name : str = Field(..., description="The name of the Client")
    email : str = Field(..., description="The email of the Client")
    date: str = Field(..., description="The date of the meeting")
    meeting_start : str = Field(..., description="The start time of the meeting")
    meeting_end : str = Field(..., description="The end time of the meeting")

struct_output = PydanticOutputParser(pydantic_object=StructureOutput)

prompt = PromptTemplate(
    template="""
    -You are a meeting planner. You will be given a meeting request from a client.
    -You will take all the relative information from the request and plan a meeting.
    -the output should always be as defined below 
    The Structure:
    {format_instructions}

    The Meeting Request:
    {meeting_request}
    /n
    """,
    partial_variables={"format_instructions": struct_output.get_format_instructions()},
    input_variables=["meeting_request"]
)

client = MultiServerMCPClient(
            {
                "AI Scedule Tool": {
                    "command": "python",
                    "args" : ["scedule_tool.py"],
                    "transport" : "stdio"
                }
            }
        )

tools = client.get_tools()


llm = ChatGroq(
        model = "llama3-70b-8192",
        temperature= 0
    )

llm = llm.bind_tools(tools)




def llm_call(state: AgentState) -> AgentState:
    prompt1 = prompt.invoke({"meeting_request": state["messages"][-1]})
    response = llm.invoke(prompt1)

    return {"messages" : [response]}


def should_continue(state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end_edge"
        else:
            return "continue_edge"
        

    

graph = StateGraph(AgentState)

graph.add_node("llm call", llm_call)

tool_node = ToolNode(tools=tools)

graph.add_node("tools", tool_node)


graph.add_edge(START, "llm call")

graph.add_conditional_edges(
        "llm call", 
        should_continue,
        {
            "continue_edge": "tools",
            "end_edge": END
        }
    )

graph.add_edge("tools", "llm call")




app = graph.compile()

async def test_scheduler():
    """Test the meeting scheduler"""
    
    # Start MCP client
    await client.start()
    
    try:
        # Test input
        input_msg = {
            "messages": [HumanMessage(
                content="Schedule a meeting for a client named Tom, with email: 0T9QF@example.com, starting at 10:00 and ending at 11:00, on 2025-09-18"
            )]
        }
        
        print("🚀 Scheduling meeting...")
        result = await app.ainvoke(input_msg)
        
        print("📋 Result:")
        for message in result["messages"]:
            if hasattr(message, 'content'):
                print(f"   {message.content}")
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"   Tool calls: {message.tool_calls}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up
        await client.close()

# Alternative synchronous version for testing
def test_scheduler_sync():
    """Synchronous test version"""
    
    input_msg = {
        "messages": [HumanMessage(
            content="Schedule a meeting for a client named Tom, with email: 0T9QF@example.com, starting at 10:00 and ending at 11:00, on 2025-09-18"
        )]
    }
    
    print("🚀 Scheduling meeting...")
    try:
        result = app.invoke(input_msg)
        
        print("📋 Result:")
        for message in result["messages"]:
            print(f"   Type: {type(message)}")
            if hasattr(message, 'content'):
                print(f"   Content: {message.content}")
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"   Tool calls made: {len(message.tool_calls)}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use async version if possible, sync for testing
    try:
        asyncio.run(test_scheduler())
    except:
        print("Trying synchronous version...")
        test_scheduler_sync()



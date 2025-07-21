# Langchain Imports
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
# langgraph Imports
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import asyncio

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


async def main():
    client = MultiServerMCPClient(
            {
                "image_generation": {
                    "command": "python",
                    "args" : ["image_generation_tools.py"],
                    "transport" : "stdio"
                }
            }
        )

    tools = await client.get_tools()

    llm = ChatGroq(
        model = "llama3-70b-8192",
        temperature= 0
    )

    llm = llm.bind_tools(tools)



    # Make a Node for Prompt and User Query
    async def model_call(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(
            content= """You are my AI Assisstant, Please answer my query to the best of your ability
            - You are an expert in Image Generation
            - Use the tools to generate images, if the User Demands


            """
        )

        response = await llm.ainvoke([system_prompt] + state["messages"])

        return {"messages" : [response]}  # Return the response and add it to the state

    # Router
    def should_continue(state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end_edge"
        else:
            return "continue_edge"
        


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


    app = graph.compile()

    # Just for Display
    import os
    import subprocess
    import re

    async def print_stream(stream):
        async for s in stream:
            message = s["messages"][-1]
            if isinstance(message, ToolMessage):
                print("Tool response:", message.content)

                # Try to find a filename in the message
                match = re.search(r"saved as ([\w_\-\.]+)", message.content)
                if match:
                    image_path = match.group(1)
                    if os.path.exists(image_path):
                        print(f"Opening image: {image_path}")
                        subprocess.run(["start", image_path], shell=True)  # Windows
                    else:
                        print(f"Image file not found: {image_path}")
            else:
                message.pretty_print()


    inputs = {"messages": [("user", "make an image of a cat")]}

    await print_stream(app.astream(inputs, stream_mode="values"))


if __name__ == "__main__":
    asyncio.run(main())
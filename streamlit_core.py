import os
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from modules.utils import BasicToolNode, quantity_optimizer, get_nutrients, system_message
# REFERENCES
# 1. groq model-> https://groq.com/introducing-llama-3-groq-tool-use-models/
# 1.b groq pricing -> https://groq.com/pricing/
# 2. langraph with memory -> https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-3-adding-memory-to-the-chatbot

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = os.getenv("LLAMA_MODEL")

# Define state dictionary for the graph
class State(TypedDict):
    messages: Annotated[List, add_messages]
    pippo: Annotated[List, add_messages]

# Initialize memory saver
memory = MemorySaver()
# Configuration for graph updates
config = {"configurable": {"thread_id": "1"}}
# Tool initialization
my_tools = [get_nutrients,quantity_optimizer]

# Initialize the LLM with tools
llm = ChatGroq(
    model=LLAMA_MODEL,
    temperature=0.0,
    max_retries=2,
    api_key=GROQ_API_KEY,
)
llm_with_tools = llm.bind_tools(my_tools)
# Graph definition
graph_builder = StateGraph(State)


# Chatbot node logic
def chatbot(state: State) -> Dict[str, List]:
    """
    Handles user interaction and integrates tool usage when required.
    """
    messages_with_context = [SystemMessage(system_message)] + state["messages"]
    response_to_append = {"messages": [llm_with_tools.invoke(messages_with_context)]}
    return response_to_append

graph_builder.add_node("chatbot", chatbot)

# Streaming graph updates
def stream_graph_updates(user_input: str):
    """
    Streams graph updates based on user input.
    """
    new_message = HumanMessage(user_input)
    for event in graph.stream(
        {"messages": [new_message]}, config, stream_mode="values"
    ):
        message = event["messages"][-1]
        # Check if the message is either AIMessage or HumanMessage
        if isinstance(message, (ToolMessage)):
            continue  # Skip ToolMessage or other types you don't want to display

        else:
            message.pretty_print()

tool_node = BasicToolNode(tools=my_tools)
graph_builder.add_node("tools", tool_node)

# Graph conditional edges
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")  # Return to chatbot after tool execution
graph_builder.set_entry_point("chatbot")
# Compile the graph with memory saver
graph = graph_builder.compile(checkpointer=memory)

# Main loop for user interaction
if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            print(f"Error: {e}")
            break
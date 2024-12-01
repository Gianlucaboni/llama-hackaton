import streamlit as st
from typing import Annotated, List, Dict, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from modules.utils_old import BasicToolNode, quantity_optimizer, get_nutrients, diet_explorer, diet_manager ,system_message, llm
# from langgraph.graph import END
from ast import literal_eval
import json

# Streamlit session state initialization
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "memory_saver" not in st.session_state:
    st.session_state.memory_saver = MemorySaver()

# Initialize memory saver and tools
config = {"configurable": {"thread_id": "1"}}
my_tools = [get_nutrients, quantity_optimizer,diet_explorer,diet_manager]
tool_fatsecret = [get_nutrients]
tool_optimizer = [quantity_optimizer]


llm_with_tools = llm.bind_tools(my_tools)

# Graph definition
# Define state dictionary for the graph
class State(TypedDict):
    messages: Annotated[List, add_messages]
graph_builder = StateGraph(State)

def chatbot(state: State) -> Dict[str, List]:
    messages_with_context = [SystemMessage(system_message)] + state["messages"]
    response_to_append = {"messages": [llm_with_tools.invoke(messages_with_context)]}
    return response_to_append

# def should_continue(state: State) -> Literal["fatsecret", "__end__"]:
#     messages = state['messages']
#     last_message = messages[-1]

#     # If the LLM makes a tool call, then we route to the "tools" node
#     if 'tool_calls' in last_message.additional_kwargs:
#         return "fatsecret"
#     # Otherwise, we stop (reply to the user)
#     return "__end__"

# def should_continue2(state: State) -> Literal["chatbot", "optimizer"]:
#     messages = state['messages']
#     last_message = messages[-1]
#     # If the LLM makes a tool call, then we route to the "tools" node
#     if 'tool_calls' in last_message.additional_kwargs:
#         return "optimizer"
#     # Otherwise, we stop (reply to the user)
#     return "chatbot"

# setting nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", BasicToolNode(my_tools))

# setting edges
graph_builder.add_conditional_edges("chatbot", tools_condition)  # Move to fatsecret if needed
graph_builder.add_edge("tools", "chatbot")  # Return to chatbot after tools

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=st.session_state.memory_saver)

# ~~~~~~~~~~~~~~~~~~ Function to handle user input and append new messages
def handle_user_input(user_input: str):
    new_message = HumanMessage(user_input)
    st.session_state.conversation.append(("You", user_input,None))  # Add user input to conversation

    # Process the input using the chatbot
    state = {"messages": [new_message]}
    events = graph.stream(state, config, stream_mode="values")
    for i,e in enumerate(events):
        messages_to_update = []
        i = -1
        last_message = e["messages"][i]
        messages_to_update.append(last_message)

        if isinstance(last_message, ToolMessage) and last_message.content!="":
            while True:
                i -=1
                previous_message = e["messages"][i]
                if not (isinstance(previous_message, ToolMessage)):
                    break
                else:
                    messages_to_update.append(previous_message)

            tools_used = []
            for call in previous_message.additional_kwargs['tool_calls']:
                tools_used.append(call['function']['name'])
        for i,m in enumerate(messages_to_update):
            m.pretty_print()
            if isinstance(m, ToolMessage) and m.content!="":
                st.session_state.conversation.append(("Tool", m.content,tools_used[i]))  # Add tool response
            elif isinstance(m, AIMessage) and m.content!="":
                st.session_state.conversation.append(("Agent", m.content,None))  # Add agent response

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Streamlit UI setup
st.title("Nutrition Chatbot")
user_input = st.text_input("You: ", "",None)
if st.button("Send"):
    if user_input:
        handle_user_input(user_input)

# ~~~~~~~~~~~~~~~~~~~ Display the conversation
for role, content, tool_name in st.session_state.conversation:
    if role == "You":
        st.markdown(f"**You:** {content}")
    elif role == "Tool":
        try:
            # Attempt to parse content as JSON or evaluate it to handle dictionary-like strings
            parsed_content = literal_eval(content) if isinstance(content, str) else content

            if isinstance(parsed_content, (dict, list)):
                # Format the parsed JSON/dict for readability
                formatted_content = json.dumps(parsed_content, indent=4)
                st.markdown(f"**Tool [{tool_name}]:**\n```json\n{formatted_content}\n```")  # Render as JSON block
            else:
                # If the content is not JSON, render as markdown
                st.markdown(f"**Tool [{tool_name}]:**\n```markdown\n{content}\n```")
        except (ValueError, SyntaxError):
            # Fallback if content cannot be parsed or evaluated
            st.markdown(f"**Tool [{tool_name}]:**\n```markdown\n{content}\n```")
    elif role == "Agent":
        st.markdown(f"**Agent:** {content}")

import streamlit as st
from typing import Annotated, List, Dict, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from modules.utils import BasicToolNode, RecipeSchema, quantity_optimizer, food_info, diet_explorer, retrieve_recipes, recipe_generator ,insert_into_diet,system_message, llm
from langgraph.graph import END
from typing import Annotated, List, Dict, Literal, Optional
from ast import literal_eval
import json
from pydantic import BaseModel, Field


# Streamlit session state initialization
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "memory_saver" not in st.session_state:
    st.session_state.memory_saver = MemorySaver()

if "recipe" not in st.session_state:
    st.session_state.recipe = None


# Initialize memory saver and tools
config = {"configurable": {"thread_id": "1"}}
my_tools = [food_info, quantity_optimizer, diet_explorer, retrieve_recipes,recipe_generator]
llm_with_tools = llm.bind_tools(my_tools)
def add_recipe(left:List[RecipeSchema],right:List[RecipeSchema]) -> List[RecipeSchema]:
    if right is not None:   
        return left+right

# Graph definition
class State(TypedDict):
    messages: Annotated[List, add_messages]
    recipes: Annotated[List[RecipeSchema],add_recipe]
    verification_needed: bool  


graph_builder = StateGraph(State)

def chatbot(state: State) -> Dict[str, List]:
    print(f"STATE NOW - {state.keys()}")
    messages_with_context = [SystemMessage(system_message)] + state["messages"]
    response_to_append = {"messages": [llm_with_tools.invoke(messages_with_context)]}
    return response_to_append


def routing(state):
    messages = state["messages"]
    last_message = messages[-1]
        
    # Check if the last message is an AIMessage with tool_calls
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        destination_node = last_message.tool_calls[0]["name"]
        return destination_node
    
    # If no tool calls or not an AIMessage, end the graph
    return END

# Setting nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("food_info", BasicToolNode([food_info]))
graph_builder.add_node("diet_explorer", BasicToolNode([diet_explorer]))
graph_builder.add_node("quantity_optimizer", BasicToolNode([quantity_optimizer]))
graph_builder.add_node("retrieve_recipes", BasicToolNode([retrieve_recipes],modify_recipe=True))
graph_builder.add_node("recipe_generator",  BasicToolNode([recipe_generator],modify_recipe=True))

# Setting edges
graph_builder.add_conditional_edges("chatbot", routing)
# graph_builder.add_edge("food_info", "is_correct")  # Ask if it's correct

# Direct node connections
for tool_node in ["food_info","diet_explorer", "quantity_optimizer", "retrieve_recipes","recipe_generator"]:
    graph_builder.add_edge(tool_node, "chatbot")  # Return to chatbot after tools

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=st.session_state.memory_saver)

# Verification handling function

# Modified handle_user_input function
def handle_user_input(user_input: str):
    new_message = HumanMessage(user_input)
    st.session_state.conversation.append(("You", user_input,None))  # Add user input to conversation

    # Process the input using the chatbot
    state = {"messages": [new_message]}
    events = graph.stream(state, config, stream_mode="values")
    last_recipe = None

    for i,e in enumerate(events):
        messages_to_update = []
        i = -1
        last_message = e["messages"][i]
        try:
            last_recipe = e["recipes"][i]
        except:
            pass
        st.session_state.recipe = last_recipe
        # print("MY LAST RECIPE: ",last_recipe)
        # st.session_state.recipe = last_recipe
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
                # After food_info tool, set verification state and add verification buttons
                if tools_used[i] in ["recipe_generator","retrieve_recipes"]:
                    st.session_state.verification_needed = True
            elif isinstance(m, AIMessage) and m.content!="":
                st.session_state.conversation.append(("Agent", m.content,None)) 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Streamlit UI setup
st.title("ChefChatBot")

# Add a container for verification message at the top
verification_container = st.empty()

user_input = st.text_input("You: ", "", key="main_user_input")
if st.button("Send"):
    if user_input:
        handle_user_input(user_input)

# ~~~~~~~~~~~~~~~~~~~ Display the conversation
for role, content, tool_name in st.session_state.conversation:
    if role == "You":
        st.markdown(f"**You:** {content}")
    elif role == "Tool":
        try:
            parsed_content = literal_eval(content) if isinstance(content, str) else content

            if isinstance(parsed_content, (dict, list)):
                formatted_content = json.dumps(parsed_content, indent=4)
                st.markdown(f"**Tool [{tool_name}]:**\n```json\n{formatted_content}\n```")
            else:
                st.markdown(f"**Tool [{tool_name}]:**\n```markdown\n{content}\n```")
        except (ValueError, SyntaxError):
            st.markdown(f"**Tool [{tool_name}]:**\n```markdown\n{content}\n```")
    elif role == "Agent":
        st.markdown(f"**Agent:** {content}")

# Add verification buttons after food_info tool
if hasattr(st.session_state, 'verification_needed') and st.session_state.verification_needed:
    st.write("Do you like it ?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, let's save it in my diary!"):
            # Show success message
            retrieved_recipe = st.session_state.recipe
            insert_into_diet(retrieved_recipe)
            verification_container.success("Meal saved successfully!")
            st.session_state.verification_needed = False
    
    with col2:
        if st.button("Mmm, not sure..."):
            # Show warning message
            # verification_container.warning("Me")
            st.session_state.verification_needed = False
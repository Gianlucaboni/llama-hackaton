import operator
from datetime import datetime
from typing import Annotated, TypedDict, Union

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

load_dotenv()


@tool
def sum(a: int, b: int):
    """
    Returns the result of summing 2 numbers, a and b
    """
    return a + b


tools = [sum]

tool_executor = ToolExecutor(tools)


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


OLLAMA_MODEL = "tinyllama"  # openhermes
model = ChatOllama(model=OLLAMA_MODEL)
prompt = hub.pull("hwchase17/react")


agent_runnable = create_react_agent(model, tools, prompt)


def execute_tools(state):
    print("Called `execute_tools`")
    messages = [state["agent_outcome"]]
    last_message = messages[-1]

    tool_name = last_message.tool

    print(f"Calling tool: {tool_name}")

    action = ToolInvocation(
        tool=tool_name,
        tool_input=last_message.tool_input,
    )
    response = tool_executor.invoke(action)
    return {"intermediate_steps": [(state["agent_outcome"], response)]}


def run_agent(state):
    """
    #if you want to better manages intermediate steps
    inputs = state.copy()
    if len(inputs['intermediate_steps']) > 5:
        inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
    """
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


def should_continue(state):
    messages = [state["agent_outcome"]]
    last_message = messages[-1]
    if "Action" not in last_message.log:
        return "end"
    else:
        return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)


workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)


workflow.add_edge("action", "agent")
app = workflow.compile()

input_text = "What is the result of summin eight with seven?"

inputs = {"input": input_text, "chat_history": []}
results = []
for s in app.stream(inputs):
    result = list(s.values())[0]
    results.append(result)
    print(result)

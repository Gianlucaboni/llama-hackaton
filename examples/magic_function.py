from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
print(prompt)


@tool
def magic_function(input: int) -> int:
    "A function that simply adds 2 to the input. Input must be integer"
    return input + 2


tools = [magic_function]

agent = ChatOllama(model="tinyllama", temperature=0.0, num_predict=256)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "what is the value of magic_function(3)?",
        "chat_history": [],
        "agent_scratchpad": [],
    }
)

# # Using with chat history
# from langchain_core.messages import AIMessage, HumanMessage
# agent_executor.invoke(
#     {
#         "input": "what's my name?",
#         "chat_history": [
#             HumanMessage(content="hi! my name is bob"),
#             AIMessage(content="Hello Bob! How can I assist you today?"),
#         ],
#     }
# )

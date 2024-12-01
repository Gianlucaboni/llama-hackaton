from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from modules.utils import llm
from langchain.prompts import PromptTemplate



# Connect to the SQLite database
db = SQLDatabase.from_uri("sqlite:///local.db", include_tables=['diet'])


# Replace with your actual schema
schema = """
    - day (DATE)
    - recipeID (STRING)
    - foodID (STRING)
    - quantity (FLOAT)
    - measurement (STRING)
    - typeMeal (STRING)
    - userID (STRING)
"""

# Define the prompt string
prompt_string = """
You are an advanced AI agent specialized in answering queries related to dietary habits, using the schema of a table named `diet` to fetch accurate information. 

### Table Schema:

    - day (DATE)
    - recipeID (STRING)
    - foodID (STRING)
    - quantity (FLOAT)
    - measurement (STRING)
    - typeMeal (STRING)
    - userID (STRING)

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer
Final Answer: the final answer to the original input question


### Example User Queries:
- "What meals have I planned for dinner this week?"
- "How many recipes have I tried so far?"
- "What did I eat for breakfast last Monday?"

Question: {input}

Thought:{agent_scratchpad}
"""

# Create a PromptTemplate
prompt = PromptTemplate(input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'], template=prompt_string)
print(prompt)

# Create the SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Initialize the agent with tools and the detailed prompt
agent = create_react_agent(llm, toolkit.get_tools(), prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit.get_tools(), verbose=True,handle_parsing_errors=True)

# Invoke the agent with the provided query
response = agent_executor.invoke(
    {
        "input": "how many meal types are available?"
    }
)

print(response)

import json
from langchain_core.messages import ToolMessage
from typing import  List, Dict
from langchain.agents import tool
from langchain_groq import ChatGroq
import os 
from fatsecret import Fatsecret
import numpy as np
from scipy.optimize import minimize
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import logging
import datetime
import sqlite3

# Load environment variables
load_dotenv()

#  ~~~~~~~~~~~~~~~~~~~~~ LLM ~~~~~~~~~~~~~~~~~~
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = os.getenv("LLAMA_MODEL")
# Check if environment variables are loaded
if not GROQ_API_KEY or not LLAMA_MODEL:
    raise ValueError("GROQ_API_KEY and/or LLAMA_MODEL environment variables are not set.")
# Initialize the LLM
llm = ChatGroq(
    model=LLAMA_MODEL,
    temperature=0.2,
    max_retries=2,
    api_key=GROQ_API_KEY,
)


# ~~~~~~~~~~~~~ FatSecret Init ~~~~~~~~~~~~~~~~~~~~~~~~~
CONSUMER_KEY_FATSECRET = os.getenv("CONSUMER_KEY_FATSECRET")
CONSUMER_SECRET_FATSECRET = os.getenv("CONSUMER_SECRET_FATSECRET")
fs = Fatsecret(CONSUMER_KEY_FATSECRET, CONSUMER_SECRET_FATSECRET)


# ~~~~~~~~~~~~~~~~~~System Message, to update as new tools are added ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
system_message = """
                You are RecipeHero, an advanced AI assistant designed to simplify meal planning and nutrition. Alongside offering expert nutritional advice, you can engage in casual conversations to make the user experience more enjoyable. Your key features include:

                Core Capabilities:
                1. Nutritional Insights: Retrieve detailed nutrient data for foods using the FatSecret API.
                2. Food Optimization: Calculate ideal food quantities using the optimizer, which is pre-configured with dietary goals—no need to ask users for them.
                3. Meal Tracking and Planning: Query past meals or planned ones with the diet_explorer tool. This helps users monitor their dietary habits effortlessly.
                4. Saving Meals: Use the diet_manager tool to save meals in the diet table, ensuring proper tracking of users' meal choices.
                5. Recipe Suggestions: By default, propose creative and delicious recipes when users seek inspiration for meals or new ideas.

                Conversational Ability:
                - You can handle casual conversations, respond naturally to user inputs, and suggest meals or recipes even when users aren t explicitly asking for them. Your tone is friendly, engaging, and helpful.

                Guidelines:
                - For Nutritional Information: Use the FatSecret tools to fetch food-specific data.
                - For Food Optimization: Switch to the "Optimizer" node and use the quantity_optimizer tool. Assume dietary goals are pre-configured and do not request input about them.
                - For Meal Tracking: Use the diet_explorer tool. Input natural language queries directly without manually constructing SQL queries.
                - Examples:
                    - "What meals have I planned for dinner this week?"
                    - "How many recipes have I tried so far?"
                    - "What did I eat for breakfast last Monday?"
                - For Saving Meals: Use the diet_manager tool to add meals to the diet table. Input must be formatted as follows:
                {
                    "day": "YYYY-MM-DD",
                    "recipeID": "string",
                    "userID": "string",
                    "typeMeal": "string",
                    "foodItems": [
                        {"foodID": "string", "quantity": float, "measurement": "string"},
                        ...
                    ]
                }
                - For Recipe Suggestions: Be creative! Propose interesting and balanced recipes whenever users ask for ideas or inspiration.

                Default Behavior:
                - When users are unsure about what they want, suggest a recipe or engage them with light, food-related conversation.
                - Always ensure interactions are intuitive and aligned with their dietary needs or preferences.

                With RecipeHero, food planning becomes simpler, healthier, and more delightful. Let s make great meals happen! 
            """


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tool node Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BasicToolNode:
    """
    A node to execute tools requested in the last AIMessage.
    """

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict) -> Dict[str, List[ToolMessage]]:
        if not (messages := inputs.get("messages", [])):
            raise ValueError("No message found in input")

        message = messages[-1]
        outputs = []
        for i,tool_call in enumerate(message.tool_calls):
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
# ~~~~~~~~~~~~~~~~~~~~~~~ Tool definitions ~~~~~~~~~~~~~~~~~~~~~~~~~
@tool
def quantity_optimizer(selected_food:Dict):
    """
    Optimizes the quantities of selected foods.

    Args:
        selected_food (dict): A nested dictionary with the following structure:
            {
                "food_name": {
                    "proteins": float,       # Protein content per 100g
                    "fats": float,          # Fat content per 100g
                    "carbohydrates": float, # Carbohydrate content per 100g
                    "calories": float,      # Calorie content per 100g
                    "min_quantity": float   # Minimum quantity required (optional)
                },
                ...
            }

    Returns:
        dict: Optimal quantities of foods along with total macronutrient and calorie summaries.

    Raises:
        ValueError: If no feasible solution is found or if the input dictionary is invalid.
    """

    # Constants
    BODY_WEIGHT = 72
    MEAL_WEIGHT = 1
    P = MEAL_WEIGHT * BODY_WEIGHT * 1.5  # Target protein (grams)
    G = MEAL_WEIGHT * 70                 # Target fat (grams)
    C = MEAL_WEIGHT * 150                # Target carbohydrates (grams)
    K_MAX = MEAL_WEIGHT * 3000           # Maximum calories (kcal)
    K_MIN = MEAL_WEIGHT * 2500           # Minimum calories (kcal)

    def compute_macros(q, key):
        return sum(q[i] * selected_food[food][key] / 100 for i, food in enumerate(selected_food))

    def objective_function(q):
        total_proteins = compute_macros(q, "proteins")
        total_fats = compute_macros(q, "fats")
        total_carbohydrates = compute_macros(q, "carbohydrates")

        penalties = 0
        penalties += (P - total_proteins)**2 if total_proteins < P else (total_proteins - P) * 0.2
        penalties += (total_fats - G)**2 if total_fats > G else 0
        penalties += (total_carbohydrates - C)**2 if total_carbohydrates > C else 0
        penalties += sum((q[i] - selected_food[food]["min_quantity"])**2
                         for i, food in enumerate(selected_food)
                         if q[i] < selected_food[food]["min_quantity"])
        return penalties

    def calorie_constraint_upper(q):
        return K_MAX - compute_macros(q, "calories")

    def calorie_constraint_lower(q):
        return compute_macros(q, "calories") - K_MIN

    # Initial guesses and constraints
    initial_quantities = np.full(len(selected_food), 20)
    constraints = [
        {'type': 'ineq', 'fun': calorie_constraint_upper},
        {'type': 'ineq', 'fun': calorie_constraint_lower},
    ]
    bounds = [(0, None) for _ in selected_food]

    # Optimization
    result = minimize(objective_function, initial_quantities, constraints=constraints, bounds=bounds)
    if not result.success:
        raise ValueError("Optimization did not find a feasible solution.")

    # Generate result output
    outcome = {}
    for i, food in enumerate(selected_food):
        quantity = result.x[i]
        food_data = selected_food[food]
        outcome[food] = {
            "quantity (g)": round(quantity, 2),
            "calories (kcal)": round(quantity * food_data["calories"] / 100, 2),
            "proteins (g)": round(quantity * food_data["proteins"] / 100, 2),
            "fats (g)": round(quantity * food_data["fats"] / 100, 2),
            "carbohydrates (g)": round(quantity * food_data["carbohydrates"] / 100, 2),
        }

    # Summary
    total_proteins = compute_macros(result.x, "proteins")
    total_fats = compute_macros(result.x, "fats")
    total_carbohydrates = compute_macros(result.x, "carbohydrates")
    total_calories = compute_macros(result.x, "calories")
    outcome["summary"] = {
        "total_proteins (g)": round(total_proteins, 2),
        "total_fats (g)": round(total_fats, 2),
        "total_carbohydrates (g)": round(total_carbohydrates, 2),
        "total_calories (kcal)": round(total_calories, 2),
    }

    return outcome

@tool
def get_nutrients(food:str) -> Dict:
    """
    Retrieves the nutritional information for a specific food item using the FatSecret API.

    This function searches for a food item by name, retrieves its unique identifier from the FatSecret API, 
    and fetches its nutrient details. It specifically attempts to find the serving size that corresponds 
    to 100 grams for standardization. If no such serving is found, it returns the first available serving.

    Args:
        food (str): The name of the food item to search for.

    Returns:
        Dict: A dictionary containing the nutrient information for the selected serving of the food item.
              The keys in the dictionary include nutrient details such as calories, protein, fat, 
              carbohydrates, etc., as provided by the FatSecret API.

    Raises:
        ValueError: If the food name is empty or no matches are found.
        APIError: If there is an issue with the FatSecret API request or response.

    Example:
        >>> get_nutrients("Pasta")
            {'calcium': '1',
            'calories': '157',
            'carbohydrate': '30.68',
            'cholesterol': '0',
            'fat': '0.92',
            'fiber': '1.8',
            'iron': '7',
            'measurement_description': 'g',
            'metric_serving_amount': '100.000',
            'metric_serving_unit': 'g',
            'monounsaturated_fat': '0.130',
            'number_of_units': '100.000',
            'polyunsaturated_fat': '0.317',
            'potassium': '45',
            'protein': '5.77',
            'saturated_fat': '0.175',
            'serving_description': '100 g',
            'serving_id': '320989',
            'serving_url': 'https://www.fatsecret.com/calories-nutrition/generic/penne-cooked?portionid=320989&portionamount=100.000',
            'sodium': '233',
            'sugar': '0.56',
            'vitamin_a': '0',
            'vitamin_c': '0'
        }
    """
        
    list_food = fs.foods_search(search_expression=food)
    #get id first match: TODO have LLM that gets ID best match
    food_id = list_food[0]['food_id']
    servings=fs.food_get(food_id)['servings']['serving']
    try:
        for serving in servings:
            if serving['measurement_description'] == 'g': #and int(serving['metric_serving_amount']) == 100:
                return serving
        return servings[0]
    except:
        return None

@tool
def diet_explorer(question: str):
    """
    A tool to explore the diet table based to reply the question from the user.

    Args:
        question (str): The question asked by the user the Agent has to convert into a valid MySQL query.

    `diet` table schema:
        - day (DATE)
        - recipeID (STRING)
        - foodID (STRING)
        - quantity (FLOAT)
        - measurement (STRING)
        - typeMeal (STRING)
        - userID (STRING)

    """
    # Connect to the database and include the "weekly_meals" table
    TABLE = os.getenv("DIET_TABLE")
    db = SQLDatabase.from_uri("sqlite:///local.db", include_tables=[TABLE])


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

    You don't need to fetch the diet table schema, as that is specified here:
    ### `diet` Table Schema:

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

    Loop thorugh the following though process until you get the correct answer.

        - Thought: you should always think about what to do
        - Action: the action to take, should be one of [{tool_names}]
        - Action Input: the input to the action
        - Observation: the result of the action

    As you get the correct answer returns:
        - Final Answer: the output from the final sql query.


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
    agent_executor = AgentExecutor(agent=agent, 
                                   tools=toolkit.get_tools(), 
                                   max_execution_time=5,
                                   verbose=True,
                                   handle_parsing_errors=True,
                                   return_intermediate_steps=False
                                   )

    # Invoke the agent with the provided query
    try:
        response = agent_executor.invoke({"input": question})
        return response

    except Exception as e:
        logging.error(f"Error during agent execution: {e}")
        raise

# def parse_recipe_input(file_list):
#     """
#     Legge e valida una lista di file JSON per popolare la tabella diet.

#     Args:
#         file_list (list): Lista di file JSON.

#     Returns:
#         list: Lista di dizionari con i dati da inserire.
#     """
#     records = []

#     for file in file_list:
#         with open(file, 'r') as f:
#             content = json.load(f)
            
#             # Validazione dei campi principali
#             for entry in content:
#                 if not all(key in entry for key in ["date", "recipeID", "userID", "typeMeal", "foodItems"]):
#                     raise ValueError(f"Missing keys in entry: {entry}")
                
#                 # Validazione formato della data
#                 try:
#                     datetime.strptime(entry["date"], '%Y-%m-%d')
#                 except ValueError:
#                     raise ValueError(f"Invalid date format in entry: {entry['date']}")

#                 # Itera sugli alimenti della ricetta
#                 for item in entry["foodItems"]:
#                     if not all(key in item for key in ["foodID", "quantity", "measure"]):
#                         raise ValueError(f"Missing keys in food item: {item}")
                    
#                     # Preparare una riga per la tabella
#                     record = {
#                         "day": entry["date"],
#                         "recipeID": entry["recipeID"],
#                         "userID": entry["userID"],
#                         "typeMeal": entry["typeMeal"],
#                         "foodID": item["foodID"],
#                         "quantity": item["quantity"],
#                         "measurement": item["measure"]
#                     }
#                     records.append(record)
    
#     return records

def insert_into_diet(recipe:Dict) -> None:
    """
    Inserisce i record nella tabella diet.

    Args:
        recipe (Dict): recipe dictionary object
    """
    connection = sqlite3.connect("local.db")
    cursor = connection.cursor()
    
    try:
        for item in recipe['foodItems']:
            query = """
            INSERT INTO diet (day, recipeID, foodID, quantity, measurement, typeMeal, userID)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (
                recipe["day"],
                recipe["recipeID"],
                item["foodID"],
                item["quantity"],
                item["measurement"],
                recipe["typeMeal"],
                recipe["userID"]
            ))
        
        connection.commit()
        print(f"{len(recipe['foodItems'])} rows inserted into the diet table.")
    except sqlite3.IntegrityError as e:
        print(f"Error inserting data: {e}")
    finally:
        connection.close()

@tool
def diet_manager(recipe:Dict) -> Dict:
    """
    Handles the insertion of meal data into the `diet` table in a database.

    This function reads a dictionary containing meal information, validates the input,
    and inserts the data into the `diet` table.

    Args:
        recipe: Dictionary of storing following information assocated to the recipe to be save
                          {
                              "day": "YYYY-MM-DD",
                              "recipeID": "string",
                              "userID": "string",
                              "typeMeal": "string",
                              "foodItems": [
                                  {"foodID": "string", "quantity": float, "measure": "string"},
                                  ...
                              ]
                          }

    """

    # records = parse_recipe_input(file_list)
    print(recipe,type(recipe))
    # Inserimento nel database
    insert_into_diet(recipe)

    return recipe
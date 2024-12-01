from langchain_core.messages import ToolMessage
from typing import  List, Dict, Any
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
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import logging
from pinecone.grpc import PineconeGRPC as Pinecone
from openai import OpenAI
import pandas as pd
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
from typing import Annotated, List, Dict, Literal, Optional
from pydantic import BaseModel, Field


# ~~~~~~~~~~~~~ FatSecret Init ~~~~~~~~~~~~~~~~~~~~~~~~~
CONSUMER_KEY_FATSECRET = os.getenv("CONSUMER_KEY_FATSECRET")
CONSUMER_SECRET_FATSECRET = os.getenv("CONSUMER_SECRET_FATSECRET")
fs = Fatsecret(CONSUMER_KEY_FATSECRET, CONSUMER_SECRET_FATSECRET)

PINECONE_KEY = os.getenv("PINECONE_KEY")
pc = Pinecone(api_key=PINECONE_KEY)
# Define a Pydantic model for the recipe schema
# Recipe Schema Definition
class FoodItem(BaseModel):
    foodID: str
    quantity: float
    measurement: str

class RecipeSchema(BaseModel):
    day: str
    recipeID: str
    typeMeal: str
    userID: str
    foodItems: List[FoodItem]
    description: str


system_message = """
                You are ChefChatBot, an advanced AI assistant designed to simplify meal planning and nutrition. 
                Beyond providing expert nutritional advice, you can engage in casual conversations to make 
                the user experience more enjoyable. Your key features include:
                Explain your capabilites in detail if asked.

                Core Capabilities:
                1. **Nutritional Insights**: Retrieve detailed nutrient data for foods using the FatSecret API.
                2. **Food Optimization**: Calculate ideal food quantities using the optimizer, pre-configured with 
                   dietary goals—no need to ask users for them.
                3. **Meal Tracking and Planning**: Use the diet_explorer tool to query past meals or planned ones, 
                   helping users monitor their dietary habits effortlessly. **IMPORTANT** NEVER TRY TO GENERATE SQL QUERIES: just pass what user asked.
                4. **Retrieve Well Known Recipes**: Retrieve standard recipes from a catalog of well-known dishes.
                5. **Recipe Creation**: Generate creative and balanced recipes from scratch tailored to users' dietary 
                   needs or preferences.

                **How You Can Help**:
                - You can either **create a new recipe** from scratch based on the user’s needs or preferences, 
                  or **retrieve a well-known recipe** from a catalogue. 
                - Please ask the user whether they want to create a recipe or find an existing one when they request 
                  a recipe.

                Conversational Ability:
                - Handle casual conversations and respond naturally to user inputs.
                - Proactively suggest meals or recipes even when users aren’t explicitly asking for them.
                - Maintain a friendly, engaging, and helpful tone.

                Guidelines:
                - **Nutritional Information**: Use the FatSecret tools to fetch food-specific data.
                
                - **Food Optimization**: Switch to the "Optimizer" node and use the quantity_optimizer tool. Dietary 
                  goals are pre-configured, so avoid asking users about them.
                
                - **Meal Tracking**: Use the diet_explorer tool to explore past or planned meals. Input natural language 
                  queries directly. **Do NOT manually attempt constructing SQL queries**.
                    Example of valid arguemnts to pass:
                    - "What meals have I planned for dinner this week?"
                    - "How many recipes have I tried so far?"
                    - "What did I eat for breakfast last Monday?"
                
                - **Recipe Suggestions**: Be creative! Propose interesting, balanced recipes whenever users seek 
                  inspiration.

                Default Behavior:
                - When users are uncertain about what they want, suggest a recipe or engage them in light, 
                  food-related conversation.
                - Always ensure interactions are intuitive, aligned with their dietary needs, and respectful of their preferences.

                With ChefChatBot, food planning becomes simpler, healthier, and more delightful. Let's make great meals happen!
            """

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tool node Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BasicToolNode:
    def __init__(self, tools: list,modify_recipe: bool=False) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.modify_recipe = modify_recipe

    def __call__(self, inputs: dict) -> Dict[str, List[Any]]:
        if not (messages := inputs.get("messages", [])):
            raise ValueError("No message found in input")

        message = messages[-1]
        outputs = []
        
        # Ensure we're working with an AIMessage with tool_calls
        if not hasattr(message, 'tool_calls'):
            return {"messages": []}

        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            
            # Create ToolMessage with additional context
            tool_message = ToolMessage(
                content=str(tool_result),  # Convert to string instead of JSON
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                # Add tool_calls to maintain context
                additional_kwargs={
                    'tool_calls': [tool_call]
                }
            )
            outputs.append(tool_message)
        if self.modify_recipe:
           return {"messages": outputs,"recipes":[tool_result]}
        return {"messages": outputs,"recipes":None}
    
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
    K_MAX = MEAL_WEIGHT * 2500           # Maximum calories (kcal)
    K_MIN = MEAL_WEIGHT * 1800           # Minimum calories (kcal)

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
def food_info(food:str) -> Dict:
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
        - userID (STRING) - user "user1"

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
    You are an advanced AI agent specialized in answering queries related to dietary habits of userID = "user1"
    Using the schema of a table named `diet` to fetch accurate information. 

    **IMPORTANT **: When possible returns list of unique items rather than many duplicates making use of DISTINCT key word

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

    # Create the SQLDatabaseToolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Initialize the agent with tools and the detailed prompt
    agent = create_react_agent(llm, toolkit.get_tools(), prompt)
    agent_executor = AgentExecutor(agent=agent, 
                                   tools=toolkit.get_tools(), 
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


# Recipe Generation Function
def generate_recipe_with_llm(
    userID: str,
    day: str,
    typeMeal: str,
    requirements: str = "",
    description: str = "",
    **kwargs
) -> RecipeSchema:
    """
    Generate a recipe using an LLM with explicit schema instructions.
    """
    if not userID:
        raise ValueError("userID is a required parameter")
    if not day:
        raise ValueError("day is a required parameter")
    if not typeMeal:
        raise ValueError("typeMeal is a required parameter")

    # Create prompt template
    prompt_template = f"""
    You are a professional nutritionist and chef tasked with generating a recipe.
    If a recipe is already provided, sitcj with it, otherwise make one based on the 
    requirements

    REQUIRED OUTPUT SCHEMA:
    {{
        "day": "YYYY-MM-DD",
        "recipeID": "unique_identifier",
        "typeMeal": "breakfast/lunch/dinner/snack",
        "userID": "user_identifier",
        "foodItems": [
            {{
                "foodID": "string",
                "quantity": float,
                "measurement": "string"
            }}
        "description": "detailed directions step by step"
        ]
    }}

    User Inputs:
    - Date: {day}
    - Type of Meal: {typeMeal}
    - userID : {userID}
    - Additional Requirements: {requirements}
    - Description: {description}

    Guidelines:
    1. STRICTLY FOLLOW the schema.
    2. Populate all fields with realistic, available data.
    3. Ensure a nutritionally balanced recipe.
    4. Provide detailed directions and follow the description if any is given

    IMPORTANT:
    - Respond ONLY with a valid JSON following the schema.
    """


    # Add LLM invocation here (example for structure, replace `llm.invoke`)
    try:
        # Replace with actual LLM integration
        llm_response = llm.invoke(prompt_template)  # Simulated response
        #error here !
        recipe_data = JsonOutputParser(pydantic_object=RecipeSchema).parse(llm_response.content)

        # Return parsed schema
        return RecipeSchema(**recipe_data)
    
    except Exception as e:
        print(f"Error generating recipe: {e}")
        # Fallback recipe
        return RecipeSchema(
            day=day,
            recipeID=f"fallback_recipe_{day}",
            typeMeal=typeMeal,
            userID=userID,
            foodItems=[
                FoodItem(foodID="fallback_protein", quantity=150, measurement="grams"),
                FoodItem(foodID="fallback_carb", quantity=100, measurement="grams"),
            ],
            description = ""
        )

# Tool Wrapper
@tool
def recipe_generator(
    userID: str,
    day: str,
    meal_type: str,
    additional_requirements: str = ""
) -> Dict:
    """
    Recipe Generation Tool.

    Generates a personalized recipe tailored to the user's preferences and dietary needs. 
    This tool uses advanced algorithms to create creative, balanced, and delicious meal ideas.
    
    Args:
        userID (str): User identifier.
        day (str): Date for the recipe.
        meal_type (str): Type of meal (breakfast, lunch, dinner, snack).
        additional_requirements (str): Custom requirements for the recipe.

    Returns:
        Dict: Generated recipe as a dictionary.
    """
    recipe = generate_recipe_with_llm(
        userID='user1',
        day=day,
        typeMeal=meal_type,
        requirements=additional_requirements,
        description=""
    )
    return recipe

def insert_into_diet(recipe: RecipeSchema) -> None:
    """
    Inserisce i record nella tabella diet.
    """
    connection = sqlite3.connect("local.db")
    cursor = connection.cursor()
    
    print(recipe.day)  # Accessing the 'day' attribute directly
    
    try:
        for item in recipe.foodItems:  # Accessing the foodItems attribute directly
            query = """
            INSERT INTO diet (day, recipeID, foodID, quantity, measurement, typeMeal, userID)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (
                recipe.day,
                recipe.recipeID,
                item.foodID,
                item.quantity,
                item.measurement,
                recipe.typeMeal,
                recipe.userID
            ))
        
        connection.commit()
        print(f"{len(recipe.foodItems)} rows inserted into the diet table.")
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

@tool
def retrieve_recipes(
    query: str,
    namespace: str ="recipes",
    top_k: int = 1,
    metadata_filters=None,
) -> pd.DataFrame:
    """
    Retrieves the most relevant recipes based on a user's query using a semantic search. 
    This function performs retrieval-augmented generation (RAG) by searching for recipes 
    stored in a Pinecone vector database, filtering results using optional metadata, and 
    ranking the results by similarity.

    The function:
    - Embeds the user's query into a vector representation.
    - Normalizes the vector to ensure consistent similarity comparison.
    - Queries the Pinecone database within the specified namespace.
    - Retrieves the top-k matching results, including their metadata.

    :param query: The user's search query, which is semantically encoded for retrieval.
    :param namespace: The namespace within Pinecone to search (default: "recipes").
    :param top_k: The number of top matching results to return (default: 10).
    :param metadata_filters: Optional metadata filters to apply when querying the database.
    """

    def normalize_l2(x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x
            return x / norm
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm)

    # Initialize Pinecone index and query embeddings
    index = pc.Index("llama-hackathon-256")

    # Perform vector search using the embedded query
    client = OpenAI()
    # Create embeddings using the client
    response = client.embeddings.create(
        model="text-embedding-3-small", input=query, encoding_format="float"
    )
    
    # Cut and normalize the embedding
    cut_dim = response.data[0].embedding[:256]
    embedded_query = normalize_l2(cut_dim)
    search_results = index.query(
        vector=embedded_query,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter=metadata_filters,
    )
    description = search_results.matches[0]["metadata"]["description"]
    print(description)
    recipe = generate_recipe_with_llm(
        userID='user1',
        day='2024-12-01',
        typeMeal='lunch',
        requirements=None,
        description=description
    )

    return recipe

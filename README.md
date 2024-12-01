# ChefChatBot with LangGraph Agent and Groq-powered LLAMA Model

## Overview

**ChefChatBot** is an intelligent conversational agent that uses advanced AI to assist users with dietary needs, recipe generation, and nutritional information. Powered by **Groq's LLAMA Model** (`llama3-groq-70b-8192-tool-use-preview`), the bot leverages **LangGraph**, a flexible framework for orchestrating multi-tool conversational agents, enabling dynamic interactions through various tools and memory capabilities.

---

## Features

1. **Groq's LLAMA Model**:
   - High-performance natural language understanding and generation.
   - Supports tool integration for diverse tasks.

2. **LangGraph Framework**:
   - Implements a **StateGraph** for stateful conversations.
   - Allows modular addition of tools and flexible routing based on user input.

## 🛠 Integrated Tools and Functionality

### Tool Overview

The project leverages several powerful tools and APIs to create an intelligent meal planning and nutrition assistance system:

#### 1. Quantity Optimizer Tool (`quantity_optimizer`)
- **Purpose**: Optimize food quantities based on nutritional goals
- **Key Features**:
  - Calculates optimal quantities of foods
  - Considers target macronutrients (proteins, fats, carbohydrates)
  - Respects calorie constraints
  - Utilizes scipy's minimize function for optimization
- **Input**: Dictionary of foods with nutritional information
- **Output**: Optimized food quantities with detailed nutritional breakdown

#### 2. Food Information Tool (`food_info`)
- **Purpose**: Retrieve detailed nutritional information for food items
- **Key Features**:
  - Integrates with FatSecret API
  - Searches for food items by name
  - Retrieves standardized (100g) nutrient details
- **Input**: Food name (string)
- **Output**: Comprehensive nutritional information dictionary

#### 3. Diet Explorer Tool (`diet_explorer`)
- **Purpose**: Query and explore past dietary habits
- **Key Features**:
  - Uses SQLAlchemy for database interactions
  - Supports natural language queries about diet history
  - Handles complex query transformations
- **Input**: Natural language question about diet
- **Output**: Relevant dietary information from the database

#### 4. Recipe Generator Tool (`recipe_generator`)
- **Purpose**: Create personalized, nutritionally balanced recipes
- **Key Features**:
  - Generates recipes based on user preferences
  - Follows a strict JSON schema for recipe creation
  - Integrates with Language Model for creative recipe generation
- **Input**: User ID, date, meal type, additional requirements
- **Output**: Structured recipe object

#### 5. Diet Manager Tool (`diet_manager`)
- **Purpose**: Manage and persist meal information
- **Key Features**:
  - Inserts recipe data into SQLite database
  - Validates and transforms recipe data
- **Input**: Recipe dictionary
- **Output**: Confirmed recipe after database insertion

#### 6. Recipe Retrieval Tool (`retrieve_recipes`)
- **Purpose**: Semantic search for recipes
- **Key Features**:
  - Uses Pinecone vector database for similarity search
  - Generates embeddings with OpenAI's embedding model
  - Supports metadata-based filtering
- **Input**: Search query, optional filters
- **Output**: Most relevant recipe based on semantic similarity

### Technology Stack
- **Language Models**: Groq (LLaMA)
- **Vector Database**: Pinecone
- **Nutritional API**: FatSecret
- **Optimization**: SciPy
- **Database**: SQLite
- **Embedding**: OpenAI

### System Architecture
The system uses a modular, tool-based approach with:
- Intelligent agent framework
- Retrieval-augmented generation
- Optimization algorithms
- Semantic search capabilities

3. **Integrated Tools**:
   - **Food Info**: Provides nutritional information about ingredients.
   - **Diet Explorer**: Helps explore diets based on preferences and constraints.
   - **Quantity Optimizer**: Suggests optimized quantities for recipes.
   - **Retrieve Recipes**: Fetches recipes from external sources or databases.
   - **Recipe Generator**: Creates customized recipes based on user input.

4. **Memory Management**:
   - Uses `MemorySaver` for storing session data.
   - Supports long-term stateful interactions with users.

5. **Streamlit User Interface**:
   - Interactive UI for real-time conversation and visualization of tool responses.
   - Supports user input, tool-based suggestions, and feedback loops.

---

## How It Works

### Architecture

The core logic is built around a **StateGraph** that defines the flow of messages and tool invocations. The bot processes user inputs, invokes necessary tools, and routes the response back to the user, ensuring conversational coherence.

1. **Session Initialization**:
   - Initializes conversation history, memory saver, and tools in the Streamlit session state.

2. **Graph Definition**:
   - Nodes represent the chatbot logic and tool invocations.
   - Conditional edges handle routing based on AI-generated `tool_calls`.

3. **Tool Integration**:
   - Each tool is wrapped as a node in the graph, ensuring modularity and reusability.

4. **Conversation Handling**:
   - Captures user input as `HumanMessage`.
   - Processes the conversation state using the compiled graph and tools.
   - Outputs responses as `AIMessage` or `ToolMessage`.

5. **Feedback and Verification**:
   - Prompts users to verify recipe suggestions.
   - Saves approved recipes to a personal diary for future reference.

---

## Streamlit Interface

### Features

- **Dynamic Conversation Display**:
  - Displays user inputs, tool responses, and agent replies.
  - Renders tool responses in JSON or Markdown for better readability.

- **Interactive Feedback Buttons**:
  - Allows users to save or reject generated recipes with a single click.

- **Real-Time Updates**:
  - Updates conversation state and recipe suggestions seamlessly.

---

## Usage

### Running the App

1. Install the required dependencies:
   ```bash
   poetry install
   ```

2. Launch the app:
   ```bash
   streamlit run chatbot.py
   ```

3. Interact with the chatbot through the Streamlit interface.

## Future Enhancements

1. **Expand Toolset**:
   - Incorporate tools for grocery shopping and allergy checks.
   - Enhance semantic search capabilities
   - Improve recipe personalization algorithms

---

## Acknowledgments

- **Groq Inc.** for providing the `llama3-groq-70b-8192-tool-use-preview` model.
- **LangGraph** for enabling a scalable and modular conversational architecture.

--- 

Enjoy your journey with **ChefChatBot**—your AI-powered culinary assistant! 🎉
import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta

# Database and table setup
DB_NAME = "local.db"
TABLE_NAME = "weekly_meals"

def generate_mock_data():
    """Generates 300 random records for the diet table."""
    # Define random data generation ranges
    start_date = datetime.now()
    meal_types = ["Breakfast", "Lunch", "Dinner", "Snack"]
    measurements = ["grams", "liters", "pieces", "cups"]
    user_ids = ["user1", "user2", "user3", "test"]  # Simulated user IDs
    
    data = []
    for _ in range(300):
        record = {
            "day": (start_date + timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
            "recipeID": str(random.randint(1, 20)),  # Random recipe IDs as strings
            "foodID": str(random.randint(1, 50)),  # Random food IDs as strings
            "quantity": round(random.uniform(0.5, 5.0), 2),  # Random float for quantity
            "measurement": random.choice(measurements),
            "typeMeal": random.choice(meal_types),
            "userID": random.choice(user_ids),
        }
        data.append(record)
    
    return pd.DataFrame(data)

def create_table_and_insert_data(df, db_name, table_name):
    """Creates a table in SQLite and inserts the data."""
    with sqlite3.connect(db_name) as conn:
        
        # Insert the data
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Inserted {len(df)} records into the {table_name} table.")

def read_df(db_name: str, query: str) -> pd.DataFrame:
    """
    Executes a SQL query on the specified database and returns the result as a pandas DataFrame.

    Args:
        db_name (str): The name of the SQLite database file.
        query (str): The SQL query to execute.

    Returns:
        pd.DataFrame: The result of the query as a pandas DataFrame.
    """
    with sqlite3.connect(db_name) as conn:
        df = pd.read_sql_query(query, conn)
    return df

if __name__ == "__main__":
    # Generate mock data
    df = generate_mock_data()
    
    # Insert the data into the SQLite database
    create_table_and_insert_data(df, DB_NAME, TABLE_NAME)
    
    print(f"Data successfully saved to {DB_NAME} in table {TABLE_NAME}.")

import pandas as pd
from dotenv import load_dotenv
import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from time import time
from typing import List
from time import time

load_dotenv()
PINECONE_KEY = os.getenv("PINECONE_KEY")
pc = Pinecone(api_key=PINECONE_KEY)

def create_index(index_name: str = "eu",
                 embedding_dim:int = 3072,
                 metric:str = "cosine"
                 ) -> None:
    """
    Create an index with the specified name and wait until it is ready.

    Args:
        index_name (str): The name of the index to create. Defaults to 'eu'.

    Returns:
        None
    """
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

def from_df_to_pinecone(
    df: pd.DataFrame, 
    index_name: str, 
    namespace: str, 
    batch_size: int = 10
) -> List[str]:
    """
    Load data from a DataFrame into a Pinecone index in batches.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be loaded.
        index_name (str): The name of the Pinecone index.
        namespace (str): The namespace for the Pinecone index.
        batch_size (int): The number of records to load in each batch. Default is 150.

    Returns:
        List[str]: A list of IDs of the loaded records.
    """

    # checking Dataframe format
    assert all(
        col in df.columns for col in ["id", "embedding", "metadata"]
    ), "DataFrame must have columns 'id', 'embedding', and 'metadata'"

    ids_loaded = []
    index = pc.Index(index_name)

    for i in range(0, len(df), batch_size):
        try:
            batch = df.iloc[i : i + batch_size]
            new_documents = []
            ids_batch = batch["id"].tolist()

            # Process each document in the batch
            for _, row in batch.iterrows():
                new_documents.append(
                    {
                        "id": row["id"],
                        "values": row["embedding"],
                        "metadata": row["metadata"], #it must a python dictionary !
                    }
                )

            # Upsert the documents if there are any
            if new_documents:
                index.upsert(vectors=new_documents, namespace=namespace)
                # extend ids_loaded list
                ids_loaded.extend(ids_batch)
                print(f"Batch {i} pushed")

        except Exception as e:
            print(f"Error {e} - batch {i}")
            continue
    return ids_loaded

def retrieve_documents(
    namespace: str,
    query: str,
    top_k: int = 10,
    metadata_filters=None,
) -> pd.DataFrame:
    """
    Retrieves documents from Pinecone, filters out undesired keys, and returns a DataFrame.

    :param namespace: The namespace to search within.
    :param query: The query to search for.
    :param top_k: The number of top documents to retrieve.
    :param keys_to_drop: List of keys to drop from the document metadata.
    :param metadata_filters: Filters to apply to the metadata.
    """
    # Initialize Pinecone index and query embeddings
    index = pc.Index("eu")

    # Perform vector search using the embedded query
    #TODO embedded_query = # embedding_model.embed_query(query)
    search_results = index.query(
        vector=embedded_query,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter=metadata_filters,
    )
    return search_results
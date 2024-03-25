from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel


import tabulate
import operator
import os
import pandas as pd
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv, dotenv_values 

load_dotenv()


def initilize_api_keys():
    """ This function sets environment variable """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = 'true'
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] ="multi-agent"
    os.environ['OPENAI_API_KEY'] = openai_api_key
    
def create_csv_agent_function(file_path):

    file_path= file_path

    csv_agent= create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-4"),
            file_path, 
            verbose=True,
            stop=["\nObservation:"],
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True
        )
        
    return csv_agent

def create_prompt():
    prompt = """ You are a question-answering bot over the data you queried from dataframes.
            Give responses to the question on the end of the text :
                
                
                Given a dataset from an e-commerce platform detailing sales transactions over a year, analyze the data to understand consumer behavior and sales patterns. This dataset originates from a UK-based online retail shop specializing in gifts and homewares for both adults and children, operating since 2007. With customers worldwide, the business model includes direct consumer sales and bulk purchases by small businesses for resale.

                Dataset Description:

                - TransactionNo (categorical): A unique six-digit number for each transaction. Transactions beginning with "C" denote cancellations.
                - Date (numeric): The date of each transaction.
                - ProductNo (categorical): A unique identifier for each product, consisting of five or six digits.
                - Product (categorical): The name of the product/item.
                - Price (numeric): The price per unit of each product in pound sterling (£).
                - Quantity (numeric): The number of units per product in each transaction. Negative quantities indicate cancelled transactions.
                - CustomerNo (categorical): A unique five-digit number identifying each customer.
                - Country (categorical): The customer's country of residence.

                A notable aspect of this dataset is the presence of order cancellations, often due to products being out of stock. Customers generally prefer all items in their order to be delivered simultaneously, leading to cancellations when this isn't possible.

                
                Below is the question
                Question:"""
    return prompt

class InputQuery(BaseModel):
    query: str

app = FastAPI()

@app.put("/csv_agent/")
def csv_agent_api(query: InputQuery):
    initilize_api_keys()
    csv_agent = create_csv_agent_function("data.csv")
    prompt = create_prompt()
    output = csv_agent(prompt + query.query)
    return {'Query': query.query, 'Output': output['output']}
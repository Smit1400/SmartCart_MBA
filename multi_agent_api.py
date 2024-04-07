from fastapi import FastAPI
from pydantic import BaseModel
import streamlit as st
import os
from io import StringIO
import sys
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_agent_executor

# from langchain_core.pydantic_v1 import BaseModel
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict, List, Tuple
from dotenv import load_dotenv, dotenv_values
import operator
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import tool
import random
import tabulate
import pandas as pd
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langgraph.prebuilt.tool_executor import ToolExecutor

# from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentFinish
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = "multi-agent"
os.environ["OPENAI_API_KEY"] = openai_api_key

model = ChatOpenAI(temperature=0, max_tokens=1024, model_name="gpt-4")

file_path = "data.csv"

csv_agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-4"),
    file_path,
    verbose=True,
    stop=["\nObservation:"],
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
)


@tool("insights", return_direct=False)
def get_actionable_insights(input: str):
    """Return actionable business insights from the dataset either for a country or the whole dataset"""
    question = f"Give me actionable business insights for {input}"
    prompt = """
        Given a specific country or the entire dataset, generate actionable business insights, including but not limited to overall revenue, top-selling product, customer with the highest number of purchases, and the most popular product in the current period.
    """
    insights = csv_agent.run(prompt)
    return insights


@tool("chart_pdf", return_direct=False)
def get_charts_pdf(input: str):
    """Returns python executable code that first loads data.csv and generates a pdf with charts"""
    question = f"Give me a detailed report for {input}"

    prompt = (
        """ Generate Python code that creates a comprehensive series of graphs to analyze business data for country. Also include the code for loading the data (data.csv)
    The visualization should potentially include, but not be limited to, a bar plot of the top-selling products, a line plot showing sales trends over time, and a pie chart detailing the sales distribution among top customers. 
    Additionally, consider including other relevant analyses such as sales by category, regional sales distribution, and seasonality effects on sales, if the data permits. 

    The graphs should be stored in a PDF named 'sales_report.pdf'. Utilize pandas for data handling, matplotlib for plotting, and the matplotlib.backends.backend_pdf module for generating the PDF file. 
    The code should be well-commented to explain the purpose of each graph and how it contributes to understanding the business climate in `country_name`. Include brief insights or interpretations for each plot to highlight the key takeaways and business implications derived from the data analysis.
    """
        + question
    )
    charts = csv_agent.run(prompt)
    charts = charts.split("python")[1].split("```")[0]
    exec(charts)
    return "pdf"


tools = [get_actionable_insights, get_charts_pdf]

tool_executor = ToolExecutor(tools)

functions = [convert_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    print(f"The agent action is {action}")
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    print(f"The tool result is: {response}")
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)

workflow.add_edge("action", "agent")

graph = workflow.compile()


class Question(BaseModel):
    query: str


app = FastAPI()


@app.put("/multi_agent/")
def csv_agent_api(query: Question):
    system_message = SystemMessage(content="you are a helpful assistant")
    user_01 = HumanMessage(content=query.query)
    inputs = {"messages": [system_message, user_01]}
    result = graph.invoke(inputs)
    output = result["messages"][-1].content
    return {"Query": query.query, "Output": output}

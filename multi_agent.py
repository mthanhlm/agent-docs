from langchain.agents import create_agent
from pydantic import BaseModel, Field
from tools import web_search, get_current_weather, calculator
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, START, END
from typing import Literal
from langgraph.types import Command
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}
model = ChatOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("GROQ_API_KEY"), model=os.getenv("MODEL_NAME", "gpt-4o"))


class TeamState(BaseModel):
    messages: Annotated[list, add_messages]
    next_agent: Literal["search_agent", "calculator_agent", "weather_agent"] 

parser = PydanticOutputParser(pydantic_object=TeamState)

prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. 

QUESTION:
{messages}

FORMAT:
{format}
"""
)

prompt = prompt.partial(format=parser.get_format_instructions())


def communicate_agent(state: TeamState) -> Command[Literal[END]]:
    communicate_agent = create_agent(
        model=model,
        system_prompt="You are a helpful AI assistant. You receive user queries and delegate tasks to specialized agents: search_agent, calculator_agent, and weather_agent. Use the responses from these agents to formulate a final answer to the user. Always use context from Knowledge Base (RAG) and available tools to answer precisely. Answer in short all needed information",
        checkpointer=checkpointer,
        tools=[web_search, calculator, get_current_weather],
    )
    response = communicate_agent.invoke(
        {"messages": prompt.format_messages(messages=state["messages"])}
    )
    # return Command(
    #     goto=response["next_agent"],
    #     update={"messages": [response["content"]]},
    # )
    # return print(response['structured_response'])
    return print(response)
def search_agent(state: State) -> Command[Literal['communicate_agent','calculator_agent', 'weather_agent' ,END ]]:
    search_agent = create_agent(
        model=model,
        tools=[web_search],
        system_prompt="You are a helpful AI assistant. Your task is search information. Answer in short all needed information",
        checkpointer=checkpointer,
        response_format=State
    )
    response = search_agent.invoke(
    {"messages": state["messages"]}
)
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

def calculator_agent(state: State) -> Command[Literal['communicate_agent', 'search_agent', 'weather_agent', END ]]:
    calculator_agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt="You are a helpful AI assistant. Your task is to perform calculations. Answer in short all needed information",
        checkpointer=checkpointer,
        response_format=State
    )
    response = calculator_agent.invoke(
    {"messages": state["messages"]}
)
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

def weather_agent(state: State) -> Command[Literal['communicate_agent', 'search_agent', 'calculator_agent', END]]:
    weather_agent = create_agent(
        model=model,
        tools=[get_current_weather],
        system_prompt="You are a helpful AI assistant. Your task is to provide current weather information. Answer in short all needed information",
        checkpointer=checkpointer,
        response_format=State
    )
    response = weather_agent.invoke(
    {"messages": state["messages"]}
)
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

builder = StateGraph(State)
builder.add_node('communicate_agent',communicate_agent)
builder.add_node('search_agent',search_agent)
builder.add_node('calculator_agent',calculator_agent)
builder.add_node('weather_agent',weather_agent)
builder.add_edge(START, 'communicate_agent')

graph = builder.compile()

user_query = (
    "What is the current weather in New York City? "
)
messages = graph.invoke({"messages": user_query})

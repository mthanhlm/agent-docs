"""Experimental multi-agent graph with state-based routing."""

import os
from typing import Literal, Annotated
from dotenv import load_dotenv
from pydantic import Field
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tools import perform_search, fetch_weather, execute_math

load_dotenv()

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

model = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("MODEL_NAME", "gpt-4o")
)


class State(TypedDict):
    """Graph state with message history and routing information."""
    messages: Annotated[list, add_messages]
    next_agent: str


def communicate_agent(state: State) -> Command[Literal['search_agent', 'calculator_agent', 'weather_agent']]:
    """Coordinator agent that routes tasks to specialized agents."""
    agent = create_agent(
        model=model,
        system_prompt=(
            "You are a coordinator that delegates tasks to specialized agents: "
            "search_agent, calculator_agent, and weather_agent."
        ),
        checkpointer=checkpointer,
        response_format=State
    )
    response = agent.invoke({"messages": state["messages"]})
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )


def search_agent(state: State) -> Command[Literal['communicate_agent', 'calculator_agent', 'weather_agent']]:
    """Agent specialized in web search operations."""
    agent = create_agent(
        model=model,
        tools=[perform_search],
        system_prompt="You are a search specialist. Use web search to find information.",
        checkpointer=checkpointer,
        response_format=State
    )
    response = agent.invoke({"messages": state["messages"]})
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )


def calculator_agent(state: State) -> Command[Literal['communicate_agent', 'search_agent', 'weather_agent']]:
    """Agent specialized in mathematical calculations."""
    agent = create_agent(
        model=model,
        tools=[execute_math],
        system_prompt="You are a math specialist. Perform calculations as requested.",
        checkpointer=checkpointer,
        response_format=State
    )
    response = agent.invoke({"messages": state["messages"]})
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )


def weather_agent(state: State) -> Command[Literal['communicate_agent', 'search_agent', 'calculator_agent']]:
    """Agent specialized in weather information retrieval."""
    agent = create_agent(
        model=model,
        tools=[fetch_weather],
        system_prompt="You are a weather specialist. Provide current weather information.",
        checkpointer=checkpointer,
        response_format=State
    )
    response = agent.invoke({"messages": state["messages"]})
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )


builder = StateGraph(State)
builder.add_node('communicate_agent', communicate_agent)
builder.add_node('search_agent', search_agent)
builder.add_node('calculator_agent', calculator_agent)
builder.add_node('weather_agent', weather_agent)
builder.add_edge(START, 'communicate_agent')

graph = builder.compile()


if __name__ == "__main__":
    user_query = (
        "Based on the knowledge base, what is the historical average temperature in Da Lat? "
        "Now, use a tool to get the current real-time temperature in Da Lat. "
        "Then, use the calculator to find the difference between the current temperature and the historical average. "
        "Finally, search for one interesting fact about Da Lat's flower festival."
    )
    
    messages = graph.invoke({"messages": user_query})
    for m in messages["messages"]:
        m.pretty_print()

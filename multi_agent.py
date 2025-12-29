from langchain.agents import create_agent
from tools import web_search, get_current_weather, calculator
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
import os
import re
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from typing import Literal, Annotated, Union, Any, cast
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import JsonOutputParser
import json
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig


checkpointer = InMemorySaver()
config: RunnableConfig = {"configurable": {"thread_id": "1"}}
model = ChatOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("GROQ_API_KEY", ""), model=os.getenv("MODEL_NAME", "gpt-4o"))


class ResponseFormat(BaseModel):
    content: str = Field(..., description="The content/answer or reasoning")
    next_agent: Literal["supervisor", "search_agent", "weather_agent", "calculator_agent", "END"] = Field(..., description="The next agent to handle the response")

parser = JsonOutputParser(pydantic_object=ResponseFormat)
format_prompt = parser.get_format_instructions()

def parse_cleaned_json(text: str):
    # Remove <think> blocks
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return parser.parse(cleaned_text)

def supervisor_agent(state: MessagesState) -> Command[Literal["search_agent","weather_agent","calculator_agent", "__end__" ]]:
    agent = create_agent(
        model=model,
        checkpointer=checkpointer,
        system_prompt=(
            "You are a supervisor agent. Your job is to decide which specialized agent should handle the next part of the user request. "
            "Specialized agents: search_agent, weather_agent, calculator_agent. "
            "If the task is complete, return 'END' as the next_agent. "
            f"\n\nCRITICAL: You MUST respond ONLY with a JSON object in this format:\n{format_prompt}"
        )
    )
    response = agent.invoke(
        {"messages": state["messages"]},
        config=config)
    
    last_message = response['messages'][-1]
    response_data = parse_cleaned_json(last_message.content)
    
    goto = response_data['next_agent']
    if goto == "END":
        goto = END
        
    return Command(
        goto=cast(Any, goto),
        update={"messages": response['messages'][len(state['messages']):]}
    )


def search_agent(state: MessagesState) -> Command[Literal["supervisor","weather_agent","calculator_agent", "__end__" ]]:
    agent = create_agent(
        model=model,
        checkpointer=checkpointer,
        system_prompt=(
            "You are a web search agent. Use the tool to perform web searches. "
            "After getting results, summarize them and decide the next agent. "
            f"\n\nCRITICAL: You MUST respond ONLY with a JSON object in this format after your tool use:\n{format_prompt}"
        ),
        tools=[web_search]
    )
    response = agent.invoke(
        {"messages": state["messages"]},config=config
    )
    
    last_message = response['messages'][-1]
    response_data = parse_cleaned_json(last_message.content)
    
    goto = response_data['next_agent']
    if goto == "END":
        goto = END

    return Command(
        goto=cast(Any, goto),
        update={"messages": response['messages'][len(state['messages']):]}
    )


def weather_agent(state: MessagesState) -> Command[Literal["supervisor","search_agent","calculator_agent", "__end__" ]]:
    agent = create_agent(
        model=model,
        checkpointer=checkpointer,
        system_prompt=(
            "You are a weather agent. Use the tool to provide current weather. "
            f"\n\nCRITICAL: You MUST respond ONLY with a JSON object in this format after your tool use:\n{format_prompt}"
        ),
        tools=[get_current_weather]
    )
    response = agent.invoke(
        {"messages": state["messages"]},config=config
    )
    
    last_message = response['messages'][-1]
    response_data = parse_cleaned_json(last_message.content)
    
    goto = response_data['next_agent']
    if goto == "END":
        goto = END

    return Command(
        goto=cast(Any, goto),
        update={"messages": response['messages'][len(state['messages']):]}
    )

def calculator_agent(state: MessagesState) -> Command[Literal["supervisor","search_agent","weather_agent", "__end__" ]]:
    agent = create_agent(
        model=model,
        checkpointer=checkpointer,
        system_prompt=(
            "You are a calculator agent. Use the tool to perform math. "
            f"\n\nCRITICAL: You MUST respond ONLY with a JSON object in this format after your tool use:\n{format_prompt}"
        ),
        tools=[calculator]
    )
    response = agent.invoke(
        {"messages": state["messages"]},config=config
    )
    
    last_message = response['messages'][-1]
    response_data = parse_cleaned_json(last_message.content)
    
    goto = response_data['next_agent']
    if goto == "END":
        goto = END

    return Command(
        goto=cast(Any, goto),
        update={"messages": response['messages'][len(state['messages']):]}
    )


builder = StateGraph(MessagesState)
builder.add_node('supervisor',supervisor_agent)
builder.add_node('search_agent', search_agent)
builder.add_node('weather_agent', weather_agent)
builder.add_node('calculator_agent', calculator_agent)
builder.add_edge(START, 'supervisor')


graph = builder.compile()

if __name__ == "__main__":
    user_query = (
        "I need a comprehensive climate and tourism analysis for Da Lat. "
        "First, search the historical average temperatures for each of the four seasons from the knowledge base. "
        "Second, get the current real-time temperature and wind speed for Da Lat. "
        "Third, for each season, calculate the difference between the historical average and the current temperature. "
        "Fourth, search for three distinct, specific historical facts about the Da Lat Flower Festival's origin. "
        "Fifth, search for current visitor guidelines and ticket prices for the upcoming festival. "
        "Sixth, calculate the total estimated cost for a group of 5 people based on those ticket prices. "
        "Finally, provide a summary comparing the current weather to the best season for visiting, including the festival facts and total cost."
    )
    messages = graph.invoke({"messages": [HumanMessage(content=user_query)]})
    for m in messages['messages']:
        m.pretty_print()

"""Prototype: Supervisor-based multi-agent routing with structured output."""

import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from tools import perform_search, fetch_weather, execute_math

load_dotenv()

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

model = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("MODEL_NAME", "gpt-4o")
)


class State(MessagesState):
    """Extended state with routing information."""
    next_agent: str


class ResponseFormat(BaseModel):
    """Structured output for agent responses."""
    content: str = Field(..., description="The content to pass to the next agent")
    next_agent: Literal["agent_1", "agent_2", "agent_3"] = Field(
        ..., description="The next agent to handle the response"
    )


parser = JsonOutputParser(pydantic_object=State)
format_instructions = parser.get_format_instructions()

supervisor_prompt = SystemMessage(
    content=(
        "You are a supervisor agent that decides which specialized agent to "
        "delegate tasks to based on user queries.\n"
        "Available agents:\n"
        "1. search_agent: Performs web searches\n"
        "2. weather_agent: Provides weather information\n"
        "3. calculator_agent: Performs calculations\n"
        "4. END: Task delegation complete\n\n"
        f"Output format: {format_instructions}\n"
    )
)

response_parser = JsonOutputParser(pydantic_object=ResponseFormat)
response_format_prompt = response_parser.get_format_instructions()

agent = create_agent(
    model=model,
    tools=[perform_search, execute_math, fetch_weather],
    system_prompt=f"You are a routing assistant. Decide which agent should handle the task.\n#Format: {response_format_prompt}\n",
    checkpointer=checkpointer,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the current weather in New York"}]},
    config=config
)

print(response['messages'][-1].content)

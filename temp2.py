from langchain.agents import create_agent
from tools import web_search, get_current_weather, calculator
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from typing import Literal, Annotated
from langgraph.types import Command
from langgraph.graph.message import add_messages

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage
import json

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}
model = ChatOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("GROQ_API_KEY"), model=os.getenv("MODEL_NAME", "gpt-4o"))


class State(MessagesState):
    next_agent: Literal["search_agent", "weather_agent", "calculator_agent", END]


parser = JsonOutputParser(pydantic_object=State)

format_instructions = parser.get_format_instructions()

supervisor_prompt = SystemMessage(
    content=(
        "You are a supervisor agent that decides which specialized agent to "
        "delegate tasks to based on user queries.\n"
        "You have access to the following specialized agents:\n"
        "1. search_agent: Performs web searches to retrieve information.\n"
        "2. weather_agent: Provides current weather information for "
        "specified locations.\n"
        "3. calculator_agent: Performs mathematical calculations.\n"
        "4. END: Indicates the end of the task delegation.\n"
        "When given a user query, analyze the request and determine the most "
        "appropriate agent to handle it.\n"
        f"Format: {format_instructions}\n\n"
    )
)

print(supervisor_prompt.content)

# def supervisor_agent(state: AgentResponse) -> Command[Literal["search_agent","weather_agent","calculator_agent", END ]]:
#     supervisor_agent = create_agent(
#         model=model,
#         checkpointer=checkpointer,
#         system_prompt=supervisor_prompt.content,
#     )
#     response = supervisor_agent.invoke(
#         {"messages": state["messages"]}
#     )
#     response_data = json.loads(response['messages'][1].content)
#     return Command(
#         goto=response_data['next_agent'],
#         update={"messages": response_data["messages"]}
#     )


# search_agent_prompt = SystemMessage(
#     content=(
#         "You are a specialized agent that performs web searches to retrieve "
#         "information based on user queries. Use the web_search tool to find "
#         "relevant information."
#     )
# )


# def search_agent(state: MessagesState) -> Command[Literal['supervisor']]:
#     search_agent = create_agent(
#         model=model,
#         checkpointer=checkpointer,
#         system_prompt=search_agent_prompt.content,
#         tools=[web_search],
#     )
#     response = search_agent.invoke(
#         {"messages": state["messages"]}
#     )
#     response_data = json.loads(response['messages'][1].content)
#     return Command(
#         goto='supervisor',
#         update={"messages": response_data['messages']}
#     )


# weather_agent_prompt = SystemMessage(
#     content=(
#         "You are a specialized agent that provides current weather information "
#         "for specified locations. Use the get_current_weather tool to fetch "
#         "real-time weather data."
#     )
# )


# def weather_agent(state: MessagesState) -> Command[Literal['supervisor']]:
#     weather_agent = create_agent(
#         model=model,
#         checkpointer=checkpointer,
#         system_prompt=weather_agent_prompt.content,
#         tools=[get_current_weather],
#     )
#     response = weather_agent.invoke(
#         {"messages": state["messages"]}
#     )
#     response_data = json.loads(response['messages'][1].content)
#     return Command(
#         goto='supervisor',
#         update={"messages": response_data['messages']}
#     )


# calculator_agent_prompt = SystemMessage(
#     content=(
#         "You are a specialized agent that performs mathematical calculations. "
#         "Use the calculator tool to compute results based on user queries."
#     )
# )


# def calculator_agent(state: MessagesState) -> Command[Literal['supervisor']]:
#     calculator_agent = create_agent(
#         model=model,
#         checkpointer=checkpointer,
#         system_prompt=calculator_agent_prompt.content,
#         tools=[calculator],
#     )
#     response = calculator_agent.invoke(
#         {"messages": state["messages"]}
#     )
#     response_data = json.loads(response['messages'][1].content)
#     return Command(
#         goto='supervisor',
#         update={"messages": response_data['messages']}
#     )


# builder = StateGraph(MessagesState)
# builder.add_node('supervisor',supervisor_agent)
# builder.add_node('search_agent', search_agent)
# builder.add_node('weather_agent', weather_agent)
# builder.add_node('calculator_agent', calculator_agent)
# builder.add_edge(START, 'supervisor')

# graph = builder.compile()

# if __name__ == "__main__":
#     user_query = (
#         "Based on the knowledge base, what is the historical average temperature in Da Lat? "
#         "Now, use a tool to get the current real-time temperature in Da Lat. "
#         "Then, use the calculator to find the difference between the current temperature and the historical average. "
#         "Finally, search for one interesting fact about Da Lat's flower festival."
#     )
#     messages = graph.invoke({"messages": [{"role": "user", "content": user_query}]})
#     print("Final Response:")
#     print(messages['messages'][-1].content)




class ResponseFormat(BaseModel):
    content: str = Field(..., description="The content give to agent")
    next_agemt: Literal["agent_1", "agent_2", "agent_3"] = Field(..., description="The next agent to handle the response")

parser = JsonOutputParser(pydantic_object=ResponseFormat)

format_prompt = parser.get_format_instructions()


agent = create_agent(
    model=model,
    tools=[web_search, calculator, get_current_weather],
    system_prompt=f"You are a helpful AI assistant. You help to decide which agent should handle the task \n #Format:{format_prompt}\n",
    checkpointer=checkpointer,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the current weather in New York "}]},
    config=config)

print(response['messages'][-1].content)
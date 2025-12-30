from tools import perform_search, fetch_weather, execute_math
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
import os
import re
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from typing import Literal, Annotated, Union, Any, cast
from langgraph.types import Command
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent


checkpointer = InMemorySaver()
model = ChatOpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("GROQ_API_KEY", ""), model=os.getenv("MODEL_NAME", "gpt-4o"))


# Agent configuration for easy management and scaling
# Renamed to experts to avoid confusion with tool names
AGENT_ROLES = {
    "research_expert": "searching for historical facts, average temperatures, and general research information",
    "weather_expert": "fetching current real-time weather data, temperature, and wind speed",
    "math_expert": "performing mathematical calculations and comparing numerical values",
}

def supervisor_agent(state: MessagesState) -> Command[Literal["research_expert","weather_expert","math_expert", "__end__" ]]:
    # Dynamically build the prompt from AGENT_ROLES
    agent_descriptions = "\n".join([f"- {name}: Delegate to this expert for {desc}." for name, desc in AGENT_ROLES.items()])
    next_agent_options = list(AGENT_ROLES.keys()) + ["END"]
    
    system_prompt = (
        "You are a supervisor agent (Coordinator). You DO NOT have any tools and CANNOT perform tasks yourself.\n"
        "Your ONLY job is to delegate tasks to the following specialized experts ONE BY ONE:\n\n"
        f"{agent_descriptions}\n\n"
        "CRITICAL RULES:\n"
        "1. NEVER attempt to call a tool or function directly. You do not have access to them.\n"
        "2. NEVER use your internal knowledge to provide facts or perform calculations.\n"
        "3. You MUST respond ONLY with a raw JSON object. No other text, no markdown blocks, no 'think' tags.\n"
        f"4. The JSON format MUST be: {{\"next_agent\": \"one of {next_agent_options}\", \"content\": \"your reasoning or final answer\"}}\n\n"
        "Instructions:\n"
        "- Review the conversation history to see what has been done.\n"
        "- If more information is needed, set 'next_agent' to the appropriate expert.\n"
        "- If all tasks are completed, provide the final summary in 'content' and set 'next_agent' to 'END'."
    )
    
    agent = create_agent(model=model, system_prompt=system_prompt)
    response = agent.invoke({"messages": state["messages"]})
    last_message = response['messages'][-1]
    
    try:
        # Robust JSON parsing
        cleaned_text = re.sub(r'<think>.*?</think>', '', last_message.content, flags=re.DOTALL).strip()
        json_match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
        response_data = json.loads(json_match.group(1)) if json_match else json.loads(cleaned_text)
        goto = response_data.get('next_agent', 'END')
    except Exception:
        goto = "END"

    if goto == "END":
        goto = END
    return Command(goto=cast(Any, goto), update={"messages": response['messages'][len(state['messages']):]})

def research_expert(state: MessagesState) -> Command[Literal["supervisor"]]:
    agent = create_agent(
        model=model,
        system_prompt="You are a research expert. Use the 'perform_search' tool to find accurate information as requested by the supervisor.",
        tools=[perform_search]
    )
    response = agent.invoke({"messages": state["messages"]})
    return Command(goto="supervisor", update={"messages": response['messages'][len(state['messages']):]})

def weather_expert(state: MessagesState) -> Command[Literal["supervisor"]]:
    agent = create_agent(
        model=model,
        system_prompt="You are a weather expert. Use the 'fetch_weather' tool to retrieve real-time data for locations requested by the supervisor.",
        tools=[fetch_weather]
    )
    response = agent.invoke({"messages": state["messages"]})
    return Command(goto="supervisor", update={"messages": response['messages'][len(state['messages']):]})

def math_expert(state: MessagesState) -> Command[Literal["supervisor"]]:
    agent = create_agent(
        model=model,
        system_prompt="You are a math expert. Use the 'execute_math' tool to perform precise mathematical operations as requested by the supervisor.",
        tools=[execute_math]
    )
    response = agent.invoke({"messages": state["messages"]})
    return Command(goto="supervisor", update={"messages": response['messages'][len(state['messages']):]})



builder = StateGraph(MessagesState)
builder.add_node('supervisor',supervisor_agent)
builder.add_node('research_expert', research_expert)
builder.add_node('weather_expert', weather_expert)
builder.add_node('math_expert', math_expert)
builder.add_edge(START, 'supervisor')


graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    user_query = (
        "Process the following request step-by-step:\n"
        "1. Fetch the current weather in Da Lat.\n"
        "2. Find the historical average temperature in Da Lat for December.\n"
        "3. Calculate the difference between the current temperature and that historical average."
    )
    
    main_config = {"configurable": {"thread_id": "main_conversation"}}
    
    print("Starting graph execution...")
    try:
        for chunk in graph.stream({"messages": [HumanMessage(content=user_query)]}, config=main_config, stream_mode="values"):
            if "messages" in chunk:
                last_message = chunk["messages"][-1]
                if not isinstance(last_message, HumanMessage):
                    last_message.pretty_print()
    except Exception as e:
        print(f"Error during execution: {e}")

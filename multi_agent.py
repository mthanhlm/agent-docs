"""Multi-agent system with supervisor coordination pattern."""

import os
from typing import Literal, Any, cast
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tools import perform_search, fetch_weather, execute_math

load_dotenv()

checkpointer = InMemorySaver()
model = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY", ""),
    model=os.getenv("MODEL_NAME", "gpt-4o")
)

AGENT_ROLES = {
    "research_expert": "searching for historical facts, average temperatures, and general research information",
    "weather_expert": "fetching current real-time weather data, temperature, and wind speed",
    "math_expert": "performing mathematical calculations and comparing numerical values",
}


class SupervisorDecision(BaseModel):
    """Structured output for supervisor routing decisions."""
    next_agent: Literal["research_expert", "weather_expert", "math_expert", "END"] = Field(
        description="The next agent to delegate to, or 'END' if all tasks are completed"
    )
    content: str = Field(
        description="Reasoning for the decision or the final answer summary"
    )


def supervisor_agent(state: MessagesState) -> Command[Literal["research_expert", "weather_expert", "math_expert", "__end__"]]:
    """Route tasks to specialized agents based on the current conversation state."""
    agent_descriptions = "\n".join([
        f"- {name}: Delegate to this expert for {desc}." 
        for name, desc in AGENT_ROLES.items()
    ])
    
    system_prompt = (
        "You are a supervisor agent (Coordinator). You DO NOT have any tools and CANNOT perform tasks yourself.\n"
        "Your ONLY job is to delegate tasks to the following specialized experts ONE BY ONE:\n\n"
        f"{agent_descriptions}\n\n"
        "CRITICAL RULES:\n"
        "1. NEVER attempt to call a tool or function directly. You do not have access to them.\n"
        "2. NEVER use your internal knowledge to provide facts or perform calculations.\n"
        "3. Always delegate to the appropriate expert based on the task.\n\n"
        "Instructions:\n"
        "- Review the conversation history to see what has been done.\n"
        "- If more information is needed, set 'next_agent' to the appropriate expert.\n"
        "- If all tasks are completed, provide the final summary in 'content' and set 'next_agent' to 'END'."
    )
    
    agent = create_agent(
        model=model,
        system_prompt=system_prompt,
        response_format=SupervisorDecision
    )
    
    response = agent.invoke({"messages": state["messages"]})
    structured_response: SupervisorDecision = response.get('structured_response')
    
    goto = structured_response.next_agent if structured_response else "END"
    if goto == "END":
        goto = END
        
    return Command(goto=cast(Any, goto), update={"messages": response['messages'][len(state['messages']):]})


def research_expert(state: MessagesState) -> Command[Literal["supervisor"]]:
    """Handle research and web search tasks."""
    agent = create_agent(
        model=model,
        system_prompt="You are a research expert. Use the 'perform_search' tool to find accurate information as requested.",
        tools=[perform_search]
    )
    response = agent.invoke({"messages": state["messages"]})
    return Command(goto="supervisor", update={"messages": response['messages'][len(state['messages']):]})


def weather_expert(state: MessagesState) -> Command[Literal["supervisor"]]:
    """Handle real-time weather data retrieval."""
    agent = create_agent(
        model=model,
        system_prompt="You are a weather expert. Use the 'fetch_weather' tool to retrieve real-time data for locations.",
        tools=[fetch_weather]
    )
    response = agent.invoke({"messages": state["messages"]})
    return Command(goto="supervisor", update={"messages": response['messages'][len(state['messages']):]})


def math_expert(state: MessagesState) -> Command[Literal["supervisor"]]:
    """Handle mathematical calculations."""
    agent = create_agent(
        model=model,
        system_prompt="You are a math expert. Use the 'execute_math' tool to perform precise mathematical operations.",
        tools=[execute_math]
    )
    response = agent.invoke({"messages": state["messages"]})
    return Command(goto="supervisor", update={"messages": response['messages'][len(state['messages']):]})


builder = StateGraph(MessagesState)
builder.add_node('supervisor', supervisor_agent)
builder.add_node('research_expert', research_expert)
builder.add_node('weather_expert', weather_expert)
builder.add_node('math_expert', math_expert)
builder.add_edge(START, 'supervisor')

graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    user_query = (
        "Process the following request step-by-step:\n"
        "1. Use web search to find Da Lat's historical December average high and low temperatures (°C) "
        "from a credible source, and extract the two numbers.\n"
        "2. Fetch the current temperature in Da Lat (°C).\n"
        "3. Compute:\n"
        "   historical_mean = (avg_high + avg_low) / 2\n"
        "   anomaly = current_temp - historical_mean\n"
        "4. Return anomaly rounded to 1 decimal place, and include the source you used for the historical averages."
    )
    
    main_config = {"configurable": {"thread_id": "main_conversation"}}
    
    print("Starting graph execution...")
    try:
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=user_query)]},
            config=main_config,
            stream_mode="values"
        ):
            if "messages" in chunk:
                last_message = chunk["messages"][-1]
                if not isinstance(last_message, HumanMessage):
                    last_message.pretty_print()
    except Exception as e:
        print(f"Error during execution: {e}")

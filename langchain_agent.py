"""LangChain-based agent with tool support for search, weather, and calculations."""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
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

agent = create_agent(
    model=model,
    tools=[perform_search, execute_math, fetch_weather],
    system_prompt="You are a helpful AI assistant. Use available tools to answer precisely and concisely.",
    checkpointer=checkpointer,
)


if __name__ == "__main__":
    user_query = (
        "Based on the knowledge base, what is the historical average temperature in Da Lat? "
        "Now, use a tool to get the current real-time temperature in Da Lat. "
        "Then, use the calculator to find the difference between the current temperature and the historical average. "
        "Finally, search for one interesting fact about Da Lat's flower festival."
    )

    agent_response = None
    debug = False
    
    for chunk in agent.stream(  
        {"messages": [{"role": "user", "content": user_query}]},
        stream_mode="updates",
        config=config,
    ):
        for step, data in chunk.items():
            if debug:
                print(f"step: {step}\n")
                print(f"content: {data['messages'][-1].content_blocks}\n")
            agent_response = data['messages'][-1].content_blocks

    print(f"\nFinal Answer:\n\n {agent_response[0]['text']}")

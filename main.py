from agent import Agent
from tools import web_search, get_current_weather, calculator
from memory import ShortTermMemory
from rag import RAGSystem
import logging

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    memory = ShortTermMemory(max_messages=10)
    
    # Initialize RAG with some example data
    kb = [
        "The project 'agent-docs' is a documentation and implementation of an AI agent system.",
        "The agent supports web search, weather queries, and calculations.",
        "RAG stands for Retrieval-Augmented Generation.",
        "This system uses Groq for fast LLM inference."
    ]
    rag = RAGSystem(documents=kb, threshold=0.2)

    agent = Agent(
        tools=[web_search, get_current_weather, calculator], 
        system_prompt="You are a helpful AI assistant. Always respond in English. Use tools to verify data.",
        debug=True,
        memory=memory,
        rag=rag
    )

    print("--- Interactive Chat Mode (Type 'exit' or 'quit' to stop) ---")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            response = agent.chat(user_input)
            print(f"\nAI: {response}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

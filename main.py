from agent import Agent
from tools import web_search, get_current_weather, calculator
from memory import ShortTermMemory

def main():
    memory = ShortTermMemory(max_messages=10)

    agent = Agent(
        tools=[web_search, get_current_weather, calculator], 
        system_prompt="You are a helpful AI assistant. Always respond in English. Use tools to verify data.",
        debug=True,
        memory=memory
    )

    print("--- Interactive Chat Mode (Type 'exit' or 'quit' to stop) ---")
    
    # Old test query for reference:
    """
    user_query = (
        "1. What is the current temperature in Da Lat right now? "
        "2. Search for the average annual temperature in Da Lat. "
        "3. Calculate the difference between the current temperature and that average annual temperature."
        "4. Provide the final answer in English and short all results of each step."
    )
    response = agent.chat(user_query)
    print(f"\nFinal Answer:\n\n {response}")
    """

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

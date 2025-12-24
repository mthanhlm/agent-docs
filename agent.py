import inspect
import json
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, tools=None, system_prompt="You are a helpful assistant.", debug=False, memory=None):
        self.client = OpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = os.getenv("MODEL_NAME", "gpt-4o")
        self.tools_map = {tool.__name__: tool for tool in (tools or [])}
        self.memory = memory
        self.system_prompt = system_prompt
        self.tool_schemas = self.get_tool_schemas()
        self.debug = debug

    def get_tool_schemas(self):
        if not self.tools_map: 
            return None
        schemas = []
        for name, function_object in self.tools_map.items():
            signature = inspect.signature(function_object)
            docstring = inspect.getdoc(function_object) or "No description available."
            properties = {
                parameter_name: {
                    "type": {int: "integer", float: "number", bool: "boolean", list: "array", dict: "object", str: "string"}.get(parameter.annotation, "string"),
                    "description": f"The {parameter_name} parameter."
                } for parameter_name, parameter in signature.parameters.items()
            }
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": docstring,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": [parameter_name for parameter_name, parameter in signature.parameters.items() if parameter.default == inspect.Parameter.empty]
                    }
                }
            })
        return schemas

    def chat(self, user_input=None):
        # Initialize context for this turn
        if self.memory:
            messages = list(self.memory.get_messages())
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": self.system_prompt})
        else:
            messages = [{"role": "system", "content": self.system_prompt}]
        
        if user_input:
            user_message = {"role": "user", "content": str(user_input)}
            messages.append(user_message)
            if self.memory:
                self.memory.add_message(user_message)

        while True:

            api_arguments = {
                "model": self.model, 
                "messages": messages, 
                "tools": self.tool_schemas, 
                "tool_choice": "auto" if self.tool_schemas else None
            }
            api_arguments = {key: value for key, value in api_arguments.items() if value is not None}

            # [MODEL] Decide next step
            response = self.client.chat.completions.create(**api_arguments) # type: ignore
            model_message = response.choices[0].message
            
            assistant_message = {"role": "assistant", "content": model_message.content or ""}
            if model_message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.id, 
                        "type": "function", 
                        "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
                    } for tool_call in model_message.tool_calls
                ]
            
            # Append to context
            messages.append(assistant_message)
            if self.memory:
                self.memory.add_message(assistant_message)

            if self.debug:
                if model_message.content:
                    logger.info(f"Thought: {model_message.content}")
                if model_message.tool_calls:
                    for tool_call in model_message.tool_calls:
                        logger.info(f"Action: {tool_call.function.name}({tool_call.function.arguments})")

            # If no tool calls, model has decided it's finished
            if not model_message.tool_calls:
                if self.debug:
                    logger.info(f"Final Answer: {model_message.content}")
                return model_message.content

            # [TOOL] Execute tool(s)
            for tool_call in model_message.tool_calls:
                selected_function = self.tools_map.get(tool_call.function.name)
                if not selected_function:
                    result = f"Error: Tool '{tool_call.function.name}' not found."
                else:
                    arguments = json.loads(tool_call.function.arguments)
                    result = selected_function(**arguments)
                
                if self.debug:
                    logger.info(f"Perception: {result}")

                tool_message = {
                    "role": "tool", 
                    "tool_call_id": tool_call.id, 
                    "name": tool_call.function.name, 
                    "content": str(result)
                }
                
                # [RETURN RESULT TO MODEL] 
                messages.append(tool_message)
                if self.memory:
                    self.memory.add_message(tool_message)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    from tools import web_search, get_current_weather, calculator
    from memory import ShortTermMemory

    memory = ShortTermMemory(max_messages=10)

    agent = Agent(
        tools=[web_search, get_current_weather, calculator], 
        system_prompt="You are a helpful AI assistant. Always respond in English. Use tools to verify data.",
        debug=True,
        memory=memory
    )

    user_query = (
        "1. What is the current temperature in Da Lat right now? "
        "2. Search for the average annual temperature in Da Lat. "
        "3. Calculate the difference between the current temperature and that average annual temperature."
        "4. Provide the final answer in English and short all results of each step."
    )
    response = agent.chat(user_query)
    print(f"\nFinal Answer:\n\n {response}")    


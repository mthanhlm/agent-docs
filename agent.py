import inspect
import json
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from rag import RAGSystem

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, tools=None, system_prompt="You are a helpful assistant.", debug=False, memory=None, rag: RAGSystem = None):
        self.client = OpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = os.getenv("MODEL_NAME", "gpt-4o")
        self.tools_map = {tool.__name__: tool for tool in (tools or [])}
        self.memory = memory
        self.rag = rag
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
            content = str(user_input)
            if self.rag:
                rag_context = self.rag.query(content)
                if rag_context:
                    if self.debug:
                        logger.info(f"--- RAG CONTEXT RETRIEVED ---\n{rag_context}\n---------------------------")
                    content = f"Context from knowledge base:\n{rag_context}\n\nUser query: {user_input}"
            
            user_message = {"role": "user", "content": content}
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
            
            # If there are tool calls, handle only the first one
            if model_message.tool_calls:
                tool_call = model_message.tool_calls[0]
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.id, 
                        "type": "function", 
                        "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
                    }
                ]
                
                # Append assistant message with tool_calls to context
                messages.append(assistant_message)
                if self.memory:
                    self.memory.add_message(assistant_message)

                if self.debug:
                    if model_message.content:
                        logger.info(f"Thought: {model_message.content}")
                    logger.info(f"Action: {tool_call.function.name}({tool_call.function.arguments})")

                # [TOOL] Execute the selected tool
                selected_function = self.tools_map.get(tool_call.function.name)
                if not selected_function:
                    result = f"Error: Tool '{tool_call.function.name}' not found."
                else:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        result = selected_function(**arguments)
                    except Exception as e:
                        result = f"Error executing tool: {str(e)}"
                
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
                
                # Continue the while loop to let the model process the tool result
                continue

            # If no tool calls, model has decided it's finished
            messages.append(assistant_message)
            if self.memory:
                self.memory.add_message(assistant_message)
                
            if self.debug:
                logger.info(f"Final Answer: {model_message.content}")
            
            return model_message.content


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    from tools import web_search, get_current_weather, calculator
    from memory import ShortTermMemory
    from rag import RAGSystem

    memory = ShortTermMemory(max_messages=10)
    
    # Khởi tạo RAG với top_k=2 và XÓA tri thức cũ
    rag = RAGSystem(threshold=0.5, top_k=1, clear_history=True)
    
    # Nạp kiến thức cố định vào Knowledge Base
    kb_data = [
        "Historical data shows that the average annual temperature in Da Lat is exactly 18.0 degrees Celsius.",
        "The highest recorded temperature in Da Lat's history was 31.5 degrees Celsius.",
        "Da Lat is famous for Arabica coffee, which requires temperatures between 15-24 degrees Celsius to grow well."
    ]
    rag.add_knowledge(kb_data)

    agent = Agent(
        tools=[web_search, get_current_weather, calculator], 
        system_prompt="You are a helpful AI assistant. Always use context from Knowledge Base (RAG) and available tools to answer precisely. Answer in short all needed information",
        debug=True,
        memory=memory,
        rag=rag
    )

    # Câu hỏi yêu cầu dùng: RAG + Weather Tool + Calculator + Search Tool
    user_query = (
        "Based on the knowledge base, what is the historical average temperature in Da Lat? "
        "Now, use a tool to get the current real-time temperature in Da Lat. "
        "Then, use the calculator to find the difference between the current temperature and the historical average. "
        "Finally, search for one interesting fact about Da Lat's flower festival."
    )
    
    response = agent.chat(user_query)
    print(f"\nFinal Answer:\n\n {response}")    


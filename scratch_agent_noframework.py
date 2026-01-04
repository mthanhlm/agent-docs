"""Lightweight AI agent implementation without external frameworks."""

import inspect
import json
import logging
import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from rag import RAGSystem

load_dotenv()

logger = logging.getLogger(__name__)


class Agent:
    """AI agent with tool execution, memory, and RAG capabilities."""
    
    def __init__(
        self,
        tools=None,
        system_prompt: str = "You are a helpful assistant.",
        debug: bool = False,
        memory=None,
        rag: Optional[RAGSystem] = None
    ):
        self.client = OpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = os.getenv("MODEL_NAME", "gpt-4o")
        self.tools_map = {tool.__name__: tool for tool in (tools or [])}
        self.memory = memory
        self.rag = rag
        self.system_prompt = system_prompt
        self.tool_schemas = self._generate_tool_schemas()
        self.debug = debug

    def _generate_tool_schemas(self) -> Optional[list]:
        """Generate OpenAI-compatible tool schemas from function signatures."""
        if not self.tools_map: 
            return None
            
        schemas = []
        type_mapping = {
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            str: "string"
        }
        
        for name, func in self.tools_map.items():
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or "No description available."
            
            properties = {
                param_name: {
                    "type": type_mapping.get(param.annotation, "string"),
                    "description": f"The {param_name} parameter."
                } 
                for param_name, param in signature.parameters.items()
            }
            
            required = [
                param_name 
                for param_name, param in signature.parameters.items() 
                if param.default == inspect.Parameter.empty
            ]
            
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": docstring,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        return schemas

    def chat(self, user_input: Optional[str] = None) -> str:
        """Process user input through the agent loop with tool execution."""
        messages = self._initialize_messages()
        
        if user_input:
            content = self._prepare_content(user_input)
            user_message = {"role": "user", "content": content}
            messages.append(user_message)
            if self.memory:
                self.memory.add_message(user_message)

        while True:
            response = self._call_model(messages)
            model_message = response.choices[0].message
            assistant_message = {"role": "assistant", "content": model_message.content or ""}
            
            if model_message.tool_calls:
                self._handle_tool_call(model_message, assistant_message, messages)
                continue

            messages.append(assistant_message)
            if self.memory:
                self.memory.add_message(assistant_message)
                
            if self.debug:
                logger.info(f"Final Answer: {model_message.content}")
            
            return model_message.content

    def _initialize_messages(self) -> list:
        """Set up the initial message context."""
        if self.memory:
            messages = list(self.memory.get_messages())
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": self.system_prompt})
        else:
            messages = [{"role": "system", "content": self.system_prompt}]
        return messages

    def _prepare_content(self, user_input: str) -> str:
        """Augment user input with RAG context if available."""
        content = str(user_input)
        if self.rag:
            rag_context = self.rag.query(content)
            if rag_context:
                if self.debug:
                    logger.info(f"--- RAG CONTEXT RETRIEVED ---\n{rag_context}\n---------------------------")
                content = f"Context from knowledge base:\n{rag_context}\n\nUser query: {user_input}"
        return content

    def _call_model(self, messages: list):
        """Make API call to the language model."""
        api_args = {
            "model": self.model,
            "messages": messages,
            "tools": self.tool_schemas,
            "tool_choice": "auto" if self.tool_schemas else None
        }
        api_args = {k: v for k, v in api_args.items() if v is not None}
        return self.client.chat.completions.create(**api_args)

    def _handle_tool_call(self, model_message, assistant_message: dict, messages: list):
        """Execute tool call and update message history."""
        tool_call = model_message.tool_calls[0]
        assistant_message["tool_calls"] = [{
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            }
        }]
        
        messages.append(assistant_message)
        if self.memory:
            self.memory.add_message(assistant_message)

        if self.debug:
            if model_message.content:
                logger.info(f"Thought: {model_message.content}")
            logger.info(f"Action: {tool_call.function.name}({tool_call.function.arguments})")

        result = self._execute_tool(tool_call)
        
        if self.debug:
            logger.info(f"Observation: {result}")

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": str(result)
        }
        
        messages.append(tool_message)
        if self.memory:
            self.memory.add_message(tool_message)

    def _execute_tool(self, tool_call) -> str:
        """Execute a tool and return the result."""
        func = self.tools_map.get(tool_call.function.name)
        if not func:
            return f"Error: Tool '{tool_call.function.name}' not found."
        
        try:
            arguments = json.loads(tool_call.function.arguments)
            return func(**arguments)
        except Exception as e:
            return f"Error executing tool: {str(e)}"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    from tools import perform_search, fetch_weather, execute_math
    from memory import ShortTermMemory

    knowledge_base = [
        "Historical data shows that the average annual temperature in Da Lat is exactly 18.0 degrees Celsius.",
        "The highest recorded temperature in Da Lat's history was 31.5 degrees Celsius.",
        "Da Lat is famous for Arabica coffee, which requires temperatures between 15-24 degrees Celsius to grow well."
    ]

    memory = ShortTermMemory(max_messages=10)
    rag = RAGSystem(threshold=0.5, top_k=1, clear_history=True)
    rag.add_knowledge(knowledge_base)

    agent = Agent(
        tools=[perform_search, fetch_weather, execute_math],
        system_prompt="You are a helpful AI assistant. Use context from Knowledge Base (RAG) and tools to answer precisely.",
        debug=True,
        memory=memory,
        rag=rag
    )

    user_query = (
        "Based on the knowledge base, what is the historical average temperature in Da Lat? "
        "Now, use a tool to get the current real-time temperature in Da Lat. "
        "Then, use the calculator to find the difference between the current temperature and the historical average. "
        "Finally, search for one interesting fact about Da Lat's flower festival."
    )
    
    response = agent.chat(user_query)
    print(f"\nFinal Answer:\n\n {response}")

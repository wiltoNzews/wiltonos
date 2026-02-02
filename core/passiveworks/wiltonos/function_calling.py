"""
WiltonOS Function Calling Integration
Implements structured OpenAI function calling for improved inference and agent capabilities
"""

import json
import logging
import datetime
import inspect
import os
from typing import Dict, Any, List, Callable, Optional, Union, TypeVar, get_type_hints, get_origin, get_args
from pydantic import BaseModel, create_model, Field
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [FUNCTION_CALLING] %(message)s",
    handlers=[
        logging.FileHandler("logs/function_calling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("function_calling")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

T = TypeVar('T')

class FunctionRegistry:
    """Registry for functions that can be called by the LLM"""
    
    def __init__(self):
        self.functions = {}
        logger.info("Function Registry initialized")
    
    def register(self, func: Callable) -> Callable:
        """Register a function with the registry"""
        function_name = func.__name__
        
        if function_name in self.functions:
            logger.warning(f"Function {function_name} already registered, overwriting")
        
        # Extract function metadata
        function_metadata = self._extract_function_metadata(func)
        
        self.functions[function_name] = {
            "function": func,
            "metadata": function_metadata
        }
        
        logger.info(f"Registered function: {function_name}")
        return func
    
    def _extract_function_metadata(self, func: Callable) -> Dict[str, Any]:
        """Extract metadata from a function for OpenAI function calling"""
        # Get the function signature
        sig = inspect.signature(func)
        
        # Get type hints for parameters
        type_hints = get_type_hints(func)
        
        # Extract docstring
        docstring = inspect.getdoc(func) or "No description provided"
        
        # Build parameters schema
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            # Skip self for instance methods
            if param_name == "self":
                continue
            
            param_type = type_hints.get(param_name, type(None))
            param_schema = self._type_to_json_schema(param_type)
            
            # Get param description from docstring if available
            param_desc = self._extract_param_description(docstring, param_name)
            if param_desc:
                param_schema["description"] = param_desc
            
            parameters["properties"][param_name] = param_schema
            
            # Mark as required if no default value
            if param.default is inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        # Build function metadata
        function_metadata = {
            "name": func.__name__,
            "description": self._extract_function_description(docstring),
            "parameters": parameters
        }
        
        return function_metadata
    
    def _extract_function_description(self, docstring: str) -> str:
        """Extract the function description from a docstring"""
        if not docstring:
            return "No description provided"
        
        # Take the first paragraph of the docstring
        lines = docstring.strip().split("\n\n")
        return lines[0].strip()
    
    def _extract_param_description(self, docstring: str, param_name: str) -> Optional[str]:
        """Extract parameter description from a docstring"""
        if not docstring:
            return None
        
        # Look for param in docstring
        lines = docstring.split("\n")
        for i, line in enumerate(lines):
            if f":{param_name}:" in line or f"@param {param_name}" in line or f"Args:\n    {param_name}" in line:
                # Get the description
                desc_start = line.find(":", line.find(param_name)) + 1
                desc = line[desc_start:].strip()
                
                # Check if description continues on next lines
                j = i + 1
                while j < len(lines) and lines[j].strip() and not (lines[j].strip().startswith(":") or lines[j].strip().startswith("@")):
                    desc += " " + lines[j].strip()
                    j += 1
                
                return desc
        
        return None
    
    def _type_to_json_schema(self, typ):
        """Convert a Python type to a JSON schema"""
        origin = get_origin(typ)
        args = get_args(typ)
        
        # Handle Union types (Optional is Union[T, None])
        if origin is Union:
            if type(None) in args:  # Optional
                return self._type_to_json_schema(next(a for a in args if a is not type(None)))
            else:
                # This is a simplified approach; in a real implementation,
                # you might want to handle multiple types differently
                return {"type": "object", "description": f"One of {args}"}
        
        # Handle List types
        if origin is list:
            item_type = args[0] if args else Any
            return {
                "type": "array",
                "items": self._type_to_json_schema(item_type)
            }
        
        # Handle Dict types
        if origin is dict:
            return {"type": "object"}
        
        # Handle basic types
        if typ is str:
            return {"type": "string"}
        elif typ is int:
            return {"type": "integer"}
        elif typ is float:
            return {"type": "number"}
        elif typ is bool:
            return {"type": "boolean"}
        elif issubclass(typ, BaseModel):
            # Handle Pydantic models
            schema = typ.schema()
            return {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        else:
            # Default to object for complex types
            return {"type": "object"}
    
    def get_openai_schema(self) -> List[Dict[str, Any]]:
        """Get the OpenAI function calling schema for all registered functions"""
        return [func_data["metadata"] for _, func_data in self.functions.items()]
    
    def execute(self, function_name: str, **kwargs) -> Any:
        """Execute a registered function with the provided arguments"""
        if function_name not in self.functions:
            raise ValueError(f"Function {function_name} not registered")
        
        func = self.functions[function_name]["function"]
        
        try:
            logger.info(f"Executing function: {function_name}")
            return func(**kwargs)
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {str(e)}")
            raise

class FunctionCaller:
    """Handles calling functions through the OpenAI API"""
    
    def __init__(self, registry: FunctionRegistry, model: str = "gpt-4o"):
        self.registry = registry
        self.model = model
        logger.info(f"Function Caller initialized with model: {model}")
    
    async def process_with_functions(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Process a conversation with function calling"""
        if functions is None:
            functions = self.registry.get_openai_schema()
        
        try:
            # Create the API request
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=functions,
                function_call="auto",  # Let the model decide when to call functions
                temperature=temperature
            )
            
            message = response.choices[0].message
            
            # Check if the model wants to call a function
            if hasattr(message, 'function_call') and message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                logger.info(f"Model requested function call: {function_name}")
                
                # Execute the function
                try:
                    function_response = self.registry.execute(function_name, **function_args)
                    
                    # Convert the response to a string if it's not already
                    if not isinstance(function_response, str):
                        if hasattr(function_response, "model_dump_json"):  # Pydantic v2
                            function_response = function_response.model_dump_json()
                        elif hasattr(function_response, "json"):  # Pydantic v1
                            function_response = function_response.json()
                        else:
                            function_response = json.dumps(function_response)
                    
                    # Append the function response to messages
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": message.function_call.arguments
                        }
                    })
                    
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": function_response
                    })
                    
                    # Get the final response from the model
                    second_response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature
                    )
                    
                    return {
                        "response": second_response.choices[0].message.content,
                        "function_call": {
                            "name": function_name,
                            "arguments": function_args,
                            "result": function_response
                        }
                    }
                
                except Exception as e:
                    logger.error(f"Error executing function {function_name}: {str(e)}")
                    return {
                        "response": f"Error executing function {function_name}: {str(e)}",
                        "error": str(e)
                    }
            
            # If no function call, just return the response
            return {
                "response": message.content,
                "function_call": None
            }
        
        except Exception as e:
            logger.error(f"Error in function calling: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "error": str(e)
            }

# Create a singleton registry
function_registry = FunctionRegistry()

# Decorator for registering functions
def openai_function(func):
    """Decorator to register a function with the OpenAI function registry"""
    return function_registry.register(func)

# Create a singleton function caller
function_caller = FunctionCaller(function_registry)

# Example function registration
@openai_function
def get_current_time() -> str:
    """
    Get the current date and time
    
    Returns:
        The current date and time in ISO format
    """
    return datetime.datetime.now().isoformat()

@openai_function
def calculate_cognitive_resonance(
    memory_waves: float,
    emotional_viscosity: float,
    perturbation: float
) -> Dict[str, Any]:
    """
    Calculate the cognitive resonance score based on quantum metrics
    
    Args:
        memory_waves: The memory wave amplitude (0-10)
        emotional_viscosity: The emotional viscosity coefficient (0-10)
        perturbation: The perturbation factor (0-10)
    
    Returns:
        A dictionary with the cognitive resonance score and components
    """
    # Simple formula for demonstration
    cr_score = (memory_waves * 0.4 + emotional_viscosity * 0.3 + perturbation * 0.3) / 10
    
    return {
        "cognitive_resonance": cr_score,
        "components": {
            "memory_waves": memory_waves,
            "emotional_viscosity": emotional_viscosity,
            "perturbation": perturbation
        },
        "timestamp": datetime.datetime.now().isoformat()
    }
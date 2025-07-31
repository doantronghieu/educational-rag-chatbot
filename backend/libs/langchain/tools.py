"""LangChain tools integration for function calling and structured outputs."""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.language_models.chat_models import BaseChatModel
from utils.decorators import handle_errors

class ToolManager:
    """Unified tool management class for all tool-related operations."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def create_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        **kwargs
    ) -> BaseTool:
        """
        Create a tool from a function.
        
        Args:
            func: Function to convert to tool
            name: Tool name (defaults to function name)
            description: Tool description (from docstring if not provided)
            return_direct: Whether to return tool output directly
            **kwargs: Additional tool parameters
        
        Returns:
            LangChain tool instance
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool for {tool_name}"
        
        # Use @tool decorator for automatic schema generation
        if hasattr(func, '__annotations__'):
            langchain_tool = tool(return_direct=return_direct)(func)
            langchain_tool.name = tool_name
            langchain_tool.description = tool_description
        else:
            # Fallback for functions without type hints
            langchain_tool = StructuredTool.from_function(
                func=func,
                name=tool_name,
                description=tool_description,
                return_direct=return_direct,
                **kwargs
            )
        
        self._tools[tool_name] = langchain_tool
        return langchain_tool
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register an existing tool."""
        self._tools[tool.name] = tool
    
    def get_tools(self, names: Optional[Union[str, List[str]]] = None) -> Union[BaseTool, List[BaseTool], None]:
        """
        Get tools by name(s) or all tools.
        
        Args:
            names: Tool name, list of names, or None for all tools
        
        Returns:
            Single tool, list of tools, or None if not found
        """
        if names is None:
            return list(self._tools.values())
        elif isinstance(names, str):
            return self._tools.get(names)
        else:
            return [self._tools.get(name) for name in names if name in self._tools]
    
    def _resolve_tools(self, tools: Optional[List[Union[str, BaseTool]]] = None) -> List[BaseTool]:
        """
        Helper method to resolve tools from names/objects to BaseTool list.
        
        Args:
            tools: List of tool names or tool instances (all tools if None)
        
        Returns:
            List of resolved BaseTool instances
        """
        if tools is None:
            return self.get_tools()
        
        tool_list = []
        for tool in tools:
            if isinstance(tool, str):
                tool_obj = self.get_tools(tool)
                if tool_obj:
                    tool_list.append(tool_obj)
            else:
                tool_list.append(tool)
        
        return tool_list
    
    @handle_errors("bind tools to model", return_on_error=None, raise_on_error=True)
    def bind_tools_to_model(
        self,
        model: Optional[Union[BaseChatModel]] = None,
        tools: Optional[List[Union[str, BaseTool]]] = None,
        **model_kwargs
    ) -> BaseChatModel:
        """
        Bind tools to a chat model or create new model with tools.
        
        Args:
            model: Chat model to bind tools to (creates new ChatModel if None)
            tools: List of tool names or tool instances (all tools if None)
            **model_kwargs: Model parameters if creating new model
        
        Returns:
            Model with tools bound
        """
        if model is None:
            from core.dependencies import get_llm
            model = get_llm()
        
        # Get the actual LangChain model if it's a ChatModel wrapper
        actual_model = getattr(model, 'model', model)
        
        tool_list = self._resolve_tools(tools)
        return actual_model.bind_tools(tool_list)
    
    
    # Few-shot prompting functionality
    def create_few_shot_prompt(
        self,
        examples: List[Dict[str, Any]],
        input_variables: List[str] = None
    ) -> ChatPromptTemplate:
        """Create a few-shot prompt template with tool usage examples."""
        from libs.langchain.messages import create_few_shot_prompt
        return create_few_shot_prompt(examples, input_variables)
    
    async def execute_tool_chain(
        self,
        input_text: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Union[str, BaseTool]]] = None,
        **kwargs
    ) -> Union[List[BaseMessage], List[ToolMessage]]:
        """
        Execute tool operations - either complete chain or just tool calls.
        
        Args:
            input_text: User input for complete chain (if provided, runs full chain)
            tool_calls: Direct tool calls to execute (if provided, runs only tool execution)
            tools: Tools to make available
            **kwargs: Additional model parameters
        
        Returns:
            List of messages from interaction (if input_text) or tool results (if tool_calls)
        """
        # Mode 1: Execute only tool calls (low-level operation)
        if tool_calls is not None and input_text is None:
            # Resolve tools for execution
            tool_list = self._resolve_tools(tools)
            
            # Create tool lookup for efficiency
            tool_lookup = {tool.name: tool for tool in tool_list}
            
            results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", "")
                
                tool = tool_lookup.get(tool_name)
                if tool:
                    try:
                        # async tool execution with fallback handling
                        result = await self._execute_tool(tool, tool_args, **kwargs)
                        results.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_id
                        ))
                    except Exception as e:
                        results.append(ToolMessage(
                            content=f"Error executing {tool_name}: {str(e)}",
                            tool_call_id=tool_id
                        ))
                else:
                    results.append(ToolMessage(
                        content=f"Tool {tool_name} not found",
                        tool_call_id=tool_id
                    ))
            
            return results
        
        # Mode 2: Execute complete tool chain (high-level operation)
        elif input_text is not None:
            from core.dependencies import get_llm
            model = get_llm()
            model_with_tools = self.bind_tools_to_model(model, tools)
            
            # Resolve tools for execution
            tool_list = self._resolve_tools(tools)
            
            # Get initial response with tool calls
            response = await model_with_tools.ainvoke([HumanMessage(content=input_text)])
            messages = [HumanMessage(content=input_text), response]
            
            # Execute any tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_results = await self.execute_tool_chain(
                    tool_calls=response.tool_calls, 
                    tools=tool_list
                )
                messages.extend(tool_results)
                
                # Get final response
                final_response = await model_with_tools.ainvoke(messages)
                messages.append(final_response)
            
            return messages
        
        else:
            raise ValueError("Must provide either input_text (for chain) or tool_calls (for execution)")
    
    
    def create_few_shot_tool_chain(
        self,
        examples: List[Dict[str, Any]],
        tools: List[BaseTool],
        **model_kwargs
    ) -> BaseChatModel:
        """
        Create a tool-enabled model with few-shot examples.
        
        Args:
            examples: Few-shot examples
            tools: Available tools
            **model_kwargs: Model parameters
        
        Returns:
            Configured model with few-shot prompting
        """
        from core.dependencies import get_llm
        model = get_llm()
        model_with_tools = self.bind_tools_to_model(model, tools)
        
        # Create few-shot prompt
        from libs.langchain.messages import create_few_shot_prompt
        prompt = create_few_shot_prompt(examples)
        
        # Chain prompt with model
        return prompt | model_with_tools
    
    
    @handle_errors("execute tool", return_on_error="Tool execution failed", raise_on_error=False)
    async def _execute_tool(
        self,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Execute tool with async fallback handling.
        
        Args:
            tool: Tool to execute
            tool_args: Arguments for the tool
            **kwargs: Additional parameters
        
        Returns:
            Tool execution result
        """
        import asyncio
        
        # Helper function to run sync tool safely
        async def _run_sync_tool():
            try:
                loop = asyncio.get_running_loop()
                # Run sync tool in thread pool to avoid blocking
                return await loop.run_in_executor(None, lambda: tool.invoke(tool_args, **kwargs))
            except RuntimeError:
                # No running loop, execute synchronously
                return tool.invoke(tool_args, **kwargs)
        
        # Try async first, then sync as fallback
        if hasattr(tool, 'ainvoke'):
            try:
                return await tool.ainvoke(tool_args, **kwargs)
            except Exception as async_error:
                # Async failed, try sync fallback if available
                if hasattr(tool, 'invoke'):
                    try:
                        return await _run_sync_tool()
                    except Exception:
                        # Both failed, raise original async error
                        raise async_error
                else:
                    raise async_error
        elif hasattr(tool, 'invoke'):
            # Only sync invoke available
            return await _run_sync_tool()
        else:
            # Neither method available
            raise AttributeError(f"Tool {tool.name} has no invoke or ainvoke method")
    
    def parse_tool_output(
        self,
        response: BaseMessage,
        tools: List[BaseTool]
    ) -> List[BaseModel]:
        """Parse tool calls from model response into structured objects."""
        parser = PydanticToolsParser(tools=tools)
        return parser.parse_result([response])

# Common tool implementations
@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for information.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
    
    Returns:
        Search results as formatted string
    """
    # Placeholder implementation
    return f"Search results for '{query}': [Mock results - implement actual search]"

@tool
def calculate(expression: str) -> Union[float, str]:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
    
    Returns:
        Result of the calculation or error message
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return float(result)
        else:
            return "Error: Expression contains invalid characters"
    except Exception as e:
        return f"Error: {str(e)}"

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: 72°F, sunny"

def calculate_tip(bill: float, percentage: float = 15.0) -> float:
    """Calculate tip amount for a bill."""
    return round(bill * (percentage / 100), 2)

# Example Pydantic models for structured tool outputs
class CalculationResult(BaseModel):
    """Result of a mathematical calculation."""
    expression: str = Field(description="The mathematical expression")
    result: float = Field(description="The calculated result")
    explanation: str = Field(description="Explanation of the calculation")

class SearchResult(BaseModel):
    """Result of a web search."""
    query: str = Field(description="The search query")
    results: List[str] = Field(description="List of search results")
    total_count: int = Field(description="Total number of results found")


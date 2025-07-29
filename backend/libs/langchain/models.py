"""LangChain chat models integration with OpenAI."""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Any, AsyncIterator, Dict, List, Optional, Type, TypeVar, Union, Literal, get_type_hints, TypedDict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from libs.langchain.tools import ToolManager
from core.config import settings
import time
import uuid

T = TypeVar('T', bound=BaseModel)

# Type aliases for input data and modes
ChatInput = Union[str, List[BaseMessage], List[Union[str, List[BaseMessage]]]]
ModeInput = Literal["invoke", "batch", "structured", "multimodal", "stream", "stream_events", "stream_tools"]

class ChatModel:
    """
    Unified chat model class with comprehensive LangChain integration.
    
    Provides a single interface for all LangChain chat model operations including
    streaming, tool calling, structured output, and multimodal support with 
    async/await throughout and runtime model switching capabilities.
    """
    
    model: BaseChatModel
    
    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        model_provider: str = "openai",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize chat model using LangChain's init_chat_model.
        
        Features:
        - Universal model initialization with init_chat_model
        - Runtime model switching via config parameter (use config param in run())
        - Async/await support throughout
        - Type safety with custom type aliases
        
        Args:
            model_name: Model name (defaults to "gpt-4.1-nano")
            model_provider: Provider name (defaults to "openai")
            temperature: Controls randomness (0.0-1.0, defaults to 0.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional model parameters
        """
        
        # Prepare kwargs for init_chat_model
        init_kwargs = {
            "temperature": temperature,
            "api_key": settings.openai_api_key,  # Pass API key explicitly
            **kwargs
        }
        
        # Handle max_tokens separately as it might not be supported by init_chat_model
        if max_tokens:
            init_kwargs["max_tokens"] = max_tokens
        
        # Use init_chat_model for initialization with model/provider specified
        # This still allows runtime switching via config parameter in run() calls
        self.model = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            **init_kwargs
        )
        
        # Initialize tool management
        self.tool_manager = ToolManager()
    
    # Unified async operation method
    def run(
        self,
        input_data: ChatInput,
        mode: ModeInput = "invoke",
        schema: Optional[Type[T]] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Unified method for all chat model operations.
        
        Streaming Capabilities:
        - Content streaming (stream)
        - Event streaming (stream_events) with filtering
        - Tool streaming (stream_tools) with chunk accumulation
        
        Data Processing:
        - Batch processing for efficiency
        - Structured output (Pydantic, TypedDict, JSON schema)
        - Multimodal input support (text + images)
        
        Args:
            input_data: Input message(s) or batch of inputs
            mode: Operation mode (invoke, batch, structured, multimodal, stream, stream_events, stream_tools)
            schema: Pydantic model for structured output
            image_url: URL for multimodal input
            image_base64: Base64 image for multimodal input
            config: Configuration for runtime model switching (e.g., {"configurable": {"model": "gpt-4o"}})
            **kwargs: Additional parameters
        
        Returns:
            AsyncIterator for streaming modes, Coroutine for non-streaming modes
        """
        # Handle streaming modes - return async generators directly
        if mode == "stream":
            return self._stream_data(input_data, "content", config=config, **kwargs)
        elif mode == "stream_events":
            return self._stream_data(input_data, "events", config=config, **kwargs)
        elif mode == "stream_tools":
            return self._stream_tools(input_data, config=config, **kwargs)
        
        # Handle non-streaming modes - return coroutine
        return self._process_non_streaming(input_data, mode, schema, image_url, image_base64, config=config, **kwargs)
    
    async def _stream_data(self, input_data: ChatInput, stream_type: Literal["content", "events"], config: Optional[Dict[str, Any]] = None, event_filter: Optional[List[str]] = None, **kwargs):
        """Stream content chunks or detailed events with functionality."""
        if stream_type == "content":
            async for chunk in self.model.astream(input_data, config=config, **kwargs):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        elif stream_type == "events":
            async for event in self.model.astream_events(input_data, config=config, **kwargs):
                # Basic event filtering if specified
                if event_filter:
                    event_type = event.get('event', '')
                    if event_type not in event_filter:
                        continue
                yield event
    
    async def _stream_tools(self, input_data: ChatInput, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Stream with tool call information.
        
        Important Notes:
        - Tools must be registered first: model.tool_manager.create_tool(your_function)
        - Models often choose tool-only responses without text content when tools are available
        - For text + tools together, use model.tool_manager.execute_tool_chain() instead
        - Tool arguments accumulate from empty {} to complete objects during streaming
        - Duplicate tool calls are automatically filtered out
        
        Troubleshooting:
        - Use explicit instructions: "Use the calculate tool to compute 2+2" 
        - Ensure tool names in tools=["tool_name"] match registered tool names
        - Tools need clear docstrings for model to understand when to use them
        """
        tools = kwargs.pop('tools', [])
        if not tools:
            raise ValueError("Tools required for stream_tools mode")
        
        model_with_tools = self.tool_manager.bind_tools_to_model(self, tools)
        
        # Track accumulated tool calls for streaming
        state = {
            'accumulated_tool_calls': {},
            'accumulated_args_strings': {}, # Track raw argument strings for each call
            'call_id_by_index': {}, # Map chunk index to call_id
            'previously_shown_calls': {} # Track what we've already shown to avoid duplicates
        }
        
        async for chunk in model_with_tools.astream(input_data, config=config, **kwargs):
            new_or_updated_calls = []
            
            # Process tool call chunks
            if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                for tool_chunk in chunk.tool_call_chunks:
                    call_id = self._resolve_call_id(tool_chunk, chunk, state['call_id_by_index'])
                    if call_id:
                        updated_call = self._process_tool_chunk(tool_chunk, call_id, state)
                        if updated_call:
                            new_or_updated_calls.append(updated_call)
            
            # Process complete tool calls
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    call_id = tool_call.get('id')
                    if self._is_new_complete_call(call_id, tool_call, state['previously_shown_calls']):
                        new_or_updated_calls.append(tool_call)
                        state['previously_shown_calls'][call_id] = tool_call['args'].copy()
            
            # Yield chunk with filtered tool calls
            chunk_data = self._create_tool_chunk_data(chunk)
            chunk_data['tool_calls'] = new_or_updated_calls
            yield chunk_data
    
    def _resolve_call_id(self, tool_chunk: Dict[str, Any], chunk: Any, call_id_by_index: Dict[int, str]) -> Optional[str]:
        """Resolve call ID from tool chunk, chunk context, or index mapping."""
        call_id = tool_chunk.get('id')
        chunk_index = tool_chunk.get('index', 0)
        
        # Try index mapping first
        if not call_id and chunk_index in call_id_by_index:
            call_id = call_id_by_index[chunk_index]
        
        # Fall back to main tool_calls
        if not call_id and hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            for main_call in chunk.tool_calls:
                if main_call.get('id'):
                    call_id = main_call['id']
                    call_id_by_index[chunk_index] = call_id
                    break
        
        # Generate call_id if still missing (for chunks without IDs)
        if not call_id:
            call_id = f"call_{chunk_index}"
            call_id_by_index[chunk_index] = call_id
        
        return call_id
    
    def _process_tool_chunk(self, tool_chunk: Dict[str, Any], call_id: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a tool chunk and return updated call if newly complete."""
        accumulated_calls = state['accumulated_tool_calls']
        accumulated_strings = state['accumulated_args_strings'] 
        previously_shown = state['previously_shown_calls']
        
        # Store previous state for comparison
        previous_args = accumulated_calls.get(call_id, {}).get('args', {})
        
        # Initialize accumulated call if needed
        if call_id not in accumulated_calls:
            accumulated_calls[call_id] = {
                'name': tool_chunk.get('name', ''),
                'args': {},
                'id': call_id,
                'type': 'tool_call'
            }
            accumulated_strings[call_id] = ''
        
        # Update name if provided (prioritize non-empty names)
        chunk_name = tool_chunk.get('name')
        if chunk_name and chunk_name.strip():
            accumulated_calls[call_id]['name'] = chunk_name
        
        # Accumulate and parse arguments
        if 'args' in tool_chunk and tool_chunk['args']:
            accumulated_strings[call_id] += str(tool_chunk['args'])
            
            # Try parsing accumulated JSON
            try:
                import json
                args_dict = json.loads(accumulated_strings[call_id])
                if isinstance(args_dict, dict):
                    accumulated_calls[call_id]['args'] = args_dict
                    
                    # Check if newly complete and not previously shown
                    current_args = accumulated_calls[call_id]['args']
                    if (current_args != previous_args and 
                        current_args and 
                        call_id not in previously_shown):
                        previously_shown[call_id] = current_args.copy()
                        return accumulated_calls[call_id]
            except:
                # JSON not complete yet
                pass
        
        return None
    
    def _is_new_complete_call(self, call_id: str, tool_call: Dict[str, Any], previously_shown: Dict[str, Any]) -> bool:
        """Check if this is a new complete tool call that hasn't been shown."""
        return (call_id and 
                tool_call.get('args') and 
                call_id not in previously_shown)
    
    def _create_tool_chunk_data(self, chunk) -> Dict[str, Any]:
        """Create standardized tool chunk data structure."""
        return {
            "content": getattr(chunk, "content", ""),
            "tool_calls": getattr(chunk, "tool_calls", []),
            "tool_call_chunks": getattr(chunk, "tool_call_chunks", [])
        }
    
    async def _process_non_streaming(
        self,
        input_data: ChatInput,
        mode: ModeInput,
        schema: Optional[Type[T]],
        image_url: Optional[str],
        image_base64: Optional[str],
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[AIMessage, List[AIMessage], T]:
        """Process non-streaming operations."""
        # Handle multimodal input
        if mode == "multimodal" or image_url or image_base64:
            if isinstance(input_data, str):
                input_data = [self.create_multimodal_message(input_data, image_url, image_base64)]
            return await self.model.ainvoke(input_data, config=config, **kwargs)
        
        # Handle structured output
        if mode == "structured" or schema:
            if not schema:
                raise ValueError("Schema required for structured output")
            return await self.with_structured_output(schema).ainvoke(input_data, config=config, **kwargs)
        
        # Handle batch processing
        if mode == "batch" or isinstance(input_data, list) and all(isinstance(item, (str, list)) for item in input_data):
            return await self.model.abatch(input_data, config=config, **kwargs)
        
        # Default: regular invoke
        return await self.model.ainvoke(input_data, config=config, **kwargs)
    
    # Tool functionality - delegate to tool_manager
    def bind_tools(self, tools: Optional[List[Union[str, BaseTool]]] = None, **kwargs) -> BaseChatModel:
        """
        Bind tools to the model for tool calling functionality.
        
        Tool Management Features:
        - Tool calling workflow
        - Async tool execution with sync fallback 
        - Few-shot prompting with tools
        - Tool management
        
        Note: Tools must be registered first using model.tool_manager.create_tool(your_function)
        or model.tool_manager.register_tool(existing_tool) before binding.
        """
        return self.tool_manager.bind_tools_to_model(self, tools)
    
    def with_structured_output(
        self, 
        schema: Union[Type[BaseModel], Type[Dict], Dict[str, Any]],
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs
    ) -> BaseChatModel:
        """
        Create a model that outputs structured data according to the given schema.
        
        Supports three schema types:
        - Pydantic models (recommended): class MyModel(BaseModel): field: str
        - TypedDict: class MyDict(TypedDict): field: str  
        - JSON schema: {"type": "object", "properties": {"field": {"type": "string"}}}
        
        Args:
            schema: Pydantic model class, TypedDict, or JSON schema dict
            method: "function_calling" (default, more reliable) or "json_mode"
            include_raw: Whether to include raw response alongside parsed output
            **kwargs: Additional parameters
        
        Returns:
            Model configured for structured output
        
        Note: For JSON mode, include "Please respond in JSON format" in your message.
        """
        return self.model.with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            **kwargs
        )
    
    # Helper method for multimodal messages
    def create_multimodal_message(
        self,
        text: str,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        **kwargs
    ) -> HumanMessage:
        """
        Create a multimodal message with text and optional image.
        
        Args:
            text: Text content
            image_url: URL to image
            image_base64: Base64 encoded image
            **kwargs: Additional content parameters
        
        Returns:
            HumanMessage with multimodal content
        """
        content = [{"type": "text", "text": text}]
        
        # Add image if provided
        image_data = image_url or (f"data:image/jpeg;base64,{image_base64}" if image_base64 else None)
        if image_data:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })
        
        return HumanMessage(content=content)
    

# Export type aliases for external use
__all__ = ['ChatModel', 'ChatInput', 'ModeInput']


"""
# 🚀 LangChain ChatModel Wrapper - Developer Guide

# 📋 Quick Start

from libs.langchain.models import ChatModel

# Initialize model (uses LangChain's init_chat_model internally)
model = ChatModel()  # Defaults: openai, gpt-4.1-nano, temperature=0.0

# Simple text completion
response = await model.run("Hello, how are you?")
print(response.content)

# 🔄 Runtime Model Switching

# Switch models at runtime without re-initialization
response = await model.run(
    "What's the weather like?",
    config={"configurable": {"model": "gpt-4o-mini"}}
)
print(response.content)

# 🌊 Streaming Modes

## Content Streaming
# Stream tokens as they're generated
async for token in model.run("Tell me a story", mode="stream"):
    print(token, end="", flush=True)

## Event Streaming
# Stream detailed generation events
async for event in model.run("Analyze this data", mode="stream_events"):
    print(f"Event: {event['event']}")
    
# Filter specific events
async for event in model.run(
    "Tell me about AI", 
    mode="stream_events",
    event_filter=["on_chat_model_stream", "on_chat_model_end"]
):
    print(f"Model Event: {event['event']}")

## Tool Streaming
from libs.langchain.tools import calculate
model.tool_manager.register_tool(calculate)

async for chunk in model.run(
    "Use the calculate tool to compute 50 * 0.15 for a tip calculation", 
    mode="stream_tools", 
    tools=["calculate"]
):
    if chunk.get('tool_calls'):
        for call in chunk['tool_calls']:
            print(f"🔧 {call['name']}: {call['args']}")
    if chunk.get('content'):
        print(f"💬 {chunk['content']}")

# Output: 🔧 calculate: {'expression': '50 * 0.15'}

# 📦 Batch Processing

# Process multiple inputs efficiently
inputs = ["What is Python?", "Explain machine learning", "How does AI work?"]
responses = await model.run(inputs, mode="batch")
for response in responses:
    print(response.content)

# 🏗️ Structured Output

## Using Pydantic Models
from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's job")

# Method 1: Via run()
person = await model.run(
    "John is a 30-year-old software engineer",
    mode="structured",
    schema=PersonInfo
)
print(person)

# Method 2: Via with_structured_output()
structured_model = model.with_structured_output(PersonInfo)
person = await structured_model.ainvoke("Jane is a 25-year-old designer")
print(person)

## Using TypedDict
from typing import TypedDict

class PersonDict(TypedDict):
    name: str
    age: int
    occupation: str

typed_model = model.with_structured_output(PersonDict)
result = await typed_model.ainvoke("Bob is a 35-year-old teacher")
print(result)

## Using JSON Schema
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "occupation": {"type": "string"}
    }
}
json_model = model.with_structured_output(json_schema, method="json_mode")
print(json_model)

# 🖼️ Multimodal Input

# With image URL
response = await model.run(
    "Describe this image",
    mode="multimodal",
    image_url="https://example.com/image.jpg"
)

# With base64 image
response = await model.run(
    "What's in this picture?",
    mode="multimodal",
    image_base64="iVBORw0KGgoAAAANSUhEUgAA..."
)

# 🛠️ Tool Calling

## Basic Tool Setup
from libs.langchain.tools import get_weather, calculate_tip

# Register tools (must be done before using in streaming)
model.tool_manager.create_tool(get_weather)
model.tool_manager.create_tool(calculate_tip)

## Tool Streaming
async for chunk in model.run(
    "What's the weather in NYC and calculate 18% tip on $45?",
    mode="stream_tools", 
    tools=["get_weather", "calculate_tip"]
):
    if chunk.get('tool_calls'):
        for call in chunk['tool_calls']:
            print(f"🔧 {call['name']}: {call['args']}")
    if chunk.get('content'):
        print(f"💬 {chunk['content']}")

# Output: 🔧 calculate: {'expression': '50 * 0.15'}

## Complete Tool Chain
# Execute full tool workflow
messages = await model.tool_manager.execute_tool_chain(
    "Get weather for San Francisco and help me tip on a $32 meal"
)
for msg in messages:
    print(f"{msg.__class__.__name__}: {msg.content}")

## Direct Tool Execution
# Execute specific tool calls
tool_calls = [
    {"name": "calculate_tip", "args": {"bill": 50.0, "percentage": 20.0}, "id": "call1"}
]
results = await model.tool_manager.execute_tool_chain(tool_calls=tool_calls)
print(results)

# 🎯 Few-Shot Prompting

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Create examples
examples = [
    HumanMessage(content="What's 2 + 3?"),
    AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "1"}]),
    ToolMessage(content="5", tool_call_id="1"),
    AIMessage(content="2 + 3 equals 5.")
]

# Create few-shot chain
few_shot_chain = model.tool_manager.create_few_shot_tool_chain(
    examples=examples,
    tools=[add, multiply],
    temperature=0.0
)

# 🔗 LangChain Integration

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create processing chain
prompt = ChatPromptTemplate.from_template("Translate to {language}: {text}")
chain = prompt | model.model | StrOutputParser()

# Use with runtime model switching
result = await chain.ainvoke(
    {"language": "Spanish", "text": "Hello, how are you today?"},
    config={"configurable": {"model": "gpt-4o"}}
)
print(result)  # "Hola, ¿cómo estás hoy?"

# ⚡ Performance Optimization

## Batch Processing

questions = ["What is Python?", "What is JavaScript?", "What is Go?"]
responses = await model.run(questions, mode="batch")
print(responses)

# Structured batch processing
structured_model = model.with_structured_output(TechExplanation)
structured_responses = await structured_model.abatch(questions)
print(structured_responses)

# 🏛️ Advanced Configuration

# Custom initialization
model = ChatModel(
    model_name="gpt-4o-mini",
    model_provider="openai",
    temperature=0.3,
    max_tokens=1000,
    request_timeout=60,
    max_retries=3
)

# Advanced tool binding
bound_model = model.tool_manager.bind_tools_to_model(
    model=model,
    tools=["calculate_tip", get_weather]  # Mix names and objects
)

# Create model with all registered tools
model_with_tools = model.tool_manager.bind_tools_to_model(
    temperature=0.3,
    tools=None  # Uses all registered tools
)

This implementation provides a unified, developer-friendly interface that covers all major LangChain chat model patterns while maintaining clean, maintainable code.
"""

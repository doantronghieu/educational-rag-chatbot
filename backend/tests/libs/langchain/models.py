from pathlib import Path
import sys
import asyncio
sys.path.append(str(Path(__file__).resolve().parents[4]))

from backend.libs.langchain.models import ChatModel
from backend.libs.langchain.tools import calculate
from backend.libs.langchain.tools import get_weather, calculate_tip
from pydantic import BaseModel, Field
from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



# Common test variables
DEFAULT_MODEL_CONFIG = {
    "model_name": "gpt-4.1-nano",
    "model_provider": "openai", 
    "temperature": 0.0
}

# Sample Pydantic model for structured output
class PersonInfo(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's job")

# Sample TypedDict for structured output
class PersonDict(TypedDict):
    name: str
    age: int
    occupation: str

# JSON Schema for structured output
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "occupation": {"type": "string"}
    }
}


def create_chat_model():
    """Initialize ChatModel for testing."""
    return ChatModel()


async def test_basic_chat():
    """Test basic chat functionality."""
    print("Testing basic chat functionality...")
    model = create_chat_model()
    response = await model.run("Hello, how are you?")
    print(f"Basic chat response: {response.content}")


async def test_runtime_model_switching():
    """Test runtime model switching."""
    print("Testing runtime model switching...")
    model = create_chat_model()
    response = await model.run(
        "What's the weather like?",
        config={"configurable": {"model": "gpt-4o-mini"}}
    )
    print(f"Model switching response: {response.content}")


async def test_content_streaming():
    """Test content streaming mode."""
    print("Testing content streaming...")
    model = create_chat_model()
    print("Streaming response: ", end="")
    async for token in model.run("Tell me a short story", mode="stream"):
        print(token, end="", flush=True)
    print("\n")


async def test_event_streaming():
    """Test event streaming mode."""
    print("Testing event streaming...")
    model = create_chat_model()
    event_count = 0
    async for event in model.run("Analyze this data", mode="stream_events"):
        event_count += 1
        print(f"Event {event_count}: {event['event']}")
        if event_count >= 5:  # Limit output for demo
            break
    print(f"Processed {event_count} events")


async def test_tool_streaming():
    """Test tool streaming functionality."""
    print("Testing tool streaming...")
    model = create_chat_model()
    model.tool_manager.register_tool(calculate)
    
    chunk_count = 0
    async for chunk in model.run(
        "Use the calculate tool to compute 50 * 0.15 for a tip calculation", 
        mode="stream_tools", 
        tools=["calculate"]
    ):
        chunk_count += 1
        if chunk.get('tool_calls'):
            for call in chunk['tool_calls']:
                print(f"🔧 {call.get('name', 'Unknown')}: {call.get('args', {})}")
        if chunk.get('content'):
            print(f"💬 {chunk['content']}")
        if chunk_count >= 3:  # Limit output for demo
            break


async def test_batch_processing():
    """Test batch processing functionality."""
    print("Testing batch processing...")
    model = create_chat_model()
    inputs = ["What is Python?", "Explain machine learning", "How does AI work?"]
    responses = await model.run(inputs, mode="batch")
    for i, response in enumerate(responses):
        print(f"Batch response {i+1}: {response.content[:50]}...")


async def test_structured_output_pydantic():
    """Test structured output with Pydantic model."""
    print("Testing structured output with Pydantic...")
    model = create_chat_model()
    
    # Method 1: Via run()
    person = await model.run(
        "John is a 30-year-old software engineer",
        mode="structured",
        schema=PersonInfo
    )
    print(f"Structured output (run): {person}")
    
    # Method 2: Via with_structured_output()
    structured_model = model.with_structured_output(PersonInfo)
    person = await structured_model.ainvoke("Jane is a 25-year-old designer")
    print(f"Structured output (with_structured_output): {person}")


async def test_structured_output_typed_dict():
    """Test structured output with TypedDict."""
    print("Testing structured output with TypedDict...")
    model = create_chat_model()
    typed_model = model.with_structured_output(PersonDict)
    result = await typed_model.ainvoke("Bob is a 35-year-old teacher")
    print(f"TypedDict output: {result}")


def test_structured_output_json_schema():
    """Test structured output with JSON schema."""
    print("Testing structured output with JSON schema...")
    model = create_chat_model()
    json_model = model.with_structured_output(JSON_SCHEMA, method="json_mode")
    print(f"JSON schema model: {json_model}")


async def test_multimodal_input():
    """Test multimodal input functionality."""
    print("Testing multimodal input...")
    model = create_chat_model()
    
    # With image URL
    response = await model.run(
        "Describe this image",
        mode="multimodal",
        image_url="https://www.nylabone.com/-/media/project/oneweb/nylabone/images/dog101/10-intelligent-dog-breeds/golden-retriever-tongue-out.jpg?h=430&w=710&hash=7FEB820D235A44B76B271060E03572C7"
    )
    print(f"Multimodal response (URL): {response.content}")
    
    # With base64 image (shortened for demo)
    # response = await model.run(
    #     "What's in this picture?",
    #     mode="multimodal",
    #     image_base64="iVBORw0KGgoAAAANSUhEUgAA..."
    # )
    # print(f"Multimodal response (base64): {response.content}")


async def test_tool_calling():
    """Test tool calling functionality."""
    print("Testing tool calling...")
    model = create_chat_model()
    
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
                print(f"🔧 {call.get('name', 'Unknown')}: {call.get('args', {})}")
        if chunk.get('content'):
            print(f"💬 {chunk['content']}")

    # Output: 🔧 calculate: {'expression': '50 * 0.15'}


async def test_complete_tool_chain():
    """Test complete tool chain execution."""
    print("Testing complete tool chain execution...")
    model = create_chat_model()
    
    # Register tools
    model.tool_manager.create_tool(get_weather)
    model.tool_manager.create_tool(calculate_tip)
    
    # Execute full tool workflow
    messages = await model.tool_manager.execute_tool_chain(
        "Get weather for San Francisco and help me tip on a $32 meal"
    )
    for msg in messages:
        print(f"{msg.__class__.__name__}: {msg.content}")


async def test_direct_tool_execution():
    """Test direct tool execution with specific tool calls."""
    print("Testing direct tool execution...")
    model = create_chat_model()
    
    # Register tools
    model.tool_manager.create_tool(calculate_tip)
    
    # Execute specific tool calls
    tool_calls = [
        {"name": "calculate_tip", "args": {"bill": 50.0, "percentage": 20.0}, "id": "call1"}
    ]
    results = await model.tool_manager.execute_tool_chain(tool_calls=tool_calls)
    print(f"Direct tool execution results: {results}")


async def test_few_shot_prompting():
    """Test few-shot prompting functionality."""
    print("Testing few-shot prompting...")
    model = create_chat_model()
    
    # Create examples
    examples = [
        HumanMessage(content="What's 2 + 3?"),
        AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "1"}]),
        ToolMessage(content="5", tool_call_id="1"),
        AIMessage(content="2 + 3 equals 5.")
    ]
    
    print(f"Few-shot examples created: {len(examples)} messages")


async def test_langchain_integration():
    """Test LangChain integration."""
    print("Testing LangChain integration...")
    model = create_chat_model()
    
    # Create processing chain
    prompt = ChatPromptTemplate.from_template("Translate to {language}: {text}")
    chain = prompt | model.model | StrOutputParser()
    
    # Use with runtime model switching
    result = await chain.ainvoke(
        {"language": "Spanish", "text": "Hello, how are you today?"},
        config={"configurable": {"model": "gpt-4o"}}
    )
    print(f"LangChain integration result: {result}")


async def test_batch_structured_processing():
    """Test batch processing with structured output."""
    print("Testing batch structured processing...")
    model = create_chat_model()
    structured_model = model.with_structured_output(PersonInfo)
    
    questions = [
        "John is a 30-year-old software engineer",
        "Mary is a 28-year-old doctor", 
        "Steve is a 45-year-old teacher"
    ]
    structured_responses = await structured_model.abatch(questions)
    for i, response in enumerate(structured_responses):
        print(f"Structured batch {i+1}: {response}")


def test_advanced_configuration():
    """Test advanced configuration options."""
    print("Testing advanced configuration...")
    
    # Custom initialization
    model = ChatModel(
        model_name="gpt-4o-mini",
        model_provider="openai",
        temperature=0.3,
        max_tokens=1000,
        request_timeout=60,
        max_retries=3
    )
    print(f"Custom model created: {model}")
    
    # Advanced tool binding
    bound_model = model.tool_manager.bind_tools_to_model(
        model=model,
        tools=["calculate"]  # Mix names and objects
    )
    print(f"Bound model created: {bound_model}")


async def main():
    """Run all ChatModel tests."""
    print("=== Running ChatModel Tests ===\n")
    
    await test_basic_chat()
    print()
    
    await test_runtime_model_switching()
    print()
    
    await test_content_streaming()
    print()
    
    await test_event_streaming()
    print()
    
    await test_tool_streaming()
    print()
    
    await test_batch_processing()
    print()
    
    await test_structured_output_pydantic()
    print()
    
    await test_structured_output_typed_dict()
    print()
    
    test_structured_output_json_schema()
    print()
    
    await test_multimodal_input()
    print()
    
    await test_tool_calling()
    print()
    
    await test_complete_tool_chain()
    print()
    
    await test_direct_tool_execution()
    print()
    
    await test_few_shot_prompting()
    print()
    
    await test_langchain_integration()
    print()
    
    await test_batch_structured_processing()
    print()
    
    test_advanced_configuration()
    print()
    
    print("=== All ChatModel tests completed ===")


if __name__ == "__main__":
    asyncio.run(main())
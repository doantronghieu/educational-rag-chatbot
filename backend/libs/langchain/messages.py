"""LangChain message and prompt utilities."""

from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate


def create_few_shot_prompt(
    examples: List[Dict[str, Any]],
    input_variables: List[str] = None
) -> ChatPromptTemplate:
    """
    Create a few-shot prompt template with tool usage examples.
    
    Args:
        examples: List of example interactions with 'human' and 'ai' keys
        input_variables: List of input variable names (defaults to ["input"])
    
    Returns:
        ChatPromptTemplate configured with examples and input placeholder
    """
    if input_variables is None:
        input_variables = ["input"]
    
    messages = []
    
    # Add examples
    for example in examples:
        if "human" in example:
            messages.append(("human", example["human"]))
        if "ai" in example:
            messages.append(("ai", example["ai"]))
    
    # Add input placeholder
    messages.append(("human", "{input}"))
    
    return ChatPromptTemplate.from_messages(messages)

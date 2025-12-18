#!/usr/bin/env python3
"""
benchmark_llms.py

A script to benchmark GPT-5.2, Gemini-3-Flash, and Grok-4-1-Fast-Non-Reasoning
on a query with web search enabled and "no reasoning" (or minimal).
Executes tests sequentially.
"""

import asyncio
import logging
import os
import sys
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# SDK Imports
from openai import AsyncOpenAI
from xai_sdk import AsyncClient as XAIClient
from xai_sdk.chat import user, system
from xai_sdk.tools import x_search, web_search as xai_web_search
from google import genai
from google.genai import types

# Configure Logging to Lowest Level (DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant."
TEST_QUESTION = "What is the current stock price of Google? Explain your reasoning."

async def call_gpt():
    """
    Calls GPT-5.2 with web search enabled and reasoning effort 'none'.
    """
    print("\n--- Testing GPT-5.2 ---")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    client = AsyncOpenAI(api_key=api_key)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TEST_QUESTION}
    ]

    create_kwargs = {
        "model": "gpt-5.2",
        "input": messages,
        "tools": [{"type": "web_search"}],
        "reasoning": {"effort": "none"}
    }

    try:
        logger.debug(f"Calling GPT with kwargs: {create_kwargs}")
        response = await client.responses.create(**create_kwargs)
        print("RAW RESPONSE:")
        print(response)
        
        # Extract and print reasoning tokens
        reasoning_tokens = 0
        if response.usage:
            # Check output_tokens_details first (Responses API standard)
            output_details = getattr(response.usage, 'output_tokens_details', None)
            completion_details = getattr(response.usage, 'completion_tokens_details', None)

            if output_details and hasattr(output_details, 'reasoning_tokens'):
                reasoning_tokens = output_details.reasoning_tokens
            elif completion_details and hasattr(completion_details, 'reasoning_tokens'):
                reasoning_tokens = completion_details.reasoning_tokens
            elif hasattr(response.usage, 'reasoning_tokens'):
                reasoning_tokens = response.usage.reasoning_tokens
                
        print(f"Reasoning Tokens: {reasoning_tokens}")
        
    except Exception as e:
        logger.error(f"GPT Error: {e}")

async def call_gemini():
    """
    Calls Gemini-3-Flash-Preview with web search and thinking level MINIMAL.
    """
    print("\n--- Testing Gemini-3-Flash-Preview ---")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY/GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)
    
    # Configure tools and thinking
    tools = [types.Tool(google_search=types.GoogleSearch())]
    config = types.GenerateContentConfig(
        tools=tools,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        system_instruction=SYSTEM_PROMPT
    )
    
    prompt = TEST_QUESTION

    try:
        logger.debug(f"Calling Gemini with model='gemini-3-flash-preview', config={config}")
        
        # Use native async via client.aio
        response = await client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=config
        )
        print("RAW RESPONSE:")
        print(response)
        
        reasoning_tokens = 0
        if response.usage_metadata:
            reasoning_tokens = getattr(response.usage_metadata, 'thoughts_token_count', 0)
        print(f"Reasoning Tokens: {reasoning_tokens}")
        
    except Exception as e:
        logger.error(f"Gemini Error: {e}")

async def call_grok():
    """
    Calls Grok-4-1-Fast-Non-Reasoning with web search (image understanding disabled) and NO reasoning effort.
    """
    print("\n--- Testing Grok-4-1-Fast-Non-Reasoning ---")
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY not found.")
        return

    async with XAIClient(api_key=api_key) as client:
        # Configure tools - disable image understanding
        tools = [
            xai_web_search(enable_image_understanding=False),
            x_search(enable_image_understanding=False)
        ]
        
        # Create kwargs - NO reasoning_effort
        create_kwargs = {
            "model": "grok-4-1-fast-non-reasoning",
            "tools": tools,
        }

        try:
            logger.debug(f"Creating Grok chat with kwargs: {create_kwargs}")
            chat = client.chat.create(**create_kwargs)
            
            chat.append(system(SYSTEM_PROMPT))
            chat.append(user(TEST_QUESTION))
            
            logger.debug("Sampling Grok response...")
            response = await chat.sample()
            print("RAW RESPONSE:")
            print(response)
            
            reasoning_tokens = 0
            if response.usage:
                reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)
            print(f"Reasoning Tokens: {reasoning_tokens}")
                
        except Exception as e:
            logger.error(f"Grok Error: {e}")

async def main():
    logger.debug("Starting benchmark...")
    # Sequential execution
    await call_gpt()
    await call_gemini()
    await call_grok()
    logger.debug("Benchmark complete.")

if __name__ == "__main__":
    asyncio.run(main())

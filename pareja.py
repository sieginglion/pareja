#!/usr/bin/env python3
"""
two_llm_synth_chainlit.py

Simplified flow:
  1) Ask GPT and Gemini the question.
  2) Pass both answers to GPT as reference.
  3) Take GPT's synthesis as the final answer.

Uses Chainlit for the UI and LangChain for LLM calls.

Requires:
  pip install langchain-openai langchain-core python-dotenv chainlit
  export OPENAI_API_KEY=...
"""

import asyncio
import os
from typing import List, Tuple, Dict

import chainlit as cl
from openai import OpenAI
try:
    from xai_sdk import AsyncClient, Client
    from xai_sdk.chat import user, system, assistant
    from xai_sdk.tools import x_search, web_search
    from google import genai
    from google.genai import types

except ImportError:
    # Fallback or strict requirement? Plan implied strict but let's be safe or just standard import
    pass 
from dotenv import load_dotenv

# Load env immediately
load_dotenv()

SYSTEM_PROMPT = "You are a buy-side analyst."

# --- Model variants ---
# --- Model variants ---
MODEL_GPT_BASE = "gpt-5.2"
MODEL_GEMINI_BASE = "gemini-3-flash-preview"
MODEL_GEMINI_REASONING = "gemini-3-pro-preview"
MODEL_GROK_BASE = "grok-4-1-fast-non-reasoning"
MODEL_GROK_REASONING = "grok-4-1-fast-reasoning"

HistoryItem = Tuple[str, str]  # (q, final)

def invoke_gpt(
    model: str,
    question: str,
    history: List[HistoryItem],
    web_search: bool = True,
    reasoning_effort: str = None
) -> Tuple[str, int]:
    """
    Invoke GPT using OpenAI SDK responses.create
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        return "Error: OPENAI_API_KEY is required.", 0
        
    client = OpenAI(
        api_key=api_key
    )
    
    # Format input as messages list
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if history:
         for h_q, h_a in history:
             messages.append({"role": "user", "content": h_q})
             messages.append({"role": "assistant", "content": h_a})
    
    messages.append({"role": "user", "content": question})
    
    # Tools config
    tools = [{"type": "web_search"}] if web_search else None
    
    # Reasoning config
    reasoning = {"effort": reasoning_effort} if reasoning_effort else None

    try:
        # Note: 'responses' is experimental/specific to the user's provider (OpenAI-next)
        # We assume client.responses.create exists and matches the user's snippet.
        response = client.responses.create(
            model=model,
            tools=tools,
            input=messages,
            reasoning=reasoning
        )
        # Accessing output_text as per user example
        text = getattr(response, 'output_text', str(response))
        
        # Extract reasoning tokens (OpenAI style)
        usage = getattr(response, 'usage', None)
        reasoning_tokens = 0
        if usage:
            # Common OpenAI structure: usage.completion_tokens_details.reasoning_tokens
            details = getattr(usage, 'completion_tokens_details', None)
            if details:
                reasoning_tokens = getattr(details, 'reasoning_tokens', 0)
            else:
                # Fallback for Some providers: usage.reasoning_tokens
                reasoning_tokens = getattr(usage, 'reasoning_tokens', 0)
        
        return text, reasoning_tokens
    except Exception as e:
        return f"Error invoking GPT: {e}", 0

async def invoke_grok(
    model: str,
    question: str,
    history: List[HistoryItem],
    web_search: bool = True,
    reasoning_effort: str = None
) -> Tuple[str, int]:
    """
    Invoke Grok using xAI SDK directly.
    """
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        return "Error: XAI_API_KEY not found."

    client = AsyncClient(api_key=api_key)
    
    # Tools
    tools = []
    if web_search:
        # Enable image understanding only when thinking (reasoning_effort is set)
        image_understanding = reasoning_effort is not None
        tools.append(web_search(enable_image_understanding=image_understanding))
        tools.append(x_search(enable_image_understanding=image_understanding))
    
    # Reconstruct history for Grok
    # xAI SDK chat expects a specific history object or methods to append.
    # It seems we create a chat session.
    create_kwargs = {
        "model": model,
        "tools": tools,
    }
    if reasoning_effort:
        create_kwargs["reasoning_effort"] = reasoning_effort

    chat = client.chat.create(**create_kwargs)
    
    # Add system prompt
    chat.append(system(SYSTEM_PROMPT))
    
    for q_hist, a_hist in history:
        chat.append(user(q_hist))
        chat.append(assistant(a_hist))
        
    chat.append(user(question))
    
    # print(f"DEBUG: invoking Grok {model}...")
    try:
        response = await chat.sample()
        text = response.content.strip() if response.content else ""
        
        # Extract tokens from Grok (xAI SDK)
        usage = getattr(response, 'usage', None)
        reasoning_tokens = 0
        if usage:
            reasoning_tokens = getattr(usage, 'reasoning_tokens', 0)
            
        return text, reasoning_tokens
    except Exception as e:
        return f"Error invoking Grok: {e}", 0

async def invoke_gemini(
    model: str,
    question: str,
    history: List[HistoryItem],
    thinking_mode: bool,
    web_search: bool = True
) -> Tuple[str, int]:
    """
    Invoke Gemini using google-genai SDK.
    """
    api_key = os.getenv("GOOGLE_API_KEY") 
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
         return "Error: GEMINI_API_KEY/GOOGLE_API_KEY not found."

    client = genai.Client(api_key=api_key)
    
    # Tools
    tools = []
    if web_search:
         tools.append(types.Tool(google_search=types.GoogleSearch()))
    
    # Configuration
    # thinking_level: 'high' if thinking_mode else 'minimal'
    # Note: 'minimal' might warn but is the requested setting for low reasoning.
    t_level = "high" if thinking_mode else "minimal"
    
    config = types.GenerateContentConfig(
        tools=tools,
        thinking_config=types.ThinkingConfig(thinking_level=t_level)
    )
    
    # Construct History + Prompt
    # Gemini SDK history is usually handled by keeping a chat session or passing contents list.
    # We'll concatenate for simplicity or use role-based contents if supported nicely.
    # For 'generate_content', we can pass a list of parts/messages.
    
    # Simple prompt construction as seen in test script, but preserving history context:
    full_prompt = ""
    if SYSTEM_PROMPT:
        full_prompt += f"System: {SYSTEM_PROMPT}\n\n"
    
    for q_h, a_h in history:
        full_prompt += f"User: {q_h}\nModel: {a_h}\n\n"
    
    full_prompt += f"User: {question}"

    try:
        # We wrap in to_thread if it was blocking, but the new SDK might support async?
        # The new SDK has an async client too usually, but 'genai.Client' is sync?
        # Re-checking imports.. 'from google import genai'. 
        # The documentation showed `client.models.generate_content`.
        # To be safe with async loop, we'll run it in a thread or check for async client.
        # The test script used sync `client`. `pareja.py` is async.
        # Let's wrap in to_thread.
        
        def _run_gemini():
            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=config
            )
            # Extract text and thoughts_token_count
            text = response.text
            usage = getattr(response, 'usage_metadata', None)
            reasoning_tokens = 0
            if usage:
                reasoning_tokens = getattr(usage, 'thoughts_token_count', 0)
            return text, reasoning_tokens

        return await asyncio.to_thread(_run_gemini)

    except Exception as e:
        return f"Error invoking Gemini: {e}", 0

async def first_pass(
    gpt_model_name: str,
    gpt_reasoning_effort: str,
    gemini_model: str,
    gemini_thinking: bool,
    grok_model_name: str,
    grok_reasoning_effort: str,
    q: str,
    history: List[HistoryItem]
) -> Tuple[Tuple[str, int], Tuple[str, int], Tuple[str, int]]:
    # Run three distinct models in parallel
    # Note: invoke_gpt is synchronous in the user example (client.responses.create)
    # wraps in to_thread for non-blocking async
    a0, b0, c0 = await asyncio.gather(
        asyncio.to_thread(invoke_gpt, gpt_model_name, q, history, web_search=True, reasoning_effort=gpt_reasoning_effort),
        invoke_gemini(gemini_model, q, history, gemini_thinking),
        invoke_grok(grok_model_name, q, history, web_search=True, reasoning_effort=grok_reasoning_effort)
    )
    return a0, b0, c0

async def synthesize_final(
    gpt_model: str,
    question: str,
    answer_a0: str,
    answer_b0: str,
    answer_c0: str,
    history: List[HistoryItem],
    reasoning_effort: str = None
) -> str:
    prompt = f"""<prompt>
{question}
</prompt>
<response model="GPT">
{answer_a0}
</response>
<response model="Gemini">
{answer_b0}
</response>
<response model="Grok">
{answer_c0}
</response>
Here are three responses to the prompt. Generate a final one based on them
"""
    # For synthesis, we can use the same invoke_gpt mechanism
    # web_search=False for synthesis usually
    text, _ = await asyncio.to_thread(invoke_gpt, gpt_model, prompt, history, web_search=False, reasoning_effort=reasoning_effort)
    return text


@cl.on_chat_start
def start():
    # Initialize history in user session
    cl.user_session.set("history", [])

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    
    raw = message.content.strip()
    
    # Check for /think prefix
    thinking_mode = raw.startswith("/think")
    if thinking_mode:
        q = raw[6:].strip()
        if not q:
            await cl.Message(content="Usage: /think <your question>").send()
            return
        await cl.Message(content="🧠 **Reasoning mode enabled**").send()
    else:
        q = raw

    # Build LLMs
    # gpt = _make_llm(MODEL_GPT_BASE, thinking_mode) # NO longer used via LangChain
    # gemini = _make_llm(MODEL_GEMINI_BASE, thinking_mode) # NO longer used via LangChain
    
    grok_model = MODEL_GROK_REASONING if thinking_mode else MODEL_GROK_BASE
    grok_reasoning = "high" if thinking_mode else None
    
    gemini_model = MODEL_GEMINI_REASONING if thinking_mode else MODEL_GEMINI_BASE

    gpt_reasoning = "xhigh" if thinking_mode else None

    # gpt_synth = _make_llm(MODEL_GPT_BASE, thinking=False, web_search=False) # No longer used

    # Step 1: First Pass
    async with cl.Step(name="Step 1") as step:
        res_a0, res_b0, res_c0 = await first_pass(MODEL_GPT_BASE, gpt_reasoning, gemini_model, thinking_mode, grok_model, grok_reasoning, q, history)
        a0, t_a0 = res_a0
        b0, t_b0 = res_b0
        c0, t_c0 = res_c0
        step.output = f'# gpt (reasoning: {t_a0})\n{a0}\n\n# gemini (reasoning: {t_b0})\n{b0}\n\n# grok (reasoning: {t_c0})\n{c0}'

    # Send intermediate outputs as their own messages if preferred, 
    # but Elements are cleaner for "First Pass" content to avoid clutter.
    # However, the user might want to see them directly.
    # Let's send them as collapsible messages (Actions) or just simple Messages?
    # The requirement is just "interface".
    
    # Let's pop out the answers as well so they are visible in the chat stream 
    # but maybe in a cleaner way. 
    # Actually, appending them to the next message or just showing them above.
    
    # The intermediate answers are already shown via the inline Text elements attached to the step.
    # explicit message removed to avoid duplication.

    # Step 2: Synthesis
    async with cl.Step(name="Step 2") as step:
        final = await synthesize_final(MODEL_GPT_BASE, q, a0, b0, c0, history, reasoning_effort=gpt_reasoning)
        step.output = "Done"

    # Final Answer
    await cl.Message(content=final).send()

    # Update History
    history.append((q, final))
    cl.user_session.set("history", history)

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
  export OPENROUTER_API_KEY=...
"""

import asyncio
import os
from typing import List, Tuple, Dict

import chainlit as cl
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load env immediately
load_dotenv()

SYSTEM_PROMPT = "You are a buy-side analyst."

# --- Model variants ---
MODEL_GPT_BASE = "openai/gpt-5.2"
MODEL_GEMINI_BASE = "google/gemini-3-pro-preview"
TEMPERATURE = 0.7

HistoryItem = Tuple[str, str]  # (q, final)

def _build_messages(history: List[HistoryItem], user_prompt: str) -> List[BaseMessage]:
    msgs: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for q, a in history:
        msgs.append(HumanMessage(content=q))
        msgs.append(AIMessage(content=a))
    msgs.append(HumanMessage(content=user_prompt))
    return msgs

def _make_llm(model: str, thinking: bool = False, web_search: bool = True) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required.")

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

    default_headers: Dict[str, str] = {}
    if os.getenv("OPENROUTER_HTTP_REFERER", "").strip():
        default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    if os.getenv("OPENROUTER_X_TITLE", "").strip():
        default_headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE", "").strip()

    # Reasoning config: "high" effort when /think is used, otherwise disabled
    reasoning_config = {"effort": "high"} if thinking else None

    # Web search tool config
    model_kwargs = {"tools": [{"type": "web_search"}]} if web_search else {}

    return ChatOpenAI(
        model=model,
        temperature=TEMPERATURE,
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers or None,
        use_responses_api=True,
        reasoning=reasoning_config,
        model_kwargs=model_kwargs,
    )

async def invoke_with_history(llm: ChatOpenAI, question: str, history: List[HistoryItem]) -> str:
    messages = _build_messages(history, question)
    resp = await llm.ainvoke(messages)
    content = resp.content
    print(f"DEBUG: invoking {llm.model_name}, raw_content type={type(content)}")
    
    # Handle Responses API: content can be a list of blocks
    if isinstance(content, list):
        text_parts = [block.get("text", "") for block in content if isinstance(block, dict)]
        content = "\n".join(text_parts)
    return (content or "").strip()

async def first_pass(
    gpt: ChatOpenAI,
    gemini: ChatOpenAI,
    q: str,
    history: List[HistoryItem]
) -> Tuple[str, str]:
    # Run both distinct models in parallel
    a0, b0 = await asyncio.gather(
        invoke_with_history(gpt, q, history),
        invoke_with_history(gemini, q, history),
    )
    return a0, b0

async def synthesize_final(
    gpt: ChatOpenAI,
    question: str,
    answer_a0: str,
    answer_b0: str,
    history: List[HistoryItem],
) -> str:
    prompt = f"""<prompt>
{question}
</prompt>
<response>
{answer_a0}
</response>
<response>
{answer_b0}
</response>
Here are two responses to the prompt. Generate a final one based on them
"""
    return await invoke_with_history(gpt, prompt, history)


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
    gpt = _make_llm(MODEL_GPT_BASE, thinking_mode)
    gemini = _make_llm(MODEL_GEMINI_BASE, thinking_mode)
    gpt_synth = _make_llm(MODEL_GPT_BASE, thinking=False, web_search=False)

    # Step 1: First Pass
    async with cl.Step(name="Step 1") as step:
        a0, b0 = await first_pass(gpt, gemini, q, history)
        step.output = f'# gpt\n{a0}\n# gemini\n{b0}'

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
        final = await synthesize_final(gpt_synth, q, a0, b0, history)
        step.output = "Done"

    # Final Answer
    await cl.Message(content=final).send()

    # Update History
    history.append((q, final))
    cl.user_session.set("history", history)

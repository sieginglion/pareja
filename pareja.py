#!/usr/bin/env python3
"""
two_llm_synth_chainlit.py

Simplified flow:
  1) Ask GPT, Gemini, Grok, and Claude the question.
  2) Pass all answers to GPT as reference.
  3) Take GPT's synthesis as the final answer.

Uses Chainlit for the UI.

Requires:
  pip install python-dotenv chainlit anthropic
  export OPENAI_API_KEY=...
"""

import asyncio
import os
from typing import Dict, List, Tuple

import anthropic
import chainlit as cl
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from xai_sdk import AsyncClient, Client
from xai_sdk.chat import assistant, system, user
from xai_sdk.tools import web_search as xai_web_search
from xai_sdk.tools import x_search

# Load env immediately
load_dotenv()

SYSTEM_PROMPT = "You are a buy-side analyst."

# --- Model variants ---
# --- Model variants ---
MODEL_GPT_BASE = "gpt-5.4"
MODEL_GEMINI = "gemini-3.1-pro-preview"
MODEL_GROK_BASE = "grok-4-1-fast-non-reasoning"
MODEL_CLAUDE = "claude-opus-4-6"
ENABLE_GROK = False
ENABLE_SEARCH = False

HistoryItem = Tuple[str, str]  # (q, final)


def _openai_to_tuples(messages: List[Dict[str, str]]) -> List[HistoryItem]:
    history = []
    current_user = None
    current_assistant = None

    for m in messages:
        if m['role'] == 'user':
            # If we were building a turn, save it
            if current_user is not None and current_assistant is not None:
                history.append((current_user, current_assistant))

            # Start new turn
            current_user = m['content']
            current_assistant = None

        elif m['role'] == 'assistant':
            # If we have a user message waiting, update the answer
            # This ensures we get the *last* assistant message (e.g. ignoring intermediate steps if present)
            current_assistant = m['content']

    # Append the final turn if complete
    if current_user is not None and current_assistant is not None:
        history.append((current_user, current_assistant))

    return history


async def invoke_gpt(
    model: str,
    question: str,
    history: List[HistoryItem],
    thinking_mode: bool = False,
    web_search: bool = True,
) -> Tuple[str, int]:
    """
    Invoke GPT using AsyncOpenAI SDK responses.create
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        return "Error: OPENAI_API_KEY is required.", 0

    client = AsyncOpenAI(api_key=api_key)

    # Keep conversation state client-side; pass system guidance via top-level instructions.
    messages: List[Dict[str, str]] = []

    if history:
        for h_q, h_a in history:
            messages.append({"role": "user", "content": h_q})
            messages.append({"role": "assistant", "content": h_a})

    messages.append({"role": "user", "content": question})

    # Build kwargs dynamically to omit tools if not needed (best practice)
    create_kwargs = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": messages,
    }

    if web_search:
        create_kwargs["tools"] = [{"type": "web_search"}]

    create_kwargs["reasoning"] = {"effort": "high" if thinking_mode else "low"}

    try:
        # Note: 'responses' is experimental/specific to the user's provider (OpenAI-next)
        response = await client.responses.create(**create_kwargs)

        # Accessing output_text as per user example
        text = getattr(response, 'output_text', str(response))

        # Extract reasoning tokens (OpenAI Responses API style)
        # Usage is typically response.usage
        usage = getattr(response, 'usage', None)
        reasoning_tokens = 0
        if usage:
            # Check output_tokens_details first (Responses API standard)
            output_details = getattr(usage, 'output_tokens_details', None)
            completion_details = getattr(
                usage, 'completion_tokens_details', None
            )  # Fallback

            if output_details and hasattr(output_details, 'reasoning_tokens'):
                reasoning_tokens = output_details.reasoning_tokens
            elif completion_details and hasattr(completion_details, 'reasoning_tokens'):
                reasoning_tokens = completion_details.reasoning_tokens
            elif hasattr(usage, 'reasoning_tokens'):
                reasoning_tokens = usage.reasoning_tokens

        return text, reasoning_tokens
    except Exception as e:
        return f"Error invoking GPT: {e}", 0


async def invoke_grok(
    model: str,
    question: str,
    history: List[HistoryItem],
    thinking_mode: bool = False,
    web_search: bool = True,
) -> Tuple[str, int]:
    """
    Invoke Grok using xAI SDK directly.
    """
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        return "Error: XAI_API_KEY not found.", 0

    async with AsyncClient(api_key=api_key) as client:
        # Tools
        tools = []
        if web_search:
            tools.append(xai_web_search())
            tools.append(x_search())

        # Reconstruct history for Grok
        create_kwargs = {
            "model": model,
            "tools": tools,
        }

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
    thinking_mode: bool = False,
    web_search: bool = True,
) -> Tuple[str, int]:
    """
    Invoke Gemini using google-genai SDK.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Error: GEMINI_API_KEY/GOOGLE_API_KEY not found.", 0

    client = genai.Client(api_key=api_key)

    # Tools
    tools = []
    if web_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    t_level = types.ThinkingLevel.HIGH if thinking_mode else types.ThinkingLevel.LOW

    config = types.GenerateContentConfig(
        tools=tools,
        thinking_config=types.ThinkingConfig(thinking_level=t_level),
        system_instruction=SYSTEM_PROMPT,
    )

    # Construct History + Prompt using structured Content objects
    # Convert OpenAI-style history to Google GenAI Content objects
    contents = []

    # System prompt is handled via config, so we skip it if it was in history (though cl.chat_context usually doesn't include system)

    for q_h, a_h in history:
        role_u = "user"
        role_m = "model"

        contents.append(types.Content(role=role_u, parts=[types.Part(text=q_h)]))
        contents.append(types.Content(role=role_m, parts=[types.Part(text=a_h)]))

    # Add the current question
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    try:
        # Use native async client.aio
        response = await client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )

        # Extract text and thoughts_token_count
        text = response.text
        usage = getattr(response, 'usage_metadata', None)
        reasoning_tokens = 0
        if usage:
            reasoning_tokens = getattr(usage, 'thoughts_token_count', 0)
        return text, reasoning_tokens

    except Exception as e:
        return f"Error invoking Gemini: {e}", 0


async def invoke_claude(
    model: str,
    question: str,
    history: List[Tuple[str, str]],
    thinking_mode: bool = False,
    web_search: bool = True,
) -> Tuple[str, int]:
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

    messages = []
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})

    kwargs = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "max_tokens": 16384,
        "messages": messages,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high" if thinking_mode else "low"},
    }

    if web_search:
        kwargs["tools"] = [{"type": "web_search_20260209", "name": "web_search"}]

    try:
        response = await client.messages.create(**kwargs)
        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        return text, getattr(response.usage, "output_tokens", 0)
    except Exception as e:
        return f"Error: {e}", 0


async def first_pass(
    gpt_model_name: str,
    gemini_model: str,
    thinking_mode: bool,
    claude_model_name: str,
    grok_enabled: bool,
    grok_model_name: str,
    q: str,
    history: List[HistoryItem],
) -> Tuple[Tuple[str, int], Tuple[str, int], Tuple[str, int], Tuple[str, int]]:
    grok_call = (
        invoke_grok(
            grok_model_name,
            q,
            history,
            thinking_mode=thinking_mode,
            web_search=ENABLE_SEARCH,
        )
        if grok_enabled
        else asyncio.sleep(0, result=("Grok disabled.", 0))
    )

    a0, b0, c0, d0 = await asyncio.gather(
        invoke_gpt(
            gpt_model_name,
            q,
            history,
            thinking_mode=thinking_mode,
            web_search=ENABLE_SEARCH,
        ),
        invoke_gemini(
            gemini_model,
            q,
            history,
            thinking_mode=thinking_mode,
            web_search=ENABLE_SEARCH,
        ),
        invoke_claude(
            claude_model_name,
            q,
            history,
            thinking_mode=thinking_mode,
            web_search=ENABLE_SEARCH,
        ),
        grok_call,
    )
    return a0, b0, c0, d0


async def synthesize_final(
    gpt_model: str,
    question: str,
    answer_a0: str,
    answer_b0: str,
    answer_c0: str,
    answer_d0: str,
    grok_enabled: bool,
    history: List[HistoryItem],
    thinking_mode: bool = False,
) -> str:
    response_blocks = [
        f"""<response model="gpt">
{answer_a0}
</response>""",
        f"""<response model="gemini">
{answer_b0}
</response>""",
        f"""<response model="claude">
{answer_c0}
</response>""",
    ]
    if grok_enabled:
        response_blocks.append(
            f"""<response model="grok">
{answer_d0}
</response>"""
        )

    prompt = f"""<prompt>
{question}
</prompt>
{chr(10).join(response_blocks)}
The above are responses to the prompt. Merge them. If there are major conflicts, list them.
"""
    # Synthesis should only merge the first-pass answers, so search stays off here.
    text, _ = await invoke_gpt(
        gpt_model,
        prompt,
        history,
        thinking_mode=thinking_mode,
        web_search=False,
    )
    return text


@cl.on_chat_start
def start():
    # No manual history initialization needed
    pass


@cl.on_message
async def main(message: cl.Message):
    # Get history from Chainlit context (handles edits)
    openai_history = cl.chat_context.to_openai()
    history = _openai_to_tuples(openai_history)
    print(f"DEBUG HISTORY: {history}")

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

    grok_enabled = ENABLE_GROK
    grok_model = MODEL_GROK_BASE
    claude_model = MODEL_CLAUDE

    # Step 1: First Pass
    async with cl.Step(name="Step 1") as step:
        res_a0, res_b0, res_c0, res_d0 = await first_pass(
            MODEL_GPT_BASE,
            MODEL_GEMINI,
            thinking_mode,
            claude_model,
            grok_enabled,
            grok_model,
            q,
            history,
        )
        a0, t_a0 = res_a0
        b0, t_b0 = res_b0
        c0, t_c0 = res_c0
        d0, t_d0 = res_d0
        step_output_parts = [
            f'# gpt (reasoning_tokens: {t_a0})\n{a0}',
            f'# gemini (thoughts_token_count: {t_b0})\n{b0}',
            f'# claude (output_tokens: {t_c0})\n{c0}',
        ]
        if grok_enabled:
            step_output_parts.append(f'# grok (reasoning_tokens: {t_d0})\n{d0}')
        step.output = "\n\n".join(step_output_parts)

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
        final = await synthesize_final(
            MODEL_GPT_BASE,
            q,
            a0,
            b0,
            c0,
            d0,
            grok_enabled,
            history,
            thinking_mode=False,
        )
        step.output = "Done"

    # Final Answer
    await cl.Message(content=final).send()

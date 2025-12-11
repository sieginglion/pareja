#!/usr/bin/env python3
"""
two_llm_synth_langchain.py

Simplified flow:
  1) Ask GPT and Gemini the question.
  2) Pass both answers to GPT as reference.
  3) Take GPT's synthesis as the final answer.

Uses LangChain (OpenRouter OpenAI-compatible endpoint) and `rich` for output.

Requires:
  pip install langchain-openai langchain-core python-dotenv rich
  export OPENROUTER_API_KEY=...

Optional:
  export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
  export OPENROUTER_HTTP_REFERER=...
  export OPENROUTER_X_TITLE=...
"""

import asyncio
import os
import sys
from typing import List, Tuple, Dict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

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
Respond to the prompt based on the two responses
"""
    return await invoke_with_history(gpt, prompt, history)


def _print_block(console: Console, title: str, text: str, style: str = "white") -> None:
    console.print(Panel(Markdown(text), title=title, style=style))


async def main() -> int:
    load_dotenv()
    console = Console()

    history: List[HistoryItem] = []

    while True:
        try:
            raw = input().strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            break

        if not raw or raw.lower() in ("exit", "quit"):
            break

        # Check for /think prefix to enable reasoning mode
        thinking_mode = raw.startswith("/think")
        if thinking_mode:
            q = raw[6:].strip()  # Remove "/think" prefix
            if not q:
                console.print("[yellow]Usage: /think <your question>[/yellow]")
                continue
            console.print("[bold magenta]🧠 Reasoning mode enabled[/bold magenta]")
        else:
            q = raw

        # Build LLMs with web search via tools API, reasoning enabled if /think
        gpt = _make_llm(MODEL_GPT_BASE, thinking_mode)
        gemini = _make_llm(MODEL_GEMINI_BASE, thinking_mode)
        # Synthesis GPT: no web search or reasoning (already has reference answers)
        gpt_synth = _make_llm(MODEL_GPT_BASE, thinking=False, web_search=False)

        try:
            # First pass: both models answer the question.
            a0, b0 = await first_pass(gpt, gemini, q, history)
            _print_block(console, "GPT (First Pass)", a0, style="bold blue")
            _print_block(console, "Gemini (First Pass)", b0, style="bold green")

            # Synthesis step: GPT sees both answers and produces the final.
            final = await synthesize_final(gpt_synth, q, a0, b0, history)
            _print_block(console, "Final Answer", final, style="bold yellow")

            # Store final in history for conversational context.
            history.append((q, final))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)

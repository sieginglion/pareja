#!/usr/bin/env python3
"""
two_llm_synth_langchain.py

Same spec/flow, but implemented with LangChain (OpenRouter OpenAI-compatible endpoint).
Uses `rich` for conversational output.

Requires:
  pip install langchain-openai langchain-core python-dotenv rich
  export OPENROUTER_API_KEY=...

Optional:
  export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
  export OPENROUTER_HTTP_REFERER=...
  export OPENROUTER_X_TITLE=...
  export ENABLE_WEB_SEARCH=1   (best-effort; depends on route/provider)
"""

import asyncio
import os
import sys
import signal
from typing import List, Tuple, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

SYSTEM_PROMPT = "You are a buy-side analyst."

MODEL_GPT = "openai/gpt-5.1"
MODEL_GEMINI = "google/gemini-3-pro-preview"
TEMPERATURE = 0.8

HistoryItem = Tuple[str, str]  # (q, final)


def _extra_body_for_web_search() -> Dict[str, Any]:
    # Best-effort OpenRouter pattern (not universal across providers/routes).
    if os.getenv("ENABLE_WEB_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return {"plugins": [{"id": "web"}]}
    return {}


def _build_messages(history: List[HistoryItem], user_prompt: str) -> List[BaseMessage]:
    msgs: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for q, a in history:
        msgs.append(HumanMessage(content=q))
        msgs.append(AIMessage(content=a))
    msgs.append(HumanMessage(content=user_prompt))
    return msgs


def _make_llm(model: str) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required.")

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

    default_headers: Dict[str, str] = {}
    if os.getenv("OPENROUTER_HTTP_REFERER", "").strip():
        default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    if os.getenv("OPENROUTER_X_TITLE", "").strip():
        default_headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE", "").strip()

    extra_body = _extra_body_for_web_search()

    return ChatOpenAI(
        model=model,
        temperature=TEMPERATURE,
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers or None,
        extra_body=extra_body or None,
    )


async def invoke_with_history(llm: ChatOpenAI, question: str, history: List[HistoryItem]) -> str:
    messages = _build_messages(history, question)
    resp = await llm.ainvoke(messages)
    return (resp.content or "").strip()


async def first_pass(gpt: ChatOpenAI, gemini: ChatOpenAI, q: str, history: List[HistoryItem]) -> Tuple[str, str]:
    a0, b0 = await asyncio.gather(
        invoke_with_history(gpt, q, history),
        invoke_with_history(gemini, q, history)
    )
    return a0, b0


async def second_pass(
    gpt: ChatOpenAI,
    gemini: ChatOpenAI,
    question: str,
    answer_a0: str,
    answer_b0: str,
    history: List[HistoryItem],
) -> Tuple[str, str]:
    prompt = f"""<text>
{answer_a0}
</text>
<text>
{answer_b0}
</text>

Based on the two texts, answer the question: {question}
"""
    a1, b1 = await asyncio.gather(
        invoke_with_history(gpt, prompt, history),
        invoke_with_history(gemini, prompt, history)
    )
    return a1, b1


async def final_merge(gpt: ChatOpenAI, answer_a1: str, answer_b1: str, history: List[HistoryItem]) -> str:
    prompt = f"""<text>
{answer_a1}
</text>
<text>
{answer_b1}
</text>

Merge the two texts.
"""
    return await invoke_with_history(gpt, prompt, history)


def _print_block(console: Console, title: str, text: str, style: str = "white") -> None:
    console.print(Panel(Markdown(text), title=title, style=style))


async def main() -> int:
    load_dotenv()
    console = Console()
    gpt = _make_llm(MODEL_GPT)
    gemini = _make_llm(MODEL_GEMINI)

    history: List[HistoryItem] = []

    def handle_sigint(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        try:
            q = input().strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            break

        if not q or q.lower() in ("exit", "quit"):
            break

        try:
            a0, b0 = await first_pass(gpt, gemini, q, history)
            _print_block(console, "GPT (First Pass)", a0, style="bold blue")
            _print_block(console, "Gemini (First Pass)", b0, style="bold green")
            a1, b1 = await second_pass(gpt, gemini, q, a0, b0, history)
            _print_block(console, "GPT (Second Pass)", a1, style="blue")
            _print_block(console, "Gemini (Second Pass)", b1, style="green")
            final = await final_merge(gpt, a1, b1, history)
            _print_block(console, "Final Answer", final, style="bold yellow")

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

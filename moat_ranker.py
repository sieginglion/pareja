#!/usr/bin/env python3
"""Rank the moat width of CSV-style stock lists using the Pareja model panel.

Read one security per line from standard input, for example:

    NVDA,NVIDIA Corporation
    PLTR,Palantir Technologies Inc.
    NFLX,"Netflix, Inc."

The script processes four securities at a time.  For each batch it asks the
four Pareja models about their moats, has GPT synthesize their answers, asks
each model to "think deeper" with that synthesis as shared history, and
synthesizes again.  It then asks all four models to rank the combined batch
findings into four moat-width tiers and performs one final GPT synthesis.

Required environment variables:
  OPENAI_API_KEY, GOOGLE_API_KEY (or GEMINI_API_KEY), XAI_API_KEY,
  ANTHROPIC_API_KEY

Install dependencies with:
  pip install anthropic google-genai openai python-dotenv xai-sdk
"""

import argparse
import asyncio
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from xai_sdk import AsyncClient
from xai_sdk.chat import assistant, user

# These are the model variants used by pareja.py.
MODEL_GPT_BASE = "gpt-5.6-terra"
MODEL_GEMINI = "gemini-3.1-pro-preview"
MODEL_GROK_BASE = "grok-4.5"
MODEL_CLAUDE = "claude-opus-4-8"

HistoryItem = Tuple[str, str]


class GPTInvocationError(RuntimeError):
    """Raised when GPT cannot provide a usable response for a required step."""


@dataclass(frozen=True)
class Models:
    gpt: str
    gemini: str
    grok: str
    claude: str


def progress(message: str) -> None:
    """Keep stdout reserved for the final synthesis."""
    print(message, file=sys.stderr, flush=True)


async def invoke_gpt(
    model: str, question: str, history: Sequence[HistoryItem]
) -> Tuple[str, int]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise GPTInvocationError("OPENAI_API_KEY is required.")

    messages: List[Dict[str, str]] = []
    for prior_question, prior_answer in history:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})

    async def request() -> Tuple[str, int]:
        # Disable SDK retries so every GPT request is attempted exactly once.
        async with AsyncOpenAI(api_key=api_key, max_retries=0) as client:
            response = await client.responses.create(
                model=model,
                input=messages,
                reasoning={"effort": "xhigh"},
            )
        response_error = getattr(response, "error", None)
        if response_error is not None:
            code = getattr(response_error, "code", "") or ""
            message = getattr(response_error, "message", "") or "Unknown model failure."
            code_text = f" [{code}]" if code else ""
            raise GPTInvocationError(f"GPT model response{code_text}: {message}")

        status = getattr(response, "status", None)
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) or "unknown reason"
            raise GPTInvocationError(f"GPT incomplete response ({reason}).")
        if status and status != "completed":
            raise GPTInvocationError(f"GPT response status {status!r}.")

        output_text = getattr(response, "output_text", "") or ""
        if not output_text.strip():
            raise GPTInvocationError("GPT response contained no text.")

        usage = getattr(response, "usage", None)
        output_details = getattr(usage, "output_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = (
            getattr(output_details, "reasoning_tokens", None)
            or getattr(completion_details, "reasoning_tokens", None)
            or getattr(usage, "reasoning_tokens", 0)
            or 0
        )
        return output_text, reasoning_tokens

    try:
        return await request()
    except GPTInvocationError:
        raise
    except Exception as exc:
        raise GPTInvocationError(f"Error invoking OpenAI API: {exc}") from exc


async def invoke_gemini(
    model: str, question: str, history: Sequence[HistoryItem]
) -> Tuple[str, int]:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY/GOOGLE_API_KEY not found.", 0

    contents = []
    for prior_question, prior_answer in history:
        contents.append(
            types.Content(role="user", parts=[types.Part(text=prior_question)])
        )
        contents.append(
            types.Content(role="model", parts=[types.Part(text=prior_answer)])
        )
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH)
    )

    async def request() -> Tuple[str, int]:
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )
        usage = getattr(response, "usage_metadata", None)
        return response.text, getattr(usage, "thoughts_token_count", 0) or 0

    try:
        return await request()
    except Exception as exc:
        return f"Error invoking Gemini: {exc}", 0


async def invoke_grok(
    model: str, question: str, history: Sequence[HistoryItem]
) -> Tuple[str, int]:
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        return "Error: XAI_API_KEY not found.", 0

    async def request() -> Tuple[str, int]:
        async with AsyncClient(api_key=api_key) as client:
            chat = client.chat.create(model=model, tools=[], reasoning_effort="high")
            for prior_question, prior_answer in history:
                chat.append(user(prior_question))
                chat.append(assistant(prior_answer))
            chat.append(user(question))
            response = await chat.sample()
        content = response.content.strip() if response.content else ""
        return (
            content,
            getattr(getattr(response, "usage", None), "reasoning_tokens", 0) or 0,
        )

    try:
        return await request()
    except Exception as exc:
        return f"Error invoking Grok: {exc}", 0


async def invoke_claude(
    model: str, question: str, history: Sequence[HistoryItem]
) -> Tuple[str, int]:
    messages = []
    for prior_question, prior_answer in history:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})

    async def request() -> Tuple[str, int]:
        client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""), max_retries=0
        )
        response = await client.messages.create(
            model=model,
            max_tokens=16384,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "xhigh"},
        )
        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        return text, getattr(response.usage, "output_tokens", 0)

    try:
        return await request()
    except Exception as exc:
        return f"Error invoking Claude: {exc}", 0


async def call_llm(
    question: str, models: Models, history: Sequence[HistoryItem] = ()
) -> str:
    """Ask the full model panel concurrently and return GPT's merged response."""
    gpt, gemini, claude, grok = await asyncio.gather(
        invoke_gpt(models.gpt, question, history),
        invoke_gemini(models.gemini, question, history),
        invoke_claude(models.claude, question, history),
        invoke_grok(models.grok, question, history),
    )
    response_blocks = [
        f'''<response model="gpt">
{gpt[0]}
</response>''',
        f'''<response model="gemini">
{gemini[0]}
</response>''',
        f'''<response model="claude">
{claude[0]}
</response>''',
        f'''<response model="grok">
{grok[0]}
</response>''',
    ]
    joined_responses = "\n".join(response_blocks)
    prompt = f'''<prompt>
{question}
</prompt>
{joined_responses}
Merge the responses into a coherent one. List any major conflicts.
'''
    text, _ = await invoke_gpt(models.gpt, prompt, history)
    return text


async def process_batch(batch: Sequence[str], batch_number: int, models: Models) -> str:
    initial_question = "\n".join(batch) + "\nWhat are their moats?"
    progress(f"Batch {batch_number}: first-pass moat analysis")
    first_synthesis = await call_llm(initial_question, models)

    # In pareja.py, the next turn's history contains the prior user question and
    # GPT's displayed synthesis. Give every model that same shared history.
    # The requested next user turn is deliberately just these two words.
    deeper_history = ((initial_question, first_synthesis),)
    progress(f"Batch {batch_number}: deeper analysis")
    return await call_llm("think deeper", models, deeper_history)


def read_records() -> List[str]:
    records = []
    for line_number, raw_line in enumerate(sys.stdin, start=1):
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        try:
            fields = next(csv.reader([line], strict=True))
        except csv.Error as exc:
            raise ValueError(f"Invalid CSV on input line {line_number}: {exc}") from exc
        if len(fields) < 2 or not fields[0].strip() or not fields[1].strip():
            raise ValueError(
                f"Input line {line_number} must contain a ticker and company name."
            )
        # Preserve the supplied CSV spelling (including quoted company names) in prompts.
        records.append(line)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank stock moats from CSV records supplied on standard input."
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Records per batch (default: 4)."
    )
    parser.add_argument("--gpt-model", default=MODEL_GPT_BASE)
    parser.add_argument("--gemini-model", default=MODEL_GEMINI)
    parser.add_argument("--grok-model", default=MODEL_GROK_BASE)
    parser.add_argument("--claude-model", default=MODEL_CLAUDE)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> str:
    if args.batch_size != 4:
        raise ValueError("batch size must be 4 for this workflow")

    records = read_records()
    if not records:
        raise ValueError("No non-empty CSV records were supplied on standard input.")

    models = Models(
        gpt=args.gpt_model,
        gemini=args.gemini_model,
        grok=args.grok_model,
        claude=args.claude_model,
    )
    batches = [
        records[index : index + args.batch_size]
        for index in range(0, len(records), args.batch_size)
    ]
    batch_semaphore = asyncio.Semaphore(4)

    async def process_batch_with_limit(batch_number: int, batch: Sequence[str]) -> str:
        async with batch_semaphore:
            return await process_batch(batch, batch_number, models)

    batch_finals = await asyncio.gather(
        *(
            process_batch_with_limit(batch_number, batch)
            for batch_number, batch in enumerate(batches, start=1)
        )
    )

    ranking_question = (
        "<context>\n"
        + "\n".join(batch_finals)
        + "\n</context>\nRank them into 4 tiers by moat width"
    )
    progress("Final ranking: sending all batch findings to the model panel")
    return await call_llm(ranking_question, models)


def main() -> int:
    load_dotenv()
    args = parse_args()
    try:
        print(asyncio.run(run(args)))
    except (ValueError, csv.Error, GPTInvocationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

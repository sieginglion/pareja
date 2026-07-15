"""Decorator for turning typed Pydantic function stubs into LLM calls."""

from __future__ import annotations

import inspect
import os
from functools import wraps
from typing import Awaitable, Callable, ParamSpec, TypeVar, get_type_hints

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

P = ParamSpec("P")
T = TypeVar("T", bound=BaseModel)


def _pydantic_model(value: object) -> bool:
    return inspect.isclass(value) and issubclass(value, BaseModel)


def llm(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    docstring = inspect.getdoc(func)
    if not docstring:
        raise ValueError(f"{func.__name__} needs a docstring.")

    params = list(signature.parameters.values())
    if len(params) != 1:
        raise TypeError(f"{func.__name__} must take exactly one argument.")

    input_param = params[0]
    input_model = type_hints.get(input_param.name)
    output_model = type_hints.get("return")
    if not _pydantic_model(input_model) or not _pydantic_model(output_model):
        raise TypeError(f"{func.__name__} must use Pydantic input and output types.")

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        bound = signature.bind(*args, **kwargs)
        input_object = input_model.model_validate(bound.arguments[input_param.name])

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required.")

        client = AsyncOpenAI(api_key=api_key)
        response = await client.responses.parse(
            model="gpt-5.5",
            input=[
                {
                    "role": "user",
                    "content": (
                        f"{inspect.cleandoc(docstring)}\n"
                        f"Input:\n{input_object.model_dump_json(indent=2)}"
                    ),
                }
            ],
            reasoning={"effort": "medium"},
            tools=[{"type": "web_search"}],
            text_format=output_model,
        )

        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI response did not parse.")

        return parsed

    return wrapper


class SummarizeInput(BaseModel):
    text: str = Field(description="Text to summarize.")


class SummarizeOutput(BaseModel):
    summary: str = Field(description="Concise summary of the input text.")


@llm
async def summarize(item: SummarizeInput) -> SummarizeOutput:
    """Summarize the input text concisely."""

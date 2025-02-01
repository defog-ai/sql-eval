from typing import Callable
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

LLM_COSTS_PER_TOKEN = {
    "gpt-4o": {"input_cost_per1k": 0.0025, "output_cost_per1k": 0.01},
    "gpt-4o-mini": {"input_cost_per1k": 0.00015, "output_cost_per1k": 0.0006},
    "o1": {"input_cost_per1k": 0.015, "output_cost_per1k": 0.06},
    "o1-preview": {"input_cost_per1k": 0.015, "output_cost_per1k": 0.06},
    "o1-mini": {"input_cost_per1k": 0.003, "output_cost_per1k": 0.012},
    "o3-mini": {"input_cost_per1k": 0.0011, "output_cost_per1k": 0.0044},
    "gpt-4-turbo": {"input_cost_per1k": 0.01, "output_cost_per1k": 0.03},
    "gpt-3.5-turbo": {"input_cost_per1k": 0.0005, "output_cost_per1k": 0.0015},
    "claude-3-5-sonnet": {"input_cost_per1k": 0.003, "output_cost_per1k": 0.015},
    "claude-3-5-haiku": {"input_cost_per1k": 0.00025, "output_cost_per1k": 0.00125},
    "claude-3-opus": {"input_cost_per1k": 0.015, "output_cost_per1k": 0.075},
    "claude-3-sonnet": {"input_cost_per1k": 0.003, "output_cost_per1k": 0.015},
    "claude-3-haiku": {"input_cost_per1k": 0.00025, "output_cost_per1k": 0.00125},
    "gemini-1.5-pro": {"input_cost_per1k": 0.00125, "output_cost_per1k": 0.005},
    "gemini-1.5-flash": {"input_cost_per1k": 0.000075, "output_cost_per1k": 0.0003},
    "gemini-1.5-flash-8b": {
        "input_cost_per1k": 0.0000375,
        "output_cost_per1k": 0.00015,
    },
    "gemini-2.0-flash": {
        "input_cost_per1k": 0.000075,
        "output_cost_per1k": 0.0003,
    },
}


@dataclass
class LLMResponse:
    content: Any
    model: str
    time: float
    input_tokens: int
    output_tokens: int
    output_tokens_details: Optional[Dict[str, int]] = None
    cost: Optional[float] = None

    def __post_init__(self):
        if self.model in LLM_COSTS_PER_TOKEN:
            model_name = self.model
        else:
            model_name = None
            potential_model_names = []

            for mname in LLM_COSTS_PER_TOKEN.keys():
                if mname in self.model:
                    potential_model_names.append(mname)

            if len(potential_model_names) > 0:
                model_name = max(potential_model_names, key=len)

        if model_name:
            self.cost = (
                self.input_tokens
                / 1000
                * LLM_COSTS_PER_TOKEN[model_name]["input_cost_per1k"]
                + self.output_tokens
                / 1000
                * LLM_COSTS_PER_TOKEN[model_name]["output_cost_per1k"]
            )


def chat_anthropic(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
    timeout=100,
) -> LLMResponse:
    """
    Returns the response from the Anthropic API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    Note that anthropic doesn't have explicit json mode api constraints, nor does it have a seed parameter.
    """
    from anthropic import Anthropic

    client_anthropic = Anthropic()
    t = time.time()
    if len(messages) >= 1 and messages[0].get("role") == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        sys_msg = ""
    response = client_anthropic.messages.create(
        system=sys_msg,
        messages=messages,
        model=model,
        max_tokens=max_completion_tokens,
        temperature=temperature,
        stop_sequences=stop,
        timeout=timeout,
    )
    if response.stop_reason == "max_tokens":
        raise Exception("Max tokens reached")
    if len(response.content) == 0:
        raise Exception("Max tokens reached")
    return LLMResponse(
        model=model,
        content=response.content[0].text,
        time=round(time.time() - t, 3),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def chat_openai(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    max_completion_tokens: int = 16384,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
    timeout=100,
) -> LLMResponse:
    """
    Returns the response from the OpenAI API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    We use max_completion_tokens here, instead of using max_tokens. This is to support o1 models.
    """
    from openai import OpenAI

    client_openai = OpenAI()
    t = time.time()
    if model.startswith("o"):
        if messages[0].get("role") == "system":
            sys_msg = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = sys_msg + messages[0]["content"]

        response = client_openai.chat.completions.create(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            store=store,
            metadata=metadata,
            timeout=timeout,
        )
    else:
        if response_format or json_mode:
            response = client_openai.beta.chat.completions.parse(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stop=stop,
                response_format=(
                    {"type": "json_object"} if json_mode else response_format
                ),
                seed=seed,
                store=store,
                metadata=metadata,
            )
        else:
            response = client_openai.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stop=stop,
                seed=seed,
                store=store,
                metadata=metadata,
            )

    if response_format and not model.startswith("o1"):
        content = response.choices[0].message.parsed
    else:
        content = response.choices[0].message.content

    if response.choices[0].finish_reason == "length":
        print("Max tokens reached")
        raise Exception("Max tokens reached")
    if len(response.choices) == 0:
        print("Empty response")
        raise Exception("No response")
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        output_tokens_details=response.usage.completion_tokens_details,
    )


def chat_gemini(
    messages: List[Dict[str, str]],
    model: str = "gemini-2.0-flash-exp",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
    timeout=100,  # does not have timeout method
) -> LLMResponse:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    t = time.time()
    if messages[0]["role"] == "system":
        system_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        system_msg = None

    message = "\n".join([i["content"] for i in messages])

    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_msg,
        max_output_tokens=max_completion_tokens,
        stop_sequences=stop,
    )

    if response_format:
        # use Pydantic classes for response_format
        generation_config.response_mime_type = "application/json"
        generation_config.response_schema = response_format

    try:
        response = client.models.generate_content(
            model=model,
            contents=message,
            config=generation_config,
        )
        content = response.text
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

    if response_format:
        # convert the content into Pydantic class
        content = response_format.parse_raw(content)

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count,
    )


def map_model_to_chat_fn(model: str) -> Callable:
    """
    Returns the appropriate chat function based on the model.
    """
    if model.startswith("claude"):
        return chat_anthropic
    if model.startswith("gemini"):
        return chat_gemini
    if model.startswith("gpt") or model.startswith("o1"):
        return chat_openai
    raise ValueError(f"Unknown model: {model}")


async def chat(
    model,
    messages,
    max_completion_tokens=4096,
    temperature=0.0,
    stop=[],
    json_mode=False,
    response_format=None,
    seed=0,
    store=True,
    metadata=None,
    timeout=100,  # in seconds
) -> LLMResponse:
    """
    Returns the response from the LLM API for a single model that is passed in.
    Includes retry logic with exponential backoff for up to 3 attempts.
    """
    llm_function = map_model_to_chat_fn(model)
    max_retries = 3
    base_delay = 1  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            return llm_function(
                model=model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stop=stop,
                json_mode=json_mode,
                response_format=response_format,
                seed=seed,
                store=store,
                metadata=metadata,
                timeout=timeout,
            )
        except Exception as e:
            delay = base_delay * (2**attempt)  # Exponential backoff
            print(
                f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...",
                flush=True,
            )
            print(f"Error: {e}", flush=True)
            time.sleep(delay)

    # If we get here, all attempts failed
    raise Exception("All attempts at calling the chat function failed")

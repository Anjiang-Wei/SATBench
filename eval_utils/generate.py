# from litellm import completion
import anthropic
import together
import openai

from eval_utils.type import EvalInput
from eval_utils.utils import log_error_wrapper,  handle_error_openai, handle_error_anthropic, handle_error_together

async def llm_generate(input: EvalInput) -> str:
    match input.model_platform:
        case "openai":
            return await generate_openai(input=input)
        case "anthropic":
            return await generate_anthropic(input=input)
        case _:
            return await generate_together(input=input)

@handle_error_openai
@log_error_wrapper
async def generate_openai(input: EvalInput) -> str:
    client = openai.AsyncClient()
    chat_completion = await client.chat.completions.create(
        messages = input.messages,
        model    = input.model_name,
        # timeout  = 60 * 5
    )
    return chat_completion.choices[0].message.content

@handle_error_anthropic
@log_error_wrapper
async def generate_anthropic(input: EvalInput) -> str:
    client = anthropic.AsyncClient()
    message = await client.messages.create(
        max_tokens = 8192*2,
        model      = input.model_name,
        messages   = input.messages
    )
    return "".join([block.text for block in message.content])

@handle_error_together
@log_error_wrapper
async def generate_together(input: EvalInput) -> str:
    #print(input)
    client = together.AsyncClient()
    chat_completion = await client.chat.completions.create(
        max_tokens=8192*2,
        messages=input.messages,
        model   =input.model_name
    )
    return chat_completion.choices[0].message.content

from pydantic import BaseModel, ValidationError

from teacherlm_core.llm.ollama_client import OllamaClient


async def generate_structured[T: BaseModel](
    client: OllamaClient,
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2,
    options: dict | None = None,
) -> T:
    """Call the model with a schema-constrained format and parse the reply.

    On validation failure, retries up to `max_retries` times, feeding the
    previous malformed output and the validation error back to the model.
    `options` (e.g. {"temperature": 0.2}) is forwarded to ollama on every
    attempt — useful when the caller needs a colder/hotter sampling than
    the model's default.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    last_content: str | None = None

    for attempt in range(max_retries + 1):
        try:
            return await client.chat_structured(
                messages=messages, schema=schema, options=options
            )
        except (ValidationError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            repair_prompt = _build_repair_prompt(last_content, exc)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "user", "content": repair_prompt},
            ]

    raise RuntimeError(
        f"Failed to produce valid {schema.__name__} after {max_retries + 1} attempts"
    ) from last_error


def _build_repair_prompt(previous: str | None, error: Exception) -> str:
    prev_block = f"\nPrevious output:\n{previous}\n" if previous else ""
    return (
        "Your previous response did not match the required JSON schema."
        f"{prev_block}"
        f"\nValidation error:\n{error}\n"
        "\nReturn ONLY valid JSON conforming to the schema — no prose, no code fences."
    )

import httpx
import llm
from pydantic import Field


@llm.hookimpl
def register_models(register):
    register(Huggingface())


class Huggingface(llm.Model):
    model_id = "huggingface"

    class Options(llm.Options):
        max_new_tokens: int = Field(
            default=60,
            ge=0,
            le=250,
            description="Maximum number of tokens to generate.",
        )
        temperature: float = Field(
            default=0.2,
            gt=0.0,
            le=100.0,
            description="Randomness in the response.",
        )
        top_p: float = Field(
            default=0.95,
            gt=0.0,
            lt=1.0,
            description="Nucleus sampling threshold.",
        )

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "huggingface", "LLM_HUGGINGFACE_KEY")
        headers = {"Authorization": f"Bearer {key}"}
        url = "https://api-inference.huggingface.co/models/bigcode/starcoder2-15b"
        inputs = prompt.prompt
        parameters = {
            "max_new_tokens": prompt.options.max_new_tokens,
            "temperature": prompt.options.temperature,
            "top_p": prompt.options.top_p,
        }
        body= {
            "inputs": inputs,
            "parameters": parameters,
        }
        with httpx.Client() as client:
            api_response = client.post(
                url,
                headers=headers,
                json=body,
            )
            api_response.raise_for_status()
            yield api_response.json()[0]["generated_text"]
            response.response_json = api_response.json()

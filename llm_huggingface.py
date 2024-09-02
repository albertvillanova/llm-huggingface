import httpx
import llm


@llm.hookimpl
def register_models(register):
    register(Huggingface())


class Huggingface(llm.Model):
    model_id = "huggingface"

    class Options(llm.Options):
        max_new_tokens: int = 60
        temperature: float = 0.2
        top_p: float = 0.95

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

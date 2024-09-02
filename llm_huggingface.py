import httpx
import llm


@llm.hookimpl
def register_models(register):
    register(Huggingface())


class Huggingface(llm.Model):
    model_id = "huggingface"

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "huggingface", "LLM_HUGGINGFACE_KEY")
        headers = {"Authorization": f"Bearer {key}"}
        url = "https://api-inference.huggingface.co/models/bigcode/starcoder2-15b"
        body= {
            "inputs": prompt.prompt,
            "parameters": {
                "max_new_tokens": 60,
                "temperature": 0.2,
                "top_p": 0.95,
            },
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

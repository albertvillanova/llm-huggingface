import llm


@llm.hookimpl
def register_models(register):
    register(Huggingface())


class Huggingface(llm.Model):
    model_id = "huggingface"

    def execute(self, prompt, stream, response, conversation):
        return ["hello world"]

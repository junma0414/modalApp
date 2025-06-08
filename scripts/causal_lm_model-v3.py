
import modal

    
app = modal.App("causal_lm_model-v3")

volume = modal.Volume.from_name("llm-models")
image = modal.Image.debian_slim().pip_install(
    "torch>=2.0.0",
    "transformers[torch]",
    "accelerate>=0.21.0",
    "fastapi",
    "pydantic"
)


@app.function(image=image, volumes={"/model": volume}, gpu="A10G", timeout=600)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import Union, List, Optional
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = "/model/causal_lm_model v3"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")


    web_app = FastAPI()

    class InputRequest(BaseModel):
        input_text: Union[str, List[str], dict, List[dict]]
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_k: Optional[int] = None
        top_p: Optional[float] = None
        do_sample: Optional[bool] = None
        return_probs: Optional[bool] = None

    @web_app.post("/inference")
    def predict(req: InputRequest):
        
        texts = req.input_text if isinstance(req.input_text, list) else [req.input_text]
        generation_config = {
            "max_new_tokens": req.max_tokens or 100,
               
"temperature": req.temperature or 0.7,
    "top_k": req.top_k or 50,
    "top_p": req.top_p or 0.9,
    "do_sample": req.do_sample if req.do_sample is not None else True,
        }
        responses = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, **generation_config)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(decoded)
        return {"responses": responses}
    

    return web_app

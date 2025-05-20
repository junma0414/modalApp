
import modal

    
app = modal.App("distill_model_v3")

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
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_path = "/model/distill_model - 副本"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")


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
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        outputs = model(**inputs)
        
        if req.return_probs:
            probs = torch.softmax(outputs.logits, dim=-1)
            return {"predictions": probs.tolist()}
    
        predictions = outputs.logits.argmax(dim=-1).tolist()
        return {"predictions": predictions}
    

    return web_app

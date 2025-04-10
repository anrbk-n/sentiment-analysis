from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import os

model_name = "anrbk/sent-analysis"  
try:
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval() 
except Exception as e:
    print(f"Error loading the model: {e}")

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class PredictionInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    label: str
    confidence: float

@app.get("/", response_class=HTMLResponse)
async def render_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_model=PredictionOutput)
async def predict(input: PredictionInput):
    try:
        inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction_idx].item()

        labels = ["positive", "negative", "neutral"]  
        predicted_label = labels[prediction_idx]

        return PredictionOutput(label=predicted_label, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

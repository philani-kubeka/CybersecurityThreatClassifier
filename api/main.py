from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import Predictor
app = FastAPI(title="Cyber Threat Classifier", version="1.0")
predictor = Predictor(Path(__file__).resolve().parent.parent / "artifacts")
class PredictRequest(BaseModel):
    text: str
    entity: Optional[str] = None # exact words from the text; required if the model uses spans
    explain: bool = True
@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/info")
def info():
    return {"classes": predictor.classes, "uses_entity_span": predictor.use_span}
@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    s = e = None
    if req.entity:
        s = req.text.find(req.entity)
        if s < 0:
            raise HTTPException(400, "entity was not found inside the text")
        e = s + len(req.entity)
    elif predictor.use_span:
        raise HTTPException(400, "this model classifies an entity: fill in the entity field")
    return predictor.predict(req.text, s, e, req.explain)

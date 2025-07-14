# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from EcommerceClassifier.model import load_model, predict
import uvicorn

app = FastAPI()
model = load_model()

class Item(BaseModel):
    title: str

@app.post("/predict")
def classify(item: Item):
    result = predict(model, None, item.title)  # if you ignore image
    return {"result": result}

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

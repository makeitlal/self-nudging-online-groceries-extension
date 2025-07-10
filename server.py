from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from EcommerceClassifier.model import load_model, predict
import uvicorn
import tempfile
import requests
from PIL import Image
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow your extension to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, restrict this in production!
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Load model once on startup
model = load_model()

class PredictRequest(BaseModel):
    title: str
    image_url: str = None  # optional

def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

@app.post("/predict")
def predict_api(req: PredictRequest):
    print("Received request:", req)
    # Download image if URL provided, else use a blank white image (fallback)
    if req.image_url:
        image_path = download_image(req.image_url)
        if image_path is None:
            raise HTTPException(status_code=400, detail="Failed to download image")
    else:
        # create a blank white image file for dummy input (224x224)
        blank_path = "blank.jpg"
        if not os.path.exists(blank_path):
            Image.new('RGB', (224, 224), color='white').save(blank_path)
        image_path = blank_path

    label, score = predict(model, image_path, req.title)

    # Cleanup downloaded image file if needed
    if req.image_url and image_path and os.path.exists(image_path):
        os.unlink(image_path)

    return {"label": label, "score": score}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

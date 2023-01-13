# fastpi app for the kitchenware image classification

from fastapi import FastAPI
from pydantic import BaseModel
from train import predict_single

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

app = FastAPI()

class ImageDetails(BaseModel):
    image : str

    class Config:
        schema_extra = {
            "example": {
                "image": "https://www.baladeo.com/medias/produits/1630258919/1710_1280-security-kinfe-emergency-yellow.jpg"
            }
        }

@app.on_event("startup")
async def startup_event():
    from tensorflow import keras
    global model
    logging.info("Starting up...")
    logging.info("Loading model...")
    model = keras.models.load_model('./kitchenware-classification/models/model_kaggle.h5')


@app.get("/")
async def root():
    return {"message": "Welcome to the mlzoomcamp-2022-capstone-project-2 kitchenware classification API"}

@app.post("/predict")
async def predict(image: ImageDetails):
    logging.info(f"Predicting image: {image.image}")
    return predict_single(model, image.image, web=True)




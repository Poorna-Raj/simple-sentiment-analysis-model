from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()

class Sentiment(BaseModel):
    message:str
    expected_result:bool
    result:bool = False

model = joblib.load("../model/model.pkl")

@app.post("/Sentiment/")
async def predict(sentiment:Sentiment):
    prediction = model.predict([sentiment.message])[0]
    sentiment.result = bool(prediction)
    return sentiment
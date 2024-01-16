import pickle

with open("spam_classifier.pickle", 'rb') as file:
    model = pickle.load(file)

from fastapi import FastAPI

from pydantic import BaseModel
class Data(BaseModel):
    email: str

app = FastAPI()

@app.get("/")
def home():
    return {"msg": "Spam email Classifier API is working"}

@app.post("/classify")
def classify(item: Data):
    email = item.email
    y_pred =model.predict([email])
    return {'label': y_pred[0]}
from fastapi import FastAPI
from pydantic import BaseModel
from utils import mask_pii
import joblib

model = joblib.load("saved_model/email_classifier.pkl")
app = FastAPI()

class EmailRequest(BaseModel):
    email_body: str

@app.post("/")
def classify_email(request: EmailRequest):
    original_email = request.email_body
    masked_email, masked_entities = mask_pii(original_email)
    category = model.predict([masked_email])[0]

    return {
        "input_email_body": original_email,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
from models import train_classifier
import uvicorn

# Train model if needed
train_classifier("data/combined_emails_with_natural_pii.csv")

# Launch the API
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
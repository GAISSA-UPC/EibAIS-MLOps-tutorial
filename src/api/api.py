from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import ValidationError
from transformers import pipeline

from src.api.schemas import PredictionRequest, PredictionResponse
from src.config import MODELS_DIR

# Create a FastAPI instance
app = FastAPI(title="IMDB Reviews API", version="1.0.0")

# Load the trained model and tokenizer from local directory
pipeline = pipeline(task="text-classification", model=MODELS_DIR / "distilbert-imdb")
# Load a Hugging Face model and tokenizer
# pipe = pipeline("zero-shot-classification", model="sileod/deberta-v3-base-tasksource-nli")


# Root route to return basic information
@app.get("/")
def root():
    """
    Root endpoint that returns a welcome message.

    Returns
    -------
        dict: A welcome message with information on the app's purpose.
    """
    return {"message": "Welcome to the IMDB reviews app!"}


@app.post("/prediction", response_model=list[PredictionResponse])
def predict_sentiment(requests: PredictionRequest) -> list[PredictionResponse]:
    """
    Predict the sentiment of a single or multiple reviews.

    This endpoint accepts a single review or a list of reviews and returns the predicted label
    and confidence score for each review.
    The input review(s) are validated using Pydantic models, and the predictions are made using
    a pre-trained model loaded from the specified directory.

    Parameters
    ----------
    requests : PredictRequest
        The input request containing the review(s) to be predicted.
        The request is validated using Pydantic models to ensure the input format is correct.

    Returns
    -------
    list[PredictResponse]
        A list of PredictResponse objects containing the review text, predicted label, and
        confidence score for each review.

    Raises
    ------
    HTTPException
        If the input review(s) exceed the maximum length or if any other validation error occurs,
        a 400 Bad Request error is raised with a detailed message.
        If an unexpected error occurs during prediction, a 500 Internal Server Error is raised
        with a detailed message.
    """
    try:
        reviews = [review.review for review in requests.reviews]
        labeled_reviews = pipeline(reviews)
        return [
            PredictionResponse(review=review, label=out["label"], score=out["score"])
            for review, out in zip(reviews, labeled_reviews, strict=False)
        ]
    except Exception as exception:
        # Log the exception and return a 500 error
        logger.error(f"Unexpected error: {str(exception)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exception)}") from exception

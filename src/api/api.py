from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from transformers import pipeline
from loguru import logger

from src.api.schemas import PredictRequest, PredictResponse
from src.config import MODELS_DIR

# Create a FastAPI instance
app = FastAPI()

# Load the trained model and tokenizer
pipeline = pipeline(task="text-classification", model=MODELS_DIR / "distilbert-imdb")


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


@app.post("/predict", response_model=list[PredictResponse])
def predict_sentiment(requests: PredictRequest | list[PredictRequest]) -> list[PredictResponse]:
    """
    Predict the sentiment of a single or multiple reviews.

    This endpoint accepts a single review or a list of reviews and returns the predicted label
    and confidence score for each review.
    The input review(s) are validated using Pydantic models, and the predictions are made using
    a pre-trained model loaded from the specified directory.

    Parameters
    ----------
        requests (PredictRequest | list[PredictRequest]): A single or a list of Pydantic models
            containing the review text(s) to be classified.

    Returns
    -------
        list[PredictResponse]: A list of Pydantic models containing the review text, predicted label,
            and confidence score for each review.
    Raises
    ------
        HTTPException: If validation fails or an unexpected error occurs during processing.
    """
    labeled_reviews = []
    try:
        if isinstance(requests, PredictRequest):
            out = pipeline(requests.review)[0]
            return [PredictResponse(review=requests.review, label=out["label"], score=out["score"])]

        labeled_reviews = []
        for request in requests:
            out = pipeline(request.review)[0]
            labeled_reviews.append(PredictResponse(review=request.review, label=out["label"], score=out["score"]))

        return labeled_reviews

    except ValidationError as exception:
        logger.error(f"Validation error: {str(exception)}")
        raise HTTPException(status_code=400, detail=f"Validation Error: {str(exception)}")
    except Exception as exception:
        # Log the exception and return a 500 error
        logger.error(f"Unexpected error: {str(exception)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exception)}")

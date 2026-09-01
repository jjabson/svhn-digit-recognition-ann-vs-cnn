from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    """
    API response returned after classifying an SVHN digit image.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_digit": 7,
                "confidence": 0.9842,
                "confidence_percent": "98.42%",
                "probabilities": {
                    "0": 0.0011,
                    "1": 0.0004,
                    "2": 0.0020,
                    "3": 0.0007,
                    "4": 0.0015,
                    "5": 0.0003,
                    "6": 0.0010,
                    "7": 0.9842,
                    "8": 0.0041,
                    "9": 0.0047,
                },
            }
        }
    )

    predicted_digit: int
    confidence: float
    confidence_percent: str
    probabilities: dict[str, float]
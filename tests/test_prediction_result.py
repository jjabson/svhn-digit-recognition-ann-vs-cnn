from src.schemas.prediction import PredictionResult


def test_prediction_result_formats_confidence_percent():
    result = PredictionResult(
        predicted_digit=8,
        confidence=0.818678081035614,
        probabilities={
            "8": 0.818678081035614,
        },
    )

    assert result.predicted_digit == 8
    assert result.confidence == 0.818678081035614
    assert result.probabilities["8"] == 0.818678081035614
    assert result.confidence_percent == "81.87%"
from __future__ import annotations

import os
from collections import defaultdict

from pymongo import MongoClient
from pymongo.server_api import ServerApi


MONGODB_URI_ENV = "SVHN_MONGODB_URI"

DATABASE_NAME = "svhn_ml"
COLLECTION_NAME = "predictions"
MODEL_VERSION = "cnn_v2"


def get_mongodb_uri() -> str:
    uri = os.environ.get(
        MONGODB_URI_ENV,
        "",
    ).strip()

    if not uri:
        raise RuntimeError(
            f"{MONGODB_URI_ENV} is not configured."
        )

    return uri


def main() -> None:
    uri = get_mongodb_uri()

    with MongoClient(
        uri,
        server_api=ServerApi("1"),
        serverSelectionTimeoutMS=5000,
    ) as client:
        client.admin.command("ping")

        predictions = (
            client[DATABASE_NAME][COLLECTION_NAME]
        )

        documents = list(
            predictions.find(
                {
                    "model_version": MODEL_VERSION
                },
                {
                    "_id": 0,
                    "true_label": 1,
                    "predicted_label": 1,
                    "confidence": 1,
                },
            )
        )

    grouped = defaultdict(list)

    for document in documents:
        grouped[
            document["true_label"]
        ].append(document)

    print("Python Validation")
    print("-----------------")

    total_correct = 0

    for digit in sorted(grouped):
        digit_documents = grouped[digit]

        total = len(digit_documents)

        correct = sum(
            document["true_label"]
            == document["predicted_label"]
            for document in digit_documents
        )

        avg_confidence = sum(
            document["confidence"]
            for document in digit_documents
        ) / total

        accuracy = correct / total

        total_correct += correct

        print(
            f"Digit {digit}: "
            f"total={total}, "
            f"correct={correct}, "
            f"accuracy={accuracy:.4f}, "
            f"avg_confidence={avg_confidence:.4f}"
        )

    overall_accuracy = (
        total_correct / len(documents)
    )

    print()
    print(
        f"Total documents: {len(documents)}"
    )
    print(
        f"Correct predictions: {total_correct}"
    )
    print(
        f"Overall accuracy: {overall_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
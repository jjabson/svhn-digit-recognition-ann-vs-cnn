from __future__ import annotations

import os

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
    pipeline = [
        {
            "$match": {
                "model_version": MODEL_VERSION
            }
        },
        {
            "$group": {
                "_id": "$true_label",
                "total": {
                    "$sum": 1
                },
                "correct": {
                    "$sum": {
                        "$cond": [
                            {
                                "$eq": [
                                    "$true_label",
                                    "$predicted_label",
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "avg_confidence": {
                    "$avg": "$confidence"
                },
            }
        },
        {
            "$project": {
                "total": 1,
                "correct": 1,
                "avg_confidence": 1,
                "accuracy": {
                    "$divide": [
                        "$correct",
                        "$total",
                    ]
                },
            }
        },
        {
            "$sort": {
                "accuracy": 1
            }
        },
    ]

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

        results = list(
            predictions.aggregate(pipeline)
        )

    print(
        "Per-Digit MongoDB Aggregation"
    )
    print(
        "-----------------------------"
    )

    for result in results:
        digit = result["_id"]
        total = result["total"]
        correct = result["correct"]
        accuracy = result["accuracy"]
        avg_confidence = result["avg_confidence"]

        print(
            f"Digit {digit}: "
            f"total={total}, "
            f"correct={correct}, "
            f"accuracy={accuracy:.4f}, "
            f"avg_confidence={avg_confidence:.4f}"
        )


if __name__ == "__main__":
    main()
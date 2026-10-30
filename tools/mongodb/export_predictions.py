from __future__ import annotations

import os

from pymongo import MongoClient
from pymongo.server_api import ServerApi

from tools.mongodb.prediction_documents import (
    DEFAULT_EVALUATION_PROTOCOL,
    DEFAULT_MODEL_VERSION,
    build_balanced_prediction_documents,
    summarize_document_distribution,
)
from tools.reliability.reliability_store import (
    load_reliability_observations,
)


MONGODB_URI_ENV = "SVHN_MONGODB_URI"

DATABASE_NAME = "svhn_ml"
COLLECTION_NAME = "predictions"


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
    observations = load_reliability_observations(
        DEFAULT_EVALUATION_PROTOCOL
    )

    print(
        "Loaded real reliability observations: "
        f"{len(observations)}"
    )

    documents = build_balanced_prediction_documents(
        observations=observations,
        samples_per_digit=50,
        random_seed=42,
        model_version=DEFAULT_MODEL_VERSION,
        evaluation_protocol=(
            DEFAULT_EVALUATION_PROTOCOL
        ),
    )

    distribution = summarize_document_distribution(
        documents
    )

    print(
        "MongoDB prediction documents: "
        f"{len(documents)}"
    )

    print(
        "True-label distribution: "
        f"{distribution}"
    )

    uri = get_mongodb_uri()

    with MongoClient(
        uri,
        server_api=ServerApi("1"),
        serverSelectionTimeoutMS=5000,
    ) as client:
        client.admin.command("ping")

        print(
            "MongoDB Atlas connection: OK"
        )

        predictions = (
            client[DATABASE_NAME][COLLECTION_NAME]
        )

        predictions.delete_many(
            {
                "model_version": (
                    DEFAULT_MODEL_VERSION
                ),
                "evaluation_protocol": (
                    DEFAULT_EVALUATION_PROTOCOL
                ),
            }
        )

        result = predictions.insert_many(
            documents
        )

        print(
            "Inserted documents: "
            f"{len(result.inserted_ids)}"
        )

        stored_count = predictions.count_documents(
            {
                "model_version": (
                    DEFAULT_MODEL_VERSION
                ),
                "evaluation_protocol": (
                    DEFAULT_EVALUATION_PROTOCOL
                ),
            }
        )

        print(
            "Stored matching documents: "
            f"{stored_count}"
        )


if __name__ == "__main__":
    main()
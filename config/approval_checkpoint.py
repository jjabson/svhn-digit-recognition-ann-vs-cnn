from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from config.project_paths import (
    APPROVAL_CHECKPOINT_DATABASE_FILE,
)


def create_approval_checkpointer() -> SqliteSaver:
    APPROVAL_CHECKPOINT_DATABASE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        APPROVAL_CHECKPOINT_DATABASE_FILE,
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(
        connection,
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=None,
        ),
    )

    checkpointer.setup()

    return checkpointer
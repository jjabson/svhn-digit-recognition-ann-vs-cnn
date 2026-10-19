from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from config.approval_checkpoint import create_approval_checkpointer


def test_approval_checkpointer_uses_strict_msgpack_serializer():
    checkpointer = create_approval_checkpointer()

    try:
        assert isinstance(checkpointer.serde, JsonPlusSerializer)
        assert checkpointer.serde._allowed_msgpack_modules is None
        assert checkpointer.serde.pickle_fallback is False
    finally:
        checkpointer.conn.close()
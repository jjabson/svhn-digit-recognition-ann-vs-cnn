import asyncio

import pytest

from mcp_server.http_auth import (
    McpHttpAuthConfig,
    McpHttpAuthConfigurationError,
    StaticBearerTokenVerifier,
    create_auth_settings,
)
from mcp_server.server import (
    validate_http_bind_security,
)


def test_disabled_auth_requires_no_credentials():
    config = McpHttpAuthConfig(
        enabled=False,
    )

    config.validate()


def test_enabled_auth_requires_bearer_token():
    config = McpHttpAuthConfig(
        enabled=True,
        bearer_token=None,
        issuer_url="https://auth.example.test",
        resource_server_url=(
            "http://127.0.0.1:8001/mcp"
        ),
    )

    with pytest.raises(
        McpHttpAuthConfigurationError
    ):
        config.validate()


def test_auth_settings_require_expected_scope():
    config = McpHttpAuthConfig(
        enabled=True,
        bearer_token="secret-token",
        issuer_url="https://auth.example.test",
        resource_server_url=(
            "http://127.0.0.1:8001/mcp"
        ),
    )

    config.validate()

    settings = create_auth_settings(
        config
    )

    assert settings.required_scopes == [
        "svhn:mcp",
    ]

    assert (
        str(settings.resource_server_url)
        == "http://127.0.0.1:8001/mcp"
    )

    assert (
        settings.validate_token_resource
        is True
    )


def test_static_verifier_accepts_valid_token():
    verifier = StaticBearerTokenVerifier(
        expected_token="secret-token",
        resource_server_url=(
            "http://127.0.0.1:8001/mcp"
        ),
        required_scope="svhn:mcp",
    )

    access_token = asyncio.run(
        verifier.verify_token(
            "secret-token"
        )
    )

    assert access_token is not None
    assert (
        access_token.client_id
        == "svhn-mcp-client"
    )
    assert access_token.scopes == [
        "svhn:mcp",
    ]
    assert (
        access_token.resource
        == "http://127.0.0.1:8001/mcp"
    )


def test_static_verifier_rejects_invalid_token():
    verifier = StaticBearerTokenVerifier(
        expected_token="secret-token",
        resource_server_url=(
            "http://127.0.0.1:8001/mcp"
        ),
        required_scope="svhn:mcp",
    )

    access_token = asyncio.run(
        verifier.verify_token(
            "wrong-token"
        )
    )

    assert access_token is None


@pytest.mark.parametrize(
    ("host", "auth_enabled"),
    [
        ("127.0.0.1", False),
        ("localhost", False),
        ("::1", False),
        ("0.0.0.0", True),
    ],
)
def test_http_bind_security_allows_safe_modes(
    host,
    auth_enabled,
):
    validate_http_bind_security(
        host=host,
        auth_enabled=auth_enabled,
    )


def test_http_bind_security_rejects_unauthenticated_network_bind():
    with pytest.raises(
        ValueError,
        match="Refusing to start unauthenticated",
    ):
        validate_http_bind_security(
            host="0.0.0.0",
            auth_enabled=False,
        )
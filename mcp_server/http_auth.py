from __future__ import annotations

from dataclasses import dataclass, field
import os
import secrets

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl


AUTH_ENABLED_ENV = "SVHN_MCP_HTTP_AUTH_ENABLED"
BEARER_TOKEN_ENV = "SVHN_MCP_BEARER_TOKEN"
AUTH_ISSUER_URL_ENV = "SVHN_MCP_AUTH_ISSUER_URL"
RESOURCE_SERVER_URL_ENV = "SVHN_MCP_RESOURCE_SERVER_URL"

DEFAULT_REQUIRED_SCOPE = "svhn:mcp"


class McpHttpAuthConfigurationError(ValueError):
    """Raised when MCP HTTP authentication is configured incorrectly."""


@dataclass(frozen=True)
class McpHttpAuthConfig:
    enabled: bool
    bearer_token: str | None = field(
        default=None,
        repr=False,
    )
    issuer_url: str | None = None
    resource_server_url: str | None = None
    required_scope: str = DEFAULT_REQUIRED_SCOPE

    @classmethod
    def from_environment(cls) -> "McpHttpAuthConfig":
        enabled = (
            os.getenv(AUTH_ENABLED_ENV, "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )

        config = cls(
            enabled=enabled,
            bearer_token=os.getenv(BEARER_TOKEN_ENV),
            issuer_url=os.getenv(AUTH_ISSUER_URL_ENV),
            resource_server_url=os.getenv(
                RESOURCE_SERVER_URL_ENV
            ),
        )

        config.validate()
        return config

    def validate(self) -> None:
        if not self.enabled:
            return

        if not self.bearer_token:
            raise McpHttpAuthConfigurationError(
                f"{BEARER_TOKEN_ENV} is required when "
                f"{AUTH_ENABLED_ENV}=true."
            )

        if not self.issuer_url:
            raise McpHttpAuthConfigurationError(
                f"{AUTH_ISSUER_URL_ENV} is required when "
                f"{AUTH_ENABLED_ENV}=true."
            )

        if not self.resource_server_url:
            raise McpHttpAuthConfigurationError(
                f"{RESOURCE_SERVER_URL_ENV} is required when "
                f"{AUTH_ENABLED_ENV}=true."
            )


class StaticBearerTokenVerifier:
    """
    Verify a preconfigured bearer token.

    This verifier establishes the MCP HTTP authentication
    boundary. It does not provide trusted approval authority
    for controlled runtime actions.
    """

    def __init__(
        self,
        *,
        expected_token: str,
        resource_server_url: str,
        required_scope: str,
    ) -> None:
        self._expected_token = expected_token
        self._resource_server_url = (
            resource_server_url
        )
        self._required_scope = required_scope

    async def verify_token(
        self,
        token: str,
    ) -> AccessToken | None:
        if not secrets.compare_digest(
            token,
            self._expected_token,
        ):
            return None

        return AccessToken(
            token=token,
            client_id="svhn-mcp-client",
            scopes=[self._required_scope],
            resource=self._resource_server_url,
        )


def create_auth_settings(
    config: McpHttpAuthConfig,
) -> AuthSettings:
    if not config.enabled:
        raise McpHttpAuthConfigurationError(
            "Cannot create AuthSettings when "
            "HTTP authentication is disabled."
        )

    assert config.issuer_url is not None
    assert config.resource_server_url is not None

    return AuthSettings(
        issuer_url=AnyHttpUrl(
            config.issuer_url
        ),
        resource_server_url=AnyHttpUrl(
            config.resource_server_url
        ),
        required_scopes=[
            config.required_scope,
        ],
        validate_token_resource=True,
    )
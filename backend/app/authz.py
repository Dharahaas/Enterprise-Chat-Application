"""Server-side document access control for retrieval (Phase 4).

Effective permissions are derived **only** from the authenticated session and
from stored document metadata. The frontend is never the security boundary:
filtering happens in the retrieval path, before any chunk text reaches the
prompt, the model, the citations or the streaming response.

What the current Chroma metadata actually represents:

* ``agent``  — the document's department/namespace (also the collection name).
  This is the only access-relevant classification every ingested chunk carries.
* ``visibility`` — **not present on any existing document**. When present it may
  only tighten access: ``public`` stays available to every authenticated
  employee (preserving existing behaviour), ``restricted``/``admin``/``private``
  are administrator-only, and any unrecognised value is refused.
* There is no per-document ownership or per-user grant in the store. Introducing
  one requires a metadata migration, which is deliberately **not** performed
  here.

Fail-closed rules: no session, unknown role, empty permission set, a chunk
without a valid ``agent``, a chunk from a different namespace than the one being
queried, or an unverifiable ``visibility`` marker all deny access.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Mapping

from app.auth import DEMO_ADMIN_USERNAME, AuthenticatedUser
from app.config import settings

PUBLIC_VISIBILITY = "public"
RESTRICTED_VISIBILITIES = frozenset({"restricted", "admin", "private"})
KNOWN_ROLES = frozenset({"employee", "admin", "service"})


@dataclass(frozen=True)
class AccessContext:
    """Effective permissions for the principal handling a request."""

    username: str | None
    role: str | None
    allowed_agents: frozenset[str]
    allow_restricted: bool = False

    @property
    def verified(self) -> bool:
        """False when the principal or its permission set could not be verified."""
        return self.role in KNOWN_ROLES and bool(self.allowed_agents)


_ACCESS_CONTEXT: ContextVar[AccessContext | None] = ContextVar(
    "eka_access_context",
    default=None,
)


def set_access_context(context: AccessContext) -> Token:
    return _ACCESS_CONTEXT.set(context)


def reset_access_context(token: Token) -> None:
    _ACCESS_CONTEXT.reset(token)


def current_access_context() -> AccessContext | None:
    return _ACCESS_CONTEXT.get()


def enabled_agents() -> frozenset[str]:
    return frozenset(
        agent.strip()
        for agent in settings.enabled_agents.split(",")
        if agent.strip()
    )


def access_context_for(user: AuthenticatedUser | None) -> AccessContext:
    """Build the effective permissions for a session user.

    Missing or unrecognised principals fail closed with an empty permission set.
    Every authenticated employee may retrieve from every enabled department —
    that is the product's existing, explicitly company-wide knowledge base.
    """
    if user is None:
        return AccessContext(username=None, role=None, allowed_agents=frozenset())
    if user.role not in {"employee", "admin"}:
        return AccessContext(
            username=user.username,
            role=None,
            allowed_agents=frozenset(),
        )
    # The public demo identity must never see restricted material.
    allow_restricted = (
        user.role == "admin" and user.username != DEMO_ADMIN_USERNAME
    )
    return AccessContext(
        username=user.username,
        role=user.role,
        allowed_agents=enabled_agents(),
        allow_restricted=allow_restricted,
    )


def service_context() -> AccessContext:
    """Explicit server-side principal for code that runs without an end user.

    Used by the admin-triggered evaluation runner and by tests. It must always
    be constructed deliberately — it is never inferred from a missing session.
    """
    return AccessContext(
        username=None,
        role="service",
        allowed_agents=enabled_agents(),
        allow_restricted=True,
    )


def chunk_permitted(
    metadata: Mapping[str, Any] | None,
    context: AccessContext | None,
    namespace: str | None = None,
) -> bool:
    """Decide whether one retrieved chunk may be shown to this principal.

    ``namespace`` is the collection being queried; a chunk whose ``agent``
    metadata disagrees with it is refused (defence against cross-department
    contamination).
    """
    if context is None or not context.verified:
        return False
    if not isinstance(metadata, Mapping):
        return False
    agent = metadata.get("agent")
    if not isinstance(agent, str) or not agent:
        # Required permission information is missing → fail closed.
        return False
    if agent not in context.allowed_agents:
        return False
    if namespace is not None and agent != namespace:
        return False
    visibility = metadata.get("visibility")
    if visibility is None:
        # No visibility marker: the document is part of the company knowledge
        # base available to every authenticated employee (existing behaviour).
        return True
    if not isinstance(visibility, str):
        return False
    normalized = visibility.strip().casefold()
    if normalized == PUBLIC_VISIBILITY:
        return True
    if normalized in RESTRICTED_VISIBILITIES:
        return bool(context.allow_restricted)
    # Unrecognised marker → cannot be verified → fail closed.
    return False
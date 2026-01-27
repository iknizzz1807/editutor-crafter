#!/usr/bin/env python3
"""
Add Authentication & Security projects - foundational for all production systems.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

auth_security_projects = {
    "oauth2-provider": {
        "name": "OAuth2/OIDC Provider",
        "description": "Build a complete OAuth2 and OpenID Connect identity provider supporting authorization code flow, PKCE, refresh tokens, and JWT access tokens.",
        "why_expert": "Identity is the foundation of security. Understanding OAuth2/OIDC internals helps debug auth issues, design secure systems, and integrate with any identity provider.",
        "difficulty": "expert",
        "tags": ["security", "authentication", "identity", "oauth2", "jwt"],
        "estimated_hours": 50,
        "prerequisites": ["build-http-server"],
        "milestones": [
            {
                "name": "Client Registration & Authorization Endpoint",
                "description": "Implement client registration and the authorization endpoint with PKCE support",
                "skills": ["OAuth2 flows", "PKCE", "Cryptographic challenges"],
                "hints": {
                    "level1": "Start with client_id/client_secret storage and the /authorize endpoint",
                    "level2": "PKCE: client generates code_verifier, sends SHA256(code_verifier) as code_challenge",
                    "level3": """
```python
import hashlib
import base64
import secrets
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class OAuthClient:
    client_id: str
    client_secret_hash: str
    redirect_uris: list[str]
    grant_types: list[str]  # authorization_code, refresh_token, client_credentials
    scopes: list[str]

class AuthorizationServer:
    def __init__(self):
        self.clients: dict[str, OAuthClient] = {}
        self.authorization_codes: dict[str, dict] = {}  # code -> {client_id, user_id, scopes, code_challenge, expires_at}
        self.refresh_tokens: dict[str, dict] = {}

    def register_client(self, redirect_uris: list[str], grant_types: list[str]) -> tuple[str, str]:
        client_id = secrets.token_urlsafe(16)
        client_secret = secrets.token_urlsafe(32)
        client_secret_hash = hashlib.sha256(client_secret.encode()).hexdigest()

        self.clients[client_id] = OAuthClient(
            client_id=client_id,
            client_secret_hash=client_secret_hash,
            redirect_uris=redirect_uris,
            grant_types=grant_types,
            scopes=["openid", "profile", "email"]
        )
        return client_id, client_secret

    def authorize(self, client_id: str, redirect_uri: str, response_type: str,
                  scope: str, state: str, code_challenge: Optional[str] = None,
                  code_challenge_method: Optional[str] = None, user_id: str = None) -> str:
        client = self.clients.get(client_id)
        if not client:
            raise ValueError("Invalid client_id")
        if redirect_uri not in client.redirect_uris:
            raise ValueError("Invalid redirect_uri")
        if response_type != "code":
            raise ValueError("Only authorization_code flow supported")

        # Generate authorization code
        code = secrets.token_urlsafe(32)
        self.authorization_codes[code] = {
            "client_id": client_id,
            "user_id": user_id,
            "redirect_uri": redirect_uri,
            "scopes": scope.split(),
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "expires_at": time.time() + 600  # 10 minutes
        }

        return f"{redirect_uri}?code={code}&state={state}"

    def verify_pkce(self, code_verifier: str, code_challenge: str, method: str) -> bool:
        if method == "S256":
            computed = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).rstrip(b'=').decode()
            return secrets.compare_digest(computed, code_challenge)
        elif method == "plain":
            return secrets.compare_digest(code_verifier, code_challenge)
        return False
```
"""
                },
                "pitfalls": [
                    "Authorization codes must be single-use and short-lived (10 min max)",
                    "Always validate redirect_uri exactly - no partial matches",
                    "PKCE is mandatory for public clients (mobile, SPA)",
                    "State parameter prevents CSRF - must be unpredictable"
                ]
            },
            {
                "name": "Token Endpoint & JWT Generation",
                "description": "Implement the token endpoint with JWT access tokens and refresh token rotation",
                "skills": ["JWT signing", "Token rotation", "Secure token storage"],
                "hints": {
                    "level1": "Token endpoint exchanges authorization code for access_token + refresh_token",
                    "level2": "Use RS256 (asymmetric) for JWTs so resource servers can verify without shared secret",
                    "level3": """
```python
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from datetime import datetime, timedelta

class TokenService:
    def __init__(self):
        # Generate RSA key pair for JWT signing
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        self.issuer = "https://auth.example.com"

    def get_jwks(self) -> dict:
        '''Return JWKS for token verification'''
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        import json

        public_bytes = self.public_key.public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        # Convert to JWK format
        numbers = self.public_key.public_numbers()

        def int_to_base64(n: int) -> str:
            length = (n.bit_length() + 7) // 8
            return base64.urlsafe_b64encode(n.to_bytes(length, 'big')).rstrip(b'=').decode()

        return {
            "keys": [{
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": "key-1",
                "n": int_to_base64(numbers.n),
                "e": int_to_base64(numbers.e)
            }]
        }

    def create_access_token(self, user_id: str, client_id: str, scopes: list[str]) -> str:
        now = datetime.utcnow()
        payload = {
            "iss": self.issuer,
            "sub": user_id,
            "aud": client_id,
            "exp": now + timedelta(hours=1),
            "iat": now,
            "scope": " ".join(scopes),
            "jti": secrets.token_urlsafe(16)  # Unique token ID
        }

        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        return jwt.encode(payload, private_pem, algorithm="RS256", headers={"kid": "key-1"})

    def create_id_token(self, user_id: str, client_id: str, nonce: str, user_info: dict) -> str:
        '''OpenID Connect ID Token'''
        now = datetime.utcnow()
        payload = {
            "iss": self.issuer,
            "sub": user_id,
            "aud": client_id,
            "exp": now + timedelta(hours=1),
            "iat": now,
            "auth_time": int(now.timestamp()),
            "nonce": nonce,
            **user_info  # name, email, etc.
        }

        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        return jwt.encode(payload, private_pem, algorithm="RS256", headers={"kid": "key-1"})

    def create_refresh_token(self) -> str:
        return secrets.token_urlsafe(32)
```
"""
                },
                "pitfalls": [
                    "Never include sensitive data in JWT payload - it's base64, not encrypted",
                    "Refresh token rotation: issue new refresh token on each use, invalidate old one",
                    "Access tokens should be short-lived (15 min - 1 hour)",
                    "Always use constant-time comparison for token validation"
                ]
            },
            {
                "name": "Token Introspection & Revocation",
                "description": "Implement RFC 7662 token introspection and RFC 7009 token revocation",
                "skills": ["Token lifecycle", "Revocation strategies", "Cache invalidation"],
                "hints": {
                    "level1": "Introspection lets resource servers validate opaque tokens",
                    "level2": "Revocation must handle both access tokens and refresh tokens",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class TokenMetadata:
    active: bool
    client_id: str
    username: str
    scope: str
    sub: str
    aud: str
    iss: str
    exp: int
    iat: int
    token_type: str = "Bearer"

class TokenManager:
    def __init__(self, token_service: TokenService):
        self.token_service = token_service
        self.revoked_tokens: set[str] = set()  # Set of revoked JTIs
        self.refresh_tokens: dict[str, dict] = {}  # token -> metadata
        self.token_families: dict[str, str] = {}  # refresh_token -> family_id

    def introspect(self, token: str, token_type_hint: Optional[str] = None) -> dict:
        '''RFC 7662 - Token Introspection'''
        # Check if it's a JWT access token
        try:
            # Verify JWT signature
            public_pem = self.token_service.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            payload = jwt.decode(token, public_pem, algorithms=["RS256"],
                               options={"verify_aud": False})

            # Check if revoked
            if payload.get("jti") in self.revoked_tokens:
                return {"active": False}

            # Check expiration
            if payload["exp"] < time.time():
                return {"active": False}

            return {
                "active": True,
                "client_id": payload.get("aud"),
                "username": payload.get("sub"),
                "scope": payload.get("scope"),
                "sub": payload.get("sub"),
                "aud": payload.get("aud"),
                "iss": payload.get("iss"),
                "exp": payload.get("exp"),
                "iat": payload.get("iat"),
                "token_type": "Bearer"
            }
        except jwt.InvalidTokenError:
            pass

        # Check refresh tokens
        if token in self.refresh_tokens:
            meta = self.refresh_tokens[token]
            if meta["expires_at"] > time.time():
                return {"active": True, **meta}

        return {"active": False}

    def revoke(self, token: str, token_type_hint: Optional[str] = None) -> bool:
        '''RFC 7009 - Token Revocation'''
        # Try to decode as JWT to get JTI
        try:
            # Decode without verification to get JTI
            payload = jwt.decode(token, options={"verify_signature": False})
            if "jti" in payload:
                self.revoked_tokens.add(payload["jti"])
                return True
        except:
            pass

        # Check if it's a refresh token
        if token in self.refresh_tokens:
            # Revoke entire token family (prevents refresh token reuse attacks)
            family_id = self.token_families.get(token)
            if family_id:
                tokens_to_revoke = [t for t, fid in self.token_families.items() if fid == family_id]
                for t in tokens_to_revoke:
                    del self.refresh_tokens[t]
                    del self.token_families[t]
            else:
                del self.refresh_tokens[token]
            return True

        return False
```
"""
                },
                "pitfalls": [
                    "Introspection endpoint must be protected (client authentication required)",
                    "Revoked JWTs need tracking until expiry (use jti claim)",
                    "Refresh token families: if old token reused, revoke entire family (theft detection)",
                    "Consider using Redis for revocation list with TTL matching token expiry"
                ]
            },
            {
                "name": "UserInfo Endpoint & Consent Management",
                "description": "Implement OIDC UserInfo endpoint and user consent flows for scope approval",
                "skills": ["OIDC compliance", "Consent UX", "Scope management"],
                "hints": {
                    "level1": "UserInfo returns claims based on granted scopes (profile, email, address, phone)",
                    "level2": "Store user consents per client - don't re-ask for already-granted scopes",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class Scope(Enum):
    OPENID = "openid"
    PROFILE = "profile"  # name, family_name, given_name, picture, etc.
    EMAIL = "email"      # email, email_verified
    ADDRESS = "address"  # address claim
    PHONE = "phone"      # phone_number, phone_number_verified

@dataclass
class UserConsent:
    user_id: str
    client_id: str
    granted_scopes: set[str]
    granted_at: float

class UserInfoService:
    # Standard OIDC claims per scope
    SCOPE_CLAIMS = {
        "profile": ["name", "family_name", "given_name", "middle_name", "nickname",
                   "preferred_username", "profile", "picture", "website", "gender",
                   "birthdate", "zoneinfo", "locale", "updated_at"],
        "email": ["email", "email_verified"],
        "address": ["address"],
        "phone": ["phone_number", "phone_number_verified"]
    }

    def __init__(self):
        self.user_consents: dict[tuple[str, str], UserConsent] = {}  # (user_id, client_id) -> consent
        self.user_profiles: dict[str, dict] = {}  # user_id -> profile data

    def get_userinfo(self, access_token: str, token_manager: TokenManager) -> dict:
        '''OIDC UserInfo Endpoint'''
        introspection = token_manager.introspect(access_token)
        if not introspection.get("active"):
            raise ValueError("Invalid or expired token")

        user_id = introspection["sub"]
        scopes = introspection.get("scope", "").split()

        if "openid" not in scopes:
            raise ValueError("openid scope required")

        profile = self.user_profiles.get(user_id, {})
        claims = {"sub": user_id}

        for scope in scopes:
            if scope in self.SCOPE_CLAIMS:
                for claim in self.SCOPE_CLAIMS[scope]:
                    if claim in profile:
                        claims[claim] = profile[claim]

        return claims

    def check_consent(self, user_id: str, client_id: str, requested_scopes: set[str]) -> set[str]:
        '''Check which scopes need user consent'''
        key = (user_id, client_id)
        if key not in self.user_consents:
            return requested_scopes  # All scopes need consent

        existing = self.user_consents[key].granted_scopes
        return requested_scopes - existing  # Return scopes needing consent

    def record_consent(self, user_id: str, client_id: str, scopes: set[str]):
        '''Record user's consent for scopes'''
        key = (user_id, client_id)
        if key in self.user_consents:
            self.user_consents[key].granted_scopes.update(scopes)
        else:
            self.user_consents[key] = UserConsent(
                user_id=user_id,
                client_id=client_id,
                granted_scopes=scopes,
                granted_at=time.time()
            )

    def revoke_consent(self, user_id: str, client_id: str):
        '''Allow user to revoke consent for a client'''
        key = (user_id, client_id)
        if key in self.user_consents:
            del self.user_consents[key]
            # Also revoke all tokens for this user/client pair
            return True
        return False
```
"""
                },
                "pitfalls": [
                    "UserInfo must use access token, not ID token",
                    "Only return claims for scopes actually granted, not requested",
                    "Consent screen must clearly show what data will be shared",
                    "Allow users to revoke consent and see all authorized applications"
                ]
            }
        ]
    },

    "rbac-system": {
        "name": "RBAC/ABAC Authorization System",
        "description": "Build a flexible authorization system supporting Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC) with policy evaluation.",
        "why_expert": "Authorization logic is often scattered and inconsistent. A centralized policy engine prevents security bugs and simplifies auditing.",
        "difficulty": "expert",
        "tags": ["security", "authorization", "rbac", "abac", "policy"],
        "estimated_hours": 40,
        "prerequisites": [],
        "milestones": [
            {
                "name": "Role & Permission Model",
                "description": "Implement hierarchical roles with permission inheritance and efficient lookup",
                "skills": ["Role hierarchies", "Permission modeling", "Graph traversal"],
                "hints": {
                    "level1": "Roles can inherit from other roles - use DAG (no cycles)",
                    "level2": "Denormalize effective permissions for O(1) lookup at runtime",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import time

class Action(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "*"  # All actions

@dataclass
class Permission:
    resource: str      # e.g., "documents", "users", "reports"
    action: Action
    conditions: Optional[dict] = None  # For ABAC conditions

    def __hash__(self):
        return hash((self.resource, self.action))

    def matches(self, resource: str, action: Action) -> bool:
        if self.action == Action.ADMIN:
            return self.resource == resource or self.resource == "*"
        return self.resource == resource and self.action == action

@dataclass
class Role:
    name: str
    permissions: set[Permission] = field(default_factory=set)
    parent_roles: set[str] = field(default_factory=set)  # Inheritance

class RBACEngine:
    def __init__(self):
        self.roles: dict[str, Role] = {}
        self.user_roles: dict[str, set[str]] = {}  # user_id -> role names
        self._effective_permissions_cache: dict[str, set[Permission]] = {}

    def create_role(self, name: str, permissions: list[Permission],
                    parent_roles: list[str] = None) -> Role:
        # Validate no cycles
        if parent_roles:
            for parent in parent_roles:
                if self._would_create_cycle(name, parent):
                    raise ValueError(f"Role inheritance would create cycle: {name} -> {parent}")

        role = Role(
            name=name,
            permissions=set(permissions),
            parent_roles=set(parent_roles) if parent_roles else set()
        )
        self.roles[name] = role
        self._invalidate_cache()
        return role

    def _would_create_cycle(self, new_role: str, parent: str, visited: set = None) -> bool:
        if visited is None:
            visited = set()
        if parent == new_role:
            return True
        if parent in visited:
            return False
        visited.add(parent)

        parent_role = self.roles.get(parent)
        if not parent_role:
            return False
        for grandparent in parent_role.parent_roles:
            if self._would_create_cycle(new_role, grandparent, visited):
                return True
        return False

    def get_effective_permissions(self, role_name: str) -> set[Permission]:
        if role_name in self._effective_permissions_cache:
            return self._effective_permissions_cache[role_name]

        role = self.roles.get(role_name)
        if not role:
            return set()

        # Start with direct permissions
        effective = set(role.permissions)

        # Add inherited permissions (BFS)
        visited = {role_name}
        queue = list(role.parent_roles)

        while queue:
            parent_name = queue.pop(0)
            if parent_name in visited:
                continue
            visited.add(parent_name)

            parent = self.roles.get(parent_name)
            if parent:
                effective.update(parent.permissions)
                queue.extend(parent.parent_roles)

        self._effective_permissions_cache[role_name] = effective
        return effective

    def assign_role(self, user_id: str, role_name: str):
        if role_name not in self.roles:
            raise ValueError(f"Role not found: {role_name}")
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role_name)

    def check_permission(self, user_id: str, resource: str, action: Action) -> bool:
        user_role_names = self.user_roles.get(user_id, set())

        for role_name in user_role_names:
            permissions = self.get_effective_permissions(role_name)
            for perm in permissions:
                if perm.matches(resource, action):
                    return True
        return False

    def _invalidate_cache(self):
        self._effective_permissions_cache.clear()
```
"""
                },
                "pitfalls": [
                    "Role inheritance cycles cause infinite loops - validate DAG property",
                    "Cache invalidation needed when roles change - use versioning",
                    "Wildcard permissions (*) need careful handling to avoid over-granting",
                    "Role explosion: too many fine-grained roles become unmanageable"
                ]
            },
            {
                "name": "ABAC Policy Engine",
                "description": "Extend to attribute-based policies with conditions on user/resource/environment attributes",
                "skills": ["Policy languages", "Condition evaluation", "Context propagation"],
                "hints": {
                    "level1": "ABAC evaluates conditions like 'user.department == resource.owner_department'",
                    "level2": "Policies combine: allow if ANY policy allows, deny if ANY policy denies (deny wins)",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum
import operator

class Effect(Enum):
    ALLOW = "allow"
    DENY = "deny"

class Operator(Enum):
    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    MATCHES = "matches"  # Regex

@dataclass
class Condition:
    attribute: str      # e.g., "user.department", "resource.classification"
    operator: Operator
    value: Any

    def evaluate(self, context: dict) -> bool:
        # Navigate dot notation: "user.department" -> context["user"]["department"]
        attr_value = self._get_nested(context, self.attribute)
        if attr_value is None:
            return False

        ops = {
            Operator.EQUALS: lambda a, b: a == b,
            Operator.NOT_EQUALS: lambda a, b: a != b,
            Operator.GREATER_THAN: lambda a, b: a > b,
            Operator.LESS_THAN: lambda a, b: a < b,
            Operator.IN: lambda a, b: a in b,
            Operator.NOT_IN: lambda a, b: a not in b,
            Operator.CONTAINS: lambda a, b: b in a,
        }

        return ops[self.operator](attr_value, self.value)

    def _get_nested(self, obj: dict, path: str) -> Any:
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current

@dataclass
class Policy:
    id: str
    effect: Effect
    resources: list[str]    # Patterns like "documents/*", "users/{self}"
    actions: list[str]
    conditions: list[Condition] = field(default_factory=list)
    priority: int = 0       # Higher priority evaluated first

    def matches(self, resource: str, action: str, context: dict) -> tuple[bool, Effect]:
        # Check resource pattern
        if not self._matches_resource(resource, context):
            return False, None

        # Check action
        if action not in self.actions and "*" not in self.actions:
            return False, None

        # Evaluate all conditions (AND logic)
        for condition in self.conditions:
            if not condition.evaluate(context):
                return False, None

        return True, self.effect

    def _matches_resource(self, resource: str, context: dict) -> bool:
        for pattern in self.resources:
            # Handle {self} placeholder
            if "{self}" in pattern:
                user_id = context.get("user", {}).get("id", "")
                pattern = pattern.replace("{self}", user_id)

            # Simple glob matching
            if pattern == "*" or pattern == resource:
                return True
            if pattern.endswith("/*"):
                prefix = pattern[:-2]
                if resource.startswith(prefix):
                    return True
        return False

class ABACEngine:
    def __init__(self):
        self.policies: list[Policy] = []

    def add_policy(self, policy: Policy):
        self.policies.append(policy)
        # Keep sorted by priority (descending)
        self.policies.sort(key=lambda p: p.priority, reverse=True)

    def evaluate(self, resource: str, action: str, context: dict) -> bool:
        '''
        Evaluate all policies. Logic:
        1. If any DENY policy matches -> deny
        2. If any ALLOW policy matches -> allow
        3. Default deny
        '''
        allow_matched = False

        for policy in self.policies:
            matches, effect = policy.matches(resource, action, context)
            if matches:
                if effect == Effect.DENY:
                    return False  # Deny wins immediately
                elif effect == Effect.ALLOW:
                    allow_matched = True

        return allow_matched  # Default deny if no allow matched

# Example usage:
# policy = Policy(
#     id="owner-full-access",
#     effect=Effect.ALLOW,
#     resources=["documents/*"],
#     actions=["*"],
#     conditions=[
#         Condition("user.id", Operator.EQUALS, "resource.owner_id")
#     ]
# )
```
"""
                },
                "pitfalls": [
                    "Deny-by-default: no matching policy = deny",
                    "Explicit deny always wins over allow (principle of least privilege)",
                    "Context must include all attributes needed for evaluation",
                    "Policy ordering matters - define clear priority rules"
                ]
            },
            {
                "name": "Resource-Based & Multi-tenancy",
                "description": "Implement resource-level permissions and tenant isolation for SaaS applications",
                "skills": ["Multi-tenancy", "Resource ownership", "Tenant isolation"],
                "hints": {
                    "level1": "Every resource belongs to a tenant; users can only access resources in their tenant",
                    "level2": "Cross-tenant access requires explicit sharing with capability-based tokens",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import secrets

class ShareLevel(Enum):
    VIEWER = "viewer"    # Read-only
    EDITOR = "editor"    # Read + Write
    ADMIN = "admin"      # Read + Write + Share + Delete

@dataclass
class Resource:
    id: str
    tenant_id: str
    owner_id: str
    resource_type: str
    shares: dict[str, ShareLevel] = field(default_factory=dict)  # user_id -> level

@dataclass
class ShareLink:
    token: str
    resource_id: str
    level: ShareLevel
    expires_at: Optional[float] = None
    max_uses: Optional[int] = None
    uses: int = 0

class MultiTenantAuthz:
    def __init__(self, rbac_engine: RBACEngine, abac_engine: ABACEngine):
        self.rbac = rbac_engine
        self.abac = abac_engine
        self.resources: dict[str, Resource] = {}
        self.share_links: dict[str, ShareLink] = {}
        self.user_tenants: dict[str, str] = {}  # user_id -> tenant_id

    def check_access(self, user_id: str, resource_id: str, action: str) -> bool:
        resource = self.resources.get(resource_id)
        if not resource:
            return False

        user_tenant = self.user_tenants.get(user_id)

        # 1. Check tenant isolation
        if user_tenant != resource.tenant_id:
            # Cross-tenant - only allowed via explicit share
            return self._check_share(user_id, resource, action)

        # 2. Check if owner (owners have full access)
        if resource.owner_id == user_id:
            return True

        # 3. Check direct share
        if self._check_share(user_id, resource, action):
            return True

        # 4. Check RBAC (tenant-scoped roles)
        context = {
            "user": {"id": user_id, "tenant_id": user_tenant},
            "resource": {
                "id": resource_id,
                "tenant_id": resource.tenant_id,
                "owner_id": resource.owner_id,
                "type": resource.resource_type
            }
        }

        if self.abac.evaluate(f"{resource.resource_type}/{resource_id}", action, context):
            return True

        return False

    def _check_share(self, user_id: str, resource: Resource, action: str) -> bool:
        share_level = resource.shares.get(user_id)
        if not share_level:
            return False

        action_requirements = {
            "read": [ShareLevel.VIEWER, ShareLevel.EDITOR, ShareLevel.ADMIN],
            "write": [ShareLevel.EDITOR, ShareLevel.ADMIN],
            "delete": [ShareLevel.ADMIN],
            "share": [ShareLevel.ADMIN]
        }

        required = action_requirements.get(action, [ShareLevel.ADMIN])
        return share_level in required

    def create_share_link(self, user_id: str, resource_id: str,
                          level: ShareLevel, expires_in: int = None,
                          max_uses: int = None) -> str:
        # Verify user can share
        if not self.check_access(user_id, resource_id, "share"):
            raise PermissionError("Cannot share this resource")

        token = secrets.token_urlsafe(32)
        expires_at = time.time() + expires_in if expires_in else None

        self.share_links[token] = ShareLink(
            token=token,
            resource_id=resource_id,
            level=level,
            expires_at=expires_at,
            max_uses=max_uses
        )

        return token

    def redeem_share_link(self, token: str, user_id: str) -> bool:
        link = self.share_links.get(token)
        if not link:
            return False

        # Check expiry
        if link.expires_at and time.time() > link.expires_at:
            del self.share_links[token]
            return False

        # Check uses
        if link.max_uses and link.uses >= link.max_uses:
            return False

        # Grant access
        resource = self.resources.get(link.resource_id)
        if resource:
            resource.shares[user_id] = link.level
            link.uses += 1
            return True

        return False
```
"""
                },
                "pitfalls": [
                    "Tenant isolation must be enforced at database query level too",
                    "Admin roles should be tenant-scoped, not global",
                    "Share links need expiry and use limits to prevent abuse",
                    "Revoking share should be immediate - no caching of permissions"
                ]
            },
            {
                "name": "Audit Logging & Policy Testing",
                "description": "Implement comprehensive audit logging and policy simulation for testing",
                "skills": ["Security auditing", "Policy testing", "Compliance"],
                "hints": {
                    "level1": "Log every authorization decision with full context for forensics",
                    "level2": "Policy simulation lets admins test 'what if' scenarios before deploying",
                    "level3": """
```python
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
import json

@dataclass
class AuthzDecision:
    timestamp: datetime
    user_id: str
    resource: str
    action: str
    decision: bool
    policies_evaluated: list[str]
    matching_policy: Optional[str]
    context: dict
    latency_ms: float

class AuthzAuditLog:
    def __init__(self):
        self.decisions: list[AuthzDecision] = []

    def log_decision(self, decision: AuthzDecision):
        self.decisions.append(decision)
        # In production: send to SIEM, write to immutable log
        print(json.dumps({
            "type": "authz_decision",
            "timestamp": decision.timestamp.isoformat(),
            "user": decision.user_id,
            "resource": decision.resource,
            "action": decision.action,
            "allowed": decision.decision,
            "policy": decision.matching_policy
        }))

    def query(self, user_id: str = None, resource: str = None,
              action: str = None, decision: bool = None,
              start_time: datetime = None, end_time: datetime = None) -> list[AuthzDecision]:
        results = self.decisions

        if user_id:
            results = [d for d in results if d.user_id == user_id]
        if resource:
            results = [d for d in results if resource in d.resource]
        if action:
            results = [d for d in results if d.action == action]
        if decision is not None:
            results = [d for d in results if d.decision == decision]
        if start_time:
            results = [d for d in results if d.timestamp >= start_time]
        if end_time:
            results = [d for d in results if d.timestamp <= end_time]

        return results

class PolicySimulator:
    def __init__(self, abac_engine: ABACEngine):
        self.abac = abac_engine

    def simulate(self, policies: list[Policy], test_cases: list[dict]) -> list[dict]:
        '''
        Run policies against test cases without affecting production.

        test_cases format:
        [
            {
                "description": "Owner can read own document",
                "resource": "documents/123",
                "action": "read",
                "context": {"user": {"id": "user1"}, "resource": {"owner_id": "user1"}},
                "expected": True
            }
        ]
        '''
        # Create isolated engine for simulation
        sim_engine = ABACEngine()
        for policy in policies:
            sim_engine.add_policy(policy)

        results = []
        for case in test_cases:
            actual = sim_engine.evaluate(
                case["resource"],
                case["action"],
                case["context"]
            )

            results.append({
                "description": case["description"],
                "passed": actual == case["expected"],
                "expected": case["expected"],
                "actual": actual,
                "resource": case["resource"],
                "action": case["action"]
            })

        return results

    def find_violations(self, policies: list[Policy],
                        security_invariants: list[dict]) -> list[dict]:
        '''
        Check policies against security invariants.

        invariants format:
        [
            {
                "description": "No user can delete audit logs",
                "resource_pattern": "audit-logs/*",
                "action": "delete",
                "should_be": "denied",
                "for_any_context": True
            }
        ]
        '''
        violations = []

        for invariant in security_invariants:
            # Generate test contexts
            test_contexts = self._generate_test_contexts(invariant)

            for context in test_contexts:
                result = self.simulate(policies, [{
                    "description": invariant["description"],
                    "resource": invariant["resource_pattern"].replace("*", "test"),
                    "action": invariant["action"],
                    "context": context,
                    "expected": invariant["should_be"] == "denied"
                }])

                if not result[0]["passed"]:
                    violations.append({
                        "invariant": invariant["description"],
                        "context": context,
                        "expected": invariant["should_be"],
                        "actual": "allowed" if result[0]["actual"] else "denied"
                    })

        return violations

    def _generate_test_contexts(self, invariant: dict) -> list[dict]:
        # Generate various contexts to test the invariant
        return [
            {"user": {"id": "admin", "role": "admin"}},
            {"user": {"id": "user", "role": "user"}},
            {"user": {"id": "owner", "role": "owner"}, "resource": {"owner_id": "owner"}},
        ]
```
"""
                },
                "pitfalls": [
                    "Audit logs must be immutable - use append-only storage",
                    "Include enough context to understand decision without leaking secrets",
                    "Simulation environment must match production exactly",
                    "Test negative cases (should be denied) not just positive"
                ]
            }
        ]
    },

    "secret-management": {
        "name": "Secret Management System",
        "description": "Build a Vault-like secret management system with encryption, access control, dynamic secrets, and audit logging.",
        "why_expert": "Hardcoded secrets cause breaches. Understanding secret management helps design secure systems and properly integrate with existing solutions.",
        "difficulty": "expert",
        "tags": ["security", "secrets", "encryption", "vault", "infrastructure"],
        "estimated_hours": 45,
        "prerequisites": ["build-http-server"],
        "milestones": [
            {
                "name": "Encrypted Secret Storage",
                "description": "Implement envelope encryption with master key and data encryption keys",
                "skills": ["Envelope encryption", "Key management", "Secure storage"],
                "hints": {
                    "level1": "Master Key encrypts Data Encryption Keys (DEKs). DEKs encrypt actual secrets.",
                    "level2": "Never store master key with encrypted data. Use HSM or external KMS in production.",
                    "level3": """
```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import base64
import os
import json
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class EncryptedSecret:
    encrypted_dek: bytes      # DEK encrypted with master key
    encrypted_value: bytes    # Value encrypted with DEK
    created_at: float
    version: int
    metadata: dict

class SecretStore:
    def __init__(self, master_key: bytes = None):
        # In production: master key from HSM, Kubernetes secret, or cloud KMS
        if master_key:
            self.master_key = master_key
        else:
            # Derive from password for demo (use proper KMS in prod!)
            self.master_key = self._derive_key(b"demo-password", b"salt")

        self.master_fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
        self.secrets: dict[str, list[EncryptedSecret]] = {}  # path -> versions

    def _derive_key(self, password: bytes, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
            backend=default_backend()
        )
        return kdf.derive(password)

    def _generate_dek(self) -> tuple[bytes, Fernet]:
        '''Generate new Data Encryption Key'''
        dek = Fernet.generate_key()
        return dek, Fernet(dek)

    def put(self, path: str, value: str, metadata: dict = None) -> int:
        '''Store a secret with envelope encryption'''
        # Generate new DEK for this secret
        dek, dek_fernet = self._generate_dek()

        # Encrypt value with DEK
        encrypted_value = dek_fernet.encrypt(value.encode())

        # Encrypt DEK with master key
        encrypted_dek = self.master_fernet.encrypt(dek)

        # Determine version
        existing = self.secrets.get(path, [])
        version = len(existing) + 1

        secret = EncryptedSecret(
            encrypted_dek=encrypted_dek,
            encrypted_value=encrypted_value,
            created_at=time.time(),
            version=version,
            metadata=metadata or {}
        )

        if path not in self.secrets:
            self.secrets[path] = []
        self.secrets[path].append(secret)

        return version

    def get(self, path: str, version: int = None) -> Optional[str]:
        '''Retrieve and decrypt a secret'''
        versions = self.secrets.get(path)
        if not versions:
            return None

        # Get requested version or latest
        if version:
            if version < 1 or version > len(versions):
                return None
            secret = versions[version - 1]
        else:
            secret = versions[-1]

        # Decrypt DEK with master key
        dek = self.master_fernet.decrypt(secret.encrypted_dek)
        dek_fernet = Fernet(dek)

        # Decrypt value with DEK
        value = dek_fernet.decrypt(secret.encrypted_value)
        return value.decode()

    def rotate_master_key(self, new_master_key: bytes):
        '''Re-encrypt all DEKs with new master key'''
        new_master_fernet = Fernet(base64.urlsafe_b64encode(new_master_key))

        for path, versions in self.secrets.items():
            for secret in versions:
                # Decrypt DEK with old key
                dek = self.master_fernet.decrypt(secret.encrypted_dek)
                # Re-encrypt with new key
                secret.encrypted_dek = new_master_fernet.encrypt(dek)

        self.master_key = new_master_key
        self.master_fernet = new_master_fernet
```
"""
                },
                "pitfalls": [
                    "Master key in memory can be extracted - consider memory encryption",
                    "Secret versioning needed for rotation without breaking consumers",
                    "Backup encrypted data AND keys separately (but both needed for recovery)",
                    "Key derivation: use high iteration count (480k+) for PBKDF2"
                ]
            },
            {
                "name": "Access Policies & Authentication",
                "description": "Implement path-based ACLs and multiple authentication methods (token, AppRole, mTLS)",
                "skills": ["Path-based ACLs", "Authentication methods", "Token management"],
                "hints": {
                    "level1": "Policies define what paths a token can access with what capabilities",
                    "level2": "AppRole: role_id (public) + secret_id (private) = token for apps",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import secrets
import hashlib
import time
import fnmatch

class Capability(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    SUDO = "sudo"     # Override policies
    DENY = "deny"     # Explicit deny

@dataclass
class Policy:
    name: str
    path_rules: dict[str, set[Capability]]  # path pattern -> capabilities

    def check(self, path: str, capability: Capability) -> bool:
        for pattern, caps in self.path_rules.items():
            if fnmatch.fnmatch(path, pattern):
                if Capability.DENY in caps:
                    return False
                if capability in caps or Capability.SUDO in caps:
                    return True
        return False

@dataclass
class Token:
    id: str
    policies: list[str]
    created_at: float
    expires_at: Optional[float]
    renewable: bool
    metadata: dict = field(default_factory=dict)

@dataclass
class AppRole:
    role_id: str
    secret_id_hash: str
    policies: list[str]
    token_ttl: int = 3600
    secret_id_ttl: Optional[int] = None
    secret_id_num_uses: Optional[int] = None
    bind_secret_id: bool = True

class AuthManager:
    def __init__(self):
        self.policies: dict[str, Policy] = {}
        self.tokens: dict[str, Token] = {}
        self.app_roles: dict[str, AppRole] = {}
        self.secret_id_uses: dict[str, int] = {}  # secret_id_hash -> uses

    def create_policy(self, name: str, rules: dict[str, list[str]]) -> Policy:
        path_rules = {}
        for path, caps in rules.items():
            path_rules[path] = {Capability(c) for c in caps}

        policy = Policy(name=name, path_rules=path_rules)
        self.policies[name] = policy
        return policy

    def create_token(self, policies: list[str], ttl: int = 3600,
                     renewable: bool = True, metadata: dict = None) -> str:
        token_id = "hvs." + secrets.token_urlsafe(32)

        token = Token(
            id=token_id,
            policies=policies,
            created_at=time.time(),
            expires_at=time.time() + ttl if ttl else None,
            renewable=renewable,
            metadata=metadata or {}
        )

        self.tokens[token_id] = token
        return token_id

    def validate_token(self, token_id: str) -> Optional[Token]:
        token = self.tokens.get(token_id)
        if not token:
            return None
        if token.expires_at and time.time() > token.expires_at:
            del self.tokens[token_id]
            return None
        return token

    def check_permission(self, token_id: str, path: str, capability: Capability) -> bool:
        token = self.validate_token(token_id)
        if not token:
            return False

        for policy_name in token.policies:
            policy = self.policies.get(policy_name)
            if policy and policy.check(path, capability):
                return True
        return False

    # AppRole authentication
    def create_app_role(self, name: str, policies: list[str],
                        token_ttl: int = 3600) -> str:
        role_id = secrets.token_urlsafe(16)

        self.app_roles[name] = AppRole(
            role_id=role_id,
            secret_id_hash="",  # No secret yet
            policies=policies,
            token_ttl=token_ttl
        )

        return role_id

    def generate_secret_id(self, role_name: str) -> str:
        role = self.app_roles.get(role_name)
        if not role:
            raise ValueError("Role not found")

        secret_id = secrets.token_urlsafe(32)
        role.secret_id_hash = hashlib.sha256(secret_id.encode()).hexdigest()

        return secret_id

    def login_approle(self, role_id: str, secret_id: str) -> Optional[str]:
        '''Authenticate with AppRole, return token'''
        # Find role by role_id
        role = None
        for r in self.app_roles.values():
            if r.role_id == role_id:
                role = r
                break

        if not role:
            return None

        # Verify secret_id
        secret_hash = hashlib.sha256(secret_id.encode()).hexdigest()
        if not secrets.compare_digest(secret_hash, role.secret_id_hash):
            return None

        # Check secret_id uses
        if role.secret_id_num_uses:
            uses = self.secret_id_uses.get(secret_hash, 0)
            if uses >= role.secret_id_num_uses:
                return None
            self.secret_id_uses[secret_hash] = uses + 1

        # Issue token
        return self.create_token(
            policies=role.policies,
            ttl=role.token_ttl,
            metadata={"auth_method": "approle"}
        )
```
"""
                },
                "pitfalls": [
                    "role_id can be embedded in code; secret_id must be delivered securely",
                    "Token lookup table grows unbounded - need periodic cleanup",
                    "Glob patterns in paths: ensure * doesn't match too broadly",
                    "Constant-time comparison for secret validation prevents timing attacks"
                ]
            },
            {
                "name": "Dynamic Secrets",
                "description": "Generate short-lived credentials on-demand for databases, cloud providers, etc.",
                "skills": ["Dynamic credentials", "Lease management", "Secret rotation"],
                "hints": {
                    "level1": "Dynamic secrets are generated per-request with automatic expiration",
                    "level2": "Lease: tracks TTL and allows renewal or revocation",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Callable, Any, Optional
from abc import ABC, abstractmethod
import secrets
import string
import time
import threading

@dataclass
class Lease:
    lease_id: str
    secret_type: str
    ttl: int
    renewable: bool
    created_at: float
    expires_at: float
    data: dict  # Actual credentials
    revoke_callback: Callable[[], None]

class SecretEngine(ABC):
    @abstractmethod
    def generate(self, role: str) -> tuple[dict, Callable[[], None]]:
        '''Generate credentials and return (creds, revoke_fn)'''
        pass

class PostgresSecretEngine(SecretEngine):
    def __init__(self, conn_string: str):
        self.conn_string = conn_string

    def generate(self, role: str) -> tuple[dict, Callable[[], None]]:
        # Generate random username/password
        username = f"v-{role}-{secrets.token_hex(4)}"
        password = ''.join(secrets.choice(string.ascii_letters + string.digits)
                          for _ in range(32))

        # In production: actually create the user in Postgres
        # CREATE USER {username} WITH PASSWORD '{password}';
        # GRANT {role} TO {username};

        def revoke():
            # DROP USER {username};
            print(f"Revoking Postgres user: {username}")

        return {
            "username": username,
            "password": password,
            "connection_string": f"postgresql://{username}:{password}@..."
        }, revoke

class AWSSecretEngine(SecretEngine):
    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key

    def generate(self, role: str) -> tuple[dict, Callable[[], None]]:
        # In production: use AWS STS AssumeRole or IAM CreateAccessKey
        temp_access_key = f"AKIA{secrets.token_hex(8).upper()}"
        temp_secret_key = secrets.token_urlsafe(32)

        def revoke():
            # IAM DeleteAccessKey
            print(f"Revoking AWS key: {temp_access_key}")

        return {
            "access_key": temp_access_key,
            "secret_key": temp_secret_key,
            "session_token": None  # For STS
        }, revoke

class LeaseManager:
    def __init__(self):
        self.leases: dict[str, Lease] = {}
        self.engines: dict[str, SecretEngine] = {}
        self._start_reaper()

    def register_engine(self, name: str, engine: SecretEngine):
        self.engines[name] = engine

    def generate(self, engine_name: str, role: str, ttl: int = 3600) -> Lease:
        engine = self.engines.get(engine_name)
        if not engine:
            raise ValueError(f"Unknown engine: {engine_name}")

        creds, revoke_fn = engine.generate(role)

        lease_id = f"{engine_name}/{role}/{secrets.token_hex(8)}"
        now = time.time()

        lease = Lease(
            lease_id=lease_id,
            secret_type=engine_name,
            ttl=ttl,
            renewable=True,
            created_at=now,
            expires_at=now + ttl,
            data=creds,
            revoke_callback=revoke_fn
        )

        self.leases[lease_id] = lease
        return lease

    def renew(self, lease_id: str, increment: int = None) -> Optional[Lease]:
        lease = self.leases.get(lease_id)
        if not lease or not lease.renewable:
            return None

        # Extend TTL
        extension = increment or lease.ttl
        lease.expires_at = time.time() + extension
        return lease

    def revoke(self, lease_id: str) -> bool:
        lease = self.leases.get(lease_id)
        if not lease:
            return False

        # Call revoke callback (delete user, invalidate key, etc.)
        try:
            lease.revoke_callback()
        except Exception as e:
            print(f"Error revoking {lease_id}: {e}")

        del self.leases[lease_id]
        return True

    def _start_reaper(self):
        '''Background thread to revoke expired leases'''
        def reap():
            while True:
                time.sleep(60)
                now = time.time()
                expired = [lid for lid, l in self.leases.items()
                          if l.expires_at < now]
                for lease_id in expired:
                    self.revoke(lease_id)

        thread = threading.Thread(target=reap, daemon=True)
        thread.start()
```
"""
                },
                "pitfalls": [
                    "Lease reaper must handle engine failures gracefully",
                    "Max TTL should be enforced even for renewals",
                    "Revocation can fail - need retry logic and alerting",
                    "Connection pooling: don't create new DB user for every request"
                ]
            },
            {
                "name": "Unsealing & High Availability",
                "description": "Implement Shamir's secret sharing for master key unsealing and HA replication",
                "skills": ["Shamir's secret sharing", "Consensus", "Replication"],
                "hints": {
                    "level1": "Split master key into N shares; require K shares to reconstruct (K-of-N)",
                    "level2": "Sealed state: encrypted data exists but can't be decrypted without unsealing",
                    "level3": """
```python
from typing import List, Tuple
import secrets
from functools import reduce

class ShamirSecretSharing:
    '''
    Shamir's Secret Sharing Scheme
    Split secret into n shares, requiring k shares to reconstruct
    '''
    PRIME = 2**127 - 1  # Mersenne prime for finite field

    @classmethod
    def split(cls, secret: int, k: int, n: int) -> List[Tuple[int, int]]:
        '''Split secret into n shares with threshold k'''
        if k > n:
            raise ValueError("Threshold cannot exceed total shares")

        # Generate random polynomial coefficients
        # f(x) = secret + a1*x + a2*x^2 + ... + a(k-1)*x^(k-1)
        coefficients = [secret] + [secrets.randbelow(cls.PRIME) for _ in range(k - 1)]

        # Generate shares: (x, f(x)) for x = 1, 2, ..., n
        shares = []
        for x in range(1, n + 1):
            y = cls._evaluate_polynomial(coefficients, x)
            shares.append((x, y))

        return shares

    @classmethod
    def reconstruct(cls, shares: List[Tuple[int, int]]) -> int:
        '''Reconstruct secret from k shares using Lagrange interpolation'''
        k = len(shares)
        secret = 0

        for i, (xi, yi) in enumerate(shares):
            # Compute Lagrange basis polynomial at x=0
            numerator = 1
            denominator = 1

            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % cls.PRIME
                    denominator = (denominator * (xi - xj)) % cls.PRIME

            # Modular multiplicative inverse
            lagrange = (yi * numerator * pow(denominator, -1, cls.PRIME)) % cls.PRIME
            secret = (secret + lagrange) % cls.PRIME

        return secret

    @classmethod
    def _evaluate_polynomial(cls, coefficients: List[int], x: int) -> int:
        result = 0
        for i, coef in enumerate(coefficients):
            result = (result + coef * pow(x, i, cls.PRIME)) % cls.PRIME
        return result

class SealableVault:
    def __init__(self, threshold: int = 3, shares: int = 5):
        self.threshold = threshold
        self.total_shares = shares
        self.sealed = True
        self.master_key: bytes = None
        self.unseal_shares: List[Tuple[int, int]] = []
        self.secret_store: SecretStore = None

    def initialize(self) -> List[bytes]:
        '''Initialize vault, return key shares (distribute to operators)'''
        # Generate master key
        master_key_int = int.from_bytes(secrets.token_bytes(16), 'big')

        # Split using Shamir
        shares = ShamirSecretSharing.split(
            master_key_int, self.threshold, self.total_shares
        )

        # Encode shares for distribution
        encoded_shares = []
        for x, y in shares:
            # Encode as "x:y" in hex
            share_bytes = f"{x}:{y:032x}".encode()
            encoded_shares.append(share_bytes)

        # Store encrypted version of master key (for verification later)
        self._master_key_hash = hashlib.sha256(
            master_key_int.to_bytes(16, 'big')
        ).hexdigest()

        return encoded_shares

    def unseal(self, share: bytes) -> bool:
        '''Provide an unseal key. Returns True when vault is unsealed.'''
        if not self.sealed:
            return True

        # Decode share
        parts = share.decode().split(':')
        x = int(parts[0])
        y = int(parts[1], 16)

        # Add to collected shares (avoid duplicates)
        if not any(s[0] == x for s in self.unseal_shares):
            self.unseal_shares.append((x, y))

        # Try to reconstruct if we have enough shares
        if len(self.unseal_shares) >= self.threshold:
            master_key_int = ShamirSecretSharing.reconstruct(
                self.unseal_shares[:self.threshold]
            )
            master_key = master_key_int.to_bytes(16, 'big')

            # Verify
            if hashlib.sha256(master_key).hexdigest() == self._master_key_hash:
                self.master_key = master_key
                self.secret_store = SecretStore(master_key)
                self.sealed = False
                self.unseal_shares = []  # Clear shares from memory
                return True
            else:
                # Invalid reconstruction - wrong shares
                self.unseal_shares = []
                raise ValueError("Invalid unseal keys")

        return False

    def seal(self):
        '''Seal the vault - clear master key from memory'''
        self.master_key = None
        self.secret_store = None
        self.sealed = True
```
"""
                },
                "pitfalls": [
                    "Never store all shares together - defeats the purpose",
                    "Clear shares from memory after reconstruction",
                    "Sealed state must reject all secret operations",
                    "HA: only one node should be active writer to prevent split-brain"
                ]
            }
        ]
    },

    "session-management": {
        "name": "Distributed Session Management",
        "description": "Build a production-grade session management system with distributed storage, secure cookies, and session fixation prevention.",
        "why_expert": "Session management bugs (fixation, hijacking) are common vulnerabilities. Understanding internals helps prevent security issues.",
        "difficulty": "advanced",
        "tags": ["security", "sessions", "distributed", "cookies", "authentication"],
        "estimated_hours": 30,
        "prerequisites": ["build-redis"],
        "milestones": [
            {
                "name": "Secure Session Creation & Storage",
                "description": "Implement cryptographically secure session IDs with distributed storage backend",
                "skills": ["Session security", "Distributed storage", "Cookie handling"],
                "hints": {
                    "level1": "Session ID must be cryptographically random (128+ bits of entropy)",
                    "level2": "Store session data server-side, only session ID in cookie",
                    "level3": """
```python
import secrets
import hashlib
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Any
from abc import ABC, abstractmethod

@dataclass
class Session:
    id: str
    user_id: Optional[str]
    data: dict
    created_at: float
    last_accessed: float
    expires_at: float
    ip_address: str
    user_agent: str

class SessionStore(ABC):
    @abstractmethod
    def save(self, session: Session) -> None: pass

    @abstractmethod
    def load(self, session_id: str) -> Optional[Session]: pass

    @abstractmethod
    def delete(self, session_id: str) -> None: pass

    @abstractmethod
    def delete_user_sessions(self, user_id: str) -> int: pass

class RedisSessionStore(SessionStore):
    def __init__(self, redis_client, prefix: str = "session:"):
        self.redis = redis_client
        self.prefix = prefix

    def save(self, session: Session) -> None:
        key = f"{self.prefix}{session.id}"
        ttl = int(session.expires_at - time.time())

        data = {
            "id": session.id,
            "user_id": session.user_id,
            "data": session.data,
            "created_at": session.created_at,
            "last_accessed": session.last_accessed,
            "expires_at": session.expires_at,
            "ip_address": session.ip_address,
            "user_agent": session.user_agent
        }

        self.redis.setex(key, ttl, json.dumps(data))

        # Index by user for "logout all devices"
        if session.user_id:
            user_key = f"{self.prefix}user:{session.user_id}"
            self.redis.sadd(user_key, session.id)
            self.redis.expire(user_key, ttl)

    def load(self, session_id: str) -> Optional[Session]:
        key = f"{self.prefix}{session_id}"
        data = self.redis.get(key)
        if not data:
            return None

        d = json.loads(data)
        return Session(**d)

    def delete(self, session_id: str) -> None:
        key = f"{self.prefix}{session_id}"
        self.redis.delete(key)

    def delete_user_sessions(self, user_id: str) -> int:
        user_key = f"{self.prefix}user:{user_id}"
        session_ids = self.redis.smembers(user_key)

        for sid in session_ids:
            self.delete(sid.decode() if isinstance(sid, bytes) else sid)

        self.redis.delete(user_key)
        return len(session_ids)

class SessionManager:
    def __init__(self, store: SessionStore,
                 session_ttl: int = 86400,  # 24 hours
                 idle_timeout: int = 1800):  # 30 minutes
        self.store = store
        self.session_ttl = session_ttl
        self.idle_timeout = idle_timeout

    def create_session(self, ip_address: str, user_agent: str,
                       user_id: str = None) -> Session:
        # Generate cryptographically secure session ID
        session_id = secrets.token_urlsafe(32)  # 256 bits

        now = time.time()
        session = Session(
            id=session_id,
            user_id=user_id,
            data={},
            created_at=now,
            last_accessed=now,
            expires_at=now + self.session_ttl,
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.store.save(session)
        return session

    def get_session(self, session_id: str, ip_address: str = None) -> Optional[Session]:
        session = self.store.load(session_id)
        if not session:
            return None

        now = time.time()

        # Check absolute expiry
        if now > session.expires_at:
            self.store.delete(session_id)
            return None

        # Check idle timeout
        if now - session.last_accessed > self.idle_timeout:
            self.store.delete(session_id)
            return None

        # Optional: IP binding for high-security applications
        # if ip_address and session.ip_address != ip_address:
        #     return None  # Session hijacking attempt?

        # Update last accessed
        session.last_accessed = now
        self.store.save(session)

        return session

    def regenerate_id(self, old_session_id: str) -> Optional[Session]:
        '''Regenerate session ID (call after login to prevent fixation)'''
        session = self.store.load(old_session_id)
        if not session:
            return None

        # Delete old session
        self.store.delete(old_session_id)

        # Create new session with same data but new ID
        session.id = secrets.token_urlsafe(32)
        session.last_accessed = time.time()

        self.store.save(session)
        return session
```
"""
                },
                "pitfalls": [
                    "Session fixation: ALWAYS regenerate session ID after login",
                    "Session ID in URL is insecure (referer leaks, logs) - use cookies only",
                    "Idle timeout and absolute timeout are different - need both",
                    "Race condition: concurrent requests can cause session data loss"
                ]
            },
            {
                "name": "Cookie Security & Transport",
                "description": "Implement secure cookie handling with proper flags and encryption",
                "skills": ["Cookie security", "CSRF prevention", "Secure transport"],
                "hints": {
                    "level1": "Set Secure, HttpOnly, SameSite flags on session cookies",
                    "level2": "Sign cookies to detect tampering; encrypt if storing data in cookie",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional
import hmac
import hashlib
import base64
import time
import json
from cryptography.fernet import Fernet

@dataclass
class CookieOptions:
    secure: bool = True          # HTTPS only
    http_only: bool = True       # No JavaScript access
    same_site: str = "Lax"       # CSRF protection: Strict, Lax, None
    domain: Optional[str] = None
    path: str = "/"
    max_age: Optional[int] = None  # Seconds until expiry

    def to_header_string(self, name: str, value: str) -> str:
        parts = [f"{name}={value}"]

        if self.max_age is not None:
            parts.append(f"Max-Age={self.max_age}")
        if self.domain:
            parts.append(f"Domain={self.domain}")
        parts.append(f"Path={self.path}")
        if self.secure:
            parts.append("Secure")
        if self.http_only:
            parts.append("HttpOnly")
        parts.append(f"SameSite={self.same_site}")

        return "; ".join(parts)

class SecureCookieManager:
    def __init__(self, secret_key: bytes, encryption_key: bytes = None):
        self.secret_key = secret_key
        self.fernet = Fernet(encryption_key) if encryption_key else None

    def sign(self, value: str) -> str:
        '''Sign a value for integrity verification'''
        timestamp = str(int(time.time()))
        message = f"{timestamp}|{value}"

        signature = hmac.new(
            self.secret_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{message}|{signature}"

    def verify_signature(self, signed_value: str, max_age: int = None) -> Optional[str]:
        '''Verify signature and optionally check age'''
        try:
            parts = signed_value.rsplit("|", 2)
            if len(parts) != 3:
                return None

            timestamp_str, value, signature = parts
            timestamp = int(timestamp_str)

            # Check age
            if max_age and (time.time() - timestamp) > max_age:
                return None

            # Verify signature
            message = f"{timestamp_str}|{value}"
            expected = hmac.new(
                self.secret_key,
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            if hmac.compare_digest(signature, expected):
                return value
            return None

        except (ValueError, TypeError):
            return None

    def encrypt(self, data: dict) -> str:
        '''Encrypt data for cookie storage'''
        if not self.fernet:
            raise ValueError("Encryption key not configured")

        json_data = json.dumps(data)
        encrypted = self.fernet.encrypt(json_data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_value: str) -> Optional[dict]:
        '''Decrypt cookie data'''
        if not self.fernet:
            return None

        try:
            encrypted = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self.fernet.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except Exception:
            return None

    def create_session_cookie(self, session_id: str,
                              options: CookieOptions = None) -> str:
        '''Create a secure session cookie'''
        if options is None:
            options = CookieOptions()

        # Sign the session ID
        signed_id = self.sign(session_id)

        return options.to_header_string("session", signed_id)

    def parse_session_cookie(self, cookie_value: str,
                             max_age: int = 86400) -> Optional[str]:
        '''Parse and verify session cookie'''
        return self.verify_signature(cookie_value, max_age)

# CSRF Token management
class CSRFProtection:
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key

    def generate_token(self, session_id: str) -> str:
        '''Generate CSRF token bound to session'''
        random_part = secrets.token_urlsafe(16)
        message = f"{session_id}|{random_part}"

        signature = hmac.new(
            self.secret_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()[:16]

        return f"{random_part}.{signature}"

    def validate_token(self, token: str, session_id: str) -> bool:
        '''Validate CSRF token'''
        try:
            random_part, signature = token.split(".")
            message = f"{session_id}|{random_part}"

            expected = hmac.new(
                self.secret_key,
                message.encode(),
                hashlib.sha256
            ).hexdigest()[:16]

            return hmac.compare_digest(signature, expected)
        except ValueError:
            return False
```
"""
                },
                "pitfalls": [
                    "SameSite=None requires Secure flag (HTTPS)",
                    "Cookie size limit is ~4KB - don't store too much data",
                    "CSRF tokens must be tied to session - not global",
                    "Double-submit cookie pattern needs both cookie AND header/form"
                ]
            },
            {
                "name": "Multi-Device & Concurrent Sessions",
                "description": "Handle multiple sessions per user, device tracking, and forced logout",
                "skills": ["Device fingerprinting", "Session listing", "Forced logout"],
                "hints": {
                    "level1": "Track all active sessions per user for 'logout all devices'",
                    "level2": "Device fingerprint (IP, User-Agent, etc.) helps detect session theft",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional
import hashlib
import time
from user_agents import parse as parse_ua  # pip install user-agents

@dataclass
class DeviceInfo:
    device_id: str
    device_type: str      # mobile, tablet, desktop
    os: str
    browser: str
    ip_address: str
    location: Optional[str]  # From IP geolocation
    first_seen: float
    last_seen: float
    is_current: bool = False

@dataclass
class UserSession:
    session_id: str
    device: DeviceInfo
    created_at: float
    last_active: float
    is_current: bool

class MultiDeviceSessionManager:
    def __init__(self, session_manager: SessionManager, store: SessionStore):
        self.sessions = session_manager
        self.store = store
        self.max_sessions_per_user = 10

    def create_session_with_device(self, user_id: str, ip_address: str,
                                   user_agent: str) -> Session:
        # Parse user agent
        ua = parse_ua(user_agent)

        # Create device fingerprint
        device_id = self._fingerprint_device(ip_address, user_agent)

        # Check session limit
        existing = self.get_user_sessions(user_id)
        if len(existing) >= self.max_sessions_per_user:
            # Remove oldest session
            oldest = min(existing, key=lambda s: s.last_active)
            self.store.delete(oldest.session_id)

        # Create session
        session = self.sessions.create_session(
            ip_address=ip_address,
            user_agent=user_agent,
            user_id=user_id
        )

        # Store device info in session
        session.data["device"] = {
            "device_id": device_id,
            "device_type": self._get_device_type(ua),
            "os": f"{ua.os.family} {ua.os.version_string}",
            "browser": f"{ua.browser.family} {ua.browser.version_string}",
            "ip_address": ip_address
        }

        self.store.save(session)
        return session

    def _fingerprint_device(self, ip: str, user_agent: str) -> str:
        '''Create stable device fingerprint'''
        # In production: add more signals (screen size, timezone, etc.)
        data = f"{ip}|{user_agent}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _get_device_type(self, ua) -> str:
        if ua.is_mobile:
            return "mobile"
        elif ua.is_tablet:
            return "tablet"
        return "desktop"

    def get_user_sessions(self, user_id: str) -> list[UserSession]:
        '''List all active sessions for a user'''
        # Get all session IDs for user from Redis set
        user_key = f"session:user:{user_id}"
        session_ids = self.store.redis.smembers(user_key)

        sessions = []
        for sid in session_ids:
            sid_str = sid.decode() if isinstance(sid, bytes) else sid
            session = self.store.load(sid_str)
            if session:
                device_data = session.data.get("device", {})
                sessions.append(UserSession(
                    session_id=session.id,
                    device=DeviceInfo(
                        device_id=device_data.get("device_id", "unknown"),
                        device_type=device_data.get("device_type", "unknown"),
                        os=device_data.get("os", "unknown"),
                        browser=device_data.get("browser", "unknown"),
                        ip_address=device_data.get("ip_address", "unknown"),
                        location=None,  # Add geolocation lookup
                        first_seen=session.created_at,
                        last_seen=session.last_accessed,
                        is_current=False
                    ),
                    created_at=session.created_at,
                    last_active=session.last_accessed,
                    is_current=False
                ))

        return sorted(sessions, key=lambda s: s.last_active, reverse=True)

    def logout_session(self, user_id: str, session_id: str,
                       current_session_id: str) -> bool:
        '''Logout a specific session (from settings page)'''
        session = self.store.load(session_id)
        if not session or session.user_id != user_id:
            return False

        # Don't allow logging out current session via this method
        if session_id == current_session_id:
            return False

        self.store.delete(session_id)
        return True

    def logout_all_except_current(self, user_id: str,
                                  current_session_id: str) -> int:
        '''Logout all sessions except current'''
        sessions = self.get_user_sessions(user_id)
        count = 0

        for session in sessions:
            if session.session_id != current_session_id:
                self.store.delete(session.session_id)
                count += 1

        return count

    def force_logout_user(self, user_id: str) -> int:
        '''Admin: force logout all sessions for a user'''
        return self.store.delete_user_sessions(user_id)

    def detect_anomaly(self, session: Session, ip_address: str,
                       user_agent: str) -> list[str]:
        '''Detect suspicious session activity'''
        warnings = []
        device_data = session.data.get("device", {})

        # IP changed significantly
        if device_data.get("ip_address") != ip_address:
            # Could check if same /16 subnet or geolocation
            warnings.append("ip_changed")

        # User agent changed (browser update is normal, complete change is suspicious)
        if device_data.get("browser") != user_agent:
            new_fp = self._fingerprint_device(ip_address, user_agent)
            if new_fp != device_data.get("device_id"):
                warnings.append("device_changed")

        return warnings
```
"""
                },
                "pitfalls": [
                    "Device fingerprinting has false positives (VPN, browser updates)",
                    "Session limit without cleanup leads to locked-out users",
                    "Time-based anomaly detection needs to consider time zones",
                    "Don't show full session ID in UI - use hash or partial"
                ]
            }
        ]
    }
}

# Load and update
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

if 'expert_projects' not in data:
    data['expert_projects'] = {}

for project_id, project in auth_security_projects.items():
    data['expert_projects'][project_id] = project
    print(f"Added: {project_id} - {project['name']}")

with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nAdded {len(auth_security_projects)} Authentication & Security projects")

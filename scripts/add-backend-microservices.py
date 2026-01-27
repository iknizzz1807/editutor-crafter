#!/usr/bin/env python3
"""
Add Backend & Microservices projects.
Essential for production backend development and enterprise systems.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

backend_projects = {
    "rest-api-design": {
        "id": "rest-api-design",
        "name": "Production REST API",
        "description": "Build a production-grade REST API with authentication, validation, rate limiting, and proper error handling.",
        "difficulty": "beginner",
        "estimated_hours": "20-30",
        "prerequisites": ["HTTP basics", "JSON", "Database basics"],
        "languages": {
            "recommended": ["Go", "Python", "Node.js"],
            "also_possible": ["Rust", "Java"]
        },
        "resources": [
            {"name": "REST API Design Best Practices", "url": "https://restfulapi.net/", "type": "tutorial"},
            {"name": "OpenAPI Specification", "url": "https://swagger.io/specification/", "type": "documentation"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "CRUD Operations",
                "description": "Implement Create, Read, Update, Delete operations with proper HTTP methods.",
                "acceptance_criteria": [
                    "POST /resources creates new resource, returns 201",
                    "GET /resources returns paginated list",
                    "GET /resources/:id returns single resource or 404",
                    "PUT/PATCH /resources/:id updates resource",
                    "DELETE /resources/:id removes resource"
                ],
                "hints": {
                    "level1": "Use proper HTTP methods: GET for read, POST for create, PUT for replace, PATCH for update, DELETE for remove.",
                    "level2": "Return appropriate status codes: 200 OK, 201 Created, 204 No Content, 400 Bad Request, 404 Not Found.",
                    "level3": """from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4

app = FastAPI()

class Item(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    price: float

# In-memory storage
items_db: dict[str, Item] = {}

@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(item: Item) -> Item:
    item.id = str(uuid4())
    items_db[item.id] = item
    return item

@app.get("/items")
def list_items(skip: int = 0, limit: int = 10) -> List[Item]:
    items = list(items_db.values())
    return items[skip:skip + limit]

@app.get("/items/{item_id}")
def get_item(item_id: str) -> Item:
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]

@app.put("/items/{item_id}")
def update_item(item_id: str, item: Item) -> Item:
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    item.id = item_id
    items_db[item_id] = item
    return item

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: str):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del items_db[item_id]"""
                },
                "pitfalls": [
                    "Using POST for updates or GET for mutations",
                    "Returning 200 for resource creation instead of 201",
                    "Not handling concurrent updates",
                    "Missing Content-Type headers"
                ],
                "concepts": ["REST principles", "HTTP methods", "Status codes", "Resource modeling"],
                "estimated_hours": "4-6"
            },
            {
                "id": 2,
                "name": "Input Validation",
                "description": "Validate all input data and return meaningful errors.",
                "acceptance_criteria": [
                    "Validate request body against schema",
                    "Validate query parameters",
                    "Validate path parameters",
                    "Return detailed error messages",
                    "Sanitize input for security"
                ],
                "hints": {
                    "level1": "Use Pydantic (Python), Zod (TypeScript), or similar for schema validation.",
                    "level2": "Return 400 Bad Request with field-level errors. Include error codes for i18n.",
                    "level3": """from pydantic import BaseModel, Field, validator, ValidationError
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import List
import re

class CreateUserRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128)
    username: str = Field(..., min_length=3, max_length=50)

    @validator('email')
    def validate_email(cls, v):
        pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
        if not re.match(pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()

    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v

    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscores')
        return v

class ErrorDetail(BaseModel):
    field: str
    message: str
    code: str

class ErrorResponse(BaseModel):
    error: str
    details: List[ErrorDetail]

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = []
    for error in exc.errors():
        field = '.'.join(str(loc) for loc in error['loc'][1:])  # Skip 'body'
        errors.append(ErrorDetail(
            field=field,
            message=error['msg'],
            code=f"validation.{error['type']}"
        ))

    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Validation failed",
            details=errors
        ).dict()
    )

# Query parameter validation
@app.get("/users")
def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", regex="^(created_at|name|email)$")
):
    # Validated parameters are guaranteed to be valid here
    pass"""
                },
                "pitfalls": [
                    "Only validating types, not business rules",
                    "Exposing internal error details",
                    "Not validating query/path params",
                    "SQL/NoSQL injection through unvalidated input"
                ],
                "concepts": ["Schema validation", "Error handling", "Input sanitization", "Security"],
                "estimated_hours": "4-6"
            },
            {
                "id": 3,
                "name": "Authentication & Authorization",
                "description": "Implement JWT authentication and role-based access control.",
                "acceptance_criteria": [
                    "User registration and login",
                    "JWT token generation and validation",
                    "Token refresh mechanism",
                    "Role-based access control (RBAC)",
                    "Protected routes"
                ],
                "hints": {
                    "level1": "JWT = header.payload.signature. Store user ID and roles in payload.",
                    "level2": "Use short-lived access tokens (15min) + long-lived refresh tokens (7d).",
                    "level3": """import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from functools import wraps

SECRET_KEY = "your-secret-key"  # Use env var in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def create_access_token(user_id: str, roles: list[str]) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "roles": roles,
        "exp": expire,
        "type": "access"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = verify_token(credentials.credentials)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")
    return payload

def require_roles(*required_roles):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: dict = Depends(get_current_user), **kwargs):
            user_roles = set(current_user.get("roles", []))
            if not user_roles.intersection(required_roles):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

@app.post("/auth/login")
def login(email: str, password: str):
    user = get_user_by_email(email)
    if not user or not pwd_context.verify(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "access_token": create_access_token(user.id, user.roles),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer"
    }

@app.post("/auth/refresh")
def refresh(refresh_token: str):
    payload = verify_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user = get_user_by_id(payload["sub"])
    return {
        "access_token": create_access_token(user.id, user.roles),
        "token_type": "bearer"
    }

@app.get("/admin/users")
@require_roles("admin")
async def list_all_users(current_user: dict = Depends(get_current_user)):
    return get_all_users()"""
                },
                "pitfalls": [
                    "Storing secrets in code",
                    "No token expiration or too long expiry",
                    "Not invalidating tokens on password change",
                    "RBAC bypass through direct object access"
                ],
                "concepts": ["JWT tokens", "Password hashing", "RBAC", "Token refresh"],
                "estimated_hours": "6-8"
            },
            {
                "id": 4,
                "name": "Rate Limiting & Throttling",
                "description": "Protect API from abuse with rate limiting.",
                "acceptance_criteria": [
                    "Per-user rate limits",
                    "Per-endpoint rate limits",
                    "Sliding window algorithm",
                    "Rate limit headers in response",
                    "Graceful degradation"
                ],
                "hints": {
                    "level1": "Token bucket or sliding window. Store counts in Redis for distributed.",
                    "level2": "Return X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset headers.",
                    "level3": """import time
import redis
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> tuple[bool, dict]:
        now = time.time()
        window_start = now - window_seconds

        pipe = self.redis.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current entries
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(now): now})

        # Set expiry
        pipe.expire(key, window_seconds)

        results = pipe.execute()
        current_count = results[1]

        remaining = max(0, limit - current_count - 1)
        reset_time = int(now + window_seconds)

        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time)
        }

        return current_count < limit, headers

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_client: redis.Redis, default_limit: int = 100):
        super().__init__(app)
        self.limiter = RateLimiter(redis_client)
        self.default_limit = default_limit

        # Per-endpoint limits
        self.endpoint_limits = {
            "/auth/login": (5, 60),     # 5 requests per minute
            "/auth/register": (3, 60),  # 3 requests per minute
            "/api/search": (30, 60),    # 30 requests per minute
        }

    async def dispatch(self, request: Request, call_next):
        # Get identifier (user ID or IP)
        user_id = getattr(request.state, 'user_id', None)
        identifier = user_id or request.client.host

        # Get limit for endpoint
        path = request.url.path
        limit, window = self.endpoint_limits.get(path, (self.default_limit, 60))

        # Check rate limit
        key = f"ratelimit:{identifier}:{path}"
        allowed, headers = self.limiter.is_allowed(key, limit, window)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": headers["X-RateLimit-Reset"]},
                headers=headers
            )

        response = await call_next(request)

        # Add rate limit headers
        for header, value in headers.items():
            response.headers[header] = value

        return response

# Distributed rate limiting with token bucket
class TokenBucket:
    def __init__(self, redis_client: redis.Redis, capacity: int, refill_rate: float):
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second

    def consume(self, key: str, tokens: int = 1) -> bool:
        now = time.time()

        # Lua script for atomic operation
        script = '''
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local current_tokens = tonumber(bucket[1]) or capacity
        local last_update = tonumber(bucket[2]) or now

        local elapsed = now - last_update
        local refill = elapsed * refill_rate
        current_tokens = math.min(capacity, current_tokens + refill)

        if current_tokens >= tokens then
            current_tokens = current_tokens - tokens
            redis.call('HMSET', key, 'tokens', current_tokens, 'last_update', now)
            redis.call('EXPIRE', key, 3600)
            return 1
        end
        return 0
        '''

        result = self.redis.eval(script, 1, key, self.capacity, self.refill_rate, tokens, now)
        return result == 1"""
                },
                "pitfalls": [
                    "Rate limits per IP bypass with proxies",
                    "Not handling Redis failures gracefully",
                    "Clock skew in distributed systems",
                    "Rate limit headers revealing too much info"
                ],
                "concepts": ["Rate limiting algorithms", "Sliding window", "Token bucket", "Redis"],
                "estimated_hours": "5-7"
            }
        ]
    },

    "api-gateway": {
        "id": "api-gateway",
        "name": "API Gateway",
        "description": "Build an API gateway that handles routing, authentication, rate limiting, and request transformation.",
        "difficulty": "advanced",
        "estimated_hours": "50-70",
        "prerequisites": ["REST APIs", "Networking", "Load balancing concepts"],
        "languages": {
            "recommended": ["Go", "Rust"],
            "also_possible": ["Node.js", "Java"]
        },
        "resources": [
            {"name": "Kong Gateway", "url": "https://docs.konghq.com/", "type": "reference"},
            {"name": "NGINX as API Gateway", "url": "https://www.nginx.com/blog/deploying-nginx-plus-as-an-api-gateway/", "type": "article"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Reverse Proxy & Routing",
                "description": "Route requests to appropriate backend services.",
                "acceptance_criteria": [
                    "Path-based routing (/api/users -> user-service)",
                    "Host-based routing (api.example.com)",
                    "Load balancing across instances",
                    "Health checks for backends",
                    "Circuit breaker for failing services"
                ],
                "hints": {
                    "level1": "Use regex patterns for path matching. Maintain pool of healthy backends.",
                    "level2": "Round-robin for basic LB. Add weights, least-connections for advanced.",
                    "level3": """package main

import (
    "net/http"
    "net/http/httputil"
    "net/url"
    "sync"
    "time"
)

type Backend struct {
    URL      *url.URL
    Alive    bool
    Weight   int
    mu       sync.RWMutex
}

func (b *Backend) SetAlive(alive bool) {
    b.mu.Lock()
    b.Alive = alive
    b.mu.Unlock()
}

func (b *Backend) IsAlive() bool {
    b.mu.RLock()
    defer b.mu.RUnlock()
    return b.Alive
}

type Route struct {
    PathPrefix string
    Backends   []*Backend
    current    uint64
}

func (r *Route) NextBackend() *Backend {
    for i := 0; i < len(r.Backends); i++ {
        idx := int(atomic.AddUint64(&r.current, 1)) % len(r.Backends)
        if r.Backends[idx].IsAlive() {
            return r.Backends[idx]
        }
    }
    return nil
}

type Gateway struct {
    routes  map[string]*Route
    mu      sync.RWMutex
}

func (g *Gateway) AddRoute(pathPrefix string, backends []string) {
    route := &Route{PathPrefix: pathPrefix}

    for _, b := range backends {
        u, _ := url.Parse(b)
        route.Backends = append(route.Backends, &Backend{URL: u, Alive: true, Weight: 1})
    }

    g.mu.Lock()
    g.routes[pathPrefix] = route
    g.mu.Unlock()
}

func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // Find matching route
    g.mu.RLock()
    var matchedRoute *Route
    var longestMatch int
    for prefix, route := range g.routes {
        if strings.HasPrefix(r.URL.Path, prefix) && len(prefix) > longestMatch {
            matchedRoute = route
            longestMatch = len(prefix)
        }
    }
    g.mu.RUnlock()

    if matchedRoute == nil {
        http.Error(w, "No route found", http.StatusNotFound)
        return
    }

    // Get healthy backend
    backend := matchedRoute.NextBackend()
    if backend == nil {
        http.Error(w, "No healthy backends", http.StatusServiceUnavailable)
        return
    }

    // Reverse proxy
    proxy := httputil.NewSingleHostReverseProxy(backend.URL)

    // Modify request
    r.URL.Host = backend.URL.Host
    r.URL.Scheme = backend.URL.Scheme
    r.Header.Set("X-Forwarded-Host", r.Host)
    r.Header.Set("X-Real-IP", r.RemoteAddr)

    proxy.ServeHTTP(w, r)
}

// Health checker
func (g *Gateway) HealthCheck(interval time.Duration) {
    ticker := time.NewTicker(interval)
    for range ticker.C {
        g.mu.RLock()
        for _, route := range g.routes {
            for _, backend := range route.Backends {
                go func(b *Backend) {
                    resp, err := http.Get(b.URL.String() + "/health")
                    b.SetAlive(err == nil && resp.StatusCode == 200)
                }(backend)
            }
        }
        g.mu.RUnlock()
    }
}"""
                },
                "pitfalls": [
                    "Not forwarding original client IP",
                    "Health checks blocking request handling",
                    "Memory leaks from unclosed connections",
                    "Thundering herd on backend recovery"
                ],
                "concepts": ["Reverse proxy", "Load balancing", "Health checks", "Service discovery"],
                "estimated_hours": "12-18"
            },
            {
                "id": 2,
                "name": "Request/Response Transformation",
                "description": "Modify requests and responses as they pass through.",
                "acceptance_criteria": [
                    "Header manipulation (add, remove, modify)",
                    "Request body transformation",
                    "Response body transformation",
                    "URL rewriting",
                    "Request aggregation (combine multiple backend calls)"
                ],
                "hints": {
                    "level1": "Intercept request/response streams. Use middleware pattern.",
                    "level2": "For body transformation, buffer entire body, transform, forward.",
                    "level3": """import (
    "bytes"
    "encoding/json"
    "io"
)

type Transformer interface {
    TransformRequest(r *http.Request) error
    TransformResponse(resp *http.Response) error
}

type HeaderTransformer struct {
    AddHeaders    map[string]string
    RemoveHeaders []string
}

func (t *HeaderTransformer) TransformRequest(r *http.Request) error {
    for key, value := range t.AddHeaders {
        r.Header.Set(key, value)
    }
    for _, key := range t.RemoveHeaders {
        r.Header.Del(key)
    }
    return nil
}

type BodyTransformer struct {
    Transform func(body []byte) ([]byte, error)
}

func (t *BodyTransformer) TransformRequest(r *http.Request) error {
    if r.Body == nil || t.Transform == nil {
        return nil
    }

    body, err := io.ReadAll(r.Body)
    if err != nil {
        return err
    }
    r.Body.Close()

    transformed, err := t.Transform(body)
    if err != nil {
        return err
    }

    r.Body = io.NopCloser(bytes.NewReader(transformed))
    r.ContentLength = int64(len(transformed))
    return nil
}

// URL Rewriting
type URLRewriter struct {
    Rules []RewriteRule
}

type RewriteRule struct {
    Pattern *regexp.Regexp
    Replace string
}

func (u *URLRewriter) TransformRequest(r *http.Request) error {
    path := r.URL.Path
    for _, rule := range u.Rules {
        if rule.Pattern.MatchString(path) {
            path = rule.Pattern.ReplaceAllString(path, rule.Replace)
            break
        }
    }
    r.URL.Path = path
    return nil
}

// Request aggregation
type Aggregator struct {
    Endpoints []AggregateEndpoint
}

type AggregateEndpoint struct {
    Name    string
    URL     string
    Extract func(response []byte) interface{}
}

func (a *Aggregator) Aggregate(ctx context.Context) (map[string]interface{}, error) {
    results := make(map[string]interface{})
    var mu sync.Mutex
    var wg sync.WaitGroup

    for _, endpoint := range a.Endpoints {
        wg.Add(1)
        go func(ep AggregateEndpoint) {
            defer wg.Done()

            req, _ := http.NewRequestWithContext(ctx, "GET", ep.URL, nil)
            resp, err := http.DefaultClient.Do(req)
            if err != nil {
                return
            }
            defer resp.Body.Close()

            body, _ := io.ReadAll(resp.Body)
            extracted := ep.Extract(body)

            mu.Lock()
            results[ep.Name] = extracted
            mu.Unlock()
        }(endpoint)
    }

    wg.Wait()
    return results, nil
}"""
                },
                "pitfalls": [
                    "Large body buffering causes memory issues",
                    "Transformation errors not handled gracefully",
                    "Aggregation timeout causes partial responses",
                    "Content-Length mismatch after transformation"
                ],
                "concepts": ["Request transformation", "Response transformation", "URL rewriting", "API aggregation"],
                "estimated_hours": "10-14"
            },
            {
                "id": 3,
                "name": "Authentication & Authorization Layer",
                "description": "Centralized auth handling at gateway level.",
                "acceptance_criteria": [
                    "JWT validation at gateway",
                    "API key authentication",
                    "OAuth2 token introspection",
                    "Pass user context to backends",
                    "Auth caching for performance"
                ],
                "hints": {
                    "level1": "Validate JWT signature and expiry. Cache valid tokens briefly.",
                    "level2": "Add user info to X-User-ID, X-User-Roles headers for backends.",
                    "level3": """type AuthMiddleware struct {
    JWTSecret     []byte
    APIKeyStore   APIKeyStore
    TokenCache    *lru.Cache // github.com/hashicorp/golang-lru
    CacheTTL      time.Duration
}

func (m *AuthMiddleware) Authenticate(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        var userCtx *UserContext

        // Try Bearer token first
        authHeader := r.Header.Get("Authorization")
        if strings.HasPrefix(authHeader, "Bearer ") {
            token := strings.TrimPrefix(authHeader, "Bearer ")
            userCtx = m.validateJWT(token)
        }

        // Try API key
        if userCtx == nil {
            apiKey := r.Header.Get("X-API-Key")
            if apiKey != "" {
                userCtx = m.validateAPIKey(apiKey)
            }
        }

        if userCtx == nil {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }

        // Add user context to headers for backends
        r.Header.Set("X-User-ID", userCtx.UserID)
        r.Header.Set("X-User-Roles", strings.Join(userCtx.Roles, ","))
        r.Header.Set("X-Auth-Method", userCtx.AuthMethod)

        next.ServeHTTP(w, r)
    })
}

func (m *AuthMiddleware) validateJWT(token string) *UserContext {
    // Check cache first
    if cached, ok := m.TokenCache.Get(token); ok {
        return cached.(*UserContext)
    }

    // Parse and validate
    claims := jwt.MapClaims{}
    parsed, err := jwt.ParseWithClaims(token, claims, func(t *jwt.Token) (interface{}, error) {
        return m.JWTSecret, nil
    })

    if err != nil || !parsed.Valid {
        return nil
    }

    userCtx := &UserContext{
        UserID:     claims["sub"].(string),
        Roles:      toStringSlice(claims["roles"]),
        AuthMethod: "jwt",
    }

    // Cache it
    m.TokenCache.Add(token, userCtx)

    return userCtx
}

// OAuth2 token introspection for external tokens
func (m *AuthMiddleware) introspectToken(token string) *UserContext {
    // Check cache
    if cached, ok := m.TokenCache.Get(token); ok {
        return cached.(*UserContext)
    }

    // Call OAuth server
    resp, err := http.PostForm(m.IntrospectionURL, url.Values{
        "token":           {token},
        "client_id":       {m.ClientID},
        "client_secret":   {m.ClientSecret},
    })
    if err != nil {
        return nil
    }
    defer resp.Body.Close()

    var result struct {
        Active bool   `json:"active"`
        Sub    string `json:"sub"`
        Scope  string `json:"scope"`
    }
    json.NewDecoder(resp.Body).Decode(&result)

    if !result.Active {
        return nil
    }

    userCtx := &UserContext{
        UserID:     result.Sub,
        Roles:      strings.Split(result.Scope, " "),
        AuthMethod: "oauth2",
    }

    m.TokenCache.Add(token, userCtx)
    return userCtx
}"""
                },
                "pitfalls": [
                    "Caching tokens too long misses revocations",
                    "Not validating token audience/issuer",
                    "API keys in logs or error messages",
                    "Auth failures not rate limited (brute force)"
                ],
                "concepts": ["Centralized authentication", "Token introspection", "Auth caching", "Header propagation"],
                "estimated_hours": "10-15"
            },
            {
                "id": 4,
                "name": "Observability & Plugins",
                "description": "Add logging, metrics, tracing, and plugin system.",
                "acceptance_criteria": [
                    "Structured access logs",
                    "Prometheus metrics",
                    "Distributed tracing (OpenTelemetry)",
                    "Plugin architecture for extensibility",
                    "Dynamic configuration reload"
                ],
                "hints": {
                    "level1": "Wrap handlers with logging/metrics middleware. Use OpenTelemetry SDK.",
                    "level2": "Plugin = interface with hooks. Load plugins at startup or dynamically.",
                    "level3": """import (
    "github.com/prometheus/client_golang/prometheus"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
)

// Metrics
var (
    requestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "gateway_requests_total",
            Help: "Total HTTP requests",
        },
        []string{"method", "path", "status"},
    )

    requestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "gateway_request_duration_seconds",
            Help:    "Request duration in seconds",
            Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
        },
        []string{"method", "path"},
    )
)

func MetricsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()

        wrapped := &statusRecorder{ResponseWriter: w, status: 200}
        next.ServeHTTP(wrapped, r)

        duration := time.Since(start).Seconds()
        path := sanitizePath(r.URL.Path)  // Avoid cardinality explosion

        requestsTotal.WithLabelValues(r.Method, path, strconv.Itoa(wrapped.status)).Inc()
        requestDuration.WithLabelValues(r.Method, path).Observe(duration)
    })
}

// Tracing
func TracingMiddleware(next http.Handler) http.Handler {
    tracer := otel.Tracer("gateway")

    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        ctx, span := tracer.Start(r.Context(), r.URL.Path,
            trace.WithAttributes(
                attribute.String("http.method", r.Method),
                attribute.String("http.url", r.URL.String()),
            ),
        )
        defer span.End()

        // Propagate trace context to backends
        r = r.WithContext(ctx)
        otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(r.Header))

        next.ServeHTTP(w, r)
    })
}

// Plugin system
type Plugin interface {
    Name() string
    Init(config map[string]interface{}) error
    PreRequest(r *http.Request) error
    PostResponse(resp *http.Response) error
}

type PluginManager struct {
    plugins []Plugin
}

func (pm *PluginManager) Load(name string, config map[string]interface{}) error {
    // Dynamic loading (could use go plugins or scripting)
    var plugin Plugin

    switch name {
    case "cors":
        plugin = &CORSPlugin{}
    case "request-id":
        plugin = &RequestIDPlugin{}
    case "compression":
        plugin = &CompressionPlugin{}
    default:
        return fmt.Errorf("unknown plugin: %s", name)
    }

    if err := plugin.Init(config); err != nil {
        return err
    }

    pm.plugins = append(pm.plugins, plugin)
    return nil
}

func (pm *PluginManager) RunPreRequest(r *http.Request) error {
    for _, p := range pm.plugins {
        if err := p.PreRequest(r); err != nil {
            return err
        }
    }
    return nil
}"""
                },
                "pitfalls": [
                    "High cardinality labels in metrics",
                    "Logging sensitive data (passwords, tokens)",
                    "Trace sampling too aggressive loses important traces",
                    "Plugin panics crash entire gateway"
                ],
                "concepts": ["Observability", "Prometheus metrics", "OpenTelemetry tracing", "Plugin architecture"],
                "estimated_hours": "12-18"
            }
        ]
    },

    "grpc-service": {
        "id": "grpc-service",
        "name": "gRPC Microservice",
        "description": "Build a gRPC service with streaming, error handling, and interceptors.",
        "difficulty": "intermediate",
        "estimated_hours": "25-40",
        "prerequisites": ["Protocol Buffers", "RPC concepts", "Go or similar"],
        "languages": {
            "recommended": ["Go", "Rust"],
            "also_possible": ["Java", "Python", "C++"]
        },
        "resources": [
            {"name": "gRPC Official Docs", "url": "https://grpc.io/docs/", "type": "documentation"},
            {"name": "Protocol Buffers", "url": "https://developers.google.com/protocol-buffers", "type": "documentation"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Proto Definition & Code Generation",
                "description": "Define service contract with Protocol Buffers.",
                "acceptance_criteria": [
                    "Define message types",
                    "Define service methods (unary, streaming)",
                    "Generate server and client code",
                    "Version proto files properly",
                    "Document APIs with comments"
                ],
                "hints": {
                    "level1": "proto3 syntax. Use well-known types (Timestamp, Duration). Generate with protoc.",
                    "level2": "Package versioning: myservice.v1.MyService. Never break backwards compatibility.",
                    "level3": """syntax = "proto3";

package userservice.v1;

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

option go_package = "github.com/example/userservice/v1;userservicev1";

// UserService manages user accounts
service UserService {
  // Creates a new user account
  rpc CreateUser(CreateUserRequest) returns (User);

  // Gets a user by ID
  rpc GetUser(GetUserRequest) returns (User);

  // Lists users with pagination
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);

  // Watches for user updates (server streaming)
  rpc WatchUsers(WatchUsersRequest) returns (stream UserEvent);

  // Bulk import users (client streaming)
  rpc ImportUsers(stream ImportUserRequest) returns (ImportUsersResponse);

  // Chat (bidirectional streaming)
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

message User {
  string id = 1;
  string email = 2;
  string name = 3;
  UserStatus status = 4;
  google.protobuf.Timestamp created_at = 5;
  google.protobuf.Timestamp updated_at = 6;

  // Nested type for profile
  Profile profile = 7;

  message Profile {
    string bio = 1;
    string avatar_url = 2;
  }
}

enum UserStatus {
  USER_STATUS_UNSPECIFIED = 0;
  USER_STATUS_ACTIVE = 1;
  USER_STATUS_SUSPENDED = 2;
  USER_STATUS_DELETED = 3;
}

message CreateUserRequest {
  string email = 1;
  string name = 2;
  string password = 3;
}

message GetUserRequest {
  string id = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
  string filter = 3;  // e.g., "status=active"
}

message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
  int32 total_count = 3;
}

message UserEvent {
  enum EventType {
    EVENT_TYPE_UNSPECIFIED = 0;
    EVENT_TYPE_CREATED = 1;
    EVENT_TYPE_UPDATED = 2;
    EVENT_TYPE_DELETED = 3;
  }

  EventType type = 1;
  User user = 2;
  google.protobuf.Timestamp occurred_at = 3;
}

// Generate: protoc --go_out=. --go-grpc_out=. user.proto"""
                },
                "pitfalls": [
                    "Changing field numbers breaks compatibility",
                    "Using required fields (proto3 doesn't have them)",
                    "Not setting default enum value to UNSPECIFIED",
                    "Large messages exceed gRPC size limits"
                ],
                "concepts": ["Protocol Buffers", "Service definition", "Code generation", "API versioning"],
                "estimated_hours": "4-6"
            },
            {
                "id": 2,
                "name": "Server Implementation",
                "description": "Implement gRPC server with all RPC types.",
                "acceptance_criteria": [
                    "Unary RPC implementation",
                    "Server streaming RPC",
                    "Client streaming RPC",
                    "Bidirectional streaming",
                    "Graceful shutdown"
                ],
                "hints": {
                    "level1": "Implement the generated interface. For streaming, use Send() and Recv() loops.",
                    "level2": "Handle context cancellation. Return proper gRPC status codes.",
                    "level3": """package main

import (
    "context"
    "io"
    "sync"

    pb "github.com/example/userservice/v1"
    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

type userServer struct {
    pb.UnimplementedUserServiceServer
    users    map[string]*pb.User
    mu       sync.RWMutex
    watchers map[chan *pb.UserEvent]struct{}
}

func (s *userServer) CreateUser(ctx context.Context, req *pb.CreateUserRequest) (*pb.User, error) {
    // Validate
    if req.Email == "" {
        return nil, status.Error(codes.InvalidArgument, "email is required")
    }

    s.mu.Lock()
    defer s.mu.Unlock()

    // Check duplicate
    for _, u := range s.users {
        if u.Email == req.Email {
            return nil, status.Error(codes.AlreadyExists, "email already registered")
        }
    }

    user := &pb.User{
        Id:        uuid.New().String(),
        Email:     req.Email,
        Name:      req.Name,
        Status:    pb.UserStatus_USER_STATUS_ACTIVE,
        CreatedAt: timestamppb.Now(),
    }

    s.users[user.Id] = user

    // Notify watchers
    s.notify(&pb.UserEvent{
        Type:       pb.UserEvent_EVENT_TYPE_CREATED,
        User:       user,
        OccurredAt: timestamppb.Now(),
    })

    return user, nil
}

func (s *userServer) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()

    user, ok := s.users[req.Id]
    if !ok {
        return nil, status.Error(codes.NotFound, "user not found")
    }

    return user, nil
}

// Server streaming
func (s *userServer) WatchUsers(req *pb.WatchUsersRequest, stream pb.UserService_WatchUsersServer) error {
    ch := make(chan *pb.UserEvent, 10)

    s.mu.Lock()
    s.watchers[ch] = struct{}{}
    s.mu.Unlock()

    defer func() {
        s.mu.Lock()
        delete(s.watchers, ch)
        s.mu.Unlock()
        close(ch)
    }()

    for {
        select {
        case event := <-ch:
            if err := stream.Send(event); err != nil {
                return err
            }
        case <-stream.Context().Done():
            return stream.Context().Err()
        }
    }
}

// Client streaming
func (s *userServer) ImportUsers(stream pb.UserService_ImportUsersServer) error {
    var count int32

    for {
        req, err := stream.Recv()
        if err == io.EOF {
            return stream.SendAndClose(&pb.ImportUsersResponse{
                ImportedCount: count,
            })
        }
        if err != nil {
            return err
        }

        // Process each user
        _, err = s.CreateUser(stream.Context(), &pb.CreateUserRequest{
            Email: req.Email,
            Name:  req.Name,
        })
        if err == nil {
            count++
        }
    }
}

// Bidirectional streaming
func (s *userServer) Chat(stream pb.UserService_ChatServer) error {
    for {
        msg, err := stream.Recv()
        if err == io.EOF {
            return nil
        }
        if err != nil {
            return err
        }

        // Echo back (in real app, broadcast to other participants)
        response := &pb.ChatMessage{
            UserId:    msg.UserId,
            Content:   "Received: " + msg.Content,
            Timestamp: timestamppb.Now(),
        }

        if err := stream.Send(response); err != nil {
            return err
        }
    }
}

func main() {
    lis, _ := net.Listen("tcp", ":50051")

    srv := grpc.NewServer()
    pb.RegisterUserServiceServer(srv, &userServer{
        users:    make(map[string]*pb.User),
        watchers: make(map[chan *pb.UserEvent]struct{}),
    })

    // Graceful shutdown
    go func() {
        sigCh := make(chan os.Signal, 1)
        signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
        <-sigCh
        srv.GracefulStop()
    }()

    srv.Serve(lis)
}"""
                },
                "pitfalls": [
                    "Blocking in streaming without timeout",
                    "Not handling context cancellation",
                    "Forgetting UnimplementedServer for forward compatibility",
                    "Memory leaks from unclosed streams"
                ],
                "concepts": ["gRPC server", "Streaming RPCs", "Graceful shutdown", "Status codes"],
                "estimated_hours": "8-12"
            },
            {
                "id": 3,
                "name": "Interceptors & Middleware",
                "description": "Add cross-cutting concerns with interceptors.",
                "acceptance_criteria": [
                    "Logging interceptor",
                    "Authentication interceptor",
                    "Rate limiting",
                    "Error handling and recovery",
                    "Request/response validation"
                ],
                "hints": {
                    "level1": "UnaryInterceptor and StreamInterceptor. Chain multiple with grpc.ChainUnaryInterceptor.",
                    "level2": "Extract metadata (headers) with metadata.FromIncomingContext(ctx).",
                    "level3": """import (
    "google.golang.org/grpc"
    "google.golang.org/grpc/metadata"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// Logging interceptor
func LoggingInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    start := time.Now()

    // Extract metadata
    md, _ := metadata.FromIncomingContext(ctx)
    requestID := md.Get("x-request-id")

    log.Printf("[%s] %s started", requestID, info.FullMethod)

    resp, err := handler(ctx, req)

    duration := time.Since(start)
    code := codes.OK
    if err != nil {
        code = status.Code(err)
    }

    log.Printf("[%s] %s completed in %v with code %s",
        requestID, info.FullMethod, duration, code)

    return resp, err
}

// Auth interceptor
func AuthInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    // Skip auth for certain methods
    if info.FullMethod == "/userservice.v1.UserService/Health" {
        return handler(ctx, req)
    }

    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return nil, status.Error(codes.Unauthenticated, "missing metadata")
    }

    tokens := md.Get("authorization")
    if len(tokens) == 0 {
        return nil, status.Error(codes.Unauthenticated, "missing token")
    }

    // Validate token
    userID, err := validateToken(tokens[0])
    if err != nil {
        return nil, status.Error(codes.Unauthenticated, "invalid token")
    }

    // Add user to context
    ctx = context.WithValue(ctx, "user_id", userID)

    return handler(ctx, req)
}

// Recovery interceptor
func RecoveryInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
    defer func() {
        if r := recover(); r != nil {
            log.Printf("panic recovered: %v\\n%s", r, debug.Stack())
            err = status.Error(codes.Internal, "internal error")
        }
    }()

    return handler(ctx, req)
}

// Validation interceptor
func ValidationInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    // Check if request implements Validate()
    if v, ok := req.(interface{ Validate() error }); ok {
        if err := v.Validate(); err != nil {
            return nil, status.Error(codes.InvalidArgument, err.Error())
        }
    }

    return handler(ctx, req)
}

// Stream interceptor for server streaming
func StreamLoggingInterceptor(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
    start := time.Now()
    log.Printf("%s stream started", info.FullMethod)

    err := handler(srv, &wrappedStream{ss})

    log.Printf("%s stream ended after %v", info.FullMethod, time.Since(start))
    return err
}

type wrappedStream struct {
    grpc.ServerStream
}

func (w *wrappedStream) SendMsg(m interface{}) error {
    log.Printf("Sending message: %T", m)
    return w.ServerStream.SendMsg(m)
}

func (w *wrappedStream) RecvMsg(m interface{}) error {
    err := w.ServerStream.RecvMsg(m)
    log.Printf("Received message: %T", m)
    return err
}

// Server setup with chained interceptors
func main() {
    srv := grpc.NewServer(
        grpc.ChainUnaryInterceptor(
            RecoveryInterceptor,
            LoggingInterceptor,
            AuthInterceptor,
            ValidationInterceptor,
        ),
        grpc.ChainStreamInterceptor(
            StreamLoggingInterceptor,
        ),
    )
}"""
                },
                "pitfalls": [
                    "Interceptor order matters (auth before logging?)",
                    "Recovery interceptor not catching all panics",
                    "Context values lost in interceptor chain",
                    "Stream interceptors more complex than unary"
                ],
                "concepts": ["gRPC interceptors", "Middleware pattern", "Context propagation", "Error handling"],
                "estimated_hours": "6-10"
            },
            {
                "id": 4,
                "name": "Client & Testing",
                "description": "Build robust client and comprehensive tests.",
                "acceptance_criteria": [
                    "Client with retry and backoff",
                    "Connection pooling",
                    "Deadline/timeout handling",
                    "Unit tests with mocks",
                    "Integration tests"
                ],
                "hints": {
                    "level1": "Use grpc.WithDefaultServiceConfig for retry policy. Set deadlines with context.",
                    "level2": "Mock server for unit tests. Bufconn for in-process testing.",
                    "level3": """import (
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    "google.golang.org/grpc/backoff"
)

// Client with retry
func NewUserServiceClient(target string) (pb.UserServiceClient, error) {
    retryPolicy := `{
        "methodConfig": [{
            "name": [{"service": "userservice.v1.UserService"}],
            "retryPolicy": {
                "maxAttempts": 3,
                "initialBackoff": "0.1s",
                "maxBackoff": "1s",
                "backoffMultiplier": 2,
                "retryableStatusCodes": ["UNAVAILABLE", "RESOURCE_EXHAUSTED"]
            }
        }]
    }`

    conn, err := grpc.Dial(target,
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithDefaultServiceConfig(retryPolicy),
        grpc.WithConnectParams(grpc.ConnectParams{
            Backoff: backoff.Config{
                BaseDelay:  100 * time.Millisecond,
                Multiplier: 1.6,
                MaxDelay:   10 * time.Second,
            },
            MinConnectTimeout: 5 * time.Second,
        }),
    )
    if err != nil {
        return nil, err
    }

    return pb.NewUserServiceClient(conn), nil
}

// Client wrapper with deadline
type UserClient struct {
    client  pb.UserServiceClient
    timeout time.Duration
}

func (c *UserClient) GetUser(ctx context.Context, id string) (*pb.User, error) {
    ctx, cancel := context.WithTimeout(ctx, c.timeout)
    defer cancel()

    return c.client.GetUser(ctx, &pb.GetUserRequest{Id: id})
}

// Testing with bufconn
import (
    "google.golang.org/grpc/test/bufconn"
    "testing"
)

const bufSize = 1024 * 1024

var lis *bufconn.Listener

func init() {
    lis = bufconn.Listen(bufSize)
    srv := grpc.NewServer()
    pb.RegisterUserServiceServer(srv, &userServer{
        users: make(map[string]*pb.User),
    })
    go srv.Serve(lis)
}

func bufDialer(context.Context, string) (net.Conn, error) {
    return lis.Dial()
}

func TestCreateUser(t *testing.T) {
    ctx := context.Background()
    conn, err := grpc.DialContext(ctx, "bufnet",
        grpc.WithContextDialer(bufDialer),
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    )
    if err != nil {
        t.Fatalf("Failed to dial: %v", err)
    }
    defer conn.Close()

    client := pb.NewUserServiceClient(conn)

    resp, err := client.CreateUser(ctx, &pb.CreateUserRequest{
        Email: "test@example.com",
        Name:  "Test User",
    })

    if err != nil {
        t.Fatalf("CreateUser failed: %v", err)
    }

    if resp.Email != "test@example.com" {
        t.Errorf("Expected email test@example.com, got %s", resp.Email)
    }
}

// Mock for unit testing
type mockUserServiceClient struct {
    pb.UserServiceClient
    getUser func(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error)
}

func (m *mockUserServiceClient) GetUser(ctx context.Context, req *pb.GetUserRequest, opts ...grpc.CallOption) (*pb.User, error) {
    return m.getUser(ctx, req)
}

func TestBusinessLogic(t *testing.T) {
    mock := &mockUserServiceClient{
        getUser: func(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
            return &pb.User{Id: req.Id, Name: "Mocked"}, nil
        },
    }

    // Test code that uses the client
    service := NewMyService(mock)
    result, err := service.DoSomething("user-123")
    // assertions...
}"""
                },
                "pitfalls": [
                    "Not setting deadline causes hanging requests",
                    "Retry on non-idempotent methods causes duplicates",
                    "Connection not reused (creating new per request)",
                    "Tests not cleaning up connections"
                ],
                "concepts": ["gRPC client", "Retry policies", "Connection management", "Testing strategies"],
                "estimated_hours": "6-10"
            }
        ]
    },

    "circuit-breaker": {
        "id": "circuit-breaker",
        "name": "Circuit Breaker Pattern",
        "description": "Implement circuit breaker for resilient microservices communication.",
        "difficulty": "intermediate",
        "estimated_hours": "15-25",
        "prerequisites": ["Microservices basics", "Concurrency"],
        "languages": {
            "recommended": ["Go", "Java"],
            "also_possible": ["Python", "TypeScript"]
        },
        "resources": [
            {"name": "Circuit Breaker Pattern", "url": "https://martinfowler.com/bliki/CircuitBreaker.html", "type": "article"},
            {"name": "Hystrix", "url": "https://github.com/Netflix/Hystrix/wiki", "type": "reference"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Basic Circuit Breaker",
                "description": "Implement closed, open, half-open states.",
                "acceptance_criteria": [
                    "Closed: requests pass through normally",
                    "Open: requests fail immediately",
                    "Half-open: test requests allowed",
                    "State transitions based on failure threshold",
                    "Thread-safe implementation"
                ],
                "hints": {
                    "level1": "Track consecutive failures. Open circuit after threshold. Reset after timeout.",
                    "level2": "Half-open: allow one request, if success -> closed, if fail -> open again.",
                    "level3": """import (
    "sync"
    "time"
)

type State int

const (
    StateClosed State = iota
    StateOpen
    StateHalfOpen
)

type CircuitBreaker struct {
    name          string
    maxFailures   int
    timeout       time.Duration
    halfOpenMax   int

    state         State
    failures      int
    successes     int
    lastFailure   time.Time
    halfOpenCount int
    mu            sync.RWMutex
}

func NewCircuitBreaker(name string, maxFailures int, timeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        name:        name,
        maxFailures: maxFailures,
        timeout:     timeout,
        halfOpenMax: 3,
        state:       StateClosed,
    }
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    if !cb.allowRequest() {
        return ErrCircuitOpen
    }

    err := fn()

    cb.recordResult(err)
    return err
}

func (cb *CircuitBreaker) allowRequest() bool {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    switch cb.state {
    case StateClosed:
        return true

    case StateOpen:
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = StateHalfOpen
            cb.halfOpenCount = 0
            cb.successes = 0
            return true
        }
        return false

    case StateHalfOpen:
        if cb.halfOpenCount < cb.halfOpenMax {
            cb.halfOpenCount++
            return true
        }
        return false
    }

    return false
}

func (cb *CircuitBreaker) recordResult(err error) {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    if err != nil {
        cb.onFailure()
    } else {
        cb.onSuccess()
    }
}

func (cb *CircuitBreaker) onFailure() {
    cb.failures++
    cb.lastFailure = time.Now()

    switch cb.state {
    case StateClosed:
        if cb.failures >= cb.maxFailures {
            cb.state = StateOpen
        }
    case StateHalfOpen:
        cb.state = StateOpen
    }
}

func (cb *CircuitBreaker) onSuccess() {
    switch cb.state {
    case StateClosed:
        cb.failures = 0
    case StateHalfOpen:
        cb.successes++
        if cb.successes >= cb.halfOpenMax {
            cb.state = StateClosed
            cb.failures = 0
        }
    }
}

var ErrCircuitOpen = errors.New("circuit breaker is open")"""
                },
                "pitfalls": [
                    "Race conditions without proper locking",
                    "Not resetting failure count on success",
                    "Half-open allows too many requests",
                    "Timer not being reset properly"
                ],
                "concepts": ["Circuit breaker states", "Failure detection", "Recovery testing", "Thread safety"],
                "estimated_hours": "5-8"
            },
            {
                "id": 2,
                "name": "Advanced Features",
                "description": "Add sliding window, metrics, and fallbacks.",
                "acceptance_criteria": [
                    "Sliding window for failure rate calculation",
                    "Configurable error classification",
                    "Fallback function support",
                    "Metrics and observability",
                    "Bulkhead (concurrency limit)"
                ],
                "hints": {
                    "level1": "Use ring buffer for sliding window. Classify errors (timeout vs 5xx).",
                    "level2": "Fallback: return cached/default value. Bulkhead: limit concurrent calls.",
                    "level3": """type SlidingWindow struct {
    size    int
    buckets []bucket
    current int
    mu      sync.Mutex
}

type bucket struct {
    successes int
    failures  int
    timestamp time.Time
}

func NewSlidingWindow(size int, bucketDuration time.Duration) *SlidingWindow {
    sw := &SlidingWindow{
        size:    size,
        buckets: make([]bucket, size),
    }

    // Rotate buckets periodically
    go func() {
        ticker := time.NewTicker(bucketDuration)
        for range ticker.C {
            sw.rotate()
        }
    }()

    return sw
}

func (sw *SlidingWindow) rotate() {
    sw.mu.Lock()
    defer sw.mu.Unlock()

    sw.current = (sw.current + 1) % sw.size
    sw.buckets[sw.current] = bucket{timestamp: time.Now()}
}

func (sw *SlidingWindow) RecordSuccess() {
    sw.mu.Lock()
    sw.buckets[sw.current].successes++
    sw.mu.Unlock()
}

func (sw *SlidingWindow) RecordFailure() {
    sw.mu.Lock()
    sw.buckets[sw.current].failures++
    sw.mu.Unlock()
}

func (sw *SlidingWindow) FailureRate() float64 {
    sw.mu.Lock()
    defer sw.mu.Unlock()

    var total, failures int
    for _, b := range sw.buckets {
        total += b.successes + b.failures
        failures += b.failures
    }

    if total == 0 {
        return 0
    }
    return float64(failures) / float64(total)
}

// Enhanced circuit breaker
type EnhancedCircuitBreaker struct {
    *CircuitBreaker
    window      *SlidingWindow
    fallback    func() (interface{}, error)
    bulkhead    chan struct{}
    classifier  func(error) bool  // true = should count as failure
    metrics     *Metrics
}

func (cb *EnhancedCircuitBreaker) ExecuteWithFallback(fn func() (interface{}, error)) (interface{}, error) {
    // Check bulkhead (concurrency limit)
    select {
    case cb.bulkhead <- struct{}{}:
        defer func() { <-cb.bulkhead }()
    default:
        cb.metrics.BulkheadRejected.Inc()
        return cb.executeFallback(ErrBulkheadFull)
    }

    if !cb.allowRequest() {
        cb.metrics.CircuitRejected.Inc()
        return cb.executeFallback(ErrCircuitOpen)
    }

    start := time.Now()
    result, err := fn()
    duration := time.Since(start)

    cb.metrics.RequestDuration.Observe(duration.Seconds())

    // Classify error
    if err != nil && cb.classifier(err) {
        cb.window.RecordFailure()
        cb.metrics.Failures.Inc()

        // Check if we should open circuit
        if cb.window.FailureRate() > 0.5 {  // 50% threshold
            cb.openCircuit()
        }

        return cb.executeFallback(err)
    }

    cb.window.RecordSuccess()
    cb.metrics.Successes.Inc()
    return result, nil
}

func (cb *EnhancedCircuitBreaker) executeFallback(originalErr error) (interface{}, error) {
    if cb.fallback != nil {
        cb.metrics.FallbackExecuted.Inc()
        return cb.fallback()
    }
    return nil, originalErr
}

// Error classifier
func DefaultClassifier(err error) bool {
    // Ignore client errors, count server errors and timeouts
    var netErr net.Error
    if errors.As(err, &netErr) && netErr.Timeout() {
        return true
    }

    var httpErr *HTTPError
    if errors.As(err, &httpErr) {
        return httpErr.StatusCode >= 500
    }

    return true
}"""
                },
                "pitfalls": [
                    "Sliding window bucket rotation timing issues",
                    "Fallback also failing (need fallback for fallback)",
                    "Bulkhead too small causes unnecessary rejections",
                    "Error classifier too aggressive"
                ],
                "concepts": ["Sliding window", "Fallback patterns", "Bulkhead pattern", "Error classification"],
                "estimated_hours": "6-10"
            },
            {
                "id": 3,
                "name": "Integration & Testing",
                "description": "Integrate with HTTP/gRPC clients and test chaos scenarios.",
                "acceptance_criteria": [
                    "HTTP client wrapper with circuit breaker",
                    "gRPC interceptor with circuit breaker",
                    "Per-service circuit breakers",
                    "Chaos testing (inject failures)",
                    "Dashboard for circuit states"
                ],
                "hints": {
                    "level1": "Wrap http.Client.Do(). Key circuit breaker by host or service name.",
                    "level2": "Chaos: randomly fail N% of requests. Test state transitions.",
                    "level3": """// HTTP client with circuit breaker
type ResilientHTTPClient struct {
    client   *http.Client
    breakers map[string]*EnhancedCircuitBreaker
    mu       sync.RWMutex
}

func (c *ResilientHTTPClient) getBreaker(host string) *EnhancedCircuitBreaker {
    c.mu.RLock()
    cb, ok := c.breakers[host]
    c.mu.RUnlock()

    if ok {
        return cb
    }

    c.mu.Lock()
    defer c.mu.Unlock()

    // Double check
    if cb, ok = c.breakers[host]; ok {
        return cb
    }

    cb = NewEnhancedCircuitBreaker(host, Config{
        MaxFailures:   5,
        Timeout:       30 * time.Second,
        HalfOpenMax:   3,
        BulkheadSize:  100,
    })
    c.breakers[host] = cb
    return cb
}

func (c *ResilientHTTPClient) Do(req *http.Request) (*http.Response, error) {
    cb := c.getBreaker(req.Host)

    result, err := cb.ExecuteWithFallback(func() (interface{}, error) {
        return c.client.Do(req)
    })

    if err != nil {
        return nil, err
    }
    return result.(*http.Response), nil
}

// gRPC interceptor
func CircuitBreakerInterceptor(breakers map[string]*EnhancedCircuitBreaker) grpc.UnaryClientInterceptor {
    return func(ctx context.Context, method string, req, reply interface{},
                cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {

        // Get service name from method
        service := strings.Split(method, "/")[1]
        cb := breakers[service]

        _, err := cb.ExecuteWithFallback(func() (interface{}, error) {
            return nil, invoker(ctx, method, req, reply, cc, opts...)
        })

        return err
    }
}

// Chaos testing
type ChaosMiddleware struct {
    failureRate float64
    latencyMs   int
    enabled     bool
}

func (c *ChaosMiddleware) Wrap(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if !c.enabled {
            next.ServeHTTP(w, r)
            return
        }

        // Inject latency
        if c.latencyMs > 0 {
            time.Sleep(time.Duration(c.latencyMs) * time.Millisecond)
        }

        // Inject failures
        if rand.Float64() < c.failureRate {
            http.Error(w, "Chaos: injected failure", http.StatusInternalServerError)
            return
        }

        next.ServeHTTP(w, r)
    })
}

// Tests
func TestCircuitOpensOnFailures(t *testing.T) {
    cb := NewCircuitBreaker("test", 3, 1*time.Second)

    // 3 failures should open circuit
    for i := 0; i < 3; i++ {
        cb.Execute(func() error {
            return errors.New("fail")
        })
    }

    // Next call should fail immediately
    err := cb.Execute(func() error {
        t.Fatal("Should not execute")
        return nil
    })

    if !errors.Is(err, ErrCircuitOpen) {
        t.Errorf("Expected ErrCircuitOpen, got %v", err)
    }
}

func TestCircuitRecovery(t *testing.T) {
    cb := NewCircuitBreaker("test", 1, 100*time.Millisecond)

    // Open circuit
    cb.Execute(func() error { return errors.New("fail") })

    // Wait for timeout
    time.Sleep(150 * time.Millisecond)

    // Should be half-open now, success should close
    err := cb.Execute(func() error { return nil })
    if err != nil {
        t.Errorf("Expected success, got %v", err)
    }
}"""
                },
                "pitfalls": [
                    "Chaos testing in production without safeguards",
                    "Circuit breaker per-request instead of per-service",
                    "Not exposing circuit state for debugging",
                    "Test flakiness due to timing"
                ],
                "concepts": ["Client integration", "Chaos engineering", "Per-service isolation", "Observability"],
                "estimated_hours": "6-10"
            }
        ]
    },

    "distributed-tracing": {
        "id": "distributed-tracing",
        "name": "Distributed Tracing System",
        "description": "Build a distributed tracing system to track requests across microservices.",
        "difficulty": "advanced",
        "estimated_hours": "40-60",
        "prerequisites": ["Microservices", "Networking", "Data storage"],
        "languages": {
            "recommended": ["Go", "Java"],
            "also_possible": ["Rust", "Python"]
        },
        "resources": [
            {"name": "OpenTelemetry", "url": "https://opentelemetry.io/docs/", "type": "documentation"},
            {"name": "Jaeger Architecture", "url": "https://www.jaegertracing.io/docs/architecture/", "type": "documentation"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Trace Context & Propagation",
                "description": "Implement trace/span IDs and context propagation.",
                "acceptance_criteria": [
                    "Generate unique trace and span IDs",
                    "Propagate via HTTP headers (W3C Trace Context)",
                    "Propagate via gRPC metadata",
                    "Parent-child span relationships",
                    "Context injection and extraction"
                ],
                "hints": {
                    "level1": "Trace ID = request ID across services. Span ID = operation within service.",
                    "level2": "W3C traceparent: version-traceid-spanid-flags (00-{32hex}-{16hex}-01).",
                    "level3": """import (
    "crypto/rand"
    "encoding/hex"
    "context"
    "net/http"
)

type TraceID [16]byte
type SpanID [8]byte

func (t TraceID) String() string { return hex.EncodeToString(t[:]) }
func (s SpanID) String() string  { return hex.EncodeToString(s[:]) }

func NewTraceID() TraceID {
    var id TraceID
    rand.Read(id[:])
    return id
}

func NewSpanID() SpanID {
    var id SpanID
    rand.Read(id[:])
    return id
}

type SpanContext struct {
    TraceID    TraceID
    SpanID     SpanID
    ParentID   SpanID
    TraceFlags byte
}

// W3C Trace Context format
const (
    traceparentHeader = "traceparent"
    tracestateHeader  = "tracestate"
)

type Propagator struct{}

func (p *Propagator) Inject(ctx context.Context, carrier http.Header) {
    sc := SpanContextFromContext(ctx)
    if sc == nil {
        return
    }

    // Format: 00-{trace-id}-{span-id}-{trace-flags}
    traceparent := fmt.Sprintf("00-%s-%s-%02x",
        sc.TraceID.String(),
        sc.SpanID.String(),
        sc.TraceFlags,
    )

    carrier.Set(traceparentHeader, traceparent)
}

func (p *Propagator) Extract(ctx context.Context, carrier http.Header) context.Context {
    traceparent := carrier.Get(traceparentHeader)
    if traceparent == "" {
        return ctx
    }

    // Parse: 00-{trace-id}-{span-id}-{trace-flags}
    parts := strings.Split(traceparent, "-")
    if len(parts) != 4 || parts[0] != "00" {
        return ctx
    }

    traceID, _ := hex.DecodeString(parts[1])
    spanID, _ := hex.DecodeString(parts[2])
    flags, _ := strconv.ParseUint(parts[3], 16, 8)

    var tid TraceID
    var sid SpanID
    copy(tid[:], traceID)
    copy(sid[:], spanID)

    sc := &SpanContext{
        TraceID:    tid,
        ParentID:   sid,  // Incoming span becomes parent
        SpanID:     NewSpanID(),  // Create new span for this service
        TraceFlags: byte(flags),
    }

    return ContextWithSpanContext(ctx, sc)
}

// Context helpers
type spanContextKey struct{}

func ContextWithSpanContext(ctx context.Context, sc *SpanContext) context.Context {
    return context.WithValue(ctx, spanContextKey{}, sc)
}

func SpanContextFromContext(ctx context.Context) *SpanContext {
    sc, _ := ctx.Value(spanContextKey{}).(*SpanContext)
    return sc
}

// HTTP middleware
func TracingMiddleware(next http.Handler) http.Handler {
    propagator := &Propagator{}

    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Extract or create trace context
        ctx := propagator.Extract(r.Context(), r.Header)

        if SpanContextFromContext(ctx) == nil {
            // No incoming trace, start new one
            sc := &SpanContext{
                TraceID:    NewTraceID(),
                SpanID:     NewSpanID(),
                TraceFlags: 0x01,  // Sampled
            }
            ctx = ContextWithSpanContext(ctx, sc)
        }

        r = r.WithContext(ctx)
        next.ServeHTTP(w, r)
    })
}"""
                },
                "pitfalls": [
                    "Trace ID not propagated to async operations",
                    "Invalid trace context parsing crashes",
                    "Generating new trace ID when should continue existing",
                    "Not handling malformed traceparent"
                ],
                "concepts": ["Trace context", "W3C Trace Context spec", "Context propagation", "Span hierarchy"],
                "estimated_hours": "8-12"
            },
            {
                "id": 2,
                "name": "Span Recording",
                "description": "Record span data with timing, tags, and logs.",
                "acceptance_criteria": [
                    "Start/end spans with timing",
                    "Add tags/attributes to spans",
                    "Add log events within spans",
                    "Record errors and exceptions",
                    "Span status (OK, Error)"
                ],
                "hints": {
                    "level1": "Span = name + start time + end time + attributes + events.",
                    "level2": "Buffer spans in memory, flush to collector periodically or on threshold.",
                    "level3": """type Span struct {
    TraceID     TraceID
    SpanID      SpanID
    ParentID    SpanID
    Name        string
    StartTime   time.Time
    EndTime     time.Time
    Status      SpanStatus
    Attributes  map[string]interface{}
    Events      []SpanEvent
    mu          sync.Mutex
}

type SpanStatus struct {
    Code    StatusCode
    Message string
}

type StatusCode int

const (
    StatusUnset StatusCode = iota
    StatusOK
    StatusError
)

type SpanEvent struct {
    Name       string
    Timestamp  time.Time
    Attributes map[string]interface{}
}

type Tracer struct {
    serviceName string
    exporter    SpanExporter
    spans       chan *Span
}

func NewTracer(serviceName string, exporter SpanExporter) *Tracer {
    t := &Tracer{
        serviceName: serviceName,
        exporter:    exporter,
        spans:       make(chan *Span, 1000),
    }

    // Background exporter
    go t.exportLoop()

    return t
}

func (t *Tracer) StartSpan(ctx context.Context, name string) (context.Context, *Span) {
    parentSC := SpanContextFromContext(ctx)

    span := &Span{
        Name:       name,
        StartTime:  time.Now(),
        Attributes: make(map[string]interface{}),
    }

    if parentSC != nil {
        span.TraceID = parentSC.TraceID
        span.ParentID = parentSC.SpanID
    } else {
        span.TraceID = NewTraceID()
    }
    span.SpanID = NewSpanID()

    // Add default attributes
    span.SetAttribute("service.name", t.serviceName)

    sc := &SpanContext{
        TraceID:  span.TraceID,
        SpanID:   span.SpanID,
        ParentID: span.ParentID,
    }

    return ContextWithSpanContext(ctx, sc), span
}

func (s *Span) SetAttribute(key string, value interface{}) {
    s.mu.Lock()
    s.Attributes[key] = value
    s.mu.Unlock()
}

func (s *Span) AddEvent(name string, attrs map[string]interface{}) {
    s.mu.Lock()
    s.Events = append(s.Events, SpanEvent{
        Name:       name,
        Timestamp:  time.Now(),
        Attributes: attrs,
    })
    s.mu.Unlock()
}

func (s *Span) RecordError(err error) {
    s.mu.Lock()
    s.Status = SpanStatus{Code: StatusError, Message: err.Error()}
    s.Events = append(s.Events, SpanEvent{
        Name:      "exception",
        Timestamp: time.Now(),
        Attributes: map[string]interface{}{
            "exception.type":    fmt.Sprintf("%T", err),
            "exception.message": err.Error(),
        },
    })
    s.mu.Unlock()
}

func (s *Span) End() {
    s.mu.Lock()
    s.EndTime = time.Now()
    s.mu.Unlock()

    // Send to exporter
    tracer.spans <- s
}

func (t *Tracer) exportLoop() {
    batch := make([]*Span, 0, 100)
    ticker := time.NewTicker(5 * time.Second)

    for {
        select {
        case span := <-t.spans:
            batch = append(batch, span)
            if len(batch) >= 100 {
                t.exporter.ExportSpans(batch)
                batch = batch[:0]
            }

        case <-ticker.C:
            if len(batch) > 0 {
                t.exporter.ExportSpans(batch)
                batch = batch[:0]
            }
        }
    }
}"""
                },
                "pitfalls": [
                    "Span End() never called (memory leak)",
                    "Too many attributes causes large payloads",
                    "High cardinality attribute values",
                    "Blocking on export channel"
                ],
                "concepts": ["Span recording", "Attributes and events", "Error recording", "Batch export"],
                "estimated_hours": "8-12"
            },
            {
                "id": 3,
                "name": "Collector & Storage",
                "description": "Build collector to receive, process, and store traces.",
                "acceptance_criteria": [
                    "Receive spans via HTTP/gRPC",
                    "Process and enrich spans",
                    "Store in time-series database",
                    "Sampling strategies",
                    "Tail-based sampling"
                ],
                "hints": {
                    "level1": "Collector receives spans, buffers, writes to storage. Use Cassandra/ClickHouse for traces.",
                    "level2": "Tail sampling: wait for trace completion, sample based on latency/errors.",
                    "level3": """// Collector service
type Collector struct {
    storage   TraceStorage
    processor SpanProcessor
    sampler   Sampler
}

type SpanProcessor interface {
    Process(span *Span) *Span
}

type EnrichmentProcessor struct {
    geoIP   *geoip.DB
}

func (p *EnrichmentProcessor) Process(span *Span) *Span {
    // Add derived attributes
    if ip, ok := span.Attributes["http.client_ip"].(string); ok {
        if location, err := p.geoIP.Lookup(ip); err == nil {
            span.SetAttribute("geo.country", location.Country)
            span.SetAttribute("geo.city", location.City)
        }
    }

    // Calculate duration
    span.SetAttribute("duration_ms", span.EndTime.Sub(span.StartTime).Milliseconds())

    return span
}

// Sampling
type Sampler interface {
    ShouldSample(span *Span) bool
}

type ProbabilisticSampler struct {
    rate float64
}

func (s *ProbabilisticSampler) ShouldSample(span *Span) bool {
    // Use trace ID for deterministic sampling
    hash := fnv.New64a()
    hash.Write(span.TraceID[:])
    return float64(hash.Sum64()%10000)/10000 < s.rate
}

// Tail-based sampling - sample complete traces based on their properties
type TailSampler struct {
    pending     map[TraceID]*pendingTrace
    sampleRules []SampleRule
    mu          sync.Mutex
}

type pendingTrace struct {
    spans       []*Span
    firstSeen   time.Time
    rootSpan    *Span
}

type SampleRule struct {
    Name      string
    Condition func(trace *pendingTrace) bool
    Rate      float64
}

func (s *TailSampler) AddSpan(span *Span) {
    s.mu.Lock()
    defer s.mu.Unlock()

    pt, ok := s.pending[span.TraceID]
    if !ok {
        pt = &pendingTrace{firstSeen: time.Now()}
        s.pending[span.TraceID] = pt
    }

    pt.spans = append(pt.spans, span)

    if span.ParentID == (SpanID{}) {
        pt.rootSpan = span
    }
}

func (s *TailSampler) Evaluate() []*Span {
    s.mu.Lock()
    defer s.mu.Unlock()

    var sampled []*Span

    for traceID, pt := range s.pending {
        // Wait for trace to complete (no new spans for 10s)
        if time.Since(pt.firstSeen) < 10*time.Second {
            continue
        }

        // Evaluate sampling rules
        shouldSample := false
        for _, rule := range s.sampleRules {
            if rule.Condition(pt) && rand.Float64() < rule.Rate {
                shouldSample = true
                break
            }
        }

        if shouldSample {
            sampled = append(sampled, pt.spans...)
        }

        delete(s.pending, traceID)
    }

    return sampled
}

// Example rules
func ErrorTraceRule(pt *pendingTrace) bool {
    for _, span := range pt.spans {
        if span.Status.Code == StatusError {
            return true
        }
    }
    return false
}

func SlowTraceRule(threshold time.Duration) func(*pendingTrace) bool {
    return func(pt *pendingTrace) bool {
        if pt.rootSpan == nil {
            return false
        }
        return pt.rootSpan.EndTime.Sub(pt.rootSpan.StartTime) > threshold
    }
}"""
                },
                "pitfalls": [
                    "Memory exhaustion holding pending traces",
                    "Sampling bias toward short traces",
                    "Clock skew affecting duration calculations",
                    "Hot partition in storage by trace ID"
                ],
                "concepts": ["Span collection", "Enrichment", "Head vs tail sampling", "Trace storage"],
                "estimated_hours": "12-18"
            },
            {
                "id": 4,
                "name": "Query & Visualization",
                "description": "Query traces and visualize in timeline view.",
                "acceptance_criteria": [
                    "Search traces by service, operation, tags",
                    "Time range queries",
                    "Trace timeline visualization",
                    "Service dependency graph",
                    "Latency percentiles"
                ],
                "hints": {
                    "level1": "Index by service, operation, and time. Use ClickHouse for fast queries.",
                    "level2": "Dependency graph: aggregate parent -> child service pairs from spans.",
                    "level3": """// Query API
type TraceQuery struct {
    Service     string
    Operation   string
    Tags        map[string]string
    MinDuration time.Duration
    MaxDuration time.Duration
    StartTime   time.Time
    EndTime     time.Time
    Limit       int
}

type TraceQueryService struct {
    storage TraceStorage
}

func (s *TraceQueryService) Search(q TraceQuery) ([]*Trace, error) {
    // Build query
    query := `
        SELECT trace_id, span_id, parent_id, name, start_time, end_time, attributes
        FROM spans
        WHERE start_time >= ? AND start_time <= ?
    `
    args := []interface{}{q.StartTime, q.EndTime}

    if q.Service != "" {
        query += " AND service_name = ?"
        args = append(args, q.Service)
    }

    if q.Operation != "" {
        query += " AND name = ?"
        args = append(args, q.Operation)
    }

    if q.MinDuration > 0 {
        query += " AND duration_ms >= ?"
        args = append(args, q.MinDuration.Milliseconds())
    }

    for key, value := range q.Tags {
        query += fmt.Sprintf(" AND attributes['%s'] = ?", key)
        args = append(args, value)
    }

    query += " ORDER BY start_time DESC LIMIT ?"
    args = append(args, q.Limit)

    // Execute and group by trace
    rows, _ := s.storage.Query(query, args...)
    return s.groupSpansToTraces(rows)
}

// Dependency graph
type ServiceDependency struct {
    Parent      string
    Child       string
    CallCount   int64
    ErrorCount  int64
    AvgDuration float64
}

func (s *TraceQueryService) GetDependencies(startTime, endTime time.Time) ([]ServiceDependency, error) {
    query := `
        SELECT
            parent.service_name as parent,
            child.service_name as child,
            count(*) as call_count,
            countIf(child.status_code = 'ERROR') as error_count,
            avg(child.duration_ms) as avg_duration
        FROM spans child
        JOIN spans parent ON child.parent_id = parent.span_id
            AND child.trace_id = parent.trace_id
        WHERE child.start_time >= ? AND child.start_time <= ?
            AND parent.service_name != child.service_name
        GROUP BY parent.service_name, child.service_name
    `

    // Return for D3.js force-directed graph
    return s.storage.Query(query, startTime, endTime)
}

// Timeline data for visualization
type TimelineSpan struct {
    SpanID    string        `json:"spanId"`
    ParentID  string        `json:"parentId"`
    Name      string        `json:"name"`
    Service   string        `json:"service"`
    StartTime int64         `json:"startTime"` // microseconds from trace start
    Duration  int64         `json:"duration"`  // microseconds
    Status    string        `json:"status"`
    Tags      []TagPair     `json:"tags"`
    Logs      []LogEntry    `json:"logs"`
    Depth     int           `json:"depth"`
    Children  []*TimelineSpan `json:"children,omitempty"`
}

func (s *TraceQueryService) GetTraceTimeline(traceID string) (*TimelineSpan, error) {
    spans, _ := s.storage.GetSpans(traceID)

    // Build tree
    spanMap := make(map[string]*TimelineSpan)
    var root *TimelineSpan

    for _, span := range spans {
        ts := &TimelineSpan{
            SpanID:   span.SpanID.String(),
            ParentID: span.ParentID.String(),
            Name:     span.Name,
            Service:  span.Attributes["service.name"].(string),
            Duration: span.EndTime.Sub(span.StartTime).Microseconds(),
            Status:   span.Status.Code.String(),
        }
        spanMap[ts.SpanID] = ts

        if span.ParentID == (SpanID{}) {
            root = ts
        }
    }

    // Link parents
    for _, ts := range spanMap {
        if ts.ParentID != "" {
            if parent, ok := spanMap[ts.ParentID]; ok {
                parent.Children = append(parent.Children, ts)
            }
        }
    }

    // Calculate relative times and depths
    if root != nil {
        s.calculateDepths(root, 0, root.StartTime)
    }

    return root, nil
}"""
                },
                "pitfalls": [
                    "Query without time range scans entire database",
                    "Deep traces cause stack overflow in tree building",
                    "Timezone issues in timestamp display",
                    "Missing spans leave gaps in visualization"
                ],
                "concepts": ["Trace querying", "Service dependencies", "Timeline visualization", "Aggregations"],
                "estimated_hours": "12-16"
            }
        ]
    }
}

# Load YAML
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# Check if backend domain exists, if not create it
backend_domain = None
for domain in data['domains']:
    if domain['id'] == 'app-dev':
        backend_domain = domain
        break

# Add projects to app-dev domain (backend projects fit there)
if backend_domain:
    # Add to beginner
    existing_ids = [p['id'] for p in backend_domain['projects'].get('beginner', [])]
    if 'rest-api-design' not in existing_ids:
        backend_domain['projects']['beginner'].append({
            'id': 'rest-api-design',
            'name': 'Production REST API',
            'detailed': True
        })

    # Add to intermediate
    existing_ids = [p['id'] for p in backend_domain['projects'].get('intermediate', [])]
    for proj_id in ['grpc-service', 'circuit-breaker']:
        if proj_id not in existing_ids:
            backend_domain['projects']['intermediate'].append({
                'id': proj_id,
                'name': backend_projects[proj_id]['name'],
                'detailed': True
            })

    # Add to advanced
    existing_ids = [p['id'] for p in backend_domain['projects'].get('advanced', [])]
    for proj_id in ['api-gateway', 'distributed-tracing']:
        if proj_id not in existing_ids:
            backend_domain['projects']['advanced'].append({
                'id': proj_id,
                'name': backend_projects[proj_id]['name'],
                'detailed': True
            })

# Add expert_projects
expert_projects = data.get('expert_projects', {})
for proj_id, proj_data in backend_projects.items():
    expert_projects[proj_id] = proj_data
    print(f"Added: {proj_id}")

data['expert_projects'] = expert_projects

# Save
with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nTotal expert_projects: {len(expert_projects)}")

# AUDIT & FIX: multi-tenant-saas

## CRITIQUE
- The project is well-structured with a logical milestone progression: data model -> request context -> RLS -> customization -> billing.
- M1 AC 'All application data tables include a tenant_id column' — good, but no specification of what the 'application data tables' are. The project needs a sample domain (e.g., a simple task management app) to make this concrete.
- M1 AC 'Tenant deletion soft-deletes all data' — no specification for how soft delete cascades (does deleting a tenant mark ALL their rows as deleted? Or just the tenant record?).
- M2 AC 'Database queries are automatically filtered by the current tenant context without manual WHERE clauses' — this is the correct approach but technically challenging. Should specify mechanism (ORM global filter, PostgreSQL RLS, or custom query rewriter).
- M2 and M3 overlap: M2 says 'Cross-tenant data access attempts are blocked' and M3 implements RLS. The relationship should be clearer — M2 does application-level filtering, M3 adds database-level RLS as defense-in-depth.
- M2 pitfall about thread-local context leaking is critical for async frameworks (asyncio, Go goroutines) but the AC doesn't test this scenario.
- M3 AC 'No data leakage occurs when multiple tenants query the same tables simultaneously' — how is this tested? Need a specific test scenario.
- M3 deliverable says 'Migration bypass mechanism that disables RLS for schema migration scripts' — this is a security risk if not carefully scoped. Should specify that bypass only applies to DDL migrations, not data migrations.
- M4 AC about feature flags and branding is reasonable but the scope is very broad for a single milestone.
- M5 AC 'Usage-based billing computes charges from metered events' — no specification of how metering handles clock skew or out-of-order events.
- M5 mentions Stripe integration but no AC verifies webhook handling (idempotency, signature verification).
- No mention of tenant data export or GDPR-related data deletion requirements, which is a major gap for SaaS.
- No mention of rate limiting per tenant, which is essential for shared infrastructure.
- Overall the project is solid but would benefit from a concrete sample domain to ground the abstractions.

## FIXED YAML
```yaml
id: multi-tenant-saas
name: Multi-tenant SaaS Backend
description: >-
  Multi-tenant SaaS backend with tenant isolation via row-level security,
  request-scoped tenant context, per-tenant configuration, and usage-based
  billing. Built around a sample task management domain.
difficulty: advanced
estimated_hours: 65
essence: >-
  Request-scoped tenant context propagation through middleware layers coupled
  with defense-in-depth isolation — application-level ORM filters backed by
  PostgreSQL row-level security policies — enforcing data boundaries while
  managing per-tenant configuration, usage metering, and billing integration.
why_important: >-
  Building this teaches production SaaS architecture patterns critical for any
  backend engineer working on B2B platforms, combining database security,
  distributed systems design, and business logic that directly impacts revenue
  and compliance.
learning_outcomes:
  - Design scalable multi-tenant database schemas with tenant isolation
  - Implement request-scoped tenant context with automatic query filtering
  - Configure PostgreSQL row-level security as defense-in-depth
  - Build per-tenant feature flags and configuration management
  - Implement usage metering and billing integration with Stripe
  - Handle tenant lifecycle (provisioning, suspension, deletion)
skills:
  - Multi-tenancy patterns
  - Row-level security
  - Tenant isolation
  - Database design
  - Request context propagation
  - Billing integration
tags:
  - advanced
  - backend
  - isolation
  - multi-tenancy
  - row-level-security
  - security
architecture_doc: architecture-docs/multi-tenant-saas/index.md
languages:
  recommended:
    - Python
    - Go
    - Java
  also_possible: []
resources:
  - name: PostgreSQL Row-Level Security
    url: https://www.postgresql.org/docs/current/ddl-rowsecurity.html
    type: documentation
  - name: AWS Multi-Tenant Architectures Guidance
    url: https://aws.amazon.com/solutions/guidance/multi-tenant-architectures-on-aws/
    type: documentation
  - name: Designing Postgres for Multi-Tenancy
    url: https://www.crunchydata.com/blog/designing-your-postgres-database-for-multi-tenancy
    type: article
  - name: SaaS Architecture Fundamentals
    url: https://docs.aws.amazon.com/whitepapers/latest/saas-architecture-fundamentals/re-defining-multi-tenancy.html
    type: documentation
prerequisites:
  - type: skill
    name: REST API development
  - type: skill
    name: PostgreSQL database design
  - type: skill
    name: Authentication (JWT)
milestones:
  - id: multi-tenant-saas-m1
    name: Tenant Data Model & Provisioning
    description: >-
      Design the multi-tenant database schema with tenant_id on all data
      tables, tenant provisioning, and a sample domain (task management) to
      make isolation concrete and testable.
    estimated_hours: 13
    concepts:
      - "Shared database, shared schema: all tenants in same tables, distinguished by tenant_id"
      - "Composite indexes with tenant_id prefix for efficient tenant-scoped queries"
      - "Soft delete: mark records as deleted without physical removal"
      - "Sample domain: tenants, users, projects, tasks — all scoped by tenant_id"
    skills:
      - Schema design with tenant isolation
      - Foreign key and index design
      - Tenant provisioning workflow
    acceptance_criteria:
      - "Tenant table has columns: id (UUID), name, slug (unique), subdomain (unique), plan (free/pro/enterprise), settings (JSONB), created_at, deleted_at (nullable for soft delete)"
      - "Sample domain tables (users, projects, tasks) all include a tenant_id column with NOT NULL constraint and foreign key to tenant table"
      - "Composite indexes on (tenant_id, id) and (tenant_id, created_at) exist on all tenant-scoped tables; EXPLAIN ANALYZE confirms index scan for tenant-scoped queries"
      - "Tenant provisioning API creates a tenant record with default settings, creates an admin user, and returns credentials — all within a single database transaction"
      - "Tenant soft delete sets deleted_at timestamp on tenant and all associated data records; soft-deleted records are excluded from all queries by default"
      - "Slug and subdomain uniqueness is enforced at the database level (unique constraint); attempting to create a duplicate returns 409 Conflict"
      - "Junction tables (e.g., project_members) include tenant_id to prevent cross-tenant joins"
    pitfalls:
      - Missing tenant_id on junction/association tables breaking isolation
      - Forgetting composite indexes causing full table scans when querying within a tenant
      - UUID primary keys without considering B-tree index fragmentation (consider ULIDs)
      - Circular foreign key dependencies when modeling tenant-user relationships
      - Not including tenant_id in all unique constraints (e.g., task name unique per tenant, not globally)
    deliverables:
      - Tenant table with plan, settings JSONB, slug, subdomain
      - Sample domain tables (users, projects, tasks) with tenant_id FK
      - Composite indexes with tenant_id prefix on all data tables
      - Tenant provisioning API (create tenant + admin user in transaction)
      - Soft delete implementation on tenant and all data tables
      - Database migration scripts for the complete schema

  - id: multi-tenant-saas-m2
    name: Request Context & Application-Level Isolation
    description: >-
      Implement request-scoped tenant context injection and automatic ORM-level
      query filtering as the first layer of tenant isolation.
    estimated_hours: 13
    concepts:
      - "Tenant resolution: determine tenant from subdomain, header, or JWT claim"
      - "Request-scoped context: store tenant_id for the duration of the request"
      - "ORM global filter: automatically append WHERE tenant_id = X to all queries"
      - "Context propagation: ensure tenant context reaches background jobs and async tasks"
    skills:
      - Middleware implementation
      - Request context management
      - ORM query hooks/filters
    acceptance_criteria:
      - "Tenant is resolved from (in priority order): JWT tenant_id claim, X-Tenant-ID header, or request subdomain; missing tenant identification returns 400 Bad Request"
      - "Tenant context is stored in request scope and accessible to all downstream handlers, services, and database queries within the same request"
      - "All ORM queries automatically include WHERE tenant_id = :current_tenant without manual specification; a query for tasks returns only the current tenant's tasks"
      - "Cross-tenant data access is blocked: manually constructing a query with a different tenant_id returns zero rows (application-level filter overrides)"
      - "All log entries include tenant_id for request tracing and debugging"
      - "Background jobs triggered by a request carry the originating tenant_id in their payload and set the tenant context before execution"
      - "An async test: two concurrent requests from different tenants executing simultaneously each see only their own data (no context leakage)"
      - "Admin superuser endpoint allows platform operators to query across all tenants by explicitly bypassing the tenant filter"
    pitfalls:
      - Thread-local/context-local tenant ID leaking between async requests in event-loop frameworks
      - Setting tenant context AFTER database queries already executed in middleware ordering
      - Not validating that the authenticated user belongs to the resolved tenant
      - Forgetting to propagate tenant context into background jobs or message queue consumers
      - Raw SQL queries bypassing ORM filter — must also apply tenant scoping
    deliverables:
      - Tenant resolver middleware (JWT claim, header, subdomain)
      - Request-scoped tenant context storage
      - ORM global query filter automatically scoping by tenant_id
      - Background job tenant context propagation
      - Cross-tenant access prevention guard
      - Admin bypass for platform-level operations
      - Tenant context in structured logs

  - id: multi-tenant-saas-m3
    name: PostgreSQL Row-Level Security
    description: >-
      Add PostgreSQL row-level security as a defense-in-depth layer ensuring
      tenant isolation at the database level, independent of application code.
    estimated_hours: 13
    concepts:
      - "RLS policy: CREATE POLICY tenant_isolation ON tasks USING (tenant_id = current_setting('app.current_tenant')::uuid)"
      - "Session variable: SET app.current_tenant = 'tenant-uuid' before each query"
      - "Defense in depth: RLS catches bugs in application-level filtering"
      - "RLS applies to SELECT, INSERT, UPDATE, DELETE independently"
      - "Superuser and table owner bypass RLS — use a non-owner application role"
    skills:
      - PostgreSQL RLS policy creation
      - Session variable management
      - Database role and permission design
    acceptance_criteria:
      - "RLS is enabled on all tenant-scoped tables (ALTER TABLE ... ENABLE ROW LEVEL SECURITY)"
      - "RLS policies restrict SELECT, INSERT, UPDATE, and DELETE to rows where tenant_id matches the session variable (current_setting('app.current_tenant'))"
      - "Application sets the session variable (SET app.current_tenant = X) at the beginning of each database connection/transaction before executing any queries"
      - "Cross-tenant isolation test: with RLS enabled, a query executed with tenant A's context against a table containing both tenant A and tenant B data returns ONLY tenant A's rows"
      - "INSERT with a mismatched tenant_id (different from session variable) is rejected by the RLS policy"
      - "A dedicated application database role (not superuser, not table owner) is used for all application queries, ensuring RLS is enforced"
      - "Schema migrations run as a superuser role that bypasses RLS; data migrations run as the application role with RLS enforced"
      - "Concurrent requests from 5 different tenants all receive correct isolated data (load test with parallel execution)"
    pitfalls:
      - Using superuser or table owner connection for application queries — RLS is silently bypassed
      - Forgetting to SET session variable before query — RLS sees NULL tenant and returns no rows (or all rows depending on policy)
      - Complex RLS policies with subqueries causing full table scans on every query
      - Not testing RLS with INSERT/UPDATE/DELETE — only testing SELECT
      - RLS bypass through database functions defined with SECURITY DEFINER
    deliverables:
      - RLS enablement migration for all tenant-scoped tables
      - RLS policies for SELECT, INSERT, UPDATE, DELETE on each table
      - Session variable setter in connection/transaction setup
      - Dedicated application database role (non-superuser, non-owner)
      - Superuser migration role for schema changes
      - RLS isolation test suite (multi-tenant concurrent queries)
      - RLS bypass documentation for platform admin operations

  - id: multi-tenant-saas-m4
    name: Tenant Customization & Feature Flags
    description: >-
      Implement per-tenant feature flags, plan-based feature gating, and
      branding customization.
    estimated_hours: 13
    concepts:
      - "Feature flags: boolean or percentage-based toggles per tenant"
      - "Plan-based gating: free plan gets features A/B, pro gets A/B/C/D"
      - "Configuration hierarchy: global defaults -> plan defaults -> tenant overrides"
      - "Configuration caching: cache in Redis with TTL, invalidate on update"
    skills:
      - Feature flag system design
      - Hierarchical configuration
      - Cache management
    acceptance_criteria:
      - "Feature flags are evaluated per-tenant with a hierarchy: global default -> plan default -> tenant-specific override; the most specific value wins"
      - "Plan-based feature gating: accessing a feature not included in the tenant's plan returns 403 Forbidden with a message indicating the required plan"
      - "Tenant branding (logo URL, primary color, accent color) is stored per tenant and returned in a branding API endpoint; the frontend applies these values"
      - "Configuration changes take effect within 30 seconds (cached in Redis with 30s TTL, invalidated on explicit update)"
      - "Feature flag check is fast: <1ms with warm cache (Redis lookup); verified under load"
      - "Settings management API allows tenant admins to view and update their configurable options with validation (e.g., color must be valid hex, URL must be valid)"
      - "Feature flag changes are audit-logged with who changed what and when"
    pitfalls:
      - Configuration cache not invalidated across distributed instances after update
      - Feature flag checks bypassing tenant context validation (checking flag for wrong tenant)
      - Allowing dangerous tenant configurations (e.g., setting webhook URL to internal network addresses — SSRF)
      - Not versioning configuration changes making rollback impossible
      - Performance degradation from uncached per-request configuration lookups
    deliverables:
      - Feature flag evaluation engine with global/plan/tenant hierarchy
      - Plan-based feature access control middleware
      - Tenant branding storage and retrieval API
      - Configuration caching in Redis with TTL and invalidation
      - Settings management API with validation
      - Feature flag and configuration audit log

  - id: multi-tenant-saas-m5
    name: Usage Metering & Billing
    description: >-
      Track per-tenant usage metrics, enforce plan quotas, and integrate with
      Stripe for subscription billing and usage-based invoicing.
    estimated_hours: 13
    concepts:
      - "Usage metering: record each billable event (API call, storage byte, compute minute) with idempotency key"
      - "Aggregation: roll up raw events into hourly/daily totals per tenant"
      - "Quota enforcement: check current usage against plan limits before allowing action"
      - "Stripe integration: create customer, subscription, and usage records via API"
      - "Webhook handling: process Stripe events (payment succeeded, failed, subscription updated) idempotently"
    skills:
      - Usage event tracking
      - Billing API integration (Stripe)
      - Quota enforcement
      - Webhook processing
    acceptance_criteria:
      - "Billable events (API calls, storage bytes, active users) are recorded per tenant with an idempotency key preventing double-counting; duplicate event submissions are silently ignored"
      - "Usage aggregation pipeline rolls up raw events into hourly and daily totals per tenant per metric type; aggregation runs within 5 minutes of the period end"
      - "Plan quota enforcement checks current period usage before allowing the action; exceeding the limit returns 429 Too Many Requests with a Retry-After header and a message indicating the limit and current usage"
      - "Stripe integration creates a Stripe customer and subscription on tenant provisioning; usage-based line items are reported to Stripe at the end of each billing period"
      - "Stripe webhook handler processes events (invoice.paid, invoice.payment_failed, customer.subscription.updated) idempotently using the event ID; webhook signature is verified using Stripe's signing secret"
      - "A billing dashboard API returns current period usage, plan limits, and projected cost for the current billing cycle"
      - "Tenant suspension: when payment fails after configurable retry period (default 7 days), tenant access is suspended (read-only mode) until payment is resolved"
    pitfalls:
      - Double-counting usage events without idempotency keys
      - Race condition when checking quota under high concurrency (check-then-act)
      - Not verifying Stripe webhook signatures allowing spoofed events
      - Incorrect timezone handling in billing period boundaries
      - Missing usage events due to async processing failures (use durable queue)
      - Not handling Stripe webhook retry storms (same event delivered multiple times)
    deliverables:
      - Usage event recorder with idempotency key deduplication
      - Hourly and daily usage aggregation pipeline
      - Quota enforcement middleware checking usage against plan limits
      - Stripe customer and subscription management
      - Stripe usage reporting at billing period end
      - Stripe webhook handler with signature verification and idempotent processing
      - Billing dashboard API (usage, limits, projected cost)
      - Tenant suspension on payment failure

```
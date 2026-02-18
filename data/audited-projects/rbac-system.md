# AUDIT & FIX: rbac-system

## CRITIQUE
- **Missing Policy Enforcement Point (PEP)**: The project defines a Policy Decision Point (PDP) but completely ignores the PEP—the actual middleware/interceptor that sits in the request path, extracts subject/object/action, calls the PDP, and enforces the decision. Without a PEP, the entire system is an academic exercise with no integration point.
- **ABAC ≠ DAG Traversal**: The essence conflates RBAC hierarchical traversal (DAG) with ABAC evaluation. ABAC evaluates boolean predicates over attribute tuples (subject attributes × object attributes × environment attributes). It does NOT traverse graphs. This is a fundamental conceptual error.
- **Missing Explicit Deny Priority**: The RBAC model doesn't address deny rules. In any real system, explicit deny MUST override allow (AWS IAM model). The milestones mention 'deny-overrides' in M2 pitfalls but it's not in M1's acceptance criteria where role permissions are defined.
- **No Permission Evaluation API**: There's no acceptance criterion for the core 'can user X do action Y on resource Z?' API. This is the most fundamental operation.
- **Bitmap Optimization Premature**: Permission bitmap encoding is mentioned as a concept in M1 but there's no benchmark or performance requirement that would motivate it. Premature optimization without measured need.
- **M3 Multi-tenancy Is Underspecified**: 'Enforce tenant isolation' is vague. What does enforcement mean? Database-level RLS? Application-level checks? Both? The AC doesn't specify.
- **XACML Mentioned but Not Implemented**: Learning outcomes mention 'XACML-style policy evaluation with combining algorithms' but no milestone implements this.
- **Missing Constraint-Based Separation of Duty**: NIST RBAC model includes Separation of Duty constraints (Static and Dynamic). These are absent.
- **Audit Log Integrity**: M4 mentions 'immutable' audit logs in pitfalls but acceptance criteria just say 'log every access control decision' with no integrity guarantees.

## FIXED YAML
```yaml
id: rbac-system
name: RBAC/ABAC Authorization System
description: >-
  Role-based and attribute-based access control system with policy decision
  point, policy enforcement point, multi-tenant isolation, and audit logging.
difficulty: expert
estimated_hours: "50-65"
essence: >-
  Hierarchical role permission graph with DAG-based inheritance for RBAC,
  boolean predicate evaluation over subject-object-environment attribute
  tuples for ABAC policy decisions, policy enforcement point middleware for
  request interception, deny-override conflict resolution, multi-tenant
  resource isolation, and tamper-evident audit logging for compliance.
why_important: >-
  Building this teaches production authorization patterns critical for SaaS
  platforms, combining security policy enforcement with performance-sensitive
  permission checks that scale across millions of users and resources—the
  same patterns used by AWS IAM, Google Cloud IAM, and Kubernetes RBAC.
learning_outcomes:
  - Implement hierarchical RBAC with DAG-based role inheritance and efficient permission resolution
  - Design ABAC policy engines evaluating boolean predicates over subject, resource, and environment attributes
  - Build a Policy Decision Point (PDP) with configurable combining algorithms (deny-overrides, permit-overrides)
  - Implement a Policy Enforcement Point (PEP) as middleware intercepting requests and enforcing PDP decisions
  - Design explicit deny rules that ALWAYS override allow rules regardless of evaluation order
  - Implement multi-tenant resource isolation with tenant-scoped roles and row-level security patterns
  - Build tamper-evident audit logging capturing authorization decisions for compliance
  - Create policy testing frameworks with simulation, conflict detection, and regression testing
  - Optimize permission lookups using caching with proper invalidation on role/policy changes
skills:
  - RBAC with Role Hierarchies
  - ABAC Policy Evaluation
  - Policy Decision Point (PDP)
  - Policy Enforcement Point (PEP)
  - Multi-tenancy Architecture
  - Deny-Override Conflict Resolution
  - Audit Logging with Integrity
  - Permission Caching & Invalidation
tags:
  - abac
  - access-control
  - authorization
  - expert
  - multi-tenancy
  - permissions
  - policy
  - rbac
  - roles
  - security
architecture_doc: architecture-docs/rbac-system/index.md
languages:
  recommended:
    - Go
    - Python
    - Java
  also_possible:
    - Rust
    - TypeScript
resources:
  - name: NIST RBAC Model SP 800-162
    url: https://csrc.nist.gov/pubs/sp/800/162/upd2/final
    type: documentation
  - name: NIST RBAC Project
    url: https://csrc.nist.gov/projects/role-based-access-control
    type: documentation
  - name: Casbin Authorization Library
    url: https://www.casbin.org/
    type: tool
  - name: AWS IAM Policy Evaluation Logic
    url: https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_evaluation-logic.html
    type: documentation
  - name: Open Policy Agent
    url: https://www.openpolicyagent.org/docs/latest/
    type: documentation
prerequisites:
  - type: project
    id: http-server-basic
    name: HTTP Server (Basic)
  - type: skill
    name: Database operations (SQL or equivalent)
  - type: skill
    name: HTTP middleware concepts
  - type: skill
    name: Graph data structures (DAG)
milestones:
  - id: rbac-system-m1
    name: Role & Permission Model with Hierarchy
    description: >-
      Implement the core RBAC model with named roles, typed permissions,
      DAG-based role inheritance, and a permission resolution API.
    acceptance_criteria:
      - Define permissions as (resource_type, action) tuples (e.g., 'document: read', 'document:write', 'user:delete')
      - Define roles with a name, description, and a set of directly assigned permissions
      - Support role hierarchy where child roles inherit all permissions from parent roles
      - Validate role hierarchy is a DAG (Directed Acyclic Graph); reject any role assignment that would create a cycle
      - Assign one or more roles to individual users with role-user mapping
      - Implement the core authorization check API: hasPermission(user_id, resource_type, action) → bool
      - Permission resolution computes transitive closure of inherited permissions through role hierarchy
      - Support explicit DENY permissions that override any inherited or direct ALLOW
      - Deny always wins: if any role (direct or inherited) has deny for a permission, the final result is deny
      - Default-deny: if no matching permission (allow or deny) exists, the result is deny
      - Wildcard permissions (e.g., 'document: *') are supported but resolved explicitly (expanded, not pattern-matched at check time)
    pitfalls:
      - Role hierarchy cycles cause infinite loops in permission resolution; MUST validate DAG property on every role relationship change
      - Cache invalidation required when roles or permissions change; use versioned cache keys or event-based invalidation
      - Wildcard permissions can accidentally grant more access than intended; require explicit confirmation
      - Role explosion (too many fine-grained roles) makes the system unmanageable; provide role templates
      - Transitive closure computation can be expensive for deep hierarchies; precompute and cache
      - Deny permissions must be checked at ALL levels of the hierarchy, not just the directly assigned role
    concepts:
      - Directed Acyclic Graph (DAG) for role hierarchies
      - Transitive closure computation for inherited permissions
      - Explicit deny override semantics
      - Default-deny security model
    skills:
      - Graph algorithms (cycle detection, transitive closure)
      - Permission modeling and resolution
      - Database schema design for RBAC
      - Cache invalidation strategies
    deliverables:
      - Permission CRUD with (resource_type, action) tuples and optional deny flag
      - Role CRUD with name, description, and direct permission assignments
      - Role hierarchy management with DAG cycle validation
      - User-to-role assignment supporting multiple roles per user
      - Permission resolution API computing effective permissions including inherited and deny
      - Core hasPermission(user_id, resource_type, action) check returning allow/deny
    estimated_hours: "12-15"

  - id: rbac-system-m2
    name: ABAC Policy Engine & Policy Decision Point
    description: >-
      Extend the system with attribute-based policies evaluated by a Policy
      Decision Point (PDP) using boolean predicate logic over attribute tuples.
    acceptance_criteria:
      - Define ABAC policies with: effect (allow/deny), target (resource_type + action), and conditions
      - Conditions evaluate boolean expressions over subject attributes (user.department, user.clearance_level), resource attributes (resource.classification, resource.owner), and environment attributes (request.time, request.ip)
      - Support comparison operators: equals, not_equals, greater_than, less_than, in_list, contains
      - Support logical operators: AND, OR, NOT for combining conditions
      - Implement deny-overrides combining algorithm: if ANY policy evaluates to deny, final decision is deny
      - Implement permit-overrides combining algorithm as an alternative: if ANY policy evaluates to permit, final decision is permit
      - Default decision is DENY when no policies match (closed-world assumption)
      - PDP evaluates RBAC permissions FIRST, then ABAC policies, with deny from either resulting in final deny
      - PDP returns structured decision: {allowed: bool, reason: string, matched_policies: [ids]}
      - Policy evaluation completes within 10ms for typical request with ≤100 active policies
    pitfalls:
      - ABAC is NOT graph traversal; it's boolean predicate evaluation over attribute tuples
      - All attributes needed for evaluation must be present in the request context; missing attributes should result in deny
      - Policy ordering matters for performance but should NOT affect the final decision (combining algorithm is deterministic)
      - Deny-overrides is safer (least privilege) but permit-overrides may be needed for exception handling
      - Caching ABAC decisions is hard because attribute values can change between requests
      - Policy conflicts (same resource, one allow one deny) must be resolved deterministically by combining algorithm
    concepts:
      - Boolean predicate evaluation over attribute tuples
      - Policy combining algorithms (deny-overrides, permit-overrides, first-applicable)
      - Subject-Object-Environment attribute model
      - Policy Decision Point (PDP) architecture
    skills:
      - Policy expression parsing and evaluation
      - Attribute resolution and context building
      - Combining algorithm implementation
      - Performance optimization for policy evaluation
    deliverables:
      - ABAC policy definition with effect, target, and condition expressions
      - Condition evaluator supporting comparison and logical operators over attributes
      - Policy Decision Point (PDP) evaluating RBAC + ABAC and returning structured decisions
      - Combining algorithm implementations (deny-overrides, permit-overrides)
      - Request context builder aggregating subject, resource, and environment attributes
      - Policy evaluation performance benchmark meeting <10ms target
    estimated_hours: "12-15"

  - id: rbac-system-m3
    name: Policy Enforcement Point & Multi-Tenancy
    description: >-
      Implement the PEP as HTTP middleware that intercepts requests, builds
      authorization context, calls PDP, and enforces decisions. Add
      multi-tenant isolation.
    acceptance_criteria:
      - PEP middleware intercepts every HTTP request before it reaches the application handler
      - PEP extracts subject identity from authentication token (JWT or session)
      - PEP maps HTTP method + URL path to (resource_type, action) for PDP evaluation
      - PEP builds request context with subject attributes, inferred resource attributes, and environment attributes (timestamp, IP)
      - PEP calls PDP and returns HTTP 403 Forbidden with error details if decision is deny
      - PEP passes request to application handler only if PDP decision is allow
      - Tenant isolation: every resource has a tenant_id; PEP injects tenant_id filter into all queries
      - Cross-tenant access is denied by default unless an explicit cross-tenant policy grant exists
      - Roles are tenant-scoped: admin in Tenant A has NO permissions in Tenant B
      - Resource ownership model tracks creator and tenant; creator gets full control by default
      - PEP adds authorization decision metadata to response headers for debugging (in dev mode only)
    pitfalls:
      - PEP MUST be non-bypassable; all request paths must go through it (no backdoor endpoints)
      - Tenant isolation must be enforced at BOTH application level (PEP) AND database query level (WHERE tenant_id = ?)
      - Admin roles must be tenant-scoped, not global; a global admin is a massive security risk
      - PEP performance is critical: it's in the hot path of every request; cache PDP decisions where safe
      - Authorization context must be built atomically; partial context can lead to incorrect decisions
      - Don't leak authorization internals in production error responses (403 should be generic)
    concepts:
      - Policy Enforcement Point (PEP) as middleware
      - Request-to-permission mapping
      - Tenant isolation at application and database layers
      - Row-level security (RLS) patterns
    skills:
      - HTTP middleware implementation
      - Request interception and context extraction
      - Multi-tenant database design
      - Tenant-scoped role management
    deliverables:
      - PEP HTTP middleware intercepting all requests and calling PDP
      - Request-to-permission mapper translating HTTP method + path to (resource_type, action)
      - Authorization context builder aggregating subject, resource, and environment attributes
      - Tenant isolation enforcement at application and database query levels
      - Tenant-scoped role model restricting role permissions to owning tenant
      - Resource ownership tracking with creator and tenant metadata
    estimated_hours: "12-15"

  - id: rbac-system-m4
    name: Audit Logging & Policy Testing
    description: >-
      Implement tamper-evident audit logging for all authorization decisions
      and a policy simulation framework for testing changes before deployment.
    acceptance_criteria:
      - Log every authorization decision with: timestamp, user_id, tenant_id, resource, action, decision (allow/deny), matched_policies, and latency_ms
      - Audit logs are append-only; once written, log entries cannot be modified or deleted
      - Audit log entries include a chained hash (each entry includes hash of previous entry) for tamper detection
      - Log integrity verification tool detects any modified, deleted, or inserted log entries
      - Track all permission and role changes as separate audit events with actor, before-state, and after-state
      - Policy simulation endpoint accepts a hypothetical request context and returns the PDP decision WITHOUT enforcing it
      - Policy diff tool compares two policy sets and reports permissions that change (gained, lost, modified)
      - Automated policy test suite with YAML-defined test cases: {context: {...}, expected_decision: allow|deny}
      - Test suite includes NEGATIVE test cases (should-be-denied scenarios) not just positive cases
      - Generate compliance reports summarizing: who accessed what, denied access attempts, and policy changes over a time period
    pitfalls:
      - Audit logs MUST be immutable; use append-only storage or write-once media
      - Include enough context to understand the decision but NEVER log sensitive data (passwords, tokens, PII beyond user_id)
      - Chained hashing has performance cost; batch hash computation if needed
      - Policy simulation must use an ISOLATED copy of the policy set, not production
      - Testing only positive cases (should-be-allowed) misses the most dangerous bugs (should-be-denied but isn't)
      - Audit log volume can be enormous; implement log rotation with archival to cold storage
    concepts:
      - Tamper-evident logging with hash chains
      - Policy simulation and what-if analysis
      - Differential policy analysis
      - Compliance audit reporting
    skills:
      - Append-only storage implementation
      - Hash chain integrity verification
      - Policy testing and simulation
      - Compliance report generation
    deliverables:
      - Authorization decision logger recording every PEP decision with full context
      - Append-only audit log with chained hash integrity protection
      - Log integrity verification tool detecting tampering
      - Policy change audit trail recording role/permission modifications
      - Policy simulation endpoint for testing hypothetical authorization requests
      - Policy diff tool comparing permission effects of policy set changes
      - YAML-based policy test suite with positive and negative test cases
      - Compliance report generator summarizing access patterns and policy changes
    estimated_hours: "12-15"
```
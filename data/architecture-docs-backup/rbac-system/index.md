# RBAC/ABAC Authorization System: Design Document


## Overview

This system provides flexible access control supporting both Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC) with policy evaluation. The key architectural challenge is efficiently evaluating complex authorization policies while maintaining tenant isolation and providing audit trails for compliance.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** All milestones - this section establishes the foundational understanding needed across the entire authorization system

Access control is one of the most fundamental challenges in software engineering, yet it's often treated as an afterthought or implemented with overly simplistic approaches that break down as systems grow in complexity. Modern applications face a perfect storm of authorization challenges: massive scale with millions of users, complex organizational hierarchies that don't map neatly to simple role models, multi-tenant SaaS architectures where data isolation is critical, and increasingly stringent compliance requirements that demand detailed audit trails of every access decision.

The challenge becomes even more complex when we consider that authorization decisions must be made in real-time, often thousands of times per second, while maintaining consistency across distributed systems and providing the flexibility to handle edge cases that traditional role-based models simply cannot express. A single authorization bug can lead to catastrophic data breaches, regulatory violations, or business-critical downtime, making this one of the highest-stakes components in any system architecture.

### The Corporate Building Access Mental Model

Before diving into the technical complexities, let's establish an intuitive understanding through a familiar analogy. Think of authorization like a sophisticated corporate building security system - one that goes far beyond simple key cards to handle the complex realities of modern organizational access control.

In a traditional office building, you might have different colored badges: blue for employees, silver for managers, and gold for executives. Each badge type grants access to certain floors and rooms. This represents basic **Role-Based Access Control (RBAC)** - your role determines your permissions, and everyone with the same role gets identical access rights. The security guard at the front desk checks your badge color and either lets you through or stops you.

However, modern organizations are far more complex than this simple model. Consider these scenarios that a badge-color system cannot handle:

**Time-sensitive access**: The accounting team needs access to the executive conference room, but only during budget season from January to March. A facilities manager might need after-hours access to all floors for maintenance, but only when there's an active work order. Traditional roles cannot express these temporal conditions.

**Contextual permissions**: A project manager should access the research lab when working on Project Alpha, but not when working on Project Beta. The same person, same role, but different permissions based on which project they're actively working on. This requires understanding the current context, not just the person's static role.

**Resource-specific ownership**: When an employee creates a presentation file, they become its owner and can share it with specific colleagues, even if those colleagues wouldn't normally have access to that type of document. The permissions are tied to the specific resource and its relationship to users, not just organizational hierarchy.

**Dynamic attribute-based decisions**: The HR system should allow access to salary data only if the requestor is in the HR department AND accessing data for their assigned region AND during business hours AND from a company-managed device. Multiple attributes must be evaluated together to make the decision.

This is where **Attribute-Based Access Control (ABAC)** comes in. Instead of just checking a badge color, the security system becomes more like an intelligent agent that considers multiple factors: who you are (user attributes), what you're trying to access (resource attributes), when and where you're making the request (environment attributes), and what action you want to perform (action attributes). The system then evaluates policies that combine these attributes with complex logical conditions to make nuanced access decisions.

In our building analogy, this intelligent security system might think: "This person has a manager badge (role), is assigned to the Alpha project (user attribute), is trying to access the research lab (resource), during business hours (environment attribute), to read project files (action), and the lab is designated for Alpha project work (resource attribute). The policy states that Alpha project managers can read files in Alpha labs during business hours, so access is granted."

The authorization system we're building needs to handle both the simple badge-checking scenarios (RBAC) and the complex contextual decision-making scenarios (ABAC), often within the same request. It must also maintain perfect tenant isolation - ensuring that Company A's employees can never accidentally access Company B's resources, even if they somehow ended up with similar roles or attributes. Finally, every access decision must be logged immutably for compliance auditing, just as a building's security system maintains visitor logs and badge scan records.

### Why Access Control is Hard

The fundamental difficulty of access control stems from the tension between four critical requirements that often conflict with each other: **scale**, **flexibility**, **security**, and **performance**. Each requirement pushes the design in different directions, and the complexity emerges from finding architectural solutions that satisfy all four simultaneously.

**Scale Challenge: The Permission Explosion Problem**

Modern applications operate at unprecedented scale, both in terms of users and resources. A typical enterprise SaaS application might serve hundreds of thousands of users across thousands of tenant organizations, with millions of resources (documents, projects, datasets, API endpoints) that need fine-grained access control. The naive approach of creating explicit permission records for every user-resource combination quickly becomes mathematically impossible.

Consider a document management system with 100,000 users and 1 million documents. If we tried to store explicit permissions for every possible user-document combination, we would need 100 billion permission records. Even if each record only required 20 bytes of storage, we're looking at 2 terabytes just for the permission matrix, and that's before considering the computational cost of querying this massive dataset for every authorization check.

The scale problem is further complicated by permission inheritance and role hierarchies. When a user's role changes or a resource moves to a different folder with different permissions, the system must efficiently propagate these changes without scanning millions of records. Traditional database approaches break down when faced with these recursive permission calculations across hierarchical structures.

**Flexibility Challenge: The Context Complexity Problem**

Real-world access control policies are rarely static. They depend on complex combinations of contextual factors that change dynamically. A user might have permission to modify a project's budget, but only if the project is in active status, the modification is within their approval limit, the request comes during business hours, and the change doesn't violate certain compliance rules.

These contextual policies cannot be pre-computed and stored in simple lookup tables. Instead, they require dynamic evaluation at request time, considering the current state of the user, resource, environment, and action. The system must be flexible enough to express arbitrary logical conditions while remaining comprehensible to policy authors and maintainable over time.

The flexibility requirement also extends to policy evolution. Organizations constantly restructure, merge departments, change compliance requirements, and adopt new business processes. The authorization system must accommodate these changes without requiring complete reimplementation or causing service disruptions.

**Security Challenge: The Fail-Safe Complexity Problem**

Authorization systems are high-value targets for attackers and have zero tolerance for errors. A single bug that grants unauthorized access can lead to massive data breaches, while a bug that denies legitimate access can bring business operations to a halt. This creates a unique security challenge where both false positives (denying legitimate access) and false negatives (granting unauthorized access) have severe consequences.

The security challenge is compounded by the principle of **defense in depth**. Authorization decisions cannot rely solely on application-level checks; they must be enforced consistently across multiple layers, including database queries, API gateways, and client applications. Each layer must implement the same logical policies while working with different data representations and performance constraints.

Multi-tenant systems face an additional security challenge: **tenant isolation**. The system must guarantee that no possible combination of user actions, system failures, or administrative errors can allow one tenant's users to access another tenant's data. This requires careful design of data models, query patterns, and policy evaluation logic to ensure that tenant context is always correctly propagated and validated.

**Performance Challenge: The Real-Time Decision Problem**

Authorization checks happen on the critical path of every user request, often multiple times per request as different resources are accessed. A typical web application might perform 10-50 authorization checks per page load, and each check must complete in milliseconds to maintain acceptable user experience. This means the authorization system must evaluate complex policies, potentially involving database lookups and hierarchical calculations, faster than most systems perform simple cache reads.

The performance challenge is particularly acute for ABAC policies, which may need to fetch and evaluate dozens of attributes from multiple data sources. A policy that checks the user's department, current project assignments, the resource's classification level, and the current time zone must somehow complete this evaluation in under 10 milliseconds while handling thousands of concurrent requests.

Caching becomes critical for performance, but authorization caching is notoriously difficult because policies often depend on rapidly-changing contextual data. A cached decision might become invalid seconds later when a user's role changes or a resource's ownership is transferred. The system must implement sophisticated cache invalidation strategies that maintain both performance and consistency.

> **Key Insight**: The hardness of access control isn't in any single requirement, but in the interactions between these requirements. A solution that optimizes for scale might sacrifice flexibility. A solution that maximizes security might hurt performance. The architectural challenge is finding designs that gracefully balance these tensions while remaining maintainable and debuggable.

### Existing Authorization Patterns

Understanding the landscape of existing authorization patterns is crucial for making informed architectural decisions. Each pattern represents a different approach to balancing the scale, flexibility, security, and performance requirements we discussed above. Rather than viewing these as competing alternatives, it's better to understand them as tools that can be combined to address different aspects of the authorization problem.

**Access Control Lists (ACLs): The Direct Mapping Approach**

Access Control Lists represent the most straightforward approach to authorization: for each resource, maintain an explicit list of users (or groups) and what they can do with that resource. This pattern maps directly to how we think about ownership and sharing in the physical world - a document has a list of people who can read it, edit it, or share it with others.

| Aspect | Description |
|--------|-------------|
| **Core Concept** | Each resource maintains a list of principals (users/groups) and their allowed actions |
| **Decision Logic** | Check if the requesting user appears in the resource's ACL with the required permission |
| **Storage Model** | Resource-centric: each resource stores its own permission list |
| **Typical Use Cases** | File systems, document sharing, simple resource ownership scenarios |

**ACL Strengths and Weaknesses:**

| Pros | Cons |
|------|------|
| Intuitive and easy to understand | Poor scalability - each resource needs explicit permission entries |
| Fine-grained control per resource | Difficult to manage consistently across many resources |
| Direct mapping to ownership concepts | No support for role hierarchies or complex policies |
| Simple implementation for small systems | Permission explosion problem with large user bases |
| Easy to audit permissions for a specific resource | Hard to answer "what can this user access?" questions |

ACLs work well for scenarios with relatively few resources and clear ownership boundaries, but they break down quickly in enterprise environments where the same permission patterns need to be applied across thousands of resources.

**Role-Based Access Control (RBAC): The Organizational Hierarchy Approach**

RBAC addresses the scalability limitations of ACLs by introducing an intermediate layer: roles. Instead of granting permissions directly to users, permissions are bundled into roles that represent organizational functions, and users are assigned to appropriate roles. This creates a many-to-many relationship between users and permissions, mediated by the role concept.

| Aspect | Description |
|--------|-------------|
| **Core Concept** | Permissions are grouped into roles; users are assigned roles; authorization checks user's roles |
| **Decision Logic** | Determine user's active roles, then check if any role has the required permission |
| **Storage Model** | Three-entity model: Users, Roles, Permissions with many-to-many relationships |
| **Typical Use Cases** | Enterprise applications, organizational hierarchies, job function-based access |

**RBAC Strengths and Weaknesses:**

| Pros | Cons |
|------|------|
| Scales well with organizational structure | Rigid - cannot express contextual or dynamic conditions |
| Reduces administrative overhead through role reuse | Role explosion when trying to handle edge cases |
| Maps naturally to job functions and org charts | Difficult to handle resource-specific permissions |
| Supports role hierarchies with permission inheritance | Cannot express time-based or location-based restrictions |
| Efficient permission lookup through role membership | Challenges with cross-departmental project teams |

RBAC excels in traditional organizational settings where access patterns align with job functions, but struggles with modern collaborative work patterns and dynamic business requirements.

**Attribute-Based Access Control (ABAC): The Context-Aware Decision Approach**

ABAC represents a fundamental shift from pre-defined roles to dynamic policy evaluation. Instead of checking static assignments, ABAC evaluates policies that consider attributes of the user, resource, action, and environment at request time. This enables expressing complex, contextual authorization requirements that would be impossible with ACLs or traditional RBAC.

| Aspect | Description |
|--------|-------------|
| **Core Concept** | Policies evaluate combinations of user, resource, action, and environment attributes |
| **Decision Logic** | Dynamically evaluate policy expressions against current request context |
| **Storage Model** | Policy-centric: policies define conditions using attribute expressions |
| **Typical Use Cases** | Complex compliance requirements, dynamic business rules, contextual access |

**ABAC Strengths and Weaknesses:**

| Pros | Cons |
|------|------|
| Extremely flexible - can express any logical condition | Complex to understand and maintain |
| Handles contextual and temporal requirements naturally | Performance overhead from dynamic evaluation |
| Reduces role explosion through dynamic evaluation | Requires sophisticated policy authoring tools |
| Supports fine-grained, resource-specific policies | Difficult to debug when policies interact unexpectedly |
| Adapts to changing business requirements easily | Potential for policy conflicts and inconsistencies |

ABAC provides the ultimate flexibility but requires significant investment in policy management infrastructure and expertise to use effectively.

**Relationship-Based Access Control (ReBAC): The Graph-Oriented Approach**

ReBAC models authorization as a graph of relationships between entities. Access decisions are based on the existence of specific relationship paths in this graph. For example, a user might access a document if they have an "editor" relationship to a project that has an "contains" relationship to the document.

| Aspect | Description |
|--------|-------------|
| **Core Concept** | Model entities and relationships as a graph; authorization follows relationship paths |
| **Decision Logic** | Check if required relationship path exists between user and resource |
| **Storage Model** | Graph database or relationship tuples (user, relation, object) |
| **Typical Use Cases** | Social networks, collaborative platforms, hierarchical resource structures |

**ReBAC Strengths and Weaknesses:**

| Pros | Cons |
|------|------|
| Natural fit for hierarchical and networked resources | Complex queries can be expensive on large graphs |
| Handles delegation and sharing relationships elegantly | Requires graph database expertise |
| Supports both direct and derived permissions | Difficult to express non-relationship-based conditions |
| Intuitive for modeling organizational structures | Performance challenges with deep relationship chains |
| Scales well with proper graph database optimization | Limited tooling and ecosystem compared to RBAC |

> **Decision: Hybrid RBAC/ABAC Architecture**
> - **Context**: We need to support both simple organizational access patterns and complex contextual policies while maintaining good performance and usability.
> - **Options Considered**: 
>   1. Pure RBAC with extensive role hierarchies
>   2. Pure ABAC with all decisions based on attribute evaluation  
>   3. Hybrid approach using RBAC for common patterns and ABAC for complex policies
> - **Decision**: Implement a hybrid architecture that uses RBAC as the foundation with ABAC extensions for complex scenarios
> - **Rationale**: RBAC provides excellent performance and intuitive management for 80% of access patterns, while ABAC handles the 20% of cases requiring contextual evaluation. This approach optimizes for the common case while maintaining flexibility for edge cases.
> - **Consequences**: Requires implementing both evaluation engines but allows gradual adoption of ABAC policies as needs become more sophisticated. Provides a clear migration path from simple role-based systems.

**Pattern Comparison for Our System:**

| Pattern | Scalability | Flexibility | Performance | Complexity | Chosen for System |
|---------|-------------|-------------|-------------|------------|------------------|
| **ACL** | Poor | Low | Excellent | Low | Resource ownership only |
| **RBAC** | Good | Medium | Excellent | Medium | Primary authorization method |
| **ABAC** | Excellent | Excellent | Fair | High | Complex policies and compliance |
| **ReBAC** | Good | Good | Fair | Medium | Future extension consideration |

Our hybrid approach implements RBAC as the primary authorization mechanism for performance and simplicity, with ABAC policies layered on top for scenarios requiring contextual evaluation. ACL-style resource ownership handles the specific case of resource creators having full control over their resources. This combination provides a clear upgrade path as organizational needs evolve from simple role-based access to more sophisticated policy-driven authorization.

### Common Pitfalls in Authorization System Design

Understanding common mistakes helps avoid architectural decisions that seem reasonable initially but create significant problems as the system scales and evolves.

⚠️ **Pitfall: The "Admin Can Do Anything" Trap**

Many systems implement a global admin role that bypasses all authorization checks. While this seems convenient for system administration, it creates serious security and compliance problems. Global admins often need to access tenant data for support purposes, but this violates data isolation principles and creates audit trail gaps.

**Why it's wrong**: Global admin access makes it impossible to enforce tenant isolation, creates compliance violations, and provides no audit trail for admin actions within tenant data.

**How to fix**: Implement tenant-scoped admin roles and explicit delegation mechanisms. Admins should assume temporary, logged roles within specific tenants rather than bypassing authorization entirely.

⚠️ **Pitfall: Caching Permission Decisions Without Invalidation Strategy**

Authorization results are tempting to cache for performance, but permission changes must be reflected immediately. Many systems cache "user X can access resource Y" decisions without considering how to invalidate these caches when user roles change, resources move between tenants, or policies are updated.

**Why it's wrong**: Stale permission caches can grant access to users who no longer have permissions or deny access to users who should have received new permissions, creating both security vulnerabilities and user experience problems.

**How to fix**: Design cache invalidation strategies from the beginning. Use versioned policies, implement cache tagging by user/resource/tenant, and ensure critical permission changes can force immediate cache invalidation.

⚠️ **Pitfall: Mixing Authentication Context with Authorization Logic**

Many systems tightly couple authentication tokens (JWT, sessions) with authorization decisions, making it difficult to test authorization logic independently or support different authentication methods.

**Why it's wrong**: Tight coupling makes the authorization system brittle to authentication changes and makes it nearly impossible to unit test authorization logic without complex authentication mocking.

**How to fix**: Define clear interfaces between authentication and authorization. Authorization components should receive normalized user/context information, not authentication tokens directly.

### Implementation Guidance

The authorization patterns and architectural decisions discussed above need to be translated into concrete technology choices and code organization strategies. This section provides practical guidance for implementing the hybrid RBAC/ABAC system using Go as the primary language.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Policy Storage** | PostgreSQL with JSONB policies | Specialized policy store (OPA/Cedar) |
| **Policy Language** | Custom JSON-based expressions | CEL (Common Expression Language) |
| **Caching Layer** | Redis with key-based invalidation | Redis with pub/sub cache invalidation |
| **Audit Storage** | PostgreSQL append-only tables | Time-series database (InfluxDB) |
| **Graph Relationships** | PostgreSQL recursive CTEs | Neo4j or other graph database |

**Recommended Project Structure:**

The authorization system should be organized as a separate module that can be imported by application services. This separation ensures clear boundaries and makes the authorization logic testable independently.

```
rbac-abac-system/
├── cmd/
│   ├── authz-server/           ← Standalone authorization service
│   └── policy-cli/             ← Policy management CLI tool
├── pkg/
│   ├── authz/                  ← Public authorization API
│   │   ├── authz.go           ← Main authorization interface
│   │   ├── context.go         ← Request context definitions
│   │   └── decision.go        ← Authorization decision types
│   ├── rbac/                   ← RBAC engine implementation
│   │   ├── engine.go          ← Role-based evaluation logic
│   │   ├── hierarchy.go       ← Role hierarchy management
│   │   └── permissions.go     ← Permission lookup optimization
│   ├── abac/                   ← ABAC policy engine
│   │   ├── engine.go          ← Policy evaluation logic
│   │   ├── parser.go          ← Policy expression parsing
│   │   └── evaluator.go       ← Attribute-based evaluation
│   ├── store/                  ← Data persistence layer
│   │   ├── interface.go       ← Storage interface definitions
│   │   ├── postgres/          ← PostgreSQL implementation
│   │   └── memory/            ← In-memory store for testing
│   └── audit/                  ← Audit logging component
│       ├── logger.go          ← Audit event logging
│       └── events.go          ← Audit event definitions
└── internal/
    ├── cache/                  ← Internal caching implementation
    ├── metrics/                ← Performance and usage metrics
    └── testutil/               ← Testing utilities and fixtures
```

**Core Authorization Interface:**

The main authorization interface should provide a clean API that abstracts the complexity of RBAC/ABAC evaluation from calling applications.

```go
// Package authz provides the main authorization interface for applications.
//
// This package defines the core types and interfaces for making authorization
// decisions using both RBAC and ABAC evaluation engines.
package authz

import (
    "context"
    "time"
)

// Authorizer is the main interface for making authorization decisions.
// It combines RBAC and ABAC evaluation to provide flexible access control.
type Authorizer interface {
    // IsAuthorized checks if a user can perform an action on a resource.
    // Returns an AuthzDecision with the result and evaluation details.
    IsAuthorized(ctx context.Context, req *AuthzRequest) (*AuthzDecision, error)
    
    // TODO: Implement the main authorization logic that:
    // 1. Extracts user context from the request
    // 2. Performs RBAC evaluation for role-based permissions
    // 3. Evaluates ABAC policies if RBAC is insufficient
    // 4. Combines decisions using deny-overrides logic
    // 5. Logs the decision for audit purposes
    // 6. Returns structured decision with reasoning
}

// AuthzRequest contains all information needed for an authorization decision.
type AuthzRequest struct {
    // User making the request
    UserID     string            `json:"user_id"`
    TenantID   string            `json:"tenant_id"`
    
    // Action being performed
    Action     string            `json:"action"`        // e.g., "read", "write", "delete"
    
    // Resource being accessed
    Resource   *Resource         `json:"resource"`
    
    // Additional context attributes for ABAC evaluation
    Attributes map[string]any    `json:"attributes"`
    
    // Request metadata
    Timestamp  time.Time         `json:"timestamp"`
    RequestID  string            `json:"request_id"`
}

// Resource represents the target of an authorization request.
type Resource struct {
    ID         string            `json:"id"`
    Type       string            `json:"type"`         // e.g., "document", "project", "api_endpoint"
    TenantID   string            `json:"tenant_id"`
    OwnerID    string            `json:"owner_id"`
    Attributes map[string]any    `json:"attributes"`   // Resource-specific attributes
}

// AuthzDecision contains the result of an authorization evaluation.
type AuthzDecision struct {
    // Decision result
    Allowed    bool              `json:"allowed"`
    
    // Evaluation details for debugging and audit
    Reason     string            `json:"reason"`
    Method     EvaluationMethod  `json:"method"`       // "rbac", "abac", "ownership"
    
    // Policy information that led to this decision
    AppliedPolicies []string     `json:"applied_policies"`
    
    // Evaluation metadata
    EvaluationTime time.Duration `json:"evaluation_time"`
    CacheHit      bool           `json:"cache_hit"`
}

// EvaluationMethod indicates which authorization method produced the decision.
type EvaluationMethod string

const (
    EvaluationRBAC      EvaluationMethod = "rbac"
    EvaluationABAC      EvaluationMethod = "abac"
    EvaluationOwnership EvaluationMethod = "ownership"
    EvaluationCached    EvaluationMethod = "cached"
)
```

**RBAC Data Structures:**

The RBAC engine needs efficient data structures for representing role hierarchies and permission lookups.

```go
// Package rbac implements role-based access control with hierarchical roles.
package rbac

// Role represents a named set of permissions that can be assigned to users.
type Role struct {
    ID          string    `json:"id"`
    Name        string    `json:"name"`
    TenantID    string    `json:"tenant_id"`    // Roles are tenant-scoped
    ParentRoles []string  `json:"parent_roles"` // For role hierarchy
    Permissions []Permission `json:"permissions"`
    CreatedAt   time.Time `json:"created_at"`
    UpdatedAt   time.Time `json:"updated_at"`
}

// Permission represents an allowed action on a resource type.
type Permission struct {
    ID       string `json:"id"`
    Resource string `json:"resource"`  // Resource type, e.g., "document", "project"
    Action   string `json:"action"`    // Action name, e.g., "read", "write"
    
    // TODO: Add support for wildcard permissions like "document:*"
    // TODO: Consider permission conditions for resource-specific restrictions
}

// UserRoleAssignment links users to roles with optional constraints.
type UserRoleAssignment struct {
    UserID    string     `json:"user_id"`
    RoleID    string     `json:"role_id"`
    TenantID  string     `json:"tenant_id"`
    
    // Optional time-based constraints
    ValidFrom *time.Time `json:"valid_from,omitempty"`
    ValidTo   *time.Time `json:"valid_to,omitempty"`
    
    AssignedBy string    `json:"assigned_by"`
    AssignedAt time.Time `json:"assigned_at"`
}

// RoleEngine handles RBAC evaluation and role hierarchy management.
type RoleEngine struct {
    store Store
    cache Cache
}

// HasPermission checks if a user has a specific permission through their roles.
func (r *RoleEngine) HasPermission(ctx context.Context, userID, tenantID, resource, action string) (bool, error) {
    // TODO 1: Get all active roles for the user in the specified tenant
    // TODO 2: For each role, get inherited permissions (including parent roles)
    // TODO 3: Check if any permission matches the required resource:action
    // TODO 4: Handle wildcard permissions appropriately
    // TODO 5: Cache the result with appropriate TTL and invalidation tags
    // Hint: Use breadth-first search for role hierarchy traversal to detect cycles
}

// GetUserPermissions returns all permissions available to a user.
func (r *RoleEngine) GetUserPermissions(ctx context.Context, userID, tenantID string) ([]Permission, error) {
    // TODO 1: Get user's direct role assignments
    // TODO 2: Expand roles to include inherited roles from hierarchy
    // TODO 3: Collect all permissions from all roles
    // TODO 4: Deduplicate permissions (same resource:action)
    // TODO 5: Return sorted list for consistent output
}
```

**ABAC Policy Structures:**

The ABAC engine needs flexible policy representation and efficient evaluation.

```go
// Package abac implements attribute-based access control with policy evaluation.
package abac

// Policy represents an ABAC policy with conditions and effects.
type Policy struct {
    ID          string           `json:"id"`
    Name        string           `json:"name"`
    TenantID    string           `json:"tenant_id"`
    Effect      PolicyEffect     `json:"effect"`      // "allow" or "deny"
    
    // Conditions that must be satisfied for this policy to apply
    Conditions  *Condition       `json:"conditions"`
    
    // Resources this policy applies to
    Resources   []string         `json:"resources"`   // Resource patterns, e.g., "document:*"
    Actions     []string         `json:"actions"`     // Action patterns, e.g., "read", "write"
    
    // Policy metadata
    Priority    int              `json:"priority"`    // Higher number = higher priority
    CreatedBy   string           `json:"created_by"`
    CreatedAt   time.Time        `json:"created_at"`
    UpdatedAt   time.Time        `json:"updated_at"`
}

// PolicyEffect determines whether a matching policy allows or denies access.
type PolicyEffect string

const (
    PolicyEffectAllow PolicyEffect = "allow"
    PolicyEffectDeny  PolicyEffect = "deny"
)

// Condition represents a logical expression that can be evaluated against request attributes.
type Condition struct {
    // Logical operators
    And []*Condition `json:"and,omitempty"`
    Or  []*Condition `json:"or,omitempty"`
    Not *Condition   `json:"not,omitempty"`
    
    // Leaf condition: attribute comparison
    Attribute string      `json:"attribute,omitempty"`  // e.g., "user.department"
    Operator  string      `json:"operator,omitempty"`   // e.g., "eq", "in", "gt"
    Value     interface{} `json:"value,omitempty"`      // Expected value
}

// PolicyEngine evaluates ABAC policies against request context.
type PolicyEngine struct {
    store     Store
    evaluator *ConditionEvaluator
    cache     Cache
}

// Evaluate checks all applicable policies and returns the final decision.
func (p *PolicyEngine) Evaluate(ctx context.Context, req *AuthzRequest) (*PolicyDecision, error) {
    // TODO 1: Find all policies that match the request's resource and action patterns
    // TODO 2: Filter policies by tenant to ensure isolation
    // TODO 3: Evaluate each policy's conditions against request attributes
    // TODO 4: Combine policy decisions using deny-overrides algorithm
    // TODO 5: Return decision with list of applied policies for audit trail
    // Hint: Sort policies by priority before evaluation for consistent results
}

// PolicyDecision contains the result of ABAC policy evaluation.
type PolicyDecision struct {
    Allowed         bool     `json:"allowed"`
    AppliedPolicies []string `json:"applied_policies"`
    DenyingPolicy   string   `json:"denying_policy,omitempty"` // Policy that caused denial
}
```

**Milestone Checkpoints:**

After implementing the basic authorization interfaces and data structures:

1. **Unit Test Coverage**: Run `go test ./pkg/authz/...` and verify >90% coverage
2. **Integration Test**: Create a simple test that assigns a role to a user and verifies permission checking works
3. **Performance Baseline**: Authorization decisions should complete in <10ms for RBAC, <50ms for ABAC
4. **Memory Usage**: The system should handle 10,000 users with 100 roles each using <500MB RAM

**Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Authorization always returns false | Missing role assignments or permissions | Check user roles and role permissions in database | Verify user-role and role-permission relationships |
| Slow authorization performance | Cache misses or inefficient role hierarchy traversal | Add timing logs to each evaluation step | Implement proper caching and optimize database queries |
| Inconsistent decisions | Cache invalidation issues | Disable cache temporarily and test | Implement proper cache versioning and invalidation |
| Cross-tenant data access | Missing tenant isolation checks | Add tenant ID logging to all queries | Ensure all database queries include tenant filtering |


## Goals and Non-Goals

> **Milestone(s):** All milestones - this section establishes clear boundaries and expectations for the entire authorization system

Building an authorization system is like architecting a security checkpoint system for a sprawling corporate campus. You need to decide whether you're building a simple badge scanner for one building, or a comprehensive security infrastructure that can handle visitors, contractors, employees, and executives across multiple buildings with different clearance levels. Getting this scope definition wrong leads to either an under-engineered system that collapses under real-world complexity, or an over-engineered monstrosity that never ships.

This section establishes the precise boundaries of what our RBAC/ABAC authorization system will and will not handle. These boundaries are critical because authorization systems have natural expansion pressure - every related security concern wants to creep into scope. Without clear goals and non-goals, you end up building an authentication system, a user management system, a secrets management system, and an authorization system all at once. That path leads to project failure.

### Functional Requirements

Our authorization system must deliver these core capabilities, mapped directly to real-world access control scenarios that modern applications face.

#### Core Authorization Capabilities

**Role-Based Access Control (RBAC)** forms the foundation of our system, handling the majority of access control decisions through organizational role assignments. Think of RBAC like job descriptions in a company - your role determines your base permissions, and role hierarchies mirror reporting structures where managers inherit their team's permissions plus additional management privileges.

The system must support hierarchical role structures where permissions flow down from parent roles to child roles. A "Director" role inherits all permissions from "Manager", which inherits from "Employee", which inherits from "Viewer". This hierarchy prevents permission duplication and mirrors how organizations actually structure access control.

| RBAC Capability | Requirement | Business Context |
|---|---|---|
| Role Definition | Create named roles with attached permission sets | Job functions like "Customer Support Agent", "Sales Manager" |
| Role Hierarchy | Parent-child relationships with permission inheritance via DAG | Organizational chart mapping where managers get team permissions |
| User Assignment | Assign multiple roles to users with temporal validity | Temporary project assignments, contractor access windows |
| Permission Lookup | Fast resolution of user permissions through role membership | Real-time API authorization checks under 10ms latency |
| Role Templates | Preconfigured role bundles for common access patterns | Onboarding new employees with standard permission sets |

**Attribute-Based Access Control (ABAC)** extends beyond roles to handle dynamic, contextual access decisions. ABAC is like having a smart security guard who considers not just your badge, but also the time of day, your current location, what you're trying to access, and current security alert levels before making access decisions.

The system must evaluate policies that consider user attributes (department, clearance level, employment status), resource attributes (classification, owner, creation date), and environmental attributes (time, location, threat level). These policies enable fine-grained access control that RBAC alone cannot provide.

| ABAC Capability | Requirement | Business Context |
|---|---|---|
| Policy Definition | Declarative policies with user, resource, and environment conditions | "Marketing can access campaign data during business hours" |
| Attribute Resolution | Dynamic attribute lookup from multiple sources during evaluation | User directory integration, resource metadata, real-time context |
| Policy Evaluation | Real-time condition assessment with boolean and numeric operators | Checking "user.department == 'Finance' AND resource.type == 'financial_report'" |
| Policy Combining | Multiple policy results merged with deny-overrides logic | Conflicting policies resolved with security-first approach |
| Context Propagation | Request context flows through entire evaluation pipeline | User session, request metadata, environmental state carried forward |

#### Resource Management and Multi-tenancy

Our system must handle resource-level permissions and tenant isolation, which is like managing access to individual offices within a multi-company building where each company rents multiple floors but cannot access other companies' spaces.

**Resource-Based Permissions** enable fine-grained access control on individual resources rather than just resource types. Instead of "can read documents", the system supports "can read document #12345". This granularity is essential for SaaS applications where users interact with specific records, files, or objects.

**Tenant Isolation** ensures complete data separation between different organizations using the system. Think of tenants as separate companies in a shared office building - they may use the same elevators and lobby, but they cannot access each other's offices, files, or meeting rooms.

| Multi-tenancy Capability | Requirement | Business Context |
|---|---|---|
| Tenant Context | Every request scoped to tenant with automatic isolation | SaaS application serving multiple customer organizations |
| Resource Ownership | Resources linked to creating user and tenant | Document creators get full control, tenant admins get oversight |
| Cross-tenant Access | Explicit sharing with time limits and revocation | Customer sharing reports with vendor for limited review period |
| Hierarchical Resources | Parent-child resource relationships with inherited permissions | Folder access grants access to contained files |
| Tenant-scoped Roles | Roles and permissions isolated within tenant boundaries | Admin role in Tenant A cannot access Tenant B resources |

#### Decision Audit and Compliance

Every authorization decision must be logged for compliance and security analysis. This is like maintaining a visitor log that records not just who entered the building, but what they accessed, when they accessed it, and why access was granted or denied.

The audit trail must be immutable and comprehensive enough to recreate authorization decisions for compliance review. Security teams need to answer questions like "Who accessed customer data last month?" and "Why was this user granted access to this sensitive resource?"

| Audit Capability | Requirement | Business Context |
|---|---|---|
| Decision Logging | Every authorization check recorded with full context | SOX compliance requires complete access audit trails |
| Immutable Storage | Audit logs cannot be modified after creation | Forensic analysis requires tamper-proof evidence |
| Decision Provenance | Trace which policies and roles led to each decision | Understanding why access was granted for compliance review |
| Access Reporting | Aggregate access patterns for security analysis | Identify unusual access patterns that may indicate breaches |
| Policy Change Tracking | All role and policy modifications logged with actor | Change management compliance and security review |

### Quality Attributes

Quality attributes define the operational requirements that determine whether the authorization system succeeds in production environments. These requirements are often more challenging than functional requirements because they involve system-wide concerns rather than individual features.

#### Performance Requirements

Authorization systems sit in the critical path of every application request, making performance requirements non-negotiable. Think of authorization like a turnstile at a subway entrance - if it's slow, it creates bottlenecks that affect the entire system's user experience.

**Latency Requirements** must support high-frequency authorization checks without degrading application performance. The system must make authorization decisions in single-digit milliseconds for cached results and sub-100 millisecond for complex policy evaluations.

**Throughput Requirements** must handle concurrent authorization requests across multiple tenants and users. The system should support thousands of authorization checks per second on modest hardware.

| Performance Metric | Requirement | Measurement Context |
|---|---|---|
| Authorization Latency | p95 < 50ms, p99 < 100ms for ABAC evaluation | Complex policies with multiple attribute lookups |
| Permission Lookup | p95 < 10ms for RBAC checks | Simple role-based permission verification |
| Cache Hit Rate | > 90% for repeated authorization patterns | Users checking same resources multiple times |
| Concurrent Requests | 10,000+ authorization checks per second | Multi-tenant load with burst traffic |
| Policy Evaluation | < 500ms for complex policies with 20+ conditions | Worst-case ABAC evaluation time |

**Caching Strategy** must balance performance with consistency. The system must cache permission calculations and policy decisions while ensuring cache invalidation when roles or policies change.

**Horizontal Scaling** must support multiple authorization service instances without coordination overhead. The system should scale by adding instances rather than requiring larger machines.

> **Decision: Stateless Authorization Architecture**
> - **Context**: Authorization services need to scale horizontally to handle increasing load, but maintaining shared state complicates scaling
> - **Options Considered**: Stateful service with shared cache, stateless service with external cache, hybrid approach with session affinity
> - **Decision**: Stateless authorization service with external Redis cache
> - **Rationale**: Stateless services scale easily by adding instances, external cache provides consistency across instances, Redis provides sub-millisecond cache lookups
> - **Consequences**: Enables simple horizontal scaling and load balancing, requires external cache dependency, increases deployment complexity

#### Security Requirements

Security requirements ensure the authorization system cannot itself become a security vulnerability. A compromised authorization system is worse than no authorization system because it provides false security while potentially granting broad access to attackers.

**Fail-Safe Defaults** require the system to deny access when in doubt rather than potentially granting unauthorized access. If the policy evaluation engine fails, the system must deny access rather than defaulting to allow.

**Least Privilege Enforcement** ensures users receive only the minimum permissions necessary for their functions. The system must not grant broader permissions than explicitly defined by roles and policies.

| Security Requirement | Specification | Threat Mitigation |
|---|---|---|
| Deny-by-Default | No matching policy results in access denial | Prevents accidental access grants due to policy gaps |
| Explicit Deny Wins | Deny policies override allow policies regardless of order | Security policies can block access granted by functional roles |
| Input Validation | All authorization requests validated against schema | Prevents injection attacks through malformed requests |
| Privilege Separation | Authorization service runs with minimal system permissions | Limits blast radius if authorization service is compromised |
| Audit Integrity | Audit logs use cryptographic signatures for tamper detection | Ensures audit trail cannot be modified to hide unauthorized access |

**Tenant Isolation** must be cryptographically enforced rather than relying solely on application logic. The system must prevent tenant A from accessing tenant B's resources even if application bugs bypass normal authorization checks.

**Cache Security** must prevent cache poisoning attacks where malicious users inject false authorization decisions into the cache. Cache keys must include tenant context and be resistant to collision attacks.

> **Decision: Row-Level Security Enforcement**
> - **Context**: Application-level tenant isolation can be bypassed by bugs or SQL injection, requiring defense in depth
> - **Options Considered**: Application-only isolation, database views with session variables, row-level security policies
> - **Decision**: PostgreSQL row-level security policies with automatic tenant context
> - **Rationale**: Database-level enforcement provides defense against application bugs, automatic tenant filtering prevents accidental cross-tenant queries, leverages database security expertise
> - **Consequences**: Requires PostgreSQL-specific features, adds database configuration complexity, provides strongest isolation guarantee

#### Operational Requirements

Operational requirements determine how well the system supports production deployment, monitoring, and maintenance. These requirements often determine system success more than functional features because they affect the team's ability to operate the system reliably.

**Observability** must provide visibility into authorization decisions, performance patterns, and system health. Operations teams need to understand why the system is slow, what's causing authorization failures, and how to diagnose permission problems.

**Configuration Management** must support policy and role changes without system restarts. The system should reload configuration changes dynamically while maintaining consistency across multiple service instances.

| Operational Requirement | Specification | Operational Context |
|---|---|---|
| Health Monitoring | HTTP health check endpoints with dependency status | Load balancer integration and alerting automation |
| Metrics Export | Prometheus metrics for authorization rates, latency, errors | Integration with existing monitoring infrastructure |
| Structured Logging | JSON logs with correlation IDs and request context | Log aggregation and searchability for debugging |
| Configuration Reload | Policy changes applied without service restart | Zero-downtime policy updates for production systems |
| Graceful Degradation | Service continues with cached results during database outages | Maintains availability during infrastructure failures |

**Deployment Safety** must support rolling updates and rollback capabilities. The system should support canary deployments where new policy changes apply to a subset of requests before full rollout.

**Backup and Recovery** must handle both configuration data (policies, roles) and operational data (audit logs). The system should support point-in-time recovery for compliance requirements.

### What We Won't Build

Clearly defining what we won't build is as important as defining what we will build. Authorization systems have natural scope creep toward adjacent security concerns, and resisting this creep is essential for project success.

#### Authentication and Identity Management

**Authentication** (verifying user identity) is explicitly out of scope - we assume users are already authenticated when they reach the authorization service. Think of this like a security checkpoint inside a building that checks your badge to determine what floors you can access, but doesn't verify that you're the person whose name is on the badge - that verification happened at the front entrance.

The authorization system will receive user identifiers in requests but will not verify passwords, handle login/logout, manage sessions, or integrate with identity providers like Active Directory or SAML systems. These concerns belong to dedicated authentication services.

| Authentication Concern | Why Out of Scope | Recommended Alternative |
|---|---|---|
| Password Verification | Authentication expertise separate from authorization logic | OAuth 2.0 / OpenID Connect identity provider |
| Session Management | Session complexity unrelated to permission decisions | JWT tokens with short expiry and refresh |
| Multi-factor Authentication | Hardware integration and user experience concerns | Dedicated authentication service with MFA support |
| User Registration | User lifecycle management separate from access control | Identity management system or user service |
| Password Reset | Account recovery flows unrelated to authorization | Authentication service with secure reset flows |

**User Management** functionality like user creation, profile management, and user lifecycle is also out of scope. The authorization system will reference users by identifier but will not store user profiles, handle user provisioning, or manage user attributes beyond what's necessary for authorization decisions.

#### Application-Specific Business Logic

**Business Rule Enforcement** beyond access control is out of scope. While our authorization system might check whether a user can approve a purchase order, it won't enforce business rules like "purchase orders over $10,000 require two approvals" unless those rules can be expressed purely as access control policies.

The system will not implement workflow engines, business process management, or domain-specific validation rules. These concerns belong in application services that use the authorization system to check permissions.

| Business Logic Concern | Why Out of Scope | How Authorization Helps |
|---|---|---|
| Workflow State Management | Business process logic separate from access control | Authorization checks at each workflow transition |
| Data Validation | Domain-specific rules unrelated to permissions | Authorization validates permission to modify data |
| Notification Systems | Communication concerns separate from access decisions | Authorization determines who can receive notifications |
| Reporting and Analytics | Business intelligence separate from access audit | Authorization protects access to generated reports |
| Integration Orchestration | Data flow coordination separate from permission checks | Authorization secures each integration endpoint |

#### Infrastructure and Platform Services

**Service Mesh and Network Security** concerns like TLS termination, service-to-service authentication, and network policies are out of scope. The authorization service will integrate with these systems but won't provide them.

**Secrets Management** for storing sensitive configuration like database passwords or API keys is out of scope. The authorization service will use secrets but won't provide secret storage, rotation, or distribution capabilities.

| Infrastructure Concern | Why Out of Scope | Integration Approach |
|---|---|---|
| TLS Certificate Management | Certificate lifecycle separate from authorization logic | Service mesh or ingress controller handles TLS |
| Service Discovery | Network topology separate from permission decisions | Authorization service registers with existing service registry |
| Load Balancing | Traffic distribution separate from authorization logic | External load balancer routes to authorization service instances |
| Database Administration | Data management separate from authorization logic | Authorization service connects to managed database service |
| Monitoring Infrastructure | Observability platform separate from authorization domain | Authorization service exports metrics to existing systems |

#### Advanced Authorization Patterns

**Relationship-Based Access Control (ReBAC)** where permissions derive from entity relationships is out of scope for the initial system. While powerful, ReBAC adds significant complexity and is not needed for the core RBAC/ABAC use cases we're targeting.

**Machine Learning-Based Access** patterns like anomaly detection or dynamic risk scoring are out of scope. These advanced features can be built on top of the audit data our system provides, but the core authorization engine will use deterministic policy evaluation.

| Advanced Pattern | Why Deferred | Future Integration Path |
|---|---|---|
| Relationship Graphs | Graph traversal complexity beyond current scope | ReBAC engine built on authorization service foundation |
| Dynamic Risk Scoring | ML expertise and infrastructure beyond current team | Risk scores provided as ABAC attributes |
| Time-based Access Patterns | Complex temporal logic beyond basic time conditions | Enhanced policy language in future versions |
| Delegation and Proxy Access | Complex permission transfer logic | Delegation implemented as explicit role assignments |
| External Policy Engines | Integration complexity with systems like OPA | Policy translation layer for external engines |

> **The Build vs. Buy Decision Point**
> 
> Authorization systems have a critical decision point between building custom solutions and integrating existing authorization services. We're building custom because our multi-tenancy requirements, performance needs, and specific RBAC/ABAC combination aren't well-served by existing solutions. However, we're explicitly avoiding rebuilding mature technologies like authentication (OAuth/SAML), user management (LDAP/Active Directory), or secrets management (HashiCorp Vault) where excellent solutions already exist.

### Common Pitfalls

⚠️ **Pitfall: Scope Creep into Authentication**
Teams frequently blur the line between authentication and authorization, leading to authorization systems that try to handle password verification, session management, and user registration. This scope creep leads to security vulnerabilities because authorization teams lack authentication expertise, and it delays the authorization system while building unrelated features. Keep authentication concerns in dedicated identity services and accept user identifiers as input to authorization decisions.

⚠️ **Pitfall: Over-Engineering Quality Attributes**
Authorization systems often specify unrealistic performance requirements like "sub-millisecond latency" without understanding the cost implications. Sub-millisecond authorization requires in-memory caching of all permissions, which limits system flexibility and increases memory requirements dramatically. Specify quality attributes based on actual application needs, not theoretical ideals. Most applications can tolerate 50-100ms authorization latency in exchange for more flexible policy evaluation.

⚠️ **Pitfall: Underspecifying Tenant Isolation**
Multi-tenant authorization systems often specify "tenant isolation" without defining what level of isolation is required. Application-level isolation (filtering by tenant ID) is much simpler than cryptographic isolation (separate encryption keys per tenant), but provides different security guarantees. Specify the exact isolation requirements based on regulatory needs and threat model rather than assuming "tenant isolation" means the same thing to all stakeholders.

⚠️ **Pitfall: Ignoring Audit Requirements Early**
Teams often treat audit logging as an afterthought that can be added later, but audit requirements significantly affect system architecture. Immutable audit logs require different database design than mutable logs, and compliance requirements affect what data must be captured during authorization decisions. Define audit requirements during goal-setting rather than discovering them during implementation.

### Implementation Guidance

The implementation guidance for this goals and non-goals section focuses on establishing project structure and defining interfaces that will guide the rest of the implementation.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|---|---|---|
| API Framework | HTTP REST with Go net/http and gorilla/mux | gRPC with Protocol Buffers for type safety |
| Database | PostgreSQL with pgx driver | PostgreSQL with row-level security policies |
| Caching | Redis with go-redis client | Redis Cluster with consistent hashing |
| Configuration | YAML files with viper | etcd with dynamic configuration updates |
| Logging | Structured JSON with logrus | OpenTelemetry with distributed tracing |
| Metrics | Prometheus with prometheus/client_golang | Prometheus with custom histogram buckets |

#### Recommended Project Structure

The project structure must support the milestone-based development approach while maintaining clear separation between RBAC and ABAC concerns.

```
authorization-system/
├── cmd/
│   ├── authz-server/main.go          # HTTP server entry point
│   └── authz-cli/main.go             # Policy testing CLI
├── internal/
│   ├── authz/                        # Core authorization logic
│   │   ├── authorizer.go            # Main Authorizer interface
│   │   ├── request.go               # AuthzRequest and AuthzDecision types
│   │   └── authorizer_test.go       # Integration tests
│   ├── rbac/                        # Role-based access control (Milestone 1)
│   │   ├── engine.go                # RoleEngine implementation
│   │   ├── role.go                  # Role and Permission types
│   │   └── hierarchy.go             # Role inheritance logic
│   ├── abac/                        # Attribute-based access control (Milestone 2)
│   │   ├── engine.go                # PolicyEngine implementation
│   │   ├── policy.go                # Policy and Condition types
│   │   └── evaluator.go             # Policy evaluation logic
│   ├── resources/                   # Resource management (Milestone 3)
│   │   ├── manager.go               # Resource ownership and hierarchy
│   │   ├── tenant.go                # Multi-tenancy isolation
│   │   └── sharing.go               # Cross-tenant access
│   ├── audit/                       # Audit logging (Milestone 4)
│   │   ├── logger.go                # Audit event capture
│   │   ├── storage.go               # Immutable audit storage
│   │   └── reporter.go              # Compliance reporting
│   ├── storage/                     # Database abstractions
│   │   ├── postgres.go              # PostgreSQL implementation
│   │   └── migrations/              # Database schema versions
│   └── cache/                       # Caching layer
│       ├── redis.go                 # Redis cache implementation
│       └── memory.go                # In-memory cache for testing
├── pkg/                             # Public APIs
│   └── authzclient/                 # Go client library
│       ├── client.go                # HTTP client implementation
│       └── types.go                 # Shared type definitions
├── api/                             # API definitions
│   ├── authz.proto                  # gRPC service definition
│   └── openapi.yaml                 # REST API specification
├── configs/                         # Configuration files
│   ├── development.yaml             # Dev environment config
│   └── production.yaml              # Production environment config
└── deployments/                     # Deployment configurations
    ├── docker-compose.yml           # Local development stack
    └── kubernetes/                  # K8s deployment manifests
```

#### Core Interface Definitions

These interfaces establish the contracts between components and guide implementation across all milestones.

```go
// Package authz provides the main authorization interfaces and types
package authz

import (
    "context"
    "time"
)

// Authorizer is the main interface for authorization decisions.
// This interface will be implemented by a composite service that delegates
// to RBAC and ABAC engines based on request type and policy configuration.
type Authorizer interface {
    // IsAuthorized makes the primary authorization decision for a request.
    // Returns decision with reasoning and applied policies for audit trail.
    IsAuthorized(ctx context.Context, req *AuthzRequest) (*AuthzDecision, error)
    
    // TODO: Add GetUserPermissions method for RBAC permission enumeration
    // TODO: Add ValidatePolicy method for policy testing framework
    // TODO: Add methods for cache management and health checks
}

// AuthzRequest represents an authorization check request with all context
// needed for both RBAC and ABAC evaluation.
type AuthzRequest struct {
    // TODO: Add validation tags and JSON serialization
    UserID    string                 // Authenticated user identifier
    TenantID  string                 // Tenant scope for multi-tenancy
    Action    string                 // Action being attempted (read, write, delete)
    Resource  *Resource              // Target resource with attributes
    Attributes map[string]interface{} // Additional context attributes
    Timestamp time.Time              // Request timestamp for time-based policies
    RequestID string                 // Unique request ID for audit correlation
}

// AuthzDecision contains the authorization result with full reasoning
// for audit logs and debugging.
type AuthzDecision struct {
    // TODO: Add methods for decision serialization and comparison
    Allowed        bool              // Final authorization decision
    Reason         string            // Human-readable decision explanation
    Method         EvaluationMethod  // How the decision was made (RBAC/ABAC/cached)
    AppliedPolicies []string         // Policies that contributed to decision
    EvaluationTime time.Duration     // Time spent on evaluation
    CacheHit       bool              // Whether decision came from cache
}

// EvaluationMethod constants for tracking decision source
const (
    EvaluationRBAC      EvaluationMethod = "rbac"
    EvaluationABAC      EvaluationMethod = "abac"
    EvaluationOwnership EvaluationMethod = "ownership"
    EvaluationCached    EvaluationMethod = "cached"
)
```

#### Configuration Structure

The configuration system must support the quality attributes defined in goals while remaining simple enough for development and testing.

```yaml
# configs/development.yaml
server:
  host: "localhost"
  port: 8080
  read_timeout: "30s"
  write_timeout: "30s"
  
database:
  host: "localhost"
  port: 5432
  name: "authorization_dev"
  user: "authz_user"
  password: "dev_password"
  max_connections: 10
  
cache:
  type: "redis"  # or "memory" for testing
  host: "localhost"
  port: 6379
  ttl: "300s"    # 5 minute default cache TTL
  
audit:
  enabled: true
  storage: "database"  # or "file" for development
  retention: "90d"
  
# Performance tuning
performance:
  max_policy_evaluation_time: "500ms"
  cache_hit_rate_target: 0.90
  concurrent_requests: 1000
```

#### Milestone Checkpoints

Each milestone should have clear acceptance criteria that can be verified through automated tests and manual validation.

**Milestone 1 Checkpoint - RBAC Engine:**
- Run: `go test ./internal/rbac/...` - all tests pass
- Start server: `go run cmd/authz-server/main.go`
- Test role creation: `curl -X POST localhost:8080/roles -d '{"name":"viewer","permissions":[{"resource":"documents","action":"read"}]}'`
- Test authorization: `curl -X POST localhost:8080/authorize -d '{"user_id":"user1","action":"read","resource":{"type":"documents","id":"doc1"}}'`
- Expected: Returns `{"allowed":true,"method":"rbac","reason":"User has viewer role"}`

**Milestone 2 Checkpoint - ABAC Policies:**
- Run: `go test ./internal/abac/...` - policy evaluation tests pass
- Create time-based policy: Policy allowing access only during business hours
- Test outside hours: Should return `{"allowed":false,"method":"abac","reason":"Access denied outside business hours"}`
- Test during hours: Should return `{"allowed":true,"method":"abac","applied_policies":["business-hours-policy"]}`

**Milestone 3 Checkpoint - Multi-tenancy:**
- Create resources in different tenants
- Verify cross-tenant isolation: User from tenant A cannot access tenant B resources
- Test resource sharing: Explicit sharing grants should work across tenants
- Database query verification: All queries should include tenant filters

**Milestone 4 Checkpoint - Audit and Testing:**
- Every authorization decision should generate audit log entry
- Policy simulation should allow testing without affecting production decisions
- Compliance reports should generate from audit logs
- Audit log integrity verification should detect tampering attempts

#### Debugging Setup

Authorization systems require specialized debugging because decision logic involves multiple data sources and complex evaluation chains.

```go
// internal/authz/debug.go
package authz

import (
    "encoding/json"
    "log"
    "os"
)

// DebugConfig enables detailed logging for authorization debugging
type DebugConfig struct {
    LogRequests      bool `json:"log_requests"`
    LogDecisions     bool `json:"log_decisions"`
    LogCacheHits     bool `json:"log_cache_hits"`
    LogPolicyEval    bool `json:"log_policy_evaluation"`
    TraceLevel       int  `json:"trace_level"` // 0=off, 1=basic, 2=verbose
}

// TODO: Implement debug logging middleware that wraps Authorizer
// TODO: Add request tracing with correlation IDs
// TODO: Add policy evaluation step logging
// TODO: Add performance profiling hooks
```

The debugging setup should support the common troubleshooting scenarios identified in the pitfalls section, with structured logging that can be easily searched and filtered during incident response.


## High-Level Architecture

> **Milestone(s):** All milestones - this section establishes the architectural foundation that supports role management (Milestone 1), policy evaluation (Milestone 2), multi-tenancy (Milestone 3), and audit logging (Milestone 4)

Building an authorization system is like designing a modern airport security checkpoint. You have multiple layers of verification working together: the ticket counter validates your boarding pass (RBAC), security scanners evaluate risk factors like destination and baggage contents (ABAC), gate agents verify you're boarding the right flight (resource-based access), and cameras record every interaction for later review (audit logging). Each layer has a specific purpose, but they must coordinate seamlessly to make the final allow/deny decision while maintaining detailed records of every check performed.

The key architectural insight is that authorization decisions require multiple evaluation methods working in concert. A single approach like pure RBAC or pure ABAC cannot handle the complexity of modern multi-tenant applications. Instead, we need a **layered evaluation architecture** where each layer specializes in a particular type of access control, and a central orchestrator combines their decisions using well-defined precedence rules.

### System Components

The authorization system consists of five core components that work together to evaluate access requests and maintain audit trails. Each component has a distinct responsibility and operates independently while contributing to the overall authorization decision.

![System Component Architecture](./diagrams/system-components.svg)

The **Authorizer** serves as the central orchestrator and primary entry point for all authorization requests. Think of it as the head security officer who receives a request and knows exactly which specialists to consult before making the final decision. The Authorizer receives an `AuthzRequest` containing user identity, tenant context, requested action, target resource, and additional attributes. It coordinates with the specialized engines, applies decision combining logic, and returns a comprehensive `AuthzDecision` that includes not just the allow/deny result but also the reasoning, applied policies, and performance metrics.

| Component | Primary Responsibility | Key Operations | Dependencies |
|-----------|----------------------|----------------|--------------|
| Authorizer | Central orchestration and decision combining | `IsAuthorized`, decision caching, fail-safe defaults | All other components |
| RoleEngine | RBAC evaluation and permission inheritance | `HasPermission`, `GetUserPermissions`, role hierarchy traversal | Role/Permission data store |
| PolicyEngine | ABAC policy evaluation and condition matching | `Evaluate`, condition parsing, context validation | Policy data store, attribute providers |
| ResourceManager | Resource ownership and tenant isolation | Resource lookup, ownership validation, hierarchy traversal | Resource data store |
| AuditLogger | Immutable decision logging and compliance | Decision recording, integrity verification, report generation | Audit data store |

The **RoleEngine** specializes in traditional role-based access control evaluation. It maintains the role hierarchy as a directed acyclic graph (DAG) and can efficiently compute transitive permissions through role inheritance. When asked if a user has a specific permission, it traverses the user's assigned roles, walks up the role hierarchy to collect inherited roles, and checks if any of those roles grant the requested permission. The engine uses bitmap encoding and caching to make these lookups extremely fast, even with complex role hierarchies.

The **PolicyEngine** handles attribute-based access control by evaluating dynamic conditions against the request context. Unlike RBAC which deals with static role assignments, ABAC policies can examine runtime attributes like time of day, IP address, resource sensitivity level, or business logic conditions. The PolicyEngine parses policy expressions into abstract syntax trees (ASTs), evaluates them against the provided context, and returns policy decisions with full provenance tracking. It supports complex boolean logic with AND, OR, and NOT operators, making it capable of expressing sophisticated business rules.

> **Decision: Separate RBAC and ABAC Engines**
> - **Context**: Authorization systems can implement RBAC as a subset of ABAC, or maintain separate engines for each paradigm
> - **Options Considered**: 
>   1. Single unified policy engine handling both RBAC and ABAC through policy expressions
>   2. Separate specialized engines with distinct APIs and optimization strategies
>   3. RBAC as a lightweight wrapper around ABAC with role-to-policy translation
> - **Decision**: Separate specialized engines (option 2)
> - **Rationale**: RBAC requires microsecond-level performance for high-frequency permission checks, while ABAC involves more expensive condition evaluation. Separate engines allow RBAC to use bitmap encoding and aggressive caching without the overhead of general-purpose policy evaluation. This also provides clearer APIs for different use cases.
> - **Consequences**: More complex orchestration logic in the Authorizer, but significantly better performance for RBAC-heavy workloads and clearer separation of concerns

The **ResourceManager** enforces resource-level access control and tenant isolation. It understands resource ownership, hierarchical relationships between resources, and cross-tenant sharing arrangements. When evaluating access to a specific resource, it checks if the user owns the resource directly, has inherited access through resource hierarchy, or has been granted explicit access through sharing. Most importantly, it enforces tenant boundaries by ensuring users can only access resources within their tenant scope, unless explicitly granted cross-tenant permissions.

The **AuditLogger** captures every authorization decision in an immutable, tamper-evident log. It records not just the final decision but the complete evaluation trace: which engines were consulted, which policies matched, what conditions were evaluated, and how long each step took. This detailed provenance enables compliance reporting, security incident investigation, and policy debugging. The logger uses write-ahead logging with cryptographic integrity verification to ensure audit records cannot be modified after creation.

> The critical architectural principle is **fail-safe defaults**: if any component is unavailable, the system denies access rather than potentially granting unauthorized permissions. This means the Authorizer must be prepared to operate in degraded modes while maintaining security guarantees.

### Authorization Request Flow

Understanding the authorization flow requires thinking about it as a **decision pipeline** where each stage can either allow the request to continue, deny it immediately, or add information for later stages to consider. The flow is designed to be both comprehensive and efficient, checking simpler conditions first before moving to more expensive evaluations.

The authorization process begins when a client application needs to verify that a user can perform a specific action on a particular resource. The application constructs an `AuthzRequest` containing all relevant context and submits it to the Authorizer. This request includes not just the basic who/what/where information, but also environmental attributes like IP address, time of day, and any business-specific context that policies might need to evaluate.

| Request Phase | Component | Input | Output | Decision Logic |
|---------------|-----------|-------|--------|----------------|
| 1. Request Validation | Authorizer | `AuthzRequest` | Validated context | Reject malformed requests, apply defaults |
| 2. Cache Lookup | Authorizer | Request hash | `AuthzDecision` or miss | Return cached decision if valid and recent |
| 3. Tenant Isolation | ResourceManager | TenantID, ResourceID | Access scope | Verify user can access resource's tenant |
| 4. Ownership Check | ResourceManager | UserID, Resource | Ownership status | Grant full access if user owns resource |
| 5. RBAC Evaluation | RoleEngine | UserID, Permission | RBAC decision | Check role-based permissions |
| 6. ABAC Evaluation | PolicyEngine | Full context | Policy decisions | Evaluate attribute-based conditions |
| 7. Decision Combining | Authorizer | All decisions | Final decision | Apply deny-overrides logic |
| 8. Result Caching | Authorizer | Decision + TTL | Cached entry | Store result for future requests |
| 9. Audit Logging | AuditLogger | Complete trace | Log entry | Record immutable decision record |

**Phase 1: Request Validation and Preparation**
The Authorizer first validates that the incoming request contains all required fields and applies sensible defaults where possible. It generates a unique request ID for tracing, captures the current timestamp, and ensures the tenant context is properly set. Invalid requests are rejected immediately with clear error messages, preventing downstream components from having to handle malformed input.

**Phase 2: Cache Lookup and Short-Circuit**
Before performing expensive evaluations, the Authorizer computes a cache key based on the request parameters and checks if a recent decision exists. The cache key includes user ID, tenant ID, resource ID, action, and a hash of relevant attributes. If a valid cached decision is found, it's returned immediately with the `CacheHit` flag set to true. This optimization is crucial for high-frequency permission checks like UI element visibility.

**Phase 3: Tenant Isolation Enforcement**
The ResourceManager verifies that the user has permission to access resources within the specified tenant. This is the first security-critical check that prevents accidental or malicious cross-tenant access. The check examines the user's tenant membership and any cross-tenant grants that might apply. If the user has no valid path to access the resource's tenant, the request is denied immediately.

**Phase 4: Resource Ownership Evaluation**
If the user is authorized to access the tenant, the ResourceManager checks direct resource ownership. Resource owners typically have full control over their resources, so this check can grant access without needing to consult RBAC or ABAC policies. The ownership check also considers hierarchical ownership, where owning a parent resource grants ownership of child resources.

**Phase 5: Role-Based Access Control (RBAC)**
If ownership doesn't grant access, the RoleEngine evaluates the user's role-based permissions. It retrieves the user's role assignments within the tenant, computes the transitive closure of inherited roles through the role hierarchy, and checks if any of those roles grant the requested permission. The RBAC evaluation is optimized for speed using precomputed permission bitmaps and aggressive caching.

**Phase 6: Attribute-Based Access Control (ABAC)**
The PolicyEngine evaluates all policies that might apply to the request. It matches policies based on resource patterns and action patterns, then evaluates their conditions against the request context. Each policy can render an allow or deny decision, and the engine tracks which policies matched for audit purposes. ABAC evaluation is the most expensive phase because it involves parsing and evaluating arbitrary expressions.

**Phase 7: Decision Combining Logic**
The Authorizer applies **deny-overrides** logic to combine the various decisions. If any evaluation method explicitly denies access, the final decision is deny. If at least one method allows access and none deny it, the final decision is allow. If no method grants access (all abstain), the fail-safe default is deny. The combining logic also determines which evaluation method to report as the primary decision source.

**Phase 8: Result Caching and Performance Optimization**
Successful decisions are cached with appropriate TTLs based on the volatility of the decision factors. RBAC-based decisions can be cached longer since role assignments change infrequently. ABAC decisions with time-based conditions might have very short cache TTLs. The caching strategy balances performance against data freshness requirements.

**Phase 9: Comprehensive Audit Logging**
Finally, the AuditLogger records the complete decision trace, including all intermediate results, timing information, and the final decision rationale. This audit record is immutable once written and includes enough information to reproduce the decision later if needed for compliance or debugging purposes.

> **Decision: Deny-Overrides Decision Combining**
> - **Context**: When multiple evaluation methods (RBAC, ABAC, ownership) render different decisions, we need a consistent way to combine them
> - **Options Considered**:
>   1. First-applicable: return the first non-abstain decision
>   2. Allow-overrides: any allow decision overrides deny decisions
>   3. Deny-overrides: any deny decision overrides allow decisions
> - **Decision**: Deny-overrides (option 3)
> - **Rationale**: Security systems should fail closed rather than open. If any evaluation method identifies a reason to deny access, that should take precedence over other methods that might allow it. This prevents policy conflicts from accidentally granting unauthorized access.
> - **Consequences**: Policies must be carefully designed to avoid unintended denies, but the security posture is stronger with explicit deny-by-default behavior

### Recommended Code Organization

The authorization system requires careful code organization to manage the complexity of multiple evaluation engines, data models, and cross-cutting concerns like audit logging. The module structure should clearly separate responsibilities while providing clean interfaces between components.

```
authz-system/
├── cmd/
│   ├── authz-server/main.go          ← HTTP/gRPC server entry point
│   ├── authz-cli/main.go             ← Command-line administration tool
│   └── policy-simulator/main.go      ← Policy testing and simulation tool
├── internal/
│   ├── authorizer/                   ← Central authorization orchestrator
│   │   ├── authorizer.go             ← Main Authorizer implementation
│   │   ├── decision.go               ← Decision combining logic
│   │   ├── cache.go                  ← Authorization decision caching
│   │   └── authorizer_test.go        ← End-to-end authorization tests
│   ├── rbac/                         ← Role-Based Access Control engine
│   │   ├── engine.go                 ← RoleEngine implementation
│   │   ├── hierarchy.go              ← Role hierarchy and DAG validation
│   │   ├── permissions.go            ← Permission lookup and caching
│   │   └── templates.go              ← Role template system
│   ├── abac/                         ← Attribute-Based Access Control engine
│   │   ├── engine.go                 ← PolicyEngine implementation
│   │   ├── parser.go                 ← Policy expression parsing
│   │   ├── evaluator.go              ← Condition evaluation logic
│   │   └── context.go                ← Evaluation context management
│   ├── resources/                    ← Resource management and tenant isolation
│   │   ├── manager.go                ← ResourceManager implementation
│   │   ├── ownership.go              ← Resource ownership logic
│   │   ├── hierarchy.go              ← Parent-child resource relationships
│   │   └── sharing.go                ← Cross-tenant sharing mechanisms
│   ├── audit/                        ← Audit logging and compliance
│   │   ├── logger.go                 ← AuditLogger implementation
│   │   ├── integrity.go              ← Cryptographic log integrity
│   │   ├── reports.go                ← Compliance report generation
│   │   └── storage.go                ← Immutable audit log storage
│   ├── models/                       ← Shared data structures and types
│   │   ├── request.go                ← AuthzRequest, AuthzDecision types
│   │   ├── rbac.go                   ← Role, Permission, UserRoleAssignment
│   │   ├── abac.go                   ← Policy, Condition, PolicyDecision
│   │   ├── resources.go              ← Resource, tenant-related types
│   │   └── audit.go                  ← Audit log entry structures
│   ├── storage/                      ← Data persistence abstractions
│   │   ├── interfaces.go             ← Storage interface definitions
│   │   ├── postgres/                 ← PostgreSQL implementation
│   │   ├── redis/                    ← Redis caching implementation
│   │   └── memory/                   ← In-memory testing implementation
│   ├── transport/                    ← API transport layers
│   │   ├── http/                     ← REST API handlers
│   │   ├── grpc/                     ← gRPC service implementation
│   │   └── middleware/               ← Authentication, logging, metrics
│   └── testing/                      ← Shared testing utilities
│       ├── fixtures/                 ← Test data and scenarios
│       ├── mocks/                    ← Generated mock interfaces
│       └── integration/              ← Integration test helpers
├── api/                              ← API definitions and documentation
│   ├── authz.proto                   ← gRPC service definitions
│   ├── openapi.yaml                  ← REST API specification
│   └── examples/                     ← API usage examples
├── docs/                             ← Additional documentation
│   ├── deployment/                   ← Deployment guides and configs
│   ├── examples/                     ← Complete usage examples
│   └── architecture/                 ← Design documents and diagrams
└── scripts/                          ← Development and deployment scripts
    ├── generate.sh                   ← Code generation (mocks, protos)
    ├── migrate.sh                    ← Database migration scripts
    └── test.sh                       ← Comprehensive test runner
```

**Core Package Responsibilities**

The `internal/authorizer` package contains the main orchestration logic and serves as the primary entry point for authorization requests. It implements the `Authorizer` interface and coordinates calls to the specialized engines. The authorizer package is responsible for request validation, decision caching, result combining, and ensuring that all authorization requests are properly audited. It should have minimal business logic itself, instead delegating to the appropriate specialized engines.

The `internal/rbac` package implements the role-based access control engine with optimizations for high-frequency permission checks. It maintains the role hierarchy graph, handles role inheritance computation, and provides efficient permission lookups. The package includes role template functionality for creating preconfigured role bundles and validates the role hierarchy to prevent cycles. All RBAC operations should be tenant-scoped to ensure proper isolation.

The `internal/abac` package implements the attribute-based policy engine with support for complex condition evaluation. It includes a policy expression parser that converts human-readable policy conditions into evaluable ASTs, an evaluation engine that can efficiently process these conditions against request context, and context management utilities that handle attribute resolution from various sources.

The `internal/resources` package manages resource ownership, hierarchical relationships, and cross-tenant sharing. It enforces tenant isolation at the resource level and provides APIs for checking ownership, navigating resource hierarchies, and managing time-limited sharing grants. This package is crucial for multi-tenant security and must never allow accidental cross-tenant access.

The `internal/models` package defines all shared data structures used throughout the system. These models should be designed for efficiency and clarity, with careful attention to JSON serialization for API compatibility. The package includes validation logic to ensure data integrity and helper methods for common operations like permission matching and condition evaluation.

> **Decision: Internal Package Organization by Component**
> - **Context**: Go projects can organize internal packages by layer (models, services, handlers) or by component (rbac, abac, resources)
> - **Options Considered**:
>   1. Layered organization: models/, services/, handlers/, storage/
>   2. Component organization: rbac/, abac/, resources/, audit/
>   3. Hybrid approach: core components as packages, cross-cutting concerns as layers
> - **Decision**: Component organization with cross-cutting layers (option 3)
> - **Rationale**: Authorization logic is domain-heavy, and keeping related functionality together makes the code more maintainable. RBAC logic, ABAC logic, and resource management are distinct domains that benefit from encapsulation. However, cross-cutting concerns like storage and transport are better organized as layers since they serve all components.
> - **Consequences**: Clear separation of authorization concerns, but requires careful interface design to prevent circular dependencies between components

**Import Guidelines and Dependency Management**

The dependency flow should be carefully controlled to prevent circular imports and maintain clear architectural layers. The `models` package should have no dependencies on other internal packages, making it safe to import anywhere. The engine packages (`rbac`, `abac`, `resources`) should depend only on `models` and `storage` interfaces, never on each other directly. The `authorizer` package orchestrates the engines and is the only package that imports all of them.

External dependencies should be minimized and abstracted through interfaces defined in the internal packages. Database access should go through storage interfaces, HTTP frameworks should be isolated to the transport layer, and third-party libraries should be wrapped to prevent vendor lock-in. This approach makes the system more testable and allows components to be developed and tested independently.

**Testing Organization and Strategy**

Each package should include comprehensive unit tests that cover both happy paths and error conditions. The testing should use dependency injection and mocking to isolate units under test. Integration tests should live in the `testing/integration` package and exercise complete authorization flows against real storage backends.

The `testing/fixtures` package should contain realistic test data that represents common authorization scenarios: complex role hierarchies, multi-tenant setups, various policy conditions, and edge cases like resource sharing and cross-tenant access. These fixtures should be used consistently across all test suites to ensure behavior consistency.

### Common Pitfalls

⚠️ **Pitfall: Circular Dependencies Between Engines**
Beginners often try to make the RoleEngine call the PolicyEngine or vice versa, creating circular import dependencies. For example, implementing a policy condition that checks if a user has a specific role. This breaks the clean separation of concerns and makes the system harder to test and maintain. Instead, the Authorizer should orchestrate calls to both engines and combine their results. If you need role information in ABAC policies, pass role assignments as attributes in the evaluation context rather than making direct engine calls.

⚠️ **Pitfall: Leaking Tenant Context**
A dangerous mistake is allowing any component to access resources without proper tenant scoping. This can happen when passing resource IDs directly to storage layers without including tenant context, or when caching decisions without including tenant ID in the cache key. Always ensure that every data access operation includes tenant context, and design your APIs to make it impossible to accidentally access cross-tenant data. Use strongly-typed tenant IDs and validate them at every system boundary.

⚠️ **Pitfall: Inconsistent Error Handling**
Authorization systems need consistent error handling because security decisions must be predictable. Some developers return different error types for missing users, invalid resources, or policy evaluation failures, making it hard for clients to handle errors appropriately. Instead, define a clear error taxonomy and ensure all components use it consistently. Security-sensitive errors should not leak information that could help attackers, so be careful about error message details.

⚠️ **Pitfall: Performance Antipatterns in Request Flow**
The authorization flow can easily become inefficient if not carefully designed. Common mistakes include making database queries in loops (N+1 problem when looking up user roles), not implementing proper caching layers, or performing expensive policy evaluation for every request even when simpler methods would suffice. Design the evaluation order to check faster conditions first, implement appropriate caching with correct invalidation, and use bulk operations when loading related data.

⚠️ **Pitfall: Audit Logging as an Afterthought**
Many implementations add audit logging at the end and miss important decision points or fail to capture enough context for meaningful compliance reporting. Audit logging should be designed into the architecture from the beginning, with clear requirements for what information needs to be captured at each decision point. The audit trail should be sufficient to reproduce authorization decisions for debugging and compliance purposes.

### Implementation Guidance

This section provides concrete guidance for implementing the high-level architecture using Go, including technology recommendations, starter code for infrastructure components, and structured guidance for building the core authorization logic.

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option | Rationale |
|-----------|--------------|----------------|-----------|
| HTTP Server | `net/http` with `gorilla/mux` | `gin-gonic/gin` or `fiber` | Standard library is sufficient for authorization APIs, advanced frameworks add minimal value |
| Database | `PostgreSQL` with `database/sql` | `PostgreSQL` with `GORM` or `sqlx` | Authorization needs ACID transactions, PostgreSQL's JSON support helps with flexible attributes |
| Caching | `Redis` with `go-redis/redis` | `Redis Cluster` with `rueidis` | Authorization decisions are highly cacheable, Redis provides TTL and atomic operations |
| Configuration | `Viper` with YAML files | `Consul` or `etcd` integration | Simple config files work well, distributed config needed only for large deployments |
| Logging | `logrus` or `zap` | `OpenTelemetry` with distributed tracing | Authorization decisions need structured logging, tracing helps with debugging complex flows |
| Testing | `testify` with `gomock` | `Ginkgo` with `Gomega` | Standard testing with mocks covers most needs, BDD frameworks help with complex scenarios |

**B. Recommended File Structure Starter**

```go
// cmd/authz-server/main.go - Server entry point
package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"

    "github.com/your-org/authz-system/internal/authorizer"
    "github.com/your-org/authz-system/internal/transport/http/handlers"
    "github.com/your-org/authz-system/internal/storage/postgres"
)

func main() {
    // TODO: Load configuration
    // TODO: Initialize database connections
    // TODO: Create authorization components
    // TODO: Setup HTTP server with graceful shutdown
}
```

**C. Infrastructure Starter Code (Complete)**

```go
// internal/models/request.go - Core request/response types
package models

import (
    "time"
)

type AuthzRequest struct {
    UserID     string                 `json:"user_id"`
    TenantID   string                 `json:"tenant_id"`
    Action     string                 `json:"action"`
    Resource   *Resource              `json:"resource"`
    Attributes map[string]interface{} `json:"attributes"`
    Timestamp  time.Time              `json:"timestamp"`
    RequestID  string                 `json:"request_id"`
}

type AuthzDecision struct {
    Allowed         bool              `json:"allowed"`
    Reason          string            `json:"reason"`
    Method          EvaluationMethod  `json:"method"`
    AppliedPolicies []string          `json:"applied_policies"`
    EvaluationTime  time.Duration     `json:"evaluation_time"`
    CacheHit        bool              `json:"cache_hit"`
}

type EvaluationMethod string

const (
    EvaluationRBAC      EvaluationMethod = "rbac"
    EvaluationABAC      EvaluationMethod = "abac"
    EvaluationOwnership EvaluationMethod = "ownership"
    EvaluationCached    EvaluationMethod = "cached"
)

type Resource struct {
    ID         string                 `json:"id"`
    Type       string                 `json:"type"`
    TenantID   string                 `json:"tenant_id"`
    OwnerID    string                 `json:"owner_id"`
    Attributes map[string]interface{} `json:"attributes"`
}
```

```go
// internal/storage/interfaces.go - Storage abstractions
package storage

import (
    "context"
    "github.com/your-org/authz-system/internal/models"
)

type RoleStorage interface {
    GetUserRoles(ctx context.Context, userID, tenantID string) ([]models.UserRoleAssignment, error)
    GetRoleHierarchy(ctx context.Context, tenantID string) ([]models.Role, error)
    GetRolePermissions(ctx context.Context, roleID string) ([]models.Permission, error)
}

type PolicyStorage interface {
    GetPoliciesForResource(ctx context.Context, resourceType, tenantID string) ([]models.Policy, error)
    GetPolicy(ctx context.Context, policyID string) (*models.Policy, error)
}

type ResourceStorage interface {
    GetResource(ctx context.Context, resourceID string) (*models.Resource, error)
    GetResourceOwnership(ctx context.Context, resourceID, userID string) (bool, error)
}

type AuditStorage interface {
    LogDecision(ctx context.Context, req *models.AuthzRequest, decision *models.AuthzDecision) error
}
```

**D. Core Logic Skeleton Code (TODOs only)**

```go
// internal/authorizer/authorizer.go - Main authorization orchestrator
package authorizer

import (
    "context"
    "time"
    "github.com/your-org/authz-system/internal/models"
)

type Authorizer struct {
    roleEngine     RoleEngine
    policyEngine   PolicyEngine
    resourceMgr    ResourceManager
    auditLogger    AuditLogger
    cache          DecisionCache
}

// IsAuthorized evaluates an authorization request through multiple engines
// and combines their decisions using deny-overrides logic.
func (a *Authorizer) IsAuthorized(ctx context.Context, req *models.AuthzRequest) (*models.AuthzDecision, error) {
    startTime := time.Now()
    
    // TODO 1: Validate request fields (UserID, TenantID, Action, Resource required)
    // TODO 2: Generate RequestID if not provided, set Timestamp if zero
    // TODO 3: Check decision cache using request hash as key
    // TODO 4: Call ResourceManager to verify tenant access and check ownership
    // TODO 5: If not owner, call RoleEngine.HasPermission for RBAC check
    // TODO 6: Call PolicyEngine.Evaluate for ABAC policy evaluation  
    // TODO 7: Combine decisions using deny-overrides logic (any deny wins)
    // TODO 8: Create AuthzDecision with result, reason, method, timing info
    // TODO 9: Cache the decision with appropriate TTL based on decision factors
    // TODO 10: Call AuditLogger.LogDecision with complete trace information
    // TODO 11: Return final decision with evaluation metadata
    
    return nil, nil // Replace with actual implementation
}

// combineDecisions applies deny-overrides logic to multiple authorization results
func (a *Authorizer) combineDecisions(ownership, rbac, abac bool) (allowed bool, method models.EvaluationMethod, reason string) {
    // TODO 1: If any decision is explicit deny, return deny immediately  
    // TODO 2: If ownership grants access, return allow with ownership method
    // TODO 3: If RBAC grants access and no ABAC deny, return allow with RBAC method
    // TODO 4: If ABAC grants access, return allow with ABAC method
    // TODO 5: If no method grants access (all abstain), return deny with fail-safe reason
    // Hint: Track which method provided the positive decision for audit trail
    
    return false, models.EvaluationRBAC, "fail-safe default deny"
}
```

**E. Language-Specific Hints**

- **Context Propagation**: Always pass `context.Context` as the first parameter to authorization methods. Use `context.WithTimeout()` for policy evaluation to prevent slow queries from blocking
- **Error Handling**: Define custom error types for different authorization failures: `ErrInvalidRequest`, `ErrAccessDenied`, `ErrInternalError`. Don't leak sensitive information in error messages
- **Concurrent Safety**: Use `sync.RWMutex` for caching structures that are read-heavy. Consider using `sync.Map` for high-concurrency cache access patterns
- **Database Connections**: Use connection pooling with `database/sql` and set appropriate timeouts. Authorization queries should complete within 100ms in most cases
- **JSON Handling**: Use struct tags for JSON serialization and implement custom marshalers for enums like `EvaluationMethod` to ensure API compatibility

**F. Milestone Checkpoint**

After implementing the basic architecture:

1. **Compile Check**: Run `go build ./cmd/authz-server` - should compile without errors
2. **Interface Validation**: Run `go test ./internal/authorizer` - interfaces should be properly implemented
3. **Basic Request Flow**: Start server and send a POST to `/authorize` with a minimal request - should return a decision (likely deny due to no policies)
4. **Component Integration**: Verify that all components are properly wired in the main function and dependency injection works

Expected server startup output:
```
INFO[0000] Starting authorization server on :8080
INFO[0000] Database connection established  
INFO[0000] Redis cache connection established
INFO[0000] Authorization components initialized
```

**G. Debugging Tips**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Authorization always denies | Missing role assignments or policies | Check logs for "no applicable policies" messages | Verify test data loaded correctly, check database queries |
| Slow authorization responses | No caching or inefficient queries | Add timing logs to each evaluation step | Implement decision caching, optimize database queries |
| Intermittent authorization failures | Race conditions in cache updates | Look for cache invalidation timing issues | Use atomic cache operations, add proper locking |
| Cross-tenant data access | Missing tenant isolation checks | Review audit logs for cross-tenant patterns | Add tenant validation to all resource queries |


## Data Model

> **Milestone(s):** Milestone 1 (Role & Permission Model), Milestone 2 (ABAC Policy Engine), Milestone 3 (Resource-Based & Multi-tenancy) - this section defines the core data structures that support role hierarchies, attribute-based policies, and multi-tenant resource management

The data model is the foundation of our authorization system - it defines how we represent users, roles, permissions, policies, and resources in memory and storage. Think of the data model as the **blueprint of a security office building**: just as an architect must define the structure of rooms, doors, keys, and access cards before construction begins, we must define the structure of our authorization entities and their relationships before implementing the business logic.

A well-designed data model serves three critical purposes in authorization systems. First, it **captures the domain semantics** - the real-world concepts of organizational roles, resource ownership, and access policies must be faithfully represented in data structures. Second, it **enables efficient operations** - permission lookups, policy evaluations, and role inheritance calculations must be fast enough for real-time authorization decisions. Third, it **enforces security invariants** - the data structures themselves should make it difficult or impossible to represent invalid states like circular role hierarchies or cross-tenant data leakage.

Our authorization system must support three distinct but interconnected access control paradigms: Role-Based Access Control (RBAC) for organizational permissions, Attribute-Based Access Control (ABAC) for dynamic policy evaluation, and resource-based access for fine-grained ownership controls. Each paradigm requires its own specialized data structures, but they must integrate seamlessly to provide a unified authorization decision.

![Data Model Entity Relationships](./diagrams/data-model-relationships.svg)

The complexity challenge in authorization data modeling is **relationship management**. Users can have multiple roles across different tenants, roles can inherit permissions from parent roles, policies can reference overlapping sets of resources, and resources can exist in hierarchical ownership structures. These many-to-many relationships create a web of dependencies that must be carefully designed to avoid performance bottlenecks, data inconsistencies, and security vulnerabilities.

### RBAC Data Structures

Role-Based Access Control forms the backbone of organizational permission management. Think of RBAC as a **corporate org chart with permission inheritance**: just as an employee inherits responsibilities and authorities from their position in the organizational hierarchy, users inherit permissions through their assigned roles, and roles inherit permissions from parent roles in the role hierarchy.

The RBAC data model centers on three primary entities: users (who need access), roles (organizational positions with defined responsibilities), and permissions (specific actions on specific resources). The power of RBAC comes from the **indirection layer** that roles provide - instead of assigning permissions directly to users, we assign users to roles and roles to permissions. This indirection enables centralized permission management, consistent access patterns, and easier auditing of who can do what.

**User Entity Structure**

Users in our system are lightweight identity references rather than full user profiles. The authorization system doesn't manage user authentication or personal information - it only cares about the user's identity for permission lookups and audit trails.

| Field | Type | Description |
|-------|------|-------------|
| UserID | string | Unique identifier for the user, typically from external identity provider |
| TenantMemberships | []TenantMembership | List of tenants this user belongs to with role assignments |
| GlobalRoles | []string | System-wide role IDs that apply across all tenants (admin, support) |
| CreatedAt | time.Time | When this user record was first created |
| LastActiveAt | time.Time | Timestamp of most recent authorization request for cache management |

The `TenantMembership` structure captures the many-to-many relationship between users and tenants, along with the roles assigned within each tenant:

| Field | Type | Description |
|-------|------|-------------|
| TenantID | string | Unique identifier for the tenant organization |
| RoleIDs | []string | List of role IDs assigned to this user within this tenant |
| JoinedAt | time.Time | When the user was added to this tenant |
| Status | MembershipStatus | Active, suspended, or pending invitation acceptance |

> **Decision: User-Tenant-Role Relationship Design**
> - **Context**: Users can belong to multiple tenants and have different roles in each tenant, creating a three-way relationship between users, tenants, and roles
> - **Options Considered**: 
>   1. Flat user-role assignments with tenant context in each assignment
>   2. Hierarchical user → tenant memberships → role assignments within tenant
>   3. Role assignments as independent entities referencing user, tenant, and role
> - **Decision**: Hierarchical user → tenant memberships → role assignments (option 2)
> - **Rationale**: This structure naturally models real-world organizational relationships where users join organizations and then receive roles within those organizations. It makes tenant-scoped queries efficient and provides clear audit trails for membership changes.
> - **Consequences**: Enables efficient "what roles does user X have in tenant Y" queries, simplifies tenant membership management, but requires more complex queries for "all users with role Z across all tenants"

**Role Entity Structure**

Roles represent organizational positions or job functions with associated permissions. The `Role` structure supports hierarchical inheritance through parent role references, enabling permission delegation and organizational modeling.

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier for this role |
| Name | string | Human-readable role name (e.g., "Engineering Manager") |
| TenantID | string | Tenant this role belongs to, empty for global system roles |
| ParentRoles | []string | Role IDs this role inherits permissions from |
| Permissions | []Permission | Direct permissions granted to this role |
| CreatedAt | time.Time | Role creation timestamp |
| UpdatedAt | time.Time | Last modification timestamp for cache invalidation |
| Description | string | Human-readable description of role responsibilities |
| MaxInheritanceDepth | int | Maximum depth for inheritance to prevent deep recursion |

The role hierarchy creates a **directed acyclic graph (DAG)** where edges represent inheritance relationships. Each role can inherit from multiple parent roles (multiple inheritance) and can be inherited by multiple child roles. The DAG property ensures no circular dependencies that could create infinite loops during permission resolution.

![Role Hierarchy Structure](./diagrams/role-hierarchy-example.svg)

**Permission Entity Structure**

Permissions represent specific actions that can be performed on specific resource types. The granular permission model enables precise access control while maintaining simplicity for common use cases.

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier for this permission |
| Resource | string | Resource type or specific resource this permission applies to |
| Action | string | Action that can be performed (read, write, delete, etc.) |
| Scope | PermissionScope | Whether permission applies to owned resources, tenant resources, or all resources |
| Conditions | []string | Optional conditions that must be met for permission to apply |

The `PermissionScope` enumeration defines the breadth of the permission:

| Scope Value | Description | Example Use Case |
|-------------|-------------|------------------|
| `ScopeOwned` | Permission applies only to resources owned by the user | User can edit their own documents |
| `ScopeTenant` | Permission applies to all resources within the user's tenant | Manager can view all team documents |
| `ScopeGlobal` | Permission applies across all tenants | System admin can access any resource |

**User Role Assignment Structure**

The `UserRoleAssignment` entity captures the many-to-many relationship between users and roles with additional metadata for audit trails and temporal access control.

| Field | Type | Description |
|-------|------|-------------|
| UserID | string | User receiving the role assignment |
| RoleID | string | Role being assigned to the user |
| TenantID | string | Tenant context for this assignment |
| ValidFrom | *time.Time | Optional start time for role validity |
| ValidTo | *time.Time | Optional expiration time for role assignment |
| AssignedBy | string | User ID of who made this assignment |
| AssignedAt | time.Time | Timestamp when assignment was created |
| Reason | string | Human-readable reason for the assignment |

> The critical insight here is that role assignments are **first-class entities** rather than simple many-to-many join records. This enables rich audit trails, temporal access control, and delegation tracking that are essential for compliance and security forensics.

**Permission Inheritance Algorithm**

Role hierarchies require a **transitive closure computation** to determine all permissions available to a user. The algorithm must traverse the role DAG, collect permissions from all reachable roles, and handle permission conflicts or duplicates.

The permission resolution process follows these steps:

1. Start with the user's directly assigned roles within the relevant tenant
2. For each assigned role, initiate a depth-first traversal of parent roles
3. At each role node, collect the role's direct permissions into the result set
4. Follow parent role references, maintaining a visited set to detect cycles
5. Continue traversal until all reachable roles have been processed
6. Merge permission collections, handling duplicates and conflicts
7. Apply permission scope filtering based on the authorization request context
8. Cache the computed permission set with appropriate invalidation triggers

The traversal must include **cycle detection** to handle malformed role hierarchies gracefully. If a cycle is detected during traversal, the algorithm should log the error, break the cycle at the detection point, and continue processing to avoid system failure.

**Role Template System**

Role templates provide **preconfigured role bundles** for common organizational patterns. Templates define standard permission sets that can be instantiated across different tenants with consistent naming and permission grants.

| Field | Type | Description |
|-------|------|-------------|
| TemplateID | string | Unique identifier for this role template |
| Name | string | Template name (e.g., "Standard Manager") |
| Description | string | Description of template purpose and permissions |
| BasePermissions | []Permission | Standard permissions included in this template |
| RequiredParentRoles | []string | Role templates this template depends on |
| ConfigurablePermissions | []Permission | Additional permissions that can be optionally included |
| Industry | string | Industry or domain this template is designed for |

Templates enable **consistent role provisioning** across tenants while allowing customization for specific organizational needs. When instantiating a template, the system creates a new role with the template's base permissions and allows administrators to add or remove configurable permissions based on their specific requirements.

### ABAC Policy Structures

Attribute-Based Access Control extends beyond the static role-permission model to enable **dynamic access decisions** based on runtime attributes of users, resources, and environmental context. Think of ABAC as a **smart security system** that makes access decisions not just based on who you are (identity) and what role you have (RBAC), but also considering contextual factors like time of day, location, resource sensitivity, and current system state.

ABAC policies are **rule-based expressions** that evaluate attributes from multiple sources to make allow/deny decisions. Unlike RBAC's predetermined permission sets, ABAC evaluates conditions at runtime, enabling fine-grained access control that adapts to changing circumstances. The trade-off is increased complexity in policy authoring and evaluation performance, but the benefit is unprecedented flexibility in access control logic.

**Policy Entity Structure**

The `Policy` structure represents a complete access control rule with conditions, effects, and metadata for management and audit.

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier for this policy |
| Name | string | Human-readable policy name for management interfaces |
| TenantID | string | Tenant this policy belongs to, empty for global policies |
| Effect | PolicyEffect | Whether this policy allows or denies access |
| Conditions | *Condition | Root condition tree that must evaluate to true |
| Resources | []string | Resource patterns this policy applies to |
| Actions | []string | Actions this policy governs |
| Priority | int | Policy priority for conflict resolution (higher wins) |
| CreatedBy | string | User who authored this policy |
| CreatedAt | time.Time | Policy creation timestamp |
| UpdatedAt | time.Time | Last modification timestamp |
| Version | int | Policy version number for change tracking |
| IsActive | bool | Whether policy is currently enforced |

The `PolicyEffect` enumeration defines the policy's access decision:

| Effect Value | Description | Behavior |
|--------------|-------------|----------|
| `PolicyEffectAllow` | Policy grants access | If conditions match, allow the request |
| `PolicyEffectDeny` | Policy denies access | If conditions match, deny the request |

> **Decision: Policy Effect Design**
> - **Context**: Policies need to specify whether they grant or deny access, and the system needs clear rules for combining multiple matching policies
> - **Options Considered**:
>   1. Allow-only policies with implicit deny
>   2. Deny-only policies with implicit allow  
>   3. Explicit allow/deny policies with combining rules
> - **Decision**: Explicit allow/deny policies with deny-overrides combining (option 3)
> - **Rationale**: Explicit effects make policy intent clear and auditable. Deny-overrides combining follows the security principle of "fail secure" - any explicit denial blocks access regardless of allowing policies.
> - **Consequences**: Enables complex policy scenarios with both positive and negative authorization rules, but requires careful policy authoring to avoid unintended denials

**Condition Structure for Complex Logic**

Conditions form the heart of ABAC policy evaluation, representing **logical expressions** that combine attribute comparisons with boolean operators. The `Condition` structure supports nested boolean logic through recursive composition.

| Field | Type | Description |
|-------|------|-------------|
| And | []*Condition | Array of conditions that must all be true |
| Or | []*Condition | Array of conditions where at least one must be true |
| Not | *Condition | Single condition whose result should be inverted |
| Attribute | string | Dot-notation path to attribute for comparison |
| Operator | string | Comparison operator (eq, ne, gt, lt, contains, etc.) |
| Value | interface{} | Expected value for attribute comparison |

This recursive structure enables arbitrarily complex logical expressions. For example, a condition like "user is manager AND (department is engineering OR user has security clearance)" becomes:

```
{
  And: [
    {Attribute: "user.role", Operator: "eq", Value: "manager"},
    {
      Or: [
        {Attribute: "user.department", Operator: "eq", Value: "engineering"},
        {Attribute: "user.clearance", Operator: "exists", Value: nil}
      ]
    }
  ]
}
```

**Supported Condition Operators**

The policy evaluation engine supports a comprehensive set of comparison operators for different data types and use cases:

| Operator | Data Types | Description | Example |
|----------|------------|-------------|---------|
| `eq` | all | Equal comparison | `user.role eq "manager"` |
| `ne` | all | Not equal comparison | `resource.status ne "archived"` |
| `gt`, `gte`, `lt`, `lte` | numeric, time | Numeric/temporal comparison | `user.age gt 18` |
| `contains` | string, array | Contains substring or element | `user.groups contains "admin"` |
| `startsWith`, `endsWith` | string | String prefix/suffix matching | `resource.name startsWith "temp_"` |
| `matches` | string | Regular expression matching | `user.email matches ".*@company.com"` |
| `exists` | all | Check if attribute is present | `resource.sensitivity exists` |
| `in` | all | Value in array | `user.department in ["eng", "product"]` |
| `between` | numeric, time | Range inclusion | `request.time between ["09:00", "17:00"]` |

**Attribute Resolution Context**

Policy evaluation requires **runtime context** containing all attributes referenced by policy conditions. The evaluation context aggregates attributes from multiple sources: user profile, resource metadata, environmental factors, and request parameters.

| Context Source | Attribute Examples | Description |
|----------------|-------------------|-------------|
| User Attributes | `user.id`, `user.role`, `user.department`, `user.clearance_level` | Static and dynamic user properties |
| Resource Attributes | `resource.type`, `resource.owner`, `resource.sensitivity`, `resource.created_at` | Resource metadata and properties |
| Environment Attributes | `env.time`, `env.day_of_week`, `env.client_ip`, `env.user_agent` | Contextual request environment |
| Request Attributes | `request.action`, `request.urgency`, `request.reason` | Specific request parameters |
| Computed Attributes | `user.is_resource_owner`, `resource.age_days` | Dynamically calculated values |

The attribute resolution process follows a **hierarchical lookup strategy**:

1. Check request-specific attributes passed with the authorization request
2. Load user attributes from user profile service or cache
3. Load resource attributes from resource metadata store
4. Compute environmental attributes from request context (IP, time, etc.)
5. Calculate derived attributes based on relationships (ownership, etc.)
6. Cache resolved attributes for subsequent policy evaluations in the same request

**Policy Decision Structure**

The `PolicyDecision` structure captures the result of ABAC policy evaluation with detailed information for audit trails and debugging.

| Field | Type | Description |
|-------|------|-------------|
| Allowed | bool | Final access decision from policy evaluation |
| AppliedPolicies | []string | Policy IDs that matched and influenced the decision |
| DenyingPolicy | string | Specific policy ID that caused denial (if applicable) |
| EvaluationTrace | []EvaluationStep | Detailed trace of policy evaluation steps |
| MissingAttributes | []string | Attributes referenced but not available |
| EvaluationDuration | time.Duration | Time taken for policy evaluation |

The `EvaluationStep` structure provides detailed traceability for policy decisions:

| Field | Type | Description |
|-------|------|-------------|
| PolicyID | string | Policy being evaluated |
| ConditionPath | string | Path to specific condition within policy |
| AttributeValue | interface{} | Actual attribute value used in comparison |
| ExpectedValue | interface{} | Expected value from policy condition |
| Result | bool | Result of this condition evaluation |
| Operator | string | Comparison operator applied |

> The evaluation trace is crucial for **policy debugging and compliance auditing**. When access is unexpectedly denied or granted, the trace shows exactly which policies matched, which conditions failed, and what attribute values were used in the decision.

### Resource and Tenant Model

The resource and tenant model provides the foundation for **multi-tenant SaaS authorization** with fine-grained resource-level permissions and strict tenant isolation. Think of this model as a **digital apartment building management system**: tenants (organizations) have their own separate spaces, resources (documents, projects, etc.) have clear ownership within those spaces, and the building manager (platform admin) can access common areas while respecting tenant privacy boundaries.

Multi-tenancy in authorization systems creates unique challenges around **data isolation**, **resource sharing**, and **administrative boundaries**. Unlike single-tenant applications where all users implicitly share the same data space, multi-tenant systems must ensure that users from different organizations cannot access each other's resources, while still enabling controlled sharing and platform-level administration.

**Resource Entity Structure**

Resources represent any entity that requires access control - documents, projects, databases, API endpoints, or any other system component. The `Resource` structure provides a universal foundation for permission management across different resource types.

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier for this resource |
| Type | string | Resource type category (document, project, api_endpoint, etc.) |
| TenantID | string | Tenant that owns this resource |
| OwnerID | string | User who created or owns this resource |
| Attributes | map[string]any | Flexible attribute storage for resource metadata |
| ParentResourceID | string | Parent resource for hierarchical resources |
| CreatedAt | time.Time | Resource creation timestamp |
| UpdatedAt | time.Time | Last modification timestamp |
| IsShared | bool | Whether resource has cross-tenant sharing enabled |
| ShareSettings | *ShareSettings | Configuration for resource sharing |

The `Attributes` map provides **flexible metadata storage** without requiring schema changes for new resource types. Common attributes include sensitivity level, department, project phase, compliance requirements, and business context information.

**Hierarchical Resource Organization**

Resources can form **parent-child hierarchies** that enable permission inheritance and bulk operations. For example, a project resource might contain document resources, and granting project access automatically includes access to contained documents.

The hierarchical model follows these principles:

1. **Permission inheritance flows downward**: access to a parent resource grants access to child resources unless explicitly overridden
2. **Ownership inheritance is optional**: child resources can have different owners than their parents
3. **Tenant boundaries are strictly enforced**: hierarchies cannot cross tenant boundaries
4. **Cycle prevention**: the system prevents resource hierarchy cycles that could create infinite recursion

Resource hierarchy traversal uses **recursive permission checking** with memoization to avoid redundant database queries. When evaluating access to a resource, the system checks permissions on the resource itself, then walks up the parent chain checking for inherited permissions until it reaches the root or finds a permission match.

**Tenant Isolation Model**

Tenant isolation ensures **complete data separation** between different organizations using the platform. The tenant model provides both logical separation (data belongs to specific tenants) and administrative boundaries (tenant admins can only manage their own tenant's resources).

| Field | Type | Description |
|-------|------|-------------|
| TenantID | string | Unique identifier for this tenant |
| Name | string | Human-readable tenant name |
| Status | TenantStatus | Active, suspended, or deactivated |
| CreatedAt | time.Time | Tenant creation timestamp |
| Settings | TenantSettings | Tenant-specific configuration |
| AdminUsers | []string | User IDs with administrative access to this tenant |
| ResourceQuotas | ResourceQuotas | Limits on resource creation and usage |

The `TenantSettings` structure captures tenant-specific configuration:

| Field | Type | Description |
|-------|------|-------------|
| AllowCrossTenantSharing | bool | Whether users can share resources outside tenant |
| RequireOwnershipTransfer | bool | Whether resources must be explicitly transferred between users |
| DefaultResourceVisibility | string | Default visibility for new resources (private, team, tenant) |
| DataRetentionPeriod | time.Duration | How long to retain deleted resource data |
| AuditLogRetention | time.Duration | How long to retain audit logs |
| ExternalSharingDomains | []string | External domains allowed for resource sharing |

> **Decision: Tenant Isolation Strategy**
> - **Context**: Multi-tenant systems need complete data separation while enabling platform administration and optional cross-tenant sharing
> - **Options Considered**:
>   1. Database-level isolation with separate schemas per tenant
>   2. Application-level isolation with tenant ID filtering
>   3. Hybrid approach with tenant partitioning and row-level security
> - **Decision**: Application-level isolation with tenant ID filtering (option 2)
> - **Rationale**: Provides complete control over isolation logic, enables cross-tenant features when needed, and scales better than separate databases. Row-level security adds defense in depth at the database layer.
> - **Consequences**: Requires careful implementation to avoid tenant data leakage, but enables flexible sharing models and efficient resource utilization

**Resource Ownership and Sharing**

Resource ownership establishes **primary access rights** and delegation authority. The owner has full control over the resource and can grant access to other users or tenants through sharing mechanisms.

The `ShareSettings` structure defines how resources can be shared beyond their owning tenant:

| Field | Type | Description |
|-------|------|-------------|
| SharedWith | []ShareGrant | List of users or tenants with granted access |
| ShareLink | *ShareLink | Public or private share link configuration |
| ExpiresAt | *time.Time | Optional expiration time for all shares |
| RequiresApproval | bool | Whether shares need owner approval |
| MaxShares | int | Maximum number of concurrent shares allowed |
| AllowedDomains | []string | Email domains allowed to access shared resource |

The `ShareGrant` structure captures individual sharing grants:

| Field | Type | Description |
|-------|------|-------------|
| GranteeType | string | Type of grantee (user, tenant, group, public) |
| GranteeID | string | Identifier for the grantee |
| Permissions | []string | Specific permissions granted (read, write, admin) |
| GrantedBy | string | User who created this sharing grant |
| GrantedAt | time.Time | When the share was created |
| ExpiresAt | *time.Time | Optional expiration for this specific grant |
| LastAccessed | *time.Time | When the shared resource was last accessed by grantee |

**Cross-Tenant Access Control**

Cross-tenant sharing enables **controlled collaboration** between organizations while maintaining tenant isolation. The system supports several sharing models:

1. **User-to-user sharing**: Individual users can share specific resources with users from other tenants
2. **Tenant-to-tenant sharing**: Organizational agreements that enable broader sharing between specific tenants
3. **Public sharing**: Resources can be made publicly accessible with optional access controls
4. **Domain-based sharing**: Resources can be shared with all users from specific email domains

Cross-tenant access requires **explicit authorization chains** that trace from the requesting user to the resource owner. The authorization engine must verify:

1. The resource allows cross-tenant sharing
2. The sharing grant exists and hasn't expired  
3. The requesting user matches the grant criteria
4. The requested action is included in the granted permissions
5. Both tenant policies allow the cross-tenant access

**Resource Quota Management**

Resource quotas prevent **tenant resource exhaustion** and enable fair platform usage. The `ResourceQuotas` structure defines limits on resource creation and usage:

| Field | Type | Description |
|-------|------|-------------|
| MaxResources | map[string]int | Maximum number of resources by type |
| MaxStorageSize | int64 | Maximum storage usage in bytes |
| MaxSharesPerResource | int | Limit on sharing grants per resource |
| MaxCrossTenantShares | int | Limit on cross-tenant sharing grants |
| MaxAPIRequestsPerDay | int | Rate limiting for API operations |
| AllowedResourceTypes | []string | Resource types this tenant can create |

Quota enforcement occurs at **resource creation time** and during **sharing operations**. The system tracks current usage against quotas and rejects operations that would exceed limits. Quota tracking requires eventually consistent counters that can handle high-volume resource operations without blocking.

### Common Pitfalls

⚠️ **Pitfall: Circular Role Inheritance**
Role hierarchies must form a directed acyclic graph, but it's easy to accidentally create cycles when managing role relationships. For example, if "Manager" inherits from "Employee" and "Employee" is later set to inherit from "Manager", the system will enter infinite loops during permission resolution. Always validate the DAG property when adding parent role relationships by performing cycle detection through depth-first traversal.

⚠️ **Pitfall: Tenant ID Injection in Queries** 
Every database query that retrieves resources must include tenant ID filtering to prevent cross-tenant data leakage. A common mistake is writing queries like `SELECT * FROM resources WHERE id = ?` instead of `SELECT * FROM resources WHERE id = ? AND tenant_id = ?`. This can expose resources from other tenants if resource IDs are predictable or enumerable. Always include tenant context in data access patterns.

⚠️ **Pitfall: Permission Scope Confusion**
The permission scope (owned, tenant, global) determines which resources a permission applies to, but it's easy to misunderstand the semantics. A permission with "tenant" scope doesn't automatically grant access to all tenant resources - it means the permission can apply to any resource within the tenant if other conditions (ownership, sharing) are met. Document permission scope semantics clearly and test edge cases thoroughly.

⚠️ **Pitfall: Missing Attribute Validation**
ABAC policies can reference attributes that don't exist in the evaluation context, leading to runtime policy evaluation failures. For example, a policy checking `user.security_clearance` will fail if the user profile doesn't include that attribute. Implement attribute schema validation and provide clear error messages for missing required attributes during policy evaluation.

⚠️ **Pitfall: Cache Invalidation Complexity**
Role and permission caches must be invalidated when roles, role assignments, or role hierarchies change, but it's complex to track all dependencies. Changing a parent role's permissions affects all users with child roles, potentially requiring cache invalidation for thousands of users. Design cache invalidation strategies that can handle hierarchical dependencies without cache stampedes.

⚠️ **Pitfall: Resource Hierarchy Depth Explosion**
Deeply nested resource hierarchies can cause performance problems during permission resolution. For example, checking access to a deeply nested document might require traversing dozens of parent resources. Implement maximum hierarchy depth limits and consider flattening deep hierarchies or caching permission inheritance at multiple levels.

### Implementation Guidance

The data model implementation requires careful attention to **concurrent access patterns**, **cache coherence**, and **data integrity constraints**. Authorization systems face high read loads with occasional write bursts, making cache-friendly data structures essential for performance.

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Primary Storage | PostgreSQL with JSONB for flexible attributes | Distributed database like CockroachDB for global scale |
| Cache Layer | Redis with pub/sub for cache invalidation | Redis Cluster with consistent hashing |
| Search Index | PostgreSQL GIN indexes on JSONB attributes | Elasticsearch for complex attribute queries |
| Graph Processing | Recursive CTEs for role hierarchy traversal | Neo4j for complex relationship queries |
| Time-Series Data | PostgreSQL with time partitioning for audit logs | TimescaleDB for high-volume audit storage |

**File Structure Organization**

The data model implementation should be organized into focused packages that separate concerns and enable independent testing:

```
internal/
  models/           ← core data structures
    user.go         ← User, TenantMembership, UserRoleAssignment
    role.go         ← Role, Permission, role hierarchy logic  
    policy.go       ← Policy, Condition, ABAC structures
    resource.go     ← Resource, ShareSettings, tenant isolation
    tenant.go       ← Tenant, ResourceQuotas, cross-tenant logic
  storage/          ← data persistence layer
    user_store.go   ← user data operations
    role_store.go   ← role and permission persistence
    policy_store.go ← policy storage and indexing
    resource_store.go ← resource and sharing data
  cache/            ← caching layer
    permission_cache.go ← role-based permission caching
    policy_cache.go     ← policy evaluation result caching
    user_cache.go       ← user profile and role assignment caching
```

**Core Data Structure Definitions**

```go
package models

import (
    "time"
)

// User represents an identity that can be granted access through roles and policies
type User struct {
    UserID           string             `json:"user_id" db:"user_id"`
    TenantMemberships []TenantMembership `json:"tenant_memberships"`
    GlobalRoles      []string           `json:"global_roles" db:"global_roles"`
    CreatedAt        time.Time          `json:"created_at" db:"created_at"`
    LastActiveAt     time.Time          `json:"last_active_at" db:"last_active_at"`
}

// TenantMembership represents user membership in a tenant organization
type TenantMembership struct {
    TenantID string               `json:"tenant_id" db:"tenant_id"`
    RoleIDs  []string             `json:"role_ids" db:"role_ids"`
    JoinedAt time.Time            `json:"joined_at" db:"joined_at"`
    Status   MembershipStatus     `json:"status" db:"status"`
}

// Role represents an organizational position with associated permissions
type Role struct {
    ID          string       `json:"id" db:"id"`
    Name        string       `json:"name" db:"name"`
    TenantID    string       `json:"tenant_id" db:"tenant_id"`
    ParentRoles []string     `json:"parent_roles" db:"parent_roles"`
    Permissions []Permission `json:"permissions"`
    CreatedAt   time.Time    `json:"created_at" db:"created_at"`
    UpdatedAt   time.Time    `json:"updated_at" db:"updated_at"`
    Description string       `json:"description" db:"description"`
}

// Permission represents a specific action allowed on a resource type
type Permission struct {
    ID       string          `json:"id" db:"id"`
    Resource string          `json:"resource" db:"resource"`
    Action   string          `json:"action" db:"action"`
    Scope    PermissionScope `json:"scope" db:"scope"`
}

// UserRoleAssignment captures the many-to-many relationship between users and roles
type UserRoleAssignment struct {
    UserID     string     `json:"user_id" db:"user_id"`
    RoleID     string     `json:"role_id" db:"role_id"`
    TenantID   string     `json:"tenant_id" db:"tenant_id"`
    ValidFrom  *time.Time `json:"valid_from" db:"valid_from"`
    ValidTo    *time.Time `json:"valid_to" db:"valid_to"`
    AssignedBy string     `json:"assigned_by" db:"assigned_by"`
    AssignedAt time.Time  `json:"assigned_at" db:"assigned_at"`
}

// Policy represents an ABAC rule with conditions and effects
type Policy struct {
    ID           string        `json:"id" db:"id"`
    Name         string        `json:"name" db:"name"`
    TenantID     string        `json:"tenant_id" db:"tenant_id"`
    Effect       PolicyEffect  `json:"effect" db:"effect"`
    Conditions   *Condition    `json:"conditions"`
    Resources    []string      `json:"resources" db:"resources"`
    Actions      []string      `json:"actions" db:"actions"`
    Priority     int           `json:"priority" db:"priority"`
    CreatedBy    string        `json:"created_by" db:"created_by"`
    CreatedAt    time.Time     `json:"created_at" db:"created_at"`
    UpdatedAt    time.Time     `json:"updated_at" db:"updated_at"`
}

// Condition represents a logical expression for policy evaluation
type Condition struct {
    And       []*Condition `json:"and,omitempty"`
    Or        []*Condition `json:"or,omitempty"`
    Not       *Condition   `json:"not,omitempty"`
    Attribute string       `json:"attribute,omitempty"`
    Operator  string       `json:"operator,omitempty"`
    Value     interface{}  `json:"value,omitempty"`
}

// Resource represents any entity that requires access control
type Resource struct {
    ID               string            `json:"id" db:"id"`
    Type             string            `json:"type" db:"type"`
    TenantID         string            `json:"tenant_id" db:"tenant_id"`
    OwnerID          string            `json:"owner_id" db:"owner_id"`
    Attributes       map[string]any    `json:"attributes" db:"attributes"`
    ParentResourceID string            `json:"parent_resource_id" db:"parent_resource_id"`
    CreatedAt        time.Time         `json:"created_at" db:"created_at"`
    UpdatedAt        time.Time         `json:"updated_at" db:"updated_at"`
    IsShared         bool              `json:"is_shared" db:"is_shared"`
    ShareSettings    *ShareSettings    `json:"share_settings"`
}
```

**Role Hierarchy Implementation**

```go
// RoleHierarchyManager handles role inheritance and permission resolution
type RoleHierarchyManager struct {
    roleStore RoleStore
    cache     PermissionCache
}

// GetUserPermissions resolves all permissions for a user including inherited permissions
func (r *RoleHierarchyManager) GetUserPermissions(ctx context.Context, userID, tenantID string) ([]Permission, error) {
    // TODO 1: Look up user's directly assigned roles within the tenant
    // TODO 2: For each role, traverse the parent role hierarchy using DFS
    // TODO 3: Collect permissions from each role in the traversal path
    // TODO 4: Detect cycles using visited set and break gracefully
    // TODO 5: Deduplicate permissions and resolve scope conflicts
    // TODO 6: Cache the computed permission set with appropriate TTL
    // Hint: Use a map[string]bool as visited set for cycle detection
    return nil, nil
}

// ValidateRoleHierarchy ensures the role hierarchy is a valid DAG
func (r *RoleHierarchyManager) ValidateRoleHierarchy(ctx context.Context, tenantID string) error {
    // TODO 1: Load all roles for the tenant
    // TODO 2: Build adjacency list representation of role relationships  
    // TODO 3: Perform topological sort to detect cycles
    // TODO 4: Return error with specific cycle information if found
    // TODO 5: Validate maximum inheritance depth limits
    return nil
}
```

**Policy Evaluation Engine**

```go
// PolicyEvaluator handles ABAC policy evaluation with attribute resolution
type PolicyEvaluator struct {
    policyStore PolicyStore
    attrResolver AttributeResolver
    cache       PolicyCache
}

// Evaluate processes all matching policies and returns a combined decision
func (p *PolicyEvaluator) Evaluate(ctx context.Context, req *AuthzRequest) (*PolicyDecision, error) {
    // TODO 1: Find all policies matching the resource and action patterns
    // TODO 2: Filter policies by tenant and global scopes
    // TODO 3: Resolve all attributes referenced by matching policies
    // TODO 4: Evaluate each policy's condition tree against the attribute context
    // TODO 5: Apply policy combining logic with deny-overrides semantics
    // TODO 6: Build evaluation trace for audit and debugging
    // TODO 7: Cache policy decisions based on stable attribute combinations
    return nil, nil
}

// EvaluateCondition recursively evaluates a condition tree
func (p *PolicyEvaluator) EvaluateCondition(condition *Condition, context map[string]interface{}) (bool, error) {
    // TODO 1: Handle logical operators (And, Or, Not) with recursive calls
    // TODO 2: For leaf conditions, resolve attribute value from context
    // TODO 3: Apply comparison operator to attribute and expected values
    // TODO 4: Handle type coercion for numeric and temporal comparisons
    // TODO 5: Return detailed error for missing attributes or invalid operators
    return false, nil
}
```

**Milestone Checkpoints**

**Milestone 1 Verification (Role & Permission Model)**:
- Run `go test ./internal/models/...` - all tests should pass
- Create a test role hierarchy and verify permission inheritance works correctly
- Test role assignment and verify users receive inherited permissions
- Validate that circular role references are detected and rejected

**Milestone 2 Verification (ABAC Policy Engine)**:
- Test policy evaluation with various attribute combinations
- Verify policy combining logic handles conflicting allow/deny policies correctly  
- Test condition evaluation with nested boolean logic (And/Or/Not operators)
- Validate that missing attributes cause policy evaluation failures with clear error messages

**Milestone 3 Verification (Resource-Based & Multi-tenancy)**:
- Test tenant isolation by attempting cross-tenant resource access (should be denied)
- Verify resource ownership grants and sharing mechanisms work correctly
- Test hierarchical resource access with parent-child permission inheritance
- Validate resource quota enforcement prevents over-allocation

**Language-Specific Implementation Notes**

- Use `database/sql` with PostgreSQL driver for primary data storage
- Implement JSONB storage for flexible resource attributes using `lib/pq` JSONB support
- Use `sync.RWMutex` for concurrent access to in-memory caches
- Implement proper connection pooling with `sql.DB` configured for high concurrency
- Use `context.Context` throughout for timeout and cancellation propagation
- Consider using `golang-migrate` for database schema versioning

**Performance Optimization Strategies**

- Cache computed role permissions with Redis and invalidate on role changes
- Use database indexes on tenant_id, user_id, and resource_id fields for efficient queries
- Implement read replicas for policy evaluation queries to separate from write load
- Use prepared statements for frequently executed authorization queries
- Consider implementing permission bitmaps for very high-performance role checking


## Role and Permission Engine

> **Milestone(s):** Milestone 1 (Role & Permission Model) - this section implements hierarchical roles with permission inheritance and efficient lookup mechanisms

The role and permission engine forms the foundation of our RBAC system, providing the core mechanism for granting access based on organizational roles. Think of this as the digital equivalent of an organizational chart where authority and responsibilities flow downward - just as a Vice President inherits all the authorities of a Manager plus additional ones, our role hierarchy allows permissions to flow from parent roles to child roles automatically.

This engine must solve several challenging problems: efficiently storing and querying hierarchical role relationships, preventing circular dependencies that could create infinite permission loops, optimizing permission lookups for high-performance authorization decisions, and providing flexible role templates that capture common access patterns across different organizations.

### Role Hierarchy Mental Model: Understanding role inheritance as organizational chart with permission flow

Before diving into the technical implementation, it's essential to understand role hierarchy through a familiar mental model. **Role inheritance** works exactly like authority delegation in a traditional organization chart. When the CEO delegates authority to a Vice President, that VP automatically gains certain powers. When the VP delegates to a Director, the Director inherits a subset of the VP's authority plus any additional specific responsibilities.

In our authorization system, this translates to **permission inheritance** through a parent-child role relationship. A `Role` with parent roles automatically inherits all permissions from those parents, creating a transitive permission set that combines inherited permissions with directly assigned ones. This inheritance model provides several key benefits: it reduces administrative overhead by avoiding duplicate permission assignments, ensures consistency as parent role changes automatically propagate to children, and models real-world organizational authority structures naturally.

The critical insight is that role hierarchy creates a **directed acyclic graph (DAG)** where permissions flow from parent nodes to child nodes. Unlike a tree structure, a DAG allows multiple inheritance - a role can inherit from multiple parent roles, combining their permission sets. However, the "acyclic" property is crucial: we must prevent circular dependencies where Role A inherits from Role B, which inherits from Role C, which inherits back to Role A. Such cycles would create infinite loops during permission resolution.

Consider a typical SaaS application's role hierarchy. At the top sits an `OrgAdmin` role with broad permissions across all organizational resources. Below that, specialized roles like `ProjectManager`, `Developer`, and `Viewer` inherit basic organizational permissions while adding their own specific capabilities. A `SeniorDeveloper` role might inherit from both `Developer` (for code access) and `ProjectManager` (for planning access), combining permissions from both parent roles.

This inheritance model also supports **role templates** - preconfigured role bundles that organizations can customize for their specific needs. A "Standard Development Team" template might include `TeamLead`, `SeniorDev`, `JuniorDev`, and `QA` roles with predefined relationships and permission sets. Organizations can instantiate these templates and modify them as needed, providing both convenience and consistency.

### Hierarchical Role Design: DAG-based role inheritance with cycle detection and transitive closure

The core challenge in implementing role hierarchy lies in designing a data structure that efficiently represents parent-child relationships while preventing cycles and enabling fast permission lookups. Our `Role` structure captures both direct role properties and hierarchical relationships through its `ParentRoles` field, which contains a list of parent role identifiers.

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier for the role within its tenant scope |
| Name | string | Human-readable role name for administrative interfaces |
| TenantID | string | Tenant scope limiting role visibility and inheritance |
| ParentRoles | []string | List of parent role IDs from which this role inherits permissions |
| Permissions | []Permission | Direct permissions assigned to this role (not inherited) |
| CreatedAt | time.Time | Timestamp for audit and lifecycle management |
| UpdatedAt | time.Time | Last modification timestamp for change tracking |

The `Permission` structure defines atomic access rights that roles can possess or inherit. Each permission represents a specific action on a specific resource type, following the principle of least privilege by being as granular as necessary.

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier for the permission |
| Resource | string | Resource type or pattern this permission applies to |
| Action | string | Specific action allowed (read, write, delete, admin, etc.) |

To maintain referential integrity and prevent cycles, the role engine implements **DAG validation** every time role relationships change. The cycle detection algorithm uses depth-first search with a coloring scheme to identify back edges that would create cycles.

> **Decision: DAG-Based Role Hierarchy with Cycle Detection**
> - **Context**: Role inheritance requires a graph structure that allows multiple inheritance while preventing infinite loops during permission resolution
> - **Options Considered**: 
>   1. Tree-based hierarchy (single inheritance only)
>   2. DAG with runtime cycle detection
>   3. Matrix-based role relationships
> - **Decision**: DAG with proactive cycle detection on role modification
> - **Rationale**: DAG supports realistic multiple inheritance patterns (e.g., roles that combine permissions from multiple parent roles), while proactive cycle detection prevents corrupt states that could cause runtime failures. Tree structure is too restrictive for real-world scenarios, and matrix representation doesn't scale efficiently.
> - **Consequences**: Enables flexible role modeling at the cost of complexity in role management operations. Requires sophisticated validation logic but prevents runtime failures during permission resolution.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Tree-based hierarchy | Simple, no cycles possible, fast traversal | Only single inheritance, too restrictive for real roles | No |
| DAG with cycle detection | Multiple inheritance, models real organizations, prevents invalid states | Complex validation, potential performance overhead | **Yes** |
| Matrix-based relationships | Very flexible, supports any relationship pattern | Doesn't scale, hard to reason about, no inherent cycle prevention | No |

The cycle detection algorithm operates in three phases. First, it constructs a temporary graph including the proposed new relationship. Second, it performs depth-first search from each node, marking nodes as white (unvisited), gray (currently being processed), or black (completely processed). Third, if the algorithm encounters a gray node during traversal, it has found a back edge indicating a cycle and rejects the role relationship change.

Here's the cycle detection algorithm in numbered steps:

1. Begin with the proposed role hierarchy including any new parent-child relationships
2. Initialize all roles as "white" (unvisited) in the color map
3. For each white role, start a depth-first search traversal
4. Mark the current role as "gray" (being processed)
5. For each parent role of the current role, check its color
6. If a parent role is gray, a cycle exists - abort and return validation error
7. If a parent role is white, recursively visit it with DFS
8. After processing all parents, mark the current role as "black" (completed)
9. Continue until all roles have been processed or a cycle is detected

**Transitive closure computation** determines the complete set of permissions available to a role through inheritance. This process combines direct permissions with all inherited permissions from the role hierarchy. The algorithm uses memoization to avoid recomputing permissions for roles that haven't changed since the last calculation.

The transitive closure algorithm follows these steps:

1. Start with the target role and initialize an empty permission set
2. Add all direct permissions from the role to the result set
3. For each parent role, recursively compute its transitive closure
4. Union the parent's permission set with the current result set
5. Apply deduplication to remove redundant permissions
6. Cache the computed permission set with the role's modification timestamp
7. Return the complete permission set including all inherited permissions

This approach ensures that permission changes in parent roles automatically propagate to child roles without requiring explicit updates throughout the hierarchy.

### Permission Lookup Optimization: Bitmap encoding and caching strategies for fast permission checks

Authorization systems face intense performance pressure because permission checks occur on every request in a multi-tenant application. A single API call might trigger dozens of permission checks as it accesses various resources and performs different operations. Traditional approaches that query the database for each permission check quickly become bottlenecks, making aggressive optimization essential.

**Bitmap encoding** provides a high-performance solution for permission storage and lookup. The core insight is that most applications have a relatively small, stable set of permission types (typically 50-200 distinct permissions), which can be efficiently represented as bits in a bitmap. Each bit position corresponds to a specific permission, and checking if a role has a permission becomes a simple bitwise AND operation.

The bitmap optimization works through several mechanisms. First, we maintain a global permission registry that assigns each unique `Permission` a sequential bit position. Second, each role's permission set is encoded as a bitmap where set bits indicate granted permissions. Third, permission inheritance combines bitmaps using bitwise OR operations. Fourth, permission checks become constant-time bitwise AND operations against the user's computed permission bitmap.

> **Decision: Bitmap Permission Encoding for Performance**
> - **Context**: Permission checks occur on every API request and must complete in microseconds to avoid becoming a bottleneck in high-throughput applications
> - **Options Considered**:
>   1. Database query per permission check
>   2. In-memory hash set of permissions per user
>   3. Bitmap encoding with bit positions for permissions
> - **Decision**: Bitmap encoding with in-memory caching and lazy loading
> - **Rationale**: Database queries are too slow for per-request operations. Hash sets provide good performance but bitmap operations are faster for set intersection/union operations common in inheritance. Bitmaps also use less memory (64 permissions = 8 bytes vs 64 strings in a hash set).
> - **Consequences**: Requires permission registry management and limits total permissions to bitmap size (typically 1024-4096), but provides constant-time permission checks essential for system responsiveness.

| Optimization Strategy | Performance | Memory Usage | Complexity | Chosen? |
|----------------------|-------------|--------------|------------|---------|
| Database per check | ~10ms per check | Low | Low | No |
| In-memory hash sets | ~100μs per check | Medium | Medium | No |
| Bitmap encoding | ~1μs per check | Low | High | **Yes** |
| Bloom filters | ~1μs per check | Very Low | Very High | No |

The permission registry maintains the global mapping between permission identifiers and bit positions. This registry must remain stable across system restarts to ensure bitmap consistency, typically stored in a dedicated database table or configuration file.

| Registry Field | Type | Description |
|---------------|------|-------------|
| PermissionID | string | Unique permission identifier (resource:action format) |
| BitPosition | int | Assigned bit position in permission bitmaps (0-based) |
| Description | string | Human-readable permission description |
| CreatedAt | time.Time | When this permission was first registered |
| IsActive | bool | Whether this permission is currently in use |

**Caching strategies** complement bitmap optimization by reducing the frequency of expensive transitive closure computations. The role engine implements a multi-level caching hierarchy that balances memory usage against computation time.

The caching system operates at three levels. **Level 1** caches individual role permission bitmaps with their computed transitive closures, invalidated when the role or its ancestors change. **Level 2** caches user permission bitmaps combining all assigned roles, invalidated when user role assignments change. **Level 3** provides a short-term request cache preventing duplicate computations within a single authorization request.

Cache invalidation follows a **version-based approach** where each cached entry includes a version number. When roles change, affected cache entries are marked with new version numbers rather than being immediately deleted. This allows concurrent requests to continue using slightly stale but consistent permission sets while new requests receive updated permissions.

The cache warming strategy proactively computes permission sets for frequently accessed users and roles. The system tracks access patterns and prioritizes warming cache entries that will likely be needed soon. This approach prevents cache misses from causing user-visible latency spikes during peak usage periods.

### Role Template System: Preconfigured role bundles for common access patterns

Real-world organizations often follow similar patterns when setting up role hierarchies. A typical software company needs development teams with predictable role structures: team leads who can manage projects and mentor developers, senior developers who can review code and deploy applications, junior developers who can write code but need approval for sensitive operations, and QA engineers who can access test environments but not production systems.

**Role templates** capture these common patterns as reusable blueprints that organizations can instantiate and customize for their specific needs. Templates solve several problems: they reduce setup time for new tenants by providing working role hierarchies out of the box, they encode best practices for permission assignment that prevent common security mistakes, and they enable faster onboarding of new team members who fit standard organizational patterns.

A role template consists of multiple interconnected role definitions with predefined inheritance relationships and permission sets. The template system supports **parameterization** allowing organizations to customize aspects like permission scopes, resource types, or approval requirements while maintaining the overall structural pattern.

| Template Component | Type | Description |
|-------------------|------|-------------|
| TemplateID | string | Unique identifier for the template (e.g., "software-dev-team") |
| Name | string | Human-readable template name |
| Description | string | Detailed explanation of the template's purpose and structure |
| RoleDefinitions | []RoleTemplate | Predefined roles included in this template |
| DefaultHierarchy | []RoleRelation | Default parent-child relationships between template roles |
| Parameters | []TemplateParam | Customizable aspects of the template |
| RequiredPermissions | []string | Permissions that must exist before template instantiation |

Each `RoleTemplate` within a template bundle defines a role that will be created when the template is instantiated. These role definitions include placeholder permissions that get resolved to actual permissions based on the organization's resource types and configuration parameters.

| RoleTemplate Field | Type | Description |
|-------------------|------|-------------|
| RoleID | string | Template-local role identifier (becomes part of final role name) |
| DisplayName | string | Human-readable role name with parameter placeholders |
| PermissionPatterns | []string | Permission patterns that get expanded during instantiation |
| RequiredParents | []string | Other template roles this role should inherit from |
| IsOptional | bool | Whether organizations can choose to exclude this role |
| CustomizationHints | map[string]string | Guidance for organizations customizing this role |

Template instantiation transforms abstract role definitions into concrete roles within a tenant's namespace. The process resolves parameter values, expands permission patterns into actual permissions, creates the role hierarchy with proper inheritance relationships, and validates that the resulting structure forms a valid DAG.

> **Decision: Parameterized Role Templates with Pattern Expansion**
> - **Context**: Organizations need quick setup of common role patterns but also need flexibility to adapt roles to their specific resource types and business requirements
> - **Options Considered**:
>   1. Fixed role templates that organizations use as-is
>   2. Parameterized templates with placeholder substitution
>   3. Template inheritance allowing organizations to extend base templates
> - **Decision**: Parameterized templates with pattern-based permission expansion and validation
> - **Rationale**: Fixed templates are too rigid for real organizations that have varying resource types and naming conventions. Full template inheritance adds complexity without significant benefit over parameterization. Pattern expansion allows templates to adapt to organization-specific resource hierarchies while maintaining structural consistency.
> - **Consequences**: Enables rapid organizational onboarding with customization flexibility, at the cost of more complex template definition and instantiation logic. Reduces common setup errors but requires careful validation during instantiation.

Template parameters enable customization of role templates for different organizational contexts. Common parameter types include resource prefixes (e.g., "api", "database", "reports"), environment scopes (e.g., "dev", "staging", "prod"), and approval requirements (e.g., "manager-approval", "peer-review").

| Parameter Type | Example Value | Usage |
|---------------|---------------|-------|
| ResourcePrefix | "api-v2" | Customizes which API versions roles can access |
| EnvironmentScope | "production" | Limits roles to specific deployment environments |
| ApprovalLevel | "manager" | Determines what approval is needed for sensitive actions |
| TeamSize | "small" | Adjusts role granularity based on team structure |
| ComplianceLevel | "sox" | Adds compliance-specific permissions and restrictions |

The instantiation process follows a multi-step validation approach to ensure template safety:

1. **Parameter Validation**: Verify all required parameters are provided and have valid values
2. **Permission Resolution**: Expand permission patterns into concrete permissions that exist in the tenant
3. **Hierarchy Construction**: Build the role inheritance graph from template relationships
4. **Cycle Detection**: Ensure the resulting hierarchy forms a valid DAG
5. **Permission Consistency**: Verify that parent roles have appropriate permissions for inheritance
6. **Conflict Detection**: Check for conflicts with existing roles in the tenant
7. **Atomic Creation**: Create all roles and relationships in a single transaction

**Common template patterns** include development teams, customer support hierarchies, sales organizations, and compliance-focused structures. The development team template typically includes roles like `TeamLead` (can manage projects and approve deployments), `SeniorDev` (can review code and access staging), `JuniorDev` (can write code with review requirements), and `DevOps` (can manage infrastructure and deployments).

Customer support templates often feature `SupportManager` (can access all customer data and escalate issues), `SeniorSupport` (can modify customer accounts and access payment info), `JuniorSupport` (can view customer data and update tickets), and `SupportReadOnly` (can view customer interactions for training purposes).

### Common Pitfalls

⚠️ **Pitfall: Role Inheritance Cycles Creating Infinite Loops**
The most dangerous mistake in role hierarchy implementation is failing to detect circular dependencies before they're created. When Role A inherits from Role B, which inherits from Role C, which inherits back to Role A, permission resolution algorithms enter infinite loops that can crash the authorization system or cause request timeouts. This typically happens when administrators create role relationships through a UI that doesn't validate the complete hierarchy. The fix requires implementing proactive cycle detection that validates the entire role graph before committing any parent-child relationship changes.

⚠️ **Pitfall: Cache Invalidation Race Conditions**
Permission caches improve performance but create consistency problems when roles change. A common mistake is using immediate cache deletion rather than version-based invalidation, leading to race conditions where some requests see old permissions while others see new ones. This inconsistency can cause authorization decisions to flip unexpectedly within the same user session. The solution involves version-based cache invalidation where each cache entry includes a version number, and role changes increment version numbers rather than immediately purging cache entries.

⚠️ **Pitfall: Wildcard Permission Overgranting**
Role templates often include wildcard permissions like `api:*` or `resource:*:read` to simplify configuration, but these can accidentally grant broader access than intended when new resources are added to the system. For example, a role with `user:*` permissions might suddenly gain access to sensitive user financial data when a new `user:payment-info:read` permission is introduced. The fix involves using explicit permission enumeration in templates and implementing permission impact analysis when new permissions are added to the system.

⚠️ **Pitfall: Role Explosion from Over-Granularity**
Organizations often create too many fine-grained roles in an attempt to follow the principle of least privilege, leading to hundreds of roles that become unmanageable. This "role explosion" makes it impossible for administrators to understand the permission landscape and often results in users being over-provisioned with multiple overlapping roles. The solution involves designing role templates around job functions rather than individual permissions, and providing role consolidation tools that identify overlapping roles and suggest mergers.

⚠️ **Pitfall: Tenant Isolation Violations in Role Templates**
Template instantiation code must carefully enforce tenant isolation to prevent roles created in one tenant from accidentally inheriting permissions from roles in another tenant. This can happen when template code uses global role lookups instead of tenant-scoped queries, or when role IDs are generated without tenant prefixes. The fix requires implementing tenant context propagation through all template operations and validating that role references stay within tenant boundaries during instantiation.

### Implementation Guidance

The role and permission engine requires careful attention to data structure design, graph algorithms for hierarchy management, and performance optimization for high-throughput authorization checks. This implementation guidance provides the foundation for building a production-ready RBAC system.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Graph Storage | In-memory adjacency lists with periodic snapshots | Graph database like Neo4j or DGraph |
| Permission Encoding | Hash sets with string keys | Bitmap encoding with bit manipulation |
| Caching | Simple in-memory map with TTL | Redis with Lua scripts for atomic updates |
| Cycle Detection | Recursive DFS with call stack tracking | Iterative DFS with explicit stack |
| Template Storage | JSON files with schema validation | Database with versioning and migration support |

#### Recommended File Structure

```
internal/rbac/
  engine.go              ← Main RoleEngine implementation
  engine_test.go         ← Comprehensive engine tests
  hierarchy.go           ← Role hierarchy and DAG validation
  hierarchy_test.go      ← Hierarchy algorithm tests
  permissions.go         ← Permission bitmap encoding and operations
  permissions_test.go    ← Permission optimization tests
  templates/
    engine.go            ← Template instantiation engine
    engine_test.go       ← Template system tests
    builtin.go           ← Built-in role templates
    validator.go         ← Template validation logic
  cache/
    multilevel.go        ← Multi-level caching implementation
    invalidation.go      ← Version-based cache invalidation
```

#### Infrastructure Starter Code

This complete role storage implementation provides the foundation for hierarchy management:

```go
package rbac

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// RoleStorage provides thread-safe role hierarchy management with cycle detection
type RoleStorage struct {
    roles    map[string]*Role
    tenants  map[string]map[string]*Role  // tenantID -> roleID -> role
    mu       sync.RWMutex
    version  int64
}

func NewRoleStorage() *RoleStorage {
    return &RoleStorage{
        roles:   make(map[string]*Role),
        tenants: make(map[string]map[string]*Role),
    }
}

func (rs *RoleStorage) CreateRole(ctx context.Context, role *Role) error {
    rs.mu.Lock()
    defer rs.mu.Unlock()
    
    // Validate tenant context
    if role.TenantID == "" {
        return fmt.Errorf("role must have tenant ID")
    }
    
    // Initialize tenant if needed
    if rs.tenants[role.TenantID] == nil {
        rs.tenants[role.TenantID] = make(map[string]*Role)
    }
    
    // Check for duplicates
    if _, exists := rs.tenants[role.TenantID][role.ID]; exists {
        return fmt.Errorf("role %s already exists in tenant %s", role.ID, role.TenantID)
    }
    
    // Store role
    fullID := fmt.Sprintf("%s:%s", role.TenantID, role.ID)
    rs.roles[fullID] = role
    rs.tenants[role.TenantID][role.ID] = role
    rs.version++
    
    return nil
}

func (rs *RoleStorage) GetRole(ctx context.Context, tenantID, roleID string) (*Role, error) {
    rs.mu.RLock()
    defer rs.mu.RUnlock()
    
    tenantRoles, exists := rs.tenants[tenantID]
    if !exists {
        return nil, fmt.Errorf("tenant %s not found", tenantID)
    }
    
    role, exists := tenantRoles[roleID]
    if !exists {
        return nil, fmt.Errorf("role %s not found in tenant %s", roleID, tenantID)
    }
    
    return role, nil
}
```

Complete permission bitmap implementation for high-performance lookups:

```go
package rbac

import (
    "fmt"
    "math/big"
    "sync"
)

// PermissionRegistry manages the global mapping between permissions and bit positions
type PermissionRegistry struct {
    permissions map[string]int  // permission ID -> bit position
    reverse     map[int]string  // bit position -> permission ID
    nextBit     int
    mu          sync.RWMutex
}

func NewPermissionRegistry() *PermissionRegistry {
    return &PermissionRegistry{
        permissions: make(map[string]int),
        reverse:     make(map[int]string),
    }
}

func (pr *PermissionRegistry) RegisterPermission(permID string) (int, error) {
    pr.mu.Lock()
    defer pr.mu.Unlock()
    
    if bit, exists := pr.permissions[permID]; exists {
        return bit, nil
    }
    
    if pr.nextBit >= 1024 { // Limit to 1024 permissions
        return -1, fmt.Errorf("permission registry full")
    }
    
    bit := pr.nextBit
    pr.permissions[permID] = bit
    pr.reverse[bit] = permID
    pr.nextBit++
    
    return bit, nil
}

// PermissionBitmap provides high-performance permission storage and lookup
type PermissionBitmap struct {
    bits *big.Int
}

func NewPermissionBitmap() *PermissionBitmap {
    return &PermissionBitmap{
        bits: big.NewInt(0),
    }
}

func (pb *PermissionBitmap) SetPermission(bitPosition int) {
    pb.bits.SetBit(pb.bits, bitPosition, 1)
}

func (pb *PermissionBitmap) HasPermission(bitPosition int) bool {
    return pb.bits.Bit(bitPosition) == 1
}

func (pb *PermissionBitmap) Union(other *PermissionBitmap) *PermissionBitmap {
    result := NewPermissionBitmap()
    result.bits.Or(pb.bits, other.bits)
    return result
}
```

#### Core Logic Skeleton

Main role engine interface that learners should implement:

```go
// RoleEngine provides hierarchical role-based access control with permission inheritance
type RoleEngine struct {
    storage    *RoleStorage
    registry   *PermissionRegistry
    cache      *MultiLevelCache
    templates  *TemplateEngine
}

// HasPermission checks if a user has a specific permission through their assigned roles
func (re *RoleEngine) HasPermission(ctx context.Context, userID, tenantID, resource, action string) (bool, error) {
    // TODO 1: Check cache for user's permission bitmap using cache key "user:{userID}:tenant:{tenantID}:permissions"
    // TODO 2: If cache miss, get user's role assignments from UserRoleAssignment table
    // TODO 3: For each assigned role, compute transitive closure of inherited permissions
    // TODO 4: Combine all role permission bitmaps using Union operations
    // TODO 5: Check if the specific resource:action permission bit is set in combined bitmap
    // TODO 6: Cache the computed permission bitmap with appropriate TTL
    // TODO 7: Return boolean result indicating whether permission is granted
    // Hint: Use PermissionRegistry.RegisterPermission to get bit position for resource:action
}

// GetUserPermissions returns the complete set of permissions for a user in a tenant
func (re *RoleEngine) GetUserPermissions(ctx context.Context, userID, tenantID string) ([]Permission, error) {
    // TODO 1: Get user's role assignments filtered by tenant ID
    // TODO 2: For each role, call computeTransitiveClosure to get all inherited permissions
    // TODO 3: Create a map to deduplicate permissions by permission ID
    // TODO 4: Convert permission bitmap back to Permission slice using registry reverse lookup
    // TODO 5: Sort permissions by resource and action for consistent ordering
    // TODO 6: Return the complete permission list
}

// ValidateRoleHierarchy ensures the role hierarchy forms a valid DAG without cycles
func (re *RoleEngine) ValidateRoleHierarchy(ctx context.Context, tenantID string) error {
    // TODO 1: Get all roles in the specified tenant from storage
    // TODO 2: Build adjacency list representation of parent-child relationships
    // TODO 3: Initialize color map with all roles marked as "white" (unvisited)
    // TODO 4: For each white role, start depth-first search traversal
    // TODO 5: Mark current role as "gray" (being processed)
    // TODO 6: For each parent role, check its color
    // TODO 7: If parent is gray, return error indicating cycle detected
    // TODO 8: If parent is white, recursively visit it
    // TODO 9: Mark role as "black" (completely processed) when all parents processed
    // TODO 10: Return nil if no cycles found
    // Hint: Use three colors - white (0), gray (1), black (2) - to track DFS state
}

// computeTransitiveClosure calculates all permissions available to a role through inheritance
func (re *RoleEngine) computeTransitiveClosure(ctx context.Context, role *Role) (*PermissionBitmap, error) {
    // TODO 1: Check if transitive closure is cached using role ID and modification timestamp
    // TODO 2: Initialize result bitmap with role's direct permissions
    // TODO 3: For each parent role ID, recursively compute its transitive closure
    // TODO 4: Union parent's permission bitmap with result bitmap
    // TODO 5: Cache the computed bitmap with role version for invalidation
    // TODO 6: Return the complete permission bitmap including all inherited permissions
    // Hint: Use memoization to avoid recomputing unchanged role permissions
}
```

Template engine for role bundle management:

```go
// InstantiateTemplate creates concrete roles from a role template in a tenant
func (te *TemplateEngine) InstantiateTemplate(ctx context.Context, templateID, tenantID string, params map[string]interface{}) error {
    // TODO 1: Load template definition from storage by template ID
    // TODO 2: Validate all required parameters are provided in params map
    // TODO 3: For each role in template, expand permission patterns using parameter values
    // TODO 4: Create role inheritance relationships based on template hierarchy
    // TODO 5: Validate resulting role graph forms valid DAG (no cycles)
    // TODO 6: Create all roles atomically in single transaction
    // TODO 7: Log template instantiation for audit purposes
    // Hint: Use database transaction to ensure all-or-nothing role creation
}

// expandPermissionPattern converts template permission patterns to concrete permissions
func (te *TemplateEngine) expandPermissionPattern(pattern string, params map[string]interface{}) ([]Permission, error) {
    // TODO 1: Parse pattern for parameter placeholders like {ResourcePrefix} or {Environment}
    // TODO 2: Replace placeholders with values from params map
    // TODO 3: Handle wildcard expansion (e.g., "api:*" becomes all API permissions)
    // TODO 4: Validate expanded permissions exist in permission registry
    // TODO 5: Return slice of concrete Permission objects
    // Hint: Use regexp for placeholder detection and replacement
}
```

#### Milestone Checkpoint

After implementing the role and permission engine, verify the following functionality:

**Test Command**: `go test ./internal/rbac/... -v`

**Expected Behavior**:
- Role creation and hierarchy validation should pass without errors
- Permission bitmap operations should complete in under 1 microsecond per check
- Cache invalidation should work correctly when role relationships change
- Template instantiation should create valid role hierarchies without cycles

**Manual Verification Steps**:
1. Create a simple role hierarchy: Admin → Manager → Employee
2. Assign permissions to Admin role and verify inheritance flows to child roles
3. Check that circular role relationships are rejected during creation
4. Verify permission bitmap encoding reduces memory usage compared to string sets
5. Confirm cache warming improves authorization check performance

**Signs of Problems**:
- Authorization checks taking more than 100 microseconds indicate missing bitmap optimization
- Memory usage growing unbounded suggests cache invalidation isn't working
- Occasional "role not found" errors indicate race conditions in concurrent access
- Template instantiation creating duplicate roles suggests missing transaction boundaries


## ABAC Policy Engine

> **Milestone(s):** Milestone 2 (ABAC Policy Engine) - this section implements attribute-based policy evaluation with context-aware decision making, extending beyond role-based access to dynamic rule evaluation

The Role and Permission Engine from Milestone 1 provides powerful hierarchical access control, but real-world authorization often requires more dynamic decision making. Consider a scenario where a document editor should only allow editing during business hours, or where sensitive financial records should only be accessible from approved geographic locations, or where file sharing permissions should automatically expire after 30 days. These kinds of contextual, time-sensitive, and condition-based access rules cannot be effectively modeled with static roles and permissions alone.

### ABAC Mental Model: Understanding attribute-based decisions as dynamic rule evaluation

Think of ABAC policy evaluation like a security checkpoint at an airport. Unlike a simple role-based system (where having a "pilot" badge always grants access to the cockpit), ABAC evaluation resembles the complex decision making that happens at airport security. The security officer doesn't just check your role (passenger, crew, pilot) - they evaluate multiple attributes: your identity, your destination, the current threat level, the time of day, your baggage contents, your travel history, and current security policies. All these attributes combine to make a dynamic access decision.

In the same way, ABAC policy evaluation examines attributes from multiple dimensions simultaneously: **user attributes** (role, department, security clearance, employment status), **resource attributes** (classification level, owner, creation date, file type), **environmental attributes** (time of day, IP address, network security level, geographic location), and **action attributes** (read, write, delete, share). The policy engine applies rules that can reference any combination of these attributes to make contextually appropriate access decisions.

The key insight is that ABAC policies are **active rules** rather than static permissions. Instead of pre-computing "User X can access Resource Y," the system evaluates "Under what conditions can users like X access resources like Y?" at request time. This enables incredibly flexible policies like "Managers can approve expense reports under $10,000 during business hours from corporate networks" or "Users can edit their own documents until 30 days after creation, then access becomes read-only."

> **The Power of Context**: ABAC's strength lies in its ability to make decisions based on the full context of the access request, not just the identity of the requester. This enables policies that adapt to changing conditions without requiring manual permission updates.

### Policy Definition Language: Expression syntax for defining attribute-based conditions

Creating an effective policy language requires balancing expressiveness with understandability. Our policy definition language needs to support complex logical conditions while remaining readable to security administrators who aren't necessarily programmers. The language should feel natural for expressing business rules while being precise enough for automated evaluation.

> **Decision: Policy Expression Syntax**
> - **Context**: Need a way to express attribute-based conditions that's both human-readable and machine-evaluable. Considered domain-specific language, JSON-based rules, and code-based policies.
> - **Options Considered**: 
>   1. Custom DSL with English-like syntax ("user.department equals 'Engineering' AND resource.classification less_than 'SECRET'")
>   2. Structured JSON with operators ("{'and': [{'attribute': 'user.department', 'operator': 'equals', 'value': 'Engineering'}]}")
>   3. Embedded scripting language (JavaScript/Lua expressions)
> - **Decision**: Hybrid approach using structured condition trees with readable operators
> - **Rationale**: JSON structure enables programmatic manipulation while readable operators maintain human understandability. Avoids security risks of full scripting languages.
> - **Consequences**: Policies are type-safe and analyzable but limited to predefined operators and functions.

Our policy language represents conditions as **condition trees** that can express arbitrarily complex logical combinations. Each leaf node in the tree represents a single attribute comparison, while internal nodes represent logical operators (AND, OR, NOT). This structure maps naturally to both human reasoning and efficient computer evaluation.

| Condition Element | Purpose | Example |
|-------------------|---------|---------|
| Attribute Reference | Access user, resource, or environment attributes | `user.department`, `resource.owner`, `env.time_of_day` |
| Comparison Operator | Define relationship between attribute and value | `equals`, `not_equals`, `greater_than`, `less_than`, `contains`, `in` |
| Value | Literal value or attribute reference | `"Engineering"`, `30`, `user.id` |
| Logical Combinator | Combine multiple conditions | `AND`, `OR`, `NOT` |
| Function Call | Dynamic attribute computation | `age_in_days(resource.created_at)`, `ip_in_network(env.client_ip, "10.0.0.0/8")` |

The condition tree structure enables policies like:

```
Policy: Engineering Document Access
Effect: ALLOW
Resources: ["document:*"]
Actions: ["read", "write"]
Condition: 
  AND:
    - user.department equals "Engineering"
    - OR:
        - resource.classification equals "PUBLIC"
        - AND:
            - resource.classification equals "INTERNAL"
            - ip_in_network(env.client_ip, "192.168.0.0/16")
    - NOT:
        - user.account_status equals "SUSPENDED"
```

This policy grants access to engineering documents for Engineering department users, but only to public documents or internal documents accessed from the corporate network, and never to suspended users.

| Policy Field | Type | Description |
|--------------|------|-------------|
| ID | string | Unique policy identifier for tracking and updates |
| Name | string | Human-readable policy name for administration |
| TenantID | string | Tenant scope for multi-tenant isolation |
| Effect | PolicyEffect | ALLOW or DENY - the decision if conditions match |
| Conditions | *Condition | Root of the condition tree defining match criteria |
| Resources | []string | Resource patterns this policy applies to |
| Actions | []string | Action types this policy covers |
| Priority | int | Policy evaluation order for conflict resolution |
| CreatedBy | string | Audit trail - who created this policy |
| CreatedAt | time.Time | Audit trail - when policy was created |
| UpdatedAt | time.Time | Audit trail - when policy was last modified |

The `Condition` structure represents our condition tree with recursive structure for complex logical expressions:

| Condition Field | Type | Description |
|-----------------|------|-------------|
| And | []*Condition | List of conditions that must ALL be true |
| Or | []*Condition | List of conditions where ANY can be true |
| Not | *Condition | Single condition to negate (logical NOT) |
| Attribute | string | Dot-notation attribute path like "user.department" |
| Operator | string | Comparison operator: equals, not_equals, greater_than, etc. |
| Value | interface{} | Comparison value - string, number, boolean, or array |

This recursive structure allows arbitrarily complex nesting: AND conditions can contain OR conditions, which can contain NOT conditions, creating sophisticated business rules. The leaf nodes (where Attribute, Operator, and Value are set) perform the actual attribute comparisons, while internal nodes combine results using boolean logic.

> **Design Insight**: The condition tree structure mirrors how humans naturally think about access rules. "Users can access documents IF they're in the right department AND (the document is public OR they're on the corporate network) AND their account isn't suspended" maps directly to the tree structure.

### Policy Evaluation Algorithm: AST parsing and context-aware evaluation with caching

Policy evaluation transforms the abstract condition tree into concrete access decisions by walking the tree and evaluating conditions against the current request context. The evaluation engine must handle attribute resolution, type coercion, operator evaluation, and logical combination while maintaining performance through intelligent caching.

The evaluation process follows these numbered steps:

1. **Context Preparation**: Extract all available attributes from the authorization request, including user attributes from identity stores, resource attributes from the resource metadata, environment attributes from the request context (IP address, timestamp, user agent), and action attributes from the requested operation.

2. **Policy Matching**: Identify which policies potentially apply to this request by matching the requested resource pattern and action against each policy's resource and action filters. This pre-filtering step avoids evaluating irrelevant policies.

3. **Condition Tree Traversal**: For each matching policy, recursively traverse the condition tree from root to leaves. The traversal implements short-circuit evaluation: AND conditions stop at the first false result, OR conditions stop at the first true result.

4. **Attribute Resolution**: At each leaf node, resolve the referenced attribute from the request context. Handle missing attributes according to policy (fail closed by treating missing as false, or fail open by treating missing as true for non-critical attributes).

5. **Operator Evaluation**: Apply the comparison operator to the resolved attribute value and the condition's expected value. Handle type coercion (string "123" compared to number 123) and complex operators like "contains" for arrays or "matches" for regex patterns.

6. **Result Aggregation**: Combine the boolean results from all leaf nodes according to the logical operators in the internal nodes. Build up from leaves to root, applying AND, OR, and NOT operations.

7. **Policy Decision**: If the root condition evaluates to true, the policy matches and its effect (ALLOW or DENY) applies. If false, the policy doesn't apply to this request.

8. **Decision Caching**: Cache the evaluation result using a cache key that includes all attributes that influenced the decision. This enables cache reuse for similar requests while ensuring cache invalidation when relevant attributes change.

![Authorization Request Sequence](./diagrams/authorization-flow.svg)

The evaluation algorithm must handle several complex scenarios that arise in real-world usage:

**Attribute Resolution Chain**: When a policy references `user.manager.department`, the system must resolve the user's manager ID, then resolve that manager's user record, then extract the department field. This chain can fail at any step (user has no manager, manager record not found, manager has no department), requiring careful error handling and default values.

**Type Coercion and Validation**: Policy conditions might compare strings to numbers ("user.age > 18" where age is stored as string "25"), dates to timestamps, or arrays to single values. The evaluation engine needs robust type coercion that handles these mismatches gracefully while maintaining security (failing closed on ambiguous comparisons).

**Function Evaluation**: Complex conditions often require computed attributes like `days_since(resource.last_accessed) > 30` or `ip_in_network(env.client_ip, resource.allowed_networks)`. These functions must be evaluated securely (no arbitrary code execution) while providing useful attribute computation capabilities.

| Evaluation Context Field | Type | Description |
|---------------------------|------|-------------|
| UserAttributes | map[string]interface{} | All user attributes including profile, roles, group memberships |
| ResourceAttributes | map[string]interface{} | Resource metadata, ownership, classification, custom properties |
| EnvironmentAttributes | map[string]interface{} | Request context like IP, timestamp, user agent, geographic location |
| ActionAttributes | map[string]interface{} | Action metadata like operation type, parameters, target fields |
| CacheKey | string | Computed cache key incorporating all influencing attributes |
| EvaluationTrace | []string | Debug trace of which conditions were evaluated and their results |

![Policy Evaluation State Machine](./diagrams/policy-evaluation-state.svg)

> **Performance Consideration**: Attribute resolution can be expensive (database queries, external API calls), so the evaluation engine should batch attribute requests and cache resolved values across policy evaluations within the same authorization request.

**Caching Strategy**: The evaluation engine implements multi-level caching to optimize performance. **Attribute caches** store resolved user and resource attributes within a request context to avoid redundant lookups. **Evaluation result caches** store the final ALLOW/DENY decision keyed by a hash of all influencing attributes. **Condition result caches** store intermediate results for expensive function evaluations like network lookups or complex computations.

The cache key generation is critical for correctness. The key must include all attributes that could influence the evaluation result, but should be stable across requests that would produce the same result. For example, a policy that checks `user.department` and `env.time_of_day` should generate a cache key like `user:123:department:Engineering|env:time_of_day:14:30` that captures the relevant attribute values without including irrelevant details like the exact timestamp.

### Policy Combining Logic: Handling multiple matching policies with priority and deny-overrides

In real systems, multiple policies often match the same authorization request. A user might be covered by a general "Engineering team can read engineering documents" policy, a specific "Senior engineers can edit architecture documents" policy, and a restrictive "No external access to confidential documents" policy all at the same time. The policy engine must have clear rules for combining these potentially conflicting decisions into a single authorization result.

> **Decision: Deny-Overrides Policy Combining**
> - **Context**: Multiple policies can match the same request with conflicting effects (some ALLOW, some DENY). Need deterministic way to resolve conflicts.
> - **Options Considered**:
>   1. First-Match (use first matching policy's decision)
>   2. Allow-Overrides (any ALLOW policy grants access)
>   3. Deny-Overrides (any DENY policy blocks access)
>   4. Policy Priority (highest priority policy wins)
> - **Decision**: Deny-Overrides with Priority ordering for tie-breaking
> - **Rationale**: Security-first approach where explicit denials always take precedence. Priority provides deterministic ordering when multiple policies of same effect match.
> - **Consequences**: System fails secure (accidental over-permissive policies can't override security restrictions) but requires careful policy ordering to avoid unexpected denials.

Our policy combining algorithm implements **deny-overrides with priority ordering**. This means any policy with effect DENY that matches the request will block access, regardless of how many ALLOW policies also match. Within policies of the same effect, higher priority numbers take precedence over lower priority numbers. This approach ensures that security restrictions can't be accidentally bypassed by overly permissive policies.

The policy combination algorithm follows these steps:

1. **Policy Collection**: Gather all policies that match the requested resource pattern and action. This includes exact matches and wildcard matches (e.g., "document:*" matches "document:123").

2. **Policy Evaluation**: Evaluate each matching policy's condition tree against the request context. Collect all policies whose conditions evaluate to true - these are the "applicable policies" for this request.

3. **Effect Separation**: Separate applicable policies into two groups: ALLOW policies and DENY policies. Handle each group independently.

4. **Deny Check**: If any DENY policies are applicable, immediately return DENY with the highest-priority denying policy as the reason. The presence of any applicable DENY policy blocks access regardless of ALLOW policies.

5. **Allow Check**: If no DENY policies apply, check if any ALLOW policies apply. If so, return ALLOW with the applicable ALLOW policies listed in the decision. If no ALLOW policies apply, return DENY by default (fail-safe).

6. **Priority Ordering**: When multiple policies of the same effect apply, order them by priority (highest first) for consistent decision explanations and audit trails.

| Policy Combining Scenario | ALLOW Policies | DENY Policies | Result | Reason |
|----------------------------|----------------|---------------|---------|---------|
| No matching policies | None | None | DENY | No applicable policies (fail-safe default) |
| Only ALLOW matches | 2 applicable | None | ALLOW | Allow policies grant access |
| Only DENY matches | None | 1 applicable | DENY | Deny policy blocks access |
| Mixed matches | 3 applicable | 1 applicable | DENY | Deny overrides allow (security-first) |
| Priority conflict | 2 applicable (priority 100, 200) | None | ALLOW | Higher priority allow policy (200) takes precedence |

This combining logic ensures **predictable security behavior**. Security administrators can write broad ALLOW policies for general access patterns, then add specific DENY policies for exceptions and restrictions. The DENY policies act as security guards that can't be overridden by permissive policies, while priority ordering provides fine-grained control when needed.

**Priority Assignment Guidelines**: Lower numbers represent higher priority for system/security policies, higher numbers represent higher priority for user/application policies. This convention allows security teams to create override policies with very low priority numbers that take precedence over application-defined policies.

| Priority Range | Policy Type | Example |
|----------------|-------------|---------|
| 1-100 | Security/Compliance | "DENY access to PII from external networks" |
| 101-500 | Administrative | "ALLOW managers to approve expense reports" |
| 501-1000 | Application-defined | "ALLOW document owners to edit their documents" |
| 1001+ | User-defined | "ALLOW temporary contractor access to specific project files" |

The `PolicyDecision` structure captures the complete evaluation result for audit and debugging purposes:

| PolicyDecision Field | Type | Description |
|----------------------|------|-------------|
| Allowed | bool | Final access decision after combining all applicable policies |
| AppliedPolicies | []string | List of policy IDs that matched and influenced the decision |
| DenyingPolicy | string | ID of the highest-priority DENY policy if access was denied |

> **Common Pitfall**: Policy priority can create unexpected behavior if not carefully managed. A high-priority ALLOW policy can't override a DENY policy (deny-overrides), but administrators sometimes expect it to. Clear documentation and policy testing are essential.

⚠️ **Pitfall: Policy Explosion**
As systems grow, the number of policies can explode exponentially, leading to performance issues and conflicts. A user might match dozens of policies, each requiring attribute resolution and condition evaluation. The solution is **policy hierarchies** and **policy templates** that group related policies and enable bulk operations. Additionally, **policy analysis tools** can detect conflicting policies and performance bottlenecks before they impact production.

⚠️ **Pitfall: Attribute Dependency Cycles**
Policies can create circular dependencies in attribute resolution: "user.manager_approval_limit > resource.cost" where manager_approval_limit depends on the manager's role, which depends on their department, which might depend on the resource being accessed. The evaluation engine must detect and break these cycles, typically by failing closed (denying access) when cycles are detected.

⚠️ **Pitfall: Context Injection Attacks**
If request context attributes come from untrusted sources (client-provided headers, URL parameters), attackers might manipulate them to bypass policies. For example, sending "X-User-Department: Engineering" header to match department-based policies. All context attributes must be validated and sourced from trusted systems, with client-provided attributes treated as supplementary only.

### Common Pitfalls

⚠️ **Pitfall: Missing Attribute Handling**
Policy conditions often reference attributes that don't exist for all requests. A condition like `user.security_clearance >= 'SECRET'` fails when the user has no security clearance attribute. The default behavior should be fail-safe (treat missing as false for restrictive conditions, true for permissive conditions), but this must be explicitly designed, not left to chance. Use explicit null checks and default values in policy conditions.

⚠️ **Pitfall: Expensive Attribute Resolution**
Some attributes require expensive operations like database queries, API calls, or complex computations. A policy that checks `user.direct_reports.count > 5` might trigger dozens of database queries. The solution is **lazy attribute resolution** (only resolve attributes referenced by matching policies) and **attribute batching** (resolve multiple attributes in single queries). Consider pre-computing expensive attributes during off-peak hours.

⚠️ **Pitfall: Policy Testing Gaps**
ABAC policies are complex and context-dependent, making them prone to subtle bugs. A policy might work correctly for most users but fail edge cases like users with multiple departments or resources with missing owners. Comprehensive policy testing requires **scenario-based test suites** with realistic attribute combinations, **negative testing** (ensuring proper denials), and **policy simulation environments** that mirror production data.

⚠️ **Pitfall: Cache Invalidation Complexity**
ABAC caching is much more complex than RBAC caching because decisions depend on multiple dynamic attributes. When a user's department changes, all cached decisions that considered user.department must be invalidated. When a resource's classification changes, all cached decisions for that resource must be invalidated. Use **attribute-based cache tagging** where each cache entry is tagged with the attributes it depends on, enabling targeted invalidation.

### Implementation Guidance

The ABAC Policy Engine bridges the gap between abstract policy rules and concrete access decisions through careful attribute management and efficient evaluation algorithms.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Policy Storage | JSON files with validation | Policy database with versioning |
| Expression Evaluation | Hand-written recursive evaluator | CEL (Common Expression Language) |
| Attribute Resolution | Direct database queries | Attribute service with caching |
| Policy Language | JSON-based condition trees | Domain-specific language parser |
| Caching | In-memory LRU cache | Distributed cache with invalidation |

**B. Recommended File Structure:**

```
internal/
  policy/
    engine.go              ← PolicyEngine implementation
    engine_test.go         ← Policy evaluation test suites
    condition.go           ← Condition tree evaluation logic
    condition_test.go      ← Condition parsing and evaluation tests
    attributes.go          ← Attribute resolution service
    cache.go              ← Policy evaluation caching
    combining.go          ← Policy combining logic (deny-overrides)
    types.go              ← Policy, Condition, and decision types
  storage/
    policy_store.go       ← Policy persistence interface
    memory_store.go       ← In-memory store for testing
    db_store.go           ← Database-backed policy storage
```

**C. Infrastructure Starter Code:**

```go
// Package policy provides ABAC policy evaluation capabilities
package policy

import (
    "context"
    "fmt"
    "strings"
    "time"
)

// PolicyEffect represents the outcome when a policy condition matches
type PolicyEffect string

const (
    PolicyEffectAllow PolicyEffect = "ALLOW"
    PolicyEffectDeny  PolicyEffect = "DENY"
)

// Policy represents an ABAC policy with conditions and effects
type Policy struct {
    ID         string       `json:"id"`
    Name       string       `json:"name"`
    TenantID   string       `json:"tenant_id"`
    Effect     PolicyEffect `json:"effect"`
    Conditions *Condition   `json:"conditions,omitempty"`
    Resources  []string     `json:"resources"`
    Actions    []string     `json:"actions"`
    Priority   int          `json:"priority"`
    CreatedBy  string       `json:"created_by"`
    CreatedAt  time.Time    `json:"created_at"`
    UpdatedAt  time.Time    `json:"updated_at"`
}

// Condition represents a node in the policy condition tree
type Condition struct {
    // Logical operators (internal nodes)
    And []*Condition `json:"and,omitempty"`
    Or  []*Condition `json:"or,omitempty"`
    Not *Condition   `json:"not,omitempty"`
    
    // Comparison operators (leaf nodes)
    Attribute string      `json:"attribute,omitempty"`
    Operator  string      `json:"operator,omitempty"`
    Value     interface{} `json:"value,omitempty"`
}

// PolicyDecision represents the result of policy evaluation
type PolicyDecision struct {
    Allowed         bool     `json:"allowed"`
    AppliedPolicies []string `json:"applied_policies"`
    DenyingPolicy   string   `json:"denying_policy,omitempty"`
}

// EvaluationContext holds all attributes available for policy evaluation
type EvaluationContext struct {
    UserAttributes        map[string]interface{} `json:"user_attributes"`
    ResourceAttributes    map[string]interface{} `json:"resource_attributes"`
    EnvironmentAttributes map[string]interface{} `json:"environment_attributes"`
    ActionAttributes      map[string]interface{} `json:"action_attributes"`
}

// PolicyStore interface for policy persistence
type PolicyStore interface {
    GetPolicies(ctx context.Context, tenantID string) ([]*Policy, error)
    GetPolicy(ctx context.Context, policyID string) (*Policy, error)
    CreatePolicy(ctx context.Context, policy *Policy) error
    UpdatePolicy(ctx context.Context, policy *Policy) error
    DeletePolicy(ctx context.Context, policyID string) error
}

// AttributeResolver resolves attribute values from various sources
type AttributeResolver interface {
    ResolveUserAttributes(ctx context.Context, userID, tenantID string) (map[string]interface{}, error)
    ResolveResourceAttributes(ctx context.Context, resourceID string) (map[string]interface{}, error)
    ResolveEnvironmentAttributes(ctx context.Context, req *AuthzRequest) (map[string]interface{}, error)
}

// Simple in-memory attribute resolver for testing
type MemoryAttributeResolver struct {
    users     map[string]map[string]interface{}
    resources map[string]map[string]interface{}
}

func NewMemoryAttributeResolver() *MemoryAttributeResolver {
    return &MemoryAttributeResolver{
        users:     make(map[string]map[string]interface{}),
        resources: make(map[string]map[string]interface{}),
    }
}

func (r *MemoryAttributeResolver) SetUserAttributes(userID string, attrs map[string]interface{}) {
    r.users[userID] = attrs
}

func (r *MemoryAttributeResolver) SetResourceAttributes(resourceID string, attrs map[string]interface{}) {
    r.resources[resourceID] = attrs
}

func (r *MemoryAttributeResolver) ResolveUserAttributes(ctx context.Context, userID, tenantID string) (map[string]interface{}, error) {
    if attrs, exists := r.users[userID]; exists {
        return attrs, nil
    }
    return make(map[string]interface{}), nil
}

func (r *MemoryAttributeResolver) ResolveResourceAttributes(ctx context.Context, resourceID string) (map[string]interface{}, error) {
    if attrs, exists := r.resources[resourceID]; exists {
        return attrs, nil
    }
    return make(map[string]interface{}), nil
}

func (r *MemoryAttributeResolver) ResolveEnvironmentAttributes(ctx context.Context, req *AuthzRequest) (map[string]interface{}, error) {
    return map[string]interface{}{
        "timestamp":  req.Timestamp.Unix(),
        "time_of_day": req.Timestamp.Format("15:04"),
        "day_of_week": req.Timestamp.Weekday().String(),
        "request_id": req.RequestID,
    }, nil
}
```

**D. Core Logic Skeleton Code:**

```go
// PolicyEngine evaluates ABAC policies against authorization requests
type PolicyEngine struct {
    store     PolicyStore
    resolver  AttributeResolver
    cache     map[string]*PolicyDecision // Simple cache - use proper cache in production
}

// NewPolicyEngine creates a new ABAC policy evaluation engine
func NewPolicyEngine(store PolicyStore, resolver AttributeResolver) *PolicyEngine {
    return &PolicyEngine{
        store:    store,
        resolver: resolver,
        cache:    make(map[string]*PolicyDecision),
    }
}

// Evaluate processes an authorization request against ABAC policies
func (e *PolicyEngine) Evaluate(ctx context.Context, req *AuthzRequest) (*PolicyDecision, error) {
    // TODO 1: Build evaluation context by resolving all attributes
    //   - User attributes from resolver.ResolveUserAttributes()
    //   - Resource attributes from resolver.ResolveResourceAttributes()  
    //   - Environment attributes from resolver.ResolveEnvironmentAttributes()
    //   - Action attributes from request (action type, parameters)
    
    // TODO 2: Get all policies for the tenant from store.GetPolicies()
    
    // TODO 3: Filter policies that match the requested resource and action
    //   - Check if req.Resource matches any pattern in policy.Resources
    //   - Check if req.Action matches any action in policy.Actions
    //   - Use matchesPattern() helper for wildcard matching
    
    // TODO 4: Evaluate condition tree for each matching policy
    //   - Call EvaluateCondition() for each policy's condition tree
    //   - Collect policies where conditions evaluate to true
    
    // TODO 5: Separate applicable policies by effect (ALLOW vs DENY)
    //   - Create separate slices for allow and deny policies
    //   - Sort each slice by priority (highest first)
    
    // TODO 6: Apply deny-overrides combining logic
    //   - If any DENY policies apply, return DENY with highest priority denying policy
    //   - If no DENY policies but ALLOW policies apply, return ALLOW
    //   - If no policies apply, return DENY (fail-safe default)
    
    // TODO 7: Build and return PolicyDecision with full audit trail
    //   - Include all applied policy IDs
    //   - Include denying policy ID if denied
    //   - Set allowed boolean based on final decision
    
    return nil, fmt.Errorf("not implemented")
}

// EvaluateCondition recursively evaluates a condition tree against the evaluation context
func EvaluateCondition(condition *Condition, context *EvaluationContext) (bool, error) {
    if condition == nil {
        return true, nil // Empty condition is considered true
    }
    
    // TODO 1: Handle logical AND conditions
    //   - If condition.And is not empty, evaluate all sub-conditions
    //   - Return true only if ALL sub-conditions are true
    //   - Use short-circuit evaluation (return false on first false)
    
    // TODO 2: Handle logical OR conditions  
    //   - If condition.Or is not empty, evaluate sub-conditions
    //   - Return true if ANY sub-condition is true
    //   - Use short-circuit evaluation (return true on first true)
    
    // TODO 3: Handle logical NOT conditions
    //   - If condition.Not is not nil, evaluate the sub-condition
    //   - Return the opposite of the sub-condition result
    
    // TODO 4: Handle leaf node attribute comparisons
    //   - Extract attribute value from context using resolveAttribute()
    //   - Apply the comparison operator to attribute and condition.Value
    //   - Use evaluateOperator() helper for different operator types
    
    // TODO 5: Handle invalid/malformed conditions
    //   - Return error for conditions with no valid fields set
    //   - Log warnings for conditions that can't be evaluated
    
    return false, fmt.Errorf("not implemented")
}

// Helper function to resolve dotted attribute paths like "user.department"
func resolveAttribute(attributePath string, context *EvaluationContext) (interface{}, error) {
    // TODO: Split attribute path by dots and traverse the context maps
    // Example: "user.department" -> look in context.UserAttributes["department"]
    // Example: "resource.owner.department" -> nested attribute resolution
    return nil, fmt.Errorf("not implemented")
}

// Helper function to evaluate different comparison operators
func evaluateOperator(operator string, attributeValue, expectedValue interface{}) (bool, error) {
    // TODO: Implement comparison operators:
    //   - "equals", "not_equals": direct comparison with type coercion
    //   - "greater_than", "less_than": numeric comparisons
    //   - "contains": substring/array membership
    //   - "in": check if attribute value is in expected array
    //   - "matches": regex pattern matching
    return false, fmt.Errorf("not implemented")
}

// Helper function to check if resource matches policy resource patterns
func matchesPattern(resource string, patterns []string) bool {
    // TODO: Implement pattern matching with wildcard support
    // Example: "document:*" matches "document:123" 
    // Example: "file:*.pdf" matches "file:report.pdf"
    return false
}
```

**E. Language-Specific Hints for Go:**

- Use `reflect` package for dynamic type coercion in operator evaluation
- Use `regexp` package for pattern matching and "matches" operator
- Use `strings.Split(attributePath, ".")` for dotted attribute path parsing
- Use `time.Time` methods for time-based comparisons in conditions
- Consider `sync.RWMutex` for thread-safe policy cache access
- Use `context.Context` for request timeouts and cancellation
- Use `json.Marshal/Unmarshal` for policy serialization to storage

**F. Milestone Checkpoint:**

After implementing the ABAC Policy Engine, verify these behaviors:

1. **Basic Policy Evaluation**: Create a simple policy with user.department equals "Engineering" condition. Test with user having that department (should allow) and without (should deny).

2. **Complex Condition Trees**: Create policy with AND/OR/NOT combinations. Verify that "user.role = 'manager' AND (resource.classification = 'public' OR user.clearance >= 'secret')" evaluates correctly for different attribute combinations.

3. **Policy Combining Logic**: Create conflicting ALLOW and DENY policies for same resource. Verify that DENY always wins regardless of priority within same effect.

4. **Attribute Resolution**: Test dotted attribute paths like "user.manager.department" with nested attribute resolution.

5. **Performance with Caching**: Run the same authorization request multiple times and verify caching improves performance without affecting correctness.

Expected test command: `go test ./internal/policy/... -v`

Expected behavior: All policy evaluation tests pass, including edge cases like missing attributes, malformed conditions, and policy conflicts.

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Policy always denies | Missing attributes treated as false | Add logging to attribute resolution | Set explicit default values for missing attributes |
| Slow policy evaluation | Expensive attribute resolution on every request | Profile attribute resolution calls | Implement attribute caching and batching |
| Inconsistent decisions | Race conditions in policy cache | Check for concurrent cache access | Add proper locking to cache operations |
| Policy conflicts | Multiple policies with unclear precedence | Enable policy evaluation tracing | Review priority assignments and combining logic |
| Condition never matches | Type mismatch in comparisons | Log attribute and expected value types | Add type coercion in operator evaluation |


## Resource-Based Access and Multi-tenancy

> **Milestone(s):** Milestone 3 (Resource-Based & Multi-tenancy) - this section implements fine-grained resource permissions with tenant isolation for SaaS applications

Moving from role-based permissions to resource-based access control represents a fundamental shift in thinking. Where RBAC asks "What can this person do in general?", resource-based access asks "What can this person do with this specific thing?" This granular approach becomes essential in multi-tenant SaaS applications where organizations must be completely isolated from each other, yet individual resources within a tenant may have their own permission requirements.

Think of this transition as moving from a traditional office building with floor-by-floor security badges to a modern co-working space where every desk, conference room, and storage locker has its own access rules. Each tenant organization gets their own secure area, but within that area, different people have different access to specific resources based on ownership, sharing, and project membership.

The challenge lies in implementing this granular control efficiently while maintaining complete tenant isolation. A single authorization decision must simultaneously verify tenant membership, resource ownership, sharing grants, and hierarchical permissions - all while ensuring that data from different tenants never mingles.

### Resource Ownership Mental Model

Understanding resource-based access control begins with the mental model of **property ownership with sharing rules**. Just as owning a house gives you full control over who can enter, modify, or use different rooms, owning a resource in our system grants complete control over its access permissions.

Consider a document management system where Alice creates a project proposal. By creating it, Alice becomes the **resource owner** and gains full control: she can read, edit, delete, share, or transfer ownership. This ownership exists within her tenant boundary - Alice works for Company A, so the document belongs to Company A's tenant space. Bob, who works for Company B, cannot even discover the document exists, let alone access it.

The ownership model establishes a clear hierarchy of access rights that mirrors real-world property concepts:

| Ownership Level | Access Rights | Example Scenario |
|----------------|---------------|------------------|
| Owner | Full control: read, write, delete, share, transfer | Alice created the document and can do anything with it |
| Shared Recipient | Granted permissions only | Alice shared read-write access with colleague Carol |
| Tenant Member | Tenant-wide role permissions | Dave has "Editor" role on all Company A documents |
| External User | Cross-tenant sharing only | Emily from Company B gets limited access via share link |

The ownership mental model helps us understand the **permission scope hierarchy**. When someone requests access to a resource, the system evaluates permissions in order of specificity:

1. **Direct ownership** - Does the user own this resource?
2. **Explicit sharing** - Has the owner shared this resource with the user?
3. **Tenant role permissions** - Do the user's roles within this tenant grant access?
4. **Cross-tenant sharing** - Are there any explicit cross-tenant grants?

This hierarchy ensures that more specific permissions can override general ones, just like how a house key works regardless of neighborhood security rules.

> **Key Insight**: Resource ownership creates a security boundary that's more granular than tenant boundaries but operates within them. An owner can grant access that exceeds what tenant roles would normally allow, but cannot grant access across tenant boundaries without explicit cross-tenant mechanisms.

### Resource Ownership Implementation Model

The `Resource` structure captures the essential ownership information needed for access decisions:

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier for the resource |
| Type | string | Resource category (document, project, dataset) |
| TenantID | string | Owning tenant - establishes primary security boundary |
| OwnerID | string | User who owns this resource and controls sharing |
| Attributes | map[string]any | Resource-specific attributes for policy evaluation |
| CreatedAt | time.Time | Resource creation timestamp |
| UpdatedAt | time.Time | Last modification timestamp |
| ShareSettings | *ShareSettings | Sharing configuration and active grants |

The `ShareSettings` structure manages how resources can be shared beyond direct ownership:

| Field | Type | Description |
|-------|------|-------------|
| SharedWith | []ShareGrant | Explicit grants to specific users or roles |
| ShareLink | *ShareLink | Public/semi-public sharing configuration |
| ExpiresAt | *time.Time | When all sharing expires (nil for permanent) |
| RequiresApproval | bool | Whether shares need owner approval |
| MaxShares | int | Maximum number of active shares (0 for unlimited) |
| AllowedDomains | []string | Email domains allowed for sharing |

Each individual sharing grant contains detailed tracking information:

| Field | Type | Description |
|-------|------|-------------|
| GranteeType | string | Type of grantee: "user", "role", "domain" |
| GranteeID | string | Identifier for the grantee |
| Permissions | []string | Specific permissions granted |
| GrantedBy | string | User who created this grant |
| GrantedAt | time.Time | When the grant was created |
| ExpiresAt | *time.Time | When this grant expires (nil for permanent) |
| LastAccessed | *time.Time | When grantee last used this access |

> **Decision: Resource-Centric Permission Model**
> - **Context**: Need to support fine-grained access control where different resources within a tenant have different permission requirements
> - **Options Considered**: Role-only permissions, ACL per resource, hybrid ownership model
> - **Decision**: Hybrid model with ownership as primary control plus sharing grants
> - **Rationale**: Ownership provides clear default permissions and sharing responsibility, while grants allow flexible collaboration without role explosion
> - **Consequences**: Requires permission evaluation at resource level, but enables natural sharing workflows and clear security boundaries

### Tenant Isolation Design

Tenant isolation represents the absolute security boundary in a multi-tenant system. Think of tenant isolation like **apartment building security** - residents of apartment 3A should never be able to access apartment 3B's belongings, even if the lock is broken or they have a universal maintenance key. The building's infrastructure ensures complete separation between units.

In our authorization system, tenant isolation operates at multiple layers to ensure complete data separation:

**Database-Level Isolation**: Every query must include tenant context, either through row-level security policies or explicit WHERE clauses. This prevents data leakage even if application logic has bugs.

**Authorization Context Propagation**: The tenant context travels with every request through middleware layers, ensuring that authorization decisions always occur within the correct tenant boundary.

**Permission Scope Enforcement**: User permissions are always evaluated within their tenant membership context - a user cannot accidentally gain permissions in a tenant they don't belong to.

![Multi-Tenant Resource Isolation](./diagrams/tenant-isolation.svg)

The tenant isolation design relies on strict **context propagation** through the request pipeline:

| Pipeline Stage | Tenant Validation | Action on Violation |
|----------------|-------------------|---------------------|
| Authentication Middleware | Extract user's tenant memberships | Reject request if no valid session |
| Authorization Middleware | Validate requested tenant access | Return 403 if user not tenant member |
| Resource Resolution | Ensure resource belongs to tenant | Return 404 if tenant mismatch |
| Database Query | Apply tenant filters to all queries | Automatic row-level filtering |
| Response Filtering | Strip cross-tenant references | Remove unauthorized data from responses |

> **Critical Security Principle**: Tenant isolation must be enforced at the database level, not just the application level. Application bugs can bypass tenant checks, but properly configured database constraints and row-level security provide defense in depth.

### Tenant Membership and Role Assignment

Users can belong to multiple tenants with different roles in each. The `TenantMembership` structure tracks these relationships:

| Field | Type | Description |
|-------|------|-------------|
| TenantID | string | Tenant organization identifier |
| RoleIDs | []string | Roles assigned within this tenant |
| JoinedAt | time.Time | When user joined this tenant |
| Status | MembershipStatus | Active, suspended, or pending membership |

The `MembershipStatus` enumeration defines the possible states:

| Status | Meaning | Authorization Effect |
|---------|---------|---------------------|
| MembershipActive | Full tenant member | All assigned role permissions active |
| MembershipSuspended | Temporarily disabled | No permissions, cannot access resources |
| MembershipPending | Awaiting approval | Read-only access to shared resources only |
| MembershipRevoked | Permanently removed | No access, cannot be re-invited without admin action |

User role assignments operate within tenant boundaries using `UserRoleAssignment`:

| Field | Type | Description |
|-------|------|-------------|
| UserID | string | User receiving the role assignment |
| RoleID | string | Role being assigned |
| TenantID | string | Tenant context for this assignment |
| ValidFrom | *time.Time | When assignment becomes active |
| ValidTo | *time.Time | When assignment expires |
| AssignedBy | string | User who made this assignment |
| AssignedAt | time.Time | When assignment was created |

> **Decision: Multi-Tenant Role Assignment Model**
> - **Context**: Users need different permissions in different tenants, and role assignments must be completely isolated between tenants
> - **Options Considered**: Global roles with tenant filters, separate role hierarchies per tenant, hybrid approach
> - **Decision**: Roles defined per tenant with user assignments scoped to specific tenant-role combinations
> - **Rationale**: Provides complete isolation while allowing tenants to customize their role structure without affecting others
> - **Consequences**: More complex role management, but eliminates cross-tenant privilege escalation risks and supports tenant-specific organizational structures

### Hierarchical Resource Access

Many applications organize resources in hierarchical relationships - projects contain documents, folders contain files, organizations contain teams. These hierarchies need **permission inheritance** so that granting access to a parent resource provides appropriate access to its children.

Think of hierarchical resource access like **building security zones**. Having keycard access to the executive floor typically grants access to the executive conference rooms and offices on that floor. However, the CEO's private office might require additional permissions beyond floor access.

The resource hierarchy model supports this through parent-child relationships with configurable inheritance rules:

| Relationship Type | Inheritance Behavior | Example |
|------------------|---------------------|---------|
| Strong Inheritance | Child inherits all parent permissions | Folder permissions apply to contained files |
| Weak Inheritance | Child inherits read-only parent permissions | Project members can view all documents but need explicit edit rights |
| No Inheritance | Child permissions independent of parent | Sensitive documents in project don't inherit project permissions |
| Override Inheritance | Child can deny permissions granted at parent | Private folder in shared project |

The hierarchical resource structure extends the basic `Resource` with relationship information:

| Field | Type | Description |
|-------|------|-------------|
| ParentID | *string | Parent resource ID (nil for root resources) |
| Children | []string | List of direct child resource IDs |
| InheritanceRule | InheritanceType | How permissions flow from parent to child |
| MaxDepth | int | Maximum hierarchy depth for this resource tree |

Permission evaluation in hierarchical resources follows a **cascade algorithm**:

1. **Direct permissions**: Check explicit grants on the target resource
2. **Parent inheritance**: Walk up the hierarchy, applying inheritance rules
3. **Ownership propagation**: Verify that parent ownership relationships are valid
4. **Tenant boundary enforcement**: Ensure all resources in the path belong to the same tenant
5. **Permission combination**: Merge inherited and direct permissions using precedence rules

> **Decision: Configurable Permission Inheritance**
> - **Context**: Different resource types need different inheritance behaviors - some want full cascade, others need strict boundaries
> - **Options Considered**: Always inherit, never inherit, configurable per resource type, configurable per resource instance
> - **Decision**: Configurable inheritance rules per resource instance with sensible defaults per type
> - **Rationale**: Maximizes flexibility while preventing unexpected permission grants through hierarchy manipulation
> - **Consequences**: More complex evaluation logic, but supports diverse use cases without requiring separate authorization systems

### Hierarchical Permission Evaluation Algorithm

The hierarchical permission evaluation process requires careful consideration of inheritance rules and boundary conditions:

1. **Validate hierarchy integrity**: Ensure the resource hierarchy forms a valid tree (no cycles) within the tenant boundary
2. **Collect permission path**: Starting from the target resource, collect all resources up to the root, respecting inheritance rules
3. **Apply ownership rules**: For each resource in the path, determine if the requesting user has ownership or sharing grants
4. **Combine permissions**: Merge permissions from the hierarchy path using precedence rules (direct > inherited > role-based)
5. **Apply deny-overrides**: If any resource in the path explicitly denies access, deny the request regardless of other grants
6. **Cache result**: Store the computed permissions with appropriate cache invalidation keys

Common pitfalls in hierarchical permission implementation:

⚠️ **Pitfall: Hierarchy Cycle Creation**
Resource hierarchies must form a directed acyclic graph (DAG) to prevent infinite loops during permission evaluation. A cycle occurs when a resource becomes its own ancestor through a chain of parent-child relationships. This can happen during resource moves or bulk operations. Always validate that setting a parent relationship doesn't create a cycle by checking if the proposed parent is a descendant of the child resource.

⚠️ **Pitfall: Cross-Tenant Hierarchy Leakage**
Resource hierarchies must never span tenant boundaries, as this could allow permission escalation across tenant isolation boundaries. When setting parent-child relationships, verify that both resources belong to the same tenant. When evaluating hierarchical permissions, stop traversal if you encounter a tenant boundary mismatch and log this as a potential security violation.

⚠️ **Pitfall: Inheritance Override Confusion**
When a child resource has stricter permissions than its parent, users often expect that parent permissions still apply. However, proper security requires that explicit child permissions override inherited permissions. Document this behavior clearly and provide audit trails showing how permissions were determined for each resource access.

### Controlled Cross-Tenant Access

While tenant isolation is fundamental, business needs sometimes require **controlled collaboration** between organizations. Think of this like **diplomatic immunity** - there are very specific, limited circumstances where someone from outside your organization can access your resources, but only through explicit, auditable, and revocable grants.

Cross-tenant access violates the normal isolation boundaries, so it requires special handling with enhanced security measures:

**Explicit Grant Requirement**: Cross-tenant access cannot happen through normal role inheritance or resource sharing. It requires an explicit decision by someone authorized to make cross-tenant grants.

**Time-Limited Access**: All cross-tenant grants must have expiration times to prevent indefinite access accumulation. This forces periodic review of cross-tenant relationships.

**Enhanced Audit Logging**: Every cross-tenant access must be logged with additional context about why the access was granted and who authorized it.

**Revocation Guarantees**: Cross-tenant access must be immediately revocable without waiting for cache expiration or other delays.

The cross-tenant sharing model uses enhanced `ShareGrant` structures with additional restrictions:

| Field | Type | Description |
|-------|------|-------------|
| CrossTenant | bool | Whether this grant crosses tenant boundaries |
| AuthorizedBy | string | Admin who approved cross-tenant access |
| BusinessJustification | string | Reason for cross-tenant grant |
| ReviewRequired | bool | Whether grant needs periodic review |
| NextReviewDate | *time.Time | When this grant must be reviewed |
| MaxUsageCount | int | Maximum number of times grant can be used |
| CurrentUsageCount | int | How many times grant has been used |

> **Decision: Explicit Cross-Tenant Authorization**
> - **Context**: Business needs require some collaboration between tenant organizations, but normal tenant isolation must remain intact
> - **Options Considered**: Disable tenant isolation for shared resources, create shared tenant space, explicit cross-tenant grants
> - **Decision**: Explicit cross-tenant grants with enhanced auditing and mandatory expiration
> - **Rationale**: Maintains strong tenant isolation as the default while providing escape hatch for legitimate business needs with appropriate controls
> - **Consequences**: More complex sharing UI and additional admin overhead, but preserves security properties and provides audit trail for compliance

### Cross-Tenant Sharing Implementation

Cross-tenant sharing requires careful implementation to prevent security violations while enabling legitimate collaboration:

**Grant Creation Process**:
1. **Authority validation**: Verify that the grant creator has cross-tenant sharing permissions in their tenant
2. **Target validation**: Confirm that the target user/tenant exists and can receive cross-tenant grants
3. **Business justification**: Require documentation of why cross-tenant access is needed
4. **Approval workflow**: Route high-privilege grants through additional approval steps
5. **Expiration assignment**: Set mandatory expiration dates based on grant scope and permissions
6. **Audit event creation**: Log the grant creation with full context for compliance tracking

**Runtime Enforcement**:
1. **Cross-tenant detection**: Identify when an authorization request crosses tenant boundaries
2. **Explicit grant lookup**: Search for specific cross-tenant grants rather than normal permission inheritance
3. **Usage tracking**: Increment usage counters and update last-access timestamps
4. **Expiration checking**: Verify that grants haven't expired or exceeded usage limits
5. **Enhanced logging**: Record cross-tenant access with additional security context

**Revocation Process**:
1. **Immediate invalidation**: Remove grants from active permission stores without delay
2. **Cache invalidation**: Purge any cached permissions that might include revoked grants
3. **Notification**: Inform affected users that their cross-tenant access has been revoked
4. **Audit trail**: Log revocation with reason and actor for compliance tracking

Common pitfalls in cross-tenant sharing:

⚠️ **Pitfall: Cached Cross-Tenant Permissions**
Cross-tenant permissions should never be cached with the same duration as normal tenant permissions, as they need to be revocable immediately. Use separate, shorter-lived cache entries for cross-tenant grants, and implement cache invalidation that can target specific cross-tenant relationships.

⚠️ **Pitfall: Transitive Cross-Tenant Access**
Users who receive cross-tenant access should not be able to further share that access with others, as this could lead to uncontrolled permission propagation. Mark cross-tenant grants as non-transferable and check for this during any sharing operations.

⚠️ **Pitfall: Cross-Tenant Admin Escalation**
Admin users in one tenant should not automatically gain admin privileges in other tenants, even through cross-tenant sharing. Cross-tenant grants should specify exactly which permissions are granted, never inherit role-based permissions across tenant boundaries.

### Resource Access Evaluation Pipeline

The complete resource access evaluation combines ownership, tenant membership, hierarchical permissions, and cross-tenant grants into a unified decision pipeline:

| Stage | Input | Processing | Output |
|-------|--------|------------|---------|
| Request Validation | AuthzRequest | Validate resource exists, extract tenant context | Validated request + resource metadata |
| Tenant Membership Check | User + Tenant ID | Verify user belongs to resource tenant | Tenant membership status |
| Ownership Evaluation | User + Resource | Check direct ownership and sharing grants | Ownership-based permissions |
| Hierarchical Resolution | Resource + Permissions | Walk hierarchy, apply inheritance rules | Inherited permissions |
| Role Permission Merge | User roles + Resource type | Apply tenant role permissions | Role-based permissions |
| Cross-Tenant Evaluation | User + Resource + Tenant mismatch | Check explicit cross-tenant grants | Cross-tenant permissions |
| Permission Combination | All permission sources | Apply precedence and deny-overrides | Final permission set |
| Decision Generation | Permissions + Requested action | Determine allow/deny with audit info | AuthzDecision |

The evaluation pipeline implements **defense in depth** by checking multiple permission sources and applying the principle of least privilege at each stage.

### Common Pitfalls in Resource-Based Access

⚠️ **Pitfall: Resource Tenant Mismatch**
When resources reference other resources (like a document linking to an image), ensure all referenced resources belong to the same tenant. Cross-resource references that span tenants can leak information about resource existence in other tenants. Validate tenant consistency for all resource relationships.

⚠️ **Pitfall: Ownership Transfer Security**
When transferring resource ownership, the old owner loses all control including the ability to revoke shares they previously granted. Implement ownership transfer with a confirmation step that warns about losing sharing control, and consider requiring explicit re-approval of existing shares after ownership transfer.

⚠️ **Pitfall: Bulk Operation Tenant Violations**
Bulk operations (like moving multiple resources to a new folder) can accidentally cross tenant boundaries if the resources being moved belong to different tenants. Validate tenant consistency for all resources in bulk operations and reject operations that would create cross-tenant relationships.

⚠️ **Pitfall: Share Link Tenant Scope**
Share links that allow anonymous access must still respect tenant boundaries. A share link created in Tenant A should never grant access to resources in Tenant B, even if the user later gains membership in Tenant B. Bind share links to their originating tenant context.

### Implementation Guidance

The resource-based access and multi-tenancy system requires careful implementation to maintain security boundaries while providing flexible resource sharing. This implementation guidance focuses on the Go language patterns that support secure multi-tenant architectures.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Tenant Context | Context values + middleware | Custom context type with compile-time checking |
| Resource Storage | PostgreSQL with tenant_id columns | Separate databases per tenant |
| Permission Caching | Redis with tenant-prefixed keys | Dedicated cache cluster with tenant isolation |
| Hierarchical Queries | Recursive CTEs in SQL | Specialized graph database |
| Cross-Tenant Audit | Structured logging to files | Dedicated audit database with write-only access |

#### Recommended File Structure

```
internal/
  resource/
    resource.go              ← Resource entity and basic operations
    ownership.go             ← Ownership and sharing logic
    hierarchy.go             ← Hierarchical permission evaluation
    tenant_isolation.go      ← Tenant boundary enforcement
    cross_tenant.go          ← Cross-tenant sharing implementation
    resource_test.go         ← Resource access tests
  middleware/
    tenant_context.go        ← Tenant context propagation middleware
    tenant_isolation.go      ← Request-level tenant validation
  storage/
    resource_store.go        ← Resource persistence with tenant filtering
    tenant_store.go          ← Tenant membership and role storage
pkg/
  authorization/
    resource_evaluator.go    ← Resource-based authorization logic
    tenant_validator.go      ← Tenant isolation validation
```

#### Infrastructure Starter Code

**Tenant Context Propagation**:
```go
package middleware

import (
    "context"
    "net/http"
    "strings"
)

type TenantContext struct {
    TenantID     string
    UserID       string
    Memberships  []TenantMembership
    CrossTenant  bool
}

type tenantContextKey struct{}

func WithTenantContext(ctx context.Context, tc *TenantContext) context.Context {
    return context.WithValue(ctx, tenantContextKey{}, tc)
}

func GetTenantContext(ctx context.Context) (*TenantContext, bool) {
    tc, ok := ctx.Value(tenantContextKey{}).(*TenantContext)
    return tc, ok
}

// TenantIsolationMiddleware extracts tenant context and validates membership
func TenantIsolationMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Extract tenant ID from path, header, or subdomain
        tenantID := extractTenantID(r)
        if tenantID == "" {
            http.Error(w, "Tenant context required", http.StatusBadRequest)
            return
        }

        // Get user context (from previous auth middleware)
        userID := r.Header.Get("X-User-ID")
        if userID == "" {
            http.Error(w, "User authentication required", http.StatusUnauthorized)
            return
        }

        // Validate tenant membership
        memberships, err := validateTenantMembership(r.Context(), userID, tenantID)
        if err != nil {
            http.Error(w, "Tenant access denied", http.StatusForbidden)
            return
        }

        tc := &TenantContext{
            TenantID:    tenantID,
            UserID:      userID,
            Memberships: memberships,
        }

        ctx := WithTenantContext(r.Context(), tc)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

func extractTenantID(r *http.Request) string {
    // Priority: header > path parameter > subdomain
    if tenantID := r.Header.Get("X-Tenant-ID"); tenantID != "" {
        return tenantID
    }
    
    if tenantID := r.URL.Query().Get("tenant"); tenantID != "" {
        return tenantID
    }
    
    // Extract from subdomain (tenant.domain.com)
    host := r.Host
    if idx := strings.Index(host, "."); idx > 0 {
        return host[:idx]
    }
    
    return ""
}
```

**Resource Storage with Tenant Filtering**:
```go
package storage

import (
    "context"
    "database/sql"
    "fmt"
)

type ResourceStore struct {
    db *sql.DB
}

func NewResourceStore(db *sql.DB) *ResourceStore {
    return &ResourceStore{db: db}
}

// GetResource retrieves a resource with automatic tenant filtering
func (rs *ResourceStore) GetResource(ctx context.Context, resourceID string) (*Resource, error) {
    tc, ok := GetTenantContext(ctx)
    if !ok {
        return nil, fmt.Errorf("tenant context required")
    }

    query := `
        SELECT id, type, tenant_id, owner_id, attributes, created_at, updated_at
        FROM resources 
        WHERE id = $1 AND tenant_id = $2`
    
    var r Resource
    err := rs.db.QueryRowContext(ctx, query, resourceID, tc.TenantID).Scan(
        &r.ID, &r.Type, &r.TenantID, &r.OwnerID, &r.Attributes, &r.CreatedAt, &r.UpdatedAt,
    )
    
    if err == sql.ErrNoRows {
        return nil, fmt.Errorf("resource not found")
    }
    if err != nil {
        return nil, fmt.Errorf("database error: %w", err)
    }
    
    return &r, nil
}

// CreateResource creates a resource within the current tenant context
func (rs *ResourceStore) CreateResource(ctx context.Context, resource *Resource) error {
    tc, ok := GetTenantContext(ctx)
    if !ok {
        return fmt.Errorf("tenant context required")
    }

    // Enforce tenant isolation - resource must belong to current tenant
    if resource.TenantID != tc.TenantID {
        return fmt.Errorf("resource tenant mismatch")
    }

    query := `
        INSERT INTO resources (id, type, tenant_id, owner_id, attributes, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, NOW(), NOW())`
    
    _, err := rs.db.ExecContext(ctx, query, 
        resource.ID, resource.Type, resource.TenantID, resource.OwnerID, resource.Attributes)
    
    return err
}
```

#### Core Logic Skeleton Code

**Resource-Based Authorization Evaluator**:
```go
package authorization

import (
    "context"
    "time"
)

// ResourceEvaluator handles resource-based authorization decisions
type ResourceEvaluator struct {
    resourceStore   ResourceStore
    hierarchyEngine HierarchyEngine
    shareStore      ShareStore
}

// EvaluateResourceAccess determines if a user can perform an action on a resource
func (re *ResourceEvaluator) EvaluateResourceAccess(ctx context.Context, req *AuthzRequest) (*AuthzDecision, error) {
    start := time.Now()
    
    // TODO 1: Validate tenant context and extract tenant information
    // Use GetTenantContext(ctx) to get tenant context
    // Return error if no tenant context or user not member of tenant
    
    // TODO 2: Load resource and validate it belongs to the request tenant
    // Use re.resourceStore.GetResource to load the resource
    // Verify resource.TenantID matches tenant context
    
    // TODO 3: Check direct ownership
    // If req.UserID == resource.OwnerID, grant full access immediately
    // Return AuthzDecision with Allowed=true, Method=EvaluationOwnership
    
    // TODO 4: Check explicit sharing grants
    // Use re.shareStore.GetSharesForResource to get active shares
    // Check if user has explicit grants for the requested action
    // Verify share hasn't expired and hasn't exceeded usage limits
    
    // TODO 5: Evaluate hierarchical permissions
    // Use re.hierarchyEngine.EvaluateHierarchy to check parent permissions
    // Apply inheritance rules based on resource hierarchy configuration
    // Combine inherited permissions with direct grants
    
    // TODO 6: Check cross-tenant grants if tenant mismatch detected
    // If user tenant != resource tenant, look for cross-tenant grants
    // Verify grants are still valid and haven't expired
    // Log cross-tenant access for enhanced auditing
    
    // TODO 7: Apply deny-overrides policy combining
    // If any evaluation stage returns explicit deny, deny the request
    // Otherwise, allow if any stage granted the requested permission
    
    // TODO 8: Build authorization decision with provenance
    // Create AuthzDecision with detailed reasoning and applied policies
    // Include evaluation time and cache hit status
    // Set appropriate cache TTL based on permission types involved
    
    return &AuthzDecision{
        Allowed:        false, // Replace with actual evaluation result
        Reason:         "",    // Set based on evaluation outcome
        Method:         EvaluationRBAC, // Set based on which method granted access
        AppliedPolicies: nil,  // List policies that influenced the decision
        EvaluationTime:  time.Since(start),
        CacheHit:       false,
    }, nil
}

// ValidateTenantAccess checks if user can access resources in the specified tenant
func (re *ResourceEvaluator) ValidateTenantAccess(ctx context.Context, userID, tenantID string) (bool, error) {
    // TODO 1: Load user's tenant memberships
    // Query user's tenant membership records
    // Check if user has active membership in specified tenant
    
    // TODO 2: Validate membership status
    // Ensure membership status is MembershipActive
    // Check membership hasn't expired if time-limited
    
    // TODO 3: Handle cross-tenant admin scenarios
    // Check if user has global admin role allowing cross-tenant access
    // Verify admin access is for legitimate purposes (support, compliance)
    // Log cross-tenant admin access for audit trail
    
    return false, nil // Replace with actual validation result
}

// EvaluateHierarchicalPermissions walks resource hierarchy to determine inherited permissions
func (re *ResourceEvaluator) EvaluateHierarchicalPermissions(ctx context.Context, resource *Resource, userID string, action string) (bool, error) {
    // TODO 1: Build resource hierarchy path
    // Starting from target resource, walk up to root parent
    // Collect all resources in the hierarchy path
    // Validate no cycles exist in the hierarchy
    
    // TODO 2: Apply inheritance rules at each level
    // For each resource in path, check inheritance configuration
    // Apply strong/weak/no inheritance rules appropriately
    // Accumulate permissions from parent to child
    
    // TODO 3: Check for permission overrides
    // Look for explicit deny rules that override inherited allows
    // Handle override inheritance where child blocks parent permissions
    
    // TODO 4: Validate tenant consistency across hierarchy
    // Ensure all resources in hierarchy belong to same tenant
    // Log security violation if cross-tenant hierarchy detected
    
    // TODO 5: Combine inherited and direct permissions
    // Merge permissions from hierarchy with direct resource grants
    // Apply precedence rules (direct > inherited > role-based)
    
    return false, nil // Replace with actual evaluation result
}
```

**Cross-Tenant Sharing Manager**:
```go
package authorization

import (
    "context"
    "fmt"
    "time"
)

// CrossTenantShareManager handles cross-tenant resource sharing
type CrossTenantShareManager struct {
    shareStore  ShareStore
    auditLogger AuditLogger
}

// CreateCrossTenantShare creates a new cross-tenant sharing grant
func (ctsm *CrossTenantShareManager) CreateCrossTenantShare(ctx context.Context, grant *ShareGrant) error {
    // TODO 1: Validate cross-tenant sharing authority
    // Check if requesting user has permission to create cross-tenant shares
    // Verify business justification is provided and meets requirements
    
    // TODO 2: Validate target tenant and user existence
    // Confirm target tenant exists and can receive cross-tenant grants
    // Verify target user exists and has active membership in target tenant
    
    // TODO 3: Apply mandatory expiration and usage limits
    // Set expiration date if not provided (default to policy maximum)
    // Apply usage limits based on grant scope and permissions
    // Schedule automatic review dates for high-privilege grants
    
    // TODO 4: Create approval workflow for high-privilege grants
    // Route grants exceeding threshold permissions through approval process
    // Send notifications to tenant admins for cross-tenant grant approval
    
    // TODO 5: Store grant with enhanced audit information
    // Save grant with complete provenance and justification
    // Include business context and approval chain in audit trail
    
    // TODO 6: Log cross-tenant grant creation
    // Create audit event with enhanced security context
    // Include business justification and authorization chain
    
    return fmt.Errorf("not implemented")
}

// ValidateCrossTenantAccess checks if cross-tenant access is authorized
func (ctsm *CrossTenantShareManager) ValidateCrossTenantAccess(ctx context.Context, userID, sourceTenant, targetTenant, resourceID, action string) (bool, *ShareGrant, error) {
    // TODO 1: Load active cross-tenant grants
    // Query grants where user has access to target tenant resources
    // Filter to grants that cover the requested resource and action
    
    // TODO 2: Validate grant expiration and usage limits
    // Check grant hasn't expired or exceeded maximum usage count
    // Verify grant is still within valid date range
    
    // TODO 3: Update usage tracking
    // Increment usage counter for successful access
    // Update last access timestamp for audit trail
    
    // TODO 4: Check for grant revocation
    // Ensure grant hasn't been explicitly revoked
    // Verify granting user still has authority to maintain this grant
    
    // TODO 5: Log cross-tenant access
    // Create detailed audit log entry for cross-tenant access
    // Include business context and grant details for compliance
    
    return false, nil, fmt.Errorf("not implemented")
}
```

#### Language-Specific Hints

- **Context Propagation**: Use Go's `context.Context` to propagate tenant information through request chains. Store tenant context using typed context keys to avoid collisions.

- **Database Isolation**: Use PostgreSQL row-level security (RLS) policies to enforce tenant isolation at the database level. Create policies like `CREATE POLICY tenant_isolation ON resources FOR ALL TO app_role USING (tenant_id = current_setting('app.current_tenant'))`.

- **Concurrent Access**: Use `sync.RWMutex` for resource hierarchy caches that need frequent reads but infrequent updates. Consider using `sync.Map` for tenant-specific permission caches.

- **Time Handling**: Use `time.Time` with UTC timezone for all timestamps. Handle time comparisons carefully when checking grant expiration with `time.Now().After(grant.ExpiresAt)`.

- **Error Wrapping**: Use `fmt.Errorf("context: %w", err)` to wrap errors with context while preserving error types for tenant isolation violations.

#### Milestone Checkpoint

After implementing resource-based access and multi-tenancy:

**Verification Commands**:
```bash
# Test tenant isolation
go test -v ./internal/resource/ -run TestTenantIsolation

# Test hierarchical permissions
go test -v ./internal/resource/ -run TestHierarchicalAccess

# Test cross-tenant sharing
go test -v ./internal/resource/ -run TestCrossTenantSharing
```

**Expected Behavior**:
- Users can only access resources within their tenant memberships
- Resource owners have full control over their resources
- Hierarchical permissions flow according to inheritance rules
- Cross-tenant shares work only with explicit grants
- All access decisions are logged with tenant context

**Manual Testing Scenarios**:
1. Create resources in different tenants and verify isolation
2. Test resource sharing within tenant and cross-tenant
3. Verify hierarchy permission inheritance works correctly
4. Confirm cross-tenant access requires explicit grants
5. Check that tenant membership changes affect resource access immediately

**Signs of Implementation Issues**:
- Users can access resources from other tenants (tenant isolation failure)
- Resource hierarchy permissions don't inherit properly (hierarchy evaluation bug)
- Cross-tenant grants don't expire or can't be revoked (sharing lifecycle issue)
- Audit logs missing tenant context or cross-tenant access details (logging gap)


## Audit Logging and Policy Testing

> **Milestone(s):** Milestone 4 (Audit Logging & Policy Testing) - this section implements comprehensive audit trails and policy simulation for compliance and testing

Authorization systems are not just about making access decisions in the present—they're about creating an immutable historical record of those decisions for compliance, security analysis, and continuous improvement. Think of audit logging as the "black box recorder" of your authorization system. Just as airplane black boxes help investigators understand what happened during an incident, authorization audit logs help security teams understand who accessed what, when, why the decision was made, and whether the system is behaving correctly.

But unlike airplane black boxes that only matter after something goes wrong, authorization audit logs serve multiple ongoing purposes: compliance reporting for regulations like SOX and GDPR, security incident investigation, access pattern analysis for optimization, and most importantly, providing the confidence to make policy changes without fear of unintended consequences.

This creates a dual challenge: we need both comprehensive audit trails that capture every authorization decision with sufficient context for later analysis, and we need a safe testing environment where policy changes can be validated before they affect real users. The testing challenge is particularly complex because authorization policies have emergent behavior—the interaction between multiple policies, role hierarchies, and resource relationships can produce unexpected results that are only apparent when tested against real-world scenarios.

### Audit Trail Mental Model: Understanding audit logs as immutable decision history for compliance

Think of authorization audit logs like a court stenographer's transcript. The stenographer records not just what people said, but who said it, when they said it, and the context in which it was said. Most importantly, the transcript is immutable—once recorded, it cannot be changed, only supplemented with corrections or clarifications that are themselves part of the record.

Authorization audit logs serve the same function for access control decisions. Every time the system makes an authorization decision, we record not just the outcome (allowed/denied) but the complete context: who made the request, what resource they wanted to access, what action they wanted to perform, which policies were evaluated, why the decision was made, and when it happened. This creates an immutable chain of evidence that can be used for compliance reporting, security investigation, and system analysis.

The "immutable" aspect is crucial for compliance and legal requirements. Many regulations require that audit logs be tamper-evident, meaning any attempt to modify or delete audit records must itself be detectable. This is why audit logs are often stored in append-only systems with cryptographic integrity verification—like a blockchain where each new audit record includes a cryptographic hash of all previous records, making any tampering immediately detectable.

But audit logs are not just about compliance—they're also about system understanding. Good audit logs should tell the story of how your authorization system behaves in practice. They should reveal patterns like "users in the Marketing department frequently try to access Engineering resources" or "90% of cross-tenant access requests are denied due to missing business justification." These patterns inform both security improvements and user experience enhancements.

The challenge is balancing completeness with performance. Comprehensive audit logging means capturing every authorization decision, which can generate enormous volumes of data in high-traffic systems. But incomplete audit logs defeat the purpose—you can't investigate a security incident if the relevant decisions weren't logged. This requires careful design of both what to log and how to log it efficiently without impacting authorization performance.

> **Design Principle: Audit Completeness**
> Every authorization decision must be logged before the decision is returned to the caller. A failed audit write should fail the authorization request rather than allowing unaudited access.

### Access Decision Logging: Immutable audit trail with decision provenance and integrity verification

Access decision logging captures the complete story of every authorization decision made by the system. This goes beyond simply recording "allowed" or "denied"—it captures the decision provenance, meaning the complete chain of reasoning that led to the final decision, including which policies were evaluated, which roles were considered, what attributes were examined, and how conflicts were resolved.

The core audit record structure captures both the request context and the decision details. Every audit record includes a unique identifier that can be used to correlate related events, a precise timestamp for chronological ordering, and the complete authorization request that triggered the decision. The request context includes not just the user and resource, but also environmental attributes like source IP address, user agent, and any additional context attributes that were considered during policy evaluation.

| Field | Type | Description |
|-------|------|-------------|
| AuditID | string | Globally unique identifier for this audit record |
| Timestamp | time.Time | Precise timestamp when decision was made (microsecond precision) |
| RequestID | string | Correlation ID linking this audit to the original request |
| UserID | string | User who made the authorization request |
| TenantID | string | Tenant context in which request was made |
| Action | string | Action the user attempted to perform |
| ResourceID | string | Specific resource that was accessed |
| ResourceType | string | Type of resource (document, project, etc.) |
| Decision | bool | Whether access was allowed or denied |
| EvaluationMethod | string | Primary method used to make decision (RBAC, ABAC, ownership) |
| AppliedPolicies | []string | List of all policies that contributed to decision |
| DenyingPolicy | string | Specific policy that caused denial (if denied) |
| UserRoles | []string | Roles assigned to user at time of request |
| EvaluatedAttributes | map[string]interface{} | All attributes examined during evaluation |
| EvaluationTime | time.Duration | Time taken to compute authorization decision |
| CacheHit | bool | Whether decision was served from cache |
| SourceIP | string | IP address of request origin |
| UserAgent | string | User agent string from request |
| SessionID | string | User session identifier |
| CrossTenant | bool | Whether this was a cross-tenant access request |
| BusinessJustification | string | Business justification provided for access |
| IntegrityHash | string | Cryptographic hash for tamper detection |
| PreviousHash | string | Hash of previous audit record for chain integrity |

The decision provenance captures not just what decision was made, but why it was made. This includes the complete policy evaluation trace showing which conditions were evaluated, what their results were, and how multiple policy decisions were combined. For RBAC decisions, this includes which roles were considered and which permissions were found. For ABAC decisions, this includes which attributes were resolved and how each condition was evaluated. For resource ownership decisions, this includes ownership verification and any inherited permissions that were considered.

> **Decision: Comprehensive Decision Provenance**
> - **Context**: Compliance requirements demand that we can explain any access decision made by the system, often months after it occurred
> - **Options Considered**: 
>   1. Log only final decision (allowed/denied)
>   2. Log decision plus primary reason
>   3. Log complete evaluation trace with all intermediate steps
> - **Decision**: Log complete evaluation trace with all intermediate steps
> - **Rationale**: Compliance investigations often need to understand not just what decision was made, but why alternative decisions were not made. Complete traces enable this analysis.
> - **Consequences**: Higher storage requirements and more complex audit record structure, but enables comprehensive compliance reporting and security analysis.

| Provenance Component | Information Captured | Example Value |
|---------------------|---------------------|---------------|
| Policy Evaluation | All policies matched, conditions evaluated, results | `["policy-marketing-read": true, "policy-cross-tenant-deny": false]` |
| Role Resolution | User roles at decision time, inherited permissions | `["marketing-analyst", "temp-contractor"]` |
| Attribute Resolution | All attributes accessed, their values, resolution source | `{"user.department": "marketing", "resource.classification": "public"}` |
| Conflict Resolution | How multiple policy results were combined | `"deny-overrides: policy-cross-tenant-deny caused final denial"` |
| Performance Metrics | Evaluation time breakdown by component | `{"rbac": "2ms", "abac": "15ms", "total": "17ms"}` |

The immutability of audit records is enforced through cryptographic integrity verification using a hash chain structure similar to blockchain technology. Each audit record includes a cryptographic hash of its own content plus the hash of the previous audit record, creating a chain where any modification of past records would break the chain integrity. This provides mathematical proof that audit logs have not been tampered with.

The integrity verification system maintains a continuous hash chain across all audit records. When a new audit record is created, the system computes a cryptographic hash of the record content using a secure hash function like SHA-256. This hash is then combined with the hash of the immediately previous audit record to create a chain hash that links all records together. Any attempt to modify a past record would require recomputing all subsequent hashes, which is computationally infeasible and would be immediately detectable.

| Integrity Component | Purpose | Implementation |
|---------------------|---------|----------------|
| Content Hash | Ensures record content hasn't been modified | SHA-256 of all field values |
| Chain Hash | Links records together to prevent insertion/deletion | SHA-256(current_content + previous_chain_hash) |
| Merkle Tree Root | Provides efficient integrity verification for large batches | Computed daily over all records in 24-hour period |
| Digital Signature | Provides non-repudiation of audit system authenticity | RSA signature of Merkle root using system private key |

Performance considerations are critical for audit logging since it occurs on the critical path of every authorization decision. The audit system must not significantly impact authorization latency, which requires asynchronous audit writing with careful attention to reliability. The system uses a write-ahead log pattern where audit records are first written to a fast, local buffer, then asynchronously flushed to permanent storage with integrity verification.

The audit pipeline uses multiple stages to balance performance with reliability. Authorization decisions are logged synchronously to an in-memory buffer with persistence guarantees, then asynchronously transferred to long-term storage with full integrity verification. This ensures that audit logging doesn't block authorization decisions while maintaining the guarantee that all decisions are captured.

> **Critical Implementation Note**
> Authorization requests must wait for audit records to be written to the persistent buffer before returning a decision. This ensures that authorized access is never granted without a corresponding audit trail, even if the system crashes immediately after the authorization decision.

⚠️ **Pitfall: Audit Logging Failures**
A common mistake is treating audit logging as "best effort" where authorization decisions can succeed even if audit logging fails. This violates compliance requirements and creates security blind spots. Instead, audit logging failures should cause authorization denial with appropriate error responses. The system should fail secure rather than create unaudited access.

### Policy Testing Framework: Sandbox environment for testing policy changes before production

Policy testing is like having a flight simulator for your authorization system. Just as pilots use simulators to practice emergency procedures without risking real passengers, authorization administrators need a safe environment to test policy changes without risking real user access or creating security vulnerabilities. The policy testing framework provides this capability through snapshot-based simulation that replays real authorization scenarios against modified policies.

The fundamental challenge of policy testing is that authorization policies have complex emergent behavior. A simple change to one policy can have unexpected effects when combined with other policies, role hierarchies, and resource relationships. For example, granting a new permission to a parent role might inadvertently give access to sensitive resources through unexpected inheritance paths. Or modifying an ABAC policy condition might interact with other policies in ways that were not anticipated during the change design.

The testing framework addresses this through snapshot-based simulation. The system captures complete snapshots of the authorization state including all users, roles, policies, resources, and recent authorization requests. This snapshot can then be loaded into a testing environment where policy modifications can be applied and their effects analyzed by replaying historical authorization requests and running synthetic test scenarios.

The snapshot capture process creates a point-in-time copy of all authorization-relevant data. This includes not just the policy definitions themselves, but also the complete context needed to evaluate those policies: user role assignments, resource ownership relationships, cross-tenant sharing grants, and attribute values. The snapshot also includes a representative sample of recent authorization requests that can be replayed to test how policy changes would have affected real user interactions.

| Snapshot Component | Data Captured | Purpose |
|-------------------|---------------|---------|
| Policy Definitions | All RBAC and ABAC policies with versions | Test policy modifications |
| Role Hierarchy | Complete role inheritance structure | Test role-based changes |
| User Assignments | User-to-role mappings with validity periods | Test user impact |
| Resource State | Resource ownership, sharing, attributes | Test resource access changes |
| Request History | Sample of recent authorization requests | Replay testing scenarios |
| Attribute Values | Current user, resource, environment attributes | Ensure realistic evaluation context |
| System Configuration | Policy combining rules, defaults, timeouts | Maintain evaluation consistency |

> **Decision: Snapshot-Based Testing vs. Live Shadowing**
> - **Context**: Need to test policy changes safely without affecting production decisions
> - **Options Considered**:
>   1. Live shadowing: Run new policies alongside production in real-time
>   2. Snapshot-based testing: Capture state and replay in isolated environment
>   3. Synthetic testing only: Generate test scenarios without real data
> - **Decision**: Snapshot-based testing with synthetic scenario support
> - **Rationale**: Live shadowing risks performance impact and data exposure, while synthetic-only testing misses real-world complexity. Snapshots provide safety with realism.
> - **Consequences**: Requires snapshot storage and management, but provides safe, realistic testing environment.

The simulation engine loads snapshots into an isolated testing environment that mirrors the production authorization system architecture. This testing environment includes all the same components—Role Engine, Policy Engine, Resource Manager—but operates against the snapshot data rather than live production data. Policy administrators can then modify policies within this environment and observe the effects through various testing mechanisms.

The testing workflow follows a structured process designed to catch both obvious and subtle policy interactions. The process begins with loading a baseline snapshot and establishing baseline behavior by running a comprehensive test suite against the current policies. Policy modifications are then applied incrementally, with the test suite run after each change to identify the specific impact of each modification.

1. **Snapshot Loading**: The testing environment loads a complete snapshot including all policies, roles, users, and recent request history
2. **Baseline Establishment**: The system runs the complete test suite against unmodified policies to establish expected behavior
3. **Incremental Modification**: Policy changes are applied one at a time to isolate the impact of each change
4. **Impact Analysis**: After each modification, the system runs both regression tests and new scenario tests
5. **Differential Reporting**: The system generates detailed reports showing exactly what changed between baseline and modified policy behavior
6. **Rollback Testing**: The system verifies that policy changes can be safely reverted if problems are discovered
7. **Deployment Validation**: Before applying changes to production, the system validates that the modified policies can be successfully loaded and deployed

The differential analysis component compares authorization decisions between baseline and modified policies to identify exactly what changes. This analysis goes beyond simple allowed/denied comparisons to examine changes in decision reasoning, performance characteristics, and edge case behavior. The system generates detailed reports showing which users would gain or lose access to which resources, and why those changes would occur.

| Analysis Dimension | Comparison Performed | Report Generated |
|-------------------|---------------------|------------------|
| Access Changes | Which decisions changed from allow to deny or vice versa | User impact summary with affected resources |
| Reasoning Changes | How decision logic changed even when final outcome is same | Policy evaluation trace differences |
| Performance Impact | Changes in evaluation time and resource usage | Performance regression analysis |
| Coverage Changes | Which policies/rules are no longer being exercised | Dead policy detection |
| Edge Case Behavior | How boundary conditions and error cases are handled | Edge case impact assessment |
| Compliance Impact | Changes affecting audit trails and compliance reporting | Compliance requirement validation |

The synthetic scenario generation capability supplements snapshot replay with targeted test cases designed to exercise specific policy behaviors. This is particularly important for testing edge cases that might not appear in the historical request sample, such as unusual attribute combinations, resource hierarchy edge cases, or cross-tenant scenarios that occur infrequently.

⚠️ **Pitfall: Incomplete Test Coverage**
A common mistake is testing only the "happy path" scenarios where policies work as expected. Authorization systems must be tested for edge cases, error conditions, and malicious attempts to circumvent policies. The test framework should include scenarios for invalid attributes, missing data, conflicting policies, and attempts to exploit inheritance or sharing mechanisms.

⚠️ **Pitfall: Stale Snapshot Data**
Testing against outdated snapshots can miss critical interactions with recent changes to users, roles, or resources. Snapshots should be refreshed regularly and the testing process should validate that snapshot data is sufficiently recent to be representative of current system state.

### Compliance and Analytics: Generating audit reports and access pattern analysis

Compliance reporting transforms raw audit logs into structured reports that demonstrate adherence to regulatory requirements and organizational policies. Think of this as translating the raw "court stenographer transcript" of your audit logs into executive summaries, legal briefs, and compliance certificates that different stakeholders can understand and act upon. Each audience needs different views of the same underlying audit data: executives want high-level risk summaries, compliance officers want detailed regulatory attestations, and security teams want operational metrics and anomaly detection.

The compliance reporting system must address multiple regulatory frameworks simultaneously. SOX compliance requires demonstrating that financial systems have appropriate access controls and that changes are properly authorized and documented. GDPR compliance requires showing that personal data access is limited to authorized users with legitimate business purposes and that data subject access can be tracked and reported. HIPAA compliance requires audit trails showing that healthcare information access follows the minimum necessary principle and that all access is logged for potential investigation.

Each regulatory framework has specific requirements for what must be logged, how long logs must be retained, and what reports must be generated. The system must be flexible enough to generate framework-specific reports from the same underlying audit data while ensuring that the audit logs contain all necessary information for all applicable frameworks.

| Compliance Framework | Key Requirements | Required Reports |
|---------------------|-----------------|------------------|
| SOX (Sarbanes-Oxley) | Financial system access controls, change authorization | User access reviews, privilege escalation reports, segregation of duties |
| GDPR | Personal data access tracking, lawful basis documentation | Data subject access reports, purpose limitation compliance, retention compliance |
| HIPAA | Healthcare data minimum necessary, access logging | Patient data access logs, minimum necessary compliance, breach investigation support |
| SOC 2 | Security control effectiveness, access management | Access review evidence, security incident response, control testing results |
| PCI DSS | Cardholder data protection, access monitoring | Cardholder data access reports, security testing evidence, access control validation |

The automated report generation system produces compliance reports on regular schedules and on-demand for audits or investigations. These reports aggregate audit log data according to compliance framework requirements, applying appropriate filtering, grouping, and statistical analysis to produce meaningful compliance evidence. The reports include both summary metrics and detailed evidence that auditors can drill down into for verification.

Regular compliance reports follow standardized templates that map directly to regulatory requirements. For example, SOX user access reviews require reports showing all users with access to financial systems, when that access was granted, who authorized it, and when it was last reviewed. The system generates these reports automatically by analyzing audit logs for access patterns, role assignments, and administrative actions.

| Report Type | Generation Schedule | Content Included | Primary Audience |
|-------------|-------------------|------------------|------------------|
| User Access Review | Monthly | All user permissions, recent access activity, dormant accounts | Access administrators |
| Privilege Escalation | Weekly | New role assignments, permission changes, administrative actions | Security team |
| Cross-Tenant Activity | Daily | Cross-tenant access requests, sharing grants, business justifications | Compliance officers |
| Data Access Summary | On-demand | Resource access patterns, data classification compliance | Data protection officers |
| Anomaly Detection | Real-time | Unusual access patterns, policy violations, security events | Security operations center |

The access pattern analysis system identifies trends, anomalies, and optimization opportunities in authorization behavior. This goes beyond compliance reporting to provide operational insights that can improve both security and user experience. The analysis identifies patterns like users who frequently request access to resources they don't have permission for (suggesting role definitions might need adjustment) or resources that are accessed by many users but not formally shared (suggesting sharing mechanisms might need improvement).

Pattern analysis uses statistical methods to identify normal and abnormal access behavior. The system establishes baselines for normal access patterns by analyzing historical audit data, then identifies deviations from these patterns that might indicate security issues, policy problems, or user experience difficulties. This analysis is particularly valuable for identifying insider threats, compromised accounts, or policy misconfigurations that might not be apparent from individual authorization decisions.

| Analysis Type | Pattern Detected | Security Implication | Recommended Action |
|---------------|-----------------|---------------------|-------------------|
| Unusual Access Volume | User suddenly accessing many more resources | Possible account compromise | Investigate user activity, consider access suspension |
| Off-Hours Access | Access outside normal business hours | Potential unauthorized access | Review business justification, implement time-based controls |
| Cross-Tenant Anomalies | Unusual patterns in cross-tenant access | Possible data exfiltration | Review sharing grants, investigate business need |
| Permission Escalation | Users gaining significantly more permissions | Privilege creep or role inflation | Review role assignments, implement regular access reviews |
| Denied Access Patterns | High denial rates for specific users/resources | Policy misconfiguration or training need | Analyze policy effectiveness, provide user guidance |

The analytics system also provides policy effectiveness metrics that help administrators understand how well their authorization policies are working in practice. This includes metrics like policy coverage (what percentage of authorization requests are handled by each policy), policy performance (how long different policies take to evaluate), and policy accuracy (how often policies produce the intended access decisions based on business requirements).

> **Decision: Real-Time vs. Batch Analytics Processing**
> - **Context**: Need to balance real-time anomaly detection with efficient large-scale pattern analysis
> - **Options Considered**:
>   1. Real-time only: Process all audit events immediately for instant insights
>   2. Batch only: Process audit data in scheduled batches for efficiency
>   3. Hybrid: Real-time for anomalies, batch for comprehensive analysis
> - **Decision**: Hybrid approach with real-time anomaly detection and batch pattern analysis
> - **Rationale**: Security anomalies require immediate detection, but comprehensive pattern analysis is more efficient in batches and doesn't require real-time response
> - **Consequences**: More complex architecture with two processing paths, but optimal balance of security responsiveness and analytical depth

The anomaly detection system operates in near real-time to identify potential security incidents as they occur. This system uses machine learning techniques to understand normal access patterns for each user, resource, and tenant, then flags deviations that might indicate security threats. The detection system generates alerts with different priority levels based on the severity and confidence of the anomaly detection.

Anomaly detection considers multiple dimensions simultaneously to reduce false positives while maintaining sensitivity to real threats. The system analyzes temporal patterns (when access occurs), behavioral patterns (what resources are accessed), relationship patterns (interactions between users and tenants), and outcome patterns (success and failure rates) to build comprehensive profiles of normal behavior.

| Anomaly Type | Detection Method | False Positive Mitigation | Alert Priority |
|-------------|------------------|-------------------------|---------------|
| Volume Anomaly | Statistical deviation from historical access volume | Consider business context (month-end, deadlines) | Medium |
| Temporal Anomaly | Access outside established time patterns | Account for time zone changes, special projects | Medium |
| Geographic Anomaly | Access from unusual locations | Consider business travel, VPN usage | High |
| Permission Anomaly | Access to resources outside normal scope | Factor in role changes, project assignments | High |
| Failure Anomaly | Unusual patterns of access denials | Distinguish between attacks and misconfigurations | Variable |

⚠️ **Pitfall: Alert Fatigue**
Anomaly detection systems that generate too many false positive alerts quickly become ignored by security teams. The system must be carefully tuned to balance sensitivity with specificity, and alerts should include sufficient context for security analysts to quickly determine whether they represent real threats.

⚠️ **Pitfall: Compliance Interpretation**
Different auditors may interpret compliance requirements differently, and regulatory requirements evolve over time. The compliance reporting system should be flexible enough to accommodate different interpretations and should include detailed evidence that supports multiple possible compliance approaches rather than assuming a single "correct" interpretation.

### Implementation Guidance

The audit logging and policy testing system bridges high-performance authorization with comprehensive compliance and safety requirements. This implementation guidance provides complete, production-ready code for the supporting infrastructure while focusing your implementation effort on the core audit and testing logic.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|----------------|
| Audit Storage | SQLite with WAL mode | PostgreSQL with append-only tables |
| Integrity Verification | SHA-256 hash chains | Merkle tree with periodic commitment |
| Analytics Processing | In-memory aggregation | Apache Kafka + stream processing |
| Report Generation | Template-based HTML/PDF | Business intelligence integration |
| Anomaly Detection | Statistical thresholds | Machine learning with TensorFlow |

**Recommended File Structure:**
```
internal/audit/
  audit.go                    ← main audit logging interface
  logger.go                   ← audit record creation and storage
  integrity.go                ← hash chain and integrity verification
  analytics.go                ← access pattern analysis
  compliance.go               ← compliance report generation
  audit_test.go              ← comprehensive audit system tests
internal/testing/
  framework.go                ← policy testing framework
  snapshot.go                 ← snapshot capture and management
  simulation.go               ← policy simulation engine
  differential.go             ← baseline vs. modified policy analysis
  framework_test.go          ← policy testing validation
internal/reports/
  generator.go                ← compliance report generation
  templates/                  ← report templates by compliance framework
    sox_access_review.html    ← SOX user access review template
    gdpr_data_access.html     ← GDPR data access report template
```

**Complete Audit Infrastructure (ready to use):**

```go
// Package audit provides comprehensive authorization audit logging with integrity verification
package audit

import (
    "context"
    "crypto/sha256"
    "database/sql"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "time"
    
    _ "github.com/lib/pq" // PostgreSQL driver
)

// AuditRecord represents a complete authorization decision audit entry
type AuditRecord struct {
    AuditID             string                 `json:"audit_id"`
    Timestamp           time.Time              `json:"timestamp"`
    RequestID           string                 `json:"request_id"`
    UserID              string                 `json:"user_id"`
    TenantID            string                 `json:"tenant_id"`
    Action              string                 `json:"action"`
    ResourceID          string                 `json:"resource_id"`
    ResourceType        string                 `json:"resource_type"`
    Decision            bool                   `json:"decision"`
    EvaluationMethod    string                 `json:"evaluation_method"`
    AppliedPolicies     []string               `json:"applied_policies"`
    DenyingPolicy       string                 `json:"denying_policy,omitempty"`
    UserRoles           []string               `json:"user_roles"`
    EvaluatedAttributes map[string]interface{} `json:"evaluated_attributes"`
    EvaluationTime      time.Duration          `json:"evaluation_time"`
    CacheHit            bool                   `json:"cache_hit"`
    SourceIP            string                 `json:"source_ip"`
    UserAgent           string                 `json:"user_agent"`
    SessionID           string                 `json:"session_id"`
    CrossTenant         bool                   `json:"cross_tenant"`
    BusinessJustification string               `json:"business_justification,omitempty"`
    IntegrityHash       string                 `json:"integrity_hash"`
    PreviousHash        string                 `json:"previous_hash"`
}

// AuditLogger provides immutable audit logging with integrity verification
type AuditLogger struct {
    db           *sql.DB
    lastHash     string
    hashChan     chan string
}

// NewAuditLogger creates a new audit logging system with integrity verification
func NewAuditLogger(dbURL string) (*AuditLogger, error) {
    db, err := sql.Open("postgres", dbURL)
    if err != nil {
        return nil, fmt.Errorf("failed to open audit database: %w", err)
    }
    
    if err := createAuditTables(db); err != nil {
        return nil, fmt.Errorf("failed to create audit tables: %w", err)
    }
    
    logger := &AuditLogger{
        db:       db,
        hashChan: make(chan string, 1),
    }
    
    // Initialize hash chain from last audit record
    if err := logger.initializeHashChain(); err != nil {
        return nil, fmt.Errorf("failed to initialize hash chain: %w", err)
    }
    
    return logger, nil
}

// LogAuthzDecision records an authorization decision with complete context
func (al *AuditLogger) LogAuthzDecision(ctx context.Context, req *AuthzRequest, decision *AuthzDecision) error {
    record := &AuditRecord{
        AuditID:             generateAuditID(),
        Timestamp:           time.Now().UTC(),
        RequestID:           req.RequestID,
        UserID:              req.UserID,
        TenantID:            req.TenantID,
        Action:              req.Action,
        ResourceID:          req.Resource.ID,
        ResourceType:        req.Resource.Type,
        Decision:            decision.Allowed,
        EvaluationMethod:    string(decision.Method),
        AppliedPolicies:     decision.AppliedPolicies,
        UserRoles:           getUserRoles(ctx, req.UserID, req.TenantID),
        EvaluatedAttributes: req.Attributes,
        EvaluationTime:      decision.EvaluationTime,
        CacheHit:            decision.CacheHit,
        SourceIP:            getSourceIP(ctx),
        UserAgent:           getUserAgent(ctx),
        SessionID:           getSessionID(ctx),
        CrossTenant:         req.TenantID != req.Resource.TenantID,
        PreviousHash:        al.lastHash,
    }
    
    // Compute integrity hash
    record.IntegrityHash = al.computeIntegrityHash(record)
    
    // Store record with integrity verification
    if err := al.storeAuditRecord(ctx, record); err != nil {
        return fmt.Errorf("failed to store audit record: %w", err)
    }
    
    // Update hash chain
    al.lastHash = record.IntegrityHash
    
    return nil
}

func (al *AuditLogger) computeIntegrityHash(record *AuditRecord) string {
    // Create deterministic hash of record content
    content, _ := json.Marshal(record)
    hash := sha256.Sum256(append(content, []byte(record.PreviousHash)...))
    return hex.EncodeToString(hash[:])
}

// Policy testing framework infrastructure
type PolicyTestFramework struct {
    authorizer   Authorizer
    snapshotDB   *sql.DB
    testResults  map[string]*TestResult
}

// TestScenario represents a single authorization test case
type TestScenario struct {
    Name        string        `json:"name"`
    Request     *AuthzRequest `json:"request"`
    Expected    *AuthzDecision `json:"expected"`
    Description string        `json:"description"`
}

// TestResult contains the outcome of running a test scenario
type TestResult struct {
    Scenario     *TestScenario `json:"scenario"`
    Actual       *AuthzDecision `json:"actual"`
    Passed       bool          `json:"passed"`
    ErrorMessage string        `json:"error_message,omitempty"`
    ExecutedAt   time.Time     `json:"executed_at"`
}
```

**Core Audit Logic Skeleton (implement these):**

```go
// LogAuthzDecision creates and stores a complete audit record for an authorization decision
// This is called after every authorization check to ensure comprehensive audit coverage
func (al *AuditLogger) LogAuthzDecision(ctx context.Context, req *AuthzRequest, decision *AuthzDecision) error {
    // TODO 1: Generate unique audit ID and capture precise timestamp
    // TODO 2: Extract all context information from request (IP, user agent, session)
    // TODO 3: Gather additional context like user roles and evaluated attributes
    // TODO 4: Compute integrity hash linking to previous audit record
    // TODO 5: Store audit record in append-only storage with immediate persistence
    // TODO 6: Update hash chain state for next audit record
    // TODO 7: Handle storage failures by failing the authorization request
    // Hint: Audit logging failures should cause authorization denial for security
}

// VerifyIntegrityChain validates that audit records have not been tampered with
// This is used during compliance audits to prove log integrity
func (al *AuditLogger) VerifyIntegrityChain(ctx context.Context, startTime, endTime time.Time) error {
    // TODO 1: Retrieve all audit records in specified time range in chronological order
    // TODO 2: Start with expected first hash and verify each subsequent hash
    // TODO 3: For each record, recompute integrity hash and compare to stored value
    // TODO 4: Verify that each record's previous hash matches the prior record's hash
    // TODO 5: Return detailed error if any hash verification fails
    // TODO 6: Generate integrity verification report for compliance evidence
    // Hint: Use constant-time comparison for hash verification to prevent timing attacks
}

// CreateSnapshot captures complete authorization state for testing
// This enables safe policy testing without affecting production
func (ptf *PolicyTestFramework) CreateSnapshot(ctx context.Context, name string) error {
    // TODO 1: Begin database transaction to ensure consistent snapshot
    // TODO 2: Capture all users, roles, and role assignments
    // TODO 3: Capture all policies with their current versions
    // TODO 4: Capture all resources with ownership and sharing state
    // TODO 5: Capture sample of recent authorization requests for replay
    // TODO 6: Store snapshot with metadata (timestamp, description, creator)
    // TODO 7: Commit transaction to finalize consistent snapshot
    // Hint: Use database-level snapshots if available for consistency
}

// RunSimulation executes test scenarios against modified policies
// This validates policy changes before production deployment
func (ptf *PolicyTestFramework) RunSimulation(ctx context.Context, snapshotID string, scenarios []TestScenario) (*SimulationReport, error) {
    // TODO 1: Load specified snapshot into isolated testing environment
    // TODO 2: Initialize test authorization engine with snapshot data
    // TODO 3: Execute each test scenario and capture actual results
    // TODO 4: Compare actual results to expected results for each scenario
    // TODO 5: Generate detailed report showing pass/fail status and differences
    // TODO 6: Include performance metrics and policy coverage analysis
    // TODO 7: Clean up test environment resources
    // Hint: Run scenarios in parallel but ensure test isolation
}

// GenerateComplianceReport creates regulatory compliance reports from audit data
// This transforms raw audit logs into compliance evidence
func (cr *ComplianceReporter) GenerateComplianceReport(ctx context.Context, framework string, startTime, endTime time.Time) (*ComplianceReport, error) {
    // TODO 1: Validate compliance framework and time range parameters
    // TODO 2: Query audit logs for all relevant authorization decisions
    // TODO 3: Apply framework-specific filtering and aggregation rules
    // TODO 4: Generate summary statistics (access patterns, policy usage)
    // TODO 5: Identify compliance violations or unusual patterns
    // TODO 6: Format report according to regulatory requirements
    // TODO 7: Include supporting evidence and detailed audit trails
    // Hint: Use templates for different compliance frameworks (SOX, GDPR, HIPAA)
}

// DetectAnomalies identifies unusual access patterns in real-time
// This provides security monitoring and early threat detection
func (ad *AnomalyDetector) DetectAnomalies(ctx context.Context, record *AuditRecord) []Anomaly {
    // TODO 1: Load user's historical access patterns and normal behavior baseline
    // TODO 2: Compare current access against temporal patterns (time of day, day of week)
    // TODO 3: Check for volume anomalies (accessing many more resources than normal)
    // TODO 4: Detect geographic anomalies (access from unusual locations)
    // TODO 5: Identify permission anomalies (accessing unusual resource types)
    // TODO 6: Score anomalies by severity and confidence level
    // TODO 7: Generate alerts for high-confidence anomalies above threshold
    // Hint: Use sliding windows for baseline calculation to adapt to changing patterns
}
```

**Language-Specific Implementation Hints:**

- Use `database/sql` with PostgreSQL for production audit storage with proper WAL configuration
- Implement hash chains using `crypto/sha256` for integrity verification
- Use `encoding/json` for audit record serialization with consistent field ordering
- Use `time.Now().UTC()` for all timestamps to avoid timezone confusion
- Consider `sync.Mutex` for protecting hash chain state during concurrent audit logging
- Use `context.Context` for request tracing and timeout management in audit operations
- Implement audit record batching for high-throughput scenarios using channels and goroutines

**Milestone Checkpoint:**
After implementing this milestone, verify:
1. Run `go test ./internal/audit/...` - all audit logging tests should pass
2. Create an audit record: authorization decisions should generate complete audit entries
3. Verify integrity: `VerifyIntegrityChain` should detect any tampering with audit records
4. Test policy simulation: create snapshot, modify policies, run test scenarios
5. Generate compliance report: produce SOX user access review from audit data
6. Check anomaly detection: unusual access patterns should trigger appropriate alerts

**Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Audit logging fails silently | Database connection issues or table missing | Check database connectivity and table schema | Implement proper error handling and database initialization |
| Hash chain verification fails | Concurrent audit writes corrupting hash sequence | Check for race conditions in hash chain updates | Add mutex protection around hash chain state |
| Policy tests give inconsistent results | Test environment not properly isolated from production | Verify snapshot isolation and test data cleanup | Ensure complete test environment reset between runs |
| Compliance reports show missing data | Audit logging not capturing all required fields | Review audit record structure against compliance requirements | Add missing fields and regenerate reports |
| Anomaly detection has too many false positives | Baseline calculation doesn't account for business patterns | Analyze false positive patterns and business context | Adjust anomaly thresholds and add business context filtering |


## Interactions and Data Flow

> **Milestone(s):** All milestones - this section demonstrates how components communicate throughout the authorization lifecycle, from role-based permission checks (Milestone 1) through policy evaluation (Milestone 2), resource validation (Milestone 3), to audit logging (Milestone 4)

Authorization systems are like air traffic control towers - they coordinate multiple specialized systems to make split-second decisions based on constantly changing information. Just as air traffic control receives flight plans, weather data, and radar information to decide whether a plane can land, our authorization system receives user context, resource metadata, and policy definitions to decide whether access should be granted. The key difference is that our system must handle thousands of authorization requests per second while maintaining perfect accuracy - there's no room for "near misses" in security decisions.

Understanding how authorization components interact requires thinking about the system as a **decision pipeline** where each stage can allow, deny, or enrich the request with additional context. Unlike a simple function call that returns true or false, authorization involves orchestrating multiple engines, resolving dynamic attributes, checking caches at multiple levels, and ensuring every decision is audited for compliance. This section explores the intricate dance of components that makes authorization decisions both fast and secure.

### Authorization Request Processing

The authorization request flow represents the core orchestration of our system - how a single `IsAuthorized` call triggers a cascade of specialized evaluations across multiple engines. Think of this like a medical diagnosis process: the doctor (main `Authorizer`) coordinates with specialists (Role Engine, Policy Engine, Resource Manager) to gather evidence from different perspectives before making a final decision. Each specialist contributes expertise from their domain, but the primary doctor synthesizes everything into a coherent treatment plan.

![System Component Architecture](./diagrams/system-components.svg)

The authorization flow follows a **fail-safe default** principle - any uncertainty or error results in access denial rather than risking unauthorized access. This creates a multi-stage evaluation pipeline where each stage must explicitly approve the request for it to proceed. The pipeline also implements **short-circuiting** - if any stage issues a definitive deny decision, the remaining stages are skipped to minimize latency.

![Authorization Request Sequence](./diagrams/authorization-flow.svg)

**Stage 1: Request Validation and Context Building**

The authorization process begins with comprehensive request validation and context assembly. This stage acts like a security checkpoint - verifying that the request contains all required information and building a complete picture of the authorization context before expensive evaluations begin.

The `Authorizer` receives an `AuthzRequest` and immediately performs structural validation to ensure all required fields are present and properly formatted. This includes validating that the `UserID` and `TenantID` are non-empty strings, the `Resource` contains valid identifiers, and the `Action` matches expected patterns. The system also generates a unique `RequestID` if not provided, enabling complete request tracing through logs and metrics.

Context building involves resolving the `TenantContext` to understand the user's relationship to the requested tenant. The system queries user membership information to determine if the user belongs to the target tenant, has any cross-tenant grants, or requires special handling for guest access. This context propagation ensures that all downstream components operate with consistent tenant isolation boundaries.

| Validation Check | Purpose | Failure Action |
|-----------------|---------|----------------|
| UserID non-empty | Ensure authenticated user | Return immediate deny with "invalid user" reason |
| TenantID validation | Confirm target tenant exists | Return immediate deny with "unknown tenant" reason |
| Resource structure | Verify resource fields complete | Return immediate deny with "malformed resource" reason |
| Action format | Ensure action matches expected pattern | Return immediate deny with "invalid action" reason |
| Tenant membership | Validate user can access tenant | Proceed to cross-tenant evaluation if needed |

**Stage 2: Cache Lookup and Hit Evaluation**

Before expensive computation, the system performs multi-level cache lookups to find recent authorization decisions for identical or similar requests. Think of this like consulting a doctor's notes from a recent visit - if the patient's condition and symptoms are identical, the previous diagnosis may still apply without repeating all the tests.

The cache lookup uses a composite key combining `UserID`, `TenantID`, `Resource.ID`, `Action`, and relevant attribute hashes. The system maintains separate cache tiers with different expiration policies:

| Cache Tier | Scope | TTL | Hit Rate Target |
|-----------|-------|-----|----------------|
| L1 In-Memory | Identical requests | 5 minutes | >80% for repeated access |
| L2 Redis | User-resource pairs | 30 minutes | >60% for active users |
| L3 Database | Policy decisions | 2 hours | >40% for stable policies |

Cache hits must validate that the cached decision remains valid by checking policy versions, role assignments, and resource metadata timestamps. If any component has been modified since the cached decision, the cache entry is invalidated and fresh evaluation proceeds.

When a valid cache hit is found, the system constructs an `AuthzDecision` with `CacheHit: true` and returns immediately. This dramatically reduces latency for repeated access patterns while ensuring security properties are maintained through cache validation.

**Stage 3: RBAC Permission Evaluation**

The Role Engine evaluation handles hierarchical role resolution and permission inheritance computation. This stage operates like checking an employee's organizational chart and job description - understanding what permissions they inherit through their position in the company hierarchy.

The `RoleEngine.HasPermission` method begins by loading the user's role assignments within the target tenant. For users with multiple roles, the system computes the union of all permissions using bitmap operations for efficiency. Role hierarchy traversal uses the precomputed transitive closure to avoid recursive evaluation during request processing.

| RBAC Evaluation Step | Operation | Performance Optimization |
|---------------------|-----------|-------------------------|
| Load user roles | Query `UserRoleAssignment` table | Batch load with tenant-scoped index |
| Resolve role hierarchy | Follow parent role chains | Use precomputed transitive closure bitmap |
| Compute permission union | Combine permissions from all roles | Bitwise OR operations on `PermissionBitmap` |
| Check specific permission | Test if permission exists in union | Single bit test operation O(1) |
| Handle wildcards | Expand wildcard permissions | Pattern matching against permission registry |

The RBAC evaluation can result in three outcomes: explicit allow (permission found), explicit deny (conflicting negative permission), or no decision (permission not found). The system records which roles contributed to the decision for audit logging and troubleshooting purposes.

**Stage 4: ABAC Policy Evaluation**

When RBAC evaluation doesn't yield a definitive decision, or when attribute-based policies are configured, the Policy Engine performs dynamic policy evaluation. This stage operates like applying complex business rules - checking not just what role the user has, but contextual factors like time of day, resource sensitivity, and environmental conditions.

The `PolicyEngine.Evaluate` method loads all policies that could apply to the request based on resource patterns and action matching. Policy evaluation follows a **deny-overrides** algorithm where any policy that denies access immediately terminates evaluation with a deny decision.

| Policy Evaluation Phase | Purpose | Implementation |
|------------------------|---------|----------------|
| Policy loading | Find applicable policies | Query policies matching resource/action patterns |
| Attribute resolution | Gather evaluation context | Resolve user, resource, environment attributes |
| Condition evaluation | Test policy conditions | Recursive evaluation of condition tree |
| Policy combining | Handle multiple matches | Apply deny-overrides with priority ordering |
| Decision synthesis | Generate final policy decision | Record applied policies and reasoning |

The attribute resolution process queries multiple sources to build the complete `EvaluationContext`. User attributes come from identity providers, resource attributes from the resource metadata, and environment attributes from the request context (IP address, time, device information).

**Stage 5: Resource-Based Authorization**

The Resource Manager evaluation handles ownership-based permissions and hierarchical resource access. This stage operates like checking property ownership and inheritance - determining if the user owns the resource, has been explicitly granted access, or inherits permissions through resource hierarchy.

The `ResourceEvaluator.EvaluateResourceAccess` method first checks direct ownership by comparing the resource's `OwnerID` with the requesting `UserID`. Resource owners automatically receive full permissions unless explicitly restricted by policies. The system then evaluates explicit grants through the `ShareSettings` mechanism.

| Resource Evaluation Check | Purpose | Grant Conditions |
|---------------------------|---------|-----------------|
| Direct ownership | Owner has full control | `Resource.OwnerID == UserID` |
| Explicit sharing | Granted specific permissions | Valid `ShareGrant` with required permission |
| Hierarchical inheritance | Parent resource permissions | Parent allows and inheritance enabled |
| Cross-tenant sharing | Inter-tenant collaboration | Valid cross-tenant grant with approval |
| Time-limited access | Temporary permissions | Grant within valid time window |

For hierarchical resources, the system walks up the resource tree to find inherited permissions. Each level of the hierarchy can grant additional permissions or impose restrictions, with the final decision being the intersection of all applicable permissions.

**Stage 6: Decision Synthesis and Audit Logging**

The final stage synthesizes all evaluation results into a coherent authorization decision and ensures comprehensive audit logging. This stage operates like a judge writing a court decision - documenting not just the outcome, but the complete reasoning process that led to the conclusion.

The `Authorizer` combines results from all engines using a **conservative decision algorithm**:

| Engine Results | Final Decision | Reasoning |
|---------------|----------------|-----------|
| Any engine denies | Deny | Fail-safe: explicit deny overrides all allows |
| All engines allow | Allow | Consensus: all evaluation methods approve |
| Mixed allow/no-decision | Allow | Optimistic: at least one method approves |
| All engines no-decision | Deny | Fail-safe: no explicit approval found |
| Any engine errors | Deny | Fail-safe: uncertainty treated as deny |

The decision synthesis includes performance metadata such as evaluation time, cache hit status, and which engines were consulted. This information proves invaluable for performance optimization and troubleshooting authorization issues.

Audit logging occurs asynchronously to avoid impacting authorization latency, but the audit record generation happens synchronously to capture complete request context before it's lost. The `AuditLogger.LogAuthzDecision` method creates an immutable `AuditRecord` with cryptographic integrity verification.

> **Key Insight**: The authorization pipeline implements **defense in depth** through multiple independent evaluation stages. Even if one engine has a bug or misconfiguration, other engines provide overlapping security coverage. This redundancy is intentional - authorization systems must be paranoid about security while remaining performant for user experience.

### Inter-Component Messaging

Authorization components communicate through well-defined interfaces using structured message passing rather than sharing mutable state. Think of this like a courtroom where different parties (lawyers, witnesses, experts) present evidence through formal procedures rather than having private conversations that could compromise the integrity of the decision-making process. Each component speaks a specific "language" optimized for its domain while maintaining clear contracts with other components.

The messaging architecture follows **immutable message passing** principles where components exchange read-only data structures rather than sharing mutable objects. This eliminates race conditions and makes the system's behavior predictable under concurrent load. Messages carry complete context rather than requiring recipients to perform additional lookups, reducing coupling between components.

**Request Message Structures**

The primary request messages flowing through the authorization system carry comprehensive context to enable stateless evaluation. Each message includes not just the basic authorization parameters, but also metadata for tracing, caching, and audit purposes.

| Message Type | Purpose | Required Fields | Optional Fields |
|-------------|---------|----------------|-----------------|
| `AuthzRequest` | Main authorization query | UserID, TenantID, Action, Resource | Attributes, Timestamp, RequestID |
| `PermissionQuery` | RBAC permission check | UserID, TenantID, Resource, Action | CachePolicy, TraceEnabled |
| `PolicyEvaluationRequest` | ABAC policy evaluation | Request context, Applicable policies | AttributeOverrides, SimulationMode |
| `ResourceAccessQuery` | Resource-based check | Resource, UserID, Action, TenantContext | HierarchyDepth, ShareLookup |
| `AuditLogRequest` | Decision logging | AuthzRequest, AuthzDecision | IntegrityCheck, ComplianceFlags |

The `AuthzRequest` serves as the canonical representation of an authorization question throughout the system. Its design carefully balances completeness with performance - including enough information to make accurate decisions without requiring additional network calls, while remaining lightweight enough for high-frequency usage.

```
AuthzRequest Structure:
- UserID: string (required) - authenticated user identifier
- TenantID: string (required) - target tenant for authorization
- Action: string (required) - operation being attempted
- Resource: *Resource (required) - target resource with full metadata
- Attributes: map[string]any (optional) - additional context attributes
- Timestamp: time.Time (optional) - request time for temporal policies
- RequestID: string (optional) - unique identifier for request tracing
```

**Response Message Structures**

Authorization responses carry not just the binary decision, but rich metadata explaining how the decision was reached. This information serves multiple purposes: enabling intelligent caching, supporting audit requirements, and providing debugging information when authorization behaves unexpectedly.

| Response Type | Purpose | Decision Fields | Metadata Fields |
|--------------|---------|----------------|-----------------|
| `AuthzDecision` | Final authorization result | Allowed, Reason, Method | AppliedPolicies, EvaluationTime, CacheHit |
| `PermissionResult` | RBAC evaluation outcome | HasPermission, EffectiveRoles | InheritedPermissions, HierarchyPath |
| `PolicyDecision` | ABAC evaluation result | Allowed, AppliedPolicies | DenyingPolicy, AttributeValues |
| `ResourceAccessResult` | Resource check outcome | AccessGranted, AccessType | OwnershipStatus, ShareGrants |
| `AuditRecord` | Decision audit trail | All request/response data | IntegrityHash, ComplianceMetadata |

The `AuthzDecision` structure provides transparency into the authorization decision-making process. The `Method` field indicates which evaluation approach was used (RBAC, ABAC, ownership, cached), while `AppliedPolicies` lists specific policies that contributed to the decision. This information proves crucial for troubleshooting unexpected authorization behavior and demonstrating compliance with regulatory requirements.

```
AuthzDecision Structure:
- Allowed: bool (required) - authorization decision
- Reason: string (required) - human-readable explanation
- Method: EvaluationMethod (required) - how decision was reached
- AppliedPolicies: []string (optional) - policies that applied
- EvaluationTime: time.Duration (required) - processing time
- CacheHit: bool (required) - whether result came from cache
```

**Component Interface Contracts**

Each authorization component exposes a focused interface that encapsulates its domain expertise while remaining composable with other components. These interfaces follow the **principle of least knowledge** - each component only exposes the minimum surface area necessary for its role in the authorization pipeline.

| Component | Primary Interface | Input Message | Output Message | Error Conditions |
|-----------|------------------|---------------|----------------|------------------|
| `Authorizer` | Authorization orchestration | `AuthzRequest` | `AuthzDecision` | Invalid request, evaluation timeout |
| `RoleEngine` | RBAC evaluation | `PermissionQuery` | `PermissionResult` | Role hierarchy cycles, missing roles |
| `PolicyEngine` | ABAC evaluation | `PolicyEvaluationRequest` | `PolicyDecision` | Policy syntax errors, attribute resolution failures |
| `ResourceEvaluator` | Resource-based checks | `ResourceAccessQuery` | `ResourceAccessResult` | Missing resources, broken hierarchy |
| `AuditLogger` | Decision logging | `AuditLogRequest` | Log confirmation | Storage failures, integrity violations |

The interface contracts specify not just method signatures, but also behavioral guarantees such as idempotency, timeout handling, and error recovery. For example, the `RoleEngine.HasPermission` method guarantees that it will return within 100ms or return an error, and that calling it multiple times with identical parameters will yield identical results.

**Message Routing and Error Propagation**

The authorization system implements **structured error propagation** where errors carry enough context to enable intelligent retry logic and meaningful user feedback. Rather than generic "access denied" messages, the system provides specific error codes and remediation hints.

| Error Category | HTTP Status | Error Code | User Message | Admin Context |
|---------------|-------------|------------|--------------|---------------|
| Invalid Request | 400 | `INVALID_REQUEST` | "Request missing required fields" | Field validation details |
| Unauthorized User | 401 | `USER_NOT_AUTHENTICATED` | "Please log in to continue" | Authentication provider info |
| Forbidden Access | 403 | `ACCESS_DENIED` | "Insufficient permissions for this resource" | Required vs. actual permissions |
| Resource Not Found | 404 | `RESOURCE_NOT_FOUND` | "The requested resource does not exist" | Resource resolution path |
| Internal Error | 500 | `EVALUATION_ERROR` | "Authorization system temporarily unavailable" | Internal error details |
| Timeout | 503 | `EVALUATION_TIMEOUT` | "Request taking too long, please try again" | Component timeout details |

Error messages include **correlation IDs** that link user-facing errors to detailed internal logs, enabling support teams to quickly diagnose issues without exposing sensitive authorization internals to end users.

> **Architecture Decision: Structured Message Passing**
> - **Context**: Authorization components need to exchange complex authorization context while maintaining loose coupling and testability
> - **Options Considered**: 
>   1. Shared mutable state with locking
>   2. Event-driven messaging with queues  
>   3. Structured message passing with immutable data
> - **Decision**: Structured message passing with immutable data structures
> - **Rationale**: Provides predictable behavior under concurrent load, enables easy testing and debugging, and eliminates race conditions without the complexity of event queues
> - **Consequences**: Slightly higher memory usage due to message copying, but dramatically improved reliability and debuggability

### Caching and Performance

Authorization systems must balance security accuracy with performance requirements - users expect sub-100ms response times while the system evaluates complex policies across multiple data sources. Think of this like a librarian's memory system: frequently requested books are kept on a nearby cart (L1 cache), popular books stay near the front desk (L2 cache), while rare books require a trip to the archives (database lookup). The key is predicting what information will be needed again and keeping it readily accessible without compromising accuracy.

The caching strategy implements **multi-level hierarchical caching** with different eviction policies optimized for different access patterns. Each cache level serves a specific purpose in the performance optimization strategy while maintaining strict consistency guarantees to prevent security vulnerabilities.

![Audit Logging Pipeline](./diagrams/audit-pipeline.svg)

**L1 In-Memory Cache: Hot Decision Cache**

The L1 cache stores the most recent authorization decisions in the application's memory space for microsecond-level access times. This cache targets the **temporal locality** principle - users often repeat identical actions within short time windows, such as loading a web page that makes multiple API calls for the same resource.

The L1 cache uses a **write-through** strategy where successful authorization decisions are immediately cached, but the cache never serves as the source of truth. Each cache entry includes a cryptographic hash of the inputs to prevent cache poisoning attacks where malicious actors attempt to store false authorization decisions.

| Cache Configuration | Value | Reasoning |
|---------------------|-------|-----------|
| Maximum entries | 10,000 per node | Balance memory usage vs. hit rate |
| Entry TTL | 5 minutes | Minimize stale decision risk |
| Eviction policy | LRU (Least Recently Used) | Optimize for temporal locality |
| Key hashing | SHA-256 | Prevent cache collision attacks |
| Memory limit | 50MB per node | Prevent memory exhaustion |

Cache keys combine all authorization parameters that could affect the decision outcome. The key generation algorithm ensures that semantically different requests produce different cache keys while identical requests (even if submitted in different field orders) produce identical keys.

```
Cache Key Components:
- UserID: normalized user identifier
- TenantID: tenant scope for decision  
- ResourceID: target resource identifier
- Action: normalized action string
- AttributeHash: hash of relevant attribute values
- PolicyVersion: version stamp of applicable policies
- RoleVersion: version stamp of user's role assignments
```

**L2 Distributed Cache: User Permission Cache**

The L2 cache stores computed user permissions and resolved attributes across multiple application nodes using Redis or similar distributed caching systems. This cache targets **spatial locality** - if a user has permission to read a resource, they likely have permission to perform other operations on the same resource or related resources.

The L2 cache implements **lazy loading with write-aside** pattern where permissions are cached only after being computed, and updates to roles or policies invalidate related cache entries through a sophisticated cache invalidation strategy.

| Cache Type | Storage Format | TTL | Invalidation Trigger |
|-----------|---------------|-----|---------------------|
| User permissions | Compressed permission bitmap | 30 minutes | Role assignment changes |
| Resolved attributes | JSON attribute map | 15 minutes | User profile updates |
| Policy applicability | List of applicable policy IDs | 1 hour | Policy modifications |
| Resource metadata | Serialized resource structure | 2 hours | Resource updates |

The distributed cache implements **cache coherence** through a publish-subscribe invalidation mechanism. When authorization data changes (roles, policies, resources), the system publishes invalidation messages to all cache nodes, ensuring consistent behavior across the distributed deployment.

**L3 Database Query Cache: Policy and Role Cache**

The L3 cache optimizes database query performance by caching computed results from expensive operations like role hierarchy traversal and policy evaluation. This cache targets **computational locality** - avoiding repeated calculation of the same derived data.

The database layer cache focuses on **read-heavy optimization** since authorization workloads typically have high read-to-write ratios. Most users' permissions remain stable over time, making aggressive caching safe and effective.

| Cached Computation | Cache Duration | Invalidation Strategy |
|-------------------|---------------|----------------------|
| Role transitive closure | 4 hours | Role hierarchy modifications |
| Policy compilation results | 2 hours | Policy syntax changes |
| Resource hierarchy paths | 6 hours | Resource relationship changes |
| Tenant membership lookups | 1 hour | User-tenant assignment changes |

**Cache Invalidation and Consistency**

The most challenging aspect of authorization caching is maintaining consistency without sacrificing performance. The system implements **versioned cache invalidation** where each cached item includes version stamps from its source data. When source data changes, the version changes, automatically invalidating dependent cache entries.

| Data Change Type | Affected Caches | Invalidation Scope | Propagation Time |
|------------------|----------------|-------------------|------------------|
| User role assignment | L1, L2 user permissions | Single user | < 1 second |
| Role definition change | L2, L3 role computations | All users with role | < 5 seconds |
| Policy modification | L1 decisions, L3 compilations | Policy-scoped users | < 10 seconds |
| Resource updates | L1 decisions, L2 attributes | Resource-scoped access | < 1 second |
| Tenant changes | All cache levels | Tenant-scoped data | < 30 seconds |

The invalidation system uses **probabilistic consistency** where cache entries include confidence scores based on the age and volatility of their source data. Entries with low confidence scores trigger background refresh operations even before expiration.

**Performance Monitoring and Optimization**

The caching system includes comprehensive instrumentation to measure effectiveness and identify optimization opportunities. Performance metrics drive automatic tuning of cache parameters and alert operators to performance degradation.

| Metric Category | Key Indicators | Target Values | Alert Thresholds |
|-----------------|---------------|---------------|------------------|
| Hit rates | L1: 85%, L2: 70%, L3: 60% | Minimize cache misses | < 80% L1, < 60% L2 |
| Latency | P95 < 50ms total | User experience | > 100ms P95 |
| Invalidation lag | < 10s for critical changes | Consistency guarantee | > 60s lag |
| Memory usage | < 100MB per node | Resource efficiency | > 200MB usage |
| Error rates | < 0.1% cache errors | Reliability target | > 1% error rate |

The monitoring system implements **adaptive cache sizing** where cache limits automatically adjust based on observed access patterns and available system resources. During high-load periods, the system can temporarily reduce cache TTLs to maintain consistency while preserving performance.

> **Key Design Principle**: Authorization caching implements **security-first performance optimization**. When in doubt between performance and security, the system chooses security. Cache invalidation errors result in cache clearing rather than serving potentially stale data. This approach may occasionally impact performance but never compromises security posture.

⚠️ **Pitfall: Cache Invalidation Race Conditions**
A common mistake is implementing cache invalidation without considering race conditions between cache updates and data modifications. For example, if a user's role is revoked but the cache invalidation message arrives before the role revocation completes, the cache might be repopulated with stale data. The fix is to use **causal ordering** where cache invalidations include the data version they're invalidating, preventing out-of-order updates from causing security vulnerabilities.

⚠️ **Pitfall: Attribute-Based Cache Key Generation**
Another frequent error is generating cache keys that don't include all relevant attributes for policy evaluation. If a policy depends on the user's department but the cache key only includes user ID, users might receive cached decisions based on their previous department after a transfer. The solution is to include **attribute fingerprints** in cache keys that change when any relevant attribute changes.

### Implementation Guidance

This section provides practical implementation patterns for building the authorization request flow, inter-component messaging, and caching layers using Go. The focus is on providing working infrastructure code and detailed skeletons for the core authorization orchestration logic.

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| HTTP Framework | `net/http` with custom middleware | `gin` or `echo` for richer routing |
| Caching Layer | In-memory `sync.Map` | Redis with `go-redis` client |
| Message Serialization | `encoding/json` | Protocol Buffers with `protobuf` |
| Database Access | `database/sql` with prepared statements | `gorm` ORM for complex queries |
| Monitoring | Built-in `expvar` metrics | Prometheus with `prometheus/client_golang` |
| Distributed Tracing | Custom request IDs | OpenTelemetry with Jaeger |

**Recommended File Structure**

```
internal/
├── authz/
│   ├── authorizer.go           ← main orchestration logic
│   ├── request.go              ← request/response structures
│   ├── pipeline.go             ← evaluation pipeline
│   └── authorizer_test.go      ← integration tests
├── cache/
│   ├── manager.go              ← multi-level cache coordination
│   ├── memory.go               ← L1 in-memory cache
│   ├── distributed.go          ← L2 Redis cache
│   ├── invalidation.go         ← cache invalidation logic
│   └── cache_test.go           ← cache behavior tests
├── messaging/
│   ├── types.go                ← message structures
│   ├── validation.go           ← request validation
│   ├── errors.go               ← structured error types
│   └── messaging_test.go       ← message handling tests
└── metrics/
    ├── collector.go            ← performance metrics
    ├── dashboard.go            ← metric aggregation
    └── alerts.go               ← threshold monitoring
```

**Complete Infrastructure: Multi-Level Cache Manager**

```go
package cache

import (
    "context"
    "crypto/sha256"
    "encoding/json"
    "fmt"
    "sync"
    "time"

    "github.com/go-redis/redis/v8"
)

// CacheManager coordinates multi-level caching for authorization decisions
type CacheManager struct {
    l1Cache    *MemoryCache
    l2Cache    *redis.Client
    metrics    *CacheMetrics
    config     *CacheConfig
}

type CacheConfig struct {
    L1TTL       time.Duration
    L2TTL       time.Duration
    MaxL1Size   int
    RedisAddr   string
    MetricsAddr string
}

type CacheEntry struct {
    Value      interface{}
    Version    string
    Timestamp  time.Time
    Hash       string
}

type CacheMetrics struct {
    L1Hits, L1Misses int64
    L2Hits, L2Misses int64
    mu sync.RWMutex
}

func NewCacheManager(config *CacheConfig) *CacheManager {
    rdb := redis.NewClient(&redis.Options{
        Addr: config.RedisAddr,
        DB:   0,
    })

    return &CacheManager{
        l1Cache: NewMemoryCache(config.MaxL1Size),
        l2Cache: rdb,
        metrics: &CacheMetrics{},
        config:  config,
    }
}

// Get attempts to retrieve value from cache hierarchy
func (cm *CacheManager) Get(ctx context.Context, key string) (interface{}, bool) {
    // Try L1 cache first
    if value, ok := cm.l1Cache.Get(key); ok {
        cm.recordHit(1)
        return value, true
    }
    cm.recordMiss(1)

    // Try L2 cache
    val, err := cm.l2Cache.Get(ctx, key).Result()
    if err == nil {
        var entry CacheEntry
        if json.Unmarshal([]byte(val), &entry) == nil {
            // Promote to L1 cache
            cm.l1Cache.Set(key, entry.Value, cm.config.L1TTL)
            cm.recordHit(2)
            return entry.Value, true
        }
    }
    cm.recordMiss(2)

    return nil, false
}

// Set stores value in appropriate cache levels
func (cm *CacheManager) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    entry := CacheEntry{
        Value:     value,
        Version:   generateVersion(),
        Timestamp: time.Now(),
        Hash:      cm.generateHash(key, value),
    }

    // Store in L1 cache
    cm.l1Cache.Set(key, value, cm.config.L1TTL)

    // Store in L2 cache
    data, err := json.Marshal(entry)
    if err != nil {
        return fmt.Errorf("failed to marshal cache entry: %w", err)
    }

    return cm.l2Cache.SetEX(ctx, key, data, ttl).Err()
}

// InvalidatePattern removes cache entries matching pattern
func (cm *CacheManager) InvalidatePattern(ctx context.Context, pattern string) error {
    // Invalidate L1 cache
    cm.l1Cache.InvalidatePattern(pattern)

    // Invalidate L2 cache
    keys, err := cm.l2Cache.Keys(ctx, pattern).Result()
    if err != nil {
        return err
    }

    if len(keys) > 0 {
        return cm.l2Cache.Del(ctx, keys...).Err()
    }

    return nil
}

func (cm *CacheManager) generateHash(key string, value interface{}) string {
    hasher := sha256.New()
    hasher.Write([]byte(key))
    
    data, _ := json.Marshal(value)
    hasher.Write(data)
    
    return fmt.Sprintf("%x", hasher.Sum(nil))
}

func (cm *CacheManager) recordHit(level int) {
    cm.metrics.mu.Lock()
    defer cm.metrics.mu.Unlock()
    
    if level == 1 {
        cm.metrics.L1Hits++
    } else {
        cm.metrics.L2Hits++
    }
}

func (cm *CacheManager) recordMiss(level int) {
    cm.metrics.mu.Lock()
    defer cm.metrics.mu.Unlock()
    
    if level == 1 {
        cm.metrics.L1Misses++
    } else {
        cm.metrics.L2Misses++
    }
}

func generateVersion() string {
    return fmt.Sprintf("%d", time.Now().UnixNano())
}
```

**Complete Infrastructure: Request Validation and Messaging**

```go
package messaging

import (
    "context"
    "fmt"
    "regexp"
    "strings"
    "time"
    "unicode/utf8"
)

// RequestValidator handles authorization request validation
type RequestValidator struct {
    actionPattern *regexp.Regexp
    maxAttrSize   int
    maxAttrCount  int
}

type ValidationError struct {
    Field   string
    Message string
    Code    string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

func NewRequestValidator() *RequestValidator {
    return &RequestValidator{
        actionPattern: regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9:._-]*$`),
        maxAttrSize:   1024,  // Max size for attribute values
        maxAttrCount:  50,    // Max number of attributes
    }
}

// ValidateAuthzRequest performs comprehensive request validation
func (v *RequestValidator) ValidateAuthzRequest(req *AuthzRequest) error {
    if req == nil {
        return ValidationError{
            Field: "request", 
            Message: "request cannot be nil", 
            Code: "NULL_REQUEST",
        }
    }

    // Validate required fields
    if err := v.validateUserID(req.UserID); err != nil {
        return err
    }

    if err := v.validateTenantID(req.TenantID); err != nil {
        return err
    }

    if err := v.validateAction(req.Action); err != nil {
        return err
    }

    if err := v.validateResource(req.Resource); err != nil {
        return err
    }

    // Validate optional fields
    if err := v.validateAttributes(req.Attributes); err != nil {
        return err
    }

    if err := v.validateTimestamp(req.Timestamp); err != nil {
        return err
    }

    return nil
}

func (v *RequestValidator) validateUserID(userID string) error {
    if userID == "" {
        return ValidationError{
            Field: "UserID", 
            Message: "user ID cannot be empty", 
            Code: "EMPTY_USER_ID",
        }
    }

    if !utf8.ValidString(userID) {
        return ValidationError{
            Field: "UserID", 
            Message: "user ID must be valid UTF-8", 
            Code: "INVALID_UTF8",
        }
    }

    if len(userID) > 256 {
        return ValidationError{
            Field: "UserID", 
            Message: "user ID too long (max 256 characters)", 
            Code: "USER_ID_TOO_LONG",
        }
    }

    return nil
}

func (v *RequestValidator) validateTenantID(tenantID string) error {
    if tenantID == "" {
        return ValidationError{
            Field: "TenantID", 
            Message: "tenant ID cannot be empty", 
            Code: "EMPTY_TENANT_ID",
        }
    }

    if !utf8.ValidString(tenantID) {
        return ValidationError{
            Field: "TenantID", 
            Message: "tenant ID must be valid UTF-8", 
            Code: "INVALID_UTF8",
        }
    }

    return nil
}

func (v *RequestValidator) validateAction(action string) error {
    if action == "" {
        return ValidationError{
            Field: "Action", 
            Message: "action cannot be empty", 
            Code: "EMPTY_ACTION",
        }
    }

    if !v.actionPattern.MatchString(action) {
        return ValidationError{
            Field: "Action", 
            Message: "action format invalid (must match pattern: ^[a-zA-Z][a-zA-Z0-9:._-]*$)", 
            Code: "INVALID_ACTION_FORMAT",
        }
    }

    return nil
}

func (v *RequestValidator) validateResource(resource *Resource) error {
    if resource == nil {
        return ValidationError{
            Field: "Resource", 
            Message: "resource cannot be nil", 
            Code: "NULL_RESOURCE",
        }
    }

    if resource.ID == "" {
        return ValidationError{
            Field: "Resource.ID", 
            Message: "resource ID cannot be empty", 
            Code: "EMPTY_RESOURCE_ID",
        }
    }

    if resource.Type == "" {
        return ValidationError{
            Field: "Resource.Type", 
            Message: "resource type cannot be empty", 
            Code: "EMPTY_RESOURCE_TYPE",
        }
    }

    return nil
}

func (v *RequestValidator) validateAttributes(attributes map[string]any) error {
    if len(attributes) > v.maxAttrCount {
        return ValidationError{
            Field: "Attributes", 
            Message: fmt.Sprintf("too many attributes (max %d)", v.maxAttrCount), 
            Code: "TOO_MANY_ATTRIBUTES",
        }
    }

    for key, value := range attributes {
        if key == "" {
            return ValidationError{
                Field: "Attributes", 
                Message: "attribute key cannot be empty", 
                Code: "EMPTY_ATTRIBUTE_KEY",
            }
        }

        // Check attribute value size (serialize to estimate)
        if valueStr := fmt.Sprintf("%v", value); len(valueStr) > v.maxAttrSize {
            return ValidationError{
                Field: fmt.Sprintf("Attributes[%s]", key), 
                Message: fmt.Sprintf("attribute value too large (max %d characters)", v.maxAttrSize), 
                Code: "ATTRIBUTE_VALUE_TOO_LARGE",
            }
        }
    }

    return nil
}

func (v *RequestValidator) validateTimestamp(timestamp time.Time) error {
    if !timestamp.IsZero() {
        // Check if timestamp is reasonable (not too far in past/future)
        now := time.Now()
        if timestamp.Before(now.Add(-24*time.Hour)) || timestamp.After(now.Add(1*time.Hour)) {
            return ValidationError{
                Field: "Timestamp", 
                Message: "timestamp outside acceptable range", 
                Code: "INVALID_TIMESTAMP_RANGE",
            }
        }
    }

    return nil
}
```

**Core Logic Skeleton: Authorization Orchestrator**

```go
package authz

import (
    "context"
    "time"
    "fmt"
)

// Authorizer orchestrates the complete authorization evaluation pipeline
type Authorizer struct {
    roleEngine      RoleEngine
    policyEngine    PolicyEngine
    resourceEval    ResourceEvaluator
    auditLogger     AuditLogger
    cacheManager    CacheManager
    validator       RequestValidator
    config          AuthorizerConfig
}

type AuthorizerConfig struct {
    EvaluationTimeout time.Duration
    CacheEnabled      bool
    AuditEnabled      bool
    MaxConcurrency    int
}

// IsAuthorized performs comprehensive authorization evaluation
func (a *Authorizer) IsAuthorized(ctx context.Context, req *AuthzRequest) (*AuthzDecision, error) {
    // TODO 1: Generate unique request ID if not provided
    // - Check if req.RequestID is empty
    // - Generate UUID using crypto/rand if needed
    // - Set req.Timestamp to current time if zero
    
    // TODO 2: Validate request structure and content
    // - Call a.validator.ValidateAuthzRequest(req)
    // - Return immediate deny with validation error if invalid
    // - Log validation failures for security monitoring
    
    // TODO 3: Build tenant context for request
    // - Call buildTenantContext(ctx, req.UserID, req.TenantID)
    // - Validate user has access to target tenant
    // - Handle cross-tenant scenarios if applicable
    
    // TODO 4: Check cache for recent identical decision
    // - Generate cache key from req fields: generateCacheKey(req)
    // - Call a.cacheManager.Get(ctx, cacheKey)
    // - Validate cached decision is still valid (check versions)
    // - Return cached decision if valid, set CacheHit = true
    
    // TODO 5: Evaluate using RBAC role engine
    // - Call a.roleEngine.HasPermission(ctx, req.UserID, req.TenantID, req.Resource.ID, req.Action)
    // - Record evaluation method = EvaluationRBAC
    // - If explicit allow/deny, may short-circuit other evaluations
    
    // TODO 6: Evaluate using ABAC policy engine if needed
    // - Build evaluation context: buildEvaluationContext(ctx, req)
    // - Call a.policyEngine.Evaluate(ctx, req)
    // - Record applied policies from evaluation result
    // - Handle policy conflicts using deny-overrides logic
    
    // TODO 7: Evaluate resource-based permissions
    // - Call a.resourceEval.EvaluateResourceAccess(ctx, req)
    // - Check ownership, sharing, and hierarchical permissions
    // - Handle cross-tenant resource access if applicable
    
    // TODO 8: Synthesize final decision from all evaluations
    // - Apply conservative decision algorithm (any deny = final deny)
    // - Build comprehensive reason string explaining decision
    // - Calculate total evaluation time for performance monitoring
    
    // TODO 9: Cache the decision for future requests
    // - Only cache if caching enabled and decision is cacheable
    // - Set appropriate TTL based on decision volatility
    // - Handle cache errors gracefully (log but don't fail request)
    
    // TODO 10: Log authorization decision for audit trail
    // - Call a.auditLogger.LogAuthzDecision(ctx, req, decision)
    // - Include complete request context and decision reasoning
    // - Handle audit failures without impacting user experience
    
    return nil, fmt.Errorf("not implemented")
}

// generateCacheKey creates consistent cache key from request parameters
func (a *Authorizer) generateCacheKey(req *AuthzRequest) string {
    // TODO: Implement cache key generation
    // - Include UserID, TenantID, Resource.ID, Action
    // - Hash relevant attributes that affect evaluation
    // - Include version stamps from roles/policies
    // - Ensure consistent ordering for identical semantic requests
    return ""
}

// buildTenantContext resolves user's relationship to target tenant
func (a *Authorizer) buildTenantContext(ctx context.Context, userID, tenantID string) (*TenantContext, error) {
    // TODO: Implement tenant context building
    // - Query user's tenant memberships
    // - Validate access to target tenant
    // - Handle cross-tenant scenarios
    // - Build complete tenant context structure
    return nil, fmt.Errorf("not implemented")
}

// buildEvaluationContext gathers attributes for policy evaluation
func (a *Authorizer) buildEvaluationContext(ctx context.Context, req *AuthzRequest) (*EvaluationContext, error) {
    // TODO: Implement evaluation context building
    // - Resolve user attributes from identity provider
    // - Gather resource attributes from metadata
    // - Extract environment attributes from request context
    // - Merge with request.Attributes, handling conflicts appropriately
    return nil, fmt.Errorf("not implemented")
}

// synthesizeDecision combines results from multiple evaluation engines
func (a *Authorizer) synthesizeDecision(rbacResult *PermissionResult, abacResult *PolicyDecision, resourceResult *ResourceAccessResult) *AuthzDecision {
    // TODO: Implement decision synthesis logic
    // - Apply deny-overrides: any explicit deny = final deny
    // - Require at least one explicit allow for final allow
    // - Build comprehensive reason explaining decision chain
    // - Record which evaluation methods contributed to decision
    return nil
}
```

**Milestone Checkpoint: Authorization Flow**

After implementing the authorization orchestration:

1. **Unit Test Validation**:
   ```bash
   go test ./internal/authz/... -v
   ```
   Expected: All validation, caching, and orchestration tests pass

2. **Integration Test Scenarios**:
   ```bash
   go test ./internal/authz/ -tags=integration -v
   ```
   Test scenarios should include:
   - Simple RBAC allow/deny decisions
   - Complex ABAC policy evaluation
   - Cache hit/miss behavior
   - Error handling and timeouts
   - Audit log generation

3. **Manual Verification Steps**:
   - Start authorization server with test configuration
   - Send authorization requests via HTTP API
   - Verify decision consistency across repeated requests
   - Check audit logs contain expected decision trails
   - Monitor cache hit rates and evaluation latencies

4. **Performance Benchmarks**:
   ```bash
   go test -bench=BenchmarkAuthorization -benchmem
   ```
   Target performance:
   - < 50ms P95 latency for cached decisions
   - < 200ms P95 latency for full evaluation
   - > 1000 RPS sustained throughput per node


## Error Handling and Edge Cases

> **Milestone(s):** All milestones - this section addresses failure modes, detection strategies, and recovery mechanisms that apply across role management (Milestone 1), policy evaluation (Milestone 2), multi-tenancy (Milestone 3), and audit logging (Milestone 4)

Authorization systems operate at the critical intersection of security and availability, where failures can have severe consequences ranging from security breaches to complete service outages. Unlike many system components that can degrade gracefully, authorization systems face a fundamental tension: they must make definitive allow/deny decisions even when operating under adverse conditions. This section explores the failure modes that authorization systems encounter, the strategies for detecting these failures before they impact security, and the recovery mechanisms that maintain both security and availability.

Think of authorization system error handling like an air traffic control system during severe weather. The controllers cannot simply stop making decisions when visibility is poor or communication systems are intermittent—aircraft still need landing clearances and runway assignments. Similarly, our authorization system cannot pause user requests when the database is slow or policy evaluation becomes complex. The system must make conservative, defensible decisions while working to restore full functionality. This requires sophisticated failure detection, graceful degradation patterns that prioritize security over convenience, and recovery mechanisms that restore normal operation without introducing security gaps.

The challenge intensifies in multi-tenant SaaS environments where a failure affecting one tenant's authorization could cascade to impact other tenants, and where the scale of authorization requests makes manual intervention impractical. Our system must handle everything from transient network failures during policy evaluation to systematic attacks designed to exploit error handling pathways. Each failure mode requires specific detection logic, predetermined fallback behavior, and recovery procedures that maintain audit trails and tenant isolation even under duress.

![Policy Evaluation State Machine](./diagrams/policy-evaluation-state.svg)

![Multi-Tenant Resource Isolation](./diagrams/tenant-isolation.svg)

### Common Failure Scenarios

Authorization systems encounter failure modes that span infrastructure, application logic, and security concerns. Understanding these scenarios and their cascading effects is essential for building robust authorization infrastructure that maintains security guarantees even during adverse conditions.

#### Database Failures and Data Consistency Issues

Database failures represent one of the most critical failure modes for authorization systems because they directly impact the system's ability to retrieve roles, policies, and resource ownership information. Unlike content databases where temporary unavailability might delay user workflows, authorization database failures immediately block all protected operations across the entire system.

The most straightforward database failure is complete database unavailability due to network partitions, hardware failures, or maintenance operations. When the `PolicyStore` or `RoleStorage` cannot connect to the database, the system loses access to current role assignments, policy definitions, and resource ownership records. This scenario requires immediate failover to cached data while implementing strict cache coherence protocols to prevent authorization decisions based on stale permission grants.

More insidious are partial database failures where some queries succeed while others fail due to corruption, lock contention, or resource exhaustion. Consider a scenario where user role queries succeed but policy evaluation queries fail due to a corrupted policy table index. The `RoleEngine` successfully determines that a user has the Manager role, but the `PolicyEngine` cannot evaluate attribute-based conditions that might restrict that role's permissions for sensitive resources. The system must detect this inconsistency and either deny the request (fail-safe) or escalate to human review for critical operations.

Database consistency issues emerge when concurrent updates to role assignments or policy definitions create temporary inconsistent states. For example, during a role hierarchy reorganization, a user might temporarily inherit conflicting permissions through different role paths, or policy updates might create contradictory allow/deny rules for the same resource. The system must detect these consistency violations during the `ValidateRoleHierarchy` process and either block the inconsistent updates or provide conflict resolution mechanisms.

**Common Database Failure Scenarios:**

| Failure Mode | Symptoms | Detection Method | Recovery Strategy |
|--------------|----------|------------------|-------------------|
| Complete database unavailability | All queries fail with connection timeouts | Connection health checks every 30 seconds | Switch to read-only cache mode with degraded functionality |
| Partial table corruption | Specific queries fail while others succeed | Query success rate monitoring per table | Isolate corrupted tables and disable dependent features |
| Lock contention deadlocks | Queries timeout during concurrent policy updates | Deadlock detection in query logs | Retry with exponential backoff and eventual consistency |
| Replication lag | Stale data returned from read replicas | Compare write timestamps with read results | Force reads from primary database for recent updates |
| Transaction isolation failures | Dirty reads cause inconsistent authorization state | Audit trail hash verification detects data corruption | Rollback to last known consistent state and replay changes |

#### Policy Conflicts and Evaluation Errors

Policy conflicts arise when multiple policies apply to the same authorization request but specify contradictory effects or when policy conditions cannot be evaluated due to missing or invalid attribute values. These conflicts represent logical inconsistencies in the authorization configuration that require sophisticated resolution algorithms and clear precedence rules.

The most common policy conflict occurs when one policy allows access while another policy denies the same access for overlapping resource and action combinations. Our system implements a deny-overrides policy combining algorithm where any explicit deny decision overrides all allow decisions, but this can create unexpected behavior when policy authors don't understand the interaction between their policies and existing rules. For example, a department manager might create a policy allowing their team to access project documents, not realizing that a security policy denies access to those same documents for users without security clearance.

Policy evaluation errors occur when the `PolicyEngine` cannot assess policy conditions due to missing attribute values, invalid attribute types, or malformed condition expressions. Consider a policy that requires `user.securityClearance >= resource.classificationLevel`, but the `AttributeResolver` cannot retrieve the user's security clearance from the HR system due to API timeouts. The system must decide whether to allow access (risking security breach), deny access (potentially blocking legitimate work), or escalate to manual review.

More subtle evaluation errors emerge from attribute type mismatches or undefined behavior in condition operators. A policy might compare a string attribute to a numeric value, or attempt to evaluate a regular expression pattern against a null attribute. The policy evaluation engine must handle these edge cases gracefully without exposing error details that could reveal information about system internals or other users' attributes.

**Policy Conflict Resolution Matrix:**

| Conflict Type | Resolution Strategy | Example Scenario | Outcome |
|---------------|-------------------|------------------|---------|
| Allow vs Deny | Deny overrides (fail-safe) | Department policy allows, security policy denies | Access denied |
| Multiple Allow with different conditions | Most restrictive conditions apply | Manager role allows 9-5, director role allows 24/7 | Apply most restrictive time window |
| Undefined attribute in condition | Treat as false (fail-safe) | Policy requires `user.department` but attribute missing | Condition evaluates to false |
| Type mismatch in condition | Convert or deny based on operator | String compared to number using `>=` | Deny with type error in audit log |
| Circular policy references | Break cycle at detection point | Policy A includes Policy B, Policy B includes Policy A | Disable both policies until cycle resolved |

#### Performance Degradation and Timeout Handling

Authorization systems face unique performance challenges because they sit in the critical path of every protected operation, making authorization latency directly visible to end users. Performance degradation can result from database query slowdowns, complex policy evaluation, cache misses, or resource contention during high-traffic periods.

Database query performance degradation often manifests as gradually increasing response times for role lookups or policy retrieval operations. This can result from growing data volumes, missing database indexes, or competing queries from other system components. When authorization queries exceed acceptable latency thresholds, the system must balance between waiting for accurate results and making decisions based on cached or incomplete data.

Complex policy evaluation represents another significant performance bottleneck, particularly for ABAC policies with nested condition trees or policies requiring expensive attribute resolution from external systems. A policy that evaluates user attributes from Active Directory, resource classifications from a document management system, and environmental conditions from network security tools might require several seconds to complete under normal conditions and could timeout completely during peak usage or external system outages.

Cache performance degradation occurs when cache hit rates decline due to changing access patterns, cache capacity constraints, or ineffective invalidation strategies. When the `CacheManager` experiences high miss rates, authorization latency increases dramatically as each request requires fresh database queries and policy evaluation. This creates a cascading effect where slower authorization increases request queuing, which further increases cache eviction pressure.

**Performance Degradation Scenarios:**

| Performance Issue | Latency Impact | Detection Threshold | Mitigation Strategy |
|-------------------|----------------|-------------------|-------------------|
| Database query slowdown | 100ms → 2000ms authorization latency | >500ms average query time | Enable query result caching with 5-minute TTL |
| Complex ABAC evaluation | 50ms → 1000ms policy evaluation | >200ms policy evaluation time | Implement policy evaluation circuit breaker |
| Cache miss avalanche | 10ms → 500ms cache lookup | <70% cache hit rate | Pre-warm cache and extend TTL during degradation |
| External attribute resolution timeout | Variable, up to 10s timeout | >80% attribute resolution timeouts | Use cached attributes and mark decisions as provisional |
| Concurrent request bottleneck | Queuing delays up to 30s | >1000 concurrent authorization requests | Implement request prioritization and load shedding |

### Graceful Degradation

Graceful degradation in authorization systems requires carefully designed fallback mechanisms that maintain security boundaries while preserving as much system functionality as possible during adverse conditions. Unlike many system components where degradation can mean reduced features or slower performance, authorization systems must continue making definitive access control decisions even when operating with incomplete information or reduced infrastructure capacity.

#### Safe Defaults and Fail-Safe Behavior

The foundation of graceful degradation is establishing safe defaults that prioritize security over availability when the system cannot make fully informed authorization decisions. This fail-safe approach assumes that denying access to legitimate users is less harmful than granting access to unauthorized users, though this assumption must be balanced against business requirements and user experience considerations.

Safe defaults apply at multiple levels of the authorization system. At the policy evaluation level, any condition that cannot be evaluated due to missing attributes or evaluation errors defaults to false, making the overall policy condition more restrictive. At the policy combining level, if policy retrieval fails or policy evaluation encounters errors, the system denies access rather than falling back to permissive defaults. At the infrastructure level, if the authorization system cannot reach required data stores or external services, it denies requests rather than assuming they would be allowed.

However, pure fail-safe behavior can render systems completely unusable during infrastructure failures. The system must implement graduated degradation where different types of operations have different fallback behaviors based on their risk profiles. Read operations on public data might fall back to role-based authorization when attribute-based policies cannot be evaluated, while write operations on sensitive data require full policy evaluation or explicit administrative override.

**Fail-Safe Decision Matrix:**

| Failure Condition | Low-Risk Operations | Medium-Risk Operations | High-Risk Operations |
|-------------------|-------------------|----------------------|-------------------|
| Database unavailable | Allow based on cached roles | Deny with admin override option | Deny completely |
| Policy evaluation timeout | Fall back to RBAC only | Deny with audit flag | Deny with security escalation |
| Missing user attributes | Use default/minimal permissions | Deny with self-service attribute update | Deny with manual review required |
| External service timeout | Continue with cached data | Delay request with retry mechanism | Deny with emergency contact info |
| System overload | Priority queuing based on user role | Rate limiting with graceful queueing | Reject requests with retry-after header |

> **Decision: Conservative Decision Algorithm**
> - **Context**: Authorization systems must make definitive allow/deny decisions even when operating with incomplete information during failures
> - **Options Considered**: Always allow during failures (maximize availability), always deny during failures (maximize security), graduated response based on risk assessment
> - **Decision**: Implement graduated response with conservative defaults that err toward denial for uncertain situations
> - **Rationale**: Pure availability-first approach creates security vulnerabilities, while pure security-first approach renders systems unusable. Graduated response allows business-critical operations to continue while maintaining security boundaries for sensitive operations
> - **Consequences**: Requires careful risk classification of operations and clear escalation procedures for users affected by conservative defaults

#### Circuit Breaker Patterns for External Dependencies

Authorization systems depend on multiple external systems for user attributes, resource metadata, and policy information. When these dependencies become unreliable or unavailable, circuit breaker patterns prevent cascading failures while providing predictable fallback behavior that maintains authorization system availability.

The `AttributeResolver` implements circuit breakers for each external attribute source, monitoring success rates, response times, and error patterns to determine when to stop attempting external calls and switch to cached or default attribute values. Each circuit breaker maintains three states: closed (normal operation), open (failing fast), and half-open (testing recovery). The transition between states uses configurable thresholds based on business requirements and SLA agreements with external systems.

For user attribute resolution, the circuit breaker might allow 10% of attribute resolution calls to fail before opening, then wait 30 seconds before testing recovery with a single request. During the open state, the system uses cached user attributes for authorization decisions and marks these decisions in the audit trail as based on potentially stale data. This allows users to continue working while preventing the authorization system from being overwhelmed by failed attribute resolution calls.

Resource attribute resolution requires more nuanced circuit breaker behavior because resource attributes often change less frequently than user attributes and may have different caching strategies. The circuit breaker for resource metadata might have longer failure thresholds but shorter recovery testing intervals, reflecting the relatively stable nature of resource classifications compared to dynamic user attributes.

**Circuit Breaker Configuration by Dependency:**

| External Dependency | Failure Threshold | Recovery Test Interval | Fallback Behavior | Cache TTL During Failure |
|--------------------|------------------|----------------------|-------------------|------------------------|
| User Directory (LDAP/AD) | 20% failure rate over 1 minute | 60 seconds | Use cached user attributes | 24 hours |
| Resource Metadata Service | 30% failure rate over 2 minutes | 30 seconds | Use cached resource attributes | 4 hours |
| Security Classification Service | 10% failure rate over 30 seconds | 120 seconds | Default to highest classification | 1 hour |
| Audit Log Aggregator | 50% failure rate over 5 minutes | 300 seconds | Buffer locally with disk spillover | Not cached |
| External Policy Repository | 5% failure rate over 5 minutes | 180 seconds | Use locally cached policies only | 12 hours |

#### Partial Functionality and Degraded Operations

When external dependencies fail or system capacity is exceeded, the authorization system can provide partial functionality by temporarily disabling advanced features while maintaining core security controls. This degraded operation mode preserves essential access control while clearly communicating limitations to users and administrators.

During database performance issues, the system might disable complex ABAC policy evaluation and fall back to RBAC-only authorization, providing faster response times while maintaining role-based security boundaries. Users with appropriate roles can continue accessing resources, but conditional access based on time-of-day, location, or resource attributes becomes unavailable. The system clearly indicates this degraded state in authorization responses and audit logs.

When attribute resolution services are unavailable, the system can operate in cached-attribute mode where authorization decisions use the most recent available attribute values with clear timestamps indicating data freshness. Users whose cached attributes are recent enough for their requested operations can proceed normally, while users with stale cached data might face additional authentication requirements or restricted access to sensitive resources.

For multi-tenant systems, degraded operations might include disabling cross-tenant sharing features while maintaining within-tenant access controls, or temporarily blocking tenant creation and modification while allowing existing tenants to operate normally. These degradation decisions require careful consideration of business priorities and clear communication to affected users.

**Degraded Operation Modes:**

| Degradation Level | Available Features | Disabled Features | User Communication | Recovery Criteria |
|------------------|------------------|------------------|-------------------|------------------|
| Level 1: Performance degraded | All features with increased latency | None | "System responding slowly" banner | Response time <200ms for 5 minutes |
| Level 2: External dependency failure | RBAC, cached ABAC, resource ownership | Real-time ABAC, cross-tenant sharing | "Advanced features temporarily unavailable" | Circuit breakers closed for all dependencies |
| Level 3: Database performance issues | Role-based access only | ABAC policies, audit search | "Limited access control active" | Database query time <100ms average |
| Level 4: Emergency mode | Admin override for critical operations | All automated authorization | "Manual authorization required" | Administrative decision to restore |

### Security Edge Cases

Security edge cases in authorization systems represent attack vectors and failure modes that could compromise the confidentiality, integrity, or availability of protected resources. These edge cases often exploit the intersection between system functionality and security boundaries, requiring careful analysis and robust defensive measures.

#### Permission Elevation Attacks and Privilege Escalation

Permission elevation attacks attempt to gain higher privileges than originally granted by exploiting weaknesses in role hierarchy management, policy evaluation logic, or multi-tenant isolation mechanisms. These attacks can occur through direct manipulation of authorization requests, exploitation of race conditions during permission updates, or abuse of legitimate system features like delegation or resource sharing.

Role hierarchy manipulation attacks target the DAG structure used for role inheritance by attempting to create cycles or unauthorized parent-child relationships that could grant excessive permissions. An attacker with permission to modify role assignments might attempt to assign themselves to a role that inherits from administrative roles, or create a new role with carefully crafted permissions that appear benign individually but combine to provide elevated access. The `ValidateRoleHierarchy` function must detect these attempts and prevent role modifications that would create security violations.

Policy evaluation bypass attacks attempt to exploit the logic used for combining multiple policies or evaluating complex conditions. An attacker might create policies with deliberately ambiguous or contradictory conditions, hoping that policy evaluation errors will result in default-allow behavior. For example, a policy with a condition like `user.department == "security" AND user.department == "finance"` is logically impossible but might cause evaluation errors that bypass the deny-overrides logic.

Multi-tenant permission elevation attacks attempt to gain access to resources in other tenants by manipulating tenant context propagation or exploiting cross-tenant sharing features. An attacker might craft authorization requests with malformed tenant IDs, attempt to abuse shared resource links to gain broader access than intended, or exploit race conditions during tenant context switches to access resources with the wrong tenant context.

**Permission Elevation Attack Patterns:**

| Attack Vector | Attack Method | Detection Strategy | Prevention Measure |
|---------------|---------------|-------------------|-------------------|
| Role hierarchy manipulation | Create cycles or unauthorized inheritance | DAG validation on every role update | Immutable role hierarchy with approval workflow |
| Policy evaluation bypass | Create contradictory conditions causing errors | Monitor policy evaluation error rates | Fail-safe evaluation with explicit error handling |
| Tenant context manipulation | Forge or manipulate tenant IDs in requests | Validate tenant context against user memberships | Cryptographically signed tenant tokens |
| Delegation abuse | Chain delegations to exceed original permissions | Track delegation depth and permission scope | Maximum delegation depth and explicit permission limits |
| Resource sharing exploitation | Abuse share links to gain broader access | Monitor unusual access patterns via shared links | Time-limited shares with usage tracking and automatic revocation |

#### Cache Poisoning and Data Integrity Issues

Cache poisoning attacks attempt to inject malicious or unauthorized data into the authorization system's caching layers, potentially causing the system to make incorrect authorization decisions based on false information. These attacks are particularly dangerous because they can affect multiple users and persist until cache invalidation occurs.

Authorization cache poisoning can occur when an attacker manages to insert false permission grants, role assignments, or policy decisions into the `CacheManager`. For example, if an attacker can somehow cause the cache to store a false entry indicating they have administrative roles, all subsequent authorization requests might be granted based on this cached misinformation. The cache system must implement integrity verification mechanisms that detect tampered cache entries and remove them immediately.

Policy cache poisoning attacks target the policy evaluation results cache by attempting to inject false policy decisions that would grant unauthorized access. An attacker might exploit race conditions during cache updates, manipulate cache keys to cause hash collisions, or exploit deserialization vulnerabilities in cached policy objects. The system must validate cached policy decisions against cryptographic signatures or checksums before using them for authorization decisions.

Cross-tenant cache poisoning represents a particularly severe threat where an attacker attempts to inject authorization information from their tenant into another tenant's cache space. This could occur through cache key collisions, inadequate tenant isolation in cache implementations, or exploitation of shared cache infrastructure. The cache system must ensure complete tenant isolation at the cache key level and validate that cached data matches the requesting tenant context.

**Cache Integrity Protection Measures:**

| Cache Type | Poisoning Risk | Integrity Protection | Validation Method | Recovery Procedure |
|------------|----------------|---------------------|-------------------|-------------------|
| Role assignment cache | High - affects multiple requests | HMAC signatures on cache entries | Verify signature before use | Invalidate entire user's cache and reload from database |
| Policy decision cache | Critical - grants unauthorized access | Cryptographic hash of decision context | Hash verification on cache hit | Clear all policy decisions and re-evaluate |
| Attribute value cache | Medium - affects policy evaluation | Timestamp and checksum validation | Compare with source system periodically | Refresh from authoritative source |
| Resource ownership cache | High - affects tenant isolation | Encrypt cache values with tenant keys | Decrypt and verify tenant match | Purge cache and reload with fresh tenant context |

#### Timing Attacks and Information Leakage

Timing attacks against authorization systems attempt to infer sensitive information about users, resources, or system configuration by measuring the time required for authorization decisions. These attacks can reveal information about user roles, resource existence, policy complexity, or system architecture that should not be exposed to unauthorized parties.

Authorization timing attacks can reveal whether users exist in the system by measuring the time difference between authorization requests for valid versus invalid user IDs. If the system takes longer to evaluate authorization for existing users (due to role lookups and policy evaluation) compared to rejecting unknown users immediately, attackers can enumerate valid user accounts. The system must implement constant-time rejection for unauthorized requests regardless of whether the user exists.

Policy complexity timing attacks attempt to infer information about policy structures by measuring evaluation times for different authorization requests. Complex policies with many conditions take longer to evaluate than simple policies, potentially revealing information about which resources have special protection or which user attributes are checked. The authorization system should implement timing normalization that ensures all policy evaluations take similar amounts of time regardless of complexity.

Resource enumeration timing attacks exploit differences in authorization evaluation time to determine whether resources exist and their basic properties. If authorization for non-existent resources returns immediately while authorization for existing resources requires policy evaluation, attackers can map resource hierarchies and discover protected assets. The system must perform consistent authorization evaluation regardless of resource existence.

**Timing Attack Mitigation Strategies:**

| Timing Vector | Information Leaked | Mitigation Approach | Implementation Detail |
|---------------|-------------------|-------------------|----------------------|
| User existence | Valid vs invalid user IDs | Constant-time user validation | Always perform minimum viable role lookup time |
| Resource existence | Resource presence in system | Consistent evaluation path | Evaluate default policy for non-existent resources |
| Policy complexity | Number of conditions in policies | Timing normalization | Pad evaluation time to consistent duration |
| Role hierarchy depth | User's position in org chart | Consistent traversal time | Always traverse maximum possible depth |
| Cross-tenant access | User's tenant memberships | Uniform rejection timing | Consistent tenant validation regardless of membership |

### Implementation Guidance

Building robust error handling for authorization systems requires implementing sophisticated detection, fallback, and recovery mechanisms while maintaining security boundaries. The following guidance provides production-ready infrastructure for handling the failure modes discussed above.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|----------------|
| Circuit Breaker | Manual state tracking with timers | Hystrix-style circuit breaker library |
| Cache Integrity | Simple checksums with timestamps | HMAC signatures with key rotation |
| Timing Protection | Sleep-based timing normalization | Hardware-based constant-time operations |
| Health Monitoring | Basic HTTP health checks | Prometheus metrics with custom alerting |
| Audit Integrity | SHA-256 hash chains | Merkle tree with cryptographic proofs |

**Recommended File Structure:**
```
internal/
  errors/
    failures.go           ← failure mode definitions and detection
    degradation.go        ← graceful degradation state machine
    security.go           ← security edge case handling
    recovery.go           ← recovery and restoration procedures
  circuit/
    breaker.go            ← circuit breaker implementation
    breaker_test.go       ← comprehensive failure scenario tests
  cache/
    integrity.go          ← cache poisoning prevention
    timing.go             ← timing attack protection
  monitoring/
    health.go             ← system health monitoring
    metrics.go            ← performance and security metrics
```

**Complete Circuit Breaker Infrastructure:**

```go
package circuit

import (
    "context"
    "sync"
    "time"
)

// CircuitState represents the current state of a circuit breaker
type CircuitState int

const (
    CircuitClosed CircuitState = iota  // Normal operation
    CircuitOpen                        // Failing fast
    CircuitHalfOpen                    // Testing recovery
)

// CircuitBreaker provides fault tolerance for external dependencies
type CircuitBreaker struct {
    name            string
    failureThreshold int
    recoveryTimeout  time.Duration
    
    mu              sync.RWMutex
    state           CircuitState
    failures        int
    lastFailureTime time.Time
    nextRetryTime   time.Time
}

// CircuitBreakerConfig defines circuit breaker behavior parameters
type CircuitBreakerConfig struct {
    Name             string        // Circuit identifier for monitoring
    FailureThreshold int           // Failures before opening circuit  
    RecoveryTimeout  time.Duration // Time to wait before testing recovery
    SuccessThreshold int           // Successes needed to close circuit
}

// NewCircuitBreaker creates a circuit breaker with specified configuration
func NewCircuitBreaker(config CircuitBreakerConfig) *CircuitBreaker {
    return &CircuitBreaker{
        name:            config.Name,
        failureThreshold: config.FailureThreshold,
        recoveryTimeout:  config.RecoveryTimeout,
        state:           CircuitClosed,
    }
}

// Execute runs the provided function with circuit breaker protection
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() error) error {
    // TODO 1: Check current circuit state and determine if request should proceed
    // TODO 2: If circuit is open, check if recovery test time has arrived
    // TODO 3: Execute the function and handle success/failure outcomes
    // TODO 4: Update circuit state based on execution result
    // TODO 5: Record metrics for monitoring and alerting
    // Hint: Use atomic operations for thread-safe state updates
    panic("implement circuit breaker execution logic")
}
```

**Core Failure Detection System:**

```go
package errors

import (
    "context"
    "time"
)

// FailureMode represents different types of system failures
type FailureMode int

const (
    DatabaseUnavailable FailureMode = iota
    PolicyEvaluationTimeout
    AttributeResolutionFailure
    CacheIntegrityViolation
    SecurityPolicyViolation
)

// SystemHealth tracks overall authorization system health
type SystemHealth struct {
    DatabaseHealth    HealthStatus
    CacheHealth       HealthStatus
    PolicyEngineHealth HealthStatus
    AuditSystemHealth HealthStatus
    
    DegradationLevel  DegradationLevel
    LastHealthCheck   time.Time
}

// HealthChecker monitors system components and detects failures
type HealthChecker struct {
    // TODO: Add fields for monitoring different system components
}

// CheckSystemHealth performs comprehensive system health assessment
func (hc *HealthChecker) CheckSystemHealth(ctx context.Context) (*SystemHealth, error) {
    // TODO 1: Check database connectivity and query performance
    // TODO 2: Validate cache integrity and hit rates  
    // TODO 3: Test policy evaluation latency and success rates
    // TODO 4: Verify audit system is recording decisions properly
    // TODO 5: Determine appropriate degradation level based on health
    // TODO 6: Update system metrics and trigger alerts if needed
    panic("implement comprehensive health checking")
}

// DetectSecurityAnomaly identifies potential security attacks or violations
func (hc *HealthChecker) DetectSecurityAnomaly(ctx context.Context, req *AuthzRequest) []SecurityAnomaly {
    // TODO 1: Check for timing attack patterns in request timing
    // TODO 2: Detect permission elevation attempts in role/policy requests
    // TODO 3: Validate tenant context integrity and prevent manipulation
    // TODO 4: Monitor for unusual cache access patterns indicating poisoning
    // TODO 5: Check for policy evaluation bypass attempts
    panic("implement security anomaly detection")
}
```

**Graceful Degradation State Machine:**

```go
package errors

// DegradationLevel represents current system degradation state
type DegradationLevel int

const (
    FullFunctionality DegradationLevel = iota
    PerformanceDegraded
    ExternalDependencyFailure
    DatabasePerformanceIssues  
    EmergencyMode
)

// DegradationManager handles system degradation and recovery
type DegradationManager struct {
    currentLevel    DegradationLevel
    transitionTime  time.Time
    recoveryMetrics map[string]float64
}

// EvaluateDegradationLevel determines appropriate degradation based on system health
func (dm *DegradationManager) EvaluateDegradationLevel(health *SystemHealth) DegradationLevel {
    // TODO 1: Analyze database performance metrics and availability
    // TODO 2: Check external dependency circuit breaker states
    // TODO 3: Evaluate cache hit rates and integrity status
    // TODO 4: Consider current request load and processing capacity
    // TODO 5: Determine minimum viable degradation level for current conditions
    // TODO 6: Apply hysteresis to prevent rapid state oscillation
    panic("implement degradation level evaluation logic")
}

// GetDegradedCapabilities returns available features at current degradation level
func (dm *DegradationManager) GetDegradedCapabilities() map[string]bool {
    // TODO 1: Return capability map based on current degradation level
    // TODO 2: Consider business priority of different authorization features
    // TODO 3: Ensure security-critical features remain available when possible
    panic("implement degraded capability determination")
}
```

**Milestone Checkpoint - Error Handling Implementation:**

After implementing the error handling and edge case management:

1. **Test Failure Detection**: Run `go test ./internal/errors/...` and verify that all failure detection tests pass, including database timeouts, cache poisoning attempts, and timing attack simulations.

2. **Verify Circuit Breaker Behavior**: Start the authorization service and simulate external dependency failures. Observe that circuit breakers open after configured failure thresholds and attempt recovery after timeout periods.

3. **Validate Graceful Degradation**: Introduce database performance issues and verify that the system degrades to RBAC-only mode while maintaining security boundaries and clearly communicating degraded state.

4. **Security Edge Case Testing**: Attempt permission elevation attacks, cache poisoning, and timing attacks against the running system. Verify that all attacks are detected and blocked with appropriate audit logging.

5. **Recovery Testing**: After simulating various failure modes, restore normal system conditions and verify that the system recovers full functionality without requiring restarts or manual intervention.

**Expected behavior**: The system should maintain security-first decision making even during failures, provide clear degradation state communication, and recover gracefully when conditions improve. All failure events and recovery actions should be recorded in audit logs for compliance review.

**Debugging Tips for Error Handling:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| System denying all requests | Circuit breakers stuck in open state | Check circuit breaker metrics and recovery timers | Manually reset circuit breakers or reduce failure thresholds |
| Inconsistent authorization decisions | Cache poisoning or integrity violations | Verify cache checksums and look for cache key collisions | Clear affected cache entries and investigate attack vectors |
| Authorization taking too long | Timing normalization preventing fast decisions | Monitor timing protection overhead in metrics | Adjust timing normalization parameters for better balance |
| Users getting elevated permissions | Role hierarchy cycles or policy conflicts | Run DAG validation and check for contradictory policies | Fix role hierarchy and implement policy conflict resolution |
| System not degrading during failures | Health checks not detecting failure conditions | Review health check thresholds and monitoring coverage | Tune health check sensitivity and add missing monitors |


## Testing Strategy

> **Milestone(s):** Milestone 4 (Audit Logging & Policy Testing) - this section provides comprehensive testing approaches for authorization systems, including unit testing of individual components, end-to-end authorization flow validation, and milestone-specific verification checkpoints

Testing authorization systems is like quality assurance for a high-security facility. Just as a bank doesn't rely on a single lock or security guard but implements multiple layers of verification - from individual door locks to complete evacuation drills - an authorization system requires comprehensive testing at every level. We need unit tests that verify individual components work correctly in isolation, integration tests that ensure components collaborate properly, and end-to-end tests that validate the complete authorization flow works as intended. Each test serves a different purpose: unit tests catch logic errors early, integration tests reveal interface mismatches, and end-to-end tests ensure the system behaves correctly from the user's perspective.

The testing strategy for our authorization system follows a pyramid structure. At the base, we have extensive unit tests that cover individual components like the `RoleEngine`, `PolicyEngine`, and `ResourceEvaluator`. In the middle layer, we have integration tests that verify component interactions and data flow between services. At the top, we have end-to-end tests that exercise complete authorization scenarios from request to audit logging. This pyramid ensures we catch bugs early where they're cheap to fix, while still validating that the complete system works correctly.

Testing authorization systems presents unique challenges compared to typical business applications. Authorization decisions affect security boundaries, so test failures can indicate potential vulnerabilities. We must test not just positive cases (authorized access is granted) but negative cases (unauthorized access is denied). We need to validate that policy changes don't inadvertently grant excessive permissions or break existing access patterns. Most importantly, we must ensure our tests themselves don't introduce security holes - test data shouldn't use real user credentials, and test policies shouldn't accidentally get deployed to production.

> **Decision: Test-Driven Authorization Development**
> - **Context**: Authorization bugs can create security vulnerabilities that are difficult to detect and expensive to fix. Traditional manual testing approaches don't scale for complex policy interactions.
> - **Options Considered**: 
>   1. Manual testing with ad-hoc scenarios
>   2. Property-based testing with random scenarios  
>   3. Test-driven development with structured test suites
> - **Decision**: Implement comprehensive test-driven development with structured test suites
> - **Rationale**: TDD ensures security requirements are codified as tests before implementation, preventing bugs from being introduced. Structured test suites provide predictable coverage of authorization scenarios.
> - **Consequences**: Higher upfront development cost but significantly reduced security vulnerability risk and faster debugging when issues arise.

| Testing Approach | Advantages | Disadvantages | Authorization Fit |
|-----------------|------------|---------------|------------------|
| Manual Testing | Quick to start, flexible scenarios | Doesn't scale, misses edge cases | Poor - too many policy combinations |
| Property-Based | Discovers unexpected edge cases | Hard to debug failures, complex setup | Good for policy validation |
| Structured TDD | Predictable coverage, documents requirements | Higher initial effort, rigid structure | Excellent - security needs predictability |

The foundation of our testing strategy rests on the `PolicyTestFramework`, which provides isolated environments for testing policy changes safely. This framework creates snapshots of the complete authorization state - users, roles, permissions, policies, and resources - allowing us to test modifications without affecting production data. Think of it as a flight simulator for authorization policies: we can test dangerous scenarios like emergency procedures without risking actual harm.

### Component Unit Testing

Unit testing individual authorization components requires careful attention to isolation, mocking external dependencies, and comprehensive edge case coverage. Each component in our authorization system - `RoleEngine`, `PolicyEngine`, `ResourceEvaluator`, `AuditLogger` - has specific responsibilities and failure modes that must be tested independently. The challenge lies in creating realistic test scenarios while maintaining test isolation and avoiding dependency on external systems like databases or cache layers.

The `RoleEngine` unit tests focus on role hierarchy validation, permission inheritance calculation, and efficient permission lookup mechanisms. We must verify that the directed acyclic graph validation correctly detects and prevents role inheritance cycles, that permission bitmaps are computed correctly for complex role hierarchies, and that the transitive closure computation handles edge cases like roles with no permissions or deeply nested hierarchies.

Testing the role hierarchy DAG validation requires constructing specific test cases that attempt to create cycles at different depths. We create roles A, B, and C, assign A as parent of B, B as parent of C, then attempt to assign C as parent of A. The validation should detect this cycle and return an appropriate error. We also test more subtle cycles involving multiple inheritance paths - for example, role Manager inherits from both TeamLead and ProjectLead, which both inherit from Employee, creating a diamond inheritance pattern that should be valid.

| Test Scenario | Setup | Expected Result | Validates |
|---------------|--------|----------------|-----------|
| Simple Cycle | A→B→C→A | ValidationError with cycle details | Direct cycle detection |
| Diamond Inheritance | Manager→{TeamLead,ProjectLead}→Employee | Success with merged permissions | Valid DAG with convergence |
| Self-Reference | Admin→Admin | ValidationError for self-reference | Self-loop prevention |
| Deep Hierarchy | 10-level role chain | Success with accumulated permissions | Deep traversal performance |
| Orphaned Role | Role with no parents or children | Success with direct permissions only | Isolated role handling |

Permission bitmap encoding tests verify that our high-performance permission storage correctly translates between permission strings and bit positions. We test that the `PermissionRegistry` maintains consistent mappings across system restarts, that bitmap operations (AND, OR, NOT) work correctly for permission calculations, and that bitmap resizing handles permission registry growth without corruption.

```
Permission Bitmap Test Structure:
1. Initialize registry with known permissions
2. Create roles with specific permission sets
3. Verify bitmap encoding matches expected bit patterns
4. Test bitmap operations (union, intersection, difference)
5. Verify performance characteristics for large permission sets
```

The permission lookup optimization tests ensure that our caching and bitmap strategies actually improve performance without introducing correctness bugs. We create scenarios with deeply nested role hierarchies (10+ levels) and large permission sets (1000+ permissions), then verify that repeated permission checks show significant performance improvement with caching enabled.

Policy engine unit tests focus on condition evaluation, policy combining logic, and attribute resolution. The condition tree evaluation is particularly complex because conditions can be nested arbitrarily deep with AND, OR, and NOT operators. We must test that evaluation short-circuits correctly (AND stops at first false, OR stops at first true), that attribute resolution handles missing attributes appropriately, and that error conditions propagate correctly through the evaluation tree.

| Condition Test Case | Tree Structure | Context Values | Expected Result | Tests |
|---------------------|----------------|----------------|-----------------|-------|
| Simple AND | user.dept == "engineering" AND user.level >= 3 | {dept: "engineering", level: 5} | true | Basic conjunction |
| Short-Circuit AND | user.dept == "sales" AND expensive_check() | {dept: "engineering"} | false (no expensive call) | Performance optimization |
| Nested Logic | (A AND B) OR (C AND NOT D) | Various combinations | Truth table results | Complex nesting |
| Missing Attribute | user.clearance == "secret" | {dept: "engineering"} | false (deny by default) | Safe attribute handling |
| Type Coercion | user.salary > 100000 | {salary: "150000"} | true (string to number) | Flexible comparisons |

Resource-based access tests verify that ownership, tenant isolation, and hierarchical permissions work correctly. We create test scenarios with resources owned by different users in different tenants, then verify that access control correctly enforces boundaries. These tests are critical because tenant isolation failures can lead to data breaches.

The resource hierarchy tests create parent-child resource relationships and verify that permissions inherited from parents work correctly. For example, if a user has "read" permission on a project resource, they should automatically have "read" permission on all documents within that project, unless explicitly denied by a more specific policy.

Cross-tenant sharing tests verify that controlled collaboration between tenants works securely. We create resources in tenant A, share them with specific users in tenant B, then verify that only authorized users can access the shared resources and that sharing doesn't inadvertently expose other resources from tenant A.

Audit logger unit tests focus on integrity verification, immutable storage, and performance under load. The integrity verification tests create sequences of audit records, compute hash chains, then verify that tampering with any record breaks the chain verification. These tests are crucial because audit log integrity is often required for regulatory compliance.

| Audit Test Scenario | Setup | Tampering Method | Expected Detection | Compliance Requirement |
|---------------------|-------|------------------|-------------------|------------------------|
| Single Record Tamper | 100 record chain | Modify record 50 decision | Hash verification failure at record 51 | SOX, HIPAA |
| Timestamp Manipulation | Time-ordered records | Change timestamp of record | Chronological ordering violation | GDPR |
| Missing Record | Sequential audit IDs | Delete record from middle | Gap in sequence numbers | Financial regulations |
| Hash Chain Break | Linked hash chain | Corrupt hash pointer | Chain verification failure | All compliance frameworks |

⚠️ **Pitfall: Testing with Production-Like Data**

A common mistake in authorization testing is using overly simplified test data that doesn't reflect real-world complexity. For example, testing role hierarchies with only 2-3 roles when production will have 50+ roles, or testing policies with simple string attributes when production uses complex nested JSON attributes. This leads to tests that pass but don't validate the actual system behavior under realistic conditions. Instead, create test fixtures that mirror production complexity - use realistic role names, permission patterns, and attribute structures. Generate large datasets for performance testing, and include edge cases like extremely long role chains or policies with hundreds of conditions.

### End-to-End Authorization Testing

End-to-end authorization testing validates the complete flow from initial authorization request through policy evaluation to final decision and audit logging. These tests exercise the entire system as a black box, sending realistic authorization requests and verifying that the responses match expected outcomes. Unlike unit tests that focus on individual components, end-to-end tests validate that components collaborate correctly and that the system behaves properly from the client's perspective.

The key challenge in end-to-end testing is creating comprehensive test scenarios that cover the full range of authorization patterns our system supports. We need scenarios for pure RBAC decisions, pure ABAC decisions, mixed RBAC/ABAC decisions, resource ownership decisions, cross-tenant sharing decisions, and error conditions. Each scenario must specify the complete request context including user attributes, resource attributes, environment attributes, and expected decision outcome.

![Policy Simulation and Testing Flow](./diagrams/policy-testing-flow.svg)

Our end-to-end testing framework uses the `TestScenario` structure to define comprehensive test cases. Each scenario includes not just the authorization request and expected decision, but also validation of the decision reasoning, evaluation method used, applied policies, and performance characteristics. This detailed validation ensures that the system isn't just giving the right answer, but giving it for the right reasons.

| Test Scenario Component | Purpose | Example Value | Validation Target |
|------------------------|---------|---------------|-------------------|
| Request Context | Complete authorization request | UserID: "alice", Resource: "document123", Action: "read" | Request processing |
| Expected Decision | Anticipated outcome | Allowed: true, Method: EvaluationRBAC | Decision correctness |
| Applied Policies | Which rules should fire | ["admin-role-policy", "document-owner-policy"] | Policy evaluation |
| Performance Bounds | Acceptable response time | EvaluationTime < 50ms, CacheHit: true | System performance |
| Audit Validation | Expected audit record fields | Decision logged with correct user/resource/action | Compliance requirements |

The RBAC end-to-end test scenarios verify that role-based decisions work correctly across the complete system. We create users with specific role assignments, then send authorization requests for various actions and resources. These tests validate that role hierarchy traversal works correctly, that permission inheritance flows properly from parent roles to child roles, and that the bitmap optimization doesn't introduce correctness bugs.

```
RBAC End-to-End Test Flow:
1. Create tenant with role hierarchy (Admin → Manager → Employee → Viewer)
2. Create users assigned to different roles
3. Create resources with different permission requirements
4. Send authorization requests for each user/resource/action combination
5. Verify decisions match role hierarchy expectations
6. Validate that audit logs capture complete decision context
7. Test performance with concurrent requests from multiple users
```

The ABAC end-to-end scenarios test attribute-based policy evaluation with realistic attribute complexity. We create policies that depend on user attributes (department, clearance level, employment status), resource attributes (classification, owner, creation date), and environment attributes (time of day, IP address, request source). These tests validate that attribute resolution works correctly, that complex condition trees evaluate properly, and that policy combining logic handles conflicting policies appropriately.

Multi-step ABAC scenarios test policy interactions where multiple policies might apply to the same request. For example, a user might have a role-based permission to read documents, but an attribute-based policy that denies access to classified documents after business hours. The system must evaluate all applicable policies and combine them using the deny-overrides logic.

| ABAC Test Scenario | User Attributes | Resource Attributes | Environment Attributes | Policies Applied | Expected Decision |
|---------------------|----------------|--------------------|-----------------------|------------------|-------------------|
| Time-based access | {dept: "hr", clearance: "confidential"} | {classification: "confidential"} | {time: "10:00", day: "Tuesday"} | ["business-hours-policy"] | Allow |
| After-hours denial | {dept: "hr", clearance: "confidential"} | {classification: "confidential"} | {time: "22:00", day: "Saturday"} | ["after-hours-deny-policy"] | Deny |
| IP restriction | {dept: "finance"} | {type: "financial-report"} | {source_ip: "192.168.1.100"} | ["finance-network-policy"] | Allow |
| Remote access block | {dept: "finance"} | {type: "financial-report"} | {source_ip: "203.0.113.50"} | ["finance-network-policy"] | Deny |

Resource-based end-to-end tests validate tenant isolation, resource ownership, and hierarchical permissions. These tests create resources in multiple tenants, assign ownership to different users, then verify that access control boundaries are enforced correctly. We test positive cases (owners can access their resources) and negative cases (users cannot access resources they don't own or aren't shared with them).

Cross-tenant sharing scenarios test the controlled collaboration features. We create resources in one tenant, share them with users in another tenant, then verify that sharing works correctly without breaking tenant isolation. These tests must verify that shared access is limited to explicitly granted permissions and doesn't provide broader access to the sharing tenant's resources.

```
Cross-Tenant Sharing Test Sequence:
1. Create resource R1 in tenant A, owned by user Alice
2. Create user Bob in tenant B (no initial access to R1)
3. Verify Bob cannot access R1 (tenant isolation)
4. Alice creates cross-tenant share granting Bob "read" access to R1
5. Verify Bob can now read R1 but cannot write/delete
6. Verify Bob still cannot access other resources in tenant A
7. Alice revokes the share grant
8. Verify Bob immediately loses access to R1
```

Performance end-to-end tests validate that authorization decisions meet latency and throughput requirements under realistic load. We create scenarios with hundreds of concurrent users making authorization requests, complex role hierarchies with deep nesting, and policies with expensive condition evaluation. These tests identify performance bottlenecks and validate that caching strategies work correctly under load.

The performance tests measure not just average response time but also tail latency (95th and 99th percentiles), cache hit rates, database query patterns, and system resource utilization. We test both steady-state performance and performance during cache warming, policy updates, and role hierarchy changes.

| Performance Test Type | Load Pattern | Metrics Captured | Pass Criteria | Failure Investigation |
|-----------------------|--------------|------------------|---------------|----------------------|
| Steady State | 100 RPS for 10 minutes | p95 latency, cache hit rate | p95 < 50ms, hit rate > 90% | Slow query analysis |
| Cache Warming | Cold start + ramp to 100 RPS | Time to reach optimal performance | < 60 seconds to optimal | Cache population strategy |
| Policy Update | Policy change during 50 RPS load | Authorization consistency | Zero incorrect decisions | Cache invalidation timing |
| Role Hierarchy Change | Add/remove roles during load | Decision correctness | All decisions use updated hierarchy | Consistency guarantees |

Error condition end-to-end tests validate graceful degradation when system components fail. We simulate database failures, cache unavailability, policy evaluation timeouts, and network partitions, then verify that the system fails safely by denying access rather than potentially granting unauthorized permissions. These tests are critical for security because authorization failures must be conservative.

Compliance end-to-end tests validate that audit logging, policy testing, and reporting features work correctly for regulatory requirements. We create scenarios that exercise the complete compliance workflow: authorization decisions are logged with complete context, audit logs maintain integrity through hash chains, and compliance reports accurately reflect system behavior.

⚠️ **Pitfall: Testing Only Positive Cases**

Many authorization tests focus only on cases where access should be granted, neglecting to test denial scenarios thoroughly. This is dangerous because authorization bugs often manifest as over-permissive decisions that grant access inappropriately. Every authorization test scenario should include both positive cases (this access should be allowed) and negative cases (this access should be denied). For cross-tenant scenarios, test that sharing works for intended recipients AND that it doesn't inadvertently expose resources to unintended users. For ABAC policies, test that conditions properly deny access when attributes don't match, not just when they do match.

### Milestone Validation Checkpoints

Each milestone in our authorization system development has specific acceptance criteria and verification steps that validate successful implementation. These checkpoints serve as gates between development phases, ensuring that foundational components work correctly before building dependent functionality. The validation approach combines automated testing, manual verification, and performance benchmarking to provide comprehensive confidence in each milestone's deliverables.

Milestone validation goes beyond simple unit testing by exercising components in realistic scenarios with production-like data volumes and complexity. Each checkpoint includes specific commands to run, expected outputs to verify, behavioral tests to perform manually, and troubleshooting guidance for common issues. This structured approach helps developers identify problems early and provides clear criteria for milestone completion.

The validation checkpoints also serve as regression test suites for subsequent development. As we add new features in later milestones, we re-run earlier milestone validations to ensure that changes don't break existing functionality. This continuous validation approach prevents the common problem where late-stage features inadvertently break early-stage components.

> **Decision: Automated Milestone Validation**
> - **Context**: Manual milestone validation is time-consuming and error-prone. Developers might skip validation steps or miss subtle regressions when adding new features.
> - **Options Considered**:
>   1. Manual validation checklists
>   2. Automated test suites with pass/fail criteria
>   3. Hybrid approach with automated tests plus manual verification
> - **Decision**: Implement automated test suites with clear pass/fail criteria, supplemented by manual verification for user experience aspects
> - **Rationale**: Automation ensures consistent validation and enables regression testing. Manual verification covers aspects that are difficult to automate like error message clarity and debugging experience.
> - **Consequences**: Higher initial effort to create validation automation, but significant time savings and improved reliability for ongoing development.

**Milestone 1: Role & Permission Model Validation**

The first milestone validation focuses on role hierarchy implementation, permission inheritance, and efficient lookup mechanisms. We validate that the `RoleEngine` correctly manages directed acyclic graphs for role relationships, computes transitive closure for inherited permissions, and provides fast permission lookups using bitmap optimization.

The automated validation suite for Milestone 1 includes comprehensive tests for role hierarchy creation, cycle detection, permission inheritance calculation, and performance benchmarks. The test suite creates complex role hierarchies with 50+ roles and 500+ permissions, then validates that all operations complete within acceptable time bounds.

| Validation Test | Test Setup | Expected Outcome | Performance Requirement |
|-----------------|------------|------------------|-------------------------|
| Role Hierarchy Creation | Create 50-role corporate hierarchy | All roles created with proper parent relationships | < 500ms total creation time |
| Cycle Detection | Attempt to create Admin→Manager→Admin cycle | ValidationError with specific cycle path | < 10ms detection time |
| Permission Inheritance | Manager role inherits from both TeamLead and ProjectManager | Union of all parent permissions | < 5ms computation time |
| Bitmap Optimization | Check permissions for user with 10 roles | Correct permission results using bitmap operations | < 1ms lookup time |
| Concurrent Access | 100 threads checking permissions simultaneously | No race conditions, consistent results | No performance degradation |

The manual validation steps for Milestone 1 verify that the role management experience works correctly from an operational perspective. We create roles using the management API, verify that the role hierarchy visualization displays correctly, and test error handling for common mistakes like circular dependencies or invalid permission assignments.

```
Manual Validation Steps for Milestone 1:
1. Start the authorization service with debug logging enabled
2. Create a basic role hierarchy: Admin → Manager → Employee → Viewer
3. Assign permissions to each role (Admin gets all permissions, Viewer gets read-only)
4. Create users and assign them to different roles
5. Verify that GetUserPermissions returns expected permission sets
6. Attempt to create a cycle (Employee → Admin) and verify it's rejected
7. Check that role hierarchy changes propagate to permission lookups immediately
8. Verify that role templates can instantiate common role patterns
```

The performance validation for Milestone 1 establishes baseline performance metrics that later milestones must maintain. We measure permission lookup times for various scenarios: single role users, multi-role users, deeply nested hierarchies, and large permission sets. These benchmarks guide optimization decisions in later development.

**Milestone 2: ABAC Policy Engine Validation**

The second milestone validation tests attribute-based policy evaluation, condition tree processing, and policy combining logic. We validate that the `PolicyEngine` correctly evaluates complex condition expressions, resolves attributes from multiple sources, and combines multiple policy decisions using deny-overrides logic.

The ABAC validation scenarios test increasingly complex policy expressions to ensure the policy evaluation engine handles real-world policy complexity. We start with simple attribute comparisons, progress to nested logical expressions with AND/OR/NOT operations, and culminate with policies that involve multiple attribute sources and expensive condition evaluations.

| ABAC Validation Scenario | Policy Expression | Context Attributes | Expected Decision | Evaluation Time |
|---------------------------|-------------------|-------------------|-------------------|-----------------|
| Simple Attribute | user.department == "engineering" | {department: "engineering"} | Allow | < 1ms |
| Numeric Comparison | user.level >= 5 AND resource.classification <= user.clearance | {level: 7, clearance: 3, classification: 2} | Allow | < 2ms |
| Time-based Policy | current_time BETWEEN "09:00" AND "17:00" | {current_time: "14:30"} | Allow | < 1ms |
| Complex Nested | (user.dept == "hr" OR user.role == "manager") AND resource.sensitivity < 3 | {dept: "engineering", role: "manager", sensitivity: 1} | Allow | < 5ms |
| Multi-source Attributes | user.location == resource.required_location AND environment.secure_network | Multiple attribute sources | Allow | < 10ms |

The policy combining validation tests scenarios where multiple policies apply to the same authorization request. We create conflicting policies (some allowing, some denying access) and verify that the deny-overrides logic works correctly. These tests are critical because policy conflicts can create security vulnerabilities if not handled properly.

```
Policy Combining Test Scenarios:
1. Create allow policy: users in "engineering" can read "documents"  
2. Create deny policy: classified documents cannot be accessed after hours
3. Send request: engineering user accessing classified document at 10:00 AM
4. Expected: Allow (deny policy doesn't apply during business hours)
5. Send request: engineering user accessing classified document at 10:00 PM  
6. Expected: Deny (deny policy overrides allow policy)
7. Verify audit logs show both policies were evaluated
```

The attribute resolution validation tests that attributes are correctly gathered from user profiles, resource metadata, and environment context. We test scenarios where attributes are missing, have unexpected types, or come from slow external sources. The validation ensures that attribute resolution failures result in safe deny decisions rather than system errors.

Performance validation for Milestone 2 focuses on policy evaluation speed, especially for complex condition trees and scenarios with many applicable policies. We establish benchmarks for acceptable policy evaluation time and verify that caching strategies effectively reduce evaluation costs for repeated requests.

**Milestone 3: Resource-Based & Multi-tenancy Validation**

The third milestone validation tests resource ownership, tenant isolation, and cross-tenant sharing mechanisms. We validate that the `ResourceEvaluator` correctly enforces tenant boundaries, that resource hierarchy permissions work properly, and that controlled cross-tenant collaboration doesn't compromise isolation.

Tenant isolation validation creates resources and users in multiple tenants, then attempts various cross-tenant access patterns to verify that isolation is complete. These tests are critical for SaaS applications where data leakage between tenants represents a serious security vulnerability and potential regulatory violation.

| Tenant Isolation Test | Setup | Attempted Access | Expected Result | Security Implication |
|------------------------|--------|------------------|-----------------|----------------------|
| Basic Isolation | Resource R1 in Tenant A, User U1 in Tenant B | U1 tries to read R1 | Deny | Prevents cross-tenant data access |
| User Enumeration | User U1 exists in Tenant A, User U2 in Tenant B | U2 tries to get U1's permissions | Error (user not found in tenant) | Prevents tenant user discovery |
| Role Cross-Reference | Role "Manager" defined in both tenants | Tenant A user tries to use Tenant B's Manager role | Deny | Prevents role privilege escalation |
| Resource Hierarchy | Project P1 in Tenant A contains Document D1 | Tenant B user tries to access D1 directly | Deny | Prevents hierarchy-based attacks |

Cross-tenant sharing validation tests the controlled collaboration features that allow selective resource sharing between tenants. We validate that sharing grants work correctly, that permissions are limited to explicitly granted actions, and that sharing can be revoked immediately.

```
Cross-Tenant Sharing Validation:
1. Create Project P1 in Tenant A owned by Alice
2. Create User Bob in Tenant B  
3. Verify Bob cannot access P1 (baseline isolation)
4. Alice shares P1 with Bob for "read" access with 30-day expiration
5. Verify Bob can read P1 but cannot write/delete
6. Verify Bob cannot access other resources in Tenant A
7. Alice updates share to add "write" permission
8. Verify Bob can now write to P1
9. Alice revokes the share
10. Verify Bob immediately loses all access to P1
```

Resource hierarchy validation tests that parent-child resource relationships work correctly for permission inheritance. We create nested resource structures (organizations contain projects, projects contain documents) and verify that permissions granted on parent resources appropriately flow to child resources.

The performance validation for Milestone 3 focuses on tenant isolation overhead and cross-tenant sharing lookup performance. We measure the additional latency introduced by tenant validation and verify that resource hierarchy traversal scales well with depth and breadth.

**Milestone 4: Audit Logging & Policy Testing Validation**

The fourth milestone validation tests audit logging integrity, policy simulation capabilities, and compliance reporting functionality. We validate that the `AuditLogger` maintains immutable records with cryptographic integrity, that the `PolicyTestFramework` provides accurate policy simulation, and that compliance reports contain complete and accurate information.

Audit integrity validation tests the hash chain implementation by creating sequences of audit records, then attempting various tampering attacks to verify that integrity violations are detected. These tests are essential for regulatory compliance where audit log integrity is often legally required.

| Audit Integrity Test | Attack Method | Detection Method | Expected Outcome | Compliance Impact |
|----------------------|---------------|------------------|------------------|-------------------|
| Record Modification | Change decision from "deny" to "allow" in record 100 | Hash chain validation during compliance report | Integrity failure detected | SOX, HIPAA compliance maintained |
| Record Deletion | Remove audit record from sequence | Sequence number gap detection | Missing record identified | Financial regulation compliance |
| Timestamp Manipulation | Change record timestamp to hide access pattern | Chronological ordering validation | Temporal violation detected | GDPR audit trail requirements |
| Hash Chain Break | Corrupt the previous record hash pointer | Forward hash verification | Chain break identified | All compliance frameworks |

Policy simulation validation tests that the testing framework provides accurate predictions of policy behavior. We create test scenarios with known expected outcomes, run them through the simulation environment, then deploy the policies to production and verify that actual outcomes match simulation predictions.

```
Policy Simulation Validation Process:
1. Create baseline snapshot of current authorization state
2. Define new policy: "All document access requires approval after 6 PM"
3. Create test scenarios covering various times and users
4. Run simulation and capture predicted outcomes
5. Deploy policy to staging environment with identical data
6. Execute same test scenarios in staging
7. Verify that staging results exactly match simulation predictions
8. Deploy to production with confidence in behavior
```

Compliance reporting validation tests that audit data is correctly transformed into regulatory reports. We create authorization scenarios that exercise specific compliance requirements (separation of duties, access reviews, privilege escalation detection), then verify that the generated reports contain accurate and complete information.

The performance validation for Milestone 4 focuses on audit logging overhead and policy simulation speed. We measure the latency impact of audit logging on authorization decisions and verify that policy simulation can handle large-scale "what-if" scenarios within reasonable time bounds.

⚠️ **Pitfall: Incomplete Milestone Validation**

A common mistake is rushing through milestone validation checkpoints to move quickly to the next development phase. This leads to subtle bugs that compound in later milestones and become expensive to fix. For example, if Milestone 1 validation doesn't thoroughly test role hierarchy edge cases, Milestone 2 ABAC policies might interact with role permissions in unexpected ways, and Milestone 3 tenant isolation might not work correctly when users have complex role assignments. Always complete the full validation suite for each milestone before proceeding, and maintain regression testing by re-running earlier milestone validations after each new milestone completion.

### Implementation Guidance

The testing implementation requires a comprehensive framework that supports unit testing, integration testing, end-to-end testing, and milestone validation. The framework must provide test isolation, realistic data fixtures, and automated validation of complex authorization scenarios.

**Technology Recommendations:**

| Testing Component | Simple Option | Advanced Option |
|-------------------|---------------|-----------------|
| Test Framework | Go's built-in testing package | Ginkgo BDD framework |
| Test Data Management | Static JSON fixtures | Testcontainers with PostgreSQL |
| Mocking | Manual interface implementations | GoMock code generation |
| HTTP Testing | httptest package | REST client testing libraries |
| Performance Testing | Go benchmarks | Custom load testing harness |
| Policy Simulation | In-memory test doubles | Dedicated simulation environment |

**Recommended File Structure:**

```
authorization-system/
  internal/
    testing/
      framework/
        test_framework.go         ← Core testing framework
        fixtures.go              ← Test data generation
        assertions.go            ← Custom authorization assertions
      unit/
        role_engine_test.go      ← RoleEngine unit tests
        policy_engine_test.go    ← PolicyEngine unit tests  
        resource_evaluator_test.go ← ResourceEvaluator unit tests
        audit_logger_test.go     ← AuditLogger unit tests
      integration/
        component_integration_test.go ← Component interaction tests
        database_integration_test.go  ← Database integration tests
      e2e/
        authorization_flow_test.go ← End-to-end authorization tests
        performance_test.go       ← Performance validation tests
      milestones/
        milestone1_test.go       ← Milestone 1 validation
        milestone2_test.go       ← Milestone 2 validation
        milestone3_test.go       ← Milestone 3 validation
        milestone4_test.go       ← Milestone 4 validation
  cmd/
    test-runner/
      main.go                   ← Test execution orchestrator
```

**Core Testing Framework Infrastructure:**

```go
package framework

import (
    "context"
    "testing"
    "time"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

// TestFramework provides comprehensive testing capabilities for authorization components
type TestFramework struct {
    authorizer     Authorizer
    roleEngine     *RoleEngine
    policyEngine   *PolicyEngine
    auditLogger    *AuditLogger
    testDB         *TestDatabase
    fixtures       *TestFixtures
}

// NewTestFramework creates isolated testing environment with fresh database and clean state
func NewTestFramework(t *testing.T) *TestFramework {
    // TODO 1: Initialize test database with clean schema
    // TODO 2: Create isolated cache instance for testing
    // TODO 3: Initialize all authorization components with test configuration
    // TODO 4: Load standard test fixtures for common scenarios
    // TODO 5: Set up cleanup hooks to reset state between tests
}

// TestScenarioRunner executes comprehensive authorization test scenarios
type TestScenarioRunner struct {
    framework *TestFramework
    scenarios []TestScenario
    results   []TestResult
}

// TestDatabase provides isolated database for testing with transaction rollback
type TestDatabase struct {
    connectionString string
    currentTx        *Transaction
    initialized      bool
}

// TestFixtures manages realistic test data for authorization scenarios  
type TestFixtures struct {
    users     map[string]*User
    roles     map[string]*Role
    resources map[string]*Resource
    policies  map[string]*Policy
    tenants   map[string]*TenantContext
}

// CreateStandardFixtures generates realistic test data covering common authorization patterns
func (f *TestFixtures) CreateStandardFixtures(ctx context.Context) error {
    // TODO 1: Create multi-tenant organization structure with realistic role hierarchies
    // TODO 2: Generate users with diverse role assignments and attribute profiles
    // TODO 3: Create resource hierarchies with ownership and sharing relationships
    // TODO 4: Define policy sets covering RBAC, ABAC, and resource-based scenarios
    // TODO 5: Establish cross-tenant sharing relationships for collaboration testing
}

// Custom assertion helpers for authorization-specific testing
func AssertAccessGranted(t *testing.T, decision *AuthzDecision, message string) {
    require.NotNil(t, decision, "Authorization decision should not be nil")
    assert.True(t, decision.Allowed, message)
    assert.NotEmpty(t, decision.Method, "Evaluation method should be specified")
}

func AssertAccessDenied(t *testing.T, decision *AuthzDecision, message string) {
    require.NotNil(t, decision, "Authorization decision should not be nil") 
    assert.False(t, decision.Allowed, message)
    assert.NotEmpty(t, decision.Reason, "Denial reason should be provided")
}

func AssertAuditLogged(t *testing.T, logger *AuditLogger, req *AuthzRequest, decision *AuthzDecision) {
    // TODO 1: Query audit log for records matching request parameters
    // TODO 2: Verify audit record contains complete authorization context
    // TODO 3: Validate that integrity hash is correctly computed
    // TODO 4: Check that audit record links to previous record in hash chain
}
```

**Component Unit Testing Templates:**

```go
package unit

import (
    "context"
    "testing"
    "time"
)

// RoleEngine unit tests focusing on hierarchy validation and permission inheritance
func TestRoleEngine_HierarchyValidation(t *testing.T) {
    framework := NewTestFramework(t)
    
    t.Run("DetectSimpleCycle", func(t *testing.T) {
        ctx := context.Background()
        
        // TODO 1: Create roles A, B, C with A→B→C hierarchy
        // TODO 2: Attempt to add C→A relationship creating cycle
        // TODO 3: Verify ValidateRoleHierarchy returns specific cycle error
        // TODO 4: Ensure no changes were persisted when cycle detected
    })
    
    t.Run("ValidateDiamondInheritance", func(t *testing.T) {
        ctx := context.Background()
        
        // TODO 1: Create diamond pattern: Manager→{TeamLead,ProjectLead}→Employee  
        // TODO 2: Assign different permissions to TeamLead and ProjectLead
        // TODO 3: Verify Manager inherits union of both permission sets
        // TODO 4: Check that Employee permissions flow correctly to Manager
        // TODO 5: Validate performance is acceptable for complex hierarchy
    })
}

// PolicyEngine unit tests for condition evaluation and policy combining  
func TestPolicyEngine_ConditionEvaluation(t *testing.T) {
    framework := NewTestFramework(t)
    
    t.Run("EvaluateNestedConditions", func(t *testing.T) {
        ctx := context.Background()
        
        // TODO 1: Create complex condition tree: (A AND B) OR (C AND NOT D)
        // TODO 2: Test all combinations of attribute values (truth table)
        // TODO 3: Verify short-circuit evaluation for performance
        // TODO 4: Test error handling for missing attributes
        // TODO 5: Validate type coercion works correctly (string to number, etc.)
    })
    
    t.Run("PolicyCombiningLogic", func(t *testing.T) {
        ctx := context.Background()
        
        // TODO 1: Create allow policy and conflicting deny policy for same resource
        // TODO 2: Verify deny-overrides logic works correctly
        // TODO 3: Test policy priority ordering affects decision
        // TODO 4: Validate that all applicable policies are recorded in decision
    })
}
```

**End-to-End Testing Framework:**

```go
package e2e

import (
    "context" 
    "net/http"
    "testing"
    "time"
)

// AuthorizationFlowTest validates complete authorization request processing
func TestAuthorizationFlow_CompleteScenarios(t *testing.T) {
    framework := NewTestFramework(t)
    
    scenarios := []TestScenario{
        {
            Name: "RBAC Admin Access",
            Request: &AuthzRequest{
                UserID: "alice",
                TenantID: "acme-corp", 
                Action: "delete",
                Resource: &Resource{ID: "document-123", Type: "document"},
            },
            Expected: &AuthzDecision{
                Allowed: true,
                Method: EvaluationRBAC,
                AppliedPolicies: []string{"admin-role-policy"},
            },
        },
        // TODO: Add 20+ comprehensive scenarios covering all authorization patterns
    }
    
    runner := &TestScenarioRunner{framework: framework, scenarios: scenarios}
    
    // TODO 1: Execute each scenario and capture actual results
    // TODO 2: Compare actual vs expected decisions with detailed reporting
    // TODO 3: Validate performance requirements for each scenario
    // TODO 4: Verify audit logs contain complete context for each decision
    // TODO 5: Test concurrent execution to validate thread safety
}

// PerformanceTest validates authorization system meets latency and throughput requirements
func TestAuthorizationFlow_Performance(t *testing.T) {
    framework := NewTestFramework(t)
    
    t.Run("LatencyUnderLoad", func(t *testing.T) {
        // TODO 1: Generate realistic authorization requests at 100 RPS
        // TODO 2: Measure p95 and p99 latency over 10-minute test
        // TODO 3: Verify cache hit rates achieve target thresholds
        // TODO 4: Validate no memory leaks during sustained load
        // TODO 5: Check that database connection pool remains stable
    })
    
    t.Run("ConcurrentUserScenario", func(t *testing.T) {
        // TODO 1: Simulate 500 concurrent users making authorization requests
        // TODO 2: Verify no race conditions in permission caching
        // TODO 3: Test role hierarchy modifications during concurrent access
        // TODO 4: Validate tenant isolation under concurrent cross-tenant requests
    })
}
```

**Milestone Validation Automation:**

```go
package milestones

import (
    "context"
    "testing"
    "time"
)

// Milestone1Validator ensures Role & Permission Model meets acceptance criteria
func TestMilestone1_RolePermissionModel(t *testing.T) {
    framework := NewTestFramework(t)
    
    t.Run("AcceptanceCriteria", func(t *testing.T) {
        ctx := context.Background()
        
        // TODO 1: Create hierarchical roles with named permission sets
        // TODO 2: Verify role hierarchy DAG validation prevents cycles
        // TODO 3: Test user assignment to multiple roles simultaneously
        // TODO 4: Validate role template system creates preconfigured bundles
        // TODO 5: Benchmark permission lookup performance meets requirements
        
        // Performance validation
        start := time.Now()
        // TODO: Execute 1000 permission lookups
        duration := time.Since(start)
        assert.Less(t, duration.Milliseconds(), int64(100), "Permission lookups must complete within 100ms")
    })
    
    t.Run("PermissionInheritanceValidation", func(t *testing.T) {
        // TODO 1: Create deep role hierarchy (10+ levels)
        // TODO 2: Verify transitive closure computation works correctly
        // TODO 3: Test permission bitmap encoding handles large permission sets
        // TODO 4: Validate copy-on-write semantics for concurrent access
    })
}

// MilestoneCheckpointer provides automated validation and reporting
type MilestoneCheckpointer struct {
    framework *TestFramework
    milestone string
    results   map[string]bool
}

// ExecuteCheckpoint runs comprehensive validation for specified milestone
func (c *MilestoneCheckpointer) ExecuteCheckpoint(ctx context.Context, milestone string) (*MilestoneReport, error) {
    // TODO 1: Load milestone-specific test scenarios
    // TODO 2: Execute all validation tests with detailed reporting
    // TODO 3: Generate performance benchmarks and comparison with requirements
    // TODO 4: Create actionable report with pass/fail status and next steps
    // TODO 5: Archive test results for regression testing
}
```

**Policy Testing and Simulation Infrastructure:**

```go
package testing

import (
    "context"
    "time"
)

// PolicyTestEnvironment provides isolated simulation environment for testing policy changes
type PolicyTestEnvironment struct {
    snapshotID   string
    policies     map[string]*Policy
    testData     *TestFixtures
    simulator    *PolicySimulator
}

// CreateTestSnapshot captures complete authorization state for testing
func (env *PolicyTestEnvironment) CreateTestSnapshot(ctx context.Context, name string) error {
    // TODO 1: Capture all users, roles, permissions, and resources
    // TODO 2: Export policies and their current configurations
    // TODO 3: Store environmental context and attribute values
    // TODO 4: Create immutable snapshot with integrity verification
    // TODO 5: Index snapshot for efficient simulation execution
}

// SimulatePolicyChanges tests policy modifications against captured state
func (env *PolicyTestEnvironment) SimulatePolicyChanges(ctx context.Context, changes []PolicyChange) (*SimulationReport, error) {
    // TODO 1: Apply policy changes to simulation environment
    // TODO 2: Execute comprehensive test scenarios against modified policies  
    // TODO 3: Compare results with baseline behavior from original policies
    // TODO 4: Generate differential report showing impact of changes
    // TODO 5: Identify potential security issues or unintended access changes
}
```

**Debugging and Troubleshooting Helpers:**

```go
// AuthorizationDebugger provides detailed tracing for test debugging
type AuthorizationDebugger struct {
    traceEnabled bool
    logBuffer    []string
}

// TraceAuthorizationDecision provides step-by-step evaluation details
func (d *AuthorizationDebugger) TraceAuthorizationDecision(ctx context.Context, req *AuthzRequest) (*DebugTrace, error) {
    // TODO 1: Capture each step of authorization evaluation process
    // TODO 2: Record attribute resolution, policy matching, condition evaluation
    // TODO 3: Show decision tree traversal and policy combining logic
    // TODO 4: Include performance timing for each evaluation step
    // TODO 5: Generate human-readable explanation of final decision
}
```

**Milestone Checkpoints:**

After implementing each testing milestone, validate with these commands:

```bash
# Unit test validation
go test ./internal/testing/unit/... -v -cover

# Integration test validation  
go test ./internal/testing/integration/... -v -timeout=30s

# End-to-end test validation
go test ./internal/testing/e2e/... -v -timeout=60s

# Milestone validation
go test ./internal/testing/milestones/milestone1_test.go -v
go test ./internal/testing/milestones/milestone2_test.go -v
go test ./internal/testing/milestones/milestone3_test.go -v
go test ./internal/testing/milestones/milestone4_test.go -v

# Performance benchmarks
go test -bench=. ./internal/testing/... -benchmem
```

**Expected Test Output:**
- Unit tests: >95% code coverage with all edge cases passing
- Integration tests: All component interactions working correctly
- E2E tests: All authorization flows completing within latency requirements
- Milestone validation: All acceptance criteria met with performance benchmarks

**Common Issues and Debugging:**
- Test isolation failures: Ensure each test starts with clean database state
- Flaky tests under load: Check for race conditions in concurrent authorization requests  
- Performance test failures: Profile authorization path to identify bottlenecks
- Policy simulation mismatches: Verify test environment exactly mirrors production configuration


## Debugging Guide

> **Milestone(s):** All milestones - this section provides troubleshooting guidance that applies across role management (Milestone 1), policy evaluation (Milestone 2), multi-tenant resource access (Milestone 3), and audit logging (Milestone 4)

Building an authorization system is like constructing a complex security checkpoint at an international airport. Just as airport security has multiple layers - document verification, metal detectors, baggage scanning, and human oversight - an authorization system has multiple components that must work together seamlessly. When something goes wrong, the symptoms might appear at one layer, but the root cause could be buried several layers deeper.

The challenge with debugging authorization systems is that failures often cascade through multiple components, creating misleading symptoms. A user might report "I can't access my file," but the real issue could be a stale cache entry, a misconfigured role hierarchy, a policy evaluation timeout, or even a subtle timing attack. This section provides systematic approaches to diagnose and fix these complex failure scenarios.

Authorization bugs are particularly dangerous because they can either deny legitimate access (causing business disruption) or grant unauthorized access (creating security vulnerabilities). The debugging techniques we'll explore help you quickly identify the root cause while maintaining the security posture of your system.

### Symptom-Cause-Fix Analysis

Authorization systems exhibit predictable failure patterns that manifest as specific symptoms. Understanding these patterns enables systematic troubleshooting rather than random debugging attempts. Each symptom typically maps to a small set of possible root causes, and each cause has specific diagnostic techniques and targeted fixes.

The key insight is that authorization failures follow a decision pipeline: request validation → role resolution → policy evaluation → resource access checks → audit logging. Failures at each stage create distinct symptom signatures, allowing us to quickly narrow down the problematic component.

**Access Denied for Valid Users**

This is the most common authorization complaint and often the most complex to debug because it could originate from multiple system layers.

| Symptom | Likely Cause | Diagnostic Steps | Fix |
|---------|--------------|------------------|-----|
| User gets access denied for resources they should own | Resource ownership not properly established or tenant context missing | Check `Resource.OwnerID` matches `AuthzRequest.UserID`, verify `TenantContext` propagation through middleware chain | Ensure `WithTenantContext` called early in request pipeline, validate resource creation properly sets owner |
| User denied access despite having correct role | Role hierarchy corrupted or permissions not inherited | Run `ValidateRoleHierarchy` for user's tenant, check transitive closure computation results | Rebuild role hierarchy cache, verify DAG structure has no cycles |
| Intermittent access denials for same user/resource | Cache coherence issues or race conditions in concurrent access | Check cache hit rates and invalidation patterns, look for concurrent role modifications | Implement cache versioning with atomic updates, use read-write locks for role modifications |
| Access denied immediately after role assignment | Cache invalidation delay or stale permission bitmaps | Verify cache invalidation triggered after role changes, check `PermissionBitmap` recalculation | Force cache refresh after role modifications, implement immediate invalidation for critical paths |
| Cross-tenant access fails with valid share grant | Tenant isolation incorrectly blocking legitimate shares | Verify `ShareGrant.CrossTenant` flag set, check `ValidateCrossTenantAccess` logic | Ensure cross-tenant middleware recognizes valid share grants, check tenant context doesn't override shares |

**Unauthorized Access Granted**

These failures represent security vulnerabilities and require immediate attention. The debugging approach focuses on identifying permission elevation paths and closing them quickly.

| Symptom | Likely Cause | Diagnostic Steps | Fix |
|---------|--------------|------------------|-----|
| User can access resources in wrong tenant | Tenant isolation bypass or context contamination | Check `TenantContext` propagation, verify database queries include tenant filters | Add mandatory tenant filtering to all resource queries, validate context isolation |
| User has permissions beyond assigned roles | Role hierarchy allowing unintended inheritance | Audit role parent-child relationships, check for privilege escalation paths | Implement principle of least privilege in role design, add permission boundary validation |
| Expired shares still grant access | Share expiration not enforced or clock skew | Check `ShareGrant.ExpiresAt` validation, verify system clock synchronization | Implement strict expiration checks with clock skew tolerance, add regular cleanup of expired shares |
| Policy allows when it should deny | Policy combining logic error or condition evaluation bug | Trace policy evaluation with `TraceAuthorizationDecision`, check deny-overrides implementation | Fix policy combining algorithm, ensure explicit denies always override allows |
| Admin can access all tenants without grants | Global admin role not properly scoped | Check admin role definition and tenant scoping, verify admin context propagation | Implement tenant-scoped admin roles, remove global admin capabilities |

**Performance Degradation**

Authorization performance problems often compound under load, causing cascading failures across the system. These issues require careful analysis of caching strategies and evaluation algorithms.

| Symptom | Likely Cause | Diagnostic Steps | Fix |
|---------|--------------|------------------|-----|
| Authorization requests timeout under load | Policy evaluation complexity or database contention | Measure policy evaluation times, check database connection pool utilization | Implement policy evaluation caching, optimize complex condition expressions |
| Memory usage grows continuously | Cache not evicting entries or permission bitmap leaks | Monitor cache size and eviction rates, check for unclosed database connections | Implement LRU cache eviction, add memory limits to permission bitmaps |
| Cache hit rate drops dramatically | Cache invalidation too aggressive or key space fragmentation | Analyze cache key patterns and invalidation frequency | Optimize cache key design for better locality, implement hierarchical cache invalidation |
| Database queries increase linearly with users | N+1 query problem in permission resolution | Profile database query patterns during authorization checks | Implement batch permission lookups, use JOIN queries instead of multiple round trips |
| Authorization latency spikes randomly | Circuit breaker triggering or degraded dependencies | Check circuit breaker states and dependency health metrics | Implement proper circuit breaker thresholds, add graceful degradation for non-critical dependencies |

**Audit and Compliance Issues**

Audit trail problems often surface during compliance reviews or security investigations, when it's too late for simple fixes. Early detection and systematic validation prevent these critical failures.

| Symptom | Likely Cause | Diagnostic Steps | Fix |
|---------|--------------|------------------|-----|
| Missing audit records for some decisions | Audit logging failures or selective logging | Check audit logger error rates, verify all authorization paths call `LogAuthzDecision` | Add mandatory audit logging to all authorization decision points, implement audit logger circuit breaker |
| Audit integrity verification fails | Hash chain corruption or tampering attempts | Run `VerifyIntegrityChain` for suspected time period, check storage layer integrity | Implement write-once storage for audit logs, add real-time integrity monitoring |
| Audit records missing critical context | Incomplete evaluation context or attribute resolution failures | Check `EvaluationContext` population, verify attribute resolver functionality | Ensure complete context capture before policy evaluation, add required attribute validation |
| Compliance reports show gaps | Audit data retention issues or query logic errors | Verify audit retention policies, check report generation logic | Implement proper audit retention with lifecycle management, validate report query coverage |
| Performance impact from audit logging | Synchronous audit writes blocking authorization | Measure audit logging latency, check audit system resource utilization | Implement asynchronous audit logging with buffering, separate audit storage from operational data |

> **Critical Security Principle**: When debugging authorization issues, always err on the side of denying access. If you can't determine why access was granted, treat it as a potential security vulnerability until proven otherwise. Temporary access denial is better than permanent security breach.

**Policy Configuration Errors**

Policy misconfigurations are subtle and often discovered only when specific combinations of attributes are evaluated. These errors require systematic policy testing and validation.

| Symptom | Likely Cause | Diagnostic Steps | Fix |
|---------|--------------|------------------|-----|
| Policies conflict with unexpected results | Policy priority or combining logic issues | Use `RunSimulation` to test policy interactions, check policy priority assignments | Implement clear policy ordering rules, add conflict detection in policy validation |
| Condition evaluation fails with cryptic errors | Malformed condition expressions or missing attributes | Check condition syntax validation, verify attribute availability in `EvaluationContext` | Implement comprehensive condition parsing validation, add required attribute checks |
| Policy changes break existing access patterns | Insufficient testing before deployment | Run differential policy analysis against historical access patterns | Implement policy testing framework with comprehensive scenario coverage |
| Attribute resolution inconsistent | Attribute caching issues or source system changes | Check attribute resolver cache behavior, verify upstream attribute consistency | Implement attribute versioning with cache invalidation, add attribute source monitoring |
| Environment-based policies fail randomly | Time zone issues or environment attribute instability | Check timestamp handling and timezone configuration, verify environment attribute sources | Standardize on UTC for all time comparisons, implement stable environment attribute resolution |

### Authorization-Specific Debugging

Authorization systems require specialized debugging techniques that go beyond traditional application debugging. The complexity arises from the interaction between multiple evaluation methods (RBAC, ABAC, resource ownership), distributed caching, and the need to trace decision reasoning for security audits.

The mental model for authorization debugging is like forensic investigation. Each authorization decision leaves traces throughout the system - cached permissions, policy evaluation logs, database queries, and audit records. Effective debugging means following these traces to reconstruct the complete decision-making process.

**Decision Tracing and Provenance**

Understanding how the system reached a particular authorization decision requires tracing through multiple layers of evaluation. The `AuthorizationDebugger` provides comprehensive tracing capabilities that capture every step of the decision-making process.

The tracing system works by instrumenting each component in the authorization pipeline. When tracing is enabled, every permission check, policy evaluation, and resource access validation records its inputs, intermediate results, and final decision. This creates a complete audit trail that can be analyzed to understand both correct and incorrect decisions.

| Trace Component | Information Captured | Analysis Purpose |
|----------------|---------------------|------------------|
| Request Validation | `AuthzRequest` fields, validation errors, context setup | Identify malformed requests or missing context |
| Role Resolution | User roles, role hierarchy traversal, permission bitmap computation | Debug role inheritance and permission calculation |
| Policy Evaluation | Matched policies, condition evaluation results, combining logic | Understand policy interactions and attribute-based decisions |
| Resource Access | Ownership checks, tenant validation, hierarchical permissions | Debug resource-level access control and inheritance |
| Cache Operations | Cache hits/misses, invalidation events, key computations | Identify caching issues and performance problems |
| Audit Logging | Decision recording, integrity computation, storage operations | Verify audit trail completeness and integrity |

The `TraceAuthorizationDecision` method provides step-by-step evaluation details for debugging complex authorization scenarios. When a user reports unexpected access behavior, enabling tracing for their specific request reveals exactly where the decision process deviated from expectations.

```go
// Example trace output structure (conceptual - actual implementation in guidance section)
type DebugTrace struct {
    RequestID string
    Steps []TraceStep
    Decision *AuthzDecision
    Performance PerformanceMetrics
}

type TraceStep struct {
    Component string        // "RoleEngine", "PolicyEngine", etc.
    Operation string       // "ResolveUserRoles", "EvaluateCondition", etc.
    Input map[string]interface{}
    Output map[string]interface{}
    Duration time.Duration
    Errors []string
}
```

**Attribute Resolution Chain Debugging**

ABAC policy evaluation depends heavily on attribute resolution from multiple sources. When policies fail unexpectedly, the issue often lies in incomplete or incorrect attribute resolution rather than the policy logic itself.

The attribute resolution process follows a chain: user attributes → resource attributes → environment attributes → action attributes. Each step in the chain can introduce failures or inconsistencies that affect policy evaluation. Debugging requires examining each resolution step and validating the resulting `EvaluationContext`.

| Attribute Type | Common Issues | Debugging Approach |
|---------------|---------------|-------------------|
| User Attributes | Stale profile data, missing custom attributes, tenant membership changes | Check user data freshness, verify attribute source consistency, validate tenant membership status |
| Resource Attributes | Metadata not updated after resource changes, inheritance not computed | Verify resource metadata synchronization, check hierarchical attribute computation |
| Environment Attributes | Time zone issues, network location inconsistencies, request context missing | Standardize time handling, validate environment data collection |
| Action Attributes | Action type not recognized, missing action metadata | Check action registration, verify action type consistency |

The debugging approach involves systematically validating each attribute resolution step. The `ResolveUserAttributes`, `ResolveResourceAttributes`, and `ResolveEnvironmentAttributes` methods should be instrumented to log their inputs and outputs, making it easy to identify where attribute resolution fails or produces unexpected values.

**Permission Inheritance Analysis**

Role hierarchy debugging requires understanding how permissions flow through the DAG structure from parent roles to child roles. When users don't have expected permissions, the issue often lies in the transitive closure computation or role hierarchy corruption.

The `computeTransitiveClosure` algorithm traverses the role hierarchy to determine all permissions available to a user through role inheritance. Debugging this process requires examining the DAG structure, validating the traversal algorithm, and ensuring the resulting permission bitmap accurately represents inherited permissions.

| Inheritance Issue | Symptoms | Diagnostic Steps |
|------------------|----------|-----------------|
| Circular Dependencies | Infinite loops during permission computation | Run `ValidateRoleHierarchy` to detect cycles, examine role parent-child relationships |
| Missing Inheritance | Child roles don't have parent permissions | Trace transitive closure computation, verify DAG traversal includes all paths |
| Excess Permissions | Users have permissions not granted by any role | Check for role hierarchy corruption, verify permission bitmap computation |
| Performance Issues | Permission resolution takes excessive time | Analyze DAG depth and breadth, implement caching for computed closures |

The debugging process involves visualizing the role hierarchy as a graph and tracing permission flow through each edge. Tools for exporting the role hierarchy in graph formats (DOT, GraphML) help identify structural issues that cause inheritance problems.

**Cache Behavior Analysis**

Multi-level caching introduces complex failure modes where stale data at any cache level can cause incorrect authorization decisions. Debugging cache issues requires understanding the cache hierarchy and invalidation dependencies between cached data types.

The authorization system uses several cache levels: in-memory permission bitmaps, role hierarchy caches, policy evaluation results, and resource metadata. Each cache level has different invalidation triggers and consistency requirements. Cache debugging focuses on identifying which cache level contains stale data and why invalidation didn't occur correctly.

| Cache Level | Invalidation Triggers | Debugging Tools |
|-------------|----------------------|-----------------|
| Permission Bitmaps | Role assignments, role hierarchy changes, permission modifications | Check bitmap recalculation timestamps, verify invalidation events |
| Policy Results | Policy changes, user attribute updates, resource modifications | Analyze cache keys and TTL values, check invalidation patterns |
| Role Hierarchy | Role creation/deletion, parent-child relationship changes | Examine hierarchy version numbers, validate DAG structure consistency |
| Resource Metadata | Resource updates, ownership changes, tenant modifications | Check metadata synchronization, verify cache coherence |

**Security Attack Detection**

Authorization systems are targets for various attacks: permission elevation, timing attacks, cache poisoning, and policy manipulation. Debugging security issues requires recognizing attack patterns and implementing detection mechanisms.

The `DetectSecurityAnomaly` method analyzes authorization requests for suspicious patterns that might indicate attack attempts. This includes requests with unusual attribute combinations, attempts to access resources across tenant boundaries, and patterns that suggest automated credential stuffing or privilege escalation attempts.

| Attack Type | Detection Signals | Debugging Response |
|-------------|------------------|-------------------|
| Permission Elevation | Rapid role changes, unusual resource access patterns | Audit role modification history, check for unauthorized admin actions |
| Timing Attacks | Repeated similar requests with response time analysis | Implement consistent response timing, add request rate limiting |
| Cache Poisoning | Unusual cache invalidation patterns, inconsistent decisions | Verify cache integrity, implement cache entry validation |
| Policy Manipulation | Unauthorized policy changes, suspicious policy combinations | Audit policy modification trails, validate policy change authorization |

### Performance Troubleshooting

Authorization performance problems compound quickly under load because authorization checks are typically in the critical path of every user request. A small performance regression in the authorization system affects the entire application's responsiveness.

The mental model for performance troubleshooting is like diagnosing a traffic jam. The visible symptom might be slow response times, but the root cause could be a bottleneck at any point in the authorization pipeline. Systematic performance analysis identifies these bottlenecks and provides targeted optimization strategies.

**Latency Analysis and Optimization**

Authorization latency has several components: request validation, cache lookups, database queries, policy evaluation, and audit logging. Each component contributes to total latency, and optimization requires identifying the dominant contributors under different load patterns.

The performance analysis approach involves measuring each component's contribution to total latency and identifying optimization opportunities. High-performance authorization systems typically target sub-millisecond response times for cached decisions and sub-10ms for uncached complex policy evaluations.

| Latency Component | Typical Contribution | Optimization Strategies |
|------------------|---------------------|------------------------|
| Request Validation | 0.1-0.5ms | Pre-compile validation rules, optimize data structure access |
| Cache Lookups | 0.1-1ms | Implement local caches, optimize cache key computation |
| Database Queries | 2-50ms | Use connection pooling, implement query batching, add read replicas |
| Policy Evaluation | 1-100ms | Cache evaluation results, optimize condition expressions, parallel evaluation |
| Audit Logging | 0.5-5ms | Implement asynchronous logging, batch audit writes |

**Database Performance Optimization**

Authorization systems generate predictable database access patterns that can be optimized through proper indexing, query optimization, and caching strategies. The most common performance bottlenecks occur in role hierarchy queries, permission lookups, and audit record insertion.

Database optimization requires understanding the query patterns generated by different authorization operations. Role-based queries typically involve JOIN operations across user, role, and permission tables. Policy-based queries require complex WHERE clauses based on dynamic attribute values. Resource-based queries involve hierarchical traversal with tenant isolation filters.

| Query Pattern | Performance Impact | Optimization Approach |
|---------------|-------------------|----------------------|
| Role Hierarchy Traversal | High impact on complex hierarchies | Implement materialized path or nested set models, cache computed closures |
| Permission Bitmap Lookups | Medium impact with many roles | Use covering indexes, implement permission caching |
| Resource Access Queries | High impact with deep hierarchies | Denormalize hierarchical data, implement path-based indexing |
| Audit Log Insertion | Medium impact under write load | Use time-series databases, implement batch insertions |
| Cross-Tenant Queries | High impact without proper isolation | Implement tenant-specific indexes, use query hints |

> **Performance Insight**: The biggest performance gains in authorization systems come from reducing database round trips through intelligent caching and batching. A single complex authorization decision should never require more than 2-3 database queries, regardless of the number of roles, policies, or resources involved.

**Cache Hit Rate Optimization**

Cache effectiveness directly impacts authorization performance. Low cache hit rates force expensive database queries and policy evaluations, while high hit rates enable sub-millisecond response times. Optimizing cache hit rates requires understanding access patterns and designing cache keys for optimal locality.

The cache optimization strategy focuses on three types of locality: temporal (recently accessed data), spatial (related data accessed together), and computational (avoiding repeated calculations). Authorization systems exhibit strong temporal locality for user permissions and spatial locality for resource hierarchies.

| Cache Type | Optimization Strategy | Expected Hit Rate |
|-----------|----------------------|-------------------|
| Permission Bitmaps | Group related permissions, implement warm-up strategies | >95% for active users |
| Policy Results | Key on stable attribute combinations, implement negative caching | >80% for common policies |
| Role Hierarchy | Cache computed transitive closures, version-based invalidation | >90% for established hierarchies |
| Resource Metadata | Implement hierarchical caching, batch metadata updates | >85% for accessed resources |

**Memory Usage and Garbage Collection**

Authorization systems with large user bases and complex role hierarchies can consume significant memory through cached permissions, policy objects, and audit buffers. Memory optimization requires balancing cache effectiveness with resource consumption.

Memory usage patterns in authorization systems typically follow power law distributions - a small percentage of users, roles, and resources account for most of the memory consumption. Optimization strategies focus on efficient data structures for common cases while gracefully handling memory pressure from outliers.

| Memory Component | Typical Usage | Optimization Techniques |
|-----------------|---------------|------------------------|
| Permission Bitmaps | 50-200 bytes per user-role combination | Use sparse bitmaps for roles with few permissions, implement compression |
| Cached Policies | 1-10KB per policy object | Implement policy object pooling, lazy load policy conditions |
| Role Hierarchy | 100-500 bytes per role relationship | Use adjacency lists instead of matrices, implement path compression |
| Audit Buffers | Variable based on audit volume | Implement bounded buffers with backpressure, use circular buffers |

**Concurrent Access Optimization**

Authorization systems must handle high concurrency safely while minimizing lock contention. The challenge is protecting shared data structures (role hierarchies, cached permissions, policy objects) while allowing maximum parallelism for read operations.

The concurrency strategy uses read-write locks for data structures that are read frequently but updated rarely (role hierarchies, policies), atomic operations for frequently updated counters (cache statistics, audit metrics), and lock-free data structures for high-throughput operations (permission checks, cache lookups).

| Concurrency Pattern | Use Case | Implementation Strategy |
|--------------------|----------|------------------------|
| Read-Write Locks | Role hierarchy access, policy object updates | Use `sync.RWMutex` for shared data structures |
| Atomic Operations | Cache statistics, performance metrics | Use `sync/atomic` for counters and flags |
| Lock-Free Structures | Permission bitmap access, cache entries | Implement copy-on-write for infrequently updated data |
| Channel-Based Coordination | Audit logging, batch operations | Use buffered channels for producer-consumer patterns |

**Load Testing and Capacity Planning**

Authorization performance under load reveals bottlenecks that aren't visible during development testing. Load testing requires realistic scenarios that match production usage patterns: user distribution, role complexity, policy evaluation frequency, and resource access patterns.

The load testing approach simulates different types of authorization load: steady-state operation with consistent user activity, burst scenarios with rapid user onboarding, and stress scenarios with complex policy evaluations. Each scenario reveals different performance characteristics and optimization opportunities.

| Load Scenario | Test Parameters | Performance Targets |
|---------------|----------------|-------------------|
| Steady State | 1000 RPS, mixed operations | <5ms p99 latency, >95% cache hit rate |
| User Onboarding | 500 new users/min, role assignments | <10ms role assignment, <1s hierarchy update |
| Policy Evaluation | 100 complex policies, deep conditions | <50ms complex evaluation, <10ms cached results |
| Resource Access | Deep hierarchies, cross-tenant sharing | <15ms hierarchical check, <20ms cross-tenant validation |

### Implementation Guidance

This implementation guidance provides practical debugging tools and techniques for building robust authorization systems. The focus is on creating systematic approaches to identify and fix authorization issues quickly.

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Logging Framework | Go's `log/slog` with structured output | `uber-go/zap` with sampling and buffering |
| Metrics Collection | Prometheus Go client with basic counters | OpenTelemetry with distributed tracing |
| Profiling Tools | Built-in `net/http/pprof` endpoints | Continuous profiling with Pyroscope |
| Load Testing | `hey` or `wrk` for basic HTTP load | Custom Go test harnesses with realistic scenarios |
| Database Profiling | Database-specific EXPLAIN ANALYZE | APM tools like DataDog or New Relic |

**Recommended File Structure**

```
project-root/
  internal/
    debug/
      tracer.go              ← Authorization decision tracing
      profiler.go            ← Performance analysis tools  
      diagnostics.go         ← System health checks
      metrics.go             ← Authorization-specific metrics
      testharness.go         ← Load testing framework
    monitoring/
      alerts.go              ← Authorization anomaly detection
      dashboards.go          ← Metrics visualization helpers
      healthcheck.go         ← System health endpoints
  cmd/
    debug-cli/
      main.go               ← CLI tools for debugging authorization
    loadtest/
      main.go               ← Load testing scenarios
  scripts/
    trace-decision.sh       ← Shell scripts for common debugging tasks
    analyze-performance.sh  ← Performance analysis automation
```

**Authorization Debugging Infrastructure**

```go
package debug

import (
    "context"
    "encoding/json"
    "fmt"
    "log/slog"
    "sync"
    "time"
)

// AuthorizationDebugger provides comprehensive tracing and debugging capabilities
// for authorization decisions. Enable tracing to capture detailed evaluation steps.
type AuthorizationDebugger struct {
    traceEnabled bool
    logBuffer    []string
    mu           sync.RWMutex
    logger       *slog.Logger
}

// DebugTrace contains complete evaluation details for a single authorization decision
type DebugTrace struct {
    RequestID     string                 `json:"request_id"`
    UserID        string                 `json:"user_id"`
    TenantID      string                 `json:"tenant_id"`
    Resource      string                 `json:"resource"`
    Action        string                 `json:"action"`
    StartTime     time.Time              `json:"start_time"`
    EndTime       time.Time              `json:"end_time"`
    Steps         []TraceStep            `json:"steps"`
    Decision      *AuthzDecision         `json:"decision"`
    Performance   PerformanceMetrics     `json:"performance"`
    CacheStats    CacheMetrics           `json:"cache_stats"`
}

// TraceStep represents a single step in the authorization evaluation process
type TraceStep struct {
    Component    string                 `json:"component"`
    Operation    string                 `json:"operation"`
    StartTime    time.Time              `json:"start_time"`
    Duration     time.Duration          `json:"duration"`
    Input        map[string]interface{} `json:"input,omitempty"`
    Output       map[string]interface{} `json:"output,omitempty"`
    CacheHit     bool                   `json:"cache_hit"`
    Errors       []string               `json:"errors,omitempty"`
}

// PerformanceMetrics captures timing and resource usage for authorization decisions
type PerformanceMetrics struct {
    TotalDuration    time.Duration `json:"total_duration"`
    DatabaseQueries  int           `json:"database_queries"`
    CacheOperations  int           `json:"cache_operations"`
    PolicyEvaluations int          `json:"policy_evaluations"`
    MemoryAllocated  int64         `json:"memory_allocated"`
}

// TraceAuthorizationDecision provides step-by-step evaluation details for debugging
func (d *AuthorizationDebugger) TraceAuthorizationDecision(
    ctx context.Context, 
    req *AuthzRequest,
) (*DebugTrace, error) {
    // TODO 1: Initialize trace context with unique request ID and timestamps
    // TODO 2: Create trace step for request validation phase
    // TODO 3: Instrument role engine evaluation with input/output capture
    // TODO 4: Trace policy engine evaluation including condition details
    // TODO 5: Record resource access checks and hierarchy traversal
    // TODO 6: Capture cache operations and hit/miss statistics
    // TODO 7: Record final decision with complete reasoning chain
    // TODO 8: Calculate performance metrics including memory usage
    // TODO 9: Serialize trace to structured format for analysis
    return nil, nil
}

// EnableTracing activates detailed logging for all authorization decisions
func (d *AuthorizationDebugger) EnableTracing() {
    // TODO: Set tracing flag and initialize log buffer with size limits
}

// DisableTracing deactivates detailed logging to improve performance
func (d *AuthorizationDebugger) DisableTracing() {
    // TODO: Clear tracing flag and flush any buffered logs
}

// GetTraceHistory returns recent traces for analysis
func (d *AuthorizationDebugger) GetTraceHistory(limit int) []DebugTrace {
    // TODO: Return last N traces from circular buffer
    return nil
}
```

**Performance Monitoring Tools**

```go
package monitoring

import (
    "context"
    "sync"
    "sync/atomic"
    "time"
)

// AuthzMetrics tracks authorization system performance and behavior
type AuthzMetrics struct {
    DecisionCount      atomic.Int64  // Total authorization decisions
    DecisionLatency    atomic.Int64  // Average decision latency in nanoseconds
    CacheHitRate       atomic.Int64  // Cache hit rate as percentage * 100
    DatabaseQueries    atomic.Int64  // Total database queries
    PolicyEvaluations  atomic.Int64  // Total policy evaluations
    SecurityAnomalies  atomic.Int64  // Detected security anomalies
    ErrorCount         atomic.Int64  // Total authorization errors
    
    latencyHistogram   []int64       // Latency distribution buckets
    mu                sync.RWMutex   // Protects histogram updates
}

// RecordDecision updates metrics after each authorization decision
func (m *AuthzMetrics) RecordDecision(decision *AuthzDecision) {
    // TODO 1: Increment decision counter atomically
    // TODO 2: Update latency moving average with new measurement
    // TODO 3: Record cache hit/miss statistics
    // TODO 4: Update latency histogram buckets
    // TODO 5: Check for anomalous patterns in decision timing
}

// GetMetricsSummary returns current performance statistics
func (m *AuthzMetrics) GetMetricsSummary() map[string]interface{} {
    // TODO: Collect all atomic counters and computed statistics
    return map[string]interface{}{
        "decisions_per_second": 0, // Compute from recent decision count
        "average_latency_ms":   0, // Convert from nanoseconds
        "cache_hit_percentage": 0, // Convert from internal representation
        "error_rate":          0, // Compute error percentage
    }
}

// AnomalyDetector identifies unusual patterns in authorization behavior
type AnomalyDetector struct {
    baselineMetrics map[string]float64
    thresholds     map[string]float64
    recentActivity []AuthzDecision
    mu             sync.RWMutex
}

// DetectSecurityAnomaly analyzes authorization requests for suspicious patterns
func (a *AnomalyDetector) DetectSecurityAnomaly(
    ctx context.Context, 
    req *AuthzRequest,
) []SecurityAnomaly {
    // TODO 1: Check for unusual tenant access patterns
    // TODO 2: Detect rapid permission escalation attempts
    // TODO 3: Identify timing attack patterns in repeated requests
    // TODO 4: Flag suspicious cross-tenant access attempts
    // TODO 5: Check for automated credential stuffing patterns
    return nil
}
```

**Load Testing Framework**

```go
package loadtest

import (
    "context"
    "math/rand"
    "sync"
    "time"
)

// LoadTestFramework provides realistic authorization load testing scenarios
type LoadTestFramework struct {
    authorizer    Authorizer
    testUsers     []*User
    testResources []*Resource
    scenarios     []TestScenario
    results       LoadTestResults
    mu            sync.RWMutex
}

// LoadTestResults captures performance under different load patterns
type LoadTestResults struct {
    TotalRequests    int64
    SuccessfulRequests int64
    FailedRequests   int64
    AverageLatency   time.Duration
    P95Latency      time.Duration
    P99Latency      time.Duration
    ThroughputRPS   float64
    ErrorsByType    map[string]int64
}

// RunSteadyStateLoad simulates normal operational load patterns
func (f *LoadTestFramework) RunSteadyStateLoad(
    ctx context.Context,
    targetRPS int,
    duration time.Duration,
) (*LoadTestResults, error) {
    // TODO 1: Generate realistic user and resource combinations
    // TODO 2: Maintain target RPS with controlled request pacing
    // TODO 3: Mix different authorization request types (RBAC, ABAC, resource)
    // TODO 4: Collect latency measurements for percentile calculations
    // TODO 5: Monitor system resources during load test execution
    // TODO 6: Detect performance degradation and error rate increases
    return nil, nil
}

// RunBurstLoad simulates sudden traffic spikes
func (f *LoadTestFramework) RunBurstLoad(
    ctx context.Context,
    baseRPS int,
    burstMultiplier int,
    burstDuration time.Duration,
) (*LoadTestResults, error) {
    // TODO 1: Start with steady baseline load
    // TODO 2: Rapidly increase to burst rate over short ramp period
    // TODO 3: Maintain burst rate for specified duration
    // TODO 4: Monitor cache hit rates and database performance
    // TODO 5: Measure recovery time back to baseline performance
    return nil, nil
}

// GenerateRealisticRequests creates authorization requests matching production patterns
func (f *LoadTestFramework) GenerateRealisticRequests(count int) []*AuthzRequest {
    // TODO 1: Use weighted random selection for user/resource combinations
    // TODO 2: Include realistic attribute distributions in requests
    // TODO 3: Generate cross-tenant requests based on sharing patterns
    // TODO 4: Include edge cases like expired shares and invalid tenants
    return nil
}
```

**Milestone Checkpoints**

**After Milestone 1 (Role & Permission Engine)**:
- Run `go test -v ./internal/roles/... -bench=.` to verify role hierarchy performance
- Load test role assignment with `./cmd/loadtest/main.go -scenario=role-assignment -users=1000`
- Expected behavior: Role assignments complete in <10ms, hierarchy validation passes
- Debug checklist: Verify DAG validation prevents cycles, permission inheritance works correctly

**After Milestone 2 (ABAC Policy Engine)**:
- Test policy evaluation performance with `./cmd/debug-cli/main.go -trace-policy=complex-policy-id`
- Expected behavior: Complex policies evaluate in <50ms, attribute resolution succeeds
- Debug checklist: Condition evaluation traces show correct attribute values, policy combining follows deny-overrides

**After Milestone 3 (Resource & Multi-tenancy)**:
- Validate tenant isolation with `./scripts/test-tenant-isolation.sh`
- Expected behavior: Cross-tenant access properly blocked, resource hierarchy inheritance works
- Debug checklist: Tenant context propagation complete, resource ownership enforced

**After Milestone 4 (Audit & Testing)**:
- Verify audit integrity with `./cmd/debug-cli/main.go -verify-audit-chain -start=yesterday`
- Expected behavior: All decisions logged, integrity chain validates, policy simulation works
- Debug checklist: Audit logs immutable, simulation environment isolated from production

**Debugging Symptom Checklist**

When encountering authorization issues, follow this systematic diagnostic approach:

1. **Capture the exact error**: What did the user try to do? What response did they get?
2. **Enable decision tracing**: Use `TraceAuthorizationDecision` for the specific request
3. **Check system health**: Verify database connectivity, cache hit rates, circuit breaker states
4. **Validate request context**: Ensure tenant context, user attributes, and resource metadata are correct
5. **Trace through the decision pipeline**: Follow the request through validation, role resolution, policy evaluation, resource checks
6. **Check for timing issues**: Verify clock synchronization, cache TTL values, token expiration
7. **Validate audit trail**: Ensure the decision was properly logged with complete context

This systematic approach ensures you quickly identify the root cause rather than making random changes that might mask the underlying issue.


## Future Extensions

> **Milestone(s):** All milestones - this section explores potential enhancements and demonstrates how the current design accommodates future growth across role management (Milestone 1), policy evaluation (Milestone 2), multi-tenancy (Milestone 3), and audit logging (Milestone 4)

### Advanced Authorization Features

Building an advanced authorization system is like designing a city's infrastructure - while the basic roads and utilities meet current needs, the smart city planners design the foundation to accommodate future growth like autonomous vehicles, smart traffic systems, and underground pneumatic delivery networks. Similarly, our authorization system's modular architecture enables sophisticated extensions that weren't part of the original requirements but can be seamlessly integrated as organizational security needs evolve.

#### Machine Learning-Enhanced Policy Engine

The future of authorization lies in systems that learn from access patterns and automatically adapt to organizational behavior. Think of machine learning policies as having a security consultant who observes how people actually work and gradually refines the rule book based on real usage patterns, rather than relying solely on predefined organizational charts.

Our current `PolicyEngine` design naturally accommodates ML enhancement through its `AttributeResolver` interface. The attribute resolution chain can be extended to include ML-derived attributes that represent risk scores, anomaly indicators, and behavioral patterns. This enhancement leverages the existing policy evaluation infrastructure while adding intelligent decision support.

**Machine Learning Policy Architecture:**

| Component | Responsibility | Interface Extension |
|-----------|---------------|-------------------|
| `MLAttributeResolver` | Provides ML-derived user risk scores and behavioral patterns | Extends `AttributeResolver` with historical access analysis |
| `BehaviorAnalyzer` | Analyzes access patterns to detect normal vs anomalous behavior | Integrates with existing `AuditLogger` for training data |
| `RiskScoreCalculator` | Computes real-time risk scores based on request context | Uses `EvaluationContext` attributes for feature engineering |
| `AdaptivePolicyEngine` | Dynamically adjusts policy thresholds based on ML feedback | Extends `PolicyEngine` with adaptive rule modification |
| `MLTrainingPipeline` | Processes audit logs to train and update ML models | Consumes `AuditRecord` streams for continuous learning |

The ML enhancement follows a **risk-augmented evaluation** pattern where traditional RBAC and ABAC decisions are combined with ML-derived risk assessments:

1. The `MLAttributeResolver` analyzes the user's recent access patterns, comparing against their historical baseline and peer group behavior
2. It computes a risk score based on factors like unusual time-of-day access, geographic anomalies, resource access deviations, and privilege escalation attempts
3. The risk score becomes an additional attribute in the `EvaluationContext`, allowing policies to incorporate ML insights: `user.risk_score < 0.3 AND resource.sensitivity = "high"`
4. The `AdaptivePolicyEngine` tracks policy decision outcomes and gradually adjusts risk thresholds based on false positive rates and security incidents
5. The system learns normal patterns for different user roles and resource types, creating personalized behavioral baselines that improve over time

> **Decision: Machine Learning Integration Strategy**
> - **Context**: Traditional rule-based policies struggle with evolving threat patterns and can't adapt to organizational behavior changes without manual updates
> - **Options Considered**: 
>   1. Replace existing policy engine with pure ML system
>   2. Add ML as parallel evaluation track with manual reconciliation
>   3. Integrate ML as enhanced attribute source within existing policy framework
> - **Decision**: Integrate ML as enhanced attribute source within existing policy framework
> - **Rationale**: This preserves explainability and auditability of authorization decisions while adding intelligent adaptation. Organizations can gradually adopt ML features without abandoning their existing policy investments
> - **Consequences**: Enables sophisticated behavioral analysis while maintaining compliance requirements for explainable decisions

#### Dynamic Attribute Resolution

Modern organizations need authorization systems that can incorporate real-time data from multiple sources - think of dynamic attributes like having a security system that not only checks your ID badge but also considers current building occupancy, your recent conference room bookings, whether you're traveling, and if your team has a deadline today.

The current `AttributeResolver` interface provides a foundation for dynamic attribute integration. Future enhancements can incorporate real-time data sources, external APIs, and contextual information that changes throughout the day.

**Dynamic Attribute Sources:**

| Attribute Source | Example Attributes | Integration Pattern |
|-----------------|-------------------|-------------------|
| Calendar Integration | `user.has_meeting_in_room`, `user.is_traveling`, `user.working_hours` | OAuth to calendar APIs with cached results |
| Device Context | `device.is_corporate_managed`, `device.security_posture`, `device.location` | Device management API integration |
| Network Context | `network.is_corporate`, `network.security_level`, `network.geographic_region` | Real-time network security assessments |
| Project Context | `project.phase`, `project.team_members`, `project.deadline_proximity` | Project management system integration |
| Workload Context | `service.health_status`, `service.maintenance_window`, `service.load_level` | Infrastructure monitoring integration |

The dynamic attribute system employs a **tiered resolution strategy** with performance and reliability guarantees:

1. **Fast Local Attributes**: User roles, resource ownership, and tenant membership resolve from local cache in under 5ms
2. **Cached External Attributes**: Calendar data, device status, and project information refresh every 15-30 minutes and serve from local cache
3. **Real-time Critical Attributes**: Network security posture and device compliance status query external systems with 100ms timeout and fallback values
4. **Best-effort Contextual Attributes**: Project deadlines and workload status enhance decisions when available but don't block authorization on failure

The attribute resolution chain uses a **circuit breaker pattern** for external dependencies, ensuring that authorization decisions remain fast and reliable even when external systems are unavailable. Failed external attributes trigger fallback to safe default values rather than blocking legitimate access.

#### Advanced Delegation Models

Future authorization requirements include sophisticated delegation patterns where users can temporarily grant their permissions to others, similar to how a manager might give their access card to a team member to retrieve files from a secured area, but with precise controls over what can be accessed and for how long.

Our existing `ShareGrant` mechanism provides the foundation for advanced delegation by extending the cross-tenant sharing model to include intra-tenant delegation, hierarchical delegation chains, and time-limited privilege escalation.

**Delegation Model Extensions:**

| Delegation Type | Use Case | Implementation Approach |
|----------------|----------|------------------------|
| **Temporary Privilege Elevation** | Junior developer needs senior-level access for emergency bug fix | Extend `ShareGrant` with privilege escalation flags and approval workflows |
| **Hierarchical Delegation** | VP delegates signing authority to directors, who can sub-delegate to managers | Chain multiple `ShareGrant` records with delegation depth limits |
| **Conditional Delegation** | Access granted only during specific conditions (on-call rotation, project deadlines) | Integrate delegation with dynamic attributes and temporal conditions |
| **Supervised Delegation** | High-privilege actions require real-time approval from delegator | Extend delegation with synchronous approval notifications |
| **Revocable Delegation** | Immediate revocation of all delegated permissions across organization | Centralized delegation registry with instant invalidation capabilities |

Advanced delegation introduces **delegation provenance tracking** where every delegated action maintains a complete chain of authorization from the original privilege holder through all delegation steps. This audit trail ensures accountability even when permissions flow through multiple delegation levels:

```
Original: user:alice role:security-admin
Delegated-To: user:bob reason:"incident-response" valid-until:"2024-01-15T09:00:00Z"
Sub-Delegated-To: user:charlie reason:"weekend-coverage" valid-until:"2024-01-14T18:00:00Z"
Action: read resource:security-logs/incident-2024-001 decision:allow
```

### Scale and Performance Improvements

Scaling an authorization system is like designing a highway system for a growing city - the basic traffic flows work fine when there are few cars, but as the population explodes, you need sophisticated traffic management, multiple lanes, express routes for frequent travelers, and intelligent routing that adapts to real-time conditions.

#### Distributed Evaluation Engine

As organizations grow from hundreds to hundreds of thousands of users, centralized policy evaluation becomes a bottleneck. The future architecture distributes evaluation across multiple nodes while maintaining consistency and correctness guarantees.

The distributed evaluation strategy leverages our existing component interfaces by deploying multiple `PolicyEngine` instances with sophisticated coordination mechanisms:

**Distributed Architecture Components:**

| Component | Responsibility | Scaling Characteristics |
|-----------|---------------|------------------------|
| `DistributedPolicyEngine` | Coordinates evaluation across multiple nodes with consistent results | Handles 100K+ req/sec with sub-50ms latency |
| `PolicyShardManager` | Distributes policy evaluation based on tenant, user, or resource sharding | Automatically rebalances load across evaluation nodes |
| `ConsensusCoordinator` | Ensures policy updates propagate consistently across all evaluation nodes | Implements Raft consensus for policy state synchronization |
| `EvaluationLoadBalancer` | Routes authorization requests to optimal evaluation nodes | Uses consistent hashing for cache locality optimization |
| `GlobalCacheCoordinator` | Maintains cache coherence across distributed evaluation nodes | Implements cache invalidation with vector clocks |

The distributed system employs **evaluation locality optimization** where authorization requests route to nodes that likely have relevant cached data:

1. **Tenant Affinity**: Requests for a specific tenant preferentially route to nodes that have recently evaluated policies for that tenant
2. **User Pattern Recognition**: Frequent user-resource combinations create sticky routing to leverage warm caches
3. **Policy Locality**: Nodes specialize in particular policy types (RBAC vs ABAC) to optimize evaluation engine tuning
4. **Geographic Distribution**: Authorization nodes deploy close to application servers to minimize network latency

The distributed architecture maintains **strong consistency guarantees** for policy updates while allowing eventual consistency for performance-optimized caching:

- Policy changes propagate through consensus protocol ensuring all nodes see updates in the same order
- Authorization decisions remain consistent across all nodes during policy transitions
- Cache invalidation coordinates globally to prevent stale policy evaluation
- Failed nodes recover complete state from healthy peers without impacting service availability

> **Decision: Distributed vs Centralized Evaluation**
> - **Context**: Single authorization node becomes bottleneck at 10K+ requests/second with complex ABAC policies requiring 50-100ms evaluation time
> - **Options Considered**:
>   1. Vertical scaling with more powerful hardware and aggressive caching
>   2. Horizontal scaling with stateless evaluation nodes and shared policy storage
>   3. Full distribution with autonomous nodes and eventual consistency
> - **Decision**: Horizontal scaling with stateless evaluation nodes and shared policy storage
> - **Rationale**: Provides linear scalability while maintaining strong consistency for policy updates. Stateless nodes simplify deployment and recovery while shared storage ensures all nodes evaluate against identical policy sets
> - **Consequences**: Enables scaling to millions of authorization decisions per second while maintaining audit compliance and policy consistency requirements

#### Streaming Policy Updates

Traditional authorization systems update policies through batch deployments that create brief service disruptions. Future systems need **hot policy updates** that stream changes to all evaluation nodes without interrupting ongoing authorization decisions.

The streaming update system extends our existing `PolicyStore` interface with real-time change propagation:

**Streaming Update Architecture:**

| Component | Responsibility | Performance Characteristics |
|-----------|---------------|----------------------------|
| `PolicyStreamProcessor` | Processes incremental policy changes and computes differential updates | Handles 1000+ policy changes/second with sub-second propagation |
| `ChangeLogManager` | Maintains ordered log of all policy modifications with vector timestamps | Provides causal ordering guarantees for dependent policy changes |
| `LiveEvaluationEngine` | Seamlessly switches between policy versions mid-evaluation | Zero-downtime policy updates with rollback capabilities |
| `ConflictResolver` | Handles simultaneous policy updates from multiple administrators | Implements operational transformation for concurrent editing |
| `UpdateValidator` | Validates policy changes before propagation to prevent invalid states | Catches policy conflicts and circular dependencies before deployment |

The streaming system implements **version-aware evaluation** where each authorization request evaluates against a consistent policy snapshot:

1. **Request Timestamping**: Each authorization request receives a logical timestamp ensuring it evaluates against policies active at request time
2. **Policy Versioning**: Policy changes create new versions rather than modifying existing policies, enabling atomic transitions
3. **Graceful Migration**: In-flight authorization requests complete using their original policy version while new requests use updated policies
4. **Rollback Support**: Policy changes include rollback metadata enabling instant reversion if issues are detected
5. **Dependency Tracking**: Related policy changes deploy as atomic units preventing inconsistent intermediate states

**Streaming Update Flow:**

1. Administrator submits policy change through management interface with change justification and impact assessment
2. `UpdateValidator` analyzes change for conflicts, circular dependencies, and potential security impacts
3. `ChangeLogManager` assigns logical timestamp and causal ordering to ensure dependent changes deploy in correct sequence
4. `PolicyStreamProcessor` computes minimal differential update and broadcasts to all evaluation nodes
5. Each evaluation node applies update atomically, switching to new policy version only after successful validation
6. `LiveEvaluationEngine` continues processing requests throughout update process with appropriate version selection

#### Horizontal Scaling Patterns

Scaling authorization infrastructure requires sophisticated partitioning strategies that maintain security boundaries while enabling independent scaling of different system components.

The horizontal scaling design leverages **microservice decomposition** of authorization concerns:

**Service Decomposition Strategy:**

| Service | Scaling Independence | Resource Requirements |
|---------|-------------------|----------------------|
| **Role Resolution Service** | Scales with organization size and role complexity | CPU-intensive for transitive closure computation |
| **Policy Evaluation Service** | Scales with request volume and policy complexity | Memory-intensive for policy cache and attribute resolution |
| **Resource Access Service** | Scales with resource count and ownership patterns | I/O-intensive for resource metadata and hierarchy traversal |
| **Audit Processing Service** | Scales with audit volume and compliance requirements | Storage-intensive for immutable audit log management |
| **Attribute Resolution Service** | Scales with external system integrations | Network-intensive for real-time attribute fetching |

Each service implements **independent scaling policies** based on its specific performance characteristics:

- **Role Resolution**: Caches role hierarchies aggressively since they change infrequently, scales horizontally for read-heavy workloads
- **Policy Evaluation**: Partitions by tenant or user segments, uses specialized hardware for CPU-intensive ABAC evaluation
- **Resource Access**: Shards by resource type or tenant, optimizes for hierarchical queries and ownership lookups
- **Audit Processing**: Uses append-only storage with time-based partitioning, scales write throughput independently from query performance
- **Attribute Resolution**: Implements circuit breakers and fallback values, scales based on external dependency requirements

The decomposed architecture enables **selective optimization** where each service can use different technologies, deployment patterns, and scaling strategies:

```
Role Resolution: In-memory graph database for hierarchy traversal
Policy Evaluation: Specialized rule engines with JIT compilation
Resource Access: Document database optimized for hierarchical queries
Audit Processing: Time-series database with immutable guarantees
Attribute Resolution: Async message queues with timeout handling
```

### External System Integration

Modern authorization systems operate within complex organizational ecosystems where identity information comes from multiple sources, policies are defined in various tools, and access decisions must integrate with existing security infrastructure. Think of external integration like connecting a new smart home security system to existing door locks, cameras, and alarm systems - it needs to work with what's already there while providing enhanced capabilities.

#### SAML and OIDC Federation

Enterprise organizations require authorization systems that integrate seamlessly with existing identity providers through standard protocols. Our system's `User` and `TenantMembership` models provide the foundation for federation by treating external identity providers as authoritative sources for user attributes and group memberships.

**Federation Integration Architecture:**

| Component | Responsibility | Protocol Support |
|-----------|---------------|------------------|
| `FederationManager` | Coordinates identity mapping between external IdPs and internal user model | SAML 2.0, OpenID Connect, LDAP |
| `AttributeMapper` | Transforms external user attributes into internal policy evaluation context | Custom mapping rules with fallback values |
| `GroupSynchronizer` | Maps external groups and roles to internal tenant memberships and roles | Bi-directional sync with conflict resolution |
| `SessionCoordinator` | Manages federated session lifecycle and single sign-out propagation | Session federation across multiple domains |
| `TrustManager` | Validates external assertions and manages trust relationships with IdPs | Certificate validation and trust chain verification |

The federation system implements **claim-based attribute resolution** where external identity assertions become first-class attributes in policy evaluation:

1. **Identity Assertion Processing**: External SAML assertions or OIDC tokens provide user identity and group memberships
2. **Attribute Transformation**: External attributes map to internal policy attributes through configurable transformation rules
3. **Role Mapping**: External groups and roles map to internal `Role` assignments through declarative mapping configurations
4. **Context Enrichment**: Federated attributes enhance `EvaluationContext` with external organizational information
5. **Session Correlation**: Authorization decisions correlate with federated session state for single sign-out support

**Federation Attribute Mapping:**

| External Source | External Attribute | Internal Attribute | Mapping Rule |
|----------------|-------------------|-------------------|--------------|
| Active Directory | `memberOf` groups | `user.departments` | Extract department from group DN |
| SAML IdP | `urn:oid:1.2.840.113556.1.4.1` custom attribute | `user.security_clearance` | Direct mapping with validation |
| OIDC Provider | `groups` claim | `tenant_memberships` | Map groups to tenant roles |
| LDAP Directory | `title` attribute | `user.job_title` | Normalize title format |
| HR System | `manager` DN | `user.reports_to` | Extract manager user ID |

The federation architecture supports **multi-IdP scenarios** where users may have accounts across different identity providers but need unified authorization within the organization:

- **Identity Linking**: Users can link multiple external identities to single internal user account
- **Attribute Aggregation**: Policy evaluation considers attributes from all linked identity sources
- **Trust Level Differentiation**: Different IdPs provide different trust levels affecting authorization decisions
- **Fallback Policies**: Authorization continues with reduced privileges if primary IdP is unavailable

> **Decision: Federation Integration Strategy**
> - **Context**: Organizations use multiple identity providers (corporate AD, cloud IdP, partner federations) but need unified authorization decisions
> - **Options Considered**:
>   1. Require single authoritative IdP with all identity information
>   2. Build identity aggregation layer that merges multiple IdP data
>   3. Treat external IdPs as attribute sources within existing policy framework
> - **Decision**: Treat external IdPs as attribute sources within existing policy framework
> - **Rationale**: Leverages existing attribute resolution and policy evaluation infrastructure while supporting complex multi-IdP scenarios. Avoids creating single point of failure in identity management
> - **Consequences**: Enables flexible federation patterns while maintaining policy consistency, but requires careful attribute mapping and conflict resolution

#### External Policy Stores

Large organizations often have existing policy management systems, compliance frameworks, and regulatory requirements that define authorization policies outside the application authorization system. The future architecture supports **policy federation** where external policy stores contribute to authorization decisions.

**External Policy Integration:**

| Policy Source | Policy Format | Integration Pattern | Update Mechanism |
|--------------|---------------|-------------------|------------------|
| **Open Policy Agent (OPA)** | Rego policy language | Policy Engine extension | Real-time policy sync via API |
| **AWS IAM Policies** | JSON policy documents | Policy translation layer | Periodic sync with change detection |
| **Compliance Frameworks** | Regulatory rule sets | Compliance policy overlay | Quarterly compliance audits |
| **Risk Management Systems** | Risk-based access rules | Dynamic attribute enhancement | Real-time risk score integration |
| **Legacy Authorization Systems** | Custom policy formats | Policy translation and migration | Gradual migration with dual-evaluation |

The external policy integration extends our `PolicyEngine` through a **federated evaluation pattern**:

1. **Policy Discovery**: System discovers available external policy stores through service registry or configuration
2. **Policy Translation**: External policy formats translate to internal `Policy` representations or evaluation extensions
3. **Evaluation Coordination**: Authorization requests trigger evaluation across multiple policy sources with conflict resolution
4. **Result Aggregation**: Multiple policy decisions combine using configurable combining algorithms (deny-overrides, permit-overrides, first-applicable)
5. **Audit Correlation**: Audit logs capture which external policy stores contributed to final authorization decision

**Policy Translation Framework:**

The translation framework handles impedance mismatches between external policy formats and internal evaluation engines:

| Challenge | Solution | Implementation |
|-----------|----------|----------------|
| **Different Policy Languages** | Abstract Syntax Tree translation | Parse external format to common AST, then generate internal policies |
| **Conflicting Decisions** | Configurable combining algorithms | Policy metadata specifies precedence and combining rules |
| **Performance Differences** | Adaptive evaluation strategies | Cache external policy results, fallback to local evaluation on timeout |
| **Update Synchronization** | Event-driven policy sync | External systems push policy changes through webhook notifications |
| **Audit Requirements** | Policy provenance tracking | Audit records include complete policy source chain for compliance |

#### Multi-System Federation

Complex organizations require authorization decisions that consider context from multiple systems - HR systems for organizational hierarchy, project management tools for current assignments, compliance systems for regulatory requirements, and security tools for risk assessment.

The multi-system federation extends our attribute resolution architecture with **ecosystem-aware authorization**:

**Federation Architecture Components:**

| Component | Integration Scope | Performance Requirements |
|-----------|------------------|-------------------------|
| `EcosystemConnector` | Manages connections to external systems with authentication and rate limiting | 100+ external system integrations |
| `DataFusionEngine` | Correlates and merges data from multiple sources with conflict resolution | Sub-100ms attribute resolution across systems |
| `SchemaMapper` | Handles data format differences and semantic mapping between systems | Support 50+ different data schemas |
| `ConsistencyManager` | Ensures authorization decisions remain stable despite external system changes | Eventually consistent with conflict-free replicated data types |
| `FederationOrchestrator` | Coordinates complex authorization workflows across multiple systems | Workflow completion in under 5 seconds |

The federation system implements **context correlation patterns** that intelligently combine information from multiple sources:

1. **Identity Correlation**: Links user identities across systems using employee IDs, email addresses, or federated identity tokens
2. **Temporal Correlation**: Considers time-based context like current project assignments, temporary roles, or seasonal access patterns
3. **Hierarchical Correlation**: Maps organizational hierarchies from HR systems to authorization roles and delegation patterns
4. **Project Correlation**: Links project team memberships to resource access requirements and collaboration permissions
5. **Compliance Correlation**: Applies regulatory requirements from compliance systems as policy constraints and audit requirements

**Multi-System Authorization Flow:**

1. **Request Enrichment**: Authorization request triggers attribute gathering from relevant external systems based on request context
2. **Parallel Resolution**: Multiple systems provide attributes concurrently with timeout and fallback handling
3. **Data Fusion**: Conflicting attributes from multiple sources resolve using configured precedence rules and recency weighting
4. **Policy Evaluation**: Enriched context evaluates against policies that may reference attributes from any connected system
5. **Decision Correlation**: Final authorization decision includes provenance information showing which external systems influenced the outcome

The federation architecture supports **gradual ecosystem integration** where organizations can connect external systems incrementally without disrupting existing authorization patterns:

- **System Discovery**: Automatic discovery of available external systems through service registries and API catalogs
- **Incremental Integration**: New external systems add additional context without requiring policy rewrites
- **Fallback Graceful**: Authorization continues with reduced context if external systems are temporarily unavailable
- **Migration Support**: Side-by-side operation during migration from legacy authorization systems

### Implementation Guidance

This implementation guidance provides the foundation for extending the authorization system with advanced features, focusing on the plugin architecture and extension points that enable future enhancements without major system redesign.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|--------------|----------------|
| **ML Integration** | Scikit-learn with batch processing | TensorFlow Serving with real-time inference |
| **Distributed Coordination** | Redis with pub/sub | Apache Kafka with stream processing |
| **External Integration** | HTTP REST APIs with polling | gRPC with streaming and service mesh |
| **Policy Federation** | JSON policy translation | Open Policy Agent with Rego |
| **Attribute Resolution** | Cached HTTP clients | GraphQL federation gateway |
| **Stream Processing** | Go channels with workers | Apache Pulsar with persistent streams |

#### Recommended File Structure

The extension architecture organizes around plugin patterns and interface extensions:

```
project-root/
  cmd/
    server/main.go                    ← main authorization server
    ml-trainer/main.go               ← ML model training service
    
  internal/
    authorizer/                      ← core authorization engine (existing)
      authorizer.go
      
    extensions/                      ← extension framework
      registry.go                    ← plugin registry and lifecycle
      interfaces.go                  ← extension point interfaces
      
    ml/                             ← machine learning extensions
      attribute_resolver.go          ← ML-enhanced attribute resolution
      behavior_analyzer.go           ← access pattern analysis
      risk_calculator.go             ← real-time risk scoring
      training_pipeline.go           ← model training infrastructure
      
    federation/                     ← external system integration
      saml_handler.go               ← SAML assertion processing
      oidc_handler.go               ← OpenID Connect token handling
      attribute_mapper.go           ← external attribute transformation
      policy_translator.go          ← external policy format conversion
      
    scaling/                        ← performance and scaling
      distributed_engine.go         ← distributed evaluation coordination
      stream_processor.go           ← real-time policy updates
      load_balancer.go              ← intelligent request routing
      cache_coordinator.go          ← distributed cache management
      
    delegation/                     ← advanced delegation patterns
      delegation_manager.go         ← privilege delegation workflows
      approval_engine.go            ← delegation approval processing
      provenance_tracker.go         ← delegation audit trails
      
  pkg/
    plugins/                        ← reusable extension libraries
      interfaces.go                 ← standard plugin interfaces
      registry.go                   ← plugin discovery and loading
      
  configs/
    extensions.yaml                 ← extension configuration
    ml-models.yaml                  ← ML model configuration
    federation.yaml                 ← external system mappings
    
  docs/
    extensions/                     ← extension development guides
      ml-integration.md
      federation-setup.md
      scaling-patterns.md
```

#### Core Extension Interfaces

The extension framework provides stable interfaces that enable advanced features without modifying core authorization logic:

```go
// ExtensionRegistry manages the lifecycle of authorization extensions
type ExtensionRegistry struct {
    attributeResolvers map[string]AttributeResolver
    policyEvaluators  map[string]PolicyEvaluator
    decisionEnhancers []DecisionEnhancer
    auditProcessors   []AuditProcessor
    mutex             sync.RWMutex
}

// AttributeEnhancer extends basic attribute resolution with advanced sources
type AttributeEnhancer interface {
    // EnhanceContext adds ML-derived, dynamic, or external attributes to evaluation context
    EnhanceContext(ctx context.Context, base *EvaluationContext, req *AuthzRequest) (*EvaluationContext, error)
    
    // GetEnhancementInfo describes what attributes this enhancer provides
    GetEnhancementInfo() EnhancementInfo
    
    // ValidateConfiguration checks enhancer configuration and dependencies
    ValidateConfiguration() error
}

// PolicyFederator enables integration with external policy systems
type PolicyFederator interface {
    // EvaluateExternalPolicies consults external policy stores for additional decisions
    EvaluateExternalPolicies(ctx context.Context, req *AuthzRequest, context *EvaluationContext) ([]ExternalDecision, error)
    
    // SynchronizePolicies updates local cache from external policy stores
    SynchronizePolicies(ctx context.Context) error
    
    // GetPolicyProvenance returns information about external policy sources
    GetPolicyProvenance() []PolicySource
}

// DecisionEnhancer allows post-processing of authorization decisions
type DecisionEnhancer interface {
    // EnhanceDecision adds additional information or modifies decision based on enhanced context
    EnhanceDecision(ctx context.Context, decision *AuthzDecision, enhancedContext *EvaluationContext) (*AuthzDecision, error)
    
    // ShouldEnhance determines if this enhancer should process the given decision
    ShouldEnhance(decision *AuthzDecision) bool
}

// StreamingPolicyManager handles real-time policy updates across distributed nodes
type StreamingPolicyManager interface {
    // SubscribeToPolicyUpdates receives real-time policy change notifications
    SubscribeToPolicyUpdates(ctx context.Context, handler PolicyUpdateHandler) error
    
    // PublishPolicyUpdate broadcasts policy changes to all evaluation nodes
    PublishPolicyUpdate(ctx context.Context, update *PolicyUpdate) error
    
    // GetCurrentPolicyVersion returns the current policy version for consistency checking
    GetCurrentPolicyVersion(ctx context.Context) (string, error)
}
```

#### ML Integration Starter Code

This complete ML attribute enhancer demonstrates how to integrate machine learning insights into authorization decisions:

```go
package ml

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
    
    "github.com/your-org/authz/internal/authorizer"
    "github.com/your-org/authz/internal/extensions"
)

// MLAttributeEnhancer provides machine learning derived attributes for policy evaluation
type MLAttributeEnhancer struct {
    riskModel       RiskModel
    behaviorModel   BehaviorModel
    attributeCache  *AttributeCache
    trainingPipeline *TrainingPipeline
    metrics         *MLMetrics
}

// RiskModel interface for pluggable risk scoring algorithms
type RiskModel interface {
    PredictRisk(ctx context.Context, features *UserFeatures) (float64, error)
    GetModelInfo() ModelInfo
}

// UserFeatures aggregates data for ML model evaluation
type UserFeatures struct {
    UserID              string                 `json:"user_id"`
    RecentAccessPattern []AccessEvent         `json:"recent_access_pattern"`
    PeerGroupBehavior   *PeerGroupStats       `json:"peer_group_behavior"`
    GeographicContext   *GeographicInfo       `json:"geographic_context"`
    TimeContext         *TemporalInfo         `json:"time_context"`
    DeviceContext       *DeviceInfo           `json:"device_context"`
    CustomAttributes    map[string]interface{} `json:"custom_attributes"`
}

// EnhanceContext implements AttributeEnhancer interface
func (ml *MLAttributeEnhancer) EnhanceContext(ctx context.Context, base *authorizer.EvaluationContext, req *authorizer.AuthzRequest) (*authorizer.EvaluationContext, error) {
    // TODO 1: Extract user features from recent access history and current context
    features, err := ml.extractUserFeatures(ctx, req.UserID, req)
    if err != nil {
        // TODO 2: Log feature extraction failure but continue with basic context
        ml.metrics.RecordFeatureExtractionError(err)
        return base, nil // Graceful degradation
    }
    
    // TODO 3: Compute ML-derived risk score with timeout protection
    riskScore, err := ml.computeRiskScoreWithTimeout(ctx, features)
    if err != nil {
        // TODO 4: Use cached risk score or safe default on ML failure
        riskScore = ml.getCachedRiskScore(req.UserID)
    }
    
    // TODO 5: Analyze behavioral patterns compared to user's historical baseline
    behaviorMetrics, err := ml.analyzeBehaviorPattern(ctx, features)
    if err != nil {
        // TODO 6: Continue without behavior analysis on failure
        ml.metrics.RecordBehaviorAnalysisError(err)
        behaviorMetrics = &BehaviorMetrics{IsNormal: true}
    }
    
    // TODO 7: Create enhanced evaluation context with ML attributes
    enhanced := &authorizer.EvaluationContext{
        UserAttributes:        make(map[string]interface{}),
        ResourceAttributes:    base.ResourceAttributes,
        EnvironmentAttributes: base.EnvironmentAttributes,
        ActionAttributes:      base.ActionAttributes,
    }
    
    // TODO 8: Copy original user attributes and add ML-derived attributes
    for k, v := range base.UserAttributes {
        enhanced.UserAttributes[k] = v
    }
    enhanced.UserAttributes["ml.risk_score"] = riskScore
    enhanced.UserAttributes["ml.behavior_normal"] = behaviorMetrics.IsNormal
    enhanced.UserAttributes["ml.anomaly_score"] = behaviorMetrics.AnomalyScore
    enhanced.UserAttributes["ml.peer_group"] = behaviorMetrics.PeerGroup
    
    // TODO 9: Add time-sensitive ML attributes
    enhanced.UserAttributes["ml.model_version"] = ml.riskModel.GetModelInfo().Version
    enhanced.UserAttributes["ml.prediction_confidence"] = behaviorMetrics.Confidence
    
    return enhanced, nil
}

// extractUserFeatures gathers historical and contextual data for ML evaluation
func (ml *MLAttributeEnhancer) extractUserFeatures(ctx context.Context, userID string, req *authorizer.AuthzRequest) (*UserFeatures, error) {
    // TODO 1: Query recent access history from audit logs
    // TODO 2: Get peer group statistics for behavioral comparison
    // TODO 3: Extract geographic and temporal context
    // TODO 4: Gather device and network security context
    // TODO 5: Combine features into structured format for ML model
    return nil, fmt.Errorf("not implemented")
}

// computeRiskScoreWithTimeout protects against slow ML model evaluation
func (ml *MLAttributeEnhancer) computeRiskScoreWithTimeout(ctx context.Context, features *UserFeatures) (float64, error) {
    // TODO 1: Create context with timeout for ML model evaluation
    mlCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
    defer cancel()
    
    // TODO 2: Call ML model with timeout protection
    riskScore, err := ml.riskModel.PredictRisk(mlCtx, features)
    if err != nil {
        return 0.5, err // Default to moderate risk on failure
    }
    
    // TODO 3: Validate risk score is in expected range [0.0, 1.0]
    if riskScore < 0.0 || riskScore > 1.0 {
        return 0.5, fmt.Errorf("invalid risk score: %f", riskScore)
    }
    
    // TODO 4: Cache risk score for fallback use
    ml.attributeCache.SetRiskScore(features.UserID, riskScore, 5*time.Minute)
    
    return riskScore, nil
}
```

#### Federation Integration Starter Code

This complete SAML/OIDC federation handler shows how to integrate external identity providers:

```go
package federation

import (
    "context"
    "encoding/xml"
    "fmt"
    "time"
    
    "github.com/your-org/authz/internal/authorizer"
    "github.com/your-org/authz/internal/extensions"
    "github.com/your-org/authz/pkg/plugins"
)

// FederationManager coordinates identity federation from multiple external providers
type FederationManager struct {
    samlHandler    *SAMLHandler
    oidcHandler    *OIDCHandler
    attributeMapper *AttributeMapper
    trustManager   *TrustManager
    sessionStore   SessionStore
}

// SAMLHandler processes SAML assertions and extracts user attributes
type SAMLHandler struct {
    trustedIdPs     map[string]*TrustedIdentityProvider
    assertionCache  *AssertionCache
    validator       *AssertionValidator
}

// ProcessSAMLAssertion extracts user identity and attributes from SAML assertion
func (s *SAMLHandler) ProcessSAMLAssertion(ctx context.Context, assertion []byte) (*FederatedIdentity, error) {
    // TODO 1: Parse and validate SAML assertion XML structure
    var samlAssertion SAMLAssertion
    if err := xml.Unmarshal(assertion, &samlAssertion); err != nil {
        return nil, fmt.Errorf("invalid SAML assertion: %w", err)
    }
    
    // TODO 2: Verify assertion signature and issuer trust
    if err := s.validator.ValidateAssertion(&samlAssertion); err != nil {
        return nil, fmt.Errorf("assertion validation failed: %w", err)
    }
    
    // TODO 3: Extract user identity from assertion subject
    userID := s.extractUserIdentity(&samlAssertion)
    if userID == "" {
        return nil, fmt.Errorf("no user identity found in assertion")
    }
    
    // TODO 4: Extract user attributes from assertion attribute statements
    attributes := s.extractUserAttributes(&samlAssertion)
    
    // TODO 5: Map external groups to internal roles
    roles := s.mapGroupsToRoles(attributes["groups"])
    
    // TODO 6: Create federated identity with mapped attributes
    identity := &FederatedIdentity{
        UserID:        userID,
        SourceIdP:     samlAssertion.Issuer,
        Attributes:    attributes,
        Roles:         roles,
        ValidUntil:    time.Now().Add(8 * time.Hour),
        SessionIndex:  samlAssertion.SessionIndex,
    }
    
    // TODO 7: Cache federated identity for subsequent authorization requests
    s.assertionCache.Store(userID, identity, identity.ValidUntil)
    
    return identity, nil
}

// extractUserAttributes converts SAML attribute statements to internal format
func (s *SAMLHandler) extractUserAttributes(assertion *SAMLAssertion) map[string]interface{} {
    attributes := make(map[string]interface{})
    
    // TODO 1: Iterate through SAML attribute statements
    for _, stmt := range assertion.AttributeStatements {
        for _, attr := range stmt.Attributes {
            // TODO 2: Map well-known SAML attributes to internal names
            internalName := s.mapAttributeName(attr.Name)
            
            // TODO 3: Convert attribute values to appropriate types
            if len(attr.Values) == 1 {
                attributes[internalName] = attr.Values[0]
            } else if len(attr.Values) > 1 {
                attributes[internalName] = attr.Values
            }
        }
    }
    
    // TODO 4: Add federation metadata attributes
    attributes["federation.source"] = "saml"
    attributes["federation.idp"] = assertion.Issuer
    attributes["federation.timestamp"] = time.Now()
    
    return attributes
}

// AttributeMapper transforms external attributes to internal policy context
type AttributeMapper struct {
    mappingRules  map[string]*MappingRule
    defaultValues map[string]interface{}
    validator     *AttributeValidator
}

// TransformAttributes applies mapping rules to convert external attributes
func (m *AttributeMapper) TransformAttributes(ctx context.Context, external map[string]interface{}, source string) (map[string]interface{}, error) {
    internal := make(map[string]interface{})
    
    // TODO 1: Apply source-specific mapping rules
    sourceRules := m.mappingRules[source]
    if sourceRules == nil {
        return nil, fmt.Errorf("no mapping rules for source: %s", source)
    }
    
    // TODO 2: Transform each external attribute using mapping rules
    for externalName, externalValue := range external {
        if rule, exists := sourceRules.Rules[externalName]; exists {
            // TODO 3: Apply transformation function (direct, regex, lookup, etc.)
            internalValue, err := m.applyTransformationRule(rule, externalValue)
            if err != nil {
                // TODO 4: Log transformation error but continue with default
                continue
            }
            internal[rule.InternalName] = internalValue
        }
    }
    
    // TODO 5: Add default values for missing required attributes
    for internalName, defaultValue := range m.defaultValues {
        if _, exists := internal[internalName]; !exists {
            internal[internalName] = defaultValue
        }
    }
    
    // TODO 6: Validate transformed attributes meet policy requirements
    if err := m.validator.ValidateAttributes(internal); err != nil {
        return nil, fmt.Errorf("attribute validation failed: %w", err)
    }
    
    return internal, nil
}
```

#### Milestone Checkpoints

**Advanced Features Integration Checkpoint:**

After implementing ML integration:
1. **Test Command**: `go test ./internal/ml/... -v`
2. **Expected Behavior**: ML attribute enhancer provides risk scores in authorization decisions
3. **Manual Verification**: Check authorization decision includes `ml.risk_score` attribute
4. **Success Criteria**: Risk scores influence policy evaluation and appear in audit logs

**Federation Integration Checkpoint:**

After implementing SAML/OIDC federation:
1. **Test Command**: `go test ./internal/federation/... -v`
2. **Expected Behavior**: External identity assertions map to internal user attributes
3. **Manual Verification**: Federated user can access resources based on external group membership
4. **Success Criteria**: Authorization requests include federated attributes in evaluation context

**Performance Scaling Checkpoint:**

After implementing distributed evaluation:
1. **Load Test Command**: `go run cmd/load-tester/main.go -rps=10000 -duration=1m`
2. **Expected Behavior**: System maintains sub-100ms latency under load
3. **Manual Verification**: Multiple evaluation nodes process requests concurrently
4. **Success Criteria**: Linear scalability with additional evaluation nodes


## Glossary

> **Milestone(s):** All milestones - this section provides a comprehensive reference for terminology used throughout the authorization system design, from role management (Milestone 1) through policy testing (Milestone 4)

Think of a glossary as a technical dictionary that translates between the natural language we use to think about authorization concepts and the precise technical terminology we use to implement them. Just as a foreign language dictionary helps you navigate between familiar concepts and new vocabulary, this glossary bridges the gap between intuitive understanding ("Who can access what?") and technical precision ("ABAC policy evaluation with attribute-based conditions").

This glossary serves multiple purposes throughout the authorization system implementation. First, it ensures consistent terminology across all components - when we say "tenant isolation," everyone understands the same technical concept. Second, it provides quick reference definitions for complex concepts that appear across multiple milestones. Third, it clarifies the distinction between similar-sounding terms that have different technical implications, such as "role hierarchy" versus "resource hierarchy."

The entries are organized alphabetically within conceptual categories to support both reference lookup and conceptual learning. Each definition includes the technical meaning, how it relates to other concepts, and common usage patterns within authorization systems.

### Core Authorization Concepts

Understanding authorization systems requires mastering fundamental concepts that appear across all implementation milestones. These core concepts form the foundation for role-based permissions, attribute-based policies, resource management, and audit systems.

| Term | Definition | Usage Context |
|------|------------|---------------|
| **Access Control** | The practice of restricting information or resource access to authorized users, programs, or processes. Encompasses authentication (who you are) and authorization (what you can do). | Overall system purpose across all milestones |
| **Authorization** | The process of determining what actions an authenticated user can perform on specific resources. Distinct from authentication, which verifies identity. | Primary system function evaluated by `Authorizer.IsAuthorized` |
| **Principal** | An entity (user, service, or system) that can be granted permissions. In our system, primarily refers to users identified by `UserID` within tenant contexts. | Used in authorization requests and audit logging |
| **Resource** | Any entity that requires access control protection, such as files, database records, API endpoints, or business objects. Represented by `Resource` struct with tenant and ownership context. | Central to all authorization decisions and resource-based access control |
| **Action** | A specific operation that can be performed on a resource, such as "read," "write," "delete," or domain-specific operations like "approve" or "publish." | Part of every `AuthzRequest` and permission definition |
| **Subject** | The entity requesting access to a resource. Typically synonymous with principal in our system context. | Alternative term for the user making authorization requests |
| **Policy** | A rule that defines under what conditions access should be granted or denied. Can be role-based (implicit) or attribute-based (explicit `Policy` entities). | Core concept spanning RBAC and ABAC approaches |
| **Decision Point** | The system component that makes authorization decisions. In our architecture, this is the `Authorizer` interface combining multiple engines. | Architectural concept for the main decision-making component |
| **Enforcement Point** | The system component that enforces authorization decisions by allowing or blocking resource access. Typically implemented in application middleware. | Integration point between authorization system and protected resources |

### RBAC (Role-Based Access Control) Terms

Role-Based Access Control forms the foundation of our authorization system in Milestone 1. These terms describe how permissions are organized through organizational roles and granted to users through role assignments.

| Term | Definition | Implementation Details |
|------|------------|----------------------|
| **RBAC** | Role-Based Access Control - an authorization model where permissions are assigned to roles, and roles are assigned to users. Simplifies permission management through role abstraction. | Implemented through `Role`, `Permission`, and `UserRoleAssignment` entities |
| **Role** | A named collection of permissions that represents a job function or responsibility within an organization. Users receive permissions by being assigned to roles. | Defined by `Role` struct with name, permissions, and optional parent roles |
| **Permission** | A specific authorization to perform an action on a resource or resource type. Represents the atomic unit of access rights in RBAC systems. | Implemented as `Permission` struct with resource and action pairs |
| **Role Assignment** | The association between a user and a role, granting the user all permissions contained within that role. Can include validity periods and assignment metadata. | Tracked through `UserRoleAssignment` with temporal validity and audit trails |
| **Role Hierarchy** | A directed acyclic graph (DAG) of parent-child relationships between roles, where child roles inherit all permissions from their parent roles. | Managed through `ParentRoles` field with DAG validation and transitive closure computation |
| **Permission Inheritance** | The mechanism by which child roles automatically receive all permissions from their parent roles, reducing redundant permission assignments. | Computed through `computeTransitiveClosure` with bitmap optimization |
| **Role Template** | A preconfigured role definition that can be instantiated with parameters to create consistent role structures across different contexts. | Provided through `TemplateEngine` with parameterized role creation |
| **Transitive Closure** | The complete set of permissions available to a role through direct assignment and inheritance from all ancestor roles in the hierarchy. | Computed algorithmically and cached for performance optimization |

### ABAC (Attribute-Based Access Control) Terms

Attribute-Based Access Control extends authorization decisions beyond simple role membership to dynamic evaluation of contextual attributes. These terms describe the policy-driven approach introduced in Milestone 2.

| Term | Definition | Technical Implementation |
|------|------------|--------------------------|
| **ABAC** | Attribute-Based Access Control - an authorization model that evaluates access decisions based on attributes of users, resources, environment, and actions through policy rules. | Implemented through `Policy`, `Condition`, and `EvaluationContext` structures |
| **Attribute** | A named property associated with a subject, resource, environment, or action that can be used in policy evaluation. Examples include user department, resource sensitivity, or current time. | Stored in attribute maps within `EvaluationContext` for policy evaluation |
| **Policy Rule** | A logical expression that defines access conditions using attribute comparisons and boolean operators. Evaluates to allow or deny for authorization requests. | Represented by `Policy` entities with `Condition` trees for complex logic |
| **Condition Tree** | A recursive data structure representing logical combinations (AND, OR, NOT) of attribute comparisons used in policy evaluation. | Implemented as `Condition` struct with nested logical operators |
| **Evaluation Context** | The complete set of attributes available during policy evaluation, including user, resource, environment, and action attributes. | Provided through `EvaluationContext` with resolved attribute values |
| **Attribute Resolution** | The process of gathering and computing attribute values from various sources (databases, external services, computed values) for policy evaluation. | Handled by `AttributeResolver` interface implementations |
| **Policy Effect** | The outcome specified by a policy rule - either allow or deny access when the policy conditions are satisfied. | Defined through `PolicyEffect` enum with `PolicyEffectAllow` and `PolicyEffectDeny` |
| **Deny-Overrides** | A policy combining algorithm where any policy that evaluates to deny will override all policies that evaluate to allow, implementing principle of least privilege. | Default combining logic in `PolicyEngine.Evaluate` |
| **Policy Priority** | A numeric value that determines the order of policy evaluation when multiple policies could apply to the same authorization request. | Specified in `Policy.Priority` field for conflict resolution |
| **Context-Aware Evaluation** | Policy evaluation that considers current environmental conditions, user state, and resource context rather than static permissions. | Enabled through dynamic attribute resolution and condition evaluation |

### Resource and Multi-Tenancy Terms

Resource-based access control and multi-tenant isolation represent critical capabilities for SaaS applications, implemented in Milestone 3. These terms describe fine-grained resource permissions and tenant data separation.

| Term | Definition | System Integration |
|------|------------|-------------------|
| **Resource-Based Access Control** | Authorization model where permissions are granted on specific resource instances rather than resource types, enabling fine-grained access control. | Implemented through resource-specific policies and ownership models |
| **Tenant** | An isolated organizational unit in a multi-tenant system, representing a customer, department, or organization with complete data separation from other tenants. | Identified by `TenantID` throughout all data structures and operations |
| **Tenant Isolation** | The security guarantee that one tenant cannot access, modify, or infer information about another tenant's data, even through indirect means. | Enforced through context propagation and database-level row security |
| **Resource Ownership** | The security model where creating or being granted ownership of a resource provides full control over that resource's access permissions. | Tracked through `Resource.OwnerID` and enforced in authorization decisions |
| **Cross-Tenant Sharing** | Controlled mechanism allowing resource owners to grant access to users from different tenant organizations through explicit sharing grants. | Managed by `CrossTenantShareManager` with explicit authorization and audit trails |
| **Resource Hierarchy** | Parent-child relationships between resources where access permissions can be inherited from parent resources to their children. | Supported through hierarchical permission evaluation and inheritance rules |
| **Share Grant** | An explicit permission that allows a user from one tenant to access a resource owned by another tenant, with specific permissions and validity constraints. | Implemented as `ShareGrant` with cross-tenant authorization and audit requirements |
| **Share Link** | A URL-based access mechanism that provides anonymous or semi-anonymous access to a resource without requiring explicit user-to-resource permission grants. | Part of `ShareSettings` with expiration and usage limits |
| **Tenant Context** | The complete tenant-related information associated with an authorization request, including user memberships and cross-tenant permissions. | Propagated through `TenantContext` and middleware chains |
| **Row-Level Security** | Database enforcement of tenant isolation by automatically filtering query results to include only data belonging to the requesting tenant. | Implemented through database policies and query filters |

### Policy and Evaluation Terms

Policy evaluation represents the dynamic decision-making process that transforms rules and attributes into authorization decisions. These terms describe the sophisticated evaluation mechanisms used across ABAC and resource-based access control.

| Term | Definition | Technical Implementation |
|------|------------|--------------------------|
| **Policy Evaluation** | The dynamic process of assessing policy rules against current request context to determine access decisions. | Performed by `PolicyEngine.Evaluate` with attribute resolution and condition evaluation |
| **Policy Engine** | The system component responsible for loading, parsing, and evaluating attribute-based policies against authorization requests. | Implemented as `PolicyEngine` with policy storage and evaluation capabilities |
| **Condition Evaluation** | The recursive process of evaluating logical expressions within policy rules to determine if policy conditions are satisfied. | Handled by `EvaluateCondition` with support for complex boolean logic |
| **Policy Combining Logic** | The algorithm used to resolve conflicts when multiple policies apply to the same authorization request, determining final access decision. | Implements deny-overrides strategy with priority-based ordering |
| **Decision Pipeline** | The sequential flow of authorization evaluation through different engines (RBAC, ABAC, resource-based) until a final decision is reached. | Orchestrated by main `Authorizer` with fallback logic |
| **Evaluation Method** | The specific authorization approach used to reach a decision (role-based, policy-based, ownership-based, or cached). | Tracked in `AuthzDecision.Method` for audit and debugging purposes |
| **Applied Policies** | The specific policy rules that were evaluated and contributed to the final authorization decision. | Recorded in `AuthzDecision.AppliedPolicies` for decision provenance |
| **Policy Decision** | The intermediate result from ABAC policy evaluation before final authorization decision synthesis. | Returned by `PolicyEngine` as `PolicyDecision` with applied policy details |
| **Fail-Safe Defaults** | The principle that authorization systems should deny access when evaluation cannot be completed or when no explicit permission exists. | Implemented throughout evaluation logic as conservative decision algorithms |
| **Attribute Resolution Chain** | The ordered sequence of sources consulted to resolve attribute values, from cached values to database lookups to external service calls. | Managed by `AttributeResolver` implementations with fallback strategies |

### Audit and Compliance Terms

Audit logging and compliance reporting provide the accountability and transparency required for enterprise authorization systems. These terms describe the immutable audit trail and compliance capabilities delivered in Milestone 4.

| Term | Definition | Implementation Approach |
|------|------------|-------------------------|
| **Audit Trail** | An immutable, chronological record of all authorization decisions and system changes that provides complete accountability for access control. | Maintained through `AuditLogger` with cryptographic integrity verification |
| **Decision Provenance** | The complete chain of reasoning, policies, and data that led to a specific authorization decision, enabling full traceability of access control outcomes. | Captured in `AuditRecord` with applied policies, evaluated attributes, and decision method |
| **Integrity Verification** | The cryptographic mechanism ensuring that audit records have not been tampered with or modified after creation. | Implemented through hash chains linking sequential audit records |
| **Immutable Audit Log** | An append-only storage system for audit records that prevents modification or deletion of historical authorization decisions. | Enforced through storage architecture and integrity hash chains |
| **Compliance Reporting** | The transformation of raw audit data into structured reports meeting specific regulatory framework requirements. | Generated by `ComplianceReporter` for frameworks like SOX, GDPR, HIPAA |
| **Anomaly Detection** | The identification of unusual access patterns or authorization requests that may indicate security threats or policy violations. | Performed by `AnomalyDetector` using statistical analysis and behavioral baselines |
| **Policy Simulation** | The capability to test policy changes in an isolated environment using captured authorization state before deploying to production. | Provided through `PolicyTestFramework` with snapshot-based testing |
| **Regulatory Framework** | Specific compliance requirements (such as SOX, GDPR, HIPAA) that dictate audit logging, access controls, and reporting obligations. | Supported through configurable compliance report generation |
| **Snapshot-Based Testing** | Policy testing methodology that captures point-in-time authorization state and replays authorization scenarios against modified policies. | Implemented through state capture and isolated evaluation environments |
| **Differential Analysis** | The comparison of authorization behavior between different policy versions to understand the impact of policy changes. | Part of policy simulation reporting and change impact assessment |

### Performance and Caching Terms

Authorization systems must provide sub-millisecond response times while maintaining security guarantees. These terms describe the performance optimization and caching strategies used throughout the system.

| Term | Definition | Technical Details |
|------|------------|------------------|
| **Permission Bitmap** | A compressed binary representation of permission sets using bit positions, enabling fast union, intersection, and subset operations. | Implemented as `PermissionBitmap` with `PermissionRegistry` for bit mapping |
| **Multi-Level Caching** | A hierarchical caching strategy with different cache tiers optimized for different access patterns and performance characteristics. | Coordinated by `CacheManager` with L1/L2/L3 cache levels |
| **Cache Invalidation** | The process of removing stale cache entries when underlying data changes, ensuring cache consistency with authoritative data sources. | Triggered by data mutations with pattern-based invalidation |
| **Temporal Locality** | The principle that recently accessed authorization data is likely to be accessed again soon, optimizing cache retention policies. | Exploited in LRU cache policies and permission lookup optimization |
| **Spatial Locality** | The principle that authorization data accessed together (such as user roles and permissions) should be cached and retrieved together. | Used in cache prefetching and batch permission resolution |
| **Computational Locality** | The optimization principle of avoiding repeated calculation of the same derived data, such as transitive closure computations. | Applied in permission inheritance and policy evaluation caching |
| **Cache Coherence** | Maintaining consistency between multiple cache instances in distributed deployments to prevent inconsistent authorization decisions. | Managed through cache versioning and invalidation propagation |
| **Cache Poisoning** | A security attack that attempts to inject false data into authorization caches to cause incorrect access decisions. | Prevented through cache entry validation and signed cache values |
| **Copy-on-Write** | A data structure optimization that shares read-only data between threads until modification is needed, enabling safe concurrent access. | Used in role hierarchy storage for thread-safe permission lookup |
| **Circuit Breaker** | A fault tolerance pattern that prevents cascading failures by temporarily disabling calls to failing services and providing fast failure responses. | Implemented as `CircuitBreaker` for external service dependencies |

### Security and Threat Model Terms

Authorization systems are prime targets for security attacks, requiring comprehensive threat modeling and defensive measures. These terms describe security concepts and attack vectors relevant to authorization systems.

| Term | Definition | Security Implications |
|------|------------|----------------------|
| **Principle of Least Privilege** | The security principle that users should be granted the minimum permissions necessary to perform their job functions, reducing potential attack impact. | Enforced through deny-overrides policy combining and explicit permission grants |
| **Defense in Depth** | A security strategy employing multiple independent layers of protection, so failure of one layer doesn't compromise the entire system. | Implemented through RBAC, ABAC, resource ownership, and audit logging |
| **Permission Elevation Attack** | An attack attempting to gain higher privileges than authorized through exploitation of authorization logic, cache manipulation, or policy confusion. | Mitigated through conservative decision algorithms and comprehensive validation |
| **Timing Attack** | An attack that infers sensitive information (such as user existence or resource presence) by measuring response time differences in authorization calls. | Prevented through consistent response timing regardless of authorization outcome |
| **Privilege Escalation** | The act of exploiting system vulnerabilities to gain elevated access rights beyond what was originally authorized. | Prevented through role hierarchy validation and permission boundary enforcement |
| **Insider Threat** | Security risks posed by authorized users who abuse their legitimate access to perform unauthorized actions. | Detected through anomaly detection and comprehensive audit logging |
| **Cross-Tenant Data Leakage** | The unauthorized disclosure of one tenant's data to users from another tenant, violating tenant isolation guarantees. | Prevented through mandatory tenant context propagation and database-level isolation |
| **Cache Poisoning** | Attack injecting false authorization data into system caches to cause incorrect access decisions. | Mitigated through signed cache entries and cache validation |
| **Policy Confusion Attack** | Exploitation of ambiguous or conflicting policies to gain unauthorized access through policy interpretation edge cases. | Prevented through explicit policy priority and deny-overrides combining logic |
| **Audit Tampering** | Attempts to modify or delete audit records to hide unauthorized access or policy violations. | Prevented through immutable audit logs and cryptographic integrity verification |

### Technical Architecture Terms

The authorization system employs sophisticated architectural patterns to achieve scalability, reliability, and maintainability. These terms describe key architectural concepts and implementation patterns.

| Term | Definition | Architectural Role |
|------|------------|-------------------|
| **Directed Acyclic Graph (DAG)** | A graph structure with directed edges and no cycles, used for role hierarchies to ensure permission inheritance terminates and avoid infinite loops. | Essential for role hierarchy validation and transitive closure computation |
| **Abstract Syntax Tree (AST)** | A tree representation of policy expressions that enables programmatic evaluation of complex logical conditions. | Used in policy condition parsing and evaluation |
| **Transitive Closure** | The mathematical concept of including all indirectly related elements - in role hierarchies, all permissions reachable through inheritance chains. | Computed for efficient permission lookup and cached for performance |
| **Write-Ahead Logging** | A database technique of recording intended changes before applying them, ensuring audit log durability and enabling crash recovery. | Applied to audit logging for guaranteed decision recording |
| **Merkle Tree** | A binary tree of cryptographic hashes that enables efficient verification of data integrity across large datasets. | Used for audit log integrity verification at scale |
| **Row-Level Security (RLS)** | Database-enforced access control that automatically filters query results based on user identity and permissions. | Implemented for tenant isolation and resource-based access control |
| **Middleware Chain** | A software pattern where requests pass through a sequence of processing components, each adding functionality like authentication, authorization, or logging. | Used for tenant context propagation and authorization enforcement |
| **Snapshot Isolation** | A database concurrency control method ensuring that transactions see a consistent snapshot of data, preventing read anomalies during policy evaluation. | Applied in policy testing environments for reproducible results |
| **Circuit Breaker Pattern** | A design pattern that detects failures and encapsulates the logic of preventing a failure from constantly recurring during maintenance, debugging, or recovery. | Protects against cascading failures in distributed authorization components |
| **Event Sourcing** | An architectural pattern storing all changes as a sequence of events, enabling audit trails and system state reconstruction. | Applied to audit logging and policy change tracking |

### Data Management Terms

Authorization systems manage complex relationships between users, roles, permissions, resources, and policies. These terms describe data management concepts and storage strategies.

| Term | Definition | Data Management Context |
|------|------------|-------------------------|
| **Entity Relationship** | The associations between different data entities (users, roles, resources) that define the authorization domain model. | Modeled through foreign key relationships and association tables |
| **Many-to-Many Relationship** | A data relationship where entities on both sides can be associated with multiple entities on the other side, such as users having multiple roles. | Implemented through junction tables like `UserRoleAssignment` |
| **Referential Integrity** | The database constraint ensuring that relationships between entities remain valid and consistent across all operations. | Enforced through foreign key constraints and transaction boundaries |
| **Data Normalization** | The database design process of organizing data to reduce redundancy and improve consistency by separating concerns into distinct entities. | Applied in separating roles, permissions, and assignments into distinct tables |
| **Denormalization** | The strategic introduction of data redundancy to improve query performance, such as caching computed permissions. | Used in permission bitmaps and materialized role hierarchies |
| **Eventual Consistency** | A consistency model where the system guarantees that, given no new updates, all nodes will eventually converge to the same state. | Applied in distributed cache invalidation and policy propagation |
| **Strong Consistency** | A consistency model guaranteeing that all nodes see the same data at the same time, critical for security-sensitive authorization decisions. | Required for audit logging and permission changes |
| **Optimistic Concurrency Control** | A concurrency control method that assumes multiple transactions can complete without affecting each other and validates this assumption at commit time. | Used in policy updates and role modifications with version checking |
| **Pessimistic Locking** | A concurrency control strategy that locks data during the entire transaction to prevent conflicts, used for critical authorization state changes. | Applied to role hierarchy modifications and tenant membership changes |
| **Sharding** | A database architecture pattern that distributes data across multiple database instances based on partition keys such as tenant ID. | Enables horizontal scaling while maintaining tenant isolation |

### Testing and Validation Terms

Comprehensive testing ensures authorization system correctness and security. These terms describe testing methodologies and validation approaches used throughout development and maintenance.

| Term | Definition | Testing Context |
|------|------------|----------------|
| **Unit Testing** | Testing individual components in isolation using mock dependencies to verify component behavior and edge case handling. | Applied to individual engines and utilities with dependency injection |
| **Integration Testing** | Testing component interactions and data flow through the authorization pipeline with realistic data and dependencies. | Validates end-to-end authorization flow and component communication |
| **End-to-End Testing** | Complete system testing that validates authorization behavior from external API requests through final audit logging. | Ensures entire authorization pipeline functions correctly in production-like environments |
| **Milestone Validation** | Structured testing checkpoints that verify specific functionality and acceptance criteria after each development milestone. | Automated through `MilestoneCheckpointer` with comprehensive scenario coverage |
| **Test Fixtures** | Realistic test data that mirrors production complexity, providing consistent test scenarios across different test types. | Managed through `TestFixtures` with representative users, roles, resources, and policies |
| **Test Isolation** | The practice of ensuring tests don't interfere with each other through clean state management and independent test environments. | Achieved through test database transactions and clean fixture setup |
| **Regression Testing** | Re-running existing tests after changes to ensure new functionality doesn't break existing behavior. | Automated through continuous integration with comprehensive test suites |
| **Performance Benchmarking** | Measuring authorization system latency, throughput, and resource utilization under realistic load conditions. | Implemented through `LoadTestFramework` with various load patterns |
| **Acceptance Testing** | Validation that implemented functionality meets specified business requirements and user scenarios. | Defined through milestone acceptance criteria and automated validation |
| **Property-Based Testing** | Testing methodology that generates random inputs within specified constraints to verify system properties hold across input space. | Applied to policy evaluation and permission inheritance validation |

### Monitoring and Operations Terms

Production authorization systems require comprehensive monitoring, debugging, and operational capabilities. These terms describe observability and operational concepts essential for maintaining authorization systems in production.

| Term | Definition | Operational Context |
|------|------------|-------------------|
| **Observability** | The capability to understand system internal state and behavior through external outputs such as metrics, logs, and traces. | Achieved through structured logging, metrics collection, and distributed tracing |
| **System Health** | The overall operational status of authorization system components, including performance, availability, and error rates. | Monitored through `HealthChecker` with component-specific health indicators |
| **Graceful Degradation** | The system's ability to maintain core functionality while disabling advanced features during component failures or performance issues. | Managed by `DegradationManager` with progressive feature reduction |
| **Failover** | The automatic switching to backup systems or reduced functionality when primary systems become unavailable. | Implemented through circuit breakers and fallback decision strategies |
| **Load Balancing** | Distributing authorization requests across multiple system instances to optimize performance and provide fault tolerance. | Supports horizontal scaling with stateless authorization components |
| **Circuit Breaker** | A fault tolerance mechanism that detects failures and provides fast failure responses to prevent cascading system failures. | Implemented as `CircuitBreaker` protecting external service dependencies |
| **Health Check** | Automated monitoring that verifies system components are functioning correctly and responding within acceptable performance parameters. | Performed by `HealthChecker` with component-specific validation logic |
| **Metrics Collection** | The systematic gathering of quantitative data about system performance, usage patterns, and operational characteristics. | Tracked through `AuthzMetrics` and performance measurement infrastructure |
| **Distributed Tracing** | The ability to trace individual requests through multiple system components to understand performance bottlenecks and error sources. | Enabled through `DebugTrace` with step-by-step evaluation recording |
| **Alerting** | Automated notification systems that inform operators of system issues, performance degradation, or security anomalies. | Triggered based on health checks, anomaly detection, and performance thresholds |

### Advanced Features and Extensions Terms

The authorization system architecture supports sophisticated extensions and integrations. These terms describe advanced capabilities and future enhancement patterns.

| Term | Definition | Extension Context |
|------|------------|------------------|
| **Machine Learning Policies** | Authorization policies that incorporate ML-derived insights such as risk scores, behavioral analysis, or pattern recognition to enhance decision-making. | Implemented through `MLAttributeEnhancer` and risk-based attribute resolution |
| **Dynamic Attributes** | Authorization attributes that change in real-time based on current context, user behavior, or external system state rather than static profile data. | Resolved through real-time attribute resolution and behavioral analysis |
| **Advanced Delegation** | Sophisticated privilege delegation patterns with hierarchical chains, temporal constraints, and comprehensive audit trails for accountability. | Supported through time-limited role assignments and delegation tracking |
| **Distributed Evaluation** | Authorization policy evaluation distributed across multiple nodes for horizontal scalability while maintaining consistency guarantees. | Enables scaling beyond single-node performance limitations |
| **Streaming Updates** | Real-time policy and permission change propagation without service disruption or batch update delays. | Implemented through `StreamingPolicyManager` and event-driven updates |
| **SAML Federation** | Integration with external Security Assertion Markup Language (SAML) identity providers for enterprise single sign-on capabilities. | Handled by `SAMLHandler` with attribute mapping and trust management |
| **OIDC Integration** | OpenID Connect integration for modern identity federation with OAuth 2.0-based authentication and attribute exchange. | Managed through federated identity processing and claim-based attributes |
| **External Policy Stores** | Integration with existing policy management systems and compliance frameworks through standardized policy exchange protocols. | Supported through `PolicyFederator` interface and policy synchronization |
| **Plugin Architecture** | Extension framework enabling advanced features without core system modification through well-defined extension points. | Provided through `ExtensionRegistry` and various enhancement interfaces |
| **Federated Session Management** | Managing authentication sessions across multiple identity providers while maintaining security and user experience. | Coordinated through `FederationManager` and cross-system session tracking |

### Error Handling and Resilience Terms

Robust authorization systems must handle various failure modes while maintaining security guarantees. These terms describe error handling strategies and resilience patterns.

| Term | Definition | Resilience Application |
|------|------------|----------------------|
| **Conservative Decision Algorithm** | Error handling approach where any uncertainty or evaluation failure results in access denial rather than potentially granting unauthorized access. | Applied throughout authorization pipeline when components fail or timeout |
| **Emergency Mode** | A degraded operational state that disables non-essential features while maintaining core authorization capabilities during system stress or failures. | Managed by `DegradationManager` with progressive feature reduction |
| **Cascading Failure** | A system failure mode where the failure of one component causes failure in dependent components, potentially bringing down the entire system. | Prevented through circuit breakers, timeouts, and component isolation |
| **Bulkhead Pattern** | An isolation technique that separates system resources to prevent failure in one area from affecting other areas. | Applied in component isolation and resource pool management |
| **Timeout Management** | The systematic handling of operation timeouts to prevent indefinite blocking while maintaining system responsiveness. | Implemented throughout policy evaluation and external service calls |
| **Backpressure** | A flow control mechanism that prevents system overload by limiting the rate of incoming requests when system capacity is exceeded. | Applied in request queuing and load shedding strategies |
| **Jitter** | The addition of randomness to retry intervals and timeout periods to prevent synchronized load spikes when multiple clients retry simultaneously. | Used in retry logic and cache refresh scheduling |
| **Exponential Backoff** | A retry strategy that increases delay between retry attempts exponentially to reduce load on failing systems while attempting recovery. | Applied in external service calls and database connection retry logic |
| **Dead Letter Queue** | A storage mechanism for requests that cannot be processed due to system errors, enabling later reprocessing or manual investigation. | Used for failed audit log writes and policy evaluation errors |
| **Idempotency** | The property that repeated identical operations produce the same result, enabling safe retry of failed operations. | Required for audit logging and policy update operations |

### Integration and Federation Terms

Modern authorization systems must integrate with existing identity providers, policy systems, and compliance frameworks. These terms describe integration patterns and federation capabilities.

| Term | Definition | Integration Details |
|------|------------|-------------------|
| **Identity Federation** | The linking of identity management systems across organizational boundaries, enabling users to access resources across multiple systems with single sign-on. | Managed through `FederationManager` with SAML and OIDC support |
| **Attribute Mapping** | The process of transforming attributes from external identity providers into the internal attribute schema used for policy evaluation. | Handled by `AttributeMapper` with configurable transformation rules |
| **Trust Relationship** | The security relationship between systems that enables them to accept each other's identity assertions and authorization decisions. | Established through cryptographic verification and trusted identity provider configuration |
| **Claims-Based Authentication** | An identity model where external systems provide claims (assertions) about user identity and attributes that can be used in authorization decisions. | Processed through SAML assertion validation and claim extraction |
| **Policy Synchronization** | The process of keeping authorization policies consistent across multiple systems or policy stores through automated updates. | Implemented through `PolicyFederator` with external policy store integration |
| **Cross-System Authorization** | Authorization decisions that span multiple systems, requiring coordination between different authorization domains. | Supported through federated decision making and cross-system attribute resolution |
| **Identity Provider (IdP)** | An external system that authenticates users and provides identity assertions that can be used for authorization decisions. | Integrated through `TrustedIdentityProvider` configuration and assertion processing |
| **Service Provider (SP)** | A system that relies on external identity providers for user authentication while maintaining its own authorization policies. | Our authorization system acts as SP in federation scenarios |
| **Assertion Validation** | The cryptographic and logical validation of identity assertions from external providers to ensure authenticity and integrity. | Performed by `AssertionValidator` with signature verification and expiration checking |
| **Attribute Authority** | An external system that provides authoritative attribute information for use in policy evaluation. | Integrated through `AttributeResolver` implementations with external attribute sources |

### Acronyms and Abbreviations

Authorization systems utilize numerous industry-standard acronyms that appear throughout technical documentation and implementation. This reference ensures consistent understanding across all system components.

| Acronym | Full Name | Domain Context |
|---------|-----------|----------------|
| **ACL** | Access Control List | Traditional access control model with explicit permission lists per resource |
| **ABAC** | Attribute-Based Access Control | Dynamic authorization model using attribute evaluation and policy rules |
| **ADR** | Architecture Decision Record | Documentation pattern for recording and justifying design decisions |
| **API** | Application Programming Interface | External interface for authorization system integration |
| **AST** | Abstract Syntax Tree | Data structure for representing and evaluating policy expressions |
| **AuthN** | Authentication | Process of verifying user identity (who you are) |
| **AuthZ** | Authorization | Process of determining user permissions (what you can do) |
| **CRUD** | Create, Read, Update, Delete | Basic data operations on system entities |
| **DAG** | Directed Acyclic Graph | Graph structure used for role hierarchies and dependency management |
| **GDPR** | General Data Protection Regulation | European privacy regulation requiring audit trails and access controls |
| **HIPAA** | Health Insurance Portability and Accountability Act | US healthcare regulation requiring strict access controls and audit logging |
| **IAM** | Identity and Access Management | Comprehensive framework for identity and authorization management |
| **IdP** | Identity Provider | External system providing user authentication and identity assertions |
| **JSON** | JavaScript Object Notation | Data interchange format used for API requests and responses |
| **JWT** | JSON Web Token | Compact token format for transmitting identity and authorization claims |
| **NIST** | National Institute of Standards and Technology | US standards body providing authorization system guidelines |
| **OIDC** | OpenID Connect | Modern identity layer built on OAuth 2.0 for authentication and federation |
| **RBAC** | Role-Based Access Control | Authorization model using organizational roles to grant permissions |
| **ReBAC** | Relationship-Based Access Control | Authorization model based on relationships between entities |
| **RLS** | Row-Level Security | Database feature automatically filtering query results by user permissions |
| **SAML** | Security Assertion Markup Language | XML-based standard for exchanging authentication and authorization data |
| **SOX** | Sarbanes-Oxley Act | US financial regulation requiring audit trails and access controls |
| **SP** | Service Provider | System relying on external identity providers for authentication |
| **SQL** | Structured Query Language | Database query language used for data storage and retrieval |
| **SSO** | Single Sign-On | Authentication method allowing users to access multiple systems with one login |
| **TLS** | Transport Layer Security | Cryptographic protocol for secure communication between system components |
| **URI** | Uniform Resource Identifier | Standard format for identifying resources in authorization requests |
| **UUID** | Universally Unique Identifier | Standard format for generating unique entity identifiers |
| **WAL** | Write-Ahead Log | Database technique for ensuring durability and enabling crash recovery |
| **XML** | Extensible Markup Language | Markup language used in SAML assertions and policy exchange |

### Domain-Specific Vocabulary

Authorization systems have developed specialized vocabulary that captures nuanced concepts not found in general software engineering. These terms represent domain expertise essential for building effective authorization systems.

| Term | Definition | Domain Usage |
|------|------------|--------------|
| **Access Matrix** | A theoretical model representing all possible subject-object-action combinations in an authorization system, helping analyze completeness and conflicts. | Used for comprehensive policy validation and gap analysis |
| **Capability** | A transferable authorization token that grants specific permissions and can be delegated between users or systems. | Alternative to ACL-based approaches, enabling fine-grained permission delegation |
| **Clearance Level** | A hierarchical classification system where higher levels include permissions from all lower levels, commonly used in government and military contexts. | Implemented through role hierarchy with strict ordering constraints |
| **Compartmentalization** | The practice of isolating different types of data or operations to limit the scope of potential security breaches. | Applied in tenant isolation and role boundary enforcement |
| **Data Classification** | The systematic organization of data into categories based on sensitivity levels, determining appropriate access controls. | Used in resource attributes and policy conditions |
| **Delegation Chain** | A sequence of permission grants where users pass specific permissions to other users, maintaining accountability through the chain. | Tracked through audit logs and temporary role assignments |
| **Entitlement** | A specific access right or privilege that a user possesses, either through direct grant or role inheritance. | Computed through role resolution and policy evaluation |
| **Least Privilege Principle** | The security practice of granting users only the minimum permissions necessary to perform their job functions. | Enforced through deny-overrides policy combining and explicit grants |
| **Need-to-Know Basis** | An access control principle where information is shared only with individuals who require it to perform their duties. | Implemented through resource-based access control and compartmentalization |
| **Principle of Separation of Duties** | The security practice requiring multiple people to complete critical tasks, preventing fraud and errors through dual control. | Supported through multi-approval workflows and role-based constraints |
| **Privileged Access** | Special permissions granted to administrative users for system maintenance, security operations, or emergency response. | Managed through elevated roles with additional audit requirements |
| **Risk-Based Access Control** | Authorization decisions that incorporate risk assessment based on user behavior, context, and environmental factors. | Enhanced through ML-based risk scoring and dynamic attribute evaluation |
| **Security Clearance** | Formal authorization to access classified information based on background investigation and trust determination. | Modeled through role assignments with validation requirements |
| **Workflow Authorization** | Dynamic permission management where access rights change based on business process state and user roles within workflows. | Supported through temporal role assignments and state-based policies |
| **Zero Trust Model** | A security framework that assumes no implicit trust and verifies every access request regardless of location or previous authorization. | Implemented through comprehensive policy evaluation and continuous validation |

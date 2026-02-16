# AUDIT & FIX: blog-platform

## CRITIQUE
- **Audit Finding Confirmed (Session Store):** M1 defines the User table and database schema, but if the auth system (M2) uses session-based auth, M1 needs to provision a session store (e.g., a sessions table or Redis configuration). This is a logical sequencing gap.
- **Audit Finding Confirmed (Markdown Rendering):** M3 AC says 'Markdown content is rendered to HTML for display while preserving the raw markdown in storage.' This implies server-side rendering in the API layer, which couples the API to a specific display format. The API should store raw markdown and optionally return rendered HTML as a separate field or let the client render. The AC as written is architecturally opinionated without acknowledging the trade-off.
- **Difficulty vs Scope Mismatch:** The project includes frontend UI (M4) with split-pane markdown editor, responsive design, and component architecture. Combined with backend auth, CRUD, and database design, this is a full-stack intermediate project. The description ('Full CRUD, auth, markdown') is uselessly terse.
- **M2 Deliverable Drift:** M2 deliverables include 'Password reset functionality with time-limited token sent via email' but no AC mentions password reset. This is scope creep in the deliverables.
- **M3 Deliverable Drift:** M3 deliverables mention 'cursor-based pagination' and 'soft-delete option' but the ACs mention neither. ACs say 'configurable page size and total count' which implies offset pagination, and DELETE returning 403 for wrong user but no mention of soft-delete.
- **M4 Missing Backend Dependency:** M4 (Frontend UI) assumes all backend endpoints are complete, but there's no AC verifying the frontend correctly handles loading states, error states, and authentication flow end-to-end.
- **XSS Pitfall in M3:** Correctly identified—rendering user-supplied markdown to HTML does create an XSS risk. However, the mitigation strategy (sanitize HTML output) is mentioned only in a pitfall, not in an AC. This should be a measurable AC.
- **Missing Comments Milestone:** The tags mention 'comments' and M4 deliverables reference 'threaded comments section' but there is no milestone or AC for implementing comments CRUD. This is a significant gap.
- **M1 Missing Index AC:** Pitfall says 'Not indexing frequently queried columns' but no AC requires index creation. ACs should enforce what pitfalls warn about.

## FIXED YAML
```yaml
id: blog-platform
name: Blog Platform
description: >-
  Build a full-stack blog platform with user authentication, markdown-based
  post authoring with comments, and a responsive frontend interface.
difficulty: intermediate
estimated_hours: "30-40"
essence: >-
  HTTP request handling with stateful session management, relational data
  modeling with foreign key constraints, secure password hashing with
  token-based authentication flows, and markdown content processing with
  server-side sanitization.
why_important: >-
  Building a blog platform teaches full-stack development fundamentals used
  in most web applications—database schema design, RESTful API patterns,
  authentication security, content processing, and connecting frontend
  interfaces to backend services.
learning_outcomes:
  - Design normalized database schemas with foreign key relationships for users, posts, and comments
  - Implement secure authentication with password hashing and JWT token validation
  - Build RESTful API endpoints following HTTP method conventions for CRUD operations
  - Parse and sanitize markdown content to prevent XSS vulnerabilities in rendered HTML
  - Handle authorization ensuring users can only modify their own resources
  - Create dynamic frontend components that consume REST API data
  - Implement pagination, form validation, and error handling across client and server
  - Deploy a full-stack application with environment-based configuration
skills:
  - RESTful API Design
  - JWT Authentication
  - Database Schema Design
  - Markdown Parsing & Sanitization
  - Password Hashing
  - Session Management
  - HTTP Request Handling
  - SQL Queries
  - Frontend Component Architecture
tags:
  - comments
  - crud
  - full-stack
  - intermediate
  - javascript
  - markdown
  - python
  - typescript
architecture_doc: architecture-docs/blog-platform/index.md
languages:
  recommended:
    - JavaScript
    - Python
    - TypeScript
  also_possible:
    - Go
    - Ruby
    - PHP
resources:
  - name: Build a Blog with Next.js
    url: https://nextjs.org/learn/basics/create-nextjs-app
    type: tutorial
  - name: Flask Mega-Tutorial
    url: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
    type: tutorial
prerequisites:
  - type: skill
    name: HTML/CSS/JavaScript
  - type: skill
    name: Basic backend knowledge (Node.js or Python)
  - type: skill
    name: Database basics (SQL)
milestones:
  - id: blog-platform-m1
    name: Project Setup & Database Schema
    description: >-
      Set up the project structure, design the database schema for users,
      posts, and comments, and configure session/token storage infrastructure.
    acceptance_criteria:
      - "Project is initialized with the chosen web framework and all dependencies are installable via a single command"
      - "Database connection is established and verified with a health check query on startup"
      - "User table includes id, email (unique, indexed), password_hash, display_name, and created_at columns"
      - "Post table includes id, title, content (markdown text), author_id (FK to users), status (draft/published), created_at, and updated_at columns"
      - "Comment table includes id, post_id (FK to posts), author_id (FK to users), body, created_at columns"
      - "Indexes are created on user.email, post.author_id, post.created_at, and comment.post_id for query performance"
      - "Migration system supports running pending migrations forward and rolling back the most recent migration"
      - "Session store is configured (sessions table, Redis, or JWT secret loaded from environment) to support authentication in M2"
    pitfalls:
      - "Not indexing frequently queried columns (email lookups, post listing by date) causes full table scans"
      - "Storing passwords in plain text instead of using a one-way hash"
      - "Not using foreign key constraints allowing orphaned posts/comments"
      - "Hardcoding database credentials instead of using environment variables"
    concepts:
      - Normalized database design with foreign keys
      - Database migrations and version control
      - Connection pooling
      - Index design for query patterns
    skills:
      - Database schema design
      - SQL migrations
      - ORM configuration
      - Environment configuration management
    deliverables:
      - Database schema DDL for users, posts, and comments with foreign keys and indexes
      - Migration files supporting forward and rollback operations
      - Database connection pool with configurable size and timeout
      - Project structure with route stubs, middleware setup, and configuration files
    estimated_hours: "3-5"

  - id: blog-platform-m2
    name: User Authentication
    description: >-
      Implement user registration, login, and logout with secure password
      handling and token/session management.
    acceptance_criteria:
      - "POST /auth/register validates email format and password strength (min 8 chars), creates user if email is unique, returns 201"
      - "Password is hashed with bcrypt (cost >= 10) or argon2id before storing in the database"
      - "POST /auth/login verifies credentials and returns a signed JWT (or sets an HttpOnly secure cookie) with configurable expiration"
      - "Login failure returns 401 with a generic message that does not reveal whether the email exists"
      - "Protected routes return 401 Unauthorized when the request lacks a valid, non-expired authentication token"
      - "POST /auth/logout invalidates the session or token so subsequent requests with that credential are rejected"
    pitfalls:
      - "Storing JWT in localStorage exposes it to XSS; prefer HttpOnly cookies for browser clients"
      - "Revealing whether an email exists on failed login enables user enumeration attacks"
      - "Not enforcing password strength requirements allows trivially guessable passwords"
      - "Using a fixed string comparison for password verification is vulnerable to timing attacks; use constant-time comparison"
    concepts:
      - Password hashing with bcrypt/argon2id
      - JWT structure and validation
      - HttpOnly secure cookies
      - Constant-time comparison
    skills:
      - Secure authentication implementation
      - Session/token management
      - Password security best practices
      - API endpoint protection
    deliverables:
      - Registration endpoint with input validation and password hashing
      - Login endpoint returning JWT or setting secure session cookie
      - Logout endpoint invalidating the active session/token
      - Auth middleware rejecting unauthenticated requests with 401
    estimated_hours: "5-6"

  - id: blog-platform-m3
    name: Blog Post & Comment CRUD
    description: >-
      Implement create, read, update, delete for blog posts and comments,
      with authorization, pagination, and markdown handling.
    acceptance_criteria:
      - "POST /posts creates a new post storing raw markdown content with the authenticated user as author; returns 201"
      - "GET /posts returns paginated results (offset or cursor-based) with configurable page size; response includes total count"
      - "GET /posts/:id returns the full post with author display_name, raw markdown, and rendered HTML (sanitized)"
      - "PUT /posts/:id returns 403 Forbidden when a non-author attempts to modify the post"
      - "DELETE /posts/:id returns 403 Forbidden when a non-author attempts to delete the post; author receives 204"
      - "Rendered HTML output is sanitized using an allowlist-based HTML sanitizer to prevent stored XSS from malicious markdown"
      - "POST /posts/:id/comments creates a comment by the authenticated user; returns 201 with the comment"
      - "GET /posts/:id/comments returns paginated comments for the post, ordered by created_at ascending"
      - "DELETE /comments/:id allows only the comment author to delete; returns 403 for non-authors"
      - "Queries for posts with authors use JOIN or eager loading to avoid N+1 query patterns"
    pitfalls:
      - "Rendering unsanitized user markdown to HTML creates stored XSS vulnerabilities; always sanitize rendered output"
      - "N+1 queries when loading posts with author names; use JOIN or DataLoader-style batching"
      - "Not checking resource ownership on edit/delete allows any authenticated user to modify any post"
      - "Returning rendered HTML from the API couples it to a display format; consider returning both raw and sanitized HTML"
    concepts:
      - CRUD operations with authorization
      - Markdown parsing and HTML sanitization
      - Pagination (offset vs cursor)
      - N+1 query prevention
    skills:
      - RESTful API design
      - Authorization and access control
      - Content sanitization
      - Database query optimization
    deliverables:
      - Post CRUD endpoints with authorization checks
      - Comment CRUD endpoints nested under posts
      - Markdown-to-HTML rendering with allowlist-based sanitization
      - Paginated list endpoints with efficient JOIN queries
    estimated_hours: "7-9"

  - id: blog-platform-m4
    name: Frontend UI
    description: >-
      Build the frontend interface consuming the backend API, with post
      browsing, authoring, authentication flows, and responsive design.
    acceptance_criteria:
      - "Homepage displays a paginated list of posts sorted by newest first, showing title, excerpt, author, and date"
      - "Post detail page renders sanitized HTML content with author name and publish date"
      - "Login and registration forms validate input client-side and display server error messages on failure"
      - "Post editor provides a markdown input area with a live preview panel rendering the markdown to HTML"
      - "Comment section on post detail page displays existing comments and allows authenticated users to submit new ones"
      - "Loading states are displayed during API calls; error states show user-friendly messages on failure"
      - "Responsive layout adapts to mobile widths without horizontal scrolling or overlapping elements"
      - "Unauthenticated users are redirected to login when attempting to create/edit posts or comments"
    pitfalls:
      - "Not handling loading and error states creates a broken UX during slow or failed API calls"
      - "Client-side rendering without SSR or pre-rendering hurts SEO for a content-heavy blog"
      - "Not implementing proper auth state management causes stale UI after token expiration"
      - "Live markdown preview without debouncing causes excessive re-renders on every keystroke"
    concepts:
      - Component architecture
      - Client-side routing
      - Form handling and validation
      - State management for auth
    skills:
      - Frontend component development
      - Client-side routing
      - Form validation
      - Responsive design
    deliverables:
      - Post listing page with pagination and post previews
      - Post detail page with rendered content and comments section
      - Post editor with markdown input and live HTML preview
      - Auth forms (login/register) with client-side validation and error display
      - Responsive CSS layout for desktop and mobile viewports
    estimated_hours: "7-9"
```
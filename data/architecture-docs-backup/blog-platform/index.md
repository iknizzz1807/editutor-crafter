# Blog Platform: Design Document


## Overview

This document describes the architecture for a full-featured blog platform supporting user authentication, markdown-based content creation, and CRUD operations. The key architectural challenge is designing a secure, scalable system that cleanly separates frontend and backend concerns while providing an intuitive learning path for full-stack development.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


> **Milestone(s):** Milestone 1 (Project Setup & Database Schema)

## Context and Problem Statement

This document outlines the architecture for a full-featured blog platform designed as an educational project. The goal is to provide a practical learning vehicle for intermediate developers to grasp full-stack web development fundamentals, contrasting with production systems that prioritize scale, reliability, and advanced features. The core challenge lies in balancing pedagogical clarity with the implementation of realistic, secure patterns for user management, content storage, and presentation. This section establishes the conceptual foundation for the system by introducing a mental model, defining the technical problem space, and comparing common architectural approaches.

### Mental Model: The Digital Printing Press

Imagine the blog platform as a **digital printing press and publishing house**. This analogy helps map abstract software components to tangible roles and workflows:

*   **Writers (Users):** Individuals who create content. They must register with the publishing house (user registration) and receive credentials (authentication) to submit manuscripts.
*   **Manuscripts (Blog Posts):** The core content, written in a specific formatting language (Markdown). The publishing house stores these in an organized archive (database), allows writers to edit or withdraw them (CRUD operations), and typesets them for readers (HTML rendering).
*   **Readers (Public Visitors):** The audience who browse the published catalog (post listing) and read individual articles (post detail page). They do not need credentials for consumption.
*   **The Printing Press (Backend API):** The machinery that coordinates all activity. It accepts submissions from writers, retrieves manuscripts from the archive on request, and enforces house rules (e.g., only the original author can edit a post).
*   **The Archive (Database):** A vast, organized library storing all manuscripts (posts), writer records (users), and reader notes (comments) with a precise cataloging system (schema and indexes).
*   **The Storefront (Frontend UI):** The public-facing magazine stand and reading room. It displays the latest issues (paginated post list) in an attractive, readable layout that adapts to whether the reader is on a desktop or mobile device (responsive design).

This mental model clarifies responsibilities: the printing press (backend) handles logic and rules, the archive (database) ensures safe storage and retrieval, and the storefront (frontend) presents the final product. Understanding these roles is the first step in decomposing the system into manageable components.

### Technical Problem Space

Building a functional blog platform requires solving four interconnected technical problems, each with its own subtleties and trade-offs.

**1. User Management & Secure Access Control**
The system must distinguish between anonymous readers, authenticated writers, and administrators. This involves:
*   **Identity Establishment:** Creating a persistent user record with validated, unique credentials (email/password).
*   **Credential Security:** Storing passwords in a non-reversible format (hashing) to prevent exposure during a data breach.
*   **Session Management:** Issuing a temporary, verifiable proof of identity (a token or session cookie) after successful login, and validating this proof on every subsequent request to protected resources (like post creation).
*   **Authorization:** Enforcing business rules such as "a user can only edit their own posts," which requires comparing the authenticated user's identity with the resource's owner.

**2. Content Storage & Retrieval**
Blog posts are the primary data entity. The design must address:
*   **Structured Storage:** Deciding on a database schema that efficiently stores post metadata (title, author, timestamps) and content, while maintaining relationships (e.g., linking a post to its author and comments).
*   **Format Handling:** Choosing how to store and process Markdown-formatted text—whether to store the raw Markdown, pre-rendered HTML, or both—which impacts editing flexibility, rendering performance, and storage overhead.
*   **Efficient Listing:** Implementing pagination to handle a growing number of posts without overwhelming the database or frontend, requiring decisions on sorting (newest first) and filtering (by author, tag).
*   **Data Integrity:** Using database constraints (foreign keys, unique indexes) to prevent orphaned records and inconsistent state.

**3. Secure Operation Execution**
All user interactions, especially mutations (create, update, delete), must be guarded against common web vulnerabilities:
*   **Injection Attacks:** Ensuring user input (like post content or URL parameters) cannot be interpreted as executable code by the database (SQL injection) or browser (Cross-Site Scripting - XSS).
*   **Cross-Site Request Forgery (CSRF):** Preventing malicious websites from tricking a logged-in user's browser into submitting unauthorized requests to the blog platform.
*   **Broken Access Control:** Rigorously verifying a user's permission to perform an action on a specific resource *every time*, not just checking if they are logged in.

**4. Responsive & Intuitive Presentation**
The user interface must be functional and pleasant across devices:
*   **Dynamic Content Rendering:** Choosing a strategy for converting stored data (like Markdown) into interactive HTML pages. This decision affects initial page load speed, search engine optimization (SEO), and development complexity.
*   **State Management:** Handling the user's authentication state, form inputs, and UI loading states predictably across different pages and components.
*   **Layout Adaptation:** Using CSS techniques (Flexbox, Grid) to ensure the interface rearranges itself gracefully for mobile screens without manual zooming or horizontal scrolling.

### Existing Approaches Comparison

Several architectural and technological decisions present themselves at the outset of the project. The choices made here set the trajectory for implementation. Below is a comparison of common patterns, evaluated for their suitability in an *educational* context where the goals are learning fundamental concepts, maintaining simplicity, and building a complete, working system.

> **Decision: Monolithic Application Architecture**
> - **Context**: We are building a complete but bounded application (a blog) with a small team (often a single learner). The primary goal is learning full-stack fundamentals, not building a massively scalable service.
> - **Options Considered**:
>     1. **Monolithic Architecture**: A single codebase and deployment unit containing the user interface, business logic, and data access layers.
>     2. **Microservices Architecture**: Decomposing the system into independently deployable services (e.g., Auth Service, Post Service, Comment Service).
>     3. **Serverless/Function-as-a-Service**: Implementing business logic as stateless functions triggered by HTTP events.
> - **Decision**: Use a monolithic architecture.
> - **Rationale**: A monolith drastically reduces operational and cognitive complexity for learners. They can reason about the entire data flow within one codebase, debug with a single process, and manage transactions and data consistency without distributed systems concerns. This aligns perfectly with the goal of learning foundational patterns without the overhead of service discovery, inter-service communication, and distributed tracing.
> - **Consequences**: The system will be easier to develop, test, and deploy initially. However, it will not demonstrate patterns for scaling horizontally or managing independent lifecycles for different system capabilities. These are considered advanced topics beyond the project's learning objectives.

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **Monolithic** | Single codebase, simple debugging, easy data consistency, straightforward deployment. | Coupled components, limits horizontal scaling, can become complex as code grows. | **Yes** - Best fit for learning fundamentals. |
| **Microservices** | Independent scaling, technology flexibility, clear service boundaries. | High operational complexity, requires network communication, hard to debug, eventual consistency challenges. | No - Introduces advanced distributed systems concepts prematurely. |
| **Serverless** | No server management, automatic scaling, pay-per-use. | Vendor lock-in, cold starts, limited execution time, debugging complexity. | No - Abstracts away server concepts learners need to understand. |

> **Decision: Hybrid Server-Side Rendering (SSR) & Client-Side Rendering (CSR)**
> - **Context**: Blog content must be publicly accessible to search engines (good SEO) while providing a dynamic, app-like experience for authenticated users (e.g., live Markdown preview in the editor).
> - **Options Considered**:
>     1. **Client-Side Rendering (CSR)**: The server sends a minimal HTML shell and a JavaScript bundle. The browser fetches data via API and renders the page.
>     2. **Server-Side Rendering (SSR)**: The server generates the complete HTML for a page and sends it to the browser.
>     3. **Hybrid/Static Generation**: Pre-rendering public pages at build time (SSG) and using CSR for authenticated, dynamic pages.
> - **Decision**: Use a hybrid approach: SSR for public pages (homepage, post detail) and CSR for authenticated pages (editor, dashboard). This can be achieved with a modern meta-framework (like Next.js).
> - **Rationale**: Public pages benefit from SSR's fast initial render and SEO advantages. Authenticated pages are user-specific and highly interactive, making CSR a better fit. Using a framework that supports both patterns allows learners to experience and compare both rendering models, which is a key frontend concept.
> - **Consequences**: Requires a slightly more advanced frontend setup (e.g., Next.js, Nuxt, SvelteKit) compared to a simple Create-React-App setup. The build process becomes more important, and routing logic must account for rendering strategies.

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **Client-Side (CSR)** | Rich interactivity, fast navigation after load, simple frontend architecture. | Poor initial SEO, blank page during initial load, content not visible without JS. | No - Unsuitable for public blog SEO. |
| **Server-Side (SSR)** | Excellent SEO, fast initial perceived load, content always available. | Higher server load, slower time-to-interactive (TTI), more complex server setup. | **Partial** - For public pages. |
| **Hybrid (SSR/CSR)** | Best of both worlds: SEO for public content, interactivity for private. | Most complex implementation, requires framework support. | **Yes** - Balanced approach for learning and function. |

> **Decision: SQL-based Relational Database**
> - **Context**: Data is highly structured (Users, Posts, Comments) with clear relationships (a User *has many* Posts). We require strong consistency for operations like user registration and post ownership.
> - **Options Considered**:
>     1. **SQL (e.g., PostgreSQL, SQLite)**: A relational database with a predefined schema, ACID transactions, and JOIN operations.
>     2. **Document NoSQL (e.g., MongoDB)**: A schema-less database storing JSON-like documents, favoring flexibility and horizontal scalability.
> - **Decision**: Use an SQL database (PostgreSQL for production-like, SQLite for simplicity).
> - **Rationale**: SQL databases force learners to think carefully about data design through schema definition. Concepts like foreign key constraints, indexes, and normalized data are fundamental to backend development. The blog's data model is inherently relational, and SQL provides powerful, declarative querying (via an ORM or raw SQL) that is a critical industry skill. ACID transactions simplify operations like creating a user and their first post in a single, atomic unit.
> - **Consequences**: Requires upfront schema design and migrations for changes. Less flexible if the data model changes frequently, but this is a positive constraint for learning. Performance for highly hierarchical or unstructured data might be less optimal than a document store, but this is not a concern for this project.

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **SQL (PostgreSQL)** | Strong consistency, ACID transactions, powerful query language, explicit schema. | Schema rigidity, scaling can be complex, requires understanding of relationships. | **Yes** - Excellent for learning data modeling fundamentals. |
| **NoSQL (MongoDB)** | Schema flexibility, horizontal scalability, JSON-native for JavaScript. | Eventual consistency, no joins (must manage relationships in code), less query power. | No - Sacrifices foundational data integrity concepts for flexibility not needed here. |

By anchoring the project with a monolithic, SQL-based backend and a hybrid-rendering frontend, we create a constrained environment that prioritizes learning core, transferable web development concepts over exploring the vast landscape of modern architectural patterns. This focused approach provides a solid foundation upon which more advanced topics can be built in future iterations or projects.

### Implementation Guidance

This section bridges the high-level design to initial project setup, providing concrete technology recommendations and a foundational code structure.

**A. Technology Recommendations Table**

| Component | Simple Option (Rapid Start) | Advanced Option (Production-like) |
| :--- | :--- | :--- |
| **Backend Framework** | Express.js (Node.js) / Flask (Python) | NestJS (Node.js/TS) / Django (Python) / Fiber (Go) |
| **Database & ORM** | SQLite with `better-sqlite3` or `sqlite3` / SQLAlchemy (Python) | PostgreSQL with `node-postgres` (pg) / Prisma (Node.js/TS) / SQLAlchemy (Python) |
| **Authentication** | Manual JWT with `jsonwebtoken` & `bcrypt` | Passport.js (Node.js) / `authlib` (Python) with OAuth2 support |
| **Frontend Framework** | Create-React-App (CRA) with React Router | Next.js (React) / Nuxt (Vue) / SvelteKit |
| **Styling** | Plain CSS / CSS Modules | Tailwind CSS / Styled-Components |
| **Markdown Processing** | `marked` (Node.js) / `markdown2` (Python) | `remark` and `rehype` ecosystem with sanitization |

**B. Recommended File/Module Structure**

A well-organized project from the start prevents "spaghetti code" and makes the architecture tangible. Below is a structure for a **Node.js/Express + React** monorepo, which is a common and effective setup for full-stack JavaScript projects.

```
blog-platform/
├── server/                    # Backend API (Express.js)
│   ├── package.json
│   ├── src/
│   │   ├── config/           # Environment variables, database config
│   │   │   └── database.js
│   │   ├── models/           # Database models (ORM) or data access layer
│   │   │   ├── User.js
│   │   │   ├── Post.js
│   │   │   └── Comment.js
│   │   ├── migrations/       # Database migration files (versioned)
│   │   │   ├── 001_create_users_table.js
│   │   │   └── 002_create_posts_table.js
│   │   ├── routes/           # API route handlers grouped by resource
│   │   │   ├── auth.js
│   │   │   ├── posts.js
│   │   │   └── index.js      # Combines and exports all routes
│   │   ├── middleware/       # Custom Express middleware
│   │   │   ├── auth.js       # JWT validation middleware
│   │   │   ├── errorHandler.js
│   │   │   └── notFound.js
│   │   ├── utils/            # Helper functions (hashing, validation)
│   │   │   ├── bcrypt.js
│   │   │   ├── jwt.js
│   │   │   └── validation.js
│   │   └── app.js            # Express app setup (middleware, routes)
│   └── server.js             # Entry point (starts the server)
├── client/                   # Frontend React application
│   ├── package.json
│   ├── public/
│   └── src/
│       ├── components/       # Reusable UI components
│       │   ├── Layout/
│       │   ├── PostCard/
│       │   └── MarkdownEditor/
│       ├── pages/           # Page-level components (mapped to routes)
│       │   ├── HomePage.js
│       │   ├── LoginPage.js
│       │   ├── PostPage.js
│       │   └── EditorPage.js
│       ├── context/         # React Context for global state (e.g., Auth)
│       │   └── AuthContext.js
│       ├── hooks/           # Custom React hooks
│       ├── utils/           # Frontend utilities (API client, formatters)
│       │   └── api.js
│       ├── App.js           # Main app component with router
│       └── index.js         # React DOM render
└── package.json             # (Optional) Root package.json for shared scripts
```

**C. Infrastructure Starter Code: Database Connection & Configuration**

Setting up a robust database connection is a prerequisite. Below is a complete, reusable module for connecting to PostgreSQL using connection pooling.

*File: `server/src/config/database.js`*
```javascript
const { Pool } = require('pg');
require('dotenv').config(); // Loads environment variables from .env file

/**
 * Configuration for PostgreSQL connection pool.
 * Environment variables should be set in a `.env` file.
 */
const poolConfig = {
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT) || 5432,
  database: process.env.DB_NAME || 'blog_platform',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || '',
  // Connection pool settings
  max: parseInt(process.env.DB_POOL_MAX) || 20, // max number of clients in the pool
  idleTimeoutMillis: parseInt(process.env.DB_POOL_IDLE_TIMEOUT) || 30000, // how long a client is allowed to remain idle
  connectionTimeoutMillis: parseInt(process.env.DB_POOL_CONN_TIMEOUT) || 2000, // how long to wait for a connection
};

// Create the pool instance
const pool = new Pool(poolConfig);

// Log connection success/errors (useful for debugging)
pool.on('connect', () => {
  console.log('Database connection established.');
});
pool.on('error', (err) => {
  console.error('Unexpected database pool error:', err);
  // In a production app, you might want to gracefully shut down or alert
});

/**
 * Helper function to execute a SQL query with parameters.
 * This is the primary way to interact with the database.
 * @param {string} text - The SQL query string.
 * @param {Array} params - The query parameters (for parameterized queries).
 * @returns {Promise<QueryResult>} The result from the database.
 */
const query = (text, params) => {
  return pool.query(text, params);
};

/**
 * Helper function to get a client from the pool for transactions.
 * Remember to call `release()` on the client when done.
 * @returns {Promise<PoolClient>} A single client from the pool.
 */
const getClient = () => {
  return pool.connect();
};

module.exports = {
  query,
  getClient,
  // Export the pool for direct access if needed (e.g., for shutting down)
  pool,
};
```

**D. Core Logic Skeleton: Health Check Endpoint**

A health check endpoint is a critical first route to verify the database connection and server status. Implement this in your main app file.

*File: `server/src/app.js` (snippet)*
```javascript
const express = require('express');
const { query } = require('./config/database');

const app = express();

// ... other middleware setup (json parsing, etc.)

/**
 * Health check endpoint.
 * TODO 1: Define a GET route at the path '/api/health'
 * TODO 2: Inside the handler, attempt a simple database query (e.g., `SELECT 1`)
 * TODO 3: If the query succeeds, respond with status 200 and a JSON: { status: 'ok', timestamp: new Date().toISOString() }
 * TODO 4: If the query fails (catches an error), respond with status 503 (Service Unavailable) and a JSON: { status: 'error', error: err.message }
 * TODO 5: Add a catch-all middleware at the end of your route definitions to handle 404 errors for undefined routes.
 */
app.get('/api/health', async (req, res) => {
  // TODO: Implement the steps above.
  // Hint: Use `await query('SELECT 1')` and wrap it in a try/catch block.
});

// ... rest of your routes

// TODO 5: Catch-all 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

module.exports = app;
```

**E. Language-Specific Hints (JavaScript/Node.js)**
*   **Environment Variables:** Use the `dotenv` package to load configuration from a `.env` file. Never commit this file to version control. Add `.env` to your `.gitignore`.
*   **Database Security:** Always use **parameterized queries** (as shown in the `query` helper) or an ORM to prevent SQL injection. Never concatenate user input directly into SQL strings.
*   **Connection Pooling:** The `pg` library's `Pool` is essential for handling multiple concurrent database requests efficiently. Configure the pool size based on your expected load.

**F. Milestone 1 Checkpoint**
After setting up the database configuration and health check endpoint, verify your progress:
1.  **Run the server:** `cd server && npm run dev` (assuming you have a script `"dev": "nodemon server.js"`).
2.  **Test the health endpoint:** Use `curl http://localhost:3000/api/health` or open the URL in your browser.
    *   **Expected Success Output:** `{"status":"ok","timestamp":"2023-10-01T12:00:00.000Z"}`
    *   **Expected Database Error Output:** `{"status":"error","error":"connection refused"}`
3.  **Signs of Trouble:**
    *   **`ECONNREFUSED`:** Your database server is not running. Start PostgreSQL (`brew services start postgresql` on macOS, `sudo service postgresql start` on Linux).
    *   **`role "username" does not exist` or authentication errors:** Check your `.env` file credentials or create the database/user specified.
    *   **`Cannot find module 'pg'`:** Run `npm install pg dotenv` in your `server/` directory.

---


## Goals and Non-Goals

> **Milestone(s):** Milestone 1 (Project Setup & Database Schema), Milestone 2 (User Authentication), Milestone 3 (Blog CRUD Operations), Milestone 4 (Frontend UI)

This section establishes the boundaries of our blog platform by defining what the system must accomplish and, equally important, what it will not include. Clear scope definition prevents feature creep and ensures we build a cohesive, learnable system rather than an unwieldy collection of features. Think of this as the project's constitution—it defines the fundamental rights and responsibilities of the system, leaving room for future amendments (extensions) but establishing a stable foundation for development.

### Functional Goals

The functional goals represent the core user-facing capabilities our blog platform must provide. These align directly with the four milestones and can be visualized as a publishing workflow: users join the platform, create content, publish it for consumption, and engage with readers through comments. Each function serves a specific role in this workflow.

| Function | Description | User Role | Corresponding Milestone |
|----------|-------------|-----------|-------------------------|
| **User Registration** | Allows new users to create an account by providing email, password, and name. The system validates email format, enforces password strength, and ensures email uniqueness. | Reader → Author | Milestone 2 |
| **User Login/Logout** | Enables registered users to authenticate using credentials and obtain a session. Provides secure session termination. | Author | Milestone 2 |
| **Password Recovery** | Allows users to reset forgotten passwords via time-limited email tokens, maintaining security while providing practical access recovery. | Author | Milestone 2 |
| **Post Creation** | Enables authenticated users to compose new blog posts with title, markdown content, and optional tags. Stores raw markdown for editing flexibility. | Author | Milestone 3 |
| **Post Listing** | Displays paginated posts sorted chronologically (newest first) with titles, excerpts, author names, and dates. Supports basic filtering by tags. | Reader | Milestone 3, 4 |
| **Post Detail View** | Renders a complete post with markdown converted to sanitized HTML, showing full content, author attribution, publish date, and comment thread. | Reader | Milestone 3, 4 |
| **Post Editing** | Allows the original author to modify their existing posts' title and content while preserving the creation date. | Author | Milestone 3 |
| **Post Deletion** | Allows authors to remove their posts, with optional soft-delete functionality (marking as deleted without physical removal). | Author | Milestone 3 |
| **Comment Creation** | Enables authenticated users to add threaded comments to posts, supporting discussions while maintaining civil discourse through basic moderation. | Reader/Author | Milestone 3 (implied) |
| **Responsive Presentation** | Provides a usable interface across desktop, tablet, and mobile devices without horizontal scrolling or content overlap. | All users | Milestone 4 |

These functions represent the minimum viable product for a blog platform. The system prioritizes **clarity over complexity**—each feature serves a distinct purpose in the content creation and consumption lifecycle. Notice the pattern: all content modification operations (create, edit, delete posts) require authentication, while content consumption (listing, reading) remains publicly accessible. This follows the principle of least privilege while maintaining an open reading experience.

### Non-Functional Goals

Beyond what the system does, we define how well it must perform through measurable quality attributes. These non-functional goals represent the system's "-ilities"—characteristics that determine user satisfaction and long-term viability. Think of these as the building codes for our digital structure: they don't change what rooms we build, but ensure they're safe, accessible, and durable.

#### Performance Requirements

| Requirement | Target Metric | Rationale | Measurement Method |
|-------------|---------------|-----------|--------------------|
| **Page Load Time** | < 2 seconds for above-the-fold content on desktop with broadband connection. | User attention spans are short; delayed loading increases bounce rates. | Browser DevTools Network panel, Lighthouse audit. |
| **API Response Time** | < 200ms for 95% of authenticated requests under normal load. | Interactive applications feel responsive when feedback occurs within human perception thresholds. | Server-side request timing middleware, logging percentiles. |
| **Database Query Performance** | < 50ms for indexed primary key lookups; < 100ms for simple joins with proper indexes. | Database latency directly impacts all higher-level operations; slow queries bottleneck the entire system. | Database query execution plans, EXPLAIN ANALYZE. |
| **Concurrent Users** | Support 100 simultaneous authenticated users without significant degradation (< 20% increase in response times). | Educational projects need headroom beyond single-user testing; this ensures architecture can handle realistic classroom use. | Load testing with tools like Artillery or k6. |
| **First Contentful Paint** | < 1.5 seconds on mobile devices over 3G networks. | Mobile users represent significant traffic; progressive enhancement ensures accessibility across network conditions. | Lighthouse mobile simulation, WebPageTest. |

#### Security Requirements

| Requirement | Implementation Approach | Rationale | Verification Method |
|-------------|------------------------|-----------|---------------------|
| **Authentication Bypass Protection** | All protected routes validate JWT signatures and expiration before processing. | Prevents unauthorized access to privileged operations like post modification. | Manual testing with modified/expired tokens, automated security scanning. |
| **Injection Prevention** | All database queries use parameterized queries; all user-generated HTML is sanitized before rendering. | SQL injection and XSS remain top OWASP vulnerabilities; parameterization separates code from data. | Static analysis tools, manual penetration testing with payloads like `' OR '1'='1`. |
| **Credential Protection** | Passwords hashed with bcrypt (work factor 12); never stored or logged in plaintext. | Password breaches have severe consequences; strong hashing protects users even if database is compromised. | Code review of password handling, verification that plaintext passwords never appear in logs. |
| **Session Security** | JWTs signed with strong secret (HS256) or asymmetric crypto; tokens transmitted via HTTP-only cookies (with SameSite=Strict). | Prevents token theft via XSS and CSRF attacks while maintaining stateless scalability. | Browser DevTools inspection for token storage, CSRF testing with unauthorized origin requests. |
| **Data Privacy** | Users can only access their own posts for modification; authorization checks verify resource ownership. | Ensures users cannot edit or delete others' content, maintaining trust in the platform. | Manual testing with multiple user accounts attempting cross-user operations. |

#### Maintainability Requirements

| Requirement | Implementation Approach | Rationale | Measurement Indicator |
|-------------|------------------------|-----------|-----------------------|
| **Code Organization** | Consistent file/folder structure following framework conventions; separation of concerns (routes, models, services). | New developers can navigate the codebase quickly; reduces cognitive load during maintenance. | Existence of documented project structure, consistency across modules. |
| **Test Coverage** | > 80% line coverage for core business logic (authentication, post operations, markdown rendering). | Tests catch regressions during refactoring and provide living documentation of expected behavior. | Test runner coverage reports, CI pipeline enforcement. |
| **Documentation** | All public APIs documented with usage examples; complex algorithms include inline comments explaining intent. | Reduces the learning curve for contributors and future maintainers. | Presence of API documentation, meaningful commit messages. |
| **Configuration Management** | Environment-specific configuration (database URLs, secrets) externalized from code; sensitive values never committed. | Enables deployment across environments (dev, staging, production) without code changes. | Absence of hardcoded credentials, use of environment variables or config files. |
| **Error Handling** | Consistent error response format (JSON with code, message, details); all unexpected exceptions caught and logged. | Provides debuggable production issues while giving users actionable feedback. | Structured error logs, uniform API error responses. |

> **Architecture Decision: Quality Attribute Prioritization**
> 
> **Context**: We must balance competing quality attributes (performance, security, maintainability) within the constraints of an educational project.
> 
> **Options Considered**:
> 1. **Security-First**: Maximize security at the expense of developer experience and some performance (e.g., requiring 2FA, extensive input validation slowing requests).
> 2. **Performance-First**: Optimize for speed and responsiveness, potentially sacrificing security rigor (e.g., skipping some validation layers).
> 3. **Balanced Approach**: Meet minimum thresholds for all attributes while prioritizing security for authentication/data protection and performance for user-facing operations.
> 
> **Decision**: Balanced approach with security as the non-negotiable foundation.
> 
> **Rationale**: As an educational platform handling user data, security failures would undermine trust and create real risks. However, we must avoid overwhelming learners with overly complex security implementations. We implement industry-standard practices (bcrypt, JWT, parameterized queries) that provide strong protection without excessive complexity. Performance targets are set at "good enough" levels that encourage thoughtful design without premature optimization.
> 
> **Consequences**: The system will be secure against common web vulnerabilities while remaining performant for its intended scale. Learners will experience implementing production-grade security patterns but won't need to master advanced topics like cryptographic rotation or DDoS mitigation initially.

### Explicit Non-Goals

Clearly stating what we will not build is as important as defining what we will build. These non-goals establish boundaries that prevent scope creep and allow focused implementation on core learning objectives. Consider these the "not now, maybe later" features—they might be valuable additions but would distract from foundational full-stack concepts.

| Feature Category | Specific Exclusions | Rationale for Exclusion | Potential Future Extension |
|------------------|---------------------|-------------------------|----------------------------|
| **Real-time Features** | Live collaboration (multiple authors editing simultaneously), real-time comment updates (WebSocket push), typing indicators, presence awareness. | Real-time systems introduce significant complexity (WebSocket management, conflict resolution, state synchronization) that distracts from core HTTP request/response patterns. | Add WebSocket endpoints for comment live updates after mastering REST APIs. |
| **Advanced Search** | Full-text search with relevance ranking, faceted filtering (by date ranges, multiple tags), search suggestions/autocomplete. | Search implementation (especially with relevance algorithms) is a specialized domain requiring dedicated search engines (Elasticsearch) or advanced database features. | Integrate a dedicated search service (Elasticsearch, Algolia) via API calls. |
| **Recommendation Engines** | "Readers who liked this also liked..." suggestions, personalized content feeds, machine learning-based content discovery. | Recommendation algorithms require collecting and analyzing extensive user behavior data, introducing privacy considerations and algorithmic complexity. | Implement simple tag-based related posts or most-popular ranking based on view counts. |
| **Native Mobile Applications** | iOS or Android native apps, React Native/Flutter cross-platform mobile applications. | Mobile app development involves distinct toolchains, deployment processes, and platform-specific concerns that diverge from web fundamentals. | Progressive Web App (PWA) capabilities for mobile web experience. |
| **Social Features** | User profiles with avatars/bios, following/follower relationships, social sharing integrations (beyond basic URLs), notifications system. | Social features create complex relationship models and notification delivery systems that would overwhelm the core content-focused architecture. | Add user profile pages with bios and gravatar integration. |
| **Monetization** | Subscription paywalls, advertisement integration, sponsored content management, e-commerce for digital products. | Monetization introduces payment processing, access tiers, and compliance requirements (PCI, GDPR) far beyond educational scope. | Integrate a third-party payment processor (Stripe) for premium features. |
| **Advanced Media Management** | Image/video upload and processing, CDN integration, media galleries, embedded media transcoding. | Media handling requires storage infrastructure, processing pipelines, and optimization techniques that constitute a separate domain. | Integrate with cloud storage (S3) and image processing service (Imgix). |
| **Multi-language Support** | Internationalization (i18n) of UI text, content translation, locale-specific formatting. | i18n adds complexity to UI rendering and content management without enhancing core architectural learning. | Add React-Intl or i18next for UI translation framework. |
| **Administration Dashboard** | Comprehensive admin interface for user management, content moderation, analytics reporting, system configuration. | Admin tools require additional authorization layers (role-based access) and complex UI for data visualization. | Build simple admin endpoints protected by role checks, with basic UI. |
| **API Versioning** | Multiple concurrent API versions, backward compatibility layers, version negotiation. | API versioning is important for public APIs but adds complexity premature for a learning project with controlled clients. | Add version prefix (/v1/) to routes with plan for future /v2. |

> **Key Insight**: The most dangerous non-goals are those left unstated. By explicitly declaring these exclusions, we create a "negative space" that defines our system's shape more clearly than any list of features could. This clarity prevents the common beginner mistake of trying to build everything at once and instead focuses effort on mastering foundational patterns that can later support these extensions.

### Common Pitfalls in Scope Definition

⚠️ **Pitfall: Feature Creep During Implementation**
- **Description**: Adding "just one more small feature" that seems quick to implement but introduces dependencies and complexity that ripple through the system.
- **Why It's Wrong**: Each additional feature increases testing surface, creates new edge cases, and can compromise the clean separation of concerns established in the architecture.
- **How to Avoid**: Refer back to this Goals and Non-Goals section when considering new features. If a feature isn't listed here, document it in the "Future Extensions" section of the design doc rather than implementing it immediately.

⚠️ **Pitfall: Over-Engineering Non-Functional Requirements**
- **Description**: Implementing complex performance optimizations (caching layers, database read replicas) or security measures (HMAC request signing) before verifying they're needed.
- **Why It's Wrong**: Premature optimization adds complexity without measurable benefit and distracts from delivering working core functionality. Security theater (implementing security features incorrectly) can create false confidence.
- **How to Avoid**: Implement the simplest solution that meets the stated non-functional goals. Use monitoring to identify actual bottlenecks before optimizing. Follow established security patterns rather than inventing custom solutions.

⚠️ **Pitfall: Neglecting One Category of Goals**
- **Description**: Focusing exclusively on functional goals while ignoring security, or prioritizing performance while sacrificing maintainability.
- **Why It's Wrong**: Systems that work but are insecure, slow, or impossible to maintain fail in production. The interplay between these categories determines overall system success.
- **How to Avoid**: Review all three goal categories during each milestone. For example, when implementing user authentication (functional goal), simultaneously implement password hashing (security goal) and clean error handling (maintainability goal).

### Implementation Guidance

> **Technology Recommendations Table**

| Component | Simple Option | Advanced Option | Rationale for Simple Choice |
|-----------|---------------|-----------------|----------------------------|
| **Backend Framework** | Express.js (Node.js) | Nest.js (TypeScript) | Express has minimal abstraction, making HTTP and middleware concepts visible to learners. |
| **Database** | PostgreSQL with node-postgres | MongoDB with Mongoose | PostgreSQL's relational model teaches schema design, foreign keys, and ACID transactions. |
| **Authentication** | JWT with jsonwebtoken library | OAuth 2.0 with Passport.js | JWT demonstrates stateless authentication without session storage complexity. |
| **Markdown Processing** | marked library | Unified.js ecosystem (remark, rehype) | marked provides simple conversion with optional HTML sanitization via DOMPurify. |
| **Frontend Framework** | React with Create React App | Next.js with SSR/SSG | React's component model is foundational; CRA provides sensible defaults without configuration overhead. |
| **CSS Styling** | Plain CSS with CSS Grid/Flexbox | Tailwind CSS utility classes | Writing CSS directly teaches fundamentals before abstracting with utility frameworks. |
| **Testing** | Jest + Supertest | Cypress for E2E + Jest | Jest handles both unit and API tests; Supertest provides HTTP assertion interface. |
| **Deployment** | Heroku/Render (PaaS) | Docker + AWS/GCP | PaaS abstracts infrastructure while demonstrating deployment workflows. |

> **Recommended File/Module Structure**

The following structure organizes code by concern, separating infrastructure from business logic and frontend from backend. This organization scales well and teaches industry-standard patterns.

```
blog-platform/
├── backend/
│   ├── src/
│   │   ├── config/                 # Configuration management
│   │   │   ├── database.js         # DB connection pooling with `PoolConfig`
│   │   │   └── environment.js      # Environment variable validation
│   │   ├── db/
│   │   │   ├── migrations/         # Forward/rollback migration files
│   │   │   ├── seeds/              # Development seed data
│   │   │   └── index.js            # Database client with `query()` and `getClient()`
│   │   ├── models/                 # Data access layer
│   │   │   ├── User.js             # User model with authentication methods
│   │   │   ├── Post.js             # Post model with CRUD operations
│   │   │   └── Comment.js          # Comment model with thread support
│   │   ├── middleware/             # Express middleware
│   │   │   ├── auth.js             # JWT validation middleware
│   │   │   ├── errorHandler.js     # Structured error responses
│   │   │   └── validation.js       # Request schema validation
│   │   ├── routes/                 # API route definitions
│   │   │   ├── auth.js             # Register, login, logout, password reset
│   │   │   ├── posts.js            # Post CRUD operations
│   │   │   └── comments.js         # Comment operations
│   │   ├── services/               # Business logic layer
│   │   │   ├── AuthService.js      # Authentication business logic
│   │   │   ├── PostService.js      # Post-related business rules
│   │   │   └── MarkdownService.js  # Markdown rendering and sanitization
│   │   ├── utils/                  # Shared utilities
│   │   │   ├── logger.js           # Structured logging
│   │   │   ├── security.js         # bcrypt/JWT helpers
│   │   │   └── pagination.js       # Pagination parameter parsing
│   │   └── app.js                  # Express app configuration
├── frontend/
│   ├── public/                     # Static assets
│   └── src/
│       ├── components/             # Reusable UI components
│       │   ├── common/
│       │   │   ├── Button.js
│       │   │   ├── Input.js
│       │   │   └── Loader.js
│       │   ├── layout/
│       │   │   ├── Header.js       # Navigation with auth state
│       │   │   └── Layout.js       # Main layout wrapper
│       │   ├── posts/
│       │   │   ├── PostList.js     # Paginated post listing
│       │   │   ├── PostItem.js     # Individual post preview
│       │   │   └── PostEditor.js   # Markdown editor with preview
│       │   └── auth/
│       │       ├── LoginForm.js
│       │       └── RegisterForm.js
│       ├── context/                # React context for state sharing
│       │   └── AuthContext.js      # Authentication state provider
│       ├── hooks/                  # Custom React hooks
│       │   ├── useAuth.js          # Auth context consumer
│       │   └── usePosts.js         # Posts data fetching
│       ├── pages/                  # Page-level components
│       │   ├── HomePage.js         # Post listing page
│       │   ├── PostPage.js         # Single post view
│       │   ├── EditPostPage.js     # Post editor
│       │   ├── LoginPage.js
│       │   └── RegisterPage.js
│       ├── services/               # API communication
│       │   └── api.js              # Axios instance with interceptors
│       ├── utils/                  # Frontend utilities
│       │   └── formatDate.js       # Date formatting helpers
│       ├── App.js                  # Router configuration
│       └── index.js                # React entry point
├── package.json                    # Backend dependencies
├── package-lock.json
├── package.frontend.json           # Frontend dependencies (or use separate folder)
└── README.md                       # Project setup and goals
```

> **Infrastructure Starter Code**

**Database Connection Pool Configuration (`backend/src/config/database.js`)**:
This complete implementation provides production-ready connection pooling with the exact `PoolConfig` structure and methods specified in naming conventions.

```javascript
const { Pool } = require('pg');

// Configuration constants matching naming conventions
const DB_POOL_MAX = process.env.DB_POOL_MAX || 20;
const DB_POOL_IDLE_TIMEOUT = process.env.DB_POOL_IDLE_TIMEOUT || 30000;
const DB_POOL_CONN_TIMEOUT = process.env.DB_POOL_CONN_TIMEOUT || 2000;

/**
 * @typedef {Object} PoolConfig
 * @property {string} host - Database host
 * @property {number} port - Database port
 * @property {string} database - Database name
 * @property {string} user - Database user
 * @property {string} password - Database password
 * @property {number} max - Maximum number of clients in pool
 * @property {number} idleTimeoutMillis - Idle client timeout in milliseconds
 * @property {number} connectionTimeoutMillis - Connection timeout in milliseconds
 */

/**
 * Creates and configures a PostgreSQL connection pool
 * @returns {Pool} Configured connection pool
 */
function createPool() {
  const poolConfig = {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432'),
    database: process.env.DB_NAME || 'blogdb',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'postgres',
    max: parseInt(DB_POOL_MAX),
    idleTimeoutMillis: parseInt(DB_POOL_IDLE_TIMEOUT),
    connectionTimeoutMillis: parseInt(DB_POOL_CONN_TIMEOUT),
  };

  const pool = new Pool(poolConfig);

  // Test connection on startup
  pool.on('connect', (client) => {
    console.log('Database client connected');
  });

  pool.on('error', (err) => {
    console.error('Unexpected database pool error', err);
    process.exit(-1);
  });

  return pool;
}

const pool = createPool();

/**
 * @typedef {Object} QueryResult
 * @property {Array<any>} rows - Query result rows
 * @property {number} rowCount - Number of rows returned
 * @property {string} command - SQL command executed
 */

/**
 * Executes a parameterized SQL query using the connection pool
 * @param {string} text - SQL query text with parameter placeholders ($1, $2, etc.)
 * @param {Array<any>} params - Parameter values for the query
 * @returns {Promise<QueryResult>} Query execution result
 */
async function query(text, params) {
  const start = Date.now();
  try {
    const result = await pool.query(text, params);
    const duration = Date.now() - start;
    console.log('Executed query', { text, duration, rows: result.rowCount });
    return result;
  } catch (error) {
    console.error('Query error', { text, params, error: error.message });
    throw error;
  }
}

/**
 * Acquires a single client from the pool for transactions
 * @returns {Promise<PoolClient>} Database client for transaction management
 */
async function getClient() {
  const client = await pool.connect();
  
  // Add query method to client for consistent interface
  const originalQuery = client.query;
  const originalRelease = client.release;
  
  // Track query count for this client
  let queryCount = 0;
  
  client.query = (...args) => {
    queryCount++;
    return originalQuery.apply(client, args);
  };
  
  // Override release to log query count
  client.release = () => {
    console.log(`Client released after ${queryCount} queries`);
    originalRelease.apply(client);
  };
  
  return client;
}

module.exports = {
  query,
  getClient,
  pool, // Direct pool access for advanced use cases
};
```

> **Core Logic Skeleton Code**

**Main Application Setup (`backend/src/app.js`)**:
This skeleton establishes the Express application structure with placeholder middleware and route mounting.

```javascript
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');

// Import route modules
const authRoutes = require('./routes/auth');
const postRoutes = require('./routes/posts');
const commentRoutes = require('./routes/comments');

// Import middleware
const errorHandler = require('./middleware/errorHandler');
const { notFound } = require('./middleware/notFound');

/**
 * Creates and configures the Express application
 * @returns {express.Application} Configured Express app
 */
function createApp() {
  const app = express();
  
  // TODO 1: Apply security middleware (Helmet for security headers)
  // TODO 2: Apply CORS middleware (configure allowed origins)
  // TODO 3: Apply request logging middleware (Morgan for HTTP logging)
  // TODO 4: Apply JSON body parsing middleware
  // TODO 5: Apply URL-encoded body parsing middleware
  
  // Health check endpoint
  app.get('/health', (req, res) => {
    // TODO 6: Add database connection check
    res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
  });
  
  // API routes
  app.use('/api/auth', authRoutes);
  app.use('/api/posts', postRoutes);
  app.use('/api/comments', commentRoutes);
  
  // TODO 7: Add static file serving for frontend in production
  
  // 404 handler for undefined routes
  app.use(notFound);
  
  // Global error handler (must be last middleware)
  app.use(errorHandler);
  
  return app;
}

module.exports = createApp;
```

> **Language-Specific Hints**

- **Environment Variables**: Use the `dotenv` package to load environment variables from a `.env` file during development. Remember to never commit `.env` files to version control.
- **Promise Handling**: Always use `async/await` with try-catch blocks or `.catch()` for Promise-based operations. Unhandled Promise rejections can crash Node.js applications.
- **Error Propagation**: In Express middleware, call `next(error)` to pass errors to the centralized error handler rather than throwing synchronously.
- **Connection Pool Management**: The `node-postgres` pool automatically manages connections. Always release clients acquired via `getClient()` by calling `client.release()` in a finally block.
- **Type Checking**: Even without TypeScript, consider using JSDoc comments (as shown above) to document parameter and return types for better IDE support and code clarity.

> **Milestone Checkpoint for Milestone 1**

After implementing the project setup and database schema:

**Command to Run**:
```bash
# Start the database (using Docker for simplicity)
docker run --name blog-db -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:14

# Install dependencies
npm install

# Run migrations
npm run migrate:up

# Start the development server
npm run dev
```

**Expected Output**:
- Server starts on port 3000 (or specified PORT)
- Database connection logs show successful connection
- Visiting `http://localhost:3000/health` returns `{"status":"ok","timestamp":"..."}`
- Database contains `users`, `posts`, and `comments` tables with proper indexes

**Verification Steps**:
1. Check database tables exist: `psql -h localhost -U postgres -d blogdb -c "\dt"`
2. Verify foreign key constraints: `psql -h localhost -U postgres -d blogdb -c "\d+ posts"` should show `author_id` foreign key to `users(id)`
3. Test connection pool by making multiple concurrent requests to `/health` endpoint

**Signs Something Is Wrong**:
- Server fails to start: Check database connection string and ensure PostgreSQL is running
- Migrations fail: Verify SQL syntax in migration files; check for existing tables with same names
- Health check shows database error: Verify PostgreSQL credentials and network connectivity
- High latency on health endpoint: Adjust `DB_POOL_CONN_TIMEOUT` or check database resource limits

> **Debugging Tips**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **"Cannot find module" errors** | Missing dependencies or incorrect import paths | Check `package.json` for missing dependencies; verify file paths are correct relative to project root | Run `npm install`, check import statements match file structure |
| **Database connection timeout** | PostgreSQL not running, wrong port, or firewall blocking | Try connecting manually with `psql` using same credentials; check if port 5432 is listening with `netstat -tuln` | Start PostgreSQL service, correct connection string, adjust firewall rules |
| **"Table does not exist" errors** | Migrations not run or failed silently | Check migration logs; query `SELECT * FROM pg_migrations` or similar tracking table | Run migrations manually, check SQL syntax in migration files |
| **Connection pool exhausted** | Too many concurrent requests or connections not being released | Monitor pool usage with `SELECT * FROM pg_stat_activity`; add logging to `getClient()` release | Increase `DB_POOL_MAX`, ensure `client.release()` is called in finally blocks |
| **Inconsistent data across requests** | Missing transaction boundaries for multi-step operations | Check if related operations are wrapped in BEGIN/COMMIT blocks | Use `getClient()` for transactions: `const client = await getClient(); try { await client.query('BEGIN'); ... await client.query('COMMIT'); } finally { client.release(); }` |
| **Environment variables undefined** | `.env` file not loaded or variables misspelled | Log `process.env` to see what's loaded; check `.env` file exists in project root | Load `dotenv` early in entry point: `require('dotenv').config()` |


## High-Level Architecture

> **Milestone(s):** Milestone 1 (Project Setup & Database Schema), Milestone 2 (User Authentication), Milestone 3 (Blog CRUD Operations), Milestone 4 (Frontend UI)

This section outlines the overall architecture of the blog platform, describing how the system's components are organized, their responsibilities, and how they interact. Think of this as the blueprint for a publishing house—it shows the different departments (components), what each one handles, and how information flows between them to turn a writer's draft into a published article that readers can view.

### Component Diagram Walkthrough

Imagine the blog platform as a small, efficient publishing company with three main departments:

1.  **The Front Desk (Client Browser)**: This is the public-facing interface where readers arrive and authors work. It displays the published magazine (the blog) and provides the writing desk (editor) for authors. Its job is to present information beautifully and send requests for new content or actions to the back office.
2.  **The Back Office & Printing Press (API Server)**: This is the engine room. It receives requests from the front desk, validates them (checking author credentials, ensuring requests are proper), processes the logic (fetching an article, saving a new draft), and interacts with the massive archive to retrieve or store data. It then sends formatted responses back to the front desk.
3.  **The Archive (Database)**: This is the secure, organized vault where every article, author profile, and reader comment is permanently stored. It doesn't do any processing itself; it only responds to precise requests from the back office to store or retrieve specific records.

![System Component Diagram](./diagrams/system-component.svg)

**Interaction Patterns:**

The system follows a request-response pattern over HTTP(S), which is the standard "conversation" protocol of the web.

*   **Client → API Server (HTTP Request)**: Every user action—loading the homepage, submitting a login, publishing a post—triggers an HTTP request from the client browser to the API server. This request contains:
    *   **Method & Path**: What action to perform (GET, POST, PUT, DELETE) and on what resource (`/api/posts`, `/api/login`).
    *   **Headers**: Metadata, such as the `Content-Type` (e.g., `application/json`) and, crucially, the `Authorization` header carrying the user's `JWT` for authenticated requests.
    *   **Body (Optional)**: The data for the request, like the JSON object `{"email": "user@example.com", "password": "secret"}` for login.

*   **API Server → Database (Parameterized Query)**: Upon receiving a valid request, the API server translates it into one or more database operations. It never concatenates user input directly into SQL strings. Instead, it uses **Parameterized Query** placeholders (like `$1`, `$2`) to separate instructions from data, which is the primary defense against SQL injection attacks. The server uses a connection `PoolConfig` to efficiently manage multiple simultaneous conversations with the database.

*   **Database → API Server (QueryResult)**: The database executes the query and returns a `QueryResult` object containing the affected `rows` (data) and `rowCount`.

*   **API Server → Client (HTTP Response)**: The API server formats the `QueryResult` into a clean JSON response, sets the appropriate HTTP status code (200 for success, 404 for not found, 401 for unauthorized), and sends it back. For operations like login, it may also attach a secure, HTTP-only cookie containing the signed `JWT`.

This separation of concerns—presentation (client), business logic and routing (API server), and data persistence (database)—is the hallmark of a **Monolithic Architecture** where all components are part of a single deployable unit. This is ideal for our learning project due to its simplicity and coherence.

### Recommended File/Module Structure

Organizing code well is like setting up a clean, logical workshop. Each tool has its place, making development faster and maintenance easier. Below is the recommended structure for our Node.js/Express backend and a React frontend, grouped by concern.

**Backend (API Server) Structure:**

```
blog-platform-backend/
├── package.json
├── .env                          # Environment variables (DB_URL, JWT_SECRET)
├── src/
│   ├── index.js                  # Application entry point: creates Express app, connects to DB, mounts routes.
│   ├── app.js                    # Express app configuration (middleware setup: CORS, JSON parsing, logging).
│   ├── config/
│   │   └── database.js           # Exports the configured database `PoolConfig` and helper functions (`query`, `getClient`).
│   ├── routes/
│   │   ├── index.js              # Main router that imports and mounts all feature-specific route modules.
│   │   ├── auth.routes.js        # Authentication routes: POST /register, POST /login, POST /logout, POST /reset-password.
│   │   ├── posts.routes.js       # Blog post CRUD routes: GET /posts, POST /posts, GET /posts/:id, etc.
│   │   └── comments.routes.js    # Comment routes (optional extension).
│   ├── controllers/
│   │   ├── auth.controller.js    # Functions that handle auth request logic: validate, hash password, generate JWT.
│   │   ├── posts.controller.js   # Functions for post CRUD logic: validate ownership, render markdown.
│   │   └── comments.controller.js
│   ├── models/
│   │   ├── User.model.js         # Data access layer for users. Methods like `findByEmail`, `create`.
│   │   ├── Post.model.js         # Data access layer for posts. Methods like `findAllPaginated`, `update`.
│   │   └── Comment.model.js
│   ├── middleware/
│   │   ├── auth.middleware.js    # `authenticateToken`: Validates JWT, attaches user to `req.user`.
│   │   ├── validate.middleware.js# Validation using a library like Joi or express-validator.
│   │   └── error.middleware.js   # Centralized error handler; catches errors and sends formatted 4xx/5xx responses.
│   ├── utils/
│   │   ├── security.js           # Password hashing (`bcrypt` helpers), JWT signing/verification.
│   │   ├── markdown.js           # Function to convert markdown to sanitized HTML (using `marked` and `DOMPurify`).
│   │   └── pagination.js         # Helper to build pagination metadata (page, limit, total, next/prev links).
│   └── migrations/               # SQL files or a framework (like `node-pg-migrate`) to manage schema changes.
│       ├── 001_create_users_table.sql
│       └── 002_create_posts_table.sql
└── tests/                        # Unit and integration tests mirroring the `src/` structure.
```

**Frontend (Client) Structure:**

```
blog-platform-frontend/
├── package.json
├── public/
│   └── index.html
└── src/
    ├── index.js                  # React app entry point, renders the root `App` component.
    ├── App.js                    # Root component: sets up Router, provides global Context (Auth).
    ├── components/               # Reusable, presentational components (buttons, cards, inputs).
    │   ├── Button.jsx
    │   ├── Card.jsx
    │   └── MarkdownPreview.jsx
    ├── layouts/                  # Layout components that define page structure.
    │   └── MainLayout.jsx        # Contains header, navigation, and renders child page content.
    ├── pages/                    # Page-level components, each corresponding to a route.
    │   ├── HomePage.jsx          # Lists blog posts.
    │   ├── LoginPage.jsx         # Login form.
    │   ├── RegisterPage.jsx      # Registration form.
    │   ├── PostPage.jsx          # Displays a single post.
    │   ├── EditPostPage.jsx      # Form to create/edit a post (with markdown editor).
    │   └── DashboardPage.jsx     # User's post management (optional).
    ├── hooks/                    # Custom React hooks for reusable logic.
    │   ├── useAuth.js            # Hook to access authentication state and methods (login, logout).
    │   └── useApi.js             # Hook to make authenticated API calls with error handling.
    ├── contexts/                 # React Context providers for global state.
    │   └── AuthContext.jsx       # Provides `{ user, login, logout }` to entire app.
    ├── services/                 # Modules for interacting with the backend API.
    │   └── api.js                # Axios instance with base URL and request/response interceptors for attaching JWTs.
    └── styles/                   # CSS modules or styled-components.
        └── global.css
```

### Technology Stack Recommendations

Choosing technologies is about balancing ease of learning, development speed, and production readiness. For an intermediate learning project, we recommend the "Simple Option" column for its gentle learning curve and strong community support. The "Advanced Option" introduces more powerful, specialized tools that are valuable to learn as you grow.

> **Decision: Full-Stack JavaScript Monolith**
> - **Context**: We need a cohesive, beginner-to-intermediate friendly stack that allows a single developer to understand the entire flow from database to UI. The project has clear frontend and backend components but doesn't require microservices.
> - **Options Considered**:
>     1.  **Monolithic Full-Stack JS (Node.js + React)**: A single language across the stack, unified tooling, vast ecosystem.
>     2.  **Split-Stack (Python/Flask + React)**: Leverages Python's simplicity on the backend but requires context-switching between languages.
>     3.  **Server-Side Rendered Framework (Next.js/Nuxt)**: Blurs the frontend/backend boundary, excellent for SEO and performance, but abstracts away traditional client-server separation.
> - **Decision**: Option 1: Monolithic Full-Stack JavaScript (Node.js + Express + React).
> - **Rationale**: Using JavaScript/TypeScript across the stack drastically reduces cognitive load for learners. They can apply knowledge of language syntax, Promises, and modules consistently. The separation between a RESTful Express API and a React client is a fundamental, transferable pattern in web development. It teaches clear API design and state management.
> - **Consequences**: We get a clean separation of concerns. The backend is a standalone API that could later serve mobile apps. The downside is managing two separate projects (backend and frontend) and dealing with Cross-Origin Resource Sharing (CORS) during development.

The table below compares specific technology choices within this stack.

| Component | Simple Option (Recommended) | Advanced Option (For Exploration) | Rationale for Recommendation |
| :--- | :--- | :--- | :--- |
| **Backend Framework** | **Express.js** | Fastify, NestJS | Express is minimal, unopinionated, and has the largest ecosystem. It forces you to understand middleware composition and routing fundamentals. Fastify offers better performance; NestJS provides a full, Angular-like structure. |
| **Database ORM/Driver** | **`node-postgres` (pg) + raw SQL** | Prisma, TypeORM | Using the raw `pg` driver with `Parameterized Query` teaches SQL fundamentals and prevents "magic" abstraction. ORMs like Prisma are excellent for productivity and type safety but hide SQL details crucial for learning. |
| **Authentication Library** | **`jsonwebtoken` + `bcrypt`** | Passport.js | Manually implementing JWT signing/verification and password hashing with `bcrypt` provides invaluable security insight. Passport.js is a robust middleware but abstracts many steps. |
| **Frontend Framework** | **React (with Create React App)** | Next.js, Vue 3 | React's component model is industry-standard. Create React App offers a zero-config start. Next.js (the **SSR**/**CSR** hybrid) is a natural upgrade for better SEO and performance. |
| **Frontend State Management** | **React Context API + `useReducer`** | Redux Toolkit, Zustand | For the scale of this app (user auth, post data), React Context is sufficient and introduces less complexity than Redux. It teaches prop drilling avoidance without a new paradigm. |
| **Styling** | **CSS Modules / Plain CSS** | Tailwind CSS, Styled-Components | CSS Modules keep styles scoped and simple, reinforcing core CSS skills. Utility-first CSS (Tailwind) is highly productive but has a unique learning curve. |
| **API Client** | **Fetch API** | Axios | The native `fetch` API is sufficient and avoids an external dependency. Axios provides slightly nicer defaults and interceptors, which are helpful for global JWT attachment. |
| **Validation** | **Joi or express-validator** | Class-validator (with TypeScript) | Server-side validation is non-negotiable. Joi provides a rich, readable schema API. Using it in middleware keeps controllers clean. |

### Implementation Guidance

#### A. Technology Recommendations Table
(See the comprehensive table in the section above.)

#### B. Recommended File/Module Structure
(See the detailed directory trees in the section above.)

#### C. Infrastructure Starter Code

**1. Database Configuration (`src/config/database.js`):**
This is a complete, reusable module for setting up a PostgreSQL connection pool. It abstracts the pool creation and provides the crucial `query` helper function that ensures all queries are parameterized.

```javascript
// src/config/database.js
const { Pool } = require('pg');
require('dotenv').config(); // Loads DB_URL from .env file

/**
 * @type {PoolConfig} Configuration for the PostgreSQL connection pool.
 * In practice, these values are read from environment variables.
 */
const poolConfig = {
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT, 10) || 5432,
  database: process.env.DB_NAME || 'blog_platform',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || '',
  max: parseInt(process.env.DB_POOL_MAX, 10) || 20, // DB_POOL_MAX
  idleTimeoutMillis: parseInt(process.env.DB_POOL_IDLE_TIMEOUT, 10) || 30000, // DB_POOL_IDLE_TIMEOUT
  connectionTimeoutMillis: parseInt(process.env.DB_POOL_CONN_TIMEOUT, 10) || 2000, // DB_POOL_CONN_TIMEOUT
  // Optional: Enable SSL in production
  // ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
};

// Create the global connection pool
const pool = new Pool(poolConfig);

// Log connection events (optional, for debugging)
pool.on('connect', () => console.log('New client connected to DB'));
pool.on('error', (err) => console.error('Unexpected database pool error', err));

/**
 * Executes a parameterized SQL query using the shared connection pool.
 * @param {string} text - The SQL query text with placeholders ($1, $2, etc.).
 * @param {Array<any>} params - The values to substitute for the placeholders.
 * @returns {Promise<QueryResult>} The result from the database.
 */
const query = async (text, params) => {
  const start = Date.now();
  try {
    const result = await pool.query(text, params); // This is the SAFE parameterized call
    const duration = Date.now() - start;
    console.log('Executed query', { text, duration, rows: result.rowCount });
    return result;
  } catch (error) {
    console.error('Database query error', { text, params, error });
    throw error; // Re-throw for the route handler to catch
  }
};

/**
 * Acquires a client from the pool for transactions.
 * @returns {Promise<PoolClient>} A single client. YOU MUST CALL .release() on it when done.
 */
const getClient = async () => {
  const client = await pool.connect();
  // Optionally add a method to the client to ensure it's always released.
  const originalRelease = client.release.bind(client);
  client.release = () => {
    client.release = originalRelease;
    return originalRelease();
  };
  return client;
};

module.exports = {
  query,
  getClient,
  // Export the pool for graceful shutdown (in index.js)
  pool
};
```

**2. Express App Configuration (`src/app.js`):**
This sets up the core Express application with essential middleware. It's a prerequisite before defining routes.

```javascript
// src/app.js
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');

const app = express();

// Security: Set various HTTP headers to protect against well-known vulnerabilities
app.use(helmet());

// Enable Cross-Origin Resource Sharing (for frontend communication)
// Configure appropriately for production (specify origin)
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true // If using cookies for auth
}));

// Parse incoming JSON request bodies
app.use(express.json({ limit: '10mb' })); // Limit to prevent large payload attacks

// HTTP request logging
app.use(morgan('combined')); // Use 'dev' for more concise logs during development

// Basic health check endpoint (for Milestone 1)
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Error handling middleware will be added last (after routes are defined)
// This is a placeholder. The actual error middleware will be attached in index.js
// app.use(errorMiddleware);

module.exports = app;
```

#### D. Core Logic Skeleton Code

**1. Main Application Entry Point (`src/index.js`):**
This file ties everything together: creates the app, connects to the database, mounts routes, and starts the server.

```javascript
// src/index.js
const app = require('./app');
const { pool } = require('./config/database');
const errorMiddleware = require('./middleware/error.middleware');

// Import route modules
const authRoutes = require('./routes/auth.routes');
const postRoutes = require('./routes/posts.routes');
// const commentRoutes = require('./routes/comments.routes');

// Mount routes
app.use('/api/auth', authRoutes);
app.use('/api/posts', postRoutes);
// app.use('/api/comments', commentRoutes);

// Mount the centralized error handling middleware LAST.
// It must catch errors from all preceding routes and middleware.
app.use(errorMiddleware);

const PORT = process.env.PORT || 4000;

// Graceful shutdown and startup
const server = app.listen(PORT, async () => {
  console.log(`Server is running on port ${PORT}`);
  // Verify database connection on startup (Milestone 1 Acceptance Criteria)
  try {
    await pool.query('SELECT 1+1 AS result');
    console.log('Database connection established and verified.');
  } catch (dbError) {
    console.error('Failed to verify database connection:', dbError);
    process.exit(1); // Exit if DB is not reachable
  }
});

// Handle graceful shutdown
const shutdown = async (signal) => {
  console.log(`${signal} received. Starting graceful shutdown...`);
  server.close(async () => {
    console.log('HTTP server closed.');
    try {
      await pool.end(); // Close all database connections in the pool
      console.log('Database pool closed.');
      process.exit(0);
    } catch (err) {
      console.error('Error during database pool shutdown:', err);
      process.exit(1);
    }
  });
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
```

**2. Authentication Middleware Skeleton (`src/middleware/auth.middleware.js`):**
This is a core piece learners will implement in Milestone 2. The TODO comments map directly to the authentication algorithm.

```javascript
// src/middleware/auth.middleware.js
const jwt = require('jsonwebtoken');

/**
 * Middleware to authenticate requests using a JWT.
 * If valid, it attaches the decoded user payload to `req.user`.
 * If invalid or missing, it passes an error to the next middleware (the error handler).
 *
 * @param {Object} req - The Express request object.
 * @param {Object} res - The Express response object.
 * @param {Function} next - The Express next middleware function.
 */
const authenticateToken = (req, res, next) => {
  // TODO 1: Extract the token from the request.
  //   - Check the `Authorization` header (format: "Bearer <token>").
  //   - Fallback to checking cookies (if using HTTP-only cookies for auth).
  //   - If no token is found, create an error with status 401 and message "Authentication token required".

  // TODO 2: Verify the token using `jsonwebtoken.verify`.
  //   - Use the `JWT_SECRET` environment variable.
  //   - This function will throw an error if the token is expired or invalid.

  // TODO 3: On successful verification, attach the decoded payload (which should contain userId, email) to `req.user`.

  // TODO 4: Call `next()` to proceed to the next middleware/route handler.

  // TODO 5: Wrap steps 2-4 in a try-catch block.
  //   - In the catch block, handle JWT errors (e.g., 'jwt expired', 'invalid token').
  //   - Create a 403 Forbidden error with a clear message.
  //   - Pass the error to `next(error)`.
};

module.exports = { authenticateToken };
```

#### E. Language-Specific Hints
*   **Environment Variables**: Use the `dotenv` package to load configuration from a `.env` file. Never commit `.env` to version control. Add `.env` to your `.gitignore`.
*   **Async/Await**: Prefer `async/await` over raw Promises or callbacks for better readability. Remember to wrap async route handlers in a `try/catch` or use an express-async-errors wrapper.
*   **Parameterized Queries**: **Always** use the parameterized query feature of the `pg` library (`pool.query('SELECT * FROM users WHERE id = $1', [userId])`). Never concatenate user input into SQL strings.
*   **Error Handling in Express**: Create a centralized error-handling middleware (the last middleware) that catches all errors, logs them, and sends a structured JSON response. Use the `http-errors` package to easily create error objects with status codes.

#### F. Milestone Checkpoint
After setting up the High-Level Architecture (end of Milestone 1), you should be able to verify the following:

1.  **Command to Run**: `node src/index.js`
2.  **Expected Output**:
    ```
    Server is running on port 4000
    Database connection established and verified.
    ```
3.  **Behavior to Verify**:
    *   Open a browser or use `curl` to visit `http://localhost:4000/health`. You should receive a JSON response: `{"status":"OK","timestamp":"..."}`.
    *   The server should start without any unhandled promise rejection warnings.
4.  **Signs Something is Wrong**:
    *   `Error: listen EADDRINUSE: address already in use :::4000`: Change the `PORT` in `.env` or terminate the process using that port.
    *   `error: password authentication failed for user "postgres"`: Check your `DB_USER` and `DB_PASSWORD` in the `.env` file.
    *   `connect ECONNREFUSED 127.0.0.1:5432`: Ensure your PostgreSQL database is running. You can start it with a command like `sudo service postgresql start` (Linux) or `pg_ctl -D /usr/local/var/postgres start` (Mac).


## Data Model

> **Milestone(s):** Milestone 1 (Project Setup & Database Schema), Milestone 3 (Blog CRUD Operations)

The data model is the foundation of the blog platform, defining how information is structured, stored, and retrieved. Think of it as the **filing system of a traditional newspaper archive**. The archive contains folders for each journalist (users), each of their articles (posts), and the letters to the editor responding to each article (comments). The system's effectiveness depends entirely on how well these folders are organized, labeled, and interconnected for quick retrieval and accurate attribution.

### Core Entities and Relationships

Our platform centers around three primary entities: `User`, `Post`, and `Comment`. Their relationships form a classic one-to-many hierarchy: one user can author many posts, and one post can receive many comments. The diagram below illustrates this structure:

![Data Model Entity Relationship Diagram](./diagrams/data-model-erd.svg)

Each entity is implemented as a table in a relational database. The following tables define the exact schema, including every column's purpose, type, and constraints.

**Table: `users`**
The `users` table stores the identity and credentials for every person using the system, whether as an author or a commenter.

| Column Name | Data Type | Description | Constraints |
|-------------|-----------|-------------|-------------|
| `id` | `SERIAL` or `INTEGER AUTO_INCREMENT` | Primary key, a unique identifier for each user. | `PRIMARY KEY`, `NOT NULL` |
| `email` | `VARCHAR(255)` | The user's unique email address, used for login and communication. | `UNIQUE`, `NOT NULL` |
| `password_hash` | `VARCHAR(255)` | The cryptographically hashed representation of the user's password. | `NOT NULL` |
| `name` | `VARCHAR(100)` | The user's display name, shown on their posts and comments. | `NOT NULL` |
| `created_at` | `TIMESTAMP` | The date and time when the user account was created. | `NOT NULL`, `DEFAULT CURRENT_TIMESTAMP` |
| `updated_at` | `TIMESTAMP` | The date and time when the user record was last updated (e.g., name change). | `NOT NULL`, `DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP` |

**Table: `posts`**
The `posts` table contains the core content of the blog. Each record represents a published article, complete with its raw markdown content and metadata.

| Column Name | Data Type | Description | Constraints |
|-------------|-----------|-------------|-------------|
| `id` | `SERIAL` or `INTEGER AUTO_INCREMENT` | Primary key, a unique identifier for each blog post. | `PRIMARY KEY`, `NOT NULL` |
| `title` | `VARCHAR(200)` | The title of the blog post. | `NOT NULL` |
| `slug` | `VARCHAR(200)` | A URL-friendly version of the title (e.g., `my-awesome-post`), used for clean, SEO-friendly URLs. | `UNIQUE`, `NOT NULL` |
| `content_markdown` | `TEXT` | The full body of the post in raw Markdown format. | `NOT NULL` |
| `content_html` | `TEXT` | The HTML generated from `content_markdown`, cached for performance. | |
| `excerpt` | `VARCHAR(500)` | A short summary of the post, auto-generated from content or manually set, used in post listings. | |
| `author_id` | `INTEGER` | Foreign key linking this post to its author in the `users` table. | `NOT NULL`, `FOREIGN KEY REFERENCES users(id) ON DELETE CASCADE` |
| `status` | `ENUM('draft', 'published', 'archived')` | The publication state of the post. Controls visibility. | `NOT NULL`, `DEFAULT 'draft'` |
| `published_at` | `TIMESTAMP` | The date and time the post was moved to `published` status. Used for sorting and display. | |
| `created_at` | `TIMESTAMP` | The date and time the post record was first created. | `NOT NULL`, `DEFAULT CURRENT_TIMESTAMP` |
| `updated_at` | `TIMESTAMP` | The date and time the post was last modified (title, content, status). | `NOT NULL`, `DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP` |

**Table: `comments`**
The `comments` table enables discussion by allowing users to respond to posts. It supports a threaded structure where comments can be replies to other comments.

| Column Name | Data Type | Description | Constraints |
|-------------|-----------|-------------|-------------|
| `id` | `SERIAL` or `INTEGER AUTO_INCREMENT` | Primary key, a unique identifier for each comment. | `PRIMARY KEY`, `NOT NULL` |
| `content` | `TEXT` | The text of the comment. Supports plain text or a limited subset of Markdown. | `NOT NULL` |
| `post_id` | `INTEGER` | Foreign key linking this comment to its parent post. | `NOT NULL`, `FOREIGN KEY REFERENCES posts(id) ON DELETE CASCADE` |
| `user_id` | `INTEGER` | Foreign key linking this comment to its author. If `NULL`, the comment is by an anonymous (non-logged-in) user. | `FOREIGN KEY REFERENCES users(id) ON DELETE SET NULL` |
| `author_name` | `VARCHAR(100)` | Display name for the comment author. Populated from `users.name` if `user_id` exists, otherwise provided by an anonymous poster. | |
| `parent_comment_id` | `INTEGER` | Foreign key to `comments.id`. If not `NULL`, this comment is a direct reply to the specified parent comment, enabling threading. | `FOREIGN KEY REFERENCES comments(id) ON DELETE CASCADE` |
| `created_at` | `TIMESTAMP` | The date and time the comment was submitted. | `NOT NULL`, `DEFAULT CURRENT_TIMESTAMP` |
| `updated_at` | `TIMESTAMP` | The date and time the comment was last edited. | `NOT NULL`, `DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP` |

**Key Relationship Rules:**
*   **User to Post:** A `User` can have zero or many `Posts`. The `author_id` foreign key in `posts` enforces that every post has a valid author. The `ON DELETE CASCADE` rule means if a user account is deleted, all their posts are also deleted.
*   **Post to Comment:** A `Post` can have zero or many `Comments`. The `post_id` foreign key in `comments` ensures every comment is attached to a post. Deleting a post cascades to delete all its comments.
*   **User to Comment:** A `User` can have zero or many `Comments`. The relationship is optional (`user_id` can be `NULL`) to allow for anonymous comments. If a user is deleted, their comments are disassociated (`SET NULL`) but remain on the post with their `author_name` preserved.
*   **Comment to Comment (Threading):** A `Comment` can have zero or many child `Comments` (replies). The `parent_comment_id` self-referencing foreign key creates a tree structure. `ON DELETE CASCADE` ensures deleting a parent comment removes all its replies.

### Schema Design Decisions (ADRs)

Several critical design choices shape the data model's security, performance, and maintainability. Each decision is recorded as an **Architecture Decision Record (ADR)**.

> **Decision: Indexing Strategy for Query Performance**
> - **Context**: The blog will have frequent queries to list posts (sorted by date), find posts by author, load comments for a post, and enforce unique constraints. Without proper indexes, these operations will perform full table scans, slowing down the application as data grows.
> - **Options Considered**:
>     1.  **No Additional Indexes**: Rely only on primary key indexes.
>     2.  **Targeted Indexes on Foreign Keys and Sort Columns**: Create indexes on columns used in `WHERE`, `JOIN`, and `ORDER BY` clauses.
>     3.  **Comprehensive Indexing on All Queryable Columns**: Index every column that might appear in a query.
> - **Decision**: **Targeted Indexes on Foreign Keys and Sort Columns** (Option 2).
> - **Rationale**: Primary keys are automatically indexed. Foreign key columns (`author_id`, `post_id`, `user_id`, `parent_comment_id`) are used heavily in `JOIN` operations and should be indexed to speed up relationship traversal. The `posts.published_at` column is used for sorting the main post listing; an index there allows the database to retrieve posts in order without a costly sort operation. The `posts.slug` column has a uniqueness constraint, which is implemented via an index. This balanced approach optimizes the most critical paths without the storage and write-performance overhead of over-indexing.
> - **Consequences**: Write operations (INSERT, UPDATE, DELETE) will be slightly slower due to index maintenance. Read operations for common queries will be significantly faster. Storage requirements will increase moderately. We must be disciplined about adding new indexes only when new query patterns are proven in production.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| No Additional Indexes | Minimal storage, fastest writes. | Unacceptable read performance degradation as tables grow. | No |
| Targeted Indexes | Optimal read performance for common queries. Moderate write overhead. Requires analysis. | Requires upfront design and occasional reevaluation as features are added. | **Yes** |
| Comprehensive Indexing | Excellent read performance for all ad-hoc queries. | Significant storage overhead. Severe write performance penalty. Unnecessary for most columns. | No |

> **Decision: Password Storage with Adaptive Hashing**
> - **Context**: User passwords are highly sensitive and must be stored such that they cannot be recovered even if the database is compromised. Simple hashing algorithms (MD5, SHA-1) are insufficient as they are too fast, enabling brute-force attacks.
> - **Options Considered**:
>     1.  **bcrypt**: A widely-adopted, battle-tested adaptive hashing function with a configurable work factor to increase computational cost.
>     2.  **scrypt**: Designed to be memory-hard, offering strong resistance against ASIC-based attacks.
>     3.  **Argon2**: The winner of the Password Hashing Competition (2015), considered state-of-the-art with configurable memory, time, and parallelism costs.
> - **Decision**: **bcrypt with a work factor of 12**.
> - **Rationale**: For an intermediate learning project, bcrypt provides an excellent balance of strong security, immense popularity, and simplicity. It is available in every major programming language, has extensive documentation, and its adaptive nature allows increasing the work factor over time as hardware improves. A work factor of 12 is a secure default that balances login speed and resistance to brute-force attacks. While Argon2 is technically superior, bcrypt's maturity and ubiquity reduce implementation risk for learners.
> - **Consequences**: Password verification is intentionally slow (~250-500ms per hash), effectively throttling login attempts and making brute-force attacks infeasible. The work factor can be increased in the future if needed, but existing hashes remain valid and will be upgraded on next login.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| bcrypt | Extremely well-known, simple API, adaptive via work factor. | Less memory-hard than modern alternatives. | **Yes** |
| scrypt | Memory-hard, good resistance to specialized hardware. | Slightly more complex to configure correctly, less common than bcrypt. | No |
| Argon2 | State-of-the-art, highly configurable, memory-hard. | Relatively new, configuration can be complex, less ubiquitous in tutorials. | No |

> **Decision: Soft Deletion for Posts and Comments**
> - **Context**: When a user deletes a post or comment, we must decide whether to permanently remove the data from the database or merely mark it as deleted. Permanent deletion is irreversible and can break referential integrity (e.g., replies to a deleted comment).
> - **Options Considered**:
>     1.  **Hard Delete**: Permanently remove the row from the database using `DELETE`.
>     2.  **Soft Delete**: Add a `deleted_at` `TIMESTAMP` column. "Deletion" sets this column. All queries filter out rows where `deleted_at IS NOT NULL`.
>     3.  **Archival Table**: On delete, move the row to a separate `deleted_posts` or `deleted_comments` table.
> - **Decision**: **Soft Delete with a `deleted_at` column** (Option 2) for posts and comments.
> - **Rationale**: Soft deletion provides a safety net for accidental deletion, a common user request. It simplifies audit trails and data recovery. For a blog platform, the volume of deletions is low, so the performance impact of filtering in queries is negligible. It is simpler to implement than the archival table approach while providing similar benefits. User accounts will still use hard delete (`ON DELETE CASCADE`) for GDPR/privacy compliance, as user data is more sensitive.
> - **Consequences**: All application queries must include a `WHERE deleted_at IS NULL` clause. This adds complexity to the data access layer. Storage consumption will slowly increase with deleted content, necessitating a periodic cleanup job (e.g., permanently delete records older than 1 year) which can be added as a future extension.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Hard Delete | Simple, reclaims storage immediately, no query overhead. | Irreversible, breaks referential integrity for threaded comments. | No |
| Soft Delete | Recoverable, maintains referential integrity, simple audit trail. | Requires filtering in all queries, storage not reclaimed, potential for accidental exposure if filter is forgotten. | **Yes** |
| Archival Table | Clear separation of active/deleted data, no query filtering needed. | More complex deletion logic, requires managing two tables, foreign keys may need adjustment. | No |

### Migration Strategy

The database schema is not static; it will evolve as new features are added (e.g., post tags, user profiles). A **migration strategy** manages these changes in a controlled, reproducible, and reversible manner. Think of it as **version control for your database structure**. Each migration is a pair of scripts: one to apply the change (`up`) and one to revert it (`down`).

**Migration Lifecycle:**
1.  **Creation**: A developer writes a new migration file when a schema change is needed (e.g., `001_create_users_table.sql`, `002_add_slug_to_posts.sql`). Each file contains the `up` and `down` SQL commands.
2.  **Application**: During deployment, a migration tool runs all new `up` migrations in order, tracking which have been applied in a special `migrations` or `schema_version` table.
3.  **Rollback**: If a deployment fails, the tool can run the `down` migration for the most recent change, reverting the schema to its previous state.
4.  **Consistency**: Every environment (development, testing, production) runs the same migration sequence, guaranteeing identical schemas.

**Key Principles:**
*   **Idempotency**: Running the same migration twice should not cause errors or duplicate changes. The `CREATE TABLE IF NOT EXISTS` and `DROP TABLE IF EXISTS` patterns help achieve this.
*   **Reversibility**: Every `up` migration must have a corresponding `down` migration that perfectly undoes its effect. This is non-negotiable for safe rollbacks.
*   **Atomicity**: Each migration should be wrapped in a transaction (where supported by the database and operation) so that if any part fails, the entire migration is rolled back, leaving the schema in a known state.
*   **Small Steps**: Migrations should be small and focused on a single change. This makes them easier to review, test, and roll back if necessary.

**Example Migration Pair:**
```
-- File: migrations/002_add_slug_to_posts.sql
-- UP: Add the new column and create a unique index.
BEGIN;
ALTER TABLE posts ADD COLUMN slug VARCHAR(200);
UPDATE posts SET slug = lower(regexp_replace(title, '[^a-zA-Z0-9]+', '-', 'g'));
CREATE UNIQUE INDEX idx_posts_slug ON posts(slug);
COMMIT;

-- DOWN: Remove the index and the column.
BEGIN;
DROP INDEX IF EXISTS idx_posts_slug;
ALTER TABLE posts DROP COLUMN slug;
COMMIT;
```

**Common Pitfalls in Data Modeling & Migration:**
⚠️ **Pitfall: Not Indexing Foreign Keys**
*   **Description**: Creating foreign key constraints without also creating an index on the foreign key column.
*   **Why it's wrong**: Foreign key constraints ensure data integrity but do not automatically create indexes for performance. Queries that join tables or filter by the foreign key (e.g., `SELECT * FROM posts WHERE author_id = ?`) will perform full table scans, becoming extremely slow as the referenced table grows.
*   **How to fix**: **Always create an index on every foreign key column**. Most database analysis tools will warn you about missing indexes on foreign keys.

⚠️ **Pitfall: Storing Passwords in Plain Text or with Weak Hashing**
*   **Description**: Saving user passwords directly in the database or using a fast, cryptographically broken hash function like MD5 or SHA-1.
*   **Why it's wrong**: If the database is breached, attackers immediately have all user credentials. Fast hashes allow attackers to compute billions of hashes per second, easily cracking most passwords via rainbow tables or brute force.
*   **How to fix**: **Always use a dedicated, adaptive password hashing function like `bcrypt`, `scrypt`, or `Argon2`**. Never roll your own hashing scheme. Use a high work factor (e.g., 12 for bcrypt).

⚠️ **Pitfall: Forgetting the `ON UPDATE` clause on `updated_at`**
*   **Description**: Defining an `updated_at` column with only `DEFAULT CURRENT_TIMESTAMP`, which sets the timestamp on insert but never updates it.
*   **Why it's wrong**: The `updated_at` field becomes meaningless, as it doesn't reflect the last time the row was actually modified, breaking features that rely on this data (e.g., detecting recent activity).
*   **How to fix**: Use `DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP` (MySQL/MariaDB) or a trigger/application logic to update the field on every modification.

⚠️ **Pitfall: Writing Non-Reversible Migrations**
*   **Description**: Creating an `up` migration that, for example, drops a column, without providing a `down` migration that can restore it (e.g., because the data is lost).
*   **Why it's wrong**: Makes rollbacks impossible, turning any migration failure into a potential production outage. You cannot safely revert to a previous working state.
*   **How to fix**: **Always write and test the `down` migration before applying the `up` migration in production.** For data loss operations (like dropping a column), consider a multi-step migration: 1) copy data to a new location, 2) deploy application code that no longer uses the old column, 3) run a migration to drop the column.

### Implementation Guidance

**A. Technology Recommendations Table:**
| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Database | PostgreSQL with `node-postgres` (`pg`) library | PostgreSQL with an **ORM** like Prisma or TypeORM |
| Migration Tool | Custom script using a `migrations` table | Dedicated tool like `node-pg-migrate`, `db-migrate`, or Prisma Migrate |
| Connection Management | Basic `Pool` from `pg` with environment variables | `Pool` with dynamic configuration, health checks, and connection string parsing |

**B. Recommended File/Module Structure:**
```
blog-platform/
├── package.json
├── src/
│   ├── server.js                      # Application entry point
│   ├── config/
│   │   └── database.js                # Database configuration and pool setup
│   ├── db/
│   │   ├── migrations/                # Sequential migration SQL files
│   │   │   ├── 001_initial_schema.up.sql
│   │   │   ├── 001_initial_schema.down.sql
│   │   │   └── ...                    # More migration pairs
│   │   ├── seed.js                    # Script to populate dev database with test data
│   │   └── migrate.js                 # Custom migration runner script
│   ├── models/                        # Data access layer (optional, if not using ORM)
│   │   ├── User.js
│   │   ├── Post.js
│   │   └── Comment.js
│   └── ... (other directories for routes, middleware, etc.)
└── .env                               # Environment variables (DB connection string)
```

**C. Infrastructure Starter Code:**

**File: `src/config/database.js`**
```javascript
const { Pool } = require('pg');
require('dotenv').config();

/**
 * @typedef {Object} PoolConfig
 * @property {string} host
 * @property {number} port
 * @property {string} database
 * @property {string} user
 * @property {string} password
 * @property {number} max
 * @property {number} idleTimeoutMillis
 * @property {number} connectionTimeoutMillis
 */

// Configuration for the database connection pool
const poolConfig = {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT) || 5432,
    database: process.env.DB_NAME || 'blog_platform',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || '',
    max: parseInt(process.env.DB_POOL_MAX) || 20, // DB_POOL_MAX
    idleTimeoutMillis: parseInt(process.env.DB_POOL_IDLE_TIMEOUT) || 30000, // DB_POOL_IDLE_TIMEOUT
    connectionTimeoutMillis: parseInt(process.env.DB_POOL_CONN_TIMEOUT) || 2000, // DB_POOL_CONN_TIMEOUT
};

// Create a global connection pool
const pool = new Pool(poolConfig);

// Helper function to execute a parameterized query
/**
 * @param {string} text - SQL query text
 * @param {Array<any>} params - Query parameters
 * @returns {Promise<QueryResult>} - Result object containing rows and metadata
 */
async function query(text, params) {
    const start = Date.now();
    try {
        const res = await pool.query(text, params);
        const duration = Date.now() - start;
        console.debug('Executed query', { text, duration, rows: res.rowCount });
        return res;
    } catch (error) {
        console.error('Error executing query', { text, params, error });
        throw error;
    }
}

// Helper function to get a client for transactions
/**
 * @returns {Promise<PoolClient>} - A single client from the pool
 */
async function getClient() {
    const client = await pool.connect();
    const { query: q, release } = client;
    // Set a timeout on the client to prevent hanging queries
    const timeout = setTimeout(() => {
        console.error('A client has been checked out for more than 10 seconds!');
    }, 10000);
    // Override the release method to clear the timeout and release the client properly
    client.release = () => {
        clearTimeout(timeout);
        release.apply(client);
    };
    return client;
}

module.exports = { pool, query, getClient };
```

**File: `src/db/migrate.js`**
```javascript
const { query } = require('../config/database');
const fs = require('fs').promises;
const path = require('path');

async function runMigrations() {
    // Create migrations table if it doesn't exist
    await query(`
        CREATE TABLE IF NOT EXISTS migrations (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    `);

    // Read all migration files
    const migrationsDir = path.join(__dirname, 'migrations');
    const files = await fs.readdir(migrationsDir);
    const migrationFiles = files.filter(f => f.endsWith('.up.sql')).sort();

    // Get already applied migrations
    const { rows: appliedMigrations } = await query('SELECT name FROM migrations ORDER BY applied_at');
    const appliedNames = new Set(appliedMigrations.map(row => row.name));

    for (const upFile of migrationFiles) {
        const migrationName = upFile.replace('.up.sql', '');
        if (appliedNames.has(migrationName)) {
            console.log(`Skipping already applied migration: ${migrationName}`);
            continue;
        }

        const downFile = migrationName + '.down.sql';
        const upPath = path.join(migrationsDir, upFile);
        const downPath = path.join(migrationsDir, downFile);

        const client = await getClient(); // Use getClient for transaction
        try {
            await client.query('BEGIN');

            // Read and run the UP migration
            const upSql = await fs.readFile(upPath, 'utf8');
            await client.query(upSql);

            // Record the migration as applied
            await client.query('INSERT INTO migrations (name) VALUES ($1)', [migrationName]);

            await client.query('COMMIT');
            console.log(`Applied migration: ${migrationName}`);
        } catch (error) {
            await client.query('ROLLBACK');
            console.error(`Failed to apply migration ${migrationName}:`, error);
            throw error;
        } finally {
            client.release();
        }
    }
    console.log('All migrations applied successfully.');
}

// For command-line usage: node src/db/migrate.js
if (require.main === module) {
    runMigrations().catch(err => {
        console.error('Migration failed:', err);
        process.exit(1);
    });
}

module.exports = { runMigrations };
```

**D. Core Logic Skeleton Code:**

**File: `src/models/Post.js` (Data Access Layer without ORM)**
```javascript
const { query } = require('../config/database');

/**
 * Data model representing a Blog Post.
 */
class Post {
    /**
     * Creates a new post in the database.
     * @param {Object} postData - The post data.
     * @param {string} postData.title - Post title.
     * @param {string} postData.slug - URL-friendly slug.
     * @param {string} postData.content_markdown - Raw markdown content.
     * @param {number} authorId - ID of the author (user).
     * @returns {Promise<QueryResult>} - The result of the insert query.
     */
    static async create({ title, slug, content_markdown }, authorId) {
        // TODO 1: Generate an `excerpt` from the first 150 characters of `content_markdown`
        // TODO 2: Convert `content_markdown` to HTML (e.g., using `marked` library) and store as `content_html`
        // TODO 3: Construct a parameterized INSERT query for the `posts` table with all columns
        // TODO 4: Execute the query using the `query` function, passing title, slug, content_markdown, content_html, excerpt, authorId
        // TODO 5: Return the result, which contains the newly inserted row (including the generated ID)
    }

    /**
     * Finds a published post by its slug.
     * @param {string} slug - The post's slug.
     * @returns {Promise<Object|null>} - The post object with author name joined, or null if not found.
     */
    static async findBySlug(slug) {
        // TODO 1: Construct a SELECT query that joins `posts` with `users` on `author_id`
        // TODO 2: Include only necessary columns: post.* and user.name as author_name
        // TODO 3: Filter by `slug = $1` and `status = 'published'` and `deleted_at IS NULL`
        // TODO 4: Execute the query using the `query` function
        // TODO 5: Return the first row (post) or null if no results
    }

    /**
     * Retrieves a paginated list of published posts.
     * @param {number} page - The page number (1-indexed).
     * @param {number} limit - Number of posts per page.
     * @returns {Promise<Object>} - Object containing `posts` array and `totalCount`.
     */
    static async listPublished(page = 1, limit = 10) {
        const offset = (page - 1) * limit;
        // TODO 1: Query to get total count of published, non-deleted posts (for pagination UI)
        // TODO 2: Query to get the page of posts (join with users for author name), ordered by `published_at DESC`
        // TODO 3: Apply LIMIT and OFFSET for pagination
        // TODO 4: Return an object: { posts: [...], totalCount: number, totalPages: number }
    }

    /**
     * Updates a post. Ensures the requesting user is the author.
     * @param {number} postId - ID of the post to update.
     * @param {number} userId - ID of the user attempting the update.
     * @param {Object} updates - Fields to update (title, content_markdown, etc.).
     * @returns {Promise<boolean>} - True if update was successful, false if not authorized or not found.
     */
    static async update(postId, userId, updates) {
        // TODO 1: First, verify ownership: SELECT author_id FROM posts WHERE id = $1 AND deleted_at IS NULL
        // TODO 2: If no post found or author_id != userId, return false (not authorized)
        // TODO 3: If content_markdown is in updates, regenerate content_html and excerpt
        // TODO 4: Construct a parameterized UPDATE query for the `posts` table, setting updated fields
        // TODO 5: Include a `WHERE id = $...` clause to ensure we only update the target post
        // TODO 6: Execute the update query
        // TODO 7: Return true if the update affected a row, false otherwise
    }

    /**
     * Soft-deletes a post (sets deleted_at).
     * @param {number} postId - ID of the post to delete.
     * @param {number} userId - ID of the user attempting the deletion.
     * @returns {Promise<boolean>} - True if soft-delete was successful.
     */
    static async softDelete(postId, userId) {
        // TODO 1: Verify ownership (similar to update method)
        // TODO 2: If authorized, execute an UPDATE query: SET deleted_at = CURRENT_TIMESTAMP WHERE id = $1
        // TODO 3: Return true if a row was affected
    }
}

module.exports = Post;
```

**E. Language-Specific Hints (JavaScript/Node.js):**
*   Use the `pg` library's built-in support for parameterized queries (`client.query('SELECT * FROM users WHERE id = $1', [id])`) to prevent **SQL injection**. Never concatenate user input directly into SQL strings.
*   For `bcrypt` password hashing, use the `bcryptjs` package (pure JavaScript) to avoid native compilation issues. Hash passwords with `bcrypt.hash(password, 12)`.
*   To convert Markdown to HTML, consider the `marked` library. Always sanitize the resulting HTML with a library like `dompurify` or `sanitize-html` to prevent **XSS**.
*   Use environment variables (via `dotenv` package) for database credentials and other configuration. Never hardcode sensitive data.

**F. Milestone Checkpoint (Milestone 1):**
After implementing the database schema and migrations:
1.  **Run migrations**: `node src/db/migrate.js`. Expected output: "All migrations applied successfully." Check your database client to see the `users`, `posts`, `comments`, and `migrations` tables.
2.  **Test connection health**: Create a simple script that uses the `query` function to run `SELECT 1 as health_check;`. It should return a result without errors.
3.  **Verify foreign keys**: Try to insert a post with a non-existent `author_id`. The database should reject it with a foreign key violation error.
4.  **Signs of trouble**: If migrations fail, check: a) Database is running, b) Connection string in `.env` is correct, c) SQL syntax in migration files is valid for your database (PostgreSQL vs MySQL). Use `psql` or `pgAdmin` to manually inspect the database state.


## Component: User Authentication

> **Milestone(s):** Milestone 2 (User Authentication)

This component is the gatekeeper of the blog platform, responsible for verifying user identities, managing authenticated sessions, and securely handling credentials. It's arguably the most security-critical component, as flaws here can compromise the entire system. The design must balance security, usability, and implementation complexity for an educational context.

![User Authentication State Machine](./diagrams/auth-state-machine.svg)

### Mental Model: The Club Bouncer & Membership Desk

Imagine the blog platform as an exclusive nightclub. The **authentication system** serves two roles: the **membership desk** where new patrons sign up and get their credentials (email/password), and the **bouncer** at the entrance who checks credentials and grants access.

**The Membership Desk (Registration & Credential Management):**
- **New Membership (Registration):** A new visitor provides an email (membership ID) and chooses a secret handshake (password). The desk clerk verifies the email isn't already registered, then securely records the handshake pattern in a vault (hashed password) linked to the member's profile.
- **Lost Handshake (Password Reset):** If a member forgets their handshake, the desk issues a one-time, time-limited pass (reset token) sent to their registered email. Presenting this pass allows them to set a new handshake.
- **Member Database:** The desk maintains a secure ledger (`users` table) with each member's ID, email, hashed handshake, and membership date.

**The Bouncer (Login & Session Enforcement):**
- **Entry Check (Login):** A patron presents their email and attempts the secret handshake. The bouncer checks the vault to see if the attempted handshake matches the stored pattern. If correct, the bouncer issues a stamped wristband (JWT or session token) that grants access to the club's interior.
- **Wristband Verification (Authentication Middleware):** Inside the club, attendants at each area (protected routes) check the wristband's stamp for validity and expiration. No wristband or a forged one results in ejection.
- **Revoking Access (Logout):** When a patron leaves, their wristband is torn (token invalidated). A stolen wristband remains valid until its natural expiration, which is why wristbands have short lifespans, and patrons can get a new one (refresh token) without re-verifying at the door.

This mental model clarifies the separation between **establishing identity** (desk) and **continuously proving identity** (bouncer). It also illustrates the stateless nature of wristbands (tokens) versus the stateful alternative of maintaining a guest list at the door (server sessions).

### Authentication Interface

The authentication component exposes a RESTful API for identity management. All endpoints exchange JSON unless otherwise noted. The interface is designed to be stateless on the server side (using JWTs) and secure by default (using HTTP-only cookies).

| Endpoint Method & Path | Request Body (JSON) | Success Response (JSON) | Failure Responses | Description |
| :--- | :--- | :--- | :--- | :--- |
| **POST** `/api/auth/register` | `{ "email": "string", "password": "string", "name": "string (optional)" }` | `201 Created` <br> `{ "id": "number", "email": "string", "name": "string", "createdAt": "ISO8601" }` | `400` Invalid input <br> `409` Email already exists | Creates a new user account after validating email format and password strength. The password is hashed before storage. Returns the created user (excluding password hash). |
| **POST** `/api/auth/login` | `{ "email": "string", "password": "string" }` | `200 OK` <br> Sets `HttpOnly` cookie: `token=<JWT>` <br> Body: `{ "user": { "id", "email", "name" } }` | `401` Invalid credentials <br> `400` Missing fields | Verifies credentials. On success, issues a JWT as an HTTP-only cookie and returns minimal user data. |
| **POST** `/api/auth/logout` | *None* | `200 OK` <br> Clears cookie: `token=` | `401` If not authenticated | Invalidates the current user's session by clearing the authentication cookie. |
| **POST** `/api/auth/refresh` | *None* | `200 OK` <br> Sets new `HttpOnly` cookie: `token=<new-JWT>` <br> Body: `{ "user": { "id", "email", "name" } }` | `401` Invalid/missing refresh token | Issues a new access token using a valid refresh token, extending the session without re-authentication. |
| **POST** `/api/auth/forgot-password` | `{ "email": "string" }` | `202 Accepted` <br> `{ "message": "If the email exists, a reset link has been sent." }` | `400` Invalid email | Generates a time-limited password reset token, stores its hash, and sends a reset link to the provided email (if registered). This endpoint **must not** reveal whether the email exists. |
| **POST** `/api/auth/reset-password` | `{ "token": "string", "newPassword": "string" }` | `200 OK` <br> `{ "message": "Password reset successful." }` | `400` Invalid token or weak password <br> `410` Token expired | Validates the reset token, updates the user's password hash, and invalidates the token. |

**Protected Route Middleware:** All endpoints under `/api/posts` and `/api/users/me` are protected. The middleware extracts the JWT from the `Cookie` header, verifies its signature and expiration, and attaches the decoded user ID to the request object (`req.user = { id, email }`). If verification fails, it returns `401 Unauthorized`.

### ADR: Session Tokens vs. JWT

> **Decision: Use JSON Web Tokens (JWTs) with Refresh Tokens for Stateless Authentication**

**Context:**
The blog platform requires a way to maintain user authentication across multiple HTTP requests. The server must be able to verify a user's identity for protected operations (creating posts, commenting) without querying the database on every request. The system is expected to have a single server instance initially but should be designed to allow horizontal scaling.

**Options Considered:**
1. **Traditional Server-Side Sessions:** Store a random session ID in a cookie. The server maintains a session store (in-memory, Redis, or database) mapping session IDs to user data. Each request validates the session ID against the store.
2. **JSON Web Tokens (JWTs):** Encode user claims (ID, email) into a cryptographically signed token stored client-side (in an HTTP-only cookie). The server validates the token's signature without consulting a central store.

**Decision:**
We will use JWTs as our primary authentication token, delivered via secure, HTTP-only cookies. We will supplement them with a long-lived refresh token stored in a database (or a separate, long-lived JWT) to allow silent session renewal without forcing the user to re-enter credentials frequently.

**Rationale:**
- **Stateless Scalability:** JWTs allow any server instance to validate a request without sharing a session store. This simplifies deployment and scaling in a learning environment where introducing Redis or a shared database session table adds complexity.
- **Performance:** Eliminates a database lookup on every authenticated request (only signature verification). This is a tangible performance benefit, even for a small-scale blog.
- **Decoupling:** The token is self-contained, making it easier to implement microservices later (though that's a non-goal). The signature ensures integrity.
- **Educational Value:** Implementing JWTs teaches important concepts like cryptographic signing, token expiration, and stateless API design.

**Consequences:**
- **Immediate Revocation Challenge:** A compromised JWT cannot be individually revoked before its natural expiration. We mitigate this by keeping token lifetimes short (e.g., 15 minutes) and using refresh tokens that can be invalidated server-side upon logout or suspicious activity.
- **Token Size:** JWTs are larger than a simple session ID, increasing bandwidth slightly (negligible for blog traffic).
- **Implementation Complexity:** Properly signing, verifying, and securely storing tokens requires careful implementation to avoid common security pitfalls.

**Option Comparison Table:**

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **Server-Side Sessions** | - Immediate revocation possible<br>- No size limit on session data<br>- Mature, well-understood pattern | - Requires shared session store for scaling<br>- Adds database load (lookup per request)<br>- More complex to scale horizontally | ❌ |
| **JWTs (Stateless)** | - Stateless, easy to scale<br>- Reduced database load<br>- Self-contained claims | - No built-in revocation<br>- Larger payload size<br>- More complex crypto implementation | ✅ |

### ADR: Password Hashing Algorithm

> **Decision: Use bcrypt with a Work Factor of 12**

**Context:**
User passwords are highly sensitive and must be stored such that a database breach does not immediately compromise them. We need a one-way hashing function specifically designed to be computationally expensive (slow) to resist brute-force and rainbow table attacks.

**Options Considered:**
1. **bcrypt:** An adaptive hash function based on the Blowfish cipher. It incorporates a salt and a configurable cost (work) factor to slow down hashing.
2. **scrypt:** A memory-hard key derivation function designed to be costly in both CPU and memory, making hardware (ASIC/GPU) attacks more difficult.
3. **Argon2:** The winner of the 2015 Password Hashing Competition, designed to be resistant to GPU and side-channel attacks, with configurable time, memory, and parallelism costs.

**Decision:**
We will use `bcrypt` with a work factor (cost) of 12. The password hash will be stored in the `users.password_hash` column.

**Rationale:**
- **Proven Security:** bcrypt has withstood over 20 years of cryptanalysis and is widely trusted in the industry.
- **Adaptive:** The work factor can be increased over time as hardware improves, maintaining security without changing the algorithm.
- **Built-in Salt:** bcrypt automatically generates and manages a unique salt for each password, preventing rainbow table attacks.
- **Balanced Resistance:** It provides strong resistance against GPU/ASIC attacks due to its memory-access pattern, though not as memory-hard as scrypt or Argon2.
- **Ecosystem & Simplicity:** It has excellent library support across all our target languages (Node.js, Python, etc.) and is straightforward to implement correctly. For an educational project, this simplicity reduces the risk of misconfiguration.

**Consequences:**
- **CPU Intensive:** Hashing and verification are deliberately slow (~250ms per hash at cost 12). This protects passwords but adds overhead to login and registration. This is an acceptable trade-off for security.
- **Not Memory-Hard:** While resistant, bcrypt is less memory-hard than scrypt or Argon2, making it slightly more vulnerable to specialized hardware attacks. For a blog platform, this is a negligible risk compared to the benefits of simplicity and reliability.
- **Work Factor Tuning:** The chosen factor of 12 is a balance between security and user experience on modern hardware. It should be benchmarked in the deployment environment.

**Option Comparison Table:**

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **bcrypt** | - Time-tested, widely adopted<br>- Simple API with built-in salt<br>- Good resistance to GPU attacks | - Less memory-hard than newer algorithms<br>- Slower verification (by design) | ✅ |
| **scrypt** | - Memory-hard, excellent GPU/ASIC resistance<br>- Configurable memory cost | - More complex to configure optimally<br>- Slightly less library support | ❌ |
| **Argon2** | - State-of-the-art, most resistant<br>- Highly configurable | - Newer, less battle-tested than bcrypt<br>- Complex configuration increases risk of error | ❌ |

### Common Pitfalls in Authentication

Implementing authentication is fraught with subtle errors that can create critical security vulnerabilities. Below are the most common pitfalls learners encounter, along with their implications and corrective measures.

⚠️ **Pitfall 1: Storing JWTs in `localStorage` or `sessionStorage`**
- **Description:** Placing the JWT in browser web storage for easy access from JavaScript.
- **Why It's Wrong:** JavaScript running in the same origin can access web storage, making tokens vulnerable to Cross-Site Scripting (XSS) attacks. A malicious script can steal the token and impersonate the user.
- **How to Fix:** Always store authentication tokens in **HTTP-only cookies**. This prevents JavaScript access, significantly raising the bar for XSS attacks. Set the `Secure` flag (HTTPS only) and `SameSite=Strict` or `Lax` for additional CSRF protection.

⚠️ **Pitfall 2: Timing Attacks on User Existence**
- **Description:** Returning different error messages (e.g., "Email not found" vs. "Incorrect password") or taking measurably different times to respond during login.
- **Why It's Wrong:** An attacker can enumerate valid user emails by observing error messages or response timing, violating user privacy and aiding targeted attacks.
- **How to Fix:** Use a constant-time comparison function for password hashes (most libraries provide this). Always return a generic error message: "Invalid email or password." Additionally, hash a dummy value if the user doesn't exist to normalize response time.

⚠️ **Pitfall 3: Weak or No Password Policy**
- **Description:** Accepting passwords of any length or complexity (e.g., "123456").
- **Why It's Wrong:** Users tend to choose weak passwords, making accounts vulnerable to brute-force attacks and credential stuffing from other breaches.
- **How to Fix:** Enforce a minimum password length (8+ characters) and recommend (or require) a mix of character types. Use a library like `zxcvbn` to estimate password strength and reject overly weak passwords. **Never** impose arbitrary complexity rules (e.g., "must include a symbol") as they often lead to predictable patterns.

⚠️ **Pitfall 4: Not Invalidating Refresh Tokens on Logout**
- **Description:** When a user logs out, only the short-lived access token (JWT) is discarded, while the long-lived refresh token remains valid.
- **Why It's Wrong:** A stolen refresh token can be used to obtain new access tokens indefinitely, effectively making logout useless for that device.
- **How to Fix:** Maintain a server-side list (in a database table `refresh_tokens`) of issued refresh tokens per user. On logout, delete the specific refresh token associated with that session. Alternatively, implement token blacklisting or use a stateful refresh token that must be checked against a store.

⚠️ **Pitfall 5: Missing CSRF Protection for State-Changing Operations**
- **Description:** Relying solely on HTTP-only cookies for authentication without additional protection against Cross-Site Request Forgery.
- **Why It's Wrong:** If a user is logged in, a malicious site can trick their browser into making authenticated requests (e.g., change email, delete post) without their consent, because cookies are sent automatically.
- **How to Fix:** For operations that change state (POST, PUT, DELETE), require a CSRF token in the request header. This token should be generated server-side, stored in a separate cookie (not HTTP-only), and validated against the value sent in a header (e.g., `X-CSRF-Token`). Modern practice also leverages the `SameSite=Strict` cookie attribute, which is effective for most scenarios.

### Implementation Guidance

This section provides concrete code and structure to implement the authentication component in JavaScript (Node.js/Express), the primary language.

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
| :--- | :--- | :--- |
| **Web Framework** | Express.js | Fastify |
| **Password Hashing** | bcrypt.js | Argon2 (via `argon2` package) |
| **JWT Library** | jsonwebtoken | jose (more robust JWT handling) |
| **Database ORM/Driver** | pg (node-postgres) with raw SQL | Prisma ORM |
| **Validation** | Joi or express-validator | Zod with TypeScript integration |
| **Email Service** | Nodemailer with a test SMTP (Ethereal) | SendGrid or Amazon SES API |

**B. Recommended File/Module Structure:**

Organize authentication logic into a dedicated module within the project structure.

```
blog-platform/
├── server.js                      # Application entry point
├── package.json
├── config/
│   └── index.js                   # Centralized configuration (DB, JWT secret)
├── src/
│   ├── db/
│   │   ├── pool.js                # Database connection pool setup
│   │   └── migrations/            # SQL migration files
│   ├── middleware/
│   │   ├── auth.js                # Authentication middleware
│   │   └── errorHandler.js
│   ├── routes/
│   │   └── auth.js                # All authentication endpoints
│   ├── utils/
│   │   ├── password.js            # bcrypt wrapper functions
│   │   ├── tokens.js              # JWT generation/verification
│   │   └── validation.js          # Input validation schemas
│   └── models/
│       └── user.js                # User data access functions (not classes)
└── .env                           # Environment variables
```

**C. Infrastructure Starter Code:**

**1. Database Connection Pool (`src/db/pool.js`):**
This is a prerequisite setup. It configures and exports a reusable PostgreSQL connection pool.

```javascript
const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  // Naming conventions from the project
  max: process.env.DB_POOL_MAX || 20,
  idleTimeoutMillis: process.env.DB_POOL_IDLE_TIMEOUT || 30000,
  connectionTimeoutMillis: process.env.DB_POOL_CONN_TIMEOUT || 2000,
});

// Helper to execute a parameterized query
const query = (text, params) => pool.query(text, params);

// Helper to get a client for transactions
const getClient = () => pool.connect();

module.exports = {
  query,
  getClient,
  pool, // exported for potential health checks
};
```

**2. Password Utility (`src/utils/password.js`):**
A complete wrapper around `bcrypt` to ensure consistent usage.

```javascript
const bcrypt = require('bcryptjs');

const SALT_ROUNDS = 12; // Work factor of 12

/**
 * Hashes a plain text password.
 * @param {string} plainPassword - The user's plain text password.
 * @returns {Promise<string>} Resolves with the bcrypt hash.
 */
async function hashPassword(plainPassword) {
  if (!plainPassword || plainPassword.length < 1) {
    throw new Error('Password cannot be empty');
  }
  return await bcrypt.hash(plainPassword, SALT_ROUNDS);
}

/**
 * Compares a plain text password against a stored hash.
 * Uses constant-time comparison to prevent timing attacks.
 * @param {string} plainPassword - The attempted password.
 * @param {string} hash - The stored bcrypt hash.
 * @returns {Promise<boolean>} True if password matches.
 */
async function verifyPassword(plainPassword, hash) {
  if (!plainPassword || !hash) return false;
  return await bcrypt.compare(plainPassword, hash);
}

module.exports = {
  hashPassword,
  verifyPassword,
  SALT_ROUNDS,
};
```

**3. JWT Utility (`src/utils/tokens.js`):**
Handles signing and verifying JWTs. Uses environment variables for secrets.

```javascript
const jwt = require('jsonwebtoken');

const ACCESS_TOKEN_SECRET = process.env.ACCESS_TOKEN_SECRET;
const REFRESH_TOKEN_SECRET = process.env.REFRESH_TOKEN_SECRET;
const ACCESS_TOKEN_EXPIRY = '15m'; // Short-lived access token
const REFRESH_TOKEN_EXPIRY = '7d'; // Long-lived refresh token

if (!ACCESS_TOKEN_SECRET || !REFRESH_TOKEN_SECRET) {
  throw new Error('JWT secrets must be defined in environment variables');
}

/**
 * Generates an access JWT for a user.
 * @param {object} user - The user object (must contain at least `id` and `email`).
 * @returns {string} Signed JWT.
 */
function generateAccessToken(user) {
  return jwt.sign(
    { id: user.id, email: user.email },
    ACCESS_TOKEN_SECRET,
    { expiresIn: ACCESS_TOKEN_EXPIRY }
  );
}

/**
 * Generates a refresh JWT for a user.
 * @param {object} user - The user object (must contain at least `id`).
 * @returns {string} Signed refresh JWT.
 */
function generateRefreshToken(user) {
  return jwt.sign(
    { id: user.id, type: 'refresh' },
    REFRESH_TOKEN_SECRET,
    { expiresIn: REFRESH_TOKEN_EXPIRY }
  );
}

/**
 * Verifies an access token and returns its payload.
 * @param {string} token - The JWT to verify.
 * @returns {object} Decoded token payload if valid.
 * @throws {jwt.JsonWebTokenError} If token is invalid or expired.
 */
function verifyAccessToken(token) {
  return jwt.verify(token, ACCESS_TOKEN_SECRET);
}

/**
 * Verifies a refresh token and returns its payload.
 * @param {string} token - The refresh JWT to verify.
 * @returns {object} Decoded token payload if valid.
 * @throws {jwt.JsonWebTokenError} If token is invalid or expired.
 */
function verifyRefreshToken(token) {
  return jwt.verify(token, REFRESH_TOKEN_SECRET);
}

module.exports = {
  generateAccessToken,
  generateRefreshToken,
  verifyAccessToken,
  verifyRefreshToken,
  ACCESS_TOKEN_EXPIRY,
  REFRESH_TOKEN_EXPIRY,
};
```

**D. Core Logic Skeleton Code:**

**1. Authentication Middleware (`src/middleware/auth.js`):**
Protects routes by verifying the JWT from an HTTP-only cookie.

```javascript
const { verifyAccessToken } = require('../utils/tokens');

/**
 * Express middleware that authenticates requests using a JWT.
 * If authentication succeeds, attaches user info to `req.user`.
 * If it fails, sends a 401 Unauthorized response.
 */
function authenticate(req, res, next) {
  // TODO 1: Extract the token from the `Cookie` header.
  //   - Cookies are formatted as `key=value; key2=value2`.
  //   - Look for a cookie named `token`.
  //   - Hint: Use `req.headers.cookie` and parse it, or use a library like `cookie-parser`.

  // TODO 2: If no token is found, respond with 401 Unauthorized and a JSON error message.

  // TODO 3: Verify the token using `verifyAccessToken(token)`.
  //   - This function will throw an error if the token is invalid or expired.
  //   - Wrap the call in a try/catch block.

  // TODO 4: On successful verification, attach the decoded payload (which contains `id` and `email`) to `req.user`.

  // TODO 5: Call `next()` to proceed to the next middleware/route handler.
}

module.exports = authenticate;
```

**2. Registration Route Handler (Excerpt in `src/routes/auth.js`):**
Handles the `/api/auth/register` endpoint.

```javascript
const express = require('express');
const { query } = require('../db/pool');
const { hashPassword } = require('../utils/password');
const { validateRegistration } = require('../utils/validation'); // Assume this exists

const router = express.Router();

router.post('/register', async (req, res, next) => {
  // TODO 1: Validate input using `validateRegistration(req.body)`.
  //   - It should check email format, password strength, and name length.
  //   - If validation fails, return a 400 Bad Request with error details.

  // TODO 2: Check if a user with the given email already exists in the database.
  //   - Use a parameterized query: `SELECT id FROM users WHERE email = $1`.
  //   - If a row is returned, respond with 409 Conflict (email already exists).

  // TODO 3: Hash the provided password using `hashPassword(password)`.

  // TODO 4: Insert the new user into the database.
  //   - Use a parameterized query: `INSERT INTO users (email, password_hash, name) VALUES ($1, $2, $3) RETURNING id, email, name, created_at`.
  //   - Remember to use the hash from step 3, not the plain password.

  // TODO 5: On successful insertion, respond with 201 Created and the user data (excluding password_hash).
  //   - Format the `created_at` timestamp as an ISO string.

  // TODO 6: Wrap the entire async handler in a try/catch block.
  //   - Pass any errors to `next(error)` for the central error handler.
});
```

**3. Login Route Handler (Excerpt in `src/routes/auth.js`):**

```javascript
router.post('/login', async (req, res, next) => {
  // TODO 1: Validate that `email` and `password` are present in the request body.

  // TODO 2: Fetch the user record from the database by email.
  //   - Use a parameterized query to select `id`, `email`, `name`, `password_hash`.
  //   - **CRITICAL:** If no user is found, still proceed to simulate password verification to prevent timing attacks.
  //   - You can compare against a dummy hash (e.g., `$2b$12$...`).

  // TODO 3: Verify the provided password against the stored hash using `verifyPassword`.
  //   - This function handles the constant-time comparison.

  // TODO 4: If password verification fails (or user wasn't found), return a generic 401 Unauthorized error.
  //   - Message: "Invalid email or password."

  // TODO 5: On successful verification, generate an access token and a refresh token.
  //   - Use `generateAccessToken(user)` and `generateRefreshToken(user)`.

  // TODO 6: Store the refresh token in the database (e.g., in a `refresh_tokens` table) associated with the user ID.
  //   - This allows invalidation on logout. Use a parameterized INSERT.

  // TODO 7: Set the access token as an HTTP-only cookie.
  //   - Cookie name: `token`.
  //   - Options: `httpOnly: true`, `secure: process.env.NODE_ENV === 'production'`, `sameSite: 'strict'`, `maxAge: 15 minutes in ms`.
  //   - Optionally, set the refresh token in a separate cookie (also HTTP-only) or return it in the response body (less secure).

  // TODO 8: Respond with 200 OK and a JSON body containing minimal user info (`id`, `email`, `name`).
});
```

**4. Password Reset Token Generation (Excerpt):**

```javascript
const crypto = require('crypto');

router.post('/forgot-password', async (req, res) => {
  // TODO 1: Validate email format.

  // TODO 2: Generate a cryptographically secure random token (32 bytes hex).
  //   - Use `crypto.randomBytes(32).toString('hex')`.

  // TODO 3: Hash this token (using bcrypt or SHA-256) for safe database storage.
  //   - Store the hash, NOT the plain token.

  // TODO 4: Store the hash, user id (if user exists), and expiry (e.g., 1 hour from now) in a `password_reset_tokens` table.
  //   - Use an `UPDATE` or `INSERT` with a parameterized query.
  //   - **Do not** reveal if the email exists. If the user doesn't exist, still "succeed" without doing anything.

  // TODO 5: If a user was found, send an email with the plain (unhashed) reset token.
  //   - Construct a reset link: `https://yourapp.com/reset-password?token=<plain-token>`.
  //   - Use Nodemailer or a similar library.

  // TODO 6: Respond with 202 Accepted and a generic success message.
});
```

**E. Language-Specific Hints (Node.js/Express):**
- **Environment Variables:** Use the `dotenv` package to load `.env` files. Never commit secrets.
- **Cookie Parsing:** Use the `cookie-parser` middleware to easily access `req.cookies`.
- **Async Error Handling:** Use `express-async-errors` or wrap async route handlers in a try/catch that calls `next(error)`.
- **Password Hashing:** `bcryptjs` is a pure-JS implementation of bcrypt, compatible with `bcrypt` but without native dependencies, making it easier to deploy.
- **JWT Secrets:** Generate strong secrets using `openssl rand -base64 32` and store them in environment variables.
- **Database Queries:** Always use parameterized queries with `pg` to prevent SQL injection. The `query(text, params)` function from our pool helper does this.

**F. Milestone Checkpoint for Authentication (Milestone 2):**

After implementing the authentication component, verify its functionality with the following steps:

1. **Run the Server:** Start your development server (`npm run dev` or `node server.js`).
2. **Test Registration:** Use `curl` or Postman to send a POST request to `http://localhost:3000/api/auth/register`.
   ```bash
   curl -X POST http://localhost:3000/api/auth/register \
     -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","password":"StrongPass123!","name":"Test User"}'
   ```
   - **Expected Output:** `201 Created` with JSON user data (no password hash).
   - **Check Database:** Verify a new row exists in the `users` table with a `password_hash` column that starts with `$2b$12$` (indicating bcrypt with cost 12).
3. **Test Login:** Use the same credentials to login.
   ```bash
   curl -X POST http://localhost:3000/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","password":"StrongPass123!"}' \
     -v  # The -v flag will show response headers, including the Set-Cookie header.
   ```
   - **Expected Output:** `200 OK` with user JSON. Check the `Set-Cookie` header for an HTTP-only `token` cookie.
4. **Test Protected Route:** Attempt to access a protected endpoint (e.g., a dummy `/api/protected`) without and with the cookie.
   - Without cookie: Should return `401 Unauthorized`.
   - With cookie: Use the cookie from the login response in a subsequent request.
     ```bash
     curl -X GET http://localhost:3000/api/protected \
       -H "Cookie: token=<JWT_VALUE_FROM_LOGIN>"
     ```
     Should return a successful response (perhaps a simple message).
5. **Test Logout:** Call the logout endpoint.
   - **Expected Output:** `200 OK` and a `Set-Cookie` header that expires or clears the token cookie.
   - **Verification:** The previously valid token should now result in a `401` when accessing the protected route.

**Signs Something is Wrong:**
- **Registration fails with "email already exists" even for new emails:** Check your database unique constraint on the `email` column.
- **Login always returns "Invalid email or password":** Ensure you are comparing the correct hash. Use `console.log` to debug the hash retrieval and verification steps. Verify the password hash is being stored correctly during registration.
- **Cookie not being set in browser/curl:** Ensure you are setting the correct headers (`httpOnly`, `secure` maybe false in development). Check the `Set-Cookie` header is present in the response.
- **JWT verification fails:** Ensure the `ACCESS_TOKEN_SECRET` is identical between token generation and verification. Check token expiration.

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
| :--- | :--- | :--- | :--- |
| **Can't login even with correct password** | 1. Password hash mismatch (stored incorrectly).<br>2. `bcrypt.compare` failing due to non-string input. | 1. Log the stored hash from DB and the plain password being hashed during login simulation.<br>2. Check data types; ensure password is a string. | 1. Verify registration logic hashes the password before storage.<br>2. Explicitly cast input to string. |
| **JWT verification throws "invalid signature"** | Secret mismatch between signing and verifying. | Log the `ACCESS_TOKEN_SECRET` used in both `generateAccessToken` and `verifyAccessToken`. | Ensure the environment variable is loaded correctly and consistent across server restarts. |
| **Cookie not sent with subsequent requests** | Cookie attributes (`secure`, `domain`, `path`) preventing transmission. | Check the `Set-Cookie` header in the response. Use browser DevTools > Application > Cookies to inspect stored cookies. | In development, set `secure: false`. Ensure `sameSite` is not too restrictive (`Lax` is often sufficient). |
| **Password reset token is rejected as invalid** | 1. Token not stored correctly (hashing mismatch).<br>2. Token expired. | 1. Compare the hash of the provided token with the stored hash.<br>2. Check the `expires_at` column in the database. | 1. Ensure you hash the token with the same algorithm before storage.<br>2. Use a library like `moment` or native Date to handle expiry comparison. |
| **Slow response on login/registration** | High bcrypt work factor (this is normal, but could be excessive). | Time the `hashPassword` and `verifyPassword` functions. | Consider reducing `SALT_ROUNDS` to 10 for development if 12 is too slow, but never below 10 in production. |


## Component: Blog CRUD Operations

> **Milestone(s):** Milestone 3 (Blog CRUD Operations)

This component is the heart of the blog platform—the engine that transforms user-authored content into persistent, retrievable, and manageable articles. It handles the complete lifecycle of a blog post, from its creation as raw markdown text through its publication, display, updating, and eventual removal. The core challenge is designing a system that is simultaneously simple for learners to implement, secure against unauthorized access, and efficient for presenting content to readers.

### Mental Model: The Library Catalog System

Think of this component as a digital library catalog system for a public library. The **blog posts** are the books in the collection. The **authors** (authenticated users) are like librarians or contributing scholars who have special privileges to add new books, update existing editions, or remove outdated ones. The **readers** (any visitor) can browse the catalog, search for titles, and check out books to read, but they cannot alter the collection itself.

*   **Creating a Post (Acquiring a New Book):** A librarian (author) submits a new book to the collection. They provide a title, the book's content, and cataloging information. The library system records the book's details, assigns it a unique identifier, stamps it with the acquisition date, and links it to the librarian who processed it. Similarly, the blog platform accepts a title and markdown content, records the author and timestamp, and stores it in the database.
*   **Listing Posts (Browsing the Catalog):** A visitor wants to see what books are available. The library doesn't dump every single book onto a table; it provides a catalog with summaries, organized by shelf (newest arrivals, by genre). The visitor can flip through pages of results. Our blog's home page does the same: it shows a paginated list of post titles and excerpts, sorted by publication date.
*   **Reading a Single Post (Checking Out a Book):** A visitor selects a specific book from the catalog. The librarian retrieves the full book from the stacks. The visitor then reads the complete content. In the blog, clicking a post title fetches and renders the full post, including its formatted content and metadata.
*   **Updating a Post (Revising an Edition):** Only the original librarian who cataloged a book (or a head librarian) is authorized to make corrections or publish a revised edition. The system ensures others cannot vandalize the record. Our platform enforces that only the post's author can edit its title and content.
*   **Deleting a Post (Withdrawing a Book from Circulation):** A book might be removed due to damage or being outdated. Rather than shredding it, the library might mark it as "archived" in its records, keeping the data for historical purposes but hiding it from the public catalog. Our blog can implement a similar "soft delete," where a post is hidden but its data persists.

This mental model clarifies the distinct roles (author vs. reader), the importance of ownership, and the concept of non-destructive removal. It frames the **authorization** logic as a natural library policy, not an arbitrary technical rule.

### CRUD API Interface

The API interface is a set of RESTful endpoints that expose the blog post operations. All endpoints that modify data (`POST`, `PUT`, `DELETE`) require a valid authentication token (JWT) in the `Authorization` header. The server uses this token to identify the acting user and enforce ownership rules.

The following table details the endpoints, their methods, expected inputs, and outputs.

| Method | Endpoint | Parameters (Request Body/Query) | Returns (Response Body) | Description & Rules |
| :--- | :--- | :--- | :--- | :--- |
| **POST** | `/api/posts` | `{ "title": string, "content": string, "tags": string[] }` | `201 Created`: `{ "post": PostObject }` | Creates a new blog post. The authenticated user (from the JWT) is automatically set as the `author_id`. Validates that `title` and `content` are non-empty strings. Returns the newly created post object including its generated `id`. |
| **GET** | `/api/posts` | `?page=number&limit=number&tag=string` | `200 OK`: `{ "posts": PostObject[], "pagination": { "page": number, "limit": number, "total": number, "totalPages": number } }` | Retrieves a paginated list of published posts, sorted by `created_at` descending (newest first). The `page` query parameter defaults to 1, and `limit` defaults to 20. An optional `tag` filter returns only posts containing that tag. The response includes pagination metadata. |
| **GET** | `/api/posts/:id` | `:id` (URL parameter) | `200 OK`: `{ "post": PostObjectWithAuthor }` <br> `404 Not Found`: `{ "error": "Post not found" }` | Retrieves a single post by its unique ID. The returned `PostObjectWithAuthor` includes the post's core fields plus nested author information (e.g., `author: { id, name }`). Returns a 404 if the ID doesn't exist or the post is soft-deleted. |
| **PUT** | `/api/posts/:id` | `:id` (URL parameter), `{ "title": string, "content": string }` | `200 OK`: `{ "post": PostObject }` <br> `403 Forbidden`: `{ "error": "Not authorized to edit this post" }` <br> `404 Not Found` | Updates an existing post. The server first verifies that the authenticated user's ID matches the post's `author_id`. If not, it returns a 403. On success, it updates the `title`, `content`, and the `updated_at` timestamp. Returns the updated post object. |
| **DELETE** | `/api/posts/:id` | `:id` (URL parameter) | `204 No Content` (empty body) <br> `403 Forbidden` <br> `404 Not Found` | Deletes a post. The server verifies ownership (author_id match). If the system implements soft deletion, this endpoint sets a `deleted_at` timestamp instead of removing the row. Returns a 204 status on success. |

**Key Data Objects:**

| Field Name | Type | Description |
| :--- | :--- | :--- |
| **PostObject** (Basic) | Object | The core post representation returned by list and create/update endpoints. |
| `id` | string (UUID) or integer | The unique, immutable identifier for the post. |
| `title` | string | The post's title. |
| `content` | string | The raw markdown content as stored in the database. |
| `author_id` | string/integer | Foreign key referencing the `id` of the `users` table. |
| `created_at` | string (ISO 8601) | Timestamp of post creation. |
| `updated_at` | string (ISO 8601) | Timestamp of the most recent update. Null if never updated. |
| `tags` | Array\<string> | An array of tag labels associated with the post. |
| **PostObjectWithAuthor** (Extended) | Object | Extends `PostObject` for the single-post endpoint. |
| `author` | Object | Nested object containing author details. |
| `author.id` | string/integer | The author's user ID. |
| `author.name` | string | The author's display name. |
| `author.email` | string (optional) | The author's email, typically only included if the requesting user is the author. |
| `html_content` | string (optional) | The `content` field rendered to sanitized HTML. May be included for convenience in the single-post response. |

The interaction flow for creating a post, which involves authorization and database insertion, is illustrated in the sequence diagram: ![Post Creation Sequence Diagram](./diagrams/post-creation-sequence.svg)

### ADR: Pagination Strategy

> **Decision: Offset-Based Pagination with Page Size Limits**
>
> - **Context**: The blog platform needs to display a list of posts on the homepage and potentially in admin views. As the number of posts grows, returning all records in a single response becomes inefficient for the server, slow for the client, and overwhelming for the user. We need a strategy to break the results into manageable chunks.
> - **Options Considered**:
>     1. **Offset/Limit Pagination**: The client requests a specific "page" of results using `LIMIT` and `OFFSET` SQL clauses (e.g., `page=2&limit=20` translates to `OFFSET 20 LIMIT 20`).
>     2. **Cursor-Based Pagination**: The client requests records "after" a specific reference point (a cursor, like the `id` or `created_at` of the last seen item). The query uses a `WHERE created_at < ?` clause instead of `OFFSET`.
> - **Decision**: We will implement offset/limit pagination.
> - **Rationale**:
>     *   **Simplicity for Learning**: The concept of page numbers is intuitive and maps directly to common UI patterns (Page 1, 2, 3). The SQL is straightforward.
>     *   **Implementation Simplicity**: It requires minimal client-side state (just the current page number) and is easy to implement with common **ORM** tools.
>     *   **Adequate for Expected Scale**: For a typical learning-project blog, the number of posts will be in the hundreds or low thousands. The performance drawbacks of `OFFSET` (which must skip rows) are negligible at this scale.
> - **Consequences**:
>     *   **Positive**: Easier to understand, implement, and debug. Allows random access to any page (e.g., jumping to page 5).
>     *   **Negative**: Performance degrades with very large offsets (e.g., `OFFSET 10000`). Pages can become inconsistent if data is being added or deleted while a user is paginating (a new post on page 1 shifts all others, causing duplicates or skips).

The following table summarizes the trade-offs that led to this decision:

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **Offset/Limit** | Simple to understand and implement. Allows direct page jumps. Universal ORM support. | Performance degrades at high offsets (`OFFSET` must scan skipped rows). Results can be inconsistent with real-time data changes (e.g., a new post added while paging). | **Yes** |
| **Cursor-Based** | Consistent performance regardless of list position (uses indexed `WHERE` clause). Provides stable results during concurrent writes. | More complex client logic (must track the cursor). No random access to arbitrary pages. Requires a unique, sequential cursor field. | No |

### ADR: Markdown Processing

> **Decision: Store Raw Markdown, Render to HTML On-Demand with Caching**
>
> - **Context**: Blog posts are authored in Markdown, a lightweight markup language, but must be displayed as HTML in the browser. We must decide how to manage the transformation between these two formats and where each version resides in our system.
> - **Options Considered**:
>     1. **Store Raw Markdown Only**: Persist only the original markdown text in the database. Render it to HTML each time a post is requested for display.
>     2. **Store Both Markdown and HTML**: Persist both the raw markdown and a pre-rendered HTML version in the database (either in two columns or in a separate table). Serve the pre-computed HTML for reads.
>     3. **Store HTML Only**: Convert markdown to HTML upon post creation/update and store only the HTML. Lose the original markdown source.
> - **Decision**: We will store only the raw markdown content and render it to HTML on-demand when serving a post for reading, implementing a basic caching layer to avoid redundant rendering.
> - **Rationale**:
>     *   **Single Source of Truth**: Storing only markdown eliminates the risk of the stored HTML and markdown becoming desynchronized.
>     *   **Editing Flexibility**: The author must have access to the original markdown to make edits. Storing only HTML would make editing impossible or require a complex reverse-engineering process.
>     *   **Storage Efficiency**: Storing duplicate content (markdown + HTML) increases database size, which is a minor but unnecessary cost for this project.
>     *   **Computational Trade-off Acceptable**: Markdown rendering is a fast, CPU-bound operation. For a blog with moderate traffic, the cost of rendering on each request is minimal. A simple in-memory cache (e.g., storing the rendered HTML for the most recent N posts) can further mitigate this.
> - **Consequences**:
>     *   **Positive**: Simplifies the data model. Guarantees the edit interface always has the correct source.
>     *   **Negative**: Adds CPU overhead to each post view request. Requires careful HTML sanitization on every render to prevent **XSS** attacks from malicious markdown input.

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **Store Raw Markdown Only** | Simple data model. Always editable. No synchronization issues. | Rendering overhead on every view request. Must sanitize HTML on each render. | **Yes** |
| **Store Both Markdown and HTML** | Fast reads (serve pre-rendered HTML). Editing still possible. | Increased storage. Complexity in keeping both versions in sync on updates. | No |
| **Store HTML Only** | Fastest reads. Simple data model. | Original markdown lost, making future edits extremely difficult or impossible. | No |

### Common Pitfalls in CRUD

Learners implementing blog CRUD operations often encounter the following specific, concrete pitfalls. Understanding and avoiding them is crucial for building a correct and secure system.

⚠️ **Pitfall: N+1 Query Problem in Post Listing**
*   **Description**: When fetching a list of posts, a naive implementation issues one query to get the posts, then loops through each post and issues a separate query to fetch its author's name (`SELECT * FROM users WHERE id = post.author_id`). For a list of 20 posts, this results in 1 (get posts) + 20 (get authors) = 21 database queries.
*   **Why it's wrong**: This is extremely inefficient, causing high database load and slow page response times. The problem scales linearly with the number of posts.
*   **How to fix**: Use a **JOIN** in the initial query or your **ORM**'s eager-loading mechanism (e.g., `include` or `populate`). A single query should fetch all posts and their associated author information. For example: `SELECT posts.*, users.name as author_name FROM posts JOIN users ON posts.author_id = users.id ORDER BY posts.created_at DESC LIMIT 20`.

⚠️ **Pitfall: XSS via Unsanitized Markdown Rendering**
*   **Description**: Directly converting user-provided markdown to HTML and injecting it into the DOM without sanitization. Markdown can contain raw HTML tags (e.g., `<script>alert('xss')</script>`). If this HTML is not cleaned, the script will execute in readers' browsers.
*   **Why it's wrong**: This is a critical security vulnerability (**XSS**) that allows attackers to hijack user sessions, deface the site, or steal data.
*   **How to fix**: Always pass the rendered HTML through a reputable sanitization library (e.g., `DOMPurify` for JavaScript, `bleach` for Python) before sending it to the client. These libraries strip out dangerous tags and attributes while preserving safe formatting.

⚠️ **Pitfall: Missing Ownership Checks on Update/Delete**
*   **Description**: An endpoint like `PUT /api/posts/123` correctly verifies that the request contains a valid JWT but fails to check if the user ID encoded in that JWT matches the `author_id` of post 123. A malicious user could edit or delete anyone's post simply by knowing its ID.
*   **Why it's wrong**: This violates the core authorization principle of the blog. Users must only be able to modify their own content.
*   **How to fix**: After retrieving the post from the database, add an explicit authorization check before proceeding with the operation. The logic should be: `if (request.userId !== post.author_id) { return res.status(403).json({ error: "Forbidden" }); }`. This check must be performed on both `PUT` and `DELETE` endpoints.

⚠️ **Pitfall: Not Using Parameterized Queries**
*   **Description**: Constructing SQL queries by directly concatenating user input into the query string (e.g., `const query = "SELECT * FROM posts WHERE title = '" + userInput + "'";`).
*   **Why it's wrong**: This is the primary cause of **SQL injection** vulnerabilities. A malicious user could provide input like `' OR '1'='1` to manipulate the query logic and access unauthorized data.
*   **How to fix**: **Always use parameterized queries** (also known as prepared statements). With the `pg` library in Node.js, this looks like `pool.query('SELECT * FROM posts WHERE id = $1', [postId])`. The database driver ensures user input is treated as data, not executable SQL code.

### Implementation Guidance

This section provides concrete, actionable code and structure to implement the Blog CRUD component in JavaScript (Node.js with Express).

#### A. Technology Recommendations

| Component | Simple Option | Advanced Option |
| :--- | :--- | :--- |
| **Web Framework** | Express.js | Fastify |
| **Database Driver** | `pg` (node-postgres) | Prisma ORM |
| **Markdown Renderer** | `marked` | `remark` with `rehype` ecosystem |
| **HTML Sanitizer** | `DOMPurify` (can run in Node with `jsdom`) | `sanitize-html` |
| **Validation** | Manual checks in route handlers | Joi or Zod schemas |

#### B. Recommended File/Module Structure

Organize the blog-related code within the `routes/` and `models/` directories. The `middleware/` directory holds shared authorization logic.

```
blog-platform/
├── server.js (or index.js)
├── package.json
├── config/
│   └── database.js          # Database connection pool setup (PoolConfig)
├── middleware/
│   ├── auth.js              # authenticateToken middleware
│   └── authorizePost.js     # Optional: Specific post authorization middleware
├── models/
│   ├── index.js             # Exports all models
│   ├── user.model.js
│   └── post.model.js        # Post data access layer (queries)
├── routes/
│   ├── index.js             # Main router, combines all route files
│   ├── auth.routes.js
│   └── posts.routes.js      # All /api/posts route handlers
├── utils/
│   ├── markdown.js          # markdownToHtml(sanitized) utility
│   └── pagination.js        # calculatePaginationMetadata(total, page, limit)
└── test/
    └── posts.test.js
```

#### C. Infrastructure Starter Code

**1. Database Connection Pool (`config/database.js`):**
This is a complete, reusable setup for the `pg` pool. It defines the `PoolConfig` and exports a `query` function.

```javascript
const { Pool } = require('pg');

// Configuration matching the PoolConfig type
const poolConfig = {
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'blogdb',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || '',
  max: process.env.DB_POOL_MAX || 20,            // DB_POOL_MAX
  idleTimeoutMillis: process.env.DB_POOL_IDLE_TIMEOUT || 30000, // DB_POOL_IDLE_TIMEOUT
  connectionTimeoutMillis: process.env.DB_POOL_CONN_TIMEOUT || 2000, // DB_POOL_CONN_TIMEOUT
};

const pool = new Pool(poolConfig);

// Helper function to execute a parameterized query, returns Promise<QueryResult>
const query = (text, params) => pool.query(text, params);

// Helper to get a client for transactions (returns Promise<PoolClient>)
const getClient = () => pool.connect();

module.exports = { query, getClient };
```

**2. Markdown Rendering & Sanitization Utility (`utils/markdown.js`):**
A complete utility that safely converts markdown to HTML.

```javascript
const marked = require('marked');
const { JSDOM } = require('jsdom');
const createDOMPurify = require('dompurify');

const window = new JSDOM('').window;
const DOMPurify = createDOMPurify(window);

/**
 * Converts markdown text to sanitized HTML.
 * @param {string} markdown - The raw markdown content.
 * @returns {string} Sanitized HTML string safe for injection into the DOM.
 */
function markdownToHtml(markdown) {
  if (!markdown) return '';
  // 1. Convert markdown to (unsafe) HTML using marked
  const unsafeHtml = marked.parse(markdown);
  // 2. Sanitize the HTML to remove XSS threats
  const cleanHtml = DOMPurify.sanitize(unsafeHtml);
  return cleanHtml;
}

module.exports = { markdownToHtml };
```

#### D. Core Logic Skeleton Code

**1. Post Model (`models/post.model.js`):**
This is the data access layer. Implement the functions with the TODO steps.

```javascript
const { query } = require('../config/database');

/**
 * Represents a blog post in the system.
 * @typedef {Object} Post
 * @property {number} id
 * @property {string} title
 * @property {string} content
 * @property {number} author_id
 * @property {Date} created_at
 * @property {Date} updated_at
 * @property {string[]} tags
 */

const PostModel = {
  /**
   * Creates a new post record in the database.
   * @param {string} title
   * @param {string} content
   * @param {number} authorId
   * @param {string[]} tags
   * @returns {Promise<Post>} The newly created post.
   */
  async create(title, content, authorId, tags = []) {
    // TODO 1: Construct a parameterized INSERT query for the posts table.
    // TODO 2: Include values for title, content, author_id, tags, created_at, and updated_at.
    // TODO 3: Use the `query` function to execute the INSERT.
    // TODO 4: Return the first row from the QueryResult (the inserted post).
    // Hint: Use `RETURNING *` in your SQL to get the full inserted row.
  },

  /**
   * Finds posts with pagination and optional tag filter.
   * Performs a JOIN to include the author's name to avoid N+1 queries.
   * @param {number} page - The page number (1-indexed).
   * @param {number} limit - Number of posts per page.
   * @param {string} [tag] - Optional tag to filter by.
   * @returns {Promise<{posts: Post[], total: number}>}
   */
  async findPaginated(page = 1, limit = 20, tag = null) {
    const offset = (page - 1) * limit;
    // TODO 1: Build the base SQL: SELECT posts.*, users.name as author_name FROM posts JOIN users ON posts.author_id = users.id
    // TODO 2: If a `tag` is provided, add a WHERE clause to filter posts where the tag is in the `tags` array.
    //         Research PostgreSQL's `@>` operator or `?` operator for array containment.
    // TODO 3: Add ORDER BY posts.created_at DESC.
    // TODO 4: Add LIMIT and OFFSET clauses for pagination.
    // TODO 5: Execute the query to get the paginated posts.
    // TODO 6: Execute a separate COUNT(*) query (without LIMIT/OFFSET) to get the total number of posts matching the filter.
    // TODO 7: Return an object: { posts: rows, total: parseInt(countResult.rows[0].count) }.
  },

  /**
   * Finds a single post by its ID, including author details.
   * @param {number} postId
   * @returns {Promise<Post | null>} The post with an extra `author_name` field, or null if not found.
   */
  async findById(postId) {
    // TODO 1: Write a SELECT query that joins posts with users to get the author's name.
    // TODO 2: Use a WHERE clause on posts.id.
    // TODO 3: Execute the query.
    // TODO 4: If a row is found, return it. If no rows, return null.
  },

  /**
   * Updates a post's title and content. Also updates the `updated_at` timestamp.
   * @param {number} postId
   * @param {string} title
   * @param {string} content
   * @returns {Promise<Post>} The updated post.
   */
  async update(postId, title, content) {
    // TODO 1: Construct a parameterized UPDATE query for the posts table.
    // TODO 2: Set the title, content, and updated_at columns.
    // TODO 3: Use a WHERE clause on id.
    // TODO 4: Use `RETURNING *` to get the updated row.
    // TODO 5: Execute the query and return the first row.
  },

  /**
   * Soft-deletes a post by setting its `deleted_at` timestamp.
   * @param {number} postId
   * @returns {Promise<void>}
   */
  async softDelete(postId) {
    // TODO 1: Construct an UPDATE query to set `deleted_at` to the current timestamp.
    // TODO 2: Use a WHERE clause on id.
    // TODO 3: Execute the query. No need to return data.
  },

  /**
   * Checks if a user is the author of a post.
   * @param {number} postId
   * @param {number} userId
   * @returns {Promise<boolean>} True if the user is the author.
   */
  async isAuthor(postId, userId) {
    // TODO 1: Write a SELECT query to fetch the author_id of the post with the given id.
    // TODO 2: Execute the query.
    // TODO 3: Compare the fetched author_id with the provided userId.
    // TODO 4: Return true if they match, false otherwise (or if post not found).
  },
};

module.exports = PostModel;
```

**2. Posts Route Handler (`routes/posts.routes.js`):**
This skeleton shows the route structure with critical authorization checks. Fill in the TODOs.

```javascript
const express = require('express');
const router = express.Router();
const PostModel = require('../models/post.model');
const { authenticateToken } = require('../middleware/auth');
const { markdownToHtml } = require('../utils/markdown');
const { calculatePaginationMetadata } = require('../utils/pagination');

// All routes below require authentication
router.use(authenticateToken);

// CREATE a new post
router.post('/', async (req, res) => {
  // TODO 1: Extract title, content, and optional tags from req.body.
  // TODO 2: Validate that title and content are non-empty strings. Return 400 if not.
  // TODO 3: Get the author's ID from req.user.id (set by authenticateToken middleware).
  // TODO 4: Call PostModel.create with the validated data.
  // TODO 5: On success, respond with status 201 and the created post object.
  // TODO 6: Catch errors (e.g., database errors) and respond with status 500.
});

// READ a paginated list of posts
router.get('/', async (req, res) => {
  // TODO 1: Extract page, limit, and tag from query parameters (req.query). Provide defaults.
  // TODO 2: Call PostModel.findPaginated(page, limit, tag).
  // TODO 3: Use the utils/pagination helper to generate metadata (totalPages, etc.).
  // TODO 4: Respond with status 200 and an object containing `posts` and `pagination`.
});

// READ a single post by ID
router.get('/:id', async (req, res) => {
  // TODO 1: Extract post ID from req.params.id.
  // TODO 2: Call PostModel.findById(postId).
  // TODO 3: If no post is found, respond with status 404.
  // TODO 4: If found, add an `html_content` field by calling markdownToHtml(post.content).
  // TODO 5: Respond with status 200 and the enriched post object.
});

// UPDATE a post
router.put('/:id', async (req, res) => {
  const postId = req.params.id;
  const userId = req.user.id; // From auth middleware
  // TODO 1: Validate req.body for title and content.
  // TODO 2: Check authorization: call PostModel.isAuthor(postId, userId).
  // TODO 3: If not author, respond with status 403 Forbidden.
  // TODO 4: If authorized, call PostModel.update(postId, title, content).
  // TODO 5: Respond with status 200 and the updated post.
});

// DELETE a post
router.delete('/:id', async (req, res) => {
  const postId = req.params.id;
  const userId = req.user.id;
  // TODO 1: Check authorization using PostModel.isAuthor.
  // TODO 2: If not author, respond with status 403.
  // TODO 3: If authorized, call PostModel.softDelete(postId).
  // TODO 4: On success, respond with status 204 No Content (empty body).
});

module.exports = router;
```

#### E. Language-Specific Hints (JavaScript/Node.js)

*   **Async/Await**: All database operations are asynchronous. Use `async/await` in route handlers for cleaner code compared to promise chains. Remember to wrap logic in `try/catch` blocks.
*   **Environment Variables**: Store your `PoolConfig` details (database password, etc.) in a `.env` file and load them using the `dotenv` package. Never hard-code credentials.
*   **Parameterized Queries with `pg`**: Always use the `$1, $2, ...` placeholder syntax with the `query` function. This is your primary defense against SQL injection.
*   **JWT in Requests**: The `authenticateToken` middleware (from the Authentication component) should extract and verify the JWT from the `Authorization: Bearer <token>` header, attaching the decoded user payload to `req.user`.

#### F. Milestone Checkpoint

After implementing the CRUD operations, you can verify your work with the following steps:

1.  **Start the server**: Run `npm start` or `node server.js`.
2.  **Test Authentication First**: Use a tool like `curl` or Postman to register a user and obtain a JWT token.
3.  **Test Post Creation**:
    *   **Command**: `curl -X POST http://localhost:3000/api/posts -H "Authorization: Bearer YOUR_JWT_TOKEN" -H "Content-Type: application/json" -d '{"title":"My First Post", "content":"## Hello World\nThis is **markdown**."}'`
    *   **Expected Output**: A `201 Created` response with the full post JSON, including a new `id`.
4.  **Test Post Listing**:
    *   **Command**: `curl http://localhost:3000/api/posts`
    *   **Expected Output**: A `200 OK` response with a JSON object containing a `posts` array (with your new post) and `pagination` metadata. The author's name should be included in each post.
5.  **Test Authorization**:
    1.  Create a post with User A's token.
    2.  Try to update that post's `id` using a token for User B.
    3.  **Expected Output**: A `403 Forbidden` error. If you get a `200`, your ownership check is missing.

**Signs Something is Wrong:**
*   **`500 Internal Server Error` on all POST/PUT/DELETE**: Likely a database connection issue or syntax error in your model's SQL. Check server logs.
*   **Author name is `null` or missing in the list view**: Your `JOIN` in `findPaginated` is incorrect or you're not selecting the `author_name` field.
*   **Can edit/delete any post**: The `isAuthor` check in your route handlers is not implemented or the `req.user.id` is not being set correctly by the `authenticateToken` middleware.


## Component: Frontend UI

> **Milestone(s):** Milestone 4 (Frontend UI)

This component is the face of the blog platform—the collection of interfaces through which readers consume content and authors create it. The primary architectural challenge lies in designing a responsive, intuitive, and performant user experience that cleanly separates presentation logic from business logic, while providing a clear learning path for fundamental frontend development concepts.

### Mental Model: The Newspaper Layout

Imagine the blog platform as a modern digital newspaper. This mental model helps categorize the various user interfaces by their journalistic purpose:

*   **The Front Page (Homepage):** This is the blog's landing page, analogous to a newspaper's front page. Its primary job is to attract attention and guide readers to interesting stories. It displays a curated, paginated list of recent articles (posts) with compelling headlines (titles), engaging ledes (excerpts), and bylines (author attribution). Just as a front page is designed for public consumption and search engine discovery, this page must be fast, SEO-friendly, and accessible to all.
*   **The Feature Article (Single Post View):** This is the dedicated page for a single story, akin to a full newspaper article spread. Its responsibility is immersive content delivery. It presents the full narrative (the rendered markdown content), the author's bio, the publication date, and a space for reader letters (comments). The focus is on readability and a distraction-free environment.
*   **The Editorial Desk (Post Editor):** This is the private workspace where journalists (authors) craft their stories. It functions like a newsroom desk with writing tools. It provides a split-pane interface: a raw manuscript area (markdown input) on the left and a typeset preview (HTML render) on the right, allowing the author to see their work take shape in real time. Access to this desk is restricted to authenticated authors.
*   **The Circulation Desk (Authentication Forms):** This is the membership and access control point, similar to a newspaper's subscription desk. It handles new reader registration (sign-up) and returning subscriber login (sign-in). Its role is to verify identity and grant the appropriate credentials (a JWT "press pass") to access restricted areas like the Editorial Desk.

This model clarifies the distinct roles of each UI section, guiding design decisions about access control, rendering strategy, and state management for each.

### Frontend Component Architecture

The UI is constructed as a tree of reusable, single-responsibility components. This hierarchy promotes maintainability and facilitates a clear data flow. The component structure is visualized in the diagram below:

![Frontend Component Hierarchy](./diagrams/frontend-component-hierarchy.svg)

The root of the application is the `App` component, which initializes global providers (like authentication context) and sets up the routing infrastructure. Below it, the architecture is organized into several key component groups:

| Component Group | Primary Responsibility | Key Child Components | Statefulness |
| :--- | :--- | :--- | :--- |
| **Layout Wrapper** | Provides consistent page structure (header, footer, navigation) across all views. Manages global layout state (e.g., mobile menu toggle). | `Header`, `Footer`, `NavBar`, `MainContent` | Stateful (UI state) |
| **Post List (`HomePage`)** | Fetches and displays a paginated list of blog post previews. Handles pagination controls and optional filtering (e.g., by tag). | `PostPreviewCard`, `PaginationControls`, `TagFilter` | Stateful (data, pagination) |
| **Post Detail (`PostPage`)** | Fetches and displays a single, full blog post, including its rendered content, metadata, and associated comments. | `PostContent`, `PostMeta`, `CommentSection`, `CommentForm` | Stateful (data, comments) |
| **Post Editor (`EditorPage`)** | Provides an interface for creating new posts or editing existing ones. Manages the markdown draft, live preview, and submission. | `MarkdownEditorPane`, `HtmlPreviewPane`, `EditorToolbar` | Stateful (form data, preview) |
| **Auth Forms (`LoginPage`, `RegisterPage`)** | Present forms for user authentication and registration. Handle client-side validation and submission to the backend API. | `EmailInput`, `PasswordInput`, `ValidationErrorDisplay` | Stateful (form data, errors) |

Data flows predominantly from parent to child via props. User interactions (like clicking "Edit" or submitting a comment) trigger callback functions passed down as props, which execute logic in the parent component (often involving API calls). Shared, global state—specifically the user's authentication status—is managed via React's Context API and made available to any component in the tree that needs it, such as the `Header` (to show login/logout buttons) and the `EditorPage` (to gate access).

### ADR: Rendering Strategy

> **Decision: Hybrid Rendering (SSR for Public Content, CSR for Authenticated Interfaces)**
>
> *   **Context**: The blog has pages with different requirements. Public pages (post lists, single posts) must be fast, indexable by search engines (SEO), and accessible without JavaScript. Authenticated pages (editor, dashboard) are dynamic, user-specific, and benefit from the smooth interactivity of a Single Page Application (SPA).
> *   **Options Considered**:
>     1.  **Client-Side Rendering (CSR)**: The browser downloads a minimal HTML shell and a JavaScript bundle, which then fetches data and renders the entire UI. This is typical for React apps created with `create-react-app`.
>     2.  **Server-Side Rendering (SSR)**: The server generates the complete HTML for each page request by running the React app, fetching necessary data, and sending a fully-formed page to the browser.
>     3.  **Static Site Generation (SSG)**: HTML pages are generated at build time. Ideal for content that rarely changes.
>     4.  **Hybrid Approach**: Use SSR/SSG for public, content-heavy pages and CSR for private, interactive application shells.
> *   **Decision**: Adopt a hybrid strategy. Implement public routes (`/`, `/posts/:id`) using Server-Side Rendering (or Static Generation if build-time data fetching is sufficient). Implement authenticated routes (`/editor`, `/dashboard`) and dynamic interactions (comments, liking) using Client-Side Rendering.
> *   **Rationale**: Public content is the blog's primary asset and must be discoverable. SSR ensures search engines receive complete HTML content, improving SEO. It also provides a faster First Contentful Paint for users on slow networks. The authenticated editor is a complex, stateful application where CSR provides a smoother, app-like experience without full page reloads. This aligns with the mental model: the newspaper's public pages are pre-printed (SSR), while the editorial desk is a live workstation (CSR).
> *   **Consequences**: This approach yields excellent SEO and initial load performance for readers. However, it introduces implementation complexity, requiring a meta-framework (like Next.js) capable of both SSR and CSR, or a custom server setup. It also means the development, build, and deployment processes are more involved than a pure CSR SPA.

The table below summarizes the trade-offs for the rendering strategy decision:

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **Client-Side Rendering (CSR)** | Simple implementation, rich interactivity, clear separation of frontend/backend. | Poor SEO, slow initial load, blank page without JavaScript. | For authenticated routes only |
| **Server-Side Rendering (SSR)** | Excellent SEO, fast initial render, works without client-side JS. | Higher server load, more complex deployment, "slower" page transitions. | For public routes (post list, detail) |
| **Static Site Generation (SSG)** | Blazing fast performance, minimal server load, superb security. | Cannot handle user-specific or real-time data without client-side hydration. | Not chosen (content is dynamic) |
| **Hybrid (SSR/CSR)** | Optimizes for both SEO (public) and interactivity (private). Balances performance and user experience. | Highest implementation complexity, requires advanced framework (Next.js, Nuxt). | **Yes** |

### ADR: State Management

> **Decision: React Context API for Global Auth State, Component State for Local UI**
>
> *   **Context**: The application needs a way to share the current user's authentication state (logged in/out, user object) across many components (header, comment forms, editor, etc.) without prop drilling. It also needs to manage local UI state (form inputs, loading spinners, pagination page) within specific components.
> *   **Options Considered**:
>     1.  **Prop Drilling**: Pass state and setters down through multiple levels of component hierarchy via props.
>     2.  **React Context API**: Create a context to hold the auth state and provide it to any component in the subtree.
>     3.  **Redux (or similar library)**: Maintain all application state in a centralized store, with actions and reducers to manage updates.
> *   **Decision**: Use the React Context API to manage global authentication state. Use React's built-in `useState` and `useReducer` hooks for all other component-local state (form data, post lists, UI toggles).
> *   **Rationale**: The scale of this blog platform does not warrant the overhead of Redux. The only truly global state is authentication—a simple object (`{ user, isLoading }`). The Context API is a lightweight, built-in solution perfectly suited for passing down this type of "global" data. Using local state for everything else (like the list of posts on the homepage or the draft in the editor) keeps components independent, logic co-located, and the data flow easy to reason about. This aligns with the principle of using the simplest tool that fits the problem.
> *   **Consequences**: The application avoids the boilerplate and conceptual complexity of Redux, making it more approachable for learners. Components remain highly reusable and testable in isolation. However, for more complex future features (e.g., real-time notifications, advanced comment threading), the Context API might lead to unnecessary re-renders if not optimized, potentially requiring a move to a more sophisticated state management solution.

The comparison of state management strategies is detailed below:

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **Prop Drilling** | Simple, explicit data flow, no external APIs. | Becomes unwieldy and "noisy" for deep component trees, reduces component reusability. | No |
| **React Context API** | Built into React, avoids prop drilling for specific "global" data, relatively simple. | Can cause unnecessary re-renders if overused or not memoized, not designed for high-frequency updates. | **Yes** (for auth only) |
| **Redux** | Predictable state transitions, powerful devtools, excellent for complex, interconnected state. | Significant boilerplate, steep learning curve, overkill for small to medium applications. | No |

### Common Pitfalls in Frontend

⚠️ **Pitfall: Silent Failure - Missing Loading and Error States**
*   **Description**: Making an API call and rendering the component without handling the pending (`loading`) and rejected (`error`) states of the asynchronous operation. The user sees a blank screen, a broken layout, or an infinite spinner with no feedback.
*   **Why It's Wrong**: It creates a poor user experience. Users cannot distinguish between a slow network and a broken application. They may repeatedly refresh the page or abandon the site.
*   **How to Fix**: For every data-fetching operation, explicitly manage three states: `loading`, `error`, and `success`. Render a skeleton loader or spinner during `loading`. Render a user-friendly error message with retry options in the `error` state. Only render the primary UI in the `success` state.

⚠️ **Pitfall: Rigid Layouts - Non-Responsive Design**
*   **Description**: Building the UI with fixed pixel widths, absolute positioning, or non-flexible grids that do not adapt to different screen sizes (desktop, tablet, mobile). This leads to horizontal scrolling, overlapping elements, or unreadably small text on mobile devices.
*   **Why It's Wrong**: A significant portion of web traffic is mobile. A non-responsive site is inaccessible and unprofessional, directly harming reader engagement.
*   **How to Fix**: Use a mobile-first CSS approach. Employ flexible layout tools like CSS Flexbox and CSS Grid. Use relative units (`rem`, `%`, `vw/vh`) over fixed pixels (`px`). Utilize CSS media queries to apply conditional styles for different breakpoints. Constantly test the UI in browser developer tools' device emulation modes.

⚠️ **Pitfall: Trusting the Client - Missing Form Validation Feedback**
*   **Description**: Relying solely on HTML5 form validation (`required`, `type="email"`) or providing only generic error messages like "Invalid input" from the server response.
*   **Why It's Wrong**: HTML5 validation is easily bypassed and provides inconsistent browser styling. Generic errors force users to guess what went wrong. This leads to user frustration and failed submissions.
*   **How to Fix**: Implement **layered validation**. 1) Use client-side validation for immediate UX feedback (e.g., using libraries like `yup` or `zod` with form libraries). Highlight specific fields and provide clear, inline messages (e.g., "Password must be at least 8 characters"). 2) Always re-validate on the server for security, and return structured, field-specific error messages that the frontend can map back to the form inputs.

⚠️ **Pitfall: Memory Leaks - Not Cleaning Up Side Effects**
*   **Description**: Starting asynchronous operations (API calls, subscriptions, timers) in a component's `useEffect` hook or event listener but not cancelling them when the component unmounts. This can lead to trying to update the state of an unmounted component, causing React warnings and potential memory leaks.
*   **Why It's Wrong**: It wastes browser resources and can cause cryptic errors, especially in Single Page Applications where users navigate frequently between pages.
*   **How to Fix**: Always provide a cleanup function in your `useEffect` hooks. For API calls, use the AbortController API to cancel fetch requests. For subscriptions and timers, store the returned ID and clear/cancel it in the cleanup function.
    ```javascript
    useEffect(() => {
      const controller = new AbortController();
      fetchData(controller.signal); // Pass the signal
      return () => controller.abort(); // Cleanup on unmount
    }, []);
    ```

### Implementation Guidance

This section provides concrete, actionable code patterns and structure to implement the Frontend UI component.

**A. Technology Recommendations Table**

| Component | Simple Option | Advanced Option |
| :--- | :--- | :--- |
| **Framework** | Create React App (CRA) | Next.js (for hybrid SSR/CSR) |
| **Language** | JavaScript (ES6+) | TypeScript |
| **Styling** | Plain CSS / CSS Modules | Tailwind CSS / Styled Components |
| **Form Handling** | Component state (`useState`) | Formik + Yup / React Hook Form |
| **HTTP Client** | `window.fetch` (with wrapper) | Axios |
| **Markdown Parser** | `marked` (basic) | `remark` + `rehype` (unified ecosystem) |

For this guidance, we will use the **Simple Option** stack (CRA, JavaScript, CSS Modules, `fetch`, `marked`) to lower the initial learning curve.

**B. Recommended File/Module Structure**

Organize the frontend code by feature and concern. The `src/` directory should be structured as follows:

```
src/
├── App.js                    # Main app component, router setup
├── index.js                  # React DOM render entry point
├── styles/
│   └── globals.css           # Global CSS resets and variables
├── components/               # Reusable, presentational components
│   ├── Layout/
│   │   ├── Layout.js
│   │   ├── Header.js
│   │   ├── Footer.js
│   │   └── Layout.module.css
│   ├── Post/
│   │   ├── PostPreviewCard.js
│   │   ├── PostContent.js
│   │   └── PostMeta.js
│   └── UI/
│       ├── Button.js
│       ├── Spinner.js
│       └── ErrorMessage.js
├── pages/                    # Top-level page components (connected to routes)
│   ├── HomePage.js
│   ├── PostPage.js
│   ├── EditorPage.js
│   ├── LoginPage.js
│   └── RegisterPage.js
├── context/                  # React Context providers
│   └── AuthContext.js        # Manages global auth state (user, token)
├── hooks/                    # Custom React hooks
│   ├── useAuth.js           # Convenience hook to consume AuthContext
│   └── useApi.js            # Hook to make authenticated API calls
├── utils/
│   ├── apiClient.js         # Wrapper around fetch for consistent error handling
│   └── markdownRenderer.js  # Function using `marked` to convert markdown to sanitized HTML
└── services/                 # Modules that abstract API interactions
    ├── postService.js       # Functions: getPosts, getPost, createPost, etc.
    └── authService.js       # Functions: login, register, logout
```

**C. Infrastructure Starter Code**

1.  **API Client Wrapper (`utils/apiClient.js`)**: A reusable `fetch` wrapper that handles headers, authentication tokens, and standard error responses.

    ```javascript
    // utils/apiClient.js
    import { getToken, clearToken } from '../context/AuthContext';

    const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

    async function apiClient(endpoint, { method = 'GET', body, headers: customHeaders = {} } = {}) {
      const token = getToken();
      const defaultHeaders = {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      };

      const config = {
        method,
        headers: { ...defaultHeaders, ...customHeaders },
      };

      if (body) {
        config.body = JSON.stringify(body);
      }

      const response = await fetch(`${API_BASE_URL}${endpoint}`, config);

      // Handle HTTP errors (4xx, 5xx)
      if (!response.ok) {
        // If 401 Unauthorized, clear the local token as it's invalid
        if (response.status === 401) {
          clearToken();
          // Optionally redirect to login page
          window.location.href = '/login';
        }

        // Try to parse error message from response body
        let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } catch (e) {
          // Response body is not JSON, use default message
        }

        throw new Error(errorMessage);
      }

      // Handle successful response (204 No Content)
      if (response.status === 204) {
        return null;
      }

      return response.json();
    }

    export default apiClient;
    ```

2.  **Markdown Renderer Utility (`utils/markdownRenderer.js`)**: A function that safely converts markdown to HTML, using a library like `marked` along with a sanitizer like `DOMPurify`.

    ```javascript
    // utils/markdownRenderer.js
    import { marked } from 'marked';
    import DOMPurify from 'dompurify';

    // Configure marked options (optional)
    marked.setOptions({
      breaks: true,        // Convert newlines to <br>
      gfm: true,           // Enable GitHub Flavored Markdown
    });

    /**
     * Converts markdown text to sanitized HTML.
     * @param {string} markdown - The raw markdown text.
     * @returns {string} Sanitized HTML string safe for innerHTML usage.
     */
    export function markdownToHtml(markdown) {
      if (!markdown) return '';
      const rawHtml = marked(markdown);
      const cleanHtml = DOMPurify.sanitize(rawHtml, {
        ALLOWED_TAGS: [
          'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
          'p', 'br', 'hr',
          'strong', 'em', 'code', 'pre', 'blockquote',
          'ul', 'ol', 'li',
          'a', 'img',
          'table', 'thead', 'tbody', 'tr', 'th', 'td',
        ],
        ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'class'],
      });
      return cleanHtml;
    }
    ```

**D. Core Logic Skeleton Code**

1.  **Authentication Context (`context/AuthContext.js`)**: The global state manager for user authentication.

    ```javascript
    // context/AuthContext.js
    import React, { createContext, useState, useContext, useEffect } from 'react';
    import { jwtDecode } from 'jwt-decode'; // A small library to decode JWTs

    const AuthContext = createContext(null);

    export const useAuth = () => {
      const context = useContext(AuthContext);
      if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
      }
      return context;
    };

    export const AuthProvider = ({ children }) => {
      const [user, setUser] = useState(null);
      const [isLoading, setIsLoading] = useState(true);

      useEffect(() => {
        // TODO 1: On initial load, check localStorage for an existing JWT token.
        // TODO 2: If a token exists, decode it using `jwtDecode` to extract user info.
        // TODO 3: Validate the token's expiration date. If expired, clear it.
        // TODO 4: If valid, set the user state with the decoded information.
        // TODO 5: Set `isLoading` to false once the check is complete.
        // Hint: Store the token in localStorage with a key like 'blog_token'.
      }, []);

      const login = async (email, password) => {
        // TODO 1: Call the `/api/auth/login` endpoint with email and password.
        // TODO 2: On success, receive a JWT token from the response.
        // TODO 3: Store the received token in localStorage.
        // TODO 4: Decode the token and update the `user` state.
        // TODO 5: Return the user data or success status.
      };

      const register = async (name, email, password) => {
        // TODO 1: Call the `/api/auth/register` endpoint with user details.
        // TODO 2: On success, you may either automatically log the user in (by calling login)
        //         or redirect them to the login page.
      };

      const logout = () => {
        // TODO 1: Remove the JWT token from localStorage.
        // TODO 2: Clear the `user` state (set to null).
        // TODO 3: Optionally, redirect the user to the homepage.
      };

      const value = {
        user,
        isLoading,
        login,
        register,
        logout,
        isAuthenticated: !!user,
      };

      return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
    };

    // Helper functions for the apiClient
    export const getToken = () => localStorage.getItem('blog_token');
    export const clearToken = () => localStorage.removeItem('blog_token');
    ```

2.  **Post Editor Component (`pages/EditorPage.js`)**: The core of the authoring experience.

    ```javascript
    // pages/EditorPage.js
    import React, { useState, useEffect } from 'react';
    import { useParams, useNavigate } from 'react-router-dom';
    import { useAuth } from '../context/AuthContext';
    import apiClient from '../utils/apiClient';
    import { markdownToHtml } from '../utils/markdownRenderer';
    import styles from './EditorPage.module.css';

    function EditorPage() {
      const { id } = useParams(); // For editing an existing post, id will be present
      const navigate = useNavigate();
      const { user, isAuthenticated } = useAuth();
      const [isLoading, setIsLoading] = useState(!!id); // Load data if editing
      const [error, setError] = useState('');

      // Form state
      const [title, setTitle] = useState('');
      const [content, setContent] = useState('');
      const [tags, setTags] = useState([]);
      const [previewHtml, setPreviewHtml] = useState('');

      useEffect(() => {
        // TODO 1: If this is an edit page (id exists) and user is authenticated, fetch the existing post data.
        // TODO 2: Use `apiClient` to GET `/api/posts/${id}`.
        // TODO 3: Populate the form state (title, content, tags) with the fetched data.
        // TODO 4: Handle errors (e.g., post not found, user not authorized) and set error state.
        // TODO 5: Set `isLoading` to false after fetch completes.
      }, [id, isAuthenticated]);

      useEffect(() => {
        // Update HTML preview whenever markdown content changes
        setPreviewHtml(markdownToHtml(content));
      }, [content]);

      const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        const postData = { title, content, tags };

        try {
          // TODO 1: Determine if this is a CREATE or UPDATE operation based on `id`.
          // TODO 2: Make the appropriate API call:
          //         - CREATE: POST `/api/posts` with `postData`
          //         - UPDATE: PUT `/api/posts/${id}` with `postData`
          // TODO 3: On successful response, redirect the user to the new/updated post's page.
          // TODO 4: Handle API errors (e.g., validation errors, 403 Forbidden) and display them to the user.
        } catch (err) {
          setError(err.message || 'Failed to save the post.');
        } finally {
          setIsLoading(false);
        }
      };

      // TODO: Add a guard clause to redirect unauthenticated users to login page.

      return (
        <div className={styles.container}>
          <h1>{id ? 'Edit Post' : 'Create New Post'}</h1>
          {error && <div className={styles.error}>{error}</div>}
          <form onSubmit={handleSubmit}>
            <div className={styles.formGroup}>
              <label htmlFor="title">Title</label>
              <input
                id="title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                required
                disabled={isLoading}
              />
            </div>
            <div className={styles.editorLayout}>
              <div className={styles.editorPane}>
                <label htmlFor="content">Content (Markdown)</label>
                <textarea
                  id="content"
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  rows={20}
                  required
                  disabled={isLoading}
                />
              </div>
              <div className={styles.previewPane}>
                <label>Live Preview</label>
                <div
                  className={styles.previewContent}
                  dangerouslySetInnerHTML={{ __html: previewHtml }}
                />
              </div>
            </div>
            <div className={styles.formGroup}>
              <label htmlFor="tags">Tags (comma-separated)</label>
              <input
                id="tags"
                type="text"
                value={tags.join(', ')}
                onChange={(e) => setTags(e.target.value.split(',').map(tag => tag.trim()).filter(Boolean))}
                placeholder="javascript, webdev, tutorial"
                disabled={isLoading}
              />
            </div>
            <button type="submit" disabled={isLoading}>
              {isLoading ? 'Saving...' : (id ? 'Update Post' : 'Publish Post')}
            </button>
          </form>
        </div>
      );
    }

    export default EditorPage;
    ```

**E. Language-Specific Hints (JavaScript/React)**

*   **State Management**: Use `useState` for simple local state and `useReducer` for more complex state logic (e.g., form with many interrelated fields). For global auth, the Context API is sufficient.
*   **Side Effects**: Always pair `useEffect` with a cleanup function to prevent memory leaks, especially for subscriptions, WebSocket connections, or any asynchronous task that could resolve after component unmounting.
*   **Styling**: CSS Modules (enabled by default in Create React App) provide locally scoped class names, preventing style conflicts. Import the generated object (e.g., `import styles from './Component.module.css'`) and use `className={styles.myClass}`.
*   **Routing**: Use `react-router-dom` for client-side routing. Wrap your app in a `<BrowserRouter>` and define routes with `<Routes>` and `<Route>` components.
*   **Environment Variables**: Prefix environment variables with `REACT_APP_` (e.g., `REACT_APP_API_URL`) in your `.env` file to make them accessible in your React app via `process.env.REACT_APP_API_URL`.

**F. Milestone Checkpoint**

After implementing the Frontend UI milestone, you should be able to:
1.  **Run the Application**: Start your frontend dev server (e.g., `npm start`) and your backend API server.
2.  **Verify Public Pages**: Navigate to the homepage (`/`). You should see a list of blog post previews. Clicking on a post title should take you to a detailed view (`/posts/:id`) where the markdown content is rendered as HTML.
3.  **Test Authentication**:
    *   Navigate to `/login` and `/register`. Submit the forms with valid credentials. Upon successful login, the UI should update (e.g., the header should change from "Login" to "Logout").
    *   Try to access `/editor` while logged out. You should be redirected to the login page.
4.  **Test Authoring Flow**:
    *   While logged in, navigate to `/editor`. Type markdown in the left pane and confirm a live HTML preview appears in the right pane.
    *   Fill in the title and content, then click "Publish Post". You should be redirected to the newly created post's page.
    *   On that post's page, if you are the author, you should see an "Edit" button. Clicking it should take you back to the editor with the post's data pre-filled.
5.  **Check Responsiveness**: Use your browser's Developer Tools to toggle device emulation (e.g., iPhone 12, iPad). The layout should adapt cleanly at all screen sizes without horizontal scrolling.

**Signs of a Problem and What to Check:**
*   **Blank White Screen**: Check the browser's JavaScript console for errors. Likely an uncaught error in a component's render or useEffect. Look for undefined variables or failed API calls.
*   **Styles Not Applying**: Verify your CSS Module import path and that you are using `className={styles.myClass}` and not `class="myClass"`. Check for typos in class names.
*   **API Calls Failing with CORS Errors**: Ensure your backend API server is configured to send the correct CORS headers (`Access-Control-Allow-Origin`) and that the `REACT_APP_API_URL` environment variable points to the correct backend address.
*   **State Not Updating After Login**: Ensure the `AuthProvider` is wrapping your entire app in `App.js`. Verify that the `login` function correctly updates the context state and that consuming components are using the `useAuth` hook correctly.


## Interactions and Data Flow

> **Milestone(s):** Milestone 2 (User Authentication), Milestone 3 (Blog CRUD Operations), Milestone 4 (Frontend UI)

This section maps the theoretical architecture to concrete runtime behavior. It answers the crucial question: "When a user clicks a button, what actually happens?" We'll trace the journey of data through the system's veins—from browser clicks to database writes and back again—revealing how components collaborate to deliver functionality.

### Sequence: User Registration Flow

**Mental Model: The Club Membership Desk**

Imagine the registration process as applying for a membership card at an exclusive club. You fill out an application form (registration page), present identification (email and password), and a clerk at the desk (the API server) verifies your details against the club's existing member roster (database). If everything checks out, they issue you a unique membership card (JWT) and a welcome packet (user record). This card is your key to entering the club (protected routes) and accessing member-only areas (authorized operations).

The flow begins when a prospective user navigates to the registration form. The frontend presents a simple form with email and password fields, with client-side validation for format requirements. Upon submission, the following orchestrated sequence unfolds:

![Authentication Sequence Diagram](./diagrams/auth-sequence.svg)

1.  **Form Submission & Validation:** The user fills in the `email` and `password` fields and clicks "Register". The React component's `onSubmit` handler first runs client-side validation (e.g., checking for a valid email format and password length). It then packages the data into a JSON payload and calls `apiClient('/api/auth/register', { method: 'POST', body: { email, password } })`.

2.  **API Request & Input Sanitization:** The HTTP request arrives at the backend's registration route (`POST /api/auth/register`). The server's middleware stack processes it: body-parsing middleware extracts the JSON, then a custom validation middleware checks the payload against stricter rules. It ensures the email is unique by querying the database, and validates the password against security policies. Invalid requests are immediately rejected with a `400 Bad Request` and specific error messages.

3.  **Password Hashing & User Creation:** If validation passes, the route handler calls the `UserModel.create` function. This function uses `bcrypt` with a work factor of 12 to hash the plain-text password, transforming `"myPassword123"` into a string like `"$2b$12$Sds...J9a"`. The handler then executes a **parameterized query** to insert a new record into the `users` table with the email and the password hash.

4.  **Token Generation & Response:** Upon successful database insertion, the server generates a **JWT**. The token's payload contains the user's `id` and `email`. It is signed with a secret key and set to expire in 15 minutes. The server sends a `201 Created` response back to the client. The response body includes the user's `id` and `email`, and the JWT is included either in the JSON body (for the client to store) or, more securely, as an `HttpOnly` cookie in the response headers.

5.  **Client-State Update & Redirect:** The frontend's `apiClient` receives the successful response. It extracts the token (if not in a cookie) and stores it in memory or a secure context. The global `AuthState` is updated via the `useAuth()` hook, setting `user` to the returned user object and `isLoading` to `false`. The application then programmatically navigates the user to the homepage or dashboard, completing the registration flow.

**Error Handling & Edge Cases:**

*   **Duplicate Email:** The validation step includes a database check for existing email. If found, the server returns `409 Conflict` with a message "Email already registered." The frontend displays this error near the email field.
*   **Weak Password:** Server validation rejects passwords below a minimum length or lacking complexity. Returns `400 Bad Request` with a message detailing requirements.
*   **Network Failure:** The `apiClient` wrapper includes timeout and network error handling. It displays a generic "Network error, please try again" message.
*   **Database Unavailable:** The `PoolConfig` connection timeout (`DB_POOL_CONN_TIMEOUT`) triggers after 2 seconds. The server catches this error, logs it, and returns a `503 Service Unavailable` response.

### Sequence: Post Creation Flow

**Mental Model: The Editorial Desk Submission**

Think of creating a blog post like submitting an article to a newspaper's editorial desk. The author (authenticated user) prepares their manuscript (markdown content) and submits it to the editor (API server). The editor first checks the author's credentials (validates JWT), then reviews the submission for basic guidelines (input validation). Once approved, the editor files the manuscript in the newspaper's archive (database) under the author's name, and sends back a confirmation slip with a reference number (post ID). The published article later appears in the next edition (post listing page).

This flow is triggered when an authenticated author clicks "Publish" in the post editor. The frontend has already been maintaining local draft state in a `FormState` object. The sequence diagram below illustrates the journey:

![Post Creation Sequence Diagram](./diagrams/post-creation-sequence.svg)

1.  **Editor Submission & Token Attachment:** In the post editor component, the author clicks "Publish". The component gathers the `title`, `content` (raw markdown), and `tags` from the `FormState`. It then calls `apiClient('/api/posts', { method: 'POST', body: { title, content, tags } })`. Crucially, the `apiClient` function automatically attaches the JWT from the `AuthState` to the request's `Authorization` header as `Bearer <token>`.

2.  **Authentication Gateway:** The request hits the `POST /api/posts` endpoint, which is protected by authentication middleware. This middleware extracts the JWT from the header, verifies its signature and expiration using the server's secret key, and decodes the payload to obtain the `author_id`. If the token is invalid or missing, the middleware short-circuits the request with a `401 Unauthorized` response.

3.  **Authorization & Input Validation:** The request proceeds to the route handler, which now has access to the `req.user` object (added by the auth middleware). The handler validates the incoming `title` (non-empty, length limit) and `content` (non-empty). It may also sanitize the markdown content to remove potentially malicious scripts at this stage.

4.  **Database Transaction & Post Creation:** The handler calls `PostModel.create(title, content, authorId, tags)`. Internally, this function:
    1. Acquires a `PoolClient` from the connection pool using `getClient()` to ensure atomicity.
    2. Begins a database transaction.
    3. Executes a **parameterized query** to insert a new row into the `posts` table, setting `author_id` to the authenticated user's ID.
    4. If tags are provided, it may insert them into a `tags` table and create relationships in a `post_tags` junction table within the same transaction.
    5. Commits the transaction. On success, the newly created `PostObject` (including its auto-generated `id` and `created_at` timestamp) is returned.

5.  **Response with Enriched Data:** The route handler optionally enriches the raw `PostObject` for the response. It might call `markdownToHtml` to generate an `html_content` preview, or join with the users table to include the author's name. It then sends a `201 Created` response with the enriched `PostObjectWithAuthor` in the body.

6.  **UI Update & Navigation:** The frontend receives the response. It can update its global state cache (if using one) with the new post, display a success notification ("Post published!"), and redirect the author to the new post's detail page (`/posts/{id}`).

**Error Handling & Edge Cases:**

*   **Authentication Failure:** Invalid/missing token yields `401`. The frontend's `apiClient` intercepts this, clears the local `AuthState` (logging the user out), and redirects to the login page.
*   **Validation Failure:** Invalid title/content yields `400`. Errors are displayed inline on the form.
*   **Database Constraint Violation:** A rare duplicate `id` or foreign key error would cause the transaction to roll back and the server to return `500 Internal Server Error`. The frontend shows a generic error message and allows the user to retry.
*   **Network Timeout during Submission:** The `apiClient` can use an `AbortController` to cancel the request after a timeout (e.g., 10s). The UI should show a loading state and, on timeout, allow the user to retry without losing their drafted content (maintained in `FormState`).

### API Message Formats

Clear, consistent API contracts are the language components use to communicate. The following tables define the request and response structures for key endpoints. All requests and responses use JSON format with `Content-Type: application/json`.

**Authentication Endpoints**

| Endpoint | Method | Request Body (JSON) | Success Response (JSON) | Error Responses |
| :--- | :--- | :--- | :--- | :--- |
| `/api/auth/register` | POST | `{ "email": "string", "password": "string" }` | `201 Created`<br>`{ "user": { "id": "number", "email": "string", "name": "string" }, "token": "string" }` | `400` (Validation), `409` (Email exists), `500` (Server error) |
| `/api/auth/login` | POST | `{ "email": "string", "password": "string" }` | `200 OK`<br>`{ "user": { "id": "number", "email": "string", "name": "string" }, "token": "string" }` | `401` (Invalid credentials), `400` (Validation) |
| `/api/auth/me` | GET | *None (uses Auth header)* | `200 OK`<br>`{ "id": "number", "email": "string", "name": "string" }` | `401` (Invalid/missing token) |

**Detailed Field Descriptions for `/api/auth/register` & `/api/auth/login`:**

| Field | Type | Required | Description | Constraints |
| :--- | :--- | :--- | :--- | :--- |
| `email` | string | Yes | User's email address. | Valid email format, max 255 chars, unique. |
| `password` | string | Yes | User's chosen password. | Min 8 chars, must contain letter and number. |
| `user.id` | number/integer | Yes (response) | Unique system identifier for the user. | Auto-incremented. |
| `user.name` | string | No (response) | User's display name. Initially null or derived from email. | Max 100 chars. |
| `token` | string | Yes (response) | **JWT** for authenticating subsequent requests. | Signed, contains `sub` (user id) and `exp` (expiry). |

**Blog Post Endpoints**

| Endpoint | Method | Request Body (JSON) | Success Response (JSON) | Error Responses |
| :--- | :--- | :--- | :--- | :--- |
| `/api/posts` | POST | `{ "title": "string", "content": "string", "tags": ["string"] }` | `201 Created`<br>`PostObjectWithAuthor` | `401` (Unauth), `400` (Validation), `500` |
| `/api/posts` | GET | *None (query params: ?page=1&limit=10&tag=javascript)* | `200 OK`<br>`{ "posts": [PostObjectWithAuthor], "pagination": { "currentPage": 1, "totalPages": 5, "totalItems": 48, "pageSize": 10 } }` | `500` |
| `/api/posts/:id` | GET | *None* | `200 OK`<br>`PostObjectWithAuthor` | `404` (Not found) |
| `/api/posts/:id` | PUT | `{ "title": "string", "content": "string" }` | `200 OK`<br>`PostObject` | `401`, `403` (Not author), `400`, `404` |
| `/api/posts/:id` | DELETE | *None* | `204 No Content` (empty body) | `401`, `403`, `404`, `500` |

**Detailed Field Descriptions for `PostObject` and `PostObjectWithAuthor`:**

| Field | Type | Description | Notes |
| :--- | :--- | :--- | :--- |
| `id` | integer | Unique post identifier. | Auto-incremented primary key. |
| `title` | string | Post title. | Stored as plain text, max 200 chars. |
| `content` | string | Post body in raw markdown format. | Stored as text, no size limit (practical). |
| `author_id` | integer | Foreign key referencing `users.id`. | Links post to its author. |
| `created_at` | string (ISO 8601) | Timestamp of creation. | Set by database on insert. |
| `updated_at` | string (ISO 8601) | Timestamp of last update. | Updated on `PUT` requests. |
| `tags` | Array\<string> | Array of tag labels associated with the post. | Optional, derived from `post_tags` join. |
| `author` | Object | Nested author object. | Included in `PostObjectWithAuthor`. |
| `author.id` | integer | Author's user ID. | Same as `author_id`. |
| `author.name` | string | Author's display name. | Fetched from `users` table via **JOIN**. |
| `author.email` | string | Author's email. | Only included for the post's author on fetch. |
| `html_content` | string | **Optional** HTML rendered from markdown. | Generated on-demand via `markdownToHtml`. |

**Pagination Metadata Object:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `currentPage` | number | The page number of the current result set (1-indexed). |
| `totalPages` | number | Total number of pages available given the `pageSize`. |
| `totalItems` | number | Total number of items across all pages. |
| `pageSize` | number | Number of items returned per page (from `limit` query param). |

**Example Concrete Walk-Through: Creating a Post**

Let's trace a real example with concrete data. Author Jane Doe (user id `5`) writes a post titled "My First Blog Post" with some markdown content.

1.  **Request from Frontend:**
    ```http
    POST /api/posts HTTP/1.1
    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    Content-Type: application/json

    {
      "title": "My First Blog Post",
      "content": "# Welcome\n\nThis is my *first* post using **markdown**!",
      "tags": ["hello", "introduction"]
    }
    ```

2.  **Successful Response from Server:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "id": 42,
      "title": "My First Blog Post",
      "content": "# Welcome\n\nThis is my *first* post using **markdown**!",
      "author_id": 5,
      "created_at": "2023-10-27T14:30:00.000Z",
      "updated_at": "2023-10-27T14:30:00.000Z",
      "tags": ["hello", "introduction"],
      "author": {
        "id": 5,
        "name": "Jane Doe",
        "email": "jane@example.com"
      },
      "html_content": "<h1>Welcome</h1>\n<p>This is my <em>first</em> post using <strong>markdown</strong>!</p>"
    }
    ```

This concrete exchange demonstrates the complete data flow: structured data leaves the frontend, is validated and processed by the backend, persists in the database, and returns enriched information to update the user interface.

### Implementation Guidance

**A. Technology Recommendations Table**

| Component | Simple Option | Advanced Option |
| :--- | :--- | :--- |
| **API Contract Definition** | Manual documentation (as above) | OpenAPI/Swagger specification with auto-generated docs (Swagger UI) |
| **Request/Response Serialization** | Built-in `JSON.parse()` and `JSON.stringify()` | Validation libraries like `Joi` or `Zod` for runtime type safety |
| **API Client** | Custom `apiClient` wrapper around `fetch` | Generated client from OpenAPI spec using `openapi-typescript-codegen` |
| **Flow Visualization** | Sequence diagrams in design doc | Live tracing with distributed tracing (Jaeger, OpenTelemetry) |

**B. Recommended File/Module Structure**

Organize API-related code by domain and concern. The `routes/` directory maps directly to the API surface.

```
blog-platform/
  backend/
    src/
      index.js                    # App entry point
      config/                     # Configuration
      middleware/                 # Express middleware
        auth.js                   # JWT verification
        validate.js               # Request validation
        errorHandler.js           # Centralized error handling
      routes/                     # API route definitions
        auth.routes.js            # POST /register, /login, /logout
        posts.routes.js           # CRUD for /posts
        comments.routes.js        # (Future) comments routes
      controllers/                # Route handlers (business logic)
        auth.controller.js
        posts.controller.js
      models/                     # Data access layer
        User.model.js
        Post.model.js
        Comment.model.js
      utils/                      # Shared utilities
        apiClient.js              # Frontend fetch wrapper
        pagination.js             # calculatePaginationMetadata
        markdown.js               # markdownToHtml
      db/                         # Database connection & migrations
        pool.js                   # Exports query, getClient
        migrations/               # Migration files
  frontend/
    src/
      api/                        # Frontend API integration
        client.js                 # apiClient implementation
        endpoints.js              # Endpoint URL constants
      components/
        PostForm/                 # Reusable form for create/edit
          index.js
          styles.module.css
      contexts/
        AuthContext.js            # React Context for useAuth()
```

**C. Infrastructure Starter Code: API Client & Error Handler**

`frontend/src/api/client.js` - A robust, reusable fetch wrapper:
```javascript
import { useAuth } from '../../contexts/AuthContext';

const API_URL = process.env.REACT_APP_API_URL;

// Create a custom error class for API errors
class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

export const apiClient = async (endpoint, { method = 'GET', body, headers = {}, ...customConfig } = {}) => {
  // Get token from auth context or localStorage (simplified)
  const token = localStorage.getItem('token');
  
  const config = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(token && { Authorization: `Bearer ${token}` }),
      ...headers,
    },
    ...customConfig,
  };

  if (body) {
    config.body = JSON.stringify(body);
  }

  // Add AbortController for timeouts (optional)
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout
  config.signal = controller.signal;

  try {
    const response = await fetch(`${API_URL}${endpoint}`, config);
    clearTimeout(timeoutId);

    const data = await response.json().catch(() => null); // Handle empty responses

    if (!response.ok) {
      // Automatically logout on 401 Unauthorized
      if (response.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login'; // Force page reload to clear state
      }
      throw new ApiError(data?.message || `Request failed with status ${response.status}`, response.status, data);
    }

    return data;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new ApiError('Request timeout. Please try again.', 408);
    }
    // Re-throw ApiError instances, wrap network errors
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError('Network error. Please check your connection.', 0, { originalError: error.message });
  }
};
```

`backend/src/middleware/errorHandler.js` - Centralized Express error middleware:
```javascript
const errorHandler = (err, req, res, next) => {
  console.error('Error:', err.message, err.stack);

  // Default error
  let statusCode = err.statusCode || 500;
  let message = err.message || 'Internal Server Error';
  let details = null;

  // Handle specific error types
  if (err.name === 'ValidationError') {
    statusCode = 400;
    message = 'Validation failed';
    details = err.details; // Assuming Joi/Zod details
  } else if (err.name === 'JsonWebTokenError') {
    statusCode = 401;
    message = 'Invalid token';
  } else if (err.name === 'TokenExpiredError') {
    statusCode = 401;
    message = 'Token expired';
  }

  // In production, don't leak stack traces
  const response = {
    error: message,
    ...(process.env.NODE_ENV !== 'production' && { stack: err.stack }),
    ...(details && { details }),
  };

  res.status(statusCode).json(response);
};

export default errorHandler;
```

**D. Core Logic Skeleton Code**

`backend/src/controllers/posts.controller.js` - Post creation handler:
```javascript
import * as PostModel from '../models/Post.model.js';
import { markdownToHtml } from '../utils/markdown.js';

/**
 * Creates a new blog post.
 * @route POST /api/posts
 * @access Private
 */
export const createPost = async (req, res, next) => {
  // TODO 1: Extract title, content, and optional tags from req.body
  // TODO 2: Validate input: ensure title and content are non-empty strings
  // TODO 3: Get the authenticated user's ID from req.user (set by auth middleware)
  // TODO 4: Call PostModel.create(title, content, authorId, tags) to insert into database
  // TODO 5: On success, optionally generate html_content using markdownToHtml
  // TODO 6: Construct a PostObjectWithAuthor response (join author name from users table)
  // TODO 7: Return 201 Created with the enriched post object
  // TODO 8: Catch any errors (e.g., database errors) and pass to next() for error middleware
};

/**
 * Gets a paginated list of posts.
 * @route GET /api/posts
 * @access Public
 */
export const getPosts = async (req, res, next) => {
  // TODO 1: Extract page, limit, and optional tag filter from req.query
  // TODO 2: Set default values: page = 1, limit = 10
  // TODO 3: Call PostModel.findPaginated(page, limit, tag) to fetch posts and total count
  // TODO 4: Use calculatePaginationMetadata(total, page, limit) to build pagination info
  // TODO 5: For each post in the result, optionally generate html_content (excerpt)
  // TODO 6: Return 200 OK with { posts: [...], pagination: {...} }
};
```

**E. Language-Specific Hints (JavaScript/Node.js)**

*   **Parameterized Queries:** Always use the `pg` library's parameterized query syntax (`query('INSERT INTO posts (title) VALUES ($1)', [title])`) to prevent **SQL injection**.
*   **JWT Handling:** Use the `jsonwebtoken` library. Sign tokens with a strong secret (stored in `process.env.JWT_SECRET`) and set a short expiry (e.g., `'15m'`). Implement refresh tokens for a better user experience.
*   **Async/Await Error Handling:** Wrap async route handlers in a try/catch block, or use an express-async-errors middleware to avoid unhandled promise rejections.
*   **CORS:** For development, use the `cors` middleware and configure it to accept requests from your frontend origin (e.g., `http://localhost:3000`). In production, restrict it to your actual domain.

**F. Milestone Checkpoint: Verifying the Post Creation Flow**

After implementing Milestone 3 (Blog CRUD Operations), verify the post creation sequence works end-to-end.

1.  **Start the Backend Server:**
    ```bash
    cd backend
    npm run dev
    ```
    Check console for "Server running on port 5000" and "Database connected".

2.  **Start the Frontend Development Server:**
    ```bash
    cd frontend
    npm start
    ```

3.  **Manual Test via Browser:**
    *   Navigate to `http://localhost:3000/login` and log in with a test user.
    *   Click "New Post" to open the editor.
    *   Enter a title and markdown content, then click "Publish".
    *   **Expected Result:** You are instantly redirected to the new post's page, displaying the rendered HTML. The browser's Network tab should show a `POST /api/posts` request with a `201` response containing the full post JSON.

4.  **Verify Database State:**
    Connect to your database (e.g., `psql blog_platform`) and run:
    ```sql
    SELECT id, title, author_id FROM posts ORDER BY created_at DESC LIMIT 1;
    ```
    The returned row should match your test post's data.

**Signs of Trouble:**
*   **`401 Unauthorized` on POST:** The JWT is missing/invalid. Check that the `apiClient` is attaching the `Authorization` header correctly.
*   **`500 Internal Server Error`:** Check the backend console logs. Likely a database constraint error or uncaught exception in the model layer.
*   **Post appears but author name is missing:** The `PostModel.findPaginated` or `PostModel.findById` is not performing the **JOIN** with the `users` table to fetch the author's name.

**G. Debugging Tips**

| Symptom | Likely Cause | How to Diagnose | Fix |
| :--- | :--- | :--- | :--- |
| Registration succeeds but login fails with "Invalid credentials". | Password hash mismatch. The hash stored during registration might be incorrect. | Compare the stored `password_hash` in the database for the test user with a known-good hash of the password. | Ensure you are using `bcrypt.hash` with the correct work factor and are storing the *result* of the hash function, not the plain text. |
| Post creation returns `201` but the post list (`GET /posts`) is empty or shows old data. | **N+1 query problem** or missing **eager loading**. The list query fetches posts but not authors in a single query. | Check the SQL logs. You should see one query for posts with a `JOIN` on users. If you see many separate `SELECT * FROM users WHERE id = ...` queries, that's N+1. | Modify `PostModel.findPaginated` to use a SQL `JOIN` or your ORM's eager loading method (`.include()`). |
| `PUT /api/posts/42` returns `403 Forbidden` even when the logged-in user is the author. | The `PostModel.isAuthor` check is failing or the `req.user.id` is not being compared correctly to `post.author_id`. | Log `req.user.id` and the `post.author_id` fetched from the database in the update handler. Ensure they are the same type (both numbers). | Implement a proper ownership check in a middleware or at the start of the update controller, using `PostModel.isAuthor(postId, userId)`. |
| Frontend shows a generic "Network error" when the API is down. | The `apiClient` wrapper is not distinguishing between network errors and API errors. | Open browser DevTools > Network tab. Check if the request fails with `ERR_CONNECTION_REFUSED` (network) or returns an HTTP error status like `500`. | Improve error handling in `apiClient` to catch `TypeError` (network) separately from `Response.ok` false (API error). Provide specific user messages. |


## Error Handling and Edge Cases

> **Milestone(s):** Milestone 2 (User Authentication), Milestone 3 (Blog CRUD Operations), Milestone 4 (Frontend UI)

A robust blog platform must anticipate and gracefully handle failures, invalid inputs, and unexpected conditions. This section outlines our systematic approach to error management—from categorizing failures by their root cause to implementing layered validation and designing recovery mechanisms for edge cases. Think of this as the platform's **immune system**: it detects anomalies, contains damage, and facilitates recovery while maintaining clear communication with users about what went wrong and how to proceed.

### Error Categories and HTTP Status Codes

**Mental Model: The Traffic Light System**  
Imagine error handling as a traffic control system for web requests. **Green lights (2xx codes)** indicate smooth passage—the request reached its destination and was processed successfully. **Yellow lights (4xx codes)** signal issues with the driver's actions (invalid credentials, wrong turns, exceeding speed limits)—the driver needs to adjust their behavior. **Red lights (5xx codes)** indicate road infrastructure failures (broken bridges, power outages)—the driver must wait while maintenance crews fix the underlying problem. This mental model helps developers understand which party (client vs. server) is responsible for fixing each error type.

Our API classifies errors into two fundamental categories:

**Client Errors (4xx)**: These occur when the client sends an invalid, malformed, or unauthorized request. The client has the information needed to correct the error and retry. We use specific 4xx codes to indicate the exact nature of the problem:

| HTTP Status | Scenario | Example | Recovery Action |
|-------------|----------|---------|-----------------|
| **400 Bad Request** | General client error when request fails validation | Missing required fields, invalid JSON syntax, malformed email format | Client must fix the request structure before retrying |
| **401 Unauthorized** | Authentication credentials are missing, invalid, or expired | Missing `Authorization` header, invalid JWT, expired session token | Client must obtain valid credentials via login |
| **403 Forbidden** | Authenticated user lacks permission for the requested operation | User tries to edit another user's post, non-admin accessing admin area | Client should not retry without elevated privileges |
| **404 Not Found** | Requested resource does not exist at the specified URL | Accessing `/posts/99999` when post doesn't exist, invalid API endpoint | Client should verify the resource identifier exists |
| **409 Conflict** | Request conflicts with current state of the server | Registering with an email already in use, concurrent edit collisions | Client must resolve the conflict before retrying |
| **422 Unprocessable Entity** | Request syntax is valid but semantic validation fails | Post title exceeds 200 characters, password too weak, invalid tag format | Client must correct the semantic content |

**Server Errors (5xx)**: These indicate failures on the server side—the server encountered an unexpected condition that prevented it from fulfilling the request. The client cannot fix these errors directly but can retry later:

| HTTP Status | Scenario | Example | Recovery Strategy |
|-------------|----------|---------|-------------------|
| **500 Internal Server Error** | Generic server error when no more specific code applies | Uncaught exception, database constraint violation not handled | Server logs the error for investigation; client can retry with exponential backoff |
| **502 Bad Gateway** | Server acting as gateway/proxy received invalid response | Nginx proxy cannot reach the backend Node.js application | Check backend service health and network connectivity |
| **503 Service Unavailable** | Server temporarily unable to handle requests | Database connection pool exhausted, maintenance mode active | Client should retry after `Retry-After` header indicates |
| **504 Gateway Timeout** | Server acting as gateway did not receive timely response | Database query timeout, external API call taking too long | Investigate slow dependencies and optimize queries |

The error handling flowchart ![API Error Handling Flowchart](./diagrams/error-handling-flowchart.svg) illustrates how middleware processes requests through validation, authentication, authorization, and business logic layers, with error detection at each stage triggering appropriate HTTP responses.

> **Design Principle: Be Specific but Not Leaky**  
> Provide enough error detail for the client to understand what went wrong, but avoid leaking implementation details that could aid attackers. For example, return "Invalid credentials" (not "Password doesn't match hash"), but for validation errors, specify which field failed and why (e.g., "email: must be a valid email address").

### Input Validation Strategy

**Mental Model: The Airport Security Checkpoints**  
Input validation functions like multi-layered airport security. **Client-side validation** is the self-service check-in kiosk—fast, convenient, and catches obvious issues early (missing fields, format errors) but can be bypassed. **Server-side validation** is the TSA checkpoint—thorough, mandatory, and security-focused, examining everything that actually boards the plane (server). **Database constraints** are the final gate agent verification—a last line of defense ensuring only valid data gets stored. Each layer serves a distinct purpose: client-side for user experience, server-side for security and data integrity, database for ultimate consistency.

We implement validation at three distinct layers:

**1. Client-Side Validation (UX-Focused)**
- **Purpose**: Provide immediate feedback to users, reduce unnecessary server requests
- **Techniques**: HTML5 form validation attributes, JavaScript validation before submission
- **Scope**: Format validation, required fields, length constraints, basic pattern matching
- **Limitation**: Can be bypassed; never trust client-side validation for security

**2. Server-Side Validation (Security-Focused)**
- **Purpose**: Ensure data integrity, prevent malicious input, enforce business rules
- **Techniques**: Validation middleware, schema validation libraries, custom validation logic
- **Scope**: All inputs regardless of source, business logic validation, authorization checks
- **Implementation**: Centralized validation middleware that processes requests before reaching route handlers

**3. Database Validation (Consistency-Focused)**
- **Purpose**: Final data integrity guarantee, enforce referential integrity
- **Techniques**: SQL constraints (NOT NULL, UNIQUE, FOREIGN KEY), data type validation
- **Scope**: Data type correctness, relational integrity, uniqueness constraints
- **Role**: Safety net that prevents invalid data persistence even if application logic has bugs

**Validation Rules for Key Inputs:**

| Input Field | Validation Rules | Client-Side | Server-Side | Database Constraint |
|-------------|-----------------|-------------|-------------|---------------------|
| **User Email** | 1. Required<br>2. Valid email format<br>3. Maximum 255 characters<br>4. Unique across system | HTML5 `type="email"`, maxlength | Regex pattern, length check, uniqueness query | VARCHAR(255), UNIQUE constraint |
| **User Password** | 1. Required<br>2. Minimum 8 characters<br>3. Maximum 72 characters (bcrypt limit)<br>4. Contains letter and number | Length check, pattern matching | Length bounds, strength evaluation (optional) | VARCHAR(60) for bcrypt hash |
| **Post Title** | 1. Required<br>2. 1-200 characters<br>3. No HTML/script tags | Maxlength attribute, required attribute | Length check, HTML sanitization | VARCHAR(200), NOT NULL |
| **Post Content** | 1. Required<br>2. Maximum 50,000 characters<br>3. Markdown syntax validation (optional) | Maxlength attribute | Length check, markdown parsing attempt | TEXT, NOT NULL |
| **Post Tags** | 1. Optional<br>2. Array of strings<br>3. Each tag 1-50 characters, alphanumeric with hyphens | JavaScript array validation | Array validation, each element format check | JSONB or separate table with constraints |

> **Decision: Centralized Validation Middleware**
> - **Context**: Need consistent validation across all API endpoints with clear error messages
> - **Options Considered**:
>   1. **Inline validation in each route handler**: Duplicate code, inconsistent error formats
>   2. **Validation library per model**: Tighter coupling with data models
>   3. **Centralized middleware with validation schemas**: Single source of truth, consistent error format
> - **Decision**: Implement centralized validation middleware using JSON schema or similar
> - **Rationale**: Ensures consistent validation logic and error response format across all endpoints; separates validation concerns from business logic; makes validation rules easily testable and maintainable
> - **Consequences**: Adds dependency on validation library; requires learning schema definition; provides excellent error message consistency

**Error Response Format for Validation Failures:**
When validation fails, the API returns a consistent error structure that clients can parse programmatically:

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Input validation failed",
    "details": [
      {
        "field": "email",
        "message": "must be a valid email address"
      },
      {
        "field": "password", 
        "message": "must be at least 8 characters"
      }
    ]
  }
}
```

This format allows frontend forms to highlight specific fields with error messages next to each problematic input.

### Edge Cases and Recovery

**Mental Model: The Emergency Playbook**  
Edge cases are like emergency scenarios in aviation—rare but potentially catastrophic if not handled properly. Just as pilots train for engine failures, our system needs predefined recovery procedures for uncommon but critical scenarios. Each edge case has a **detection mechanism** (instrument warning lights), **containment strategy** (emergency protocols), and **recovery procedure** (checklist to restore normal operation). Thinking in terms of emergency playbooks ensures we handle edge cases systematically rather than reactively.

We categorize and handle edge cases across three domains: concurrency, resource lifecycle, and infrastructure failures:

**1. Concurrency and Race Conditions**
When multiple users interact with the same resources simultaneously, we must handle conflicts gracefully:

| Edge Case | Detection | Recovery Strategy | Implementation Approach |
|-----------|-----------|-------------------|-------------------------|
| **Concurrent edits to same post** | `updated_at` timestamp mismatch, optimistic locking version | Return 409 Conflict with current version, allow merge or overwrite choice | Include `version` or `last_updated` field in POST/PUT requests |
| **Duplicate registration attempts** | Unique constraint violation on email | Return 409 Conflict immediately, suggest password reset | Database UNIQUE constraint with proper error handling |
| **Simultaneous comment creation** | No conflict detection needed for comments | Allow parallel creation, rely on database transaction isolation | Use database transactions with appropriate isolation level |
| **Cached data staleness** | Cache timestamp vs. database timestamp comparison | Implement cache invalidation on write operations, return stale-while-revalidate | Set cache headers or use ETag/Last-Modified validators |

**2. Resource Lifecycle and State Transitions**
Resources (posts, users) transition through states (active, deleted, banned) that must be handled:

| Edge Case | Detection | Recovery Strategy | User Experience |
|-----------|-----------|-------------------|-----------------|
| **Accessing soft-deleted post** | `deleted_at` IS NOT NULL check | Return 404 Not Found (treat as non-existent) | Show "Post not found" without revealing deletion |
| **Editing a post after deletion** | Pre-operation check for `deleted_at` | Return 410 Gone with explanation | Display "This post has been deleted" with option to restore (if user is author) |
| **Login for deactivated account** | `is_active` flag check during authentication | Return 403 Forbidden with "Account deactivated" message | Provide reactivation instructions or contact admin |
| **Expired password reset token** | Token expiration timestamp validation | Return 400 Bad Request with "Token expired" | Prompt user to request new reset email |

**3. Infrastructure and Network Failures**
Distributed systems must handle partial failures gracefully:

| Edge Case | Detection | Recovery Strategy | System Impact |
|-----------|-----------|-------------------|---------------|
| **Database connection loss** | Connection pool timeout, query failure | Retry with exponential backoff, failover to read replica if available | Degraded performance, some operations may fail temporarily |
| **Database connection pool exhaustion** | `DB_POOL_CONN_TIMEOUT` exceeded | Queue requests, return 503 Service Unavailable | Increased latency, some requests rejected |
| **External service failure** | HTTP timeout or error response from dependency | Implement circuit breaker pattern, fallback to cached data | Reduced functionality but core features remain |
| **File upload timeout** | Multer/upload middleware timeout | Return 408 Request Timeout, clean up partial uploads | Upload fails but user can retry |
| **Memory exhaustion** | Node.js heap limit, `process.memoryUsage()` monitoring | Restart process with PM2 cluster, reject new connections | Temporary service disruption during restart |

**4. Data Integrity and Corruption**
Ensuring data remains consistent despite failures:

| Edge Case | Detection | Recovery Strategy | Prevention Measure |
|-----------|-----------|-------------------|-------------------|
| **Partial transaction failure** | Database rollback, incomplete data state | Automatic rollback, manual reconciliation from logs | Use database transactions for multi-step operations |
| **Referential integrity violation** | Foreign key constraint failure | Cascading updates/deletes or manual cleanup | Define proper foreign key constraints with ON DELETE rules |
| **Markdown rendering failure** | `marked` library exception during rendering | Fallback to raw markdown display, log error for investigation | Try-catch around `markdownToHtml()` function |
| **JWT token tampering** | Signature verification failure during `verify()` | Return 401 Unauthorized, force re-authentication | Use strong secret key, validate all token fields |

> **Decision: Soft Deletion with Audit Trail**
> - **Context**: Need to allow post deletion while preserving data for audit/recovery purposes
> - **Options Considered**:
>   1. **Hard deletion**: Permanently remove records from database
>   2. **Soft deletion with `deleted_at` flag**: Mark records as deleted without removing
>   3. **Archival to separate table**: Move deleted records to archive table
> - **Decision**: Implement soft deletion using `deleted_at` timestamp column
> - **Rationale**: Allows undeletion by authors/admins; preserves referential integrity for comments; enables analytics on deleted content; simple implementation with minimal query overhead
> - **Consequences**: All queries must filter `WHERE deleted_at IS NULL`; storage grows indefinitely; requires periodic cleanup of very old deletions

**Recovery Procedures for Critical Failures:**
For each critical failure scenario, we define a recovery checklist:

1. **Database Connection Failure Recovery**:
   1. Monitor connection health with periodic `SELECT 1` queries
   2. Implement retry logic with exponential backoff (1s, 2s, 4s, 8s)
   3. If primary database unreachable after max retries, switch to read replica for read operations
   4. Log alert to monitoring system for immediate operator intervention
   5. Attempt reconnection every 30 seconds until successful

2. **Failed User Registration Flow**:
   1. If email uniqueness check passes but user creation fails, rollback transaction
   2. If transaction rollback fails, log the orphaned email for manual cleanup
   3. Return 500 error to user with request ID for support reference
   4. Implement idempotency key to prevent duplicate charges if payment involved

3. **Partial Markdown Rendering Failure**:
   1. Catch exceptions in `markdownToHtml()` function
   2. Log the error with post ID and markdown snippet (truncated)
   3. Display raw markdown with warning banner "Formatting temporarily unavailable"
   4. Queue background job to re-render the post with updated library version

⚠️ **Pitfall: Silent Failures in Async Operations**  
A common mistake is failing to handle errors in asynchronous operations like database queries or external API calls. For example:
```javascript
// WRONG: Error swallowed, user sees infinite loading
app.post('/posts', async (req, res) => {
  const post = await PostModel.create(req.body.title, req.body.content, req.user.id);
  res.json(post);
});
```
**Why it's wrong**: Unhandled promise rejection crashes the Node.js process in production, or the request hangs indefinitely.  
**How to fix**: Always wrap async route handlers in try-catch or use Express error-handling middleware:
```javascript
// CORRECT: Errors caught and passed to error handler
app.post('/posts', async (req, res, next) => {
  try {
    const post = await PostModel.create(req.body.title, req.body.content, req.user.id);
    res.json(post);
  } catch (error) {
    next(error); // Pass to centralized error handler
  }
});
```

⚠️ **Pitfall: Overly Generic Error Messages**  
Returning the same error message for different failure modes (e.g., "Something went wrong" for both invalid credentials and database timeout) frustrates users and developers.  
**Why it's wrong**: Users can't take appropriate action; developers can't debug effectively; security information might be leaked in some cases but not others.  
**How to fix**: Create an error classification system that maps specific error types to appropriate user-facing messages while logging technical details server-side.

### Implementation Guidance

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Validation Library | `express-validator` (middleware-based) | `Joi` (schema-based) or `Zod` (TypeScript-first) |
| Error Handling Middleware | Custom Express error middleware | `http-errors` library for standardized errors |
| Logging | `winston` or `morgan` for HTTP logs | Structured logging with `pino` and log aggregation |
| Monitoring | Manual error tracking in logs | `Sentry` or `Datadog` for error tracking and alerts |

**Recommended File Structure for Error Handling:**

```
project-root/
├── src/
│   ├── middleware/
│   │   ├── errorHandler.js          # Centralized error handling middleware
│   │   ├── validation.js            # Request validation middleware  
│   │   └── notFoundHandler.js       # 404 handler for unmatched routes
│   ├── utils/
│   │   ├── errors/
│   │   │   ├── AppError.js          # Base error class
│   │   │   ├── ValidationError.js   # Validation-specific errors
│   │   │   ├── NotFoundError.js     # 404 errors
│   │   │   └── ForbiddenError.js    # 403 errors
│   │   └── logger.js                # Structured logging utility
│   ├── routes/
│   │   ├── auth.js                  # Auth routes with error handling
│   │   ├── posts.js                 # Post routes with error handling
│   │   └── users.js                 # User routes with error handling
│   └── app.js                       # App setup with middleware registration
└── package.json
```

**Infrastructure Starter Code: Custom Error Classes**

```javascript
// src/utils/errors/AppError.js
/**
 * Base application error class that all other errors extend
 * Provides consistent structure for error responses
 */
class AppError extends Error {
  constructor(message, statusCode = 500, code = 'INTERNAL_ERROR') {
    super(message);
    this.name = this.constructor.name;
    this.statusCode = statusCode;
    this.code = code;
    this.isOperational = true; // Marks errors we expect vs programming errors
    
    // Capture stack trace (excluding constructor call)
    Error.captureStackTrace(this, this.constructor);
  }
}

// src/utils/errors/ValidationError.js
class ValidationError extends AppError {
  constructor(message = 'Validation failed', errors = []) {
    super(message, 422, 'VALIDATION_FAILED');
    this.errors = errors; // Array of field-specific error objects
  }
}

// src/utils/errors/NotFoundError.js  
class NotFoundError extends AppError {
  constructor(resource = 'Resource') {
    super(`${resource} not found`, 404, 'NOT_FOUND');
  }
}

// src/utils/errors/ForbiddenError.js
class ForbiddenError extends AppError {
  constructor(message = 'You do not have permission to perform this action') {
    super(message, 403, 'FORBIDDEN');
  }
}

// src/utils/errors/UnauthorizedError.js
class UnauthorizedError extends AppError {
  constructor(message = 'Authentication required') {
    super(message, 401, 'UNAUTHORIZED');
  }
}
```

**Core Logic Skeleton: Centralized Error Handling Middleware**

```javascript
// src/middleware/errorHandler.js
const logger = require('../utils/logger');
const { AppError } = require('../utils/errors/AppError');

/**
 * Global error handling middleware
 * Catches all errors from route handlers and formats consistent responses
 */
function errorHandler(err, req, res, next) {
  // Log the error for debugging
  logger.error({
    message: err.message,
    stack: err.stack,
    path: req.path,
    method: req.method,
    ip: req.ip,
    user: req.user?.id || 'anonymous'
  });

  // Check if error is an operational error we created
  if (err instanceof AppError) {
    // TODO 1: Build error response object with code, message, and optional details
    // TODO 2: Include validation error details if error is ValidationError
    // TODO 3: Set HTTP status code from error.statusCode
    // TODO 4: Send JSON response with error structure
  } else if (err.name === 'ValidationError' && err.isJoi) {
    // TODO 5: Handle Joi validation errors by converting to our ValidationError format
    // TODO 6: Extract field-specific error messages from err.details
  } else if (err.name === 'JsonWebTokenError') {
    // TODO 7: Handle JWT errors with 401 Unauthorized
  } else if (err.name === 'TokenExpiredError') {
    // TODO 8: Handle expired tokens with 401 and specific message
  } else {
    // TODO 9: Handle unexpected errors (hide details in production)
    // TODO 10: In development, include stack trace; in production, generic message
  }
}

module.exports = errorHandler;
```

**Core Logic Skeleton: Validation Middleware**

```javascript
// src/middleware/validation.js
const { validationResult, body } = require('express-validator');
const { ValidationError } = require('../utils/errors/ValidationError');

/**
 * Validation rules for user registration
 */
const validateRegistration = [
  // TODO 1: Add validation for email field (required, email format, normalize)
  // TODO 2: Add validation for password field (required, min 8 chars, max 72)
  // TODO 3: Add validation for name field (required, max 100 chars)
  // TODO 4: Add custom validation to check password confirmation matches
  // TODO 5: Implement middleware to check validation results
  // TODO 6: If validation fails, throw ValidationError with field-specific messages
  // TODO 7: If validation passes, call next() to continue to route handler
];

/**
 * Validation rules for post creation
 */
const validatePostCreation = [
  // TODO 8: Validate title (required, 1-200 chars, no dangerous HTML)
  // TODO 9: Validate content (required, max 50000 chars)
  // TODO 10: Validate tags (optional, array, each tag alphanumeric with hyphens)
  // TODO 11: Sanitize HTML from title and content to prevent XSS
];

module.exports = {
  validateRegistration,
  validatePostCreation,
  // Export validationResult for use in other middleware
  validationResult
};
```

**Core Logic Skeleton: Route Handler with Comprehensive Error Handling**

```javascript
// src/routes/posts.js
const express = require('express');
const router = express.Router();
const { validatePostCreation } = require('../middleware/validation');
const { authenticate } = require('../middleware/auth');
const PostModel = require('../models/Post');
const { NotFoundError, ForbiddenError } = require('../utils/errors');

// GET /posts - List posts with pagination
router.get('/', async (req, res, next) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const tag = req.query.tag;
    
    // TODO 1: Validate page and limit are positive integers
    // TODO 2: Call PostModel.findPaginated(page, limit, tag)
    // TODO 3: Handle case where no posts found (empty array, not error)
    // TODO 4: Calculate pagination metadata using calculatePaginationMetadata
    // TODO 5: Return posts with pagination metadata
  } catch (error) {
    // TODO 6: Pass error to next() for centralized handling
    next(error);
  }
});

// PUT /posts/:id - Update a post
router.put('/:id', authenticate, validatePostCreation, async (req, res, next) => {
  try {
    const postId = req.params.id;
    const { title, content } = req.body;
    
    // TODO 7: Check if post exists using PostModel.findById(postId)
    // TODO 8: If post not found, throw new NotFoundError('Post')
    // TODO 9: Check if user is author using PostModel.isAuthor(postId, req.user.id)
    // TODO 10: If not author, throw new ForbiddenError()
    // TODO 11: Update post using PostModel.update(postId, title, content)
    // TODO 12: Return updated post
  } catch (error) {
    next(error);
  }
});
```

**Language-Specific Hints: JavaScript/Node.js**

1. **Async/Await Error Handling**: Always wrap async operations in try-catch blocks or use `.catch()` when using async/await in Express routes.

2. **Promise Rejection Handling**: In Node.js 15+, unhandled promise rejections terminate the process. Use `process.on('unhandledRejection', (reason, promise) => {})` to catch and log these.

3. **Error Stack Traces**: Use `Error.captureStackTrace(this, this.constructor)` in custom error classes to get clean stack traces pointing to where the error was thrown, not where the Error constructor was called.

4. **Environment-Specific Error Details**: Check `process.env.NODE_ENV` to determine whether to include stack traces in error responses (development: yes, production: no).

**Milestone Verification Checkpoint for Error Handling:**

After implementing error handling, verify with these commands:

```bash
# Test validation errors
curl -X POST http://localhost:3000/api/users/register \
  -H "Content-Type: application/json" \
  -d '{"email": "invalid-email", "password": "123"}' \
  -v
# Expected: 422 status with validation error details

# Test authentication errors  
curl -X GET http://localhost:3000/api/posts/1 \
  -H "Authorization: Bearer invalid.token.here" \
  -v
# Expected: 401 status with "Authentication required" message

# Test authorization errors (requires valid token but wrong user)
curl -X PUT http://localhost:3000/api/posts/1 \
  -H "Authorization: Bearer <VALID_TOKEN_FOR_USER_A>" \
  -H "Content-Type: application/json" \
  -d '{"title": "Updated"}' \
  -v
# Post 1 must be owned by different user. Expected: 403 status

# Test not found errors
curl http://localhost:3000/api/posts/999999 \
  -v
# Expected: 404 status with "Post not found" message

# Test server errors (simulate by temporarily breaking database connection)
# Stop database, then:
curl http://localhost:3000/api/posts \
  -v  
# Expected: 503 or 500 status with appropriate message
```

**Debugging Tips for Error Handling:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **All requests return 500** | Unhandled error in middleware | Check server logs for stack trace; add console.log at start of error handler | Wrap async middleware in try-catch; ensure error handler is last middleware |
| **Validation errors show generic message** | Error response format incorrect | Inspect error handler response structure; check ValidationError class | Ensure validation middleware throws ValidationError with errors array |
| **404 errors not caught by handler** | Route order incorrect in Express | Check app.js middleware registration order | Ensure `app.use('*', notFoundHandler)` is after all routes |
| **CORS errors on frontend** | Missing CORS headers on error responses | Check Network tab for preflight response headers | Add CORS headers in error handler, not just success paths |
| **Error logs missing request context** | Logger not receiving request info | Check logger middleware placement | Attach request ID to all logs; include path, method, user ID in error logs |
| **Client receives HTML error page** | Express default error handler active | Check if custom error handler is registered | Ensure `app.use(errorHandler)` is last middleware with all 4 parameters |


## Testing Strategy

> **Milestone(s):** Milestone 1 (Project Setup & Database Schema), Milestone 2 (User Authentication), Milestone 3 (Blog CRUD Operations), Milestone 4 (Frontend UI)

Testing is the quality assurance system for your blog platform—it verifies that each component works correctly in isolation, integrates properly with others, and provides the intended user experience. Think of it as a **three-phase construction inspection process**: unit tests check individual bricks (functions), integration tests verify the walls and joints (APIs and database), and end-to-end tests validate the entire building's livability (user workflows). This systematic approach catches bugs early, prevents regressions, and builds confidence that your platform behaves as designed under various conditions.

A robust testing strategy for the blog platform must address three critical dimensions: **correctness** (does it work as intended?), **security** (can it withstand attacks?), and **reliability** (does it handle edge cases gracefully?). This section outlines a comprehensive approach that aligns with the project milestones, providing concrete verification checkpoints at each stage of development.

### Testing Pyramid Implementation

**Mental Model: The Construction Inspection Process**

Imagine building a house. Before moving in, you conduct three levels of inspection:
1. **Material inspection** (unit tests): Check each brick, beam, and pipe individually for defects.
2. **Structural inspection** (integration tests): Verify that walls connect to foundations, plumbing connects to fixtures, and electrical circuits are properly wired.
3. **Final walkthrough** (end-to-end tests): Test the complete living experience—doors open, lights switch on, water flows—simulating real occupant behavior.

This pyramid structure ensures defects are caught at the cheapest level (a single brick) rather than the most expensive (after the house is fully built). The blog platform follows the same principle with more tests at the unit level (fast, cheap, numerous) and fewer at the E2E level (slow, expensive, critical paths only).

#### Unit Tests: Verifying Individual Components

Unit tests focus on the smallest testable units of code—individual functions, methods, or classes—in isolation from their dependencies. For the blog platform, this includes data models, utility functions, and business logic.

| Component Category | Example Units to Test | Key Assertions | Test Doubles Needed |
|-------------------|----------------------|----------------|---------------------|
| **Data Models** | `PostModel.create()`, `PostModel.findPaginated()` | Returns correct shape, respects validation rules, handles null/edge inputs | Database mock (e.g., `jest.mock()` for `query()`) |
| **Utility Functions** | `markdownToHtml()`, `calculatePaginationMetadata()` | Correct transformation, sanitization, mathematical accuracy | None (pure functions) |
| **Business Logic** | Password strength validation, post ownership checks | Logic correctness, error conditions, boundary values | Minimal mocks |
| **Middleware** | `errorHandler()`, authentication middleware | Proper error formatting, status codes, header manipulation | Mock `Request`, `Response`, `next` |

Unit tests should run quickly (milliseconds each) and in complete isolation. Use test doubles (mocks, stubs, fakes) to replace external dependencies like the database, file system, or third-party APIs. This ensures test failures point directly to bugs in the unit under test, not its dependencies.

> **Decision: Unit Testing Framework**
> - **Context**: Need a consistent, feature-rich framework for testing JavaScript/TypeScript code with good mocking capabilities and IDE integration.
> - **Options Considered**: 
>   1. **Jest**: Full-featured test runner with built-in assertions, mocking, coverage reporting, and snapshot testing.
>   2. **Mocha + Chai + Sinon**: Modular approach with separate libraries for test runner (Mocha), assertions (Chai), and mocking (Sinon).
>   3. **Node.js built-in test runner**: Minimalist option with no external dependencies.
> - **Decision**: **Jest** for the primary JavaScript implementation.
> - **Rationale**: Jest provides batteries-included testing with zero configuration for most projects, excellent TypeScript support, and powerful mocking capabilities that simplify testing database queries and external services. Its snapshot testing is valuable for React components in the frontend.
> - **Consequences**: Slightly larger dependency footprint but significantly reduced setup time and consistent testing patterns across the codebase.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| **Jest** | Built-in mocking, coverage reports, snapshot testing, zero config for most projects | Can be slower for very large test suites, opinionated configuration | ✅ Yes |
| **Mocha + Chai + Sinon** | Highly customizable, choose preferred assertion style, widespread adoption | Requires more setup and configuration, multiple dependencies | No |
| **Node.js test runner** | No dependencies, part of Node.js standard library | Limited features, no built-in mocking, early in development | No |

#### Integration Tests: Verifying Component Interactions

Integration tests verify that multiple units work together correctly, particularly at API endpoints where the server interacts with the database and external services. These tests use real database connections (or close approximations) to validate data flow through the system.

| Integration Layer | Test Focus | Setup Requirements | Teardown Requirements |
|-------------------|------------|-------------------|------------------------|
| **API Endpoints** | HTTP status codes, response body structure, error handling | Test database with seed data, running server instance | Clear test data, close connections |
| **Database Operations** | CRUD operations, transactions, constraints, JOIN queries | Isolated database (schema-only or Docker container) | Rollback transactions or truncate tables |
| **Authentication Flow** | Token generation/validation, protected route access, password reset | Test user accounts, JWT secret configuration | Clear sessions, revoke tokens |
| **Markdown Pipeline** | Markdown → sanitized HTML conversion, caching behavior | Mock external dependencies (if any) | Clear cache |

Integration tests should run against a test database that mirrors the production schema but contains isolated test data. Each test should be independent and clean up after itself to prevent test pollution.

> **Decision: Integration Testing Database Strategy**
> - **Context**: Need a database environment for integration tests that is isolated, reproducible, and fast.
> - **Options Considered**:
>   1. **Test-specific schema in shared database**: Create a unique schema per test run within a shared database instance.
>   2. **Dockerized database container**: Spin up a fresh database container for each test suite run.
>   3. **SQLite in-memory database**: Use an in-memory SQLite database for tests (if using SQL).
> - **Decision**: **Test-specific schema in shared database** for SQL databases; **MongoDB memory server** for NoSQL.
> - **Rationale**: Schema isolation provides good balance of speed and realism—tests run against the same database engine as production without permanent data pollution. For PostgreSQL/MySQL, use `CREATE SCHEMA test_xyz` and set search path; for MongoDB, use separate databases. This avoids container startup overhead while maintaining isolation.
> - **Consequences**: Requires careful connection pool management to ensure tests don't leak connections, and schema cleanup between test runs.

#### End-to-End Tests: Verifying User Journeys

End-to-end (E2E) tests simulate real user behavior by automating browser interactions that span multiple pages and API calls. These tests validate complete workflows like publishing a post, commenting, or user registration.

| Critical User Journey | Test Scenario | Key Interactions to Automate |
|----------------------|---------------|------------------------------|
| **Reader Experience** | Browse posts, read article, view comments | Visit homepage, click post link, scroll through content, verify comment section |
| **Author Publishing** | Create, edit, publish a blog post | Login, navigate to editor, input markdown, preview, save, verify publication |
| **User Registration** | Sign up, verify email, first login | Fill registration form, submit, check confirmation, login with new credentials |
| **Comment Thread** | Add comment, reply to comment, delete own comment | Authenticate, enter comment text, submit, verify display, reply flow, deletion |

E2E tests are the most realistic but also the slowest and most brittle. Focus on the **critical paths** that represent core user value. Use realistic wait conditions (not fixed sleeps) and implement retry logic for flaky operations.

> **Decision: E2E Testing Framework**
> - **Context**: Need a reliable way to automate browser interactions for critical user journeys with good debugging capabilities.
> - **Options Considered**:
>   1. **Cypress**: All-in-one framework with time-travel debugging, automatic waiting, and excellent developer experience.
>   2. **Playwright**: Cross-browser support, multiple language bindings, reliable auto-waiting and network interception.
>   3. **Selenium WebDriver**: Industry standard with widest browser support but requires more setup and manual waiting.
> - **Decision**: **Cypress** for primary E2E testing.
> - **Rationale**: Cypress provides an outstanding developer experience with real-time reloading, intuitive API, and built-in dashboard for test results. Its architecture runs in the same run-loop as the application, making tests more reliable and debugging easier with time-travel snapshots.
> - **Consequences**: Limited cross-browser support (Chromium-based browsers primarily) and requires adaptation for certain advanced scenarios like multiple tabs, but sufficient for the blog platform's testing needs.

#### Test Organization and Structure

Organize tests following the same structure as the source code to make them easy to locate and maintain:

```
project-root/
  src/
    models/
      Post.js
      Post.test.js           ← Unit tests for Post model
    routes/
      posts.js
      posts.integration.test.js ← Integration tests for post routes
    utils/
      markdown.js
      markdown.test.js       ← Unit tests for utilities
  cypress/
    integration/
      reader-journey.spec.js ← E2E tests
    fixtures/
      test-post.json
    support/
      commands.js
```

Each test file should include setup and teardown logic appropriate for its test level. Use descriptive test names that follow the pattern `"should [expected behavior] when [condition]"`.

### Milestone Verification Checkpoints

**Mental Model: The Recipe Card with Doneness Indicators**

Imagine following a recipe for baking bread. At each milestone—mixing, proofing, baking—you perform specific checks: "dough should double in size," "crust should be golden brown." These checkpoints confirm you're on track before proceeding. Similarly, each project milestone has verification commands and expected outcomes that validate progress.

The following tables provide concrete checkpoints for each milestone. Run these commands and verify the expected behaviors before considering the milestone complete.

#### Milestone 1: Project Setup & Database Schema

| Verification Aspect | Commands to Run | Expected Output | Behavior to Verify |
|-------------------|-----------------|-----------------|-------------------|
| **Dependency Installation** | `npm install` or `pip install -r requirements.txt` | No errors, dependencies downloaded | All packages from `package.json` or `requirements.txt` are installed |
| **Database Connection** | `npm run test:db-connection` or `python -m pytest tests/test_db_connection.py` | "Connection successful" or all tests pass | Application can connect to database with configured `PoolConfig` |
| **Schema Creation** | `npm run migrate:up` then `npm run migrate:status` | "No pending migrations" or similar | All tables (`users`, `posts`, `comments`) exist with correct columns |
| **Health Endpoint** | `curl http://localhost:3000/api/health` | `{"status":"ok","database":"connected"}` | API server responds with correct database status |
| **Index Verification** | `psql -U user -d blog -c "\di"` or database equivalent | Indexes on `users.email`, `posts.author_id`, `posts.created_at` visible | Frequently queried columns are properly indexed |

**Manual Verification Steps:**
1. Start the development server: `npm run dev`
2. Open browser to `http://localhost:3000` - should see placeholder page or "Server running"
3. Check server logs for "Database connection established" message
4. Attempt to insert a test user via database client: should succeed with unique email constraint
5. Run rollback migration: `npm run migrate:down` - should remove latest migration, then reapply with `npm run migrate:up`

#### Milestone 2: User Authentication

| Verification Aspect | Commands to Run | Expected Output | Behavior to Verify |
|-------------------|-----------------|-----------------|-------------------|
| **Registration Endpoint** | `curl -X POST http://localhost:3000/api/auth/register -H "Content-Type: application/json" -d '{"email":"test@example.com","password":"SecurePass123","name":"Test User"}'` | `{"user":{"id":1,"email":"test@example.com","name":"Test User"},"token":"..."}` | User created, password hashed with bcrypt, JWT returned |
| **Duplicate Registration** | Repeat same registration curl command | `{"error":"Email already exists","statusCode":400}` | Unique email constraint enforced, helpful error message |
| **Login Endpoint** | `curl -X POST http://localhost:3000/api/auth/login -H "Content-Type: application/json" -d '{"email":"test@example.com","password":"SecurePass123"}'` | `{"user":{...},"token":"..."}` | Valid credentials return user data and fresh token |
| **Invalid Login** | `curl ...` with wrong password | `{"error":"Invalid credentials","statusCode":401}` | Generic error message (doesn't reveal if email exists) |
| **Protected Route** | `curl http://localhost:3000/api/posts -H "Authorization: Bearer INVALID_TOKEN"` | `{"error":"Unauthorized","statusCode":401}` | Invalid/absent token returns 401 |
| **Valid Protected Route** | `curl http://localhost:3000/api/posts -H "Authorization: Bearer VALID_TOKEN"` | `{"posts":[],"total":0}` or posts list | Valid token grants access to protected resources |
| **Password Reset Flow** | Initiate reset via API, check email (or logs) | Reset token generated and emailed | Time-limited token created, email sent with reset link |

**Manual Verification Steps:**
1. Register a new user via the frontend form - should redirect to dashboard or show success message
2. Login with incorrect password - should show error without indicating whether email exists
3. Inspect browser cookies: authentication token should be `HttpOnly` and `Secure` (in production)
4. Logout and attempt to access protected page - should redirect to login
5. Verify password hash in database: should be bcrypt format (starts with `$2b$`)

#### Milestone 3: Blog CRUD Operations

| Verification Aspect | Commands to Run | Expected Output | Behavior to Verify |
|-------------------|-----------------|-----------------|-------------------|
| **Create Post** | `curl -X POST http://localhost:3000/api/posts -H "Authorization: Bearer TOKEN" -H "Content-Type: application/json" -d '{"title":"Test Post","content":"## Markdown content","tags":["test"]}'` | `{"post":{...},"htmlContent":"<h2>Markdown content</h2>"}` | Post created with `author_id` from token, raw markdown stored, HTML generated |
| **List Posts** | `curl "http://localhost:3000/api/posts?page=1&limit=10"` | `{"posts":[...],"total":1,"page":1,"totalPages":1}` | Paginated results with metadata, includes author names via JOIN |
| **Get Single Post** | `curl http://localhost:3000/api/posts/1` | Full post with author details and HTML content | Returns 404 for non-existent posts, includes comment count |
| **Update Own Post** | `curl -X PUT http://localhost:3000/api/posts/1 -H "Authorization: Bearer AUTHOR_TOKEN" -d '{"title":"Updated"}'` | `{"post":{...},"htmlContent":"..."}` | Only author can update, `updated_at` changes |
| **Update Others Post** | `curl -X PUT http://localhost:3000/api/posts/1 -H "Authorization: Bearer OTHER_USER_TOKEN" -d '{"title":"Hacked"}'` | `{"error":"Forbidden","statusCode":403}` | Ownership check prevents unauthorized edits |
| **Soft Delete** | `curl -X DELETE http://localhost:3000/api/posts/1 -H "Authorization: Bearer AUTHOR_TOKEN"` | `{"message":"Post deleted"}` | Post marked with `deleted_at` but not removed from database |
| **Markdown Sanitization** | Create post with `<script>alert('xss')</script>` | Script tags removed/escaped in HTML output | `DOMPurify` or similar sanitizes rendered HTML |

**Manual Verification Steps:**
1. Create multiple posts via UI, verify pagination controls appear when exceeding page limit
2. Test markdown preview in editor: headings, lists, links should render correctly
3. Attempt to edit another user's post via UI - should show permission error or hide edit button
4. View page source for post detail - verify script tags are sanitized in rendered HTML
5. Check database after delete: `deleted_at` column populated, post excluded from listings

#### Milestone 4: Frontend UI

| Verification Aspect | Commands to Run | Expected Output | Behavior to Verify |
|-------------------|-----------------|-----------------|-------------------|
| **Build Process** | `npm run build` or framework equivalent | Successful compilation, no errors | Production bundle created, TypeScript errors (if any) resolved |
| **Component Tests** | `npm test -- --testPathPattern=Component` | All component tests pass | React components render correctly with various props |
| **Lighthouse Audit** | Run Lighthouse in Chrome DevTools | Performance > 90, Accessibility > 90, SEO > 90 | Core Web Vitals met, semantic HTML used |
| **Responsive Design** | Manually resize browser or use device emulation | Layout adapts at breakpoints (mobile: <768px) | No horizontal scroll, readable text, touch-friendly buttons |
| **Form Validation** | Submit empty registration form | Client-side error messages, fields highlighted | Prevents invalid submission, clear error messages |
| **Loading States** | Throttle network in DevTools to "Slow 3G" | Loading spinners, skeleton screens during data fetch | User gets feedback during async operations |
| **Error Boundaries** | Force error in component (e.g., throw in `PostDetail`) | Fallback UI shown, error logged | Application doesn't crash entirely |

**Manual Verification Steps:**
1. Navigate through entire application without JavaScript errors in console
2. Test keyboard navigation: Tab through interactive elements in logical order
3. Verify images have alt text, buttons have descriptive labels for screen readers
4. Print stylesheet: posts should be readable when printed
5. Disable CSS: content should remain in logical order with semantic HTML

### Security Testing Considerations

**Mental Model: The Bank Vault with Multiple Locks**

Imagine securing a bank vault. You don't rely on a single lock—you use multiple layers: combination dial, time lock, seismic sensors, and armed guards. Each layer addresses different attack vectors. Security testing involves trying to bypass each layer, ensuring they work individually and together. For the blog platform, this means systematically testing authentication, authorization, input validation, and output encoding.

Security testing should be integrated throughout the development lifecycle, not tacked on at the end. Use automated tools for common vulnerabilities and manual testing for business logic flaws.

#### Common Vulnerability Testing

| Vulnerability Category | Test Method | Example Test Case | Expected Result |
|-----------------------|-------------|-------------------|-----------------|
| **SQL Injection** | Parameterized query verification, SQLi payload testing | Send `' OR '1'='1` as username in login | Authentication fails (not bypassed), error message doesn't expose SQL details |
| **XSS (Cross-Site Scripting)** | HTML/JavaScript injection in user inputs | Post content: `<script>alert(document.cookie)</script>` | Script tags removed/escaped in rendered output, cookies not accessible |
| **Authentication Bypass** | JWT manipulation, session fixation | Modify JWT payload to change `user_id`, use expired token | Requests rejected with 401, tokens validated for signature and expiry |
| **Authorization Flaws** | Horizontal privilege escalation | User A tries to delete User B's post via API | 403 Forbidden, ownership verified via `PostModel.isAuthor()` |
| **CSRF (Cross-Site Request Forgery)** | Missing CSRF token validation | Submit form from external site without CSRF token | Request rejected, state-changing operations protected |
| **Sensitive Data Exposure** | Information leakage in errors, headers | Trigger database error, check response | Generic error message, no stack traces in production |
| **Insecure Dependencies** | Dependency scanning | Run `npm audit` or `safety check` | No critical vulnerabilities, regular updates |

#### Security Testing Tools and Techniques

> **Decision: Security Testing Toolchain**
> - **Context**: Need automated tools to detect common security vulnerabilities without extensive manual penetration testing.
> - **Options Considered**:
>   1. **Static Application Security Testing (SAST)**: Analyze source code for patterns (ESLint security plugins, Bandit for Python).
>   2. **Dynamic Application Security Testing (DAST)**: Test running application (OWASP ZAP, Burp Suite Community).
>   3. **Dependency Scanning**: Check for known vulnerabilities in third-party packages (npm audit, Snyk).
> - **Decision**: **Combination of all three** with different frequencies.
> - **Rationale**: Each approach catches different issues: SAST finds code patterns (like hardcoded secrets), DAST finds runtime issues (like missing headers), dependency scanning finds known library vulnerabilities. Using all provides defense in depth.
> - **Consequences**: Additional setup and maintenance, but catches vulnerabilities early in development pipeline.

**Recommended Security Testing Pipeline:**

1. **Pre-commit Hooks**: Run ESLint with security rules (e.g., `eslint-plugin-security`) to catch dangerous patterns before code is committed.
2. **CI/CD Pipeline**:
   - Dependency scanning: `npm audit --audit-level=high` fails build on critical vulnerabilities
   - SAST scanning: Run dedicated security linter
   - Security unit tests: Test authentication/authorization logic
3. **Pre-production**:
   - DAST scan with OWASP ZAP against staging environment
   - Manual penetration testing for business logic flaws
4. **Production Monitoring**:
   - Log analysis for suspicious patterns (failed login attempts, SQL errors)
   - Regular dependency updates

#### Security Test Cases Implementation

Create dedicated security test files that verify protection mechanisms:

- **SQL Injection Tests**: Attempt injection via all user inputs (registration, search, post content)
- **XSS Tests**: Submit script payloads, verify sanitization in `markdownToHtml()`
- **Authentication Tests**: JWT tampering, token replay, session expiration
- **Authorization Tests**: Attempt unauthorized CRUD operations
- **Configuration Tests**: Check for security headers (HSTS, CSP, X-Frame-Options)

These tests should run as part of the integration test suite and fail the build if any vulnerability is detected.

#### Common Security Pitfalls and Mitigations

⚠️ **Pitfall: Storing tokens in localStorage**

- **Description**: Storing JWT tokens in `localStorage` makes them accessible to XSS attacks.
- **Why it's wrong**: If an attacker injects malicious JavaScript, they can steal tokens and impersonate users.
- **Fix**: Use `HttpOnly` cookies for token storage (server-side sessions) or ensure stringent XSS prevention if using client-side storage.

⚠️ **Pitfall: Verbose error messages**

- **Description**: Returning detailed database errors or stack traces to users.
- **Why it's wrong**: Exposes internal structure, potentially aiding attackers in crafting exploits.
- **Fix**: Use the global `errorHandler` middleware to return generic messages (`"An error occurred"`) in production while logging details internally.

⚠️ **Pitfall: Missing rate limiting**

- **Description**: Not limiting login attempts or API calls.
- **Why it's wrong**: Allows brute-force attacks on passwords or denial-of-service through resource exhaustion.
- **Fix**: Implement rate limiting middleware (e.g., `express-rate-limit`) on authentication endpoints and expensive operations.

⚠️ **Pitfall: Insecure password policies**

- **Description**: Allowing weak passwords (short length, no complexity).
- **Why it's wrong**: Makes credential guessing or brute-forcing easier.
- **Fix**: Enforce minimum length (12+ characters), but avoid excessive complexity requirements that lead to predictable patterns.

### Implementation Guidance

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Unit Testing Framework** | Jest (JavaScript) | Vitest (faster, compatible with Vite) |
| **Integration Testing** | Supertest + Jest | Testcontainers for database isolation |
| **E2E Testing** | Cypress | Playwright (cross-browser) |
| **Security Scanning** | ESLint security plugins + npm audit | Snyk + OWASP ZAP integration |
| **Test Coverage** | Istanbul/NYC (built into Jest) | Codecov integration with PR comments |
| **Mocking Library** | Jest built-in mocking | MSW (Mock Service Worker) for API mocking |

#### B. Recommended File/Module Structure

```
blog-platform/
├── src/
│   ├── __tests__/                 # Test utilities and shared setup
│   │   ├── setup.js               # Global test setup (Jest)
│   │   ├── db-test-utils.js       # Database helpers for tests
│   │   └── factories/             # Test data factories
│   │       ├── user-factory.js
│   │       └── post-factory.js
│   ├── models/
│   │   ├── User.js
│   │   ├── User.test.js           # Unit tests
│   │   ├── Post.js
│   │   └── Post.test.js
│   ├── routes/
│   │   ├── auth.js
│   │   ├── auth.integration.test.js # Integration tests
│   │   ├── posts.js
│   │   └── posts.integration.test.js
│   ├── middleware/
│   │   ├── auth.js
│   │   └── auth.test.js
│   └── utils/
│       ├── markdown.js
│       └── markdown.test.js
├── cypress/                       # E2E tests
│   ├── integration/
│   │   ├── auth.spec.js
│   │   ├── posts.spec.js
│   │   └── reader.spec.js
│   ├── fixtures/
│   └── support/
├── jest.config.js                 # Jest configuration
├── jest.integration.config.js     # Separate config for integration tests
└── cypress.config.js              # Cypress configuration
```

#### C. Infrastructure Starter Code

**Test Database Setup Utility (`src/__tests__/db-test-utils.js`):**

```javascript
import { query, getClient } from '../db';
import { Pool } from 'pg';

/**
 * Creates a test database connection with a unique schema for isolation
 */
export async function setupTestDatabase() {
  const testDbName = `test_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  
  // Create a new schema for this test run
  await query(`CREATE SCHEMA IF NOT EXISTS ${testDbName}`);
  await query(`SET search_path TO ${testDbName}, public`);
  
  // Run migrations within the test schema
  await runMigrationsInSchema(testDbName);
  
  return {
    schema: testDbName,
    cleanup: async () => {
      await query(`DROP SCHEMA IF EXISTS ${testDbName} CASCADE`);
    }
  };
}

/**
 * Global test setup/teardown for Jest
 */
beforeAll(async () => {
  global.testDb = await setupTestDatabase();
});

afterAll(async () => {
  await global.testDb.cleanup();
});

beforeEach(async () => {
  // Start transaction for test isolation
  global.testClient = await getClient();
  await global.testClient.query('BEGIN');
});

afterEach(async () => {
  // Rollback transaction to clean up test data
  await global.testClient.query('ROLLBACK');
  await global.testClient.release();
});
```

#### D. Core Logic Skeleton Code

**Unit Test Example for `PostModel.create()` (`src/models/Post.test.js`):**

```javascript
import { PostModel } from './Post';
import { query } from '../db';

// Mock the database module
jest.mock('../db', () => ({
  query: jest.fn()
}));

describe('PostModel.create', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('should create a post with valid input', async () => {
    // Arrange
    const mockPost = {
      id: 1,
      title: 'Test Post',
      content: 'Test content',
      author_id: 1,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
    
    query.mockResolvedValueOnce({ rows: [mockPost], rowCount: 1 });
    
    // Act
    const result = await PostModel.create('Test Post', 'Test content', 1, ['test']);
    
    // Assert
    expect(query).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO posts'),
      ['Test Post', 'Test content', 1, ['test']]
    );
    expect(result).toEqual(mockPost);
  });

  test('should throw validation error for empty title', async () => {
    // Act & Assert
    await expect(PostModel.create('', 'Content', 1, []))
      .rejects
      .toThrow('Title is required');
  });
});
```

**Integration Test Example for Post Creation (`src/routes/posts.integration.test.js`):**

```javascript
import request from 'supertest';
import app from '../app';
import { createTestUser, getAuthToken } from '../__tests__/factories/user-factory';

describe('POST /api/posts', () => {
  let authToken;
  
  beforeAll(async () => {
    const user = await createTestUser();
    authToken = await getAuthToken(user.id);
  });

  test('should create a post with authentication', async () => {
    // Arrange
    const postData = {
      title: 'Integration Test Post',
      content: '## Markdown Content\n\nWith **bold** text.',
      tags: ['integration', 'test']
    };
    
    // Act
    const response = await request(app)
      .post('/api/posts')
      .set('Authorization', `Bearer ${authToken}`)
      .send(postData);
    
    // Assert
    expect(response.status).toBe(201);
    expect(response.body.post).toHaveProperty('id');
    expect(response.body.post.title).toBe(postData.title);
    expect(response.body.htmlContent).toContain('<h2>');
    expect(response.body.htmlContent).not.toContain('<script>');
  });

  test('should reject unauthenticated requests', async () => {
    const response = await request(app)
      .post('/api/posts')
      .send({ title: 'Test' });
    
    expect(response.status).toBe(401);
  });
});
```

**E2E Test Example with Cypress (`cypress/integration/posts.spec.js`):**

```javascript
describe('Post Creation Flow', () => {
  beforeEach(() => {
    cy.loginTestUser(); // Custom command to handle authentication
    cy.visit('/dashboard');
  });

  it('should create and publish a new blog post', () => {
    // Navigate to editor
    cy.contains('New Post').click();
    
    // Fill in post details
    cy.get('[data-testid="post-title-input"]').type('My First Blog Post');
    cy.get('[data-testid="markdown-editor"]').type('## Introduction\n\nThis is my first post!');
    
    // Verify live preview
    cy.get('[data-testid="html-preview"]').should('contain', 'Introduction');
    
    // Add tags
    cy.get('[data-testid="tag-input"]').type('blog{enter}');
    
    // Publish
    cy.get('[data-testid="publish-button"]').click();
    
    // Verify success
    cy.url().should('include', '/posts/');
    cy.contains('My First Blog Post');
    cy.contains('This is my first post!');
  });
});
```

#### E. Language-Specific Hints

- **Jest Configuration**: Use `jest.config.js` to set up test environment, coverage thresholds, and module mapping for aliases.
- **Supertest Tips**: Remember to import your Express app instance, not start the server separately. Use `supertest.agent()` if you need cookie persistence.
- **Cypress Best Practices**: Use `data-testid` selectors instead of CSS classes for stability. Implement custom commands for common actions like `cy.login()`.
- **Database Isolation**: For PostgreSQL tests, use `SET search_path` to isolate schemas. For SQLite, use `:memory:` databases with separate connections.
- **Mocking Date/Time**: Use `jest.useFakeTimers()` or libraries like `sinon` to control time-dependent tests (token expiry, post scheduling).

#### F. Milestone Checkpoint Verification Commands

Add these scripts to your `package.json`:

```json
{
  "scripts": {
    "test": "jest --passWithNoTests",
    "test:unit": "jest src --testPathPattern=\\.test\\.js",
    "test:integration": "jest --config jest.integration.config.js",
    "test:e2e": "cypress run",
    "test:security": "npm audit --audit-level=high && eslint . --ext .js,.jsx,.ts,.tsx --config .eslintrc.security.js",
    "test:coverage": "jest --coverage",
    "test:db-connection": "node scripts/test-db-connection.js"
  }
}
```

#### G. Debugging Tips for Tests

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **Database connection timeout in tests** | Connection pool exhausted, tests not cleaning up | Check `DB_POOL_MAX` vs. number of parallel tests, examine open connections | Increase pool size, ensure `afterEach` releases connections, run tests sequentially |
| **JWT tokens expiring during test runs** | Short expiry time, slow test execution | Check token expiry (default 1h), test duration | Increase test timeout or token expiry for tests, mock time |
| **Flaky E2E tests** | Race conditions, dynamic content not loaded | Use Cypress command logs, add `cy.wait()` for specific conditions | Implement retry logic, use `cy.contains()` with timeout, avoid fixed waits |
| **Tests pass locally but fail in CI** | Environment differences, database setup | Compare environment variables, check CI logs for database errors | Use Dockerized test database in CI, ensure identical Node.js versions |
| **"N+1 query" warnings in tests** | Missing eager loading in list endpoints | Examine database query logs during tests | Implement JOINs or batch loading in `PostModel.findPaginated()` |
| **Coverage report shows untested lines** | Missing test cases for error conditions | Examine coverage report for specific untested lines | Add tests for error branches, edge cases |


## Debugging Guide

> **Milestone(s):** Milestone 1 (Project Setup & Database Schema), Milestone 2 (User Authentication), Milestone 3 (Blog CRUD Operations), Milestone 4 (Frontend UI)

Debugging is the forensic science of software development—the systematic process of examining a malfunctioning system to determine why it behaves differently than expected. For a blog platform, debugging spans multiple layers: database queries failing silently, authentication tokens expiring unexpectedly, markdown rendering producing broken HTML, or frontend components rendering blank screens. This guide provides a structured methodology for diagnosing and resolving these issues, transforming frustrating debugging sessions into systematic investigations.

Think of debugging as **medical diagnosis for software**. A doctor doesn't start with brain surgery—they follow a process: listen to symptoms (reproduce), check vital signs (isolate), form a hypothesis about the disease (hypothesize), run tests (test), and prescribe treatment (fix). Similarly, effective debugging requires moving from observing symptoms ("I can't log in") to identifying root causes ("password hash comparison failing due to encoding mismatch") through systematic investigation.

### Common Bug Symptoms and Solutions

The following table maps commonly encountered symptoms in blog platform development to their likely causes, diagnostic approaches, and specific fixes. Each entry represents a pattern observed across countless implementations—learning to recognize these patterns accelerates debugging.

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|------|
| **Can't login even with correct password** | 1. Password hash comparison failure (bcrypt work factor mismatch, salt issues) <br> 2. Database connection pool exhausted <br> 3. JWT secret mismatch between login and validation | 1. Log the hash comparison result in `bcrypt.compare()` <br> 2. Check database pool metrics (`SELECT * FROM pg_stat_activity` for PostgreSQL) <br> 3. Compare JWT_SECRET environment variable in different processes | 1. Ensure consistent Node.js version and bcrypt binding <br> 2. Increase `DB_POOL_MAX` or add connection timeout <br> 3. Use a single JWT secret source via configuration |
| **"Cannot read property 'title' of null" when loading posts** | 1. Missing JOIN in `PostModel.findPaginated` causing null author <br> 2. Database transaction not committed before query <br> 3. Soft-deleted posts returned without filtering | 1. Log the raw query output to see if `author` field is null <br> 2. Check transaction isolation level and commit timing <br> 3. Examine SQL for missing `WHERE deleted_at IS NULL` clause | 1. Use eager loading with proper JOIN in `PostModel.findPaginated` <br> 2. Ensure transactions commit before subsequent reads <br> 3. Add soft-delete filter to all post queries |
| **Posts appear duplicated in pagination** | 1. Offset/limit pagination with concurrent insertions causing skipped/duplicate rows <br> 2. Missing ORDER BY clause leading to unstable ordering | 1. Log the exact SQL with parameters being executed <br> 2. Check if `ORDER BY created_at DESC` is present in query | 1. Add stable ordering with unique secondary key: `ORDER BY created_at DESC, id DESC` <br> 2. Consider cursor-based pagination for large datasets |
| **Markdown rendering shows raw HTML tags** | 1. XSS sanitizer (DOMPurify) stripping valid HTML <br> 2. Missing Content-Type header causing browser to render as plain text <br> 3. React dangerouslySetInnerHTML not being used | 1. Check console for DOMPurify removal notifications <br> 2. Inspect network response headers for `Content-Type: text/html` <br> 3. Verify React component uses `dangerouslySetInnerHTML={{__html: content}}` | 1. Configure DOMPurify to allow safe tags (h1-h6, p, code, etc.) <br> 2. Set proper headers in API response <br> 3. Ensure markdown is converted to HTML before React rendering |
| **"Network Error" when making API calls from frontend** | 1. CORS policy blocking cross-origin requests <br> 2. `REACT_APP_API_URL` not set or incorrect <br> 3. Backend server not running or on different port | 1. Check browser console for CORS error details <br> 2. Log `REACT_APP_API_URL` value at app startup <br> 3. Verify backend is accessible via curl or browser | 1. Configure CORS middleware with proper origins <br> 2. Set `.env` file with `REACT_APP_API_URL=http://localhost:3001` <br> 3. Ensure backend server listens on correct port |
| **"JWT malformed" or "invalid token" errors** | 1. Token missing from Authorization header or cookie <br> 2. Token expired (check `exp` claim) <br> 3. Token signature verification failed due to secret change | 1. Inspect request headers in browser DevTools Network tab <br> 2. Decode JWT at jwt.io to check expiration <br> 3. Verify JWT_SECRET hasn't changed between restarts | 1. Ensure `apiClient` attaches token to `Authorization: Bearer <token>` <br> 2. Implement token refresh mechanism <br> 3. Use persistent secret or environment variable |
| **Database connections timing out after idle period** | 1. `DB_POOL_IDLE_TIMEOUT` set too low <br> 2. Database server-side idle timeout <br> 3. Connection pool not properly initialized on startup | 1. Check pool metrics for idle connections being closed <br> 2. Examine database server logs for disconnect events <br> 3. Verify pool initialization occurs before first request | 1. Increase `DB_POOL_IDLE_TIMEOUT` to 5-10 minutes <br> 2. Configure database server with longer timeout <br> 3. Implement connection health check on acquisition |
| **"N+1 query problem" slowing post listings** | 1. Loading authors separately for each post instead of JOIN <br> 2. Comments being fetched individually per post | 1. Use database query logging to see individual author queries <br> 2. Profile endpoint response time with increasing post count | 1. Modify `PostModel.findPaginated` to JOIN with users table <br> 2. Implement batch loading for comments count |
| **Mobile layout broken or overlapping elements** | 1. Missing viewport meta tag <br> 2. CSS media queries not targeting correct breakpoints <br> 3. Fixed-width elements exceeding viewport | 1. Use Chrome DevTools device emulation <br> 2. Check computed styles for mobile viewport width <br> 3. Inspect element dimensions vs. viewport | 1. Add `<meta name="viewport" content="width=device-width">` <br> 2. Implement mobile-first CSS with `min-width` media queries <br> 3. Use relative units (%, rem) instead of fixed pixels |
| **Registration allows duplicate emails despite unique constraint** | 1. Race condition between existence check and insert <br> 2. Database constraint not actually created <br> 3. Case-insensitive comparison needed | 1. Check database schema for unique constraint on email <br> 2. Simulate concurrent registration requests <br> 3. Test with "Test@example.com" vs "test@example.com" | 1. Use database-level constraint with `ON CONFLICT` handling <br> 2. Ensure migration created unique index <br> 3. Normalize email to lowercase before storage |

### Debugging Techniques and Tools

Effective debugging requires the right tools and techniques for each layer of the application stack. Think of these as **specialized medical instruments**—a stethoscope for listening to heartbeats (logging), an MRI for internal imaging (profiling), and blood tests for chemical analysis (network inspection).

#### Browser DevTools: The Frontend Diagnostic Lab

The browser's developer tools provide real-time observation of frontend behavior. For blog platform debugging:

- **Network Tab**: Inspect API requests made by `apiClient`. Check status codes, response bodies, and request headers. Look for failed requests (red status codes), missing CORS headers, or incorrect authentication tokens. The "Preview" tab shows formatted JSON responses.

- **Console Tab**: View `console.log` output from React components and `apiClient`. Errors appear in red with stack traces—click to jump to source. Use `console.table()` for inspecting arrays of objects like post listings.

- **Elements Tab**: Examine the rendered DOM structure. Check if React components rendered expected HTML, inspect CSS styles (especially media queries), and verify that `dangerouslySetInnerHTML` produced correct markup.

- **Application Tab**: Monitor localStorage, sessionStorage, and cookies. Verify JWT tokens are stored correctly (preferably in HTTP-only cookies, not localStorage for security). Clear storage to test fresh login flows.

- **Sources Tab**: Set breakpoints in JavaScript code. Use the debugger to step through `apiClient` calls, `useAuth` hook state changes, or markdown rendering logic. Add conditional breakpoints for specific post IDs or user states.

- **Performance Tab**: Record interactions to identify slow renders during post listing pagination or markdown preview updates. Look for long tasks blocking the main thread.

#### Server-Side Logging: The Application Black Box Recorder

Structured logging provides visibility into backend operations. Implement a logging middleware that captures:

| Log Field | Purpose | Example |
|-----------|---------|---------|
| `timestamp` | When event occurred | `2023-10-05T14:30:00Z` |
| `level` | Severity (error, warn, info, debug) | `"error"` |
| `message` | Human-readable description | `"Login failed for user@example.com"` |
| `method` & `path` | HTTP request details | `"POST /api/login"` |
| `statusCode` | HTTP response status | `401` |
| `userId` | Authenticated user (if any) | `"user_123"` |
| `error` | Error object details | `{message: "Invalid credentials", stack: "..."}` |
| `durationMs` | Request processing time | `45` |

Configure log levels appropriately: `DEBUG` for development (showing SQL queries), `INFO` for production (request summaries), and `ERROR` for exceptions. Use structured JSON logging for easy filtering and analysis.

#### Database Query Inspection: The Storage Layer X-Ray

Database issues often manifest as slow responses or incorrect data. Use these techniques:

- **Query Logging**: Enable SQL logging in your database client or ORM. For Node.js with `pg`, set `DEBUG=pg:*` environment variable. Examine generated SQL for missing conditions, inefficient JOINs, or N+1 patterns.

- **Explain Plan Analysis**: Prepend `EXPLAIN ANALYZE` to slow queries to see execution plan. Look for sequential scans (missing indexes), expensive hash JOINs, or large sorts. Add indexes on frequently filtered columns (`author_id`, `created_at`, `tags`).

- **Connection Pool Monitoring**: Track pool usage with metrics: active connections, idle connections, waiting clients. Implement a health check endpoint that verifies database connectivity and pool status.

- **Transaction Isolation Issues**: For concurrent operations (like comment counts), ensure proper transaction isolation levels. Use `SELECT FOR UPDATE` for critical updates or optimistic locking with version columns.

#### Network Tracing: The Communication Pathway Map

Network issues span client-server and server-database communication:

- **cURL Commands**: Test API endpoints directly without frontend: `curl -X POST -H "Content-Type: application/json" -d '{"email":"test@example.com","password":"secret"}' http://localhost:3001/api/login`. Compare results with browser requests.

- **HTTP Proxy Tools**: Use tools like mitmproxy or Charles Proxy to intercept and inspect traffic between frontend and backend. Modify requests to test error conditions.

- **Database Network Latency**: Use `pg_stat_activity` (PostgreSQL) or `SHOW PROCESSLIST` (MySQL) to see query execution states. Network latency between application and database appears as "idle in transaction" or slow query responses.

- **DNS Resolution Issues**: Ensure `localhost` resolves consistently. For Docker setups, use service names instead of `localhost`.

#### State Inspection: The System Memory Examination

Application state issues require careful observation:

- **React DevTools**: Inspect component props and state hierarchy. Track `AuthState` changes during login/logout flows. Profile component re-renders to identify unnecessary updates.

- **Redux DevTools** (if used): Visualize state changes over time with action history. Replay actions to reproduce bugs.

- **Node.js Debugger**: Use `node --inspect` to attach Chrome DevTools to backend. Set breakpoints in authentication middleware or `PostModel` methods. Inspect variable values at runtime.

- **Memory Profiling**: For memory leaks (common with event listeners or caching), use heap snapshots to find retained objects. Look for accumulating `PoolClient` instances or cached HTML strings.

### Troubleshooting Workflow

A systematic workflow transforms chaotic bug hunting into a repeatable diagnostic process. Follow these five steps like a **scientific method for debugging**:

#### Step 1: Reproduce — Create a Reliable Test Case

Goal: Consistently trigger the bug under controlled conditions.

1. **Identify the exact conditions** that cause the issue: specific user role, post content, browser, network state.
2. **Minimize variables**: Start with simplest possible scenario (fresh user, single post, clean browser session).
3. **Create a reproduction script**: Write a test that reliably triggers the bug. For API issues, create a cURL command or Postman collection. For UI issues, record step-by-step browser actions.
4. **Document the expected vs. actual behavior**: "Expected: login succeeds with valid credentials. Actual: 401 error even with correct password."
5. **Check if bug is intermittent or consistent**: Run reproduction 10 times to establish pattern.

> **Key Insight**: If you cannot reproduce a bug, you cannot debug it. Spend time creating a reliable reproduction before attempting fixes.

#### Step 2: Isolate — Narrow Down the Problem Scope

Goal: Determine which component or layer contains the defect.

1. **Frontend/Backend Separation**: Test API directly via cURL to eliminate frontend issues. If API works but frontend fails, problem is in React components or `apiClient`.
2. **Database/Application Separation**: Run the suspected SQL query directly on database. If query works but application fails, problem is in ORM or connection logic.
3. **Middleware Chain Isolation**: Temporarily bypass authentication middleware or validation to see if issue persists.
4. **Component Isolation**: For React issues, render the component in isolation (Storybook or simple test page) with controlled props.
5. **Data Isolation**: Test with minimal data (single user, single post) to eliminate data complexity.

Use a **binary search approach**: disable half the system, test, then narrow to the faulty half. Repeat until defect is isolated to specific function or module.

#### Step 3: Hypothesize — Formulate Possible Explanations

Goal: Develop testable theories about root cause based on symptoms and isolation.

1. **Brainstorm possible causes** based on symptoms and your system knowledge:
   - Authentication: token expiration, secret mismatch, cookie domain/path issues
   - Database: missing indexes, transaction deadlock, connection pool exhaustion
   - Frontend: state not updating, event handler missing, lifecycle issue
   - Network: CORS, firewall, DNS, load balancer configuration

2. **Prioritize hypotheses** by likelihood and impact:
   - High probability, easy to test: "JWT expired because clock skew"
   - Low probability, hard to test: "Database corruption in user table"

3. **Research similar issues**: Check project documentation, GitHub issues, Stack Overflow for patterns matching your symptoms.

4. **Formulate test predictions**: "If hypothesis X is correct, then when I do Y, I should see Z in the logs."

#### Step 4: Test — Gather Evidence for/against Hypotheses

Goal: Design and execute experiments to validate or refute each hypothesis.

1. **Add diagnostic instrumentation**:
   - Strategic `console.log` at system boundaries (API endpoints, database queries, component renders)
   - Temporary logging middleware capturing request/response details
   - Database query timing measurements

2. **Run controlled experiments**:
   - Modify one variable at a time (change JWT expiration from 1h to 24h)
   - Compare working vs. non-working scenarios side-by-side
   - Use A/B testing at code level (comment out suspect code, test)

3. **Collect and analyze evidence**:
   - Examine logs for patterns or anomalies
   - Measure performance metrics before/after changes
   - Compare actual system behavior against predicted behavior

4. **Use debugging tools deliberately**:
   - Set breakpoints at strategic code locations
   - Use watch expressions to monitor key variables
   - Take heap snapshots at different states

#### Step 5: Fix — Implement and Validate Correction

Goal: Apply minimal change that resolves root cause, then verify fix doesn't break existing functionality.

1. **Implement the fix** at the appropriate abstraction level:
   - Fix root cause, not symptoms (fix hash comparison, not just extend token expiry)
   - Follow existing patterns and conventions
   - Add comments explaining why fix is needed

2. **Test the fix** with reproduction case from Step 1:
   - Verify bug no longer occurs
   - Check edge cases and boundary conditions
   - Ensure fix doesn't introduce regressions

3. **Add preventive measures**:
   - Write unit test capturing the bug scenario
   - Add validation or sanity checks to catch similar issues early
   - Update documentation about the pitfall

4. **Monitor in production** (if applicable):
   - Add metrics to detect recurrence
   - Set up alerts for similar error patterns
   - Document the fix in runbooks or knowledge base

This workflow creates a **feedback loop of learning**: each debug session improves your mental model of the system, making future debugging more efficient.

### Implementation Guidance

#### Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Logging** | `console.log` with timestamps + Winston for JSON formatting | Structured logging with Winston/Pino + ELK stack (Elasticsearch, Logstash, Kibana) |
| **Error Tracking** | Custom `AppError` class with stack traces | Sentry.io or Bugsnag for real-time error monitoring with source maps |
| **API Debugging** | cURL commands + Postman collections | Swagger/OpenAPI with built-in testing + API monitoring (Postman Monitor) |
| **Database Debugging** | SQL query logging in development | Query performance monitoring with pgBadger (PostgreSQL) or Percona Monitoring |
| **Frontend Debugging** | Browser DevTools + React DevTools | LogRocket for session replay + error tracking |
| **Performance Profiling** | Chrome DevTools Performance tab | New Relic or Datadog for full-stack performance monitoring |

#### Recommended File/Module Structure

```
blog-platform/
├── backend/
│   ├── src/
│   │   ├── middleware/
│   │   │   ├── logging.js          # Request/response logging middleware
│   │   │   └── errorHandler.js     # Global error handling with AppError
│   │   ├── utils/
│   │   │   ├── debugHelpers.js     # Debug utilities (inspect queries, log contexts)
│   │   │   └── logger.js           # Winston/Pino logger configuration
│   │   ├── config/
│   │   │   └── database.js         # PoolConfig and connection debugging
│   │   └── scripts/
│   │       └── debugQueries.js     # Script to run/test database queries
│   └── test/
│       └── debug/
│           └── reproductionCases.js # Reproduction scripts for common bugs
└── frontend/
    ├── src/
    │   ├── utils/
    │   │   ├── apiClient.js        # Enhanced with request/response logging
    │   │   └── debugLogger.js      # Frontend logging utility
    │   └── hooks/
    │       └── useDebug.js         # Debug hooks for component lifecycle
    └── public/
        └── debug.html              # Isolated component testing page
```

#### Infrastructure Starter Code

**Complete Logging Middleware (`backend/src/middleware/logging.js`):**

```javascript
const logger = require('../utils/logger');

/**
 * Request logging middleware that captures method, path, status, duration
 */
function requestLogger(req, res, next) {
  const start = Date.now();
  const { method, originalUrl, ip, headers } = req;
  
  // Capture response finish
  res.on('finish', () => {
    const duration = Date.now() - start;
    const { statusCode } = res;
    const userId = req.user?.id || 'anonymous';
    
    logger.info('HTTP Request', {
      timestamp: new Date().toISOString(),
      method,
      path: originalUrl,
      ip,
      userAgent: headers['user-agent'],
      userId,
      statusCode,
      durationMs: duration,
      // Only log query params for non-sensitive endpoints
      query: originalUrl.includes('/api/posts') ? req.query : undefined
    });
  });
  
  // Capture errors
  res.on('error', (err) => {
    logger.error('HTTP Response Error', {
      error: err.message,
      stack: err.stack,
      path: originalUrl,
      method
    });
  });
  
  next();
}

/**
 * Detailed debug logging for development
 */
function debugLogger(req, res, next) {
  if (process.env.NODE_ENV === 'development') {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
    console.log('Headers:', JSON.stringify(req.headers, null, 2));
    console.log('Body:', JSON.stringify(req.body, null, 2));
    console.log('Query:', JSON.stringify(req.query, null, 2));
    console.log('User:', req.user || 'unauthenticated');
  }
  next();
}

module.exports = { requestLogger, debugLogger };
```

**Enhanced Error Class with Debug Info (`backend/src/utils/AppError.js`):**

```javascript
class AppError extends Error {
  constructor(message, statusCode, code, isOperational = true) {
    super(message);
    this.statusCode = statusCode;
    this.code = code;
    this.isOperational = isOperational;
    this.stack = new Error(message).stack;
    this.timestamp = new Date().toISOString();
    
    // Capture additional debug context
    this.context = {
      // Can be enriched by error handling middleware
    };
  }
  
  toJSON() {
    return {
      message: this.message,
      statusCode: this.statusCode,
      code: this.code,
      timestamp: this.timestamp,
      ...(process.env.NODE_ENV === 'development' && { stack: this.stack }),
      ...(this.context && Object.keys(this.context).length > 0 && { context: this.context })
    };
  }
}

// Specific error types for common scenarios
class ValidationError extends AppError {
  constructor(message, errors = []) {
    super(message, 400, 'VALIDATION_ERROR');
    this.errors = errors;
  }
}

class NotFoundError extends AppError {
  constructor(resource, id) {
    super(`${resource} with ID ${id} not found`, 404, 'NOT_FOUND');
    this.resource = resource;
    this.id = id;
  }
}

// ... other error types (UnauthorizedError, ForbiddenError, etc.)

module.exports = {
  AppError,
  ValidationError,
  NotFoundError
};
```

#### Core Logic Skeleton Code

**Database Query Debug Helper (`backend/src/utils/debugHelpers.js`):**

```javascript
const { query } = require('../config/database');

/**
 * Logs and analyzes a database query for debugging purposes
 * @param {string} sql - The SQL query with placeholders
 * @param {Array} params - Query parameters
 * @param {string} context - Context where query is executed (e.g., "PostModel.findPaginated")
 */
async function debugQuery(sql, params, context) {
  if (process.env.NODE_ENV !== 'development') {
    return query(sql, params);
  }
  
  const start = Date.now();
  console.log(`\n=== DATABASE QUERY DEBUG (${context}) ===`);
  console.log('SQL:', sql);
  console.log('Params:', params);
  
  try {
    const result = await query(sql, params);
    const duration = Date.now() - start;
    
    console.log(`Duration: ${duration}ms`);
    console.log(`Rows: ${result.rowCount}`);
    
    if (result.rowCount > 0 && result.rowCount <= 5) {
      console.log('Sample rows:', JSON.stringify(result.rows, null, 2));
    } else if (result.rowCount > 5) {
      console.log('First row:', JSON.stringify(result.rows[0], null, 2));
    }
    
    console.log('===============================\n');
    return result;
  } catch (error) {
    const duration = Date.now() - start;
    console.error(`Query failed after ${duration}ms:`, error.message);
    console.error('Full SQL with params:', {
      sql,
      params,
      error: error.message
    });
    throw error;
  }
}

/**
 * Checks database connection health and pool status
 */
async function checkDatabaseHealth() {
  // TODO 1: Attempt to acquire a client from the pool with timeout
  // TODO 2: Execute a simple query like "SELECT 1"
  // TODO 3: Measure query execution time
  // TODO 4: Release the client back to pool
  // TODO 5: Return health status object with metrics
}

module.exports = { debugQuery, checkDatabaseHealth };
```

**Frontend API Client with Debug Mode (`frontend/src/utils/apiClient.js`):**

```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

/**
 * Enhanced fetch wrapper with debugging capabilities
 */
async function apiClient(endpoint, config = {}) {
  const { method = 'GET', body, headers = {}, ...customConfig } = config;
  
  // TODO 1: Add JWT token from localStorage or cookie to Authorization header
  // TODO 2: Set Content-Type to application/json for request bodies
  // TODO 3: Stringify request body if present
  // TODO 4: Log request details in development mode (method, endpoint, body)
  // TODO 5: Make fetch request with timeout using AbortController
  // TODO 6: Parse JSON response or handle non-JSON responses
  // TODO 7: Check for HTTP error status codes (4xx, 5xx)
  // TODO 8: Log response details in development mode (status, response data)
  // TODO 9: Handle network errors with user-friendly messages
  // TODO 10: Return parsed response data
}

/**
 * Logs API call details for debugging
 */
function logApiCall(method, url, requestBody, response, duration, error = null) {
  if (process.env.NODE_ENV === 'development') {
    console.group(`API Call: ${method} ${url}`);
    console.log('Request:', requestBody);
    console.log('Response:', response);
    console.log(`Duration: ${duration}ms`);
    if (error) console.error('Error:', error);
    console.groupEnd();
  }
}

module.exports = apiClient;
```

#### Language-Specific Hints (JavaScript/Node.js)

- **Async/Await Debugging**: Use `try/catch` around `await` calls and log errors with context. For parallel operations, use `Promise.allSettled()` to get results even if some fail.

- **Node.js Inspector**: Run server with `node --inspect src/server.js`, then open `chrome://inspect` in Chrome to debug backend code with breakpoints and watch expressions.

- **Memory Leak Detection**: Use `node --inspect --expose-gc` and take heap snapshots in Chrome DevTools. Look for growing `PoolClient` or event listener counts.

- **Promise Unhandled Rejections**: Add `process.on('unhandledRejection', (reason, promise) => { console.error('Unhandled Rejection:', reason); })` to catch missing error handling.

- **Database Pool Debugging**: For `pg` pool, listen to events: `pool.on('connect', (client) => console.log('New client connected'))` and `pool.on('remove', (client) => console.log('Client removed'))`.

- **React Component Re-render Debugging**: Use `React.memo` with custom comparison or `useMemo`/`useCallback` to prevent unnecessary re-renders. The React DevTools Profiler shows which components re-render and why.

#### Milestone Checkpoint

**After implementing debugging infrastructure:**

1. **Run the health check**: Execute `node scripts/checkHealth.js` (create this script that uses `checkDatabaseHealth`). Expected output:
   ```
   Database Health: OK
   Connection Pool: 2/20 active connections
   Query Latency: 15ms
   ```

2. **Test logging middleware**: Start server and make a request to `/api/posts`. Check server logs for structured output:
   ```
   {"level":"info","message":"HTTP Request","timestamp":"2023-10-05T14:30:00Z","method":"GET","path":"/api/posts","statusCode":200,"durationMs":45}
   ```

3. **Verify frontend debugging**: In browser console, you should see API call logs when navigating between pages. The logs should show request/response details.

4. **Test error handling**: Trigger a validation error (e.g., register with invalid email). Verify the response includes structured error information and server logs capture the error context.

**Signs something is wrong:**
- No logs appear in development mode → Check `NODE_ENV` variable and logging level configuration
- Database queries not showing in logs → Ensure `debugQuery` is used instead of direct `query` calls
- Frontend API logs missing → Verify `apiClient` is used instead of direct `fetch` calls
- Memory usage growing continuously → Check for connection leaks or event listeners not cleaned up

#### Debugging Tips Table

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|------|
| Server crashes on startup with "Cannot find module" | Missing dependencies or incorrect import paths | Check `package.json` and compare with `node_modules` | Run `npm install` or fix import statement |
| "Invalid JWT token" after server restart | JWT secret changed, invalidating existing tokens | Check JWT_SECRET environment variable consistency | Use persistent secret or implement token migration |
| Posts not appearing for logged-in users | Authorization logic filtering too aggressively | Log the WHERE clause in `PostModel.findPaginated` for authenticated vs anonymous | Adjust authorization filters to include appropriate posts |
| Markdown preview not updating in real-time | Debouncing too aggressive or state not updating | Log editor component state changes and preview generation calls | Adjust debounce timing or ensure proper state synchronization |
| Database connection pool exhausted under load | Connection leaks (not releasing clients) or pool size too small | Monitor pool metrics and check for missing `client.release()` calls | Ensure all code paths release database clients, increase pool size |

This debugging guide provides a comprehensive toolkit for diagnosing and resolving issues throughout the blog platform development journey. By combining systematic methodology with appropriate tools, you transform debugging from a frustrating chore into an engaging investigation that deepens your understanding of the entire system.


## Future Extensions
> **Milestone(s):** Beyond current milestones (extensions to Milestone 1-4)

The architecture of the blog platform is intentionally designed with **extensibility** as a first-class consideration—not as an afterthought. A well-architected system should accommodate growth without requiring complete rewrites. This section outlines plausible enhancements that the current design can gracefully support, divided into **near-term features** (requiring minimal architectural changes) and **long-term evolutions** (involving more significant infrastructure additions). Each extension maintains the core principles of security, maintainability, and separation of concerns while addressing specific scalability or functionality needs.

### Near-Term Enhancements

These enhancements represent features that build directly upon the existing architecture without requiring fundamental re-engineering. They align with the natural progression of a blog platform from basic functionality to a more full-featured publishing system.

#### 1. Tags and Categories: Content Organization System

**Mental Model: The Library Dewey Decimal System**
Think of tags as flexible, user-generated labels that can be freely applied to posts (like sticky notes on books), while categories represent a fixed hierarchical taxonomy managed by administrators (like library sections: Fiction, Non-Fiction, Science). This dual approach allows for both organic discovery and structured navigation.

**Implementation Approach:**
The current `PostObject` already includes a `tags` array field. To implement full tagging functionality:

1. **Database Schema Extension:** Add a `tags` table for normalized tag storage and a `post_tags` junction table for many-to-many relationships, enabling tag reuse and efficient filtering.

| Table: `tags` | Type | Description |
|---------------|------|-------------|
| `id` | INTEGER (PK) | Unique tag identifier |
| `name` | VARCHAR(50) | Tag display name (unique, indexed) |
| `slug` | VARCHAR(50) | URL-friendly version (unique, indexed) |
| `created_at` | TIMESTAMP | When the tag was first created |

| Table: `post_tags` | Type | Description |
|-------------------|------|-------------|
| `post_id` | INTEGER (FK to posts.id) | Reference to the post |
| `tag_id` | INTEGER (FK to tags.id) | Reference to the tag |
| `created_at` | TIMESTAMP | When the tag was applied |

2. **API Extension:** New endpoints for tag management and filtered post listing:
   - `GET /api/tags` - List all tags with post counts
   - `GET /api/posts?tag=javascript` - Filter posts by tag
   - `POST /api/posts` - Accept tags array during creation
   - `PUT /api/posts/:id/tags` - Update tags for a post

3. **Frontend Components:** Tag cloud display, tag filtering interface, and tag input component with autocomplete.

> **Decision: Many-to-Many Tag Implementation**
> - **Context:** Need to associate multiple tags with posts and enable efficient tag-based filtering without denormalization.
> - **Options Considered:** 
>   - **Option A:** Store tags as JSON array in posts table (current approach)
>   - **Option B:** Normalized many-to-many relationship with junction table
>   - **Option C:** Hybrid: store array for simplicity, maintain separate tag table for metadata
> - **Decision:** Option B (normalized many-to-many)
> - **Rationale:** Enforces referential integrity, enables tag analytics (most popular tags), supports tag metadata (descriptions, colors), and allows efficient indexing for filtering operations. The junction table approach scales better as tag usage grows.
> - **Consequences:** Adds join complexity to queries but provides stronger data consistency and enables advanced tag management features.

#### 2. Post Scheduling: Time-Based Publication

**Mental Model: The Newspaper Print Queue**
Articles are written and edited in advance, then automatically published at predetermined times—similar to how newspapers are printed overnight for morning delivery. This allows content creators to maintain consistent publishing schedules without manual intervention.

**Implementation Approach:**
1. **Schema Extension:** Add `published_at` and `status` fields to the posts table:
   - `status ENUM('draft', 'scheduled', 'published', 'archived') DEFAULT 'draft'`
   - `published_at TIMESTAMP NULL` - When the post should become/public
   - `scheduled_for TIMESTAMP NULL` - When it was scheduled (for auditing)

2. **Background Worker:** Implement a scheduled job (cron job or job queue) that:
   1. Queries for posts where `status = 'scheduled'` AND `published_at <= NOW()`
   2. Updates their status to `'published'`
   3. Optionally sends notifications to subscribers

3. **API Extension:** 
   - Modify `PostModel.create()` and `PostModel.update()` to accept `published_at` parameter
   - Add validation: `published_at` must be in the future for scheduling
   - Update `PostModel.findPaginated()` to filter by `status = 'published'` by default (unless requesting user is author)

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `status` | ENUM | Publication state | 'draft', 'scheduled', 'published', 'archived' |
| `published_at` | TIMESTAMP | When post becomes publicly visible | NULL for drafts, future dates for scheduling |
| `scheduled_for` | TIMESTAMP | When scheduling was requested | Automatically set on schedule |

**Common Implementation Pitfalls:**
⚠️ **Pitfall: Timezone Confusion** - Storing naive timestamps without timezone context leads to publication at wrong times for users in different regions.
- **Why it's wrong:** Server timezone may differ from user's timezone, causing posts to publish hours early or late.
- **Fix:** Store all timestamps in UTC, convert to user's local timezone in the UI, use timezone-aware datetime libraries.

⚠️ **Pitfall: Missed Scheduled Publications** - Relying solely on cron jobs with no monitoring.
- **Why it's wrong:** If the cron job fails or server restarts, posts won't publish as scheduled.
- **Fix:** Implement idempotent job execution with logging, add health checks for the scheduler, consider using a robust job queue system.

#### 3. Basic Search: Content Discovery

**Mental Model: The Library Card Catalog**
A search system indexes key metadata (title, content, tags) and provides multiple lookup methods: simple keyword matching (like searching card catalog by title) and filtered search (like finding books by author and publication year).

**Implementation Approaches by Complexity:**

| Search Level | Implementation | Pros | Cons | Best For |
|-------------|---------------|------|------|----------|
| **Basic** | PostgreSQL `LIKE`/`ILIKE` or full-text search | Simple to implement, no new infrastructure | Poor performance on large datasets, limited relevance ranking | Small blogs (< 1,000 posts) |
| **Intermediate** | PostgreSQL full-text search with GIN indexes | Good relevance, leverages existing database | Still limited compared to dedicated search engines | Medium blogs (1,000-10,000 posts) |
| **Advanced** | Dedicated search engine (Elasticsearch, Meilisearch) | Excellent performance, advanced features (fuzzy, synonyms) | Additional infrastructure, operational complexity | Large-scale platforms |

**Recommended Path: Progressive Enhancement**
1. Start with PostgreSQL full-text search using `tsvector` columns and GIN indexes
2. Add search endpoints: `GET /api/search?q=keyword&limit=20`
3. Implement relevance scoring: title matches weighted higher than content matches
4. Add filtering: `GET /api/search?q=react&tag=javascript&author=john`

**Schema Extension for Full-Text Search:**
```sql
-- Add tsvector column for search indexing
ALTER TABLE posts ADD COLUMN search_vector tsvector;

-- Create index for faster searching  
CREATE INDEX posts_search_idx ON posts USING GIN(search_vector);

-- Update trigger to automatically maintain search_vector
CREATE TRIGGER posts_search_update BEFORE INSERT OR UPDATE ON posts
FOR EACH ROW EXECUTE PROCEDURE 
  tsvector_update_trigger(search_vector, 'pg_catalog.english', title, content);
```

#### 4. Image Uploads: Rich Media Support

**Mental Model: The Newspaper Photo Desk**
Writers submit photos to an editorial desk that processes, catalogs, and stores them in an organized archive. Each image gets resized for different contexts (thumbnail, article display, full resolution) and is tagged with metadata for retrieval.

**Implementation Architecture:**
1. **Storage Strategy Decision:**

| Option | Pros | Cons | Recommended For |
|--------|------|------|-----------------|
| **Local Filesystem** | Simple, no external dependencies | Difficult to scale, backup challenges, no CDN | Development/testing |
| **Cloud Storage (S3)** | Scalable, durable, CDN-friendly | Costs money, adds external dependency | Production deployment |
| **Database (BLOB)** | ACID transactions, easy backup | Database bloat, poor performance | Small binary files only |

2. **Processing Pipeline:**
   - Upload endpoint validates file type, size limits
   - Generate unique filename with UUID to prevent collisions
   - Create multiple resized versions (thumbnail: 300×300, medium: 800×600, original)
   - Store metadata in `images` table: `id`, `post_id`, `filename`, `alt_text`, `width`, `height`, `created_at`

3. **Security Considerations:**
   - Validate file signatures (not just extensions) to prevent disguised executables
   - Sanitize EXIF data to remove GPS coordinates and other sensitive metadata
   - Implement rate limiting to prevent storage abuse
   - Serve images with proper Content-Security-Policy headers

4. **Frontend Integration:**
   - Drag-and-drop upload component with preview
   - Image gallery selector for existing images
   - Markdown extension: `![Alt text](/api/images/abc123/medium)`

### Long-Term Architecture Evolution

These evolutions represent infrastructure-level changes that would be necessary as the platform scales to handle significant traffic, data volume, or complexity. Each addresses specific scalability bottlenecks while maintaining the core architecture's principles.

#### 1. API Versioning: Maintaining Backward Compatibility

**Mental Model: The Library Edition System**
Just as books are published in multiple editions (1st, 2nd, 3rd) with updates while older editions remain available for readers who depend on them, API versioning allows evolving the interface without breaking existing clients.

**Implementation Strategies:**

| Strategy | How it Works | Pros | Cons |
|----------|-------------|------|------|
| **URL Versioning** | `/api/v1/posts`, `/api/v2/posts` | Clear, cacheable, easy to understand | URL pollution, many versions to maintain |
| **Header Versioning** | `Accept: application/vnd.blog.v1+json` | Clean URLs, content negotiation | Less discoverable, harder to debug |
| **Query Parameter** | `/api/posts?v=1` | Simple to implement, easy testing | Not RESTful, caching challenges |
| **Media Type Versioning** | Custom media types with version | Formal specification, content negotiation | Complex, unfamiliar to many developers |

> **Decision: URL Path Versioning for Public APIs**
> - **Context:** Need to support multiple client versions simultaneously as the platform evolves, with clear deprecation paths.
> - **Options Considered:** URL path versioning vs. header-based versioning
> - **Decision:** URL path versioning (`/api/v1/`, `/api/v2/`)
> - **Rationale:** Maximizes discoverability and simplicity for API consumers. Easy to document, cache, and debug. Most developers are familiar with this pattern, and it works well with API gateway routing.
> - **Consequences:** Requires updating all client calls when migrating versions, but this is mitigated by maintaining old versions during a sunset period (e.g., v1 supported for 6 months after v2 launch).

**Implementation Guidance:**
1. Route structure: `app.use('/api/v1', v1Router); app.use('/api/v2', v2Router);`
2. Shared logic extracted to service layers to avoid duplication
3. Deprecation headers: `Deprecation: true`, `Sunset: Thu, 31 Dec 2020 23:59:59 GMT`
4. Documentation clearly indicates version lifecycle

#### 2. Caching Layer (Redis): Performance Optimization

**Mental Model: The Newspaper Kiosk vs. Printing Press**
A kiosk (cache) keeps today's popular newspapers readily available for quick purchase, while the printing press (database) can produce any newspaper on demand. The cache serves frequent requests instantly, reducing load on the primary production system.

**Caching Strategy Matrix:**

| Cache Type | What to Cache | Implementation | TTL | Invalidation Strategy |
|------------|---------------|---------------|-----|----------------------|
| **Database Query Results** | Results of expensive queries (tag clouds, popular posts) | Redis with query signature as key | 5-30 minutes | Cache-aside pattern, invalidate on writes |
| **Rendered HTML** | Fully rendered post pages for anonymous users | Redis or CDN | 1 hour | Purge on post update, comment addition |
| **Session Storage** | User session data for JWT blacklisting | Redis | Match JWT expiry | Remove on logout |
| **Rate Limit Counters** | API request counts per IP/user | Redis with increment | Sliding window | Automatic expiry |

**Architecture Integration:**
1. **Cache-Aside Pattern Implementation:**
   ```javascript
   async function getPostWithCache(postId) {
     const cacheKey = `post:${postId}`;
     const cached = await redis.get(cacheKey);
     if (cached) return JSON.parse(cached);
     
     const post = await PostModel.findById(postId);
     await redis.setex(cacheKey, 300, JSON.stringify(post)); // 5 minutes
     return post;
   }
   ```

2. **Cache Invalidation Strategy:** 
   - Write-through for critical data (update cache and DB simultaneously)
   - Write-behind for less critical data (update DB, async update cache)
   - Explicit invalidation on specific events (post update, comment addition)

3. **Monitoring Considerations:**
   - Cache hit ratio (target > 90% for hot data)
   - Memory usage and eviction policies
   - Connection pool saturation

**Common Scaling Pitfalls:**
⚠️ **Pitfall: Cache Stampede** - Multiple requests for the same expired cache key simultaneously trigger database queries.
- **Why it's wrong:** Creates thundering herd problem, overwhelming the database.
- **Fix:** Implement cache regeneration locking or probabilistic early expiration.

⚠️ **Pitfall: Stale Data Propagation** - Updates not properly invalidating cached versions.
- **Why it's wrong:** Users see outdated content even after updates.
- **Fix:** Implement explicit invalidation triggers using Redis pub/sub or database triggers.

#### 3. CDN for Static Assets: Global Content Delivery

**Mental Model: The Newspaper Distribution Network**
Instead of every reader traveling to the central printing press (your server), local distribution centers (CDN edge locations) around the world store copies of the paper, delivering it faster to readers in each region.

**Implementation Evolution:**

| Stage | Static Assets | Dynamic Content | Implementation |
|-------|---------------|----------------|---------------|
| **Initial** | Served from web server | Direct from origin | Express `static()` middleware |
| **Intermediate** | CDN for images, CSS, JS | Origin server | Configure CDN to cache static file extensions |
| **Advanced** | CDN for all cacheable content | Origin + edge computing | CDN with cache rules, edge functions for personalization |

**CDN Configuration Strategy:**
1. **Asset Fingerprinting:** `main.a1b2c3d4.css` instead of `main.css` for cache busting
2. **Cache-Control Headers:**
   - Immutable assets: `Cache-Control: public, max-age=31536000, immutable`
   - HTML pages: `Cache-Control: public, max-age=3600, stale-while-revalidate=86400`
3. **Origin Shield:** Single CDN node communicates with origin to reduce load
4. **Dynamic Content Caching:** Cache authenticated user pages with `Vary: Cookie` header

**Migration Steps:**
1. Update asset URLs to CDN domain (`cdn.blogplatform.com`)
2. Implement build process with asset hashing
3. Configure CDN with origin pull and cache rules
4. Monitor cache hit ratios and origin load reduction

#### 4. Read Replicas: Database Scaling Strategy

**Mental Model: The Library Branch System**
A central library (primary database) handles all new book acquisitions and catalog updates, while branch libraries (read replicas) maintain copies for public browsing. This distributes the read load while maintaining a single source of truth for writes.

**Architecture Evolution Path:**

| Stage | Read Capacity | Write Capacity | Implementation Complexity |
|-------|--------------|---------------|---------------------------|
| **Single Database** | ~100 QPS | ~50 WPS | Low (current implementation) |
| **Primary + Read Replica** | ~1,000 QPS | ~100 WPS | Medium (connection routing logic) |
| **Primary + Multiple Replicas** | ~10,000 QPS | ~500 WPS | High (load balancing, replication lag handling) |
| **Sharded Cluster** | 100,000+ QPS | 10,000+ WPS | Very High (application-level sharding logic) |

**Implementation Considerations:**
1. **Connection Routing:** Application logic to direct reads to replicas, writes to primary
   ```javascript
   class DatabaseRouter {
     async query(sql, params, { useReplica = false } = {}) {
       const pool = useReplica ? replicaPool : primaryPool;
       return pool.query(sql, params);
     }
   }
   ```

2. **Replication Lag Awareness:** Critical operations (reading your own writes) must use primary
   ```javascript
   // After creating a post, read from primary to ensure consistency
   await PostModel.create(title, content, authorId);
   const newPost = await primaryPool.query('SELECT * FROM posts WHERE id = $1', [postId]);
   ```

3. **Monitoring Metrics:**
   - Replication lag (seconds behind master)
   - Read/write query distribution
   - Connection pool utilization per instance

**ADR: Read Replica Implementation Strategy**
> **Decision: Connection-Level Routing with Session Consistency**
> - **Context:** Need to scale read capacity without major application rewrite, while maintaining acceptable consistency for most use cases.
> - **Options Considered:**
>   - **Option A:** Application-level routing (manual connection selection)
>   - **Option B:** Proxy-based routing (PgPool, ProxySQL)
>   - **Option C:** Driver-level routing (connection string with multiple hosts)
> - **Decision:** Option A (application-level routing)
> - **Rationale:** Provides maximum control over read/write routing logic, allows for session consistency (user's own writes always read from primary), and doesn't introduce additional infrastructure complexity at moderate scale.
> - **Consequences:** Application code becomes aware of database topology, requires careful implementation to avoid consistency issues. Migration path: start with simple routing, evolve to more sophisticated patterns as needed.

**Progressive Enhancement Path:**
1. **Phase 1:** Configure single read replica, manually route analytics queries
2. **Phase 2:** Implement middleware that automatically routes GET requests to replica
3. **Phase 3:** Add connection pool monitoring and automatic failover
4. **Phase 4:** Implement read/write splitting database proxy for transparent scaling

### Implementation Guidance

**A. Technology Recommendations for Extensions:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Tagging System** | PostgreSQL array column with GIN index | Dedicated tagging service with Redis for tag counters |
| **Post Scheduling** | Cron job with database polling | Message queue (RabbitMQ/Kafka) with delayed jobs |
| **Search Functionality** | PostgreSQL full-text search | Elasticsearch cluster with dedicated indexing pipeline |
| **Image Processing** | Sharp library with local storage | Serverless functions (AWS Lambda) + S3 + CloudFront |
| **Caching Layer** | In-memory cache (node-cache) | Redis cluster with sentinel for high availability |
| **CDN Integration** | Cloudflare free tier | AWS CloudFront/Google CDN with custom cache rules |
| **Read Replicas** | Single read replica with manual routing | Database proxy (PgBouncer + HAProxy) with auto-failover |

**B. Recommended File Structure for Extensions:**

```
blog-platform/
├── src/
│   ├── extensions/                    # New: Extension modules
│   │   ├── tagging/
│   │   │   ├── tag.model.js           # Tag database model
│   │   │   ├── tag.service.js         # Tag business logic
│   │   │   ├── tag.routes.js          # Tag API routes
│   │   │   └── tag.middleware.js      # Tag validation middleware
│   │   ├── scheduling/
│   │   │   ├── scheduler.js           # Job scheduler
│   │   │   ├── scheduled-jobs/        # Individual job definitions
│   │   │   └── worker.js              # Background job processor
│   │   ├── search/
│   │   │   ├── search-indexer.js      # Updates search indexes
│   │   │   ├── search-engine.js       # Search query processor
│   │   │   └── search.routes.js       # Search API endpoints
│   │   └── images/
│   │       ├── upload.service.js      # Image upload processing
│   │       ├── storage/               # Storage adapters (local, S3)
│   │       └── image-processor.js     # Image resizing/optimization
│   ├── cache/                         # New: Caching layer
│   │   ├── redis.client.js            # Redis connection wrapper
│   │   ├── cache.strategies.js        # Cache patterns (aside, through)
│   │   └── cache.invalidation.js      # Cache invalidation logic
│   └── api/                           # Existing API structure
│       ├── v1/                        # Current API version
│       └── v2/                        # New: Future API version
└── infrastructure/                    # New: Infrastructure configs
    ├── docker-compose.cache.yml       # Redis/Elasticsearch containers
    ├── cdn/                           # CDN configuration templates
    └── database/                      # Replication setup scripts
```

**C. Core Extension Skeletons:**

**1. Tag Service Implementation:**
```javascript
// src/extensions/tagging/tag.service.js
class TagService {
  /**
   * Creates or retrieves tags by name, associating them with a post
   * @param {string[]} tagNames - Array of tag names to associate
   * @param {number} postId - Post ID to associate tags with
   * @returns {Promise<Array<{id: number, name: string}>>} Created/found tags
   */
  async associateTagsWithPost(tagNames, postId) {
    // TODO 1: Normalize tag names (lowercase, trim whitespace)
    // TODO 2: For each tag name, find or create in tags table
    // TODO 3: Remove existing post-tag associations for this post
    // TODO 4: Create new associations in post_tags junction table
    // TODO 5: Update tag usage counters (optional optimization)
    // TODO 6: Return the complete list of associated tags with IDs
  }

  /**
   * Finds posts by tag with pagination
   * @param {string} tagSlug - URL-friendly tag identifier
   * @param {number} page - Page number (1-indexed)
   * @param {number} limit - Posts per page
   * @returns {Promise<{posts: PostObjectWithAuthor[], total: number}>}
   */
  async findPostsByTag(tagSlug, page = 1, limit = 20) {
    // TODO 1: Find tag by slug to get tag ID
    // TODO 2: Join posts, post_tags, and users tables
    // TODO 3: Apply WHERE clause for tag_id and post status = 'published'
    // TODO 4: Implement offset/limit pagination
    // TODO 5: Include author information via JOIN
    // TODO 6: Return posts array and total count for pagination UI
  }
}
```

**2. Search Service Skeleton:**
```javascript
// src/extensions/search/search-engine.js
class SearchEngine {
  /**
   * Indexes a post for search
   * @param {PostObject} post - Post to index
   * @returns {Promise<void>}
   */
  async indexPost(post) {
    // TODO 1: Extract searchable fields: title, content, tags
    // TODO 2: Remove HTML/markdown formatting for clean text
    // TODO 3: Generate search document with weights (title > content)
    // TODO 4: Store in PostgreSQL tsvector column OR Elasticsearch
    // TODO 5: Update index timestamp for caching purposes
  }

  /**
   * Searches posts by query with optional filters
   * @param {string} query - Search keywords
   * @param {Object} filters - Additional filters (author, tags, date range)
   * @returns {Promise<PostObjectWithAuthor[]>}
   */
  async searchPosts(query, filters = {}) {
    // TODO 1: Parse query string into search terms
    // TODO 2: Build PostgreSQL tsquery OR Elasticsearch query
    // TODO 3: Apply relevance scoring (title matches > content matches)
    // TODO 4: Apply additional filters from filters parameter
    // TODO 5: Execute search with pagination
    // TODO 6: Hydrate results with author information
  }
}
```

**3. Cache Strategy Implementation:**
```javascript
// src/cache/cache.strategies.js
class CacheStrategy {
  /**
   * Cache-aside pattern: Try cache first, fall back to data source
   * @param {string} key - Cache key
   * @param {Function} dataFn - Function to fetch data if not cached
   * @param {number} ttlSeconds - Time-to-live in seconds
   * @returns {Promise<any>} Cached or fresh data
   */
  async cacheAside(key, dataFn, ttlSeconds = 300) {
    // TODO 1: Attempt to get value from cache by key
    // TODO 2: If cache hit, parse and return cached value
    // TODO 3: If cache miss, execute dataFn to fetch fresh data
    // TODO 4: Store fresh data in cache with expiration (SETEX)
    // TODO 5: Return fresh data
    // TODO 6: (Optional) Implement stale-while-revalidate pattern
  }

  /**
   * Invalidates cache keys related to a post
   * @param {number} postId - Post ID to invalidate cache for
   * @returns {Promise<void>}
   */
  async invalidatePostCache(postId) {
    // TODO 1: Generate all cache key patterns for this post
    // TODO 2: Patterns: `post:${postId}`, `posts:page:*`, `tag:posts:*`
    // TODO 3: Use Redis SCAN or KEYS to find matching keys (careful with KEYS in production)
    // TODO 4: Delete all matching keys in pipeline for efficiency
    // TODO 5: Log cache invalidation for debugging
  }
}
```

**D. Milestone Extension Checkpoints:**

1. **Tags Extension Complete:**
   - Command: `npm test src/extensions/tagging/`
   - Expected: Tests pass for tag creation, association, and filtering
   - Manual verification: Create post with tags, filter posts by tag in UI
   - Failure signs: Tags not saving, filtering returning wrong posts

2. **Search Extension Complete:**
   - Command: `curl "http://localhost:3000/api/search?q=javascript"`
   - Expected: JSON response with relevant posts ranked by relevance
   - Manual verification: Search UI returns results as you type
   - Failure signs: No results for existing content, poor relevance ranking

3. **Caching Layer Operational:**
   - Command: `redis-cli monitor` (observe cache hits/misses)
   - Expected: Cache hit ratio > 80% for repeated requests
   - Manual verification: Repeated page loads show decreased response times
   - Failure signs: Redis memory full, cache misses causing DB load spikes

**E. Debugging Tips for Extensions:**

| Symptom | Likely Cause | Diagnosis Steps | Fix |
|---------|-------------|----------------|------|
| Tags not saving to database | Junction table foreign key constraints failing | Check post_tags table structure, verify post_id and tag_id exist | Ensure tags created before association, add proper error logging |
| Scheduled posts not publishing | Cron job not running or timezone mismatch | Check server logs for cron execution, verify `published_at` values in UTC | Fix cron schedule, convert user times to UTC before storing |
| Search returning no results | Index not updated or query syntax incorrect | Check search index table for entries, test raw query in database | Rebuild search index, adjust query parsing logic |
| Cache always returning misses | Redis connection failing or keys expiring too fast | Test Redis connection with `redis.ping()`, check TTL values | Fix Redis connection string, increase TTL for stable data |
| CDN serving outdated images | Cache invalidation not working or headers wrong | Check HTTP headers with browser DevTools, verify CDN purge API calls | Implement proper cache busting with file hashes, use CDN purge API |

**F. Progressive Enhancement Path:**
1. **Month 1-2:** Implement tags and basic search (PostgreSQL full-text)
2. **Month 3-4:** Add post scheduling and image uploads
3. **Month 5-6:** Introduce Redis caching for hot data
4. **Month 7-8:** Configure CDN for static assets
5. **Month 9-12:** Implement read replica and API versioning

Each extension builds upon the previous, allowing the development team to learn and adapt the architecture incrementally while maintaining system stability throughout the evolution.


## Glossary

> **Milestone(s):** All milestones (Milestone 1, Milestone 2, Milestone 3, Milestone 4)

This glossary provides definitions for technical terms, patterns, and architectural concepts used throughout this design document. Think of it as a **dictionary for the blog platform's technical vocabulary**—it helps ensure everyone uses the same language when discussing the system. Each term is defined in the context of this specific implementation, with references to where it appears in the design.

### Terms and Definitions

| Term | Definition |
|------|------------|
| **AbortController** | A JavaScript API that allows developers to abort one or more Web requests (like fetch requests) programmatically. Used in the frontend to cancel pending API calls when a user navigates away from a page, preventing memory leaks and race conditions. |
| **ACID** | **A**tomicity, **C**onsistency, **I**solation, **D**urability—a set of properties that guarantee reliable processing of database transactions. The blog platform's database operations (like creating a post with associated tags) rely on ACID properties to ensure data integrity even during concurrent access. |
| **ADR (Architecture Decision Record)** | A document that captures an important architectural decision, including the context, options considered, the decision made, rationale, and consequences. Used throughout this design document to explain why specific technical choices (like JWT over sessions) were made for the blog platform. |
| **API (Application Programming Interface)** | A set of rules and specifications that allow software components to communicate. In the blog platform, the backend exposes a RESTful API that the frontend uses to perform operations like user authentication and post management. |
| **AppError** | A custom error object type (with fields: `message`, `statusCode`, `code`, `isOperational`, `stack`) used throughout the backend to represent application errors consistently. Allows for centralized error handling and appropriate HTTP status code mapping. |
| **Asset Fingerprinting** | A technique where a unique hash is appended to static asset filenames (e.g., `styles.v123.css`) to force browsers to load the latest version when the file changes, bypassing cached copies. Important for frontend deployment to ensure users see updated CSS/JS after changes. |
| **AuthState** | A React state object (with fields: `user`, `isLoading`) that holds the current authentication status of the user. Managed by the `useAuth()` hook and shared across components via Context API to control UI rendering (e.g., showing login vs. editor buttons). |
| **bcrypt** | A password hashing function designed to be computationally expensive, making brute-force attacks impractical. Used in the blog platform's user authentication component with a work factor of 12 to securely store user passwords. |
| **Cache-Aside Pattern** | A caching strategy where the application first checks the cache for data; if missing (cache miss), it loads the data from the primary data source (database), stores it in the cache, and then returns it. Implemented by the `CacheStrategy.cacheAside()` function to improve post listing performance. |
| **Cache Config** | Configuration object (with fields: `host`, `port`, `password`, `ttl`, `keyPrefix`) for setting up a caching layer (like Redis). Used to define connection parameters and default cache behavior for the blog platform's optional performance enhancements. |
| **Cache Stampede** | A performance degradation scenario where multiple simultaneous requests for the same expired cache key all miss the cache and attempt to recompute the value simultaneously, overwhelming the data source. Mitigated by implementing cache warming or using probabilistic early expiration. |
| **CDN (Content Delivery Network)** | A geographically distributed network of servers that deliver cached static content (images, CSS, JS) to users from the nearest location. The blog platform's `CDNConfig` object configures CDN settings for serving optimized static assets in production deployments. |
| **Circuit Breaker** | A design pattern that prevents an application from repeatedly trying to execute an operation that's likely to fail (like calling an overloaded external service). After a threshold of failures, the circuit "opens" and fails fast, periodically testing if the operation has recovered. |
| **Connection Pooling** | A technique where a pool of database connections is created and reused by the application, avoiding the overhead of establishing a new connection for each query. Configured via `PoolConfig` with parameters like `DB_POOL_MAX` and `DB_POOL_IDLE_TIMEOUT`. |
| **Connection Routing** | The practice of directing database queries to appropriate database instances—typically writes to a primary database and reads to read replicas—to distribute load. An advanced optimization for scaling the blog platform's database tier. |
| **Context API** | A React feature that provides a way to pass data through the component tree without having to pass props down manually at every level (prop drilling). Used in the blog platform to share `AuthState` across all components that need to know about authentication status. |
| **CORS (Cross-Origin Resource Sharing)** | A security mechanism that allows web applications running at one origin to access resources from a different origin. The blog platform's API server must be configured with appropriate CORS headers to allow requests from the frontend when they are deployed separately. |
| **CRUD** | **C**reate, **R**ead, **U**pdate, **D**elete—the four basic operations for persistent storage. The blog platform's core functionality revolves around CRUD operations for blog posts, implemented via RESTful endpoints in the API. |
| **CSR (Client-Side Rendering)** | A rendering strategy where the browser downloads a minimal HTML page and JavaScript bundle, then uses JavaScript to generate the content and update the DOM. Used for authenticated sections of the blog platform (like the post editor) for interactivity. |
| **CSRF (Cross-Site Request Forgery)** | An attack that tricks a victim into submitting a malicious request using their authenticated session. The blog platform mitigates CSRF by using same-site cookies for session management and CSRF tokens for state-changing operations in forms. |
| **CSS Modules** | A CSS methodology where class names are scoped locally by default, preventing style conflicts across components. Recommended for the blog platform's frontend to maintain modular, maintainable stylesheets. |
| **Cursor-Based Pagination** | A pagination method that uses a unique, sequential reference point (cursor, typically an ID or timestamp) to fetch records after that point. More efficient than offset-based pagination for large datasets but more complex to implement. The blog platform uses offset-based pagination for simplicity. |
| **DAST (Dynamic Application Security Testing)** | Security testing performed on a running application to identify vulnerabilities that can be exploited (like SQL injection or XSS). Contrast with SAST (static analysis). The blog platform's testing strategy includes DAST for critical authentication and authorization flows. |
| **Database Migration** | The process of managing incremental, versioned changes to a database schema. The blog platform uses a migration system that supports running pending migrations forward and rolling back the most recent migration, ensuring schema evolution without data loss. |
| **DEFAULT_CACHE_TTL** | A constant representing the default time-to-live (in seconds) for cached data (default: 300). Used by the `CacheStrategy.cacheAside()` function to determine how long cached post listings remain valid before being refreshed. |
| **Eager Loading** | A database query optimization technique where associated data (like the author for each post) is loaded in the initial query using JOINs, avoiding the N+1 query problem. Implemented in `PostModel.findPaginated()` to include author details efficiently. |
| **End-to-End Tests** | Tests that simulate real user workflows through the entire application (frontend, backend, database) to verify that the system works as expected from the user's perspective. Used for critical paths like user registration → post creation → publishing. |
| **ETag** | An HTTP response header used for cache validation, representing a specific version of a resource. The browser can send the ETag in subsequent requests (via `If-None-Match`) to check if the content has changed, enabling efficient caching of post content. |
| **Exponential Backoff** | A retry strategy where the wait time between retry attempts increases exponentially (e.g., 1s, 2s, 4s, 8s). Used in the blog platform's API client (`apiClient`) when retrying failed requests due to network instability, preventing overwhelming the server. |
| **First Contentful Paint (FCP)** | A web performance metric measuring the time from navigation to when the browser renders the first piece of content from the DOM. The blog platform's rendering strategy (hybrid SSR/CSR) aims to optimize FCP for better user experience and SEO. |
| **Foreign Key Constraint** | A database constraint that ensures referential integrity between tables by requiring that values in a column (or columns) match values in another table's primary key. Used in the blog platform to link `posts.author_id` to `users.id`, preventing orphaned posts. |
| **FormState** | A React state object (with fields: `title`, `content`, `tags`) that holds the current values of the post editor form. Updated as the user types and validated before submission to the create/update post endpoints. |
| **Full-Text Search** | A search technique that examines all words in documents (like post content and titles) to match search criteria, rather than simple string matching. Can be implemented in PostgreSQL using `tsvector` columns and GIN indexes for advanced search functionality. |
| **GIN Index (Generalized Inverted Index)** | A type of PostgreSQL index optimized for full-text search and composite data types (like arrays). Used to accelerate search queries on post content when implementing the blog platform's optional search extension. |
| **HTTP Status Code** | A three-digit code returned by the server in an HTTP response to indicate the result of the request. The blog platform uses specific codes: 200 (OK), 201 (Created), 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden), 404 (Not Found), 500 (Internal Server Error). |
| **Idempotency** | A property of operations that can be applied multiple times without changing the result beyond the initial application. Important for safe retry of API requests (like POST requests with idempotency keys) in case of network failures. |
| **ImageObject** | A data structure (with fields: `id`, `post_id`, `filename`, `alt_text`, `width`, `height`, `created_at`) representing an uploaded image associated with a blog post. Part of the optional image upload extension for enriching post content. |
| **Index (Database)** | A database structure that improves the speed of data retrieval operations on a table at the cost of additional storage and slower writes. The blog platform creates indexes on frequently queried columns like `users.email`, `posts.author_id`, and `posts.created_at`. |
| **Integration Tests** | Tests that verify the interaction between multiple components (e.g., API endpoints interacting with the database and authentication middleware). Used to ensure that the blog platform's subsystems work correctly together. |
| **JWT (JSON Web Token)** | A compact, URL-safe token format for securely transmitting claims between parties as a JSON object. Used for authentication in the blog platform: after login, the server issues a signed JWT that the client includes in subsequent requests to prove identity. |
| **LayoutState** | A React state object (with fields: `isMobileMenuOpen`) that controls the responsive layout behavior, particularly the visibility of the mobile navigation menu. Managed by the layout component to adapt the UI for different screen sizes. |
| **MAX_IMAGE_SIZE_MB** | A constant defining the maximum allowed image upload size in megabytes (default: 5). Enforced by the backend when processing image uploads to prevent storage exhaustion and ensure reasonable upload times. |
| **Mental Model** | An intuitive analogy or conceptual framework that helps developers understand a complex system by comparing it to familiar real-world concepts. Used throughout this design document (e.g., "The Club Bouncer & Membership Desk" for authentication) to build intuition before technical details. |
| **Mobile-First CSS** | A design strategy where styling begins with the smallest screen size (mobile) and uses media queries to add or adjust styles for larger screens (tablet, desktop). Ensures the blog platform provides a good user experience across all devices. |
| **Monolithic Architecture** | An architectural style where all components of an application (presentation, business logic, data access) are combined into a single program on a single platform. The blog platform follows this approach for simplicity, as opposed to a microservices architecture. |
| **N+1 Query Problem** | A performance anti-pattern where an initial query fetches a list of items (e.g., posts), then N additional queries (one per item) are executed to fetch related data (e.g., author for each post). Solved in the blog platform by eager loading authors in the initial query. |
| **NotFoundError** | A custom error type (extending `AppError`) representing a 404 Not Found error when a requested resource (like a post or user) does not exist. Handled by the global error handler to return an appropriate HTTP response. |
| **Object-Relational Mapping (ORM)** | A technique that converts data between incompatible type systems in object-oriented programming languages and relational databases. The blog platform uses an ORM (or query builder) to interact with the database using JavaScript objects instead of raw SQL. |
| **Offset/Limit Pagination** | A pagination method using SQL's `LIMIT` and `OFFSET` clauses to skip a specified number of records (`OFFSET`) and return a fixed number (`LIMIT`). Used by `PostModel.findPaginated()` for simplicity, despite performance trade-offs with large offsets. |
| **Operational Error** | An expected error that occurs during normal operation of the application, such as invalid user input, missing resources, or network timeouts. Distinguished from programming errors; operational errors have `isOperational: true` in `AppError` objects. |
| **PaginationState** | A React state object (with fields: `currentPage`, `totalPages`, `pageSize`) that tracks the current state of paginated content (like the post list). Used to render pagination controls and trigger data fetching when the page changes. |
| **Parameterized Query** | A database query that uses placeholders (like `$1`, `$2`) for parameters, separating SQL logic from data values. Prevents SQL injection attacks by ensuring user input is treated as data, not executable code. Used by the `query()` function throughout the backend. |
| **PoolClient** | Represents a single client/connection from the database connection pool, obtained via `getClient()`. Used for transactions that require multiple queries to be executed on the same connection to maintain isolation. |
| **PoolConfig** | Configuration object (with fields: `host`, `port`, `database`, `user`, `password`, `max`, `idleTimeoutMillis`, `connectionTimeoutMillis`) for setting up the database connection pool. Defines connection limits and timeout behavior. |
| **PostObject** | The core data structure representing a blog post (with fields: `id`, `title`, `content`, `author_id`, `created_at`, `updated_at`, `tags`). Used when creating or updating posts via the API; stored in the database `posts` table. |
| **PostObjectWithAuthor** | An enriched version of `PostObject` that includes additional fields: `author` (object with `id`, `name`, `email`) and optionally `html_content` (rendered HTML). Returned by `PostModel.findById()` and `PostModel.findPaginated()` for display. |
| **Programming Error** | An unexpected bug in the code, such as type errors, undefined variable references, or logic flaws. Distinguished from operational errors; programming errors have `isOperational: false` and typically indicate a developer mistake that needs fixing. |
| **Prop Drilling** | The process of passing data through multiple layers of React components via props, which can become cumbersome in deep component trees. The blog platform avoids excessive prop drilling by using Context API for shared state like authentication. |
| **QueryResult** | The result object returned by the `query()` function (with fields: `rows` array, `rowCount`, `command`). Contains the retrieved data rows and metadata about the executed SQL command (SELECT, INSERT, etc.). |
| **Read Replica** | A copy of a database that handles read operations, distributing load away from the primary (write) database. An optional scaling technique for the blog platform to handle high traffic to public post listing pages. |
| **REACT_APP_API_URL** | An environment variable constant that defines the base URL for the backend API. Used by the frontend's `apiClient` to construct full endpoint URLs, allowing different configurations for development, testing, and production environments. |
| **Replication Lag** | The delay between when data is written to a primary database and when it becomes available on read replicas. Must be considered when implementing read replicas to ensure users see consistent data (e.g., immediately after publishing a post). |
| **SAST (Static Application Security Testing)** | Security testing performed by analyzing source code without executing it, looking for patterns that indicate vulnerabilities (like potential SQL injection). Part of the blog platform's security testing strategy, often integrated into CI/CD pipelines. |
| **SCHEDULER_INTERVAL_MINUTES** | A constant defining the interval (in minutes) at which a background scheduler runs to check for scheduled posts that need to be published (default: 5). Part of the optional post scheduling extension. |
| **SearchEngine** | A service component (with methods like `indexPost()` and `searchPosts()`) that provides full-text search capabilities over blog posts. An optional extension that can be added to the platform for improved content discovery. |
| **SearchFilters** | A data structure (with fields: `query`, `author`, `tags`, `startDate`, `endDate`, `limit`, `page`) representing the parameters for searching blog posts. Used by the search endpoint to filter and paginate results. |
| **SEARCH_PAGE_SIZE** | A constant defining the default number of search results per page (default: 20). Used by the search functionality to paginate results consistently. |
| **Soft Delete / Soft Deletion** | A pattern where a record is marked as deleted (e.g., by setting a `deleted_at` timestamp) instead of being physically removed from the database. Allows for recovery of accidentally deleted posts; implemented by `PostModel.softDelete()`. |
| **SQL Injection** | A security vulnerability where an attacker can execute arbitrary SQL code by injecting malicious input into database queries. Prevented in the blog platform by using parameterized queries exclusively via the `query()` function. |
| **SSR (Server-Side Rendering)** | A rendering strategy where the server generates the full HTML for a page and sends it to the browser. Used for public pages (like post listings and individual posts) in the blog platform to improve SEO and initial load performance. |
| **State Management** | The process of managing, sharing, and updating application state (like user authentication, form data, UI state) across components. The blog platform uses React's built-in state hooks and Context API for state management, avoiding complex libraries. |
| **TagObject** | A data structure (with fields: `id`, `name`, `slug`, `created_at`) representing a tag that can be associated with blog posts for categorization. Managed by the `TagService` for creating and associating tags with posts. |
| **Test Doubles** | A generic term for objects that replace real dependencies in tests, including mocks (pre-programmed expectations), stubs (provide canned answers), fakes (working implementations with shortcuts), and spies (record calls). Used in unit and integration tests to isolate components. |
| **tsvector** | A PostgreSQL data type that represents a document in a form optimized for full-text search—essentially a sorted list of distinct lexemes (normalized words). Used in conjunction with GIN indexes to enable efficient search on post content. |
| **Unit Tests** | Tests that verify individual functions or methods in isolation, typically by replacing dependencies with test doubles. Used for core utilities like `markdownToHtml()` and model methods like `PostModel.create()` to ensure they behave correctly. |
| **ValidationError** | A custom error type (extending `AppError` with additional `errors` array) representing input validation failures (400 Bad Request). Returned when user input fails server-side validation, with detailed error messages for each invalid field. |
| **XSS (Cross-Site Scripting)** | A security vulnerability that allows attackers to inject malicious scripts into web pages viewed by other users. Prevented in the blog platform by sanitizing rendered markdown HTML with DOMPurify and escaping user input in templates. |

> **Key Insight:** Many of these terms represent **design patterns** or **best practices** that emerge from solving common problems in web development (like pagination, caching, or authentication). Understanding these terms helps you recognize these patterns in other systems and apply the appropriate solutions to new problems.

### Implementation Guidance

Since this section is a reference glossary, implementation guidance focuses on how to effectively use these terms during development and documentation.

#### A. Technology Recommendations for Documentation

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Documentation Tool** | Markdown files in the repository | Static site generator (Docusaurus, VuePress) with search |
| **API Documentation** | Inline code comments + README | OpenAPI/Swagger specification with interactive UI |
| **Terminology Management** | Maintain this glossary as `GLOSSARY.md` | Automated extraction of terms from code comments |

#### B. Recommended File/Module Structure for Project Documentation

```
blog-platform/
├── docs/                           # Project documentation
│   ├── GLOSSARY.md                 # This glossary document
│   ├── ARCHITECTURE.md             # High-level design overview
│   ├── API.md                      # Detailed API endpoints
│   └── SETUP.md                    # Development setup instructions
├── src/                            # Source code
├── README.md                       # Main project README with quick start
└── package.json                    # Includes script for serving docs
```

#### C. Documentation Starter Code

Create a simple script to ensure terminology consistency across the codebase:

```javascript
// scripts/check-terminology.js
/**
 * Simple terminology checker that scans code for discouraged terms
 * and suggests preferred alternatives from our glossary.
 */
const fs = require('fs');
const path = require('path');

// Preferred terms mapping (discouraged → preferred)
const TERMINOLOGY_MAP = {
  'password': 'password_hash', // When referring to stored passwords
  'plain text': 'plaintext',
  'logout': 'log out', // Verb form
  'login': 'log in',   // Verb form
  'DB': 'database',    // Spell out acronyms in comments
};

function checkFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  
  lines.forEach((line, index) => {
    Object.keys(TERMINOLOGY_MAP).forEach(term => {
      if (line.toLowerCase().includes(term.toLowerCase())) {
        console.warn(
          `⚠️  ${filePath}:${index + 1}\n` +
          `   Found "${term}", consider using "${TERMINOLOGY_MAP[term]}"\n` +
          `   ${line.trim()}`
        );
      }
    });
  });
}

// Walk through source files
const srcPath = path.join(__dirname, '..', 'src');
// ... implementation to recursively check files
```

#### D. Code Commenting Guidelines

When writing code comments, reference glossary terms consistently:

```javascript
// BAD: "Check if the user can edit this post"
// GOOD: "Authorize the request: verify the authenticated user owns this post"
// BETTER: "Perform authorization check to prevent IDOR (Insecure Direct Object Reference)"
```

#### E. Language-Specific Hints for JavaScript/TypeScript

1. **TypeScript Interfaces**: Use glossary terms in interface names and comments:
   ```typescript
   /**
    * Represents a blog post with its author (eager loaded).
    * @see PostObjectWithAuthor in glossary
    */
   interface PostWithAuthor extends PostObject {
     author: User;
     html_content?: string;
   }
   ```

2. **JSDoc Comments**: Link to glossary terms using `@see` tags:
   ```javascript
   /**
    * Hashes a password using bcrypt with work factor 12.
    * @param {string} plaintext - The plaintext password (never store this!)
    * @returns {Promise<string>} The password hash for storage
    * @see bcrypt in glossary for security considerations
    */
   async function hashPassword(plaintext) {
     // Implementation
   }
   ```

#### F. Milestone Checkpoint for Terminology Consistency

After implementing each milestone, run a terminology check:

```bash
# Run the terminology checker
node scripts/check-terminology.js

# Expected output: warnings for any inconsistent terminology
# No output means all terminology is consistent

# Also verify that new terms introduced in the milestone are added to GLOSSARY.md
```

**Expected Behavior**: The script should flag any occurrences of discouraged terms (like "password" when referring to stored hashes) but not fail the build. Use this as a learning tool to reinforce consistent language.

**Signs of Issues**:
- Multiple warnings about the same term across many files
- Missing glossary entries for new concepts introduced in the milestone
- Inconsistent naming between code and documentation

#### G. Debugging Tips for Terminology Issues

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Team members use different terms for the same concept | Lack of shared vocabulary | Review code reviews and meeting notes for terminology inconsistencies | Update glossary and run terminology checker regularly |
| New developers struggle to understand documentation | Unfamiliar technical terms without definitions | Ask new developers to highlight unfamiliar terms in documentation | Add highlighted terms to glossary with simple explanations |
| API documentation doesn't match implementation | Different naming conventions between code and docs | Use automated API documentation generation from code comments | Enforce consistent naming via linter rules and code reviews |

> **Pro Tip**: Treat the glossary as a **living document**. When you encounter a new term during implementation that isn't documented, add it immediately. This practice ensures the documentation grows with the project and remains useful for all team members.

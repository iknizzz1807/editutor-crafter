#!/usr/bin/env python3
"""Add detailed milestones for new real-world projects."""

import yaml
from pathlib import Path

# Detailed project definitions with milestones
DETAILED_PROJECTS = {
    "graphql-server": {
        "name": "GraphQL Server",
        "description": "Build a production GraphQL API with schema-first design, resolvers, dataloaders for N+1 prevention, and real-time subscriptions",
        "category": "Backend",
        "difficulty": "intermediate",
        "estimated_hours": 40,
        "skills": [
            "GraphQL schema design",
            "Resolver patterns",
            "N+1 query prevention",
            "Real-time subscriptions",
            "Authentication in GraphQL",
            "Error handling"
        ],
        "prerequisites": ["REST API basics", "Database queries"],
        "learning_outcomes": [
            "Design type-safe GraphQL schemas",
            "Implement efficient data fetching with dataloaders",
            "Handle authentication and authorization in GraphQL",
            "Build real-time features with subscriptions"
        ],
        "milestones": [
            {
                "name": "Schema & Type System",
                "description": "Define GraphQL schema with types, queries, mutations",
                "skills": ["SDL", "Type definitions", "Schema design"],
                "deliverables": [
                    "GraphQL schema with Object types",
                    "Query type with field resolvers",
                    "Mutation type for CRUD operations",
                    "Input types for mutations",
                    "Custom scalar types (DateTime, JSON)",
                    "Enum types for fixed values"
                ],
                "hints": {
                    "level1": "Start with SDL (Schema Definition Language). Define your domain types first:\n```graphql\ntype User {\n  id: ID!\n  email: String!\n  posts: [Post!]!\n}\n```",
                    "level2": "Use input types for mutations to keep them clean:\n```graphql\ninput CreateUserInput {\n  email: String!\n  name: String!\n}\n\ntype Mutation {\n  createUser(input: CreateUserInput!): User!\n}\n```",
                    "level3": "For custom scalars, you need both schema definition and resolver:\n```javascript\nconst DateTimeScalar = new GraphQLScalarType({\n  name: 'DateTime',\n  serialize(value) { return value.toISOString(); },\n  parseValue(value) { return new Date(value); }\n});\n```"
                }
            },
            {
                "name": "Resolvers & Data Fetching",
                "description": "Implement resolvers that fetch data from database",
                "skills": ["Resolver functions", "Context", "Database queries"],
                "deliverables": [
                    "Root query resolvers",
                    "Field resolvers for nested types",
                    "Mutation resolvers with validation",
                    "Context setup with database connection",
                    "Error handling in resolvers",
                    "Resolver composition patterns"
                ],
                "hints": {
                    "level1": "Resolvers receive (parent, args, context, info). Use context for shared resources:\n```javascript\nconst resolvers = {\n  Query: {\n    user: (_, { id }, { db }) => db.users.findById(id)\n  }\n};\n```",
                    "level2": "Field resolvers handle nested data. They receive parent object:\n```javascript\nUser: {\n  posts: (user, _, { db }) => db.posts.findByUserId(user.id)\n}\n```",
                    "level3": "Use resolver middleware for cross-cutting concerns:\n```javascript\nconst authenticated = (resolver) => (parent, args, ctx, info) => {\n  if (!ctx.user) throw new AuthenticationError('Not logged in');\n  return resolver(parent, args, ctx, info);\n};\n```"
                }
            },
            {
                "name": "DataLoader & N+1 Prevention",
                "description": "Batch and cache database queries to prevent N+1 problem",
                "skills": ["DataLoader", "Batching", "Caching"],
                "deliverables": [
                    "DataLoader for each entity type",
                    "Batch function implementation",
                    "Per-request caching",
                    "DataLoader in resolver context",
                    "Handling batch errors",
                    "Cache invalidation strategy"
                ],
                "hints": {
                    "level1": "N+1 happens when fetching users[].posts makes N separate queries. DataLoader batches them:\n```javascript\nconst userLoader = new DataLoader(async (ids) => {\n  const users = await db.users.findByIds(ids);\n  return ids.map(id => users.find(u => u.id === id));\n});\n```",
                    "level2": "Create loaders per-request to avoid cache leaks:\n```javascript\nconst createLoaders = (db) => ({\n  userLoader: new DataLoader(ids => batchUsers(db, ids)),\n  postLoader: new DataLoader(ids => batchPosts(db, ids))\n});\n// In context: { loaders: createLoaders(db) }\n```",
                    "level3": "For has-many relations, batch by foreign key:\n```javascript\nconst postsByUserLoader = new DataLoader(async (userIds) => {\n  const posts = await db.posts.where('userId').in(userIds);\n  return userIds.map(id => posts.filter(p => p.userId === id));\n});\n```"
                }
            },
            {
                "name": "Subscriptions",
                "description": "Real-time updates via WebSocket subscriptions",
                "skills": ["WebSocket", "PubSub", "Event-driven"],
                "deliverables": [
                    "WebSocket transport setup",
                    "PubSub implementation",
                    "Subscription resolvers",
                    "Filtered subscriptions",
                    "Authentication for subscriptions",
                    "Connection lifecycle handling"
                ],
                "hints": {
                    "level1": "Subscriptions use AsyncIterator pattern:\n```javascript\nSubscription: {\n  postCreated: {\n    subscribe: () => pubsub.asyncIterator(['POST_CREATED'])\n  }\n}\n// Publish: pubsub.publish('POST_CREATED', { postCreated: post })\n```",
                    "level2": "Filter subscriptions by arguments:\n```javascript\npostCreated: {\n  subscribe: withFilter(\n    () => pubsub.asyncIterator(['POST_CREATED']),\n    (payload, variables) => payload.postCreated.authorId === variables.authorId\n  )\n}\n```",
                    "level3": "Handle connection auth in onConnect:\n```javascript\nnew WebSocketServer({\n  onConnect: async (connectionParams) => {\n    const token = connectionParams.authToken;\n    const user = await verifyToken(token);\n    return { user };\n  }\n});\n```"
                }
            }
        ]
    },
    "background-job-processor": {
        "name": "Background Job Processor",
        "description": "Build an async task queue system like Sidekiq/Celery with job scheduling, retries, priorities, and monitoring",
        "category": "Backend Infrastructure",
        "difficulty": "intermediate",
        "estimated_hours": 50,
        "skills": [
            "Message queues",
            "Worker processes",
            "Job scheduling",
            "Retry strategies",
            "Concurrency control",
            "Job persistence"
        ],
        "prerequisites": ["Redis basics", "Process management"],
        "learning_outcomes": [
            "Design reliable async job processing systems",
            "Implement exponential backoff and retry logic",
            "Handle job failures gracefully",
            "Build monitoring and observability for background jobs"
        ],
        "milestones": [
            {
                "name": "Job Queue Core",
                "description": "Basic job enqueueing and storage in Redis",
                "skills": ["Redis lists", "Job serialization", "Queue operations"],
                "deliverables": [
                    "Job class with serialization",
                    "Enqueue operation (LPUSH)",
                    "Multiple named queues",
                    "Job ID generation",
                    "Job payload validation",
                    "Queue inspection APIs"
                ],
                "hints": {
                    "level1": "Use Redis lists as queues. LPUSH to enqueue, BRPOP to dequeue:\n```python\nclass JobQueue:\n    def enqueue(self, job_class, *args):\n        job = {'id': uuid4(), 'class': job_class, 'args': args}\n        redis.lpush('queue:default', json.dumps(job))\n```",
                    "level2": "Support multiple queues with priorities:\n```python\nQUEUES = ['critical', 'default', 'low']\ndef dequeue(self):\n    # BRPOP blocks until job available, checks queues in order\n    queue, job = redis.brpop([f'queue:{q}' for q in QUEUES])\n    return json.loads(job)\n```",
                    "level3": "Add job metadata for tracking:\n```python\njob = {\n    'id': str(uuid4()),\n    'class': job_class.__name__,\n    'args': args,\n    'created_at': time.time(),\n    'retry_count': 0,\n    'max_retries': 3\n}\n```"
                }
            },
            {
                "name": "Worker Process",
                "description": "Worker that processes jobs from queue",
                "skills": ["Process management", "Job execution", "Error handling"],
                "deliverables": [
                    "Worker main loop",
                    "Job class registry and lookup",
                    "Job execution with timeout",
                    "Graceful shutdown (SIGTERM)",
                    "Worker heartbeat",
                    "Concurrent job processing"
                ],
                "hints": {
                    "level1": "Basic worker loop:\n```python\nclass Worker:\n    def run(self):\n        while self.running:\n            job = self.queue.dequeue(timeout=5)\n            if job:\n                self.process(job)\n    \n    def process(self, job):\n        klass = self.registry[job['class']]\n        klass().perform(*job['args'])\n```",
                    "level2": "Handle signals for graceful shutdown:\n```python\ndef run(self):\n    signal.signal(signal.SIGTERM, self.shutdown)\n    while self.running:\n        job = self.queue.dequeue(timeout=5)\n        if job:\n            self.current_job = job\n            self.process(job)\n            self.current_job = None\n```",
                    "level3": "Use thread/process pool for concurrency:\n```python\nfrom concurrent.futures import ThreadPoolExecutor\n\nclass Worker:\n    def __init__(self, concurrency=5):\n        self.executor = ThreadPoolExecutor(max_workers=concurrency)\n    \n    def run(self):\n        while self.running:\n            job = self.queue.dequeue()\n            self.executor.submit(self.process, job)\n```"
                }
            },
            {
                "name": "Retry & Error Handling",
                "description": "Automatic retries with exponential backoff",
                "skills": ["Retry strategies", "Dead letter queue", "Error tracking"],
                "deliverables": [
                    "Exponential backoff calculation",
                    "Retry queue with scheduled execution",
                    "Dead letter queue for failed jobs",
                    "Error serialization and storage",
                    "Max retry limits",
                    "Custom retry strategies per job"
                ],
                "hints": {
                    "level1": "Calculate exponential backoff delay:\n```python\ndef backoff_delay(retry_count):\n    # 15s, 1m, 4m, 15m, 1h...\n    return min(15 * (4 ** retry_count), 86400)\n```",
                    "level2": "Use Redis sorted sets for scheduled retries:\n```python\ndef retry_later(self, job, error):\n    job['retry_count'] += 1\n    job['error'] = str(error)\n    execute_at = time.time() + backoff_delay(job['retry_count'])\n    redis.zadd('queue:retry', {json.dumps(job): execute_at})\n```",
                    "level3": "Move to dead letter queue after max retries:\n```python\ndef handle_failure(self, job, error):\n    if job['retry_count'] >= job['max_retries']:\n        redis.lpush('queue:dead', json.dumps(job))\n    else:\n        self.retry_later(job, error)\n```"
                }
            },
            {
                "name": "Scheduling & Cron",
                "description": "Schedule jobs for future execution or recurring",
                "skills": ["Cron parsing", "Scheduled jobs", "Time handling"],
                "deliverables": [
                    "Schedule job for specific time",
                    "Recurring job definitions",
                    "Cron expression parser",
                    "Scheduler process",
                    "Timezone handling",
                    "Unique job constraints"
                ],
                "hints": {
                    "level1": "Use sorted set for scheduled jobs:\n```python\ndef schedule(self, job_class, args, run_at):\n    job = self.create_job(job_class, args)\n    redis.zadd('queue:scheduled', {json.dumps(job): run_at.timestamp()})\n```",
                    "level2": "Scheduler moves due jobs to work queue:\n```python\ndef poll_scheduled(self):\n    now = time.time()\n    jobs = redis.zrangebyscore('queue:scheduled', 0, now)\n    for job in jobs:\n        redis.lpush('queue:default', job)\n        redis.zrem('queue:scheduled', job)\n```",
                    "level3": "For cron jobs, calculate next run time:\n```python\nfrom croniter import croniter\n\ndef schedule_cron(self, job_class, cron_expr):\n    cron = croniter(cron_expr)\n    next_run = cron.get_next(datetime)\n    self.schedule(job_class, [], next_run)\n    # Re-schedule after execution\n```"
                }
            },
            {
                "name": "Monitoring & Dashboard",
                "description": "Real-time monitoring and web dashboard",
                "skills": ["Metrics", "Web UI", "Real-time updates"],
                "deliverables": [
                    "Job counts per queue",
                    "Worker status tracking",
                    "Job history and logs",
                    "Failure rate metrics",
                    "Web dashboard UI",
                    "Retry/delete failed jobs manually"
                ],
                "hints": {
                    "level1": "Track metrics in Redis:\n```python\ndef record_processed(self, job, duration):\n    redis.incr('stats:processed')\n    redis.incr(f'stats:processed:{date.today()}')\n    redis.lpush('stats:recent', json.dumps({\n        'job_id': job['id'], 'duration': duration\n    }))\n```",
                    "level2": "Worker heartbeat for status:\n```python\ndef heartbeat(self):\n    redis.hset('workers', self.id, json.dumps({\n        'pid': os.getpid(),\n        'queues': self.queues,\n        'current_job': self.current_job,\n        'last_seen': time.time()\n    }))\n```",
                    "level3": "Web dashboard endpoint:\n```python\n@app.get('/dashboard')\ndef dashboard():\n    return {\n        'queues': {q: redis.llen(f'queue:{q}') for q in QUEUES},\n        'workers': redis.hgetall('workers'),\n        'processed_today': redis.get(f'stats:processed:{date.today()}'),\n        'failed': redis.llen('queue:dead')\n    }\n```"
                }
            }
        ]
    },
    "multi-tenant-saas": {
        "name": "Multi-tenant SaaS Backend",
        "description": "Build a multi-tenant architecture with tenant isolation, row-level security, and per-tenant customization",
        "category": "Backend Architecture",
        "difficulty": "advanced",
        "estimated_hours": 60,
        "skills": [
            "Multi-tenancy patterns",
            "Row-level security",
            "Tenant isolation",
            "Database design",
            "Request context",
            "Billing integration"
        ],
        "prerequisites": ["REST API", "Database design", "Authentication"],
        "learning_outcomes": [
            "Design scalable multi-tenant architectures",
            "Implement secure tenant data isolation",
            "Handle tenant-specific customizations",
            "Build usage-based billing systems"
        ],
        "milestones": [
            {
                "name": "Tenant Data Model",
                "description": "Design database schema for multi-tenancy",
                "skills": ["Schema design", "Foreign keys", "Indexes"],
                "deliverables": [
                    "Tenant table with settings",
                    "tenant_id column on all tables",
                    "Composite indexes with tenant_id",
                    "Tenant creation and setup",
                    "Tenant subdomain/slug mapping",
                    "Soft delete for tenant data"
                ],
                "hints": {
                    "level1": "Add tenant_id to every table:\n```sql\nCREATE TABLE users (\n    id UUID PRIMARY KEY,\n    tenant_id UUID NOT NULL REFERENCES tenants(id),\n    email VARCHAR(255) NOT NULL,\n    UNIQUE(tenant_id, email)\n);\nCREATE INDEX idx_users_tenant ON users(tenant_id);\n```",
                    "level2": "Tenant table stores configuration:\n```sql\nCREATE TABLE tenants (\n    id UUID PRIMARY KEY,\n    slug VARCHAR(63) UNIQUE NOT NULL,\n    name VARCHAR(255),\n    settings JSONB DEFAULT '{}',\n    plan VARCHAR(50) DEFAULT 'free',\n    created_at TIMESTAMP DEFAULT NOW()\n);\n```",
                    "level3": "Use composite primary keys for better isolation:\n```sql\nCREATE TABLE projects (\n    tenant_id UUID NOT NULL,\n    id UUID NOT NULL,\n    name VARCHAR(255),\n    PRIMARY KEY (tenant_id, id),\n    FOREIGN KEY (tenant_id) REFERENCES tenants(id)\n);\n```"
                }
            },
            {
                "name": "Request Context & Isolation",
                "description": "Automatic tenant context injection in requests",
                "skills": ["Middleware", "Context management", "ORM hooks"],
                "deliverables": [
                    "Tenant resolution from subdomain/header",
                    "Request-scoped tenant context",
                    "Automatic tenant_id injection in queries",
                    "Tenant validation middleware",
                    "Cross-tenant access prevention",
                    "Admin/superuser bypass"
                ],
                "hints": {
                    "level1": "Resolve tenant from subdomain:\n```python\n@app.middleware('http')\nasync def tenant_middleware(request, call_next):\n    host = request.headers.get('host', '')\n    subdomain = host.split('.')[0]\n    tenant = await get_tenant_by_slug(subdomain)\n    request.state.tenant = tenant\n    return await call_next(request)\n```",
                    "level2": "Use context variables for tenant:\n```python\nfrom contextvars import ContextVar\ncurrent_tenant: ContextVar[Tenant] = ContextVar('tenant')\n\n# In middleware\ncurrent_tenant.set(tenant)\n\n# In any code\ntenant = current_tenant.get()\n```",
                    "level3": "SQLAlchemy event to auto-filter:\n```python\n@event.listens_for(Session, 'do_orm_execute')\ndef add_tenant_filter(orm_execute_state):\n    if orm_execute_state.is_select:\n        tenant = current_tenant.get(None)\n        if tenant:\n            orm_execute_state.statement = orm_execute_state.statement.filter_by(\n                tenant_id=tenant.id\n            )\n```"
                }
            },
            {
                "name": "Row-Level Security",
                "description": "Database-level tenant isolation with RLS",
                "skills": ["PostgreSQL RLS", "Policies", "Session variables"],
                "deliverables": [
                    "Enable RLS on tenant tables",
                    "Create tenant isolation policies",
                    "Set tenant context in session",
                    "Policy for SELECT/INSERT/UPDATE/DELETE",
                    "Bypass for migrations",
                    "Testing RLS policies"
                ],
                "hints": {
                    "level1": "Enable RLS and create policy:\n```sql\nALTER TABLE users ENABLE ROW LEVEL SECURITY;\n\nCREATE POLICY tenant_isolation ON users\n    USING (tenant_id = current_setting('app.tenant_id')::uuid);\n```",
                    "level2": "Set tenant in connection:\n```python\nasync def set_tenant_context(conn, tenant_id):\n    await conn.execute(\n        f\"SET app.tenant_id = '{tenant_id}'\"\n    )\n```",
                    "level3": "Separate policies for operations:\n```sql\nCREATE POLICY tenant_select ON users FOR SELECT\n    USING (tenant_id = current_setting('app.tenant_id')::uuid);\n\nCREATE POLICY tenant_insert ON users FOR INSERT\n    WITH CHECK (tenant_id = current_setting('app.tenant_id')::uuid);\n```"
                }
            },
            {
                "name": "Tenant Customization",
                "description": "Per-tenant features, branding, and configuration",
                "skills": ["Feature flags", "Configuration", "Theming"],
                "deliverables": [
                    "Tenant settings schema",
                    "Feature flag system per tenant",
                    "Plan-based feature access",
                    "Custom branding (logo, colors)",
                    "Tenant-specific webhooks",
                    "Settings API and UI"
                ],
                "hints": {
                    "level1": "Store settings as JSONB:\n```python\nclass Tenant:\n    settings = Column(JSONB, default={})\n    \n    def get_setting(self, key, default=None):\n        return self.settings.get(key, default)\n    \n    def has_feature(self, feature):\n        return feature in self.plan_features\n```",
                    "level2": "Plan-based feature access:\n```python\nPLAN_FEATURES = {\n    'free': ['basic_reports'],\n    'pro': ['basic_reports', 'api_access', 'custom_domain'],\n    'enterprise': ['basic_reports', 'api_access', 'custom_domain', 'sso', 'audit_logs']\n}\n\ndef require_feature(feature):\n    def decorator(f):\n        def wrapper(*args, **kwargs):\n            if feature not in PLAN_FEATURES[current_tenant.get().plan]:\n                raise UpgradeRequired(feature)\n            return f(*args, **kwargs)\n        return wrapper\n    return decorator\n```",
                    "level3": "Tenant-specific configuration override:\n```python\ndef get_config(key):\n    tenant = current_tenant.get(None)\n    if tenant and key in tenant.settings:\n        return tenant.settings[key]\n    return app.config[key]  # Fall back to default\n```"
                }
            },
            {
                "name": "Usage Tracking & Billing",
                "description": "Track usage metrics and integrate with billing",
                "skills": ["Metering", "Billing APIs", "Usage limits"],
                "deliverables": [
                    "Usage event tracking",
                    "Usage aggregation per tenant",
                    "Plan limits enforcement",
                    "Stripe/billing integration",
                    "Usage-based pricing calculation",
                    "Overage handling"
                ],
                "hints": {
                    "level1": "Track usage events:\n```python\nasync def track_usage(tenant_id, metric, quantity=1):\n    key = f'usage:{tenant_id}:{metric}:{date.today()}'\n    await redis.incrby(key, quantity)\n    await redis.expire(key, 86400 * 90)  # Keep 90 days\n```",
                    "level2": "Enforce limits before operations:\n```python\nasync def check_limit(tenant, metric, requested=1):\n    current = await get_usage(tenant.id, metric)\n    limit = PLAN_LIMITS[tenant.plan][metric]\n    if current + requested > limit:\n        raise LimitExceeded(metric, current, limit)\n```",
                    "level3": "Sync usage to Stripe for billing:\n```python\nasync def sync_usage_to_stripe(tenant):\n    usage = await get_monthly_usage(tenant.id)\n    stripe.SubscriptionItem.create_usage_record(\n        tenant.stripe_subscription_item_id,\n        quantity=usage['api_calls'],\n        timestamp=int(time.time()),\n        action='set'\n    )\n```"
                }
            }
        ]
    },
    "vector-database": {
        "name": "Vector Database",
        "description": "Build a vector similarity search database with HNSW indexing for AI/ML embeddings",
        "category": "Data Storage",
        "difficulty": "advanced",
        "estimated_hours": 80,
        "skills": [
            "Vector similarity",
            "HNSW algorithm",
            "Distance metrics",
            "Approximate nearest neighbor",
            "Memory-mapped storage",
            "Index persistence"
        ],
        "prerequisites": ["Data structures", "Linear algebra basics"],
        "learning_outcomes": [
            "Understand vector similarity search algorithms",
            "Implement HNSW for efficient ANN search",
            "Design memory-efficient vector storage",
            "Build production-ready vector search APIs"
        ],
        "milestones": [
            {
                "name": "Vector Storage",
                "description": "Efficient storage and retrieval of vectors",
                "skills": ["Memory layout", "Serialization", "Memory mapping"],
                "deliverables": [
                    "Fixed-dimension vector storage",
                    "Vector ID mapping",
                    "Memory-mapped file storage",
                    "Batch insert operations",
                    "Vector retrieval by ID",
                    "Storage compaction"
                ],
                "hints": {
                    "level1": "Use numpy for efficient vector storage:\n```python\nclass VectorStore:\n    def __init__(self, dim):\n        self.dim = dim\n        self.vectors = np.zeros((0, dim), dtype=np.float32)\n        self.ids = []\n    \n    def add(self, id, vector):\n        self.ids.append(id)\n        self.vectors = np.vstack([self.vectors, vector])\n```",
                    "level2": "Memory-map for large datasets:\n```python\ndef create_mmap_storage(path, dim, capacity):\n    # 4 bytes per float32\n    size = capacity * dim * 4\n    fp = np.memmap(path, dtype=np.float32, mode='w+', shape=(capacity, dim))\n    return fp\n```",
                    "level3": "Use struct for metadata header:\n```python\nimport struct\nHEADER_FORMAT = 'IIQ'  # dim, count, capacity\n\ndef write_header(f, dim, count, capacity):\n    f.write(struct.pack(HEADER_FORMAT, dim, count, capacity))\n```"
                }
            },
            {
                "name": "Distance Metrics",
                "description": "Implement various similarity measures",
                "skills": ["Cosine similarity", "Euclidean distance", "Dot product"],
                "deliverables": [
                    "Cosine similarity",
                    "Euclidean (L2) distance",
                    "Dot product similarity",
                    "SIMD-optimized implementations",
                    "Batch distance computation",
                    "Normalized vectors handling"
                ],
                "hints": {
                    "level1": "Basic distance functions:\n```python\ndef cosine_similarity(a, b):\n    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))\n\ndef euclidean_distance(a, b):\n    return np.linalg.norm(a - b)\n```",
                    "level2": "Batch computation is much faster:\n```python\ndef batch_cosine(query, vectors):\n    # Normalize\n    query_norm = query / np.linalg.norm(query)\n    vec_norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)\n    return np.dot(vec_norms, query_norm)\n```",
                    "level3": "Pre-normalize for faster cosine:\n```python\nclass VectorStore:\n    def add(self, id, vector, normalize=True):\n        if normalize:\n            vector = vector / np.linalg.norm(vector)\n        # Now cosine = dot product\n```"
                }
            },
            {
                "name": "Brute Force Search",
                "description": "Exact nearest neighbor search baseline",
                "skills": ["Linear scan", "Top-K selection", "Filtering"],
                "deliverables": [
                    "K-nearest neighbors search",
                    "Threshold-based search",
                    "Metadata filtering",
                    "Batch query support",
                    "Search with exclusions",
                    "Performance benchmarking"
                ],
                "hints": {
                    "level1": "Simple brute force search:\n```python\ndef search(self, query, k=10):\n    distances = np.dot(self.vectors, query)  # Assuming normalized\n    indices = np.argsort(distances)[-k:][::-1]\n    return [(self.ids[i], distances[i]) for i in indices]\n```",
                    "level2": "Use argpartition for efficiency:\n```python\ndef search(self, query, k=10):\n    distances = np.dot(self.vectors, query)\n    # argpartition is O(n) vs O(n log n) for argsort\n    indices = np.argpartition(distances, -k)[-k:]\n    indices = indices[np.argsort(distances[indices])[::-1]]\n    return [(self.ids[i], distances[i]) for i in indices]\n```",
                    "level3": "Add metadata filtering:\n```python\ndef search(self, query, k=10, filter_fn=None):\n    distances = np.dot(self.vectors, query)\n    if filter_fn:\n        mask = np.array([filter_fn(self.metadata[i]) for i in range(len(self.ids))])\n        distances = np.where(mask, distances, -np.inf)\n    # ... continue with top-k\n```"
                }
            },
            {
                "name": "HNSW Index",
                "description": "Hierarchical Navigable Small World graph for ANN",
                "skills": ["Graph construction", "Layer navigation", "Greedy search"],
                "deliverables": [
                    "Multi-layer graph structure",
                    "Node insertion with level assignment",
                    "Greedy search within layer",
                    "Layer traversal strategy",
                    "ef_construction parameter",
                    "Index serialization"
                ],
                "hints": {
                    "level1": "HNSW node structure:\n```python\nclass HNSWNode:\n    def __init__(self, id, vector, level):\n        self.id = id\n        self.vector = vector\n        self.level = level\n        self.neighbors = {l: [] for l in range(level + 1)}  # neighbors per layer\n```",
                    "level2": "Level assignment (exponential decay):\n```python\ndef random_level(self, ml=0.5):\n    level = 0\n    while random.random() < ml and level < self.max_level:\n        level += 1\n    return level\n```",
                    "level3": "Greedy search in layer:\n```python\ndef search_layer(self, query, entry_point, ef, layer):\n    visited = {entry_point.id}\n    candidates = [(-distance(query, entry_point.vector), entry_point)]\n    results = [(-distance(query, entry_point.vector), entry_point)]\n    \n    while candidates:\n        _, current = heapq.heappop(candidates)\n        if -candidates[0][0] > -results[0][0]:\n            break\n        for neighbor in current.neighbors[layer]:\n            if neighbor.id not in visited:\n                visited.add(neighbor.id)\n                d = distance(query, neighbor.vector)\n                if d < -results[0][0] or len(results) < ef:\n                    heapq.heappush(candidates, (-d, neighbor))\n                    heapq.heappush(results, (-d, neighbor))\n                    if len(results) > ef:\n                        heapq.heappop(results)\n    return results\n```"
                }
            },
            {
                "name": "Query API & Server",
                "description": "REST/gRPC API for vector operations",
                "skills": ["API design", "Batch operations", "Concurrent access"],
                "deliverables": [
                    "Insert/upsert vectors API",
                    "Search API with filters",
                    "Delete vectors",
                    "Collection management",
                    "Concurrent read/write handling",
                    "Batch operations"
                ],
                "hints": {
                    "level1": "REST API endpoints:\n```python\n@app.post('/collections/{name}/vectors')\nasync def upsert(name: str, vectors: List[VectorInput]):\n    collection = get_collection(name)\n    for v in vectors:\n        collection.upsert(v.id, v.values, v.metadata)\n    return {'upserted': len(vectors)}\n\n@app.post('/collections/{name}/search')\nasync def search(name: str, query: SearchQuery):\n    collection = get_collection(name)\n    results = collection.search(query.vector, k=query.top_k)\n    return {'results': results}\n```",
                    "level2": "Handle concurrent writes with locks:\n```python\nfrom asyncio import Lock\n\nclass Collection:\n    def __init__(self):\n        self.write_lock = Lock()\n    \n    async def upsert(self, id, vector, metadata):\n        async with self.write_lock:\n            # Rebuild HNSW node connections\n            self._insert(id, vector, metadata)\n```",
                    "level3": "Batch for efficiency:\n```python\n@app.post('/collections/{name}/vectors/batch')\nasync def batch_upsert(name: str, vectors: List[VectorInput]):\n    collection = get_collection(name)\n    async with collection.write_lock:\n        for v in vectors:\n            collection._insert(v.id, v.values, v.metadata)\n        collection._rebuild_index()  # Batch rebuild\n    return {'upserted': len(vectors)}\n```"
                }
            }
        ]
    },
    "workflow-orchestrator": {
        "name": "Workflow Orchestrator",
        "description": "Build a DAG-based workflow orchestration system like Airflow with scheduling, dependencies, and monitoring",
        "category": "Data Engineering",
        "difficulty": "advanced",
        "estimated_hours": 70,
        "skills": [
            "DAG scheduling",
            "Task dependencies",
            "State management",
            "Distributed execution",
            "Failure handling",
            "Monitoring"
        ],
        "prerequisites": ["Background jobs", "Database", "Process management"],
        "learning_outcomes": [
            "Design DAG-based workflow systems",
            "Implement task dependency resolution",
            "Handle failures and retries in pipelines",
            "Build workflow monitoring and alerting"
        ],
        "milestones": [
            {
                "name": "DAG Definition",
                "description": "Define workflows as directed acyclic graphs",
                "skills": ["Graph structures", "DSL design", "Validation"],
                "deliverables": [
                    "Task class definition",
                    "DAG class with dependency tracking",
                    "Operator types (Python, Bash, SQL)",
                    "DAG validation (cycle detection)",
                    "Task parameters and templating",
                    "DAG file discovery"
                ],
                "hints": {
                    "level1": "Basic task and DAG definition:\n```python\nclass Task:\n    def __init__(self, task_id, callable, dag=None):\n        self.task_id = task_id\n        self.callable = callable\n        self.upstream = []\n        self.downstream = []\n        if dag:\n            dag.add_task(self)\n\nclass DAG:\n    def __init__(self, dag_id, schedule=None):\n        self.dag_id = dag_id\n        self.schedule = schedule\n        self.tasks = {}\n```",
                    "level2": "Dependency operator overloading:\n```python\nclass Task:\n    def __rshift__(self, other):  # task1 >> task2\n        self.downstream.append(other)\n        other.upstream.append(self)\n        return other\n    \n    def __lshift__(self, other):  # task1 << task2\n        self.upstream.append(other)\n        other.downstream.append(self)\n        return self\n\n# Usage: extract >> transform >> load\n```",
                    "level3": "Cycle detection with DFS:\n```python\ndef validate_dag(self):\n    visited = set()\n    rec_stack = set()\n    \n    def has_cycle(task_id):\n        visited.add(task_id)\n        rec_stack.add(task_id)\n        for downstream in self.tasks[task_id].downstream:\n            if downstream.task_id not in visited:\n                if has_cycle(downstream.task_id):\n                    return True\n            elif downstream.task_id in rec_stack:\n                return True\n        rec_stack.remove(task_id)\n        return False\n    \n    for task_id in self.tasks:\n        if task_id not in visited:\n            if has_cycle(task_id):\n                raise CycleDetected()\n```"
                }
            },
            {
                "name": "Scheduler",
                "description": "Schedule DAG runs based on cron or triggers",
                "skills": ["Cron parsing", "Run scheduling", "Backfill"],
                "deliverables": [
                    "Cron-based scheduling",
                    "Manual trigger support",
                    "DAG run creation",
                    "Execution date handling",
                    "Backfill support",
                    "Catchup behavior"
                ],
                "hints": {
                    "level1": "Scheduler main loop:\n```python\nclass Scheduler:\n    def run(self):\n        while True:\n            for dag in self.discover_dags():\n                if self.should_run(dag):\n                    self.create_dag_run(dag)\n            self.dispatch_ready_tasks()\n            time.sleep(self.heartbeat_interval)\n```",
                    "level2": "Calculate next run time:\n```python\nfrom croniter import croniter\n\ndef get_next_run(self, dag):\n    last_run = self.get_last_run(dag.dag_id)\n    if last_run:\n        cron = croniter(dag.schedule, last_run.execution_date)\n    else:\n        cron = croniter(dag.schedule, dag.start_date)\n    return cron.get_next(datetime)\n```",
                    "level3": "Backfill past runs:\n```python\ndef backfill(self, dag, start_date, end_date):\n    cron = croniter(dag.schedule, start_date)\n    while True:\n        execution_date = cron.get_next(datetime)\n        if execution_date > end_date:\n            break\n        if not self.run_exists(dag.dag_id, execution_date):\n            self.create_dag_run(dag, execution_date)\n```"
                }
            },
            {
                "name": "Task Execution",
                "description": "Execute tasks with dependency resolution",
                "skills": ["Topological sort", "Parallel execution", "State management"],
                "deliverables": [
                    "Task instance state machine",
                    "Dependency checking",
                    "Parallel task execution",
                    "Task queuing to workers",
                    "XCom for task communication",
                    "Task timeout handling"
                ],
                "hints": {
                    "level1": "Task states:\n```python\nclass TaskState(Enum):\n    PENDING = 'pending'\n    QUEUED = 'queued'\n    RUNNING = 'running'\n    SUCCESS = 'success'\n    FAILED = 'failed'\n    SKIPPED = 'skipped'\n    UP_FOR_RETRY = 'up_for_retry'\n```",
                    "level2": "Check if task is ready to run:\n```python\ndef is_ready(self, task_instance):\n    for upstream in task_instance.task.upstream:\n        upstream_ti = self.get_task_instance(\n            upstream.task_id, \n            task_instance.dag_run_id\n        )\n        if upstream_ti.state != TaskState.SUCCESS:\n            return False\n    return True\n```",
                    "level3": "XCom for inter-task communication:\n```python\nclass XCom:\n    @staticmethod\n    def push(task_id, dag_run_id, key, value):\n        db.xcom.insert({\n            'task_id': task_id,\n            'dag_run_id': dag_run_id,\n            'key': key,\n            'value': pickle.dumps(value)\n        })\n    \n    @staticmethod\n    def pull(task_id, dag_run_id, key):\n        row = db.xcom.find_one(task_id=task_id, dag_run_id=dag_run_id, key=key)\n        return pickle.loads(row['value'])\n```"
                }
            },
            {
                "name": "Worker & Executor",
                "description": "Distributed task execution across workers",
                "skills": ["Worker processes", "Task distribution", "Resource management"],
                "deliverables": [
                    "Local executor (sequential)",
                    "Celery/Redis executor",
                    "Worker heartbeat",
                    "Task result collection",
                    "Worker pools",
                    "Resource slots management"
                ],
                "hints": {
                    "level1": "Base executor interface:\n```python\nclass BaseExecutor:\n    def execute(self, task_instance):\n        raise NotImplementedError\n    \n    def get_result(self, task_instance):\n        raise NotImplementedError\n\nclass LocalExecutor(BaseExecutor):\n    def execute(self, ti):\n        try:\n            result = ti.task.callable()\n            ti.state = TaskState.SUCCESS\n            return result\n        except Exception as e:\n            ti.state = TaskState.FAILED\n            raise\n```",
                    "level2": "Celery executor:\n```python\nfrom celery import Celery\n\napp = Celery('workflow', broker='redis://localhost')\n\n@app.task\ndef execute_task(task_id, dag_run_id):\n    ti = TaskInstance.get(task_id, dag_run_id)\n    return ti.task.callable()\n\nclass CeleryExecutor(BaseExecutor):\n    def execute(self, ti):\n        result = execute_task.delay(ti.task_id, ti.dag_run_id)\n        ti.celery_task_id = result.id\n```",
                    "level3": "Parallelism control:\n```python\nclass Executor:\n    def __init__(self, parallelism=16):\n        self.parallelism = parallelism\n        self.running_tasks = {}\n    \n    def has_slot(self):\n        return len(self.running_tasks) < self.parallelism\n    \n    def dispatch(self, ti):\n        if not self.has_slot():\n            return False\n        self.running_tasks[ti.key] = self.execute_async(ti)\n        return True\n```"
                }
            },
            {
                "name": "Web UI & Monitoring",
                "description": "Dashboard for workflow monitoring and management",
                "skills": ["Web UI", "Real-time updates", "Logging"],
                "deliverables": [
                    "DAG list view",
                    "DAG graph visualization",
                    "Task logs viewing",
                    "Manual trigger/clear",
                    "Run history",
                    "Alerting on failures"
                ],
                "hints": {
                    "level1": "API endpoints:\n```python\n@app.get('/api/dags')\ndef list_dags():\n    return [{'dag_id': d.dag_id, 'schedule': d.schedule} for d in discover_dags()]\n\n@app.get('/api/dags/{dag_id}/runs')\ndef get_runs(dag_id: str):\n    return db.dag_runs.find(dag_id=dag_id).order_by('-execution_date')\n```",
                    "level2": "Graph data for visualization:\n```python\n@app.get('/api/dags/{dag_id}/graph')\ndef get_graph(dag_id: str):\n    dag = get_dag(dag_id)\n    return {\n        'nodes': [{'id': t.task_id} for t in dag.tasks.values()],\n        'edges': [\n            {'source': t.task_id, 'target': d.task_id}\n            for t in dag.tasks.values()\n            for d in t.downstream\n        ]\n    }\n```",
                    "level3": "Stream logs:\n```python\n@app.get('/api/tasks/{task_id}/logs')\nasync def stream_logs(task_id: str, dag_run_id: str):\n    async def generate():\n        log_file = get_log_path(task_id, dag_run_id)\n        async with aiofiles.open(log_file) as f:\n            while True:\n                line = await f.readline()\n                if line:\n                    yield f'data: {line}\\n\\n'\n                else:\n                    await asyncio.sleep(0.5)\n    return StreamingResponse(generate(), media_type='text/event-stream')\n```"
                }
            }
        ]
    },
    "kubernetes-operator": {
        "name": "Kubernetes Operator",
        "description": "Build a Kubernetes operator with custom resources (CRDs) for automated application management",
        "category": "Cloud Native",
        "difficulty": "advanced",
        "estimated_hours": 60,
        "skills": [
            "Kubernetes API",
            "Custom Resources",
            "Controller pattern",
            "Reconciliation loop",
            "Leader election",
            "Webhook validation"
        ],
        "prerequisites": ["Kubernetes basics", "Go or Python"],
        "learning_outcomes": [
            "Understand Kubernetes controller patterns",
            "Implement CRDs and controllers",
            "Handle reconciliation and state management",
            "Deploy and operate custom operators"
        ],
        "milestones": [
            {
                "name": "Custom Resource Definition",
                "description": "Define custom resources for your domain",
                "skills": ["CRD schema", "OpenAPI validation", "Versioning"],
                "deliverables": [
                    "CRD YAML with schema",
                    "Status subresource",
                    "Printer columns",
                    "Validation rules",
                    "Default values",
                    "Multiple versions"
                ],
                "hints": {
                    "level1": "Basic CRD structure:\n```yaml\napiVersion: apiextensions.k8s.io/v1\nkind: CustomResourceDefinition\nmetadata:\n  name: myapps.example.com\nspec:\n  group: example.com\n  names:\n    kind: MyApp\n    plural: myapps\n  scope: Namespaced\n  versions:\n  - name: v1\n    served: true\n    storage: true\n    schema:\n      openAPIV3Schema:\n        type: object\n        properties:\n          spec:\n            type: object\n            properties:\n              replicas:\n                type: integer\n```",
                    "level2": "Add status subresource:\n```yaml\nversions:\n- name: v1\n  subresources:\n    status: {}\n  schema:\n    openAPIV3Schema:\n      properties:\n        status:\n          type: object\n          properties:\n            availableReplicas:\n              type: integer\n            conditions:\n              type: array\n              items:\n                type: object\n```",
                    "level3": "Printer columns for kubectl:\n```yaml\nadditionalPrinterColumns:\n- name: Replicas\n  type: integer\n  jsonPath: .spec.replicas\n- name: Available\n  type: integer\n  jsonPath: .status.availableReplicas\n- name: Age\n  type: date\n  jsonPath: .metadata.creationTimestamp\n```"
                }
            },
            {
                "name": "Controller Setup",
                "description": "Set up controller with client and informers",
                "skills": ["Client-go", "Informers", "Work queue"],
                "deliverables": [
                    "Kubernetes client setup",
                    "Informer for custom resource",
                    "Work queue for events",
                    "Event handlers",
                    "Controller struct",
                    "Run loop"
                ],
                "hints": {
                    "level1": "Controller structure (Go):\n```go\ntype Controller struct {\n    clientset    kubernetes.Interface\n    myappLister  listers.MyAppLister\n    myappsSynced cache.InformerSynced\n    workqueue    workqueue.RateLimitingInterface\n}\n```",
                    "level2": "Add event handlers:\n```go\nmyappInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{\n    AddFunc: func(obj interface{}) {\n        key, _ := cache.MetaNamespaceKeyFunc(obj)\n        c.workqueue.Add(key)\n    },\n    UpdateFunc: func(old, new interface{}) {\n        key, _ := cache.MetaNamespaceKeyFunc(new)\n        c.workqueue.Add(key)\n    },\n    DeleteFunc: func(obj interface{}) {\n        key, _ := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)\n        c.workqueue.Add(key)\n    },\n})\n```",
                    "level3": "Run workers:\n```go\nfunc (c *Controller) Run(workers int, stopCh <-chan struct{}) error {\n    if !cache.WaitForCacheSync(stopCh, c.myappsSynced) {\n        return fmt.Errorf(\"cache sync failed\")\n    }\n    for i := 0; i < workers; i++ {\n        go wait.Until(c.runWorker, time.Second, stopCh)\n    }\n    <-stopCh\n    return nil\n}\n```"
                }
            },
            {
                "name": "Reconciliation Loop",
                "description": "Implement the core reconciliation logic",
                "skills": ["Desired vs actual state", "Idempotency", "Error handling"],
                "deliverables": [
                    "Fetch current state",
                    "Compare with desired state",
                    "Create/Update/Delete resources",
                    "Update status",
                    "Requeue on error",
                    "Idempotent operations"
                ],
                "hints": {
                    "level1": "Basic reconcile function:\n```go\nfunc (c *Controller) reconcile(key string) error {\n    namespace, name, _ := cache.SplitMetaNamespaceKey(key)\n    \n    myapp, err := c.myappLister.MyApps(namespace).Get(name)\n    if errors.IsNotFound(err) {\n        return nil  // Deleted\n    }\n    \n    // Reconcile deployment\n    deployment := newDeployment(myapp)\n    _, err = c.clientset.AppsV1().Deployments(namespace).Create(ctx, deployment, metav1.CreateOptions{})\n    if errors.IsAlreadyExists(err) {\n        _, err = c.clientset.AppsV1().Deployments(namespace).Update(ctx, deployment, metav1.UpdateOptions{})\n    }\n    return err\n}\n```",
                    "level2": "Update status:\n```go\nfunc (c *Controller) updateStatus(myapp *v1.MyApp, deployment *appsv1.Deployment) error {\n    myappCopy := myapp.DeepCopy()\n    myappCopy.Status.AvailableReplicas = deployment.Status.AvailableReplicas\n    myappCopy.Status.Conditions = append(myappCopy.Status.Conditions, v1.Condition{\n        Type:               \"Available\",\n        Status:             \"True\",\n        LastTransitionTime: metav1.Now(),\n    })\n    _, err := c.myappclient.MyApps(myapp.Namespace).UpdateStatus(ctx, myappCopy, metav1.UpdateOptions{})\n    return err\n}\n```",
                    "level3": "Owner references for garbage collection:\n```go\nfunc newDeployment(myapp *v1.MyApp) *appsv1.Deployment {\n    return &appsv1.Deployment{\n        ObjectMeta: metav1.ObjectMeta{\n            Name:      myapp.Name,\n            Namespace: myapp.Namespace,\n            OwnerReferences: []metav1.OwnerReference{\n                *metav1.NewControllerRef(myapp, v1.SchemeGroupVersion.WithKind(\"MyApp\")),\n            },\n        },\n        // ...\n    }\n}\n```"
                }
            },
            {
                "name": "Webhooks",
                "description": "Admission webhooks for validation and mutation",
                "skills": ["Validating webhook", "Mutating webhook", "TLS"],
                "deliverables": [
                    "Webhook server",
                    "Validating webhook",
                    "Mutating webhook (defaults)",
                    "TLS certificate setup",
                    "Webhook configuration",
                    "Error responses"
                ],
                "hints": {
                    "level1": "Webhook handler:\n```go\nfunc (h *Handler) validate(ar *admissionv1.AdmissionReview) *admissionv1.AdmissionResponse {\n    myapp := &v1.MyApp{}\n    json.Unmarshal(ar.Request.Object.Raw, myapp)\n    \n    if myapp.Spec.Replicas < 1 {\n        return &admissionv1.AdmissionResponse{\n            Allowed: false,\n            Result: &metav1.Status{\n                Message: \"replicas must be >= 1\",\n            },\n        }\n    }\n    return &admissionv1.AdmissionResponse{Allowed: true}\n}\n```",
                    "level2": "Mutating webhook for defaults:\n```go\nfunc (h *Handler) mutate(ar *admissionv1.AdmissionReview) *admissionv1.AdmissionResponse {\n    myapp := &v1.MyApp{}\n    json.Unmarshal(ar.Request.Object.Raw, myapp)\n    \n    patches := []jsonPatch{}\n    if myapp.Spec.Replicas == 0 {\n        patches = append(patches, jsonPatch{\n            Op:    \"add\",\n            Path:  \"/spec/replicas\",\n            Value: 1,\n        })\n    }\n    \n    patchBytes, _ := json.Marshal(patches)\n    return &admissionv1.AdmissionResponse{\n        Allowed: true,\n        Patch:   patchBytes,\n        PatchType: func() *admissionv1.PatchType {\n            pt := admissionv1.PatchTypeJSONPatch\n            return &pt\n        }(),\n    }\n}\n```",
                    "level3": "Webhook configuration:\n```yaml\napiVersion: admissionregistration.k8s.io/v1\nkind: ValidatingWebhookConfiguration\nmetadata:\n  name: myapp-validation\nwebhooks:\n- name: validate.myapp.example.com\n  rules:\n  - apiGroups: [\"example.com\"]\n    resources: [\"myapps\"]\n    operations: [\"CREATE\", \"UPDATE\"]\n  clientConfig:\n    service:\n      name: myapp-operator\n      namespace: default\n      path: /validate\n    caBundle: ${CA_BUNDLE}\n```"
                }
            },
            {
                "name": "Testing & Deployment",
                "description": "Test and deploy the operator",
                "skills": ["Envtest", "Helm charts", "RBAC"],
                "deliverables": [
                    "Unit tests with fake client",
                    "Integration tests with envtest",
                    "RBAC rules",
                    "Deployment manifests",
                    "Helm chart",
                    "Leader election"
                ],
                "hints": {
                    "level1": "Test with fake client:\n```go\nfunc TestReconcile(t *testing.T) {\n    myapp := &v1.MyApp{\n        ObjectMeta: metav1.ObjectMeta{Name: \"test\", Namespace: \"default\"},\n        Spec:       v1.MyAppSpec{Replicas: 3},\n    }\n    \n    client := fake.NewSimpleClientset(myapp)\n    controller := NewController(client)\n    \n    err := controller.reconcile(\"default/test\")\n    assert.NoError(t, err)\n}\n```",
                    "level2": "RBAC for operator:\n```yaml\napiVersion: rbac.authorization.k8s.io/v1\nkind: ClusterRole\nmetadata:\n  name: myapp-operator\nrules:\n- apiGroups: [\"example.com\"]\n  resources: [\"myapps\", \"myapps/status\"]\n  verbs: [\"get\", \"list\", \"watch\", \"update\", \"patch\"]\n- apiGroups: [\"apps\"]\n  resources: [\"deployments\"]\n  verbs: [\"get\", \"list\", \"watch\", \"create\", \"update\", \"delete\"]\n```",
                    "level3": "Leader election:\n```go\nlock := &resourcelock.LeaseLock{\n    LeaseMeta: metav1.ObjectMeta{Name: \"myapp-operator\", Namespace: \"default\"},\n    Client:    clientset.CoordinationV1(),\n    LockConfig: resourcelock.ResourceLockConfig{Identity: hostname},\n}\n\nleaderelection.RunOrDie(ctx, leaderelection.LeaderElectionConfig{\n    Lock:          lock,\n    LeaseDuration: 15 * time.Second,\n    RenewDeadline: 10 * time.Second,\n    RetryPeriod:   2 * time.Second,\n    Callbacks: leaderelection.LeaderCallbacks{\n        OnStartedLeading: func(ctx context.Context) {\n            controller.Run(2, ctx.Done())\n        },\n    },\n})\n```"
                }
            }
        ]
    },
    "ml-model-serving": {
        "name": "ML Model Serving API",
        "description": "Build a production ML model serving system with batching, versioning, A/B testing, and monitoring",
        "category": "AI/ML Infrastructure",
        "difficulty": "intermediate",
        "estimated_hours": 45,
        "skills": [
            "Model loading",
            "Request batching",
            "Model versioning",
            "A/B testing",
            "Latency optimization",
            "Model monitoring"
        ],
        "prerequisites": ["REST API", "Basic ML", "Docker"],
        "learning_outcomes": [
            "Design scalable model serving architectures",
            "Implement efficient batching for throughput",
            "Handle model versioning and rollback",
            "Monitor model performance in production"
        ],
        "milestones": [
            {
                "name": "Model Loading & Inference",
                "description": "Load models and serve predictions",
                "skills": ["Model formats", "Memory management", "Inference"],
                "deliverables": [
                    "Model loader for different formats",
                    "Prediction endpoint",
                    "Input validation",
                    "Output formatting",
                    "GPU/CPU device handling",
                    "Warm-up requests"
                ],
                "hints": {
                    "level1": "Basic model server:\n```python\nfrom fastapi import FastAPI\nimport torch\n\napp = FastAPI()\nmodel = None\n\n@app.on_event('startup')\nasync def load_model():\n    global model\n    model = torch.load('model.pt')\n    model.eval()\n\n@app.post('/predict')\nasync def predict(data: PredictRequest):\n    with torch.no_grad():\n        input_tensor = preprocess(data.inputs)\n        output = model(input_tensor)\n        return {'predictions': output.tolist()}\n```",
                    "level2": "Support multiple model formats:\n```python\nclass ModelLoader:\n    @staticmethod\n    def load(path: str):\n        if path.endswith('.pt'):\n            return torch.load(path)\n        elif path.endswith('.onnx'):\n            import onnxruntime\n            return onnxruntime.InferenceSession(path)\n        elif path.endswith('.pkl'):\n            import joblib\n            return joblib.load(path)\n```",
                    "level3": "Warm-up to avoid cold start:\n```python\n@app.on_event('startup')\nasync def warmup():\n    global model\n    model = load_model(MODEL_PATH)\n    # Run dummy inference to warm up\n    dummy_input = torch.zeros((1, *INPUT_SHAPE))\n    for _ in range(3):\n        model(dummy_input)\n    logger.info('Model warmed up')\n```"
                }
            },
            {
                "name": "Request Batching",
                "description": "Batch requests for better throughput",
                "skills": ["Dynamic batching", "Async processing", "Timeout handling"],
                "deliverables": [
                    "Request queue",
                    "Dynamic batch formation",
                    "Batch size limits",
                    "Timeout handling",
                    "Response routing",
                    "Throughput metrics"
                ],
                "hints": {
                    "level1": "Simple batching with queue:\n```python\nimport asyncio\nfrom collections import deque\n\nclass Batcher:\n    def __init__(self, max_batch=32, max_wait=0.01):\n        self.queue = deque()\n        self.max_batch = max_batch\n        self.max_wait = max_wait\n    \n    async def add(self, request):\n        future = asyncio.Future()\n        self.queue.append((request, future))\n        return await future\n```",
                    "level2": "Batch processor:\n```python\nasync def process_batches(self):\n    while True:\n        batch = []\n        deadline = time.time() + self.max_wait\n        \n        while len(batch) < self.max_batch and time.time() < deadline:\n            if self.queue:\n                batch.append(self.queue.popleft())\n            else:\n                await asyncio.sleep(0.001)\n        \n        if batch:\n            requests = [r for r, _ in batch]\n            futures = [f for _, f in batch]\n            results = await self.inference(requests)\n            for future, result in zip(futures, results):\n                future.set_result(result)\n```",
                    "level3": "Adaptive batching based on load:\n```python\nclass AdaptiveBatcher:\n    def __init__(self):\n        self.current_qps = 0\n        self.target_latency = 0.1\n    \n    def get_optimal_batch_size(self):\n        # Larger batches for higher load\n        if self.current_qps > 1000:\n            return 64\n        elif self.current_qps > 100:\n            return 32\n        return 8\n```"
                }
            },
            {
                "name": "Model Versioning",
                "description": "Manage multiple model versions",
                "skills": ["Version management", "Hot reload", "Rollback"],
                "deliverables": [
                    "Model registry",
                    "Version metadata",
                    "Hot model reload",
                    "Rollback mechanism",
                    "Default version",
                    "Version-specific endpoints"
                ],
                "hints": {
                    "level1": "Model registry:\n```python\nclass ModelRegistry:\n    def __init__(self, storage_path):\n        self.storage_path = storage_path\n        self.models = {}  # {model_name: {version: model}}\n        self.default_versions = {}\n    \n    def load_version(self, name, version):\n        path = f'{self.storage_path}/{name}/{version}/model.pt'\n        self.models.setdefault(name, {})[version] = load_model(path)\n    \n    def get(self, name, version=None):\n        version = version or self.default_versions.get(name)\n        return self.models[name][version]\n```",
                    "level2": "Hot reload without downtime:\n```python\nasync def reload_model(self, name, version):\n    # Load new model in background\n    new_model = await asyncio.to_thread(load_model, path)\n    \n    # Atomic swap\n    old_model = self.models[name].get(version)\n    self.models[name][version] = new_model\n    \n    # Cleanup old model\n    if old_model:\n        del old_model\n        torch.cuda.empty_cache()\n```",
                    "level3": "Version in request:\n```python\n@app.post('/v1/models/{model_name}/predict')\nasync def predict(\n    model_name: str,\n    request: PredictRequest,\n    version: Optional[str] = Query(None)\n):\n    model = registry.get(model_name, version)\n    return model.predict(request.inputs)\n```"
                }
            },
            {
                "name": "A/B Testing & Canary",
                "description": "Traffic splitting for model comparison",
                "skills": ["Traffic routing", "Experiment tracking", "Statistical analysis"],
                "deliverables": [
                    "Traffic split configuration",
                    "Consistent user routing",
                    "Experiment metrics",
                    "Statistical significance",
                    "Gradual rollout",
                    "Automatic rollback"
                ],
                "hints": {
                    "level1": "Simple traffic split:\n```python\nimport random\n\nclass ABRouter:\n    def __init__(self):\n        self.experiments = {}  # {name: {version: weight}}\n    \n    def route(self, model_name, user_id=None):\n        weights = self.experiments.get(model_name, {'default': 1.0})\n        if user_id:\n            # Consistent routing based on user\n            random.seed(hash(f'{model_name}:{user_id}'))\n        return random.choices(\n            list(weights.keys()),\n            weights=list(weights.values())\n        )[0]\n```",
                    "level2": "Track experiment metrics:\n```python\nclass ExperimentTracker:\n    async def record(self, experiment, version, metrics):\n        await redis.hincrby(f'exp:{experiment}:{version}', 'count', 1)\n        await redis.hincrbyfloat(f'exp:{experiment}:{version}', 'latency_sum', metrics['latency'])\n        if metrics.get('feedback'):\n            await redis.hincrby(f'exp:{experiment}:{version}', 'positive', 1 if metrics['feedback'] > 0 else 0)\n```",
                    "level3": "Auto-rollback on degradation:\n```python\nasync def check_experiment_health(self, experiment):\n    for version, stats in self.get_stats(experiment).items():\n        error_rate = stats['errors'] / stats['count']\n        if error_rate > 0.05:  # 5% error threshold\n            await self.rollback(experiment, version)\n            await self.alert(f'Rolled back {experiment}:{version}')\n```"
                }
            },
            {
                "name": "Monitoring & Observability",
                "description": "Monitor model performance and drift",
                "skills": ["Metrics", "Logging", "Drift detection"],
                "deliverables": [
                    "Latency metrics (p50, p99)",
                    "Throughput metrics",
                    "Input/output logging",
                    "Data drift detection",
                    "Model accuracy monitoring",
                    "Alerting"
                ],
                "hints": {
                    "level1": "Prometheus metrics:\n```python\nfrom prometheus_client import Histogram, Counter\n\nREQUEST_LATENCY = Histogram(\n    'model_request_latency_seconds',\n    'Model inference latency',\n    ['model', 'version']\n)\nREQUEST_COUNT = Counter(\n    'model_request_total',\n    'Total requests',\n    ['model', 'version', 'status']\n)\n\n@app.middleware('http')\nasync def metrics_middleware(request, call_next):\n    start = time.time()\n    response = await call_next(request)\n    REQUEST_LATENCY.labels(model=model_name, version=version).observe(time.time() - start)\n    return response\n```",
                    "level2": "Log predictions for analysis:\n```python\nasync def log_prediction(self, request_id, model, version, input_data, output, latency):\n    await kafka.send('predictions', {\n        'request_id': request_id,\n        'model': model,\n        'version': version,\n        'input_hash': hash_input(input_data),\n        'output': output,\n        'latency': latency,\n        'timestamp': datetime.utcnow().isoformat()\n    })\n```",
                    "level3": "Simple drift detection:\n```python\nclass DriftDetector:\n    def __init__(self, reference_stats):\n        self.reference = reference_stats\n    \n    def check(self, recent_inputs):\n        current_mean = np.mean(recent_inputs, axis=0)\n        current_std = np.std(recent_inputs, axis=0)\n        \n        # PSI (Population Stability Index)\n        psi = np.sum(\n            (current_mean - self.reference['mean']) * \n            np.log(current_mean / self.reference['mean'])\n        )\n        \n        if psi > 0.2:\n            return DriftAlert(psi=psi)\n```"
                }
            }
        ]
    },
    "llm-finetuning-pipeline": {
        "name": "LLM Fine-tuning Pipeline",
        "description": "Build an end-to-end LLM fine-tuning system with LoRA/QLoRA, dataset preparation, and evaluation",
        "category": "AI/ML",
        "difficulty": "advanced",
        "estimated_hours": 55,
        "skills": [
            "LoRA/QLoRA",
            "Dataset preparation",
            "Training loop",
            "Evaluation metrics",
            "Model merging",
            "Quantization"
        ],
        "prerequisites": ["PyTorch basics", "Transformers library", "GPU training"],
        "learning_outcomes": [
            "Understand parameter-efficient fine-tuning methods",
            "Prepare and format datasets for instruction tuning",
            "Implement training with memory optimization",
            "Evaluate and deploy fine-tuned models"
        ],
        "milestones": [
            {
                "name": "Dataset Preparation",
                "description": "Prepare and format training data",
                "skills": ["Data formatting", "Tokenization", "Chat templates"],
                "deliverables": [
                    "Data loading from various formats",
                    "Instruction/response formatting",
                    "Chat template application",
                    "Tokenization with padding",
                    "Train/val split",
                    "Data quality filtering"
                ],
                "hints": {
                    "level1": "Format for instruction tuning:\n```python\ndef format_instruction(example):\n    return f\"\"\"### Instruction:\n{example['instruction']}\n\n### Input:\n{example.get('input', '')}\n\n### Response:\n{example['output']}\"\"\"\n\ndataset = dataset.map(lambda x: {'text': format_instruction(x)})\n```",
                    "level2": "Use chat template:\n```python\nfrom transformers import AutoTokenizer\n\ntokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')\n\ndef format_chat(example):\n    messages = [\n        {'role': 'user', 'content': example['instruction']},\n        {'role': 'assistant', 'content': example['output']}\n    ]\n    return tokenizer.apply_chat_template(messages, tokenize=False)\n```",
                    "level3": "Tokenize with labels:\n```python\ndef tokenize(example):\n    result = tokenizer(\n        example['text'],\n        truncation=True,\n        max_length=2048,\n        padding='max_length'\n    )\n    result['labels'] = result['input_ids'].copy()\n    # Mask instruction part in labels\n    response_start = example['text'].find('### Response:')\n    response_token_start = len(tokenizer(example['text'][:response_start])['input_ids'])\n    result['labels'][:response_token_start] = [-100] * response_token_start\n    return result\n```"
                }
            },
            {
                "name": "LoRA Configuration",
                "description": "Set up LoRA adapters for efficient fine-tuning",
                "skills": ["PEFT library", "LoRA math", "Target modules"],
                "deliverables": [
                    "LoRA config selection",
                    "Target module identification",
                    "Rank and alpha tuning",
                    "Adapter initialization",
                    "Trainable parameter count",
                    "Memory estimation"
                ],
                "hints": {
                    "level1": "Basic LoRA setup:\n```python\nfrom peft import LoraConfig, get_peft_model\n\nlora_config = LoraConfig(\n    r=16,  # Rank\n    lora_alpha=32,  # Scaling\n    target_modules=['q_proj', 'v_proj'],  # Attention layers\n    lora_dropout=0.05,\n    bias='none',\n    task_type='CAUSAL_LM'\n)\n\nmodel = get_peft_model(model, lora_config)\nmodel.print_trainable_parameters()\n```",
                    "level2": "Target all linear layers:\n```python\nimport re\n\ndef find_linear_modules(model):\n    modules = []\n    for name, module in model.named_modules():\n        if isinstance(module, torch.nn.Linear):\n            # Skip lm_head and embeddings\n            if 'lm_head' not in name and 'embed' not in name:\n                modules.append(name.split('.')[-1])\n    return list(set(modules))\n\n# Usually: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']\n```",
                    "level3": "Calculate memory savings:\n```python\ndef estimate_memory(model, lora_config):\n    total_params = sum(p.numel() for p in model.parameters())\n    \n    # LoRA params: 2 * r * d for each target module\n    lora_params = 0\n    for name, module in model.named_modules():\n        if any(t in name for t in lora_config.target_modules):\n            if hasattr(module, 'weight'):\n                d_in, d_out = module.weight.shape\n                lora_params += 2 * lora_config.r * (d_in + d_out)\n    \n    print(f'Total: {total_params:,}, LoRA: {lora_params:,} ({100*lora_params/total_params:.2f}%)')\n```"
                }
            },
            {
                "name": "QLoRA & Quantization",
                "description": "4-bit quantization for memory efficiency",
                "skills": ["BitsAndBytes", "4-bit quantization", "NF4"],
                "deliverables": [
                    "4-bit model loading",
                    "NF4 quantization config",
                    "Compute dtype selection",
                    "Double quantization",
                    "Memory benchmarking",
                    "Quality vs memory tradeoff"
                ],
                "hints": {
                    "level1": "Load model in 4-bit:\n```python\nfrom transformers import BitsAndBytesConfig\n\nbnb_config = BitsAndBytesConfig(\n    load_in_4bit=True,\n    bnb_4bit_quant_type='nf4',\n    bnb_4bit_compute_dtype=torch.bfloat16,\n    bnb_4bit_use_double_quant=True\n)\n\nmodel = AutoModelForCausalLM.from_pretrained(\n    model_name,\n    quantization_config=bnb_config,\n    device_map='auto'\n)\n```",
                    "level2": "Prepare for k-bit training:\n```python\nfrom peft import prepare_model_for_kbit_training\n\nmodel = prepare_model_for_kbit_training(model)\n# This:\n# - Casts layernorm to fp32\n# - Enables gradient checkpointing\n# - Enables input gradient requirement\n```",
                    "level3": "Memory comparison:\n```python\ndef print_gpu_memory():\n    for i in range(torch.cuda.device_count()):\n        mem = torch.cuda.memory_allocated(i) / 1e9\n        print(f'GPU {i}: {mem:.2f} GB')\n\n# 7B model memory:\n# FP32: ~28 GB\n# FP16: ~14 GB\n# INT8: ~7 GB\n# INT4 (QLoRA): ~3.5 GB\n```"
                }
            },
            {
                "name": "Training Loop",
                "description": "Implement training with optimization",
                "skills": ["Trainer API", "Gradient accumulation", "Checkpointing"],
                "deliverables": [
                    "Training arguments",
                    "Gradient accumulation",
                    "Learning rate schedule",
                    "Checkpointing",
                    "Logging (WandB/TensorBoard)",
                    "Early stopping"
                ],
                "hints": {
                    "level1": "Basic training setup:\n```python\nfrom transformers import TrainingArguments, Trainer\n\ntraining_args = TrainingArguments(\n    output_dir='./results',\n    num_train_epochs=3,\n    per_device_train_batch_size=4,\n    gradient_accumulation_steps=4,  # Effective batch = 16\n    learning_rate=2e-4,\n    warmup_ratio=0.03,\n    logging_steps=10,\n    save_strategy='epoch',\n    fp16=True,\n)\n\ntrainer = Trainer(\n    model=model,\n    args=training_args,\n    train_dataset=train_dataset,\n    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)\n)\n```",
                    "level2": "Custom training loop with gradient checkpointing:\n```python\nmodel.gradient_checkpointing_enable()\nmodel.enable_input_require_grads()\n\noptimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)\nscheduler = get_cosine_schedule_with_warmup(\n    optimizer,\n    num_warmup_steps=100,\n    num_training_steps=len(dataloader) * epochs\n)\n\nfor epoch in range(epochs):\n    for step, batch in enumerate(dataloader):\n        outputs = model(**batch)\n        loss = outputs.loss / gradient_accumulation_steps\n        loss.backward()\n        \n        if (step + 1) % gradient_accumulation_steps == 0:\n            optimizer.step()\n            scheduler.step()\n            optimizer.zero_grad()\n```",
                    "level3": "WandB logging:\n```python\nimport wandb\n\nwandb.init(project='llm-finetune', config={\n    'model': model_name,\n    'lora_r': lora_config.r,\n    'learning_rate': training_args.learning_rate\n})\n\ntraining_args = TrainingArguments(\n    ...,\n    report_to='wandb',\n    run_name=f'{model_name}-lora-r{lora_config.r}'\n)\n```"
                }
            },
            {
                "name": "Evaluation & Merging",
                "description": "Evaluate model and merge adapters",
                "skills": ["Evaluation metrics", "Model merging", "GGUF export"],
                "deliverables": [
                    "Perplexity calculation",
                    "Task-specific evaluation",
                    "Adapter merging",
                    "Model export (GGUF)",
                    "Inference comparison",
                    "Quality benchmarks"
                ],
                "hints": {
                    "level1": "Calculate perplexity:\n```python\nimport math\n\ndef evaluate_perplexity(model, dataloader):\n    model.eval()\n    total_loss = 0\n    total_tokens = 0\n    \n    with torch.no_grad():\n        for batch in dataloader:\n            outputs = model(**batch)\n            total_loss += outputs.loss.item() * batch['input_ids'].numel()\n            total_tokens += batch['input_ids'].numel()\n    \n    return math.exp(total_loss / total_tokens)\n```",
                    "level2": "Merge LoRA weights:\n```python\nfrom peft import PeftModel\n\n# Load base model\nbase_model = AutoModelForCausalLM.from_pretrained(base_model_name)\n\n# Load adapter\nmodel = PeftModel.from_pretrained(base_model, adapter_path)\n\n# Merge and unload\nmerged_model = model.merge_and_unload()\n\n# Save merged model\nmerged_model.save_pretrained('merged_model')\ntokenizer.save_pretrained('merged_model')\n```",
                    "level3": "Export to GGUF for llama.cpp:\n```bash\n# Install llama.cpp\ngit clone https://github.com/ggerganov/llama.cpp\ncd llama.cpp && make\n\n# Convert to GGUF\npython convert.py ../merged_model --outtype f16 --outfile model.gguf\n\n# Quantize\n./quantize model.gguf model-q4_k_m.gguf q4_k_m\n```"
                }
            }
        ]
    },
    "order-matching-engine": {
        "name": "Order Matching Engine",
        "description": "Build a low-latency trading order matching engine with order book management and price-time priority",
        "category": "Fintech",
        "difficulty": "advanced",
        "estimated_hours": 70,
        "skills": [
            "Order book data structures",
            "Matching algorithms",
            "Low-latency design",
            "Concurrency",
            "Market data feeds",
            "Risk checks"
        ],
        "prerequisites": ["Data structures", "Concurrency", "Networking"],
        "learning_outcomes": [
            "Design high-performance order matching systems",
            "Implement price-time priority matching",
            "Handle concurrent order operations safely",
            "Build market data dissemination"
        ],
        "milestones": [
            {
                "name": "Order Book Data Structure",
                "description": "Efficient order book with price levels",
                "skills": ["Tree structures", "Price levels", "Order queues"],
                "deliverables": [
                    "Order class with fields",
                    "Price level with order queue",
                    "Bid/ask sides",
                    "O(1) best bid/ask access",
                    "O(log n) price level operations",
                    "Order ID lookup"
                ],
                "hints": {
                    "level1": "Order and price level structures:\n```python\nfrom dataclasses import dataclass\nfrom collections import deque\nfrom sortedcontainers import SortedDict\n\n@dataclass\nclass Order:\n    order_id: str\n    side: str  # 'buy' or 'sell'\n    price: Decimal\n    quantity: int\n    timestamp: int\n\nclass PriceLevel:\n    def __init__(self, price):\n        self.price = price\n        self.orders = deque()  # FIFO queue\n        self.total_quantity = 0\n```",
                    "level2": "Order book with sorted price levels:\n```python\nclass OrderBook:\n    def __init__(self, symbol):\n        self.symbol = symbol\n        self.bids = SortedDict()  # price -> PriceLevel, descending\n        self.asks = SortedDict()  # price -> PriceLevel, ascending\n        self.orders = {}  # order_id -> Order\n    \n    def best_bid(self):\n        return self.bids.peekitem(-1)[0] if self.bids else None\n    \n    def best_ask(self):\n        return self.asks.peekitem(0)[0] if self.asks else None\n```",
                    "level3": "Custom tree for O(1) best price:\n```python\nclass OrderBookSide:\n    def __init__(self, is_bid):\n        self.is_bid = is_bid\n        self.levels = SortedDict()\n        self._best = None\n    \n    def add_level(self, price):\n        level = PriceLevel(price)\n        self.levels[price] = level\n        if self._best is None or (self.is_bid and price > self._best) or (not self.is_bid and price < self._best):\n            self._best = price\n        return level\n    \n    def remove_level(self, price):\n        del self.levels[price]\n        if price == self._best:\n            self._best = self.levels.peekitem(-1 if self.is_bid else 0)[0] if self.levels else None\n```"
                }
            },
            {
                "name": "Order Operations",
                "description": "Add, cancel, and modify orders",
                "skills": ["Order lifecycle", "Validation", "State management"],
                "deliverables": [
                    "Add order to book",
                    "Cancel order",
                    "Modify order quantity",
                    "Order validation",
                    "Order acknowledgment",
                    "Execution reports"
                ],
                "hints": {
                    "level1": "Add order to book:\n```python\ndef add_order(self, order: Order):\n    side = self.bids if order.side == 'buy' else self.asks\n    \n    if order.price not in side:\n        side[order.price] = PriceLevel(order.price)\n    \n    level = side[order.price]\n    level.orders.append(order)\n    level.total_quantity += order.quantity\n    self.orders[order.order_id] = order\n    \n    return OrderAck(order.order_id, 'ACCEPTED')\n```",
                    "level2": "Cancel order:\n```python\ndef cancel_order(self, order_id: str):\n    if order_id not in self.orders:\n        return OrderAck(order_id, 'REJECTED', 'Order not found')\n    \n    order = self.orders[order_id]\n    side = self.bids if order.side == 'buy' else self.asks\n    level = side[order.price]\n    \n    level.orders.remove(order)\n    level.total_quantity -= order.quantity\n    \n    if level.total_quantity == 0:\n        del side[order.price]\n    \n    del self.orders[order_id]\n    return OrderAck(order_id, 'CANCELLED')\n```",
                    "level3": "Modify preserves time priority only if quantity decreases:\n```python\ndef modify_order(self, order_id: str, new_quantity: int):\n    order = self.orders.get(order_id)\n    if not order:\n        return OrderAck(order_id, 'REJECTED')\n    \n    if new_quantity > order.quantity:\n        # Lose time priority - cancel and re-add\n        self.cancel_order(order_id)\n        order.quantity = new_quantity\n        order.timestamp = time.time_ns()\n        return self.add_order(order)\n    else:\n        # Keep time priority\n        level = self.get_level(order)\n        level.total_quantity -= (order.quantity - new_quantity)\n        order.quantity = new_quantity\n        return OrderAck(order_id, 'MODIFIED')\n```"
                }
            },
            {
                "name": "Matching Engine",
                "description": "Price-time priority matching algorithm",
                "skills": ["Matching algorithms", "Trade execution", "Partial fills"],
                "deliverables": [
                    "Price-time priority matching",
                    "Full and partial fills",
                    "Trade generation",
                    "Aggressive vs passive orders",
                    "Self-trade prevention",
                    "Match statistics"
                ],
                "hints": {
                    "level1": "Basic matching:\n```python\ndef match(self, incoming: Order):\n    trades = []\n    opposite = self.asks if incoming.side == 'buy' else self.bids\n    \n    while incoming.quantity > 0 and opposite:\n        best_price = opposite.peekitem(0 if incoming.side == 'buy' else -1)[0]\n        \n        # Check if prices cross\n        if incoming.side == 'buy' and incoming.price < best_price:\n            break\n        if incoming.side == 'sell' and incoming.price > best_price:\n            break\n        \n        level = opposite[best_price]\n        trades.extend(self.match_at_level(incoming, level))\n        \n        if level.total_quantity == 0:\n            del opposite[best_price]\n    \n    # Add remainder to book\n    if incoming.quantity > 0:\n        self.add_order(incoming)\n    \n    return trades\n```",
                    "level2": "Match at price level (FIFO):\n```python\ndef match_at_level(self, aggressor, level):\n    trades = []\n    \n    while aggressor.quantity > 0 and level.orders:\n        resting = level.orders[0]\n        \n        fill_qty = min(aggressor.quantity, resting.quantity)\n        \n        trade = Trade(\n            symbol=self.symbol,\n            price=level.price,  # Passive order's price\n            quantity=fill_qty,\n            aggressor_id=aggressor.order_id,\n            passive_id=resting.order_id\n        )\n        trades.append(trade)\n        \n        aggressor.quantity -= fill_qty\n        resting.quantity -= fill_qty\n        level.total_quantity -= fill_qty\n        \n        if resting.quantity == 0:\n            level.orders.popleft()\n            del self.orders[resting.order_id]\n    \n    return trades\n```",
                    "level3": "Self-trade prevention:\n```python\ndef match_at_level(self, aggressor, level):\n    trades = []\n    \n    for resting in list(level.orders):\n        if aggressor.quantity == 0:\n            break\n        \n        # Self-trade prevention\n        if aggressor.trader_id == resting.trader_id:\n            # Options: cancel newest, cancel oldest, cancel both\n            if self.stp_mode == 'CANCEL_NEWEST':\n                aggressor.quantity = 0\n                return trades\n            continue\n        \n        # ... rest of matching\n```"
                }
            },
            {
                "name": "Concurrency & Performance",
                "description": "Thread-safe operations and low latency",
                "skills": ["Lock-free structures", "Batching", "Latency optimization"],
                "deliverables": [
                    "Lock-free order book updates",
                    "Order batching",
                    "Memory pre-allocation",
                    "Cache optimization",
                    "Latency measurement",
                    "Throughput benchmarking"
                ],
                "hints": {
                    "level1": "Single-threaded with event loop:\n```python\nimport asyncio\n\nclass MatchingEngine:\n    def __init__(self):\n        self.order_queue = asyncio.Queue()\n        self.books = {}  # symbol -> OrderBook\n    \n    async def run(self):\n        while True:\n            order = await self.order_queue.get()\n            book = self.books[order.symbol]\n            trades = book.match(order)\n            await self.publish_trades(trades)\n```",
                    "level2": "Measure latency:\n```python\nimport time\n\nclass LatencyTracker:\n    def __init__(self):\n        self.latencies = []\n    \n    def record(self, start_ns):\n        latency = time.time_ns() - start_ns\n        self.latencies.append(latency)\n    \n    def report(self):\n        arr = sorted(self.latencies)\n        return {\n            'p50': arr[len(arr)//2] / 1000,  # microseconds\n            'p99': arr[int(len(arr)*0.99)] / 1000,\n            'p999': arr[int(len(arr)*0.999)] / 1000\n        }\n```",
                    "level3": "Object pooling:\n```python\nclass OrderPool:\n    def __init__(self, size=10000):\n        self.pool = [Order() for _ in range(size)]\n        self.available = list(range(size))\n    \n    def acquire(self):\n        if self.available:\n            idx = self.available.pop()\n            return self.pool[idx]\n        return Order()  # Fallback to allocation\n    \n    def release(self, order):\n        order.reset()\n        idx = self.pool.index(order)\n        self.available.append(idx)\n```"
                }
            },
            {
                "name": "Market Data & API",
                "description": "Market data feeds and trading API",
                "skills": ["Market data", "WebSocket", "FIX protocol"],
                "deliverables": [
                    "Level 2 market data",
                    "Trade feed",
                    "WebSocket streaming",
                    "REST order API",
                    "FIX protocol basics",
                    "Rate limiting"
                ],
                "hints": {
                    "level1": "Market data snapshot:\n```python\ndef get_depth(self, symbol, levels=10):\n    book = self.books[symbol]\n    return {\n        'symbol': symbol,\n        'bids': [\n            {'price': str(p), 'quantity': l.total_quantity}\n            for p, l in list(book.bids.items())[-levels:]\n        ][::-1],\n        'asks': [\n            {'price': str(p), 'quantity': l.total_quantity}\n            for p, l in list(book.asks.items())[:levels]\n        ]\n    }\n```",
                    "level2": "WebSocket market data:\n```python\nfrom fastapi import WebSocket\n\nclass MarketDataFeed:\n    def __init__(self):\n        self.subscribers = defaultdict(set)\n    \n    async def subscribe(self, websocket: WebSocket, symbol: str):\n        self.subscribers[symbol].add(websocket)\n        # Send snapshot\n        await websocket.send_json(self.get_depth(symbol))\n    \n    async def publish_update(self, symbol: str, update: dict):\n        for ws in self.subscribers[symbol]:\n            try:\n                await ws.send_json(update)\n            except:\n                self.subscribers[symbol].remove(ws)\n```",
                    "level3": "Incremental updates:\n```python\ndef generate_update(self, order, trades):\n    updates = []\n    \n    # Price level changes\n    level = self.get_level(order)\n    updates.append({\n        'type': 'level',\n        'side': order.side,\n        'price': str(order.price),\n        'quantity': level.total_quantity if level else 0\n    })\n    \n    # Trades\n    for trade in trades:\n        updates.append({\n            'type': 'trade',\n            'price': str(trade.price),\n            'quantity': trade.quantity,\n            'timestamp': trade.timestamp\n        })\n    \n    return updates\n```"
                }
            }
        ]
    },
    "ledger-system": {
        "name": "Double-entry Ledger System",
        "description": "Build a double-entry accounting system with journal entries, account balances, and audit trail",
        "category": "Fintech",
        "difficulty": "advanced",
        "estimated_hours": 50,
        "skills": [
            "Double-entry accounting",
            "Transaction integrity",
            "Balance calculation",
            "Audit logging",
            "Financial reporting",
            "Idempotency"
        ],
        "prerequisites": ["Database design", "Transactions", "API design"],
        "learning_outcomes": [
            "Understand double-entry accounting principles",
            "Design immutable financial transaction systems",
            "Implement balance calculations and reconciliation",
            "Build audit-compliant financial systems"
        ],
        "milestones": [
            {
                "name": "Account & Entry Model",
                "description": "Design core ledger data model",
                "skills": ["Schema design", "Account types", "Entry structure"],
                "deliverables": [
                    "Account table with types",
                    "Journal entry table",
                    "Entry line items",
                    "Currency handling",
                    "Account hierarchy",
                    "Normal balance rules"
                ],
                "hints": {
                    "level1": "Core tables:\n```sql\nCREATE TABLE accounts (\n    id UUID PRIMARY KEY,\n    code VARCHAR(50) UNIQUE NOT NULL,\n    name VARCHAR(255) NOT NULL,\n    type VARCHAR(20) NOT NULL,  -- asset, liability, equity, revenue, expense\n    currency CHAR(3) DEFAULT 'USD',\n    parent_id UUID REFERENCES accounts(id),\n    created_at TIMESTAMP DEFAULT NOW()\n);\n\nCREATE TABLE journal_entries (\n    id UUID PRIMARY KEY,\n    entry_date DATE NOT NULL,\n    description TEXT,\n    reference VARCHAR(100),\n    created_at TIMESTAMP DEFAULT NOW()\n);\n\nCREATE TABLE entry_lines (\n    id UUID PRIMARY KEY,\n    journal_entry_id UUID REFERENCES journal_entries(id),\n    account_id UUID REFERENCES accounts(id),\n    debit DECIMAL(20,4) DEFAULT 0,\n    credit DECIMAL(20,4) DEFAULT 0,\n    CHECK (debit >= 0 AND credit >= 0),\n    CHECK (debit = 0 OR credit = 0)  -- Can't have both\n);\n```",
                    "level2": "Account types and normal balances:\n```python\nclass AccountType(Enum):\n    ASSET = 'asset'          # Debit normal\n    LIABILITY = 'liability'  # Credit normal\n    EQUITY = 'equity'        # Credit normal\n    REVENUE = 'revenue'      # Credit normal\n    EXPENSE = 'expense'      # Debit normal\n\ndef normal_balance(account_type):\n    return 'debit' if account_type in [AccountType.ASSET, AccountType.EXPENSE] else 'credit'\n```",
                    "level3": "Ensure debits = credits:\n```sql\nCREATE OR REPLACE FUNCTION check_balanced_entry()\nRETURNS TRIGGER AS $$\nBEGIN\n    IF (SELECT SUM(debit) - SUM(credit) FROM entry_lines \n        WHERE journal_entry_id = NEW.journal_entry_id) != 0 THEN\n        RAISE EXCEPTION 'Entry must be balanced';\n    END IF;\n    RETURN NEW;\nEND;\n$$ LANGUAGE plpgsql;\n```"
                }
            },
            {
                "name": "Transaction Recording",
                "description": "Record transactions with double-entry",
                "skills": ["Transaction atomicity", "Validation", "Idempotency"],
                "deliverables": [
                    "Create journal entry API",
                    "Balance validation",
                    "Idempotency keys",
                    "Transaction templates",
                    "Batch entries",
                    "Entry reversal"
                ],
                "hints": {
                    "level1": "Create balanced entry:\n```python\nclass Ledger:\n    def create_entry(self, date, description, lines, reference=None):\n        # Validate balance\n        total_debit = sum(l['debit'] for l in lines)\n        total_credit = sum(l['credit'] for l in lines)\n        if total_debit != total_credit:\n            raise ValueError(f'Entry not balanced: {total_debit} != {total_credit}')\n        \n        with db.transaction():\n            entry = JournalEntry.create(\n                entry_date=date,\n                description=description,\n                reference=reference\n            )\n            for line in lines:\n                EntryLine.create(\n                    journal_entry_id=entry.id,\n                    account_id=line['account_id'],\n                    debit=line.get('debit', 0),\n                    credit=line.get('credit', 0)\n                )\n        return entry\n```",
                    "level2": "Idempotency:\n```python\ndef create_entry(self, idempotency_key, ...):\n    # Check if already processed\n    existing = db.query(\n        'SELECT id FROM journal_entries WHERE idempotency_key = %s',\n        [idempotency_key]\n    )\n    if existing:\n        return existing[0]  # Return existing entry\n    \n    with db.transaction():\n        entry = JournalEntry.create(\n            idempotency_key=idempotency_key,\n            ...\n        )\n        # ... create lines\n    return entry\n```",
                    "level3": "Common transaction templates:\n```python\nclass TransactionTemplates:\n    @staticmethod\n    def payment_received(customer_account, revenue_account, amount):\n        return [\n            {'account_id': customer_account, 'debit': amount, 'credit': 0},\n            {'account_id': revenue_account, 'debit': 0, 'credit': amount}\n        ]\n    \n    @staticmethod\n    def transfer(from_account, to_account, amount):\n        return [\n            {'account_id': from_account, 'debit': 0, 'credit': amount},\n            {'account_id': to_account, 'debit': amount, 'credit': 0}\n        ]\n```"
                }
            },
            {
                "name": "Balance Calculation",
                "description": "Calculate account balances efficiently",
                "skills": ["Running balances", "Point-in-time", "Aggregation"],
                "deliverables": [
                    "Current balance calculation",
                    "Balance at date",
                    "Running balance table",
                    "Balance caching",
                    "Trial balance report",
                    "Account reconciliation"
                ],
                "hints": {
                    "level1": "Calculate balance from entries:\n```python\ndef get_balance(self, account_id, as_of=None):\n    query = '''\n        SELECT \n            COALESCE(SUM(debit), 0) as total_debit,\n            COALESCE(SUM(credit), 0) as total_credit\n        FROM entry_lines el\n        JOIN journal_entries je ON el.journal_entry_id = je.id\n        WHERE el.account_id = %s\n    '''\n    params = [account_id]\n    \n    if as_of:\n        query += ' AND je.entry_date <= %s'\n        params.append(as_of)\n    \n    result = db.query(query, params)[0]\n    \n    account = Account.get(account_id)\n    if normal_balance(account.type) == 'debit':\n        return result.total_debit - result.total_credit\n    else:\n        return result.total_credit - result.total_debit\n```",
                    "level2": "Materialized running balance:\n```sql\nCREATE TABLE account_balances (\n    account_id UUID REFERENCES accounts(id),\n    balance_date DATE,\n    debit_total DECIMAL(20,4),\n    credit_total DECIMAL(20,4),\n    balance DECIMAL(20,4),\n    PRIMARY KEY (account_id, balance_date)\n);\n\n-- Update on entry creation\nCREATE TRIGGER update_balance\nAFTER INSERT ON entry_lines\nFOR EACH ROW\nEXECUTE FUNCTION update_account_balance();\n```",
                    "level3": "Trial balance:\n```python\ndef trial_balance(self, as_of=None):\n    balances = db.query('''\n        SELECT \n            a.code, a.name, a.type,\n            SUM(el.debit) as debits,\n            SUM(el.credit) as credits\n        FROM accounts a\n        LEFT JOIN entry_lines el ON a.id = el.account_id\n        LEFT JOIN journal_entries je ON el.journal_entry_id = je.id\n        WHERE je.entry_date <= %s OR je.entry_date IS NULL\n        GROUP BY a.id\n        ORDER BY a.code\n    ''', [as_of or date.today()])\n    \n    total_debit = sum(b.debits or 0 for b in balances)\n    total_credit = sum(b.credits or 0 for b in balances)\n    \n    return {\n        'accounts': balances,\n        'total_debit': total_debit,\n        'total_credit': total_credit,\n        'balanced': total_debit == total_credit\n    }\n```"
                }
            },
            {
                "name": "Audit Trail",
                "description": "Immutable audit logging",
                "skills": ["Immutability", "Audit logs", "Change tracking"],
                "deliverables": [
                    "Immutable entries (no UPDATE/DELETE)",
                    "Correction entries (reversals)",
                    "Change history table",
                    "User action logging",
                    "Entry approval workflow",
                    "Export for audit"
                ],
                "hints": {
                    "level1": "Prevent modifications:\n```sql\nCREATE OR REPLACE FUNCTION prevent_entry_modification()\nRETURNS TRIGGER AS $$\nBEGIN\n    RAISE EXCEPTION 'Journal entries cannot be modified. Create a reversal instead.';\nEND;\n$$ LANGUAGE plpgsql;\n\nCREATE TRIGGER no_update_entries\nBEFORE UPDATE OR DELETE ON journal_entries\nFOR EACH ROW EXECUTE FUNCTION prevent_entry_modification();\n\nCREATE TRIGGER no_update_lines\nBEFORE UPDATE OR DELETE ON entry_lines\nFOR EACH ROW EXECUTE FUNCTION prevent_entry_modification();\n```",
                    "level2": "Reversal entry:\n```python\ndef reverse_entry(self, entry_id, reason):\n    original = JournalEntry.get(entry_id)\n    original_lines = EntryLine.where(journal_entry_id=entry_id)\n    \n    # Create reversal with swapped debits/credits\n    reversal_lines = [\n        {\n            'account_id': line.account_id,\n            'debit': line.credit,  # Swap\n            'credit': line.debit   # Swap\n        }\n        for line in original_lines\n    ]\n    \n    return self.create_entry(\n        date=date.today(),\n        description=f'REVERSAL: {reason} (original: {entry_id})',\n        lines=reversal_lines,\n        reference=f'REV-{entry_id}'\n    )\n```",
                    "level3": "Audit log:\n```sql\nCREATE TABLE audit_log (\n    id BIGSERIAL PRIMARY KEY,\n    timestamp TIMESTAMP DEFAULT NOW(),\n    user_id UUID,\n    action VARCHAR(50),\n    entity_type VARCHAR(50),\n    entity_id UUID,\n    details JSONB,\n    ip_address INET\n);\n\n-- Log all entry creations\nCREATE TRIGGER log_entry_creation\nAFTER INSERT ON journal_entries\nFOR EACH ROW\nEXECUTE FUNCTION log_audit_event('CREATE', 'journal_entry');\n```"
                }
            },
            {
                "name": "Financial Reports",
                "description": "Generate standard financial reports",
                "skills": ["Report generation", "Period closing", "Multi-currency"],
                "deliverables": [
                    "Income statement",
                    "Balance sheet",
                    "Cash flow statement",
                    "Period closing entries",
                    "Multi-currency support",
                    "Report export (PDF, CSV)"
                ],
                "hints": {
                    "level1": "Income statement:\n```python\ndef income_statement(self, start_date, end_date):\n    revenue = self.sum_by_type(AccountType.REVENUE, start_date, end_date)\n    expenses = self.sum_by_type(AccountType.EXPENSE, start_date, end_date)\n    \n    return {\n        'period': {'start': start_date, 'end': end_date},\n        'revenue': revenue,\n        'expenses': expenses,\n        'net_income': revenue['total'] - expenses['total']\n    }\n\ndef sum_by_type(self, account_type, start, end):\n    accounts = db.query('''\n        SELECT a.name, SUM(el.credit) - SUM(el.debit) as amount\n        FROM accounts a\n        JOIN entry_lines el ON a.id = el.account_id\n        JOIN journal_entries je ON el.journal_entry_id = je.id\n        WHERE a.type = %s AND je.entry_date BETWEEN %s AND %s\n        GROUP BY a.id\n    ''', [account_type, start, end])\n    return {'accounts': accounts, 'total': sum(a.amount for a in accounts)}\n```",
                    "level2": "Balance sheet:\n```python\ndef balance_sheet(self, as_of):\n    assets = self.sum_by_type(AccountType.ASSET, None, as_of)\n    liabilities = self.sum_by_type(AccountType.LIABILITY, None, as_of)\n    equity = self.sum_by_type(AccountType.EQUITY, None, as_of)\n    \n    # Retained earnings = prior net income\n    retained = self.calculate_retained_earnings(as_of)\n    \n    return {\n        'as_of': as_of,\n        'assets': assets,\n        'liabilities': liabilities,\n        'equity': {\n            'accounts': equity['accounts'],\n            'retained_earnings': retained,\n            'total': equity['total'] + retained\n        },\n        'balanced': assets['total'] == liabilities['total'] + equity['total'] + retained\n    }\n```",
                    "level3": "Period closing:\n```python\ndef close_period(self, period_end):\n    # Calculate net income\n    revenue = self.sum_by_type(AccountType.REVENUE, period_start, period_end)\n    expenses = self.sum_by_type(AccountType.EXPENSE, period_start, period_end)\n    net_income = revenue['total'] - expenses['total']\n    \n    # Close revenue and expense to retained earnings\n    lines = []\n    for acc in revenue['accounts']:\n        lines.append({'account_id': acc.id, 'debit': acc.amount, 'credit': 0})\n    for acc in expenses['accounts']:\n        lines.append({'account_id': acc.id, 'debit': 0, 'credit': acc.amount})\n    lines.append({'account_id': retained_earnings_id, 'debit': 0 if net_income > 0 else -net_income, 'credit': net_income if net_income > 0 else 0})\n    \n    return self.create_entry(period_end, f'Period close {period_end}', lines)\n```"
                }
            }
        ]
    }
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    # Create a lookup for expert_projects
    if 'expert_projects' not in data:
        data['expert_projects'] = {}

    added = 0
    for project_id, project_data in DETAILED_PROJECTS.items():
        data['expert_projects'][project_id] = project_data
        print(f"Added detailed milestones: {project_id}")
        added += 1

    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"\nAdded detailed milestones for {added} projects")


if __name__ == "__main__":
    main()

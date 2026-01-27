#!/usr/bin/env python3
"""
Add Enterprise/Senior Engineering patterns to the curriculum.
Focus on patterns that senior engineers encounter in production systems.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

enterprise_projects = {
    "feature-flags": {
        "name": "Feature Flag System",
        "description": "Build a feature flag system with gradual rollouts, A/B testing support, targeting rules, and real-time updates.",
        "why_important": "Feature flags enable safe deployments, A/B testing, and gradual rollouts. Understanding flag systems helps design better release strategies.",
        "difficulty": "intermediate",
        "tags": ["backend", "architecture", "devops"],
        "estimated_hours": 35,
        "prerequisites": ["rest-api-design"],
        "learning_outcomes": [
            "Design flag evaluation with targeting rules",
            "Implement percentage-based rollouts",
            "Build real-time flag updates without restart",
            "Handle flag dependencies and conflicts"
        ],
        "milestones": [
            {
                "name": "Flag Evaluation Engine",
                "description": "Build the core flag evaluation engine with support for boolean, string, number, and JSON flags.",
                "hints": {
                    "level1": "Store flag configurations with default and variation values.",
                    "level2": "Evaluate targeting rules against user context to determine variation.",
                    "level3": """```python
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from enum import Enum
import hashlib

class FlagType(str, Enum):
    BOOLEAN = "boolean"
    STRING = "string"
    NUMBER = "number"
    JSON = "json"

@dataclass
class Variation:
    value: Any
    name: str = ""
    description: str = ""

@dataclass
class TargetingRule:
    id: str
    conditions: list['Condition']
    variation_index: int
    priority: int = 0

@dataclass
class Condition:
    attribute: str
    operator: str  # eq, neq, contains, startswith, gt, lt, in, regex
    values: list[Any]

@dataclass
class FeatureFlag:
    key: str
    type: FlagType
    variations: list[Variation]
    default_variation: int  # Index of default variation
    targeting_rules: list[TargetingRule] = field(default_factory=list)
    percentage_rollout: Optional['PercentageRollout'] = None
    enabled: bool = True
    prerequisites: list[str] = field(default_factory=list)

@dataclass
class PercentageRollout:
    bucket_by: str = "user_id"  # Attribute to hash for bucketing
    variations: list[tuple[int, int]]  # [(variation_index, percentage), ...]

@dataclass
class EvaluationContext:
    user_id: str = ""
    attributes: dict = field(default_factory=dict)

    def get(self, key: str) -> Any:
        if key == "user_id":
            return self.user_id
        return self.attributes.get(key)

@dataclass
class EvaluationResult:
    value: Any
    variation_index: int
    reason: str
    flag_key: str

class FlagEvaluator:
    def __init__(self, flags: dict[str, FeatureFlag]):
        self.flags = flags

    def evaluate(self, flag_key: str, context: EvaluationContext,
                default: Any = None) -> EvaluationResult:
        flag = self.flags.get(flag_key)

        if not flag:
            return EvaluationResult(
                value=default,
                variation_index=-1,
                reason="FLAG_NOT_FOUND",
                flag_key=flag_key
            )

        if not flag.enabled:
            return self._result(flag, flag.default_variation, "FLAG_DISABLED")

        # Check prerequisites
        for prereq_key in flag.prerequisites:
            prereq_result = self.evaluate(prereq_key, context)
            if not prereq_result.value:
                return self._result(flag, flag.default_variation, "PREREQUISITE_FAILED")

        # Evaluate targeting rules (highest priority first)
        sorted_rules = sorted(flag.targeting_rules, key=lambda r: r.priority, reverse=True)
        for rule in sorted_rules:
            if self._evaluate_rule(rule, context):
                return self._result(flag, rule.variation_index, f"RULE:{rule.id}")

        # Check percentage rollout
        if flag.percentage_rollout:
            variation_idx = self._evaluate_rollout(flag, context)
            if variation_idx is not None:
                return self._result(flag, variation_idx, "ROLLOUT")

        return self._result(flag, flag.default_variation, "DEFAULT")

    def _evaluate_rule(self, rule: TargetingRule, context: EvaluationContext) -> bool:
        return all(self._evaluate_condition(c, context) for c in rule.conditions)

    def _evaluate_condition(self, condition: Condition, context: EvaluationContext) -> bool:
        attr_value = context.get(condition.attribute)

        if condition.operator == "eq":
            return attr_value == condition.values[0]
        elif condition.operator == "neq":
            return attr_value != condition.values[0]
        elif condition.operator == "in":
            return attr_value in condition.values
        elif condition.operator == "contains":
            return condition.values[0] in str(attr_value)
        elif condition.operator == "startswith":
            return str(attr_value).startswith(condition.values[0])
        elif condition.operator == "gt":
            return float(attr_value) > float(condition.values[0])
        elif condition.operator == "lt":
            return float(attr_value) < float(condition.values[0])
        elif condition.operator == "regex":
            import re
            return bool(re.match(condition.values[0], str(attr_value)))

        return False

    def _evaluate_rollout(self, flag: FeatureFlag, context: EvaluationContext) -> Optional[int]:
        rollout = flag.percentage_rollout
        bucket_value = context.get(rollout.bucket_by)
        if bucket_value is None:
            return None

        # Hash to get bucket (0-100)
        hash_input = f"{flag.key}:{bucket_value}"
        bucket = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 100

        cumulative = 0
        for variation_idx, percentage in rollout.variations:
            cumulative += percentage
            if bucket < cumulative:
                return variation_idx

        return None

    def _result(self, flag: FeatureFlag, variation_idx: int, reason: str) -> EvaluationResult:
        return EvaluationResult(
            value=flag.variations[variation_idx].value,
            variation_index=variation_idx,
            reason=reason,
            flag_key=flag.key
        )
```"""
                },
                "pitfalls": [
                    "Inconsistent hashing causes users to flip-flop variations",
                    "Circular prerequisites cause infinite loop",
                    "Rule priority ties cause non-deterministic evaluation"
                ]
            },
            {
                "name": "Real-time Flag Updates",
                "description": "Implement real-time flag updates using SSE/WebSocket with local caching and fallback.",
                "hints": {
                    "level1": "Poll or stream flag changes from server.",
                    "level2": "Cache flags locally, use stale cache if server unavailable.",
                    "level3": """```python
import asyncio
import aiohttp
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

@dataclass
class FlagCache:
    flags: dict[str, FeatureFlag]
    version: int
    updated_at: datetime

class FlagClient:
    def __init__(self, base_url: str, sdk_key: str):
        self.base_url = base_url
        self.sdk_key = sdk_key
        self.cache: FlagCache = None
        self.evaluator: FlagEvaluator = None
        self._listeners: list[Callable] = []
        self._sse_task = None
        self._running = False

    async def initialize(self):
        \"\"\"Fetch initial flags and start streaming updates.\"\"\"
        await self._fetch_all_flags()
        self._running = True
        self._sse_task = asyncio.create_task(self._stream_updates())

    async def close(self):
        self._running = False
        if self._sse_task:
            self._sse_task.cancel()

    def evaluate(self, flag_key: str, context: EvaluationContext,
                default: Any = None) -> EvaluationResult:
        if not self.evaluator:
            return EvaluationResult(
                value=default,
                variation_index=-1,
                reason="NOT_INITIALIZED",
                flag_key=flag_key
            )
        return self.evaluator.evaluate(flag_key, context, default)

    def on_change(self, callback: Callable[[str, FeatureFlag], None]):
        \"\"\"Register listener for flag changes.\"\"\"
        self._listeners.append(callback)

    async def _fetch_all_flags(self):
        headers = {"Authorization": f"Bearer {self.sdk_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/flags", headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._update_cache(data)

    async def _stream_updates(self):
        \"\"\"Stream flag updates using Server-Sent Events.\"\"\"
        headers = {
            "Authorization": f"Bearer {self.sdk_key}",
            "Accept": "text/event-stream"
        }

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/stream",
                                          headers=headers,
                                          timeout=None) as resp:
                        async for line in resp.content:
                            if not self._running:
                                break

                            line = line.decode().strip()
                            if line.startswith("data:"):
                                event_data = json.loads(line[5:])
                                await self._handle_event(event_data)

            except aiohttp.ClientError:
                # Reconnect after delay
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break

    async def _handle_event(self, event: dict):
        event_type = event.get("type")

        if event_type == "flag_updated":
            flag_data = event.get("flag")
            flag = self._parse_flag(flag_data)
            self.cache.flags[flag.key] = flag
            self.cache.version = event.get("version", self.cache.version + 1)
            self._rebuild_evaluator()
            self._notify_listeners(flag.key, flag)

        elif event_type == "flag_deleted":
            flag_key = event.get("key")
            self.cache.flags.pop(flag_key, None)
            self._rebuild_evaluator()
            self._notify_listeners(flag_key, None)

        elif event_type == "full_sync":
            self._update_cache(event.get("flags"))

    def _update_cache(self, data: dict):
        flags = {}
        for flag_data in data.get("flags", []):
            flag = self._parse_flag(flag_data)
            flags[flag.key] = flag

        self.cache = FlagCache(
            flags=flags,
            version=data.get("version", 0),
            updated_at=datetime.utcnow()
        )
        self._rebuild_evaluator()

    def _rebuild_evaluator(self):
        self.evaluator = FlagEvaluator(self.cache.flags)

    def _notify_listeners(self, flag_key: str, flag: FeatureFlag):
        for listener in self._listeners:
            try:
                listener(flag_key, flag)
            except Exception:
                pass

    def _parse_flag(self, data: dict) -> FeatureFlag:
        # Convert dict to FeatureFlag object
        variations = [Variation(**v) for v in data.get("variations", [])]
        rules = [
            TargetingRule(
                id=r["id"],
                conditions=[Condition(**c) for c in r.get("conditions", [])],
                variation_index=r["variation_index"],
                priority=r.get("priority", 0)
            )
            for r in data.get("targeting_rules", [])
        ]

        rollout = None
        if "percentage_rollout" in data:
            rollout = PercentageRollout(**data["percentage_rollout"])

        return FeatureFlag(
            key=data["key"],
            type=FlagType(data["type"]),
            variations=variations,
            default_variation=data["default_variation"],
            targeting_rules=rules,
            percentage_rollout=rollout,
            enabled=data.get("enabled", True),
            prerequisites=data.get("prerequisites", [])
        )
```"""
                },
                "pitfalls": [
                    "SSE reconnection without backoff causes thundering herd",
                    "Stale cache served indefinitely when stream down",
                    "Large flag payloads slow down evaluation"
                ]
            },
            {
                "name": "Flag Analytics & Experiments",
                "description": "Track flag evaluations for analytics and support A/B testing with statistical significance.",
                "hints": {
                    "level1": "Log each evaluation with context, variation, and timestamp.",
                    "level2": "Calculate conversion rates per variation, check statistical significance.",
                    "level3": """```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import math
from scipy import stats

@dataclass
class EvaluationEvent:
    flag_key: str
    variation_index: int
    user_id: str
    timestamp: datetime
    context_hash: str  # For deduplication

@dataclass
class ConversionEvent:
    experiment_key: str
    user_id: str
    metric_name: str
    metric_value: float
    timestamp: datetime

@dataclass
class ExperimentResults:
    control_conversions: int
    control_total: int
    treatment_conversions: int
    treatment_total: int

    @property
    def control_rate(self) -> float:
        return self.control_conversions / self.control_total if self.control_total else 0

    @property
    def treatment_rate(self) -> float:
        return self.treatment_conversions / self.treatment_total if self.treatment_total else 0

    @property
    def relative_lift(self) -> float:
        if self.control_rate == 0:
            return 0
        return (self.treatment_rate - self.control_rate) / self.control_rate

    def is_significant(self, confidence: float = 0.95) -> bool:
        \"\"\"Check statistical significance using chi-squared test.\"\"\"
        observed = [
            [self.control_conversions, self.control_total - self.control_conversions],
            [self.treatment_conversions, self.treatment_total - self.treatment_conversions]
        ]
        chi2, p_value, _, _ = stats.chi2_contingency(observed)
        return p_value < (1 - confidence)

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        \"\"\"Calculate confidence interval for lift.\"\"\"
        # Use normal approximation
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        se_control = math.sqrt(self.control_rate * (1 - self.control_rate) / self.control_total) if self.control_total else 0
        se_treatment = math.sqrt(self.treatment_rate * (1 - self.treatment_rate) / self.treatment_total) if self.treatment_total else 0

        se_diff = math.sqrt(se_control**2 + se_treatment**2)
        diff = self.treatment_rate - self.control_rate

        return (diff - z * se_diff, diff + z * se_diff)

class ExperimentAnalyzer:
    def __init__(self, storage):
        self.storage = storage

    async def get_results(self, experiment_key: str, metric_name: str,
                         start_time: datetime, end_time: datetime) -> ExperimentResults:
        # Get users in each variation
        control_users = await self.storage.get_users_in_variation(
            experiment_key, 0, start_time, end_time
        )
        treatment_users = await self.storage.get_users_in_variation(
            experiment_key, 1, start_time, end_time
        )

        # Get conversions
        control_conversions = await self.storage.count_conversions(
            experiment_key, metric_name, control_users, start_time, end_time
        )
        treatment_conversions = await self.storage.count_conversions(
            experiment_key, metric_name, treatment_users, start_time, end_time
        )

        return ExperimentResults(
            control_conversions=control_conversions,
            control_total=len(control_users),
            treatment_conversions=treatment_conversions,
            treatment_total=len(treatment_users)
        )

    def calculate_sample_size(self, baseline_rate: float, min_detectable_effect: float,
                              power: float = 0.8, significance: float = 0.05) -> int:
        \"\"\"Calculate required sample size per variation.\"\"\"
        alpha = significance
        beta = 1 - power

        # Standard formula for two-proportion z-test
        p1 = baseline_rate
        p2 = baseline_rate * (1 + min_detectable_effect)
        p_pooled = (p1 + p2) / 2

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = (
            (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) +
             z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
            / (p2 - p1)
        ) ** 2

        return int(math.ceil(n))

class FlagAnalytics:
    def __init__(self, storage):
        self.storage = storage
        self._buffer: list[EvaluationEvent] = []
        self._buffer_size = 100
        self._lock = asyncio.Lock()

    async def track_evaluation(self, result: EvaluationResult, context: EvaluationContext):
        event = EvaluationEvent(
            flag_key=result.flag_key,
            variation_index=result.variation_index,
            user_id=context.user_id,
            timestamp=datetime.utcnow(),
            context_hash=self._hash_context(context)
        )

        async with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._buffer_size:
                await self._flush()

    async def track_conversion(self, experiment_key: str, user_id: str,
                              metric_name: str, value: float = 1.0):
        event = ConversionEvent(
            experiment_key=experiment_key,
            user_id=user_id,
            metric_name=metric_name,
            metric_value=value,
            timestamp=datetime.utcnow()
        )
        await self.storage.store_conversion(event)

    async def _flush(self):
        events = self._buffer
        self._buffer = []
        await self.storage.store_evaluations(events)

    def _hash_context(self, context: EvaluationContext) -> str:
        import hashlib
        import json
        data = json.dumps({"user_id": context.user_id, **context.attributes}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()
```"""
                },
                "pitfalls": [
                    "Peeking at results early inflates false positive rate",
                    "Sample ratio mismatch invalidates experiment",
                    "Novelty effect skews early results"
                ]
            }
        ]
    },

    "job-scheduler": {
        "name": "Distributed Job Scheduler",
        "description": "Build a distributed job scheduler with cron expressions, retries, priorities, and cluster coordination.",
        "why_important": "Background job processing is essential for async workloads. Understanding schedulers helps design reliable batch processing systems.",
        "difficulty": "advanced",
        "tags": ["distributed-systems", "backend", "reliability"],
        "estimated_hours": 45,
        "prerequisites": ["redis-clone"],
        "learning_outcomes": [
            "Parse and evaluate cron expressions",
            "Implement distributed locking for job claims",
            "Design retry strategies with backoff",
            "Handle worker failures and job recovery"
        ],
        "milestones": [
            {
                "name": "Cron Expression Parser",
                "description": "Parse cron expressions and calculate next execution times.",
                "hints": {
                    "level1": "Parse five fields: minute, hour, day, month, weekday.",
                    "level2": "Handle special characters: *, /, -, , and named values.",
                    "level3": """```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Set
import re

@dataclass
class CronField:
    values: Set[int]
    min_val: int
    max_val: int

    @classmethod
    def parse(cls, expr: str, min_val: int, max_val: int, names: dict = None) -> 'CronField':
        values = set()

        # Handle names (JAN, SUN, etc.)
        if names:
            for name, val in names.items():
                expr = expr.upper().replace(name, str(val))

        for part in expr.split(','):
            if part == '*':
                values.update(range(min_val, max_val + 1))
            elif '/' in part:
                # Step: */5 or 1-10/2
                range_part, step = part.split('/')
                step = int(step)
                if range_part == '*':
                    start, end = min_val, max_val
                elif '-' in range_part:
                    start, end = map(int, range_part.split('-'))
                else:
                    start = int(range_part)
                    end = max_val
                values.update(range(start, end + 1, step))
            elif '-' in part:
                # Range: 1-5
                start, end = map(int, part.split('-'))
                values.update(range(start, end + 1))
            else:
                values.add(int(part))

        return cls(values=values, min_val=min_val, max_val=max_val)

class CronExpression:
    MONTH_NAMES = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    DOW_NAMES = {
        'SUN': 0, 'MON': 1, 'TUE': 2, 'WED': 3, 'THU': 4, 'FRI': 5, 'SAT': 6
    }

    def __init__(self, expression: str):
        self.expression = expression
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")

        self.minute = CronField.parse(parts[0], 0, 59)
        self.hour = CronField.parse(parts[1], 0, 23)
        self.day = CronField.parse(parts[2], 1, 31)
        self.month = CronField.parse(parts[3], 1, 12, self.MONTH_NAMES)
        self.dow = CronField.parse(parts[4], 0, 6, self.DOW_NAMES)

    def next_run(self, after: datetime = None) -> datetime:
        \"\"\"Calculate next execution time after given datetime.\"\"\"
        after = after or datetime.now()
        # Start from next minute
        current = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Limit search to prevent infinite loops
        for _ in range(366 * 24 * 60):  # Max 1 year of minutes
            if self._matches(current):
                return current
            current += timedelta(minutes=1)

        raise ValueError("No valid execution time found within 1 year")

    def _matches(self, dt: datetime) -> bool:
        if dt.minute not in self.minute.values:
            return False
        if dt.hour not in self.hour.values:
            return False
        if dt.month not in self.month.values:
            return False

        # Day and DOW have OR relationship
        day_match = dt.day in self.day.values
        dow_match = dt.weekday() in self.dow.values

        # If both are restricted (not *), either can match
        day_restricted = self.day.values != set(range(1, 32))
        dow_restricted = self.dow.values != set(range(0, 7))

        if day_restricted and dow_restricted:
            return day_match or dow_match
        elif day_restricted:
            return day_match
        elif dow_restricted:
            return dow_match
        else:
            return True

    def matches(self, dt: datetime) -> bool:
        \"\"\"Check if datetime matches this cron expression.\"\"\"
        return self._matches(dt)

    def next_n_runs(self, n: int, after: datetime = None) -> list[datetime]:
        \"\"\"Get next n execution times.\"\"\"
        runs = []
        current = after or datetime.now()
        for _ in range(n):
            next_time = self.next_run(current)
            runs.append(next_time)
            current = next_time
        return runs

# Common presets
CRON_PRESETS = {
    '@yearly': '0 0 1 1 *',
    '@monthly': '0 0 1 * *',
    '@weekly': '0 0 * * 0',
    '@daily': '0 0 * * *',
    '@hourly': '0 * * * *',
}
```"""
                },
                "pitfalls": [
                    "Daylight saving time causes missed or duplicate runs",
                    "Day-of-month 31 skips months with fewer days",
                    "Infinite loop if no valid date exists"
                ]
            },
            {
                "name": "Job Queue with Priorities",
                "description": "Implement a job queue with priorities, delayed execution, and deduplication.",
                "hints": {
                    "level1": "Use sorted set for priority queue with score = priority + timestamp.",
                    "level2": "Support delayed jobs by storing with future timestamp.",
                    "level3": """```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any
import uuid
import json
import hashlib

class JobStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD = "dead"  # Max retries exceeded

@dataclass
class Job:
    id: str
    queue: str
    type: str
    payload: dict
    priority: int = 0  # Higher = more important
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    result: Any = None
    unique_key: Optional[str] = None  # For deduplication

    @classmethod
    def create(cls, queue: str, job_type: str, payload: dict,
               priority: int = 0, delay: timedelta = None,
               unique_key: str = None, max_retries: int = 3) -> 'Job':
        job = cls(
            id=str(uuid.uuid4()),
            queue=queue,
            type=job_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
            unique_key=unique_key
        )
        if delay:
            job.scheduled_at = datetime.utcnow() + delay
            job.status = JobStatus.SCHEDULED
        return job

    def score(self) -> float:
        \"\"\"Calculate priority score for sorted set.\"\"\"
        # Lower score = higher priority (processed first)
        # Base: scheduled time (or created time)
        base_time = (self.scheduled_at or self.created_at).timestamp()
        # Subtract priority to make higher priority jobs come first
        return base_time - (self.priority * 1000)

class JobQueue:
    def __init__(self, redis):
        self.redis = redis

    async def enqueue(self, job: Job) -> bool:
        \"\"\"Add job to queue. Returns False if duplicate.\"\"\"
        # Check for duplicate
        if job.unique_key:
            exists = await self.redis.exists(f"job:unique:{job.unique_key}")
            if exists:
                return False
            # Set unique key with TTL
            await self.redis.setex(
                f"job:unique:{job.unique_key}",
                86400,  # 24 hours
                job.id
            )

        # Store job data
        await self.redis.hset(f"job:{job.id}", mapping={
            "data": json.dumps(self._serialize_job(job))
        })

        # Add to queue
        if job.status == JobStatus.SCHEDULED:
            await self.redis.zadd(f"queue:{job.queue}:scheduled", {job.id: job.score()})
        else:
            await self.redis.zadd(f"queue:{job.queue}:pending", {job.id: job.score()})

        return True

    async def dequeue(self, queue: str, worker_id: str,
                     visibility_timeout: int = 300) -> Optional[Job]:
        \"\"\"Fetch next job from queue.\"\"\"
        # First, move any scheduled jobs that are due
        await self._promote_scheduled(queue)

        # Atomically pop job and add to processing set
        job_id = await self._atomic_claim(queue, worker_id, visibility_timeout)
        if not job_id:
            return None

        job = await self._get_job(job_id)
        if job:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            job.attempts += 1
            await self._update_job(job)

        return job

    async def complete(self, job: Job, result: Any = None):
        \"\"\"Mark job as completed.\"\"\"
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.result = result

        await self._update_job(job)
        await self.redis.zrem(f"queue:{job.queue}:processing", job.id)

        # Clean up unique key
        if job.unique_key:
            await self.redis.delete(f"job:unique:{job.unique_key}")

    async def fail(self, job: Job, error: str):
        \"\"\"Mark job as failed, possibly retry.\"\"\"
        job.error = error

        if job.attempts < job.max_retries:
            # Retry with exponential backoff
            delay = timedelta(seconds=2 ** job.attempts * 60)
            job.scheduled_at = datetime.utcnow() + delay
            job.status = JobStatus.SCHEDULED

            await self._update_job(job)
            await self.redis.zrem(f"queue:{job.queue}:processing", job.id)
            await self.redis.zadd(f"queue:{job.queue}:scheduled", {job.id: job.score()})
        else:
            job.status = JobStatus.DEAD
            job.completed_at = datetime.utcnow()
            await self._update_job(job)
            await self.redis.zrem(f"queue:{job.queue}:processing", job.id)
            await self.redis.zadd(f"queue:{job.queue}:dead", {job.id: job.score()})

    async def _promote_scheduled(self, queue: str):
        \"\"\"Move scheduled jobs that are due to pending.\"\"\"
        now = datetime.utcnow().timestamp()
        due_jobs = await self.redis.zrangebyscore(
            f"queue:{queue}:scheduled", 0, now, limit=100
        )

        for job_id in due_jobs:
            job = await self._get_job(job_id)
            if job:
                job.status = JobStatus.PENDING
                await self._update_job(job)
                await self.redis.zrem(f"queue:{queue}:scheduled", job_id)
                await self.redis.zadd(f"queue:{queue}:pending", {job_id: job.score()})

    async def _atomic_claim(self, queue: str, worker_id: str,
                           timeout: int) -> Optional[str]:
        # Use Lua script for atomic claim
        script = \"\"\"
        local job_id = redis.call('ZRANGE', KEYS[1], 0, 0)[1]
        if job_id then
            redis.call('ZREM', KEYS[1], job_id)
            redis.call('ZADD', KEYS[2], ARGV[1], job_id)
            redis.call('HSET', 'job:' .. job_id, 'worker', ARGV[2])
            return job_id
        end
        return nil
        \"\"\"
        deadline = datetime.utcnow().timestamp() + timeout
        return await self.redis.eval(
            script,
            2,
            f"queue:{queue}:pending",
            f"queue:{queue}:processing",
            deadline,
            worker_id
        )
```"""
                },
                "pitfalls": [
                    "Race condition between claim and processing",
                    "Visibility timeout too short causes duplicate processing",
                    "Deduplication key collision across different job types"
                ]
            },
            {
                "name": "Worker Coordination",
                "description": "Coordinate multiple workers with leader election, heartbeats, and job recovery.",
                "hints": {
                    "level1": "Workers send heartbeats, leader monitors for failures.",
                    "level2": "Recover jobs from dead workers by checking processing timeout.",
                    "level3": """```python
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional
import uuid

@dataclass
class Worker:
    id: str
    queues: list[str]
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    current_job: Optional[str] = None
    processed: int = 0
    failed: int = 0

class WorkerCoordinator:
    def __init__(self, redis, job_queue: JobQueue):
        self.redis = redis
        self.job_queue = job_queue
        self.heartbeat_interval = 30
        self.worker_timeout = 90
        self._workers: dict[str, Worker] = {}
        self._running = False
        self._is_leader = False

    async def start_worker(self, queues: list[str], handlers: dict[str, Callable]):
        \"\"\"Start a worker process.\"\"\"
        worker = Worker(
            id=str(uuid.uuid4()),
            queues=queues
        )
        self._workers[worker.id] = worker

        # Start background tasks
        self._running = True
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(worker))
        leader_task = asyncio.create_task(self._leader_election())
        process_task = asyncio.create_task(self._process_loop(worker, handlers))

        await asyncio.gather(heartbeat_task, leader_task, process_task)

    async def _heartbeat_loop(self, worker: Worker):
        while self._running:
            await self.redis.hset(
                f"worker:{worker.id}",
                mapping={
                    "last_heartbeat": datetime.utcnow().isoformat(),
                    "current_job": worker.current_job or "",
                    "processed": worker.processed,
                    "failed": worker.failed
                }
            )
            await self.redis.expire(f"worker:{worker.id}", self.worker_timeout * 2)
            await asyncio.sleep(self.heartbeat_interval)

    async def _leader_election(self):
        \"\"\"Try to become leader for cleanup tasks.\"\"\"
        leader_key = "scheduler:leader"
        while self._running:
            # Try to acquire leadership
            acquired = await self.redis.set(
                leader_key,
                self._workers[list(self._workers.keys())[0]].id,
                nx=True,
                ex=self.heartbeat_interval * 2
            )

            if acquired:
                self._is_leader = True
                await self._leader_tasks()
            else:
                self._is_leader = False

            await asyncio.sleep(self.heartbeat_interval)

    async def _leader_tasks(self):
        \"\"\"Tasks only leader performs.\"\"\"
        # Check for dead workers
        await self._recover_dead_worker_jobs()
        # Clean up old completed jobs
        await self._cleanup_old_jobs()

    async def _recover_dead_worker_jobs(self):
        \"\"\"Recover jobs from workers that haven't sent heartbeat.\"\"\"
        # Get all workers
        worker_keys = await self.redis.keys("worker:*")

        for key in worker_keys:
            worker_data = await self.redis.hgetall(key)
            if not worker_data:
                continue

            last_heartbeat = datetime.fromisoformat(worker_data.get("last_heartbeat", ""))
            if datetime.utcnow() - last_heartbeat > timedelta(seconds=self.worker_timeout):
                # Worker is dead, recover its job
                current_job = worker_data.get("current_job")
                if current_job:
                    await self._requeue_job(current_job)

                await self.redis.delete(key)

    async def _requeue_job(self, job_id: str):
        \"\"\"Put job back in pending queue.\"\"\"
        job = await self.job_queue._get_job(job_id)
        if job and job.status == JobStatus.RUNNING:
            job.status = JobStatus.PENDING
            job.started_at = None
            await self.job_queue._update_job(job)

            # Move from processing to pending
            for queue in job.queue:
                await self.redis.zrem(f"queue:{queue}:processing", job_id)
                await self.redis.zadd(f"queue:{queue}:pending", {job_id: job.score()})

    async def _process_loop(self, worker: Worker, handlers: dict[str, Callable]):
        \"\"\"Main processing loop.\"\"\"
        while self._running:
            for queue in worker.queues:
                job = await self.job_queue.dequeue(queue, worker.id)

                if job:
                    worker.current_job = job.id
                    try:
                        handler = handlers.get(job.type)
                        if handler:
                            result = await handler(job.payload)
                            await self.job_queue.complete(job, result)
                            worker.processed += 1
                        else:
                            await self.job_queue.fail(job, f"No handler for job type: {job.type}")
                            worker.failed += 1
                    except Exception as e:
                        await self.job_queue.fail(job, str(e))
                        worker.failed += 1
                    finally:
                        worker.current_job = None

            await asyncio.sleep(1)  # Polling interval

    async def _cleanup_old_jobs(self, days: int = 7):
        \"\"\"Remove old completed/dead jobs.\"\"\"
        cutoff = (datetime.utcnow() - timedelta(days=days)).timestamp()
        # Clean up each queue's completed/dead sets
        queues = await self.redis.smembers("queues")
        for queue in queues:
            await self.redis.zremrangebyscore(f"queue:{queue}:completed", 0, cutoff)
            await self.redis.zremrangebyscore(f"queue:{queue}:dead", 0, cutoff)
```"""
                },
                "pitfalls": [
                    "Leader election split-brain with network partition",
                    "Job recovered while still processing causes duplicate",
                    "Worker heartbeat failure during long job"
                ]
            }
        ]
    },

    "audit-logging": {
        "name": "Audit Logging System",
        "description": "Build an immutable audit log system for compliance with tamper detection, retention policies, and efficient querying.",
        "why_important": "Audit logs are required for compliance (SOC2, HIPAA, GDPR). Understanding audit systems helps design secure, compliant applications.",
        "difficulty": "intermediate",
        "tags": ["security", "compliance", "backend"],
        "estimated_hours": 30,
        "prerequisites": ["log-aggregator"],
        "learning_outcomes": [
            "Design immutable append-only log storage",
            "Implement hash chain for tamper detection",
            "Build efficient audit log querying",
            "Handle retention and archival policies"
        ],
        "milestones": [
            {
                "name": "Audit Event Model",
                "description": "Design audit event schema with actor, action, resource, and context information.",
                "hints": {
                    "level1": "Capture who did what to which resource and when.",
                    "level2": "Include request context: IP, user agent, session ID.",
                    "level3": """```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import uuid
import json

class AuditAction(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_CHANGE = "permission_change"
    EXPORT = "export"
    IMPORT = "import"

class AuditOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"

@dataclass
class Actor:
    id: str
    type: str  # user, service, system
    name: str
    email: Optional[str] = None
    roles: list[str] = field(default_factory=list)

@dataclass
class Resource:
    id: str
    type: str  # document, user, setting, etc.
    name: Optional[str] = None
    attributes: dict = field(default_factory=dict)

@dataclass
class RequestContext:
    ip_address: str
    user_agent: str
    session_id: Optional[str] = None
    request_id: str = ""
    correlation_id: Optional[str] = None
    geo_location: Optional[dict] = None

@dataclass
class AuditEvent:
    id: str
    timestamp: datetime
    actor: Actor
    action: AuditAction
    resource: Resource
    outcome: AuditOutcome
    context: RequestContext
    changes: Optional[dict] = None  # Before/after for updates
    metadata: dict = field(default_factory=dict)
    sequence: int = 0  # For ordering
    hash: str = ""  # Chain hash

    @classmethod
    def create(cls, actor: Actor, action: AuditAction, resource: Resource,
               outcome: AuditOutcome, context: RequestContext,
               changes: dict = None, metadata: dict = None) -> 'AuditEvent':
        return cls(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            context=context,
            changes=changes,
            metadata=metadata or {}
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "actor": {
                "id": self.actor.id,
                "type": self.actor.type,
                "name": self.actor.name,
                "email": self.actor.email,
                "roles": self.actor.roles
            },
            "action": self.action.value,
            "resource": {
                "id": self.resource.id,
                "type": self.resource.type,
                "name": self.resource.name,
                "attributes": self.resource.attributes
            },
            "outcome": self.outcome.value,
            "context": {
                "ip_address": self.context.ip_address,
                "user_agent": self.context.user_agent,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id
            },
            "changes": self.changes,
            "metadata": self.metadata,
            "sequence": self.sequence,
            "hash": self.hash
        }

    def compute_hash(self, previous_hash: str = "") -> str:
        import hashlib
        data = json.dumps({
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor.id,
            "action": self.action.value,
            "resource_id": self.resource.id,
            "outcome": self.outcome.value,
            "previous_hash": previous_hash
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

class AuditEventBuilder:
    \"\"\"Fluent builder for audit events.\"\"\"

    def __init__(self):
        self._actor = None
        self._action = None
        self._resource = None
        self._outcome = AuditOutcome.SUCCESS
        self._context = None
        self._changes = None
        self._metadata = {}

    def actor(self, id: str, type: str, name: str, **kwargs) -> 'AuditEventBuilder':
        self._actor = Actor(id=id, type=type, name=name, **kwargs)
        return self

    def action(self, action: AuditAction) -> 'AuditEventBuilder':
        self._action = action
        return self

    def resource(self, id: str, type: str, name: str = None, **kwargs) -> 'AuditEventBuilder':
        self._resource = Resource(id=id, type=type, name=name, **kwargs)
        return self

    def outcome(self, outcome: AuditOutcome) -> 'AuditEventBuilder':
        self._outcome = outcome
        return self

    def context(self, ip: str, user_agent: str, **kwargs) -> 'AuditEventBuilder':
        self._context = RequestContext(ip_address=ip, user_agent=user_agent, **kwargs)
        return self

    def changes(self, before: dict, after: dict) -> 'AuditEventBuilder':
        self._changes = {"before": before, "after": after}
        return self

    def meta(self, **kwargs) -> 'AuditEventBuilder':
        self._metadata.update(kwargs)
        return self

    def build(self) -> AuditEvent:
        return AuditEvent.create(
            actor=self._actor,
            action=self._action,
            resource=self._resource,
            outcome=self._outcome,
            context=self._context,
            changes=self._changes,
            metadata=self._metadata
        )
```"""
                },
                "pitfalls": [
                    "PII in audit logs violates GDPR right to erasure",
                    "Missing actor context makes logs useless for investigation",
                    "Async logging loses context in distributed systems"
                ]
            },
            {
                "name": "Immutable Storage with Hash Chain",
                "description": "Implement append-only storage with hash chain linking for tamper detection.",
                "hints": {
                    "level1": "Each event hash includes hash of previous event.",
                    "level2": "Store chain anchors periodically for verification.",
                    "level3": """```python
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import hashlib

@dataclass
class ChainAnchor:
    sequence: int
    hash: str
    timestamp: datetime
    event_count: int

class ImmutableAuditStore:
    def __init__(self, storage, anchor_interval: int = 1000):
        self.storage = storage
        self.anchor_interval = anchor_interval
        self._last_hash = ""
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def initialize(self):
        \"\"\"Load last hash and sequence from storage.\"\"\"
        last_event = await self.storage.get_last_event()
        if last_event:
            self._last_hash = last_event.hash
            self._sequence = last_event.sequence

    async def append(self, event: AuditEvent) -> AuditEvent:
        async with self._lock:
            self._sequence += 1
            event.sequence = self._sequence
            event.hash = event.compute_hash(self._last_hash)

            # Append to storage (immutable)
            await self.storage.append(event)

            self._last_hash = event.hash

            # Create anchor periodically
            if self._sequence % self.anchor_interval == 0:
                await self._create_anchor()

            return event

    async def _create_anchor(self):
        anchor = ChainAnchor(
            sequence=self._sequence,
            hash=self._last_hash,
            timestamp=datetime.utcnow(),
            event_count=self._sequence
        )
        await self.storage.store_anchor(anchor)

    async def verify_chain(self, start_seq: int = 1, end_seq: int = None) -> bool:
        \"\"\"Verify hash chain integrity.\"\"\"
        end_seq = end_seq or self._sequence

        events = await self.storage.get_range(start_seq, end_seq)

        previous_hash = ""
        if start_seq > 1:
            prev_event = await self.storage.get_event(start_seq - 1)
            previous_hash = prev_event.hash

        for event in events:
            expected_hash = event.compute_hash(previous_hash)
            if event.hash != expected_hash:
                return False
            previous_hash = event.hash

        return True

    async def verify_from_anchor(self, anchor: ChainAnchor) -> bool:
        \"\"\"Verify chain from anchor point.\"\"\"
        # Get events since anchor
        events = await self.storage.get_range(anchor.sequence, self._sequence)

        if not events:
            return anchor.hash == self._last_hash

        # First event should chain from anchor
        first_event = events[0]
        if first_event.sequence == anchor.sequence:
            if first_event.hash != anchor.hash:
                return False
            events = events[1:]

        return await self.verify_chain(anchor.sequence + 1, self._sequence)

    async def get_proof(self, event_id: str) -> dict:
        \"\"\"Generate proof of inclusion for an event.\"\"\"
        event = await self.storage.get_event_by_id(event_id)
        if not event:
            return None

        # Find nearest anchor before event
        anchor = await self.storage.get_nearest_anchor(event.sequence)

        # Get chain from anchor to event
        chain = await self.storage.get_range(
            anchor.sequence if anchor else 1,
            event.sequence
        )

        return {
            "event": event.to_dict(),
            "anchor": anchor,
            "chain_hashes": [e.hash for e in chain],
            "verified": await self.verify_chain(
                anchor.sequence if anchor else 1,
                event.sequence
            )
        }

class PostgresAuditStorage:
    \"\"\"PostgreSQL storage with append-only table.\"\"\"

    def __init__(self, pool):
        self.pool = pool

    async def append(self, event: AuditEvent):
        async with self.pool.acquire() as conn:
            await conn.execute(\"\"\"
                INSERT INTO audit_log (
                    id, sequence, timestamp, actor_id, actor_type, actor_name,
                    action, resource_id, resource_type, outcome,
                    context, changes, metadata, hash
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            \"\"\",
                event.id, event.sequence, event.timestamp,
                event.actor.id, event.actor.type, event.actor.name,
                event.action.value, event.resource.id, event.resource.type,
                event.outcome.value, json.dumps(event.context.__dict__),
                json.dumps(event.changes), json.dumps(event.metadata), event.hash
            )

    async def get_range(self, start: int, end: int) -> list[AuditEvent]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(\"\"\"
                SELECT * FROM audit_log
                WHERE sequence >= $1 AND sequence <= $2
                ORDER BY sequence
            \"\"\", start, end)
            return [self._row_to_event(row) for row in rows]
```"""
                },
                "pitfalls": [
                    "Hash chain breaks if events inserted out of order",
                    "Chain verification expensive on large datasets",
                    "Backup/restore must preserve hash chain integrity"
                ]
            },
            {
                "name": "Audit Query & Export",
                "description": "Build efficient querying for audit logs with filtering, search, and compliance export formats.",
                "hints": {
                    "level1": "Index on actor, resource, action, and timestamp.",
                    "level2": "Support date range, actor, and resource filtering.",
                    "level3": """```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    SIEM = "siem"  # CEF/LEEF format

@dataclass
class AuditQuery:
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    actor_ids: list[str] = None
    actor_types: list[str] = None
    actions: list[AuditAction] = None
    resource_ids: list[str] = None
    resource_types: list[str] = None
    outcomes: list[AuditOutcome] = None
    ip_addresses: list[str] = None
    search_text: Optional[str] = None
    limit: int = 100
    offset: int = 0

class AuditQueryEngine:
    def __init__(self, storage):
        self.storage = storage

    async def query(self, q: AuditQuery) -> list[AuditEvent]:
        \"\"\"Execute audit query.\"\"\"
        conditions = []
        params = []
        param_idx = 1

        if q.start_time:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(q.start_time)
            param_idx += 1

        if q.end_time:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(q.end_time)
            param_idx += 1

        if q.actor_ids:
            conditions.append(f"actor_id = ANY(${param_idx})")
            params.append(q.actor_ids)
            param_idx += 1

        if q.actions:
            conditions.append(f"action = ANY(${param_idx})")
            params.append([a.value for a in q.actions])
            param_idx += 1

        if q.resource_types:
            conditions.append(f"resource_type = ANY(${param_idx})")
            params.append(q.resource_types)
            param_idx += 1

        if q.outcomes:
            conditions.append(f"outcome = ANY(${param_idx})")
            params.append([o.value for o in q.outcomes])
            param_idx += 1

        if q.search_text:
            conditions.append(f"search_vector @@ plainto_tsquery(${param_idx})")
            params.append(q.search_text)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f\"\"\"
            SELECT * FROM audit_log
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        \"\"\"
        params.extend([q.limit, q.offset])

        rows = await self.storage.fetch(query, *params)
        return [self._row_to_event(row) for row in rows]

    async def export(self, q: AuditQuery, format: ExportFormat) -> bytes:
        \"\"\"Export audit logs in specified format.\"\"\"
        events = await self.query(q)

        if format == ExportFormat.JSON:
            return self._export_json(events)
        elif format == ExportFormat.CSV:
            return self._export_csv(events)
        elif format == ExportFormat.SIEM:
            return self._export_cef(events)

    def _export_json(self, events: list[AuditEvent]) -> bytes:
        import json
        return json.dumps([e.to_dict() for e in events], indent=2).encode()

    def _export_csv(self, events: list[AuditEvent]) -> bytes:
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'timestamp', 'actor_id', 'actor_name', 'action',
            'resource_type', 'resource_id', 'outcome', 'ip_address'
        ])

        for event in events:
            writer.writerow([
                event.timestamp.isoformat(),
                event.actor.id,
                event.actor.name,
                event.action.value,
                event.resource.type,
                event.resource.id,
                event.outcome.value,
                event.context.ip_address
            ])

        return output.getvalue().encode()

    def _export_cef(self, events: list[AuditEvent]) -> bytes:
        \"\"\"Export in Common Event Format for SIEM integration.\"\"\"
        lines = []
        for event in events:
            # CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
            severity = 3 if event.outcome == AuditOutcome.SUCCESS else 7
            extension = (
                f"src={event.context.ip_address} "
                f"suser={event.actor.name} "
                f"duser={event.resource.id} "
                f"act={event.action.value} "
                f"outcome={event.outcome.value}"
            )
            cef_line = (
                f"CEF:0|MyApp|AuditLog|1.0|{event.action.value}|"
                f"{event.action.value} {event.resource.type}|{severity}|{extension}"
            )
            lines.append(cef_line)

        return "\\n".join(lines).encode()

    async def get_activity_summary(self, actor_id: str,
                                   start: datetime, end: datetime) -> dict:
        \"\"\"Get activity summary for an actor.\"\"\"
        rows = await self.storage.fetch(\"\"\"
            SELECT action, outcome, COUNT(*) as count
            FROM audit_log
            WHERE actor_id = $1 AND timestamp BETWEEN $2 AND $3
            GROUP BY action, outcome
        \"\"\", actor_id, start, end)

        return {
            "actor_id": actor_id,
            "period": {"start": start, "end": end},
            "activity": [
                {"action": r["action"], "outcome": r["outcome"], "count": r["count"]}
                for r in rows
            ]
        }
```"""
                },
                "pitfalls": [
                    "Full text search without index causes table scan",
                    "Large exports exhaust memory (use streaming)",
                    "Time zone handling inconsistent in queries"
                ]
            }
        ]
    },

    "rate-limiter-distributed": {
        "name": "Distributed Rate Limiter",
        "description": "Build a distributed rate limiter supporting multiple algorithms with Redis backend for cluster-wide limiting.",
        "why_important": "Rate limiting protects services from abuse and ensures fair resource usage. Understanding algorithms helps choose the right approach.",
        "difficulty": "intermediate",
        "tags": ["backend", "distributed-systems", "reliability"],
        "estimated_hours": 25,
        "prerequisites": ["redis-clone"],
        "learning_outcomes": [
            "Implement token bucket and sliding window algorithms",
            "Design distributed rate limiting with Redis",
            "Handle burst traffic gracefully",
            "Build multi-tier rate limiting"
        ],
        "milestones": [
            {
                "name": "Rate Limiting Algorithms",
                "description": "Implement token bucket, sliding window log, and sliding window counter algorithms.",
                "hints": {
                    "level1": "Token bucket: refill tokens at fixed rate, consume on request.",
                    "level2": "Sliding window: track requests in time window, count or log-based.",
                    "level3": """```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import time
import math

@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_at: float  # Unix timestamp
    retry_after: float = 0  # Seconds

class RateLimiter(ABC):
    @abstractmethod
    async def is_allowed(self, key: str) -> RateLimitResult:
        pass

class TokenBucketLimiter(RateLimiter):
    def __init__(self, redis, capacity: int, refill_rate: float):
        self.redis = redis
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second

    async def is_allowed(self, key: str) -> RateLimitResult:
        now = time.time()
        bucket_key = f"ratelimit:bucket:{key}"

        # Lua script for atomic token bucket
        script = \"\"\"
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now

        -- Refill tokens
        local elapsed = now - last_refill
        local refill = elapsed * refill_rate
        tokens = math.min(capacity, tokens + refill)

        -- Try to consume
        local allowed = 0
        if tokens >= 1 then
            tokens = tokens - 1
            allowed = 1
        end

        -- Save state
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, math.ceil(capacity / refill_rate) + 1)

        return {allowed, tokens, capacity}
        \"\"\"

        result = await self.redis.eval(
            script, 1, bucket_key,
            self.capacity, self.refill_rate, now
        )

        allowed, remaining, capacity = result
        reset_at = now + (capacity - remaining) / self.refill_rate

        return RateLimitResult(
            allowed=bool(allowed),
            remaining=int(remaining),
            reset_at=reset_at,
            retry_after=0 if allowed else 1.0 / self.refill_rate
        )

class SlidingWindowLogLimiter(RateLimiter):
    \"\"\"Precise but memory-intensive sliding window.\"\"\"

    def __init__(self, redis, limit: int, window_seconds: int):
        self.redis = redis
        self.limit = limit
        self.window = window_seconds

    async def is_allowed(self, key: str) -> RateLimitResult:
        now = time.time()
        window_start = now - self.window
        log_key = f"ratelimit:log:{key}"

        # Lua script for atomic sliding window
        script = \"\"\"
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local window = tonumber(ARGV[4])

        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

        -- Count current entries
        local count = redis.call('ZCARD', key)

        local allowed = 0
        if count < limit then
            -- Add new entry
            redis.call('ZADD', key, now, now .. ':' .. math.random())
            allowed = 1
            count = count + 1
        end

        redis.call('EXPIRE', key, window + 1)

        -- Get oldest entry for reset time
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        local reset_at = oldest[2] and (tonumber(oldest[2]) + window) or (now + window)

        return {allowed, limit - count, reset_at}
        \"\"\"

        result = await self.redis.eval(
            script, 1, log_key,
            self.limit, window_start, now, self.window
        )

        allowed, remaining, reset_at = result

        return RateLimitResult(
            allowed=bool(allowed),
            remaining=max(0, int(remaining)),
            reset_at=reset_at,
            retry_after=0 if allowed else reset_at - now
        )

class SlidingWindowCounterLimiter(RateLimiter):
    \"\"\"Memory-efficient approximate sliding window.\"\"\"

    def __init__(self, redis, limit: int, window_seconds: int):
        self.redis = redis
        self.limit = limit
        self.window = window_seconds

    async def is_allowed(self, key: str) -> RateLimitResult:
        now = time.time()
        current_window = int(now // self.window)
        previous_window = current_window - 1

        current_key = f"ratelimit:counter:{key}:{current_window}"
        previous_key = f"ratelimit:counter:{key}:{previous_window}"

        # Lua script for atomic counter
        script = \"\"\"
        local current_key = KEYS[1]
        local previous_key = KEYS[2]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local current_window = tonumber(ARGV[4])

        local previous_count = tonumber(redis.call('GET', previous_key)) or 0
        local current_count = tonumber(redis.call('GET', current_key)) or 0

        -- Calculate weighted count (sliding window approximation)
        local window_start = current_window * window
        local elapsed_ratio = (now - window_start) / window
        local weighted_count = previous_count * (1 - elapsed_ratio) + current_count

        local allowed = 0
        if weighted_count < limit then
            redis.call('INCR', current_key)
            redis.call('EXPIRE', current_key, window * 2)
            allowed = 1
            weighted_count = weighted_count + 1
        end

        return {allowed, math.floor(limit - weighted_count), window_start + window}
        \"\"\"

        result = await self.redis.eval(
            script, 2, current_key, previous_key,
            self.limit, self.window, now, current_window
        )

        allowed, remaining, reset_at = result

        return RateLimitResult(
            allowed=bool(allowed),
            remaining=max(0, int(remaining)),
            reset_at=reset_at,
            retry_after=0 if allowed else reset_at - now
        )
```"""
                },
                "pitfalls": [
                    "Token bucket allows burst above sustained rate",
                    "Sliding window log uses O(n) memory per key",
                    "Counter approximation can allow 2x limit at window boundary"
                ]
            },
            {
                "name": "Multi-tier Rate Limiting",
                "description": "Implement hierarchical rate limiting with per-user, per-API, and global limits.",
                "hints": {
                    "level1": "Check limits in order: user -> API -> global.",
                    "level2": "Use different algorithms for different tiers.",
                    "level3": """```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class LimitTier(str, Enum):
    USER = "user"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    GLOBAL = "global"

@dataclass
class RateLimitConfig:
    tier: LimitTier
    limit: int
    window_seconds: int
    burst_limit: Optional[int] = None  # For token bucket

@dataclass
class MultiTierResult:
    allowed: bool
    tier_results: dict[LimitTier, RateLimitResult]
    limiting_tier: Optional[LimitTier] = None

class MultiTierRateLimiter:
    def __init__(self, redis):
        self.redis = redis
        self.limiters: dict[LimitTier, tuple[RateLimitConfig, RateLimiter]] = {}

    def configure_tier(self, config: RateLimitConfig):
        if config.burst_limit:
            limiter = TokenBucketLimiter(
                self.redis,
                capacity=config.burst_limit,
                refill_rate=config.limit / config.window_seconds
            )
        else:
            limiter = SlidingWindowCounterLimiter(
                self.redis,
                limit=config.limit,
                window_seconds=config.window_seconds
            )
        self.limiters[config.tier] = (config, limiter)

    async def is_allowed(self, context: dict) -> MultiTierResult:
        \"\"\"Check all configured rate limit tiers.\"\"\"
        tier_results = {}
        limiting_tier = None

        # Check tiers in priority order
        tier_order = [LimitTier.USER, LimitTier.API_KEY, LimitTier.ENDPOINT, LimitTier.GLOBAL]

        for tier in tier_order:
            if tier not in self.limiters:
                continue

            config, limiter = self.limiters[tier]
            key = self._build_key(tier, context)

            result = await limiter.is_allowed(key)
            tier_results[tier] = result

            if not result.allowed:
                limiting_tier = tier
                break

        return MultiTierResult(
            allowed=limiting_tier is None,
            tier_results=tier_results,
            limiting_tier=limiting_tier
        )

    def _build_key(self, tier: LimitTier, context: dict) -> str:
        if tier == LimitTier.USER:
            return f"user:{context.get('user_id', 'anonymous')}"
        elif tier == LimitTier.API_KEY:
            return f"apikey:{context.get('api_key', 'none')}"
        elif tier == LimitTier.ENDPOINT:
            return f"endpoint:{context.get('endpoint', 'unknown')}"
        elif tier == LimitTier.GLOBAL:
            return "global"
        return "unknown"

class AdaptiveRateLimiter:
    \"\"\"Rate limiter that adjusts based on system load.\"\"\"

    def __init__(self, redis, base_limit: int, window: int):
        self.redis = redis
        self.base_limit = base_limit
        self.window = window
        self.limiter = SlidingWindowCounterLimiter(redis, base_limit, window)
        self._load_factor = 1.0

    async def update_load_factor(self, cpu_usage: float, error_rate: float):
        \"\"\"Adjust limits based on system health.\"\"\"
        # Reduce limit when system is stressed
        if cpu_usage > 0.8 or error_rate > 0.05:
            self._load_factor = max(0.5, self._load_factor - 0.1)
        elif cpu_usage < 0.5 and error_rate < 0.01:
            self._load_factor = min(1.5, self._load_factor + 0.1)

        # Update limiter with adjusted limit
        adjusted_limit = int(self.base_limit * self._load_factor)
        self.limiter = SlidingWindowCounterLimiter(
            self.redis, adjusted_limit, self.window
        )

    async def is_allowed(self, key: str) -> RateLimitResult:
        return await self.limiter.is_allowed(key)

# Middleware for HTTP frameworks
class RateLimitMiddleware:
    def __init__(self, limiter: MultiTierRateLimiter):
        self.limiter = limiter

    async def __call__(self, request, call_next):
        context = {
            "user_id": request.user.id if request.user else None,
            "api_key": request.headers.get("X-API-Key"),
            "endpoint": f"{request.method}:{request.path}",
            "ip": request.client.host
        }

        result = await self.limiter.is_allowed(context)

        # Add rate limit headers
        headers = {}
        for tier, tier_result in result.tier_results.items():
            headers[f"X-RateLimit-{tier.value}-Remaining"] = str(tier_result.remaining)
            headers[f"X-RateLimit-{tier.value}-Reset"] = str(int(tier_result.reset_at))

        if not result.allowed:
            return Response(
                status_code=429,
                headers={
                    **headers,
                    "Retry-After": str(int(result.tier_results[result.limiting_tier].retry_after))
                },
                content={"error": "Rate limit exceeded", "tier": result.limiting_tier.value}
            )

        response = await call_next(request)
        for key, value in headers.items():
            response.headers[key] = value
        return response
```"""
                },
                "pitfalls": [
                    "Checking all tiers even after one fails wastes resources",
                    "Different tier windows cause confusing UX",
                    "Adaptive limiting oscillates under variable load"
                ]
            }
        ]
    },

    "saga-orchestrator": {
        "name": "Saga Orchestrator",
        "description": "Build a saga orchestrator for managing distributed transactions with compensation and recovery.",
        "why_important": "Sagas solve the distributed transaction problem in microservices. Understanding saga patterns helps design reliable distributed workflows.",
        "difficulty": "advanced",
        "tags": ["distributed-systems", "microservices", "reliability"],
        "estimated_hours": 40,
        "prerequisites": ["job-scheduler", "event-sourcing"],
        "learning_outcomes": [
            "Design saga step definitions with compensations",
            "Implement orchestration vs choreography patterns",
            "Handle partial failures and rollback",
            "Build saga state machine with persistence"
        ],
        "milestones": [
            {
                "name": "Saga Definition",
                "description": "Define saga steps with forward actions and compensation handlers.",
                "hints": {
                    "level1": "Each step has an action and a compensating action.",
                    "level2": "Steps can be sequential or parallel, with dependencies.",
                    "level3": """```python
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from enum import Enum
import uuid

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"

@dataclass
class SagaStep:
    name: str
    action: Callable[[dict], Any]  # Forward action
    compensation: Callable[[dict, Any], None]  # Rollback action
    timeout: int = 300  # seconds
    retries: int = 3
    depends_on: list[str] = field(default_factory=list)

@dataclass
class StepExecution:
    step_name: str
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0

@dataclass
class SagaDefinition:
    name: str
    steps: list[SagaStep] = field(default_factory=list)

    def add_step(self, name: str, action: Callable, compensation: Callable,
                 depends_on: list[str] = None, **kwargs) -> 'SagaDefinition':
        self.steps.append(SagaStep(
            name=name,
            action=action,
            compensation=compensation,
            depends_on=depends_on or [],
            **kwargs
        ))
        return self

    def get_step(self, name: str) -> Optional[SagaStep]:
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_execution_order(self) -> list[list[str]]:
        \"\"\"Return steps grouped by parallel execution levels.\"\"\"
        remaining = {s.name: set(s.depends_on) for s in self.steps}
        levels = []

        while remaining:
            # Find steps with no pending dependencies
            ready = [name for name, deps in remaining.items() if not deps]
            if not ready:
                raise ValueError("Circular dependency detected")

            levels.append(ready)

            # Remove completed steps from dependencies
            for name in ready:
                del remaining[name]
            for deps in remaining.values():
                deps -= set(ready)

        return levels

# Example saga definition
def create_order_saga() -> SagaDefinition:
    saga = SagaDefinition(name="create_order")

    saga.add_step(
        name="reserve_inventory",
        action=lambda ctx: inventory_service.reserve(ctx["items"]),
        compensation=lambda ctx, result: inventory_service.release(result["reservation_id"])
    )

    saga.add_step(
        name="charge_payment",
        action=lambda ctx: payment_service.charge(ctx["payment_info"], ctx["total"]),
        compensation=lambda ctx, result: payment_service.refund(result["transaction_id"]),
        depends_on=["reserve_inventory"]
    )

    saga.add_step(
        name="create_shipment",
        action=lambda ctx: shipping_service.create(ctx["address"], ctx["items"]),
        compensation=lambda ctx, result: shipping_service.cancel(result["shipment_id"]),
        depends_on=["charge_payment"]
    )

    saga.add_step(
        name="send_confirmation",
        action=lambda ctx: notification_service.send_order_confirmation(ctx["email"], ctx["order_id"]),
        compensation=lambda ctx, result: None,  # No compensation needed
        depends_on=["create_shipment"]
    )

    return saga
```"""
                },
                "pitfalls": [
                    "Compensation that fails leaves inconsistent state",
                    "Parallel steps with shared state cause race conditions",
                    "Timeout too short fails legitimate slow operations"
                ]
            },
            {
                "name": "Saga Orchestrator Engine",
                "description": "Build the orchestrator that executes sagas, tracks state, and handles failures.",
                "hints": {
                    "level1": "Execute steps in dependency order, track results.",
                    "level2": "On failure, compensate completed steps in reverse order.",
                    "level3": """```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import asyncio

class SagaStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"  # Compensation also failed

@dataclass
class SagaExecution:
    id: str
    saga_name: str
    status: SagaStatus = SagaStatus.PENDING
    context: dict = field(default_factory=dict)
    step_executions: dict[str, StepExecution] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class SagaOrchestrator:
    def __init__(self, storage, event_bus=None):
        self.storage = storage
        self.event_bus = event_bus
        self.sagas: dict[str, SagaDefinition] = {}

    def register(self, saga: SagaDefinition):
        self.sagas[saga.name] = saga

    async def execute(self, saga_name: str, context: dict) -> SagaExecution:
        saga = self.sagas.get(saga_name)
        if not saga:
            raise ValueError(f"Unknown saga: {saga_name}")

        execution = SagaExecution(
            id=str(uuid.uuid4()),
            saga_name=saga_name,
            context=context,
            started_at=datetime.utcnow()
        )

        # Initialize step executions
        for step in saga.steps:
            execution.step_executions[step.name] = StepExecution(step_name=step.name)

        await self.storage.save(execution)

        try:
            await self._run_saga(saga, execution)
            execution.status = SagaStatus.COMPLETED
        except Exception as e:
            execution.error = str(e)
            await self._compensate(saga, execution)

        execution.completed_at = datetime.utcnow()
        await self.storage.save(execution)

        return execution

    async def _run_saga(self, saga: SagaDefinition, execution: SagaExecution):
        execution.status = SagaStatus.RUNNING
        await self.storage.save(execution)

        levels = saga.get_execution_order()

        for level in levels:
            # Execute parallel steps
            tasks = [
                self._execute_step(saga.get_step(name), execution)
                for name in level
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for failures
            for name, result in zip(level, results):
                if isinstance(result, Exception):
                    raise result

    async def _execute_step(self, step: SagaStep, execution: SagaExecution):
        step_exec = execution.step_executions[step.name]
        step_exec.status = StepStatus.RUNNING
        step_exec.started_at = datetime.utcnow()
        step_exec.attempts += 1

        await self.storage.save(execution)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._call_action(step.action, execution.context),
                timeout=step.timeout
            )

            step_exec.result = result
            step_exec.status = StepStatus.COMPLETED
            step_exec.completed_at = datetime.utcnow()

            # Update context with result
            execution.context[f"{step.name}_result"] = result

        except Exception as e:
            step_exec.error = str(e)
            step_exec.status = StepStatus.FAILED

            # Retry if attempts remaining
            if step_exec.attempts < step.retries:
                await asyncio.sleep(2 ** step_exec.attempts)  # Exponential backoff
                return await self._execute_step(step, execution)

            raise

        await self.storage.save(execution)

    async def _compensate(self, saga: SagaDefinition, execution: SagaExecution):
        execution.status = SagaStatus.COMPENSATING
        await self.storage.save(execution)

        # Get completed steps in reverse order
        levels = saga.get_execution_order()
        completed_steps = []

        for level in levels:
            for name in level:
                step_exec = execution.step_executions[name]
                if step_exec.status == StepStatus.COMPLETED:
                    completed_steps.append(name)

        # Compensate in reverse order
        for step_name in reversed(completed_steps):
            step = saga.get_step(step_name)
            step_exec = execution.step_executions[step_name]

            try:
                step_exec.status = StepStatus.COMPENSATING
                await self.storage.save(execution)

                await self._call_action(
                    step.compensation,
                    execution.context,
                    step_exec.result
                )

                step_exec.status = StepStatus.COMPENSATED
            except Exception as e:
                step_exec.error = f"Compensation failed: {e}"
                execution.status = SagaStatus.FAILED
                await self.storage.save(execution)
                raise

        execution.status = SagaStatus.COMPENSATED
        await self.storage.save(execution)

    async def _call_action(self, action: Callable, *args):
        if asyncio.iscoroutinefunction(action):
            return await action(*args)
        return action(*args)

    async def resume(self, execution_id: str) -> SagaExecution:
        \"\"\"Resume a failed or interrupted saga.\"\"\"
        execution = await self.storage.load(execution_id)
        saga = self.sagas.get(execution.saga_name)

        if execution.status == SagaStatus.RUNNING:
            # Find incomplete steps and continue
            await self._run_saga(saga, execution)
        elif execution.status == SagaStatus.COMPENSATING:
            # Continue compensation
            await self._compensate(saga, execution)

        return execution
```"""
                },
                "pitfalls": [
                    "Resuming saga loses in-memory step results",
                    "Parallel compensation can conflict",
                    "Network partition causes duplicate saga execution"
                ]
            },
            {
                "name": "Saga State Persistence",
                "description": "Persist saga state for recovery and implement idempotent step execution.",
                "hints": {
                    "level1": "Store saga state after each step completion.",
                    "level2": "Use idempotency keys to prevent duplicate execution.",
                    "level3": """```python
from dataclasses import asdict
import json

class SagaStorage:
    def __init__(self, db):
        self.db = db

    async def save(self, execution: SagaExecution):
        await self.db.execute(\"\"\"
            INSERT INTO saga_executions (
                id, saga_name, status, context, step_executions,
                started_at, completed_at, error
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE SET
                status = $3, context = $4, step_executions = $5,
                completed_at = $7, error = $8
        \"\"\",
            execution.id,
            execution.saga_name,
            execution.status.value,
            json.dumps(execution.context),
            json.dumps({k: asdict(v) for k, v in execution.step_executions.items()}),
            execution.started_at,
            execution.completed_at,
            execution.error
        )

    async def load(self, execution_id: str) -> SagaExecution:
        row = await self.db.fetchrow(
            "SELECT * FROM saga_executions WHERE id = $1",
            execution_id
        )
        if not row:
            raise ValueError(f"Saga execution not found: {execution_id}")

        step_data = json.loads(row["step_executions"])
        return SagaExecution(
            id=row["id"],
            saga_name=row["saga_name"],
            status=SagaStatus(row["status"]),
            context=json.loads(row["context"]),
            step_executions={
                k: StepExecution(**v) for k, v in step_data.items()
            },
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error=row["error"]
        )

    async def list_pending(self, saga_name: str = None) -> list[SagaExecution]:
        query = \"\"\"
            SELECT * FROM saga_executions
            WHERE status IN ('pending', 'running', 'compensating')
        \"\"\"
        if saga_name:
            query += " AND saga_name = $1"
            rows = await self.db.fetch(query, saga_name)
        else:
            rows = await self.db.fetch(query)

        return [await self.load(row["id"]) for row in rows]

class IdempotentStepExecutor:
    \"\"\"Ensures steps are executed exactly once.\"\"\"

    def __init__(self, storage, lock_manager):
        self.storage = storage
        self.lock_manager = lock_manager

    async def execute(self, saga_id: str, step_name: str,
                     action: Callable, *args) -> Any:
        idempotency_key = f"saga:{saga_id}:step:{step_name}"

        # Check if already executed
        existing = await self.storage.get_step_result(idempotency_key)
        if existing:
            return existing

        # Acquire lock
        async with self.lock_manager.lock(idempotency_key, timeout=300):
            # Double-check after acquiring lock
            existing = await self.storage.get_step_result(idempotency_key)
            if existing:
                return existing

            # Execute
            result = await action(*args)

            # Store result
            await self.storage.store_step_result(idempotency_key, result)

            return result

class SagaRecoveryService:
    \"\"\"Background service that recovers interrupted sagas.\"\"\"

    def __init__(self, orchestrator: SagaOrchestrator, storage: SagaStorage):
        self.orchestrator = orchestrator
        self.storage = storage
        self.check_interval = 60  # seconds

    async def start(self):
        while True:
            await self._recover_pending()
            await asyncio.sleep(self.check_interval)

    async def _recover_pending(self):
        pending = await self.storage.list_pending()

        for execution in pending:
            # Check if saga is actually stuck (no progress for a while)
            last_update = self._get_last_update(execution)
            if datetime.utcnow() - last_update > timedelta(minutes=5):
                try:
                    await self.orchestrator.resume(execution.id)
                except Exception as e:
                    print(f"Failed to recover saga {execution.id}: {e}")

    def _get_last_update(self, execution: SagaExecution) -> datetime:
        times = [execution.started_at]
        for step_exec in execution.step_executions.values():
            if step_exec.started_at:
                times.append(step_exec.started_at)
            if step_exec.completed_at:
                times.append(step_exec.completed_at)
        return max(t for t in times if t)
```"""
                },
                "pitfalls": [
                    "JSON serialization loses datetime precision",
                    "Lock timeout shorter than step timeout causes issues",
                    "Recovery loop processes same saga repeatedly"
                ]
            }
        ]
    }
}

# Load and update YAML
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

expert_projects = data.get('expert_projects', {})

for project_id, project_data in enterprise_projects.items():
    if project_id not in expert_projects:
        expert_projects[project_id] = project_data
        print(f"Added: {project_id}")
    else:
        print(f"Skipped (exists): {project_id}")

# Save
with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nTotal expert_projects: {len(expert_projects)}")

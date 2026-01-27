#!/usr/bin/env python3
"""
Add Testing & Reliability projects - load testing and chaos engineering.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

testing_reliability_projects = {
    "load-testing-framework": {
        "name": "Distributed Load Testing Framework",
        "description": "Build a load testing tool like k6/Locust with distributed workers, realistic user simulation, and real-time metrics.",
        "why_expert": "Performance testing prevents outages. Understanding load generation, metrics collection, and bottleneck analysis is crucial for production systems.",
        "difficulty": "expert",
        "tags": ["testing", "performance", "load-testing", "distributed", "metrics"],
        "estimated_hours": 45,
        "prerequisites": ["build-http-server"],
        "milestones": [
            {
                "name": "Virtual User Simulation",
                "description": "Implement virtual users with realistic think times and behavior",
                "skills": ["User simulation", "HTTP client", "Think times"],
                "hints": {
                    "level1": "Virtual user executes scenario repeatedly with configurable think times",
                    "level2": "Simulate realistic patterns: login, browse, action, logout",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum
import asyncio
import aiohttp
import random
import time

@dataclass
class RequestMetrics:
    method: str
    url: str
    status_code: int
    response_time_ms: float
    response_size: int
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class LoadProfile(Enum):
    CONSTANT = "constant"      # Fixed number of VUs
    RAMP_UP = "ramp_up"        # Gradually increase
    STEP = "step"              # Step increases
    SPIKE = "spike"            # Sudden spike

@dataclass
class StageConfig:
    duration: int      # Seconds
    target_vus: int    # Target virtual users
    think_time: tuple[float, float] = (1.0, 3.0)  # Min, max think time

class VirtualUser:
    def __init__(self, user_id: int, scenario: 'Scenario',
                 metrics_callback: Callable[[RequestMetrics], None]):
        self.user_id = user_id
        self.scenario = scenario
        self.metrics_callback = metrics_callback
        self.session: Optional[aiohttp.ClientSession] = None
        self.cookies = {}
        self.running = False

    async def start(self):
        self.running = True
        self.session = aiohttp.ClientSession()

        try:
            while self.running:
                await self._run_iteration()
        finally:
            await self.session.close()

    async def stop(self):
        self.running = False

    async def _run_iteration(self):
        '''Run one complete scenario iteration'''
        for step in self.scenario.steps:
            if not self.running:
                break

            # Execute step
            metrics = await self._execute_step(step)
            self.metrics_callback(metrics)

            # Think time
            if step.think_time:
                await asyncio.sleep(random.uniform(*step.think_time))

    async def _execute_step(self, step: 'ScenarioStep') -> RequestMetrics:
        start_time = time.time()

        try:
            async with self.session.request(
                method=step.method,
                url=step.url,
                headers=step.headers,
                json=step.body if step.body else None,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                body = await response.read()

                return RequestMetrics(
                    method=step.method,
                    url=step.url,
                    status_code=response.status,
                    response_time_ms=(time.time() - start_time) * 1000,
                    response_size=len(body)
                )

        except Exception as e:
            return RequestMetrics(
                method=step.method,
                url=step.url,
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                response_size=0,
                error=str(e)
            )

@dataclass
class ScenarioStep:
    name: str
    method: str
    url: str
    headers: dict = field(default_factory=dict)
    body: Optional[dict] = None
    think_time: Optional[tuple[float, float]] = None
    check: Optional[Callable] = None  # Response validation

@dataclass
class Scenario:
    name: str
    steps: list[ScenarioStep]
    weight: int = 1  # For weighted random selection

class LoadRunner:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.scenarios: list[Scenario] = []
        self.stages: list[StageConfig] = []
        self.virtual_users: list[VirtualUser] = []
        self.metrics: list[RequestMetrics] = []
        self.running = False

    def add_scenario(self, scenario: Scenario):
        self.scenarios.append(scenario)

    def add_stage(self, duration: int, target_vus: int, **kwargs):
        self.stages.append(StageConfig(
            duration=duration,
            target_vus=target_vus,
            **kwargs
        ))

    def collect_metric(self, metric: RequestMetrics):
        self.metrics.append(metric)

    async def run(self):
        self.running = True
        start_time = time.time()

        for stage in self.stages:
            stage_start = time.time()

            # Adjust VU count
            await self._scale_to(stage.target_vus)

            # Wait for stage duration
            while time.time() - stage_start < stage.duration:
                if not self.running:
                    break
                await asyncio.sleep(0.1)

        # Cleanup
        await self._scale_to(0)
        self.running = False

    async def _scale_to(self, target: int):
        current = len(self.virtual_users)

        if target > current:
            # Add VUs
            for i in range(current, target):
                scenario = random.choice(self.scenarios)  # Could weight
                vu = VirtualUser(i, scenario, self.collect_metric)
                self.virtual_users.append(vu)
                asyncio.create_task(vu.start())

        elif target < current:
            # Remove VUs
            for vu in self.virtual_users[target:]:
                await vu.stop()
            self.virtual_users = self.virtual_users[:target]
```
"""
                },
                "pitfalls": [
                    "Think time prevents unrealistic load - real users pause between actions",
                    "Connection pooling affects results - configure properly",
                    "Coordinated omission: measure time from request creation, not send",
                    "Virtual user state: some tests need session/cookie persistence"
                ]
            },
            {
                "name": "Distributed Workers",
                "description": "Implement distributed load generation with coordinator and workers",
                "skills": ["Distributed coordination", "Worker management", "Aggregation"],
                "hints": {
                    "level1": "Coordinator divides work; workers generate load and report metrics",
                    "level2": "Workers heartbeat to coordinator; auto-rebalance on failure",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import json
import time
from enum import Enum

class WorkerStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class Worker:
    id: str
    host: str
    port: int
    status: WorkerStatus
    current_vus: int = 0
    last_heartbeat: float = 0
    metrics_count: int = 0

@dataclass
class TestConfig:
    scenarios: list[dict]
    stages: list[dict]
    base_url: str

class Coordinator:
    def __init__(self, port: int = 5000):
        self.port = port
        self.workers: dict[str, Worker] = {}
        self.test_config: Optional[TestConfig] = None
        self.running = False
        self.aggregated_metrics: list[RequestMetrics] = []

    async def register_worker(self, worker_id: str, host: str, port: int) -> Worker:
        worker = Worker(
            id=worker_id,
            host=host,
            port=port,
            status=WorkerStatus.IDLE,
            last_heartbeat=time.time()
        )
        self.workers[worker_id] = worker
        return worker

    async def start_test(self, config: TestConfig):
        self.test_config = config
        self.running = True

        # Divide stages among workers
        worker_count = len(self.workers)
        if worker_count == 0:
            raise ValueError("No workers available")

        for stage in config.stages:
            # Distribute VUs evenly
            vus_per_worker = stage['target_vus'] // worker_count
            remainder = stage['target_vus'] % worker_count

            for i, worker in enumerate(self.workers.values()):
                worker_vus = vus_per_worker + (1 if i < remainder else 0)

                await self._send_to_worker(worker, {
                    'command': 'run_stage',
                    'duration': stage['duration'],
                    'target_vus': worker_vus,
                    'scenarios': config.scenarios,
                    'base_url': config.base_url
                })

            # Wait for stage
            await asyncio.sleep(stage['duration'])

        await self.stop_test()

    async def stop_test(self):
        self.running = False
        for worker in self.workers.values():
            await self._send_to_worker(worker, {'command': 'stop'})

    async def _send_to_worker(self, worker: Worker, message: dict):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"http://{worker.host}:{worker.port}/command",
                json=message
            )

    async def handle_heartbeat(self, worker_id: str, metrics: list[dict]):
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = time.time()
            self.workers[worker_id].metrics_count += len(metrics)

            # Aggregate metrics
            for m in metrics:
                self.aggregated_metrics.append(RequestMetrics(**m))

    async def check_worker_health(self):
        now = time.time()
        for worker in list(self.workers.values()):
            if now - worker.last_heartbeat > 30:
                worker.status = WorkerStatus.ERROR
                # Redistribute load
                await self._rebalance()

    async def _rebalance(self):
        healthy = [w for w in self.workers.values()
                   if w.status != WorkerStatus.ERROR]
        if not healthy or not self.running:
            return

        # Recalculate VU distribution
        # ... (similar to start_test)

class WorkerNode:
    def __init__(self, worker_id: str, coordinator_url: str, port: int = 5001):
        self.worker_id = worker_id
        self.coordinator_url = coordinator_url
        self.port = port
        self.load_runner: Optional[LoadRunner] = None
        self.pending_metrics: list[RequestMetrics] = []

    async def run(self):
        # Start web server for commands
        from aiohttp import web

        app = web.Application()
        app.router.add_post('/command', self.handle_command)

        # Start heartbeat task
        asyncio.create_task(self._heartbeat_loop())

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()

    async def handle_command(self, request):
        data = await request.json()
        command = data.get('command')

        if command == 'run_stage':
            await self._run_stage(data)
        elif command == 'stop':
            if self.load_runner:
                self.load_runner.running = False

        return web.Response(text='ok')

    async def _run_stage(self, config: dict):
        self.load_runner = LoadRunner(config['base_url'])

        # Add scenarios
        for s in config['scenarios']:
            steps = [ScenarioStep(**step) for step in s['steps']]
            self.load_runner.add_scenario(Scenario(
                name=s['name'],
                steps=steps
            ))

        # Collect metrics
        self.load_runner.metrics_callback = lambda m: self.pending_metrics.append(m)

        # Add stage
        self.load_runner.add_stage(
            duration=config['duration'],
            target_vus=config['target_vus']
        )

        await self.load_runner.run()

    async def _heartbeat_loop(self):
        import aiohttp

        while True:
            await asyncio.sleep(5)

            # Send pending metrics
            metrics_to_send = self.pending_metrics[:100]
            self.pending_metrics = self.pending_metrics[100:]

            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{self.coordinator_url}/heartbeat",
                        json={
                            'worker_id': self.worker_id,
                            'metrics': [m.__dict__ for m in metrics_to_send]
                        }
                    )
            except:
                pass  # Coordinator unavailable
```
"""
                },
                "pitfalls": [
                    "Network latency between workers adds to reported response time",
                    "Time synchronization: use relative times or sync clocks",
                    "Worker failure mid-test: decide whether to continue or abort",
                    "Metric aggregation can create memory pressure"
                ]
            },
            {
                "name": "Real-time Metrics & Reporting",
                "description": "Implement live metrics dashboard with percentiles and analysis",
                "skills": ["Percentile calculation", "Time series", "Streaming aggregation"],
                "hints": {
                    "level1": "Track p50, p95, p99 latency, not just average",
                    "level2": "Use t-digest or HDR histogram for accurate percentiles",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional
import math
import time
from collections import defaultdict

class HDRHistogram:
    '''High Dynamic Range Histogram for accurate percentiles'''

    def __init__(self, lowest: int = 1, highest: int = 3600000,
                 significant_figures: int = 3):
        self.lowest = lowest
        self.highest = highest
        self.significant_figures = significant_figures

        # Calculate bucket count
        self.bucket_count = self._calculate_bucket_count()
        self.counts = [0] * self.bucket_count
        self.total_count = 0

    def _calculate_bucket_count(self) -> int:
        # Simplified - real implementation is more complex
        return int(math.log2(self.highest / self.lowest) * 1000)

    def _get_bucket(self, value: float) -> int:
        if value < self.lowest:
            return 0
        if value > self.highest:
            return self.bucket_count - 1
        return int(math.log2(value / self.lowest) * 100)

    def record(self, value: float):
        bucket = self._get_bucket(value)
        self.counts[bucket] += 1
        self.total_count += 1

    def percentile(self, p: float) -> float:
        if self.total_count == 0:
            return 0

        target_count = int(self.total_count * p / 100)
        count_so_far = 0

        for bucket, count in enumerate(self.counts):
            count_so_far += count
            if count_so_far >= target_count:
                # Convert bucket back to value
                return self.lowest * (2 ** (bucket / 100))

        return self.highest

@dataclass
class MetricsSummary:
    count: int
    error_count: int
    error_rate: float
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float

class MetricsAggregator:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size  # Seconds
        self.histograms: dict[str, HDRHistogram] = {}  # Per endpoint
        self.global_histogram = HDRHistogram()

        self.window_metrics: list[RequestMetrics] = []
        self.window_start: float = time.time()

        # Counters
        self.total_requests = 0
        self.total_errors = 0
        self.endpoint_counts: dict[str, int] = defaultdict(int)

    def record(self, metric: RequestMetrics):
        self.total_requests += 1

        if metric.error or metric.status_code >= 400:
            self.total_errors += 1

        # Record to histograms
        self.global_histogram.record(metric.response_time_ms)

        endpoint = f"{metric.method} {metric.url}"
        if endpoint not in self.histograms:
            self.histograms[endpoint] = HDRHistogram()
        self.histograms[endpoint].record(metric.response_time_ms)
        self.endpoint_counts[endpoint] += 1

        # Window metrics
        self.window_metrics.append(metric)
        self._cleanup_window()

    def _cleanup_window(self):
        now = time.time()
        cutoff = now - self.window_size
        self.window_metrics = [
            m for m in self.window_metrics
            if m.timestamp > cutoff
        ]

    def get_summary(self) -> MetricsSummary:
        if self.total_requests == 0:
            return MetricsSummary(
                count=0, error_count=0, error_rate=0,
                min_ms=0, max_ms=0, mean_ms=0,
                p50_ms=0, p95_ms=0, p99_ms=0,
                requests_per_second=0
            )

        response_times = [m.response_time_ms for m in self.window_metrics]

        return MetricsSummary(
            count=self.total_requests,
            error_count=self.total_errors,
            error_rate=self.total_errors / self.total_requests,
            min_ms=min(response_times) if response_times else 0,
            max_ms=max(response_times) if response_times else 0,
            mean_ms=sum(response_times) / len(response_times) if response_times else 0,
            p50_ms=self.global_histogram.percentile(50),
            p95_ms=self.global_histogram.percentile(95),
            p99_ms=self.global_histogram.percentile(99),
            requests_per_second=len(self.window_metrics) / self.window_size
        )

    def get_endpoint_summary(self, endpoint: str) -> Optional[MetricsSummary]:
        if endpoint not in self.histograms:
            return None

        hist = self.histograms[endpoint]
        count = self.endpoint_counts[endpoint]

        return MetricsSummary(
            count=count,
            error_count=0,  # Would need separate tracking
            error_rate=0,
            min_ms=0,
            max_ms=0,
            mean_ms=0,
            p50_ms=hist.percentile(50),
            p95_ms=hist.percentile(95),
            p99_ms=hist.percentile(99),
            requests_per_second=0
        )

class ReportGenerator:
    def __init__(self, aggregator: MetricsAggregator):
        self.aggregator = aggregator

    def generate_text_report(self) -> str:
        summary = self.aggregator.get_summary()

        lines = [
            "Load Test Results",
            "=" * 50,
            f"Total Requests:    {summary.count:,}",
            f"Failed Requests:   {summary.error_count:,} ({summary.error_rate:.1%})",
            f"Requests/sec:      {summary.requests_per_second:.1f}",
            "",
            "Response Times (ms):",
            f"  Min:    {summary.min_ms:.1f}",
            f"  Mean:   {summary.mean_ms:.1f}",
            f"  P50:    {summary.p50_ms:.1f}",
            f"  P95:    {summary.p95_ms:.1f}",
            f"  P99:    {summary.p99_ms:.1f}",
            f"  Max:    {summary.max_ms:.1f}",
        ]

        return "\\n".join(lines)

    def generate_json_report(self) -> dict:
        summary = self.aggregator.get_summary()
        return {
            'summary': {
                'total_requests': summary.count,
                'error_count': summary.error_count,
                'error_rate': summary.error_rate,
                'requests_per_second': summary.requests_per_second
            },
            'latency': {
                'min': summary.min_ms,
                'mean': summary.mean_ms,
                'p50': summary.p50_ms,
                'p95': summary.p95_ms,
                'p99': summary.p99_ms,
                'max': summary.max_ms
            },
            'endpoints': {
                endpoint: {
                    'count': self.aggregator.endpoint_counts[endpoint],
                    'p50': self.aggregator.histograms[endpoint].percentile(50),
                    'p99': self.aggregator.histograms[endpoint].percentile(99)
                }
                for endpoint in self.aggregator.histograms
            }
        }
```
"""
                },
                "pitfalls": [
                    "Average hides outliers - always report percentiles",
                    "HDR histogram more accurate than naive quantile calculation",
                    "Streaming percentiles: use t-digest for memory efficiency",
                    "Report generator should handle empty metrics gracefully"
                ]
            }
        ]
    },

    "chaos-engineering": {
        "name": "Chaos Engineering Platform",
        "description": "Build a chaos engineering tool to test system resilience through controlled failure injection.",
        "why_expert": "Systems fail in production. Chaos engineering proactively finds weaknesses. Understanding failure modes helps build resilient systems.",
        "difficulty": "expert",
        "tags": ["chaos", "reliability", "testing", "resilience", "fault-injection"],
        "estimated_hours": 50,
        "prerequisites": ["build-http-server", "container-runtime"],
        "milestones": [
            {
                "name": "Fault Injection Framework",
                "description": "Implement fault injection primitives (latency, errors, resource exhaustion)",
                "skills": ["Fault injection", "Proxy interception", "Resource limits"],
                "hints": {
                    "level1": "Inject faults at network, process, or application level",
                    "level2": "Use proxy to inject latency/errors; cgroups for resource limits",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
import random
import time
import asyncio

class FaultType(Enum):
    LATENCY = "latency"
    ERROR = "error"
    PACKET_LOSS = "packet_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    PROCESS_KILL = "process_kill"

@dataclass
class FaultConfig:
    type: FaultType
    target: str           # Service, pod, container
    duration: int         # Seconds
    probability: float = 1.0  # 0-1, for partial faults
    parameters: dict = field(default_factory=dict)

class Fault(ABC):
    def __init__(self, config: FaultConfig):
        self.config = config
        self.active = False

    @abstractmethod
    async def inject(self):
        pass

    @abstractmethod
    async def rollback(self):
        pass

class LatencyFault(Fault):
    '''Inject network latency'''

    async def inject(self):
        latency_ms = self.config.parameters.get('latency_ms', 500)
        jitter_ms = self.config.parameters.get('jitter_ms', 100)

        # Use tc (traffic control) to add latency
        import subprocess
        subprocess.run([
            'tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'netem',
            'delay', f'{latency_ms}ms', f'{jitter_ms}ms'
        ], check=True)

        self.active = True

    async def rollback(self):
        import subprocess
        subprocess.run([
            'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'
        ], check=False)
        self.active = False

class ErrorFault(Fault):
    '''Inject HTTP errors via proxy'''

    def __init__(self, config: FaultConfig, proxy: 'FaultProxy'):
        super().__init__(config)
        self.proxy = proxy

    async def inject(self):
        status_code = self.config.parameters.get('status_code', 500)
        self.proxy.add_rule(FaultRule(
            target=self.config.target,
            probability=self.config.probability,
            action='error',
            status_code=status_code
        ))
        self.active = True

    async def rollback(self):
        self.proxy.remove_rule(self.config.target)
        self.active = False

class CPUStressFault(Fault):
    '''Stress CPU to simulate resource contention'''

    async def inject(self):
        cores = self.config.parameters.get('cores', 1)
        load_percent = self.config.parameters.get('load', 80)

        # Start stress workers
        self._workers = []
        for _ in range(cores):
            task = asyncio.create_task(self._stress_worker(load_percent))
            self._workers.append(task)

        self.active = True

    async def rollback(self):
        for worker in self._workers:
            worker.cancel()
        self._workers = []
        self.active = False

    async def _stress_worker(self, load_percent: int):
        while True:
            # Busy loop for load_percent of time
            busy_time = load_percent / 100
            start = time.time()
            while time.time() - start < busy_time:
                _ = sum(i*i for i in range(1000))

            # Sleep for remainder
            await asyncio.sleep(1 - busy_time)

class ProcessKillFault(Fault):
    '''Kill process to test recovery'''

    async def inject(self):
        process_name = self.config.parameters.get('process')
        signal = self.config.parameters.get('signal', 'SIGKILL')

        import subprocess
        subprocess.run(['pkill', f'-{signal}', process_name])
        self.active = True

    async def rollback(self):
        # Process kill is permanent - can't rollback
        # Could restart process if we have that capability
        self.active = False

@dataclass
class FaultRule:
    target: str
    probability: float
    action: str
    status_code: int = 500
    latency_ms: int = 0

class FaultProxy:
    '''HTTP proxy that injects faults'''

    def __init__(self, port: int = 8080):
        self.port = port
        self.rules: dict[str, FaultRule] = {}

    def add_rule(self, rule: FaultRule):
        self.rules[rule.target] = rule

    def remove_rule(self, target: str):
        if target in self.rules:
            del self.rules[target]

    async def handle_request(self, request) -> 'Response':
        # Check if any rule matches
        for target, rule in self.rules.items():
            if target in request.url:
                if random.random() < rule.probability:
                    return self._apply_fault(request, rule)

        # Forward to upstream
        return await self._forward(request)

    def _apply_fault(self, request, rule: FaultRule) -> 'Response':
        if rule.action == 'error':
            return Response(
                status_code=rule.status_code,
                body=b'Injected fault'
            )
        elif rule.action == 'latency':
            time.sleep(rule.latency_ms / 1000)
            return self._forward(request)

    async def _forward(self, request) -> 'Response':
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=request.url.replace(f':{self.port}', ''),
                headers=request.headers,
                data=request.body
            ) as resp:
                return Response(
                    status_code=resp.status,
                    body=await resp.read()
                )
```
"""
                },
                "pitfalls": [
                    "tc commands require root/CAP_NET_ADMIN",
                    "CPU stress can affect chaos tool itself - isolate",
                    "Process kill needs restart mechanism or test fails",
                    "Probability <1.0 creates intermittent failures (realistic)"
                ]
            },
            {
                "name": "Experiment Orchestration",
                "description": "Implement experiment definition, scheduling, and safety controls",
                "skills": ["Experiment design", "Safety controls", "Rollback"],
                "hints": {
                    "level1": "Experiment: hypothesis, faults to inject, metrics to observe",
                    "level2": "Safety: abort conditions, blast radius limits, automatic rollback",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum
import asyncio
import time

class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"

@dataclass
class SteadyStateHypothesis:
    name: str
    probe: Callable[[], bool]  # Returns True if steady state
    tolerance: float = 0.95   # Acceptable success rate

@dataclass
class AbortCondition:
    name: str
    check: Callable[[], bool]  # Returns True if should abort
    message: str

@dataclass
class Experiment:
    id: str
    name: str
    description: str
    hypothesis: SteadyStateHypothesis
    faults: list[FaultConfig]
    abort_conditions: list[AbortCondition]
    duration: int  # Seconds
    blast_radius: float = 0.1  # Max % of instances affected

@dataclass
class ExperimentResult:
    experiment_id: str
    status: ExperimentStatus
    started_at: float
    ended_at: float
    steady_state_before: bool
    steady_state_after: bool
    abort_reason: Optional[str] = None
    observations: list[dict] = field(default_factory=list)

class ExperimentRunner:
    def __init__(self):
        self.active_faults: list[Fault] = []
        self.running = False

    async def run_experiment(self, experiment: Experiment) -> ExperimentResult:
        result = ExperimentResult(
            experiment_id=experiment.id,
            status=ExperimentStatus.RUNNING,
            started_at=time.time(),
            ended_at=0,
            steady_state_before=False,
            steady_state_after=False
        )

        try:
            # 1. Verify steady state before
            result.steady_state_before = await self._check_steady_state(
                experiment.hypothesis
            )
            if not result.steady_state_before:
                result.status = ExperimentStatus.FAILED
                result.abort_reason = "System not in steady state before experiment"
                return result

            # 2. Inject faults
            self.running = True
            for fault_config in experiment.faults:
                fault = self._create_fault(fault_config)
                await fault.inject()
                self.active_faults.append(fault)

            # 3. Run for duration while monitoring abort conditions
            abort_reason = await self._monitor_experiment(
                experiment.duration,
                experiment.abort_conditions
            )

            if abort_reason:
                result.status = ExperimentStatus.ABORTED
                result.abort_reason = abort_reason
            else:
                # 4. Verify steady state after
                result.steady_state_after = await self._check_steady_state(
                    experiment.hypothesis
                )
                result.status = ExperimentStatus.COMPLETED

        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.abort_reason = str(e)

        finally:
            # 5. Rollback all faults
            await self._rollback_all()
            result.ended_at = time.time()
            self.running = False

        return result

    async def _check_steady_state(self, hypothesis: SteadyStateHypothesis) -> bool:
        # Run probe multiple times to check stability
        successes = 0
        attempts = 10

        for _ in range(attempts):
            try:
                if hypothesis.probe():
                    successes += 1
            except:
                pass
            await asyncio.sleep(0.5)

        return (successes / attempts) >= hypothesis.tolerance

    async def _monitor_experiment(self, duration: int,
                                   abort_conditions: list[AbortCondition]) -> Optional[str]:
        start = time.time()

        while time.time() - start < duration:
            # Check abort conditions
            for condition in abort_conditions:
                try:
                    if condition.check():
                        return condition.message
                except:
                    pass

            await asyncio.sleep(1)

        return None

    async def _rollback_all(self):
        for fault in self.active_faults:
            try:
                await fault.rollback()
            except Exception as e:
                print(f"Rollback failed: {e}")

        self.active_faults = []

    def _create_fault(self, config: FaultConfig) -> Fault:
        fault_classes = {
            FaultType.LATENCY: LatencyFault,
            FaultType.CPU_STRESS: CPUStressFault,
            FaultType.PROCESS_KILL: ProcessKillFault,
        }
        return fault_classes[config.type](config)

class SafetyController:
    '''Global safety controls for chaos experiments'''

    def __init__(self):
        self.global_abort = False
        self.active_experiments: list[str] = []
        self.max_concurrent = 1
        self.allowed_hours: tuple[int, int] = (9, 17)  # 9 AM - 5 PM

    def can_start_experiment(self, experiment: Experiment) -> tuple[bool, str]:
        if self.global_abort:
            return False, "Global abort is active"

        if len(self.active_experiments) >= self.max_concurrent:
            return False, "Too many concurrent experiments"

        # Check time window
        hour = time.localtime().tm_hour
        if not (self.allowed_hours[0] <= hour < self.allowed_hours[1]):
            return False, f"Outside allowed hours {self.allowed_hours}"

        return True, ""

    def emergency_stop(self):
        '''Stop all experiments immediately'''
        self.global_abort = True
        # In production: send signal to all experiment runners
```
"""
                },
                "pitfalls": [
                    "Always verify steady state BEFORE injecting faults",
                    "Automatic rollback is critical - manual cleanup is error-prone",
                    "Abort conditions should check error rates, not just availability",
                    "Run during business hours initially - easier to respond to issues"
                ]
            },
            {
                "name": "GameDay Automation",
                "description": "Implement scheduled chaos experiments with runbooks and incident response",
                "skills": ["GameDay planning", "Runbooks", "Incident response"],
                "hints": {
                    "level1": "GameDay: scheduled chaos with observers and runbooks",
                    "level2": "Automated runbooks: if X happens, do Y to recover",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum
import asyncio
import time
from datetime import datetime, timedelta

@dataclass
class RunbookStep:
    name: str
    action: Callable
    rollback: Optional[Callable] = None
    timeout: int = 60

@dataclass
class Runbook:
    id: str
    name: str
    trigger_condition: Callable[[], bool]
    steps: list[RunbookStep]
    auto_execute: bool = False

class RunbookExecutor:
    async def execute(self, runbook: Runbook) -> dict:
        results = {'steps': [], 'success': True}
        executed_steps = []

        for step in runbook.steps:
            try:
                await asyncio.wait_for(
                    asyncio.create_task(step.action()),
                    timeout=step.timeout
                )
                results['steps'].append({
                    'name': step.name,
                    'status': 'success'
                })
                executed_steps.append(step)

            except Exception as e:
                results['steps'].append({
                    'name': step.name,
                    'status': 'failed',
                    'error': str(e)
                })
                results['success'] = False

                # Rollback executed steps
                await self._rollback(executed_steps)
                break

        return results

    async def _rollback(self, steps: list[RunbookStep]):
        for step in reversed(steps):
            if step.rollback:
                try:
                    await step.rollback()
                except:
                    pass

@dataclass
class GameDay:
    id: str
    name: str
    description: str
    scheduled_start: datetime
    experiments: list[Experiment]
    runbooks: list[Runbook]
    observers: list[str]  # Email/Slack of observers
    duration_hours: int = 2

class GameDayCoordinator:
    def __init__(self, experiment_runner: ExperimentRunner,
                 runbook_executor: RunbookExecutor,
                 notifier: 'Notifier'):
        self.runner = experiment_runner
        self.runbook_exec = runbook_executor
        self.notifier = notifier
        self.scheduled_gamedays: list[GameDay] = []

    async def schedule_gameday(self, gameday: GameDay):
        self.scheduled_gamedays.append(gameday)

        # Notify observers
        await self.notifier.notify(
            gameday.observers,
            f"GameDay '{gameday.name}' scheduled for {gameday.scheduled_start}"
        )

    async def run_gameday(self, gameday: GameDay):
        await self.notifier.notify(
            gameday.observers,
            f"GameDay '{gameday.name}' starting now!"
        )

        results = {
            'gameday_id': gameday.id,
            'experiments': [],
            'runbooks_triggered': []
        }

        for experiment in gameday.experiments:
            # Run experiment
            exp_result = await self.runner.run_experiment(experiment)
            results['experiments'].append({
                'id': experiment.id,
                'status': exp_result.status.value,
                'steady_state_maintained': exp_result.steady_state_after
            })

            # Check if any runbooks should trigger
            for runbook in gameday.runbooks:
                if runbook.trigger_condition():
                    if runbook.auto_execute:
                        rb_result = await self.runbook_exec.execute(runbook)
                        results['runbooks_triggered'].append({
                            'runbook': runbook.id,
                            'result': rb_result
                        })
                    else:
                        await self.notifier.notify(
                            gameday.observers,
                            f"Runbook '{runbook.name}' should be executed manually"
                        )

            # Brief pause between experiments
            await asyncio.sleep(60)

        await self.notifier.notify(
            gameday.observers,
            f"GameDay '{gameday.name}' completed. Results: {results}"
        )

        return results

@dataclass
class Observation:
    timestamp: float
    observer: str
    type: str  # 'issue', 'note', 'recovery'
    description: str
    metrics: dict = field(default_factory=dict)

class GameDayRecorder:
    '''Record observations during GameDay'''

    def __init__(self, gameday_id: str):
        self.gameday_id = gameday_id
        self.observations: list[Observation] = []
        self.timeline: list[dict] = []

    def record_observation(self, observer: str, obs_type: str,
                           description: str, metrics: dict = None):
        obs = Observation(
            timestamp=time.time(),
            observer=observer,
            type=obs_type,
            description=description,
            metrics=metrics or {}
        )
        self.observations.append(obs)

    def record_event(self, event_type: str, details: dict):
        self.timeline.append({
            'timestamp': time.time(),
            'event': event_type,
            'details': details
        })

    def generate_report(self) -> dict:
        return {
            'gameday_id': self.gameday_id,
            'timeline': self.timeline,
            'observations': [
                {
                    'time': obs.timestamp,
                    'observer': obs.observer,
                    'type': obs.type,
                    'description': obs.description
                }
                for obs in self.observations
            ],
            'issues_found': [
                obs for obs in self.observations
                if obs.type == 'issue'
            ],
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> list[str]:
        recommendations = []

        # Analyze observations for patterns
        issues = [o for o in self.observations if o.type == 'issue']
        for issue in issues:
            if 'timeout' in issue.description.lower():
                recommendations.append("Consider implementing circuit breakers")
            if 'memory' in issue.description.lower():
                recommendations.append("Review memory limits and implement backpressure")

        return recommendations
```
"""
                },
                "pitfalls": [
                    "GameDays need preparation - brief observers on experiments",
                    "Manual runbook steps may be needed for complex recovery",
                    "Record everything - observations are valuable learning",
                    "Schedule GameDays when team is available to respond"
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

for project_id, project in testing_reliability_projects.items():
    data['expert_projects'][project_id] = project
    print(f"Added: {project_id} - {project['name']}")

with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nAdded {len(testing_reliability_projects)} Testing & Reliability projects")

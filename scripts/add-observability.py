#!/usr/bin/env python3
"""
Add Observability projects to the curriculum.
Focus on metrics, logging, tracing, and alerting.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

observability_projects = {
    "metrics-collector": {
        "name": "Metrics Collection System",
        "description": "Build a Prometheus-like metrics collection system with scraping, storage, and PromQL-style querying.",
        "why_important": "Understanding metrics systems helps you design better instrumentation, optimize queries, and debug performance issues in production.",
        "difficulty": "advanced",
        "tags": ["observability", "distributed-systems", "databases"],
        "estimated_hours": 50,
        "prerequisites": ["time-series-db"],
        "learning_outcomes": [
            "Design pull-based metrics collection",
            "Implement time-series storage efficiently",
            "Build query language for aggregations",
            "Handle high-cardinality metrics"
        ],
        "milestones": [
            {
                "name": "Metrics Data Model",
                "description": "Implement the metrics data model with counters, gauges, histograms, and summaries with labels.",
                "hints": {
                    "level1": "Each metric has a name, type, labels (key-value pairs), and value.",
                    "level2": "Histograms store bucketed counts, summaries track quantiles over time.",
                    "level3": """```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import threading
import math

class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass(frozen=True)
class Labels:
    \"\"\"Immutable label set for metric identification.\"\"\"
    _labels: tuple[tuple[str, str], ...]

    def __init__(self, **kwargs):
        object.__setattr__(self, '_labels', tuple(sorted(kwargs.items())))

    def __hash__(self):
        return hash(self._labels)

    def __eq__(self, other):
        return self._labels == other._labels

    def to_dict(self) -> dict:
        return dict(self._labels)

@dataclass
class Sample:
    timestamp: float
    value: float

class Counter:
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help = help_text
        self.type = MetricType.COUNTER
        self._values: dict[Labels, float] = {}
        self._lock = threading.Lock()

    def inc(self, labels: Labels = None, value: float = 1):
        labels = labels or Labels()
        with self._lock:
            self._values[labels] = self._values.get(labels, 0) + value

    def get(self, labels: Labels = None) -> float:
        labels = labels or Labels()
        return self._values.get(labels, 0)

    def collect(self) -> list[tuple[Labels, float]]:
        with self._lock:
            return [(l, v) for l, v in self._values.items()]

class Gauge:
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help = help_text
        self.type = MetricType.GAUGE
        self._values: dict[Labels, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, labels: Labels = None):
        labels = labels or Labels()
        with self._lock:
            self._values[labels] = value

    def inc(self, labels: Labels = None, value: float = 1):
        labels = labels or Labels()
        with self._lock:
            self._values[labels] = self._values.get(labels, 0) + value

    def dec(self, labels: Labels = None, value: float = 1):
        self.inc(labels, -value)

    def collect(self) -> list[tuple[Labels, float]]:
        with self._lock:
            return [(l, v) for l, v in self._values.items()]

class Histogram:
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf'))

    def __init__(self, name: str, help_text: str = "", buckets: tuple = None):
        self.name = name
        self.help = help_text
        self.type = MetricType.HISTOGRAM
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: dict[Labels, list[int]] = {}  # bucket counts
        self._sums: dict[Labels, float] = {}
        self._totals: dict[Labels, int] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Labels = None):
        labels = labels or Labels()
        with self._lock:
            if labels not in self._counts:
                self._counts[labels] = [0] * len(self.buckets)
                self._sums[labels] = 0
                self._totals[labels] = 0

            # Increment appropriate bucket(s)
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._counts[labels][i] += 1

            self._sums[labels] += value
            self._totals[labels] += 1

    def collect(self) -> list[tuple[Labels, dict]]:
        with self._lock:
            results = []
            for labels in self._counts:
                results.append((labels, {
                    'buckets': list(zip(self.buckets, self._counts[labels])),
                    'sum': self._sums[labels],
                    'count': self._totals[labels]
                }))
            return results

class MetricRegistry:
    def __init__(self):
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help_text: str = "") -> Counter:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, help_text)
            return self._metrics[name]

    def gauge(self, name: str, help_text: str = "") -> Gauge:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, help_text)
            return self._metrics[name]

    def histogram(self, name: str, help_text: str = "", buckets: tuple = None) -> Histogram:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, help_text, buckets)
            return self._metrics[name]

    def collect_all(self) -> dict:
        with self._lock:
            return {name: metric.collect() for name, metric in self._metrics.items()}
```"""
                },
                "pitfalls": [
                    "High cardinality labels cause memory explosion",
                    "Counter resets on restart need special handling",
                    "Histogram bucket boundaries can't change after creation"
                ]
            },
            {
                "name": "Scrape Engine",
                "description": "Build a scrape engine that pulls metrics from configured targets with service discovery support.",
                "hints": {
                    "level1": "Poll targets at configured intervals, parse response format.",
                    "level2": "Handle target failures, track scrape duration and status.",
                    "level3": """```python
import asyncio
import aiohttp
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class ScrapeHealth(str, Enum):
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"

@dataclass
class ScrapeTarget:
    job_name: str
    address: str
    metrics_path: str = "/metrics"
    scrape_interval: float = 15.0
    scrape_timeout: float = 10.0
    labels: dict = field(default_factory=dict)

@dataclass
class ScrapeResult:
    target: ScrapeTarget
    timestamp: float
    samples: list[tuple[str, Labels, float]]
    health: ScrapeHealth
    scrape_duration: float
    error: Optional[str] = None

class MetricParser:
    def parse(self, text: str) -> list[tuple[str, Labels, float]]:
        \"\"\"Parse Prometheus exposition format.\"\"\"
        samples = []
        for line in text.strip().split('\\n'):
            if not line or line.startswith('#'):
                continue

            # Parse metric line: name{label="value"} value timestamp?
            if '{' in line:
                name_part, rest = line.split('{', 1)
                labels_part, value_part = rest.rsplit('}', 1)
                labels = self._parse_labels(labels_part)
            else:
                parts = line.split()
                name_part = parts[0]
                value_part = ' '.join(parts[1:])
                labels = Labels()

            value = float(value_part.split()[0])
            samples.append((name_part.strip(), labels, value))

        return samples

    def _parse_labels(self, labels_str: str) -> Labels:
        import re
        label_dict = {}
        pattern = r'(\\w+)="([^"]*)"'
        for match in re.finditer(pattern, labels_str):
            label_dict[match.group(1)] = match.group(2)
        return Labels(**label_dict)

class ScrapeEngine:
    def __init__(self, storage):
        self.storage = storage
        self.targets: list[ScrapeTarget] = []
        self.parser = MetricParser()
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False

        # Internal metrics
        self.scrape_duration = Histogram("scrape_duration_seconds", "Scrape duration")
        self.scrape_samples = Gauge("scrape_samples_scraped", "Samples scraped")
        self.up = Gauge("up", "Target health")

    def add_target(self, target: ScrapeTarget):
        self.targets.append(target)
        if self._running:
            self._start_scrape_loop(target)

    async def start(self):
        self._running = True
        for target in self.targets:
            self._start_scrape_loop(target)

    def _start_scrape_loop(self, target: ScrapeTarget):
        key = f"{target.job_name}:{target.address}"
        self._tasks[key] = asyncio.create_task(self._scrape_loop(target))

    async def stop(self):
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def _scrape_loop(self, target: ScrapeTarget):
        while self._running:
            result = await self._scrape(target)
            await self._process_result(result)
            await asyncio.sleep(target.scrape_interval)

    async def _scrape(self, target: ScrapeTarget) -> ScrapeResult:
        url = f"http://{target.address}{target.metrics_path}"
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=target.scrape_timeout)) as resp:
                    if resp.status != 200:
                        raise Exception(f"HTTP {resp.status}")

                    text = await resp.text()
                    samples = self.parser.parse(text)
                    duration = time.time() - start_time

                    return ScrapeResult(
                        target=target,
                        timestamp=start_time,
                        samples=samples,
                        health=ScrapeHealth.UP,
                        scrape_duration=duration
                    )

        except Exception as e:
            return ScrapeResult(
                target=target,
                timestamp=start_time,
                samples=[],
                health=ScrapeHealth.DOWN,
                scrape_duration=time.time() - start_time,
                error=str(e)
            )

    async def _process_result(self, result: ScrapeResult):
        target = result.target
        target_labels = Labels(job=target.job_name, instance=target.address, **target.labels)

        # Record internal metrics
        self.scrape_duration.observe(result.scrape_duration, target_labels)
        self.scrape_samples.set(len(result.samples), target_labels)
        self.up.set(1 if result.health == ScrapeHealth.UP else 0, target_labels)

        # Store samples
        for name, labels, value in result.samples:
            # Merge target labels with metric labels
            all_labels = Labels(**{**target_labels.to_dict(), **labels.to_dict()})
            await self.storage.store(name, all_labels, result.timestamp, value)
```"""
                },
                "pitfalls": [
                    "Too aggressive scraping overwhelms targets",
                    "Network timeouts block scrape loop",
                    "Label collision between target and metric labels"
                ]
            },
            {
                "name": "Time Series Storage",
                "description": "Implement efficient time-series storage with compression, retention, and downsampling.",
                "hints": {
                    "level1": "Store samples in time-ordered chunks per series.",
                    "level2": "Use delta encoding and variable-length integers for compression.",
                    "level3": """```python
import struct
from dataclasses import dataclass, field
from typing import Iterator, Optional
import mmap
import os
import threading
from bisect import bisect_left, bisect_right

@dataclass
class Chunk:
    \"\"\"Compressed chunk of time-series samples.\"\"\"
    start_time: float
    end_time: float = 0
    samples: int = 0
    data: bytearray = field(default_factory=bytearray)
    _prev_ts: float = 0
    _prev_val: float = 0

    MAX_SAMPLES = 120  # ~2 hours at 1 min interval

    def append(self, timestamp: float, value: float):
        if self.samples == 0:
            # First sample - store full values
            self.data.extend(struct.pack('<d', timestamp))
            self.data.extend(struct.pack('<d', value))
            self.start_time = timestamp
            self._prev_ts = timestamp
            self._prev_val = value
        else:
            # Delta encode timestamp
            ts_delta = int((timestamp - self._prev_ts) * 1000)  # ms precision
            self._encode_varint(ts_delta)

            # XOR encode value
            self._encode_xor_float(value)

            self._prev_ts = timestamp
            self._prev_val = value

        self.end_time = timestamp
        self.samples += 1

    def is_full(self) -> bool:
        return self.samples >= self.MAX_SAMPLES

    def _encode_varint(self, value: int):
        while value > 127:
            self.data.append((value & 0x7F) | 0x80)
            value >>= 7
        self.data.append(value)

    def _encode_xor_float(self, value: float):
        # Simple XOR encoding (real implementation uses Gorilla compression)
        prev_bits = struct.unpack('<Q', struct.pack('<d', self._prev_val))[0]
        curr_bits = struct.unpack('<Q', struct.pack('<d', value))[0]
        xor = prev_bits ^ curr_bits

        if xor == 0:
            self.data.append(0)  # Same value
        else:
            self.data.append(1)
            self.data.extend(struct.pack('<Q', xor))

    def iterate(self) -> Iterator[tuple[float, float]]:
        \"\"\"Iterate over samples in chunk.\"\"\"
        if not self.data:
            return

        offset = 0
        # First sample
        ts = struct.unpack_from('<d', self.data, offset)[0]
        offset += 8
        val = struct.unpack_from('<d', self.data, offset)[0]
        offset += 8
        yield ts, val

        prev_ts = ts
        prev_val = val

        for _ in range(self.samples - 1):
            # Decode timestamp delta
            ts_delta, bytes_read = self._decode_varint(offset)
            offset += bytes_read
            ts = prev_ts + ts_delta / 1000.0

            # Decode value
            marker = self.data[offset]
            offset += 1
            if marker == 0:
                val = prev_val
            else:
                xor = struct.unpack_from('<Q', self.data, offset)[0]
                offset += 8
                prev_bits = struct.unpack('<Q', struct.pack('<d', prev_val))[0]
                curr_bits = prev_bits ^ xor
                val = struct.unpack('<d', struct.pack('<Q', curr_bits))[0]

            yield ts, val
            prev_ts = ts
            prev_val = val

    def _decode_varint(self, offset: int) -> tuple[int, int]:
        result = 0
        shift = 0
        bytes_read = 0
        while True:
            byte = self.data[offset + bytes_read]
            bytes_read += 1
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        return result, bytes_read

@dataclass(frozen=True)
class SeriesKey:
    name: str
    labels: Labels

    def __hash__(self):
        return hash((self.name, self.labels))

class TimeSeriesStorage:
    def __init__(self, data_dir: str, retention_days: int = 15):
        self.data_dir = data_dir
        self.retention_days = retention_days
        self._series: dict[SeriesKey, list[Chunk]] = {}
        self._lock = threading.RLock()
        os.makedirs(data_dir, exist_ok=True)

    async def store(self, name: str, labels: Labels, timestamp: float, value: float):
        key = SeriesKey(name, labels)

        with self._lock:
            if key not in self._series:
                self._series[key] = [Chunk(start_time=timestamp)]

            chunks = self._series[key]
            current_chunk = chunks[-1]

            if current_chunk.is_full():
                # Start new chunk
                chunks.append(Chunk(start_time=timestamp))
                current_chunk = chunks[-1]

            current_chunk.append(timestamp, value)

    def query(self, name: str, labels: Labels,
              start_time: float, end_time: float) -> Iterator[tuple[float, float]]:
        \"\"\"Query samples in time range.\"\"\"
        key = SeriesKey(name, labels)

        with self._lock:
            chunks = self._series.get(key, [])

            for chunk in chunks:
                # Skip chunks outside time range
                if chunk.end_time < start_time or chunk.start_time > end_time:
                    continue

                for ts, val in chunk.iterate():
                    if start_time <= ts <= end_time:
                        yield ts, val

    def query_by_name(self, name: str, start_time: float,
                      end_time: float) -> dict[Labels, list[tuple[float, float]]]:
        \"\"\"Query all series with given name.\"\"\"
        results = {}

        with self._lock:
            for key, chunks in self._series.items():
                if key.name != name:
                    continue

                samples = list(self.query(name, key.labels, start_time, end_time))
                if samples:
                    results[key.labels] = samples

        return results

    def compact(self):
        \"\"\"Compact old chunks and enforce retention.\"\"\"
        cutoff = time.time() - (self.retention_days * 86400)

        with self._lock:
            for key in list(self._series.keys()):
                chunks = self._series[key]
                # Remove chunks entirely before cutoff
                self._series[key] = [c for c in chunks if c.end_time >= cutoff]

                # Remove empty series
                if not self._series[key]:
                    del self._series[key]
```"""
                },
                "pitfalls": [
                    "Chunk boundaries at exact times cause off-by-one",
                    "Concurrent writes corrupt chunk data",
                    "Compression ratio degrades with irregular timestamps"
                ]
            },
            {
                "name": "Query Engine",
                "description": "Build a PromQL-like query engine with instant queries, range queries, and aggregation functions.",
                "hints": {
                    "level1": "Parse expressions into AST, evaluate against storage.",
                    "level2": "Implement rate(), sum(), avg() and label matchers.",
                    "level3": """```python
from dataclasses import dataclass
from typing import Union, Callable
from enum import Enum
import re

class MatchType(Enum):
    EQUAL = "="
    NOT_EQUAL = "!="
    REGEX = "=~"
    NOT_REGEX = "!~"

@dataclass
class LabelMatcher:
    name: str
    value: str
    type: MatchType

    def matches(self, labels: Labels) -> bool:
        actual = labels.to_dict().get(self.name, "")
        if self.type == MatchType.EQUAL:
            return actual == self.value
        elif self.type == MatchType.NOT_EQUAL:
            return actual != self.value
        elif self.type == MatchType.REGEX:
            return bool(re.match(self.value, actual))
        elif self.type == MatchType.NOT_REGEX:
            return not bool(re.match(self.value, actual))

@dataclass
class VectorSelector:
    name: str
    matchers: list[LabelMatcher]
    range_seconds: float = 0  # 0 for instant, >0 for range

@dataclass
class AggregateExpr:
    op: str  # sum, avg, max, min, count
    expr: 'Expr'
    by: list[str] = None  # Group by labels
    without: list[str] = None  # Group by all except

@dataclass
class FunctionExpr:
    name: str  # rate, irate, increase, etc.
    args: list['Expr']

Expr = Union[VectorSelector, AggregateExpr, FunctionExpr]

class QueryEngine:
    def __init__(self, storage: TimeSeriesStorage):
        self.storage = storage
        self.functions = {
            'rate': self._rate,
            'irate': self._irate,
            'increase': self._increase,
            'sum_over_time': self._sum_over_time,
            'avg_over_time': self._avg_over_time,
        }
        self.aggregations = {
            'sum': lambda vals: sum(vals),
            'avg': lambda vals: sum(vals) / len(vals) if vals else 0,
            'max': lambda vals: max(vals) if vals else 0,
            'min': lambda vals: min(vals) if vals else 0,
            'count': lambda vals: len(vals),
        }

    def instant_query(self, query: str, timestamp: float) -> dict[Labels, float]:
        \"\"\"Execute instant query at specific time.\"\"\"
        expr = self._parse(query)
        return self._eval(expr, timestamp, timestamp)

    def range_query(self, query: str, start: float, end: float,
                   step: float) -> dict[Labels, list[tuple[float, float]]]:
        \"\"\"Execute range query with step.\"\"\"
        expr = self._parse(query)
        results = {}

        for ts in self._time_range(start, end, step):
            instant = self._eval(expr, ts - step, ts)
            for labels, value in instant.items():
                results.setdefault(labels, []).append((ts, value))

        return results

    def _parse(self, query: str) -> Expr:
        \"\"\"Parse query string into expression tree.\"\"\"
        # Simplified parser - real implementation uses proper grammar
        query = query.strip()

        # Check for aggregation: sum(metric{...})
        agg_match = re.match(r'(sum|avg|max|min|count)\\s*\\((.+)\\)\\s*(by|without)?\\s*\\(([^)]+)\\)?$', query)
        if agg_match:
            op = agg_match.group(1)
            inner = agg_match.group(2)
            modifier = agg_match.group(3)
            group_labels = [l.strip() for l in agg_match.group(4).split(',')] if agg_match.group(4) else []

            return AggregateExpr(
                op=op,
                expr=self._parse(inner),
                by=group_labels if modifier == 'by' else None,
                without=group_labels if modifier == 'without' else None
            )

        # Check for function: rate(metric{...}[5m])
        func_match = re.match(r'(\\w+)\\((.+)\\)$', query)
        if func_match and func_match.group(1) in self.functions:
            return FunctionExpr(
                name=func_match.group(1),
                args=[self._parse(func_match.group(2))]
            )

        # Vector selector: metric{label="value"}[5m]
        return self._parse_vector_selector(query)

    def _parse_vector_selector(self, query: str) -> VectorSelector:
        # Parse: metric_name{label="value",label2=~"regex"}[5m]
        range_match = re.search(r'\\[(\\d+)([smhd])\\]$', query)
        range_seconds = 0
        if range_match:
            query = query[:range_match.start()]
            value = int(range_match.group(1))
            unit = range_match.group(2)
            range_seconds = value * {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}[unit]

        if '{' in query:
            name, labels_str = query.split('{', 1)
            labels_str = labels_str.rstrip('}')
            matchers = self._parse_matchers(labels_str)
        else:
            name = query
            matchers = []

        return VectorSelector(name=name.strip(), matchers=matchers, range_seconds=range_seconds)

    def _parse_matchers(self, labels_str: str) -> list[LabelMatcher]:
        matchers = []
        pattern = r'(\\w+)\\s*(=~|!=|!~|=)\\s*"([^"]*)"'
        for match in re.finditer(pattern, labels_str):
            matchers.append(LabelMatcher(
                name=match.group(1),
                value=match.group(3),
                type=MatchType(match.group(2))
            ))
        return matchers

    def _eval(self, expr: Expr, start: float, end: float) -> dict[Labels, float]:
        if isinstance(expr, VectorSelector):
            return self._eval_selector(expr, start, end)
        elif isinstance(expr, AggregateExpr):
            return self._eval_aggregate(expr, start, end)
        elif isinstance(expr, FunctionExpr):
            return self._eval_function(expr, start, end)

    def _eval_selector(self, sel: VectorSelector, start: float, end: float) -> dict[Labels, float]:
        # Get all series matching name
        query_start = start - sel.range_seconds if sel.range_seconds else start
        all_series = self.storage.query_by_name(sel.name, query_start, end)

        results = {}
        for labels, samples in all_series.items():
            # Apply label matchers
            if all(m.matches(labels) for m in sel.matchers):
                if sel.range_seconds:
                    # Return samples in range (for rate, etc.)
                    results[labels] = samples
                else:
                    # Return latest sample
                    if samples:
                        results[labels] = samples[-1][1]

        return results

    def _rate(self, samples: list[tuple[float, float]]) -> float:
        \"\"\"Calculate per-second rate of increase.\"\"\"
        if len(samples) < 2:
            return 0

        first_ts, first_val = samples[0]
        last_ts, last_val = samples[-1]

        duration = last_ts - first_ts
        if duration <= 0:
            return 0

        # Handle counter resets
        increase = last_val - first_val
        if increase < 0:
            increase = last_val  # Counter reset

        return increase / duration
```"""
                },
                "pitfalls": [
                    "Rate calculation wrong across counter resets",
                    "Label matching with regex can be slow",
                    "Query on high-cardinality labels causes OOM"
                ]
            }
        ]
    },

    "log-aggregator": {
        "name": "Log Aggregation System",
        "description": "Build a log aggregation system like Loki with log ingestion, indexing, and LogQL-style querying.",
        "why_important": "Log systems are critical for debugging. Understanding log indexing helps optimize queries and reduce storage costs.",
        "difficulty": "intermediate",
        "tags": ["observability", "databases", "search"],
        "estimated_hours": 40,
        "prerequisites": ["shell"],
        "learning_outcomes": [
            "Design efficient log ingestion pipeline",
            "Build inverted index for label-based queries",
            "Implement log compression and chunking",
            "Handle high-volume log streams"
        ],
        "milestones": [
            {
                "name": "Log Ingestion",
                "description": "Build log ingestion endpoint that accepts structured logs with labels and timestamps.",
                "hints": {
                    "level1": "Accept logs as JSON with labels, timestamp, and line.",
                    "level2": "Batch logs for efficient writes, validate and enrich data.",
                    "level3": """```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import asyncio
import json
import time
import hashlib

@dataclass
class LogEntry:
    timestamp: datetime
    labels: dict[str, str]
    line: str
    stream_id: str = ""

    def __post_init__(self):
        if not self.stream_id:
            # Generate stream ID from labels
            label_str = json.dumps(self.labels, sort_keys=True)
            self.stream_id = hashlib.md5(label_str.encode()).hexdigest()[:16]

@dataclass
class LogBatch:
    stream_id: str
    labels: dict[str, str]
    entries: list[LogEntry] = field(default_factory=list)

class LogIngester:
    def __init__(self, storage, batch_size: int = 1000, flush_interval: float = 1.0):
        self.storage = storage
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._batches: dict[str, LogBatch] = {}
        self._lock = asyncio.Lock()
        self._flush_task = None

    async def start(self):
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        if self._flush_task:
            self._flush_task.cancel()
        await self._flush_all()

    async def ingest(self, entries: list[LogEntry]):
        \"\"\"Ingest log entries, batching by stream.\"\"\"
        async with self._lock:
            for entry in entries:
                stream_id = entry.stream_id

                if stream_id not in self._batches:
                    self._batches[stream_id] = LogBatch(
                        stream_id=stream_id,
                        labels=entry.labels
                    )

                batch = self._batches[stream_id]
                batch.entries.append(entry)

                if len(batch.entries) >= self.batch_size:
                    await self._flush_batch(stream_id)

    async def _flush_loop(self):
        while True:
            await asyncio.sleep(self.flush_interval)
            await self._flush_all()

    async def _flush_all(self):
        async with self._lock:
            for stream_id in list(self._batches.keys()):
                await self._flush_batch(stream_id)

    async def _flush_batch(self, stream_id: str):
        batch = self._batches.pop(stream_id, None)
        if batch and batch.entries:
            # Sort by timestamp
            batch.entries.sort(key=lambda e: e.timestamp)
            await self.storage.write_batch(batch)

class LogPushHandler:
    \"\"\"HTTP handler for Loki push protocol.\"\"\"

    def __init__(self, ingester: LogIngester):
        self.ingester = ingester

    async def handle_push(self, request_body: bytes) -> dict:
        \"\"\"Handle Loki push format.\"\"\"
        data = json.loads(request_body)
        entries = []

        for stream in data.get('streams', []):
            labels = self._parse_labels(stream.get('stream', {}))

            for value in stream.get('values', []):
                # Loki format: [timestamp_ns, line]
                ts_ns = int(value[0])
                line = value[1]

                entries.append(LogEntry(
                    timestamp=datetime.fromtimestamp(ts_ns / 1e9),
                    labels=labels,
                    line=line
                ))

        await self.ingester.ingest(entries)
        return {'status': 'success', 'entries': len(entries)}

    def _parse_labels(self, labels: dict) -> dict[str, str]:
        # Ensure all values are strings
        return {k: str(v) for k, v in labels.items()}
```"""
                },
                "pitfalls": [
                    "Out-of-order timestamps complicate querying",
                    "Memory pressure from unbounded batching",
                    "Label value with special characters breaks parsing"
                ]
            },
            {
                "name": "Log Index",
                "description": "Build inverted index for fast label-based log queries with bloom filters for negative lookups.",
                "hints": {
                    "level1": "Index maps label values to stream IDs containing them.",
                    "level2": "Use bloom filter per chunk to skip chunks without matches.",
                    "level3": """```python
from dataclasses import dataclass, field
import mmh3  # MurmurHash3 for bloom filter

class BloomFilter:
    def __init__(self, size: int = 10000, num_hashes: int = 7):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = bytearray((size + 7) // 8)

    def add(self, item: str):
        for i in range(self.num_hashes):
            idx = mmh3.hash(item, i) % self.size
            self.bits[idx // 8] |= (1 << (idx % 8))

    def might_contain(self, item: str) -> bool:
        for i in range(self.num_hashes):
            idx = mmh3.hash(item, i) % self.size
            if not (self.bits[idx // 8] & (1 << (idx % 8))):
                return False
        return True

@dataclass
class ChunkMeta:
    chunk_id: str
    stream_id: str
    start_time: float
    end_time: float
    entries: int
    bloom: BloomFilter

@dataclass
class StreamMeta:
    stream_id: str
    labels: dict[str, str]
    chunks: list[str] = field(default_factory=list)  # chunk IDs

class LogIndex:
    def __init__(self):
        # Label index: label_name -> label_value -> set of stream_ids
        self._label_index: dict[str, dict[str, set[str]]] = {}
        # Stream metadata
        self._streams: dict[str, StreamMeta] = {}
        # Chunk metadata
        self._chunks: dict[str, ChunkMeta] = {}

    def index_stream(self, stream_id: str, labels: dict[str, str]):
        self._streams[stream_id] = StreamMeta(stream_id=stream_id, labels=labels)

        for name, value in labels.items():
            if name not in self._label_index:
                self._label_index[name] = {}
            if value not in self._label_index[name]:
                self._label_index[name][value] = set()
            self._label_index[name][value].add(stream_id)

    def index_chunk(self, chunk: ChunkMeta):
        self._chunks[chunk.chunk_id] = chunk
        if stream := self._streams.get(chunk.stream_id):
            stream.chunks.append(chunk.chunk_id)

    def find_streams(self, matchers: list[LabelMatcher]) -> set[str]:
        \"\"\"Find stream IDs matching all label matchers.\"\"\"
        if not matchers:
            return set(self._streams.keys())

        result = None
        for matcher in matchers:
            matching = self._match_label(matcher)
            if result is None:
                result = matching
            else:
                result &= matching

        return result or set()

    def _match_label(self, matcher: LabelMatcher) -> set[str]:
        label_values = self._label_index.get(matcher.name, {})

        if matcher.type == MatchType.EQUAL:
            return label_values.get(matcher.value, set()).copy()
        elif matcher.type == MatchType.NOT_EQUAL:
            result = set()
            for value, streams in label_values.items():
                if value != matcher.value:
                    result |= streams
            return result
        elif matcher.type == MatchType.REGEX:
            result = set()
            pattern = re.compile(matcher.value)
            for value, streams in label_values.items():
                if pattern.match(value):
                    result |= streams
            return result

        return set()

    def find_chunks(self, stream_ids: set[str], start_time: float,
                   end_time: float, text_filter: str = None) -> list[str]:
        \"\"\"Find relevant chunk IDs for query.\"\"\"
        chunk_ids = []

        for stream_id in stream_ids:
            stream = self._streams.get(stream_id)
            if not stream:
                continue

            for chunk_id in stream.chunks:
                chunk = self._chunks.get(chunk_id)
                if not chunk:
                    continue

                # Time range filter
                if chunk.end_time < start_time or chunk.start_time > end_time:
                    continue

                # Bloom filter for text search
                if text_filter and not chunk.bloom.might_contain(text_filter):
                    continue

                chunk_ids.append(chunk_id)

        return chunk_ids

    def get_label_values(self, label_name: str) -> list[str]:
        \"\"\"Get all values for a label (for autocomplete).\"\"\"
        return list(self._label_index.get(label_name, {}).keys())

    def get_label_names(self) -> list[str]:
        \"\"\"Get all label names.\"\"\"
        return list(self._label_index.keys())
```"""
                },
                "pitfalls": [
                    "Bloom filter false positives require verification",
                    "High cardinality labels bloat index",
                    "Index rebuild on corruption is expensive"
                ]
            },
            {
                "name": "Log Query Engine",
                "description": "Build LogQL-style query engine with label filtering, text search, and log processing functions.",
                "hints": {
                    "level1": "Parse query into stream selector and pipeline stages.",
                    "level2": "Stream results to handle large result sets efficiently.",
                    "level3": """```python
from dataclasses import dataclass
from typing import Iterator, Callable
import re

@dataclass
class LogQueryResult:
    timestamp: datetime
    labels: dict[str, str]
    line: str

class LogQueryEngine:
    def __init__(self, index: LogIndex, storage):
        self.index = index
        self.storage = storage

    def query(self, query: str, start_time: float,
              end_time: float, limit: int = 1000) -> Iterator[LogQueryResult]:
        \"\"\"Execute LogQL query.\"\"\"
        parsed = self._parse_query(query)
        stream_ids = self.index.find_streams(parsed['matchers'])

        if not stream_ids:
            return

        # Find relevant chunks
        chunk_ids = self.index.find_chunks(
            stream_ids, start_time, end_time,
            text_filter=parsed.get('line_filter')
        )

        count = 0
        for chunk_id in chunk_ids:
            for entry in self.storage.read_chunk(chunk_id):
                # Apply time filter
                ts = entry.timestamp.timestamp()
                if ts < start_time or ts > end_time:
                    continue

                # Apply pipeline
                result = self._apply_pipeline(entry, parsed['pipeline'])
                if result:
                    yield result
                    count += 1
                    if count >= limit:
                        return

    def _parse_query(self, query: str) -> dict:
        \"\"\"Parse LogQL query.\"\"\"
        # {label="value"} |= "text" | json | line_format "{{.field}}"
        result = {
            'matchers': [],
            'pipeline': [],
            'line_filter': None
        }

        # Extract stream selector
        selector_match = re.match(r'\\{([^}]*)\\}', query)
        if selector_match:
            result['matchers'] = self._parse_matchers(selector_match.group(1))
            query = query[selector_match.end():]

        # Parse pipeline stages
        stages = query.split('|')
        for stage in stages[1:]:  # Skip empty first element
            stage = stage.strip()
            if stage.startswith('='):
                # Line filter: |= "text" or |~ "regex"
                op = stage[:2] if stage[1] in '=~' else stage[0]
                pattern = stage[len(op):].strip().strip('"')
                result['pipeline'].append(('filter', op, pattern))
                if op == '=':
                    result['line_filter'] = pattern
            elif stage == 'json':
                result['pipeline'].append(('json',))
            elif stage == 'logfmt':
                result['pipeline'].append(('logfmt',))
            elif stage.startswith('line_format'):
                template = re.search(r'"([^"]*)"', stage).group(1)
                result['pipeline'].append(('line_format', template))
            elif stage.startswith('label_format'):
                # label_format new_label=value
                assignments = self._parse_label_assignments(stage)
                result['pipeline'].append(('label_format', assignments))

        return result

    def _apply_pipeline(self, entry: LogEntry,
                       pipeline: list) -> LogQueryResult:
        \"\"\"Apply pipeline stages to log entry.\"\"\"
        line = entry.line
        labels = entry.labels.copy()

        for stage in pipeline:
            if stage[0] == 'filter':
                op, pattern = stage[1], stage[2]
                if op == '=':
                    if pattern not in line:
                        return None
                elif op == '!=':
                    if pattern in line:
                        return None
                elif op == '=~':
                    if not re.search(pattern, line):
                        return None
                elif op == '!~':
                    if re.search(pattern, line):
                        return None

            elif stage[0] == 'json':
                try:
                    parsed = json.loads(line)
                    labels.update({k: str(v) for k, v in parsed.items()})
                except json.JSONDecodeError:
                    pass  # Keep original line

            elif stage[0] == 'logfmt':
                # Parse key=value pairs
                for match in re.finditer(r'(\\w+)=("([^"]*)"|\\S+)', line):
                    key = match.group(1)
                    value = match.group(3) or match.group(2)
                    labels[key] = value

            elif stage[0] == 'line_format':
                template = stage[1]
                # Simple template substitution
                for key, value in labels.items():
                    template = template.replace(f'{{{{{key}}}}}', str(value))
                line = template

        return LogQueryResult(
            timestamp=entry.timestamp,
            labels=labels,
            line=line
        )

    def aggregate(self, query: str, start_time: float, end_time: float,
                 step: float) -> dict[str, list[tuple[float, float]]]:
        \"\"\"Execute metric query over logs (count_over_time, rate, etc.).\"\"\"
        # Parse aggregation: count_over_time({job="app"}[5m])
        match = re.match(r'(\\w+)\\((.+)\\[(\\d+)([smh])\\]\\)', query)
        if not match:
            raise ValueError("Invalid aggregation query")

        func = match.group(1)
        inner_query = match.group(2)
        interval = int(match.group(3)) * {'s': 1, 'm': 60, 'h': 3600}[match.group(4)]

        results = {}
        for ts in self._time_range(start_time, end_time, step):
            window_start = ts - interval
            count = 0
            for _ in self.query(inner_query, window_start, ts, limit=10000):
                count += 1

            key = 'total'  # Could be grouped by labels
            results.setdefault(key, []).append((ts, count))

        return results
```"""
                },
                "pitfalls": [
                    "Unbounded query scans entire log history",
                    "JSON parsing failures silently drop entries",
                    "Pipeline order matters for filter efficiency"
                ]
            }
        ]
    },

    "alerting-system": {
        "name": "Alerting System",
        "description": "Build an alerting system with rule evaluation, alert grouping, silencing, and notification routing.",
        "why_important": "Alerting is critical for on-call. Understanding alert fatigue, grouping, and routing helps design better alert systems.",
        "difficulty": "intermediate",
        "tags": ["observability", "distributed-systems"],
        "estimated_hours": 35,
        "prerequisites": ["metrics-collector"],
        "learning_outcomes": [
            "Design alert rule evaluation engine",
            "Implement alert state machine (pending, firing, resolved)",
            "Build notification routing and grouping",
            "Handle alert silencing and inhibition"
        ],
        "milestones": [
            {
                "name": "Alert Rule Evaluation",
                "description": "Build rule evaluation engine that periodically queries metrics and triggers alerts based on thresholds.",
                "hints": {
                    "level1": "Define rules with PromQL expression and threshold.",
                    "level2": "Track alert state transitions: inactive -> pending -> firing.",
                    "level3": """```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import asyncio

class AlertState(str, Enum):
    INACTIVE = "inactive"
    PENDING = "pending"
    FIRING = "firing"

@dataclass
class AlertRule:
    name: str
    expr: str  # PromQL expression
    for_duration: timedelta = timedelta(0)  # Duration before firing
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    evaluation_interval: timedelta = timedelta(seconds=60)

@dataclass
class Alert:
    rule: AlertRule
    labels: dict[str, str]
    state: AlertState = AlertState.INACTIVE
    active_at: Optional[datetime] = None
    fired_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    value: float = 0
    annotations: dict[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        import hashlib
        import json
        label_str = json.dumps(self.labels, sort_keys=True)
        return hashlib.md5(f"{self.rule.name}:{label_str}".encode()).hexdigest()

class AlertEvaluator:
    def __init__(self, query_engine, notifier):
        self.query_engine = query_engine
        self.notifier = notifier
        self.rules: list[AlertRule] = []
        self.active_alerts: dict[str, Alert] = {}
        self._running = False

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    async def start(self):
        self._running = True
        while self._running:
            await self._evaluate_all()
            await asyncio.sleep(60)  # Evaluation interval

    async def stop(self):
        self._running = False

    async def _evaluate_all(self):
        for rule in self.rules:
            await self._evaluate_rule(rule)

    async def _evaluate_rule(self, rule: AlertRule):
        now = datetime.utcnow()

        # Query metrics
        results = self.query_engine.instant_query(rule.expr, now.timestamp())

        # Track which alerts are still active
        active_fingerprints = set()

        for labels, value in results.items():
            # Merge rule labels with metric labels
            all_labels = {**labels.to_dict(), **rule.labels}
            alert = Alert(rule=rule, labels=all_labels, value=value)
            fingerprint = alert.fingerprint
            active_fingerprints.add(fingerprint)

            # Check for existing alert
            existing = self.active_alerts.get(fingerprint)

            if existing:
                # Update existing alert
                self._update_alert(existing, value, now)
            else:
                # New alert
                alert.state = AlertState.PENDING
                alert.active_at = now
                alert.annotations = self._render_annotations(rule.annotations, all_labels, value)
                self.active_alerts[fingerprint] = alert

        # Resolve alerts no longer active
        for fingerprint in list(self.active_alerts.keys()):
            if fingerprint not in active_fingerprints:
                alert = self.active_alerts[fingerprint]
                if alert.state != AlertState.INACTIVE:
                    await self._resolve_alert(alert, now)

    def _update_alert(self, alert: Alert, value: float, now: datetime):
        alert.value = value

        if alert.state == AlertState.PENDING:
            # Check if pending duration has passed
            pending_duration = now - alert.active_at
            if pending_duration >= alert.rule.for_duration:
                alert.state = AlertState.FIRING
                alert.fired_at = now
                asyncio.create_task(self.notifier.notify(alert))

        elif alert.state == AlertState.FIRING:
            # Still firing, update value
            pass

    async def _resolve_alert(self, alert: Alert, now: datetime):
        if alert.state == AlertState.FIRING:
            alert.resolved_at = now
            alert.state = AlertState.INACTIVE
            await self.notifier.notify_resolved(alert)
        del self.active_alerts[alert.fingerprint]

    def _render_annotations(self, templates: dict, labels: dict, value: float) -> dict:
        result = {}
        for key, template in templates.items():
            text = template
            for label_name, label_value in labels.items():
                text = text.replace(f'{{{{ $labels.{label_name} }}}}', str(label_value))
            text = text.replace('{{ $value }}', str(value))
            result[key] = text
        return result
```"""
                },
                "pitfalls": [
                    "Flapping alerts from noisy metrics need hysteresis",
                    "for_duration resets if metric briefly returns normal",
                    "Template rendering errors crash evaluation loop"
                ]
            },
            {
                "name": "Alert Grouping",
                "description": "Group related alerts together to reduce notification noise with configurable grouping keys.",
                "hints": {
                    "level1": "Group alerts by common labels (e.g., alertname, cluster).",
                    "level2": "Batch notifications within group_wait window.",
                    "level3": """```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import asyncio

@dataclass
class AlertGroup:
    key: str  # Hash of group labels
    labels: dict[str, str]  # Group labels
    alerts: dict[str, Alert] = field(default_factory=dict)
    first_alert_at: Optional[datetime] = None
    last_notification_at: Optional[datetime] = None

class AlertGrouper:
    def __init__(self, group_by: list[str],
                 group_wait: timedelta = timedelta(seconds=30),
                 group_interval: timedelta = timedelta(minutes=5),
                 repeat_interval: timedelta = timedelta(hours=4)):
        self.group_by = group_by
        self.group_wait = group_wait
        self.group_interval = group_interval
        self.repeat_interval = repeat_interval
        self._groups: dict[str, AlertGroup] = {}
        self._pending_notifications: dict[str, asyncio.Task] = {}

    def add_alert(self, alert: Alert) -> AlertGroup:
        \"\"\"Add alert to appropriate group.\"\"\"
        group_labels = {k: alert.labels.get(k, '') for k in self.group_by}
        group_key = self._compute_key(group_labels)

        if group_key not in self._groups:
            self._groups[group_key] = AlertGroup(
                key=group_key,
                labels=group_labels,
                first_alert_at=datetime.utcnow()
            )

        group = self._groups[group_key]
        group.alerts[alert.fingerprint] = alert

        # Schedule notification if not already pending
        if group_key not in self._pending_notifications:
            self._schedule_notification(group)

        return group

    def remove_alert(self, alert: Alert) -> Optional[AlertGroup]:
        \"\"\"Remove resolved alert from group.\"\"\"
        group_labels = {k: alert.labels.get(k, '') for k in self.group_by}
        group_key = self._compute_key(group_labels)

        group = self._groups.get(group_key)
        if group:
            group.alerts.pop(alert.fingerprint, None)
            if not group.alerts:
                # Group empty, remove it
                del self._groups[group_key]
                return None
        return group

    def _compute_key(self, labels: dict[str, str]) -> str:
        import hashlib
        import json
        return hashlib.md5(json.dumps(labels, sort_keys=True).encode()).hexdigest()

    def _schedule_notification(self, group: AlertGroup):
        \"\"\"Schedule notification after group_wait.\"\"\"
        async def notify_after_wait():
            await asyncio.sleep(self.group_wait.total_seconds())
            await self._send_notification(group)

        self._pending_notifications[group.key] = asyncio.create_task(notify_after_wait())

    async def _send_notification(self, group: AlertGroup):
        now = datetime.utcnow()

        # Check if we should notify
        if group.last_notification_at:
            since_last = now - group.last_notification_at

            # Check repeat interval for firing alerts
            if since_last < self.repeat_interval:
                # Check group interval for new alerts
                if since_last < self.group_interval:
                    # Too soon, reschedule
                    self._schedule_notification(group)
                    return

        group.last_notification_at = now
        self._pending_notifications.pop(group.key, None)

        # Yield notification to be sent
        return group

    def get_groups(self) -> list[AlertGroup]:
        return list(self._groups.values())

class NotificationBatcher:
    \"\"\"Batches notifications to reduce alert fatigue.\"\"\"

    def __init__(self, grouper: AlertGrouper, sender):
        self.grouper = grouper
        self.sender = sender

    async def on_alert(self, alert: Alert):
        group = self.grouper.add_alert(alert)
        # Notification will be sent after group_wait

    async def on_resolved(self, alert: Alert):
        group = self.grouper.remove_alert(alert)
        if group:
            # Send resolution notification
            await self.sender.send_grouped(group, resolved=[alert])

    async def process_groups(self):
        \"\"\"Periodically check and send pending notifications.\"\"\"
        for group in self.grouper.get_groups():
            notification = await self.grouper._send_notification(group)
            if notification:
                await self.sender.send_grouped(group)
```"""
                },
                "pitfalls": [
                    "Group key change orphans alert in old group",
                    "Long group_wait delays critical alerts",
                    "Memory leak from empty groups not cleaned up"
                ]
            },
            {
                "name": "Silencing & Inhibition",
                "description": "Implement alert silencing for maintenance windows and inhibition to suppress alerts when related alerts fire.",
                "hints": {
                    "level1": "Silence matches alerts by label matchers within time window.",
                    "level2": "Inhibition: if source alert fires, suppress target alerts with matching labels.",
                    "level3": """```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import re

@dataclass
class Silence:
    id: str
    matchers: list[LabelMatcher]
    starts_at: datetime
    ends_at: datetime
    created_by: str
    comment: str

    def is_active(self, now: datetime = None) -> bool:
        now = now or datetime.utcnow()
        return self.starts_at <= now <= self.ends_at

    def matches(self, labels: dict[str, str]) -> bool:
        return all(m.matches_dict(labels) for m in self.matchers)

@dataclass
class InhibitRule:
    source_matchers: list[LabelMatcher]  # Source alert must match
    target_matchers: list[LabelMatcher]  # Target alert must match
    equal: list[str]  # Labels that must be equal between source and target

class SilenceManager:
    def __init__(self):
        self.silences: dict[str, Silence] = {}

    def create(self, matchers: list[LabelMatcher], starts_at: datetime,
               ends_at: datetime, created_by: str, comment: str) -> Silence:
        import uuid
        silence = Silence(
            id=str(uuid.uuid4()),
            matchers=matchers,
            starts_at=starts_at,
            ends_at=ends_at,
            created_by=created_by,
            comment=comment
        )
        self.silences[silence.id] = silence
        return silence

    def delete(self, silence_id: str):
        self.silences.pop(silence_id, None)

    def is_silenced(self, labels: dict[str, str], now: datetime = None) -> Optional[Silence]:
        for silence in self.silences.values():
            if silence.is_active(now) and silence.matches(labels):
                return silence
        return None

    def cleanup_expired(self):
        now = datetime.utcnow()
        self.silences = {
            id: s for id, s in self.silences.items()
            if s.ends_at > now
        }

class InhibitionProcessor:
    def __init__(self, rules: list[InhibitRule]):
        self.rules = rules

    def is_inhibited(self, target: Alert, active_alerts: list[Alert]) -> bool:
        \"\"\"Check if target alert is inhibited by any active alert.\"\"\"
        for rule in self.rules:
            # Check if target matches target_matchers
            if not self._matches_all(target.labels, rule.target_matchers):
                continue

            # Find source alerts that could inhibit
            for source in active_alerts:
                if source.fingerprint == target.fingerprint:
                    continue

                if source.state != AlertState.FIRING:
                    continue

                # Check if source matches source_matchers
                if not self._matches_all(source.labels, rule.source_matchers):
                    continue

                # Check equal labels
                if self._labels_equal(source.labels, target.labels, rule.equal):
                    return True

        return False

    def _matches_all(self, labels: dict, matchers: list[LabelMatcher]) -> bool:
        return all(m.matches_dict(labels) for m in matchers)

    def _labels_equal(self, source: dict, target: dict, keys: list[str]) -> bool:
        return all(source.get(k) == target.get(k) for k in keys)

class AlertProcessor:
    \"\"\"Main processor combining silencing, inhibition, and grouping.\"\"\"

    def __init__(self, silence_manager: SilenceManager,
                 inhibitor: InhibitionProcessor,
                 grouper: AlertGrouper,
                 notifier):
        self.silences = silence_manager
        self.inhibitor = inhibitor
        self.grouper = grouper
        self.notifier = notifier
        self.active_alerts: list[Alert] = []

    async def process(self, alert: Alert):
        # Check if silenced
        silence = self.silences.is_silenced(alert.labels)
        if silence:
            alert.silenced_by = silence.id
            return

        # Check if inhibited
        if self.inhibitor.is_inhibited(alert, self.active_alerts):
            alert.inhibited = True
            return

        # Track active alert
        self.active_alerts = [a for a in self.active_alerts
                             if a.fingerprint != alert.fingerprint]
        if alert.state == AlertState.FIRING:
            self.active_alerts.append(alert)

        # Add to group for notification
        self.grouper.add_alert(alert)
```"""
                },
                "pitfalls": [
                    "Inhibition loop when alerts inhibit each other",
                    "Silence with wrong matchers misses alerts",
                    "Race between silence creation and firing alert"
                ]
            },
            {
                "name": "Notification Routing",
                "description": "Route alerts to different receivers (Slack, PagerDuty, email) based on matching rules.",
                "hints": {
                    "level1": "Define routes with matchers that direct to receivers.",
                    "level2": "Support nested routes with continue/stop semantics.",
                    "level3": """```python
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional
import aiohttp

@dataclass
class Route:
    matchers: list[LabelMatcher] = field(default_factory=list)
    receiver: str = ""
    continue_matching: bool = False
    children: list['Route'] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)

class Receiver(ABC):
    @abstractmethod
    async def send(self, group: AlertGroup):
        pass

class SlackReceiver(Receiver):
    def __init__(self, webhook_url: str, channel: str):
        self.webhook_url = webhook_url
        self.channel = channel

    async def send(self, group: AlertGroup):
        alerts = list(group.alerts.values())
        firing = [a for a in alerts if a.state == AlertState.FIRING]

        blocks = [{
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🔥 {len(firing)} alert(s) firing"
            }
        }]

        for alert in firing[:10]:  # Limit to 10
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{alert.labels.get('alertname', 'Unknown')}*\\n"
                           f"{alert.annotations.get('summary', '')}\\n"
                           f"Value: {alert.value}"
                }
            })

        payload = {
            "channel": self.channel,
            "blocks": blocks
        }

        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=payload)

class PagerDutyReceiver(Receiver):
    def __init__(self, service_key: str):
        self.service_key = service_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"

    async def send(self, group: AlertGroup):
        alerts = list(group.alerts.values())

        for alert in alerts:
            if alert.state == AlertState.FIRING:
                event_action = "trigger"
            else:
                event_action = "resolve"

            payload = {
                "routing_key": self.service_key,
                "event_action": event_action,
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": alert.annotations.get('summary', alert.labels.get('alertname')),
                    "source": alert.labels.get('instance', 'unknown'),
                    "severity": alert.labels.get('severity', 'warning'),
                    "custom_details": alert.labels
                }
            }

            async with aiohttp.ClientSession() as session:
                await session.post(self.api_url, json=payload)

class NotificationRouter:
    def __init__(self, root_route: Route, receivers: dict[str, Receiver]):
        self.root = root_route
        self.receivers = receivers

    async def route(self, group: AlertGroup):
        \"\"\"Route alert group to matching receivers.\"\"\"
        matched_receivers = self._find_receivers(group, self.root)

        for receiver_name in matched_receivers:
            receiver = self.receivers.get(receiver_name)
            if receiver:
                try:
                    await receiver.send(group)
                except Exception as e:
                    print(f"Failed to send to {receiver_name}: {e}")

    def _find_receivers(self, group: AlertGroup, route: Route) -> list[str]:
        \"\"\"Find all matching receivers for alert group.\"\"\"
        # Check if this route matches
        if not self._matches_route(group, route):
            return []

        receivers = []

        # Add this route's receiver if specified
        if route.receiver:
            receivers.append(route.receiver)

        # Check children
        for child in route.children:
            child_receivers = self._find_receivers(group, child)
            receivers.extend(child_receivers)

            # Stop if matched and not continue
            if child_receivers and not child.continue_matching:
                break

        return receivers

    def _matches_route(self, group: AlertGroup, route: Route) -> bool:
        if not route.matchers:
            return True  # Default route matches all

        # Use first alert's labels for matching
        if not group.alerts:
            return False

        first_alert = list(group.alerts.values())[0]
        return all(m.matches_dict(first_alert.labels) for m in route.matchers)
```"""
                },
                "pitfalls": [
                    "Missing default route causes unrouted alerts",
                    "continue=true on all routes sends duplicate notifications",
                    "Rate limiting by receiver prevents important alerts"
                ]
            }
        ]
    },

    "apm-system": {
        "name": "APM Tracing System",
        "description": "Build an Application Performance Monitoring system with distributed tracing, service maps, and performance analysis.",
        "why_important": "APM helps identify performance bottlenecks across microservices. Understanding trace data helps design better instrumentation.",
        "difficulty": "advanced",
        "tags": ["observability", "distributed-systems", "performance"],
        "estimated_hours": 45,
        "prerequisites": ["distributed-tracing"],
        "learning_outcomes": [
            "Design span collection and storage",
            "Build service dependency maps from traces",
            "Implement latency percentile analysis",
            "Handle high-volume trace sampling"
        ],
        "milestones": [
            {
                "name": "Trace Collection",
                "description": "Build trace collector that ingests spans from multiple services with proper parent-child linking.",
                "hints": {
                    "level1": "Collect spans with trace ID, span ID, parent ID, timestamps.",
                    "level2": "Handle out-of-order span arrival, reconstruct trace tree.",
                    "level3": """```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import defaultdict
import asyncio

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    duration_us: int  # microseconds
    status: str = "OK"  # OK, ERROR
    tags: dict = field(default_factory=dict)
    logs: list = field(default_factory=list)

@dataclass
class Trace:
    trace_id: str
    root_span: Optional[Span] = None
    spans: dict[str, Span] = field(default_factory=dict)  # span_id -> Span
    services: set = field(default_factory=set)
    start_time: Optional[datetime] = None
    duration_us: int = 0

    def add_span(self, span: Span):
        self.spans[span.span_id] = span
        self.services.add(span.service_name)

        if span.parent_id is None:
            self.root_span = span
            self.start_time = span.start_time
            self.duration_us = span.duration_us

    def get_children(self, span_id: str) -> list[Span]:
        return [s for s in self.spans.values() if s.parent_id == span_id]

    def get_depth(self) -> int:
        if not self.root_span:
            return 0

        def depth(span_id: str) -> int:
            children = self.get_children(span_id)
            if not children:
                return 1
            return 1 + max(depth(c.span_id) for c in children)

        return depth(self.root_span.span_id)

class TraceCollector:
    def __init__(self, storage, assembler_timeout: float = 30.0):
        self.storage = storage
        self.assembler_timeout = assembler_timeout
        self._pending_traces: dict[str, Trace] = {}
        self._trace_timers: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def collect(self, spans: list[Span]):
        \"\"\"Collect batch of spans.\"\"\"
        async with self._lock:
            for span in spans:
                await self._add_span(span)

    async def _add_span(self, span: Span):
        trace_id = span.trace_id

        if trace_id not in self._pending_traces:
            self._pending_traces[trace_id] = Trace(trace_id=trace_id)
            # Start assembly timeout
            self._trace_timers[trace_id] = asyncio.create_task(
                self._assembly_timeout(trace_id)
            )

        trace = self._pending_traces[trace_id]
        trace.add_span(span)

        # Check if trace is complete (has root and reasonable time passed)
        if trace.root_span and self._is_likely_complete(trace):
            await self._finalize_trace(trace_id)

    def _is_likely_complete(self, trace: Trace) -> bool:
        # Heuristic: trace is complete if we have root and all spans
        # have their parents (except root)
        for span in trace.spans.values():
            if span.parent_id and span.parent_id not in trace.spans:
                return False  # Missing parent
        return True

    async def _assembly_timeout(self, trace_id: str):
        await asyncio.sleep(self.assembler_timeout)
        async with self._lock:
            if trace_id in self._pending_traces:
                await self._finalize_trace(trace_id)

    async def _finalize_trace(self, trace_id: str):
        trace = self._pending_traces.pop(trace_id, None)
        timer = self._trace_timers.pop(trace_id, None)
        if timer:
            timer.cancel()

        if trace:
            # Calculate trace-level metrics
            self._calculate_metrics(trace)
            await self.storage.store_trace(trace)

    def _calculate_metrics(self, trace: Trace):
        if not trace.spans:
            return

        # Find actual start/end times
        start_times = [s.start_time for s in trace.spans.values()]
        end_times = [
            s.start_time.timestamp() * 1e6 + s.duration_us
            for s in trace.spans.values()
        ]

        trace.start_time = min(start_times)
        trace.duration_us = int(max(end_times) - min(s.timestamp() * 1e6 for s in start_times))

class OTLPReceiver:
    \"\"\"OpenTelemetry Protocol receiver.\"\"\"

    def __init__(self, collector: TraceCollector):
        self.collector = collector

    async def handle_traces(self, request_body: bytes) -> dict:
        # Parse OTLP protobuf (simplified - real impl uses protobuf)
        import json
        data = json.loads(request_body)

        spans = []
        for resource_span in data.get('resourceSpans', []):
            service_name = self._get_service_name(resource_span.get('resource', {}))

            for scope_span in resource_span.get('scopeSpans', []):
                for span_data in scope_span.get('spans', []):
                    spans.append(Span(
                        trace_id=span_data['traceId'],
                        span_id=span_data['spanId'],
                        parent_id=span_data.get('parentSpanId'),
                        operation_name=span_data['name'],
                        service_name=service_name,
                        start_time=datetime.fromtimestamp(span_data['startTimeUnixNano'] / 1e9),
                        duration_us=int((span_data['endTimeUnixNano'] - span_data['startTimeUnixNano']) / 1000),
                        status=span_data.get('status', {}).get('code', 'OK'),
                        tags=self._parse_attributes(span_data.get('attributes', []))
                    ))

        await self.collector.collect(spans)
        return {'accepted': len(spans)}

    def _get_service_name(self, resource: dict) -> str:
        for attr in resource.get('attributes', []):
            if attr['key'] == 'service.name':
                return attr['value'].get('stringValue', 'unknown')
        return 'unknown'

    def _parse_attributes(self, attributes: list) -> dict:
        result = {}
        for attr in attributes:
            key = attr['key']
            value = attr['value']
            if 'stringValue' in value:
                result[key] = value['stringValue']
            elif 'intValue' in value:
                result[key] = value['intValue']
            elif 'boolValue' in value:
                result[key] = value['boolValue']
        return result
```"""
                },
                "pitfalls": [
                    "Late-arriving spans miss assembly window",
                    "Memory grows unbounded with incomplete traces",
                    "Clock skew makes span ordering incorrect"
                ]
            },
            {
                "name": "Service Map",
                "description": "Build service dependency map from trace data showing call relationships and error rates.",
                "hints": {
                    "level1": "Track caller-callee relationships from parent-child spans.",
                    "level2": "Aggregate metrics per edge: latency percentiles, error rate, throughput.",
                    "level3": """```python
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

@dataclass
class ServiceEdge:
    source: str
    target: str
    request_count: int = 0
    error_count: int = 0
    latencies: list[int] = field(default_factory=list)  # microseconds

    @property
    def error_rate(self) -> float:
        return self.error_count / self.request_count if self.request_count else 0

    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0
        return statistics.median(self.latencies)

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

@dataclass
class ServiceNode:
    name: str
    request_count: int = 0
    error_count: int = 0
    latencies: list[int] = field(default_factory=list)

class ServiceMapBuilder:
    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.nodes: dict[str, ServiceNode] = {}
        self.edges: dict[str, ServiceEdge] = {}  # "source->target" -> edge
        self._lock = asyncio.Lock()

    async def process_trace(self, trace: Trace):
        \"\"\"Extract service relationships from trace.\"\"\"
        async with self._lock:
            for span in trace.spans.values():
                # Update service node
                service = span.service_name
                if service not in self.nodes:
                    self.nodes[service] = ServiceNode(name=service)

                node = self.nodes[service]
                node.request_count += 1
                if span.status == "ERROR":
                    node.error_count += 1
                node.latencies.append(span.duration_us)

                # Update edge if span has parent
                if span.parent_id and span.parent_id in trace.spans:
                    parent = trace.spans[span.parent_id]
                    if parent.service_name != span.service_name:
                        edge_key = f"{parent.service_name}->{span.service_name}"
                        if edge_key not in self.edges:
                            self.edges[edge_key] = ServiceEdge(
                                source=parent.service_name,
                                target=span.service_name
                            )

                        edge = self.edges[edge_key]
                        edge.request_count += 1
                        if span.status == "ERROR":
                            edge.error_count += 1
                        edge.latencies.append(span.duration_us)

    def get_map(self) -> dict:
        \"\"\"Return service map as graph data.\"\"\"
        return {
            'nodes': [
                {
                    'id': name,
                    'requests': node.request_count,
                    'error_rate': node.error_count / node.request_count if node.request_count else 0,
                    'p50_latency_ms': statistics.median(node.latencies) / 1000 if node.latencies else 0,
                    'p99_latency_ms': self._percentile(node.latencies, 0.99) / 1000 if node.latencies else 0
                }
                for name, node in self.nodes.items()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'requests': edge.request_count,
                    'error_rate': edge.error_rate,
                    'p50_latency_ms': edge.p50_latency / 1000,
                    'p99_latency_ms': edge.p99_latency / 1000
                }
                for edge in self.edges.values()
            ]
        }

    def _percentile(self, values: list, p: float) -> float:
        if not values:
            return 0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def get_dependencies(self, service: str) -> dict:
        \"\"\"Get upstream and downstream dependencies for a service.\"\"\"
        upstream = []
        downstream = []

        for edge in self.edges.values():
            if edge.target == service:
                upstream.append({
                    'service': edge.source,
                    'requests': edge.request_count,
                    'error_rate': edge.error_rate
                })
            elif edge.source == service:
                downstream.append({
                    'service': edge.target,
                    'requests': edge.request_count,
                    'error_rate': edge.error_rate
                })

        return {
            'service': service,
            'upstream': upstream,
            'downstream': downstream
        }
```"""
                },
                "pitfalls": [
                    "Same-service spans inflate node metrics",
                    "Async calls appear as separate traces",
                    "High cardinality operations bloat map"
                ]
            },
            {
                "name": "Trace Sampling",
                "description": "Implement adaptive sampling to reduce storage costs while preserving interesting traces.",
                "hints": {
                    "level1": "Sample N% of traces randomly, but always keep errors.",
                    "level2": "Tail-based sampling: decide after trace completes based on characteristics.",
                    "level3": """```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import hashlib

class Sampler(ABC):
    @abstractmethod
    def should_sample(self, trace: Trace) -> bool:
        pass

class ProbabilisticSampler(Sampler):
    def __init__(self, sample_rate: float = 0.1):
        self.sample_rate = sample_rate

    def should_sample(self, trace: Trace) -> bool:
        # Use trace ID for consistent sampling
        hash_val = int(hashlib.md5(trace.trace_id.encode()).hexdigest(), 16)
        return (hash_val % 10000) < (self.sample_rate * 10000)

class RateLimitingSampler(Sampler):
    def __init__(self, traces_per_second: float = 100):
        self.traces_per_second = traces_per_second
        self._tokens = traces_per_second
        self._last_refill = time.time()

    def should_sample(self, trace: Trace) -> bool:
        self._refill()
        if self._tokens >= 1:
            self._tokens -= 1
            return True
        return False

    def _refill(self):
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.traces_per_second,
            self._tokens + elapsed * self.traces_per_second
        )
        self._last_refill = now

@dataclass
class SamplingPolicy:
    name: str
    matcher: callable  # (Trace) -> bool
    sampler: Sampler
    priority: int = 0

class TailBasedSampler:
    \"\"\"Decides sampling after trace is complete.\"\"\"

    def __init__(self, default_rate: float = 0.01):
        self.default_sampler = ProbabilisticSampler(default_rate)
        self.policies: list[SamplingPolicy] = []
        self._setup_default_policies()

    def _setup_default_policies(self):
        # Always sample errors
        self.policies.append(SamplingPolicy(
            name="errors",
            matcher=lambda t: any(s.status == "ERROR" for s in t.spans.values()),
            sampler=ProbabilisticSampler(1.0),  # 100%
            priority=100
        ))

        # Sample slow traces
        self.policies.append(SamplingPolicy(
            name="slow",
            matcher=lambda t: t.duration_us > 5_000_000,  # > 5s
            sampler=ProbabilisticSampler(0.5),  # 50%
            priority=90
        ))

        # Sample traces with many spans (complex flows)
        self.policies.append(SamplingPolicy(
            name="complex",
            matcher=lambda t: len(t.spans) > 50,
            sampler=ProbabilisticSampler(0.3),
            priority=80
        ))

        # Higher rate for specific services
        self.policies.append(SamplingPolicy(
            name="payment_service",
            matcher=lambda t: "payment-service" in t.services,
            sampler=ProbabilisticSampler(0.2),
            priority=70
        ))

    def add_policy(self, policy: SamplingPolicy):
        self.policies.append(policy)
        self.policies.sort(key=lambda p: p.priority, reverse=True)

    def should_sample(self, trace: Trace) -> tuple[bool, str]:
        \"\"\"Returns (should_sample, reason).\"\"\"
        for policy in self.policies:
            if policy.matcher(trace):
                if policy.sampler.should_sample(trace):
                    return True, policy.name
                # Policy matched but sampler said no
                # Continue to check lower priority policies

        # Fall back to default
        if self.default_sampler.should_sample(trace):
            return True, "default"

        return False, "dropped"

class AdaptiveSampler:
    \"\"\"Adjusts sampling rate based on throughput.\"\"\"

    def __init__(self, target_traces_per_minute: int = 1000):
        self.target = target_traces_per_minute
        self.current_rate = 0.1
        self._count = 0
        self._last_adjust = time.time()
        self._adjust_interval = 60  # seconds

    def should_sample(self, trace: Trace) -> bool:
        self._maybe_adjust()

        if random.random() < self.current_rate:
            self._count += 1
            return True
        return False

    def _maybe_adjust(self):
        now = time.time()
        if now - self._last_adjust < self._adjust_interval:
            return

        # Calculate actual rate
        actual_per_minute = self._count

        # Adjust rate
        if actual_per_minute > self.target * 1.1:
            # Too many, decrease rate
            self.current_rate *= 0.9
        elif actual_per_minute < self.target * 0.9:
            # Too few, increase rate
            self.current_rate *= 1.1

        self.current_rate = max(0.001, min(1.0, self.current_rate))
        self._count = 0
        self._last_adjust = now
```"""
                },
                "pitfalls": [
                    "Head-based sampling loses interesting traces",
                    "Rate limiting causes bursty drops",
                    "Adaptive sampling oscillates under variable load"
                ]
            }
        ]
    }
}

# Load and update YAML
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

expert_projects = data.get('expert_projects', {})

for project_id, project_data in observability_projects.items():
    if project_id not in expert_projects:
        expert_projects[project_id] = project_data
        print(f"Added: {project_id}")
    else:
        print(f"Skipped (exists): {project_id}")

# Save
with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nTotal expert_projects: {len(expert_projects)}")

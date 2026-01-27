#!/usr/bin/env python3
"""
Add Real-time & Multiplayer projects to the curriculum.
Focus on WebSockets, game servers, collaborative editing, event sourcing.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

realtime_projects = {
    "websocket-server": {
        "name": "WebSocket Server",
        "description": "Build a WebSocket server from scratch supporting the RFC 6455 protocol with connection management, heartbeats, and message framing.",
        "why_important": "WebSockets enable real-time bidirectional communication essential for chat apps, live updates, gaming, and collaborative tools.",
        "difficulty": "intermediate",
        "tags": ["networking", "real-time", "protocols"],
        "estimated_hours": 30,
        "prerequisites": ["tcp-server"],
        "learning_outcomes": [
            "Master WebSocket handshake and frame parsing",
            "Implement connection lifecycle management",
            "Handle binary and text message types",
            "Build heartbeat/ping-pong mechanisms"
        ],
        "milestones": [
            {
                "name": "HTTP Upgrade Handshake",
                "description": "Implement WebSocket handshake by parsing HTTP upgrade request and computing Sec-WebSocket-Accept using SHA-1 and Base64.",
                "hints": {
                    "level1": "Parse HTTP headers and validate Upgrade request.",
                    "level2": "Concatenate client key with magic GUID, SHA-1 hash, Base64 encode.",
                    "level3": """```python
import hashlib
import base64

MAGIC_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

def compute_accept_key(client_key: str) -> str:
    combined = client_key.strip() + MAGIC_GUID
    sha1_hash = hashlib.sha1(combined.encode()).digest()
    return base64.b64encode(sha1_hash).decode()

def parse_upgrade_request(data: bytes) -> dict:
    lines = data.decode().split('\\r\\n')
    headers = {}
    for line in lines[1:]:
        if ': ' in line:
            key, value = line.split(': ', 1)
            headers[key.lower()] = value
    return headers

def create_handshake_response(client_key: str) -> bytes:
    accept = compute_accept_key(client_key)
    return (
        "HTTP/1.1 101 Switching Protocols\\r\\n"
        "Upgrade: websocket\\r\\n"
        "Connection: Upgrade\\r\\n"
        f"Sec-WebSocket-Accept: {accept}\\r\\n"
        "\\r\\n"
    ).encode()
```"""
                },
                "pitfalls": [
                    "Missing CRLF at end of headers causes handshake failure",
                    "Case-sensitive header parsing breaks with some clients",
                    "Not validating WebSocket version header"
                ]
            },
            {
                "name": "Frame Parsing",
                "description": "Parse WebSocket frames including opcode, masking, and payload length handling for small and large messages.",
                "hints": {
                    "level1": "Read first 2 bytes for FIN, opcode, mask bit, and initial length.",
                    "level2": "Handle 7-bit, 16-bit (126), and 64-bit (127) payload lengths.",
                    "level3": """```python
import struct
from dataclasses import dataclass
from enum import IntEnum

class Opcode(IntEnum):
    CONTINUATION = 0x0
    TEXT = 0x1
    BINARY = 0x2
    CLOSE = 0x8
    PING = 0x9
    PONG = 0xA

@dataclass
class Frame:
    fin: bool
    opcode: Opcode
    payload: bytes

def parse_frame(data: bytes) -> tuple[Frame, int]:
    if len(data) < 2:
        raise ValueError("Incomplete frame")

    byte1, byte2 = data[0], data[1]
    fin = bool(byte1 & 0x80)
    opcode = Opcode(byte1 & 0x0F)
    masked = bool(byte2 & 0x80)
    length = byte2 & 0x7F

    offset = 2
    if length == 126:
        length = struct.unpack('>H', data[2:4])[0]
        offset = 4
    elif length == 127:
        length = struct.unpack('>Q', data[2:10])[0]
        offset = 10

    if masked:
        mask = data[offset:offset+4]
        offset += 4
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(data[offset:offset+length]))
    else:
        payload = data[offset:offset+length]

    return Frame(fin, opcode, payload), offset + length
```"""
                },
                "pitfalls": [
                    "Forgetting to unmask client frames (clients MUST mask)",
                    "Integer overflow with 64-bit payload lengths",
                    "Not handling fragmented messages (FIN=0)"
                ]
            },
            {
                "name": "Connection Management",
                "description": "Manage multiple WebSocket connections with proper state tracking, broadcasting, and graceful disconnection.",
                "hints": {
                    "level1": "Track connections in a dictionary keyed by unique ID.",
                    "level2": "Use asyncio for concurrent connection handling.",
                    "level3": """```python
import asyncio
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Connection:
    id: str
    writer: asyncio.StreamWriter
    reader: asyncio.StreamReader
    is_alive: bool = True
    rooms: set = field(default_factory=set)

class ConnectionManager:
    def __init__(self):
        self.connections: dict[str, Connection] = {}
        self.rooms: dict[str, set[str]] = {}  # room -> connection_ids

    async def add(self, conn: Connection):
        self.connections[conn.id] = conn

    async def remove(self, conn_id: str):
        if conn := self.connections.pop(conn_id, None):
            for room in conn.rooms:
                self.rooms.get(room, set()).discard(conn_id)
            conn.writer.close()
            await conn.writer.wait_closed()

    async def broadcast(self, message: bytes, exclude: str = None):
        for conn_id, conn in self.connections.items():
            if conn_id != exclude and conn.is_alive:
                await self.send(conn_id, message)

    async def send(self, conn_id: str, message: bytes):
        if conn := self.connections.get(conn_id):
            frame = create_frame(message, Opcode.TEXT)
            conn.writer.write(frame)
            await conn.writer.drain()

    async def join_room(self, conn_id: str, room: str):
        if conn := self.connections.get(conn_id):
            conn.rooms.add(room)
            self.rooms.setdefault(room, set()).add(conn_id)

    async def broadcast_to_room(self, room: str, message: bytes, exclude: str = None):
        for conn_id in self.rooms.get(room, set()):
            if conn_id != exclude:
                await self.send(conn_id, message)
```"""
                },
                "pitfalls": [
                    "Not handling disconnections during broadcast causes cascade failures",
                    "Memory leak from not cleaning up closed connections",
                    "Race conditions when modifying connection dict during iteration"
                ]
            },
            {
                "name": "Ping/Pong Heartbeat",
                "description": "Implement heartbeat mechanism using ping/pong frames to detect dead connections and maintain NAT mappings.",
                "hints": {
                    "level1": "Send ping frames periodically, expect pong responses.",
                    "level2": "Track last pong time, disconnect if no response within timeout.",
                    "level3": """```python
import asyncio
import time

class HeartbeatManager:
    def __init__(self, conn_manager: ConnectionManager,
                 ping_interval: float = 30.0,
                 pong_timeout: float = 10.0):
        self.conn_manager = conn_manager
        self.ping_interval = ping_interval
        self.pong_timeout = pong_timeout
        self.last_pong: dict[str, float] = {}
        self._task = None

    def start(self):
        self._task = asyncio.create_task(self._heartbeat_loop())

    def stop(self):
        if self._task:
            self._task.cancel()

    def record_pong(self, conn_id: str):
        self.last_pong[conn_id] = time.monotonic()

    async def _heartbeat_loop(self):
        while True:
            await asyncio.sleep(self.ping_interval)
            now = time.monotonic()
            dead_connections = []

            for conn_id, conn in self.conn_manager.connections.items():
                last = self.last_pong.get(conn_id, now)
                if now - last > self.ping_interval + self.pong_timeout:
                    dead_connections.append(conn_id)
                else:
                    # Send ping
                    ping_frame = create_frame(b'', Opcode.PING)
                    try:
                        conn.writer.write(ping_frame)
                        await conn.writer.drain()
                    except Exception:
                        dead_connections.append(conn_id)

            for conn_id in dead_connections:
                await self.conn_manager.remove(conn_id)
                self.last_pong.pop(conn_id, None)
```"""
                },
                "pitfalls": [
                    "Using wall clock time fails with system time changes",
                    "Too aggressive ping interval wastes bandwidth",
                    "Not initializing last_pong on connect causes immediate disconnect"
                ]
            }
        ]
    },

    "realtime-chat": {
        "name": "Real-time Chat System",
        "description": "Build a scalable real-time chat system with rooms, presence, message history, and horizontal scaling support.",
        "why_important": "Chat systems demonstrate core real-time architecture patterns used in Slack, Discord, and messaging apps.",
        "difficulty": "intermediate",
        "tags": ["real-time", "distributed-systems", "backend"],
        "estimated_hours": 40,
        "prerequisites": ["websocket-server", "redis-clone"],
        "learning_outcomes": [
            "Design scalable real-time message delivery",
            "Implement presence detection and tracking",
            "Handle message ordering and consistency",
            "Build pub/sub for horizontal scaling"
        ],
        "milestones": [
            {
                "name": "Message Routing",
                "description": "Implement message routing between users with direct messages, room broadcasts, and delivery confirmation.",
                "hints": {
                    "level1": "Route messages based on type: DM, room, or broadcast.",
                    "level2": "Use message IDs for delivery acknowledgment.",
                    "level3": """```python
import uuid
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    DIRECT = "direct"
    ROOM = "room"
    BROADCAST = "broadcast"
    SYSTEM = "system"

@dataclass
class ChatMessage:
    id: str
    type: MessageType
    sender_id: str
    content: str
    timestamp: datetime
    target: str = None  # user_id for DM, room_id for room
    metadata: dict = None

    @classmethod
    def create(cls, type: MessageType, sender_id: str, content: str, target: str = None):
        return cls(
            id=str(uuid.uuid4()),
            type=type,
            sender_id=sender_id,
            content=content,
            timestamp=datetime.utcnow(),
            target=target
        )

class MessageRouter:
    def __init__(self, conn_manager: ConnectionManager, history_store):
        self.conn_manager = conn_manager
        self.history = history_store
        self.pending_acks: dict[str, ChatMessage] = {}

    async def route(self, message: ChatMessage):
        await self.history.store(message)

        if message.type == MessageType.DIRECT:
            await self._route_direct(message)
        elif message.type == MessageType.ROOM:
            await self._route_room(message)
        elif message.type == MessageType.BROADCAST:
            await self._route_broadcast(message)

    async def _route_direct(self, message: ChatMessage):
        payload = json.dumps(asdict(message), default=str).encode()
        # Send to recipient
        await self.conn_manager.send(message.target, payload)
        # Send to sender (echo)
        await self.conn_manager.send(message.sender_id, payload)
        self.pending_acks[message.id] = message

    async def _route_room(self, message: ChatMessage):
        payload = json.dumps(asdict(message), default=str).encode()
        await self.conn_manager.broadcast_to_room(
            message.target, payload, exclude=message.sender_id
        )

    async def acknowledge(self, message_id: str, user_id: str):
        if msg := self.pending_acks.get(message_id):
            # Mark as delivered
            await self.history.mark_delivered(message_id, user_id)
```"""
                },
                "pitfalls": [
                    "Not echoing messages to sender causes UI desync",
                    "Missing delivery confirmation loses messages silently",
                    "JSON serialization fails with datetime objects"
                ]
            },
            {
                "name": "Presence System",
                "description": "Track user online/offline status with typing indicators, last seen timestamps, and efficient presence broadcasts.",
                "hints": {
                    "level1": "Update presence on connect/disconnect and activity.",
                    "level2": "Use heartbeats to detect ghost sessions.",
                    "level3": """```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio

class PresenceStatus(str, Enum):
    ONLINE = "online"
    AWAY = "away"
    OFFLINE = "offline"
    TYPING = "typing"

@dataclass
class UserPresence:
    user_id: str
    status: PresenceStatus
    last_active: datetime
    current_room: str = None

class PresenceManager:
    def __init__(self, conn_manager: ConnectionManager):
        self.conn_manager = conn_manager
        self.presence: dict[str, UserPresence] = {}
        self.typing_timers: dict[str, asyncio.Task] = {}
        self.away_threshold = timedelta(minutes=5)

    async def set_online(self, user_id: str):
        self.presence[user_id] = UserPresence(
            user_id=user_id,
            status=PresenceStatus.ONLINE,
            last_active=datetime.utcnow()
        )
        await self._broadcast_presence(user_id)

    async def set_offline(self, user_id: str):
        if user_id in self.presence:
            self.presence[user_id].status = PresenceStatus.OFFLINE
            await self._broadcast_presence(user_id)

    async def set_typing(self, user_id: str, room_id: str):
        if user_id in self.presence:
            self.presence[user_id].status = PresenceStatus.TYPING
            self.presence[user_id].current_room = room_id
            await self._broadcast_typing(user_id, room_id)

            # Cancel previous timer
            if user_id in self.typing_timers:
                self.typing_timers[user_id].cancel()

            # Auto-clear typing after 3 seconds
            self.typing_timers[user_id] = asyncio.create_task(
                self._clear_typing(user_id, 3.0)
            )

    async def _clear_typing(self, user_id: str, delay: float):
        await asyncio.sleep(delay)
        if p := self.presence.get(user_id):
            if p.status == PresenceStatus.TYPING:
                p.status = PresenceStatus.ONLINE
                await self._broadcast_presence(user_id)

    async def heartbeat(self, user_id: str):
        if p := self.presence.get(user_id):
            p.last_active = datetime.utcnow()
            if p.status == PresenceStatus.AWAY:
                p.status = PresenceStatus.ONLINE
                await self._broadcast_presence(user_id)

    def get_room_presence(self, room_id: str) -> list[UserPresence]:
        return [p for p in self.presence.values()
                if p.current_room == room_id and p.status != PresenceStatus.OFFLINE]
```"""
                },
                "pitfalls": [
                    "Typing indicator never clears if user disconnects while typing",
                    "Presence storms when many users join simultaneously",
                    "Race condition between away detection and activity"
                ]
            },
            {
                "name": "Message History & Sync",
                "description": "Store and retrieve message history with pagination, gap detection, and client sync protocol.",
                "hints": {
                    "level1": "Store messages with timestamps, allow range queries.",
                    "level2": "Use cursor-based pagination for efficient scrolling.",
                    "level3": """```python
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class MessagePage:
    messages: list[ChatMessage]
    has_more: bool
    cursor: str = None

class MessageHistory:
    def __init__(self, storage):
        self.storage = storage  # Redis, DB, etc.

    async def store(self, message: ChatMessage):
        key = self._get_key(message)
        score = message.timestamp.timestamp()
        await self.storage.zadd(key, {message.id: score})
        await self.storage.hset(f"msg:{message.id}", mapping=asdict(message))

    async def get_history(self, room_id: str, cursor: str = None,
                         limit: int = 50) -> MessagePage:
        key = f"room:{room_id}:messages"

        if cursor:
            # Get score of cursor message
            cursor_score = await self.storage.zscore(key, cursor)
            max_score = cursor_score - 0.001
        else:
            max_score = "+inf"

        # Get message IDs in reverse chronological order
        msg_ids = await self.storage.zrevrangebyscore(
            key, max_score, "-inf", start=0, num=limit + 1
        )

        has_more = len(msg_ids) > limit
        msg_ids = msg_ids[:limit]

        # Fetch full messages
        messages = []
        for msg_id in msg_ids:
            data = await self.storage.hgetall(f"msg:{msg_id}")
            if data:
                messages.append(ChatMessage(**data))

        return MessagePage(
            messages=messages,
            has_more=has_more,
            cursor=msg_ids[-1] if msg_ids else None
        )

    async def get_since(self, room_id: str, since_id: str) -> list[ChatMessage]:
        \"\"\"Get all messages after a given message ID for sync.\"\"\"
        key = f"room:{room_id}:messages"
        since_score = await self.storage.zscore(key, since_id)

        msg_ids = await self.storage.zrangebyscore(
            key, f"({since_score}", "+inf"
        )

        messages = []
        for msg_id in msg_ids:
            data = await self.storage.hgetall(f"msg:{msg_id}")
            if data:
                messages.append(ChatMessage(**data))
        return messages
```"""
                },
                "pitfalls": [
                    "Offset-based pagination skips messages when new ones arrive",
                    "Large gaps cause client to fetch too many messages at once",
                    "Not handling deleted messages in sync causes errors"
                ]
            },
            {
                "name": "Horizontal Scaling with Pub/Sub",
                "description": "Scale to multiple server instances using pub/sub for cross-instance message delivery.",
                "hints": {
                    "level1": "Publish messages to shared channel, subscribe on all instances.",
                    "level2": "Include instance ID to avoid echo, handle reconnection.",
                    "level3": """```python
import asyncio
import json
from dataclasses import dataclass
from typing import Callable
import uuid

@dataclass
class ClusterMessage:
    type: str
    source_instance: str
    payload: dict

class ClusterBroker:
    def __init__(self, redis_client, instance_id: str = None):
        self.redis = redis_client
        self.instance_id = instance_id or str(uuid.uuid4())[:8]
        self.handlers: dict[str, Callable] = {}
        self._subscriber = None
        self._task = None

    def on(self, message_type: str):
        def decorator(func):
            self.handlers[message_type] = func
            return func
        return decorator

    async def start(self):
        self._subscriber = self.redis.pubsub()
        await self._subscriber.subscribe("chat:cluster")
        self._task = asyncio.create_task(self._listen())

    async def stop(self):
        if self._task:
            self._task.cancel()
        if self._subscriber:
            await self._subscriber.unsubscribe()

    async def publish(self, message_type: str, payload: dict):
        msg = ClusterMessage(
            type=message_type,
            source_instance=self.instance_id,
            payload=payload
        )
        await self.redis.publish("chat:cluster", json.dumps(asdict(msg)))

    async def _listen(self):
        async for message in self._subscriber.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    cluster_msg = ClusterMessage(**data)

                    # Skip messages from self
                    if cluster_msg.source_instance == self.instance_id:
                        continue

                    if handler := self.handlers.get(cluster_msg.type):
                        await handler(cluster_msg.payload)
                except Exception as e:
                    print(f"Cluster message error: {e}")

# Usage in chat server
broker = ClusterBroker(redis_client)

@broker.on("chat_message")
async def handle_cluster_message(payload):
    message = ChatMessage(**payload)
    # Deliver to local connections only
    await local_router.deliver_local(message)

# When receiving a message from WebSocket
async def on_message(ws_message):
    message = parse_message(ws_message)
    # Store and publish to cluster
    await history.store(message)
    await broker.publish("chat_message", asdict(message))
    # Also deliver locally
    await local_router.deliver_local(message)
```"""
                },
                "pitfalls": [
                    "Message duplication when publishing to self",
                    "Lost messages during Redis reconnection",
                    "Thundering herd when all instances reconnect simultaneously"
                ]
            }
        ]
    },

    "multiplayer-game-server": {
        "name": "Multiplayer Game Server",
        "description": "Build a real-time multiplayer game server with authoritative state, client prediction, lag compensation, and anti-cheat.",
        "why_important": "Game servers push real-time systems to their limits with strict latency requirements, making this excellent practice for performance-critical code.",
        "difficulty": "advanced",
        "tags": ["real-time", "networking", "game-dev", "performance"],
        "estimated_hours": 60,
        "prerequisites": ["websocket-server", "realtime-chat"],
        "learning_outcomes": [
            "Implement authoritative server architecture",
            "Build client-side prediction and reconciliation",
            "Handle lag compensation for fair gameplay",
            "Design tick-based game loops"
        ],
        "milestones": [
            {
                "name": "Game Loop & Tick System",
                "description": "Implement a fixed timestep game loop that processes input, updates state, and broadcasts at consistent intervals.",
                "hints": {
                    "level1": "Use fixed delta time (e.g., 60 ticks/second) for deterministic simulation.",
                    "level2": "Accumulate time and process multiple ticks if behind.",
                    "level3": """```python
import asyncio
import time
from dataclasses import dataclass
from typing import Callable

@dataclass
class GameState:
    tick: int = 0
    entities: dict = None

    def __post_init__(self):
        self.entities = self.entities or {}

class GameLoop:
    def __init__(self, tick_rate: int = 60):
        self.tick_rate = tick_rate
        self.tick_duration = 1.0 / tick_rate
        self.state = GameState()
        self.input_buffer: dict[int, list] = {}  # tick -> [inputs]
        self.running = False
        self._update_callbacks: list[Callable] = []
        self._broadcast_callback: Callable = None

    def on_update(self, callback: Callable):
        self._update_callbacks.append(callback)

    def set_broadcast(self, callback: Callable):
        self._broadcast_callback = callback

    def queue_input(self, player_id: str, input_data: dict, client_tick: int):
        \"\"\"Queue player input for processing at specific tick.\"\"\"
        target_tick = max(client_tick, self.state.tick)
        self.input_buffer.setdefault(target_tick, []).append({
            'player_id': player_id,
            'input': input_data,
            'client_tick': client_tick
        })

    async def start(self):
        self.running = True
        last_time = time.perf_counter()
        accumulator = 0.0

        while self.running:
            current_time = time.perf_counter()
            frame_time = current_time - last_time
            last_time = current_time

            accumulator += frame_time

            # Process fixed timestep ticks
            while accumulator >= self.tick_duration:
                self._process_tick()
                accumulator -= self.tick_duration

            # Broadcast state
            if self._broadcast_callback:
                await self._broadcast_callback(self.state)

            # Sleep to maintain tick rate
            sleep_time = self.tick_duration - (time.perf_counter() - current_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def _process_tick(self):
        # Process inputs for this tick
        inputs = self.input_buffer.pop(self.state.tick, [])
        for input_data in inputs:
            self._apply_input(input_data)

        # Run update callbacks
        for callback in self._update_callbacks:
            callback(self.state, self.tick_duration)

        self.state.tick += 1

    def _apply_input(self, input_data: dict):
        player_id = input_data['player_id']
        if entity := self.state.entities.get(player_id):
            entity.apply_input(input_data['input'])
```"""
                },
                "pitfalls": [
                    "Variable delta time causes non-deterministic simulation",
                    "Sleeping exact tick duration ignores processing time",
                    "Unbounded accumulator causes spiral of death when behind"
                ]
            },
            {
                "name": "Client Prediction & Reconciliation",
                "description": "Implement client-side prediction for responsive controls with server reconciliation to correct mispredictions.",
                "hints": {
                    "level1": "Client predicts movement locally, server is authoritative.",
                    "level2": "Store input history, replay from last acknowledged tick on correction.",
                    "level3": """```python
from dataclasses import dataclass, field
from typing import Optional
import copy

@dataclass
class InputFrame:
    tick: int
    input_data: dict
    predicted_state: dict

@dataclass
class ClientPrediction:
    entity_id: str
    current_state: dict
    pending_inputs: list[InputFrame] = field(default_factory=list)
    last_server_tick: int = 0

    def apply_input(self, tick: int, input_data: dict, physics):
        \"\"\"Apply input locally and store for reconciliation.\"\"\"
        # Predict new state
        predicted = physics.simulate(self.current_state, input_data)

        self.pending_inputs.append(InputFrame(
            tick=tick,
            input_data=input_data,
            predicted_state=copy.deepcopy(predicted)
        ))

        self.current_state = predicted
        return predicted

    def reconcile(self, server_tick: int, server_state: dict, physics):
        \"\"\"Reconcile with server state, replay unacknowledged inputs.\"\"\"
        # Remove acknowledged inputs
        self.pending_inputs = [
            inp for inp in self.pending_inputs if inp.tick > server_tick
        ]
        self.last_server_tick = server_tick

        # Check if prediction matches
        if not self.pending_inputs:
            self.current_state = server_state
            return False  # No correction needed

        # Check first pending prediction against server
        first_pending = self.pending_inputs[0]
        if self._states_match(first_pending.predicted_state, server_state):
            return False  # Prediction was correct

        # Misprediction - replay from server state
        self.current_state = copy.deepcopy(server_state)
        for inp in self.pending_inputs:
            self.current_state = physics.simulate(
                self.current_state, inp.input_data
            )
            inp.predicted_state = copy.deepcopy(self.current_state)

        return True  # Correction applied

    def _states_match(self, a: dict, b: dict, tolerance: float = 0.01) -> bool:
        for key in ['x', 'y', 'z']:
            if abs(a.get(key, 0) - b.get(key, 0)) > tolerance:
                return False
        return True

# Server-side validation
class AuthoritativeServer:
    def __init__(self):
        self.player_states: dict[str, dict] = {}
        self.physics = PhysicsEngine()

    def process_input(self, player_id: str, input_data: dict, client_tick: int):
        state = self.player_states.get(player_id, {})

        # Validate input (anti-cheat)
        if not self._validate_input(input_data):
            return None  # Reject invalid input

        # Apply authoritative simulation
        new_state = self.physics.simulate(state, input_data)
        self.player_states[player_id] = new_state

        return {
            'tick': client_tick,
            'state': new_state
        }
```"""
                },
                "pitfalls": [
                    "Not storing enough input history causes rubber-banding",
                    "Exact float comparison fails due to precision",
                    "Overcorrecting causes jittery movement"
                ]
            },
            {
                "name": "Lag Compensation",
                "description": "Implement server-side lag compensation to make hit detection fair for high-latency players.",
                "hints": {
                    "level1": "Store history of game states, rewind to player's perceived time.",
                    "level2": "Interpolate between stored states for accurate reconstruction.",
                    "level3": """```python
from dataclasses import dataclass
from collections import deque
import copy

@dataclass
class StateSnapshot:
    tick: int
    timestamp: float
    entities: dict  # entity_id -> state

class LagCompensation:
    def __init__(self, history_duration: float = 1.0, tick_rate: int = 60):
        self.history: deque[StateSnapshot] = deque()
        self.history_duration = history_duration
        self.tick_duration = 1.0 / tick_rate
        self.max_history = int(history_duration * tick_rate)

    def record(self, tick: int, timestamp: float, entities: dict):
        snapshot = StateSnapshot(
            tick=tick,
            timestamp=timestamp,
            entities=copy.deepcopy(entities)
        )
        self.history.append(snapshot)

        # Prune old history
        while len(self.history) > self.max_history:
            self.history.popleft()

    def get_state_at_time(self, target_time: float) -> dict:
        \"\"\"Get interpolated entity states at specific time.\"\"\"
        if not self.history:
            return {}

        # Find surrounding snapshots
        before = None
        after = None

        for snapshot in self.history:
            if snapshot.timestamp <= target_time:
                before = snapshot
            else:
                after = snapshot
                break

        if before is None:
            return self.history[0].entities
        if after is None:
            return self.history[-1].entities

        # Interpolate between snapshots
        t = (target_time - before.timestamp) / (after.timestamp - before.timestamp)
        return self._interpolate_entities(before.entities, after.entities, t)

    def _interpolate_entities(self, before: dict, after: dict, t: float) -> dict:
        result = {}
        for entity_id in before:
            if entity_id in after:
                result[entity_id] = self._interpolate_state(
                    before[entity_id], after[entity_id], t
                )
        return result

    def _interpolate_state(self, a: dict, b: dict, t: float) -> dict:
        return {
            'x': a['x'] + (b['x'] - a['x']) * t,
            'y': a['y'] + (b['y'] - a['y']) * t,
            'z': a.get('z', 0) + (b.get('z', 0) - a.get('z', 0)) * t,
        }

class HitDetection:
    def __init__(self, lag_comp: LagCompensation):
        self.lag_comp = lag_comp

    def check_hit(self, shooter_id: str, target_id: str,
                  shot_origin: dict, shot_direction: dict,
                  client_timestamp: float, max_rewind: float = 0.2):
        \"\"\"Check if shot hits with lag compensation.\"\"\"
        # Limit rewind to prevent abuse
        current_time = time.time()
        rewind_time = min(
            current_time - client_timestamp,
            max_rewind
        )

        # Get world state at shooter's perceived time
        perceived_time = current_time - rewind_time
        world_state = self.lag_comp.get_state_at_time(perceived_time)

        if target_id not in world_state:
            return False

        target_state = world_state[target_id]
        return self._ray_intersects_hitbox(
            shot_origin, shot_direction, target_state
        )
```"""
                },
                "pitfalls": [
                    "Unlimited rewind allows shooting around corners",
                    "Memory grows unbounded without history pruning",
                    "Interpolation fails at entity spawn/despawn boundaries"
                ]
            },
            {
                "name": "State Synchronization",
                "description": "Efficiently synchronize game state to clients using delta compression, interest management, and prioritization.",
                "hints": {
                    "level1": "Send only changed values, not full state.",
                    "level2": "Prioritize nearby entities, reduce updates for distant ones.",
                    "level3": """```python
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class EntityPriority:
    entity_id: str
    priority: float
    last_update_tick: int

class StateSynchronizer:
    def __init__(self, tick_rate: int = 60):
        self.tick_rate = tick_rate
        self.client_states: dict[str, dict] = {}  # client_id -> last sent state per entity
        self.update_rates: dict[float, int] = {
            1.0: 1,    # Priority 1.0: every tick
            0.5: 2,    # Priority 0.5: every 2 ticks
            0.25: 4,   # Priority 0.25: every 4 ticks
            0.1: 10,   # Priority 0.1: every 10 ticks
        }

    def calculate_priority(self, viewer_state: dict, entity_state: dict,
                          entity_type: str) -> float:
        # Distance-based priority
        dx = viewer_state['x'] - entity_state['x']
        dy = viewer_state['y'] - entity_state['y']
        distance = math.sqrt(dx*dx + dy*dy)

        # Base priority from distance
        if distance < 10:
            priority = 1.0
        elif distance < 50:
            priority = 0.5
        elif distance < 100:
            priority = 0.25
        else:
            priority = 0.1

        # Boost for important entity types
        if entity_type == 'player':
            priority = min(1.0, priority * 1.5)
        elif entity_type == 'projectile':
            priority = min(1.0, priority * 2.0)

        return priority

    def generate_update(self, client_id: str, viewer_id: str,
                       current_tick: int, world_state: dict) -> bytes:
        client_last = self.client_states.setdefault(client_id, {})
        viewer_state = world_state.get(viewer_id, {'x': 0, 'y': 0})

        updates = []

        for entity_id, entity_state in world_state.items():
            priority = self.calculate_priority(
                viewer_state, entity_state, entity_state.get('type', 'generic')
            )

            # Check if should update this tick based on priority
            update_interval = self._get_update_interval(priority)
            if current_tick % update_interval != 0:
                continue

            # Delta compression
            last_state = client_last.get(entity_id, {})
            delta = self._compute_delta(last_state, entity_state)

            if delta:
                updates.append({
                    'id': entity_id,
                    'delta': delta,
                    'full': len(delta) > len(entity_state) // 2
                })
                client_last[entity_id] = entity_state.copy()

        return self._encode_updates(updates)

    def _compute_delta(self, old: dict, new: dict) -> dict:
        delta = {}
        for key, value in new.items():
            if key not in old or old[key] != value:
                delta[key] = value
        return delta

    def _get_update_interval(self, priority: float) -> int:
        for p, interval in sorted(self.update_rates.items(), reverse=True):
            if priority >= p:
                return interval
        return 10  # Default slow rate
```"""
                },
                "pitfalls": [
                    "Sending full state every tick overwhelms bandwidth",
                    "Delta without baseline causes desync",
                    "Priority calculation per entity per client is O(n*m)"
                ]
            }
        ]
    },

    "collaborative-editor": {
        "name": "Collaborative Text Editor",
        "description": "Build a real-time collaborative text editor using CRDTs or Operational Transformation for conflict-free concurrent editing.",
        "why_important": "Collaborative editing is used in Google Docs, Notion, Figma. Understanding CRDTs and OT is valuable for any real-time collaborative application.",
        "difficulty": "advanced",
        "tags": ["distributed-systems", "real-time", "algorithms"],
        "estimated_hours": 50,
        "prerequisites": ["realtime-chat"],
        "learning_outcomes": [
            "Implement Operational Transformation or CRDTs",
            "Handle concurrent edits without conflicts",
            "Build cursor presence and selection sync",
            "Design efficient document synchronization"
        ],
        "milestones": [
            {
                "name": "Operation-based CRDT",
                "description": "Implement a sequence CRDT (like RGA or YATA) for conflict-free text insertion and deletion.",
                "hints": {
                    "level1": "Assign unique IDs to each character, order by ID for consistent view.",
                    "level2": "Use Lamport timestamps + site ID for globally unique, orderable IDs.",
                    "level3": """```python
from dataclasses import dataclass, field
from typing import Optional
import bisect

@dataclass(frozen=True, order=True)
class CharId:
    timestamp: int
    site_id: str
    seq: int = 0

    def __str__(self):
        return f"{self.timestamp}.{self.site_id}.{self.seq}"

@dataclass
class Char:
    id: CharId
    value: str
    deleted: bool = False

@dataclass
class RGADocument:
    \"\"\"Replicated Growable Array - a sequence CRDT.\"\"\"
    site_id: str
    chars: list[Char] = field(default_factory=list)
    timestamp: int = 0

    def _next_id(self) -> CharId:
        self.timestamp += 1
        return CharId(self.timestamp, self.site_id)

    def _find_position(self, after_id: Optional[CharId]) -> int:
        if after_id is None:
            return 0
        for i, char in enumerate(self.chars):
            if char.id == after_id:
                return i + 1
        return len(self.chars)

    def _visible_index_to_position(self, index: int) -> int:
        \"\"\"Convert visible text index to internal position.\"\"\"
        visible = 0
        for i, char in enumerate(self.chars):
            if not char.deleted:
                if visible == index:
                    return i
                visible += 1
        return len(self.chars)

    def insert(self, index: int, value: str) -> dict:
        \"\"\"Local insert at visible index.\"\"\"
        pos = self._visible_index_to_position(index)
        after_id = self.chars[pos - 1].id if pos > 0 else None

        char_id = self._next_id()
        char = Char(id=char_id, value=value)

        # Find correct position respecting ID ordering
        insert_pos = pos
        while insert_pos < len(self.chars):
            if self.chars[insert_pos].id < char_id:
                break
            insert_pos += 1

        self.chars.insert(insert_pos, char)

        return {
            'type': 'insert',
            'id': char_id,
            'after': after_id,
            'value': value
        }

    def delete(self, index: int) -> dict:
        \"\"\"Local delete at visible index (tombstone).\"\"\"
        pos = self._visible_index_to_position(index)
        self.chars[pos].deleted = True

        return {
            'type': 'delete',
            'id': self.chars[pos].id
        }

    def apply_remote(self, op: dict):
        \"\"\"Apply operation from remote site.\"\"\"
        if op['type'] == 'insert':
            self._apply_remote_insert(op)
        elif op['type'] == 'delete':
            self._apply_remote_delete(op)

        # Update local timestamp
        self.timestamp = max(self.timestamp, op['id'].timestamp)

    def _apply_remote_insert(self, op: dict):
        char = Char(id=op['id'], value=op['value'])
        pos = self._find_position(op['after'])

        # Find correct position respecting ID ordering
        while pos < len(self.chars):
            if self.chars[pos].id < char.id:
                break
            pos += 1

        self.chars.insert(pos, char)

    def _apply_remote_delete(self, op: dict):
        for char in self.chars:
            if char.id == op['id']:
                char.deleted = True
                break

    def get_text(self) -> str:
        return ''.join(c.value for c in self.chars if not c.deleted)
```"""
                },
                "pitfalls": [
                    "Timestamp collision when sites have same clock",
                    "Memory grows unbounded with tombstones",
                    "O(n) lookup for every operation is slow on large docs"
                ]
            },
            {
                "name": "Cursor Presence",
                "description": "Synchronize cursor positions and selections across all connected editors with colored indicators.",
                "hints": {
                    "level1": "Track cursor as (position, selection_start, selection_end) per user.",
                    "level2": "Transform cursor positions when local/remote edits occur.",
                    "level3": """```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CursorState:
    user_id: str
    position: int
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None
    color: str = "#3498db"
    name: str = "Anonymous"

class CursorManager:
    def __init__(self, document):
        self.document = document
        self.cursors: dict[str, CursorState] = {}

    def update_local(self, user_id: str, position: int,
                    selection: tuple[int, int] = None) -> dict:
        cursor = self.cursors.setdefault(user_id, CursorState(user_id, 0))
        cursor.position = position
        if selection:
            cursor.selection_start, cursor.selection_end = selection
        else:
            cursor.selection_start = cursor.selection_end = None

        return {
            'type': 'cursor',
            'user_id': user_id,
            'position': position,
            'selection': selection
        }

    def apply_remote_cursor(self, op: dict):
        user_id = op['user_id']
        cursor = self.cursors.setdefault(user_id, CursorState(user_id, 0))
        cursor.position = op['position']
        if op.get('selection'):
            cursor.selection_start, cursor.selection_end = op['selection']

    def transform_on_insert(self, insert_pos: int, length: int,
                           author_id: str):
        \"\"\"Transform all cursors after an insert operation.\"\"\"
        for user_id, cursor in self.cursors.items():
            # Don't transform author's cursor (they handle it locally)
            if user_id == author_id:
                continue

            if cursor.position >= insert_pos:
                cursor.position += length

            if cursor.selection_start is not None:
                if cursor.selection_start >= insert_pos:
                    cursor.selection_start += length
                if cursor.selection_end >= insert_pos:
                    cursor.selection_end += length

    def transform_on_delete(self, delete_pos: int, length: int,
                           author_id: str):
        \"\"\"Transform all cursors after a delete operation.\"\"\"
        delete_end = delete_pos + length

        for user_id, cursor in self.cursors.items():
            if user_id == author_id:
                continue

            # Cursor after deleted region
            if cursor.position >= delete_end:
                cursor.position -= length
            # Cursor within deleted region
            elif cursor.position > delete_pos:
                cursor.position = delete_pos

            # Transform selection
            if cursor.selection_start is not None:
                cursor.selection_start = self._transform_point(
                    cursor.selection_start, delete_pos, delete_end, length
                )
                cursor.selection_end = self._transform_point(
                    cursor.selection_end, delete_pos, delete_end, length
                )

    def _transform_point(self, point: int, del_start: int,
                        del_end: int, length: int) -> int:
        if point >= del_end:
            return point - length
        elif point > del_start:
            return del_start
        return point
```"""
                },
                "pitfalls": [
                    "Cursor flickers when transforming on every keystroke",
                    "Selection can become inverted (end < start) after transform",
                    "Not handling cursor at document boundaries"
                ]
            },
            {
                "name": "Operational Transformation",
                "description": "Implement OT as alternative to CRDTs, with transform functions for insert/delete operations.",
                "hints": {
                    "level1": "Transform concurrent operations so they can be applied in any order.",
                    "level2": "Use server to serialize operations and determine canonical order.",
                    "level3": """```python
from dataclasses import dataclass
from typing import Union
from enum import Enum

class OpType(Enum):
    INSERT = "insert"
    DELETE = "delete"

@dataclass
class Operation:
    type: OpType
    position: int
    text: str = ""  # For insert
    length: int = 0  # For delete
    version: int = 0

def transform(op1: Operation, op2: Operation) -> Operation:
    \"\"\"Transform op1 against op2 (op2 was applied first).\"\"\"
    if op1.type == OpType.INSERT and op2.type == OpType.INSERT:
        return transform_insert_insert(op1, op2)
    elif op1.type == OpType.INSERT and op2.type == OpType.DELETE:
        return transform_insert_delete(op1, op2)
    elif op1.type == OpType.DELETE and op2.type == OpType.INSERT:
        return transform_delete_insert(op1, op2)
    else:
        return transform_delete_delete(op1, op2)

def transform_insert_insert(op1: Operation, op2: Operation) -> Operation:
    if op1.position <= op2.position:
        return op1  # No transformation needed
    else:
        return Operation(
            type=OpType.INSERT,
            position=op1.position + len(op2.text),
            text=op1.text,
            version=op1.version
        )

def transform_insert_delete(op1: Operation, op2: Operation) -> Operation:
    if op1.position <= op2.position:
        return op1
    elif op1.position >= op2.position + op2.length:
        return Operation(
            type=OpType.INSERT,
            position=op1.position - op2.length,
            text=op1.text,
            version=op1.version
        )
    else:
        # Insert position was in deleted region
        return Operation(
            type=OpType.INSERT,
            position=op2.position,
            text=op1.text,
            version=op1.version
        )

def transform_delete_insert(op1: Operation, op2: Operation) -> Operation:
    if op2.position >= op1.position + op1.length:
        return op1
    elif op2.position <= op1.position:
        return Operation(
            type=OpType.DELETE,
            position=op1.position + len(op2.text),
            length=op1.length,
            version=op1.version
        )
    else:
        # Insert splits the delete
        # Delete text before insert, then after
        return Operation(
            type=OpType.DELETE,
            position=op1.position,
            length=op1.length + len(op2.text),
            version=op1.version
        )

def transform_delete_delete(op1: Operation, op2: Operation) -> Operation:
    # Complex case: overlapping deletes
    end1 = op1.position + op1.length
    end2 = op2.position + op2.length

    if end1 <= op2.position:
        return op1  # op1 entirely before op2
    elif op1.position >= end2:
        return Operation(
            type=OpType.DELETE,
            position=op1.position - op2.length,
            length=op1.length,
            version=op1.version
        )
    else:
        # Overlapping deletes
        new_pos = min(op1.position, op2.position)
        # Calculate remaining length after op2's deletion
        overlap_start = max(op1.position, op2.position)
        overlap_end = min(end1, end2)
        overlap = max(0, overlap_end - overlap_start)
        new_length = op1.length - overlap

        if new_length <= 0:
            return None  # op1 is completely covered by op2

        return Operation(
            type=OpType.DELETE,
            position=new_pos,
            length=new_length,
            version=op1.version
        )

class OTServer:
    def __init__(self):
        self.document = ""
        self.version = 0
        self.history: list[Operation] = []

    def apply(self, op: Operation) -> Operation:
        # Transform against all operations since client's version
        transformed = op
        for historical in self.history[op.version:]:
            transformed = transform(transformed, historical)
            if transformed is None:
                return None

        # Apply to document
        if transformed.type == OpType.INSERT:
            self.document = (
                self.document[:transformed.position] +
                transformed.text +
                self.document[transformed.position:]
            )
        else:
            self.document = (
                self.document[:transformed.position] +
                self.document[transformed.position + transformed.length:]
            )

        transformed.version = self.version
        self.history.append(transformed)
        self.version += 1

        return transformed
```"""
                },
                "pitfalls": [
                    "Transform functions must be composable: T(T(a,b),c) = T(a,T(b,c))",
                    "Overlapping delete transforms are notoriously tricky",
                    "History grows unbounded without garbage collection"
                ]
            },
            {
                "name": "Undo/Redo with Collaboration",
                "description": "Implement undo/redo that works correctly with concurrent edits from multiple users.",
                "hints": {
                    "level1": "Each user has their own undo stack of their operations.",
                    "level2": "Undoing means inverting the operation and transforming against subsequent ops.",
                    "level3": """```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class UndoManager:
    user_id: str
    undo_stack: list[Operation] = field(default_factory=list)
    redo_stack: list[Operation] = field(default_factory=list)
    document: 'OTDocument' = None

    def record(self, op: Operation):
        \"\"\"Record an operation for potential undo.\"\"\"
        self.undo_stack.append(op)
        self.redo_stack.clear()  # Clear redo on new action

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def undo(self) -> Optional[Operation]:
        if not self.can_undo():
            return None

        op = self.undo_stack.pop()

        # Create inverse operation
        inverse = self._invert(op)

        # Transform inverse against all operations that happened after op
        transformed = self._transform_against_history(inverse, op.version)

        if transformed:
            self.redo_stack.append(op)
            return transformed
        return None

    def redo(self) -> Optional[Operation]:
        if not self.can_redo():
            return None

        op = self.redo_stack.pop()

        # Transform the original operation against subsequent history
        transformed = self._transform_against_history(op, op.version)

        if transformed:
            self.undo_stack.append(op)
            return transformed
        return None

    def _invert(self, op: Operation) -> Operation:
        \"\"\"Create the inverse of an operation.\"\"\"
        if op.type == OpType.INSERT:
            return Operation(
                type=OpType.DELETE,
                position=op.position,
                length=len(op.text),
                version=self.document.version
            )
        else:
            # For delete, we need the deleted text
            # This requires storing deleted text with operation
            return Operation(
                type=OpType.INSERT,
                position=op.position,
                text=op.deleted_text,  # Need to store this
                version=self.document.version
            )

    def _transform_against_history(self, op: Operation,
                                   since_version: int) -> Optional[Operation]:
        transformed = op
        for historical in self.document.history[since_version:]:
            # Skip our own operations (they're in undo stack)
            if historical.user_id == self.user_id:
                continue
            transformed = transform(transformed, historical)
            if transformed is None:
                return None
        return transformed

    def transform_stacks_on_remote(self, remote_op: Operation):
        \"\"\"Transform all stacks when a remote operation arrives.\"\"\"
        self.undo_stack = [
            transform(op, remote_op) for op in self.undo_stack
            if transform(op, remote_op) is not None
        ]
        self.redo_stack = [
            transform(op, remote_op) for op in self.redo_stack
            if transform(op, remote_op) is not None
        ]
```"""
                },
                "pitfalls": [
                    "Undoing a delete requires storing the deleted text",
                    "Undo stack becomes invalid if user disconnects/reconnects",
                    "Group related operations (e.g., typing) into single undo unit"
                ]
            }
        ]
    },

    "event-sourcing": {
        "name": "Event Sourcing System",
        "description": "Build an event-sourced system with event store, projections, snapshots, and CQRS pattern for complex domain modeling.",
        "why_important": "Event sourcing provides audit trails, temporal queries, and robust distributed systems. Used in banking, e-commerce, and enterprise systems.",
        "difficulty": "advanced",
        "tags": ["distributed-systems", "architecture", "backend"],
        "estimated_hours": 45,
        "prerequisites": ["rest-api-design"],
        "learning_outcomes": [
            "Implement append-only event store",
            "Build projections from event streams",
            "Handle snapshots for performance",
            "Apply CQRS for read/write separation"
        ],
        "milestones": [
            {
                "name": "Event Store",
                "description": "Build an append-only event store with optimistic concurrency, stream versioning, and event metadata.",
                "hints": {
                    "level1": "Store events with stream ID, version, timestamp, and payload.",
                    "level2": "Use expected version for optimistic concurrency control.",
                    "level3": """```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import json
import uuid

@dataclass
class Event:
    id: str
    stream_id: str
    version: int
    type: str
    data: dict
    metadata: dict
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(cls, stream_id: str, version: int, event_type: str,
               data: dict, metadata: dict = None):
        return cls(
            id=str(uuid.uuid4()),
            stream_id=stream_id,
            version=version,
            type=event_type,
            data=data,
            metadata=metadata or {}
        )

class ConcurrencyError(Exception):
    pass

class EventStore:
    def __init__(self, storage):
        self.storage = storage  # Could be PostgreSQL, EventStoreDB, etc.

    async def append(self, stream_id: str, events: list[Event],
                    expected_version: int = None) -> int:
        \"\"\"
        Append events to stream with optimistic concurrency.
        Returns new stream version.
        \"\"\"
        async with self.storage.transaction() as tx:
            # Get current version
            current_version = await self._get_stream_version(tx, stream_id)

            # Check expected version for concurrency
            if expected_version is not None:
                if current_version != expected_version:
                    raise ConcurrencyError(
                        f"Expected version {expected_version}, "
                        f"but stream is at {current_version}"
                    )

            # Append events
            new_version = current_version
            for event in events:
                new_version += 1
                event.version = new_version
                await self._store_event(tx, event)

            return new_version

    async def read_stream(self, stream_id: str,
                         from_version: int = 0,
                         to_version: int = None) -> list[Event]:
        query = \"\"\"
            SELECT * FROM events
            WHERE stream_id = $1 AND version >= $2
        \"\"\"
        params = [stream_id, from_version]

        if to_version is not None:
            query += " AND version <= $3"
            params.append(to_version)

        query += " ORDER BY version ASC"

        rows = await self.storage.fetch(query, *params)
        return [self._row_to_event(row) for row in rows]

    async def read_all(self, from_position: int = 0,
                      batch_size: int = 1000) -> list[Event]:
        \"\"\"Read all events across all streams (for projections).\"\"\"
        rows = await self.storage.fetch(
            \"\"\"
            SELECT * FROM events
            WHERE global_position > $1
            ORDER BY global_position ASC
            LIMIT $2
            \"\"\",
            from_position, batch_size
        )
        return [self._row_to_event(row) for row in rows]

    async def _get_stream_version(self, tx, stream_id: str) -> int:
        result = await tx.fetchval(
            "SELECT COALESCE(MAX(version), 0) FROM events WHERE stream_id = $1",
            stream_id
        )
        return result

    async def _store_event(self, tx, event: Event):
        await tx.execute(
            \"\"\"
            INSERT INTO events (id, stream_id, version, type, data, metadata, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            \"\"\",
            event.id, event.stream_id, event.version, event.type,
            json.dumps(event.data), json.dumps(event.metadata), event.timestamp
        )
```"""
                },
                "pitfalls": [
                    "Gap in versions makes stream unreadable",
                    "Large events bloat storage without compression",
                    "Missing global ordering breaks cross-stream projections"
                ]
            },
            {
                "name": "Aggregate & Event Sourcing",
                "description": "Implement domain aggregates that are reconstituted from event history with command handlers.",
                "hints": {
                    "level1": "Aggregate state is built by replaying events in order.",
                    "level2": "Commands validate business rules, produce events.",
                    "level3": """```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic

T = TypeVar('T')

class Aggregate(ABC, Generic[T]):
    def __init__(self):
        self.id: str = None
        self.version: int = 0
        self._changes: list[Event] = []

    @abstractmethod
    def apply(self, event: Event):
        \"\"\"Apply event to update aggregate state.\"\"\"
        pass

    def load(self, events: list[Event]):
        \"\"\"Reconstitute aggregate from event history.\"\"\"
        for event in events:
            self.apply(event)
            self.version = event.version

    def _raise_event(self, event_type: str, data: dict):
        \"\"\"Record new event (to be persisted).\"\"\"
        event = Event.create(
            stream_id=self.id,
            version=self.version + len(self._changes) + 1,
            event_type=event_type,
            data=data
        )
        self._changes.append(event)
        self.apply(event)

    def get_changes(self) -> list[Event]:
        return self._changes

    def clear_changes(self):
        self._changes = []

# Example: Order aggregate
@dataclass
class OrderItem:
    product_id: str
    quantity: int
    price: float

class Order(Aggregate):
    def __init__(self):
        super().__init__()
        self.customer_id: str = None
        self.items: list[OrderItem] = []
        self.status: str = "draft"
        self.total: float = 0

    # Commands
    def create(self, order_id: str, customer_id: str):
        if self.id is not None:
            raise ValueError("Order already exists")
        self._raise_event("OrderCreated", {
            "order_id": order_id,
            "customer_id": customer_id
        })

    def add_item(self, product_id: str, quantity: int, price: float):
        if self.status != "draft":
            raise ValueError("Cannot modify non-draft order")
        self._raise_event("ItemAdded", {
            "product_id": product_id,
            "quantity": quantity,
            "price": price
        })

    def submit(self):
        if self.status != "draft":
            raise ValueError("Order already submitted")
        if not self.items:
            raise ValueError("Cannot submit empty order")
        self._raise_event("OrderSubmitted", {
            "total": self.total
        })

    # Event handlers
    def apply(self, event: Event):
        if event.type == "OrderCreated":
            self.id = event.data["order_id"]
            self.customer_id = event.data["customer_id"]
        elif event.type == "ItemAdded":
            item = OrderItem(
                product_id=event.data["product_id"],
                quantity=event.data["quantity"],
                price=event.data["price"]
            )
            self.items.append(item)
            self.total += item.quantity * item.price
        elif event.type == "OrderSubmitted":
            self.status = "submitted"

class AggregateRepository:
    def __init__(self, event_store: EventStore, aggregate_type: type):
        self.event_store = event_store
        self.aggregate_type = aggregate_type

    async def load(self, aggregate_id: str) -> Aggregate:
        events = await self.event_store.read_stream(aggregate_id)
        aggregate = self.aggregate_type()
        aggregate.load(events)
        return aggregate

    async def save(self, aggregate: Aggregate):
        changes = aggregate.get_changes()
        if changes:
            await self.event_store.append(
                aggregate.id,
                changes,
                expected_version=aggregate.version
            )
            aggregate.clear_changes()
```"""
                },
                "pitfalls": [
                    "Business logic in apply() instead of command handlers",
                    "Forgetting to clear changes after save causes duplicates",
                    "Mutable event data allows accidental modification"
                ]
            },
            {
                "name": "Projections",
                "description": "Build read models (projections) that update in response to events for efficient querying.",
                "hints": {
                    "level1": "Subscribe to events, update denormalized read model.",
                    "level2": "Track checkpoint to resume from last processed position.",
                    "level3": """```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio

@dataclass
class Checkpoint:
    projection_name: str
    position: int

class Projection(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def handle(self, event: Event):
        \"\"\"Handle a single event.\"\"\"
        pass

    @abstractmethod
    def handles(self) -> list[str]:
        \"\"\"Return list of event types this projection handles.\"\"\"
        pass

class ProjectionEngine:
    def __init__(self, event_store: EventStore, checkpoint_store):
        self.event_store = event_store
        self.checkpoint_store = checkpoint_store
        self.projections: list[Projection] = []
        self._running = False

    def register(self, projection: Projection):
        self.projections.append(projection)

    async def start(self):
        self._running = True
        while self._running:
            await self._process_batch()
            await asyncio.sleep(0.1)  # Poll interval

    async def stop(self):
        self._running = False

    async def _process_batch(self, batch_size: int = 100):
        for projection in self.projections:
            checkpoint = await self.checkpoint_store.get(projection.name)
            position = checkpoint.position if checkpoint else 0

            events = await self.event_store.read_all(
                from_position=position,
                batch_size=batch_size
            )

            for event in events:
                if event.type in projection.handles():
                    await projection.handle(event)
                position = event.global_position

            if events:
                await self.checkpoint_store.save(
                    Checkpoint(projection.name, position)
                )

# Example projection: Order summary read model
class OrderSummaryProjection(Projection):
    def __init__(self, db):
        super().__init__("order_summary")
        self.db = db

    def handles(self) -> list[str]:
        return ["OrderCreated", "ItemAdded", "OrderSubmitted"]

    async def handle(self, event: Event):
        if event.type == "OrderCreated":
            await self.db.execute(
                \"\"\"
                INSERT INTO order_summaries (id, customer_id, status, total, item_count)
                VALUES ($1, $2, 'draft', 0, 0)
                \"\"\",
                event.data["order_id"], event.data["customer_id"]
            )
        elif event.type == "ItemAdded":
            await self.db.execute(
                \"\"\"
                UPDATE order_summaries
                SET total = total + $1, item_count = item_count + 1
                WHERE id = $2
                \"\"\",
                event.data["quantity"] * event.data["price"],
                event.stream_id
            )
        elif event.type == "OrderSubmitted":
            await self.db.execute(
                \"\"\"
                UPDATE order_summaries SET status = 'submitted' WHERE id = $1
                \"\"\",
                event.stream_id
            )

# Projection for customer analytics
class CustomerAnalyticsProjection(Projection):
    def __init__(self, db):
        super().__init__("customer_analytics")
        self.db = db

    def handles(self) -> list[str]:
        return ["OrderSubmitted"]

    async def handle(self, event: Event):
        # Update customer lifetime value
        await self.db.execute(
            \"\"\"
            INSERT INTO customer_stats (customer_id, order_count, total_spent)
            VALUES ($1, 1, $2)
            ON CONFLICT (customer_id) DO UPDATE
            SET order_count = customer_stats.order_count + 1,
                total_spent = customer_stats.total_spent + $2
            \"\"\",
            event.metadata.get("customer_id"),
            event.data["total"]
        )
```"""
                },
                "pitfalls": [
                    "Processing same event twice without idempotency corrupts data",
                    "Checkpoint saved before event processed loses events on crash",
                    "Slow projection blocks all others in single-threaded engine"
                ]
            },
            {
                "name": "Snapshots",
                "description": "Implement aggregate snapshots to avoid replaying entire event history for performance.",
                "hints": {
                    "level1": "Periodically save aggregate state, load snapshot + events since.",
                    "level2": "Store snapshot version to know which events to replay.",
                    "level3": """```python
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class Snapshot:
    aggregate_id: str
    aggregate_type: str
    version: int
    state: dict

class SnapshotStore:
    def __init__(self, storage):
        self.storage = storage

    async def save(self, snapshot: Snapshot):
        await self.storage.execute(
            \"\"\"
            INSERT INTO snapshots (aggregate_id, aggregate_type, version, state)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (aggregate_id) DO UPDATE
            SET version = $3, state = $4
            WHERE snapshots.version < $3
            \"\"\",
            snapshot.aggregate_id, snapshot.aggregate_type,
            snapshot.version, json.dumps(snapshot.state)
        )

    async def load(self, aggregate_id: str) -> Optional[Snapshot]:
        row = await self.storage.fetchrow(
            "SELECT * FROM snapshots WHERE aggregate_id = $1",
            aggregate_id
        )
        if row:
            return Snapshot(
                aggregate_id=row["aggregate_id"],
                aggregate_type=row["aggregate_type"],
                version=row["version"],
                state=json.loads(row["state"])
            )
        return None

class SnapshottingRepository:
    def __init__(self, event_store: EventStore,
                 snapshot_store: SnapshotStore,
                 aggregate_type: type,
                 snapshot_frequency: int = 100):
        self.event_store = event_store
        self.snapshot_store = snapshot_store
        self.aggregate_type = aggregate_type
        self.snapshot_frequency = snapshot_frequency

    async def load(self, aggregate_id: str) -> Aggregate:
        aggregate = self.aggregate_type()

        # Try to load snapshot
        snapshot = await self.snapshot_store.load(aggregate_id)

        if snapshot:
            # Restore from snapshot
            aggregate.restore_from_snapshot(snapshot.state)
            aggregate.version = snapshot.version

            # Load events since snapshot
            events = await self.event_store.read_stream(
                aggregate_id,
                from_version=snapshot.version + 1
            )
        else:
            # Load all events
            events = await self.event_store.read_stream(aggregate_id)

        aggregate.load(events)
        return aggregate

    async def save(self, aggregate: Aggregate):
        changes = aggregate.get_changes()
        if not changes:
            return

        await self.event_store.append(
            aggregate.id,
            changes,
            expected_version=aggregate.version
        )

        new_version = aggregate.version + len(changes)
        aggregate.clear_changes()

        # Check if we should create a snapshot
        if new_version % self.snapshot_frequency == 0:
            await self._create_snapshot(aggregate, new_version)

    async def _create_snapshot(self, aggregate: Aggregate, version: int):
        snapshot = Snapshot(
            aggregate_id=aggregate.id,
            aggregate_type=type(aggregate).__name__,
            version=version,
            state=aggregate.get_snapshot_state()
        )
        await self.snapshot_store.save(snapshot)

# Add to Order aggregate
class Order(Aggregate):
    # ... previous code ...

    def get_snapshot_state(self) -> dict:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "items": [
                {"product_id": i.product_id, "quantity": i.quantity, "price": i.price}
                for i in self.items
            ],
            "status": self.status,
            "total": self.total
        }

    def restore_from_snapshot(self, state: dict):
        self.id = state["id"]
        self.customer_id = state["customer_id"]
        self.items = [
            OrderItem(**item) for item in state["items"]
        ]
        self.status = state["status"]
        self.total = state["total"]
```"""
                },
                "pitfalls": [
                    "Snapshot schema changes break deserialization",
                    "Creating snapshot on every save destroys performance",
                    "Snapshot without version causes event replay from beginning"
                ]
            }
        ]
    }
}

# Load and update YAML
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

expert_projects = data.get('expert_projects', {})

for project_id, project_data in realtime_projects.items():
    if project_id not in expert_projects:
        expert_projects[project_id] = project_data
        print(f"Added: {project_id}")
    else:
        print(f"Skipped (exists): {project_id}")

# Save
with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nTotal expert_projects: {len(expert_projects)}")

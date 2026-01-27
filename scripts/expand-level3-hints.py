#!/usr/bin/env python3
"""
Expand level3 hints that are too short (< 100 chars) with proper code examples.
Based on real-world implementations and best practices.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

# Expanded level3 hints - keyed by (project_id, milestone_name)
expanded_hints = {
    # build-git
    ("build-git", "Tree Objects"): """# Tree object format and implementation
def write_tree(directory):
    entries = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if name == '.git':
            continue
        if os.path.isfile(path):
            mode = b'100644'
            sha = hash_object(open(path, 'rb').read(), 'blob')
        else:
            mode = b'40000'
            sha = write_tree(path)  # Recurse for subdirectories
        entries.append((mode, name.encode(), bytes.fromhex(sha)))

    # Build tree content: mode + space + name + null + 20-byte SHA
    data = b''
    for mode, name, sha_bytes in entries:
        data += mode + b' ' + name + b'\\x00' + sha_bytes

    return hash_object(data, 'tree')

def ls_tree(tree_sha):
    content = read_object(tree_sha)
    entries = []
    while content:
        # Find space after mode
        space_idx = content.index(b' ')
        mode = content[:space_idx].decode()
        content = content[space_idx + 1:]
        # Find null after name
        null_idx = content.index(b'\\x00')
        name = content[:null_idx].decode()
        content = content[null_idx + 1:]
        # Next 20 bytes are SHA
        sha = content[:20].hex()
        content = content[20:]
        entries.append((mode, sha, name))
    return entries""",

    # build-interpreter
    ("build-interpreter", "Scanner (Lexer)"): """class Scanner:
    def __init__(self, source):
        self.source = source
        self.tokens = []
        self.start = 0
        self.current = 0
        self.line = 1

    def scan_tokens(self):
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        self.tokens.append(Token(TokenType.EOF, "", None, self.line))
        return self.tokens

    def scan_token(self):
        c = self.advance()
        match c:
            case '(': self.add_token(TokenType.LEFT_PAREN)
            case ')': self.add_token(TokenType.RIGHT_PAREN)
            case '-': self.add_token(TokenType.MINUS)
            case '+': self.add_token(TokenType.PLUS)
            case '!':
                self.add_token(TokenType.BANG_EQUAL if self.match('=') else TokenType.BANG)
            case '=':
                self.add_token(TokenType.EQUAL_EQUAL if self.match('=') else TokenType.EQUAL)
            case '<':
                self.add_token(TokenType.LESS_EQUAL if self.match('=') else TokenType.LESS)
            case '"': self.string()
            case c if c.isdigit(): self.number()
            case c if c.isalpha() or c == '_': self.identifier()
            case ' ' | '\\r' | '\\t': pass
            case '\\n': self.line += 1
            case _: self.error(f"Unexpected character: {c}")

    def match(self, expected):
        if self.is_at_end() or self.source[self.current] != expected:
            return False
        self.current += 1
        return True""",

    ("build-interpreter", "Parsing Expressions"): """# Recursive descent parser with precedence handling
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0

    def expression(self):
        return self.equality()

    def equality(self):  # == !=
        expr = self.comparison()
        while self.match(TokenType.BANG_EQUAL, TokenType.EQUAL_EQUAL):
            operator = self.previous()
            right = self.comparison()
            expr = Binary(expr, operator, right)
        return expr

    def comparison(self):  # > >= < <=
        expr = self.term()
        while self.match(TokenType.GREATER, TokenType.GREATER_EQUAL,
                         TokenType.LESS, TokenType.LESS_EQUAL):
            operator = self.previous()
            right = self.term()
            expr = Binary(expr, operator, right)
        return expr

    def term(self):  # + -
        expr = self.factor()
        while self.match(TokenType.MINUS, TokenType.PLUS):
            operator = self.previous()
            right = self.factor()
            expr = Binary(expr, operator, right)
        return expr

    def factor(self):  # * /
        expr = self.unary()
        while self.match(TokenType.SLASH, TokenType.STAR):
            operator = self.previous()
            right = self.unary()
            expr = Binary(expr, operator, right)
        return expr

    def unary(self):  # ! -
        if self.match(TokenType.BANG, TokenType.MINUS):
            operator = self.previous()
            right = self.unary()
            return Unary(operator, right)
        return self.primary()

    def primary(self):
        if self.match(TokenType.FALSE): return Literal(False)
        if self.match(TokenType.TRUE): return Literal(True)
        if self.match(TokenType.NIL): return Literal(None)
        if self.match(TokenType.NUMBER, TokenType.STRING):
            return Literal(self.previous().literal)
        if self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression.")
            return Grouping(expr)
        raise self.error(self.peek(), "Expect expression.")""",

    ("build-interpreter", "Statements and State"): """# Environment for variable storage
class Environment:
    def __init__(self, enclosing=None):
        self.values = {}
        self.enclosing = enclosing  # For nested scopes

    def define(self, name, value):
        self.values[name] = value

    def get(self, name):
        if name.lexeme in self.values:
            return self.values[name.lexeme]
        if self.enclosing:
            return self.enclosing.get(name)
        raise RuntimeError(f"Undefined variable '{name.lexeme}'.")

    def assign(self, name, value):
        if name.lexeme in self.values:
            self.values[name.lexeme] = value
            return
        if self.enclosing:
            self.enclosing.assign(name, value)
            return
        raise RuntimeError(f"Undefined variable '{name.lexeme}'.")

# Parsing statements
def statement(self):
    if self.match(TokenType.PRINT):
        return self.print_statement()
    if self.match(TokenType.VAR):
        return self.var_declaration()
    return self.expression_statement()

def var_declaration(self):
    name = self.consume(TokenType.IDENTIFIER, "Expect variable name.")
    initializer = None
    if self.match(TokenType.EQUAL):
        initializer = self.expression()
    self.consume(TokenType.SEMICOLON, "Expect ';' after variable declaration.")
    return Var(name, initializer)""",

    ("build-interpreter", "Control Flow"): """# Control flow implementation
class Interpreter:
    def visit_if_stmt(self, stmt):
        if self.is_truthy(self.evaluate(stmt.condition)):
            self.execute(stmt.then_branch)
        elif stmt.else_branch:
            self.execute(stmt.else_branch)

    def visit_while_stmt(self, stmt):
        while self.is_truthy(self.evaluate(stmt.condition)):
            self.execute(stmt.body)

    def visit_logical_expr(self, expr):
        left = self.evaluate(expr.left)
        # Short-circuit evaluation
        if expr.operator.type == TokenType.OR:
            if self.is_truthy(left):
                return left
        else:  # AND
            if not self.is_truthy(left):
                return left
        return self.evaluate(expr.right)

# For loop desugaring in parser:
def for_statement(self):
    self.consume(TokenType.LEFT_PAREN, "Expect '(' after 'for'.")

    initializer = None if self.match(TokenType.SEMICOLON) else \\
                  self.var_declaration() if self.match(TokenType.VAR) else \\
                  self.expression_statement()

    condition = Literal(True) if self.check(TokenType.SEMICOLON) else self.expression()
    self.consume(TokenType.SEMICOLON, "Expect ';' after loop condition.")

    increment = None if self.check(TokenType.RIGHT_PAREN) else self.expression()
    self.consume(TokenType.RIGHT_PAREN, "Expect ')' after for clauses.")

    body = self.statement()

    # Desugar to while loop
    if increment:
        body = Block([body, Expression(increment)])
    body = While(condition, body)
    if initializer:
        body = Block([initializer, body])
    return body""",

    ("build-interpreter", "Functions"): """# Function declaration and calling
class LoxFunction:
    def __init__(self, declaration, closure):
        self.declaration = declaration
        self.closure = closure  # Environment where function was defined

    def call(self, interpreter, arguments):
        # Create new environment for function scope
        environment = Environment(self.closure)

        # Bind parameters to arguments
        for i, param in enumerate(self.declaration.params):
            environment.define(param.lexeme, arguments[i])

        try:
            interpreter.execute_block(self.declaration.body, environment)
        except Return as return_value:
            return return_value.value
        return None

    def arity(self):
        return len(self.declaration.params)

class Return(Exception):
    def __init__(self, value):
        super().__init__()
        self.value = value

# In interpreter:
def visit_return_stmt(self, stmt):
    value = None
    if stmt.value:
        value = self.evaluate(stmt.value)
    raise Return(value)

def visit_call_expr(self, expr):
    callee = self.evaluate(expr.callee)
    arguments = [self.evaluate(arg) for arg in expr.arguments]

    if not hasattr(callee, 'call'):
        raise RuntimeError("Can only call functions and classes.")
    if len(arguments) != callee.arity():
        raise RuntimeError(f"Expected {callee.arity()} arguments but got {len(arguments)}.")

    return callee.call(self, arguments)""",

    ("build-interpreter", "Classes"): """# Class and instance implementation
class LoxClass:
    def __init__(self, name, superclass, methods):
        self.name = name
        self.superclass = superclass
        self.methods = methods

    def call(self, interpreter, arguments):
        instance = LoxInstance(self)
        # Call initializer if present
        initializer = self.find_method("init")
        if initializer:
            initializer.bind(instance).call(interpreter, arguments)
        return instance

    def arity(self):
        initializer = self.find_method("init")
        return 0 if not initializer else initializer.arity()

    def find_method(self, name):
        if name in self.methods:
            return self.methods[name]
        if self.superclass:
            return self.superclass.find_method(name)
        return None

class LoxInstance:
    def __init__(self, klass):
        self.klass = klass
        self.fields = {}

    def get(self, name):
        if name.lexeme in self.fields:
            return self.fields[name.lexeme]
        method = self.klass.find_method(name.lexeme)
        if method:
            return method.bind(self)  # Bind 'this'
        raise RuntimeError(f"Undefined property '{name.lexeme}'.")

    def set(self, name, value):
        self.fields[name.lexeme] = value

# Bind 'this' to method
def bind(self, instance):
    environment = Environment(self.closure)
    environment.define("this", instance)
    return LoxFunction(self.declaration, environment)""",

    # build-raytracer
    ("build-raytracer", "Output an Image"): """# PPM image output - simple format for raytracing
class Color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __add__(self, other):
        return Color(self.r + other.r, self.g + other.g, self.b + other.b)

    def __mul__(self, scalar):
        return Color(self.r * scalar, self.g * scalar, self.b * scalar)

    def clamp(self):
        return Color(
            max(0.0, min(1.0, self.r)),
            max(0.0, min(1.0, self.g)),
            max(0.0, min(1.0, self.b))
        )

    def to_ppm(self, samples_per_pixel=1):
        scale = 1.0 / samples_per_pixel
        r = int(256 * max(0.0, min(0.999, self.r * scale)))
        g = int(256 * max(0.0, min(0.999, self.g * scale)))
        b = int(256 * max(0.0, min(0.999, self.b * scale)))
        return f"{r} {g} {b}"

def write_ppm(filename, width, height, pixels):
    with open(filename, 'w') as f:
        f.write(f"P3\\n{width} {height}\\n255\\n")
        for row in pixels:
            for color in row:
                f.write(color.to_ppm() + "\\n")

# Example: gradient image
width, height = 256, 256
pixels = []
for j in range(height):
    row = []
    for i in range(width):
        r = i / (width - 1)
        g = (height - 1 - j) / (height - 1)
        b = 0.25
        row.append(Color(r, g, b))
    pixels.append(row)
write_ppm("gradient.ppm", width, height, pixels)""",

    ("build-raytracer", "Surface Normals and Multiple Objects"): """# Hit detection with surface normals
import math
from dataclasses import dataclass

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __neg__(self): return Vec3(-self.x, -self.y, -self.z)
    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, t): return Vec3(self.x*t, self.y*t, self.z*t)
    def dot(self, o): return self.x*o.x + self.y*o.y + self.z*o.z
    def length(self): return math.sqrt(self.dot(self))
    def unit(self): return self * (1.0 / self.length())

@dataclass
class HitRecord:
    p: Vec3          # Hit point
    normal: Vec3     # Surface normal (always outward)
    t: float         # Ray parameter
    front_face: bool # Are we hitting front or back?

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray, t_min, t_max):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        half_b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = half_b*half_b - a*c

        if discriminant < 0:
            return None

        sqrtd = math.sqrt(discriminant)
        root = (-half_b - sqrtd) / a
        if root < t_min or root > t_max:
            root = (-half_b + sqrtd) / a
            if root < t_min or root > t_max:
                return None

        p = ray.at(root)
        outward_normal = (p - self.center) * (1.0/self.radius)
        front_face = ray.direction.dot(outward_normal) < 0
        normal = outward_normal if front_face else -outward_normal

        return HitRecord(p, normal, root, front_face)

class HittableList:
    def __init__(self):
        self.objects = []

    def hit(self, ray, t_min, t_max):
        closest = None
        closest_t = t_max
        for obj in self.objects:
            rec = obj.hit(ray, t_min, closest_t)
            if rec:
                closest = rec
                closest_t = rec.t
        return closest""",

    # build-redis
    ("build-redis", "GET/SET/DEL Commands"): """# Redis-like command implementation
class RedisStore:
    def __init__(self):
        self.data = {}
        self.expiry = {}  # key -> expiry timestamp

    def set(self, key, value, ex=None, px=None, nx=False, xx=False):
        # NX: only set if key doesn't exist
        if nx and key in self.data:
            return None
        # XX: only set if key exists
        if xx and key not in self.data:
            return None

        self.data[key] = value

        if ex:  # Expire in seconds
            self.expiry[key] = time.time() + ex
        elif px:  # Expire in milliseconds
            self.expiry[key] = time.time() + px / 1000.0
        elif key in self.expiry:
            del self.expiry[key]

        return "OK"

    def get(self, key):
        if not self._check_expiry(key):
            return None
        return self.data.get(key)

    def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                self.expiry.pop(key, None)
                count += 1
        return count

    def _check_expiry(self, key):
        if key in self.expiry:
            if time.time() >= self.expiry[key]:
                del self.data[key]
                del self.expiry[key]
                return False
        return key in self.data

# RESP protocol parser
def parse_resp(data):
    if data[0:1] == b'+':  # Simple string
        return data[1:data.index(b'\\r\\n')].decode()
    elif data[0:1] == b'-':  # Error
        return Exception(data[1:data.index(b'\\r\\n')].decode())
    elif data[0:1] == b':':  # Integer
        return int(data[1:data.index(b'\\r\\n')])
    elif data[0:1] == b'$':  # Bulk string
        length = int(data[1:data.index(b'\\r\\n')])
        if length == -1:
            return None
        start = data.index(b'\\r\\n') + 2
        return data[start:start+length]
    elif data[0:1] == b'*':  # Array
        count = int(data[1:data.index(b'\\r\\n')])
        # Recursively parse elements...
        pass""",

    ("build-redis", "Data Structures (List, Set, Hash)"): """# Redis data structure implementations
class RedisList:
    def __init__(self):
        self.items = []

    def lpush(self, *values):
        for v in reversed(values):
            self.items.insert(0, v)
        return len(self.items)

    def rpush(self, *values):
        self.items.extend(values)
        return len(self.items)

    def lpop(self, count=1):
        result = self.items[:count]
        self.items = self.items[count:]
        return result[0] if count == 1 else result

    def lrange(self, start, stop):
        # Redis uses inclusive stop, Python uses exclusive
        if stop < 0:
            stop = len(self.items) + stop + 1
        else:
            stop += 1
        return self.items[start:stop]

class RedisSet:
    def __init__(self):
        self.members = set()

    def sadd(self, *members):
        added = sum(1 for m in members if m not in self.members)
        self.members.update(members)
        return added

    def srem(self, *members):
        removed = sum(1 for m in members if m in self.members)
        self.members -= set(members)
        return removed

    def sismember(self, member):
        return 1 if member in self.members else 0

    def smembers(self):
        return list(self.members)

class RedisHash:
    def __init__(self):
        self.fields = {}

    def hset(self, field, value):
        is_new = field not in self.fields
        self.fields[field] = value
        return 1 if is_new else 0

    def hget(self, field):
        return self.fields.get(field)

    def hgetall(self):
        result = []
        for k, v in self.fields.items():
            result.extend([k, v])
        return result

    def hincrby(self, field, increment):
        self.fields[field] = int(self.fields.get(field, 0)) + increment
        return self.fields[field]""",

    ("build-redis", "Pub/Sub"): """# Redis Pub/Sub implementation
import asyncio
from collections import defaultdict

class PubSubManager:
    def __init__(self):
        self.subscribers = defaultdict(set)  # channel -> set of clients
        self.patterns = defaultdict(set)      # pattern -> set of clients

    def subscribe(self, client, *channels):
        for channel in channels:
            self.subscribers[channel].add(client)
            client.send_message(['subscribe', channel, len(client.subscriptions)])
            client.subscriptions.add(channel)

    def unsubscribe(self, client, *channels):
        if not channels:
            channels = list(client.subscriptions)
        for channel in channels:
            self.subscribers[channel].discard(client)
            client.subscriptions.discard(channel)
            client.send_message(['unsubscribe', channel, len(client.subscriptions)])

    def publish(self, channel, message):
        count = 0
        # Direct subscribers
        for client in self.subscribers[channel]:
            client.send_message(['message', channel, message])
            count += 1
        # Pattern subscribers
        for pattern, clients in self.patterns.items():
            if self._match_pattern(pattern, channel):
                for client in clients:
                    client.send_message(['pmessage', pattern, channel, message])
                    count += 1
        return count

    def _match_pattern(self, pattern, channel):
        # Simple glob matching: * matches any sequence
        import fnmatch
        return fnmatch.fnmatch(channel, pattern)

# Async client handling for pub/sub
class RedisClient:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.subscriptions = set()
        self.message_queue = asyncio.Queue()

    async def handle_pubsub_mode(self):
        # In pub/sub mode, client can only send SUBSCRIBE/UNSUBSCRIBE
        while self.subscriptions:
            msg = await self.message_queue.get()
            self.writer.write(encode_resp_array(msg))
            await self.writer.drain()""",

    ("build-redis", "Cluster Mode (Sharding)"): """# Redis Cluster sharding implementation
import hashlib

def crc16(data):
    '''CRC16 XMODEM - used by Redis Cluster'''
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc

def key_slot(key):
    '''Calculate hash slot for a key (0-16383)'''
    # Handle hash tags: {tag}key uses only "tag" for hashing
    start = key.find(b'{')
    if start >= 0:
        end = key.find(b'}', start + 1)
        if end > start + 1:
            key = key[start + 1:end]
    return crc16(key) % 16384

class ClusterNode:
    def __init__(self, node_id, host, port):
        self.id = node_id
        self.host = host
        self.port = port
        self.slots = set()  # Set of slot numbers this node owns
        self.replicas = []  # Replica nodes

class Cluster:
    def __init__(self):
        self.nodes = {}  # node_id -> ClusterNode
        self.slot_map = [None] * 16384  # slot -> node_id

    def assign_slots(self, node_id, start_slot, end_slot):
        node = self.nodes[node_id]
        for slot in range(start_slot, end_slot + 1):
            self.slot_map[slot] = node_id
            node.slots.add(slot)

    def get_node_for_key(self, key):
        slot = key_slot(key if isinstance(key, bytes) else key.encode())
        node_id = self.slot_map[slot]
        return self.nodes[node_id]

    def handle_moved(self, slot, new_node):
        '''Handle MOVED response during resharding'''
        return f"MOVED {slot} {new_node.host}:{new_node.port}"

# Client-side routing
class ClusterClient:
    def __init__(self, startup_nodes):
        self.cluster = Cluster()
        self.refresh_slots(startup_nodes)

    def execute(self, *args):
        key = args[1]  # Most commands have key as second arg
        node = self.cluster.get_node_for_key(key)
        try:
            return node.execute(*args)
        except MovedError as e:
            self.refresh_slots()
            return self.execute(*args)""",

    # build-shell
    ("build-shell", "Built-in Commands"): """# Shell built-in commands implementation
import os
import sys

class Shell:
    def __init__(self):
        self.builtins = {
            'cd': self.builtin_cd,
            'pwd': self.builtin_pwd,
            'export': self.builtin_export,
            'exit': self.builtin_exit,
            'echo': self.builtin_echo,
            'type': self.builtin_type,
            'history': self.builtin_history,
            'alias': self.builtin_alias,
        }
        self.aliases = {}
        self.history = []
        self.env = dict(os.environ)

    def builtin_cd(self, args):
        if not args:
            path = self.env.get('HOME', '/')
        elif args[0] == '-':
            path = self.env.get('OLDPWD', '.')
        elif args[0].startswith('~'):
            path = os.path.expanduser(args[0])
        else:
            path = args[0]

        try:
            old_pwd = os.getcwd()
            os.chdir(path)
            self.env['OLDPWD'] = old_pwd
            self.env['PWD'] = os.getcwd()
            return 0
        except FileNotFoundError:
            print(f"cd: {path}: No such file or directory", file=sys.stderr)
            return 1

    def builtin_export(self, args):
        for arg in args:
            if '=' in arg:
                name, value = arg.split('=', 1)
                self.env[name] = value
                os.environ[name] = value
            else:
                # Export existing variable
                if arg in self.env:
                    os.environ[arg] = self.env[arg]
        return 0

    def builtin_type(self, args):
        for name in args:
            if name in self.builtins:
                print(f"{name} is a shell builtin")
            elif name in self.aliases:
                print(f"{name} is aliased to `{self.aliases[name]}'")
            else:
                path = self.find_executable(name)
                if path:
                    print(f"{name} is {path}")
                else:
                    print(f"{name}: not found")
                    return 1
        return 0

    def find_executable(self, name):
        for dir in self.env.get('PATH', '').split(':'):
            path = os.path.join(dir, name)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None""",

    ("build-shell", "Background Jobs"): """# Job control for background processes
import os
import signal
from enum import Enum

class JobState(Enum):
    RUNNING = "Running"
    STOPPED = "Stopped"
    DONE = "Done"

class Job:
    def __init__(self, job_id, pgid, command, processes):
        self.id = job_id
        self.pgid = pgid  # Process group ID
        self.command = command
        self.processes = processes  # List of PIDs
        self.state = JobState.RUNNING

    def __str__(self):
        return f"[{self.id}]  {self.state.value}  {self.command}"

class JobController:
    def __init__(self):
        self.jobs = {}
        self.next_id = 1
        self.foreground_pgid = None
        signal.signal(signal.SIGCHLD, self.sigchld_handler)

    def launch_job(self, command, background=False):
        pid = os.fork()
        if pid == 0:
            # Child: create new process group
            os.setpgid(0, 0)
            if not background:
                # Give terminal to foreground job
                os.tcsetpgrp(0, os.getpgrp())
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTSTP, signal.SIG_DFL)
            os.execvp(command[0], command)
        else:
            # Parent
            pgid = pid
            os.setpgid(pid, pgid)
            job = Job(self.next_id, pgid, ' '.join(command), [pid])
            self.jobs[self.next_id] = job
            self.next_id += 1

            if background:
                print(f"[{job.id}] {pid}")
            else:
                self.foreground_pgid = pgid
                os.tcsetpgrp(0, pgid)
                self.wait_for_job(job)
                os.tcsetpgrp(0, os.getpgrp())

    def wait_for_job(self, job):
        while job.state == JobState.RUNNING:
            pid, status = os.waitpid(-job.pgid, os.WUNTRACED)
            if os.WIFSTOPPED(status):
                job.state = JobState.STOPPED
                print(f"\\n{job}")
            elif os.WIFEXITED(status) or os.WIFSIGNALED(status):
                job.state = JobState.DONE

    def builtin_fg(self, args):
        job_id = int(args[0]) if args else max(self.jobs.keys())
        job = self.jobs.get(job_id)
        if job:
            print(job.command)
            os.killpg(job.pgid, signal.SIGCONT)
            job.state = JobState.RUNNING
            os.tcsetpgrp(0, job.pgid)
            self.wait_for_job(job)
            os.tcsetpgrp(0, os.getpgrp())

    def builtin_bg(self, args):
        job_id = int(args[0]) if args else max(self.jobs.keys())
        job = self.jobs.get(job_id)
        if job and job.state == JobState.STOPPED:
            os.killpg(job.pgid, signal.SIGCONT)
            job.state = JobState.RUNNING
            print(f"[{job.id}]+ {job.command} &")""",

    # build-sqlite
    ("build-sqlite", "B-tree Page Format"): """# SQLite-like B-tree page format
import struct
from enum import IntEnum

class PageType(IntEnum):
    INTERIOR_INDEX = 2
    INTERIOR_TABLE = 5
    LEAF_INDEX = 10
    LEAF_TABLE = 13

class BTreePage:
    '''
    Page format (4096 bytes default):
    - Header (8-12 bytes)
    - Cell pointer array
    - Unallocated space
    - Cell content area (grows from end)

    Header format:
    - 1 byte: page type
    - 2 bytes: first freeblock offset (0 if none)
    - 2 bytes: number of cells
    - 2 bytes: start of cell content area
    - 1 byte: fragmented free bytes
    - 4 bytes: right-most pointer (interior pages only)
    '''

    def __init__(self, page_num, page_type=PageType.LEAF_TABLE):
        self.page_num = page_num
        self.page_type = page_type
        self.cells = []
        self.right_ptr = None  # For interior pages
        self.page_size = 4096

    def serialize(self):
        # Build cell content from the end
        cell_data = b''
        cell_pointers = []
        content_start = self.page_size

        for cell in self.cells:
            cell_bytes = cell.serialize()
            content_start -= len(cell_bytes)
            cell_data = cell_bytes + cell_data
            cell_pointers.append(content_start)

        # Header
        header_size = 12 if self.page_type in (PageType.INTERIOR_INDEX, PageType.INTERIOR_TABLE) else 8
        header = struct.pack('>BHHHB',
            self.page_type,
            0,  # first freeblock
            len(self.cells),
            content_start,
            0   # fragmented bytes
        )
        if header_size == 12:
            header += struct.pack('>I', self.right_ptr or 0)

        # Cell pointers
        pointers = b''.join(struct.pack('>H', p) for p in cell_pointers)

        # Assemble page
        used = header_size + len(pointers)
        unallocated = self.page_size - used - len(cell_data)
        return header + pointers + (b'\\x00' * unallocated) + cell_data

    @classmethod
    def deserialize(cls, page_num, data):
        page_type = data[0]
        num_cells = struct.unpack('>H', data[3:5])[0]

        page = cls(page_num, PageType(page_type))
        header_size = 12 if page_type in (2, 5) else 8

        # Read cell pointers
        for i in range(num_cells):
            offset = struct.unpack('>H', data[header_size + i*2:header_size + i*2 + 2])[0]
            # Parse cell at offset...
        return page""",

    ("build-sqlite", "SELECT Execution (Table Scan)"): """# SELECT query execution with table scan
from dataclasses import dataclass
from typing import List, Any, Optional

@dataclass
class Column:
    name: str
    type: str
    primary_key: bool = False

@dataclass
class Table:
    name: str
    columns: List[Column]
    root_page: int

class QueryExecutor:
    def __init__(self, pager, schema):
        self.pager = pager
        self.schema = schema

    def execute_select(self, stmt):
        table = self.schema.get_table(stmt.table_name)
        if not table:
            raise RuntimeError(f"no such table: {stmt.table_name}")

        # Full table scan
        results = []
        for row in self.scan_table(table):
            # Apply WHERE filter
            if stmt.where_clause:
                if not self.evaluate_where(row, stmt.where_clause, table):
                    continue

            # Project columns
            if stmt.columns == ['*']:
                results.append(row)
            else:
                projected = []
                for col_name in stmt.columns:
                    idx = self.get_column_index(table, col_name)
                    projected.append(row[idx])
                results.append(tuple(projected))

        return results

    def scan_table(self, table):
        '''Iterate all rows via B-tree traversal'''
        def scan_page(page_num):
            page = self.pager.get_page(page_num)

            if page.is_leaf():
                for cell in page.cells:
                    yield cell.payload  # (rowid, col1, col2, ...)
            else:
                # Interior page: traverse children
                for cell in page.cells:
                    yield from scan_page(cell.left_child)
                if page.right_ptr:
                    yield from scan_page(page.right_ptr)

        yield from scan_page(table.root_page)

    def evaluate_where(self, row, where, table):
        '''Evaluate WHERE clause against a row'''
        if where.op == 'AND':
            return (self.evaluate_where(row, where.left, table) and
                    self.evaluate_where(row, where.right, table))
        elif where.op == 'OR':
            return (self.evaluate_where(row, where.left, table) or
                    self.evaluate_where(row, where.right, table))
        else:
            # Comparison: column op value
            col_idx = self.get_column_index(table, where.column)
            col_value = row[col_idx]

            if where.op == '=':
                return col_value == where.value
            elif where.op == '<':
                return col_value < where.value
            elif where.op == '>':
                return col_value > where.value
            # ... other operators""",

    ("build-sqlite", "INSERT/UPDATE/DELETE"): """# Data modification operations
class QueryExecutor:
    def execute_insert(self, stmt):
        table = self.schema.get_table(stmt.table_name)

        # Get next rowid
        rowid = self.get_next_rowid(table)

        # Build record
        values = [rowid]
        for i, col in enumerate(table.columns):
            if col.name in stmt.columns:
                idx = stmt.columns.index(col.name)
                values.append(stmt.values[idx])
            else:
                values.append(None)  # Default value

        # Insert into B-tree
        record = self.encode_record(values)
        self.btree_insert(table.root_page, rowid, record)

        return rowid

    def execute_update(self, stmt):
        table = self.schema.get_table(stmt.table_name)
        updated_count = 0

        # Scan and update matching rows
        for row in self.scan_table(table):
            if stmt.where_clause and not self.evaluate_where(row, stmt.where_clause, table):
                continue

            rowid = row[0]
            new_values = list(row)

            # Apply SET clauses
            for col_name, new_value in stmt.set_clauses:
                idx = self.get_column_index(table, col_name)
                new_values[idx] = new_value

            # Update in B-tree (delete + insert)
            self.btree_delete(table.root_page, rowid)
            record = self.encode_record(new_values)
            self.btree_insert(table.root_page, rowid, record)
            updated_count += 1

        return updated_count

    def execute_delete(self, stmt):
        table = self.schema.get_table(stmt.table_name)
        deleted_count = 0

        # Collect rowids to delete (can't modify during scan)
        to_delete = []
        for row in self.scan_table(table):
            if stmt.where_clause and not self.evaluate_where(row, stmt.where_clause, table):
                continue
            to_delete.append(row[0])  # rowid

        # Delete from B-tree
        for rowid in to_delete:
            self.btree_delete(table.root_page, rowid)
            deleted_count += 1

        return deleted_count

    def encode_record(self, values):
        '''SQLite record format: header + body'''
        header = []
        body = b''
        for val in values:
            if val is None:
                header.append(0)
            elif isinstance(val, int):
                if val == 0: header.append(8)
                elif val == 1: header.append(9)
                else: header.append(1); body += struct.pack('>b', val)
            elif isinstance(val, str):
                encoded = val.encode('utf-8')
                header.append(13 + len(encoded) * 2)
                body += encoded
        return self.encode_varint_header(header) + body""",

    ("build-sqlite", "WHERE Clause and Indexes"): """# Index-based query optimization
from bisect import bisect_left

class Index:
    def __init__(self, name, table_name, columns, root_page):
        self.name = name
        self.table_name = table_name
        self.columns = columns  # List of column names
        self.root_page = root_page

class QueryOptimizer:
    def __init__(self, schema, executor):
        self.schema = schema
        self.executor = executor

    def optimize_select(self, stmt):
        '''Choose best execution strategy'''
        table = self.schema.get_table(stmt.table_name)

        if not stmt.where_clause:
            return TableScan(table)

        # Look for usable indexes
        index = self.find_usable_index(stmt.where_clause, table)
        if index:
            return IndexScan(table, index, stmt.where_clause)

        return TableScan(table)

    def find_usable_index(self, where, table):
        '''Find index that covers WHERE clause'''
        if where.op in ('=', '<', '>', '<=', '>='):
            for idx in self.schema.get_indexes(table.name):
                if idx.columns[0] == where.column:
                    return idx
        return None

class IndexScan:
    def __init__(self, table, index, where):
        self.table = table
        self.index = index
        self.where = where

    def execute(self, executor):
        '''Use index to find matching rowids, then fetch rows'''
        if self.where.op == '=':
            # Point lookup
            rowids = executor.index_lookup(self.index, self.where.value)
        elif self.where.op in ('<', '<='):
            # Range scan from start
            rowids = executor.index_range(self.index, None, self.where.value,
                                         include_end=(self.where.op == '<='))
        elif self.where.op in ('>', '>='):
            # Range scan to end
            rowids = executor.index_range(self.index, self.where.value, None,
                                         include_start=(self.where.op == '>='))

        # Fetch actual rows
        for rowid in rowids:
            yield executor.fetch_row(self.table, rowid)

class QueryExecutor:
    def index_lookup(self, index, key):
        '''B-tree lookup for exact match'''
        page = self.pager.get_page(index.root_page)
        while True:
            if page.is_leaf():
                for cell in page.cells:
                    if cell.key == key:
                        yield cell.rowid
                return
            else:
                # Find child to descend
                child_page = self.find_child(page, key)
                page = self.pager.get_page(child_page)""",

    ("build-sqlite", "Query Planner"): """# Query planner with cost estimation
from dataclasses import dataclass
from typing import List
import math

@dataclass
class Plan:
    cost: float
    rows: int
    description: str

class QueryPlanner:
    def __init__(self, schema, stats):
        self.schema = schema
        self.stats = stats  # Table statistics

    def plan_select(self, stmt):
        table = self.schema.get_table(stmt.table_name)
        table_stats = self.stats.get(table.name)

        plans = []

        # Option 1: Full table scan
        scan_cost = table_stats.row_count * table_stats.avg_row_size
        plans.append(Plan(
            cost=scan_cost,
            rows=self.estimate_rows(stmt.where_clause, table_stats),
            description=f"SCAN TABLE {table.name}"
        ))

        # Option 2: Index scans
        for index in self.schema.get_indexes(table.name):
            usability = self.analyze_index_usability(index, stmt.where_clause)
            if usability:
                idx_stats = self.stats.get_index(index.name)

                # Cost = index lookup + row fetches
                selectivity = self.estimate_selectivity(usability, idx_stats)
                index_cost = math.log2(idx_stats.entries) + (selectivity * table_stats.row_count)

                plans.append(Plan(
                    cost=index_cost,
                    rows=int(selectivity * table_stats.row_count),
                    description=f"SEARCH {table.name} USING INDEX {index.name}"
                ))

        # Choose lowest cost plan
        return min(plans, key=lambda p: p.cost)

    def estimate_selectivity(self, condition, stats):
        '''Estimate fraction of rows matching condition'''
        if condition.op == '=':
            # Assume uniform distribution
            return 1.0 / stats.distinct_values
        elif condition.op in ('<', '>', '<=', '>='):
            # Assume 1/3 of range
            return 0.33
        elif condition.op == 'BETWEEN':
            return 0.25
        return 0.5  # Default

    def explain(self, stmt):
        '''Generate EXPLAIN output'''
        plan = self.plan_select(stmt)
        return [
            f"--  --  --  --  Detail",
            f"0   0   0   EXECUTE",
            f"1   0   0   {plan.description}",
            f"",
            f"Estimated rows: {plan.rows}",
            f"Estimated cost: {plan.cost:.2f}"
        ]

@dataclass
class TableStats:
    row_count: int
    avg_row_size: int
    page_count: int

@dataclass
class IndexStats:
    entries: int
    distinct_values: int
    depth: int"""
}

# Load YAML
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

expert_projects = data.get('expert_projects', {})
updated = 0

for project_id, project in expert_projects.items():
    milestones = project.get('milestones', [])
    for milestone in milestones:
        key = (project_id, milestone.get('name', ''))
        if key in expanded_hints:
            milestone['hints']['level3'] = expanded_hints[key]
            updated += 1
            print(f"Updated: {project_id}/{milestone['name']}")

# Save
with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nUpdated {updated} level3 hints")

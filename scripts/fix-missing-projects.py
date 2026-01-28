#!/usr/bin/env python3
"""Add full content for 4 missing projects."""

import yaml
from pathlib import Path

MISSING_PROJECTS = {
    "build-vpn": {
        "id": "build-vpn",
        "name": "Build Your Own VPN",
        "description": "Build a VPN that creates encrypted tunnels for secure communication. Learn TUN/TAP interfaces, encryption, key exchange, and network routing.",
        "difficulty": "expert",
        "estimated_hours": "50-80",
        "prerequisites": [
            "Network programming (sockets, TCP/UDP)",
            "Cryptography basics (AES, RSA)",
            "Linux networking (iptables, routing)",
            "TUN/TAP virtual interfaces"
        ],
        "languages": {
            "recommended": ["Go", "Rust", "C"],
            "also_possible": ["Python"]
        },
        "resources": [
            {"name": "TUN/TAP Interface Tutorial", "url": "https://www.kernel.org/doc/Documentation/networking/tuntap.txt", "type": "documentation"},
            {"name": "WireGuard Whitepaper", "url": "https://www.wireguard.com/papers/wireguard.pdf", "type": "paper"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "TUN/TAP Interface",
                "description": "Create and configure a TUN device for IP packet capture.",
                "acceptance_criteria": [
                    "Open /dev/net/tun and create TUN device",
                    "Read raw IP packets from TUN interface",
                    "Write IP packets back to TUN interface",
                    "Configure interface with IP address using ioctl",
                    "Verify with ping to TUN interface address"
                ],
                "hints": {
                    "level1": "TUN device captures IP packets. Open /dev/net/tun with specific ioctl flags to create virtual interface.",
                    "level2": "Use TUNSETIFF ioctl with IFF_TUN | IFF_NO_PI flags. Set interface IP with SIOCSIFADDR. Bring up with SIOCSIFFLAGS.",
                    "level3": """import os
import fcntl
import struct

TUNSETIFF = 0x400454ca
IFF_TUN = 0x0001
IFF_NO_PI = 0x1000

def create_tun(name='tun0'):
    '''Create TUN device and return file descriptor'''
    fd = os.open('/dev/net/tun', os.O_RDWR)

    # struct ifreq: 16 bytes name + 2 bytes flags
    ifr = struct.pack('16sH', name.encode(), IFF_TUN | IFF_NO_PI)
    fcntl.ioctl(fd, TUNSETIFF, ifr)

    # Configure IP (use subprocess for simplicity)
    import subprocess
    subprocess.run(['ip', 'addr', 'add', '10.0.0.1/24', 'dev', name])
    subprocess.run(['ip', 'link', 'set', name, 'up'])

    return fd

def read_packet(fd):
    '''Read IP packet from TUN'''
    return os.read(fd, 65535)

def write_packet(fd, packet):
    '''Write IP packet to TUN'''
    os.write(fd, packet)"""
                },
                "pitfalls": ["Forgetting IFF_NO_PI causes 4-byte header", "Must run as root", "Interface disappears when fd closed"],
                "concepts": ["Virtual network interfaces", "TUN vs TAP", "IP packet structure"],
                "estimated_hours": "6-10"
            },
            {
                "id": 2,
                "name": "UDP Transport Layer",
                "description": "Create UDP socket for tunneling encrypted packets between VPN endpoints.",
                "acceptance_criteria": [
                    "UDP server listening on VPN port",
                    "UDP client connecting to server",
                    "Packet forwarding: TUN -> UDP -> remote",
                    "Packet receiving: UDP -> TUN",
                    "Handle multiple clients (server mode)"
                ],
                "hints": {
                    "level1": "VPN tunnels packets over UDP. Read from TUN, send over UDP socket. Receive from UDP, write to TUN.",
                    "level2": "Use select/poll to multiplex TUN fd and UDP socket. Server tracks client addresses. Consider MTU (encrypt adds overhead).",
                    "level3": """import socket
import select

class VPNTunnel:
    def __init__(self, tun_fd, local_port, remote_addr=None):
        self.tun_fd = tun_fd
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', local_port))
        self.remote_addr = remote_addr  # (ip, port) for client mode
        self.clients = {}  # For server: virtual_ip -> (real_ip, port)

    def run(self):
        while True:
            readable, _, _ = select.select([self.tun_fd, self.sock], [], [])

            for fd in readable:
                if fd == self.tun_fd:
                    # Packet from local network -> send to tunnel
                    packet = os.read(self.tun_fd, 65535)
                    self.send_to_tunnel(packet)
                else:
                    # Packet from tunnel -> inject to local network
                    data, addr = self.sock.recvfrom(65535)
                    self.receive_from_tunnel(data, addr)

    def send_to_tunnel(self, packet):
        # Extract dest IP from packet, find client, send
        if self.remote_addr:
            self.sock.sendto(packet, self.remote_addr)
        else:
            # Server: route based on dest IP
            dest_ip = self.get_dest_ip(packet)
            if dest_ip in self.clients:
                self.sock.sendto(packet, self.clients[dest_ip])

    def receive_from_tunnel(self, data, addr):
        os.write(self.tun_fd, data)
        # Track client for routing
        src_ip = self.get_src_ip(data)
        self.clients[src_ip] = addr"""
                },
                "pitfalls": ["MTU issues (encryption overhead)", "NAT traversal", "Packet fragmentation"],
                "concepts": ["UDP tunneling", "Multiplexing I/O", "Client-server architecture"],
                "estimated_hours": "6-8"
            },
            {
                "id": 3,
                "name": "Encryption Layer",
                "description": "Add encryption to tunnel traffic using symmetric encryption (AES-GCM).",
                "acceptance_criteria": [
                    "Encrypt packets before sending over UDP",
                    "Decrypt packets after receiving from UDP",
                    "Use AES-256-GCM for authenticated encryption",
                    "Include nonce/IV in each packet",
                    "Verify authentication tag (reject tampered packets)"
                ],
                "hints": {
                    "level1": "Encrypt before send, decrypt after receive. AES-GCM provides encryption + authentication. Each packet needs unique nonce.",
                    "level2": "Packet format: [nonce (12 bytes)][ciphertext][tag (16 bytes)]. Use incrementing counter for nonce. Never reuse nonce with same key.",
                    "level3": """from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import struct

class EncryptedTunnel:
    def __init__(self, key):
        '''key: 32 bytes for AES-256'''
        self.aesgcm = AESGCM(key)
        self.send_counter = 0
        self.recv_window = set()  # Anti-replay

    def encrypt(self, plaintext):
        '''Encrypt packet, return nonce + ciphertext + tag'''
        # 12-byte nonce: 4 bytes sender_id + 8 bytes counter
        nonce = struct.pack('>I', 0) + struct.pack('>Q', self.send_counter)
        self.send_counter += 1

        ciphertext = self.aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext  # Tag is appended by AESGCM

    def decrypt(self, data):
        '''Decrypt packet, verify auth tag'''
        nonce = data[:12]
        ciphertext = data[12:]

        # Anti-replay check
        counter = struct.unpack('>Q', nonce[4:])[0]
        if counter in self.recv_window:
            raise ValueError("Replay attack detected")
        self.recv_window.add(counter)

        # Decrypt and verify (raises InvalidTag if tampered)
        plaintext = self.aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext

# Integration with tunnel
def send_encrypted(tunnel, packet, crypto):
    encrypted = crypto.encrypt(packet)
    tunnel.sock.sendto(encrypted, tunnel.remote_addr)

def receive_encrypted(tunnel, crypto):
    data, addr = tunnel.sock.recvfrom(65535)
    try:
        packet = crypto.decrypt(data)
        os.write(tunnel.tun_fd, packet)
    except Exception as e:
        print(f"Decrypt failed: {e}")  # Drop invalid packets"""
                },
                "pitfalls": ["Nonce reuse is catastrophic", "Forgetting auth tag verification", "Key management"],
                "concepts": ["Authenticated encryption", "Nonces and IVs", "Anti-replay protection"],
                "estimated_hours": "8-12"
            },
            {
                "id": 4,
                "name": "Key Exchange",
                "description": "Implement secure key exchange using Diffie-Hellman or similar protocol.",
                "acceptance_criteria": [
                    "Generate ephemeral key pairs",
                    "Exchange public keys over UDP",
                    "Derive shared secret using ECDH",
                    "Derive encryption key from shared secret (HKDF)",
                    "Perfect forward secrecy (new keys per session)"
                ],
                "hints": {
                    "level1": "Use Elliptic Curve Diffie-Hellman (X25519). Each side generates keypair, exchanges public key, computes shared secret.",
                    "level2": "HKDF expands shared secret into encryption key. Include session identifiers in HKDF info to bind key to session.",
                    "level3": """from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

class KeyExchange:
    def __init__(self):
        self.private_key = X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

    def get_public_bytes(self):
        '''Get public key bytes to send to peer'''
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        return self.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)

    def derive_shared_key(self, peer_public_bytes):
        '''Derive shared encryption key from peer's public key'''
        peer_public = X25519PublicKey.from_public_bytes(peer_public_bytes)
        shared_secret = self.private_key.exchange(peer_public)

        # Derive 32-byte AES key using HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'vpn-encryption-key'
        )
        return hkdf.derive(shared_secret)

# Handshake protocol
def client_handshake(sock, server_addr):
    kex = KeyExchange()

    # Send our public key
    sock.sendto(b'HELLO' + kex.get_public_bytes(), server_addr)

    # Receive server's public key
    data, _ = sock.recvfrom(65535)
    if data[:5] != b'HELLO':
        raise ValueError("Invalid handshake")

    server_public = data[5:37]
    encryption_key = kex.derive_shared_key(server_public)

    return encryption_key"""
                },
                "pitfalls": ["Not verifying peer identity (MITM)", "Reusing static keys", "Weak random number generation"],
                "concepts": ["Diffie-Hellman key exchange", "Perfect forward secrecy", "Key derivation functions"],
                "estimated_hours": "8-10"
            },
            {
                "id": 5,
                "name": "Routing and NAT",
                "description": "Configure routing tables and NAT for full VPN functionality.",
                "acceptance_criteria": [
                    "Route all traffic through VPN tunnel",
                    "Configure NAT/masquerading on server",
                    "Handle DNS through tunnel",
                    "Implement split tunneling option",
                    "Restore original routes on disconnect"
                ],
                "hints": {
                    "level1": "Add route for 0.0.0.0/0 via TUN interface. Server needs iptables MASQUERADE for NAT. Save/restore original routes.",
                    "level2": "Use 'ip route' to manage routes. Keep route to VPN server via original gateway. Set up iptables FORWARD chain for server.",
                    "level3": """import subprocess
import json

class RoutingManager:
    def __init__(self, tun_name, vpn_server_ip, local_gateway):
        self.tun_name = tun_name
        self.vpn_server_ip = vpn_server_ip
        self.local_gateway = local_gateway
        self.original_routes = []

    def setup_client_routes(self):
        '''Route all traffic through VPN'''
        # Save original default route
        result = subprocess.run(['ip', 'route', 'show', 'default'], capture_output=True, text=True)
        self.original_routes.append(result.stdout.strip())

        # Keep route to VPN server through original gateway
        subprocess.run(['ip', 'route', 'add', f'{self.vpn_server_ip}/32', 'via', self.local_gateway])

        # Route everything else through VPN
        subprocess.run(['ip', 'route', 'del', 'default'])
        subprocess.run(['ip', 'route', 'add', 'default', 'dev', self.tun_name])

        # DNS through VPN
        subprocess.run(['resolvectl', 'dns', self.tun_name, '10.0.0.1'])

    def setup_server_nat(self, external_interface='eth0'):
        '''Enable NAT on server for VPN clients'''
        # Enable IP forwarding
        with open('/proc/sys/net/ipv4/ip_forward', 'w') as f:
            f.write('1')

        # Masquerade VPN traffic
        subprocess.run([
            'iptables', '-t', 'nat', '-A', 'POSTROUTING',
            '-s', '10.0.0.0/24', '-o', external_interface, '-j', 'MASQUERADE'
        ])

        # Allow forwarding
        subprocess.run([
            'iptables', '-A', 'FORWARD',
            '-i', self.tun_name, '-o', external_interface, '-j', 'ACCEPT'
        ])
        subprocess.run([
            'iptables', '-A', 'FORWARD',
            '-i', external_interface, '-o', self.tun_name, '-m', 'state',
            '--state', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'
        ])

    def cleanup(self):
        '''Restore original routing'''
        subprocess.run(['ip', 'route', 'del', 'default'])
        for route in self.original_routes:
            subprocess.run(['ip', 'route', 'add'] + route.split())"""
                },
                "pitfalls": ["Locking yourself out (SSH)", "DNS leaks", "IPv6 leaks", "Forgetting to restore routes"],
                "concepts": ["IP routing", "NAT/masquerading", "iptables", "Split tunneling"],
                "estimated_hours": "10-15"
            }
        ]
    },

    "cache-optimized-structures": {
        "id": "cache-optimized-structures",
        "name": "Cache-Optimized Data Structures",
        "description": "Build data structures optimized for CPU cache efficiency. Learn cache-oblivious algorithms, memory layouts, and how to achieve better performance through cache-aware design.",
        "difficulty": "advanced",
        "estimated_hours": "35-55",
        "prerequisites": [
            "Data structures (arrays, trees, hash tables)",
            "Understanding of CPU cache hierarchy",
            "Memory alignment concepts",
            "Performance profiling basics"
        ],
        "languages": {
            "recommended": ["C", "C++", "Rust"],
            "also_possible": ["Go", "Zig"]
        },
        "resources": [
            {"name": "What Every Programmer Should Know About Memory", "url": "https://people.freebsd.org/~lstewart/articles/cpumemory.pdf", "type": "paper"},
            {"name": "Cache-Oblivious Algorithms", "url": "https://en.wikipedia.org/wiki/Cache-oblivious_algorithm", "type": "article"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Cache Fundamentals & Benchmarking",
                "description": "Understand cache behavior and build benchmarking tools to measure cache performance.",
                "acceptance_criteria": [
                    "Measure L1/L2/L3 cache sizes on your system",
                    "Benchmark sequential vs random memory access",
                    "Demonstrate cache line effects (64-byte access patterns)",
                    "Profile cache misses using perf or cachegrind",
                    "Document at least 10x difference between cache-friendly and cache-hostile code"
                ],
                "hints": {
                    "level1": "CPU caches are faster but smaller. L1 ~32KB, L2 ~256KB, L3 ~8MB. Access patterns matter more than algorithms for performance.",
                    "level2": "Sequential access prefetches next cache line. Random access causes cache misses. Measure with perf stat -e cache-misses,cache-references.",
                    "level3": """#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE (64 * 1024 * 1024)  // 64MB
#define CACHE_LINE 64

// Sequential access - cache friendly
double benchmark_sequential(int* arr, size_t n, int iterations) {
    clock_t start = clock();
    volatile int sum = 0;
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < n; i++) {
            sum += arr[i];
        }
    }
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Random access - cache hostile
double benchmark_random(int* arr, size_t n, int iterations) {
    // Create random permutation
    size_t* indices = malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) indices[i] = i;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
    }

    clock_t start = clock();
    volatile int sum = 0;
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < n; i++) {
            sum += arr[indices[i]];
        }
    }
    double time = (double)(clock() - start) / CLOCKS_PER_SEC;
    free(indices);
    return time;
}

// Measure cache line effect
void benchmark_stride(int* arr, size_t n) {
    printf("Stride\\tTime(ms)\\n");
    for (int stride = 1; stride <= 256; stride *= 2) {
        clock_t start = clock();
        volatile int sum = 0;
        for (int iter = 0; iter < 100; iter++) {
            for (size_t i = 0; i < n; i += stride) {
                sum += arr[i];
            }
        }
        double ms = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
        printf("%d\\t%.2f\\n", stride, ms);
    }
}

int main() {
    int* arr = aligned_alloc(CACHE_LINE, ARRAY_SIZE);
    size_t n = ARRAY_SIZE / sizeof(int);

    // Initialize
    for (size_t i = 0; i < n; i++) arr[i] = i;

    printf("Sequential: %.3f sec\\n", benchmark_sequential(arr, n, 10));
    printf("Random: %.3f sec\\n", benchmark_random(arr, n, 10));

    benchmark_stride(arr, n);

    free(arr);
    return 0;
}

// Compile: gcc -O2 -o cache_bench cache_bench.c
// Profile: perf stat -e cache-misses,cache-references ./cache_bench"""
                },
                "pitfalls": ["Compiler optimizations hiding effects", "Warmup effects", "System noise in benchmarks"],
                "concepts": ["Cache hierarchy", "Cache lines", "Spatial/temporal locality", "Prefetching"],
                "estimated_hours": "6-8"
            },
            {
                "id": 2,
                "name": "Array of Structs vs Struct of Arrays",
                "description": "Implement and compare AoS vs SoA memory layouts for better cache utilization.",
                "acceptance_criteria": [
                    "Implement particle system with AoS layout",
                    "Implement same system with SoA layout",
                    "Benchmark both with operations accessing subset of fields",
                    "Demonstrate when AoS is better (all fields accessed together)",
                    "Demonstrate when SoA is better (SIMD-friendly, single field access)"
                ],
                "hints": {
                    "level1": "AoS: struct Particle { x, y, z, vx, vy, vz }. SoA: struct Particles { float* x, *y, *z, *vx, *vy, *vz }. SoA better when accessing single field across many entities.",
                    "level2": "If you only update positions (x,y,z), AoS loads velocity too (wasted). SoA loads only what you need. SoA also enables SIMD vectorization.",
                    "level3": """#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define N 1000000

// Array of Structs
typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float mass;
    int id;
} ParticleAoS;

// Struct of Arrays
typedef struct {
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    float* mass;
    int* id;
    size_t count;
} ParticlesSoA;

// AoS: Update positions
void update_positions_aos(ParticleAoS* particles, size_t n, float dt) {
    for (size_t i = 0; i < n; i++) {
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}

// SoA: Update positions (cache-friendly, SIMD-friendly)
void update_positions_soa(ParticlesSoA* p, float dt) {
    // Scalar version
    for (size_t i = 0; i < p->count; i++) {
        p->x[i] += p->vx[i] * dt;
        p->y[i] += p->vy[i] * dt;
        p->z[i] += p->vz[i] * dt;
    }
}

// SoA with SIMD (AVX)
void update_positions_soa_simd(ParticlesSoA* p, float dt) {
    __m256 dt_vec = _mm256_set1_ps(dt);
    size_t i;
    for (i = 0; i + 8 <= p->count; i += 8) {
        __m256 x = _mm256_loadu_ps(&p->x[i]);
        __m256 vx = _mm256_loadu_ps(&p->vx[i]);
        x = _mm256_fmadd_ps(vx, dt_vec, x);
        _mm256_storeu_ps(&p->x[i], x);

        // Same for y, z...
    }
    // Handle remainder
    for (; i < p->count; i++) {
        p->x[i] += p->vx[i] * dt;
    }
}

// AoS: Access all fields (AoS wins here)
float compute_kinetic_energy_aos(ParticleAoS* particles, size_t n) {
    float total = 0;
    for (size_t i = 0; i < n; i++) {
        float v2 = particles[i].vx * particles[i].vx +
                   particles[i].vy * particles[i].vy +
                   particles[i].vz * particles[i].vz;
        total += 0.5f * particles[i].mass * v2;
    }
    return total;
}

int main() {
    // Benchmark both layouts...
    return 0;
}"""
                },
                "pitfalls": ["Not aligning SoA arrays", "Forgetting remainder in SIMD loops", "Over-optimizing when AoS is actually fine"],
                "concepts": ["Data-oriented design", "Memory layout", "SIMD vectorization", "Cache line utilization"],
                "estimated_hours": "6-10"
            },
            {
                "id": 3,
                "name": "Cache-Friendly Hash Table",
                "description": "Implement a hash table optimized for cache performance using open addressing and linear probing.",
                "acceptance_criteria": [
                    "Implement open addressing with linear probing",
                    "Store keys and values in separate arrays (better cache use for lookups)",
                    "Implement Robin Hood hashing for better probe distribution",
                    "Benchmark against std::unordered_map / HashMap",
                    "Show improvement in cache misses with perf"
                ],
                "hints": {
                    "level1": "Chaining is cache-hostile (pointer chasing). Open addressing with linear probing keeps data contiguous. Keys-only array for initial probe is cache-friendly.",
                    "level2": "Robin Hood: when inserting, if probe distance > existing element's probe distance, swap and continue with evicted element. Reduces variance in probe lengths.",
                    "level3": """#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t* keys;      // Separate array for cache-friendly probing
    uint64_t* values;
    uint8_t* distances;  // Probe distance (for Robin Hood)
    size_t capacity;
    size_t size;
} CacheFriendlyMap;

#define EMPTY_KEY 0
#define TOMBSTONE UINT64_MAX

CacheFriendlyMap* cfmap_create(size_t capacity) {
    CacheFriendlyMap* map = malloc(sizeof(CacheFriendlyMap));
    map->capacity = capacity;
    map->size = 0;
    map->keys = calloc(capacity, sizeof(uint64_t));
    map->values = calloc(capacity, sizeof(uint64_t));
    map->distances = calloc(capacity, sizeof(uint8_t));
    return map;
}

static inline size_t hash(uint64_t key) {
    // Fast hash function
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccd;
    key ^= key >> 33;
    return key;
}

void cfmap_insert(CacheFriendlyMap* map, uint64_t key, uint64_t value) {
    size_t idx = hash(key) % map->capacity;
    uint8_t dist = 0;

    while (1) {
        if (map->keys[idx] == EMPTY_KEY || map->keys[idx] == TOMBSTONE) {
            // Found empty slot
            map->keys[idx] = key;
            map->values[idx] = value;
            map->distances[idx] = dist;
            map->size++;
            return;
        }

        if (map->keys[idx] == key) {
            // Update existing
            map->values[idx] = value;
            return;
        }

        // Robin Hood: steal from rich (low probe distance)
        if (dist > map->distances[idx]) {
            // Swap
            uint64_t tmp_key = map->keys[idx];
            uint64_t tmp_val = map->values[idx];
            uint8_t tmp_dist = map->distances[idx];

            map->keys[idx] = key;
            map->values[idx] = value;
            map->distances[idx] = dist;

            key = tmp_key;
            value = tmp_val;
            dist = tmp_dist;
        }

        idx = (idx + 1) % map->capacity;
        dist++;
    }
}

uint64_t* cfmap_get(CacheFriendlyMap* map, uint64_t key) {
    size_t idx = hash(key) % map->capacity;
    uint8_t dist = 0;

    while (map->keys[idx] != EMPTY_KEY) {
        if (map->keys[idx] == key) {
            return &map->values[idx];
        }

        // Robin Hood: can stop early if we've probed further than any element could be
        if (dist > map->distances[idx]) {
            return NULL;
        }

        idx = (idx + 1) % map->capacity;
        dist++;
    }

    return NULL;
}"""
                },
                "pitfalls": ["Load factor too high (>70%)", "Poor hash function", "Not handling resize properly"],
                "concepts": ["Open addressing", "Linear probing", "Robin Hood hashing", "Cache-conscious design"],
                "estimated_hours": "8-12"
            },
            {
                "id": 4,
                "name": "Cache-Oblivious B-Tree",
                "description": "Implement a van Emde Boas layout B-tree that performs well regardless of cache size.",
                "acceptance_criteria": [
                    "Implement standard B-tree with configurable branching factor",
                    "Implement van Emde Boas memory layout",
                    "Benchmark against standard B-tree layout",
                    "Show performance scales well across different cache sizes",
                    "Measure cache misses for various tree sizes"
                ],
                "hints": {
                    "level1": "van Emde Boas layout: recursively split tree, store left subtree, then right subtree. Top half of tree in first half of memory, recursively.",
                    "level2": "For a complete binary tree of height h: split at h/2. Store top subtree, then all bottom subtrees contiguously. Achieves O(log_B N) cache misses.",
                    "level3": """#include <stdlib.h>
#include <string.h>
#include <math.h>

// Standard B-tree node
typedef struct BTreeNode {
    int* keys;
    struct BTreeNode** children;
    int num_keys;
    int is_leaf;
} BTreeNode;

// van Emde Boas layout for implicit complete binary search tree
// Maps tree position to memory position for cache-oblivious access

typedef struct {
    int* data;          // All keys in vEB layout
    size_t size;
    size_t capacity;    // Must be 2^k - 1
} vEBTree;

// Calculate vEB position for a node at given level and index
// This is the key insight: recursive layout
size_t veb_position(size_t height, size_t level, size_t index) {
    if (height == 1) {
        return 0;
    }

    size_t top_height = height / 2;
    size_t bottom_height = height - top_height;
    size_t top_size = (1 << top_height) - 1;
    size_t bottom_size = (1 << bottom_height) - 1;

    if (level < top_height) {
        // Node is in top subtree
        return veb_position(top_height, level, index);
    } else {
        // Node is in one of the bottom subtrees
        size_t bottom_index = index >> (height - level - 1);
        size_t local_index = index & ((1 << (height - level - 1)) - 1);
        size_t local_level = level - top_height;

        return top_size +
               bottom_index * bottom_size +
               veb_position(bottom_height, local_level, local_index);
    }
}

vEBTree* veb_create(size_t height) {
    vEBTree* tree = malloc(sizeof(vEBTree));
    tree->capacity = (1 << height) - 1;
    tree->data = malloc(tree->capacity * sizeof(int));
    tree->size = 0;
    memset(tree->data, -1, tree->capacity * sizeof(int));  // -1 = empty
    return tree;
}

// Build vEB tree from sorted array
void veb_build(vEBTree* tree, int* sorted, size_t n, size_t height) {
    // For each position in sorted array, compute its vEB position
    for (size_t i = 0; i < n; i++) {
        // Binary search tree position: level and index
        // Root is level 0, index 0
        // Left child of (l, i) is (l+1, 2*i), right is (l+1, 2*i+1)

        // For a balanced BST built from sorted array:
        // Element at sorted[i] goes to specific tree position
        size_t tree_pos = /* compute tree position for sorted element i */;
        size_t veb_pos = veb_position(height, /* level */, /* index */);
        tree->data[veb_pos] = sorted[i];
    }
    tree->size = n;
}

// Search in vEB layout - same traversal, different memory access pattern
int veb_search(vEBTree* tree, int key, size_t height) {
    size_t level = 0;
    size_t index = 0;

    while (level < height) {
        size_t pos = veb_position(height, level, index);

        if (tree->data[pos] == -1) return 0;  // Not found
        if (tree->data[pos] == key) return 1;  // Found

        // Go left or right
        if (key < tree->data[pos]) {
            index = 2 * index;      // Left child
        } else {
            index = 2 * index + 1;  // Right child
        }
        level++;
    }

    return 0;
}"""
                },
                "pitfalls": ["Complex index calculations", "Not handling non-power-of-2 sizes", "Overhead may exceed benefit for small trees"],
                "concepts": ["Cache-oblivious algorithms", "van Emde Boas layout", "Memory hierarchy independence"],
                "estimated_hours": "10-15"
            },
            {
                "id": 5,
                "name": "Blocked Matrix Operations",
                "description": "Implement cache-blocked matrix multiplication and other operations.",
                "acceptance_criteria": [
                    "Implement naive matrix multiplication",
                    "Implement blocked/tiled matrix multiplication",
                    "Auto-tune block size based on cache size",
                    "Achieve significant speedup (2-10x) over naive",
                    "Extend to other operations (transpose, LU decomposition)"
                ],
                "hints": {
                    "level1": "Naive matrix multiply has poor cache behavior for large matrices. Blocked/tiled approach processes submatrices that fit in cache.",
                    "level2": "Block size should fit in L1/L2 cache. For L1=32KB, 3 matrices of floats: sqrt(32KB / 3 / 4) ≈ 50. Try 32 or 64.",
                    "level3": """#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Naive O(n^3) - cache hostile for large n
void matmul_naive(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];  // B access is cache-hostile
            }
            C[i * n + j] = sum;
        }
    }
}

// Blocked/Tiled - cache friendly
void matmul_blocked(float* A, float* B, float* C, int n, int block_size) {
    memset(C, 0, n * n * sizeof(float));

    for (int i0 = 0; i0 < n; i0 += block_size) {
        for (int j0 = 0; j0 < n; j0 += block_size) {
            for (int k0 = 0; k0 < n; k0 += block_size) {
                // Multiply blocks
                int i_max = (i0 + block_size < n) ? i0 + block_size : n;
                int j_max = (j0 + block_size < n) ? j0 + block_size : n;
                int k_max = (k0 + block_size < n) ? k0 + block_size : n;

                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        float a_ik = A[i * n + k];
                        for (int j = j0; j < j_max; j++) {
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

// Auto-tune block size
int find_optimal_block_size(int n) {
    float* A = malloc(n * n * sizeof(float));
    float* B = malloc(n * n * sizeof(float));
    float* C = malloc(n * n * sizeof(float));

    // Initialize with random values
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    int best_block = 16;
    double best_time = 1e9;

    for (int block = 16; block <= 256 && block <= n; block *= 2) {
        clock_t start = clock();
        matmul_blocked(A, B, C, n, block);
        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

        printf("Block %d: %.3f sec\\n", block, elapsed);

        if (elapsed < best_time) {
            best_time = elapsed;
            best_block = block;
        }
    }

    free(A); free(B); free(C);
    return best_block;
}

int main() {
    int n = 1024;

    float* A = aligned_alloc(64, n * n * sizeof(float));
    float* B = aligned_alloc(64, n * n * sizeof(float));
    float* C = aligned_alloc(64, n * n * sizeof(float));

    // Initialize
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    clock_t start;

    start = clock();
    matmul_naive(A, B, C, n);
    printf("Naive: %.3f sec\\n", (double)(clock() - start) / CLOCKS_PER_SEC);

    start = clock();
    matmul_blocked(A, B, C, n, 64);
    printf("Blocked (64): %.3f sec\\n", (double)(clock() - start) / CLOCKS_PER_SEC);

    free(A); free(B); free(C);
    return 0;
}"""
                },
                "pitfalls": ["Block size too large (exceeds cache)", "Not handling non-block-aligned sizes", "Memory alignment issues"],
                "concepts": ["Loop tiling", "Blocking", "Cache blocking", "Auto-tuning"],
                "estimated_hours": "6-10"
            }
        ]
    },

    "sandbox": {
        "id": "sandbox",
        "name": "Process Sandbox",
        "description": "Build a process sandbox using Linux security features to isolate untrusted code. Learn namespaces, seccomp, capabilities, and cgroups for defense in depth.",
        "difficulty": "intermediate",
        "estimated_hours": "25-40",
        "prerequisites": [
            "Linux system calls",
            "Process management (fork, exec)",
            "Basic understanding of Linux security",
            "C programming"
        ],
        "languages": {
            "recommended": ["C", "Rust", "Go"],
            "also_possible": ["Python (with ctypes)"]
        },
        "resources": [
            {"name": "Linux Namespaces", "url": "https://man7.org/linux/man-pages/man7/namespaces.7.html", "type": "documentation"},
            {"name": "Seccomp BPF", "url": "https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html", "type": "documentation"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Process Namespaces",
                "description": "Use Linux namespaces to isolate process view of system resources.",
                "acceptance_criteria": [
                    "Create new PID namespace (process sees itself as PID 1)",
                    "Create new mount namespace (isolated filesystem view)",
                    "Create new network namespace (no network access)",
                    "Create new UTS namespace (isolated hostname)",
                    "Verify isolation with /proc inspection"
                ],
                "hints": {
                    "level1": "Use clone() with CLONE_NEWPID, CLONE_NEWNS, CLONE_NEWNET flags. Or unshare() in existing process.",
                    "level2": "After clone with CLONE_NEWPID, child is PID 1 in new namespace. Mount new /proc to see only namespace processes.",
                    "level3": """#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mount.h>

#define STACK_SIZE (1024 * 1024)

int child_func(void* arg) {
    printf("Child PID in namespace: %d\\n", getpid());  // Should be 1

    // Mount new /proc for this namespace
    if (mount("proc", "/proc", "proc", 0, NULL) == -1) {
        perror("mount /proc");
    }

    // Set new hostname
    sethostname("sandbox", 7);

    // Show isolation
    system("echo Hostname: $(hostname)");
    system("echo 'Processes:' && ps aux");

    // Run sandboxed program
    char* argv[] = {"/bin/sh", NULL};
    execv("/bin/sh", argv);

    return 0;
}

int main() {
    char* stack = malloc(STACK_SIZE);
    if (!stack) {
        perror("malloc");
        return 1;
    }

    int flags = CLONE_NEWPID |   // New PID namespace
                CLONE_NEWNS |    // New mount namespace
                CLONE_NEWNET |   // New network namespace (no network)
                CLONE_NEWUTS |   // New UTS namespace (hostname)
                SIGCHLD;

    pid_t pid = clone(child_func, stack + STACK_SIZE, flags, NULL);
    if (pid == -1) {
        perror("clone");
        return 1;
    }

    printf("Parent: child PID = %d\\n", pid);
    waitpid(pid, NULL, 0);

    free(stack);
    return 0;
}"""
                },
                "pitfalls": ["Forgetting SIGCHLD flag", "Not mounting /proc in new namespace", "Need root/CAP_SYS_ADMIN"],
                "concepts": ["Linux namespaces", "Process isolation", "clone() system call"],
                "estimated_hours": "5-8"
            },
            {
                "id": 2,
                "name": "Filesystem Isolation",
                "description": "Create isolated filesystem using chroot or pivot_root with minimal root filesystem.",
                "acceptance_criteria": [
                    "Create minimal rootfs with required binaries",
                    "Use chroot or pivot_root to change root",
                    "Mount /proc, /dev, /sys appropriately",
                    "Prevent escape via /proc/*/root or similar",
                    "Test that parent filesystem is inaccessible"
                ],
                "hints": {
                    "level1": "chroot changes apparent root but can be escaped. pivot_root in mount namespace is more secure. Create minimal rootfs with busybox.",
                    "level2": "pivot_root requires mount namespace. After pivot_root, unmount old root. Block /proc/*/root access with seccomp or mount restrictions.",
                    "level3": """#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/syscall.h>

int pivot_root(const char* new_root, const char* put_old) {
    return syscall(SYS_pivot_root, new_root, put_old);
}

int setup_rootfs(const char* rootfs) {
    // Make rootfs a mount point
    if (mount(rootfs, rootfs, NULL, MS_BIND | MS_REC, NULL) == -1) {
        perror("mount bind rootfs");
        return -1;
    }

    // Create put_old directory
    char put_old[256];
    snprintf(put_old, sizeof(put_old), "%s/.old_root", rootfs);
    mkdir(put_old, 0700);

    // Change to new root
    if (chdir(rootfs) == -1) {
        perror("chdir");
        return -1;
    }

    // pivot_root
    if (pivot_root(".", ".old_root") == -1) {
        perror("pivot_root");
        return -1;
    }

    // Change to new root
    if (chdir("/") == -1) {
        perror("chdir /");
        return -1;
    }

    // Unmount old root
    if (umount2("/.old_root", MNT_DETACH) == -1) {
        perror("umount old root");
        return -1;
    }
    rmdir("/.old_root");

    // Mount essential filesystems
    mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC, NULL);
    mount("tmpfs", "/tmp", "tmpfs", MS_NOSUID | MS_NODEV, "size=64M");
    mount("tmpfs", "/dev", "tmpfs", MS_NOSUID, "size=64K,mode=755");

    // Create minimal /dev entries
    mknod("/dev/null", S_IFCHR | 0666, makedev(1, 3));
    mknod("/dev/zero", S_IFCHR | 0666, makedev(1, 5));
    mknod("/dev/random", S_IFCHR | 0444, makedev(1, 8));
    mknod("/dev/urandom", S_IFCHR | 0444, makedev(1, 9));

    return 0;
}

// Create minimal rootfs (run once during setup)
void create_minimal_rootfs(const char* path) {
    char cmd[512];

    mkdir(path, 0755);
    snprintf(cmd, sizeof(cmd), "mkdir -p %s/{bin,lib,lib64,proc,dev,tmp,etc}", path);
    system(cmd);

    // Copy busybox for basic utilities
    snprintf(cmd, sizeof(cmd), "cp /bin/busybox %s/bin/", path);
    system(cmd);

    // Create symlinks for common commands
    snprintf(cmd, sizeof(cmd), "cd %s/bin && for cmd in sh ls cat echo ps; do ln -s busybox $cmd; done", path);
    system(cmd);

    // Copy required libraries
    snprintf(cmd, sizeof(cmd), "ldd /bin/busybox | grep -o '/lib[^ ]*' | xargs -I {} cp {} %s/lib/", path);
    system(cmd);
}"""
                },
                "pitfalls": ["Forgetting to unmount old root", "Missing /dev entries", "Library dependencies"],
                "concepts": ["chroot vs pivot_root", "Filesystem namespaces", "Minimal rootfs"],
                "estimated_hours": "5-8"
            },
            {
                "id": 3,
                "name": "Seccomp System Call Filtering",
                "description": "Use seccomp-BPF to restrict which system calls the sandboxed process can make.",
                "acceptance_criteria": [
                    "Create seccomp filter using BPF",
                    "Whitelist safe system calls only",
                    "Kill process on forbidden syscall",
                    "Allow read/write but block open of sensitive paths",
                    "Test filter blocks dangerous operations"
                ],
                "hints": {
                    "level1": "Seccomp BPF filters syscalls before execution. Whitelist approach: deny all, allow specific. Return SECCOMP_RET_KILL for violations.",
                    "level2": "Use libseccomp for easier filter creation. Filter can inspect syscall number and arguments. Install filter with prctl(PR_SET_SECCOMP).",
                    "level3": """#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <sys/syscall.h>

// Using raw BPF for demonstration (libseccomp is easier in practice)
int install_seccomp_filter() {
    // Allow: read, write, exit, exit_group, brk, mmap, mprotect
    // Kill on: open, openat, execve, fork, clone, socket, etc.

    struct sock_filter filter[] = {
        // Load syscall number
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),

        // Allow read (0)
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_read, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        // Allow write (1)
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_write, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        // Allow exit (60)
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        // Allow exit_group (231)
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit_group, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        // Allow brk (12) - memory allocation
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_brk, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        // Allow mmap (9), mprotect (10), munmap (11)
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mmap, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_mprotect, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_munmap, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        // Allow fstat (5), close (3)
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_fstat, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_close, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        // Kill on anything else
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL),
    };

    struct sock_fprog prog = {
        .len = sizeof(filter) / sizeof(filter[0]),
        .filter = filter,
    };

    // No new privileges
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == -1) {
        perror("prctl(NO_NEW_PRIVS)");
        return -1;
    }

    // Install filter
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) == -1) {
        perror("prctl(SECCOMP)");
        return -1;
    }

    return 0;
}

// Using libseccomp (much easier)
#include <seccomp.h>

int install_seccomp_libseccomp() {
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_KILL);  // Default: kill
    if (!ctx) return -1;

    // Whitelist safe syscalls
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(read), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit_group), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(brk), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(mmap), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(close), 0);

    // Allow write only to stdout/stderr (fd 1, 2)
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 1,
                     SCMP_A0(SCMP_CMP_EQ, 1));  // stdout
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 1,
                     SCMP_A0(SCMP_CMP_EQ, 2));  // stderr

    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
    seccomp_load(ctx);
    seccomp_release(ctx);

    return 0;
}"""
                },
                "pitfalls": ["Missing required syscalls (glibc uses many)", "Architecture-specific syscall numbers", "Not setting NO_NEW_PRIVS"],
                "concepts": ["Seccomp BPF", "System call filtering", "Whitelist vs blacklist"],
                "estimated_hours": "6-10"
            },
            {
                "id": 4,
                "name": "Resource Limits with Cgroups",
                "description": "Use cgroups to limit CPU, memory, and I/O resources for sandboxed processes.",
                "acceptance_criteria": [
                    "Create cgroup for sandbox",
                    "Set memory limit (e.g., 64MB)",
                    "Set CPU limit (e.g., 10% of one core)",
                    "Set I/O bandwidth limit",
                    "Verify limits are enforced"
                ],
                "hints": {
                    "level1": "Cgroups v2 uses unified hierarchy at /sys/fs/cgroup. Create subdirectory, write PID to cgroup.procs, set limits in controller files.",
                    "level2": "memory.max for memory limit, cpu.max for CPU (e.g., '10000 100000' = 10%), io.max for I/O limits.",
                    "level3": """#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#define CGROUP_ROOT "/sys/fs/cgroup"

typedef struct {
    char* name;
    size_t memory_bytes;
    int cpu_percent;      // 0-100
    size_t io_read_bps;
    size_t io_write_bps;
} CgroupLimits;

int cgroup_write(const char* cgroup, const char* file, const char* value) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s/%s", CGROUP_ROOT, cgroup, file);

    int fd = open(path, O_WRONLY);
    if (fd == -1) {
        perror(path);
        return -1;
    }

    write(fd, value, strlen(value));
    close(fd);
    return 0;
}

int cgroup_create(CgroupLimits* limits) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", CGROUP_ROOT, limits->name);

    // Create cgroup directory
    if (mkdir(path, 0755) == -1 && errno != EEXIST) {
        perror("mkdir cgroup");
        return -1;
    }

    // Enable controllers
    cgroup_write("", "cgroup.subtree_control", "+memory +cpu +io");

    // Set memory limit
    if (limits->memory_bytes > 0) {
        char value[64];
        snprintf(value, sizeof(value), "%zu", limits->memory_bytes);
        cgroup_write(limits->name, "memory.max", value);

        // Disable swap
        cgroup_write(limits->name, "memory.swap.max", "0");
    }

    // Set CPU limit (cpu.max: $MAX $PERIOD)
    // 10% = 10000 us per 100000 us period
    if (limits->cpu_percent > 0) {
        char value[64];
        int quota = limits->cpu_percent * 1000;  // microseconds
        snprintf(value, sizeof(value), "%d 100000", quota);
        cgroup_write(limits->name, "cpu.max", value);
    }

    // Set I/O limits (need device major:minor)
    // Example: limit /dev/sda (8:0)
    if (limits->io_read_bps > 0 || limits->io_write_bps > 0) {
        char value[128];
        snprintf(value, sizeof(value), "8:0 rbps=%zu wbps=%zu",
                 limits->io_read_bps, limits->io_write_bps);
        cgroup_write(limits->name, "io.max", value);
    }

    return 0;
}

int cgroup_add_process(const char* cgroup_name, pid_t pid) {
    char value[32];
    snprintf(value, sizeof(value), "%d", pid);
    return cgroup_write(cgroup_name, "cgroup.procs", value);
}

int cgroup_destroy(const char* name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", CGROUP_ROOT, name);

    // Move processes to parent first
    // Then rmdir
    rmdir(path);
    return 0;
}

// Example usage
int main() {
    CgroupLimits limits = {
        .name = "sandbox",
        .memory_bytes = 64 * 1024 * 1024,  // 64 MB
        .cpu_percent = 10,                   // 10% of one core
        .io_read_bps = 10 * 1024 * 1024,    // 10 MB/s read
        .io_write_bps = 5 * 1024 * 1024,    // 5 MB/s write
    };

    cgroup_create(&limits);

    pid_t pid = fork();
    if (pid == 0) {
        // Child: run in cgroup
        cgroup_add_process("sandbox", getpid());

        // Memory test - will be killed if exceeds limit
        char* mem = malloc(100 * 1024 * 1024);  // Try 100MB (over limit)
        if (mem) memset(mem, 'x', 100 * 1024 * 1024);

        exit(0);
    }

    waitpid(pid, NULL, 0);
    cgroup_destroy("sandbox");

    return 0;
}"""
                },
                "pitfalls": ["Cgroups v1 vs v2 differences", "Need root for cgroup operations", "Controller not enabled"],
                "concepts": ["Cgroups", "Resource limits", "CPU quotas", "Memory limits"],
                "estimated_hours": "5-8"
            },
            {
                "id": 5,
                "name": "Capability Dropping",
                "description": "Drop Linux capabilities to run with minimal privileges.",
                "acceptance_criteria": [
                    "List current process capabilities",
                    "Drop all capabilities except required ones",
                    "Set no-new-privileges flag",
                    "Verify capabilities are dropped",
                    "Test that privileged operations fail"
                ],
                "hints": {
                    "level1": "Linux capabilities split root power into units. Drop caps you don't need. CAP_NET_RAW for ping, CAP_SYS_ADMIN for mount, etc.",
                    "level2": "Use capset() or libcap. Clear effective, permitted, and inheritable sets. Set PR_SET_NO_NEW_PRIVS to prevent regaining caps.",
                    "level3": """#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/capability.h>
#include <sys/prctl.h>

void print_caps(const char* label) {
    cap_t caps = cap_get_proc();
    char* caps_text = cap_to_text(caps, NULL);
    printf("%s: %s\\n", label, caps_text);
    cap_free(caps_text);
    cap_free(caps);
}

int drop_capabilities(cap_value_t* keep_caps, int num_keep) {
    // Get current caps
    cap_t caps = cap_get_proc();
    if (!caps) {
        perror("cap_get_proc");
        return -1;
    }

    // Clear all caps
    if (cap_clear(caps) == -1) {
        perror("cap_clear");
        cap_free(caps);
        return -1;
    }

    // Add back only the ones we need
    if (num_keep > 0) {
        if (cap_set_flag(caps, CAP_PERMITTED, num_keep, keep_caps, CAP_SET) == -1 ||
            cap_set_flag(caps, CAP_EFFECTIVE, num_keep, keep_caps, CAP_SET) == -1) {
            perror("cap_set_flag");
            cap_free(caps);
            return -1;
        }
    }

    // Apply
    if (cap_set_proc(caps) == -1) {
        perror("cap_set_proc");
        cap_free(caps);
        return -1;
    }

    cap_free(caps);

    // Prevent regaining caps through exec
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == -1) {
        perror("prctl NO_NEW_PRIVS");
        return -1;
    }

    return 0;
}

// Drop to nobody user after dropping caps
int drop_user() {
    // Set groups
    if (setgroups(0, NULL) == -1) {
        perror("setgroups");
        return -1;
    }

    // Set GID then UID (order matters!)
    if (setgid(65534) == -1) {  // nobody
        perror("setgid");
        return -1;
    }
    if (setuid(65534) == -1) {  // nobody
        perror("setuid");
        return -1;
    }

    return 0;
}

int main() {
    print_caps("Before");

    // Keep only CAP_NET_BIND_SERVICE (for binding to ports < 1024)
    cap_value_t keep[] = { CAP_NET_BIND_SERVICE };

    // Or keep nothing for maximum restriction
    drop_capabilities(NULL, 0);

    print_caps("After drop_capabilities");

    // Also drop to unprivileged user
    drop_user();

    printf("Now running as UID %d, GID %d\\n", getuid(), getgid());

    // Test: this should fail now
    if (setuid(0) == 0) {
        printf("ERROR: Was able to become root!\\n");
        return 1;
    }
    printf("Good: Cannot become root\\n");

    // Test: this should fail
    if (unlink("/etc/passwd") == 0) {
        printf("ERROR: Was able to delete /etc/passwd!\\n");
        return 1;
    }
    printf("Good: Cannot delete protected files\\n");

    return 0;
}

// Compile: gcc -o caps caps.c -lcap"""
                },
                "pitfalls": ["Order of setgid/setuid matters", "Some caps needed for basic operations", "Ambient caps can re-enable dropped caps"],
                "concepts": ["Linux capabilities", "Principle of least privilege", "Privilege dropping"],
                "estimated_hours": "4-6"
            }
        ]
    },

    "vulnerability-scanner": {
        "id": "vulnerability-scanner",
        "name": "Vulnerability Scanner",
        "description": "Build a network vulnerability scanner that discovers hosts, identifies services, and checks for known vulnerabilities. Learn network scanning, service fingerprinting, and CVE detection.",
        "difficulty": "advanced",
        "estimated_hours": "40-60",
        "prerequisites": [
            "TCP/IP networking",
            "Socket programming",
            "Understanding of common vulnerabilities",
            "Basic security concepts"
        ],
        "languages": {
            "recommended": ["Python", "Go"],
            "also_possible": ["Rust", "C"]
        },
        "resources": [
            {"name": "Nmap Network Scanning", "url": "https://nmap.org/book/toc.html", "type": "book"},
            {"name": "NIST NVD API", "url": "https://nvd.nist.gov/developers/vulnerabilities", "type": "documentation"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Host Discovery",
                "description": "Implement various techniques to discover live hosts on a network.",
                "acceptance_criteria": [
                    "ICMP echo (ping) scan",
                    "TCP SYN scan for common ports",
                    "ARP scan for local network",
                    "Handle rate limiting to avoid detection",
                    "Report discovered hosts with response times"
                ],
                "hints": {
                    "level1": "Use raw sockets for ICMP/TCP SYN. ARP only works on local subnet. Parallelize with asyncio or threading for speed.",
                    "level2": "ICMP may be blocked by firewalls. TCP SYN to port 80/443 often gets through. ARP is most reliable for local network.",
                    "level3": """import socket
import struct
import asyncio
import time

class HostDiscovery:
    def __init__(self, timeout=1):
        self.timeout = timeout

    async def icmp_ping(self, ip):
        '''Send ICMP echo request'''
        try:
            # Need raw socket (requires root)
            sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
            sock.settimeout(self.timeout)

            # ICMP echo request
            icmp_type = 8  # Echo request
            icmp_code = 0
            checksum = 0
            identifier = 1
            sequence = 1

            # Calculate checksum
            header = struct.pack('!BBHHH', icmp_type, icmp_code, checksum, identifier, sequence)
            checksum = self.calculate_checksum(header)
            header = struct.pack('!BBHHH', icmp_type, icmp_code, checksum, identifier, sequence)

            start = time.time()
            sock.sendto(header, (ip, 0))

            # Wait for reply
            data, addr = sock.recvfrom(1024)
            rtt = (time.time() - start) * 1000
            sock.close()

            return {'ip': ip, 'alive': True, 'method': 'icmp', 'rtt_ms': rtt}
        except (socket.timeout, socket.error):
            return {'ip': ip, 'alive': False}

    async def tcp_syn_probe(self, ip, port=80):
        '''TCP connect to check if host is alive'''
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=self.timeout
            )
            writer.close()
            await writer.wait_closed()
            return {'ip': ip, 'alive': True, 'method': 'tcp', 'port': port}
        except:
            return {'ip': ip, 'alive': False}

    async def scan_network(self, network_cidr):
        '''Scan entire network'''
        import ipaddress
        network = ipaddress.ip_network(network_cidr, strict=False)

        tasks = []
        for ip in network.hosts():
            ip_str = str(ip)
            # Try multiple methods in parallel
            tasks.append(self.icmp_ping(ip_str))
            tasks.append(self.tcp_syn_probe(ip_str, 80))
            tasks.append(self.tcp_syn_probe(ip_str, 443))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Deduplicate by IP
        alive_hosts = {}
        for r in results:
            if isinstance(r, dict) and r.get('alive'):
                alive_hosts[r['ip']] = r

        return list(alive_hosts.values())

    @staticmethod
    def calculate_checksum(data):
        if len(data) % 2:
            data += b'\\x00'

        total = 0
        for i in range(0, len(data), 2):
            total += (data[i] << 8) + data[i + 1]

        total = (total >> 16) + (total & 0xffff)
        total += total >> 16
        return (~total) & 0xffff

# Usage
async def main():
    scanner = HostDiscovery(timeout=2)
    hosts = await scanner.scan_network('192.168.1.0/24')
    for host in hosts:
        print(f"Alive: {host['ip']} via {host['method']}")

asyncio.run(main())"""
                },
                "pitfalls": ["Raw sockets need root", "ICMP often blocked", "Rate limiting needed"],
                "concepts": ["ICMP protocol", "TCP handshake", "ARP protocol", "Network scanning"],
                "estimated_hours": "6-10"
            },
            {
                "id": 2,
                "name": "Port Scanning",
                "description": "Implement various port scanning techniques to identify open services.",
                "acceptance_criteria": [
                    "TCP connect scan",
                    "TCP SYN (half-open) scan",
                    "UDP scan with common port payloads",
                    "Service version detection from banner",
                    "Scan 1000 common ports in under 30 seconds"
                ],
                "hints": {
                    "level1": "Connect scan is reliable but slow. SYN scan is faster and stealthier. UDP is unreliable (no response ≠ closed).",
                    "level2": "Use asyncio for concurrent scanning. SYN scan needs raw sockets. For UDP, send protocol-specific probes to elicit response.",
                    "level3": """import asyncio
import socket

# Top 20 ports to always scan
TOP_PORTS = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139,
             143, 443, 445, 993, 995, 1723, 3306, 3389, 5900, 8080]

class PortScanner:
    def __init__(self, timeout=1, concurrency=100):
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(concurrency)

    async def tcp_connect_scan(self, ip, port):
        '''Full TCP connect scan'''
        async with self.semaphore:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip, port),
                    timeout=self.timeout
                )

                # Try to get banner
                banner = None
                try:
                    writer.write(b'\\r\\n')
                    await writer.drain()
                    banner = await asyncio.wait_for(reader.read(1024), timeout=0.5)
                    banner = banner.decode('utf-8', errors='ignore').strip()
                except:
                    pass

                writer.close()
                await writer.wait_closed()

                return {
                    'port': port,
                    'state': 'open',
                    'banner': banner
                }
            except asyncio.TimeoutError:
                return {'port': port, 'state': 'filtered'}
            except ConnectionRefusedError:
                return {'port': port, 'state': 'closed'}
            except:
                return {'port': port, 'state': 'error'}

    async def scan_host(self, ip, ports=None):
        '''Scan all ports on a host'''
        if ports is None:
            ports = TOP_PORTS + list(range(1, 1001))  # Top + first 1000
            ports = list(set(ports))

        tasks = [self.tcp_connect_scan(ip, port) for port in ports]
        results = await asyncio.gather(*tasks)

        open_ports = [r for r in results if r['state'] == 'open']
        return {'ip': ip, 'ports': open_ports}

    def identify_service(self, port, banner):
        '''Guess service from port and banner'''
        services = {
            21: 'ftp',
            22: 'ssh',
            23: 'telnet',
            25: 'smtp',
            53: 'dns',
            80: 'http',
            110: 'pop3',
            143: 'imap',
            443: 'https',
            445: 'smb',
            3306: 'mysql',
            5432: 'postgresql',
            6379: 'redis',
            27017: 'mongodb',
        }

        service = services.get(port, 'unknown')

        # Banner analysis
        if banner:
            banner_lower = banner.lower()
            if 'ssh' in banner_lower:
                service = 'ssh'
                # Extract version: SSH-2.0-OpenSSH_8.2p1
                if 'openssh' in banner_lower:
                    service = f'ssh (OpenSSH)'
            elif 'apache' in banner_lower:
                service = 'http (Apache)'
            elif 'nginx' in banner_lower:
                service = 'http (nginx)'
            elif 'mysql' in banner_lower:
                service = 'mysql'
            elif 'ftp' in banner_lower:
                service = 'ftp'

        return service

async def main():
    scanner = PortScanner(timeout=1, concurrency=200)

    result = await scanner.scan_host('192.168.1.1')

    print(f"Scan results for {result['ip']}:")
    for port_info in result['ports']:
        service = scanner.identify_service(port_info['port'], port_info.get('banner'))
        print(f"  {port_info['port']}/tcp - {port_info['state']} - {service}")
        if port_info.get('banner'):
            print(f"    Banner: {port_info['banner'][:60]}...")

asyncio.run(main())"""
                },
                "pitfalls": ["Too aggressive scanning triggers IDS", "Firewall may give false positives", "UDP scanning is slow"],
                "concepts": ["TCP/UDP ports", "Port states", "Banner grabbing", "Service detection"],
                "estimated_hours": "8-12"
            },
            {
                "id": 3,
                "name": "Service Fingerprinting",
                "description": "Identify specific service versions through probing and response analysis.",
                "acceptance_criteria": [
                    "HTTP server identification (Server header, behavior)",
                    "SSH version detection",
                    "SSL/TLS certificate analysis",
                    "Database service identification",
                    "Match against known service signatures"
                ],
                "hints": {
                    "level1": "Send protocol-specific probes. HTTP HEAD request returns Server header. SSL cert has issuer info. Each service has unique response patterns.",
                    "level2": "Create signature database with regex patterns. SSL/TLS version affects vulnerability (SSLv3 = POODLE). Check supported ciphers for weak ones.",
                    "level3": """import ssl
import socket
import re
import asyncio

class ServiceFingerprint:
    def __init__(self, timeout=5):
        self.timeout = timeout

    async def probe_http(self, ip, port):
        '''Probe HTTP service'''
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=self.timeout
            )

            # Send HTTP request
            request = f"HEAD / HTTP/1.1\\r\\nHost: {ip}\\r\\nConnection: close\\r\\n\\r\\n"
            writer.write(request.encode())
            await writer.drain()

            response = await asyncio.wait_for(reader.read(4096), timeout=self.timeout)
            response = response.decode('utf-8', errors='ignore')

            writer.close()
            await writer.wait_closed()

            return self.parse_http_response(response)
        except Exception as e:
            return {'error': str(e)}

    def parse_http_response(self, response):
        '''Extract service info from HTTP response'''
        info = {}

        # Status line
        match = re.search(r'HTTP/(\\d\\.\\d) (\\d+)', response)
        if match:
            info['http_version'] = match.group(1)
            info['status_code'] = int(match.group(2))

        # Server header
        match = re.search(r'Server: ([^\\r\\n]+)', response, re.I)
        if match:
            info['server'] = match.group(1)
            info.update(self.parse_server_header(match.group(1)))

        # X-Powered-By
        match = re.search(r'X-Powered-By: ([^\\r\\n]+)', response, re.I)
        if match:
            info['powered_by'] = match.group(1)

        return info

    def parse_server_header(self, server):
        '''Parse detailed version from Server header'''
        info = {}

        patterns = [
            (r'Apache/(\\d+\\.\\d+\\.\\d+)', 'apache', 'version'),
            (r'nginx/(\\d+\\.\\d+\\.\\d+)', 'nginx', 'version'),
            (r'Microsoft-IIS/(\\d+\\.\\d+)', 'iis', 'version'),
            (r'PHP/(\\d+\\.\\d+\\.\\d+)', 'php', 'version'),
            (r'OpenSSL/(\\d+\\.\\d+\\.\\d+\\w*)', 'openssl', 'version'),
        ]

        for pattern, name, key in patterns:
            match = re.search(pattern, server, re.I)
            if match:
                info[name] = match.group(1)

        return info

    async def probe_ssl(self, ip, port):
        '''Probe SSL/TLS service'''
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port, ssl=context),
                timeout=self.timeout
            )

            # Get SSL info
            ssl_object = writer.get_extra_info('ssl_object')

            info = {
                'protocol': ssl_object.version(),
                'cipher': ssl_object.cipher(),
            }

            # Get certificate
            cert = ssl_object.getpeercert(binary_form=True)
            if cert:
                import cryptography.x509
                x509 = cryptography.x509.load_der_x509_certificate(cert)
                info['subject'] = x509.subject.rfc4514_string()
                info['issuer'] = x509.issuer.rfc4514_string()
                info['not_after'] = str(x509.not_valid_after)

                # Check for weak signature
                sig_algo = x509.signature_algorithm_oid._name
                if 'sha1' in sig_algo.lower() or 'md5' in sig_algo.lower():
                    info['weak_signature'] = True

            writer.close()
            await writer.wait_closed()

            return info
        except Exception as e:
            return {'error': str(e)}

    async def probe_ssh(self, ip, port=22):
        '''Probe SSH service'''
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=self.timeout
            )

            # SSH sends banner first
            banner = await asyncio.wait_for(reader.readline(), timeout=2)
            banner = banner.decode('utf-8', errors='ignore').strip()

            writer.close()
            await writer.wait_closed()

            info = {'banner': banner}

            # Parse SSH banner: SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.2
            match = re.match(r'SSH-(\\d\\.\\d)-(\\S+)', banner)
            if match:
                info['protocol'] = match.group(1)
                info['software'] = match.group(2)

                # Check for old/vulnerable versions
                if 'OpenSSH' in info['software']:
                    ver_match = re.search(r'OpenSSH_(\\d+\\.\\d+)', info['software'])
                    if ver_match:
                        version = float(ver_match.group(1))
                        if version < 7.0:
                            info['vulnerable'] = 'OpenSSH < 7.0 has known vulnerabilities'

            return info
        except Exception as e:
            return {'error': str(e)}

async def main():
    fp = ServiceFingerprint()

    # Probe different services
    http_info = await fp.probe_http('example.com', 80)
    print("HTTP:", http_info)

    ssl_info = await fp.probe_ssl('example.com', 443)
    print("SSL:", ssl_info)

asyncio.run(main())"""
                },
                "pitfalls": ["Services may hide version info", "Custom banners can mislead", "SSL probing may fail on SNI"],
                "concepts": ["Service fingerprinting", "Banner analysis", "SSL/TLS analysis"],
                "estimated_hours": "8-12"
            },
            {
                "id": 4,
                "name": "Vulnerability Detection",
                "description": "Check discovered services against known vulnerability databases.",
                "acceptance_criteria": [
                    "Query NVD/CVE database for version-based vulnerabilities",
                    "Check for common misconfigurations",
                    "Test for specific vulnerabilities (e.g., default credentials)",
                    "Generate vulnerability report with severity scores",
                    "Cache CVE data for offline use"
                ],
                "hints": {
                    "level1": "NVD has API for CVE lookup by product/version (CPE). Common checks: default creds, exposed admin panels, outdated TLS.",
                    "level2": "Build CPE string from fingerprint (cpe:2.3:a:vendor:product:version). Check CVSS score for severity. Some vulns need active testing.",
                    "level3": """import json
import sqlite3
import aiohttp
import asyncio
from datetime import datetime, timedelta

class VulnerabilityScanner:
    NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    def __init__(self, db_path='cve_cache.db'):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        '''Initialize CVE cache database'''
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cve_cache (
                cpe TEXT,
                cve_id TEXT,
                description TEXT,
                cvss_score REAL,
                severity TEXT,
                cached_at TIMESTAMP,
                PRIMARY KEY (cpe, cve_id)
            )
        ''')
        conn.commit()
        conn.close()

    async def query_nvd(self, cpe_string):
        '''Query NVD for CVEs affecting a CPE'''
        async with aiohttp.ClientSession() as session:
            params = {
                'cpeName': cpe_string,
                'resultsPerPage': 100
            }

            async with session.get(self.NVD_API, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self.parse_nvd_response(data)
                return []

    def parse_nvd_response(self, data):
        '''Parse NVD API response'''
        vulnerabilities = []

        for item in data.get('vulnerabilities', []):
            cve = item.get('cve', {})

            vuln = {
                'cve_id': cve.get('id'),
                'description': self.get_description(cve),
                'cvss_score': self.get_cvss_score(cve),
                'severity': self.get_severity(cve),
                'references': self.get_references(cve)
            }

            vulnerabilities.append(vuln)

        return vulnerabilities

    def get_description(self, cve):
        descriptions = cve.get('descriptions', [])
        for desc in descriptions:
            if desc.get('lang') == 'en':
                return desc.get('value', '')
        return ''

    def get_cvss_score(self, cve):
        metrics = cve.get('metrics', {})

        # Try CVSS 3.1, then 3.0, then 2.0
        for version in ['cvssMetricV31', 'cvssMetricV30', 'cvssMetricV2']:
            if version in metrics and metrics[version]:
                return metrics[version][0].get('cvssData', {}).get('baseScore', 0)

        return 0

    def get_severity(self, cve):
        score = self.get_cvss_score(cve)
        if score >= 9.0:
            return 'CRITICAL'
        elif score >= 7.0:
            return 'HIGH'
        elif score >= 4.0:
            return 'MEDIUM'
        elif score > 0:
            return 'LOW'
        return 'UNKNOWN'

    def get_references(self, cve):
        refs = cve.get('references', [])
        return [r.get('url') for r in refs[:5]]

    def build_cpe(self, service_info):
        '''Build CPE string from service fingerprint'''
        # CPE 2.3 format: cpe:2.3:a:vendor:product:version
        cpe_mappings = {
            'apache': 'cpe:2.3:a:apache:http_server:{version}',
            'nginx': 'cpe:2.3:a:nginx:nginx:{version}',
            'openssh': 'cpe:2.3:a:openbsd:openssh:{version}',
            'mysql': 'cpe:2.3:a:oracle:mysql:{version}',
            'postgresql': 'cpe:2.3:a:postgresql:postgresql:{version}',
            'php': 'cpe:2.3:a:php:php:{version}',
        }

        for key, template in cpe_mappings.items():
            if key in service_info:
                version = service_info[key]
                return template.format(version=version)

        return None

    async def check_common_vulns(self, ip, port, service_info):
        '''Check for common misconfigurations'''
        vulns = []

        # Default credentials check
        defaults = await self.check_default_creds(ip, port, service_info)
        vulns.extend(defaults)

        # SSL/TLS checks
        if service_info.get('protocol'):
            ssl_vulns = self.check_ssl_vulns(service_info)
            vulns.extend(ssl_vulns)

        # Outdated software
        if service_info.get('openssh'):
            version = float(service_info['openssh'].split('.')[0])
            if version < 7:
                vulns.append({
                    'type': 'outdated_software',
                    'description': f'OpenSSH {service_info["openssh"]} is outdated',
                    'severity': 'HIGH'
                })

        return vulns

    async def check_default_creds(self, ip, port, service_info):
        '''Check for default credentials'''
        vulns = []

        # Common default credential pairs
        defaults = [
            ('admin', 'admin'),
            ('admin', 'password'),
            ('root', 'root'),
            ('admin', ''),
        ]

        # Would implement actual login attempts here
        # This is a placeholder

        return vulns

    def check_ssl_vulns(self, ssl_info):
        '''Check for SSL/TLS vulnerabilities'''
        vulns = []

        protocol = ssl_info.get('protocol', '')

        # Check for deprecated protocols
        if 'SSLv3' in protocol:
            vulns.append({
                'type': 'ssl_vulnerability',
                'cve': 'CVE-2014-3566',
                'description': 'SSLv3 POODLE vulnerability',
                'severity': 'HIGH'
            })

        if 'TLSv1.0' in protocol or 'TLSv1.1' in protocol:
            vulns.append({
                'type': 'ssl_vulnerability',
                'description': f'{protocol} is deprecated',
                'severity': 'MEDIUM'
            })

        # Check cipher strength
        cipher = ssl_info.get('cipher', ('', '', 0))
        if cipher[2] < 128:  # Key length
            vulns.append({
                'type': 'weak_cipher',
                'description': f'Weak cipher: {cipher[0]} ({cipher[2]} bits)',
                'severity': 'HIGH'
            })

        # Weak signature algorithm
        if ssl_info.get('weak_signature'):
            vulns.append({
                'type': 'weak_signature',
                'description': 'Certificate uses weak signature algorithm (SHA1/MD5)',
                'severity': 'MEDIUM'
            })

        return vulns

    def generate_report(self, scan_results):
        '''Generate vulnerability report'''
        report = {
            'scan_time': datetime.now().isoformat(),
            'summary': {
                'hosts_scanned': len(scan_results),
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'findings': []
        }

        for host in scan_results:
            for vuln in host.get('vulnerabilities', []):
                severity = vuln.get('severity', 'UNKNOWN')
                if severity in report['summary']:
                    report['summary'][severity.lower()] += 1

                report['findings'].append({
                    'host': host['ip'],
                    'port': host.get('port'),
                    **vuln
                })

        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        report['findings'].sort(key=lambda x: severity_order.get(x.get('severity'), 99))

        return report"""
                },
                "pitfalls": ["NVD API rate limits", "CPE matching is imprecise", "False positives from version detection"],
                "concepts": ["CVE/NVD database", "CVSS scoring", "CPE naming", "Vulnerability assessment"],
                "estimated_hours": "10-15"
            },
            {
                "id": 5,
                "name": "Report Generation",
                "description": "Generate comprehensive scan reports in multiple formats.",
                "acceptance_criteria": [
                    "HTML report with summary dashboard",
                    "JSON export for automation",
                    "Group findings by severity",
                    "Include remediation suggestions",
                    "Executive summary for management"
                ],
                "hints": {
                    "level1": "Use Jinja2 for HTML templates. JSON is easy with json.dumps. Group vulns by host, then severity. Add links to CVE details.",
                    "level2": "Include CVSS scores and severity colors (red=critical, orange=high). Add remediation from CVE references or generic advice.",
                    "level3": """from jinja2 import Template
import json
from datetime import datetime

class ReportGenerator:
    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Vulnerability Scan Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { display: flex; gap: 20px; margin-bottom: 30px; }
        .stat-card { padding: 20px; border-radius: 8px; min-width: 120px; text-align: center; }
        .critical { background: #dc3545; color: white; }
        .high { background: #fd7e14; color: white; }
        .medium { background: #ffc107; color: black; }
        .low { background: #28a745; color: white; }
        .finding { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 4px; }
        .finding-critical { border-left: 4px solid #dc3545; }
        .finding-high { border-left: 4px solid #fd7e14; }
        .finding-medium { border-left: 4px solid #ffc107; }
        .finding-low { border-left: 4px solid #28a745; }
        .severity-badge { padding: 2px 8px; border-radius: 4px; font-size: 12px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f5f5f5; }
    </style>
</head>
<body>
    <h1>Vulnerability Scan Report</h1>
    <p>Generated: {{ scan_time }}</p>

    <h2>Executive Summary</h2>
    <div class="summary">
        <div class="stat-card critical">
            <h3>{{ summary.critical }}</h3>
            <p>Critical</p>
        </div>
        <div class="stat-card high">
            <h3>{{ summary.high }}</h3>
            <p>High</p>
        </div>
        <div class="stat-card medium">
            <h3>{{ summary.medium }}</h3>
            <p>Medium</p>
        </div>
        <div class="stat-card low">
            <h3>{{ summary.low }}</h3>
            <p>Low</p>
        </div>
    </div>

    <h2>Findings</h2>
    {% for finding in findings %}
    <div class="finding finding-{{ finding.severity|lower }}">
        <h3>
            <span class="severity-badge {{ finding.severity|lower }}">{{ finding.severity }}</span>
            {% if finding.cve_id %}{{ finding.cve_id }}{% else %}{{ finding.type }}{% endif %}
        </h3>
        <p><strong>Host:</strong> {{ finding.host }}{% if finding.port %}:{{ finding.port }}{% endif %}</p>
        <p><strong>Description:</strong> {{ finding.description }}</p>
        {% if finding.cvss_score %}
        <p><strong>CVSS Score:</strong> {{ finding.cvss_score }}</p>
        {% endif %}
        {% if finding.remediation %}
        <p><strong>Remediation:</strong> {{ finding.remediation }}</p>
        {% endif %}
        {% if finding.references %}
        <p><strong>References:</strong></p>
        <ul>
        {% for ref in finding.references %}
            <li><a href="{{ ref }}" target="_blank">{{ ref }}</a></li>
        {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endfor %}

    <h2>Hosts Scanned</h2>
    <table>
        <tr>
            <th>Host</th>
            <th>Open Ports</th>
            <th>Vulnerabilities</th>
        </tr>
        {% for host in hosts %}
        <tr>
            <td>{{ host.ip }}</td>
            <td>{{ host.ports|join(', ') }}</td>
            <td>{{ host.vuln_count }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
'''

    REMEDIATION_DB = {
        'outdated_software': 'Update to the latest stable version',
        'ssl_vulnerability': 'Disable SSLv3 and TLSv1.0/1.1, use TLSv1.2 or TLSv1.3',
        'weak_cipher': 'Configure server to use strong ciphers (AES-256-GCM, CHACHA20)',
        'weak_signature': 'Renew certificate with SHA-256 or stronger signature',
        'default_credentials': 'Change default credentials immediately',
    }

    def __init__(self, scan_results):
        self.results = scan_results

    def add_remediation(self, findings):
        '''Add remediation suggestions'''
        for finding in findings:
            vuln_type = finding.get('type', '')
            if vuln_type in self.REMEDIATION_DB:
                finding['remediation'] = self.REMEDIATION_DB[vuln_type]
            elif finding.get('cve_id'):
                finding['remediation'] = f"See CVE details: https://nvd.nist.gov/vuln/detail/{finding['cve_id']}"
        return findings

    def generate_html(self, output_path):
        '''Generate HTML report'''
        template = Template(self.HTML_TEMPLATE)

        # Prepare data
        findings = self.add_remediation(self.results.get('findings', []))

        # Group hosts
        hosts = {}
        for finding in findings:
            ip = finding['host']
            if ip not in hosts:
                hosts[ip] = {'ip': ip, 'ports': set(), 'vuln_count': 0}
            if finding.get('port'):
                hosts[ip]['ports'].add(finding['port'])
            hosts[ip]['vuln_count'] += 1

        for host in hosts.values():
            host['ports'] = sorted(host['ports'])

        html = template.render(
            scan_time=self.results.get('scan_time', datetime.now().isoformat()),
            summary=self.results.get('summary', {}),
            findings=findings,
            hosts=list(hosts.values())
        )

        with open(output_path, 'w') as f:
            f.write(html)

        return output_path

    def generate_json(self, output_path):
        '''Generate JSON report'''
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        return output_path

    def generate_executive_summary(self):
        '''Generate text executive summary'''
        summary = self.results.get('summary', {})
        findings = self.results.get('findings', [])

        total_vulns = sum([summary.get(s, 0) for s in ['critical', 'high', 'medium', 'low']])

        text = f\"\"\"
VULNERABILITY SCAN EXECUTIVE SUMMARY
====================================
Scan Date: {self.results.get('scan_time', 'N/A')}
Hosts Scanned: {summary.get('hosts_scanned', 0)}

RISK OVERVIEW:
- Critical: {summary.get('critical', 0)}
- High: {summary.get('high', 0)}
- Medium: {summary.get('medium', 0)}
- Low: {summary.get('low', 0)}
- Total: {total_vulns}

TOP FINDINGS:
\"\"\"
        # Add top 5 critical/high findings
        critical_high = [f for f in findings if f.get('severity') in ['CRITICAL', 'HIGH']][:5]
        for i, finding in enumerate(critical_high, 1):
            text += f\"\"\"
{i}. [{finding.get('severity')}] {finding.get('cve_id') or finding.get('type')}
   Host: {finding.get('host')}
   {finding.get('description', '')[:100]}...
\"\"\"

        return text

# Usage
def main():
    # Example scan results
    results = {
        'scan_time': datetime.now().isoformat(),
        'summary': {
            'hosts_scanned': 5,
            'critical': 2,
            'high': 5,
            'medium': 10,
            'low': 3
        },
        'findings': [
            {
                'host': '192.168.1.10',
                'port': 443,
                'severity': 'CRITICAL',
                'cve_id': 'CVE-2021-44228',
                'description': 'Log4j RCE vulnerability',
                'cvss_score': 10.0,
                'references': ['https://nvd.nist.gov/vuln/detail/CVE-2021-44228']
            },
            # ... more findings
        ]
    }

    gen = ReportGenerator(results)
    gen.generate_html('report.html')
    gen.generate_json('report.json')
    print(gen.generate_executive_summary())

if __name__ == '__main__':
    main()"""
                },
                "pitfalls": ["Large reports slow to generate", "Missing context for findings", "Overwhelming detail for executives"],
                "concepts": ["Report generation", "Data visualization", "Risk communication"],
                "estimated_hours": "6-10"
            }
        ]
    }
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    expert_projects = data.get('expert_projects', {})

    # Add missing projects
    for project_id, project_data in MISSING_PROJECTS.items():
        if project_id not in expert_projects:
            expert_projects[project_id] = project_data
            print(f"Added: {project_id}")
        else:
            print(f"Already exists: {project_id}")

    data['expert_projects'] = expert_projects

    # Save
    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"\nTotal expert_projects: {len(expert_projects)}")


if __name__ == "__main__":
    main()

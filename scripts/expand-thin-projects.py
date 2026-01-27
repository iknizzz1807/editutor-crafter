#!/usr/bin/env python3
"""
Add additional milestones to projects with only 2 milestones:
- kv-memory: Add LRU eviction, Persistence
- vector-clocks: Add Version Pruning, Distributed Integration
- wc-clone: Add Stdin support, Max line length
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

# Additional milestones for each project
additional_milestones = {
    "kv-memory": [
        {
            "id": 3,
            "name": "LRU Eviction",
            "description": "Implement least-recently-used eviction when memory limit is reached.",
            "acceptance_criteria": [
                "Set maximum memory limit",
                "Track access order for each key",
                "Evict LRU key when limit exceeded on insert",
                "Update access order on get (not just set)",
                "Handle edge case: new item larger than limit"
            ],
            "hints": {
                "level1": "Use a doubly linked list with hash map for O(1) access and removal.",
                "level2": "Move accessed nodes to front of list. Evict from tail.",
                "level3": """// LRU Cache with O(1) get/set
typedef struct LRUNode {
    char *key;
    void *value;
    size_t size;
    struct LRUNode *prev, *next;
} LRUNode;

typedef struct LRUCache {
    HashMap *map;           // key -> LRUNode*
    LRUNode *head, *tail;   // Doubly linked list
    size_t capacity;        // Max bytes
    size_t used;            // Current bytes
} LRUCache;

void move_to_front(LRUCache *c, LRUNode *node) {
    if (node == c->head) return;

    // Remove from current position
    if (node->prev) node->prev->next = node->next;
    if (node->next) node->next->prev = node->prev;
    if (node == c->tail) c->tail = node->prev;

    // Insert at front
    node->prev = NULL;
    node->next = c->head;
    if (c->head) c->head->prev = node;
    c->head = node;
    if (!c->tail) c->tail = node;
}

void evict_lru(LRUCache *c) {
    if (!c->tail) return;

    LRUNode *victim = c->tail;
    c->tail = victim->prev;
    if (c->tail) c->tail->next = NULL;
    else c->head = NULL;

    c->used -= victim->size;
    hm_delete(c->map, victim->key);
    free(victim->key);
    free(victim->value);
    free(victim);
}

void lru_set(LRUCache *c, const char *key, void *value, size_t size) {
    // Evict until we have space
    while (c->used + size > c->capacity && c->tail) {
        evict_lru(c);
    }

    // Check if updating existing
    LRUNode *existing = hm_get(c->map, key);
    if (existing) {
        c->used -= existing->size;
        free(existing->value);
        existing->value = value;
        existing->size = size;
        c->used += size;
        move_to_front(c, existing);
        return;
    }

    // Insert new
    LRUNode *node = malloc(sizeof(LRUNode));
    node->key = strdup(key);
    node->value = value;
    node->size = size;
    hm_set(c->map, key, node);

    // Add to front
    node->prev = NULL;
    node->next = c->head;
    if (c->head) c->head->prev = node;
    c->head = node;
    if (!c->tail) c->tail = node;

    c->used += size;
}"""
            },
            "pitfalls": [
                "Forgetting to update LRU order on reads",
                "Memory accounting errors (tracked vs actual)",
                "Thread safety with concurrent access",
                "Evicting during iteration"
            ],
            "concepts": [
                "LRU cache algorithm",
                "Doubly linked list",
                "O(1) cache operations",
                "Memory management"
            ],
            "estimated_hours": "3-4"
        },
        {
            "id": 4,
            "name": "Persistence (Snapshots)",
            "description": "Add ability to save and restore cache state from disk.",
            "acceptance_criteria": [
                "SAVE command writes snapshot to disk",
                "LOAD command restores from snapshot",
                "Binary format for efficiency",
                "Handle partial writes (atomic save)",
                "Background save (fork-based, optional)"
            ],
            "hints": {
                "level1": "Write header with count, then key-value pairs with lengths.",
                "level2": "Write to temp file, rename atomically. Use fixed-size header.",
                "level3": """// Simple binary snapshot format
// Header: magic(4) + version(4) + count(8)
// Entry: key_len(4) + key + value_len(4) + value + expires(8)

int save_snapshot(HashMap *m, const char *path) {
    char tmp_path[256];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    FILE *f = fopen(tmp_path, "wb");
    if (!f) return -1;

    // Header
    uint32_t magic = 0x4B564D53;  // "KVMS"
    uint32_t version = 1;
    uint64_t count = m->size;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&count, 8, 1, f);

    // Entries
    for (size_t i = 0; i < m->capacity; i++) {
        Entry *e = m->buckets[i];
        while (e) {
            uint32_t key_len = strlen(e->key);
            uint32_t val_len = e->value_size;

            fwrite(&key_len, 4, 1, f);
            fwrite(e->key, 1, key_len, f);
            fwrite(&val_len, 4, 1, f);
            fwrite(e->value, 1, val_len, f);
            fwrite(&e->expires, 8, 1, f);

            e = e->next;
        }
    }

    fclose(f);

    // Atomic rename
    if (rename(tmp_path, path) != 0) {
        unlink(tmp_path);
        return -1;
    }

    return 0;
}

int load_snapshot(HashMap *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    // Verify header
    uint32_t magic, version;
    uint64_t count;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);
    fread(&count, 8, 1, f);

    if (magic != 0x4B564D53 || version != 1) {
        fclose(f);
        return -1;
    }

    // Read entries
    for (uint64_t i = 0; i < count; i++) {
        uint32_t key_len, val_len;
        fread(&key_len, 4, 1, f);
        char *key = malloc(key_len + 1);
        fread(key, 1, key_len, f);
        key[key_len] = '\\0';

        fread(&val_len, 4, 1, f);
        void *value = malloc(val_len);
        fread(value, 1, val_len, f);

        time_t expires;
        fread(&expires, 8, 1, f);

        hm_set_with_expiry(m, key, value, val_len, expires);
        free(key);
    }

    fclose(f);
    return 0;
}"""
            },
            "pitfalls": [
                "Partial writes leaving corrupt file",
                "Endianness issues across platforms",
                "Large snapshots blocking main thread",
                "Not handling expired keys on load"
            ],
            "concepts": [
                "Binary file formats",
                "Atomic file operations",
                "Serialization",
                "Copy-on-write (fork)"
            ],
            "estimated_hours": "3-4"
        }
    ],

    "vector-clocks": [
        {
            "id": 3,
            "name": "Version Pruning",
            "description": "Implement strategies to limit unbounded growth of version history.",
            "acceptance_criteria": [
                "Configurable max versions per key",
                "Prune oldest versions when limit exceeded",
                "Optional: prune dominated versions",
                "Garbage collection of unreferenced clocks",
                "Metrics for version count"
            ],
            "hints": {
                "level1": "Track timestamp with each version. Prune by age or count.",
                "level2": "Dominated version: another version has higher clock values for all nodes.",
                "level3": """class PrunedVersionStore:
    def __init__(self, max_versions=10, max_age_seconds=3600):
        self.data = {}  # key -> list of (value, clock, timestamp)
        self.max_versions = max_versions
        self.max_age = max_age_seconds

    def put(self, key, value, clock):
        import time
        now = time.time()

        if key not in self.data:
            self.data[key] = []

        versions = self.data[key]

        # Remove dominated versions
        new_versions = []
        for v, vc, ts in versions:
            if not self._is_dominated(vc, clock):
                new_versions.append((v, vc, ts))

        # Add new version
        new_versions.append((value, clock.copy(), now))

        # Prune by age
        new_versions = [(v, vc, ts) for v, vc, ts in new_versions
                        if now - ts < self.max_age]

        # Prune by count (keep most recent)
        if len(new_versions) > self.max_versions:
            new_versions.sort(key=lambda x: x[2])  # Sort by timestamp
            new_versions = new_versions[-self.max_versions:]

        self.data[key] = new_versions

    def _is_dominated(self, clock1, clock2):
        '''Returns True if clock1 <= clock2 (clock2 dominates clock1)'''
        if len(clock1) != len(clock2):
            return False
        return all(c1 <= c2 for c1, c2 in zip(clock1, clock2))

    def get_stats(self):
        total_versions = sum(len(v) for v in self.data.values())
        return {
            'keys': len(self.data),
            'total_versions': total_versions,
            'avg_versions_per_key': total_versions / max(1, len(self.data))
        }"""
            },
            "pitfalls": [
                "Pruning too aggressively loses conflict info",
                "Clock comparison must handle different lengths",
                "Timestamp skew across nodes",
                "Memory leaks from orphaned clocks"
            ],
            "concepts": [
                "Version pruning strategies",
                "Garbage collection",
                "Dominated versions",
                "Trade-off: consistency vs storage"
            ],
            "estimated_hours": "2-3"
        },
        {
            "id": 4,
            "name": "Distributed Key-Value Store",
            "description": "Integrate vector clocks into a working distributed key-value store.",
            "acceptance_criteria": [
                "Multi-node setup (3+ nodes)",
                "PUT/GET operations with vector clocks",
                "Read-repair on conflict detection",
                "Configurable conflict resolution (LWW, merge, manual)",
                "Simple replication (all nodes store all keys)"
            ],
            "hints": {
                "level1": "Each node maintains its own vector clock. Attach clock to every write.",
                "level2": "On GET, collect from multiple nodes. If clocks differ, return all versions.",
                "level3": """import asyncio
import aiohttp
from aiohttp import web

class DistributedKVNode:
    def __init__(self, node_id, peers):
        self.node_id = node_id
        self.peers = peers  # list of (host, port)
        self.store = VersionedStore()
        self.clock = VectorClock(node_id, len(peers) + 1)

    async def put(self, key, value):
        '''Write to local and replicate to peers'''
        clock = self.clock.send()
        self.store.put(key, value, clock)

        # Async replicate to peers
        tasks = [self._replicate_to_peer(peer, key, value, clock)
                 for peer in self.peers]
        await asyncio.gather(*tasks, return_exceptions=True)

        return {'status': 'ok', 'clock': clock}

    async def _replicate_to_peer(self, peer, key, value, clock):
        url = f'http://{peer[0]}:{peer[1]}/internal/replicate'
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={
                'key': key, 'value': value, 'clock': clock
            })

    async def get(self, key, quorum=2):
        '''Read from multiple nodes, detect conflicts'''
        results = [self.store.get(key)]

        # Query peers
        tasks = [self._get_from_peer(peer, key) for peer in self.peers]
        peer_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in peer_results:
            if not isinstance(r, Exception) and r:
                results.append(r)

        # Merge all versions
        all_versions = []
        for versions in results:
            all_versions.extend(versions)

        # Remove dominated versions
        final = self._remove_dominated(all_versions)

        if len(final) > 1:
            return {'status': 'conflict', 'versions': final}
        elif len(final) == 1:
            return {'status': 'ok', 'value': final[0][0]}
        else:
            return {'status': 'not_found'}

    def _remove_dominated(self, versions):
        result = []
        for v1, c1 in versions:
            dominated = False
            for v2, c2 in versions:
                if c1 != c2 and self._is_dominated(c1, c2):
                    dominated = True
                    break
            if not dominated:
                result.append((v1, c1))
        return result

# HTTP handlers
routes = web.RouteTableDef()

@routes.post('/kv/{key}')
async def handle_put(request):
    key = request.match_info['key']
    data = await request.json()
    result = await node.put(key, data['value'])
    return web.json_response(result)

@routes.get('/kv/{key}')
async def handle_get(request):
    key = request.match_info['key']
    result = await node.get(key)
    return web.json_response(result)"""
            },
            "pitfalls": [
                "Network partitions causing divergence",
                "Replication lag during high write load",
                "Clock synchronization across restarts",
                "Handling node failures during writes"
            ],
            "concepts": [
                "Distributed replication",
                "Quorum reads/writes",
                "Read repair",
                "Eventual consistency"
            ],
            "estimated_hours": "4-6"
        }
    ],

    "wc-clone": [
        {
            "id": 3,
            "name": "Standard Input Support",
            "description": "Read from stdin when no file is specified, like real wc.",
            "acceptance_criteria": [
                "Read from stdin when no args",
                "Support piping: cat file | wc",
                "Mix stdin with files: wc - file1 file2",
                "Dash '-' means stdin explicitly",
                "Handle binary input gracefully"
            ],
            "hints": {
                "level1": "If argc == 1 or filename is '-', read from stdin instead of file.",
                "level2": "stdin doesn't have a filename to print - use empty or no label.",
                "level3": """// Handle both files and stdin
int count_stream(FILE *f, const char *name, Counts *totals) {
    Counts c = {0, 0, 0, 0};
    int in_word = 0;
    int ch;

    while ((ch = fgetc(f)) != EOF) {
        c.bytes++;
        if (ch == '\\n') c.lines++;
        if ((unsigned char)ch < 128 || ((ch & 0xC0) != 0x80)) {
            c.chars++;  // Start of UTF-8 char
        }
        if (isspace(ch)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            c.words++;
        }
    }

    print_counts(&c, name);

    totals->lines += c.lines;
    totals->words += c.words;
    totals->bytes += c.bytes;
    totals->chars += c.chars;

    return 0;
}

int main(int argc, char **argv) {
    // Parse options (skip for brevity)
    int optind = parse_options(argc, argv);

    Counts totals = {0};
    int num_files = argc - optind;

    if (num_files == 0) {
        // No files - read stdin
        count_stream(stdin, NULL, &totals);
    } else {
        for (int i = optind; i < argc; i++) {
            if (strcmp(argv[i], "-") == 0) {
                count_stream(stdin, NULL, &totals);
            } else {
                FILE *f = fopen(argv[i], "rb");
                if (!f) {
                    fprintf(stderr, "wc: %s: No such file\\n", argv[i]);
                    continue;
                }
                count_stream(f, argv[i], &totals);
                fclose(f);
            }
        }

        if (num_files > 1) {
            print_counts(&totals, "total");
        }
    }

    return 0;
}"""
            },
            "pitfalls": [
                "stdin can only be read once",
                "Binary data in stdin (null bytes)",
                "Blocking on empty stdin",
                "Mixing stdin position with files"
            ],
            "concepts": [
                "Standard streams",
                "Unix filter pattern",
                "Piping and redirection",
                "Special filename conventions"
            ],
            "estimated_hours": "1-2"
        },
        {
            "id": 4,
            "name": "Max Line Length (-L)",
            "description": "Add the -L option to report the maximum line length.",
            "acceptance_criteria": [
                "-L flag shows max line length only",
                "Combine with other flags",
                "Handle lines without newline (last line)",
                "UTF-8 aware: count display width, not bytes",
                "Tab expansion (optional)"
            ],
            "hints": {
                "level1": "Track current line length and max seen. Reset on newline.",
                "level2": "For UTF-8, count codepoints not bytes. Tabs count as 8-(col%8).",
                "level3": """// UTF-8 aware line length counting
int max_line_length(FILE *f) {
    int max_len = 0;
    int cur_len = 0;
    int ch;

    while ((ch = fgetc(f)) != EOF) {
        if (ch == '\\n') {
            if (cur_len > max_len) max_len = cur_len;
            cur_len = 0;
        } else if (ch == '\\t') {
            // Tab expands to next 8-column boundary
            cur_len = ((cur_len / 8) + 1) * 8;
        } else if ((ch & 0xC0) != 0x80) {
            // Start of UTF-8 char (or ASCII)
            // Could use wcwidth() for proper display width
            cur_len++;
        }
        // Continuation bytes (10xxxxxx) don't add to length
    }

    // Handle last line without newline
    if (cur_len > max_len) max_len = cur_len;

    return max_len;
}

// More accurate with wcwidth for CJK chars (double-width)
#include <wchar.h>
#include <locale.h>

int display_width(FILE *f) {
    setlocale(LC_ALL, "");  // Enable locale for wcwidth

    int max_len = 0;
    int cur_len = 0;
    wint_t wc;

    while ((wc = fgetwc(f)) != WEOF) {
        if (wc == L'\\n') {
            if (cur_len > max_len) max_len = cur_len;
            cur_len = 0;
        } else if (wc == L'\\t') {
            cur_len = ((cur_len / 8) + 1) * 8;
        } else {
            int w = wcwidth(wc);
            if (w > 0) cur_len += w;  // wcwidth returns -1 for non-printable
        }
    }

    if (cur_len > max_len) max_len = cur_len;
    return max_len;
}"""
            },
            "pitfalls": [
                "Forgetting last line without newline",
                "UTF-8 vs display width (CJK is double-width)",
                "Control characters and escape sequences",
                "Very long lines causing overflow"
            ],
            "concepts": [
                "Display width vs byte length",
                "Tab expansion",
                "wcwidth() for Unicode",
                "Terminal column counting"
            ],
            "estimated_hours": "1-2"
        }
    ]
}

# Load YAML
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

expert_projects = data.get('expert_projects', {})

# Add milestones
for project_id, milestones in additional_milestones.items():
    if project_id in expert_projects:
        existing = expert_projects[project_id].get('milestones', [])
        existing.extend(milestones)
        expert_projects[project_id]['milestones'] = existing
        print(f"Added {len(milestones)} milestones to {project_id} (now has {len(existing)})")
    else:
        print(f"WARNING: {project_id} not found in expert_projects")

# Save
with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nTotal expert_projects: {len(expert_projects)}")

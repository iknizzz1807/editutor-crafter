# EduTutor Crafter

> Project-based learning platform with AI review. Build real projects, get reviewed, level up.

## Vision

Part of the **editutor ecosystem**:

```
┌─────────────────────────────────────────────────────────┐
│                    LEARNING ECOSYSTEM                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────────────┐                                  │
│   │ editutor-crafter │ ← Roadmap + Projects             │
│   │     (this)       │   Step-by-step milestones        │
│   └────────┬─────────┘   AI code review                 │
│            │                                             │
│            ▼ build project                              │
│   ┌──────────────────┐                                  │
│   │   ai-editutor    │ ← Ask questions while coding     │
│   │    (plugin)      │   Learn in context               │
│   └────────┬─────────┘                                  │
│            │                                             │
│            ▼ knowledge saved                            │
│   ┌──────────────────┐                                  │
│   │ editutor-tracker │ ← Spaced repetition tests        │
│   │    (web app)     │   Reinforce learning             │
│   └────────┬─────────┘                                  │
│            │                                             │
│            └──────────► next project ───────────────────┘
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Core Concept

**Project-centric, not curriculum-centric.**

You pick projects that excite you. The system provides structure, milestones, and AI review to ensure you actually learn (not just copy-paste).

---

## Hierarchical Structure

```
Domain
└── Level (Beginner → Intermediate → Advanced → Expert)
    └── Projects (list)
        └── Milestones (sequential steps)
            └── Submissions (code + AI review)
```

---

## Example: Game Development

```
Game Development
│
├── 🟢 Beginner
│   ├── Pong Clone
│   │   ├── M1: Game loop & window
│   │   ├── M2: Paddle movement
│   │   ├── M3: Ball physics
│   │   └── M4: Scoring & win condition
│   │
│   ├── Snake Game
│   │   ├── M1: Grid rendering
│   │   ├── M2: Snake movement
│   │   ├── M3: Food & growth
│   │   └── M4: Collision & game over
│   │
│   └── Breakout
│       ├── M1: Bricks rendering
│       ├── M2: Ball & paddle
│       ├── M3: Collision detection
│       └── M4: Levels & powerups
│
├── 🟡 Intermediate
│   ├── Platformer
│   │   ├── M1: Tile-based level
│   │   ├── M2: Character controller
│   │   ├── M3: Gravity & jumping
│   │   ├── M4: Enemies & AI
│   │   └── M5: Camera follow
│   │
│   ├── Top-down Shooter
│   │   └── ...
│   │
│   └── Puzzle Game (Sokoban-style)
│       └── ...
│
├── 🟠 Advanced
│   ├── Software 3D Renderer
│   │   ├── M1: Line drawing (Bresenham)
│   │   ├── M2: Triangle rasterization
│   │   ├── M3: Z-buffer
│   │   ├── M4: Texture mapping
│   │   ├── M5: Lighting (Phong)
│   │   └── M6: Camera & transforms
│   │
│   ├── ECS Architecture
│   │   └── ...
│   │
│   └── Multiplayer Netcode
│       └── ...
│
└── 🔴 Expert
    ├── Full Game Engine
    │   ├── M1: Core architecture
    │   ├── M2: Rendering pipeline
    │   ├── M3: Physics integration
    │   ├── M4: Audio system
    │   ├── M5: Asset pipeline
    │   ├── M6: Scripting (Lua?)
    │   └── M7: Editor tools
    │
    ├── Physics Engine
    │   └── ...
    │
    └── Custom Shading Language
        └── ...
```

---

## Example: Systems Programming

```
Systems Programming
│
├── 🟢 Beginner
│   ├── Shell (basic)
│   ├── Cat/Grep clone
│   └── HTTP client
│
├── 🟡 Intermediate
│   ├── HTTP server
│   ├── Redis clone (basic)
│   └── SQLite clone (basic)
│
├── 🟠 Advanced
│   ├── Container runtime
│   ├── TCP/IP stack
│   └── Memory allocator
│
└── 🔴 Expert
    ├── Database engine
    ├── Distributed KV (Raft)
    └── OS kernel
```

---

## Example: AI / Machine Learning

```
AI / Machine Learning
│
├── 🟢 Beginner
│   ├── Linear regression from scratch
│   ├── KNN classifier
│   └── Decision tree
│
├── 🟡 Intermediate
│   ├── Neural network (micrograd-style)
│   ├── CNN for MNIST
│   └── Word embeddings
│
├── 🟠 Advanced
│   ├── Transformer from scratch
│   ├── RL agent (Q-learning)
│   └── GAN
│
└── 🔴 Expert
    ├── LLM training pipeline
    ├── Distributed training
    └── Custom autograd framework
```

---

## Example: Compilers & Languages

```
Compilers & Languages
│
├── 🟢 Beginner
│   ├── Calculator parser
│   ├── JSON parser
│   └── Regex engine (basic)
│
├── 🟡 Intermediate
│   ├── Interpreter (Lox)
│   ├── Lisp interpreter
│   └── Bytecode VM
│
├── 🟠 Advanced
│   ├── Compiler to assembly
│   ├── Garbage collector
│   └── JIT compiler
│
└── 🔴 Expert
    ├── LLVM frontend
    ├── Type system design
    └── Language server (LSP)
```

---

## Milestone Structure

Each milestone has:

```yaml
milestone:
  id: redis-01-ping-pong
  project: redis-clone
  name: "PING/PONG Protocol"

  description: |
    Implement basic Redis server that responds to PING command.

  # Clear, testable criteria
  acceptance_criteria:
    - Server listens on TCP port 6379
    - Responds to PING with +PONG\r\n (RESP protocol)
    - Handles multiple concurrent clients
    - Clean shutdown on SIGINT

  # Optional: automated tests
  tests:
    - command: "echo 'PING' | nc localhost 6379"
      expect: "+PONG"
    - command: "redis-benchmark -t ping -n 1000"
      expect: "exit_code: 0"

  # Hints (progressive reveal if stuck)
  hints:
    - "Look into Go's net.Listen for TCP"
    - "RESP protocol: https://redis.io/docs/reference/protocol-spec/"
    - "Use goroutines for concurrent clients"

  # Context for AI review
  review_focus:
    - Error handling approach
    - Concurrency model choice
    - Code organization
```

---

## Submit & Review Flow

```
┌─────────────────────────────────────────────────────────┐
│                    SUBMIT FLOW                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. User clicks "Submit Milestone"                      │
│         │                                                │
│         ▼                                                │
│  2. Extract project code                                │
│     ├── Tree structure                                  │
│     ├── Source files (smart selection, token budget)   │
│     └── Reuse ai-editutor context extraction logic     │
│         │                                                │
│         ▼                                                │
│  3. Run automated tests (if defined)                    │
│     ├── PASS → continue to AI review                   │
│     └── FAIL → instant feedback, no AI call needed     │
│         │                                                │
│         ▼                                                │
│  4. AI Review                                           │
│     ├── Check each acceptance criterion                │
│     ├── Code quality assessment                         │
│     ├── Architecture feedback                           │
│     └── Learning suggestions                            │
│         │                                                │
│         ▼                                                │
│  5. Verdict                                             │
│     ├── ACCEPT → unlock next milestone                 │
│     │           → generate concepts for tracker        │
│     └── REJECT → specific feedback                     │
│                 → must fix and resubmit                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## AI Review Prompt Template

```markdown
# Role
You are a senior engineer reviewing a milestone submission.
Be strict but educational. Reject if criteria not met.

# Context
Project: {{project.name}}
Milestone: {{milestone.name}}
Description: {{milestone.description}}

# Acceptance Criteria
{{#each milestone.acceptance_criteria}}
- [ ] {{this}}
{{/each}}

# Automated Test Results
{{test_results}}

# Submitted Code
## Project Structure
{{tree_structure}}

## Files
{{#each files}}
### {{this.path}}
```{{this.language}}
{{this.content}}
```
{{/each}}

# Your Task

## 1. Criteria Check
For each criterion, mark PASS or FAIL with brief explanation.

## 2. Verdict
- If ALL criteria pass → **ACCEPT**
- If ANY criterion fails → **REJECT**

## 3. Code Review (regardless of verdict)
- What's done well?
- What could be improved?
- Architecture observations
- Potential issues at scale

## 4. Learning Pointers
- Concepts to explore deeper
- Related topics for ai-editutor questions
- Resources if relevant

# Response Format
{
  "verdict": "ACCEPT" | "REJECT",
  "criteria_results": [
    {"criterion": "...", "status": "PASS|FAIL", "note": "..."}
  ],
  "feedback": {
    "strengths": ["..."],
    "improvements": ["..."],
    "concerns": ["..."]
  },
  "learning": {
    "concepts": ["...", "..."],
    "questions_to_explore": ["...", "..."]
  }
}
```

---

## Data Model

```
┌─────────────┐
│   Domain    │
├─────────────┤
│ id          │
│ name        │  "Game Development"
│ icon        │
│ description │
└──────┬──────┘
       │ has many
       ▼
┌─────────────┐
│    Level    │
├─────────────┤
│ id          │
│ domain_id   │
│ name        │  "Beginner" | "Intermediate" | "Advanced" | "Expert"
│ order       │  1, 2, 3, 4
│ color       │  green, yellow, orange, red
└──────┬──────┘
       │ has many
       ▼
┌─────────────┐
│   Project   │
├─────────────┤
│ id          │
│ level_id    │
│ name        │  "Pong Clone"
│ description │
│ order       │
│ status      │  locked | available | in_progress | completed
│ repo_path   │  local path to project code
└──────┬──────┘
       │ has many
       ▼
┌─────────────┐
│  Milestone  │
├─────────────┤
│ id          │
│ project_id  │
│ name        │  "Ball physics"
│ description │
│ criteria[]  │  acceptance criteria
│ hints[]     │  progressive hints
│ tests[]     │  automated test commands
│ order       │
│ status      │  locked | available | submitted | passed
└──────┬──────┘
       │ has many
       ▼
┌─────────────┐
│ Submission  │
├─────────────┤
│ id          │
│ milestone_id│
│ code        │  JSON snapshot (tree + files)
│ test_result │  automated test output
│ ai_review   │  JSON response from AI
│ verdict     │  ACCEPT | REJECT
│ created_at  │
└─────────────┘
```

---

## Unlock Logic

```
Level unlock:
├── Beginner: always unlocked
├── Intermediate: complete ≥2 Beginner projects in domain
├── Advanced: complete ≥2 Intermediate projects in domain
└── Expert: complete ≥2 Advanced projects in domain

Project unlock:
├── First project in level: auto unlocked when level unlocked
└── Others: complete ≥1 project in same level

Milestone unlock:
└── Sequential within project (must pass M1 → M2 → M3...)
```

Alternative: Flexible mode - everything unlocked, system only **recommends** order.

---

## UI Wireframes

### Domain Overview

```
┌─────────────────────────────────────────────────────────┐
│  🎮 Game Development                    [12/28 done]    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  🟢 Beginner ████████████░░ 3/4 projects                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │  Pong   │ │  Snake  │ │Breakout │ │ Tetris  │       │
│  │   ✓     │ │   ✓     │ │   ✓     │ │  🔒    │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│                                                          │
│  🟡 Intermediate ████░░░░░░ 1/3 projects                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
│  │Platform │ │ Shooter │ │ Puzzle  │                   │
│  │  ⏳ 3/5 │ │   ○     │ │   ○     │                   │
│  └─────────┘ └─────────┘ └─────────┘                   │
│                                                          │
│  🟠 Advanced 🔒 (complete 2 intermediate to unlock)     │
│                                                          │
│  🔴 Expert 🔒                                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Project Detail

```
┌─────────────────────────────────────────────────────────┐
│  ← Back    Platformer                    🟡 Intermediate │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Progress: ████████░░░░░░░░ 3/5 milestones              │
│                                                          │
│  ✓ M1: Tile-based level                                 │
│  ✓ M2: Character controller                             │
│  ✓ M3: Gravity & jumping                                │
│  ⏳ M4: Enemies & AI              [Submit for Review]   │
│  🔒 M5: Camera follow                                    │
│                                                          │
│  ─────────────────────────────────────────────────────  │
│                                                          │
│  Current: M4 - Enemies & AI                             │
│                                                          │
│  Description:                                            │
│  Implement enemy NPCs with basic patrol AI and          │
│  player interaction (damage on contact, defeat by       │
│  jumping on head).                                       │
│                                                          │
│  Acceptance Criteria:                                    │
│  • Enemy spawns and patrols between two points         │
│  • Player takes damage on side contact                  │
│  • Enemy defeated when player jumps on head            │
│  • At least 2 different enemy types                    │
│                                                          │
│  [View Hints]  [Open Project Folder]  [Submit Code]     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Submission Review

```
┌─────────────────────────────────────────────────────────┐
│  Review Result                              ✓ ACCEPTED  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Criteria Results:                                       │
│  ✓ Enemy spawns and patrols                             │
│  ✓ Player takes damage on contact                       │
│  ✓ Enemy defeated by jumping                            │
│  ✓ 2 enemy types implemented                            │
│                                                          │
│  ─────────────────────────────────────────────────────  │
│                                                          │
│  Strengths:                                              │
│  • Clean state machine for enemy AI                     │
│  • Good separation of enemy types via inheritance      │
│                                                          │
│  Suggestions:                                            │
│  • Consider using composition over inheritance         │
│  • Patrol points could be data-driven (from tilemap)   │
│                                                          │
│  Concepts to Explore:                                    │
│  • State pattern for game AI                            │
│  • Behavior trees                                        │
│  • Entity Component System (ECS)                        │
│                                                          │
│  [Continue to M5: Camera Follow]                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack (Proposed)

```
Frontend: React + Vite (consistent with editutor-tracker)
Backend: Go + Gin (consistent with editutor-tracker)
Database: SQLite (simple, local-first)
AI: Gemini API (or configurable)
Code extraction: Port ai-editutor logic (Lua → Go)
```

---

## Integration Points

### With ai-editutor

```
Platform                          ai-editutor
   │                                   │
   │  User builds project              │
   │                                   │
   │  ──────── questions ───────────>  │
   │                                   │
   │  <─────── knowledge.json ───────  │
   │                                   │
   │  Platform reads knowledge to      │
   │  understand what user struggled   │
   │  with during this milestone       │
   │                                   │
```

### With editutor-tracker

```
Platform                          Tracker
   │                                   │
   │  AI review generates              │
   │  "concepts to reinforce"          │
   │                                   │
   │  ──────── concepts ────────────>  │
   │                                   │
   │  Tracker creates tests            │
   │  for those concepts               │
   │                                   │
   │  <─────── test results ─────────  │
   │                                   │
   │  Platform sees mastery level      │
   │                                   │
```

---

## MVP Scope

### Phase 1: Core
- [ ] Domain/Level/Project/Milestone data model
- [ ] Basic UI: browse domains, projects, milestones
- [ ] Submit milestone: extract code (tree + files)
- [ ] AI review: call Gemini, parse response
- [ ] Pass/fail logic, unlock next milestone

### Phase 2: Content
- [ ] Populate 2-3 domains with real projects
- [ ] Write detailed milestones with criteria
- [ ] Add hints for common stuck points

### Phase 3: Integration
- [ ] Read ai-editutor knowledge.json
- [ ] Push concepts to tracker
- [ ] Unified progress dashboard

---

## Open Questions

1. **Local-first or cloud?** Store submissions locally or sync to cloud?
2. **Project templates?** Provide starter code or fully from scratch?
3. **Community?** Eventually allow sharing projects/milestones?
4. **Gamification level?** XP, levels, streaks like tracker?

---

## Resources

Project ideas sources:
- [Build Your Own X](https://github.com/codecrafters-io/build-your-own-x)
- [Codecrafters](https://codecrafters.io)
- [Crafting Interpreters](https://craftinginterpreters.com)
- [Handmade Hero](https://handmadehero.org)
- [tinyrenderer](https://github.com/ssloy/tinyrenderer)
- [Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [MIT 6.824 Labs](https://pdos.csail.mit.edu/6.824/)
- [roadmap.sh](https://roadmap.sh)

# AUDIT & FIX: topdown-shooter

## CRITIQUE
- **Technical Inaccuracy (Confirmed):** The essence claims 'AI behavior trees' but the actual implementation is trivial direct-chase AI. Behavior trees are a specific, non-trivial architecture (selector/sequence/decorator nodes). The essence overpromises dramatically. Either implement actual behavior trees or correct the essence.
- **Memory Management Gap (Confirmed & Expanded):** M2 lists 'Object pooling' as a concept but the AC doesn't require it—it just says 'remove off-screen projectiles.' Without a pool, each fire creates a new allocation and each removal is a deallocation, causing GC pressure or fragmentation. The AC should explicitly require an object pool with measurable reuse.
- **Diagonal Speed Boost:** M1 mentions it as a pitfall but the AC doesn't require normalization. The AC says 'supports eight directions' without specifying that diagonal movement must be normalized to prevent √2 speed boost. This is a verification gap.
- **Missing Concept:** No mention of screen-to-world coordinate transformation for mouse aiming. The AC says 'mouse aiming rotates player sprite' but doesn't address coordinate space conversion, which is a common source of bugs.
- **M3 'Pathfinding & Navigation' Skill:** Listed as a skill but never actually required in any AC. Direct chase toward player position is not pathfinding.
- **M3 Enemy Stacking:** Listed as a pitfall but no AC addresses separation forces or collision between enemies. Enemies will pile up into a single point.
- **M4 High Score Persistence:** AC says 'persists across sessions' but doesn't specify mechanism (localStorage, file I/O). This is a non-trivial cross-cutting concern.
- **Missing Game State:** No pause, game-over screen, or restart flow defined anywhere.
- **Missing Health System for Player:** M3 introduces enemy-player collision and 'contact damage' but the player health system is never formally established with AC.

## FIXED YAML
```yaml
id: topdown-shooter
name: Top-down Shooter
description: "Top-down arena shooter with enemies, projectiles, wave-based progression, and object pooling."
difficulty: intermediate
estimated_hours: "25-35"
essence: >
  Real-time AABB and circle collision detection between dynamic entities,
  finite-state-machine-based enemy AI with chase and attack behaviors,
  object pooling for efficient projectile memory management, and
  delta-time-based physics in a constrained 2D arena.
why_important: >
  Building this teaches core real-time game programming fundamentals—
  spatial reasoning, frame-independent movement, object lifecycle
  management, and state machines—that translate directly to professional
  game development and interactive simulation systems.
learning_outcomes:
  - Implement AABB and circle collision detection between dynamic entities
  - Design enemy AI using finite state machines with chase and attack states
  - Build an object pool for projectiles to eliminate per-frame allocation
  - Implement delta-time physics for frame-independent movement
  - Normalize diagonal movement vectors to prevent speed exploits
  - Create wave-based spawning with difficulty progression
  - Handle mouse-to-world coordinate transformation for aiming
  - Implement player health, damage, and invincibility frame systems
  - Optimize game loop performance and entity update cycles
skills:
  - Collision Detection (AABB, Circle)
  - Finite State Machine AI
  - Object Pooling
  - Delta-Time Physics
  - Vector Mathematics
  - Game Loop Architecture
  - Input Handling
  - Game State Management
tags:
  - c#
  - collision
  - enemies
  - game-dev
  - intermediate
  - javascript
  - projectiles
  - python
  - sprites
architecture_doc: architecture-docs/topdown-shooter/index.md
languages:
  recommended:
    - JavaScript
    - Python
    - "C#"
  also_possible:
    - C++
    - Lua
resources:
  - name: Game Programming Patterns
    url: https://gameprogrammingpatterns.com/
    type: book
  - name: 2D Collision Detection - MDN Web Docs
    url: https://developer.mozilla.org/en-US/docs/Games/Techniques/2D_collision_detection
    type: documentation
  - name: Object Pool Pattern
    url: https://gameprogrammingpatterns.com/object-pool.html
    type: book
prerequisites:
  - type: skill
    name: Basic game loop implementation
  - type: skill
    name: 2D graphics rendering
  - type: skill
    name: Vector math (dot product, normalization)
milestones:
  - id: topdown-shooter-m1
    name: "Player Movement, Aiming & Health"
    description: >
      Implement player with 8-directional WASD movement, mouse aiming,
      and a health system that will receive damage from enemies later.
    acceptance_criteria:
      - "WASD movement supports eight directions including diagonal combinations"
      - "Diagonal movement vectors are normalized to prevent √2 speed boost (measurable: diagonal speed equals cardinal speed ±1%)"
      - "Mouse position is converted from screen coordinates to world coordinates for aiming"
      - "Player sprite rotates to face the cursor position in real time using atan2"
      - "Movement uses acceleration and deceleration curves (not instant velocity)"
      - "Player cannot move outside the arena boundaries (screen or defined play area)"
      - "Player has a health value (e.g., 100 HP) displayed in a HUD element"
      - "All movement is delta-time scaled and frame-rate independent"
    pitfalls:
      - "Diagonal speed √2 boost from non-normalized input vector"
      - "atan2 argument order wrong (atan2(dy, dx) not atan2(dx, dy))"
      - "Angle in degrees vs radians mismatch in rotation"
      - "Jittery movement from integer truncation of position"
      - "Screen-to-world coordinate mismatch when canvas is scaled or offset"
    concepts:
      - 8-directional movement with normalization
      - Mouse-to-world coordinate transformation
      - atan2 for angle calculation
      - Delta-time movement
    skills:
      - 2D vector mathematics and normalization
      - Input handling and event systems
      - Coordinate space conversion
      - Player controller implementation
      - HUD rendering basics
    deliverables:
      - "WASD movement handler with diagonal normalization"
      - "Mouse aiming with screen-to-world coordinate conversion"
      - "Player sprite rotation toward cursor"
      - "Movement with acceleration/deceleration curves"
      - "Player health system with HUD display"
      - "Arena boundary clamping"
    estimated_hours: "3-4"

  - id: topdown-shooter-m2
    name: "Projectile System with Object Pool"
    description: >
      Implement a projectile system using an object pool to manage bullet
      lifecycle without per-frame allocation. Include fire-rate limiting
      and off-screen culling.
    acceptance_criteria:
      - "Click fires a projectile from the player's weapon position in the aimed direction"
      - "Projectile pool pre-allocates a fixed number of bullet objects (e.g., 200)"
      - "Firing retrieves an inactive bullet from the pool; no new allocation occurs"
      - "Bullets travel at constant speed in their fired direction, scaled by delta time"
      - "Fire rate limiter enforces minimum cooldown (e.g., 100ms) between shots"
      - "Bullets leaving the arena bounds are returned to the pool (deactivated, not destroyed)"
      - "Pool utilization is measurable: active count and pool capacity are queryable"
      - "Projectile spawn position is offset from player center to weapon tip (not center of player)"
    pitfalls:
      - "Spawning bullet at player center causes it to visually pop out of the body"
      - "Not resetting bullet state (position, velocity) when recycling from pool causes ghost bullets"
      - "Fire rate timer using wall-clock time instead of game time breaks on pause"
      - "Pool exhaustion: firing when all bullets are active silently fails or crashes"
      - "Memory growing unbounded if pool is bypassed and new objects are allocated on overflow"
    concepts:
      - Object pooling pattern
      - Projectile physics
      - Fire rate limiting
      - Entity lifecycle management
    skills:
      - Object pool design and implementation
      - Collision detection setup
      - Timer-based cooldown systems
      - Entity state management (active/inactive)
    deliverables:
      - "Object pool pre-allocating bullet instances with acquire/release API"
      - "Projectile spawning from pool at weapon offset position"
      - "Projectile movement updated each frame along trajectory"
      - "Fire rate limiter with configurable cooldown"
      - "Off-screen bullet deactivation returning to pool"
    estimated_hours: "3-5"

  - id: topdown-shooter-m3
    name: "Enemies with FSM AI"
    description: >
      Add enemies with finite-state-machine AI (Idle, Chase, Attack states),
      health systems, and separation forces to prevent stacking.
    acceptance_criteria:
      - "Enemies have a finite state machine with at least: Idle, Chase, and Attack states"
      - "Idle state: enemy stands still or wanders until player enters detection radius"
      - "Chase state: enemy moves toward player position, activated when player is within detection range"
      - "Attack state: enemy performs attack action (melee contact or ranged) when within attack range"
      - "At least 2 enemy types with different stats (speed, health, damage, detection range)"
      - "Enemy health system: takes damage from projectile hits, dies at 0 HP with death effect"
      - "Projectile-enemy collision: bullet deals damage and is returned to pool"
      - "Enemy-player collision: deals contact damage to player; player receives brief invincibility frames"
      - "Enemy separation force prevents enemies from overlapping each other (enemies push apart)"
      - "Division-by-zero protection in direction normalization when enemy is at player position"
    pitfalls:
      - "Division by zero when normalizing zero-length vector (enemy at exact player position)"
      - "All enemies stacking on same pixel without separation forces"
      - "Enemies spawning on top of the player causing instant damage"
      - "FSM state transitions not checked every frame—enemy stuck in stale state"
      - "Invincibility frames not implemented—player takes damage every frame from contact"
    concepts:
      - Finite state machine AI
      - Entity separation forces
      - Health and damage systems
      - Invincibility frames
    skills:
      - State machine implementation
      - Basic steering behaviors
      - Health/damage system design
      - Collision response differentiation
    deliverables:
      - "Enemy FSM with Idle, Chase, Attack states and configurable transition thresholds"
      - "At least 2 enemy types with different stat profiles"
      - "Enemy health system with damage-on-hit and death"
      - "Enemy separation force preventing overlap"
      - "Player invincibility frames after taking damage"
      - "Projectile-enemy collision with pool return"
    estimated_hours: "5-7"

  - id: topdown-shooter-m4
    name: "Waves, Scoring & Game State"
    description: >
      Implement wave-based enemy spawning with difficulty progression,
      scoring, and complete game state management (playing, paused,
      game over, restart).
    acceptance_criteria:
      - "Waves spawn groups of enemies at random positions outside the visible arena"
      - "Rest period between waves (e.g., 3-5 seconds) with wave announcement UI"
      - "Difficulty scales each wave: increase enemy count, speed, and/or health by a defined formula"
      - "Score increments per enemy kill, with different enemy types worth different point values"
      - "Wave clear bonus awards extra points when all enemies in a wave are eliminated"
      - "Game state machine manages: Playing, Paused, Game Over states"
      - "Game over triggers when player HP reaches 0; displays final score and wave reached"
      - "High score persists across sessions using localStorage or file I/O"
      - "Restart resets all game state (player HP, score, wave, enemies, projectiles) without reload"
      - "HUD displays: health, score, current wave, and enemy count"
    pitfalls:
      - "Infinite spawn loop if wave completion check is wrong"
      - "No rest between waves—player is overwhelmed immediately"
      - "Difficulty scaling too aggressive—game impossible by wave 5"
      - "High score persistence failing silently (no error on storage access failure)"
      - "Restart not clearing object pool state—ghost bullets from previous game appear"
      - "Enemies from previous wave still alive when new wave spawns"
    concepts:
      - Wave-based spawning systems
      - Difficulty progression curves
      - Game state machines
      - Persistent storage
    skills:
      - Game state management
      - Progression system design
      - UI/HUD implementation
      - Local persistence (localStorage/file I/O)
    deliverables:
      - "Wave spawning system with configurable enemy composition per wave"
      - "Difficulty scaling formula applied each wave"
      - "Scoring system with per-enemy-type values and wave clear bonus"
      - "Game state machine: Playing → Paused → Game Over → Restart"
      - "High score persistence across sessions"
      - "HUD displaying health, score, wave number, enemies remaining"
    estimated_hours: "4-6"

```
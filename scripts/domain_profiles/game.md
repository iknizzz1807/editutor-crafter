# DOMAIN PROFILE: Game Development
# Applies to: game-dev
# Projects: platformer, shooter, ray tracer, ECS, 3D renderer, game engine, Vulkan, etc.

## Fundamental Tension Type
Real-time budget constraints. VISUAL FIDELITY (more polygons, better lighting) vs FRAME RATE (16.67ms for 60fps, 8.33ms for 120fps). Every technique is about fitting more work into impossibly tight time.

Secondary: physics accuracy vs speed (continuous vs discrete), client authority vs server (responsive vs cheat-proof), ECS flexibility vs cache perf, draw calls vs batching.

## Three-Level View
- **Level 1 — Game Logic**: Entity behavior, rules, state machines, events
- **Level 2 — Engine Systems**: Render pipeline, physics, audio, input, resource management
- **Level 3 — Hardware**: GPU pipeline (vertex→raster→fragment→composite), CPU cache for ECS, audio DMA

Note: Level 3 hardware MANDATORY for rendering projects (ray tracer, 3D renderer, Vulkan, game engine). OPTIONAL for gameplay projects (platformer, shooter) — focus Levels 1+2.

## Soul Section: "Frame Budget Soul"
- Milliseconds per frame for this system? % of 16.67ms budget?
- Over budget → frame drop? skip physics? reduce quality?
- Parallelizable? (physics thread 2, GPU rendering, audio thread 3)
- Memory access pattern? Sequential (prefetch) or random (cache-hostile)?
- LOD possible? (distant → fewer polygons, simplified physics)
- Entity count scaling? Budget before this becomes bottleneck?

## Alternative Reality Comparisons
Unity (ECS/DOTS), Unreal (Nanite, Lumen), Godot (scene tree), Bevy (Rust ECS), Box2D/Rapier (physics), Bullet (3D physics), FMOD/Wwise (audio), Vulkan/DX12/Metal, OpenGL, Source Engine (netcode), Quake (BSP, netcode).

## TDD Emphasis
- Game loop: MANDATORY — fixed vs variable timestep, update order, frame timing
- Component layouts: MANDATORY for ECS — struct-of-arrays, archetype storage
- Render pipeline: MANDATORY for graphics — vertex, raster, fragment, post-process
- Network protocol: MANDATORY for multiplayer — message types, tick rate, interpolation, prediction
- Physics spec: integrator (Euler/Verlet/RK4), collision phases (broad/narrow)
- Memory layout: YES for ECS arrays, vertex buffers, spatial nodes. NO for game logic classes.
- Cache line: YES for hot-loop data (ECS, particles). NO for cold (menus, saves).
- Benchmarks: FPS at target entities, draw calls/frame, physics step time, input-to-screen latency

## Cross-Domain Notes
Borrow from systems-lowlevel when: GPU pipeline internals, cache optimization, SIMD for physics/rendering.
Borrow from distributed when: multiplayer server architecture, authoritative server, lag compensation.
Borrow from ai-ml when: NPC behavior (behavior trees, ML agents), procedural generation.

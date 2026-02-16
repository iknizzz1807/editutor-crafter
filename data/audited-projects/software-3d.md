# AUDIT & FIX: software-3d

## CRITIQUE
- **Critical Missing Step (Confirmed):** Near-plane clipping is completely absent. When a triangle has one or more vertices behind the camera (w ≤ 0 after projection), the perspective divide produces inverted or infinite coordinates. This doesn't just look wrong—it crashes the rasterizer or produces garbage pixels spanning the entire screen. Near-plane clipping must occur between projection and rasterization.
- **Dependency Issue (Partially Confirmed):** The audit says perspective projection requires the depth buffer. This is partially true—you can render wireframe or flat-shaded without a z-buffer, but any scene with overlapping geometry will produce incorrect results without it. The milestones should be restructured so basic wireframe rendering works in M3, and the z-buffer is added immediately when doing filled triangles.
- **M1 Scope Creep:** M1 requires Bresenham's, Cohen-Sutherland clipping, AND Xiaolin Wu anti-aliasing. That's three distinct algorithms in one milestone. Wu's algorithm is not needed for a software renderer foundation and adds unnecessary complexity.
- **M2 Missing Fill Rule:** The fill rules concept is mentioned but not in the AC. Without a consistent fill rule (top-left rule), adjacent triangles will either have gaps or double-drawn pixels along shared edges. This is a critical rasterization correctness requirement.
- **M3 Missing Homogeneous Divide:** The AC mentions perspective projection but doesn't explicitly require the perspective divide (dividing by w). This is not the same as the projection matrix—it's a separate step that's commonly forgotten.
- **M3 Missing Viewport Transform:** After perspective divide, normalized device coordinates must be mapped to screen pixels. This is absent.
- **Missing Wireframe Mode:** There's no intermediate step between line drawing (M1) and filled triangles (M2). Wireframe rendering of 3D models is a natural validation step.
- **M4 Estimated Hours:** 8-12 hours for z-buffer AND two shading models is aggressive. These are conceptually distinct and should be separable.
- **Missing Backface Culling:** Never mentioned anywhere, but it's a fundamental optimization and correctness step for closed meshes.
- **Missing OBJ Loading:** No milestone addresses loading actual 3D models. Without this, the renderer can only show hardcoded geometry.

## FIXED YAML
```yaml
id: software-3d
name: Software 3D Renderer
description: "Pure-software scanline rasterizer implementing the core GPU rendering pipeline with no graphics API."
difficulty: advanced
estimated_hours: "40-60"
essence: >
  Scanline-based rasterization of triangles through matrix-driven
  model-view-projection coordinate transformations, homogeneous clipping,
  perspective divide, viewport mapping, z-buffer occlusion testing,
  and per-vertex attribute interpolation—implementing the core algorithms
  that underlie GPU rendering pipelines entirely in software.
why_important: >
  Building this demystifies the mathematics and algorithms underlying
  modern GPU pipelines, teaching fundamental 3D graphics concepts that
  apply to game engines, visualization tools, shader programming, and
  graphics API debugging. You will understand every step from vertex
  to pixel.
learning_outcomes:
  - Implement Bresenham's line algorithm for pixel-perfect rasterization
  - Design and apply model-view-projection matrix transformations
  - Implement homogeneous near-plane clipping to handle geometry behind the camera
  - Perform perspective divide and viewport transformation
  - Build scanline-based triangle rasterization with top-left fill rule
  - Implement z-buffer algorithm for depth-based visibility
  - Implement backface culling using surface normals
  - Develop flat and Gouraud shading with diffuse lighting
  - Load and render 3D models from OBJ files
  - Debug floating-point precision issues in homogeneous coordinates
skills:
  - Linear Algebra
  - Matrix Transformations
  - Rasterization Algorithms
  - Depth Buffering
  - Homogeneous Coordinates
  - Geometric Clipping
  - Memory Management
  - 3D Model Loading
tags:
  - 3d-graphics
  - advanced
  - c
  - c++
  - matrices
  - rasterization
  - rendering
  - rust
architecture_doc: architecture-docs/software-3d/index.md
languages:
  recommended:
    - C
    - C++
    - Rust
  also_possible:
    - Python
    - JavaScript
resources:
  - name: tinyrenderer
    url: https://github.com/ssloy/tinyrenderer/wiki
    type: tutorial
  - name: 3D Math Primer
    url: https://gamemath.com/
    type: book
  - name: Scratchapixel - Rasterization
    url: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation
    type: tutorial
prerequisites:
  - type: skill
    name: "Linear algebra (matrices, vectors, dot/cross product)"
  - type: skill
    name: Basic trigonometry
  - type: skill
    name: "2D graphics basics (pixel buffer, drawing to screen)"
milestones:
  - id: software-3d-m1
    name: "Framebuffer & Line Drawing"
    description: >
      Set up a pixel framebuffer and implement Bresenham's line algorithm
      to draw lines between any two points on screen.
    acceptance_criteria:
      - "Framebuffer allocates width×height pixel buffer writable to screen or image file"
      - "Bresenham's line algorithm draws lines between any two integer-coordinate points"
      - "All 8 octants handled correctly, including horizontal, vertical, and diagonal lines"
      - "Algorithm uses integer-only arithmetic with no floating-point operations"
      - "Lines are clipped to framebuffer bounds (pixels outside buffer are not written)"
      - "Visual test: wireframe triangle rendered from 3 line segments"
    pitfalls:
      - "Missing octant handling causes lines in certain directions to render incorrectly"
      - "Integer overflow on large coordinate differences"
      - "Writing pixels outside framebuffer bounds causes memory corruption"
      - "Forgetting to handle the degenerate case of a single-point line (start == end)"
    concepts:
      - Rasterization fundamentals
      - Bresenham's algorithm
      - Pixel framebuffers
    skills:
      - Integer arithmetic optimization
      - Pixel coordinate systems
      - Framebuffer management
    deliverables:
      - "Framebuffer with set_pixel(x, y, color) and save/display interface"
      - "Bresenham's line algorithm supporting all octants"
      - "Bounds checking or clipping for out-of-range pixels"
      - "Visual test: wireframe shapes drawn from line segments"
    estimated_hours: "3-4"

  - id: software-3d-m2
    name: "3D Transformations & Wireframe Rendering"
    description: >
      Implement 4×4 matrix math, model-view-projection pipeline, perspective
      divide, viewport transform, and render a wireframe 3D model (loaded
      from OBJ file or hardcoded).
    acceptance_criteria:
      - "4×4 matrix multiplication, identity, and inverse are implemented correctly"
      - "Model matrix supports translation, rotation (X/Y/Z), and uniform scaling"
      - "View matrix implements look-at camera (eye, target, up)"
      - "Perspective projection matrix produces correct clip-space coordinates with configurable FOV, aspect, near, far"
      - "Perspective divide converts clip-space (x,y,z,w) to NDC by dividing by w"
      - "Viewport transform maps NDC [-1,1] to screen pixel coordinates [0,width] × [0,height]"
      - "OBJ file loader reads vertex positions and face indices (at minimum)"
      - "Wireframe render: 3D model is projected and drawn as line segments on screen"
      - "Rotating the model matrix produces visible 3D rotation in output"
    pitfalls:
      - "Matrix multiplication order: MVP = Projection × View × Model, not the reverse"
      - "Forgetting perspective divide—everything looks orthographic"
      - "Y-axis flip: screen Y increases downward, NDC Y increases upward"
      - "Homogeneous w=0 vertices cause divide-by-zero (need clipping, addressed in M4)"
      - "OBJ face indices are 1-based, not 0-based"
    concepts:
      - Homogeneous coordinates
      - Model-View-Projection pipeline
      - Perspective divide
      - Viewport transformation
    skills:
      - Matrix mathematics and linear algebra
      - Coordinate space transformations
      - 3D file format parsing
      - Wireframe rendering
    deliverables:
      - "4×4 matrix library with multiply, identity, transpose, inverse"
      - "Model, View, and Projection matrix constructors"
      - "Perspective divide function"
      - "Viewport transform function"
      - "OBJ file loader (vertices and faces)"
      - "Wireframe renderer projecting 3D edges to screen lines"
    estimated_hours: "6-8"

  - id: software-3d-m3
    name: "Triangle Rasterization & Backface Culling"
    description: >
      Implement filled triangle rasterization with the top-left fill rule
      and backface culling using surface normals.
    acceptance_criteria:
      - "Solid-colored triangles are filled with correct pixel coverage using scanline or edge-function method"
      - "Top-left fill rule ensures no gaps or double-drawn pixels between adjacent triangles"
      - "Flat-top, flat-bottom, and general triangles all rasterize correctly"
      - "Degenerate triangles (zero area) are rejected without crashing"
      - "Backface culling: triangles facing away from the camera (negative winding in screen space) are skipped"
      - "Barycentric coordinates are computed for each pixel inside the triangle (used for interpolation in later milestones)"
      - "Visual test: 3D model renders as flat-colored solid faces"
    pitfalls:
      - "Inconsistent winding order between model faces—some faces culled incorrectly"
      - "Gaps between adjacent triangles from incorrect fill rule implementation"
      - "Sub-pixel precision errors causing shimmer on moving geometry"
      - "Rasterizer performance: naive bounding-box iteration is O(screen) per triangle; should iterate only within triangle bounds"
    concepts:
      - Triangle rasterization
      - Fill rules (top-left)
      - Barycentric coordinates
      - Backface culling
    skills:
      - Edge function evaluation
      - Scanline rasterization
      - Surface normal computation
      - Winding order determination
    deliverables:
      - "Triangle rasterizer with top-left fill rule"
      - "Barycentric coordinate computation per pixel"
      - "Backface culling based on screen-space winding order"
      - "Solid-colored 3D model rendering"
    estimated_hours: "5-7"

  - id: software-3d-m4
    name: "Near-Plane Clipping & Z-Buffer"
    description: >
      Implement near-plane clipping in homogeneous clip space to handle
      geometry behind the camera, and add a z-buffer for correct depth
      ordering of overlapping faces.
    acceptance_criteria:
      - "Triangles partially behind the near plane are clipped: the behind-camera portion is removed and new vertices are generated at the clip boundary"
      - "Triangles fully behind the near plane are discarded entirely"
      - "Clipping produces 0, 1, or 2 output triangles from each input triangle (Sutherland-Hodgman or equivalent)"
      - "Z-buffer stores per-pixel depth values initialized to maximum depth each frame"
      - "Depth test: only the nearest fragment at each pixel is written to the color buffer"
      - "Depth values are correctly interpolated across the triangle surface using barycentric coordinates"
      - "Visual test: overlapping geometry renders with correct occlusion (no z-fighting at normal distances)"
      - "Camera can move freely without geometry inversion or crashes when objects pass behind the camera"
    pitfalls:
      - "Skipping near-plane clipping causes inverted/exploded geometry when vertices have w ≤ 0"
      - "Clipping in screen space instead of clip space produces incorrect results"
      - "Z-fighting from insufficient depth buffer precision (use 1/z or logarithmic depth for better distribution)"
      - "Interpolating z linearly instead of using perspective-correct 1/z interpolation"
      - "Not generating correct UVs and normals at clipped vertex positions (affects later milestones)"
    concepts:
      - Homogeneous clipping
      - Near-plane clipping
      - Z-buffer algorithm
      - Depth interpolation
    skills:
      - Sutherland-Hodgman polygon clipping
      - Per-pixel depth comparison
      - Perspective-correct interpolation
      - Numerical precision management
    deliverables:
      - "Near-plane triangle clipper producing 0-2 output triangles"
      - "Z-buffer allocation and per-frame clearing"
      - "Depth test integrated into rasterizer"
      - "Perspective-correct depth interpolation"
      - "Visual test: camera moving through geometry without artifacts"
    estimated_hours: "6-8"

  - id: software-3d-m5
    name: "Lighting & Shading"
    description: >
      Implement flat shading and Gouraud shading with diffuse lighting
      from a directional light source.
    acceptance_criteria:
      - "Face normals are computed from triangle vertex positions using cross product"
      - "Flat shading: each triangle face is lit uniformly based on dot(face_normal, light_direction)"
      - "Vertex normals are computed by averaging face normals of adjacent faces (or loaded from OBJ)"
      - "Gouraud shading: per-vertex lighting is interpolated across the triangle using barycentric coordinates"
      - "Diffuse lighting intensity = max(0, dot(normal, light_dir)) (clamped, no negative light)"
      - "Light direction is configurable and consistent across the scene"
      - "Visual test: 3D model shows smooth shading gradients on curved surfaces (Gouraud) vs faceted appearance (flat)"
    pitfalls:
      - "Normals pointing inward produce inverted lighting (dark where it should be bright)"
      - "Not normalizing normals after interpolation causes brightness variation"
      - "Gouraud shading interpolation artifacts on large triangles (specular highlights missed)"
      - "Face normal computed with wrong vertex winding produces flipped normal"
    concepts:
      - Surface normals (face and vertex)
      - Lambertian diffuse lighting
      - Flat vs Gouraud shading
    skills:
      - Vector math for lighting (dot product, cross product)
      - Normal computation and averaging
      - Per-vertex attribute interpolation
      - Lighting model implementation
    deliverables:
      - "Face normal computation from triangle vertices"
      - "Flat shading with per-face diffuse lighting"
      - "Vertex normal computation (average of adjacent face normals)"
      - "Gouraud shading with per-vertex lighting interpolated across triangles"
      - "Configurable directional light source"
    estimated_hours: "5-8"

```
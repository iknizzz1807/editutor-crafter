# AUDIT & FIX: build-raytracer

## CRITIQUE
- **PPM Format Scalability (Confirmed):** P3 (ASCII PPM) is fine for M1-M4 but becomes a serious bottleneck at M5+ with high sample counts. A 1920x1080 image at 100 samples per pixel generating P3 output means writing billions of characters. The project should introduce binary PPM (P6) or PNG output before the antialiasing milestone.
- **Gamma Correction Timing (Confirmed):** Gamma correction is introduced in M6 (Diffuse Materials) but all images from M1-M5 are displayed in linear color space, which looks too dark. Gamma correction should be introduced in M1 and applied from the start, or at minimum noted as a known visual limitation.
- **M6 Hemisphere Sampling Ambiguity:** The AC says 'cosine-weighted distribution' for diffuse scattering, but the standard Ray Tracing in One Weekend approach uses uniform random-in-unit-sphere (Lambertian approximation) or random-on-hemisphere. True cosine-weighted importance sampling is an optimization. The AC should be precise about which method.
- **M8 IOR Convention:** 'Getting the IOR ratio right' is listed as a pitfall but the AC doesn't specify the convention: is it n1/n2 or n2/n1? When the ray enters vs exits the material, the ratio flips. The AC should require tracking whether the ray is entering or exiting.
- **Missing: Front-face vs Back-face Normal Determination:** This is critical for M8 (dielectrics) but should be introduced in M4. The renderer must determine whether a ray hits the outside or inside of a surface to correctly orient normals and compute refraction.
- **Missing Performance Milestone:** For a project with 'Performance Optimization' as a listed skill, there's no BVH (Bounding Volume Hierarchy) milestone, despite it being listed in the learning outcomes. BVH is essential for rendering scenes with more than a few dozen objects.
- **M3 and M4 Could Be Merged:** M3 (single sphere intersection) and M4 (normals + multiple objects) are very thin milestones. They are logically one unit of work.
- **Missing: Multi-threading or Progress Reporting:** At high sample counts, renders take minutes to hours. No milestone addresses progress reporting or parallelization, which are practical necessities.

## FIXED YAML
```yaml
id: build-raytracer
name: Build Your Own Ray Tracer
description: "Path tracing renderer implementing physically-based materials, Monte Carlo antialiasing, and camera effects."
difficulty: advanced
estimated_hours: "25-45"
essence: >
  Simulating light transport through recursive ray casting and geometric
  intersection testing, computing surface radiance by stochastic sampling
  of material scattering functions, with physically-based reflection,
  refraction, and Fresnel effects approximated via Monte Carlo integration.
why_important: >
  Building a ray tracer teaches fundamental computer graphics algorithms
  (vector math, geometric intersection, physically-based light transport)
  that underpin modern rendering engines, visual effects pipelines, and
  game development, while developing skills in numerical methods and
  performance optimization of compute-intensive algorithms.
learning_outcomes:
  - Implement ray-primitive intersection tests using quadratic equation solving
  - Design a recursive ray tracing engine with material abstraction
  - Implement Monte Carlo sampling for antialiasing and diffuse scattering
  - Build physically-based materials (Lambertian, metal, dielectric)
  - Implement Snell's law and Schlick's Fresnel approximation
  - Build a positionable camera with field of view and depth of field
  - Apply gamma correction for perceptually correct image output
  - Optimize rendering with bounding volume hierarchies
  - Debug numerical precision issues in ray-geometry calculations
skills:
  - Vector Mathematics
  - Ray-Geometry Intersection
  - Monte Carlo Integration
  - Physically-Based Rendering
  - Recursive Algorithms
  - Performance Optimization (BVH)
  - Linear Algebra
  - Image Format Handling
tags:
  - advanced
  - build-from-scratch
  - c++
  - game-dev
  - go
  - lighting
  - ray-intersection
  - rendering
  - rust
  - shading
architecture_doc: architecture-docs/build-raytracer/index.md
languages:
  recommended:
    - C++
    - Rust
    - Go
  also_possible:
    - Python
    - JavaScript
resources:
  - name: "Ray Tracing in One Weekend"
    url: https://raytracing.github.io/books/RayTracingInOneWeekend.html
    type: book
  - name: "Ray Tracing: The Next Week"
    url: https://raytracing.github.io/books/RayTracingTheNextWeek.html
    type: book
  - name: "Physically Based Rendering (PBRT)"
    url: https://pbr-book.org/
    type: reference
prerequisites:
  - type: skill
    name: "Linear algebra (vectors, dot product, cross product)"
  - type: skill
    name: "Basic geometry (spheres, planes, coordinate systems)"
  - type: skill
    name: "Understanding of light behavior (reflection, refraction)"
milestones:
  - id: build-raytracer-m1
    name: "Image Output, Vectors & Rays"
    description: >
      Set up image output (PPM), implement vector math utilities, define
      the ray class, and render a background gradient. Apply gamma
      correction from the start.
    acceptance_criteria:
      - "PPM writer outputs valid P3 (ASCII) or P6 (binary) format file openable by image viewers"
      - "Vec3 class supports add, subtract, multiply, divide, dot, cross, length, normalize, and unit_vector operations"
      - "All vector operations pass unit tests with known expected values"
      - "Color class stores RGB as floating-point [0,1]; output clamps and converts to [0,255]"
      - "Gamma correction (pow 1/2.2 or sqrt) is applied before writing pixel values"
      - "Ray class stores origin and direction; ray.at(t) returns origin + t * direction"
      - "Camera generates rays from eye point through each pixel of a virtual viewport"
      - "Background renders a blue-to-white vertical gradient based on ray y-direction"
      - "Image dimensions and aspect ratio are configurable"
    pitfalls:
      - "RGB values not clamped to [0,1] before gamma correction causes overflow"
      - "Forgetting gamma correction makes all images too dark (linear vs sRGB)"
      - "Ray direction not normalized affects later distance calculations"
      - Y-axis direction: PPM rows go top-to-bottom; ray y goes bottom-to-top
      - "Integer types for coordinates loses sub-pixel precision"
    concepts:
      - Image file formats (PPM)
      - Vector mathematics
      - Ray representation
      - Gamma correction
      - Camera model basics
    skills:
      - File I/O
      - 3D vector math implementation
      - Parametric ray equations
      - Color space conversion
    deliverables:
      - "PPM image writer (P3 or P6)"
      - "Vec3 class with full arithmetic and geometric operations"
      - "Ray class with origin, direction, and at(t) method"
      - "Camera ray generation through viewport pixels"
      - "Gamma correction applied to all output"
      - "Background gradient rendering"
    estimated_hours: "2-3"

  - id: build-raytracer-m2
    name: "Sphere Intersection, Normals & Multiple Objects"
    description: >
      Implement ray-sphere intersection, surface normal computation,
      a hittable abstraction for multiple objects, and front-face
      normal determination.
    acceptance_criteria:
      - "Ray-sphere intersection solves the quadratic equation and returns the nearest positive t"
      - Hit record stores: intersection point, surface normal, t parameter, and front_face boolean
      - "Surface normals are unit-length and always point outward from the geometry surface"
      - "front_face flag indicates whether the ray hit the outside (true) or inside (false) of the surface"
      - "Normal-to-color mapping renders sphere with visually distinct directional shading"
      - "Hittable interface/trait abstracts over different geometry types"
      - "HittableList iterates all objects and returns the closest valid intersection (smallest positive t)"
      - "t_min is set to a small epsilon (e.g., 0.001) to prevent self-intersection artifacts"
      - "Scene with ground sphere and centered sphere renders both with correct occlusion"
    pitfalls:
      - "Choosing wrong quadratic root (must select smallest positive t)"
      - Floating-point self-intersection: t_min = 0 causes shadow acne (use 0.001)
      - "Normal pointing inward for back-face hits—must flip based on ray direction"
      - "Forgetting to normalize normals after computation"
      - "Returning first hit instead of closest hit from object list"
    concepts:
      - Ray-sphere intersection (quadratic formula)
      - Surface normals
      - Front-face determination
      - Closest-hit algorithm
      - Polymorphic geometry
    skills:
      - Quadratic equation solving
      - Geometric intersection algorithms
      - Polymorphic/trait-based object design
      - Normal computation
    deliverables:
      - "Sphere class with center and radius"
      - "Ray-sphere intersection with discriminant and root selection"
      - "Hit record with point, normal, t, front_face"
      - "Hittable interface and HittableList"
      - "Closest-hit selection across all objects"
      - "Front-face normal determination"
    estimated_hours: "2-4"

  - id: build-raytracer-m3
    name: "Antialiasing & Diffuse Materials"
    description: >
      Add multi-sample antialiasing, then implement Lambertian diffuse
      material with recursive ray bouncing.
    acceptance_criteria:
      - "Multiple rays cast per pixel with random sub-pixel jitter (uniform in pixel area)"
      - "Sample count is configurable; higher counts produce visibly smoother edges"
      - "Final pixel color is the average of all sample colors"
      - "Lambertian material scatters rays in random directions on the hemisphere (using random-in-unit-sphere + normal method)"
      - "Recursive ray bouncing follows scattered rays up to a configurable max depth"
      - "At max depth, ray returns black (no light contribution)"
      - "Each bounce attenuates color by the material's albedo"
      - "Shadow regions appear naturally from indirect illumination"
      - "Gamma correction produces visually correct brightness (not too dark)"
      - "Render time scales linearly with sample count (measurable)"
    pitfalls:
      - "Not randomizing within pixel boundary causes aliasing artifacts instead of smooth edges"
      - "Infinite recursion without depth limit crashes the program"
      - "Shadow acne from t_min = 0 (must use small epsilon)"
      - "Forgetting to average samples produces overly bright image"
      - "Random-on-hemisphere vs random-in-unit-sphere produces subtly different shading"
    concepts:
      - Monte Carlo antialiasing
      - Lambertian reflection
      - Recursive ray tracing
      - Ray attenuation
    skills:
      - Monte Carlo sampling
      - Random number generation
      - Recursive algorithm with termination
      - Hemisphere sampling
    deliverables:
      - "Multi-sample per pixel with configurable count"
      - "Random sub-pixel jitter generation"
      - "Lambertian material class with albedo and scatter function"
      - "Recursive ray_color function with max depth"
      - "Color attenuation per bounce"
    estimated_hours: "3-4"

  - id: build-raytracer-m4
    name: "Metal & Dielectric Materials"
    description: >
      Implement metallic reflection with fuzz and dielectric (glass)
      refraction with Fresnel effects. Support material assignment per object.
    acceptance_criteria:
      - "Metal material reflects rays using v - 2*dot(v,n)*n formula"
      - "Fuzz parameter [0,1] adds random perturbation to reflected direction; fuzz=0 is mirror"
      - "Reflected rays pointing into surface (dot > 0 with normal) are absorbed (return black)"
      - Dielectric material refracts using Snell's law: n1*sin(θ1) = n2*sin(θ2)
      - IOR ratio correctly flips based on front_face: entering uses 1/ior, exiting uses ior/1
      - "Total internal reflection occurs when sin(θ_transmitted) > 1; ray reflects instead of refracting"
      - "Schlick approximation computes reflectance probability at grazing angles"
      - "Hollow glass sphere rendered using negative-radius inner sphere trick"
      - Material assignment: different objects in the same scene have different materials
      - "Scene demonstrates all three materials (diffuse, metal, glass) simultaneously"
    pitfalls:
      - IOR ratio inverted: entering medium should be (1.0 / ior), not (ior / 1.0)
      - "Fuzz > 1 causes scattered ray to point through surface—clamp or reject"
      - Total internal reflection check forgotten: glass appears black at steep angles
      - "Not attenuating color per bounce produces unrealistically bright reflections"
      - "Negative radius trick for hollow glass requires normal flip understanding from M2"
    concepts:
      - Specular reflection
      - Snell's law refraction
      - Fresnel effect (Schlick approximation)
      - Total internal reflection
      - Material system design
    skills:
      - Reflection vector computation
      - Refraction physics
      - Probabilistic ray path selection
      - Material abstraction design
    deliverables:
      - "Metal material with reflection and configurable fuzz"
      - "Dielectric material with refraction (Snell's law)"
      - "Schlick approximation for Fresnel reflectance"
      - "Total internal reflection handling"
      - "Material assignment per scene object"
      - "Demo scene with diffuse, metal, and glass spheres"
    estimated_hours: "4-6"

  - id: build-raytracer-m5
    name: "Positionable Camera & Depth of Field"
    description: >
      Implement a camera with arbitrary position, look-at targeting,
      configurable FOV, and thin-lens depth-of-field blur.
    acceptance_criteria:
      - "Camera positioned at arbitrary lookfrom point, aimed at lookat target"
      - "Orthonormal basis (u, v, w) computed from view direction and vup vector"
      - "Field of view parameter (vertical, in degrees) controls zoom level"
      - "Aspect ratio matches output image dimensions"
      - Thin lens model: ray origins randomly offset within circular aperture disk
      - "Focus distance parameter controls the plane of sharp focus"
      - "Objects at focus distance are sharp; objects nearer/farther are blurred proportionally"
      - "Aperture=0 produces pinhole camera (no blur); larger aperture increases blur"
      - "Camera up vector parallel to view direction is detected and reported as error"
    pitfalls:
      - FOV in radians vs degrees confusion (common: forgetting to convert)
      - "vup parallel to view direction produces degenerate basis (cross product = zero)"
      - "Aspect ratio computed as height/width instead of width/height—stretched image"
      - "Random disk sampling with bias produces visible ring artifacts in bokeh"
      - "Focus distance set incorrectly makes the intended subject blurry"
    concepts:
      - Camera coordinate system
      - Orthonormal basis construction
      - Thin lens model
      - Depth of field
    skills:
      - Camera transformation math
      - Trigonometric FOV computation
      - Random point on disk generation
      - Focal plane geometry
    deliverables:
      - "Positionable camera with lookfrom, lookat, vup"
      - "Configurable vertical FOV in degrees"
      - "Thin lens depth of field with aperture and focus distance"
      - "Random ray origin offset for defocus blur"
      - "Aspect ratio from image dimensions"
    estimated_hours: "3-4"

  - id: build-raytracer-m6
    name: "BVH, Final Scene & Output Optimization"
    description: >
      Implement Bounding Volume Hierarchy for performance, upgrade image
      output to binary format, add progress reporting, and render a
      complex final scene.
    acceptance_criteria:
      - "BVH tree constructed from scene objects using surface area heuristic or midpoint split"
      - "Ray-BVH traversal skips branches whose bounding box is not hit, reducing intersection tests"
      - "BVH reduces render time by >5x compared to linear object list on a scene with 100+ spheres (measured)"
      - "Image output uses binary PPM (P6) or PNG for practical file sizes at high resolutions"
      - "Progress reporting outputs percentage complete to stderr during rendering"
      - "Final scene contains 100+ objects with mixed materials demonstrating all features"
      - "Final scene renders at 1920×1080 with 100+ samples per pixel in under 10 minutes on modern hardware"
      - Multi-threading (optional but recommended): render tiles in parallel across CPU cores
    pitfalls:
      - "BVH with bad split heuristic degenerates to linear scan—measure speedup"
      - "AABB hit test must handle rays parallel to an axis (division by zero or NaN)"
      - "Binary PPM byte order wrong produces garbled image"
      - "Progress reporting inside tight loop causes I/O bottleneck—report per scanline, not per pixel"
      - "Multi-threading race condition on shared random number generator state"
    concepts:
      - Bounding Volume Hierarchy
      - Spatial acceleration structures
      - Performance benchmarking
      - Image format optimization
    skills:
      - Tree construction algorithms
      - AABB intersection testing
      - Performance measurement and comparison
      - Binary file I/O
      - Optional: thread-safe rendering
    deliverables:
      - "BVH tree construction with axis-aligned bounding boxes"
      - "Ray-BVH traversal with early termination"
      - "Binary PPM (P6) or PNG image output"
      - "Render progress reporting"
      - Final scene: 100+ objects, mixed materials, camera effects
      - Performance benchmark: BVH vs linear, with timing output
    estimated_hours: "5-8"
```
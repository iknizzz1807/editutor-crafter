# AUDIT & FIX: build-react

## CRITIQUE
- The project is well-structured with a clear progression: Virtual DOM -> Reconciliation -> Fiber -> Hooks.
- M1 AC 'createElement produces virtual nodes with type, props, and children properties' — good but should specify that JSX pragma integration is expected (configuring the transpiler to call your createElement).
- M1 AC is missing handling of boolean, null, and undefined children (which should be filtered out, per React behavior).
- M1 pitfalls mention 'SVG namespace' but no AC addresses SVG rendering. This is important — SVG elements require createElementNS, not createElement.
- M2 AC 'Diff algorithm identifies minimal set of changes between two virtual trees' — 'minimal' is ambiguous. React's algorithm is O(n) with heuristics, not truly minimal. Should specify the heuristic approach.
- M2 AC 'Keyed children are reordered efficiently without unnecessary recreation' — no specification of what 'efficiently' means. Should specify that keyed children maintain DOM identity (same key = same DOM node moved, not recreated).
- M2 doesn't mention the 'type change' case: if element type changes (div -> span), the entire subtree must be destroyed and recreated.
- M3 AC 'Work loop uses requestIdleCallback to process fibers without blocking main thread' — requestIdleCallback has poor browser support and React actually uses its own scheduler (MessageChannel-based). Should note this.
- M3 AC 'In-progress render work can be interrupted and resumed without visual artifacts' — how is this tested? Need a measurable scenario.
- M3 has no AC for the double-buffering (current tree and work-in-progress tree) concept which is essential for fiber.
- M4 AC 'useState returns current value and setter function triggering component re-render' — good but should specify that multiple setState calls in the same event handler are batched into a single re-render.
- M4 AC 'Hooks called in different order across renders produce a descriptive error' — good, this is the rules-of-hooks enforcement.
- M4 AC 'useEffect runs cleanup function before re-running effect on dependency changes' — should also specify that effects run AFTER the commit phase (DOM is updated), not during.
- M4 estimated hours (25-45) is a very wide range. Hooks are complex but 45 hours alone seems high given the foundation from M1-M3.
- No mention of functional component support in M1/M2 — when is the concept of components (function returning virtual DOM) introduced? It's implied but not explicit.

## FIXED YAML
```yaml
id: build-react
name: Build Your Own React
description: >-
  Build a minimal React-like library from scratch: virtual DOM, reconciliation
  with diffing, fiber-based interruptible rendering, and hooks (useState,
  useEffect).
difficulty: expert
estimated_hours: 80
essence: >-
  Component tree representation as lightweight JavaScript objects with O(n)
  heuristic tree-diffing for minimal DOM updates, fiber-based work scheduling
  for interruptible rendering with cooperative concurrency, and closure-based
  hooks implementing stateful logic in function components.
why_important: >-
  Building this demystifies React's core mechanisms — virtual DOM reconciliation,
  fiber scheduling, and hooks closures — giving deep insight into how modern UI
  frameworks optimize rendering and manage component lifecycle. Directly
  applicable to debugging production React issues and making informed
  architectural decisions.
learning_outcomes:
  - Implement virtual DOM representation using plain JavaScript objects and createElement
  - Build an O(n) heuristic reconciliation algorithm using keys and type comparison
  - Design a fiber-based scheduler with work units that pause and resume for time-slicing
  - Implement useState with closure-based state persistence and batched updates
  - Create useEffect with dependency tracking and cleanup support
  - Build a commit phase that batches DOM mutations for atomic updates
skills:
  - Virtual DOM Implementation
  - Tree Diffing Algorithms
  - Fiber Architecture
  - Closure-based State Management
  - Cooperative Scheduling
  - Reconciliation Algorithms
  - Functional Programming Patterns
tags:
  - build-from-scratch
  - expert
  - fiber
  - frontend
  - hooks
  - javascript
  - reconciliation
  - virtual-dom
architecture_doc: architecture-docs/build-react/index.md
languages:
  recommended:
    - JavaScript
    - TypeScript
  also_possible: []
resources:
  - type: article
    name: Build your own React
    url: https://pomb.us/build-your-own-react/
  - type: video
    name: React Fiber Architecture
    url: https://github.com/acdlite/react-fiber-architecture
prerequisites:
  - type: skill
    name: DOM manipulation
  - type: skill
    name: JavaScript closures, prototypes, and event loop
  - type: skill
    name: Tree data structures and recursion
milestones:
  - id: build-react-m1
    name: Virtual DOM & Initial Render
    description: >-
      Create virtual DOM representation using plain objects, implement
      createElement, and build the initial render function that creates real
      DOM from virtual DOM, including functional component support.
    estimated_hours: 12
    concepts:
      - Virtual DOM: lightweight JS objects {type, props, children} representing UI
      - createElement: function called by JSX transpiler to produce virtual nodes
      - Functional components: functions that return virtual DOM trees
      - Text nodes: wrap raw strings/numbers in a text virtual node type
      - Children normalization: filter out null, undefined, false; flatten nested arrays
    skills:
      - DOM API (createElement, createTextNode, setAttribute)
      - Recursive tree traversal
      - JSX transpiler configuration
      - Event listener attachment
    acceptance_criteria:
      - "createElement(type, props, ...children) produces a virtual node object with type (string or function), props (object including children), correctly handling zero, one, or many children"
      - "Children that are null, undefined, false, or true are filtered out; strings and numbers are wrapped as text virtual nodes"
      - "render(vdom, container) creates matching real DOM elements from the virtual node tree and appends them to the container element"
      - "Text nodes (strings, numbers) are correctly created as DOM text nodes"
      - Props are applied to DOM elements: className sets class attribute, style object sets inline styles, on* props (onClick, onChange) attach event listeners
      - "Functional components (functions returning virtual DOM) are called during render and their return value is recursively rendered"
      - JSX integration: configuring Babel/TypeScript pragma to use your createElement produces working UI from JSX syntax
    pitfalls:
      - Not handling null/undefined/boolean children causing crashes on document.createTextNode(null)
      - Event listener naming mismatch (onClick in JSX vs onclick in DOM)
      - Forgetting SVG elements require document.createElementNS with SVG namespace
      - Not normalizing children arrays (nested arrays from map() calls)
      - Directly setting innerHTML instead of using proper DOM APIs (XSS risk)
    deliverables:
      - createElement function producing virtual node objects
      - Children normalization (filter falsy, wrap primitives, flatten arrays)
      - render function recursively creating real DOM from virtual DOM tree
      - Text node creation for string and number children
      - Props application (attributes, styles, event listeners)
      - Functional component invocation and rendering
      - JSX pragma configuration example

  - id: build-react-m2
    name: Reconciliation (Diffing)
    description: >-
      Implement the O(n) heuristic reconciliation algorithm that efficiently
      updates the real DOM by diffing old and new virtual DOM trees.
    estimated_hours: 18
    concepts:
      - Heuristic: different type = destroy and recreate entire subtree (no cross-type diffing)
      - Same type element: update changed props in-place, recurse into children
      - Keyed reconciliation: match children by key to preserve DOM identity across reorders
      - O(n) complexity: compare trees level-by-level, not arbitrary tree edit distance
    skills:
      - Algorithm design (tree comparison)
      - DOM node reference tracking
      - Efficient child list reconciliation
      - Property diffing and patching
    acceptance_criteria:
      - "When element type changes (e.g., div -> span), the old subtree is completely destroyed (DOM nodes removed, event listeners cleaned up) and new subtree is created from scratch"
      - "When element type is the same, only changed props are updated on the existing DOM node; unchanged subtrees are not touched"
      - Property diffing correctly handles: added props (set on DOM), removed props (remove from DOM), changed props (update on DOM), and changed event listeners (remove old, add new)
      - "Keyed children are matched by key across renders; a reordered list of keyed items moves existing DOM nodes instead of destroying and recreating them (verified by checking DOM node identity via reference equality)"
      - "Unkeyed children are diffed by index position; inserting at the beginning of an unkeyed list recreates all children (known limitation matching React behavior)"
      - "Re-render of unchanged virtual DOM tree produces zero DOM mutations (verified by counting DOM API calls)"
      - Component re-rendering: when a functional component's parent re-renders, the component is re-invoked and its output is reconciled
    pitfalls:
      - Using array index as key causing incorrect DOM reuse when list items are reordered
      - Not cleaning up event listeners on removed elements causing memory leaks
      - DOM node reference tracking getting out of sync after moves/deletions
      - Forgetting to handle text node updates (just update textContent, don't recreate)
      - Not handling the case where children count changes (old has 3, new has 5)
    deliverables:
      - Type comparison logic (same type = update, different type = replace)
      - Property differ detecting added, removed, and changed attributes/events
      - Property patcher applying minimal DOM updates for changed props
      - Keyed child reconciliation preserving DOM node identity across reorders
      - Unkeyed child reconciliation by index
      - Subtree destruction with event listener cleanup
      - Re-render trigger function that diffs new vs old virtual DOM and patches

  - id: build-react-m3
    name: Fiber Architecture
    description: >-
      Refactor rendering into a fiber-based architecture with interruptible
      work units, enabling cooperative scheduling that doesn't block the main
      thread.
    estimated_hours: 22
    concepts:
      - Fiber node: linked list structure with child, sibling, parent pointers replacing recursive tree walk
      - Work loop: process one fiber unit of work, then check if browser needs control back
      - Double buffering: current fiber tree (what's on screen) and work-in-progress tree (being built)
      - Commit phase: after all fibers are processed, apply DOM mutations atomically in one batch
      - requestIdleCallback (or MessageChannel scheduler): yield to browser between work units
    skills:
      - Linked list data structures
      - Cooperative scheduling patterns
      - Double buffering technique
      - Work loop implementation
      - Atomic DOM update batching
    acceptance_criteria:
      - "Fiber nodes are linked via child (first child), sibling (next sibling), and parent (return) pointers, forming a traversable linked structure mirroring the component tree"
      - "Work loop processes one fiber at a time (performUnitOfWork) and yields control to the browser between units using requestIdleCallback or a MessageChannel-based scheduler"
      - "Double buffering maintains a 'current' fiber tree (reflecting what's displayed) and a 'work-in-progress' tree (being built); on commit, work-in-progress becomes current"
      - "Commit phase applies all accumulated DOM mutations (insertions, updates, deletions) in a single synchronous pass after the entire tree is reconciled — no partial updates are visible to the user"
      - "A rendering workload of 10,000 elements does not block the main thread for more than 16ms at a time (verified by measuring long task duration with PerformanceObserver or manual timing)"
      - "In-progress render work can be interrupted (simulated by reducing deadline) and resumed correctly, producing the same final DOM as an uninterrupted render"
      - Functional components are processed as fiber nodes: the component function is called during the render phase, and its returned virtual DOM is used to create child fibers
    pitfalls:
      - Forgetting to handle the 'return' (parent) pointer causing incomplete tree traversal
      - Memory leaks in the alternate (old) fiber tree if not properly cleaned up after commit
      - Effect ordering confusion — effects should be collected during render but executed after commit
      - requestIdleCallback has limited browser support — consider MessageChannel fallback
      - Interrupted render leaving stale work-in-progress state that corrupts next render
    deliverables:
      - Fiber node structure (type, props, child, sibling, parent, alternate, effectTag)
      - performUnitOfWork function processing a single fiber and returning next fiber
      - Work loop with idle callback yielding control between units
      - Double-buffered fiber trees (current and work-in-progress)
      - Commit phase walking fiber tree and applying DOM mutations atomically
      - Deletion tracking for removed fibers
      - Functional component fiber processing

  - id: build-react-m4
    name: Hooks (useState & useEffect)
    description: >-
      Implement useState and useEffect hooks using closure-based state
      persistence, with batched state updates and proper effect lifecycle.
    estimated_hours: 28
    concepts:
      - Hooks array: each fiber maintains an ordered array of hook values, indexed by call order
      - useState: returns [value, setter]; setter queues a re-render; multiple setters in one handler batch into single re-render
      - useEffect: runs AFTER commit phase (DOM is updated); cleanup runs before next effect or on unmount
      - Dependency array: effect re-runs only when deps change (shallow comparison)
      - Rules of hooks: hooks must be called in the same order every render (no conditionals)
    skills:
      - JavaScript closures
      - State management patterns
      - Dependency tracking and comparison
      - Effect lifecycle management
      - Hook ordering rules enforcement
    acceptance_criteria:
      - "useState(initialValue) returns [currentValue, setterFunction]; calling the setter with a new value triggers a re-render of the component"
      - State persists across re-renders: calling useState on re-render returns the updated value, not the initial value
      - "Multiple setState calls within the same event handler are batched into a single re-render (component renders once, not N times)"
      - "useEffect(callback, deps) runs the callback AFTER the commit phase (DOM is updated), not during render"
      - useEffect cleanup: if the callback returns a function, it is called before the effect re-runs and on component unmount
      - Dependency tracking: effect re-runs only when at least one dependency value changes (shallow equality comparison); empty deps array ([]) means effect runs only on mount
      - "Hooks called in different order across renders (e.g., inside an if statement) throw a descriptive error message"
      - Custom hooks: functions starting with 'use' that compose useState and useEffect work correctly and share no state between different component instances
      - "A counter component using useState correctly increments and re-renders; a timer component using useEffect correctly starts on mount and cleans up on unmount"
    pitfalls:
      - Stale closures capturing old state values in event handlers — setter should support functional updates: setState(prev => prev + 1)
      - Conditional hooks breaking the hook index alignment across renders
      - Effect cleanup timing: cleanup of previous effect must run BEFORE new effect, not after
      - Memory leak from useEffect without cleanup (e.g., setInterval not cleared)
      - Not implementing functional setState (setState(fn)) causing stale state in rapid updates
    deliverables:
      - Hook index tracker on fiber (wipFiber.hookIndex)
      - useState implementation with state persistence, setter, and batched re-render
      - Functional setState support (setState(prev => newValue))
      - useEffect implementation with post-commit execution
      - Effect cleanup function invocation on re-run and unmount
      - Dependency array comparison (shallow equality)
      - Hook order validation (error on conditional hooks)
      - Custom hook support (composable useState/useEffect)
```
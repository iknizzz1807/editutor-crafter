# D2 Documentation - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Core Concepts](#core-concepts)
5. [Shapes](#shapes)
6. [Connections](#connections)
7. [Containers](#containers)
8. [Styling](#styling)
9. [Themes](#themes)
10. [Layouts](#layouts)
11. [Advanced Features](#advanced-features)
12. [CLI Usage](#cli-usage)
13. [API](#api)
14. [Examples](#examples)
15. [Resources](#resources)

---

## Introduction

### What is D2?

**D2** (Declarative Diagramming) is a modern diagram scripting language that turns text into diagrams. It's designed specifically for software engineers to create software architecture diagrams through a declarative approach.

**Key Features:**
- 📝 Text-to-diagram conversion
- 🎨 Professional themes out of the box
- 🔄 Watch mode with live reload
- 🎯 Multiple layout engines (dagre, ELK, TALA)
- 📦 First-class container support
- 🖼️ Images and icons integration
- 📊 SQL tables and UML class diagrams
- 🎬 Animation support
- 🌐 Works on MacOS, Linux, and Windows
- 💻 CLI and API available

### Why D2?

D2 addresses the pain points of traditional diagramming tools:

**Traditional Diagramming (Design Tools):**
- Manual drag-and-drop
- Time-consuming updates
- Hard to version control
- No review process
- Difficult to maintain

**D2 Approach (Dev Tool):**
- Declarative syntax - describe what you want
- Automatic layout
- Version control friendly
- Easy updates and maintenance
- Code review friendly
- Modular and composable

### Design Philosophy

1. **Separation of Content and Design**: Define system structure separately from visual styling
2. **Focused Scope**: Specifically for software architecture diagrams (not general-purpose visualization)
3. **Readable Syntax**: Prioritizes clarity over terseness
4. **Fail Gracefully**: Compiles whenever possible, warnings instead of errors
5. **Good Aesthetics by Default**: Professional-looking diagrams without customization

---

## Installation

### Quick Install (Recommended)

Using the install script (works on Mac, Linux, Windows):

```bash
# Preview what the script will do
curl -fsSL https://d2lang.com/install.sh | sh -s -- --dry-run

# Install for real
curl -fsSL https://d2lang.com/install.sh | sh -s --
```

### Package Managers

**MacOS (Homebrew):**
```bash
brew install d2
```

**Windows (Winget):**
```bash
winget install d2
```

### From Source

**With Go installed:**
```bash
go install oss.terrastruct.com/d2@latest
```

### Verify Installation

```bash
d2 version
```

### Download Precompiled Binaries

Visit the GitHub releases page: https://github.com/terrastruct/d2/releases

---

## Getting Started

### Hello World

1. Create a file `input.d2`:

```d2
x -> y -> z
```

2. Run D2 with watch mode:

```bash
d2 --watch input.d2 out.svg
```

A browser window will open with `out.svg` and live-reload when you change `input.d2`.

### Basic Example

```d2
# Define shapes
NETWORK
USER
API SERVER
LOGS
CELL TOWER
ONLINE PORTAL
DATA PROCESSOR
SATELLITE
TRANSMITTER
UI
STORAGE

# Define connections
CELL TOWER.SATELLITE -> CELL TOWER.TRANSMITTER: SEND
CELL TOWER.SATELLITE -> CELL TOWER.TRANSMITTER: SEND
CELL TOWER.SATELLITE -> CELL TOWER.TRANSMITTER: SEND

USER -> NETWORK.CELL TOWER: MAKE CALL
USER -> NETWORK.ONLINE PORTAL.UI: ACCESS

API SERVER -> NETWORK.ONLINE PORTAL.UI: DISPLAY
API SERVER -> LOGS: PERSIST

NETWORK.DATA PROCESSOR -> API SERVER
```

---

## Core Concepts

### Shapes (Nodes)

Shapes are the fundamental building blocks of D2 diagrams.

**Creating Shapes:**

```d2
# Simple shape
server

# Shape with label
server: My Server

# Multiple shapes on one line (using semicolons)
x; y; z
```

### Connections (Edges)

Connections define relationships between shapes.

**Connection Types:**

```d2
# Four valid connection types
x -- y      # Undirected
x -> y      # Directed (arrow to the right)
x <- y      # Directed (arrow to the left)
x <-> y     # Bidirectional
```

**Connection Labels:**

```d2
x -> y: connection label
```

**Connection Chaining:**

```d2
a -> b -> c -> d
```

### Containers

Containers are shapes that contain other shapes.

**Two Ways to Create Containers:**

```d2
# Method 1: Dot notation
aws.server

# Method 2: Map syntax
aws: {
  server
  database
}
```

### Scoping and References

**Referencing Parent:**

```d2
birthday: {
  presents
}

christmas: {
  presents
  _.birthday.presents -> presents: regift
}
```

The underscore `_` refers to the parent scope.

---

## Shapes

### Default Shape Type

By default, all shapes are rectangles.

### Shape Types

Available shape types:

```d2
rectangle (default)
square
circle
oval
diamond
parallelogram
hexagon
triangle
cylinder
queue
package
step
callout
stored_data
person
diamond
oval
cloud
text
code
class
sql_table
image
sequence_diagram
```

**Setting Shape Type:**

```d2
user: {
  shape: person
}

database: {
  shape: cylinder
}
```

### Special Shapes

**Multiple Attribute:**

Makes shapes appear stacked (useful for representing multiple instances):

```d2
databases: {
  shape: cylinder
  style.multiple: true
}
```

**3D Shapes:**

```d2
server: {
  shape: rectangle
  style.3d: true
}
```

---

## Connections

### Basic Connections

```d2
# Simple connection
x -> y

# Connection with label
x -> y: sends data

# Undeclared shapes are auto-created
new_shape -> another_shape
```

### Connection Chaining

```d2
a -> b -> c
# Creates: a -> b and b -> c
```

### Repeated Connections

```d2
Database -> S3: backup
Database -> S3: backup

# This creates TWO separate connections, not one
```

### Connection Referencing

Reference connections by index:

```d2
x -> y: hi
x -> y: hello

# Reference the first connection
(x -> y)[0].style.stroke: red
```

### Arrowheads

**Default Arrowhead Types:**
- `triangle` (default)
- `arrow` (pointier triangle)
- `diamond`
- `circle`
- `box`
- `cf-one`, `cf-one-required` (crow's foot notation)
- `cf-many`, `cf-many-required`
- `cross`

**Customizing Arrowheads:**

```d2
x -> y: {
  source-arrowhead: 1
  target-arrowhead: * {
    shape: diamond
    style.filled: true
  }
}
```

**Arrowhead Labels:**

```d2
x -> y: {
  source-arrowhead.label: 1
  target-arrowhead.label: *
}
```

---

## Containers

### Creating Containers

**Dot Notation (Quick Method):**

```d2
server.process
# Creates server container with process inside
```

**Map Syntax (Structured Method):**

```d2
server: {
  process
  database
}
```

### Container Labels

**Method 1: Shorthand**

```d2
aws.server: My Server
```

**Method 2: Reserved Keyword**

```d2
aws: {
  label: Amazon Web Services
  server
}
```

### Nested Containers

```d2
cloud: {
  aws: {
    load_balancer
    api
    db
  }
  
  gcp: {
    compute_engine
  }
}
```

### Connecting to Containers

```d2
users -> cloud.aws.api
cloud.aws.api -> cloud.aws.db
```

---

## Styling

### Style Attributes

All styles are set under the `style` keyword.

**Available Style Properties:**

```d2
shape.style.opacity: 0.5          # Float between 0 and 1
shape.style.stroke: "#ff0000"     # CSS color name, hex, or gradient
shape.style.fill: "#0000ff"       # CSS color name, hex, or gradient
shape.style.stroke-width: 2       # Integer between 1 and 15
shape.style.stroke-dash: 5        # Integer between 0 and 10
shape.style.border-radius: 8      # Integer between 0 and 20
shape.style.shadow: true          # Boolean
shape.style.3d: true              # Boolean
shape.style.multiple: true        # Boolean (stacked appearance)
shape.style.font-size: 14         # Integer
shape.style.font-color: "#000000" # CSS color
shape.style.bold: true            # Boolean
shape.style.italic: true          # Boolean
shape.style.underline: true       # Boolean
```

### Applying Styles

**Method 1: Block Syntax**

```d2
x -> y: {
  style: {
    opacity: 0.9
    stroke-dash: 3
    shadow: true
    font-size: 10
  }
}
```

**Method 2: Dot Notation**

```d2
x -> y: hi
x.style.opacity: 0.9
x.style.stroke-dash: 3
x.style.shadow: true
x.style.font-size: 10
x.style.3d: true
```

### Gradients

D2 supports CSS gradient strings:

```d2
x: {
  style.fill: "linear-gradient(to right, #ff0000, #00ff00)"
}
```

### Root-Level Styles

```d2
# Diagram background color
style.fill: "#f0f0f0"
```

---

## Themes

### Built-in Themes

D2 includes multiple professional themes designed by professional designers.

**Available Themes:**
- Theme 0: Neutral Default
- Theme 1: Neutral Grey
- Theme 2: Flagship Terrastruct
- Theme 3: Cool Classics
- Theme 4: Mixed Berry Blue
- Theme 5: Grape Soda
- Theme 6: Aubergine
- Theme 7: Colorblind Clear
- Theme 8: Vanilla Nitro Cola
- Theme 100-104: Terminal themes
- Theme 200-203: Origami variants
- Theme 300-303: Terminal Grayscale variants

### Setting Themes

**CLI Flag:**

```bash
# Use theme 101
d2 -t 101 input.d2 out.svg

# Specify different theme for dark mode
d2 -t 0 --dark-theme 200 input.d2 out.svg
```

**Environment Variable:**

```bash
export D2_THEME=101
d2 input.d2 out.svg
```

**In D2 File:**

```d2
vars: {
  d2-config: {
    theme-id: 101
  }
}
```

### Sketch Mode

Make diagrams look hand-drawn:

```bash
d2 --sketch input.d2 out.svg
```

---

## Layouts

D2 supports multiple layout engines, each optimized for different diagram types.

### Available Layout Engines

1. **dagre** (Default, Free)
   - Fast directed graph layout
   - Hierarchical/layered layouts
   - Based on Graphviz's DOT algorithm
   - Bundled with D2

2. **ELK** (Free)
   - Directed graph layout
   - Good for node-link diagrams with ports
   - Supports hierarchical layouts
   - Bundled with D2

3. **TALA** (Commercial)
   - Advanced general-purpose layout engine
   - Developed by Terrastruct
   - Best for complex software architecture diagrams
   - Handles containers and clusters well
   - Dynamic label positioning
   - Per-container direction control

### Setting Layout Engine

**CLI Flag:**

```bash
d2 --layout elk input.d2 out.svg
```

**In D2 File:**

```d2
vars: {
  d2-config: {
    layout-engine: elk
  }
}
```

### Direction

Control the flow direction of your diagram:

```d2
direction: down    # Default
direction: right
direction: left
direction: up
```

**Per-Container Direction (TALA only):**

```d2
container: {
  direction: right
  x -> y
}
```

---

## Advanced Features

### Text and Code Blocks

**Text Shapes:**

```d2
explanation: |md
  # This is a markdown block
  
  You can write **bold** and *italic* text.
  
  - List item 1
  - List item 2
| {
  shape: text
}
```

**Code Blocks:**

```d2
server_code: |go
  func main() {
    fmt.Println("Hello, World!")
  }
| {
  shape: code
}
```

Supported languages: Most common programming languages with syntax highlighting.

### SQL Tables

```d2
users: {
  shape: sql_table
  id: int {constraint: primary_key}
  name: varchar(255)
  email: varchar(255) {constraint: unique}
  created_at: timestamp
}

posts: {
  shape: sql_table
  id: int {constraint: primary_key}
  user_id: int {constraint: foreign_key}
  title: varchar(255)
  content: text
}

users.id -> posts.user_id
```

With TALA or ELK, connections point to exact rows.

### UML Class Diagrams

```d2
MyClass: {
  shape: class
  
  # Fields
  -privateField: int
  +publicField: string
  #protectedField: bool
  
  # Methods
  +publicMethod(): void
  -privateMethod(param: string): int
}
```

### Sequence Diagrams

```d2
sequence_diagram: {
  shape: sequence_diagram
  
  alice
  bob
  charlie
  
  alice -> bob: Hello
  bob -> charlie: Hi there
  charlie -> alice: Hey!
}
```

### Icons

**Adding Icons:**

```d2
server: {
  icon: https://icons.terrastruct.com/aws/Compute/Amazon-EC2.svg
}
```

**Icon Shapes:**

```d2
logo: {
  shape: image
  icon: https://example.com/logo.png
}
```

Icon placement is automatic based on whether it's a container and label existence.

### Markdown

```d2
explanation: |md
  # Heading
  
  This is **bold** and this is *italic*.
  
  ## Code example
  ```python
  print("Hello, World!")
  ```
  
  ## Lists
  - Item 1
  - Item 2
  - Item 3
|
```

### LaTeX

```d2
formula: |latex
  E = mc^2
|
```

### Grid Diagrams

```d2
grid-diagram: {
  grid-rows: 3
  grid-columns: 3
  grid-gap: 20
  
  a: {grid-row: 1; grid-column: 1}
  b: {grid-row: 1; grid-column: 2}
  c: {grid-row: 2; grid-column: 1}
}
```

### Near Positioning

Control shape positioning:

```d2
x: {
  near: top-left
}

y: {
  near: center
}

z: {
  near: bottom-right
}
```

Available positions: `top-left`, `top-center`, `top-right`, `center-left`, `center`, `center-right`, `bottom-left`, `bottom-center`, `bottom-right`

### Interactive Features

**Tooltips:**

```d2
x: {
  tooltip: This is a tooltip that appears on hover
}
```

**Links:**

```d2
github: {
  link: https://github.com
}
```

### Variables

Define reusable values:

```d2
vars: {
  primary-color: "#3b82f6"
  secondary-color: "#10b981"
}

x: {
  style.fill: ${primary-color}
}

y: {
  style.fill: ${secondary-color}
}
```

### Globs

Apply styles to multiple objects:

```d2
# Style all shapes
*.style.fill: "#f0f0f0"

# Style all connections
**.style.stroke: "#000000"

# Style all containers
***.style.stroke-width: 2
```

### Classes

Define reusable style sets:

```d2
classes: {
  important: {
    style.fill: red
    style.stroke: darkred
    style.bold: true
  }
  
  normal: {
    style.fill: white
    style.stroke: gray
  }
}

x: {
  class: important
}

y: {
  class: normal
}
```

### Composition (Multiple Boards)

Create multiple diagrams in one file:

```d2
# Base layer
layers: {
  base: {
    x -> y
  }
  
  # Add more details in another layer
  detailed: {
    x -> y
    y -> z
    z -> x
  }
}
```

View different layers in the output.

### Scenarios

Create variants of the same diagram:

```d2
scenarios: {
  normal: {
    server.style.fill: green
  }
  
  error: {
    server.style.fill: red
    error_message: Server is down!
  }
}
```

### Imports

Split diagrams across multiple files:

**main.d2:**
```d2
...@components.d2
...@connections.d2
```

**components.d2:**
```d2
server
database
cache
```

**connections.d2:**
```d2
server -> database
server -> cache
```

### Animations

Create animated diagrams:

```d2
scenarios: {
  frame1: {
    a -> b
  }
  
  frame2: {
    a -> b
    b -> c
  }
  
  frame3: {
    a -> b
    b -> c
    c -> a
  }
}
```

Render with animation:

```bash
d2 --animate-interval 1000 input.d2 out.gif
```

---

## CLI Usage

### Basic Commands

```bash
# Compile D2 file to SVG
d2 input.d2 output.svg

# Watch mode with live reload
d2 --watch input.d2 output.svg

# Specify output format
d2 input.d2 output.png  # PNG
d2 input.d2 output.pdf  # PDF
```

### Common Flags

```bash
-w, --watch              # Watch for changes and live reload
-t, --theme INT          # Set theme ID (0-999)
--dark-theme INT         # Theme for dark mode
-l, --layout STRING      # Layout engine (dagre, elk, tala)
-s, --sketch             # Hand-drawn style
--pad INT                # Padding around diagram
--center                 # Center the diagram
--scale FLOAT            # Scale output (e.g., 0.5 for half size)
-b, --bundle             # Bundle multiple files into one
--animate-interval INT   # Animation frame duration (ms)
--timeout INT            # Timeout for rendering (seconds)
--font-regular PATH      # Custom regular font
--font-bold PATH         # Custom bold font
--font-italic PATH       # Custom italic font
-h, --host STRING        # Host for watch mode (default: localhost)
-p, --port INT           # Port for watch mode (default: auto)
```

### Environment Variables

```bash
D2_LAYOUT=elk           # Default layout engine
D2_THEME=101            # Default theme
D2_PAD=100              # Default padding
D2_SKETCH=1             # Enable sketch mode
D2_TIMEOUT=120          # Render timeout
```

### Examples

```bash
# Watch with specific theme and layout
d2 -w -t 101 -l elk input.d2 output.svg

# Sketch mode with custom port
d2 --watch --sketch --port 8080 input.d2 output.svg

# Export with high resolution
d2 --scale 2 input.d2 output.png

# Animated GIF
d2 --animate-interval 500 input.d2 animation.gif

# Custom fonts
d2 --font-regular ./fonts/custom.ttf input.d2 output.svg

# List available themes
d2 themes
```

---

## API

D2 provides a Go API for programmatically creating and manipulating diagrams.

### Basic Usage

```go
import "oss.terrastruct.com/d2/d2oracle"

// Create a new diagram
g, err := d2oracle.Create(nil)

// Add shapes
g.Create(nil, "server")
g.Create(nil, "database")

// Add connection
g.Create(nil, "server.db_connection")
g.Connect(nil, "server", "database", "db_connection")

// Set attributes
g.Set(nil, "server.shape", nil, "rectangle")
g.Set(nil, "database.shape", nil, "cylinder")

// Compile to SVG
svg, err := d2.Compile(g, &d2.CompileOptions{
    LayoutEngine: "elk",
})
```

### Oracle Operations

The D2 Oracle API provides methods for diagram manipulation:

**Create:**
```go
Create(boardPath, key string) error
```

**Set:**
```go
Set(boardPath, key, field string, value interface{}) error
```

**Delete:**
```go
Delete(boardPath, key string) error
```

**Rename:**
```go
Rename(boardPath, key, newName string) error
```

**Move:**
```go
Move(boardPath, key, newKey string) error
```

**Connect:**
```go
Connect(boardPath, srcKey, dstKey, label string) error
```

### Use Cases

- Generate diagrams from code
- Automate diagram creation from data
- Build diagram editors
- Create custom tooling
- Generate documentation programmatically

---

## Examples

### Simple Architecture Diagram

```d2
direction: right

user: {
  shape: person
}

user -> load_balancer: HTTPS

load_balancer: {
  shape: rectangle
  icon: https://icons.terrastruct.com/aws/Networking%20&%20Content%20Delivery/Elastic-Load-Balancing.svg
}

load_balancer -> api_server: distributes
load_balancer -> api_server
load_balancer -> api_server

api_server: API Servers {
  shape: rectangle
  style.multiple: true
}

api_server -> database: queries
api_server -> cache: reads/writes

database: {
  shape: cylinder
  icon: https://icons.terrastruct.com/dev/postgresql.svg
}

cache: {
  shape: cylinder
  icon: https://icons.terrastruct.com/dev/redis.svg
}
```

### Microservices Architecture

```d2
vars: {
  d2-config: {
    theme-id: 101
  }
}

direction: down

# API Gateway
gateway: API Gateway {
  shape: hexagon
}

# Microservices
user_service: User Service {
  shape: rectangle
  db: {
    shape: cylinder
  }
}

order_service: Order Service {
  shape: rectangle
  db: {
    shape: cylinder
  }
}

payment_service: Payment Service {
  shape: rectangle
  db: {
    shape: cylinder
  }
}

notification_service: Notification Service {
  shape: rectangle
  queue: {
    shape: queue
  }
}

# Connections
gateway -> user_service
gateway -> order_service
gateway -> payment_service

order_service -> payment_service: process payment
order_service -> notification_service: send notification

user_service -> user_service.db
order_service -> order_service.db
payment_service -> payment_service.db
notification_service -> notification_service.queue
```

### Sequence Diagram

```d2
shape: sequence_diagram

user
frontend
backend
database

user -> frontend: Click button
frontend -> backend: API Request
backend -> database: Query
database -> backend: Results
backend -> frontend: Response
frontend -> user: Display data
```

### ER Diagram

```d2
users: {
  shape: sql_table
  id: int {constraint: primary_key}
  email: varchar
  name: varchar
  created_at: timestamp
}

posts: {
  shape: sql_table
  id: int {constraint: primary_key}
  user_id: int {constraint: foreign_key}
  title: varchar
  content: text
  published_at: timestamp
}

comments: {
  shape: sql_table
  id: int {constraint: primary_key}
  post_id: int {constraint: foreign_key}
  user_id: int {constraint: foreign_key}
  content: text
  created_at: timestamp
}

users.id -> posts.user_id
users.id -> comments.user_id
posts.id -> comments.post_id
```

### Cloud Infrastructure

```d2
vars: {
  d2-config: {
    layout-engine: elk
    theme-id: 300
  }
}

aws: AWS {
  style.fill: "#FF9900"
  style.stroke: "#FF9900"
  
  vpc: VPC {
    public_subnet: Public Subnet {
      load_balancer: ALB
      nat_gateway: NAT Gateway
    }
    
    private_subnet: Private Subnet {
      ec2_instances: EC2 Instances {
        style.multiple: true
      }
      
      rds: RDS Database {
        shape: cylinder
      }
    }
  }
  
  s3: S3 Bucket {
    shape: cylinder
  }
  
  cloudfront: CloudFront CDN
}

internet: Internet {
  shape: cloud
}

internet -> aws.vpc.public_subnet.load_balancer: HTTPS
aws.vpc.public_subnet.load_balancer -> aws.vpc.private_subnet.ec2_instances
aws.vpc.private_subnet.ec2_instances -> aws.vpc.private_subnet.rds
aws.vpc.private_subnet.ec2_instances -> aws.s3
aws.cloudfront -> aws.s3: serves static content
```

---

## Resources

### Official Links

- **Website**: https://d2lang.com
- **Documentation**: https://d2lang.com/tour/intro
- **Playground**: https://play.d2lang.com
- **GitHub**: https://github.com/terrastruct/d2
- **GitHub Docs**: https://github.com/terrastruct/d2-docs
- **Discord**: https://discord.com/invite/pbUXgvmTpU

### Editor Support

- **VSCode Extension**: Official extension available in marketplace
- **Vim Plugin**: Official vim support
- **Tree-sitter Grammar**: Available for editors supporting tree-sitter

### Related Tools

- **D2 Studio**: Professional diagramming IDE (https://terrastruct.com/d2-studio)
- **JavaScript Wrapper**: https://github.com/Kreshnik/d2lang-js
- **Python Wrapper**: https://github.com/MrBlenny/py-d2
- **Markdown Plugins**: 
  - MkDocs: https://github.com/landmaj/mkdocs-d2-plugin
  - MdBook: https://github.com/danieleades/mdbook-d2
- **Database Importers**:
  - PostgreSQL: https://github.com/zekenie/d2-erd-from-postgres
  - MongoDB: https://github.com/novuhq/mongo-to-D2
  - MySQL: https://github.com/JDOsborne1/db_to_d2

### Community

- **Discord Server**: Active community for help and discussions
- **GitHub Discussions**: Feature requests and Q&A
- **GitHub Issues**: Bug reports and feature tracking

### Learning Resources

- **Official Tour**: https://d2lang.com/tour/intro (5-10 minute quickstart)
- **Examples Gallery**: https://d2lang.com/examples/overview
- **Blog Posts**: https://d2lang.com/blog
- **Cheat Sheet**: Available at https://d2lang.com/tour/cheat-sheet

### Commercial Products

- **D2 Studio**: Professional IDE with advanced features
- **TALA Layout Engine**: Commercial layout engine for complex diagrams
- **Terrastruct Platform**: Enterprise diagramming solutions

---

## Quick Reference Card

### Basic Syntax

```d2
# Shapes
shape_name
shape_name: Label

# Connections
a -> b          # Directed
a -- b          # Undirected
a <-> b         # Bidirectional
a -> b: label   # With label

# Containers
container.child
container: {
  child1
  child2
}

# Styling
shape.style.fill: red
shape.style.stroke: blue

# Themes
vars: {
  d2-config: {
    theme-id: 101
  }
}
```

### Common Shapes

```d2
shape: rectangle  # Default
shape: circle
shape: cylinder
shape: person
shape: cloud
shape: hexagon
shape: sql_table
shape: class
shape: sequence_diagram
```

### Common Styles

```d2
style.fill: "#ff0000"
style.stroke: "#000000"
style.opacity: 0.5
style.stroke-dash: 3
style.shadow: true
style.3d: true
style.multiple: true
style.bold: true
style.italic: true
```

---

## Changelog & Version Info

D2 is actively maintained with regular updates. Check the releases page for the latest version:
https://github.com/terrastruct/d2/releases

Current stable version focuses on:
- Improved layout engines
- Enhanced styling options
- Better performance
- Additional shape types
- Extended theme library

---

## Contributing

D2 is open source and welcomes contributions!

**Ways to Contribute:**
- Report bugs on GitHub Issues
- Suggest features on GitHub Discussions
- Submit pull requests
- Create themes
- Write documentation
- Build integrations and tools

**Development:**
- Written in Go
- Uses standard Go tooling
- Comprehensive test suite
- Active CI/CD pipeline

See https://github.com/terrastruct/d2/blob/master/docs/CONTRIBUTING.md for details.

---

## Troubleshooting

### Common Issues

**Installation Issues:**
- Ensure Go is up to date if installing from source
- Check PATH configuration
- Verify internet connection for install script

**Rendering Issues:**
- Try different layout engines (`-l elk`, `-l dagre`)
- Increase timeout for complex diagrams
- Check syntax for errors
- Use `--debug` flag for more information

**Layout Issues:**
- Use `direction` keyword to control flow
- Try different layout engines
- Adjust `pad` setting
- Use `near` for positioning hints

**Performance:**
- For large diagrams, consider splitting into multiple files
- Use TALA for complex layouts (if available)
- Increase `--timeout` for heavy rendering

### Getting Help

1. Check the official documentation
2. Search GitHub Issues
3. Ask on Discord
4. Post on GitHub Discussions
5. Review examples in the gallery

---

## Conclusion

D2 represents a paradigm shift in how engineers create and maintain diagrams. By treating diagrams as code, it brings software engineering best practices to documentation:

- **Version Control**: Track changes over time
- **Code Review**: Review diagram changes like code
- **Automation**: Generate diagrams programmatically
- **Collaboration**: Easy to share and edit
- **Maintenance**: Quick updates without manual repositioning

Whether you're documenting microservices, designing systems, creating ER diagrams, or building presentations, D2 provides a powerful, flexible, and developer-friendly solution.

Start creating diagrams today: https://d2lang.com

---

*This documentation was compiled from official D2 sources and is current as of February 2026. For the most up-to-date information, always refer to the official D2 documentation at https://d2lang.com*

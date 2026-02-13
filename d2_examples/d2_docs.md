# D2 COMPACT REFERENCE (v0.7.1)

## HELLO-WORLD

# Hello World
    
```d2
x -> y: hello world
```

<div
className="embedSVG" dangerouslySetInnerHTML={{__html: require('@site/static/img/generated/hello-world.svg2')}}></div>
This declares a connection between two shapes, `x` and `y`, with the label, `hello world`.

## SHAPES

# Shapes
## Basics
You can declare shapes like so:
    
```d2
imAShape
im_a_shape
im a shape
i'm a shape
# notice that one-hyphen is not a connection
# whereas, `a--shape` would be a connection
a-shape
```

You can also use semicolons to define multiple shapes on the same line:
```d2
SQLite; Cassandra
```
By default, a shape's label is the same as the shape's key. But if you want it to be different, assign a new label like so:
```d2
pg: PostgreSQL
```
By default, a shape's type is `rectangle`. To specify otherwise, provide the field `shape`:
```d2
Cloud: my cloud
Cloud.shape: cloud
```
## Example
    
```d2
pg: PostgreSQL
Cloud: my cloud
Cloud.shape: cloud
SQLite
Cassandra
```

Keys are case-insensitive, so `postgresql` and `postgreSQL` will reference the same shape.
 Shape catalog
There are other values that `shape` can take, but they're special types that are covered
in the next section.
## 1:1 Ratio shapes
Some shapes maintain a 1:1 aspect ratio, meaning their width and height are always equal.
- `circle`
- `square`
For these shapes, if you have a long label that make the shape wider, D2 will also make
the shape taller to maintain the 1:1 ratio.
If you manually set `width` and `height` on a 1:1 ratio shape, both dimensions will be set
to the larger of the two values to maintain the aspect ratio.

## CONNECTIONS

# Connections
Connections define relationships between shapes.
## Basics
Hyphens/arrows in between shapes define a connection.
```d2
Write Replica Canada <-> Write Replica Australia
Read Replica <- Master
Write Replica -> Master
Read Replica 1 -- Read Replica 2
```
If you reference an undeclared shape in a connection, it's created (as shown in the [hello
world](hello-world.md) example).
There are 4 valid ways to define a connection:
- `--`
- `->`
- `<-`
- `<->`
### Connection labels
```d2
Read Replica 1 -- Read Replica 2: Kept in sync
```
### Connections must reference a shape's key, not its label.
```d2
be: Backend
fe: Frontend
# This would create new shapes
Backend -> Frontend
# This would define a connection over existing labels
be -> fe
```
## Example
    
```d2
Write Replica Canada <-> Write Replica Australia

Read Replica <- Master

x -- y

super long shape id here -> super long shape id even longer here
```

## Repeated connections
Repeated connections do not override existing connections. They declare new ones.
    
```d2
Database -> S3: backup
Database -> S3
Database -> S3: backup
```

## Connection chaining
For readability, it may look more natural to define multiple connection in a single line.
    
```d2
# The label applies to each connection in the chain.
High Mem Instance -> EC2 <- High CPU Instance: Hosted By
```

## Cycles are okay
    
```d2
Stage One -> Stage Two -> Stage Three -> Stage Four
Stage Four -> Stage One: repeat
```

## Arrowheads
To override the default arrowhead shape or give a label next to arrowheads, define a special shape on connections named `source-arrowhead` and/or `target-arrowhead`.
    
```d2
a: The best way to avoid responsibility is to say, "I've got responsibilities"
b: Whether weary or unweary, O man, do not rest
c: I still maintain the point that designing a monolithic kernel in 1991 is a

a -> b: To err is human, to moo bovine {
  source-arrowhead: 1
  target-arrowhead: * {
    shape: diamond
  }
}

b <-> c: "Reality is just a crutch for people who can't handle science fiction" {
  source-arrowhead.label: 1
  target-arrowhead: * {
    shape: diamond
    style.filled: true
  }
}

d: A black cat crossing your path signifies that the animal is going somewhere

d -> a -> c
```

 Arrowhead options
- `triangle` (default)
  - Can be further styled as `style.filled: false`.
- `arrow` (like triangle but pointier)
- `diamond`
  - Can be further styled as `style.filled: true`.
- `circle`
  - Can be further styled as `style.filled: true`.
- `box`
  - Can be further styled as `style.filled: true`.
- `cf-one`, `cf-one-required` (cf stands for crows foot)
- `cf-many`, `cf-many-required`
- `cross`
It's recommended the arrowhead labels be kept short. They do not go through
autolayout for optimal positioning like regular labels do, so long arrowhead labels are
more likely to collide with surrounding objects.
If the connection does not have an endpoint, arrowheads won't do anything.
For example, the following will do nothing, because there is no source arrowhead.
```d2
x -> y: {
  source-arrowhead.shape: diamond
}
```
## Referencing connections
You can reference a connection by specifying the original ID followed by its index.
    
```d2
x -> y: hi
x -> y: hello

(x -> y)[0].style.stroke: red
(x -> y)[1].style.stroke: blue
```

## CONTAINERS

# Containers
    
```d2
server
# Declares a shape inside of another shape
server.process

# Can declare the container and child in same line
im a parent.im a child

# Since connections can also declare keys, this works too
apartment.Bedroom.Bathroom -> office.Spare Room.Bathroom: Portal
```

## Nested syntax
You can avoid repeating containers by creating nested maps.
    
```d2
clouds: {
  aws: {
    load_balancer -> api
    api -> db
  }
  gcloud: {
    auth -> db
  }

  gcloud -> aws
}
```

## Container labels
There are two ways define container labels.
### 1. Shorthand container labels
```d2-incomplete
gcloud: Google Cloud {
  ...
}
```
### 2. Reserved keyword `label`
```d2-incomplete
gcloud: {
  label: Google Cloud
  ...
}
```
## Example
    
```d2
clouds: {
  aws: AWS {
    load_balancer -> api
    api -> db
  }
  gcloud: Google Cloud {
    auth -> db
  }

  gcloud -> aws
}

users -> clouds.aws.load_balancer
users -> clouds.gcloud.auth

ci.deploys -> clouds
```

## Reference parent
Sometimes you want to reference something outside of the container from within. The
underscore (`_`) refers to parent.
    
```d2
christmas: {
  presents
}
birthdays: {
  presents
  _.christmas.presents -> presents: regift
  _.christmas.style.fill: "#ACE1AF"
}
```

## STYLE

# Styles
If you'd like to customize the style of a shape, the following reserved keywords can be
set under the `style` field.
Below is a catalog of all valid styles, applied individually to this baseline diagram.
The following SVGs are rendered with `direction: right`, but omitted from the shown scripts for
brevity.
Want to change the default styles for shapes and/or connections? See [globs to change defaults](/tour/globs/#changing-defaults).
## Style keywords
- [opacity](#opacity)
- [stroke](#stroke)
- [fill](#fill) (shape only)
- [fill-pattern](#fill-pattern) (shape only)
- [stroke-width](#stroke-width)
- [stroke-dash](#stroke-dash)
- [border-radius](#border-radius)
- [shadow](#shadow) (shape only)
- [3D](#3d) (rectangle/square only)
- [multiple](#multiple) (shape only)
- [double-border](#double-border) (rectangles and ovals)
- [font](#font)
- [font-size](#font-size)
- [font-color](#font-color)
- [animated](#animated)
- [bold, italic, underline](#bold-italic-underline)
- [text-transform](#text-transform)
- [root](#root)
## Opacity
Float between `0` and `1`.
    
```d2
direction: right
x -> y: hi {
  style: {
    opacity: 0.4
  }
}
x.style.opacity: 0
y.style.opacity: 0.7
```

## Stroke
CSS color name, hex code, or a subset of CSS gradient strings.
    
```d2
direction: right
x -> y: hi {
  style: {
    # All CSS color names are valid
    stroke: deepskyblue
  }
}
# We need quotes for hex otherwise it gets interpreted as comment
x.style.stroke: "#f4a261"
```

<br/>
For `sql_table`s and `class`es, `stroke` is applied as `fill` to the body (since `fill` is
already used to control header's `fill`).
## Fill
CSS color name, hex code, or a subset of CSS gradient strings.
    
```d2
direction: right
x -> y: hi
y -> z
x.style.fill: "#f4a261"
y.style.fill: honeydew
z.style.fill: "linear-gradient(#f69d3c, #3f87a6)"
```

<br/>
For `sql_table`s and `class`es, `fill` is applied to the header.
Want transparent?
    
```d2
x: {
  y
  y.style.fill: transparent
}
x.style.fill: PapayaWhip
```

## Fill Pattern
Available patterns:
- `dots`
- `lines`
- `grain`
- `none` (for cancelling out ones set by certain themes)
    
```d2
direction: right
style.fill-pattern: dots
x -> y: hi
x.style.fill-pattern: lines
y.style.fill-pattern: grain
```

## Stroke Width
Integer between `1` and `15`.
    
```d2
direction: right
x -> y: hi {
  style: {
    stroke-width: 8
  }
}
x.style.stroke-width: 1
```

## Stroke Dash
Integer between `0` and `10`.
    
```d2
direction: right
x -> y: hi {
  style: {
    stroke-dash: 3
  }
}
x.style.stroke-dash: 5
```

## Border Radius
Integer between `0` and `20`.
    
```d2
direction: right
x -> y: hi
x.style.border-radius: 3
y.style.border-radius: 8
```

`border-radius` works on connections too, which controls how rounded the corners are. This
only applies to layout engines that use corners (e.g. ELK), and of course, only has effect
on connections whose routes have corners.
Specifying a very high value creates a "pill" effect.
    
```d2
tylenol.style.border-radius: 999
```

## Shadow
`true` or `false`.
    
```d2
direction: right
x -> y: hi
x.style.shadow: true
```

## 3D
`true` or `false`.
    
```d2
direction: right
x -> y: hi
x.style.3d: true
```

## Multiple
`true` or `false`.
    
```d2
direction: right
x -> y: hi
x.style.multiple: true
```

## Double Border
`true` or `false`.
    
```d2
direction: right
x -> y: hi
x.style.double-border: true
y.shape: circle
y.style.double-border: true
```

## Font
Currently the only option is to specify `mono`. More coming soon.
    
```d2
direction: right
x -> y: hi {
  style: {
    font: mono
  }
}
x.style.font: mono
y.style.font: mono
```

## Font Size
Integer between `8` and `100`.
    
```d2
direction: right
x -> y: hi {
  style: {
    font-size: 28
  }
}
x.style.font-size: 8
y.style.font-size: 55
```

## Font Color
CSS color name, hex code, or a subset of CSS gradient strings.
    
```d2
direction: right
x -> y: hi {
  style: {
    font-color: red
  }
}
x.style.font-color: "#f4a261"
```

<br/>
For `sql_table`s and `class`es, `font-color` is applied to the header text only (theme
controls other colors in the body).
## Animated
`true` or `false`.
    
```d2
direction: right
x -> y: hi {
  style.animated: true
}
x.style.animated: true
```

## Bold, italic, underline
`true` or `false`.
    
```d2
direction: right
x -> y: hi {
  style: {
    bold: true
  }
}
x.style.underline: true
y.style.italic: true
# By default, shape labels are bold. Bold has precedence over italic, so unbold to see
# italic style
y.style.bold: false
```

## Text transform
`text-transform` changes the casing of labels.
- `uppercase`
- `lowercase`
- `title`
- `none` (used for negating caps lock that special themes may apply)
    
```d2
direction: right
TOM -> jerry: hi {
  style: {
    text-transform: capitalize
  }
}
TOM.style.text-transform: lowercase
jerry.style.text-transform: uppercase
```

## Root
Some styles are applicable at the root level. For example, to set a diagram background
color, use `style.fill`.
Currently the set of supported keywords are:
- `fill`: diagram background color
- `fill-pattern`: background fill pattern
- `stroke`: frame around the diagram
- `stroke-width`
- `stroke-dash`
- `double-border`: two frames, which is a popular framing method
    
```d2
direction: right
x -> y: hi
style: {
  fill: LightBlue
  stroke: FireBrick
  stroke-width: 2
}
```

All diagrams in this documentation are rendered with `pad=0`. If you're using `stroke` to
create a frame for your diagram, you'll likely want to add some padding.

## CLASSES

# Classes
Classes let you aggregate attributes and reuse them.
    
```d2
direction: right
classes: {
  load balancer: {
    label: load\nbalancer
    width: 100
    height: 200
    style: {
      stroke-width: 0
      fill: "#44C7B1"
      shadow: true
      border-radius: 5
    }
  }
  unhealthy: {
    style: {
      fill: "#FE7070"
      stroke: "#F69E03"
    }
  }
}

web traffic -> web lb
web lb.class: load balancer

web lb -> api1
web lb -> api2
web lb -> api3

api2.class: unhealthy

api1 -> cache lb
api3 -> cache lb

cache lb.class: load balancer
```

## Connection classes
As a reminder of D2 syntax, you can apply classes to connections both at the initial
declaration as well as afterwards.
On initial declaration:
```d2
a -> b: {class: something}
```
Targeting:
```d2
a -> b
# ...
(a -> b)[0].class: something
```
## Overriding classes
If your object defines an attribute that the class also has defined, the object's
attribute overrides the class attribute.
    
```d2
classes: {
  unhealthy: {
    style.fill: red
  }
}
x.class: unhealthy
x.style.fill: orange
```

## Multiple classes
You may use arrays for the value as well to apply multiple classes.
    
```d2
classes: {
  d2: {
    label: ""
    icon: https://play.d2lang.com/assets/icons/d2-logo.svg
  }
  sphere: {
    shape: circle
    style.stroke-width: 0
  }
}

logo.class: [d2; sphere]
```

### Order matters
When multiple classes are given, they are applied left-to-right.
    
```d2
classes: {
  uno: {
    label: 1
  }
  dos: {
    label: 2
  }
}

x.class: [uno; dos]
y.class: [dos; uno]
```

## Advanced: Using classes as tags
If you want to post-process D2 diagrams, you can also use classes to arbitrarily tag
objects. Any `class` you apply is written into the SVG element as a `class` attribute. So
for example, you can then apply custom CSS like `.stuff { ... }` (or use Javascript for
  onclick handlers and such) on a web page that D2 SVG is embedded in.

## VARS

# Variables & Substitutions
`vars` is a special keyword that lets you define variables. These variables can be
referenced with the substitution syntax: `${}`.
    
```d2
direction: right
vars: {
  server-name: Cat
}

server1: ${server-name}-1
server2: ${server-name}-2

server1 <-> server2
```

## Variables can be nested
Use `.` to refer to nested variables.
    
```d2
vars: {
  primaryColors: {
    button: {
      active: "#4baae5"
      border: black
    }
  }
}

button: {
  width: 100
  height: 40
  style: {
    border-radius: 5
    fill: ${primaryColors.button.active}
    stroke: ${primaryColors.button.border}
  }
}
```

## Variables are scoped
They work just like variable scopes in programming. Substitutions can refer to variables
defined in a more outer scope, but not a more inner scope. If a variable appears in two
scopes, the one closer to the substitution is used.
    
```d2
vars: {
  region: Global
}

lb: ${region} load balancer

zone1: {
  vars: {
    region: us-east-1
  }
  server: ${region} API
}
```

## Single quotes bypass substitutions
    
```d2
direction: right
vars: {
  names: John and Joyce
}
a -> b: 'Send field ${names}'
```

## Spread substitutions
If `x` is a map or an array, `...${x}` will spread the contents of `x` into either a map
or an array.
    
```d2
vars: {
  base-constraints: [NOT NULL; UNQ]
  disclaimer: DISCLAIMER {
    I am not a lawyer
    near: top-center
  }
}

data: {
  shape: sql_table
  a: int {constraint: [PK; ...${base-constraints}]}
}

custom-disclaimer: DRAFT DISCLAIMER {
  ...${disclaimer}
}
```

## Configuration variables
Some configurations can be made directly in `vars` instead of using flags or environment
variables.
    
```d2
vars: {
  d2-config: {
    theme-id: 4
    dark-theme-id: 200
    pad: 0
    center: true
    sketch: true
    layout-engine: elk
  }
}

direction: right
x -> y
```

This is equivalent to calling the following with no `vars`:
```shell
d2 --layout=elk --theme=4 --dark-theme=200 --pad=0 --sketch --center input.d2
```
 Precedence
Flags and environment variables take precedence.
In other words, if you call `D2_PAD=2 d2 --theme=1 input.d2`, it doesn't matter what
`theme-id` and `pad` are set to in `input.d2`'s `d2-config`; it will use the options from
the command.
 `data`
`data` is an anything-goes map of key-value pairs. This is used in contexts where
third-party software read configuration, such as when D2 is used as a library, or D2 is
run with an external plugin.
For example,
```d2
vars: {
  d2-config: {
    data: {
      power-level: 9000
    }
  }
}
```

## GLOBS

# Globs
 Etymology
> The glob command, short for global, originates in the earliest versions of Bell Labs' Unix... to expand wildcard characters in unquoted arguments ...
[https://en.wikipedia.org/wiki/Glob_(programming)](https://en.wikipedia.org/wiki/Glob_(programming))
Globs are a powerful language feature to make global changes in one line.
    
```d2
iphone 10
iphone 11 mini
iphone 11 pro
iphone 12 mini

*.height: 300
*.width: 140
*mini.height: 200
*pro.height: 400
```

## Globs apply backwards and forwards
In the following example, the instructions are as follows:
1. Create a shape `a`.
2. Apply a glob rule. This immediately applies to existing shapes, i.e., `a`.
3. Create a shape `b`. Existing glob rules are re-evaluated, and applied if they meet the
   criteria. This does, so it applies to `b`.
4. Same with `c`.
    
```d2
a

* -> y

b
c
```

## Globs are case insensitive
    
```d2
diddy kong
Donkey Kong

*kong.style.fill: brown
```

## Globs can appear multiple times
    
```d2
teacher
thriller
thrifter

t*h*r.shape: person
```

## Glob connections
You can use globs to create connections.
    
```d2
vars: {
  d2-config: {
    layout-engine: elk
  }
}

Spiderman 1
Spiderman 2
Spiderman 3

* -> *: 👉
```

Notice how self-connections were omitted. While not entirely consistent with what you may
expect from globs, we feel it is more pragmatic for this to be the behavior.
You can also use globs to target modifying existing connections.
    
```d2
lady 1
lady 2

barbie

lady 1 -> barbie: hi barbie
lady 2 -> barbie: hi barbie

(lady* -> barbie)[*].style.stroke: pink
```

## Scoped globs
Notice that in the below example, globs only apply to the scope they are specified in.
    
```d2
foods: {
  pizzas: {
    cheese
    sausage
    pineapple
    *.shape: circle
  }
  humans: {
    john
    james
    *.shape: person
  }
  humans.* -> pizzas.pineapple: eats
}
```

## Recursive globs
`**` means target recursively.
    
```d2
a: {
  b: {
    c
  }
}

**.style.border-radius: 7
```

    
```d2
zone-A: {
  machine A
  machine B: {
    submachine A
    submachine B
  }
}

zone-A.** -> load balancer
```

Notice how `machine B` was not captured. Similar to the exception with `* -> *` omitting
self-connections, recursive globs in connections also make an exception for practical
diagramming: it only applies to non-container (AKA leaf) shapes.
## Filters
Use `&` to filter what globs target. You may use any reserved keyword to filter on.
    
```d2
bravo team.shape: person
charlie team.shape: person
command center.shape: cloud
hq.shape: rectangle

*: {
  &shape: person
  style.multiple: true
}
```

### Property filters
Aside from reserved keywords, there are special property filters for more specific
targeting.
- `connected: true|false`
- `leaf: true|false`
    
```d2
**: {
  &connected: true
  style.fill: yellow
}

**: {
  &leaf: true
  style.stroke: red
}

container: {
  a -> b
}
c
```

### Filters on array values
If the filtered attribute has an array value, the filter will match if it matches any
element of the array.
    
```d2
the-little-cannon: {
  class: [server; deployed]
}
dino: {
  class: [internal; deployed]
}
catapult: {
  class: [server]
}

*: {
  &class: server
  style.multiple: true
}
```

### Globs as filter values
Globs can also appear in the value of a filter. `*` by itself as a value for a filter
means the key must be specified.
    
```d2
*: {
  &link: *
  style.fill: red
}

x.link: https://google.com
y
```

### AND filter
Adding multiple lines of filters counts as an AND.
    
```d2
*: {
  &shape: person
  &connected: true
  style.fill: red
}
(** -> **)[*]: {
  &src: a
  &dst: c
  style.stroke: yellow
}
a -> b
a.shape: person
a -> c
```

### Connection endpoint filters
Connections can be filtered by properties on their source and destination shapes.
    
```d2
a: {
  shape: circle
  style: {
    fill: blue
    opacity: 0.8
  }
}
b: {
  shape: rectangle
  style: {
    fill: red
    opacity: 0.5
  }
}
c: {
  shape: diamond
  style.fill: green
  style.opacity: 0.8
}
(* -> *)[*]: {
  &src.style.fill: blue
  style.stroke-dash: 3
}
(* -> *)[*]: {
  &dst.style.opacity: 0.8
  style.stroke: cyan
}
(* -> *)[*]: {
  &src.shape: rectangle
  &dst.style.fill: green
  style.stroke-width: 5
}
a -> b
b -> c
a -> c
```

Endpoint filters also work with IDs, e.g. `&src: b`.
Endpoint IDs are absolute. For example, `a.c` instead of just `c`, even if the glob is
declared within `a`.
## Inverse filters
Use `!&` to inverse-filter what globs target.
    
```d2
bravo team.shape: person
charlie team.shape: person
command center.shape: cloud
hq.shape: rectangle

*: {
  !&shape: person
  style.multiple: true
}
```

## Nested globs
You can nest globs, combining the features above.
    
```d2
conversation 1: {
  shape: sequence_diagram
  alice -> bob: hi
  bob -> alice: hi
}

conversation 2: {
  shape: sequence_diagram
  alice -> bob: hello again
  alice -> bob: hello?
  bob -> alice: hello
}

# Recursively target all shapes...
**: {
  # ... that are sequence diagrams
  &shape: sequence_diagram
  # Then recursively set all shapes in them to person
  **: {shape: person}
}
```

## Global globs
Triple globs apply globally to the whole diagram. The difference between a double glob and
a triple glob is that a triple glob will apply to nested `layers` (see the section on
[composition](/tour/composition/) for more on `layers`), as well as persist across imports.
```d2
***.style.fill: yellow
**.shape: circle
*.style.multiple: true
x: {
  y
}
layers: {
  next: {
    a
  }
}
```
<embed src={require('@site/static/img/generated/triple-glob.pdf').default} width="100%" height="800"
 type="application/pdf" />
 Imports
If you import a file, the globs declared inside it are usually not carried over. Triple
globs are the exception -- since they are global, importing a file with triple glob will
carry that glob as well.
## Changing defaults
One common use case of globs is to change the default styling of a theme.
    
```d2
# Add to the top of your diagram
***.style.fill: lightblue
(*** -> ***)[*]: {
  style.stroke: red
}

x -> y
```

## SQL-TABLES

# SQL Tables
## Basics
You can easily diagram entity-relationship diagrams (ERDs) in D2 by using the `sql_table` shape. Here's a minimal example:
    
```d2
my_table: {
  shape: sql_table
  # This is defined using the shorthand syntax for labels discussed in the containers section.
  # But here it's for the type of a constraint.
  # The id field becomes a map that looks like {type: int; constraint: primary_key}
  id: int {constraint: primary_key}
  last_updated: timestamp with time zone
}
```

Each key of a SQL Table shape defines a row. The primary value (the thing after the colon)
of each row defines its type.
The constraint value of each row defines its SQL constraint. D2 will recognize and
shorten:
| constraint  | short |
| ----------- | ----- |
| primary_key | PK    |
| foreign_key | FK    |
| unique      | UNQ   |
But you can set any constraint you'd like. It just won't be shortened if unrecognized.
You can also specify multiple constraints with an array.
```d2
x: int { constraint: [primary_key; unique] }
```
 Escaping reserved keywords
If you'd like to use a reserved keyword, wrap it in quotes.
```d2
my_table: {
  shape: sql_table
  "label": string
}
```
## Foreign Keys
Here's an example of how you'd define a foreign key connection between two tables:
    
```d2
objects: {
  shape: sql_table
  id: int {constraint: primary_key}
  disk: int {constraint: foreign_key}

  json: jsonb {constraint: unique}
  last_updated: timestamp with time zone
}

disks: {
  shape: sql_table
  id: int {constraint: primary_key}
}

objects.disk -> disks.id
```

When rendered with the [TALA layout engine](/tour/tala/) or [ELK layout engine](/tour/elk/),
connections point to the exact row.
## Example
Like all other shapes, you can nest `sql_tables` into containers and define edges
to them from other shapes. Here's an example:
    
```d2
cloud: {
  disks: {
    shape: sql_table
    id: int {constraint: primary_key}
  }
  blocks: {
    shape: sql_table
    id: int {constraint: primary_key}
    disk: int {constraint: foreign_key}
    blob: blob
  }
  blocks.disk -> disks.id

  AWS S3 Vancouver -> disks
}
```

## UML-CLASSES

# UML Classes
## Basics
D2 fully supports UML Class diagrams. Here's a minimal example:
    
```d2
MyClass: {
  shape: class

  field: "[]string"
  method(a uint64): (x, y int)
}
```

Each key of a `class` shape defines either a field or a method.
The value of a field key is its type.
Any key that contains `(` is a method, whose value is the return type.
A method key without a value has a return type of void.
 Escaping reserved keywords
If you'd like to use a reserved keyword, wrap it in quotes.
```d2
my_class: {
  shape: class
  "label": string
}
```
## Visibilities
You can also use UML-style prefixes to indicate field/method visibility.
| visibility prefix | meaning   |
| ----------------- | --------- |
| none              | public    |
| +                 | public    |
| -                 | private   |
| #                 | protected |
See https://www.uml-diagrams.org/visibility.html
Here's an example with differing visibilities and more complex types:
    
```d2
D2 Parser: {
  shape: class

  # Default visibility is + so no need to specify.
  +reader: io.RuneReader
  readerPos: d2ast.Position

  # Private field.
  -lookahead: "[]rune"

  # Protected field.
  # We have to escape the # to prevent the line from being parsed as a comment.
  \#lookaheadPos: d2ast.Position

  +peek(): (r rune, eof bool)
  rewind()
  commit()

  \#peekn(n int): (s string, eof bool)
}

"github.com/terrastruct/d2parser.git" -> D2 Parser
```

## Full example
    
```d2
DebitCard: Debit card {
  shape: class
  +cardno
  +ownedBy

  +access()
}

Bank: {
  shape: class
  +code
  +address

  +manages()
  +maintains()
}

ATMInfo: ATM info {
  shape: class
  +location
  +manageBy

  +identifies()
  +transactions()
}

Customer: {
  shape: class
  +name
  +address
  +dob

  +owns()
}

Account: {
  shape: class
  +type
  +owner
}

ATMTransaction: ATM Transaction {
  shape: class
  +transactionId
  +date
  +type

  +modifies()
}

CurrentAccount: Current account {
  shape: class
  +accountNo
  +balance

  +debit()
  +credit()
}

SavingAccount: Saving account {
  shape: class
  +accountNo
  +balance

  +debit()
  +credit()
}

WidthdrawlTransaction: Withdrawl transaction {
  shape: class
  +amount

  +Withdrawl()
}

QueryTransaction: Query transaction {
  shape: class
  +query
  +type

  +queryProcessing()
}

TransferTransaction: Transfer transaction {
  shape: class
  +account
  +accountNo
}

PinValidation: Pin validation transaction {
  shape: class
  +oldPin
  +newPin

  +pinChange()
}

DebitCard -- Bank: manages {
  source-arrowhead: 1..*
  target-arrowhead: 1
}

Bank -- ATMInfo: maintains {
  source-arrowhead: 1
  target-arrowhead: 1
}

Bank -- Customer: +has {
  source-arrowhead: 1
  target-arrowhead: 1
}

DebitCard -- Customer: +owns {
  source-arrowhead: 0..*
  target-arrowhead: 1..*
}

DebitCard -- Account: +provides access to {
  source-arrowhead: *
  target-arrowhead: 1..*
}

Customer -- Account: owns {
  source-arrowhead: 1..*
  target-arrowhead: 1..*
}

ATMInfo -- ATMTransaction: +identifies {
  source-arrowhead: 1
  target-arrowhead: *
}

ATMTransaction -> Account: modifies {
  source-arrowhead: *
  target-arrowhead: 1
}

CurrentAccount -> Account: {
  target-arrowhead.shape: triangle
  target-arrowhead.style.filled: false
}

SavingAccount -> Account: {
  target-arrowhead.shape: triangle
  target-arrowhead.style.filled: false
}

WidthdrawlTransaction -> ATMTransaction: {
  target-arrowhead.shape: triangle
  target-arrowhead.style.filled: false
}
QueryTransaction -> ATMTransaction: {
  target-arrowhead.shape: triangle
  target-arrowhead.style.filled: false
}
TransferTransaction -> ATMTransaction: {
  target-arrowhead.shape: triangle
  target-arrowhead.style.filled: false
}
PinValidation -> ATMTransaction: {
  target-arrowhead.shape: triangle
  target-arrowhead.style.filled: false
}
```

## SEQUENCE-DIAGRAMS

# Sequence Diagrams
Sequence diagrams are created by setting `shape: sequence_diagram` on an object.
    
```d2
shape: sequence_diagram
alice -> bob: What does it mean\nto be well-adjusted?
bob -> alice: The ability to play bridge or\ngolf as if they were games.
```

## Rules
Unlike other tools, there is no special syntax to learn for sequence diagrams. The rules
are also almost exactly the same as everywhere else in D2, with two notable differences.
### Scoping
Children of sequence diagrams share the same scope throughout the sequence diagram.
For example:
    
```d2
Office chatter: {
  shape: sequence_diagram
  alice: Alice
  bob: Bobby
  awkward small talk: {
    alice -> bob: uhm, hi
    bob -> alice: oh, hello
    icebreaker attempt: {
      alice -> bob: what did you have for lunch?
    }
    unfortunate outcome: {
      bob -> alice: that's personal
    }
  }
}
```

Outside of a sequence diagram, there would be multiple instances of `alice` and `bob`,
since they have different container scopes. But when nested under `shape:
sequence_diagram`, they refer to the same `alice` and `bob`.
### Ordering
Elsewhere in D2, there is no notion of order. If you define a connection after another,
there is no guarantee is will visually appear after. However, in sequence diagrams, order
matters. The order in which you define everything is the order they will appear.
This includes actors. You don't have to explicitly define actors (except when they first
appear in a group), but if you want to define a specific order, you should.
```d2
shape: sequence_diagram
# Remember that semicolons allow multiple objects to be defined in one line
# Actors will appear from left-to-right as a, b, c, d...
a; b; c; d
# ... even if the connections are in a different order
c -> d
d -> a
b -> d
```
An actor in D2 is also known elsewhere as "participant".
## Features
### Sequence diagrams are D2 objects
Like every other object in D2, they can be contained, connected, relabeled, re-styled, and
treated like any other object.
    
```d2
direction: right
Before and after becoming friends: {
  2007: Office chatter in 2007 {
    shape: sequence_diagram
    alice: Alice
    bob: Bobby
    awkward small talk: {
      alice -> bob: uhm, hi
      bob -> alice: oh, hello
      icebreaker attempt: {
        alice -> bob: what did you have for lunch?
      }
      unfortunate outcome: {
        bob -> alice: that's personal
      }
    }
  }

  2012: Office chatter in 2012 {
    shape: sequence_diagram
    alice: Alice
    bob: Bobby
    alice -> bob: Want to play with ChatGPT?
    bob -> alice: Yes!
    bob -> alice.play: Write a play...
    alice.play -> bob.play: about 2 friends...
    bob.play -> alice.play: who find love...
    alice.play -> bob.play: in a sequence diagram
  }

  2007 -> 2012: Five\nyears\nlater
}
```

### Spans
Spans convey a beginning and end to an interaction within a sequence diagram.
A span in D2 is also known elsewhere as a "lifespan", "activation box", and "activation bar".
You can specify a span by connecting a nested object on an actor.
    
```d2
shape: sequence_diagram
alice.t1 -> bob
alice.t2 -> bob.a
alice.t2.a -> bob.a
alice.t2.a <- bob.a
alice.t2 <- bob.a
```

### Groups
Groups help you label a subset of the sequence diagram.
A group in D2 is also known elsewhere as a "fragment", "edge group", and "frame".
We saw an example of this in an earlier example when explaining scoping rules. More
formally, a group is a container within a `sequence_diagram` shape which is not connected
to anything but has connections or objects inside.
    
```d2
shape: sequence_diagram
# Predefine actors
alice
bob
shower thoughts: {
  alice -> bob: A physicist is an atom's way of knowing about atoms.
  alice -> bob: Today is the first day of the rest of your life.
}
life advice: {
  bob -> alice: If all else fails, lower your standards.
}
```

Due to the unique scoping rules in sequence diagrams, when you are within a group, the
objects you reference in connections must exist at the top-level. Notice in the above
example that `alice` and `bob` are explicitly declared before group declarations.
### Notes
Notes are declared by defining a nested object on an actor with no connections going to
it.
    
```d2
shape: sequence_diagram
alice -> bob
bob."In the eyes of my dog, I'm a man."
# Notes can go into groups, too
important insight: {
  bob."Cold hands, no gloves."
}
bob -> alice: Chocolate chip.
```

### Self-messages
Self-referential messages can be declared from an actor to the themselves.
    
```d2
shape: sequence_diagram
son -> father: Can I borrow your car?
friend -> father: Never lend your car to anyone to whom you have given birth.
father -> father: internal debate ensues
```

### Customization
You can style shapes and connections like any other. Here we make some messages dashed and
set the shape on an actor.
    
```d2
shape: sequence_diagram
scorer: {shape: person}
scorer.t -> itemResponse.t: getItem()
scorer.t <- itemResponse.t: item {
  style.stroke-dash: 5
}

scorer.t -> item.t1: getRubric()
scorer.t <- item.t1: rubric {
  style.stroke-dash: 5
}

scorer.t -> essayRubric.t: applyTo(essayResp)
itemResponse -> essayRubric.t.c
essayRubric.t.c -> concept.t: match(essayResponse)
scorer <- essayRubric.t: score {
  style.stroke-dash: 5
}

scorer.t -> itemOutcome.t1: new
scorer.t -> item.t2: getNormalMinimum()
scorer.t -> item.t3: getNormalMaximum()

scorer.t -> itemOutcome.t2: setScore(score)
scorer.t -> itemOutcome.t3: setFeedback(missingConcepts)
```

Lifeline edges (those lines going from top-down) inherit the actor's `stroke` and
`stroke-dash` styles.
    
```d2
shape: sequence_diagram
alice -> bob: What does it mean\nto be well-adjusted?
bob -> alice: The ability to play bridge or\ngolf as if they were games.

alice.style: {
  stroke: red
  stroke-dash: 0
}
```

## Glossary
<WebPImage src={require('@site/static/img/screenshots/sequence_glossary.png').default} webpSrc={require('@site/static/img/screenshots/sequence_glossary.webp').default}
alt="sequence diagram glossary"/>

## GRID-DIAGRAMS

# Grid Diagrams
Grid diagrams let you display objects in a structured grid.
    
```d2
grid-rows: 5
style.fill: black

classes: {
  white square: {
    label: ""
    width: 120
    style: {
      fill: white
      stroke: cornflowerblue
      stroke-width: 10
    }
  }
  block: {
    style: {
      text-transform: uppercase
      font-color: white
      fill: darkcyan
      stroke: black
    }
  }
}

flow1.class: white square
flow2.class: white square
flow3.class: white square
flow4.class: white square
flow5.class: white square
flow6.class: white square
flow7.class: white square
flow8.class: white square
flow9.class: white square

dagger engine: {
  width: 800
  class: block
  style: {
    fill: beige
    stroke: darkcyan
    font-color: blue
    stroke-width: 8
  }
}

any docker compatible runtime: {
  width: 800
  class: block
  style: {
    fill: lightcyan
    stroke: darkcyan
    font-color: black
    stroke-width: 8
  }
  icon: https://icons.terrastruct.com/dev%2Fdocker.svg
}

any ci: {
  class: block
  style: {
    fill: gold
    stroke: maroon
    font-color: maroon
    stroke-width: 8
  }
}
windows.class: block
linux.class: block
macos.class: block
kubernetes.class: block
```

Two keywords do all the magic:
- `grid-rows`
- `grid-columns`
Setting just `grid-rows`:
    
```d2
grid-rows: 3
Executive
Legislative
Judicial
```

Setting just `grid-columns`:
    
```d2
grid-columns: 3
Executive
Legislative
Judicial
```

Setting both `grid-rows` and `grid-columns`:
    
```d2
grid-rows: 2
grid-columns: 2
Executive
Legislative
Judicial
```

## Width and height
To create specific constructions, use `width` and/or `height`.
    
```d2
grid-rows: 2
Executive
Legislative
Judicial
The American Government.width: 400
```

Notice how objects are evenly distributed within each row.
## Cells expand to fill
When you define only one of row or column, objects will expand.
    
```d2
grid-rows: 3
Executive
Legislative
Judicial
The American Government.width: 400
Voters
Non-voters
```

Notice how `Voters` and `Non-voters` fill the space.
## Dominant direction
When you apply both row and column, the first appearance is the dominant direction. The
dominant direction is the order in which cells are filled.
For example:
```d2-incomplete
grid-rows: 4
grid-columns: 2
# bunch of shapes
```
Since `grid-rows` is defined first, objects will fill rows before moving onto columns.
But if it were reversed:
```d2-incomplete
grid-columns: 2
grid-rows: 4
# bunch of shapes
```
It would do the opposite.
These animations are also pure D2, so you can animate grid diagrams being built-up. Use
the `animate-interval` flag with this
[code](https://github.com/terrastruct/d2-docs/blob/f5c762223ce192338d9d7865df3ca8533d683cdc/static/bespoke-d2/grid-row-dominant.d2#L1).
More on this later, in the [composition](/tour/composition/) section.
## Gap size
You can control the gap size of the grid with 3 keywords:
- `vertical-gap`
- `horizontal-gap`
- `grid-gap`
Setting `grid-gap` is equivalent to setting both `vertical-gap` and `horizontal-gap`.
`vertical-gap` and `horizontal-gap` can override `grid-gap`.
### Gap size 0
`grid-gap: 0` in particular can create some interesting constructions:
#### Like this map of Japan
> [D2 source](https://github.com/terrastruct/d2/blob/master/docs/examples/japan-grid/japan.d2)
#### Or a table of data
    
```d2
# Specified so that objects are written in row-dominant order
grid-rows: 2
grid-columns: 4
grid-gap: 0

classes: {
  header: {
    style.underline: true
  }
}

Element.class: header
Atomic Number.class: header
Atomic Mass.class: header
Melting Point.class: header

Hydrogen
1
"1.008"
"-259.16"

Carbon
6
"12.011"
3500

Oxygen
8
"15.999"
"-218.79"
```

You may find it easier to just use Markdown tables though, especially if there are
duplicate cells.
    
```d2
savings: ||md
  | Month    | Savings | Expenses | Balance |
  | -------- | ------- | -------- | ------- |
  | January  | $250    | $150     | $100    |
  | February | $80     | $200     | -$120   |
  | March    | $420    | $180     | $240    |
||
```

### Gap size 0
## Connections
Connections for grids themselves work normally as you'd expect.
> Source code [here](https://github.com/terrastruct/d2-docs/blob/eda2d8739ce21c656e7608be48cb9067df36eb53/static/d2/grid-connected.d2).
### Connections between grid cells
Connections between shapes inside a grid work a bit differently. Because a grid structure
imposes positioning outside what the layout engine controls, the layout engine is also
unable to make routes. Therefore, these connections are center-center straight segments,
i.e., no path-finding.
> Source code [here](https://github.com/terrastruct/d2/blob/master/e2etests/testdata/files/simple_grid_edges.d2).
> Source code [here](https://github.com/terrastruct/d2/blob/master/docs/examples/vector-grid/vector-grid.d2).
## Nesting
Currently you can nest grid diagrams within grid diagrams. Nesting other types is coming
soon.
    
```d2
grid-gap: 0
grid-columns: 1
header
body: "" {
  grid-gap: 0
  grid-columns: 2
  content
  sidebar
}
footer
```

## Aligning with invisible elements
A common technique to align grid elements to your liking is to pad the grid with invisible
elements.
Consider the following diagram.
    
```d2
grid-columns: 1
us-east-1: {
  grid-rows: 1
  a
  b
  c
  d
  e
}

us-west-1: {
  grid-rows: 1
  a
}

us-east-1.c -> us-west-1.a
```

It'd be nicer if it were centered. This can be achieved by adding 2 invisible elements.
    
```d2
classes: {
  invisible: {
    style.opacity: 0
    label: a
  }
}

grid-columns: 1
us-east-1: {
  grid-rows: 1
  a
  b
  c
  d
  e
}

us-west-1: {
  grid-rows: 1
  pad1.class: invisible
  pad2.class: invisible
  a
  # Move the label so it doesn't go through the connection
  label.near: bottom-center
}

us-east-1.c -> us-west-1.a
```

## Troubleshooting
### Why is there extra padding in one cell?
Elements in a grid column have the same width and elements in a grid row have the same
height.
So in this example, a small empty space in "Backend Node" is present.
    
```d2
classes: {
  kuber: {
    style: {
      fill: "white"
      stroke: "#aeb5bd"
      border-radius: 4
      stroke-dash: 3
    }
  }
  sys: {
    label: ""
    style: {
      fill: "#AFBFDF"
      stroke: "#aeb5bd"
    }
  }
  node: {
    grid-gap: 0
    style: {
      fill: "#ebf3e6"
      border-radius: 8
      stroke: "#aeb5bd"
    }
  }
  clust: {
    style: {
      fill: "#A7CC9E"
      stroke: "#aeb5bd"
    }
  }
  deploy: {
    grid-gap: 0
    style: {
      fill: "#ffe6d5"
      stroke: "#aeb5bd"
      # border-radius: 4
    }
  }
  nextpod: {
    icon: https://www.svgrepo.com/show/378440/nextjs-fill.svg
    style: {
      fill: "#ECECEC"
      stroke: "#aeb5bd"
      # border-radius: 4
    }
  }
  flaskpod: {
    icon: https://www.svgrepo.com/show/508915/flask.svg
    style: {
      fill: "#ECECEC"
      stroke: "#aeb5bd"
      # border-radius: 4
    }
  }
}

classes

Kubernetes: {
  grid-columns: 2
  system: {
    grid-columns: 1
    Backend Node: {
      grid-columns: 2
      ClusterIP\nService 1
      Deployment 1: {
        grid-rows: 3
        NEXT POD 1
        NEXT POD 2
        NEXT POD 3
      }
    }
    Frontend Node: {
      grid-columns: 2
      ClusterIP\nService 2
      Deployment 2: {
        grid-rows: 3
        FLASK POD 1
        FLASK POD 2
        FLASK POD 3
      }
    }
  }
}

kubernetes.class: kuber
kubernetes.system.class: sys

kubernetes.system.backend node.class: node
kubernetes.system.backend node.clusterip\nservice 1.class: clust
kubernetes.system.backend node.deployment 1.class: deploy
kubernetes.system.backend node.deployment 1.next pod*.class: nextpod

kubernetes.system.frontend node.class: node
kubernetes.system.frontend node.clusterip\nservice 2.class: clust
kubernetes.system.frontend node.deployment 2.class: deploy
kubernetes.system.frontend node.deployment 2.flask pod*.class: flaskpod
```

It's due to the label of "Flask Pod" being slightly longer than "Next Pod". So the way we
fix that is to set `width`s to match.
    
```d2
classes: {
  kuber: {
    style: {
      fill: "white"
      stroke: "#aeb5bd"
      border-radius: 4
      stroke-dash: 3
    }
  }
  sys: {
    label: ""
    style: {
      fill: "#AFBFDF"
      stroke: "#aeb5bd"
    }
  }
  node: {
    grid-gap: 0
    style: {
      fill: "#ebf3e6"
      border-radius: 8
      stroke: "#aeb5bd"
    }
  }
  clust: {
    style: {
      fill: "#A7CC9E"
      stroke: "#aeb5bd"
    }
  }
  deploy: {
    grid-gap: 0
    style: {
      fill: "#ffe6d5"
      stroke: "#aeb5bd"
      # border-radius: 4
    }
  }
  nextpod: {
    width: 180
    icon: https://www.svgrepo.com/show/378440/nextjs-fill.svg
    style: {
      fill: "#ECECEC"
      stroke: "#aeb5bd"
      # border-radius: 4
    }
  }
  flaskpod: {
    width: 180
    icon: https://www.svgrepo.com/show/508915/flask.svg
    style: {
      fill: "#ECECEC"
      stroke: "#aeb5bd"
      # border-radius: 4
    }
  }
}

classes

Kubernetes: {
  grid-columns: 2
  system: {
    grid-columns: 1
    Backend Node: {
      grid-columns: 2
      ClusterIP\nService 1
      Deployment 1: {
        grid-rows: 3
        NEXT POD 1
        NEXT POD 2
        NEXT POD 3
      }
    }
    Frontend Node: {
      grid-columns: 2
      ClusterIP\nService 2
      Deployment 2: {
        grid-rows: 3
        FLASK POD 1
        FLASK POD 2
        FLASK POD 3
      }
    }
  }
}

kubernetes.class: kuber
kubernetes.system.class: sys

kubernetes.system.backend node.class: node
kubernetes.system.backend node.clusterip\nservice 1.class: clust
kubernetes.system.backend node.deployment 1.class: deploy
kubernetes.system.backend node.deployment 1.next pod*.class: nextpod

kubernetes.system.frontend node.class: node
kubernetes.system.frontend node.clusterip\nservice 2.class: clust
kubernetes.system.frontend node.deployment 2.class: deploy
kubernetes.system.frontend node.deployment 2.flask pod*.class: flaskpod
```

## TEXT

# Text
## Standalone text is Markdown
    
```d2
explanation: |md
  # I can do headers
  - lists
  - lists

  And other normal markdown stuff
|
```

## Markdown label
If you want to set a Markdown label on a shape, you must explicitly declare the shape.
    
```d2
explanation: |md
  # I can do headers
  - lists
  - lists

  And other normal markdown stuff
|
# Explicitly declare, even though the default shape is rectangle
explanation.shape: rectangle
```

## Most languages are supported
D2 most likely supports any language you want to use, including non-Latin ones like
Chinese, Japanese, Korean, even emojis!
## LaTeX
You can use `latex` or `tex` to specify a LaTeX language block.
    
```d2
plankton -> formula: will steal
formula: |latex
  \lim_{h \rightarrow 0 } \frac{f(x+h)-f(x)}{h}
|
```

A few things to note about LaTeX blocks:
- LaTeX blocks do not respect `font-size` styling. Instead, you must style these inside
  the Latex script itself with commands:
  - `\tiny{ }`
  - `\small{ }`
  - `\normal{ }`
  - `\large{ }`
  - `\huge{ }`
- Under the hood, this is using [MathJax](https://www.mathjax.org/). It is not full LaTeX
  (full LaTeX includes a document layout engine). D2's LaTeX blocks are meant to display
  mathematical notation, but not support the format of existing LaTeX documents. See
  [here](https://docs.mathjax.org/en/latest/input/tex/macros/index.html) for a list of all
  supported commands.
D2 runs on the latest version of MathJax, which has a lot of nice things but unfortunately
does not have linebreaks. You can kind of get around this with the `displaylines` command.
See below.
## Code
Change `md` to a programming language for code blocks
    
```d2
explanation: |go
  awsSession := From(c.Request.Context())
  client := s3.New(awsSession)

  ctx, cancelFn := context.WithTimeout(c.Request.Context(), AWS_TIMEOUT)
  defer cancelFn()
|
```

 Supported syntax highlighting languages
See the [Chroma library](https://github.com/alecthomas/chroma?tab=readme-ov-file#supported-languages) for a full list of supported languages.
D2 also provides convenient short aliases:
- `md` → `markdown`
- `tex` → `latex`
- `js` → `javascript`
- `go` → `golang`
- `py` → `python`
- `rb` → `ruby`
- `ts` → `typescript`
If a language isn't recognized, D2 will fall back to plain text rendering without syntax highlighting.
## Advanced: Non-Markdown text
In some cases, you may want non-Markdown text. Maybe you just don't like Markdown, or the
GitHub-styling of Markdown that D2 uses, or you want to quickly change a shape to text.
Just set `shape: text`.
    
```d2
title: A winning strategy {
  shape: text
  near: top-center
  style: {
    font-size: 55
    italic: true
  }
}

poll the people -> results
results -> unfavorable -> poll the people
results -> favorable -> will of the people
```

## Advanced: Block strings
What if you're writing Typescript where the pipe symbol `|` is commonly used? Just add
another pipe, `||`.
```d2
my_code: ||ts
  declare function getSmallPet(): Fish | Bird;
||
```
Actually, Typescript uses `||` too, so that won't work. Let's keep going.
```d2
my_code: |||ts
  declare function getSmallPet(): Fish | Bird;
  const works = (a > 1) || (b < 2)
|||
```
There's probably some language or scenario where the triple pipe is used too. D2 actually
allows you to use any special symbols (not alphanumeric, space, or `_`) after the first pipe:
```d2
# Much cleaner!
my_code: |`ts
  declare function getSmallPet(): Fish | Bird;
  const works = (a > 1) || (b < 2)
`|
```
## Advanced: LaTeX plugins
D2 includes the following LaTeX plugins:
    
```d2
grid-columns: 3
grid-gap: 100

*.style.fill: transparent
*.style.stroke: black

amscd plugin: {
  ex: |tex
    \begin{CD} B @>{\text{very long label}}>> C S^{{\mathcal{W}}_\Lambda}\otimes T @>j>> T\\ @VVV V \end{CD}
  |
}

braket plugin: {
  ex: |tex
    \bra{a}\ket{b}
  |
}

cancel plugin: {
  ex: |tex
    \cancel{Culture + 5}
  |
}

color plugin: {
  ex: |tex
    \textcolor{red}{y} = \textcolor{green}{\sin} x
  |
}

gensymb plugin: {
  ex: |tex
    \lambda = 10.6\,\micro\mathrm{m}
  |
}

mhchem plugin: {
  ex: |tex
    \ce{SO4^2- + Ba^2+ -> BaSO4 v}
  |
}

physics plugin: {
  ex: |tex
    \var{F[g(x)]}
    \dd(\cos\theta)
  |
}

multilines: {
  ex: |tex
    \displaylines{x = a + b \\ y = b + c}
    \sum_{k=1}^{n} h_{k} \int_{0}^{1} \bigl(\partial_{k} f(x_{k-1}+t h_{k} e_{k}) -\partial_{k} f(a)\bigr) \,dt
  |
}

asm: {
  ex: |latex
    \min_{ \mathclap{\substack{ x \in \mathbb{R}^n \ x \geq 0 \ Ax \leq b }}} c^T x
  |
}
```

## ICONS

# Icons
We host a collection of icons commonly found in software architecture diagrams for free to
help you get started: [https://icons.terrastruct.com](https://icons.terrastruct.com).
Icons and images are an essential part of production-ready diagrams.
You can use any URL as value.
    
```d2
deploy: {
  icon: https://icons.terrastruct.com/aws%2FDeveloper%20Tools%2FAWS-CodeDeploy.svg
}

backup: {
  icon: https://icons.terrastruct.com/aws%2FStorage%2FAWS-Backup.svg
}

deploy -> backup: {
  icon: https://icons.terrastruct.com/infra%2F002-backup.svg
}
```

Using the D2 CLI locally? You can specify local images like `icon: ./my_cat.png`.
Icon placement is automatic. Considerations vary depending on layout engine, but things
like coexistence with a label and whether it's a container generally affect where the icon
is placed to not obstruct. Notice how the following diagram has container icons in the
top-left and non-container icons in the center.
    
```d2
vpc: VPC 1 10.1.0.0./16 {
  icon: https://icons.terrastruct.com/aws%2F_Group%20Icons%2FVirtual-private-cloud-VPC_light-bg.svg
  style: {
    stroke: green
    font-color: green
    fill: white
  }
  az: Availability Zone A {
    style: {
      stroke: blue
      font-color: blue
      stroke-dash: 3
      fill: white
    }
    firewall: Firewall Subnet A {
      icon: https://icons.terrastruct.com/aws%2FNetworking%20&%20Content%20Delivery%2FAmazon-Route-53_Hosted-Zone_light-bg.svg
      style: {
        stroke: purple
        font-color: purple
        fill: "#e1d5e7"
      }
      ec2: EC2 Instance {
        icon: https://icons.terrastruct.com/aws%2FCompute%2F_Instance%2FAmazon-EC2_C4-Instance_light-bg.svg
      }
    }
  }
}
```

Icons can be positioned with the `near` keyword [introduced later](/tour/positions/#label-and-icon-positioning).
## Add `shape: image` for standalone icon shapes
    
```d2
direction: right
server: {
  shape: image
  icon: https://icons.terrastruct.com/tech/022-server.svg
}

github: {
  shape: image
  icon: https://icons.terrastruct.com/dev/github.svg
}

server -> github
```

## LINKING

# Linking between boards
We've introduced `link` before as a way to jump to external resources. They can also be
used to create interactivity to jump to other boards. We'll call these "internal links".
Example of internal link:
    
```d2
how does the cat go?: {
  link: layers.cat
}

layers: {
  cat: {
    meoowww
  }
}
```

<embed src={require('@site/static/img/generated/cat.pdf').default} width="100%" height="800"
 type="application/pdf" />
If your board name has a `.`, use quotes to target that board.
For example:
```d2-incomplete
a.link: layers."2012.06"
layers: {
  "2012.06": {
    hello
  }
}
```
## Parent reference
The underscore `_` is used to refer to the parent scope, but when used in `link` values,
they refer not to parent containers, but to parent boards.
    
```d2
The shire

journey: {
  link: layers.rivendell
}

layers: {
  rivendell: {
    elves: {
      elrond -> frodo: gives advice
    }

    take me home sam.link: _
    go deeper: {
      link: layers.moria
    }
    
    layers: {
      moria: {
        dwarves

        take me home sam.link: _._
      }
    }
  }
}
```

<embed src={require('@site/static/img/generated/lotr.pdf').default} width="100%" height="800"
 type="application/pdf" />
## Backlinks
Notice how the navigation bar at the top is clickable. You can easily return to the root
or any ancestor page by clicking on the text.

## POSITIONS

# Positions
In general, positioning is controlled entirely by the layout engine. It's one of the
primary benefits of text-to-diagram that you don't have to manually define all the
positions of objects.
However, there are occasions where you want to have some control over positions.
Currently, there are two ways to do that.
## Near
D2 allows you to position items on set points around your diagram.
 Possible values
`top-left`, `top-center`, `top-right`,
`center-left`, `center-right`,
`bottom-left`, `bottom-center`, `bottom-right`
Let's explore some use cases:
### Giving your diagram a title
    
```d2
title: |md
  # A winning strategy
| {near: top-center}

poll the people -> results
results -> unfavorable -> poll the people
results -> favorable -> will of the people
```

### Creating a legend
    
```d2
direction: right

x -> y: {
  style.stroke: green
}

y -> z: {
  style.stroke: red
}

legend: {
  near: bottom-center
  color1: foo {
    shape: text
    style.font-color: green
  }

  color2: bar {
    shape: text
    style.font-color: red
  }
}
```

### Longform description or explanation
    
```d2
explanation: |md
  # LLMs
  The Large Language Model (LLM) is a powerful AI\
    system that learns from vast amounts of text data.\
  By analyzing patterns and structures in language,\
  it gains an understanding of grammar, facts,\
  and even some reasoning abilities. As users input text,\
  the LLM predicts the most likely next words or phrases\
  to create coherent responses. The model\
  continuously fine-tunes its output, considering both the\
  user's input and its own vast knowledge base.\
  This cutting-edge technology enables LLM to generate human-like text,\
  making it a valuable tool for various applications.
| {
  near: center-left
}

ML Platform -> Pre-trained models
ML Platform -> Model registry
ML Platform -> Compiler
ML Platform -> Validation
ML Platform -> Auditing

Model registry -> Server.Batch Predictor
Server.Online Model Server
```

## Label and icon positioning
The `near` can be nested to `label` and `icon` to specify their positions.
    
```d2
direction: right
x -> y

x: worker {
  label.near: top-center
  icon: https://icons.terrastruct.com/essentials%2F005-programmer.svg
  icon.near: outside-top-right
}

y: profits {
  label.near: bottom-right
  icon: https://icons.terrastruct.com/essentials%2Fprofits.svg
  icon.near: outside-top-left
}
```

### Outside and border
When positioning labels and icons, in addition to the values that `near` can take
elsewhere, an `outside-` prefix can be added to specify positioning outside the bounding
box of the shape.
`outside-top-left`, `outside-top-center`, `outside-top-right`,
`outside-left-center`, `outside-right-center`,
`outside-bottom-left`, `outside-bottom-center`, `outside-bottom-right`
Note that `outside-left-center` is a different order than `center-left`.
You can also add `border-x` prefix to specify the label being on the border.
    
```d2
style.fill: "#222a25"
env: Your environment {
  # --- Outside label ---
  label.near: outside-bottom-center
  style.fill: "#222a25"
  style.stroke-dash: 2
  style.double-border: false
  style.stroke: "#1e402d"
  style.font-color: "#3ddb89"
  app: Your applicaton {
    # --- Border label ---
    label.near: border-top-center
    style.stroke: "#3d9069"
    style.fill: "#222a25"
    style.font-color: "#63c08c"

    *.style.stroke: "#adf1c6"
    *.style.fill: "#306a4a"
    *.style.font-color: "#eef9f3"
    Workflow
    SDK
    Workers
    Workflow -> SDK: hello {
      style.stroke: "#fbfdfd"
      style.font-color: "#adf1c6"
    }
  }
}
```

## Tooltip near
Usually, `tooltip` is a on-hover effect. However, if you specify a `near` field, it will
permanently show.
    
```d2
ci-runner
codedeploy: {
  tooltip: |md
    God has abandoned this pipeline
  |
  tooltip.near: center-left
}
aws instance

ci-runner -> codedeploy -> aws instance: safe deploy
ci-runner -> aws instance: direct deploy
```

## Near objects
Works in TALA only. We are working on shims to make this possible in other layout engines.
You can also set `near` to the absolute ID of another shape to hint to the layout engine
that they should be in the vicinity of one another.
```d2
vars: {
  d2-config: {
    layout-engine: tala
  }
}
aws: {
  load_balancer -> api
  api -> db
}
gcloud: {
  auth -> db
}
gcloud -> aws
explanation: |md
  # Why do we use AWS?
  - It has more uptime than GCloud
  - We have free credits
| {
  near: aws
}
```
Notice how the text is positioned near the `aws` node and not the `gcloud` node.
<WebPImage src={require('@site/static/img/screenshots/text-2.png').default} webpSrc={require('@site/static/img/screenshots/text-2.webp').default} alt="text near example" width={800}/>
## Top and left
On the TALA engine, you can also directly set the `top` and `left` values for objects, and
the layout engine will only move other objects around it.
For more on this, see page 17 of the [TALA user
manual](https://github.com/terrastruct/TALA/blob/master/TALA_User_Manual.pdf).

## COMPOSITION

# Intro to Composition
This section's documentation is incomplete. We'll be adding more to this section soon.
D2 has built-in mechanisms for you to compose multiple boards into one diagram.
For example, this is a composition of 2 boards, exported as an animated SVG:
The way to define another board in a D2 diagram is to use 1 of 3 keywords. Each of these
declare boards with different inheritance rules.
| Keyword   | Inheritance                                       |
|-----------|---------------------------------------------------|
| `layers`    | Boards which do not inherit. They are a new base. |
| `scenarios` | Boards which inherit from the base layer.         |
| `steps`     | Boards which inherit from the previous step.      |
Each one serves different use cases. The example above is achieved by defining a Scenario
(the scenario of when we have to deploy a hotfix).
Thus far, all D2 diagrams we've encountered are single-board diagrams, the root board.
```d2-incomplete
# Root board
x -> y
```
Composition in D2 is when you use one of those keywords to declare another board.
```d2-incomplete
# Root board
x -> y
layers: {
  # Board named "numbers" that does not inherit anything from root
  numbers: {
    1 -> 2
  }
}
```
So now we have two boards: root and `numbers`. They cannot be visible at the same time of
course, so exports have to accommodate these more dynamic diagrams, such as the animated
SVG you see above.
Composition is one of D2's most powerful features, as you'll see from the use cases in this
section.

## LAYERS

# Layers
A "Layer" represents "a layer of abstraction". Each Layer starts off as a blank
board, since you're representing different objects at every level of abstraction.
Try clicking on the objects.
<embed src={require('@site/static/img/generated/tiktok.pdf').default} width="100%" height="800"
 type="application/pdf" />
    
```d2
explain: |md
  This is the top layer, highest level of abstraction.
| {
  near: top-center
}

Tik Tok's User Data: {
  link: layers.tiktok
}

layers: {
  tiktok: {
    explain: |md
      One layer deeper:

      Tik Tok's CEO explained that user data is stored in two places currently.
    | {
      near: top-center
    }
    Virginia data center <-> Hong Kong data center
    Virginia data center.link: layers.virginia
    Hong Kong data center.link: layers.hongkong
    
    layers: {
      virginia: {
        direction: right
        explain: |md
          Getting deeper into details:

          TikTok's CEO explains that Virginia data center has strict security measures.
        | {
          near: top-center
        }
        Oracle Databases: {
          shape: cylinder
          style.multiple: true
        }
        US residents -> Oracle Databases: access
        US residents: {
          shape: person
        }
        Third party auditors -> Oracle Databases: verify
      }
      hongkong: {
        direction: right
        explain: |md
          TikTok's CEO says data is actively being deleted and should be done by the end of the year.
        | {
          near: top-center
        }
        Legacy Databases: {
          shape: cylinder
          style.multiple: true
        }
      }
    }
  }
}
```

## SCENARIOS

# Scenarios
A "Scenario" represents a different view of the base Layer.
Each Scenario inherits from its base Layer. Any new objects are added onto all objects in
the base Layer, and you can reference any objects from the base Layer to update them.
Notice that in the below Scenario, we simply turn some objects opacity lower, and define a
new connection to show an alternate view of the deployment diagram.
    
```d2
direction: right

title: {
  label: Normal deployment
  near: bottom-center
  shape: text
  style.font-size: 40
  style.underline: true
}

local: {
  code: {
    icon: https://icons.terrastruct.com/dev/go.svg
  }
}
local.code -> github.dev: commit

github: {
  icon: https://icons.terrastruct.com/dev/github.svg
  dev
  master: {
    workflows
  }

  dev -> master.workflows: merge trigger
}

github.master.workflows -> aws.builders: upload and run

aws: {
  builders -> s3: upload binaries
  ec2 <- s3: pull binaries

  builders: {
    icon: https://icons.terrastruct.com/aws/Developer%20Tools/AWS-CodeBuild_light-bg.svg
  }
  s3: {
    icon: https://icons.terrastruct.com/aws/Storage/Amazon-S3-Glacier_light-bg.svg
  }
  ec2: {
    icon: https://icons.terrastruct.com/aws/_Group%20Icons/EC2-instance-container_light-bg.svg
  }
}

local.code -> aws.ec2: {
  style.opacity: 0.0
}

scenarios: {
  hotfix: {
    title.label: Hotfix deployment
    (local.code -> github.dev)[0].style: {
      stroke: "#ca052b"
      opacity: 0.1
    }

    github: {
      dev: {
        style.opacity: 0.1
      }
      master: {
        workflows: {
          style.opacity: 0.1
        }
        style.opacity: 0.1
      }

      (dev -> master.workflows)[0].style.opacity: 0.1
      style.opacity: 0.1
      style.fill: "#ca052b"
    }

    (github.master.workflows -> aws.builders)[0].style.opacity: 0.1

    (local.code -> aws.ec2)[0]: {
      style.opacity: 1
      style.stroke-dash: 5
      style.stroke: "#167c3c"
    }
  }
}
```

## STEPS

# Steps
A "Step" represents a step in a sequence of events.
Each Step inherits from its the Step before it. The first step inherits from its parent,
whether that's a Scenario or Layer.
Notice how in Step 3 for example, the object "Approach road" exists even though it's not
defined, because it was inherited from Step 2, which inherited it from Step 1.
    
```d2
Chicken's plan: {
  style.font-size: 35
  near: top-center
  shape: text
}

steps: {
  1: {
    Approach road
  }
  2: {
    Approach road -> Cross road
  }
  3: {
    Cross road -> Make you wonder why
  }
}
```

## IMPORTS

# Syntax
There are two ways to import. These two examples both have the same result:
> Result of running both types of imports below
In the next section, we'll see examples of common import use cases.
## Two types of imports
### 1. Regular import
- `x.d2`
```d2-incomplete
x: {
  shape: circle
}
```
- `y.d2`
```d2-incomplete
a: @x.d2
a -> b
```
This is the equivalent of giving the entire file of `x` as a map that `a` sets as its
value.
### 2. Spread import
- `x.d2`
```d2-incomplete
x: {
  shape: circle
}
```
- `y.d2`
```d2-incomplete
a: {
  ...@x.d2
}
a -> b
```
This tells D2 to take the contents of the file `x` and insert it into the map.
Spread imports only work within maps. Something like `a: ...@x.d2` is an invalid usage.
## Omit the extension
Above, we wrote the full file name for clarity, but the correct usage is to just specify
the file name without the suffix. If you run D2's autoformatter, it'll change
```d2-incomplete
x: @x.d2
```
into
```d2-incomplete
x: @x
```
D2 will not open files that don't have `.d2` extension, which means an import like
`@x.txt` won't work.
## Partial imports
You don't have to import the full file.
For example, if you have a file that defines all the people in your organization, and you
just want to show some relations between managers, you can import a specific object.
`donut-flowchart.d2`
    
```d2
...@people.management
joe -> donuts: loves
jan -> donuts: brings
```

`people.d2`
    
```d2
management: {
  joe: {
    shape: person
    label: Joe Donutlover
  }
  jan: {
    shape: person
    label: Jan Donutbaker
  }
}
# Notice how these do not appear in the rendered diagram
employees: {
  toby: {
    shape: person
    label: Toby Simonton
  }
}
```

Since `.` is used for targeting, if you want to import from a file with `.` in its name,
use string quotes.
`@"schema-v0.1.2"`
### Render of donut-flowchart.d2
## Relative imports
Relative imports are relative to the file, not the executing path.
Consider that your working directory is `/Users/You/dev`. Your D2 files:
- `/Users/you/dev/d2-stuff/x.d2`
```d2-incomplete
y: @../y.d2
```
The above import will search directory `/Users/you/dev/` for `y.d2`, not `/Users/You`.
Unnecessary relative imports are removed by autoformat.
`@./x` will be autoformatted to `@x`.
## Absolute imports
You can also use absolute paths for imports.
```d2-incomplete
# Unix/Linux/Mac
x: @/absolute/path/to/file
# Windows - must use quotes due to backslashes and colons
x: @"C:\absolute\path\to\file"
```
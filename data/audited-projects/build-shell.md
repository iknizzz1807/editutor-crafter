# AUDIT & FIX: build-shell

## CRITIQUE
- MASSIVE REDUNDANCY with mini-shell project. The projects cover identical territory: fork/exec, pipes, redirection, job control, signal handling. The milestones map almost 1:1. This wastes platform space and confuses students.
- To justify both projects existing, build-shell must differentiate significantly. Options: (a) make build-shell a strict superset covering advanced features (scripting, conditionals, loops, subshells, functions), or (b) merge into one project.
- Logical gap in M6 (Job Control): tcsetpgrp() is essential for transferring terminal control to the foreground process group, but it's not explicitly mentioned in ACs (only in critique findings). Without tcsetpgrp, Ctrl+C/Ctrl+Z are delivered to the shell's process group, not the foreground job.
- M2 mentions 'echo prints its arguments' and 'variable expansion support' but variable expansion ($VAR) is not covered in any milestone. This is the same gap identified in mini-shell.
- M5 (Background Jobs) and M6 (Job Control) should be merged or M6 should clearly depend on M5's signal infrastructure.
- No mention of heredocs, command substitution ($(cmd) or `cmd`), or subshells ((cmd)), which would differentiate this from mini-shell.
- The 'also_possible' languages list is empty for Go, which contradicts the tag list showing Go as tagged.
- Given the redundancy, I will rewrite this as an ADVANCED shell that builds upon mini-shell concepts, adding scripting features to justify its existence.

## FIXED YAML
```yaml
id: build-shell
name: "Build Your Own Shell (Advanced)"
description: "Full Unix shell with job control, scripting, subshells, and command substitution"
difficulty: advanced
estimated_hours: "40-60"
essence: >
  Process lifecycle management through fork/exec, inter-process communication
  via Unix pipes, signal-based job control using process groups and tcsetpgrp(),
  extended with shell scripting features including variables, conditionals,
  loops, command substitution, and subshell execution.
why_important: >
  Building a full-featured shell goes beyond basic process management to teach
  language design (parsing, AST, evaluation), concurrent process orchestration,
  and the subtleties of POSIX shell semantics that every systems engineer
  encounters daily.
learning_outcomes:
  - Implement process creation and management using fork/exec/wait
  - Build a proper lexer/parser for shell grammar (not just string splitting)
  - Implement I/O redirection and multi-stage pipelines with proper fd management
  - Build process group management with tcsetpgrp() for terminal control transfer
  - Handle signals (SIGINT, SIGTSTP, SIGCHLD) with async-signal-safe code
  - Implement shell variables, environment export, and command substitution
  - Build control flow (if/then/else, while/for loops) with exit-status-based conditionals
  - Implement subshell execution and function definitions
skills:
  - Process Management
  - System Calls (fork/exec)
  - Signal Handling
  - Job Control
  - Shell Parsing and Grammar
  - File Descriptor Manipulation
  - Scripting Language Design
  - POSIX Compliance
tags:
  - advanced
  - build-from-scratch
  - c
  - job-control
  - pipes
  - redirection
  - rust
  - scripting
  - signals
  - systems
architecture_doc: architecture-docs/build-shell/index.md
languages:
  recommended:
    - C
    - Rust
  also_possible:
    - Go
    - Zig
resources:
  - name: "GNU Implementing a Shell"
    url: "https://www.gnu.org/software/libc/manual/html_node/Implementing-a-Shell.html"
    type: documentation
  - name: "POSIX Shell Command Language"
    url: "https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html"
    type: specification
  - name: "Write a Shell in C"
    url: "https://brennan.io/2015/01/16/write-a-shell-in-c/"
    type: article
  - name: "Advanced Programming in Unix Environment - Ch 9"
    url: "https://www.apuebook.com/"
    type: book
prerequisites:
  - type: project
    name: "mini-shell (or equivalent basic shell experience)"
  - type: skill
    name: "C or Rust programming"
  - type: skill
    name: "Unix process model (fork, exec, wait)"
  - type: skill
    name: "File descriptors and signal basics"
milestones:
  - id: build-shell-m1
    name: "Lexer, Parser, and Basic Execution"
    description: >
      Build a proper lexer/parser for shell grammar (not just string splitting)
      that produces an AST, and execute simple commands from the AST.
    acceptance_criteria:
      - Lexer tokenizes input into: WORD, PIPE, REDIRECT_IN, REDIRECT_OUT, REDIRECT_APPEND, REDIRECT_ERR, AMPERSAND, SEMICOLON, NEWLINE, LPAREN, RPAREN, IF, THEN, ELSE, FI, WHILE, DO, DONE, FOR, IN
      - Parser constructs an AST with nodes for simple commands, pipelines, lists (;), and background (&)
      - Single-quoted strings are literal (no expansion); double-quoted strings allow $variable expansion but preserve spaces
      - Escape character (\) escapes the next character in unquoted and double-quoted contexts
      - External commands are executed via fork + execvp from the AST; exit status is captured via waitpid
      - Built-in commands (cd, exit, export, pwd, unset) are dispatched without forking; cd changes cwd and updates PWD; export sets env vars; student explains WHY these must be built-ins
      - Parser errors produce descriptive messages (e.g., 'syntax error near unexpected token |')
    pitfalls:
      - "String splitting is not parsing; 'echo \"hello world\"' must produce one argument, not two"
      - "Nested quotes and escape sequences require careful state tracking in the lexer"
      - "AST design affects extensibility; flat token lists make control flow very hard to add later"
      - execvp failure in child: must call _exit(127) not exit() to avoid flushing parent's stdio buffers
    concepts:
      - Lexical analysis and tokenization
      - Recursive descent or precedence-climbing parsing
      - Abstract syntax tree design
      - fork/exec pattern
    skills:
      - Language parser implementation
      - AST design
      - Process management
    deliverables:
      - Shell lexer producing token stream
      - Shell parser producing AST
      - AST executor for simple commands
      - Built-in command dispatcher (cd, exit, export, pwd, unset)
      - Error reporting for syntax errors
    estimated_hours: "6-9"

  - id: build-shell-m2
    name: "Pipes, Redirection, and Expansions"
    description: >
      Implement pipelines, I/O redirection, and shell expansions
      (variable, tilde, glob, command substitution).
    acceptance_criteria:
      - Pipeline 'cmd1 | cmd2 | cmd3' chains stdout→stdin; all processes run concurrently; exit status is last command's
      - Redirection: < (input), > (output/truncate), >> (append), 2> (stderr), 2>&1 (stderr to stdout), &> (both to file)
      - All pipe fds are closed in parent and non-participating children to prevent hangs
      - Variable expansion: $VAR, ${VAR}, $?, $$, $0, $1...$9 (positional params for scripts)
      - Tilde expansion: ~ → $HOME, ~user → user's home directory
      - Glob expansion: *, ?, [...] matched against files in cwd; no match passes literal
      - Command substitution: $(cmd) executes cmd in a subshell and replaces with its stdout (trailing newlines stripped)
      - Expansion order: tilde → parameter → command substitution → field splitting → glob → quote removal
    pitfalls:
      - "Command substitution requires creating a subshell (fork), capturing its stdout via pipe, and substituting the output — this is recursive shell invocation"
      - "Nested command substitution $(echo $(date)) must work; requires recursive parsing"
      - "Glob expansion in the wrong directory or not escaping glob characters in variable values"
      - 2>&1 order matters: '> file 2>&1' redirects both to file; '2>&1 > file' redirects stderr to original stdout, then stdout to file
    concepts:
      - Shell expansion pipeline
      - Command substitution as subshell
      - File descriptor duplication semantics
    skills:
      - Expansion implementation
      - Subshell execution
      - Pattern matching
      - Fd manipulation
    deliverables:
      - Pipeline executor with concurrent processes
      - Full redirection support (<, >, >>, 2>, 2>&1, &>)
      - Variable expansion ($VAR, ${VAR}, specials)
      - Tilde and glob expansion
      - Command substitution $(cmd)
    estimated_hours: "7-10"

  - id: build-shell-m3
    name: "Signal Handling and Job Control"
    description: >
      Implement process groups, terminal control transfer, signal handling,
      and full job control (fg, bg, jobs, Ctrl+Z, Ctrl+C).
    acceptance_criteria:
      - Shell ignores SIGINT, SIGTSTP, SIGTTOU in the shell process itself
      - Each pipeline is placed in its own process group using setpgid() called in BOTH parent and child to avoid race
      - Foreground job's process group is given terminal control via tcsetpgrp(STDIN_FILENO, job_pgid) before waiting
      - Shell reclaims terminal control via tcsetpgrp(STDIN_FILENO, shell_pgid) after foreground job completes or stops
      - Ctrl+C sends SIGINT only to the foreground job's process group; shell is unaffected
      - Ctrl+Z sends SIGTSTP to the foreground job's process group; shell updates job state to Stopped and shows prompt
      - SIGCHLD handler reaps background processes with waitpid(-1, ..., WNOHANG) using async-signal-safe functions only
      - Trailing '&' launches pipeline in background; shell prints '[N] PID' and returns prompt
      - 'jobs' lists all jobs with number, state (Running/Stopped/Done), and command
      - 'fg %N' sends SIGCONT if stopped, gives terminal control, and waits for completion/stop
      - 'bg %N' sends SIGCONT to stopped job, leaves it in background
      - Completed background jobs are reported at next prompt
    pitfalls:
      - tcsetpgrp() is the CRITICAL piece: without it, terminal-generated signals (Ctrl+C, Ctrl+Z) go to the shell's group, not the foreground job
      - "Race between parent's setpgid and child's exec; both must call setpgid independently"
      - "SIGCHLD handler must NOT call printf, malloc, or any non-async-signal-safe function; use write() for output"
      - "sigprocmask to block SIGCHLD during job table modifications prevents corruption"
      - "Orphaned process groups receive SIGHUP then SIGCONT when the shell exits; document this behavior"
    concepts:
      - Process groups and sessions
      - Terminal controlling process group (tcsetpgrp)
      - Async-signal safety
      - Job state machine (running → stopped → running → done)
    skills:
      - Signal handling
      - Process group management
      - Terminal session control
      - Job table implementation
    deliverables:
      - Shell signal setup (ignore SIGINT, SIGTSTP, SIGTTOU)
      - Process group creation with setpgid in parent + child
      - Terminal control transfer via tcsetpgrp
      - SIGCHLD handler with async-signal-safe waitpid
      - Job table tracking pgid, state, command
      - fg, bg, jobs built-in commands
      - Background job completion notification
    estimated_hours: "8-12"

  - id: build-shell-m4
    name: "Control Flow and Scripting"
    description: >
      Implement if/else, while, for loops, functions, and script file execution.
    acceptance_criteria:
      - 'if cmd; then body; elif cmd; then body; else body; fi' — condition is the exit status of cmd (0=true)
      - 'while cmd; do body; done' — loops while cmd exits 0; 'until' loops while cmd exits non-zero
      - 'for var in word1 word2 word3; do body; done' — iterates var over the word list; words undergo expansion
      - Shell functions: 'fname() { body; }' defines a function; calling fname executes body in the current shell; $1...$9 are function arguments
      - Script execution: 'sh script.sh' or './script.sh' reads and executes commands from a file line by line
      - 'return N' exits a function with status N; 'break' and 'continue' work in loops
      - Nested control structures work: if inside while inside for
      - '#' begins a comment; everything from '#' to end of line is ignored
    pitfalls:
      - "If condition is a command's exit status, NOT a boolean expression; 'if true; then' works because 'true' is a command that exits 0"
      - "test/[ command is external (/usr/bin/test); the shell doesn't need to implement comparison operators itself"
      - "Function definitions must be stored (name → AST body) and dispatched before external command lookup"
      - Variable scope: shell functions share the parent's variable scope by default (no local scope unless using 'local')
      - "Script execution must handle line continuations (trailing \\) and multi-line constructs"
    concepts:
      - Exit-status-based conditionals
      - Loop constructs
      - Function definitions and scope
      - Script interpretation
    skills:
      - Control flow implementation
      - AST evaluation for complex structures
      - Function/scope management
      - Script file processing
    deliverables:
      - if/elif/else/fi conditional execution
      - while/until/for loop execution
      - Function definition and invocation
      - break, continue, return built-ins
      - Script file reader and executor
      - Comment handling
    estimated_hours: "8-12"

  - id: build-shell-m5
    name: "Subshells and Advanced Features"
    description: >
      Implement subshell execution, here-documents, and logical operators.
    acceptance_criteria:
      - Parenthesized commands '(cmd1; cmd2)' execute in a subshell (forked child); variable changes don't affect parent
      - Logical AND '&&' executes right side only if left side exits 0
      - Logical OR '||' executes right side only if left side exits non-zero
      - Here-document '<<EOF ... EOF' provides multi-line input to a command's stdin; delimiter is configurable
      - Pipeline with subshell: '(cmd1; cmd2) | cmd3' works correctly
      - Semicolon ';' sequences commands: 'cmd1 ; cmd2' runs cmd1 then cmd2 regardless of exit status
      - 'set -e' causes the shell to exit on any command failure (non-zero exit status)
      - Command grouping with braces '{ cmd1; cmd2; }' executes in the CURRENT shell (unlike subshell)
    pitfalls:
      - "Subshell (parentheses) forks; brace group does not. This is a subtle but critical distinction."
      - "Here-document content must be buffered and provided via pipe or temp file to the command's stdin"
      - "&& and || have lower precedence than pipes but higher than ; — parser must handle this correctly"
      - "'set -e' interacts complexly with conditionals and loops; commands in if/while conditions don't trigger -e"
    concepts:
      - Subshell vs brace group
      - Short-circuit logical operators
      - Here-documents
      - Shell options (set -e, set -x)
    skills:
      - Subshell implementation
      - Operator precedence in parser
      - Here-document handling
      - Shell option management
    deliverables:
      - Subshell execution with (cmd) syntax
      - Brace group execution with { cmd; } syntax
      - Logical operators && and ||
      - Here-document <<EOF support
      - set -e and set -x shell options
      - Correct operator precedence in parser
    estimated_hours: "7-10"
```
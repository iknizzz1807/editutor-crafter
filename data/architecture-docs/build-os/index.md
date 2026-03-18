# 🎯 Project Charter: Build Your Own OS
## What You Are Building
A monolithic x86 operating system kernel that boots from BIOS, transitions to 32-bit protected mode, handles hardware interrupts, manages physical and virtual memory through two-level page tables, and implements preemptive multitasking with process isolation. By the end, your kernel will boot in QEMU, run multiple processes concurrently with round-robin scheduling, and provide a system call interface for user-mode programs.
## Why This Project Exists
Most developers use operating system abstractions daily—processes, virtual memory, interrupts—but treat them as black boxes. Building a kernel from scratch exposes the hardware reality beneath every abstraction: why page faults occur, how the CPU switches between processes, and what actually happens when you press a key. This project strips away the operating system layer that has stood between you and the hardware your entire career.
## What You Will Be Able to Do When Done
- Boot a custom kernel from bare metal (or QEMU) starting in 16-bit real mode
- Configure the Global Descriptor Table and transition to 32-bit protected mode
- Handle hardware interrupts (timer, keyboard) and CPU exceptions (page faults, GPFs)
- Implement a physical frame allocator and virtual memory with page tables
- Create and destroy processes with isolated address spaces
- Implement preemptive round-robin scheduling triggered by timer interrupts
- Enter user mode (Ring 3) and handle system calls via INT 0x80
- Debug kernel panics, triple faults, and memory corruption using QEMU and serial output
## Final Deliverable
~3,000-4,000 lines of C and x86 assembly across 25+ source files. Produces a bootable disk image under 1MB. Boots in under 2 seconds in QEMU. Demonstrates preemptive multitasking with 3+ kernel processes running concurrently, each writing to different screen regions, plus at least one user-mode process making system calls.
## Is This Project For You?
**You should start this if you:**
- Are comfortable with C pointers, memory layout, and bit manipulation
- Understand hexadecimal and binary arithmetic intuitively
- Have used command-line build tools (make, gcc, assemblers)
- Are curious about what happens below the application layer
- Have patience for debugging silent failures (triple faults give no error messages)
**Come back after you've learned:**
- C programming with pointers and manual memory management
- Basic computer architecture (registers, stack, memory bus)
- Hexadecimal number system and bitwise operations
- Command-line development environment usage
## Estimated Effort
| Phase | Time |
|-------|------|
| Bootloader, GDT, and Kernel Entry | ~20-30 hours |
| Interrupts, Exceptions, and Keyboard | ~15-25 hours |
| Physical and Virtual Memory Management | ~30-45 hours |
| Processes and Preemptive Scheduling | ~35-55 hours |
| **Total** | **~100-155 hours** |
## Definition of Done
The project is complete when:
- Kernel boots in QEMU from a disk image without crashing
- Timer interrupt fires at 100Hz with visible tick counter incrementing
- Keyboard input appears correctly on screen with proper scancode-to-ASCII translation
- Physical frame allocator can allocate and free frames without double-free bugs
- Paging is enabled with identity mapping and higher-half kernel mapping working
- At least 3 kernel processes run concurrently demonstrating preemptive multitasking
- At least 1 user-mode process executes system calls (sys_write, sys_getpid, sys_exit)
- User process page fault occurs when attempting to access kernel memory (isolation verified)

---

# 📚 Before You Read This: Prerequisites & Further Reading
> **Read these first.** The Atlas assumes you are familiar with the foundations below.
> Resources are ordered by when you should encounter them — some before you start, some at specific milestones.
---
## Foundations (Read BEFORE Starting)
### 1. x86 Architecture Fundamentals
**📖 Best Explanation:** *The Undocumented PC* by Frank van Gilluwe — Chapter 2 "Processor Architecture"
- **Why:** The most accessible explanation of real mode vs protected mode, segment:offset addressing, and the boot process. Van Gilluwe explains the "why" behind Intel's design decisions, not just the "what."
- **When:** Before writing a single line of assembly. You need to understand why the CPU starts in 16-bit mode.
**📄 Spec:** Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3A — Chapter 5 "Protection"
- **Sections:** 5.1-5.5 (segmentation, privilege levels, descriptor tables)
- **Why:** The authoritative source. Every OS developer references this.
- **When:** Keep open while implementing the GDT (Milestone 1).
### 2. Assembly Language
**📖 Best Explanation:** *Programming from the Ground Up* by Jonathan Bartlett — Chapters 1-4
- **Available:** Free online at https://download.savannah.gnu.org/releases/pgubook/
- **Why:** Starts from first principles. You'll understand registers, memory addressing, and the stack by actually using them, not just reading about them.
- **When:** Before Milestone 1. Work through the examples.
**💻 Code Reference:** OSDev Wiki — "Bare Bones" tutorial
- **URL:** https://wiki.osdev.org/Bare_Bones
- **Why:** A minimal working example of a multiboot kernel. Compare your bootloader approach to GRUB's multiboot handoff.
- **When:** After completing Milestone 1, to see an alternative approach.
---
## Milestone 1: Bootloader & Protected Mode
### 3. The Boot Process
**📖 Best Explanation:** *Write Great Code, Volume 2* by Randall Hyde — Chapter 4 "Memory Layout and Access"
- **Why:** Explains the memory map at boot, why 0x7C00, and the relationship between BIOS and your bootloader.
- **When:** While implementing `boot.asm`.
**📄 Spec:** IBM PC Technical Reference (1981) — BIOS listing
- **Why:** Historical context for why things are the way they are. The 512-byte MBR limit? That's IBM's decision in 1981.
- **When:** After Milestone 1, for perspective on backwards compatibility.
### 4. The A20 Line
**📖 Best Explanation:** OSDev Wiki — "A20 Line"
- **URL:** https://wiki.osdev.org/A20
- **Why:** The clearest explanation of this historical quirk, with working code for all three enable methods.
- **When:** While implementing `a20.asm`. Copy the keyboard controller sequence — it's the most reliable.
### 5. Global Descriptor Table
**📖 Best Explanation:** *Understanding the Linux Kernel* by Bovet & Cesati — Chapter 2 "Segmentation in Linux"
- **Why:** Shows how a production OS uses the GDT. The "flat model" (base=0, limit=4GB) is what everyone uses — Linux, Windows, your kernel too.
- **When:** Before Milestone 1. After reading this, the GDT entry format makes sense.
---
## Milestone 2: Interrupts
### 6. Interrupt Mechanics
**📄 Spec:** Intel SDM Volume 3A — Chapter 6 "Interrupt and Exception Handling"
- **Sections:** 6.1-6.5 (interrupt gates, IDT structure, privilege checking)
- **Why:** The authoritative explanation of what the CPU pushes on the stack and when.
- **When:** While implementing IDT entries. Reference the gate descriptor bit fields constantly.
**📖 Best Explanation:** *The Indispensable PC Hardware Guide* by Hans-Peter Messmer — Chapter 7 "Interrupts"
- **Why:** Explains the 8259 PIC in practical terms. The ICW1-ICW4 initialization sequence finally makes sense.
- **When:** Before implementing `pic.c`.
### 7. The PIC and IRQs
**💻 Code Reference:** Linux kernel source — `arch/x86/kernel/i8259.c`
- **Why:** See how a production OS initializes the PIC. The remapping sequence is nearly identical to yours.
- **When:** After completing Milestone 2, to compare your implementation.
### 8. Keyboard Driver
**📄 Spec:** IBM PS/2 Hardware Technical Reference — Keyboard Controller
- **Why:** The original specification for scancode set 1. Essential for understanding multi-byte scancodes.
- **When:** While implementing `keyboard.c`. Print raw scancodes first, then build the translation tables.
---
## Milestone 3: Memory Management
### 9. Virtual Memory Concepts
**📖 Best Explanation:** *Operating Systems: Three Easy Pieces* by Remzi Arpaci-Dusseau — Chapters 18-22 "Address Translation" and "Paging"
- **URL:** https://pages.cs.wisc.edu/~remzi/OSTEP/
- **Why:** The clearest explanation of why we need page tables, how they work, and the translation process. Remzi uses analogies that actually stick.
- **When:** BEFORE starting Milestone 3. This is required foundational knowledge.
**📄 Spec:** Intel SDM Volume 3A — Chapter 4 "Paging"
- **Sections:** 4.1-4.3 (two-level page tables, PDE/PTE formats)
- **Why:** The bit-by-bit format of page table entries. You'll reference this constantly while implementing `vmm.c`.
- **When:** Keep open during Milestone 3 implementation.
### 10. Page Table Implementation
**💻 Code Reference:** xv6 source — `vm.c`
- **URL:** https://github.com/mit-pdos/xv6-public/blob/master/vm.c
- **Why:** A teaching OS from MIT. Their `walkpgdir()` function is exactly what you're implementing, but simpler and well-commented.
- **When:** When stuck on page table walks. Compare their approach to yours.
### 11. Memory Allocators
**📖 Best Explanation:** *The C Programming Language* (K&R) by Kernighan & Ritchie — Section 8.7 "Storage Allocator"
- **Why:** The classic explanation of a linked-list heap allocator. Your `kheap.c` is essentially this implementation.
- **When:** While implementing `kmalloc` and `kfree`.
---
## Milestone 4: Processes & Scheduling
### 12. Process Concepts
**📖 Best Explanation:** *Operating Systems: Three Easy Pieces* — Chapters 3-6 "Processes" and "Direct Execution"
- **Why:** Remzi explains why context switching is hard: saving registers, the kernel stack, and the limited direct execution protocol. Essential reading.
- **When:** BEFORE starting Milestone 4. Required foundational knowledge.
### 13. Context Switching
**💻 Code Reference:** xv6 source — `swtch.S`
- **URL:** https://github.com/mit-pdos/xv6-public/blob/master/swtch.S
- **Why:** A real context switch in ~40 lines of assembly. Compare your approach: they use `pushal`/`popal`, you save registers individually.
- **When:** While debugging `context_switch.asm`. Their comments are excellent.
### 14. The Task State Segment
**📄 Spec:** Intel SDM Volume 3A — Chapter 7 "Task Management"
- **Sections:** 7.1-7.3 (TSS structure, why it exists, ESP0 usage)
- **Why:** The TSS is confusing because Intel designed it for hardware task switching (which nobody uses). This chapter explains what you actually need: just ESP0 and SS0.
- **When:** Before implementing `tss.c`. Understand that you're using 2 of 104 bytes.
### 15. User Mode & Privilege Transitions
**📖 Best Explanation:** *Understanding the Linux Kernel* — Chapter 10 "System Calls"
- **Why:** Explains the full path: user → kernel → user. The `iret` instruction, privilege level transitions, and why the kernel stack matters.
- **When:** While implementing `jump_to_user_mode()`.
### 16. Scheduling Algorithms
**📖 Best Explanation:** *Operating System Concepts* (Silberschatz) — Chapter 5 "CPU Scheduling"
- **Sections:** 5.1-5.3 (scheduling criteria, algorithms, round-robin)
- **Why:** The classic textbook explanation. Round-robin is the simplest fair algorithm — understand why before implementing.
- **When:** Before implementing `scheduler.c`.
---
## Deep Dives (After Completion)
### 17. Modern x86 Features
**📄 Spec:** Intel SDM Volume 3B — Chapter 4 "Virtual Memory" (extended)
- **Topics:** PAE (36-bit addressing), 64-bit mode page tables (4-level), PCID (avoiding TLB flushes)
- **Why:** Your kernel uses 32-bit two-level page tables. Modern CPUs use 4-level page tables for 48-bit virtual addresses. This is how we got here.
- **When:** After Milestone 3, if curious about 64-bit OS development.
### 18. Copy-on-Write Fork
**📖 Best Explanation:** *Operating Systems: Three Easy Pieces* — Chapter 5 "Beyond Atomicity: The Racy Cow"
- **Why:** The key optimization that makes `fork()` fast. Your scheduler + page tables provide the primitives.
- **When:** After completing all milestones, when thinking about process creation.
### 19. Multiboot Standard
**📄 Spec:** Multiboot Specification (version 0.6.96)
- **URL:** https://www.gnu.org/software/grub/manual/multiboot/multiboot.html
- **Why:** GRUB's standard for booting kernels. Your bootloader does everything manually; multiboot lets GRUB handle it.
- **When:** After Milestone 1, if you want to boot from GRUB instead of your own bootloader.
### 20. Advanced Debugging
**💻 Tool:** QEMU Monitor Commands
- **URL:** https://qemu.readthedocs.io/en/latest/system/monitor.html
- **Commands:** `info registers`, `info mem`, `info tlb`, `x/20x 0x7c00`
- **Why:** Your only debugging tools in kernel development. Learn to inspect memory, registers, and page tables without a traditional debugger.
- **When:** Throughout all milestones. Use `-d int` and `-no-reboot` liberally.
---
## Quick Reference (Keep Open)
| Topic | Primary Reference | Section |
|-------|-------------------|---------|
| GDT entries | Intel SDM Vol 3A | §3.4.5 |
| IDT entries | Intel SDM Vol 3A | §6.11 |
| Page table format | Intel SDM Vol 3A | §4.3 |
| TSS structure | Intel SDM Vol 3A | §7.2.1 |
| Exception vectors | Intel SDM Vol 3A | §6.15 |
| PIC programming | 8259A datasheet | Command sequence |
| PIT programming | 8254 datasheet | Mode 3 square wave |
| Scancode set 1 | IBM PS/2 Tech Ref | Keyboard appendix |
---
## Reading Timeline
```
Before M1: x86 fundamentals, assembly basics
     ↓
During M1: Boot process, GDT, A20
     ↓
Before M2: Interrupt concepts (OSTEP Ch 6)
     ↓
During M2: IDT/PIC specs, keyboard scancodes
     ↓
Before M3: Virtual memory (OSTEP Ch 18-22) ★ REQUIRED
     ↓
During M3: Page table format (Intel SDM), allocators (K&R)
     ↓
Before M4: Processes (OSTEP Ch 3-6) ★ REQUIRED
     ↓
During M4: Context switch, TSS, scheduling
     ↓
After: Copy-on-write, 64-bit paging, multiboot
```

---

# Build Your Own OS

A comprehensive journey through x86 operating system kernel development, from the moment power is applied to a fully preemptive multi-process system with virtual memory. This project strips away every abstraction between you and the hardware — you will configure the Global Descriptor Table, handle hardware interrupts, manage physical and virtual memory, and implement preemptive scheduling with ring transitions. The result is a working kernel that boots from bare metal, runs multiple processes, and provides system calls for user-mode programs.



<!-- MS_ID: build-os-m1 -->
# Milestone 1: Bootloader, GDT, and Kernel Entry

![OS Kernel Architecture Map](./diagrams/diag-satellite-os.svg)

You are about to do something that almost no programmer ever does: write the very first code that runs on a computer.
When you press the power button, the CPU doesn't know about your operating system. It doesn't know about files, processes, or memory protection. It wakes up in a primitive 16-bit mode designed in 1978, capable of addressing only 1 megabyte of memory, with no concept of multitasking or security.
Your job? Bridge the gap between that ancient hardware state and a modern 32-bit kernel running C code. You have 512 bytes to do it — less than this paragraph.
## The Fundamental Tension: Hardware Constraints Meet Software Expectations
Every abstraction you've ever relied on — `printf`, `malloc`, even the concept of "processes" — exists because someone built it. Today, that someone is you.
The tension here is physical and architectural:
- **The CPU starts in real mode** (16-bit, 1MB addressable) because Intel needed backward compatibility with the 8086
- **Your kernel needs protected mode** (32-bit, 4GB addressable, memory protection) to do anything useful
- **The transition requires precise orchestration** — a Global Descriptor Table (GDT) defining memory segments, a specific sequence of register manipulations, and a far jump to flush the pipeline
- **You have almost no code space** — the Master Boot Record (MBR) is exactly 512 bytes, with the last two bytes reserved for the boot signature
Miss one step, use one wrong instruction, forget one register — and the CPU triple-faults, resetting as if you'd pressed the power button. There's no debugger, no error message, no stack trace. Just silence.

![x86 Boot Sequence Timeline](./diagrams/diag-boot-sequence.svg)

## What Actually Happens When You Press Power
Before we write code, you need to understand the handoff you're receiving:
### Level 1 — Application View (What You're Used To)
Programs call `main()`, use `printf()`, allocate memory with `malloc()`. The OS provides process isolation, file systems, and device drivers. Crashes produce error messages.
### Level 2 — OS/Kernel View (What You're Building)
The kernel initializes hardware, manages memory through page tables, schedules processes, and handles interrupts. It runs in "kernel mode" with full hardware access.
### Level 3 — Hardware View (Where You Start)
When power is applied:
1. **CPU loads CS:IP = 0xF000:0xFFF0** — execution begins at the BIOS entry point in ROM
2. **BIOS performs POST** (Power-On Self-Test), initializes basic hardware
3. **BIOS searches for bootable disk** — reads sector 0 (the MBR) into memory at 0x7C00
4. **BIOS jumps to 0x7C00** — your bootloader code is now executing
5. **CPU is in 16-bit real mode** — 20-bit addresses (1MB max), no memory protection, no privilege separation
You are now the bootloader. The BIOS is gone. You're on your own.
```
Physical Memory at Boot:
0x000000 - 0x0003FF  : Real Mode IVT (256 interrupt vectors)
0x000400 - 0x0004FF  : BIOS Data Area
0x000500 - 0x07BFF   : Free conventional memory
0x0007C00 - 0x0007DFF: Bootloader (YOU ARE HERE - 512 bytes)
0x0007E00 - 0x0009FBFF: Free conventional memory
0x000A000 - 0x000BFFF: VGA video memory
0x000C000 - 0x000EFFF: BIOS ROM
0x000F000 - 0x000FFFF: BIOS ROM (entry point at 0xF0000)
0x00100000+          : Extended memory (your kernel goes here)
```

![Low Memory Map at Boot Time](./diagrams/diag-memory-map-boot.svg)

## The Bootloader's Mission
Your 512-byte bootloader must accomplish the following before transferring control to C code:
1. **Enable the A20 line** — allow addressing memory above 1MB
2. **Load the kernel from disk** — read sectors into memory at 0x100000
3. **Set up the GDT** — define memory segments for protected mode
4. **Enter protected mode** — switch the CPU to 32-bit operation
5. **Jump to kernel entry** — transfer control with proper segment registers
Let's build each piece.
## Step 1: The Boot Stub — Real Mode Assembly

> **🔑 Foundation: x86 assembly syntax and registers**
> 
> ## What It Is
x86 assembly is a low-level programming language where each instruction corresponds directly to a CPU operation. Unlike high-level languages, you're explicitly telling the processor which storage locations (registers) to use for every calculation.
**Registers** are the CPU's internal storage slots—extremely fast, but limited in number. In 32-bit x86 (IA-32), the general-purpose registers are:
| Register | Common Purpose |
|----------|----------------|
| `EAX` | Accumulator—arithmetic results, return values |
| `EBX` | Base—often holds pointers to data |
| `ECX` | Counter—loop iterations, string operations |
| `EDX` | Data—I/O operations, multiplication overflow |
| `ESI` | Source Index—source pointer in string/copy ops |
| `EDI` | Destination Index—destination pointer |
| `EBP` | Base Pointer—stack frame reference |
| `ESP` | Stack Pointer—top of stack |
The `E` prefix means "Extended" (32-bit). In 16-bit mode, these become `AX`, `BX`, `CX`, `DX`, etc. In 64-bit mode, they gain an `R` prefix (`RAX`, `RBX`...).
**Syntax flavors**: There are two main assembly syntaxes:
- **Intel syntax**: `mov eax, 5` (destination first)
- **AT&T syntax**: `movl $5, %eax` (source first, with sigils)
For OS development, Intel syntax is more common and generally more readable.
## Why You Need It Now
You're writing a kernel. The bootloader hands control to your code in a specific CPU state. You need assembly to:
- Set up stack pointers (`ESP`)
- Load segment registers for memory protection
- Interface with hardware ports (`in`/`out` instructions)
- Handle interrupts and save/restore register states
- Perform operations C can't express (like modifying control registers)
## Key Insight
**Think of registers as the CPU's "working desk."** You can only work on what's on your desk. To process data in memory, you first `mov` it to a register, manipulate it, then `mov` it back. This load-modify-store pattern is fundamental.
```
; Add 10 to a memory variable
mov eax, [my_variable]    ; Load from memory to register
add eax, 10               ; Modify in register
mov [my_variable], eax    ; Store back to memory
```

Create `boot/boot.asm`:
```nasm
; boot/boot.asm - Stage 1 Bootloader
; Assembled with: nasm -f bin boot/boot.asm -o boot.bin
; Must be exactly 512 bytes with 0x55AA signature at bytes 510-511
[BITS 16]           ; We start in 16-bit real mode
[ORG 0x7C00]        ; BIOS loads us at physical address 0x7C00
start:
    ; Set up segment registers and stack
    cli                 ; Disable interrupts during setup
    xor ax, ax          ; Zero AX
    mov ds, ax          ; Data segment = 0
    mov es, ax          ; Extra segment = 0
    mov ss, ax          ; Stack segment = 0
    mov sp, 0x7C00      ; Stack grows down from bootloader
    sti                 ; Re-enable interrupts
    ; Save boot drive number (BIOS puts it in DL)
    mov [boot_drive], dl
    ; Display boot message using BIOS INT 10h
    mov si, boot_msg
    call print_string
    ; Enable A20 line
    call enable_a20
    ; Load kernel from disk
    call load_kernel
    ; Set up GDT
    call setup_gdt
    ; Enter protected mode
    cli                 ; Disable interrupts for mode switch
    mov eax, cr0
    or eax, 1           ; Set PE bit (Protection Enable)
    mov cr0, eax
    ; Far jump to flush pipeline and load CS with 32-bit code segment
    jmp 0x08:protected_mode_entry
    ; --- Include GDT data ---
    %include "boot/gdt.asm"
    ; --- Data ---
    boot_drive: db 0
    boot_msg: db 'Booting MyOS...', 13, 10, 0
    ; --- Functions (16-bit) ---
    ; print_string: Print null-terminated string at SI
    print_string:
        pusha
        mov ah, 0x0E        ; BIOS teletype output
    .loop:
        lodsb               ; Load byte from [SI] into AL, increment SI
        cmp al, 0
        je .done
        int 0x10            ; BIOS video interrupt
        jmp .loop
    .done:
        popa
        ret
    ; enable_a20: Try multiple methods to enable A20 line
    %include "boot/a20.asm"
    ; load_kernel: Read kernel sectors from disk
    %include "boot/load_kernel.asm"
    ; Padding and boot signature
    times 510 - ($ - $$) db 0
    dw 0xAA55              ; Boot signature (little-endian: 0x55 at 510, 0xAA at 511)
```
### Understanding the Boot Stub
The `[ORG 0x7C00]` directive tells the assembler that our code will be loaded at address 0x7C00. This affects how it calculates label addresses — when we write `mov si, boot_msg`, the assembler encodes the absolute address 0x7C00 + offset_of_boot_msg.
The first instructions disable interrupts (`cli`), zero the segment registers, and set up a stack growing downward from 0x7C00. We can't use addresses above the bootloader for the stack because we might overwrite our own code.
## Step 2: The A20 Line — Intel's Historical Accident

![A20 Line Enable Methods](./diagrams/diag-a20-line.svg)

The A20 line is a quirk of PC history. The original 8086 had 20 address lines (A0-A19), limiting it to 1MB. Memory addresses above 1MB would "wrap around" — address 0x100000 (1MB + 0) would physically access 0x00000.
Some early software relied on this wraparound. When the 286 introduced 24 address lines (16MB), IBM needed to maintain compatibility. Their solution: a gate on the A20 line that could disable it, forcing wraparound behavior.
For your OS, this means: **the A20 line is disabled by default on many systems**. Without it, you can't access memory above 1MB — where your kernel lives.
Create `boot/a20.asm`:
```nasm
; boot/a20.asm - Enable A20 line using multiple methods
enable_a20:
    ; Method 1: BIOS INT 15h / 2401 (safest, may not exist)
    mov ax, 0x2401
    int 0x15
    jc .try_fast_a20       ; If carry set, BIOS method failed
    call test_a20
    cmp ax, 1
    je .done
.try_fast_a20:
    ; Method 2: Fast A20 via port 0x92 (System Control Port A)
    in al, 0x92
    test al, 2             ; Check if A20 already enabled
    jnz .done
    or al, 2               ; Set A20 bit
    and al, 0xFE           ; Clear reset bit (safety!)
    out 0x92, al
    call test_a20
    cmp ax, 1
    je .done
.try_kbd_a20:
    ; Method 3: Keyboard controller (slow but reliable)
    call kbd_wait_input
    mov al, 0xD0           ; Read output port command
    out 0x64, al
    call kbd_wait_output
    in al, 0x60            ; Read current output port value
    push ax
    call kbd_wait_input
    mov al, 0xD1           ; Write output port command
    out 0x64, al
    call kbd_wait_input
    pop ax
    or al, 2               ; Set A20 bit
    out 0x60, al
    call kbd_wait_input
    mov al, 0xAE           ; Re-enable keyboard
    out 0x64, al
    call kbd_wait_input
    call test_a20
    cmp ax, 1
    je .done
    ; If all methods fail, halt
    mov si, a20_error_msg
    call print_string
    jmp $
.done:
    ret
; test_a20: Returns AX=1 if A20 enabled, AX=0 if not
test_a20:
    push es
    push ds
    push di
    ; Set ES:DI to 0x0000:0x0500 (below 1MB)
    xor ax, ax
    mov es, ax
    mov di, 0x0500
    ; Set DS:SI to 0xFFFF:0x0510 (wraparound alias of 0x0000:0x0500)
    mov ax, 0xFFFF
    mov ds, ax
    mov si, 0x0510
    ; Save original values
    mov al, [es:di]
    push ax
    ; Write different values and check if they match
    mov byte [es:di], 0x00
    mov byte [ds:si], 0xFF
    cmp byte [es:di], 0xFF
    pop ax
    mov [es:di], al        ; Restore original value
    mov ax, 0
    je .a20_disabled       ; If values match, A20 is disabled
    mov ax, 1
.a20_disabled:
    pop di
    pop ds
    pop es
    ret
; kbd_wait_input: Wait until keyboard controller input buffer empty
kbd_wait_input:
    in al, 0x64
    test al, 2
    jnz kbd_wait_input
    ret
; kbd_wait_output: Wait until keyboard controller output buffer full
kbd_wait_output:
    in al, 0x64
    test al, 1
    jz kbd_wait_output
    ret
a20_error_msg: db 'ERROR: Could not enable A20 line', 13, 10, 0
```
### The A20 Test Explained
The test exploits the wraparound behavior. In real mode:
- Address 0x0000:0x0500 = 0x00500 (linear)
- Address 0xFFFF:0x0510 = 0xFFFF0 + 0x0510 = 0x100500 → wraps to 0x00500 (if A20 disabled)
If writing to one location changes the other, A20 is disabled (wraparound active). If they're independent, A20 is enabled.
## Step 3: Loading the Kernel from Disk
The kernel binary sits on disk starting at sector 2 (sector 1 is the MBR). You'll use BIOS INT 13h to read it into memory at 0x100000 (1MB mark).
Create `boot/load_kernel.asm`:
```nasm
; boot/load_kernel.asm - Load kernel from disk using BIOS INT 13h
KERNEL_LOAD_SEGMENT equ 0x1000   ; Segment for kernel load (0x1000:0x0000 = 0x10000)
                                ; We'll use a temporary buffer and copy later
KERNEL_LOAD_OFFSET  equ 0x0000
KERNEL_SECTORS      equ 64      ; Load 64 sectors = 32KB (adjust as needed)
load_kernel:
    mov si, load_kernel_msg
    call print_string
    ; Reset disk system
    xor ax, ax
    mov dl, [boot_drive]
    int 0x13
    jc .disk_error
    ; Read kernel sectors
    ; We'll read in chunks since we're limited to ~127 sectors per call
    mov ax, KERNEL_LOAD_SEGMENT
    mov es, ax
    xor bx, bx              ; ES:BX = destination buffer
    mov ah, 0x42            ; Extended read function
    mov dl, [boot_drive]
    mov si, disk_address_packet
    int 0x13
    jc .disk_error
    ret
.disk_error:
    mov si, disk_error_msg
    call print_string
    jmp $
load_kernel_msg: db 'Loading kernel...', 13, 10, 0
disk_error_msg: db 'Disk read error!', 13, 10, 0
; Disk Address Packet for extended read
disk_address_packet:
    db 0x10                 ; Packet size (16 bytes)
    db 0                    ; Reserved
    dw KERNEL_SECTORS       ; Number of sectors to read
    dw KERNEL_LOAD_OFFSET   ; Offset
    dw KERNEL_LOAD_SEGMENT  ; Segment (will be overwritten for 0x100000)
    dq 2                    ; Starting LBA (sector 2, after MBR)
```
### A Problem: Segment Addressing Can't Reach 0x100000
Real mode segments can only address up to 0xFFFF:0xFFFF = 0x10FFEF, but we want to load at 0x100000. The segment value 0x1000 only gives us 0x10000.
**Solution**: Use unreal mode or load to a lower address and copy after entering protected mode. For simplicity, we'll use the second approach:
1. Load kernel to 0x10000 in real mode
2. After entering protected mode, copy from 0x10000 to 0x100000
## Step 4: The Global Descriptor Table (GDT)

> **🔑 Foundation: Memory segmentation**
> 
> ## What It Is
Memory segmentation is a way of dividing memory into distinct regions called **segments**, each with its own purpose and access rules. Instead of one flat address space, memory is organized as a collection of segments like code, data, and stack.
Each memory access uses a **logical address** (also called a segment:offset pair) that the CPU translates to a **linear address**. The translation uses **segment descriptors**—data structures stored in tables (GDT—Global Descriptor Table, LDT—Local Descriptor Table) that define:
- **Base address**: Where the segment starts in linear memory
- **Limit**: How large the segment is
- **Access rights**: Read/write/execute permissions, privilege level
In **real mode** (16-bit legacy), segmentation is simple: `linear_address = segment × 16 + offset`. In **protected mode**, the segment register holds a **selector** (an index into the descriptor table), and the CPU looks up the base and limit from the descriptor.
## Why You Need It Now
Your kernel needs to set up segmentation before it can safely run user programs. Without proper segmentation:
- Any code can overwrite any memory (including the kernel)
- No privilege separation between user and kernel mode
- No way to isolate processes from each other
Even if you later switch to paging for virtual memory, segmentation is always active on x86—you can't turn it off, only make it "invisible" by setting up flat segments (base=0, limit=4GB).
## Key Insight
**Segmentation is like a building with keyed rooms.** Each segment is a room. The segment descriptor is the door with its lock and signage—defining who can enter, what they can do inside, and how big the room is. The segment selector is your key card, telling the system which room you're trying to access.
```
; Setting up a flat data segment selector
; Selector format: [Index(13 bits)][TI(1 bit)][RPL(2 bits)]
mov ax, 0x10    ; Index 2, Table Indicator=0 (GDT), RPL=0 (kernel)
mov ds, ax      ; Load data segment register
mov es, ax
mov ss, ax
```
The segment registers (`CS`, `DS`, `SS`, `ES`, `FS`, `GS`) hold selectors, not addresses. The CPU hardware automatically performs the translation on every memory access.


> **🔑 Foundation: Real mode vs protected mode**
> 
> ## What It Is
x86 CPUs boot into **real mode**—a backwards-compatible 16-bit mode that mimics the original 8086 processor from 1978. This ensures any OS can boot, but it's severely limited.
**Real Mode**:
- 20-bit address space (1 MB maximum memory)
- No memory protection—any code can access any memory
- No privilege levels
- Segmentation: `address = segment × 16 + offset`
- 16-bit registers and operations
**Protected Mode**:
- 32-bit address space (4 GB memory)
- Memory protection via segmentation and paging
- Four privilege levels (Ring 0-3) for isolation
- 32-bit registers (EAX, EBX, etc.)
- Modern features: virtual memory, multitasking support
## Why You Need It Now
Your kernel starts in real mode, courtesy of the bootloader. But you can't build a modern OS there—you need to switch to protected mode to:
- Access more than 1 MB of memory
- Run user programs without them crashing the kernel
- Use paging for virtual memory
- Enable 32-bit operations and addressing
The transition requires careful choreography: set up the GDT, disable interrupts, set the PE (Protection Enable) bit in CR0, then far-jump to flush the pipeline and load the new code segment.
## Key Insight
**Real mode is the CPU's "boot room"—small, exposed, and meant to be left quickly.** Protected mode is the "main building" where real work happens. The switch is one-way: once you enter protected mode, you don't go back (without complex gymnastics).
```asm
; The critical switch sequence
cli                   ; Disable interrupts (can't handle them yet)
lgdt [gdt_descriptor] ; Load our GDT
mov eax, cr0
or eax, 1             ; Set PE bit (Protection Enable)
mov cr0, eax
jmp 0x08:protected_mode_entry  ; Far jump: load CS with code segment selector
[BITS 32]
protected_mode_entry:
    ; Now in 32-bit protected mode
    mov ax, 0x10      ; Load data segment selectors
    mov ds, ax
    mov es, ax
    mov ss, ax
    ; ... continue kernel initialization
```
The far jump after setting CR0 is essential—it flushes the CPU's pipeline and loads the new code segment selector into `CS`. Without it, you'd be running 32-bit code with a 16-bit segment context.

The GDT is the cornerstone of protected mode. It defines memory segments — not just their base addresses and sizes, but their **access rights** (readable, writable, executable, privilege level).

![GDT Entry Layout and Bit Fields](./diagrams/diag-gdt-layout.svg)

In protected mode, segment registers (CS, DS, SS, etc.) no longer hold segment base addresses shifted by 4. Instead, they hold **selectors** — indices into the GDT.
Create `boot/gdt.asm`:
```nasm
; boot/gdt.asm - Global Descriptor Table
; GDT layout:
; Index 0: Null descriptor (required)
; Index 1: Kernel code segment (selector 0x08)
; Index 2: Kernel data segment (selector 0x10)
; Index 3: User code segment (selector 0x18)
; Index 4: User data segment (selector 0x20)
gdt_start:
    ; Null descriptor (required by CPU)
    dq 0x0000000000000000
gdt_kernel_code:
    ; Base=0, Limit=0xFFFFF (4KB granularity = 4GB)
    ; Access: Present, Ring 0, Code, Executable, Readable
    ; Flags: 4KB granularity, 32-bit
    dq 0x00CF9A000000FFFF
gdt_kernel_data:
    ; Base=0, Limit=0xFFFFF (4KB granularity = 4GB)
    ; Access: Present, Ring 0, Data, Writable
    ; Flags: 4KB granularity, 32-bit
    dq 0x00CF92000000FFFF
gdt_user_code:
    ; Same as kernel code but Ring 3 (DPL=3)
    ; Access byte: 0xFA = Present(1) DPL(11) S(1) Type(1010)
    dq 0x00CFFA000000FFFF
gdt_user_data:
    ; Same as kernel data but Ring 3 (DPL=3)
    ; Access byte: 0xF2 = Present(1) DPL(11) S(1) Type(0010)
    dq 0x00CFF2000000FFFF
gdt_end:
; GDT descriptor (for LGDT instruction)
gdt_descriptor:
    dw gdt_end - gdt_start - 1  ; Limit (size - 1)
    dd gdt_start                ; Base address
; Selector constants
KERNEL_CODE_SEL equ 0x08
KERNEL_DATA_SEL equ 0x10
USER_CODE_SEL   equ 0x18
USER_DATA_SEL   equ 0x20
```
### Decoding the GDT Entry
Each GDT entry is 8 bytes with a complex bit layout:
```
Byte 7    Byte 6    Byte 5    Byte 4    Byte 3    Byte 2    Byte 1    Byte 0
+--------+--------+--------+--------+--------+--------+--------+--------+
|Base 31:24|G|D|L|A|Limit|P|DPL|S| Type | Base 23:16| Base 15:0 |Limit 15:0|
|         | | | | |19:16| |   | |      |          |           |          |
+--------+--------+--------+--------+--------+--------+--------+--------+
Where:
- Base: 32-bit segment base address (split across 3 fields)
- Limit: 20-bit segment limit (split across 2 fields)
- G: Granularity (0 = byte, 1 = 4KB pages)
- D/B: Default operation size (0 = 16-bit, 1 = 32-bit)
- L: Long mode (64-bit) - must be 0 for 32-bit
- AVL: Available for system use
- P: Present (must be 1 for valid segment)
- DPL: Descriptor Privilege Level (0-3)
- S: System (0 = system segment, 1 = code/data)
- Type: Segment type (code: 1010=exec+read, data: 0010=read+write)
```
For our "flat" memory model (base=0, limit=4GB):
- Base = 0x00000000
- Limit = 0xFFFFF with G=1 (4KB granularity) = 0xFFFFF × 4096 = 4GB
- Access byte for kernel code: 0x9A = 10011010b
  - P=1 (present)
  - DPL=00 (ring 0)
  - S=1 (code/data, not system)
  - Type=1010 (executable, readable, non-conforming)
- Access byte for kernel data: 0x92 = 10010010b
  - P=1, DPL=00, S=1
  - Type=0010 (writable, expand-up)
- Flags: 0xC = 1100b (G=1, D=1 for 32-bit, L=0, AVL=0)
## Step 5: The Protected Mode Transition

![Protected Mode Entry State Machine](./diagrams/diag-protected-mode-transition.svg)

This is the moment of truth. The sequence must be exact:
1. **Disable interrupts** (`cli`) — the IDT is invalid in protected mode
2. **Load the GDT** (`lgdt [gdt_descriptor]`)
3. **Set CR0.PE** (bit 0 of CR0) — this switches the CPU to protected mode
4. **Far jump** to flush the pipeline and load CS with a 32-bit selector
5. **Load all data segment registers** with the kernel data selector
6. **Set up the 32-bit stack**
The far jump is critical. When you set CR0.PE, the CPU is in a weird hybrid state: protected mode is enabled, but CS still has the real-mode value. The far jump (`jmp 0x08:label`) loads CS with selector 0x08 (kernel code) and flushes the instruction pipeline, ensuring all subsequent instructions are decoded as 32-bit.
Continuing in `boot/boot.asm`:
```nasm
; --- 32-bit Protected Mode Code ---
[BITS 32]
protected_mode_entry:
    ; Load all data segment registers with kernel data selector
    mov ax, KERNEL_DATA_SEL
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    ; Set up 32-bit stack (below 1MB to avoid kernel)
    mov esp, 0x90000
    ; Copy kernel from 0x10000 to 0x100000 if needed
    ; (For simplicity, we'll assume kernel was loaded directly)
    ; Jump to kernel entry point
    jmp 0x08:0x100000
    ; We should never reach here
    jmp $
```
## Step 6: The Kernel Entry Point
Now you're in protected mode, running 32-bit code. But you're still in assembly — C needs some setup before it can run.
The kernel entry point must:
1. **Zero the BSS section** — uninitialized global variables must be set to 0
2. **Clear the direction flag** (`cld`) — so string operations increment
3. **Call the C entry point** with proper stack alignment
Create `kernel/entry.asm`:
```nasm
; kernel/entry.asm - Kernel entry point (called from bootloader)
[BITS 32]
[GLOBAL _start]
[EXTERN kernel_main]     ; C entry point
[EXTERN __bss_start]     ; Defined in linker script
[EXTERN __bss_end]
section .text
_start:
    ; Disable interrupts during early init
    cli
    ; Clear direction flag (string ops increment)
    cld
    ; Zero the BSS section
    mov edi, __bss_start
    mov ecx, __bss_end
    sub ecx, edi
    xor eax, eax
    rep stosb
    ; Set up stack (if not already done)
    mov esp, 0x90000
    ; Call C kernel entry point
    call kernel_main
    ; If kernel_main returns, halt
    cli
.halt:
    hlt
    jmp .halt
```
## Step 7: The Linker Script — Placing Code at the Right Addresses

![Kernel Linker Script Section Layout](./diagrams/diag-linker-script.svg)

The linker script tells the linker where each section of your kernel should be placed in memory. This is critical: the bootloader jumps to 0x100000, so your `.text` section must start there.
Create `kernel/linker.ld`:
```ld
/* kernel/linker.ld - Linker script for the kernel */
ENTRY(_start)
SECTIONS
{
    /* Kernel starts at 1MB */
    . = 0x100000;
    /* Text section (code) */
    .text ALIGN(4K) :
    {
        *(.multiboot)      /* Multiboot header if using GRUB */
        *(.text)
    }
    /* Read-only data */
    .rodata ALIGN(4K) :
    {
        *(.rodata)
        *(.rodata.*)
    }
    /* Initialized data */
    .data ALIGN(4K) :
    {
        *(.data)
    }
    /* Uninitialized data (BSS) - will be zeroed by entry.asm */
    .bss ALIGN(4K) :
    {
        __bss_start = .;
        *(COMMON)
        *(.bss)
        __bss_end = .;
    }
    /* Discard unwanted sections */
    /DISCARD/ :
    {
        *(.comment)
        *(.eh_frame)
        *(.note.*)
    }
}
```
The `ALIGN(4K)` directive ensures each section starts on a 4KB boundary — important for later when you enable paging.
## Step 8: VGA Text Mode — Your First Output Device

![VGA Text Mode Buffer Structure](./diagrams/diag-vga-buffer.svg)

The VGA text mode buffer is memory-mapped at 0xB8000. Writing to this address displays characters on screen — no drivers, no syscalls, just memory writes.
The buffer is 80 columns × 25 rows = 2000 characters. Each character is 2 bytes:
- Byte 0: ASCII character code
- Byte 1: Attribute (foreground color, background color, blink)
```
Attribute byte:
Bit 7   : Blink (0 = no blink)
Bit 6-4 : Background color (0-7)
Bit 3   : Bright (0 = normal, 1 = bright foreground)
Bit 2-0 : Foreground color (0-7)
Colors:
0 = Black     4 = Red       8 = Dark Gray    12 = Light Red
1 = Blue      5 = Magenta   9 = Light Blue   13 = Light Magenta
2 = Green     6 = Brown     10 = Light Green 14 = Yellow
3 = Cyan      7 = Light Gray 11 = Light Cyan  15 = White
```
Create `kernel/vga.h`:
```c
/* kernel/vga.h - VGA text mode driver */
#ifndef VGA_H
#define VGA_H
#include <stdint.h>
/* VGA text mode buffer address */
#define VGA_BUFFER ((volatile uint16_t *)0xB8000)
/* Screen dimensions */
#define VGA_WIDTH  80
#define VGA_HEIGHT 25
/* VGA colors */
typedef enum {
    VGA_COLOR_BLACK         = 0,
    VGA_COLOR_BLUE          = 1,
    VGA_COLOR_GREEN         = 2,
    VGA_COLOR_CYAN          = 3,
    VGA_COLOR_RED           = 4,
    VGA_COLOR_MAGENTA       = 5,
    VGA_COLOR_BROWN         = 6,
    VGA_COLOR_LIGHT_GREY    = 7,
    VGA_COLOR_DARK_GREY     = 8,
    VGA_COLOR_LIGHT_BLUE    = 9,
    VGA_COLOR_LIGHT_GREEN   = 10,
    VGA_COLOR_LIGHT_CYAN    = 11,
    VGA_COLOR_LIGHT_RED     = 12,
    VGA_COLOR_LIGHT_MAGENTA = 13,
    VGA_COLOR_LIGHT_BROWN   = 14,
    VGA_COLOR_WHITE         = 15,
} vga_color;
/* Create a VGA entry from character and color */
static inline uint16_t vga_entry(unsigned char c, uint8_t color) {
    return (uint16_t)c | ((uint16_t)color << 8);
}
/* Create a color attribute from foreground and background */
static inline uint8_t vga_color_attr(vga_color fg, vga_color bg) {
    return (uint8_t)fg | ((uint8_t)bg << 4);
}
/* Initialize VGA driver */
void vga_init(void);
/* Clear screen */
void vga_clear(void);
/* Set cursor position */
void vga_set_cursor(int x, int y);
/* Put character at position */
void vga_put_char_at(char c, uint8_t color, int x, int y);
/* Put character (scrolling if needed) */
void vga_put_char(char c, uint8_t color);
/* Write string */
void vga_write(const char *str, uint8_t color);
#endif /* VGA_H */
```
Create `kernel/vga.c`:
```c
/* kernel/vga.c - VGA text mode driver implementation */
#include "vga.h"
/* Current cursor position */
static int cursor_x = 0;
static int cursor_y = 0;
/* Current color attribute */
static uint8_t current_color;
/* I/O port for VGA cursor control */
#define VGA_CTRL_REGISTER 0x3D4
#define VGA_DATA_REGISTER 0x3D5
#define VGA_CURSOR_HIGH   14
#define VGA_CURSOR_LOW    15
/* Output byte to I/O port */
static inline void outb(uint16_t port, uint8_t value) {
    __asm__ volatile ("outb %0, %1" : : "a"(value), "Nd"(port));
}
/* Input byte from I/O port */
static inline uint8_t inb(uint16_t port) {
    uint8_t value;
    __asm__ volatile ("inb %1, %0" : "=a"(value) : "Nd"(port));
    return value;
}
/* Update hardware cursor position */
static void update_cursor(void) {
    uint16_t pos = cursor_y * VGA_WIDTH + cursor_x;
    outb(VGA_CTRL_REGISTER, VGA_CURSOR_HIGH);
    outb(VGA_DATA_REGISTER, (uint8_t)(pos >> 8));
    outb(VGA_CTRL_REGISTER, VGA_CURSOR_LOW);
    outb(VGA_DATA_REGISTER, (uint8_t)(pos & 0xFF));
}
/* Scroll screen up by one line */
static void scroll(void) {
    /* Move all lines up */
    for (int y = 0; y < VGA_HEIGHT - 1; y++) {
        for (int x = 0; x < VGA_WIDTH; x++) {
            const int src_idx = (y + 1) * VGA_WIDTH + x;
            const int dst_idx = y * VGA_WIDTH + x;
            VGA_BUFFER[dst_idx] = VGA_BUFFER[src_idx];
        }
    }
    /* Clear bottom line */
    for (int x = 0; x < VGA_WIDTH; x++) {
        const int idx = (VGA_HEIGHT - 1) * VGA_WIDTH + x;
        VGA_BUFFER[idx] = vga_entry(' ', current_color);
    }
    cursor_y = VGA_HEIGHT - 1;
}
void vga_init(void) {
    current_color = vga_color_attr(VGA_COLOR_LIGHT_GREY, VGA_COLOR_BLACK);
    vga_clear();
}
void vga_clear(void) {
    for (int y = 0; y < VGA_HEIGHT; y++) {
        for (int x = 0; x < VGA_WIDTH; x++) {
            const int idx = y * VGA_WIDTH + x;
            VGA_BUFFER[idx] = vga_entry(' ', current_color);
        }
    }
    cursor_x = 0;
    cursor_y = 0;
    update_cursor();
}
void vga_set_cursor(int x, int y) {
    if (x >= 0 && x < VGA_WIDTH && y >= 0 && y < VGA_HEIGHT) {
        cursor_x = x;
        cursor_y = y;
        update_cursor();
    }
}
void vga_put_char_at(char c, uint8_t color, int x, int y) {
    if (x >= 0 && x < VGA_WIDTH && y >= 0 && y < VGA_HEIGHT) {
        const int idx = y * VGA_WIDTH + x;
        VGA_BUFFER[idx] = vga_entry(c, color);
    }
}
void vga_put_char(char c, uint8_t color) {
    /* Handle special characters */
    if (c == '\n') {
        cursor_x = 0;
        cursor_y++;
    } else if (c == '\r') {
        cursor_x = 0;
    } else if (c == '\t') {
        cursor_x = (cursor_x + 4) & ~3;  /* Align to 4 */
        if (cursor_x >= VGA_WIDTH) {
            cursor_x = 0;
            cursor_y++;
        }
    } else if (c == '\b') {
        if (cursor_x > 0) {
            cursor_x--;
            vga_put_char_at(' ', color, cursor_x, cursor_y);
        }
    } else {
        vga_put_char_at(c, color, cursor_x, cursor_y);
        cursor_x++;
        if (cursor_x >= VGA_WIDTH) {
            cursor_x = 0;
            cursor_y++;
        }
    }
    /* Scroll if needed */
    if (cursor_y >= VGA_HEIGHT) {
        scroll();
    }
    update_cursor();
}
void vga_write(const char *str, uint8_t color) {
    while (*str) {
        vga_put_char(*str++, color);
    }
}
```
## Step 9: Serial Port Debug Output

![Serial Port Initialization Sequence](./diagrams/diag-serial-init.svg)

VGA output is great for visible feedback, but for serious debugging you need serial output. QEMU can redirect serial output to a file or terminal, giving you a persistent log of kernel messages.
The serial port (COM1 at 0x3F8) is programmed through I/O ports:
Create `kernel/serial.h`:
```c
/* kernel/serial.h - Serial port driver */
#ifndef SERIAL_H
#define SERIAL_H
#include <stdint.h>
#include <stdbool.h>
/* COM1 base port */
#define SERIAL_COM1_BASE 0x3F8
/* Port offsets */
#define SERIAL_DATA         0   /* Data register (R/W) */
#define SERIAL_INT_ENABLE   1   /* Interrupt enable register */
#define SERIAL_FIFO_CTRL    2   /* FIFO control register */
#define SERIAL_LINE_CTRL    3   /* Line control register */
#define SERIAL_MODEM_CTRL   4   /* Modem control register */
#define SERIAL_LINE_STATUS  5   /* Line status register */
/* Initialize serial port */
bool serial_init(void);
/* Check if transmit buffer is empty */
bool serial_is_transmit_empty(void);
/* Write character to serial port */
void serial_put_char(char c);
/* Write string to serial port */
void serial_write(const char *str);
#endif /* SERIAL_H */
```
Create `kernel/serial.c`:
```c
/* kernel/serial.c - Serial port driver implementation */
#include "serial.h"
/* I/O port functions */
static inline void outb(uint16_t port, uint8_t value) {
    __asm__ volatile ("outb %0, %1" : : "a"(value), "Nd"(port));
}
static inline uint8_t inb(uint16_t port) {
    uint8_t value;
    __asm__ volatile ("inb %1, %0" : "=a"(value) : "Nd"(port));
    return value;
}
bool serial_init(void) {
    uint16_t base = SERIAL_COM1_BASE;
    /* Disable interrupts */
    outb(base + SERIAL_INT_ENABLE, 0x00);
    /* Enable DLAB (Divisor Latch Access Bit) to set baud rate */
    outb(base + SERIAL_LINE_CTRL, 0x80);
    /* Set divisor to 1 (115200 baud) */
    /* Divisor = 115200 / desired_baud */
    outb(base + SERIAL_DATA, 0x01);      /* Low byte */
    outb(base + SERIAL_INT_ENABLE, 0x00); /* High byte */
    /* 8 bits, no parity, 1 stop bit (8N1) */
    outb(base + SERIAL_LINE_CTRL, 0x03);
    /* Enable FIFO, clear buffers, 14-byte threshold */
    outb(base + SERIAL_FIFO_CTRL, 0xC7);
    /* IRQs enabled, RTS/DSR set */
    outb(base + SERIAL_MODEM_CTRL, 0x0B);
    /* Test the chip by enabling loopback mode */
    outb(base + SERIAL_MODEM_CTRL, 0x1E);
    /* Write a test byte */
    outb(base + SERIAL_DATA, 0xAE);
    /* Check if we get the same byte back */
    if (inb(base + SERIAL_DATA) != 0xAE) {
        return false;  /* Serial port not working */
    }
    /* Set to normal operation mode */
    outb(base + SERIAL_MODEM_CTRL, 0x0F);
    return true;
}
bool serial_is_transmit_empty(void) {
    return inb(SERIAL_COM1_BASE + SERIAL_LINE_STATUS) & 0x20;
}
void serial_put_char(char c) {
    /* Wait for transmit buffer to be empty */
    while (!serial_is_transmit_empty());
    outb(SERIAL_COM1_BASE + SERIAL_DATA, c);
}
void serial_write(const char *str) {
    while (*str) {
        serial_put_char(*str++);
    }
}
```
## Step 10: kprintf — Formatted Output
Now let's combine VGA and serial output into a single `kprintf` function:
Create `kernel/kprintf.h`:
```c
/* kernel/kprintf.h - Kernel printf */
#ifndef KPRINTF_H
#define KPRINTF_H
#include <stdint.h>
#include <stdarg.h>
/* Print formatted string to VGA and serial */
int kprintf(const char *format, ...);
/* Print formatted string with va_list */
int kvprintf(const char *format, va_list args);
#endif /* KPRINTF_H */
```
Create `kernel/kprintf.c`:
```c
/* kernel/kprintf.c - Kernel printf implementation */
#include "kprintf.h"
#include "vga.h"
#include "serial.h"
#include <stdarg.h>
/* Helper: Print a single character */
static void print_char(char c) {
    vga_put_char(c, vga_color_attr(VGA_COLOR_LIGHT_GREY, VGA_COLOR_BLACK));
    serial_put_char(c);
}
/* Helper: Print a string */
static void print_string(const char *str) {
    while (*str) {
        print_char(*str++);
    }
}
/* Helper: Print unsigned integer in given base */
static void print_uint(uint32_t value, int base, bool uppercase) {
    static const char digits_lower[] = "0123456789abcdef";
    static const char digits_upper[] = "0123456789ABCDEF";
    const char *digits = uppercase ? digits_upper : digits_lower;
    char buffer[32];
    int i = 0;
    if (value == 0) {
        print_char('0');
        return;
    }
    while (value > 0) {
        buffer[i++] = digits[value % base];
        value /= base;
    }
    /* Print in reverse order */
    while (i > 0) {
        print_char(buffer[--i]);
    }
}
/* Helper: Print signed integer */
static void print_int(int32_t value) {
    if (value < 0) {
        print_char('-');
        value = -value;
    }
    print_uint((uint32_t)value, 10, false);
}
/* Helper: Print pointer */
static void print_pointer(void *ptr) {
    print_string("0x");
    print_uint((uint32_t)ptr, 16, false);
}
int kvprintf(const char *format, va_list args) {
    int count = 0;
    while (*format) {
        if (*format == '%') {
            format++;
            switch (*format) {
                case 'c': {
                    char c = (char)va_arg(args, int);
                    print_char(c);
                    count++;
                    break;
                }
                case 's': {
                    const char *s = va_arg(args, const char *);
                    if (s == NULL) s = "(null)";
                    print_string(s);
                    count++;  /* Approximate */
                    break;
                }
                case 'd':
                case 'i': {
                    int32_t val = va_arg(args, int32_t);
                    print_int(val);
                    count++;
                    break;
                }
                case 'u': {
                    uint32_t val = va_arg(args, uint32_t);
                    print_uint(val, 10, false);
                    count++;
                    break;
                }
                case 'x': {
                    uint32_t val = va_arg(args, uint32_t);
                    print_uint(val, 16, false);
                    count++;
                    break;
                }
                case 'X': {
                    uint32_t val = va_arg(args, uint32_t);
                    print_uint(val, 16, true);
                    count++;
                    break;
                }
                case 'p': {
                    void *ptr = va_arg(args, void *);
                    print_pointer(ptr);
                    count++;
                    break;
                }
                case '%': {
                    print_char('%');
                    count++;
                    break;
                }
                case '\0': {
                    /* Premature end of format string */
                    return count;
                }
                default: {
                    /* Unknown specifier, print literally */
                    print_char('%');
                    print_char(*format);
                    count += 2;
                    break;
                }
            }
        } else {
            print_char(*format);
            count++;
        }
        format++;
    }
    return count;
}
int kprintf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    int result = kvprintf(format, args);
    va_end(args);
    return result;
}
```
You'll also need a minimal `stdarg.h` for freestanding C. Create `include/stdarg.h`:
```c
/* include/stdarg.h - Variable argument handling (freestanding) */
#ifndef STDARG_H
#define STDARG_H
/* Use compiler builtins for variable arguments */
typedef __builtin_va_list va_list;
#define va_start(ap, last) __builtin_va_start(ap, last)
#define va_arg(ap, type)   __builtin_va_arg(ap, type)
#define va_end(ap)         __builtin_va_end(ap)
#endif /* STDARG_H */
```
Similarly, create `include/stdint.h`:
```c
/* include/stdint.h - Integer types (freestanding) */
#ifndef STDINT_H
#define STDINT_H
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef signed short       int16_t;
typedef unsigned short     uint16_t;
typedef signed int         int32_t;
typedef unsigned int       uint32_t;
typedef signed long long   int64_t;
typedef unsigned long long uint64_t;
typedef uint32_t uintptr_t;
typedef int32_t  intptr_t;
#endif /* STDINT_H */
```
## Step 11: The C Kernel Main
Finally, the C entry point:
Create `kernel/main.c`:
```c
/* kernel/main.c - Kernel entry point */
#include "vga.h"
#include "serial.h"
#include "kprintf.h"
void kernel_main(void) {
    /* Initialize VGA */
    vga_init();
    /* Initialize serial port */
    if (!serial_init()) {
        vga_write("Serial init failed!\n", 
                  vga_color_attr(VGA_COLOR_RED, VGA_COLOR_BLACK));
    }
    /* Print welcome message */
    kprintf("\n");
    kprintf("========================================\n");
    kprintf("  MyOS Kernel v0.1\n");
    kprintf("  Built on %s at %s\n", __DATE__, __TIME__);
    kprintf("========================================\n");
    kprintf("\n");
    /* Print some test output */
    kprintf("Kernel loaded at: %p\n", (void *)0x100000);
    kprintf("Stack pointer:    %p\n", (void *)0x90000);
    kprintf("\n");
    /* Test various format specifiers */
    kprintf("Integer:  %d (0x%x)\n", 42, 42);
    kprintf("Negative: %d\n", -12345);
    kprintf("Pointer:  %p\n", (void *)0xDEADBEEF);
    kprintf("String:   %s\n", "Hello, World!");
    kprintf("Char:     %c\n", 'X');
    kprintf("\nKernel initialized successfully.\n");
    kprintf("System halted.\n");
    /* Halt */
    while (1) {
        __asm__ volatile ("hlt");
    }
}
```
## Step 12: The Build System
Create `Makefile`:
```makefile
# Makefile - Build system for MyOS kernel
AS = nasm
CC = gcc
LD = ld
# Compiler flags for freestanding kernel
CFLAGS = -ffreestanding \
         -fno-stack-protector \
         -fno-pic \
         -m32 \
         -Wall \
         -Wextra \
         -nostdlib \
         -nostdinc \
         -I include \
         -O2 \
         -g
# Assembler flags
ASFLAGS = -f elf32 -g -F dwarf
# Linker flags
LDFLAGS = -m elf_i386 \
          -nostdlib \
          -T kernel/linker.ld
# Source files
ASM_SOURCES = kernel/entry.asm
C_SOURCES = kernel/main.c kernel/vga.c kernel/serial.c kernel/kprintf.c
# Object files
ASM_OBJECTS = $(ASM_SOURCES:.asm=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)
OBJECTS = $(ASM_OBJECTS) $(C_OBJECTS)
# Output files
KERNEL_BIN = kernel.bin
BOOT_BIN = boot.bin
OS_IMAGE = os.img
.PHONY: all clean run
all: $(OS_IMAGE)
# Build bootloader
$(BOOT_BIN): boot/boot.asm boot/gdt.asm boot/a20.asm boot/load_kernel.asm
	$(AS) -f bin boot/boot.asm -o $(BOOT_BIN)
# Build kernel objects
%.o: %.asm
	$(AS) $(ASFLAGS) $< -o $@
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
# Link kernel
$(KERNEL_BIN): $(OBJECTS)
	$(LD) $(LDFLAGS) -o $@ $(OBJECTS)
# Create disk image
$(OS_IMAGE): $(BOOT_BIN) $(KERNEL_BIN)
	# Create empty 1.44MB floppy image
	dd if=/dev/zero of=$@ bs=512 count=2880
	# Write bootloader to sector 0
	dd if=$(BOOT_BIN) of=$@ bs=512 count=1 conv=notrunc
	# Write kernel starting at sector 2
	dd if=$(KERNEL_BIN) of=$@ bs=512 seek=2 conv=notrunc
# Run in QEMU
run: $(OS_IMAGE)
	qemu-system-i386 -fda $(OS_IMAGE) -serial stdio
# Run with GDB support
debug: $(OS_IMAGE)
	qemu-system-i386 -fda $(OS_IMAGE) -serial stdio -s -S &
	gdb -ex "target remote localhost:1234" -ex "break kernel_main" -ex "continue"
clean:
	rm -f $(BOOT_BIN) $(KERNEL_BIN) $(OS_IMAGE) $(OBJECTS) *.o kernel/*.o
```
## Running Your Kernel
```bash
# Build
make
# Run in QEMU
make run
```
If everything works, you'll see:
```
Booting MyOS...
Loading kernel...
========================================
  MyOS Kernel v0.1
  Built on Mar 18 2026 at 10:30:00
========================================
Kernel loaded at: 0x00100000
Stack pointer:    0x00090000
Integer:  42 (0x2a)
Negative: -12345
Pointer:  0xdeadbeef
String:   Hello, World!
Char:     X
Kernel initialized successfully.
System halted.
```
## Debugging Tips
### Triple Fault Debugging
A triple fault (three consecutive exceptions) causes an immediate CPU reset. Debug by:
1. **Serial output early** — add serial output before each major step
2. **QEMU `-d int`** — logs all interrupts to `/tmp/qemu.log`
3. **QEMU `-no-reboot`** — pauses on triple fault instead of resetting
```bash
qemu-system-i386 -fda os.img -serial stdio -d int -no-reboot
```
### Common Failure Points
| Symptom | Likely Cause |
|---------|--------------|
| Immediate reset | GDT not loaded before setting CR0.PE |
| Reset after `jmp 0x08:...` | GDT entry wrong (check access byte, flags) |
| Garbage on screen | VGA buffer address wrong or wrong selector loaded |
| No output at all | Serial not initialized or kernel not loaded |
### Hardware Soul Check
Every instruction in your bootloader touches specific hardware:
```
lgdt [gdt_descriptor]
├── Memory controller: reads 6 bytes from RAM
├── CPU: loads GDTR register
└── Cache: likely cache miss (cold path)
mov eax, cr0
or eax, 1
mov cr0, eax
├── CPU: reads CR0, modifies, writes back
├── Pipeline: serialization point
└── Cache: register operations, no memory
jmp 0x08:protected_mode_entry
├── CPU: loads CS with selector 0x08
├── MMU: translates selector → GDT entry → segment base
├── Pipeline: full flush, refetch from new CS:EIP
└── Branch predictor: misprediction (unavoidable)
```
## Knowledge Cascade
You've just built the foundation for everything that follows:
**Same Domain:**
- **Interrupt Descriptor Table (next milestone)** — uses the same selector mechanism as GDT. The IDT entries reference code segments by selector.
- **Page Tables (milestone 3)** — another level of address translation. Where segmentation divides memory into segments, paging divides it into 4KB frames.
- **Task State Segment (milestone 4)** — another system descriptor in the GDT. The TSS is indexed just like code/data segments.
**Cross-Domain:**
- **Embedded systems** — use the same linker script techniques for placing code at specific addresses (flash, RAM, bootloader regions).
- **WebAssembly no_std** — the same freestanding C patterns apply. No libc, no heap, you provide everything.
- **Firmware/UEFI development** — the same boot sequence concepts, just with more features and a different entry point.
**Historical Context:**
- The 512-byte MBR limit comes from the original IBM PC (1981). We still live with it.
- The A20 line is pure legacy — modern CPUs don't need it, but the gate remains for compatibility.
- GDT "flat" model (base=0, limit=4GB) is what all modern OSes use. Segmentation was essentially abandoned in favor of paging.
**What You Can Now Build:**
- A multiboot-compliant kernel that boots from GRUB
- A two-stage bootloader with more features
- A basic bootloader for ARM or RISC-V (same concepts, different details)
---
## Acceptance Criteria
<!-- END_MS -->


<!-- MS_ID: build-os-m2 -->
<!-- MS_ID: build-os-m2 -->
# Milestone 2: Interrupts, Exceptions, and Keyboard

![OS Kernel Architecture Map](./diagrams/diag-satellite-os.svg)

You have a kernel that boots. It prints messages. It initializes hardware. But it's deaf.
Right now, your kernel can only do what it explicitly codes, in the order it codes it. It can't respond to the keyboard. It can't measure time. It can't handle errors gracefully. The CPU could be screaming about a divide-by-zero and your kernel would never know.
This milestone changes everything. You're about to give your kernel the ability to **respond to events** — not by polling in a loop, but by having the CPU itself interrupt whatever it's doing and jump to your code.
## The Fundamental Tension: The CPU Doesn't Wait
Here's the brutal truth about hardware: it operates on its own timeline, not yours.
**The keyboard controller** is a separate chip. When you press a key, it generates a scancode and signals the CPU. If the CPU is busy computing something, that signal sits there. Miss it, and the keystroke is lost forever.
**The timer chip** ticks at a fixed frequency. Every tick is an opportunity to do something — switch tasks, update a counter, check for timeouts. But only if the CPU notices.
**The CPU itself** can encounter errors — divide by zero, accessing invalid memory, executing a privileged instruction in user mode. These are **exceptions**, and they need immediate handling. The CPU can't "continue anyway" — it needs code to tell it what to do.
The tension: **hardware events are asynchronous and urgent, but your kernel code is sequential.** How do you bridge this gap?
The answer is the **interrupt mechanism** — a hardware-supported way for the CPU to save its current state, jump to a handler function you provide, and then resume exactly where it left off.

![Interrupt Service Routine Flow](./diagrams/diag-isr-flow.svg)

## Three-Level View of Interrupts
### Level 1 — Application View
Programs run sequentially. They call functions, get results, continue. The illusion of "responsiveness" comes from the OS — your keyboard input appears instantly, but the program didn't poll for it. The OS delivered it.
### Level 2 — OS/Kernel View (What You're Building)
The kernel defines **interrupt handlers** — functions that run when specific events occur. The Interrupt Descriptor Table (IDT) maps interrupt numbers to handler addresses. When an interrupt fires:
1. CPU saves current state (registers, flags, instruction pointer)
2. CPU looks up handler in IDT
3. Handler runs
4. Handler executes `iret` to restore state and resume
### Level 3 — Hardware View
- **PIC (8259 Programmable Interrupt Controller)**: Routes hardware signals (IRQs) to the CPU. There are two PICs (master and slave), each handling 8 IRQ lines.
- **CPU Interrupt Logic**: When an IRQ arrives, the CPU checks the IF (Interrupt Flag) in EFLAGS. If set, it acknowledges the interrupt and begins the handler invocation sequence.
- **Stack Operations**: The CPU pushes EFLAGS, CS, EIP (and optionally an error code) onto the stack, then loads the handler address from the IDT.
## The Revelation: Interrupts Are Not "Events"
Here's the misconception that trips up everyone:
> "Interrupts are like events in JavaScript — something happens, and my callback runs."
**No.** An interrupt is the CPU **violently** stopping what it's doing and jumping to your code. There is no event loop. There is no queue (initially). There is just:
1. The CPU pushes EFLAGS, CS, EIP to the stack
2. The CPU loads your handler's address
3. The CPU jumps
Your handler is now running. The previous code is frozen mid-instruction. And here's the critical part: **your handler must perfectly restore the CPU state before returning**, or the interrupted code will crash in bizarre ways.
Miss a register? Corrupted state. Forget to pop the error code? `iret` returns to garbage. Forget to send EOI to the PIC? **All future interrupts are blocked.**
This is not abstraction. This is hardware contract. Violate it, and the CPU doesn't throw an exception — it just does something wrong, silently.
## The Interrupt Descriptor Table (IDT)

![IDT Entry Structure and Gate Types](./diagrams/diag-idt-structure.svg)

The IDT is an array of 256 entries, each 8 bytes, describing how to handle a specific interrupt number. When interrupt N occurs, the CPU:
1. Looks up entry N in the IDT
2. Validates privileges (if the interrupt came from user mode)
3. Pushes state to the stack
4. Loads CS:EIP from the IDT entry
5. Begins executing your handler
### IDT Entry Structure
Each IDT entry (called a "gate descriptor") has this layout:
```
Bytes 7-6   Bytes 5-4   Bytes 3-2   Bytes 1-0
+---------+---------+---------+---------+
|Offset 31:16| P|DPL|0|Type|Reserved| |  Segment  |Offset 15:0|
|         | |   | |    |        | | Selector  |           |
+---------+---------+---------+---------+
- Offset (32-bit, split): Handler function address
- Segment Selector: Code segment to use (0x08 for kernel)
- P (Present): Must be 1 for valid entry
- DPL (Descriptor Privilege Level): Who can trigger this
  - 00 = Ring 0 only (kernel)
  - 11 = Any ring (including user mode for int n)
- Type:
  - 0xE = 32-bit Interrupt Gate (interrupts disabled during handler)
  - 0xF = 32-bit Trap Gate (interrupts remain enabled)
```
For now, we'll use **interrupt gates** (type 0xE) with DPL=0 for hardware interrupts and exceptions. This ensures interrupts are disabled while our handler runs, preventing reentrancy issues.
### Setting Up the IDT
Create `kernel/idt.h`:
```c
/* kernel/idt.h - Interrupt Descriptor Table */
#ifndef IDT_H
#define IDT_H
#include <stdint.h>
/* Number of IDT entries */
#define IDT_ENTRIES 256
/* IDT gate types */
#define IDT_TYPE_INTERRUPT 0x8E  /* P=1, DPL=00, Type=0xE (32-bit interrupt gate) */
#define IDT_TYPE_TRAP      0x8F  /* P=1, DPL=00, Type=0xF (32-bit trap gate) */
#define IDT_TYPE_USER_INT  0xEE  /* P=1, DPL=11, Type=0xE (user-callable interrupt) */
/* IDT entry structure (8 bytes) */
typedef struct {
    uint16_t offset_low;    /* Offset bits 0-15 */
    uint16_t selector;      /* Code segment selector */
    uint8_t  zero;          /* Reserved, must be 0 */
    uint8_t  type_attr;     /* P, DPL, Type */
    uint16_t offset_high;   /* Offset bits 16-31 */
} __attribute__((packed)) idt_entry_t;
/* IDT pointer structure (for lidt instruction) */
typedef struct {
    uint16_t limit;         /* Size of IDT - 1 */
    uint32_t base;          /* Address of IDT */
} __attribute__((packed)) idt_ptr_t;
/* Initialize the IDT */
void idt_init(void);
/* Set an IDT gate */
void idt_set_gate(uint8_t num, uint32_t handler, uint16_t selector, uint8_t type);
/* Load the IDT (assembly) */
extern void idt_load(uint32_t idt_ptr);
/* Common interrupt handler (assembly stubs call this) */
typedef void (*interrupt_handler_t)(void);
#endif /* IDT_H */
```
Create `kernel/idt.c`:
```c
/* kernel/idt.c - Interrupt Descriptor Table implementation */
#include "idt.h"
#include "kprintf.h"
/* The IDT itself - 256 entries */
static idt_entry_t idt[IDT_ENTRIES];
/* IDT pointer for lidt */
static idt_ptr_t idt_ptr;
/* External assembly stubs (defined in idt_stubs.asm) */
extern interrupt_handler_t interrupt_stubs[256];
/* Set an IDT gate entry */
void idt_set_gate(uint8_t num, uint32_t handler, uint16_t selector, uint8_t type) {
    idt[num].offset_low  = handler & 0xFFFF;
    idt[num].offset_high = (handler >> 16) & 0xFFFF;
    idt[num].selector    = selector;
    idt[num].zero        = 0;
    idt[num].type_attr   = type;
}
/* Initialize the IDT */
void idt_init(void) {
    /* Zero the IDT */
    for (int i = 0; i < IDT_ENTRIES; i++) {
        idt[i].offset_low  = 0;
        idt[i].offset_high = 0;
        idt[i].selector    = 0;
        idt[i].zero        = 0;
    }
    /* Set up exception handlers (vectors 0-31) */
    /* These are CPU-defined and must be handled */
    idt_set_gate(0,  (uint32_t)interrupt_stubs[0],  0x08, IDT_TYPE_INTERRUPT);  /* Divide Error */
    idt_set_gate(1,  (uint32_t)interrupt_stubs[1],  0x08, IDT_TYPE_INTERRUPT);  /* Debug */
    idt_set_gate(2,  (uint32_t)interrupt_stubs[2],  0x08, IDT_TYPE_INTERRUPT);  /* NMI */
    idt_set_gate(3,  (uint32_t)interrupt_stubs[3],  0x08, IDT_TYPE_INTERRUPT);  /* Breakpoint */
    idt_set_gate(4,  (uint32_t)interrupt_stubs[4],  0x08, IDT_TYPE_INTERRUPT);  /* Overflow */
    idt_set_gate(5,  (uint32_t)interrupt_stubs[5],  0x08, IDT_TYPE_INTERRUPT);  /* BOUND Range */
    idt_set_gate(6,  (uint32_t)interrupt_stubs[6],  0x08, IDT_TYPE_INTERRUPT);  /* Invalid Opcode */
    idt_set_gate(7,  (uint32_t)interrupt_stubs[7],  0x08, IDT_TYPE_INTERRUPT);  /* Device Not Available */
    idt_set_gate(8,  (uint32_t)interrupt_stubs[8],  0x08, IDT_TYPE_INTERRUPT);  /* Double Fault (has error code) */
    idt_set_gate(9,  (uint32_t)interrupt_stubs[9],  0x08, IDT_TYPE_INTERRUPT);  /* Coprocessor Segment Overrun */
    idt_set_gate(10, (uint32_t)interrupt_stubs[10], 0x08, IDT_TYPE_INTERRUPT);  /* Invalid TSS (has error code) */
    idt_set_gate(11, (uint32_t)interrupt_stubs[11], 0x08, IDT_TYPE_INTERRUPT);  /* Segment Not Present (has error code) */
    idt_set_gate(12, (uint32_t)interrupt_stubs[12], 0x08, IDT_TYPE_INTERRUPT);  /* Stack-Segment Fault (has error code) */
    idt_set_gate(13, (uint32_t)interrupt_stubs[13], 0x08, IDT_TYPE_INTERRUPT);  /* General Protection (has error code) */
    idt_set_gate(14, (uint32_t)interrupt_stubs[14], 0x08, IDT_TYPE_INTERRUPT);  /* Page Fault (has error code) */
    idt_set_gate(16, (uint32_t)interrupt_stubs[16], 0x08, IDT_TYPE_INTERRUPT);  /* x87 FPU Error */
    idt_set_gate(17, (uint32_t)interrupt_stubs[17], 0x08, IDT_TYPE_INTERRUPT);  /* Alignment Check (has error code) */
    idt_set_gate(18, (uint32_t)interrupt_stubs[18], 0x08, IDT_TYPE_INTERRUPT);  /* Machine Check */
    idt_set_gate(19, (uint32_t)interrupt_stubs[19], 0x08, IDT_TYPE_INTERRUPT);  /* SIMD Exception */
    /* Set up IRQ handlers (vectors 32-47) */
    /* IRQ0-7  -> vectors 32-39 */
    /* IRQ8-15 -> vectors 40-47 */
    for (int i = 32; i < 48; i++) {
        idt_set_gate(i, (uint32_t)interrupt_stubs[i], 0x08, IDT_TYPE_INTERRUPT);
    }
    /* Set up the IDT pointer */
    idt_ptr.limit = sizeof(idt_entry_t) * IDT_ENTRIES - 1;
    idt_ptr.base  = (uint32_t)&idt;
    /* Load the IDT */
    idt_load((uint32_t)&idt_ptr);
    kprintf("IDT initialized with %d entries\n", IDT_ENTRIES);
}
```
## The Interrupt Stack Frame

![Interrupt Stack Frame Layout](./diagrams/diag-interrupt-stack-frame.svg)

When an interrupt occurs, the CPU pushes a specific stack frame. Your handler must understand this layout to restore state correctly with `iret`.
```
High addresses
+------------------+
| SS               |  (only if privilege change: user -> kernel)
| ESP              |  (only if privilege change)
+------------------+
| EFLAGS           |
+------------------+
| CS               |
+------------------+
| EIP (return addr)|
+------------------+
| Error Code       |  (only for exceptions 8, 10-14)
+------------------+
| ... handler's stack ...
Low addresses
```
**Critical details:**
- **Error code**: Only exceptions 8, 10, 11, 12, 13, 14, 17 push an error code. All others don't. Your handler must know which type it is.
- **Privilege change**: If the interrupt transitions from Ring 3 to Ring 0, the CPU also pushes SS and ESP (the user stack). For kernel-only interrupts, these aren't pushed.
- **`iret`**: This instruction pops EIP, CS, EFLAGS (and SS:ESP if privilege changed). If there's an error code on the stack, **you must pop it first**, or `iret` will try to return to garbage.
## Assembly Stubs: Bridging C and Hardware
The CPU jumps to your handler address, but C functions don't automatically save/restore all registers. We need assembly stubs that:
1. Save all general-purpose registers
2. Call the C handler
3. Restore registers
4. Execute `iret`
Create `kernel/idt_stubs.asm`:
```nasm
; kernel/idt_stubs.asm - Assembly stubs for interrupt handlers
[BITS 32]
; External C handler
[EXTERN interrupt_handler]
; Common interrupt stub (no error code pushed by CPU)
%macro ISR_NOERRCODE 1
[GLOBAL isr%1]
isr%1:
    cli
    push byte 0          ; Push dummy error code to unify stack frame
    push byte %1         ; Push interrupt number
    jmp isr_common_stub
%endmacro
; Common interrupt stub (error code pushed by CPU)
%macro ISR_ERRCODE 1
[GLOBAL isr%1]
isr%1:
    cli
    push byte %1         ; Push interrupt number (error code already on stack)
    jmp isr_common_stub
%endmacro
; Common IRQ stub
%macro IRQ 2
[GLOBAL irq%1]
irq%1:
    cli
    push byte 0          ; Dummy error code
    push byte %2         ; Mapped interrupt number
    jmp irq_common_stub
%endmacro
; Exception handlers (0-31)
ISR_NOERRCODE 0    ; Divide Error
ISR_NOERRCODE 1    ; Debug
ISR_NOERRCODE 2    ; NMI
ISR_NOERRCODE 3    ; Breakpoint
ISR_NOERRCODE 4    ; Overflow
ISR_NOERRCODE 5    ; BOUND Range Exceeded
ISR_NOERRCODE 6    ; Invalid Opcode
ISR_NOERRCODE 7    ; Device Not Available
ISR_ERRCODE   8    ; Double Fault (has error code!)
ISR_NOERRCODE 9    ; Coprocessor Segment Overrun
ISR_ERRCODE   10   ; Invalid TSS
ISR_ERRCODE   11   ; Segment Not Present
ISR_ERRCODE   12   ; Stack-Segment Fault
ISR_ERRCODE   13   ; General Protection Fault
ISR_ERRCODE   14   ; Page Fault
ISR_NOERRCODE 15   ; Reserved
ISR_NOERRCODE 16   ; x87 FPU Error
ISR_ERRCODE   17   ; Alignment Check
ISR_NOERRCODE 18   ; Machine Check
ISR_NOERRCODE 19   ; SIMD Floating-Point Exception
ISR_NOERRCODE 20   ; Virtualization Exception
ISR_NOERRCODE 21   ; Control Protection Exception
; 22-31 reserved
; IRQ handlers (32-47)
IRQ 0,  32    ; Timer (IRQ0)
IRQ 1,  33    ; Keyboard (IRQ1)
IRQ 2,  34    ; Cascade (IRQ2)
IRQ 3,  35    ; COM2 (IRQ3)
IRQ 4,  36    ; COM1 (IRQ4)
IRQ 5,  37    ; LPT2 (IRQ5)
IRQ 6,  38    ; Floppy (IRQ6)
IRQ 7,  39    ; LPT1 (IRQ7)
IRQ 8,  40    ; RTC (IRQ8)
IRQ 9,  41    ; Free (IRQ9)
IRQ 10, 42    ; Free (IRQ10)
IRQ 11, 43    ; Free (IRQ11)
IRQ 12, 44    ; PS/2 Mouse (IRQ12)
IRQ 13, 45    ; FPU (IRQ13)
IRQ 14, 46    ; Primary ATA (IRQ14)
IRQ 15, 47    ; Secondary ATA (IRQ15)
; Export stub array for C code
[GLOBAL interrupt_stubs]
interrupt_stubs:
%assign i 0
%rep 256
    dd isr%+i
%assign i i+1
%endrep
; Common ISR stub - saves state, calls C handler, restores state
[GLOBAL isr_common_stub]
isr_common_stub:
    ; Save all general-purpose registers
    pusha               ; Pushes EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI
    ; Save segment registers
    mov ax, ds
    push ax
    mov ax, es
    push ax
    mov ax, fs
    push ax
    mov ax, gs
    push ax
    ; Load kernel data segment
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    ; Push stack pointer as argument
    push esp
    ; Call C handler
    call interrupt_handler
    ; Clean up argument
    add esp, 4
    ; Restore segment registers
    pop gs
    pop fs
    pop es
    pop ds
    ; Restore general-purpose registers
    popa
    ; Remove error code and interrupt number
    add esp, 8
    ; Return from interrupt
    iret
; Common IRQ stub - same as ISR but sends EOI
[GLOBAL irq_common_stub]
irq_common_stub:
    pusha
    mov ax, ds
    push ax
    mov ax, es
    push ax
    mov ax, fs
    push ax
    mov ax, gs
    push ax
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    push esp
    call interrupt_handler
    add esp, 4
    ; Send EOI to PIC (done in C handler for flexibility)
    pop gs
    pop fs
    pop es
    pop ds
    popa
    add esp, 8
    iret
; Load IDT
[GLOBAL idt_load]
idt_load:
    lidt [eax]          ; Load IDT pointer
    ret
```
Wait — there's a problem. The `%rep 256` macro assumes we have `isr0` through `isr255`, but we only defined exceptions 0-21 and IRQs 32-47. We need stubs for all 256 entries. Let's fix this:
```nasm
; Reserved/unused interrupt handlers (just halt for now)
%macro ISR_UNUSED 1
[GLOBAL isr%1]
isr%1:
    cli
    push byte 0
    push byte %1
    jmp isr_common_stub
%endmacro
; Fill unused vectors (22-31, 48-255)
ISR_UNUSED 22
ISR_UNUSED 23
ISR_UNUSED 24
ISR_UNUSED 25
ISR_UNUSED 26
ISR_UNUSED 27
ISR_UNUSED 28
ISR_UNUSED 29
ISR_UNUSED 30
ISR_UNUSED 31
; Vectors 48-255 (software interrupts, unused for now)
%assign i 48
%rep 208
ISR_UNUSED i
%assign i i+1
%endrep
```
## The C Interrupt Handler
Now let's write the C function that receives all interrupts:
Create `kernel/interrupts.h`:
```c
/* kernel/interrupts.h - Interrupt handling */
#ifndef INTERRUPTS_H
#define INTERRUPTS_H
#include <stdint.h>
/* Registers structure pushed by assembly stub */
typedef struct {
    uint32_t gs, fs, es, ds;                          /* Segment registers */
    uint32_t edi, esi, ebp, esp, ebx, edx, ecx, eax;  /* General purpose */
    uint32_t int_no, err_code;                        /* Interrupt number, error code */
    uint32_t eip, cs, eflags;                         /* Pushed by CPU */
    uint32_t useresp, ss;                             /* Only if privilege change */
} __attribute__((packed)) registers_t;
/* Register an interrupt handler */
typedef void (*isr_t)(registers_t *);
void register_interrupt_handler(uint8_t n, isr_t handler);
/* Initialize interrupt handling */
void interrupts_init(void);
#endif /* INTERRUPTS_H */
```
Create `kernel/interrupts.c`:
```c
/* kernel/interrupts.c - Interrupt handling implementation */
#include "interrupts.h"
#include "idt.h"
#include "kprintf.h"
#include "vga.h"
/* Array of custom interrupt handlers */
static isr_t interrupt_handlers[256] = {0};
/* Register a custom handler for an interrupt */
void register_interrupt_handler(uint8_t n, isr_t handler) {
    interrupt_handlers[n] = handler;
}
/* Called from assembly stub - dispatches to appropriate handler */
void interrupt_handler(registers_t *regs) {
    /* Check if we have a custom handler */
    if (interrupt_handlers[regs->int_no] != 0) {
        isr_t handler = interrupt_handlers[regs->int_no];
        handler(regs);
    } else {
        /* No handler - check if it's an exception or IRQ */
        if (regs->int_no < 32) {
            /* CPU exception - this is bad! */
            handle_exception(regs);
        } else if (regs->int_no >= 32 && regs->int_no < 48) {
            /* Spurious IRQ? */
            kprintf("Unhandled IRQ %d\n", regs->int_no - 32);
        }
    }
}
/* Exception names for debugging */
static const char *exception_messages[] = {
    "Division By Zero",
    "Debug",
    "Non Maskable Interrupt",
    "Breakpoint",
    "Into Detected Overflow",
    "Out of Bounds",
    "Invalid Opcode",
    "No Coprocessor",
    "Double Fault",
    "Coprocessor Segment Overrun",
    "Bad TSS",
    "Segment Not Present",
    "Stack Fault",
    "General Protection Fault",
    "Page Fault",
    "Unknown Interrupt",
    "Coprocessor Fault",
    "Alignment Check",
    "Machine Check",
    "Reserved",
    "Reserved"
};
/* Handle CPU exceptions */
void handle_exception(registers_t *regs) {
    uint8_t int_no = regs->int_no;
    /* Change color to red for error messages */
    vga_write("\n!!! EXCEPTION: ", vga_color_attr(VGA_COLOR_WHITE, VGA_COLOR_RED));
    vga_write(exception_messages[int_no], vga_color_attr(VGA_COLOR_WHITE, VGA_COLOR_RED));
    vga_write(" !!!\n", vga_color_attr(VGA_COLOR_WHITE, VGA_COLOR_RED));
    kprintf("Interrupt: %d\n", int_no);
    kprintf("Error Code: 0x%x\n", regs->err_code);
    kprintf("EIP: 0x%x  CS: 0x%x  EFLAGS: 0x%x\n", 
            regs->eip, regs->cs, regs->eflags);
    kprintf("EAX: 0x%x  EBX: 0x%x  ECX: 0x%x  EDX: 0x%x\n",
            regs->eax, regs->ebx, regs->ecx, regs->edx);
    kprintf("ESI: 0x%x  EDI: 0x%x  EBP: 0x%x  ESP: 0x%x\n",
            regs->esi, regs->edi, regs->ebp, regs->esp);
    /* For page fault, print the faulting address */
    if (int_no == 14) {
        uint32_t faulting_address;
        __asm__ volatile ("mov %%cr2, %0" : "=r"(faulting_address));
        kprintf("Page Fault Address: 0x%x\n", faulting_address);
        /* Decode error code */
        kprintf("  Present: %d  Write: %d  User: %d\n",
                regs->err_code & 0x1,
                (regs->err_code >> 1) & 0x1,
                (regs->err_code >> 2) & 0x1);
    }
    /* Double fault handler - halt instead of triple faulting */
    if (int_no == 8) {
        kprintf("\nDOUBLE FAULT - System halted to prevent triple fault\n");
        kprintf("This usually indicates a kernel stack overflow or\n");
        kprintf("an exception occurred while handling another exception.\n");
        while (1) {
            __asm__ volatile ("cli; hlt");
        }
    }
    /* For other exceptions, halt */
    kprintf("\nSystem halted.\n");
    while (1) {
        __asm__ volatile ("cli; hlt");
    }
}
```
Add the declaration to `interrupts.h`:
```c
/* Handle CPU exceptions (internal) */
void handle_exception(registers_t *regs);
```
## The Programmable Interrupt Controller (PIC)

![PIC 8259 Remapping](./diagrams/diag-pic-remap.svg)

The 8259 PIC is a legacy chip that routes hardware interrupts to the CPU. There are two PICs chained together:
- **Master PIC**: Handles IRQ0-IRQ7
- **Slave PIC**: Handles IRQ8-IRQ15, cascaded through IRQ2 of master
By default, the PIC maps IRQ0-7 to CPU vectors 0x08-0x0F and IRQ8-15 to 0x70-0x77. **This is a problem**: vectors 0-31 are reserved for CPU exceptions. IRQ0 (timer) would map to vector 8, which is the double fault exception!
We must **remap** the PIC to use vectors 32-47 instead.
### PIC Programming
The PIC is programmed through I/O ports:
- **Master PIC**: Command at 0x20, Data at 0x21
- **Slave PIC**: Command at 0xA0, Data at 0xA1
The initialization sequence uses ICW1-ICW4 (Initialization Command Words):
```
ICW1 (0x20/0xA0): Start initialization
  - Bit 0 (IC4): 1 = need ICW4
  - Bit 1 (SNGL): 0 = cascade mode
  - Bit 4: 1 = initialize
ICW2 (0x21/0xA1): Vector offset
  - Master: 0x20 (IRQ0 → vector 32)
  - Slave: 0x28 (IRQ8 → vector 40)
ICW3 (0x21/0xA1): Cascade configuration
  - Master: 0x04 (IRQ2 has slave)
  - Slave: 0x02 (slave ID is 2)
ICW4 (0x21/0xA1): Mode
  - 0x01 = 8086 mode, non-auto EOI
```
Create `kernel/pic.h`:
```c
/* kernel/pic.h - Programmable Interrupt Controller */
#ifndef PIC_H
#define PIC_H
#include <stdint.h>
/* PIC I/O ports */
#define PIC1_CMD  0x20    /* Master PIC command */
#define PIC1_DATA 0x21    /* Master PIC data */
#define PIC2_CMD  0xA0    /* Slave PIC command */
#define PIC2_DATA 0xA1    /* Slave PIC data */
/* PIC commands */
#define PIC_EOI   0x20    /* End of Interrupt */
#define PIC_READ_IRR 0x0A /* Read Interrupt Request Register */
#define PIC_READ_ISR 0x0B /* Read In-Service Register */
/* Initialize and remap the PIC */
void pic_init(void);
/* Send End of Interrupt */
void pic_send_eoi(uint8_t irq);
/* Mask (disable) an IRQ */
void pic_mask_irq(uint8_t irq);
/* Unmask (enable) an IRQ */
void pic_unmask_irq(uint8_t irq);
/* Get combined ISR and IRR */
uint16_t pic_get_isr(void);
uint16_t pic_get_irr(void);
#endif /* PIC_H */
```
Create `kernel/pic.c`:
```c
/* kernel/pic.c - PIC implementation */
#include "pic.h"
/* I/O port functions */
static inline void outb(uint16_t port, uint8_t value) {
    __asm__ volatile ("outb %0, %1" : : "a"(value), "Nd"(port));
}
static inline uint8_t inb(uint16_t port) {
    uint8_t value;
    __asm__ volatile ("inb %1, %0" : "=a"(value) : "Nd"(port));
    return value;
}
/* Initialize and remap the PIC */
void pic_init(void) {
    /* Save current masks */
    uint8_t mask1 = inb(PIC1_DATA);
    uint8_t mask2 = inb(PIC2_DATA);
    /* Start initialization sequence (ICW1) */
    outb(PIC1_CMD, 0x11);  /* ICW1: Initialize + need ICW4 */
    outb(PIC2_CMD, 0x11);
    /* ICW2: Set vector offsets */
    outb(PIC1_DATA, 0x20); /* Master: IRQ0-7 → vectors 32-39 */
    outb(PIC2_DATA, 0x28); /* Slave: IRQ8-15 → vectors 40-47 */
    /* ICW3: Tell Master there's a slave on IRQ2 */
    outb(PIC1_DATA, 0x04); /* Bit 2 set = IRQ2 has slave */
    outb(PIC2_DATA, 0x02); /* Slave ID = 2 */
    /* ICW4: 8086 mode, non-auto EOI */
    outb(PIC1_DATA, 0x01);
    outb(PIC2_DATA, 0x01);
    /* Restore masks (all IRQs disabled initially) */
    outb(PIC1_DATA, mask1);
    outb(PIC2_DATA, mask2);
}
/* Send End of Interrupt to the PIC */
void pic_send_eoi(uint8_t irq) {
    /* If IRQ came from slave (IRQ8-15), send EOI to both */
    if (irq >= 8) {
        outb(PIC2_CMD, PIC_EOI);
    }
    /* Always send EOI to master */
    outb(PIC1_CMD, PIC_EOI);
}
/* Mask (disable) an IRQ line */
void pic_mask_irq(uint8_t irq) {
    uint16_t port;
    uint8_t value;
    if (irq < 8) {
        port = PIC1_DATA;
    } else {
        port = PIC2_DATA;
        irq -= 8;
    }
    value = inb(port) | (1 << irq);
    outb(port, value);
}
/* Unmask (enable) an IRQ line */
void pic_unmask_irq(uint8_t irq) {
    uint16_t port;
    uint8_t value;
    if (irq < 8) {
        port = PIC1_DATA;
    } else {
        port = PIC2_DATA;
        irq -= 8;
    }
    value = inb(port) & ~(1 << irq);
    outb(port, value);
}
/* Get the combined In-Service Register */
uint16_t pic_get_isr(void) {
    outb(PIC1_CMD, PIC_READ_ISR);
    outb(PIC2_CMD, PIC_READ_ISR);
    return (inb(PIC2_CMD) << 8) | inb(PIC1_CMD);
}
/* Get the combined Interrupt Request Register */
uint16_t pic_get_irr(void) {
    outb(PIC1_CMD, PIC_READ_IRR);
    outb(PIC2_CMD, PIC_READ_IRR);
    return (inb(PIC2_CMD) << 8) | inb(PIC1_CMD);
}
```

![End of Interrupt (EOI) Decision Tree](./diagrams/diag-eoi-flow.svg)

### The EOI Contract
Here's the critical mistake everyone makes: forgetting to send EOI.
When an IRQ fires, the PIC marks that IRQ as "in service." It won't deliver another interrupt of the same or lower priority until you send EOI (End of Interrupt). **Forget EOI, and the system appears to hang** — no timer ticks, no keyboard input, nothing.
Rules:
- **IRQ0-7 (master only)**: Send EOI to master (port 0x20)
- **IRQ8-15 (slave)**: Send EOI to slave (0xA0) AND master (0x20)
The slave's IRQ output is connected to the master's IRQ2. When you handle an IRQ from the slave, you must clear both.
## The Timer: Measuring Time

![PIT Channel 0 Timer Configuration](./diagrams/diag-pit-timer.svg)

The Programmable Interval Timer (PIT, 8253/8254) is the oldest PC timer, but still essential. Channel 0 is connected to IRQ0, so we can get periodic interrupts.
The PIT's base frequency is 1,193,182 Hz (≈1.193 MHz). We program it with a divisor to get our desired frequency:
```
output_frequency = 1193182 / divisor
divisor = 1193182 / desired_frequency
For 100 Hz: divisor = 1193182 / 100 = 11931 (0x2E9B)
For 1000 Hz: divisor = 1193182 / 1000 = 1193 (0x4A9)
```
Create `kernel/timer.h`:
```c
/* kernel/timer.h - PIT timer driver */
#ifndef TIMER_H
#define TIMER_H
#include <stdint.h>
/* PIT I/O ports */
#define PIT_CHANNEL0 0x40
#define PIT_CHANNEL1 0x41
#define PIT_CHANNEL2 0x42
#define PIT_CMD      0x43
/* PIT frequency */
#define PIT_BASE_FREQ 1193182
/* Initialize the timer at given frequency (Hz) */
void timer_init(uint32_t freq);
/* Get current tick count */
uint64_t timer_get_ticks(void);
/* Get seconds since boot */
uint64_t timer_get_seconds(void);
/* Sleep for given milliseconds (busy-wait) */
void timer_sleep_ms(uint32_t ms);
#endif /* TIMER_H */
```
Create `kernel/timer.c`:
```c
/* kernel/timer.c - PIT timer implementation */
#include "timer.h"
#include "pic.h"
#include "interrupts.h"
#include "kprintf.h"
/* Global tick counter - 64-bit to avoid overflow */
static volatile uint64_t tick_count = 0;
/* Timer interrupt handler */
static void timer_handler(registers_t *regs) {
    (void)regs;  /* Unused */
    tick_count++;
}
/* Initialize the PIT timer */
void timer_init(uint32_t freq) {
    /* Register our timer handler for IRQ0 (vector 32) */
    register_interrupt_handler(32, timer_handler);
    /* Calculate divisor */
    uint32_t divisor = PIT_BASE_FREQ / freq;
    /* Ensure divisor fits in 16 bits */
    if (divisor > 65535) {
        divisor = 65535;
    }
    if (divisor == 0) {
        divisor = 1;
    }
    /* Send command byte:
     * 0x36 = 00110110b
     *   Bit 6-7: Channel 0
     *   Bit 4-5: Access mode: lobyte/hibyte
     *   Bit 1-3: Mode 3 (square wave generator)
     *   Bit 0:   Binary mode (not BCD)
     */
    outb(PIT_CMD, 0x36);
    /* Send divisor (low byte then high byte) */
    outb(PIT_CHANNEL0, divisor & 0xFF);
    outb(PIT_CHANNEL0, (divisor >> 8) & 0xFF);
    /* Enable IRQ0 */
    pic_unmask_irq(0);
    kprintf("Timer initialized at %d Hz (divisor: %d)\n", freq, divisor);
}
/* I/O port output */
static inline void outb(uint16_t port, uint8_t value) {
    __asm__ volatile ("outb %0, %1" : : "a"(value), "Nd"(port));
}
/* Get current tick count */
uint64_t timer_get_ticks(void) {
    return tick_count;
}
/* Get seconds since boot */
uint64_t timer_get_seconds(void) {
    /* This is approximate; assumes 100 Hz timer */
    return tick_count / 100;
}
/* Sleep for given milliseconds (busy-wait) */
void timer_sleep_ms(uint32_t ms) {
    /* At 100 Hz, each tick is 10 ms */
    uint64_t target = tick_count + (ms / 10) + 1;
    while (tick_count < target) {
        __asm__ volatile ("hlt");  /* Wait for next interrupt */
    }
}
```
Wait — we need to send EOI in the timer handler! Let's fix that:
```c
/* Timer interrupt handler */
static void timer_handler(registers_t *regs) {
    (void)regs;
    tick_count++;
    /* Send EOI to PIC - this is IRQ0 */
    pic_send_eoi(0);
}
```
Actually, we should handle EOI in a more structured way. Let's create a dedicated IRQ handling system:
Update `kernel/interrupts.c`:
```c
/* Called from assembly stub - dispatches to appropriate handler */
void interrupt_handler(registers_t *regs) {
    /* Check if we have a custom handler */
    if (interrupt_handlers[regs->int_no] != 0) {
        isr_t handler = interrupt_handlers[regs->int_no];
        handler(regs);
    } else {
        /* No handler - check if it's an exception or IRQ */
        if (regs->int_no < 32) {
            handle_exception(regs);
        } else if (regs->int_no >= 32 && regs->int_no < 48) {
            kprintf("Unhandled IRQ %d\n", regs->int_no - 32);
        }
    }
    /* Send EOI for IRQs (32-47) */
    if (regs->int_no >= 32 && regs->int_no < 48) {
        pic_send_eoi(regs->int_no - 32);
    }
}
```
This centralizes EOI handling. Each IRQ handler doesn't need to worry about it.
## The Keyboard: Human Input at Last

![PS/2 Keyboard Scancode Flow](./diagrams/diag-keyboard-scancodes.svg)

The PS/2 keyboard is deceptively simple: when you press or release a key, the keyboard controller sends a **scancode** to port 0x60 and triggers IRQ1.
The tricky part: **scancodes are not ASCII**. Scancode 0x1E means "the A key was pressed," but it doesn't tell you if Shift is held, or if Caps Lock is on. That's your job.
### Scancode Set 1 (Default)
The keyboard uses "Set 1" scancodes by default:
- **Make code**: Sent when key is pressed (e.g., 0x1E for 'A')
- **Break code**: Sent when key is released (0x9E for 'A' = 0x1E + 0x80)
Special keys send multi-byte scancodes:
- Extended keys (arrows, etc.) start with 0xE0
- Print Screen: E0 2A E0 37 (make), E0 B7 E0 AA (break)
- Pause: E1 1D 45 E1 9D C5
### The Keyboard Buffer
We need a circular buffer to store keystrokes for later retrieval by applications:
```c
#define KB_BUFFER_SIZE 256
static char kb_buffer[KB_BUFFER_SIZE];
static volatile uint32_t kb_write_pos = 0;  /* Where next char goes */
static volatile uint32_t kb_read_pos = 0;   /* Where to read from */
/* Buffer is empty when read_pos == write_pos */
/* Buffer is full when (write_pos + 1) % SIZE == read_pos */
```
Create `kernel/keyboard.h`:
```c
/* kernel/keyboard.h - PS/2 keyboard driver */
#ifndef KEYBOARD_H
#define KEYBOARD_H
#include <stdint.h>
#include <stdbool.h>
/* Initialize the keyboard driver */
void keyboard_init(void);
/* Read a character from the keyboard buffer (blocking) */
char keyboard_getchar(void);
/* Check if a character is available */
bool keyboard_has_char(void);
/* Read a character without blocking (returns 0 if none) */
char keyboard_try_getchar(void);
/* Get current shift/ctrl/alt state */
bool keyboard_shift_held(void);
bool keyboard_ctrl_held(void);
bool keyboard_alt_held(void);
#endif /* KEYBOARD_H */
```
Create `kernel/keyboard.c`:
```c
/* kernel/keyboard.c - PS/2 keyboard implementation */
#include "keyboard.h"
#include "pic.h"
#include "interrupts.h"
#include "kprintf.h"
/* Keyboard buffer */
#define KB_BUFFER_SIZE 256
static char kb_buffer[KB_BUFFER_SIZE];
static volatile uint32_t kb_write_pos = 0;
static volatile uint32_t kb_read_pos = 0;
/* Modifier key state */
static volatile bool left_shift = false;
static volatile bool right_shift = false;
static volatile bool left_ctrl = false;
static volatile bool right_ctrl = false;
static volatile bool left_alt = false;
static volatile bool right_alt = false;
static volatile bool caps_lock = false;
static volatile bool num_lock = false;
static volatile bool scroll_lock = false;
/* Keyboard LEDs */
static volatile uint8_t led_state = 0;
/* Scancode to ASCII table (Set 1, US layout) */
/* Index = scancode, value = ASCII character */
static const char scancode_to_ascii[] = {
    0,    0,   '1', '2', '3', '4', '5', '6',    /* 00-07 */
    '7', '8', '9', '0', '-', '=', '\b', '\t',    /* 08-0F: backspace, tab */
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i',      /* 10-17 */
    'o', 'p', '[', ']', '\n', 0,                 /* 18-1E: enter, left ctrl */
    'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',      /* 1E-25 */
    'l', ';', '\'', '`', 0, '\\',                /* 26-2B: left shift, backslash */
    'z', 'x', 'c', 'v', 'b', 'n', 'm', ',',      /* 2C-33 */
    '.', '/', 0, '*', 0, ' ',                    /* 34-39: right shift, numpad *, alt, space */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,               /* 3A-43: caps, F1-F10 */
    0, 0,                                         /* 44-45: num lock, scroll lock */
};
/* Shifted scancode to ASCII */
static const char scancode_to_ascii_shift[] = {
    0,    0,   '!', '@', '#', '$', '%', '^',    /* 00-07 */
    '&', '*', '(', ')', '_', '+', '\b', '\t',    /* 08-0F */
    'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I',      /* 10-17 */
    'O', 'P', '{', '}', '\n', 0,                 /* 18-1E */
    'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K',      /* 1E-25 */
    'L', ':', '"', '~', 0, '|',                  /* 26-2B */
    'Z', 'X', 'C', 'V', 'B', 'N', 'M', '<',      /* 2C-33 */
    '>', '?', 0, '*', 0, ' ',                    /* 34-39 */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,               /* 3A-43 */
    0, 0,                                         /* 44-45 */
};
/* I/O port functions */
static inline void outb(uint16_t port, uint8_t value) {
    __asm__ volatile ("outb %0, %1" : : "a"(value), "Nd"(port));
}
static inline uint8_t inb(uint16_t port) {
    uint8_t value;
    __asm__ volatile ("inb %1, %0" : "=a"(value) : "Nd"(port));
    return value;
}
/* Update keyboard LEDs */
static void update_leds(void) {
    led_state = 0;
    if (scroll_lock) led_state |= 1;
    if (num_lock)    led_state |= 2;
    if (caps_lock)   led_state |= 4;
    /* Send LED command to keyboard controller */
    outb(0x60, 0xED);         /* Set LEDs command */
    /* Wait for ACK... (simplified, should wait properly) */
    outb(0x60, led_state);
}
/* Add character to buffer */
static void buffer_put(char c) {
    uint32_t next_write = (kb_write_pos + 1) % KB_BUFFER_SIZE;
    if (next_write != kb_read_pos) {
        /* Buffer not full */
        kb_buffer[kb_write_pos] = c;
        kb_write_pos = next_write;
    }
    /* If buffer is full, character is dropped */
}
/* Keyboard interrupt handler */
static void keyboard_handler(registers_t *regs) {
    (void)regs;
    /* Read scancode from keyboard data port */
    uint8_t scancode = inb(0x60);
    /* Check for break code (key release) */
    bool is_release = (scancode & 0x80) != 0;
    scancode &= 0x7F;  /* Remove release bit */
    /* Handle modifier keys */
    switch (scancode) {
        case 0x2A: left_shift  = !is_release; return;
        case 0x36: right_shift = !is_release; return;
        case 0x1D: left_ctrl   = !is_release; return;
        /* Note: right ctrl sends E0 1D, handled in extended section */
        case 0x38: left_alt    = !is_release; return;
        case 0x3A: 
            if (!is_release) {
                caps_lock = !caps_lock;
                update_leds();
            }
            return;
        case 0x45:
            if (!is_release) {
                num_lock = !num_lock;
                update_leds();
            }
            return;
        case 0x46:
            if (!is_release) {
                scroll_lock = !scroll_lock;
                update_leds();
            }
            return;
    }
    /* Only process make codes for regular keys */
    if (is_release) return;
    /* Convert scancode to ASCII */
    char ascii = 0;
    bool shift = left_shift || right_shift;
    if (scancode < sizeof(scancode_to_ascii)) {
        if (shift) {
            ascii = scancode_to_ascii_shift[scancode];
        } else {
            ascii = scancode_to_ascii[scancode];
        }
        /* Apply caps lock to letters */
        if (caps_lock && ascii >= 'a' && ascii <= 'z') {
            ascii -= 32;
        } else if (caps_lock && ascii >= 'A' && ascii <= 'Z') {
            /* Already uppercase, caps lock + shift = lowercase */
            if (shift) ascii += 32;
        }
    }
    /* Add to buffer if valid */
    if (ascii != 0) {
        buffer_put(ascii);
    }
}
/* Initialize keyboard driver */
void keyboard_init(void) {
    /* Register keyboard handler for IRQ1 (vector 33) */
    register_interrupt_handler(33, keyboard_handler);
    /* Enable IRQ1 */
    pic_unmask_irq(1);
    kprintf("Keyboard initialized\n");
}
/* Check if character available */
bool keyboard_has_char(void) {
    return kb_read_pos != kb_write_pos;
}
/* Get character (blocking) */
char keyboard_getchar(void) {
    while (!keyboard_has_char()) {
        __asm__ volatile ("hlt");
    }
    char c = kb_buffer[kb_read_pos];
    kb_read_pos = (kb_read_pos + 1) % KB_BUFFER_SIZE;
    return c;
}
/* Try to get character (non-blocking) */
char keyboard_try_getchar(void) {
    if (keyboard_has_char()) {
        char c = kb_buffer[kb_read_pos];
        kb_read_pos = (kb_read_pos + 1) % KB_BUFFER_SIZE;
        return c;
    }
    return 0;
}
/* Modifier state queries */
bool keyboard_shift_held(void) { return left_shift || right_shift; }
bool keyboard_ctrl_held(void)  { return left_ctrl || right_ctrl; }
bool keyboard_alt_held(void)   { return left_alt || right_alt; }
```
## CPU Exceptions: When Hardware Says "No"

![CPU Exception Types and Error Codes](./diagrams/diag-exception-handlers.svg)

CPU exceptions are hardware-detected errors or special conditions:
| Vector | Name                | Error Code? | Cause |
|--------|---------------------|-------------|-------|
| 0      | Divide Error        | No          | DIV/IDIV by zero |
| 1      | Debug               | No          | Debug trap |
| 2      | NMI                 | No          | Non-maskable interrupt |
| 3      | Breakpoint          | No          | INT 3 instruction |
| 6      | Invalid Opcode      | No          | UD2 or undefined opcode |
| 7      | Device Not Avail    | No          | FPU instruction without FPU |
| 8      | **Double Fault**    | **Yes**     | Exception during exception handler |
| 10     | Invalid TSS         | Yes         | Task switch error |
| 11     | Segment Not Present | Yes         | Loading null/present=0 selector |
| 12     | Stack Fault         | Yes         | SS-related limit violation |
| 13     | **General Protection** | **Yes**  | Privilege violation, bad selector |
| 14     | **Page Fault**      | **Yes**     | Invalid page table entry |
### The Double Fault

![Double Fault and Triple Fault Chain](./diagrams/diag-double-fault.svg)

A double fault occurs when an exception happens while handling another exception. This is usually a sign of:
- **Stack overflow**: The handler's stack is corrupted
- **Bad IDT entry**: The handler address is invalid
- **Corrupted kernel state**: Something very wrong happened
If a double fault handler causes an exception, you get a **triple fault** — the CPU resets immediately. No error message, no stack trace, just silence.
Our double fault handler catches this and halts with diagnostics instead of triple-faulting:
```c
/* Already defined in interrupts.c, but let's emphasize it: */
if (int_no == 8) {
    kprintf("\nDOUBLE FAULT - System halted to prevent triple fault\n");
    kprintf("Error code: 0x%x\n", regs->err_code);
    /* Error code for double fault:
     * Bits 0-1: Which exception caused the first fault
     * Bit 2: 0 = first exception was interrupt, 1 = was exception
     * Bits 3-15: Reserved
     */
    while (1) {
        __asm__ volatile ("cli; hlt");
    }
}
```
### The Page Fault (Preview)
Page faults are special — they're the mechanism behind virtual memory. When code accesses an address without a valid page table entry, the CPU raises exception 14 and pushes:
- **CR2**: The faulting linear address
- **Error code**: Bits describing what happened
```
Error code bits:
Bit 0 (P): 0 = page not present, 1 = protection violation
Bit 1 (W): 0 = read, 1 = write
Bit 2 (U): 0 = kernel mode, 1 = user mode
Bit 3 (R): 0 = reserved bits, 1 = reserved bits caused fault
Bit 4 (I): 0 = instruction fetch, 1 = (if NX enabled)
```
For now, we just print the fault info. In the next milestone, we'll handle page faults properly for demand paging.
## Putting It All Together
Now let's update `kernel/main.c` to initialize interrupts:
```c
/* kernel/main.c - Kernel entry point */
#include "vga.h"
#include "serial.h"
#include "kprintf.h"
#include "idt.h"
#include "pic.h"
#include "timer.h"
#include "keyboard.h"
#include "interrupts.h"
void kernel_main(void) {
    /* Initialize VGA and serial */
    vga_init();
    if (!serial_init()) {
        vga_write("Serial init failed!\n", 
                  vga_color_attr(VGA_COLOR_RED, VGA_COLOR_BLACK));
    }
    kprintf("\n");
    kprintf("========================================\n");
    kprintf("  MyOS Kernel v0.2\n");
    kprintf("  Built on %s at %s\n", __DATE__, __TIME__);
    kprintf("========================================\n\n");
    /* Initialize interrupt handling */
    kprintf("Initializing interrupts...\n");
    idt_init();
    pic_init();
    /* Enable interrupts */
    kprintf("Enabling interrupts...\n");
    __asm__ volatile ("sti");
    /* Initialize timer at 100 Hz */
    timer_init(100);
    /* Initialize keyboard */
    keyboard_init();
    kprintf("\nSystem ready. Type something!\n");
    kprintf("Press keys to test keyboard (ESC to halt)...\n\n");
    /* Keyboard test loop */
    while (1) {
        char c = keyboard_getchar();
        if (c == 27) {  /* ESC */
            kprintf("\nESC pressed. Halting.\n");
            break;
        }
        /* Echo the character */
        kprintf("%c", c);
        /* Also print tick count on Enter */
        if (c == '\n') {
            kprintf("Ticks: %d, Seconds: %d\n", 
                    (uint32_t)timer_get_ticks(),
                    (uint32_t)timer_get_seconds());
        }
    }
    kprintf("\nSystem halted.\n");
    while (1) {
        __asm__ volatile ("cli; hlt");
    }
}
```
### Update the Makefile
Add the new source files:
```makefile
# Add to C_SOURCES
C_SOURCES = kernel/main.c kernel/vga.c kernel/serial.c kernel/kprintf.c \
            kernel/idt.c kernel/pic.c kernel/timer.c kernel/keyboard.c \
            kernel/interrupts.c
# Add to ASM_SOURCES
ASM_SOURCES = kernel/entry.asm kernel/idt_stubs.asm
```
## Hardware Soul: What Actually Happens
Let's trace a keystroke through the entire system:
```
1. You press 'A'
   └── Keyboard controller: Detects key press
   └── Generates scancode 0x1E
   └── Sends to output buffer at port 0x60
   └── Pulls IRQ1 line high → PIC
2. PIC (master)
   └── IRQ1 arrives
   └── Priority check (IRQ1 < current IRQ? No? Proceed)
   └── Sends INT to CPU with vector 33 (remapped)
   └── Waits for EOI
3. CPU
   └── Completes current instruction
   └── Pushes EFLAGS, CS, EIP to stack
   └── Looks up IDT entry 33
   └── Loads CS:EIP from gate descriptor
   └── Jumps to isr33 stub (in idt_stubs.asm)
4. Assembly stub (isr33)
   └── cli (disable interrupts)
   └── push byte 0 (dummy error code)
   └── push byte 33 (interrupt number)
   └── pusha (save EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI)
   └── push segment registers
   └── call interrupt_handler (C function)
5. C handler (interrupt_handler)
   └── Looks up registered handler for vector 33
   └── Calls keyboard_handler
6. keyboard_handler
   └── inb(0x60) → reads scancode 0x1E
   └── Checks shift state: false
   └── Looks up scancode_to_ascii[0x1E] → 'a'
   └── Checks caps lock: false
   └── buffer_put('a')
   └── Returns
7. Back to interrupt_handler
   └── Returns to assembly stub
8. Assembly stub
   └── pop segment registers
   └── popa (restore general registers)
   └── add esp, 8 (remove error code + int number)
   └── iret (pop EIP, CS, EFLAGS; restore CPU state)
9. CPU
   └── Resumes whatever it was doing
   └── Later: keyboard_getchar() returns 'a'
10. EOI (sent after handler returns, in our centralized code)
    └── pic_send_eoi(1)
    └── outb(0x20, 0x20) → tells master PIC "IRQ1 handled"
    └── PIC can now deliver IRQ1 again
```
**Cache behavior**: The IDT is in RAM, not cached specially. First access to an IDT entry is a cache miss (~100 cycles). The keyboard buffer is also in RAM — hot when frequently accessed.
**Interrupt latency**: From keystroke to handler entry:
- Keyboard controller: ~1 μs
- PIC propagation: ~100 ns
- CPU interrupt latency: ~10-50 cycles
- IDT lookup: ~50-100 cycles (cache miss)
- Total: ~500 ns to 1 μs
This is why interrupts are fast enough for human input — a 1 μs latency is imperceptible.
## Common Pitfalls and How to Debug Them
| Symptom | Likely Cause | Debug |
|---------|--------------|-------|
| System hangs after first keypress | Forgot EOI | Check `pic_send_eoi()` is called |
| Random crashes after interrupts | Not saving/restoring all registers | Verify `pusha`/`popa` and segment registers |
| Triple fault on first interrupt | Bad IDT entry (wrong selector/address) | Print IDT entries before `sti` |
| Wrong characters | Scancode table incorrect | Print raw scancodes to debug |
| Keyboard doesn't work | IRQ1 masked | Check `pic_unmask_irq(1)` was called |
### QEMU Debugging
```bash
# Log all interrupts
qemu-system-i386 -fda os.img -serial stdio -d int
# Don't reboot on triple fault (lets you see error)
qemu-system-i386 -fda os.img -serial stdio -no-reboot
# Connect GDB
qemu-system-i386 -fda os.img -serial stdio -s -S &
gdb -ex "target remote :1234" -ex "break *0x100000"
```
## Design Decisions
| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Interrupt Gates (type 0xE)** ✓ | Interrupts disabled during handler, no reentrancy | Can't nest interrupts | Most kernels |
| Trap Gates (type 0xF) | Can handle nested interrupts | Complex, must manage reentrancy | Some real-time OSes |
| **Centralized EOI** ✓ | Handlers don't need to remember | Less flexibility | Linux, most tutorials |
| Per-handler EOI | Handler controls timing | Easy to forget | Some embedded systems |
| **Circular buffer for keyboard** ✓ | Simple, bounded memory | Can overflow | Linux tty layer |
| Linked list buffer | Unbounded (within heap) | Fragmentation, complexity | Some advanced systems |
## Knowledge Cascade
You've just built the foundation for **asynchronous event handling** — a pattern that appears everywhere in computing.
**Same Domain:**
- **Preemptive scheduling (next milestone)** — The timer interrupt is the trigger. The scheduler runs on each tick, deciding whether to switch processes.
- **Page fault handling (milestone 3)** — Page faults use the same exception mechanism. The handler reads CR2, finds or allocates a page, updates page tables, and returns.
- **System calls** — INT 0x80 is a software-triggered interrupt. Same IDT mechanism, just initiated by `int $0x80` instead of hardware.
**Cross-Domain:**
- **JavaScript event loop** — Node.js callbacks are the software equivalent of interrupt handlers. The event loop is like a PIC, queuing events and dispatching to handlers. Understanding ISRs makes async/await intuitive: it's just `await` for the event to fire.
- **Exception handling in languages** — C++ `throw` and Java exceptions compile to stack unwinding that mirrors what the CPU does on an exception. The `catch` block is your exception handler.
- **Network interrupt coalescing** — High-performance NICs batch interrupts (process 64 packets, then interrupt once). The tradeoff is latency vs. throughput — same reason we might process multiple keyboard scancodes per interrupt.
- **Priority inversion** — Interrupts are disabled during ISR execution. If your ISR takes too long, higher-priority interrupts are blocked. This is the hardware version of holding a lock too long — the classic Mars Pathfinder priority inversion bug.
**Historical Context:**
- The 8259 PIC design dates to the 8086 era (1978). Modern systems use APIC (Advanced PIC), but the 8259 remains for compatibility.
- The PS/2 keyboard interface is from 1987. USB keyboards emulate PS/2 for backward compatibility — the OS still sees IRQ1 and port 0x60.
- The concept of "interrupt vectors" was pioneered by the IBM 360 (1964). Before that, computers polled devices.
**What You Can Now Build:**
- A preemptive scheduler that switches tasks on timer ticks
- A proper keyboard driver with full scancode set 1 support
- Mouse driver (IRQ12, similar patterns)
- Real-time clock (IRQ8) for wall-clock time
- A basic shell that reads keyboard input
---
## Acceptance Criteria
[[CRITERIA_JSON: {"milestone_id": "build-os-m2", "criteria": ["IDT contains 256 8-byte entries with proper gate descriptors loaded via lidt instruction", "CPU exception handlers (vectors 0-31) installed with type 0x8E (interrupt gate, DPL=0, 32-bit)", "Exception handlers print descriptive messages including exception name, error code, EIP, CS, EFLAGS, and general registers", "Page fault handler (vector 14) reads CR2 and prints faulting address plus error code bits (present/write/user)", "Double fault handler (vector 8) prints diagnostic information and halts with cli/hlt loop to prevent triple fault", "Interrupt handlers save all general-purpose registers (pusha) and segment registers (DS, ES, FS, GS) on entry", "Interrupt handlers restore all registers and execute iret to return; error code is removed from stack before iret", "Assembly stubs differentiate exceptions with error codes (8, 10-14) by pushing dummy error code for others", "PIC 8259 master and slave initialized with ICW1-ICW4 sequence remapping IRQs to vectors 32-47", "PIC master vector offset set to 0x20 (IRQ0→32) and slave offset to 0x28 (IRQ8→40)", "PIC cascade configured: master ICW3=0x04 (slave on IRQ2), slave ICW3=0x02 (slave ID 2)", "EOI sent to correct PIC(s) after each IRQ handler: master only for IRQ0-7, both for IRQ8-15", "EOI sent via centralized code in interrupt_handler for vectors 32-47 after custom handler returns", "PIT channel 0 programmed with mode 3 (square wave) and divisor calculated for target frequency", "Timer interrupt handler (IRQ0/vector 32) increments 64-bit global tick counter and returns", "Timer initialized at configurable frequency (e.g., 100 Hz) with IRQ0 unmasked via pic_unmask_irq(0)", "Keyboard interrupt handler (IRQ1/vector 33) reads scancode from I/O port 0x60", "Keyboard handler tracks modifier state (Shift, Ctrl, Alt, Caps Lock) and applies to character conversion", "Scancode-to-ASCII conversion using lookup tables with separate shifted and unshifted tables", "Keyboard handler implements circular buffer (256 entries) with separate read and write positions", "keyboard_getchar blocks with hlt until character available; keyboard_try_getchar returns 0 if empty", "Interrupts enabled via sti instruction only after IDT loaded, PIC remapped, and handlers registered", "System boots in QEMU and responds to keyboard input with correct character echo", "Timer tick counter visible and incrementing (e.g., printed on Enter key)"]}]
<!-- END_MS -->


<!-- MS_ID: build-os-m3 -->
<!-- MS_ID: build-os-m3 -->
# Milestone 3: Physical and Virtual Memory Management

![OS Kernel Architecture Map](./diagrams/diag-satellite-os.svg)

Your kernel boots. It handles interrupts. It responds to the keyboard. But it's still living in a lie.
Right now, every memory address in your kernel is a physical address. When you write `0xB8000` for the VGA buffer, you're writing to physical RAM at exactly that location. When your kernel sits at `0x100000`, that's where it physically resides. There's no translation, no protection, no illusion.
This milestone shatters that simplicity. You're about to build the hardware-enforced address translation layer that makes modern computing possible: **virtual memory**.
## The Fundamental Tension: Software Wants Infinity, Hardware Gives Fragments
Here's the brutal reality your kernel must confront:
**Software expectations:**
- Each process wants its own private address space starting at 0
- Programs want to allocate memory without knowing where it physically lives
- The kernel wants to protect itself from user programs
- We want the illusion of contiguous memory even when physical RAM is fragmented
**Hardware constraints:**
- Physical RAM is limited (let's say 512MB in our QEMU setup)
- Physical addresses are what the memory bus actually uses
- The CPU can only access what's physically present
- Different processes must not see each other's memory
The tension: **every memory access must be translated from what the program thinks it's accessing (virtual) to what's actually in RAM (physical)**, and this translation must happen fast enough that the CPU doesn't crawl to a halt.
The solution isn't software — it's hardware. The x86 CPU has a Memory Management Unit (MMU) that performs this translation on every single memory access. But the MMU is blind without data. Your job? Feed it the **page tables** that define the translation.

![Two-Level Page Table Structure](./diagrams/diag-page-table-structure.svg)

## The Revelation: Virtual Memory Is NOT Swap
Here's the misconception that trips up everyone:
> "Virtual memory is when the OS swaps memory to disk when RAM is full."
**No.** That's like saying "a car is when you run out of gas and walk." Swap is an optional *extension* to virtual memory, not the core mechanism.
Virtual memory is a **hardware-enforced address translation layer**. Every single memory access — every instruction fetch, every stack push, every global variable read — goes through page tables that map virtual addresses to physical frames. The CPU's MMU does this translation automatically, in hardware, on every access.
**What virtual memory actually provides:**
1. **Isolation**: User process A cannot access process B's memory because they have different page tables. The kernel's memory is protected because user page tables mark those pages as supervisor-only.
2. **Illusion of contiguity**: A process can have a contiguous virtual address space (0x00000000 to 0xFFFFFFFF) backed by scattered physical frames. The page table makes the fragmentation invisible.
3. **Demand loading**: A page can be "not present" in the page table. Accessing it triggers a page fault. The OS can then load it from disk, update the page table, and resume. This enables loading programs lazily.
4. **Memory-mapped I/O**: Files and devices can be mapped into the address space. Reading memory reads the file/device. This is how `mmap()` works.
Swap is just an optimization: when physical RAM is full, the OS can write some pages to disk and mark them "not present." But the core mechanism — page tables translating every address — is always active once paging is enabled.
## Three-Level View of a Memory Access
### Level 1 — Application View
A C program writes `*ptr = 42;` where `ptr = 0x08049ABC`. The program thinks it's writing to address 0x08049ABC. It knows nothing about physical RAM, page tables, or the MMU.
### Level 2 — OS/Kernel View (What You're Building)
The kernel sets up page tables that define the virtual→physical mapping. For address 0x08049ABC:
- Extract page directory index (bits 22-31): 0x02
- Extract page table index (bits 12-21): 0x049
- Extract page offset (bits 0-11): 0xABC
- Look up page directory entry 2 → find page table physical address
- Look up page table entry 0x049 → find physical frame address
- Add offset: physical address = frame_address + 0xABC
The kernel also handles page faults — when a program accesses an address without a valid page table entry, the CPU traps to the page fault handler.
### Level 3 — Hardware View
- **CR3 register**: Holds the physical address of the current page directory
- **MMU**: Performs the two-level lookup on every memory access
- **TLB (Translation Lookaside Buffer)**: Caches recent virtual→physical translations. Without the TLB, every memory access would require two additional memory accesses (PD lookup + PT lookup).
- **Page fault exception**: If the present bit is 0, or if access rights are violated, the CPU triggers exception 14

![Page Directory Entry Bit Fields](./diagrams/diag-page-directory-entry.svg)

## The x86 Two-Level Page Table
On 32-bit x86 with 4KB pages, a virtual address is split into three parts:
```
31          22 21          12 11           0
+-------------+-------------+-------------+
| Page Dir    | Page Table  | Page Offset  |
| Index (10b) | Index (10b) | (12 bits)    |
+-------------+-------------+-------------+
Page Directory Index:  Which entry in the page directory (0-1023)
Page Table Index:      Which entry in that page table (0-1023)
Page Offset:           Which byte within the 4KB page (0-4095)
```
**Why two levels?** A single-level page table for 4GB would need 1,048,576 entries (4GB / 4KB) × 4 bytes = 4MB per process. Two levels allow sparse allocation: a page table is only allocated when that region of virtual address space is used.

![Page Table Entry Bit Fields](./diagrams/diag-page-table-entry.svg)

### Page Directory Entry (PDE) Structure
Each PDE (4 bytes) describes a page table:
| Bits | Name | Meaning |
|------|------|---------|
| 0 | Present | 1 = page table exists in memory |
| 1 | R/W | 1 = writable, 0 = read-only |
| 2 | U/S | 1 = user-accessible, 0 = supervisor-only |
| 3 | PWT | Write-through caching |
| 4 | PCD | Cache disable |
| 5 | A | Accessed (set by CPU on read) |
| 6 | Ignored | Available for OS use |
| 7 | PS | Page size (0 = 4KB, 1 = 4MB for large pages) |
| 8-11 | Ignored | Available for OS use |
| 12-31 | Address | Physical address of page table (4KB aligned) |
### Page Table Entry (PTE) Structure
Each PTE (4 bytes) describes a 4KB physical frame:
| Bits | Name | Meaning |
|------|------|---------|
| 0 | Present | 1 = page is in memory |
| 1 | R/W | 1 = writable, 0 = read-only |
| 2 | U/S | 1 = user-accessible, 0 = supervisor-only |
| 3 | PWT | Write-through caching |
| 4 | PCD | Cache disable |
| 5 | A | Accessed (set by CPU on read) |
| 6 | D | Dirty (set by CPU on write) |
| 7 | PAT | Page attribute table index |
| 8 | G | Global page (not flushed on CR3 reload) |
| 9-11 | Available | For OS use |
| 12-31 | Address | Physical frame address (4KB aligned) |
**Critical insight**: The address fields only store bits 12-31. The lower 12 bits are implicitly 0 because page tables and frames must be 4KB-aligned. This is why you'll see code like `frame_addr & 0xFFFFF000`.
## Step 1: Getting the Physical Memory Map

![Physical Memory Map from E820/Multiboot](./diagrams/diag-physical-memory-map.svg)

Before you can allocate physical frames, you need to know which physical addresses actually contain usable RAM. The BIOS provides this via the E820 interrupt (real mode) or multiboot info structure (if using GRUB or a multiboot-compliant bootloader).
Since we're using a simple custom bootloader, we'll pass the memory map from the bootloader to the kernel. But for simplicity, let's assume a basic memory layout and enhance it later:
Create `kernel/memory.h`:
```c
/* kernel/memory.h - Physical and virtual memory management */
#ifndef MEMORY_H
#define MEMORY_H
#include <stdint.h>
#include <stdbool.h>
/* Memory region types (from E820/multiboot) */
typedef enum {
    MEMORY_TYPE_USABLE = 1,
    MEMORY_TYPE_RESERVED = 2,
    MEMORY_TYPE_ACPI_RECLAIMABLE = 3,
    MEMORY_TYPE_ACPI_NVS = 4,
    MEMORY_TYPE_BAD = 5,
} memory_type_t;
/* Memory map entry */
typedef struct {
    uint32_t base_low;
    uint32_t base_high;
    uint32_t length_low;
    uint32_t length_high;
    uint32_t type;
    uint32_t acpi_attrs;
} __attribute__((packed)) memory_map_entry_t;
/* Physical frame allocator */
void pmm_init(uint32_t total_memory_kb);
void pmm_init_from_multiboot(memory_map_entry_t *mmap, uint32_t length);
void pmm_mark_used(uint32_t physical_addr, uint32_t size);
void *pmm_alloc_frame(void);
void pmm_free_frame(void *frame);
uint32_t pmm_get_free_count(void);
uint32_t pmm_get_total_count(void);
/* Page table management */
void vmm_init(void);
void vmm_map_page(uint32_t virtual_addr, uint32_t physical_addr, 
                  uint32_t flags);
void vmm_unmap_page(uint32_t virtual_addr);
uint32_t vmm_get_physical(uint32_t virtual_addr);
void vmm_flush_tlb(uint32_t virtual_addr);
/* Page flags */
#define PAGE_PRESENT    (1 << 0)
#define PAGE_WRITABLE   (1 << 1)
#define PAGE_USER       (1 << 2)
#define PAGE_PWT        (1 << 3)
#define PAGE_PCD        (1 << 4)
#define PAGE_ACCESSED   (1 << 5)
#define PAGE_DIRTY      (1 << 6)
#define PAGE_LARGE      (1 << 7)
#define PAGE_GLOBAL     (1 << 8)
/* Kernel heap */
void kheap_init(void);
void *kmalloc(uint32_t size);
void kfree(void *ptr);
/* Convenience macro */
#define PAGE_SIZE 4096
#define PAGE_ALIGN(x) (((x) + 0xFFF) & ~0xFFF)
#define PAGE_ALIGN_DOWN(x) ((x) & ~0xFFF)
#endif /* MEMORY_H */
```
Now let's implement the physical memory manager. We'll use a **bitmap allocator** — one bit per frame, where 1 = used, 0 = free.

![Physical Frame Allocator (Bitmap)](./diagrams/diag-frame-allocator.svg)

Create `kernel/pmm.c`:
```c
/* kernel/pmm.c - Physical memory manager (bitmap allocator) */
#include "memory.h"
#include "kprintf.h"
#include <string.h>
/* Bitmap for tracking frames: 1 = used, 0 = free */
static uint32_t *frame_bitmap = NULL;
static uint32_t total_frames = 0;
static uint32_t free_frames = 0;
static uint32_t bitmap_size = 0;  /* In uint32_t elements */
/* Bitmap manipulation macros */
#define INDEX_FROM_BIT(bit)  ((bit) / 32)
#define OFFSET_FROM_BIT(bit) ((bit) % 32)
#define SET_FRAME(bit)   (frame_bitmap[INDEX_FROM_BIT(bit)] |= (1 << OFFSET_FROM_BIT(bit)))
#define CLEAR_FRAME(bit) (frame_bitmap[INDEX_FROM_BIT(bit)] &= ~(1 << OFFSET_FROM_BIT(bit)))
#define TEST_FRAME(bit)  (frame_bitmap[INDEX_FROM_BIT(bit)] & (1 << OFFSET_FROM_BIT(bit)))
/* Simple memory operations for use before kheap is available */
static void memset_simple(void *dest, uint8_t value, uint32_t count) {
    uint8_t *d = (uint8_t *)dest;
    while (count--) {
        *d++ = value;
    }
}
/* Initialize PMM with total memory size */
void pmm_init(uint32_t total_memory_kb) {
    /* Calculate total number of 4KB frames */
    total_frames = total_memory_kb / 4;
    /* Calculate bitmap size (one bit per frame, stored in uint32_t) */
    bitmap_size = (total_frames + 31) / 32;
    /* Place bitmap right after the kernel (we'll refine this later) */
    /* For now, use a fixed location that won't conflict */
    extern uint32_t _kernel_end;
    frame_bitmap = (uint32_t *)PAGE_ALIGN((uint32_t)&_kernel_end);
    /* Zero the bitmap (all frames initially "free") */
    memset_simple(frame_bitmap, 0, bitmap_size * sizeof(uint32_t));
    /* Mark first 1MB as used (BIOS, VGA, kernel, etc.) */
    /* 1MB = 256 frames */
    for (uint32_t i = 0; i < 256; i++) {
        SET_FRAME(i);
    }
    /* Mark the bitmap itself as used */
    uint32_t bitmap_start_frame = (uint32_t)frame_bitmap / PAGE_SIZE;
    uint32_t bitmap_end_frame = PAGE_ALIGN((uint32_t)frame_bitmap + 
                                            bitmap_size * sizeof(uint32_t)) / PAGE_SIZE;
    for (uint32_t i = bitmap_start_frame; i < bitmap_end_frame; i++) {
        SET_FRAME(i);
    }
    free_frames = total_frames - 256 - (bitmap_end_frame - bitmap_start_frame);
    kprintf("PMM initialized: %d MB total, %d frames, %d free\n",
            total_memory_kb / 1024, total_frames, free_frames);
    kprintf("Bitmap at 0x%x, size: %d bytes\n", 
            (uint32_t)frame_bitmap, bitmap_size * 4);
}
/* Initialize from multiboot memory map */
void pmm_init_from_multiboot(memory_map_entry_t *mmap, uint32_t length) {
    /* First, count total memory */
    uint32_t total_mem = 0;
    uint32_t entry_count = length / sizeof(memory_map_entry_t);
    kprintf("Memory map from bootloader:\n");
    for (uint32_t i = 0; i < entry_count; i++) {
        /* For simplicity, only handle 32-bit addresses */
        uint32_t base = mmap[i].base_low;
        uint32_t len = mmap[i].length_low;
        uint32_t end = base + len;
        kprintf("  [%d] 0x%08x - 0x%08x: %s\n", i, base, end,
                mmap[i].type == MEMORY_TYPE_USABLE ? "Usable" : "Reserved");
        if (mmap[i].type == MEMORY_TYPE_USABLE && end > total_mem) {
            total_mem = end;
        }
    }
    /* Initialize with total memory */
    pmm_init(total_mem / 1024);
    /* Now mark reserved regions as used */
    for (uint32_t i = 0; i < entry_count; i++) {
        if (mmap[i].type != MEMORY_TYPE_USABLE) {
            uint32_t base = PAGE_ALIGN_DOWN(mmap[i].base_low);
            uint32_t end = PAGE_ALIGN(mmap[i].base_low + mmap[i].length_low);
            for (uint32_t addr = base; addr < end; addr += PAGE_SIZE) {
                uint32_t frame = addr / PAGE_SIZE;
                if (frame < total_frames && !TEST_FRAME(frame)) {
                    SET_FRAME(frame);
                    free_frames--;
                }
            }
        }
    }
}
/* Mark a region as used */
void pmm_mark_used(uint32_t physical_addr, uint32_t size) {
    uint32_t start_frame = physical_addr / PAGE_SIZE;
    uint32_t end_frame = PAGE_ALIGN(physical_addr + size) / PAGE_SIZE;
    for (uint32_t i = start_frame; i < end_frame && i < total_frames; i++) {
        if (!TEST_FRAME(i)) {
            SET_FRAME(i);
            free_frames--;
        }
    }
}
/* Allocate a single frame, returns physical address or 0 if out of memory */
void *pmm_alloc_frame(void) {
    /* Linear search for a free frame */
    /* TODO: Optimize by tracking last allocated frame */
    for (uint32_t i = 0; i < total_frames; i++) {
        if (!TEST_FRAME(i)) {
            SET_FRAME(i);
            free_frames--;
            return (void *)(i * PAGE_SIZE);
        }
    }
    kprintf("WARNING: Out of physical memory!\n");
    return NULL;
}
/* Free a frame */
void pmm_free_frame(void *frame) {
    uint32_t frame_num = (uint32_t)frame / PAGE_SIZE;
    /* Sanity checks */
    if (frame_num >= total_frames) {
        kprintf("WARNING: Attempt to free invalid frame 0x%x\n", (uint32_t)frame);
        return;
    }
    if (!TEST_FRAME(frame_num)) {
        kprintf("WARNING: Double free of frame 0x%x\n", (uint32_t)frame);
        return;
    }
    CLEAR_FRAME(frame_num);
    free_frames++;
}
/* Get free frame count */
uint32_t pmm_get_free_count(void) {
    return free_frames;
}
uint32_t pmm_get_total_count(void) {
    return total_frames;
}
```
### Why Bitmap? The Design Decision
| Allocator Type | Pros | Cons | Used By |
|----------------|------|------|---------|
| **Bitmap** ✓ | Simple, O(n) alloc, O(1) free, deterministic | Slow allocation (linear search), fragmentation | Linux bootmem, many hobby OSes |
| Linked Free List | O(1) alloc/free | Complex, needs metadata storage | Linux buddy allocator |
| Stack Allocator | O(1) alloc/free | Can't free arbitrary frames | Some embedded systems |
| Buddy Allocator | Handles contiguous allocation, low fragmentation | Complex, internal fragmentation | Linux, FreeBSD |
For a learning kernel, the bitmap is ideal: it's simple to understand, easy to debug, and sufficient for our needs. A production kernel would use a buddy allocator for the physical frame allocator.
## Step 2: Setting Up Page Tables

![Two-Level Page Table Structure](./diagrams/diag-page-table-structure.svg)

Now we build the page table infrastructure. We need:
1. A page directory (one 4KB page = 1024 entries)
2. Page tables (as needed, each 4KB = 1024 entries)
3. Functions to map/unmap virtual addresses
Create `kernel/vmm.c`:
```c
/* kernel/vmm.c - Virtual memory manager */
#include "memory.h"
#include "pmm.h"
#include "kprintf.h"
/* Current page directory (physical address) */
static uint32_t *current_page_directory = NULL;
/* Page directory and page table are arrays of 32-bit entries */
#define PTES_PER_PT 1024
#define PDES_PER_PD 1024
/* Get page directory entry for a virtual address */
static inline uint32_t *get_pde(uint32_t virtual_addr) {
    uint32_t index = (virtual_addr >> 22) & 0x3FF;
    return &current_page_directory[index];
}
/* Get page table entry for a virtual address */
/* This returns a pointer to the PTE in virtual memory */
/* Assumes the page table is mapped into virtual memory */
static inline uint32_t *get_pte(uint32_t virtual_addr, uint32_t *page_table) {
    uint32_t index = (virtual_addr >> 12) & 0x3FF;
    return &page_table[index];
}
/* Allocate a new page table */
static uint32_t *alloc_page_table(void) {
    void *frame = pmm_alloc_frame();
    if (frame == NULL) {
        return NULL;
    }
    /* Zero the page table */
    uint32_t *pt = (uint32_t *)frame;
    for (int i = 0; i < PTES_PER_PT; i++) {
        pt[i] = 0;
    }
    return pt;
}
/* Map a virtual page to a physical frame */
void vmm_map_page(uint32_t virtual_addr, uint32_t physical_addr, 
                  uint32_t flags) {
    /* Align addresses to page boundaries */
    virtual_addr = PAGE_ALIGN_DOWN(virtual_addr);
    physical_addr = PAGE_ALIGN_DOWN(physical_addr);
    /* Get the page directory entry */
    uint32_t *pde = get_pde(virtual_addr);
    /* Check if page table exists */
    uint32_t *page_table;
    if (!(*pde & PAGE_PRESENT)) {
        /* Allocate a new page table */
        page_table = alloc_page_table();
        if (page_table == NULL) {
            kprintf("ERROR: Failed to allocate page table for 0x%x\n", virtual_addr);
            return;
        }
        /* Set the PDE to point to the new page table */
        *pde = (uint32_t)page_table | PAGE_PRESENT | PAGE_WRITABLE | 
               (flags & PAGE_USER);
    } else {
        /* Get the existing page table's physical address */
        page_table = (uint32_t *)(*pde & 0xFFFFF000);
    }
    /* Set the page table entry */
    uint32_t *pte = get_pte(virtual_addr, page_table);
    *pte = physical_addr | flags | PAGE_PRESENT;
    /* Invalidate TLB entry for this page */
    vmm_flush_tlb(virtual_addr);
}
/* Unmap a virtual page */
void vmm_unmap_page(uint32_t virtual_addr) {
    virtual_addr = PAGE_ALIGN_DOWN(virtual_addr);
    uint32_t *pde = get_pde(virtual_addr);
    if (!(*pde & PAGE_PRESENT)) {
        return;  /* No page table, nothing to unmap */
    }
    uint32_t *page_table = (uint32_t *)(*pde & 0xFFFFF000);
    uint32_t *pte = get_pte(virtual_addr, page_table);
    /* Clear the entry */
    *pte = 0;
    /* Invalidate TLB */
    vmm_flush_tlb(virtual_addr);
}
/* Get physical address for a virtual address */
uint32_t vmm_get_physical(uint32_t virtual_addr) {
    uint32_t *pde = get_pde(virtual_addr);
    if (!(*pde & PAGE_PRESENT)) {
        return 0;  /* Not mapped */
    }
    uint32_t *page_table = (uint32_t *)(*pde & 0xFFFFF000);
    uint32_t *pte = get_pte(virtual_addr, page_table);
    if (!(*pte & PAGE_PRESENT)) {
        return 0;  /* Not mapped */
    }
    /* Extract physical frame and add offset */
    uint32_t frame = *pte & 0xFFFFF000;
    uint32_t offset = virtual_addr & 0xFFF;
    return frame + offset;
}
/* Flush TLB entry for a specific page */
void vmm_flush_tlb(uint32_t virtual_addr) {
    __asm__ volatile ("invlpg (%0)" : : "r"(virtual_addr) : "memory");
}
/* Flush entire TLB by reloading CR3 */
void vmm_flush_tlb_all(void) {
    uint32_t cr3;
    __asm__ volatile ("mov %%cr3, %0" : "=r"(cr3));
    __asm__ volatile ("mov %0, %%cr3" : : "r"(cr3) : "memory");
}
```

![TLB Flush Methods](./diagrams/diag-tlb-flush.svg)

### The TLB Problem
Here's a subtle but critical issue: the CPU caches page table translations in the **Translation Lookaside Buffer (TLB)**. When you modify a page table entry, the TLB doesn't automatically update.
If you:
1. Map virtual 0x1000 → physical 0x5000
2. Map virtual 0x1000 → physical 0x6000 (without flushing)
The CPU might still use the cached translation (0x5000) because the TLB hasn't been updated!
**Two ways to flush the TLB:**
- `invlpg addr`: Flushes the TLB entry for a single page
- Reloading CR3: Flushes all non-global TLB entries
For our kernel, we use `invlpg` after each page table modification for efficiency. Reloading CR3 is overkill unless you're switching address spaces (context switch).
## Step 3: Enabling Paging — The Critical Sequence

![Paging Enable Sequence](./diagrams/diag-paging-enable.svg)

Here's the trap that catches everyone: **you cannot enable paging without identity-mapping the code that's currently executing**.
When you set CR0.PG = 1, the CPU immediately starts translating addresses through page tables. If the code you're executing isn't mapped, the CPU can't fetch the next instruction — instant page fault, triple fault, reset.
**The sequence must be:**
1. Create a page directory
2. Identity-map the first few MB (kernel code, VGA, etc.)
3. Load CR3 with the page directory address
4. Set CR0.PG = 1
5. Continue execution (now everything goes through page tables)
Let's implement the initialization:
```c
/* Add to kernel/vmm.c */
/* Identity-map a range of pages */
static void identity_map_range(uint32_t start, uint32_t end, uint32_t flags) {
    start = PAGE_ALIGN_DOWN(start);
    end = PAGE_ALIGN(end);
    for (uint32_t addr = start; addr < end; addr += PAGE_SIZE) {
        vmm_map_page(addr, addr, flags);
    }
}
/* Initialize virtual memory management */
void vmm_init(void) {
    /* Allocate the page directory */
    current_page_directory = (uint32_t *)pmm_alloc_frame();
    if (current_page_directory == NULL) {
        kprintf("PANIC: Failed to allocate page directory!\n");
        while (1) { __asm__ volatile ("hlt"); }
    }
    /* Zero the page directory */
    for (int i = 0; i < PDES_PER_PD; i++) {
        current_page_directory[i] = 0;
    }
    kprintf("Page directory at physical 0x%x\n", 
            (uint32_t)current_page_directory);
    /* Identity-map the first 4MB (kernel + boot structures + VGA) */
    /* This ensures we can continue executing after enabling paging */
    kprintf("Identity-mapping first 4MB...\n");
    identity_map_range(0x00000000, 0x00400000, PAGE_WRITABLE);
    /* Mark VGA buffer as user-accessible (for demo purposes) */
    /* Actually, we should keep it supervisor-only; this is just an example */
    /* Load CR3 with our page directory */
    kprintf("Loading CR3...\n");
    __asm__ volatile ("mov %0, %%cr3" : : "r"(current_page_directory));
    /* Enable paging by setting CR0.PG */
    kprintf("Enabling paging...\n");
    uint32_t cr0;
    __asm__ volatile ("mov %%cr0, %0" : "=r"(cr0));
    cr0 |= (1 << 31);  /* Set PG bit */
    __asm__ volatile ("mov %0, %%cr0" : : "r"(cr0));
    kprintf("Paging enabled!\n");
    /* Now we can map higher-half addresses */
    /* The kernel can be accessed at both 0x00100000 and 0xC0100000 */
}
```
## Step 4: Higher-Half Kernel Mapping

![Identity Mapping vs Higher-Half Mapping](./diagrams/diag-identity-map.svg)

Here's where the magic happens. A **higher-half kernel** maps itself to high virtual addresses (e.g., 0xC0000000+) while leaving low addresses for user processes.
**Why higher-half?**
- User processes get a clean low address space (0x00000000-0xBFFFFFFF)
- The kernel is always mapped at the same high address, regardless of which process is running
- System calls don't require changing page tables for kernel access
**The challenge:** The kernel binary is loaded at 0x100000 (physical). We want to access it at 0xC0100000 (virtual). This requires:
1. Mapping 0xC0100000 → 0x00100000
2. Running code that's aware of its virtual address
This gets tricky because the linker script must use virtual addresses, but the bootloader loads to physical addresses. Let's show a simple approach:
```c
/* Add to vmm_init() after enabling paging */
/* Map higher-half kernel (0xC0000000+) to physical 0x00000000+ */
/* This is a 1:1 mapping shifted by 0xC0000000 */
void vmm_map_higher_half(void) {
    /* Map first 4MB at 0xC0000000 */
    /* Virtual 0xC0000000-0xC0400000 → Physical 0x00000000-0x00400000 */
    for (uint32_t offset = 0; offset < 0x00400000; offset += PAGE_SIZE) {
        vmm_map_page(0xC0000000 + offset, offset, PAGE_WRITABLE);
    }
    kprintf("Higher-half kernel mapped at 0xC0000000\n");
}
```
**Important:** The linker script must be updated to place code at virtual addresses starting at 0xC0100000:
```ld
/* kernel/linker.ld - Updated for higher-half kernel */
ENTRY(_start)
/* Start at 3GB + 1MB */
KERNEL_VIRTUAL_BASE = 0xC0100000;
KERNEL_PHYSICAL_BASE = 0x00100000;
SECTIONS
{
    /* Start at the virtual address */
    . = KERNEL_VIRTUAL_BASE;
    /* Text section */
    .text ALIGN(4K) : AT(ADDR(.text) - KERNEL_VIRTUAL_BASE + KERNEL_PHYSICAL_BASE)
    {
        *(.text)
        *(.text.*)
    }
    /* Read-only data */
    .rodata ALIGN(4K) : AT(ADDR(.rodata) - KERNEL_VIRTUAL_BASE + KERNEL_PHYSICAL_BASE)
    {
        *(.rodata)
        *(.rodata.*)
    }
    /* Initialized data */
    .data ALIGN(4K) : AT(ADDR(.data) - KERNEL_VIRTUAL_BASE + KERNEL_PHYSICAL_BASE)
    {
        *(.data)
    }
    /* BSS */
    .bss ALIGN(4K) : AT(ADDR(.bss) - KERNEL_VIRTUAL_BASE + KERNEL_PHYSICAL_BASE)
    {
        __bss_start = .;
        *(COMMON)
        *(.bss)
        __bss_end = .;
    }
    /* End of kernel */
    _kernel_end = .;
}
```
The `AT()` directive tells the linker: "this section's virtual address is X, but load it at physical address Y."
## Step 5: The Page Fault Handler

![Page Fault Handler Decision Tree](./diagrams/diag-page-fault-handler.svg)

When a program accesses an address without a valid page table entry (or violates access rights), the CPU triggers exception 14 — page fault. The handler receives:
- **CR2**: The faulting linear address (what was accessed)
- **Error code**: Bits describing why the fault occurred
Let's enhance our page fault handler from Milestone 2:
```c
/* Add to kernel/interrupts.c or create kernel/pagefault.c */
#include "memory.h"
#include "kprintf.h"
/* Page fault error code bits */
#define PF_PRESENT  (1 << 0)  /* 0 = page not present, 1 = protection violation */
#define PF_WRITE    (1 << 1)  /* 0 = read, 1 = write */
#define PF_USER     (1 << 2)  /* 0 = kernel mode, 1 = user mode */
#define PF_RESERVED (1 << 3)  /* Reserved bits set */
#define PF_FETCH    (1 << 4)  /* Instruction fetch (NX bit) */
/* Page fault handler - called from exception 14 */
void page_fault_handler(registers_t *regs) {
    /* Read faulting address from CR2 */
    uint32_t faulting_addr;
    __asm__ volatile ("mov %%cr2, %0" : "=r"(faulting_addr));
    /* Decode error code */
    uint32_t err = regs->err_code;
    bool present = (err & PF_PRESENT) != 0;
    bool write = (err & PF_WRITE) != 0;
    bool user = (err & PF_USER) != 0;
    bool reserved = (err & PF_RESERVED) != 0;
    bool fetch = (err & PF_FETCH) != 0;
    kprintf("\n========== PAGE FAULT ==========\n");
    kprintf("Faulting address: 0x%08x\n", faulting_addr);
    kprintf("Error code: 0x%x\n", err);
    kprintf("  Present: %d (%s)\n", present, 
            present ? "protection violation" : "page not present");
    kprintf("  Operation: %s\n", write ? "WRITE" : "READ");
    kprintf("  Mode: %s\n", user ? "USER" : "KERNEL");
    if (reserved) kprintf("  Reserved bits were set!\n");
    if (fetch) kprintf("  Instruction fetch (NX violation)\n");
    /* Print context */
    kprintf("EIP: 0x%08x\n", regs->eip);
    kprintf("Process was executing at: 0x%08x\n", regs->eip);
    /* Check if address is mapped */
    uint32_t phys = vmm_get_physical(faulting_addr);
    if (phys != 0) {
        kprintf("Address IS mapped to physical 0x%08x\n", phys);
        kprintf("This suggests a protection violation.\n");
    } else {
        kprintf("Address is NOT mapped.\n");
    }
    /* For now, halt. In the future, this could: */
    /* - Allocate a new page (demand paging) */
    /* - Copy a page (copy-on-write) */
    /* - Load from disk (swap in) */
    /* - Kill the process (segfault) */
    kprintf("\nHalting (future: should kill process or handle fault)\n");
    while (1) {
        __asm__ volatile ("cli; hlt");
    }
}
```
Register this handler in `interrupts_init()`:
```c
/* In interrupts.c */
#include "pagefault.h"  /* or just declare page_fault_handler */
void interrupts_init(void) {
    /* ... existing initialization ... */
    /* Register page fault handler */
    register_interrupt_handler(14, page_fault_handler);
    /* ... rest of init ... */
}
```
## Step 6: The Kernel Heap Allocator

![Kernel Heap Allocator Design](./diagrams/diag-kernel-heap.svg)

Now that we have virtual memory and page allocation, we can build a heap allocator. `kmalloc` needs to:
1. Manage a region of virtual address space
2. Allocate physical frames on demand
3. Track allocated blocks for `kfree`
For simplicity, we'll implement a **linked-list allocator** (similar to K&R malloc):
Create `kernel/kheap.c`:
```c
/* kernel/kheap.c - Kernel heap allocator */
#include "memory.h"
#include "pmm.h"
#include "kprintf.h"
/* Heap starts at 0xC0400000 (after higher-half kernel) */
#define KHEAP_START    0xC0400000
#define KHEAP_INITIAL  0x100000   /* 1 MB initial size */
#define KHEAP_MAX      0x1000000  /* 16 MB max */
/* Block header */
typedef struct block_header {
    uint32_t size;                  /* Size of the block (excluding header) */
    uint8_t  free;                  /* 1 if free, 0 if allocated */
    uint8_t  magic;                 /* Magic number for corruption detection */
    struct block_header *next;      /* Next block in the list */
} block_header_t;
#define HEADER_SIZE sizeof(block_header_t)
#define MAGIC_ALLOC 0xA1
#define MAGIC_FREE  0xF1
/* Heap state */
static uint8_t *heap_start = NULL;
static uint8_t *heap_end = NULL;
static uint8_t *heap_max = NULL;
static block_header_t *free_list = NULL;
/* Align size to 8 bytes */
#define ALIGN8(x) (((x) + 7) & ~7)
/* Initialize the kernel heap */
void kheap_init(void) {
    heap_start = (uint8_t *)KHEAP_START;
    heap_end = heap_start;
    heap_max = (uint8_t *)(KHEAP_START + KHEAP_MAX);
    /* Allocate initial heap pages */
    for (uint32_t addr = KHEAP_START; 
         addr < KHEAP_START + KHEAP_INITIAL; 
         addr += PAGE_SIZE) {
        void *frame = pmm_alloc_frame();
        if (frame == NULL) {
            kprintf("WARNING: Could not allocate initial heap frame\n");
            break;
        }
        vmm_map_page(addr, (uint32_t)frame, PAGE_WRITABLE);
    }
    heap_end = (uint8_t *)(KHEAP_START + KHEAP_INITIAL);
    /* Create initial free block spanning entire heap */
    free_list = (block_header_t *)heap_start;
    free_list->size = KHEAP_INITIAL - HEADER_SIZE;
    free_list->free = 1;
    free_list->magic = MAGIC_FREE;
    free_list->next = NULL;
    kprintf("Kernel heap initialized at 0x%08x, size: %d KB\n",
            KHEAP_START, KHEAP_INITIAL / 1024);
}
/* Expand heap by allocating more pages */
static void expand_heap(uint32_t min_size) {
    /* Calculate how many pages we need */
    uint32_t pages_needed = PAGE_ALIGN(min_size) / PAGE_SIZE;
    /* Check if we have room */
    if (heap_end + pages_needed * PAGE_SIZE > heap_max) {
        kprintf("WARNING: Heap expansion would exceed max size\n");
        return;
    }
    /* Allocate and map pages */
    for (uint32_t i = 0; i < pages_needed; i++) {
        void *frame = pmm_alloc_frame();
        if (frame == NULL) {
            kprintf("WARNING: Out of memory for heap expansion\n");
            return;
        }
        vmm_map_page((uint32_t)(heap_end + i * PAGE_SIZE), 
                     (uint32_t)frame, PAGE_WRITABLE);
    }
    /* Create a new free block for the expanded region */
    block_header_t *new_block = (block_header_t *)heap_end;
    new_block->size = pages_needed * PAGE_SIZE - HEADER_SIZE;
    new_block->free = 1;
    new_block->magic = MAGIC_FREE;
    new_block->next = free_list;
    free_list = new_block;
    heap_end += pages_needed * PAGE_SIZE;
}
/* Find a free block of at least the given size */
static block_header_t *find_free_block(uint32_t size) {
    block_header_t *current = free_list;
    block_header_t *prev = NULL;
    while (current != NULL) {
        if (current->magic != MAGIC_FREE) {
            kprintf("ERROR: Heap corruption detected (magic=%02x)\n", 
                    current->magic);
            return NULL;
        }
        if (current->free && current->size >= size) {
            /* Found a suitable block */
            if (prev != NULL) {
                prev->next = current->next;
            } else {
                free_list = current->next;
            }
            return current;
        }
        prev = current;
        current = current->next;
    }
    return NULL;  /* No suitable block found */
}
/* Allocate memory */
void *kmalloc(uint32_t size) {
    if (size == 0) return NULL;
    /* Align size */
    size = ALIGN8(size);
    /* Minimum block size */
    if (size < 8) size = 8;
    /* Find a free block */
    block_header_t *block = find_free_block(size);
    if (block == NULL) {
        /* No block found, try to expand heap */
        expand_heap(size + HEADER_SIZE);
        block = find_free_block(size);
        if (block == NULL) {
            kprintf("kmalloc: Out of memory (size=%d)\n", size);
            return NULL;
        }
    }
    /* Check if we should split the block */
    if (block->size >= size + HEADER_SIZE + 8) {
        /* Split the block */
        block_header_t *new_block = (block_header_t *)((uint8_t *)block + 
                                                        HEADER_SIZE + size);
        new_block->size = block->size - size - HEADER_SIZE;
        new_block->free = 1;
        new_block->magic = MAGIC_FREE;
        new_block->next = free_list;
        free_list = new_block;
        block->size = size;
    }
    /* Mark as allocated */
    block->free = 0;
    block->magic = MAGIC_ALLOC;
    /* Return pointer to the data area (after header) */
    return (void *)((uint8_t *)block + HEADER_SIZE);
}
/* Free memory */
void kfree(void *ptr) {
    if (ptr == NULL) return;
    /* Get the block header */
    block_header_t *block = (block_header_t *)((uint8_t *)ptr - HEADER_SIZE);
    /* Sanity check */
    if (block->magic != MAGIC_ALLOC) {
        kprintf("kfree: Invalid pointer or double free (magic=%02x)\n", 
                block->magic);
        return;
    }
    /* Mark as free */
    block->free = 1;
    block->magic = MAGIC_FREE;
    /* Add to free list */
    block->next = free_list;
    free_list = block;
    /* TODO: Coalesce adjacent free blocks */
}
```
### Hardware Soul: What Happens on kmalloc
Let's trace a `kmalloc(100)` call when the heap is empty:
```
1. kmalloc(100) called
   └── size = ALIGN8(100) = 104
   └── find_free_block(104) returns NULL (empty heap)
2. expand_heap(104 + 16) called
   └── pages_needed = 1 (104 bytes fits in one page)
   └── pmm_alloc_frame() → 0x00500000 (example)
   └── vmm_map_page(0xC0400000, 0x00500000, PAGE_WRITABLE)
   └── Create free block at 0xC0400000, size = 4096 - 16 = 4080
3. find_free_block(104) returns the new block
   └── Block is large enough to split
4. Split the block:
   └── Allocated block: 0xC0400000, size = 104
   └── New free block: 0xC0400078, size = 4080 - 104 - 16 = 3960
5. Return 0xC0400010 (address after header)
Hardware touched:
- PMM bitmap: one bit set to 1
- Page table entry: PTE at virtual 0xC0400000 set to map to 0x00500000
- TLB: one entry added for 0xC0400xxx
- Physical RAM: one 4KB frame allocated
```
## Step 7: Putting It All Together
Update `kernel/main.c` to initialize memory management:
```c
/* kernel/main.c - Updated with memory management */
#include "vga.h"
#include "serial.h"
#include "kprintf.h"
#include "idt.h"
#include "pic.h"
#include "timer.h"
#include "keyboard.h"
#include "interrupts.h"
#include "memory.h"
void kernel_main(void) {
    /* Initialize VGA and serial */
    vga_init();
    serial_init();
    kprintf("\n");
    kprintf("========================================\n");
    kprintf("  MyOS Kernel v0.3 - Memory Management\n");
    kprintf("  Built on %s at %s\n", __DATE__, __TIME__);
    kprintf("========================================\n\n");
    /* Initialize physical memory manager */
    /* For QEMU with 512MB, pass 512 * 1024 = 524288 KB */
    pmm_init(512 * 1024);
    /* Mark kernel as used */
    extern uint32_t _kernel_end;
    pmm_mark_used(0x100000, (uint32_t)&_kernel_end - 0x100000 + 0x10000);
    /* Initialize interrupt handling */
    kprintf("Initializing interrupts...\n");
    idt_init();
    pic_init();
    register_interrupt_handler(14, page_fault_handler);
    /* Enable interrupts */
    __asm__ volatile ("sti");
    /* Initialize timer */
    timer_init(100);
    keyboard_init();
    /* Initialize virtual memory management */
    kprintf("\nInitializing virtual memory...\n");
    vmm_init();
    /* Initialize kernel heap */
    kheap_init();
    kprintf("\n=== Memory Management Test ===\n");
    /* Test physical frame allocation */
    kprintf("\nTesting PMM:\n");
    void *frame1 = pmm_alloc_frame();
    void *frame2 = pmm_alloc_frame();
    void *frame3 = pmm_alloc_frame();
    kprintf("  Allocated frames: 0x%x, 0x%x, 0x%x\n",
            (uint32_t)frame1, (uint32_t)frame2, (uint32_t)frame3);
    pmm_free_frame(frame2);
    kprintf("  Freed frame: 0x%x\n", (uint32_t)frame2);
    void *frame4 = pmm_alloc_frame();
    kprintf("  Allocated frame: 0x%x (should reuse freed frame)\n",
            (uint32_t)frame4);
    kprintf("  Free frames: %d / %d\n", pmm_get_free_count(), 
            pmm_get_total_count());
    /* Test virtual memory mapping */
    kprintf("\nTesting VMM:\n");
    uint32_t test_virt = 0xD0000000;
    uint32_t test_phys = (uint32_t)frame1;
    vmm_map_page(test_virt, test_phys, PAGE_WRITABLE);
    kprintf("  Mapped 0x%08x -> 0x%08x\n", test_virt, test_phys);
    uint32_t translated = vmm_get_physical(test_virt);
    kprintf("  Translation check: 0x%08x -> 0x%08x\n", test_virt, translated);
    /* Test heap allocation */
    kprintf("\nTesting KHeap:\n");
    char *str = (char *)kmalloc(100);
    kprintf("  kmalloc(100) = 0x%08x\n", (uint32_t)str);
    if (str) {
        /* Write to the allocated memory */
        for (int i = 0; i < 50; i++) {
            str[i] = 'A' + (i % 26);
        }
        str[50] = '\0';
        kprintf("  Wrote string: %s\n", str);
    }
    char *str2 = (char *)kmalloc(200);
    kprintf("  kmalloc(200) = 0x%08x\n", (uint32_t)str2);
    kfree(str);
    kprintf("  kfree(0x%08x)\n", (uint32_t)str);
    char *str3 = (char *)kmalloc(50);
    kprintf("  kmalloc(50) = 0x%08x (should reuse freed block)\n", 
            (uint32_t)str3);
    kprintf("\n=== Memory management initialized successfully ===\n");
    /* Intentionally cause a page fault to test handler */
    kprintf("\nTo test page fault handler, press 'P'\n");
    kprintf("Press ESC to halt\n\n");
    while (1) {
        char c = keyboard_getchar();
        if (c == 27) {
            kprintf("\nESC pressed. Halting.\n");
            break;
        }
        if (c == 'p' || c == 'P') {
            kprintf("\nCausing intentional page fault...\n");
            /* Access unmapped memory */
            volatile uint32_t *bad = (volatile uint32_t *)0xDEADBEEF;
            *bad = 0x12345678;
        }
        kprintf("%c", c);
    }
    kprintf("\nSystem halted.\n");
    while (1) {
        __asm__ volatile ("cli; hlt");
    }
}
```
### Update the Makefile
```makefile
# Add to C_SOURCES
C_SOURCES = kernel/main.c kernel/vga.c kernel/serial.c kernel/kprintf.c \
            kernel/idt.c kernel/pic.c kernel/timer.c kernel/keyboard.c \
            kernel/interrupts.c kernel/pmm.c kernel/vmm.c kernel/kheap.c
```
### Update the Linker Script
Add the `_kernel_end` symbol:
```ld
/* In kernel/linker.ld, at the end */
_kernel_end = .;
```
## Hardware Soul: The Full Memory Access Path
When your kernel accesses memory with paging enabled, here's what happens at the hardware level:
```
Instruction: mov eax, [0xC0401234]
1. CPU extracts virtual address: 0xC0401234
2. MMU splits the address:
   - Page directory index: 0xC0401234 >> 22 = 0x301 (769)
   - Page table index: (0xC0401234 >> 12) & 0x3FF = 0x001 (1)
   - Page offset: 0xC0401234 & 0xFFF = 0x0234
3. TLB lookup:
   - Is virtual page 0xC0401xxx cached?
   - If YES: Skip to step 6 (huge speedup!)
   - If NO: Continue with page table walk
4. Page directory access (memory read):
   - Read CR3 → PD physical address (e.g., 0x00100000)
   - Read PDE at 0x00100000 + 769*4 = 0x00100C04
   - Extract page table physical address from PDE
5. Page table access (memory read):
   - Read PTE at (page_table_addr + 1*4)
   - Extract physical frame address from PTE
   - Cache this translation in TLB
6. Physical access:
   - Physical address = frame_addr + 0x0234
   - Memory controller reads from RAM at this address
   - Data returned to CPU
Timing (approximate):
- TLB hit: ~1-2 cycles (translation is free!)
- TLB miss: ~30-50 cycles (two additional memory reads)
- Page fault: ~1000+ cycles (trap to kernel, handle, return)
```
**Cache line analysis:**
- Page directory: 1 cache line (4KB = 64 cache lines, but only accessing 1-2)
- Page table: 1 cache line per table used
- TLB: ~64 entries on older x86, ~1536 on modern CPUs with L1/L2 TLB
**The TLB is the hero:** Without it, every memory access would require two additional memory accesses (one for PDE, one for PTE). That's a 3× slowdown minimum. The TLB makes virtual memory practical by caching the most recent translations.
## Common Pitfalls and Debugging
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Triple fault immediately after enabling paging | Code not identity-mapped | Ensure identity map covers executing code |
| Can't access VGA after enabling paging | VGA not mapped | Add VGA (0xB8000) to identity map |
| Page fault on valid address | PDE/PTE not present | Check page table allocation, verify mapping |
| Wrong physical address | Address calculation error | Verify bit shifting in PDE/PTE extraction |
| kmalloc returns garbage | Heap not initialized | Call kheap_init() after vmm_init() |
| Random crashes after kmalloc | Heap corruption | Check magic numbers, verify block splitting |
| Page fault handler causes page fault | Handler uses unmapped stack | Ensure kernel stack is mapped |
### QEMU Memory Debugging
```bash
# Monitor memory operations
qemu-system-i386 -fda os.img -serial stdio -d mmu
# Dump page tables (in QEMU monitor)
info mem
info tlb
# Check physical memory
pmemsave 0 0x1000000 memory_dump.bin
```
## Design Decisions Summary
| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Bitmap PMM** ✓ | Simple, deterministic | Linear search, fragmentation | Linux bootmem, hobby OSes |
| Buddy allocator | Fast, handles contiguous | Complex | Linux, FreeBSD |
| **Identity + Higher-half** ✓ | Clean user space, kernel always mapped | Linker script complexity | Linux, Windows, macOS |
| Pure identity map | Simple | No user/kernel separation | DOS, some embedded |
| **Linked-list heap** ✓ | Simple, standard | Fragmentation, no coalescing | K&R malloc, dlmalloc |
| Slab allocator | Fast, no fragmentation | Complex, needs size classes | Linux slab, SLUB |
## Knowledge Cascade
You've just built the foundation for process isolation and memory virtualization — concepts that appear everywhere in computing.
**Same Domain:**
- **Process isolation (next milestone)** — Page tables are the hardware enforcement of isolation. Each process gets its own page directory, making it impossible to access another process's memory. When you implement `fork()`, you'll copy page tables. When you implement `exec()`, you'll create new ones.
- **Copy-on-write** — By marking pages read-only and handling page faults, you can implement efficient `fork()`. Instead of copying all memory, you share page tables and only copy when a page is written. This is how Linux `fork()` achieves O(1) time complexity for large processes.
- **Demand paging** — Loading a program doesn't require reading the entire binary. Mark pages as "not present" and load them lazily on page fault. This is why large programs can start quickly — they only load what they actually use.
- **Memory-mapped files** — The same mechanism that isolates processes enables `mmap()`. A file is mapped into the address space; accessing that memory reads the file. Page faults trigger file reads. This is zero-copy I/O.
**Cross-Domain:**
- **Browser sandboxing** — Chrome's process-per-tab model uses page tables for isolation. Each renderer process has its own page tables, making it impossible to access another tab's memory. The same primitive you just built powers web security.
- **Container isolation** — Docker's namespaces and cgroups are built on top of page table isolation. Each container sees its own address space. The hypervisor/OS enforces this via page tables.
- **Database buffer pools** — PostgreSQL, MySQL, and SQLite all use memory-mapped I/O and virtual memory tricks. Understanding page tables explains why `mmap()` can outperform `read()` for large datasets.
- **Redis and huge pages** — Redis uses 2MB huge pages to reduce TLB pressure. With standard 4KB pages, a 1GB dataset needs 262,144 TLB entries. With 2MB pages, it needs only 512. Understanding page tables explains this 512× improvement.
- **Hypervisors** — VMware, KVM, and Hyper-V all use extended page tables (EPT) or nested page tables (NPT) to virtualize memory. The guest OS thinks it's managing page tables, but the hypervisor adds another translation layer.
**Historical Context:**
- The x86 two-level page table design dates to the 80386 (1985). Before that, the 80286 used segmentation for protection.
- 64-bit x86 uses four-level page tables (PML4 → PDPT → PD → PT) to handle 48-bit virtual addresses.
- The TLB was introduced with the 80386 and has grown from ~32 entries to 1536+ entries in modern CPUs.
**What You Can Now Build:**
- A `fork()` system call that copies page tables
- A `brk()`/`sbrk()` system call for user-space heap management
- An `mmap()` system call for memory-mapped files
- A swap implementation that writes pages to disk
- A more sophisticated heap allocator (slab, buddy, jemalloc-style)
---
[[CRITERIA_JSON: {"milestone_id": "build-os-m3", "criteria": ["Physical memory manager (PMM) initialized with total memory size in KB and frame bitmap allocated at kernel end with 4-byte alignment", "PMM implements bitmap allocator with one bit per 4KB frame: 1 = used, 0 = free, with INDEX_FROM_BIT and OFFSET_FROM_BIT macros for bit manipulation", "pmm_alloc_frame() performs linear search for free frame, sets bit in bitmap, decrements free count, returns physical address or NULL if exhausted", "pmm_free_frame() validates frame number is within bounds, checks for double-free via TEST_FRAME, clears bit and increments free count", "pmm_mark_used() marks arbitrary physical address ranges as used by iterating through frames and setting bits with free count decrement", "PMM marks first 1MB (256 frames) as used at initialization for BIOS, VGA, kernel, and boot structures", "PMM marks its own bitmap memory as used to prevent overwriting", "Page directory allocated as single 4KB frame with 1024 32-bit entries, zeroed at initialization", "vmm_map_page() extracts PDE index (bits 22-31) and PTE index (bits 12-21) from virtual address", "vmm_map_page() allocates new page table via pmm_alloc_frame() if PDE present bit is 0", "vmm_map_page() sets PTE with physical frame address (bits 12-31) ORed with flags (PRESENT, WRITABLE, USER)", "vmm_unmap_page() clears PTE to 0 and calls invlpg for the virtual address", "vmm_get_physical() performs two-level page table walk to translate virtual to physical address, returns 0 if not mapped", "vmm_flush_tlb() executes invlpg instruction for single-page TLB invalidation", "vmm_flush_tlb_all() reloads CR3 to flush all non-global TLB entries", "Identity mapping implemented for first 4MB (0x00000000-0x00400000) before paging enabled", "vmm_init() creates page directory, identity-maps first 4MB, loads CR3, then sets CR0.PG bit to enable paging", "Paging enabled by loading CR3 with page directory physical address, reading CR0, setting bit 31 (PG), writing CR0", "Higher-half mapping function maps 0xC0000000+ to physical 0x00000000+ for kernel virtual addresses", "Page fault handler reads CR2 register to get faulting linear address", "Page fault handler decodes error code bits: bit 0 (present vs protection), bit 1 (read vs write), bit 2 (kernel vs user)", "Page fault handler prints diagnostic: faulting address from CR2, error code, EIP, and whether address is currently mapped", "Kernel heap initialized at fixed virtual address (e.g., 0xC0400000) with initial pages allocated via pmm_alloc_frame and vmm_map_page", "Heap uses linked-list allocator with block_header_t containing size, free flag, magic byte, and next pointer", "kmalloc() aligns request size to 8 bytes, searches free list for suitable block, splits block if remaining size >= header + 8 bytes", "kmalloc() calls expand_heap() if no suitable free block found, allocating additional pages", "kfree() validates magic byte to detect corruption and double-free, adds block to free list head", "Block headers use magic bytes (0xA1 for allocated, 0xF1 for free) for corruption detection", "VGA buffer at 0xB8000 remains accessible via identity map after paging enabled", "Linker script defines _kernel_end symbol for PMM bitmap placement", "Linker script supports higher-half kernel with AT() directive for separate virtual and physical load addresses"]}]
<!-- END_MS -->


<!-- MS_ID: build-os-m4 -->
# Milestone 4: Processes and Preemptive Scheduling

![OS Kernel Architecture Map](./diagrams/diag-satellite-os.svg)

Your kernel boots. It handles interrupts. It manages memory through page tables. But it's still a single-threaded program — one instruction stream, one stack, one destiny.
This milestone transforms your kernel into a **multitasking operating system**. You will give the illusion of parallelism on a single CPU, isolate processes from each other, and create the mechanism for user programs to request kernel services.
## The Fundamental Tension: One CPU, Many Illusions
Here's the brutal reality you must confront:
**Hardware constraint:**
- A single CPU core executes exactly one instruction at a time
- There's no "running two things at once" — only rapid switching
- Each switch costs time: save registers, change page tables, flush TLB, restore registers
**Software expectation:**
- Multiple programs want to run "simultaneously"
- Each program wants its own isolated address space
- No program should be able to crash another or the kernel
- Programs need to request privileged operations (I/O, memory allocation)
**The illusion:**
When you run three programs and they all appear to execute at once, what's actually happening is:
1. Timer interrupt fires (every 10ms at 100Hz)
2. Current process's state is frozen into its Process Control Block
3. Scheduler picks the next process
4. New process's state is restored from its PCB
5. Resume execution — the process has no idea it was paused
The tension: **the cost of switching must be low enough that the illusion holds, but the isolation must be strong enough that one process cannot corrupt another.**

![Context Switch Assembly Flow](./diagrams/diag-context-switch.svg)

## The Revelation: Multitasking Is NOT Parallelism
Here's the misconception that trips up everyone:
> "Multitasking means multiple processes run at the same time."
**No.** On a single core, **only one process runs at any instant**. Multitasking is rapid serial execution, so fast that humans perceive it as parallelism.
When you see three terminal windows each running a program, here's what's actually happening:
```
Time    What's Running
─────────────────────────────────────────────────────
0ms     Process A (terminal 1)
10ms    Process B (terminal 2)  ← timer interrupt, switch
20ms    Process C (terminal 3)  ← timer interrupt, switch
30ms    Process A (terminal 1)  ← timer interrupt, switch
40ms    Process B (terminal 2)  ← timer interrupt, switch
...
```
Each process gets a 10ms "time slice." When its slice ends, the timer interrupt triggers the scheduler, which performs a **context switch** — saving every register to the current process's PCB and loading the next process's PCB.
The magic is that **the process has no idea**. When it resumes, its registers are exactly as they were, its stack pointer points to the same location, its instruction pointer continues from the next instruction. It's as if it never stopped.
## The Revelation: The TSS Is NOT for Task Switching
Here's another misconception that causes endless confusion:
> "The Task State Segment (TSS) is used by the CPU for task switching."
**No.** Intel designed the TSS for hardware task switching in the 80286/386 era. Almost no modern OS uses this feature. Linux, Windows, macOS — they all perform **software context switching**.
So why does the TSS exist in your OS? For one specific purpose: **defining the kernel stack for privilege transitions**.
When a user-mode process (Ring 3) triggers an interrupt or system call, the CPU needs to switch to a kernel stack (Ring 0). It can't use the user stack — that would be a security disaster (user code could corrupt kernel data). The CPU looks up `TSS.ESP0` to find the kernel stack for this transition.
That's it. The TSS is a 104-byte structure that exists solely to hold `SS0` and `ESP0`. You update `ESP0` on every context switch so that the current process's kernel stack is used when it traps to the kernel.

![Task State Segment (TSS) Layout](./diagrams/diag-tss-structure.svg)

## Three-Level View of Context Switching
### Level 1 — Application View
A process calls a function, loops, computes, sleeps. It has no awareness of being paused and resumed. Time appears continuous. The only hint is that `clock()` might jump forward unexpectedly.
### Level 2 — OS/Kernel View (What You're Building)
The kernel maintains a list of processes, each with a PCB containing its frozen state. The timer interrupt triggers the scheduler, which:
1. Saves current process registers to its PCB
2. Updates `TSS.ESP0` to the next process's kernel stack
3. Switches CR3 if page directories differ
4. Loads next process's registers from its PCB
5. Returns to the new process via `iret`
### Level 3 — Hardware View
- **Timer interrupt**: IRQ0 fires at 100Hz, pushing current state to stack
- **Context switch**: Software saves additional registers, modifies CR3, updates TSS
- **TLB flush**: When CR3 changes, non-global TLB entries are invalidated
- **`iret`**: Pops EIP, CS, EFLAGS (and SS:ESP if privilege changes), resuming execution

![Round-Robin Scheduler Flow](./diagrams/diag-scheduler-flow.svg)

## Step 1: The Process Control Block
The PCB is the kernel's record of everything needed to resume a process. It must contain:
- **PID**: Unique process identifier
- **State**: READY, RUNNING, BLOCKED, TERMINATED
- **Registers**: EIP, ESP, EBP, EAX, EBX, ECX, EDX, ESI, EDI, EFLAGS
- **Page directory**: Physical address of this process's page directory (CR3)
- **Kernel stack**: Pointer to this process's kernel stack
- **User stack**: Pointer to this process's user stack (if in user mode)

![Process Control Block Structure](./diagrams/diag-pcb-structure.svg)

Create `kernel/process.h`:
```c
/* kernel/process.h - Process management */
#ifndef PROCESS_H
#define PROCESS_H
#include <stdint.h>
#include <stdbool.h>
#include "memory.h"
/* Process states */
typedef enum {
    PROCESS_STATE_READY,
    PROCESS_STATE_RUNNING,
    PROCESS_STATE_BLOCKED,
    PROCESS_STATE_TERMINATED,
} process_state_t;
/* Process Control Block */
typedef struct process {
    /* Identification */
    uint32_t pid;
    char name[32];
    /* Scheduling state */
    process_state_t state;
    uint32_t priority;
    uint32_t time_slice;          /* Remaining time slice in ticks */
    /* Register state (saved on context switch) */
    uint32_t eax, ebx, ecx, edx;
    uint32_t esi, edi, ebp, esp;
    uint32_t eip;
    uint32_t eflags;
    /* Memory management */
    uint32_t page_directory;      /* Physical address of page directory */
    uint32_t kernel_stack;        /* Kernel stack (used during syscalls) */
    uint32_t user_stack;          /* User stack (for user-mode processes) */
    /* Entry point */
    uint32_t entry_point;
    /* Linked list of processes */
    struct process *next;
    struct process *prev;
} process_t;
/* Process functions */
void process_init(void);
process_t *process_create(const char *name, void (*entry_point)(void), 
                          bool user_mode);
void process_terminate(process_t *proc);
process_t *process_get_current(void);
process_t *process_get_by_pid(uint32_t pid);
/* Scheduler functions */
void scheduler_init(void);
void scheduler_tick(registers_t *regs);
void scheduler_yield(void);
void scheduler_block(process_t *proc);
void scheduler_unblock(process_t *proc);
/* Context switch (assembly) */
extern void context_switch(process_t *current, process_t *next);
#endif /* PROCESS_H */
```
## Step 2: Creating the First Process
Before we can schedule, we need processes. Let's implement `process_create`:
Create `kernel/process.c`:
```c
/* kernel/process.c - Process management implementation */
#include "process.h"
#include "memory.h"
#include "pmm.h"
#include "vmm.h"
#include "kprintf.h"
#include "string.h"
/* Process list */
static process_t *process_list = NULL;
static process_t *current_process = NULL;
static uint32_t next_pid = 1;
/* Kernel stack size per process */
#define KERNEL_STACK_SIZE 0x2000  /* 8KB */
#define USER_STACK_SIZE   0x10000 /* 64KB */
/* Simple string functions */
static void strcpy_simple(char *dest, const char *src) {
    while (*src) *dest++ = *src++;
    *dest = '\0';
}
/* Initialize process management */
void process_init(void) {
    process_list = NULL;
    current_process = NULL;
    next_pid = 1;
    /* Create the "idle" process (PID 0) for when no other process is ready */
    /* This is actually the kernel itself, continuing after initialization */
    kprintf("Process management initialized\n");
}
/* Create a new process */
process_t *process_create(const char *name, void (*entry_point)(void), 
                          bool user_mode) {
    /* Allocate PCB */
    process_t *proc = (process_t *)kmalloc(sizeof(process_t));
    if (proc == NULL) {
        kprintf("ERROR: Failed to allocate PCB\n");
        return NULL;
    }
    /* Initialize PCB */
    memset(proc, 0, sizeof(process_t));
    proc->pid = next_pid++;
    strcpy_simple(proc->name, name);
    proc->state = PROCESS_STATE_READY;
    proc->priority = 1;
    proc->time_slice = 10;  /* 10 ticks = 100ms at 100Hz */
    /* Set entry point */
    proc->entry_point = (uint32_t)entry_point;
    proc->eip = (uint32_t)entry_point;
    /* Allocate kernel stack */
    void *kernel_stack = kmalloc(KERNEL_STACK_SIZE);
    if (kernel_stack == NULL) {
        kprintf("ERROR: Failed to allocate kernel stack\n");
        kfree(proc);
        return NULL;
    }
    proc->kernel_stack = (uint32_t)kernel_stack + KERNEL_STACK_SIZE;
    if (user_mode) {
        /* Allocate user stack (in user memory region) */
        /* For simplicity, we'll allocate it from kernel heap and map it */
        void *user_stack = kmalloc(USER_STACK_SIZE);
        if (user_stack == NULL) {
            kprintf("ERROR: Failed to allocate user stack\n");
            kfree(kernel_stack);
            kfree(proc);
            return NULL;
        }
        proc->user_stack = (uint32_t)user_stack + USER_STACK_SIZE;
        /* Create a new page directory for this process */
        proc->page_directory = (uint32_t)pmm_alloc_frame();
        if (proc->page_directory == 0) {
            kprintf("ERROR: Failed to allocate page directory\n");
            kfree(user_stack);
            kfree(kernel_stack);
            kfree(proc);
            return NULL;
        }
        /* Copy kernel page directory entries (higher half) */
        uint32_t *kernel_pd = (uint32_t *)0xFFFFF000; /* Current PD (recursive mapping) */
        uint32_t *new_pd = (uint32_t *)proc->page_directory;
        /* Identity map lower memory for now */
        for (int i = 0; i < 256; i++) {  /* First 1MB */
            new_pd[i] = kernel_pd[i];
        }
        /* Copy kernel mappings (higher half) */
        for (int i = 768; i < 1024; i++) {  /* 0xC0000000+ */
            new_pd[i] = kernel_pd[i];
        }
        /* Set EFLAGS with interrupts enabled */
        proc->eflags = 0x202;  /* IF=1, reserved bit 1 */
        /* Set up initial stack for iret to user mode */
        /* We'll push user context onto kernel stack */
        uint32_t *stack = (uint32_t *)proc->kernel_stack;
        /* User stack pointer */
        *--stack = proc->user_stack;     /* SS (user data segment) */
        *--stack = USER_DATA_SEL;        /* SS */
        *--stack = proc->eflags;         /* EFLAGS */
        *--stack = USER_CODE_SEL;        /* CS (user code segment) */
        *--stack = proc->eip;            /* EIP (entry point) */
        /* General purpose registers (initial state) */
        *--stack = 0; /* EAX */
        *--stack = 0; /* ECX */
        *--stack = 0; /* EDX */
        *--stack = 0; /* EBX */
        *--stack = 0; /* ESP (will be overwritten) */
        *--stack = proc->user_stack; /* EBP */
        *--stack = 0; /* ESI */
        *--stack = 0; /* EDI */
        proc->esp = (uint32_t)stack;
        proc->ebp = proc->user_stack;
    } else {
        /* Kernel-mode process */
        proc->page_directory = 0;  /* Use kernel's page directory */
        /* Set up initial stack for kernel process */
        uint32_t *stack = (uint32_t *)proc->kernel_stack;
        /* For kernel processes, we set up to return to entry_point */
        /* Push a fake "return address" that points to process_exit */
        *--stack = (uint32_t)process_exit_handler;
        *--stack = proc->eip;  /* "Return" to entry point */
        /* General purpose registers */
        *--stack = 0; /* EAX */
        *--stack = 0; /* ECX */
        *--stack = 0; /* EDX */
        *--stack = 0; /* EBX */
        *--stack = (uint32_t)stack + 20; /* ESP */
        *--stack = (uint32_t)stack + 20; /* EBP */
        *--stack = 0; /* ESI */
        *--stack = 0; /* EDI */
        /* EFLAGS with interrupts enabled */
        proc->eflags = 0x202;
        /* Push EFLAGS, CS, EIP for iret */
        *--stack = proc->eflags;
        *--stack = KERNEL_CODE_SEL;
        *--stack = proc->eip;
        proc->esp = (uint32_t)stack;
        proc->ebp = (uint32_t)stack + 20;
    }
    /* Add to process list */
    proc->next = process_list;
    proc->prev = NULL;
    if (process_list != NULL) {
        process_list->prev = proc;
    }
    process_list = proc;
    kprintf("Created process '%s' (PID %d) at 0x%x, %s mode\n",
            name, proc->pid, proc->eip, 
            user_mode ? "user" : "kernel");
    return proc;
}
/* Process exit handler (called when process returns from entry point) */
void process_exit_handler(void) {
    kprintf("Process %d exited\n", current_process->pid);
    process_terminate(current_process);
    scheduler_yield();  /* Never returns */
    while (1);          /* Should never reach here */
}
/* Terminate a process */
void process_terminate(process_t *proc) {
    if (proc == NULL) return;
    proc->state = PROCESS_STATE_TERMINATED;
    /* Remove from process list */
    if (proc->prev != NULL) {
        proc->prev->next = proc->next;
    } else {
        process_list = proc->next;
    }
    if (proc->next != NULL) {
        proc->next->prev = proc->prev;
    }
    /* Free resources (TODO: free stacks, page directory) */
    /* For now, just mark as terminated */
    /* If this was the current process, schedule another */
    if (proc == current_process) {
        current_process = NULL;
    }
}
/* Get current process */
process_t *process_get_current(void) {
    return current_process;
}
/* Get process by PID */
process_t *process_get_by_pid(uint32_t pid) {
    process_t *proc = process_list;
    while (proc != NULL) {
        if (proc->pid == pid) return proc;
        proc = proc->next;
    }
    return NULL;
}
```
## Step 3: The Context Switch — Assembly Magic
The context switch must be written in assembly because:
1. We need to save/restore ALL registers atomically
2. We can't rely on C's calling convention (it uses some registers)
3. We need precise control over stack manipulation

![Context Switch Assembly Flow](./diagrams/diag-context-switch.svg)

Create `kernel/context_switch.asm`:
```nasm
; kernel/context_switch.asm - Context switch implementation
[BITS 32]
; External symbols
[EXTERN current_process]
[EXTERN tss_esp0]
; context_switch(process_t *current, process_t *next)
; Saves current process state and loads next process state
[GLOBAL context_switch]
context_switch:
    ; Arguments:
    ;   [esp+4]  = current process PCB (may be NULL)
    ;   [esp+8]  = next process PCB
    push ebp
    mov ebp, esp
    ; Save current process state (if current != NULL)
    mov eax, [ebp+8]        ; current process
    test eax, eax
    jz .load_next           ; Skip save if no current process
    ; Save general-purpose registers
    mov [eax+16], ebx       ; pcb.ebx (offset 16)
    mov [eax+20], ecx       ; pcb.ecx
    mov [eax+24], edx       ; pcb.edx
    mov [eax+28], esi       ; pcb.esi
    mov [eax+32], edi       ; pcb.edi
    mov [eax+36], ebp       ; pcb.ebp
    ; Save ESP (stack pointer after we pushed ebp)
    mov [eax+40], esp       ; pcb.esp
    ; Save EIP by calculating return address
    ; When we return from this function, we'll go back to scheduler
    ; For now, save the return address
    mov ecx, [ebp+4]        ; Return address
    mov [eax+44], ecx       ; pcb.eip
    ; Save EFLAGS
    pushfd
    pop ecx
    mov [eax+48], ecx       ; pcb.eflags
    ; EAX is saved in the PCB structure's first register field
    mov ecx, [ebp+12]       ; Save original EAX value from stack
    ; Actually, we need to save EAX before we used it...
    ; Let's restructure this
.load_next:
    ; Load next process
    mov eax, [ebp+12]       ; next process PCB
    test eax, eax
    jz .panic               ; No process to run!
    ; Update current_process pointer
    mov [current_process], eax
    ; Check if we need to switch page directories
    mov ecx, [eax+56]       ; pcb.page_directory
    test ecx, ecx
    jz .same_pd             ; 0 means use kernel's PD
    ; Switch page directory
    mov cr3, ecx
.same_pd:
    ; Update TSS.ESP0 for this process's kernel stack
    mov ecx, [eax+60]       ; pcb.kernel_stack
    mov [tss_esp0], ecx
    ; Restore next process's registers
    mov ebx, [eax+16]       ; pcb.ebx
    mov ecx, [eax+20]       ; pcb.ecx
    mov edx, [eax+24]       ; pcb.edx
    mov esi, [eax+28]       ; pcb.esi
    mov edi, [eax+32]       ; pcb.edi
    mov ebp, [eax+36]       ; pcb.ebp
    ; Restore EFLAGS
    push dword [eax+48]     ; pcb.eflags
    popfd
    ; Restore EIP and ESP by returning to the saved context
    mov esp, [eax+40]       ; pcb.esp
    jmp [eax+44]            ; pcb.eip
.panic:
    ; No process to run - this should never happen
    cli
    hlt
    jmp .panic
```
Wait, this approach has issues. The context switch is tricky because we're switching stacks mid-function. Let me rewrite this more carefully:
```nasm
; kernel/context_switch.asm - Context switch implementation (revised)
[BITS 32]
; PCB field offsets (must match struct process in process.h)
PCB_PID           equ 0
PCB_STATE         equ 4
PCB_EAX           equ 8
PCB_EBX           equ 12
PCB_ECX           equ 16
PCB_EDX           equ 20
PCB_ESI           equ 24
PCB_EDI           equ 28
PCB_EBP           equ 32
PCB_ESP           equ 36
PCB_EIP           equ 40
PCB_EFLAGS        equ 44
PCB_PAGE_DIR      equ 48
PCB_KERNEL_STACK  equ 52
; External symbols
[GLOBAL context_switch]
[EXTERN tss_update_esp0]
; context_switch(process_t *old_proc, process_t *new_proc)
; This function does NOT return normally!
; It saves old state, then resumes at new_proc's saved EIP
context_switch:
    ; Prologue - save old base pointer
    push ebp
    mov ebp, esp
    push edi
    push esi
    push ebx
    ; Get arguments
    mov eax, [ebp+8]        ; old_proc
    mov edx, [ebp+12]       ; new_proc
    ; === SAVE OLD PROCESS STATE ===
    test eax, eax
    jz .load_new            ; Skip if old_proc is NULL
    ; Save callee-saved registers to PCB
    mov [eax+PCB_EBX], ebx
    mov [eax+PCB_ESI], esi
    mov [eax+PCB_EDI], edi
    mov [eax+PCB_EBP], ebp
    ; Save ESP (current stack pointer after saved registers)
    lea ecx, [ebp+12]       ; ESP value before we pushed anything
    mov [eax+PCB_ESP], ecx
    ; Save return address as EIP
    mov ecx, [ebp+4]        ; Return address from stack
    mov [eax+PCB_EIP], ecx
    ; Save EFLAGS
    pushfd
    pop ecx
    mov [eax+PCB_EFLAGS], ecx
    ; Save EAX (caller-saved, but we need to preserve it)
    mov ecx, [ebp-4]        ; Actually this is wrong...
    ; Let's save EAX before we modify it
    mov [eax+PCB_EAX], eax  ; This saves the PCB pointer, not EAX!
    ; Actually, EAX is the PCB pointer now. We need to save the original
    ; EAX value from before the function call. This is tricky...
    ; The standard approach: caller saves caller-saved registers,
    ; callee saves callee-saved registers. EAX, ECX, EDX are caller-saved.
    ; For simplicity, we'll save all registers explicitly
.load_new:
    ; === LOAD NEW PROCESS STATE ===
    test edx, edx
    jz .panic               ; Can't switch to NULL process
    ; Check if page directory needs to change
    mov ecx, [edx+PCB_PAGE_DIR]
    test ecx, ecx
    jz .skip_cr3            ; 0 = use current (kernel) PD
    ; Switch page directory
    mov cr3, ecx
.skip_cr3:
    ; Update TSS.ESP0 for ring transitions
    mov ecx, [edx+PCB_KERNEL_STACK]
    push ecx
    call tss_update_esp0
    add esp, 4
    ; Restore callee-saved registers from new PCB
    mov ebx, [edx+PCB_EBX]
    mov esi, [edx+PCB_ESI]
    mov edi, [edx+PCB_EDI]
    mov ebp, [edx+PCB_EBP]
    ; Restore EFLAGS
    push dword [edx+PCB_EFLAGS]
    popfd
    ; Restore ESP and return to saved EIP
    mov esp, [edx+PCB_ESP]
    ret                     ; "Return" to new process's EIP
.panic:
    ; No process to switch to - halt
    cli
.halt_loop:
    hlt
    jmp .halt_loop
```
This is getting complex. Let me take a cleaner approach — the context switch should be simpler:
```nasm
; kernel/context_switch.asm - Clean implementation
[BITS 32]
; Offsets into process_t structure (bytes)
; Must match process.h exactly
OFFSET_EAX        equ 8
OFFSET_EBX        equ 12
OFFSET_ECX        equ 16
OFFSET_EDX        equ 20
OFFSET_ESI        equ 24
OFFSET_EDI        equ 28
OFFSET_EBP        equ 32
OFFSET_ESP        equ 36
OFFSET_EIP        equ 40
OFFSET_EFLAGS     equ 44
OFFSET_PAGE_DIR   equ 48
OFFSET_KSTACK     equ 52
section .text
[GLOBAL context_switch]
[EXTERN current_process]
[EXTERN tss_esp0_ptr]
; void context_switch(process_t *old, process_t *new)
; Does not return in the normal sense - resumes at new->eip
context_switch:
    ; [esp+4] = old process (may be NULL)
    ; [esp+8] = new process
    mov eax, [esp+4]        ; old process
    mov edx, [esp+8]        ; new process
    ; === SAVE OLD PROCESS STATE ===
    test eax, eax
    jz .load_new
    ; Save all general-purpose registers
    mov [eax+OFFSET_EBX], ebx
    mov [eax+OFFSET_ECX], ecx
    mov [eax+OFFSET_EDX], edx
    ; EAX saved later (contains PCB pointer)
    mov [eax+OFFSET_ESI], esi
    mov [eax+OFFSET_EDI], edi
    mov [eax+OFFSET_EBP], ebp
    ; Save stack pointer (after return address)
    mov ecx, esp
    add ecx, 4              ; Point to return address
    mov [eax+OFFSET_ESP], ecx
    ; Save return address as EIP
    mov ecx, [esp]
    mov [eax+OFFSET_EIP], ecx
    ; Save EFLAGS
    pushfd
    pop ecx
    mov [eax+OFFSET_EFLAGS], ecx
    ; Save EAX (current PCB pointer) - we're done with old process
    mov ecx, eax
    mov eax, [esp+4]        ; Get old PCB pointer again
    mov [eax+OFFSET_EAX], ecx
.load_new:
    ; === LOAD NEW PROCESS STATE ===
    test edx, edx
    jz .panic
    ; Update current_process global
    ; (This is done in C before calling context_switch)
    ; Switch page directory if needed
    mov ecx, [edx+OFFSET_PAGE_DIR]
    test ecx, ecx
    jz .no_pd_switch
    mov cr3, ecx
.no_pd_switch:
    ; Update TSS.ESP0
    mov ecx, [edx+OFFSET_KSTACK]
    mov [tss_esp0_ptr], ecx
    ; Restore registers
    mov ebx, [edx+OFFSET_EBX]
    mov ecx, [edx+OFFSET_ECX]
    mov esi, [edx+OFFSET_ESI]
    mov edi, [edx+OFFSET_EDI]
    mov ebp, [edx+OFFSET_EBP]
    ; Restore EFLAGS
    push dword [edx+OFFSET_EFLAGS]
    popfd
    ; Restore EAX
    mov eax, [edx+OFFSET_EAX]
    ; Restore EDX (we're about to clobber it anyway)
    ; Actually, we need EDX for the PCB pointer still
    ; Switch stacks and jump to new EIP
    mov esp, [edx+OFFSET_ESP]
    ret                     ; Jumps to [new->eip]
.panic:
    cli
.loop:
    hlt
    jmp .loop
```
## Step 4: The Task State Segment (TSS)
The TSS is required for Ring 3 → Ring 0 transitions. When a user-mode process executes `int 0x80` for a system call, the CPU:
1. Looks up the IDT entry for vector 0x80
2. Sees it's a Ring 0 handler
3. Loads SS:ESP from TSS.SS0:TSS.ESP0
4. Pushes user SS, ESP, EFLAGS, CS, EIP onto the kernel stack
5. Loads CS:EIP from the IDT gate

![Ring 3 → Ring 0 Transition Stack Change](./diagrams/diag-ring-transition.svg)

Create `kernel/tss.h`:
```c
/* kernel/tss.h - Task State Segment */
#ifndef TSS_H
#define TSS_H
#include <stdint.h>
/* TSS structure (104 bytes) */
typedef struct {
    uint16_t link;     uint16_t _pad0;
    uint32_t esp0;            /* Stack pointer for ring 0 */
    uint16_t ss0;     uint16_t _pad1;
    uint32_t esp1;            /* Stack pointer for ring 1 (unused) */
    uint16_t ss1;     uint16_t _pad2;
    uint32_t esp2;            /* Stack pointer for ring 2 (unused) */
    uint16_t ss2;     uint16_t _pad3;
    uint32_t cr3;             /* Page directory (not used for software switching) */
    uint32_t eip;
    uint32_t eflags;
    uint32_t eax, ecx, edx, ebx;
    uint32_t esp, ebp, esi, edi;
    uint16_t es;      uint16_t _pad4;
    uint16_t cs;      uint16_t _pad5;
    uint16_t ss;      uint16_t _pad6;
    uint16_t ds;      uint16_t _pad7;
    uint16_t fs;      uint16_t _pad8;
    uint16_t gs;      uint16_t _pad9;
    uint16_t ldtr;    uint16_t _pad10;
    uint16_t _pad11;
    uint16_t iomap_base;      /* I/O permission bitmap offset */
} __attribute__((packed)) tss_t;
/* Initialize the TSS */
void tss_init(void);
/* Update ESP0 for the current process */
void tss_update_esp0(uint32_t esp0);
/* Get pointer to TSS.ESP0 for assembly access */
extern uint32_t *tss_esp0_ptr;
#endif /* TSS_H */
```
Create `kernel/tss.c`:
```c
/* kernel/tss.c - Task State Segment implementation */
#include "tss.h"
#include "gdt.h"
#include "kprintf.h"
/* The TSS */
static tss_t tss;
/* Pointer to ESP0 for assembly access */
uint32_t *tss_esp0_ptr = &tss.esp0;
/* Initialize the TSS */
void tss_init(void) {
    /* Zero the TSS */
    for (uint32_t i = 0; i < sizeof(tss_t); i++) {
        ((uint8_t *)&tss)[i] = 0;
    }
    /* Set up kernel stack segment */
    tss.ss0 = 0x10;  /* Kernel data segment selector */
    tss.esp0 = 0x90000;  /* Will be updated on context switch */
    /* Set I/O bitmap base (outside TSS = no I/O bitmap) */
    tss.iomap_base = sizeof(tss_t);
    /* Add TSS to GDT (as a system segment) */
    /* We need to add a TSS descriptor to the GDT */
    /* GDT index 5 = selector 0x28 */
    gdt_set_tss_gate(5, (uint32_t)&tss, sizeof(tss_t) - 1);
    /* Load TR (Task Register) with TSS selector */
    __asm__ volatile ("ltr %0" : : "r"((uint16_t)0x28));
    kprintf("TSS initialized at 0x%x, TR=0x28\n", (uint32_t)&tss);
}
/* Update ESP0 for the current process */
void tss_update_esp0(uint32_t esp0) {
    tss.esp0 = esp0;
}
```
We need to add a function to set the TSS gate in the GDT. Update `kernel/gdt.h` and `kernel/gdt.c`:
```c
/* Add to kernel/gdt.h */
void gdt_set_tss_gate(uint32_t index, uint32_t base, uint32_t limit);
/* Add to kernel/gdt.c */
void gdt_set_tss_gate(uint32_t index, uint32_t base, uint32_t limit) {
    /* TSS descriptor format (system segment, type 0x89 = available 32-bit TSS) */
    uint64_t entry = 0;
    /* Base address (bits 0-23 and 56-63) */
    entry |= (base & 0xFF000000) << 32;  /* Base 31:24 at bits 56-63 */
    entry |= (base & 0x00FF0000) << 16;  /* Base 23:16 at bits 40-47 */
    entry |= (base & 0x0000FFFF);        /* Base 15:0 at bits 16-31 */
    /* Limit (bits 0-15 and 48-51) */
    entry |= (limit & 0x000F0000) << 32; /* Limit 19:16 at bits 48-51 */
    entry |= (limit & 0x0000FFFF) << 32; /* Limit 15:0 at bits 32-47 */
    /* Access byte: Present=1, DPL=0, System=0 (system segment), Type=1001 (32-bit TSS) */
    entry |= 0x890000000000ULL;          /* At bits 40-47 */
    /* Flags: granularity=0 (byte), 32-bit=1 */
    entry |= 0x000000000000ULL;
    /* Write to GDT */
    gdt[index] = entry;
}
```
Wait, we need access to the GDT array. Let me restructure this. The GDT was defined in assembly. We need to make it accessible from C.
Let's create a cleaner GDT interface:
```c
/* kernel/gdt.h - Revised */
#ifndef GDT_H
#define GDT_H
#include <stdint.h>
/* Selector constants */
#define KERNEL_CODE_SEL 0x08
#define KERNEL_DATA_SEL 0x10
#define USER_CODE_SEL   0x18
#define USER_DATA_SEL   0x20
#define TSS_SEL         0x28
/* GDT entry type */
typedef struct {
    uint16_t limit_low;
    uint16_t base_low;
    uint8_t  base_middle;
    uint8_t  access;
    uint8_t  granularity;
    uint8_t  base_high;
} __attribute__((packed)) gdt_entry_t;
/* GDT pointer */
typedef struct {
    uint16_t limit;
    uint32_t base;
} __attribute__((packed)) gdt_ptr_t;
/* Initialize GDT */
void gdt_init(void);
/* Set a GDT entry */
void gdt_set_gate(uint32_t index, uint32_t base, uint32_t limit, 
                  uint8_t access, uint8_t granularity);
/* Set TSS entry */
void gdt_set_tss(uint32_t index, uint32_t base, uint32_t limit);
/* Load GDT */
extern void gdt_flush(uint32_t gdt_ptr);
#endif /* GDT_H */
```
```c
/* kernel/gdt.c - GDT implementation */
#include "gdt.h"
#include "kprintf.h"
/* GDT with 6 entries: null, kernel code, kernel data, user code, user data, TSS */
#define GDT_ENTRIES 6
static gdt_entry_t gdt[GDT_ENTRIES];
static gdt_ptr_t gdt_ptr;
/* Set a GDT entry */
void gdt_set_gate(uint32_t index, uint32_t base, uint32_t limit, 
                  uint8_t access, uint8_t granularity) {
    gdt[index].base_low    = base & 0xFFFF;
    gdt[index].base_middle = (base >> 16) & 0xFF;
    gdt[index].base_high   = (base >> 24) & 0xFF;
    gdt[index].limit_low   = limit & 0xFFFF;
    gdt[index].granularity = (limit >> 16) & 0x0F;
    gdt[index].granularity |= granularity & 0xF0;
    gdt[index].access = access;
}
/* Set TSS entry (system segment, different format) */
void gdt_set_tss(uint32_t index, uint32_t base, uint32_t limit) {
    /* For TSS, access byte = 0x89 (present, DPL=0, system, type=0x89) */
    /* Granularity = 0x00 (byte granularity, 16-bit) */
    gdt_set_gate(index, base, limit, 0x89, 0x00);
}
/* Initialize GDT */
void gdt_init(void) {
    /* Null descriptor */
    gdt_set_gate(0, 0, 0, 0, 0);
    /* Kernel code segment: base=0, limit=4GB, 32-bit, ring 0 */
    gdt_set_gate(1, 0, 0xFFFFFFFF, 0x9A, 0xCF);
    /* Kernel data segment: base=0, limit=4GB, 32-bit, ring 0 */
    gdt_set_gate(2, 0, 0xFFFFFFFF, 0x92, 0xCF);
    /* User code segment: base=0, limit=4GB, 32-bit, ring 3 */
    gdt_set_gate(3, 0, 0xFFFFFFFF, 0xFA, 0xCF);
    /* User data segment: base=0, limit=4GB, 32-bit, ring 3 */
    gdt_set_gate(4, 0, 0xFFFFFFFF, 0xF2, 0xCF);
    /* TSS entry (index 5) - will be set by tss_init() */
    gdt_set_gate(5, 0, 0, 0, 0);  /* Placeholder */
    /* Set up GDT pointer */
    gdt_ptr.limit = sizeof(gdt) - 1;
    gdt_ptr.base = (uint32_t)&gdt;
    /* Load GDT */
    gdt_flush((uint32_t)&gdt_ptr);
    kprintf("GDT initialized with %d entries at 0x%x\n", GDT_ENTRIES, 
            (uint32_t)&gdt);
}
```
Add the assembly stub:
```nasm
; kernel/gdt_flush.asm
[BITS 32]
[GLOBAL gdt_flush]
gdt_flush:
    mov eax, [esp+4]    ; GDT pointer
    lgdt [eax]          ; Load GDT
    ; Reload segment registers
    mov ax, 0x10        ; Kernel data segment
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    ; Far jump to reload CS
    jmp 0x08:.flush
.flush:
    ret
```

![TSS.ESP0 Update on Context Switch](./diagrams/diag-tss-esp0-update.svg)

## Step 5: The Scheduler — Round-Robin
The scheduler is called on every timer interrupt. It selects the next READY process and performs a context switch.

![Round-Robin Scheduler Flow](./diagrams/diag-scheduler-flow.svg)

Create `kernel/scheduler.c`:
```c
/* kernel/scheduler.c - Round-robin scheduler */
#include "scheduler.h"
#include "process.h"
#include "tss.h"
#include "kprintf.h"
/* Scheduler state */
static process_t *ready_queue = NULL;
static uint32_t tick_count = 0;
/* Initialize scheduler */
void scheduler_init(void) {
    ready_queue = NULL;
    tick_count = 0;
    kprintf("Scheduler initialized (round-robin)\n");
}
/* Add process to ready queue */
void scheduler_add(process_t *proc) {
    if (proc == NULL) return;
    proc->state = PROCESS_STATE_READY;
    /* Add to end of queue */
    if (ready_queue == NULL) {
        ready_queue = proc;
        proc->next = proc;
        proc->prev = proc;  /* Circular */
    } else {
        proc->next = ready_queue;
        proc->prev = ready_queue->prev;
        ready_queue->prev->next = proc;
        ready_queue->prev = proc;
    }
}
/* Remove process from ready queue */
void scheduler_remove(process_t *proc) {
    if (proc == NULL || ready_queue == NULL) return;
    if (proc->next == proc) {
        /* Only process in queue */
        ready_queue = NULL;
    } else {
        proc->prev->next = proc->next;
        proc->next->prev = proc->prev;
        if (ready_queue == proc) {
            ready_queue = proc->next;
        }
    }
    proc->next = NULL;
    proc->prev = NULL;
}
/* Timer tick - called from IRQ0 handler */
void scheduler_tick(registers_t *regs) {
    tick_count++;
    /* Get current process */
    process_t *current = process_get_current();
    if (current == NULL) {
        /* No current process - try to start one */
        if (ready_queue != NULL) {
            process_t *next = ready_queue;
            current_process = next;
            next->state = PROCESS_STATE_RUNNING;
            /* Start running the first process */
            /* This is a special case - we need to "return" to it */
        }
        return;
    }
    /* Decrement time slice */
    if (current->time_slice > 0) {
        current->time_slice--;
    }
    /* Check if time slice expired */
    if (current->time_slice == 0) {
        /* Time to switch! */
        scheduler_yield();
    }
}
/* Yield the CPU - switch to next ready process */
void scheduler_yield(void) {
    process_t *current = process_get_current();
    if (ready_queue == NULL) {
        /* No other process to run */
        if (current != NULL) {
            current->time_slice = 10;  /* Reset time slice */
        }
        return;
    }
    /* Get next process */
    process_t *next;
    if (current == NULL) {
        next = ready_queue;
    } else {
        /* Put current process back in ready queue */
        current->state = PROCESS_STATE_READY;
        current->time_slice = 10;
        /* Next process in circular queue */
        next = current->next;
        /* Skip non-ready processes */
        while (next != current && next->state != PROCESS_STATE_READY) {
            next = next->next;
        }
        if (next == current || next->state != PROCESS_STATE_READY) {
            /* No ready process found */
            return;
        }
    }
    /* Update ready queue head */
    ready_queue = next;
    /* Mark next as running */
    next->state = PROCESS_STATE_RUNNING;
    /* Perform context switch */
    if (current != next) {
        /* Update current_process before switch */
        current_process = next;
        /* Update TSS.ESP0 */
        if (next->kernel_stack != 0) {
            tss_update_esp0(next->kernel_stack);
        }
        /* Do the context switch */
        context_switch(current, next);
    }
}
/* Block a process */
void scheduler_block(process_t *proc) {
    if (proc == NULL) return;
    proc->state = PROCESS_STATE_BLOCKED;
    scheduler_remove(proc);
    /* If this is the current process, yield */
    if (proc == process_get_current()) {
        scheduler_yield();
    }
}
/* Unblock a process */
void scheduler_unblock(process_t *proc) {
    if (proc == NULL) return;
    scheduler_add(proc);
}
```
## Step 6: The System Call Interface
System calls are the mechanism for user-mode programs to request kernel services. The user program executes `int 0x80`, the CPU transitions to Ring 0, and the kernel handles the request.

![INT 0x80 System Call Interface](./diagrams/diag-syscall-interface.svg)

Create `kernel/syscall.h`:
```c
/* kernel/syscall.h - System call interface */
#ifndef SYSCALL_H
#define SYSCALL_H
#include <stdint.h>
#include "process.h"
/* System call numbers */
#define SYS_EXIT    0
#define SYS_READ    1
#define SYS_WRITE   2
#define SYS_YIELD   3
#define SYS_GETPID  4
/* System call handler (called from assembly) */
void syscall_handler(registers_t *regs);
/* System call implementations */
int sys_exit(int status);
int sys_read(int fd, char *buf, int count);
int sys_write(int fd, const char *buf, int count);
int sys_yield(void);
int sys_getpid(void);
/* Initialize system calls */
void syscall_init(void);
#endif /* SYSCALL_H */
```
Create `kernel/syscall.c`:
```c
/* kernel/syscall.c - System call implementation */
#include "syscall.h"
#include "process.h"
#include "scheduler.h"
#include "vga.h"
#include "kprintf.h"
#include "idt.h"
/* System call table */
typedef int (*syscall_func_t)(int, int, int);
static syscall_func_t syscall_table[] = {
    [SYS_EXIT]    = (syscall_func_t)sys_exit,
    [SYS_READ]    = (syscall_func_t)sys_read,
    [SYS_WRITE]   = (syscall_func_t)sys_write,
    [SYS_YIELD]   = (syscall_func_t)sys_yield,
    [SYS_GETPID]  = (syscall_func_t)sys_getpid,
};
#define NUM_SYSCALLS (sizeof(syscall_table) / sizeof(syscall_table[0]))
/* System call handler */
void syscall_handler(registers_t *regs) {
    /* Syscall number in EAX, arguments in EBX, ECX, EDX */
    int syscall_num = regs->eax;
    int arg1 = regs->ebx;
    int arg2 = regs->ecx;
    int arg3 = regs->edx;
    int result = -1;
    if (syscall_num >= 0 && syscall_num < (int)NUM_SYSCALLS) {
        syscall_func_t func = syscall_table[syscall_num];
        if (func != NULL) {
            result = func(arg1, arg2, arg3);
        }
    }
    /* Return result in EAX */
    regs->eax = result;
}
/* sys_exit - terminate the current process */
int sys_exit(int status) {
    process_t *current = process_get_current();
    if (current != NULL) {
        kprintf("Process %d exited with status %d\n", current->pid, status);
        process_terminate(current);
        scheduler_yield();  /* Never returns */
    }
    return 0;
}
/* sys_read - read from file descriptor */
int sys_read(int fd, char *buf, int count) {
    /* For now, only support stdin (fd=0) from keyboard */
    if (fd != 0) return -1;
    /* TODO: Read from keyboard buffer */
    (void)buf;
    (void)count;
    return -1;  /* Not implemented */
}
/* sys_write - write to file descriptor */
int sys_write(int fd, const char *buf, int count) {
    /* For now, only support stdout (fd=1) and stderr (fd=2) */
    if (fd != 1 && fd != 2) return -1;
    /* Write to VGA console */
    for (int i = 0; i < count; i++) {
        vga_put_char(buf[i], vga_color_attr(VGA_COLOR_LIGHT_GREY, VGA_COLOR_BLACK));
    }
    return count;
}
/* sys_yield - voluntarily yield the CPU */
int sys_yield(void) {
    scheduler_yield();
    return 0;
}
/* sys_getpid - get current process ID */
int sys_getpid(void) {
    process_t *current = process_get_current();
    if (current != NULL) {
        return (int)current->pid;
    }
    return -1;
}
/* Initialize system calls */
void syscall_init(void) {
    /* Register syscall handler for INT 0x80 */
    /* We need to set up IDT entry 0x80 as a trap gate with DPL=3 */
    /* This allows user mode to call it */
    idt_set_gate(0x80, (uint32_t)syscall_asm_entry, 0x08, 0xEE);
    /* 0xEE = P(1) DPL(11) 0 Type(1110) = user-accessible trap gate */
    kprintf("System calls initialized (INT 0x80, DPL=3)\n");
}
```
Create the assembly entry point for syscalls:
```nasm
; kernel/syscall_entry.asm - System call entry point
[BITS 32]
[EXTERN syscall_handler]
[GLOBAL syscall_asm_entry]
syscall_asm_entry:
    ; Save all registers
    pusha               ; EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI
    ; Save segment registers
    mov ax, ds
    push ax
    mov ax, es
    push ax
    ; Load kernel data segment
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    ; Call C handler
    push esp            ; Pointer to register structure
    call syscall_handler
    add esp, 4
    ; Restore segment registers
    pop ax
    mov es, ax
    pop ax
    mov ds, ax
    ; Restore general registers
    popa
    ; Return from interrupt
    iret
```

![System Call Dispatch Table](./diagrams/diag-syscall-table.svg)

## Step 7: Entering User Mode
The most delicate operation is transitioning from kernel mode to user mode. We use `iret` to do this atomically — it pops EIP, CS, EFLAGS, and (for privilege changes) ESP and SS.

![Entering User Mode (Ring 3)](./diagrams/diag-user-mode-entry.svg)

Create `kernel/usermode.h`:
```c
/* kernel/usermode.h - User mode transitions */
#ifndef USERMODE_H
#define USERMODE_H
#include <stdint.h>
/* Jump to user mode */
void jump_to_user_mode(uint32_t entry_point, uint32_t stack_ptr);
/* Start a user process */
void start_user_process(process_t *proc);
#endif /* USERMODE_H */
```
Create `kernel/usermode.c`:
```c
/* kernel/usermode.c - User mode implementation */
#include "usermode.h"
#include "process.h"
#include "tss.h"
#include "gdt.h"
/* Jump to user mode using iret */
void jump_to_user_mode(uint32_t entry_point, uint32_t stack_ptr) {
    /* Set up the stack for iret */
    /* We need to push: SS, ESP, EFLAGS, CS, EIP */
    /* Set EFLAGS with interrupts enabled */
    uint32_t eflags;
    __asm__ volatile ("pushfl; popl %0" : "=r"(eflags));
    eflags |= 0x200;  /* Set interrupt flag */
    /* Set TSS.ESP0 to our kernel stack */
    uint32_t kernel_esp;
    __asm__ volatile ("mov %%esp, %0" : "=r"(kernel_esp));
    tss_update_esp0(kernel_esp);
    /* Push user mode context onto stack */
    __asm__ volatile (
        "cli;"                    /* Disable interrupts during transition */
        "mov $0x23, %%ax;"        /* User data segment selector (0x20 | RPL=3) */
        "mov %%ax, %%ds;"         /* Load data segments */
        "mov %%ax, %%es;"
        "mov %%ax, %%fs;"
        "mov %%ax, %%gs;"
        /* Push stack for iret */
        "push $0x23;"             /* SS = user data segment */
        "push %0;"                /* ESP = user stack */
        "push %1;"                /* EFLAGS */
        "push $0x1B;"             /* CS = user code segment (0x18 | RPL=3) */
        "push %2;"                /* EIP = entry point */
        "iret;"                   /* Jump to user mode! */
        :
        : "r"(stack_ptr), "r"(eflags), "r"(entry_point)
        : "eax", "memory"
    );
}
```
## Step 8: Process Memory Isolation
Each user process needs its own page directory to ensure isolation. The kernel is mapped into every process's address space (at higher-half addresses), but user pages are private.

![Process Memory Isolation via Page Tables](./diagrams/diag-process-isolation.svg)

Create a function to create a process page directory:
```c
/* Add to kernel/process.c */
/* Create a page directory for a new user process */
static uint32_t create_user_page_directory(void) {
    /* Allocate a page for the page directory */
    void *pd_frame = pmm_alloc_frame();
    if (pd_frame == NULL) {
        return 0;
    }
    uint32_t *pd = (uint32_t *)pd_frame;
    /* Zero the page directory */
    for (int i = 0; i < 1024; i++) {
        pd[i] = 0;
    }
    /* Copy kernel mappings (higher half) from current page directory */
    /* The kernel is at 0xC0000000+ (entries 768-1023) */
    extern uint32_t *current_page_directory;
    for (int i = 768; i < 1024; i++) {
        pd[i] = current_page_directory[i];
        /* Mark as supervisor-only for security */
        pd[i] &= ~PAGE_USER;
    }
    /* Identity map first 1MB for VGA and BIOS (read-only for user) */
    /* Actually, user processes shouldn't access VGA directly */
    /* They should use syscalls */
    return (uint32_t)pd;
}
```
## Step 9: Testing with Multiple Processes
Let's create a test that demonstrates preemptive multitasking:

![Multi-Process Demo Memory Layout](./diagrams/diag-multi-process-demo.svg)

```c
/* kernel/test_processes.c - Process tests */
#include "process.h"
#include "scheduler.h"
#include "vga.h"
#include "kprintf.h"
/* Test process 1 - prints 'A' continuously */
void process_a(void) {
    int count = 0;
    while (1) {
        vga_put_char_at('A', vga_color_attr(VGA_COLOR_RED, VGA_COLOR_BLACK), 
                        count % 20, 10);
        count++;
        for (volatile int i = 0; i < 1000000; i++);  /* Delay */
        scheduler_yield();  /* Be nice, yield CPU */
    }
}
/* Test process 2 - prints 'B' continuously */
void process_b(void) {
    int count = 0;
    while (1) {
        vga_put_char_at('B', vga_color_attr(VGA_COLOR_GREEN, VGA_COLOR_BLACK), 
                        count % 20, 12);
        count++;
        for (volatile int i = 0; i < 1000000; i++);
        scheduler_yield();
    }
}
/* Test process 3 - prints 'C' continuously */
void process_c(void) {
    int count = 0;
    while (1) {
        vga_put_char_at('C', vga_color_attr(VGA_COLOR_BLUE, VGA_COLOR_BLACK), 
                        count % 20, 14);
        count++;
        for (volatile int i = 0; i < 1000000; i++);
        scheduler_yield();
    }
}
/* User mode test - uses syscalls */
void user_process_test(void) {
    /* This runs in user mode (ring 3) */
    /* We'll use syscalls to print */
    const char *msg = "Hello from user mode!\n";
    /* sys_write(1, msg, strlen(msg)) */
    int len = 0;
    while (msg[len]) len++;
    __asm__ volatile (
        "int $0x80"
        :
        : "a"(2), "b"(1), "c"(msg), "d"(len)  /* sys_write, fd=1, buf, count */
        : "memory"
    );
    /* sys_getpid */
    int pid;
    __asm__ volatile (
        "int $0x80"
        : "=a"(pid)
        : "a"(4)  /* sys_getpid */
    );
    /* Print PID */
    char pid_msg[] = "My PID is: X\n";
    pid_msg[11] = '0' + (pid % 10);
    __asm__ volatile (
        "int $0x80"
        :
        : "a"(2), "b"(1), "c"(pid_msg), "d"(13)
        : "memory"
    );
    /* Try to access kernel memory - should page fault! */
    /* Uncomment to test: */
    // volatile int *kernel_mem = (volatile int *)0xC0100000;
    // *kernel_mem = 0xDEAD;  /* Page fault! */
    /* Exit */
    __asm__ volatile (
        "int $0x80"
        :
        : "a"(0), "b"(0)  /* sys_exit, status=0 */
    );
}
/* Run process tests */
void test_processes(void) {
    kprintf("\n=== Process Tests ===\n\n");
    /* Initialize scheduler */
    scheduler_init();
    /* Create kernel processes */
    process_t *proc_a = process_create("ProcessA", process_a, false);
    process_t *proc_b = process_create("ProcessB", process_b, false);
    process_t *proc_c = process_create("ProcessC", process_c, false);
    /* Add to scheduler */
    scheduler_add(proc_a);
    scheduler_add(proc_b);
    scheduler_add(proc_c);
    /* Create a user-mode process */
    process_t *user_proc = process_create("UserTest", user_process_test, true);
    scheduler_add(user_proc);
    kprintf("\nCreated 4 processes. Starting scheduler...\n");
    kprintf("Press any key to watch them run!\n\n");
    /* Enable timer interrupts */
    /* The timer handler will call scheduler_tick */
    /* Start the first process */
    current_process = proc_a;
    proc_a->state = PROCESS_STATE_RUNNING;
    /* Jump to the first process */
    /* This never returns - we're now in process context */
    /* Actually, we need to set up the return properly */
    /* For now, just enable interrupts and let the timer do its work */
    __asm__ volatile ("sti");
    /* Wait forever - processes will run via timer interrupts */
    while (1) {
        __asm__ volatile ("hlt");
    }
}
```
## Step 10: Putting It All Together
Update `kernel/main.c`:
```c
/* kernel/main.c - Updated with process management */
#include "vga.h"
#include "serial.h"
#include "kprintf.h"
#include "idt.h"
#include "pic.h"
#include "timer.h"
#include "keyboard.h"
#include "interrupts.h"
#include "memory.h"
#include "gdt.h"
#include "tss.h"
#include "process.h"
#include "scheduler.h"
#include "syscall.h"
/* External test function */
extern void test_processes(void);
void kernel_main(void) {
    /* Initialize VGA and serial */
    vga_init();
    serial_init();
    kprintf("\n");
    kprintf("========================================\n");
    kprintf("  MyOS Kernel v0.4 - Processes & Scheduling\n");
    kprintf("  Built on %s at %s\n", __DATE__, __TIME__);
    kprintf("========================================\n\n");
    /* Initialize GDT (must be done before TSS) */
    kprintf("Initializing GDT...\n");
    gdt_init();
    /* Initialize TSS */
    kprintf("Initializing TSS...\n");
    tss_init();
    /* Initialize physical memory */
    kprintf("Initializing physical memory...\n");
    pmm_init(512 * 1024);
    extern uint32_t _kernel_end;
    pmm_mark_used(0x100000, (uint32_t)&_kernel_end - 0x100000 + 0x10000);
    /* Initialize interrupt handling */
    kprintf("Initializing interrupts...\n");
    idt_init();
    pic_init();
    register_interrupt_handler(14, page_fault_handler);
    /* Enable interrupts */
    __asm__ volatile ("sti");
    /* Initialize timer */
    timer_init(100);
    /* Initialize keyboard */
    keyboard_init();
    /* Initialize virtual memory */
    kprintf("Initializing virtual memory...\n");
    vmm_init();
    /* Initialize kernel heap */
    kheap_init();
    /* Initialize process management */
    kprintf("Initializing process management...\n");
    process_init();
    /* Initialize system calls */
    kprintf("Initializing system calls...\n");
    syscall_init();
    kprintf("\n=== All subsystems initialized ===\n");
    kprintf("\nStarting process tests...\n");
    /* Run process tests - this never returns */
    test_processes();
    /* Should never reach here */
    kprintf("\nSystem halted.\n");
    while (1) {
        __asm__ volatile ("cli; hlt");
    }
}
```
## Hardware Soul: The Complete Context Switch
Let's trace exactly what happens during a context switch from Process A to Process B:
```
1. Timer interrupt fires (IRQ0)
   └── PIC sends INT to CPU with vector 32
   └── CPU checks IF (Interrupt Flag) - enabled
   └── CPU pushes SS, ESP (if privilege change), EFLAGS, CS, EIP to kernel stack
   └── CPU loads IDT entry 32 → CS:EIP = isr32
   └── CPU jumps to isr32 (in idt_stubs.asm)
2. ISR stub saves registers
   └── pusha (EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI)
   └── push DS, ES, FS, GS
   └── Load kernel data segment (0x10) into DS, ES, FS, GS
   └── Push ESP as argument
   └── Call interrupt_handler
3. interrupt_handler dispatches to timer_handler
   └── timer_handler calls scheduler_tick
4. scheduler_tick decides to switch
   └── Process A's time slice expired
   └── Call scheduler_yield
5. scheduler_yield selects next process
   └── Process B is next in ready queue
   └── Update current_process = Process B
   └── Call tss_update_esp0(Process B's kernel stack)
       └── TSS.ESP0 = Process B's kernel stack top
   └── Call context_switch(Process A, Process B)
6. context_switch (assembly)
   └── Save Process A's registers to its PCB:
       - EBX, ECX, EDX, ESI, EDI, EBP, ESP, EIP, EFLAGS
   └── Load Process B's page directory if different:
       - mov cr3, Process B->page_directory
       - TLB is flushed (non-global entries)
   └── Load Process B's registers from its PCB:
       - EBX, ECX, EDX, ESI, EDI, EBP
       - EFLAGS
       - ESP
   └── ret (jumps to Process B's saved EIP)
7. Process B resumes
   └── It was last in the middle of its timer interrupt handler!
   └── Or it was at the beginning, never run before
   └── Eventually returns from interrupt via iret
   └── If Process B was in user mode: iret pops EIP, CS, EFLAGS, ESP, SS
   └── Process B continues executing user code
Cache effects:
- Process A's stack: likely in cache (hot)
- Process B's stack: may be cold (cache miss on first access)
- Page tables: TLB miss when CR3 changes, needs page table walk
- Code: Process B's code may not be cached
Timing:
- Timer interrupt to ISR entry: ~50 cycles
- Saving registers: ~20 cycles
- Scheduler logic: ~100-500 cycles (depending on complexity)
- Context switch assembly: ~30-50 cycles
- TLB refill after CR3 change: ~100-500 cycles (depends on locality)
- Total: ~500-1500 cycles = 0.5-1.5 microseconds at 1GHz
```
## Common Pitfalls and Debugging
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Triple fault on first context switch | PCB fields misaligned with assembly offsets | Verify offset macros match struct layout |
| User process crashes immediately | TSS.ESP0 not set correctly | Ensure tss_update_esp0 is called before first user process |
| Page fault in scheduler | Kernel stack not mapped | Verify kernel stack pages are mapped |
| Processes don't switch | Timer interrupt not calling scheduler | Check timer handler calls scheduler_tick |
| User syscall doesn't work | IDT entry DPL=0 | Set IDT gate type to 0xEE (DPL=3) |
| Random register corruption | Missing register save/restore | Verify pusha/popa cover all registers |
### QEMU Debugging
```bash
# Debug context switching
qemu-system-i386 -fda os.img -serial stdio -s -S &
gdb -ex "target remote :1234" \
    -ex "break context_switch" \
    -ex "break scheduler_yield" \
    -ex "continue"
# In GDB:
# info registers     - see current register state
# x/20x $esp        - examine stack
# stepi             - single step through assembly
```
## Design Decisions
| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Round-robin** ✓ | Simple, fair, predictable latency | No priority, no real-time | Most hobby OSes |
| Priority-based | Important processes run first | Starvation possible | Windows, Linux (CFS) |
| **Software context switch** ✓ | Full control, portable | More code | Linux, Windows, macOS |
| Hardware task switch (TSS) | Built-in, less code | Slow, inflexible | Rarely used |
| **Per-process kernel stack** ✓ | Isolation, safety | Memory overhead | All modern OSes |
| Shared kernel stack | Memory efficient | Reentrancy issues | Some embedded systems |
| **INT 0x80 for syscalls** ✓ | Simple, works everywhere | ~1000 cycles overhead | Linux (x86), older systems |
| sysenter/syscall | ~100 cycles | CPU-specific | Linux (modern), Windows |
## Knowledge Cascade
You've just built the foundation for all concurrent programming — the illusion of parallelism through rapid context switching.
**Same Domain:**
- **Wait queues and blocking** — When a process waits for I/O, it's removed from the ready queue and placed on a wait queue. When I/O completes, the interrupt handler moves it back. This is how `read()`, `wait()`, and `sleep()` work.
- **Priority scheduling** — Instead of round-robin, give each process a priority. Higher-priority processes run first. This enables real-time guarantees.
- **Copy-on-write fork()** — When forking, share page tables instead of copying. Mark pages read-only. On write, page fault handler copies the page. This makes `fork()` O(1) instead of O(n) for memory.
- **Kernel threads** — The same mechanism can run kernel threads (no user address space). This is how Linux handles background work like flushing buffers.
**Cross-Domain:**
- **Green threads / fibers** — Go goroutines, Java green threads, and async/await patterns all use the same concept: user-space context switching. Instead of the kernel saving registers, the runtime does it. The difference: kernel preemption is involuntary (timer interrupt), user-space yielding is voluntary (must call `yield` or use async).
- **Container runtimes** — Docker, runc, and containerd all use the same primitives you just built: processes, namespaces, and cgroups. A container is just a process (or process tree) with restricted views of resources. The PCB and page tables are the hardware-level namespace.
- **Real-time scheduling** — Audio processing, game loops, and industrial control need guarantees. Round-robin can't provide them. RTOSes use priority-based scheduling with priority inheritance to prevent unbounded latency.
- **System call overhead** — INT 0x80 costs ~1000 cycles. Modern OSes use `sysenter` (Intel) or `syscall` (AMD) for ~100 cycles. This 10× improvement drove designs like io_uring (batch syscalls) and DPDK (kernel bypass for networking).
- **Signal handling** — Unix signals are "software interrupts." They use the same mechanism: interrupt current execution, run handler, resume. Understanding ISRs makes signal behavior intuitive.
- **Coroutines in languages** — Python `async/await`, Rust `async`, C++20 coroutines — they all implement cooperative multitasking at the language level. The same concept of saving state and resuming later, just without hardware support.
**Historical Context:**
- The concept of "process" dates to MULTICS (1960s). Before that, computers ran one program at a time.
- Time-sharing was revolutionary: users could interact with the computer while it was "running" other users' programs.
- The TSS was Intel's attempt at hardware task switching (80286). It failed because it was too slow and inflexible. All modern OSes use software context switching.
- The 100Hz timer tick is a common choice. Linux uses 100Hz, 250Hz, or 1000Hz depending on configuration. Higher frequency = better responsiveness, higher overhead.
**What You Can Now Build:**
- A `fork()` system call that creates child processes
- An `exec()` system call that loads new programs
- A `wait()` system call for parent-child synchronization
- A pipe implementation for inter-process communication
- A shell that launches and manages processes
- A real-time scheduler with priorities
- Kernel threads for background maintenance
---
[[CRITERIA_JSON: {"milestone_id": "build-os-m4", "criteria": ["Process control block (PCB) structure defined with PID (uint32_t), name (char[32]), state enum (READY/RUNNING/BLOCKED/TERMINATED), priority (uint32_t), time_slice (uint32_t)", "PCB contains saved register state: EAX, EBX, ECX, EDX, ESI, EDI, EBP, ESP, EIP, EFLAGS as uint32_t fields", "PCB contains memory management fields: page_directory (uint32_t physical address), kernel_stack (uint32_t), user_stack (uint32_t)", "PCB contains linked list pointers (next/prev) for scheduler queue management", "process_create() allocates PCB via kmalloc, assigns unique PID, initializes state to READY, sets time_slice to 10 ticks", "process_create() allocates kernel stack (8KB) via kmalloc for each process", "process_create() for user mode allocates separate user stack (64KB) and creates new page directory", "process_create() sets up initial stack frame with EIP pointing to entry point and EFLAGS with IF=1 (interrupts enabled)", "process_terminate() sets state to TERMINATED and removes PCB from process list", "Task State Segment (TSS) structure is 104 bytes with SS0, ESP0 fields for kernel stack during ring transitions", "TSS initialized with SS0=0x10 (kernel data segment) and ESP0 updated on every context switch", "TSS descriptor added to GDT at index 5 (selector 0x28) with type 0x89 (32-bit available TSS)", "Task Register (TR) loaded with TSS selector via ltr instruction during TSS initialization", "tss_update_esp0() updates TSS.ESP0 field to current process kernel stack top before returning to user mode", "Context switch implemented in assembly: saves all callee-saved registers (EBX, ESI, EDI, EBP, ESP) to old PCB", "Context switch saves EIP by capturing return address from stack", "Context switch saves EFLAGS via pushfd/pop to PCB.EFLAGS field", "Context switch loads CR3 with new process page directory if non-zero (triggers TLB flush)", "Context switch updates TSS.ESP0 via call to tss_update_esp0 before restoring new process registers", "Context switch restores all registers from new PCB and executes ret to jump to new process EIP", "Round-robin scheduler maintains circular ready queue of processes", "scheduler_tick() called from timer interrupt handler (IRQ0), decrements current process time_slice", "scheduler_yield() called when time_slice reaches zero, selects next READY process from queue", "scheduler_yield() updates current_process global, updates TSS.ESP0, and calls context_switch", "scheduler_add() adds process to ready queue with state=READY in circular linked list", "scheduler_remove() removes process from ready queue when blocked or terminated", "User mode entry uses iret instruction with stack: SS (user data 0x23), ESP (user stack), EFLAGS, CS (user code 0x1B), EIP", "jump_to_user_mode() disables interrupts during setup, loads data segments with user selector, pushes user context, executes iret", "System call interface uses INT 0x80 with syscall number in EAX, arguments in EBX/ECX/EDX", "IDT entry 0x80 configured with type 0xEE (DPL=3, user-callable trap gate) allowing ring 3 invocation", "syscall_handler() reads EAX for syscall number, dispatches via syscall_table array to handler functions", "sys_write(fd, buf, count) implemented for fd=1 (stdout) writing to VGA console", "sys_exit(status) terminates current process via process_terminate() and calls scheduler_yield()", "sys_getpid() returns current process PID from PCB", "sys_yield() voluntarily yields CPU by calling scheduler_yield()", "System call entry stub saves all registers (pusha), segment registers, calls syscall_handler, restores and irets", "System call result returned in EAX register to user process", "Process page directory copies kernel mappings (entries 768-1023) marked supervisor-only for isolation", "User process page fault on kernel memory access demonstrates isolation (user bit not set)", "At least 3 kernel-mode processes run concurrently demonstrating preemptive multitasking with different screen regions", "Timer interrupt at 100Hz provides 10ms time slices, triggering scheduler on each tick", "Current process pointer (current_process) global updated atomically before context switch", "Kernel stack for each process is 8KB, user stack is 64KB for user-mode processes"]}]
<!-- END_MS -->


## System Overview

![System Overview](./diagrams/system-overview.svg)




# TDD

A 32-bit x86 operating system kernel that boots from bare metal, transitions through real mode to protected mode, handles hardware interrupts and CPU exceptions, manages physical and virtual memory through page tables, and implements preemptive multitasking with process isolation and system calls. The kernel demonstrates the complete software stack from firmware handoff to user-mode process execution, revealing how hardware constraints (cache lines, TLB entries, interrupt latency) shape every design decision.



<!-- TDD_MOD_ID: build-os-m1 -->
# Technical Design Document: Bootloader, GDT, and Kernel Entry
**Module ID:** `build-os-m1`
---
## Module Charter
The bootloader module bridges the gap between BIOS firmware handoff and a fully operational 32-bit protected mode kernel. It is responsible for transitioning the CPU from 16-bit real mode (the state left by BIOS) to 32-bit protected mode with proper memory segmentation defined by a Global Descriptor Table. This module loads the kernel binary from disk into memory at 1MB, enables the A20 address line for accessing memory above 1MB, configures five GDT entries (null, kernel code, kernel data, user code, user data), performs the protected mode transition via CR0.PE manipulation and a far jump, and finally transfers control to the kernel's C entry point with BSS zeroed, stack initialized, and console output available via VGA and serial ports.
**This module does NOT:** handle interrupts (IDT comes later), set up paging (CR3/CR0.PG), load user programs, or implement any system calls. It is purely the bootstrap sequence from power-on to kernel_main().
**Upstream dependencies:** BIOS firmware (provides boot drive in DL, loads MBR to 0x7C00), disk containing kernel binary at sectors 2+.
**Downstream consumers:** Kernel C code (kernel_main), interrupt handlers (milestone 2), memory manager (milestone 3), process scheduler (milestone 4).
**Invariants:**
- Bootloader code + data fits in 512 bytes with 0x55AA signature at bytes 510-511
- GDT is loaded before CR0.PE is set
- Far jump follows CR0.PE modification before any 32-bit code executes
- All data segment registers (DS, ES, FS, GS, SS) are reloaded after mode switch
- BSS section is zeroed before any C code runs
- Kernel is loaded at physical address 0x100000 (1MB mark)
---
## File Structure
```
project/
├── 01_boot/
│   ├── 01_boot.asm          # Main bootloader (MBR, 512 bytes)
│   ├── 02_gdt.asm           # GDT definitions and descriptor
│   ├── 03_a20.asm           # A20 line enable routines
│   ├── 04_load_kernel.asm   # Disk read routines
│   └── 05_protected.asm     # Protected mode transition (32-bit)
├── 02_kernel/
│   ├── 01_entry.asm         # Kernel entry point, BSS zeroing
│   ├── 02_main.c            # C kernel entry (kernel_main)
│   ├── 03_vga.h             # VGA text mode interface
│   ├── 04_vga.c             # VGA implementation
│   ├── 05_serial.h          # Serial port interface
│   ├── 06_serial.c          # Serial implementation
│   ├── 07_kprintf.h         # Kernel printf interface
│   ├── 08_kprintf.c         # kprintf implementation
│   └── 09_linker.ld         # Linker script
├── 03_include/
│   ├── 01_stdint.h          # Fixed-width integer types
│   └── 02_stdarg.h          # Variable argument handling
└── 04_Makefile              # Build system
```
---
## Complete Data Model
### Memory Layout at Boot
| Address Range | Size | Purpose | Notes |
|---------------|------|---------|-------|
| 0x000000-0x0003FF | 1KB | Real Mode IVT | 256 interrupt vectors × 4 bytes |
| 0x000400-0x0004FF | 256B | BIOS Data Area | BIOS workspace |
| 0x000500-0x07BFF | ~30KB | Free conventional memory | Available for bootloader use |
| 0x0007C00-0x07DFF | 512B | Bootloader (MBR) | **YOU ARE HERE** |
| 0x0007E00-0x9FBFF | ~622KB | Free conventional memory | Stage 2 loader, temporary buffers |
| 0x000A000-0x000BFFF | 8KB | VGA video memory | Text mode buffer at 0xB8000 |
| 0x000C000-0x000EFFF | 12KB | BIOS ROM expansion | Video BIOS, etc. |
| 0x000F000-0x000FFFF | 4KB | BIOS ROM | Entry at 0xF0000:0xFFF0 |
| 0x00100000+ | — | Extended memory | Kernel loads here |
### GDT Entry Structure (8 bytes each)
```
Byte 7    Byte 6    Byte 5    Byte 4    Byte 3    Byte 2    Byte 1    Byte 0
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│Base    │ G│D│L│A│ Limit   │ P│DPL│S│  Type   │Base    │Base    │Limit   │
│31:24   │  │ │ │ │19:16    │  │   │ │         │23:16   │15:0    │15:0    │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Base Address: 32-bit value split across 3 fields (bytes 2, 3-4, 7)
  - Byte 2:     Base[15:0]  (low 16 bits)
  - Byte 3-4:   Base[23:16] (next 8 bits)
  - Byte 7:     Base[31:24] (high 8 bits)
Segment Limit: 20-bit value split across 2 fields (bytes 0-1, byte 6)
  - Bytes 0-1:  Limit[15:0] (low 16 bits)
  - Byte 6[3:0]:Limit[19:16] (high 4 bits)
Access Byte (Byte 5):
  - Bit 7 (P):  Present (1 = segment is valid)
  - Bit 6-5 (DPL): Descriptor Privilege Level (00 = Ring 0, 11 = Ring 3)
  - Bit 4 (S):  System (0 = system segment, 1 = code/data segment)
  - Bit 3-0 (Type):
    - For code: E(Execute), C(Conforming), R(Readable), A(Accessed)
    - For data: W(Writable), E(Expand-down), W(Write), A(Accessed)
Flags (Byte 6, bits 4-7):
  - Bit 7 (G):  Granularity (0 = byte, 1 = 4KB pages)
  - Bit 6 (D/B): Default operation size (0 = 16-bit, 1 = 32-bit)
  - Bit 5 (L):  Long mode (64-bit) - must be 0 for 32-bit
  - Bit 4 (AVL): Available for system software
```
### GDT Entries (5 Required)
| Index | Selector | Base | Limit | G | D | DPL | Type | Purpose |
|-------|----------|------|-------|---|---|-----|------|---------|
| 0 | 0x00 | 0x00000000 | 0x00000 | 0 | 0 | 0 | 0x00 | Null descriptor (required) |
| 1 | 0x08 | 0x00000000 | 0xFFFFF | 1 | 1 | 0 | 0x9A | Kernel code (32-bit, exec+read) |
| 2 | 0x10 | 0x00000000 | 0xFFFFF | 1 | 1 | 0 | 0x92 | Kernel data (32-bit, read+write) |
| 3 | 0x18 | 0x00000000 | 0xFFFFF | 1 | 1 | 3 | 0xFA | User code (Ring 3) |
| 4 | 0x20 | 0x00000000 | 0xFFFFF | 1 | 1 | 3 | 0xF2 | User data (Ring 3) |
**Decoded Access Bytes:**
- `0x9A` = 10011010b: Present(1), DPL(00), System(1), Exec(1), Read(1)
- `0x92` = 10010010b: Present(1), DPL(00), System(1), Write(1)
- `0xFA` = 11111010b: Present(1), DPL(11), System(1), Exec(1), Read(1)
- `0xF2` = 11110010b: Present(1), DPL(11), System(1), Write(1)
**Decoded Flag Bytes (with limit high nibble):**
- `0xCF` = 11001111b: G(1), D(1), L(0), AVL(0), Limit[19:16]=0xF
### GDT Descriptor (for LGDT instruction)
```
Offset  Size    Field
0x00    2 bytes Limit (size of GDT - 1, so 5 entries × 8 bytes - 1 = 39)
0x02    4 bytes Base address of GDT (linear address)
Total: 6 bytes
```
### VGA Text Mode Character Entry (2 bytes)
```
Byte 1    Byte 0
┌────────┬────────┐
│Attrib  │ ASCII  │
└────────┴────────┘
Attribute Byte:
  - Bit 7:   Blink (0 = no blink)
  - Bit 6-4: Background color (0-7)
  - Bit 3:   Bright foreground (0 = normal, 1 = bright)
  - Bit 2-0: Foreground color (0-7)
Colors:
  0=Black, 1=Blue, 2=Green, 3=Cyan, 4=Red, 5=Magenta, 6=Brown, 7=Light Grey
  8=Dark Grey, 9=Light Blue, 10=Light Green, 11=Light Cyan, 12=Light Red
  13=Light Magenta, 14=Yellow, 15=White
```
### Serial Port Register Map (COM1 = 0x3F8)
| Offset | Register | Read/Write | Description |
|--------|----------|------------|-------------|
| +0 | RBR/THR | R/W | Receive Buffer Register / Transmit Holding Register |
| +1 | IER | R/W | Interrupt Enable Register |
| +2 | IIR/FCR | R/W | Interrupt Identification / FIFO Control |
| +3 | LCR | R/W | Line Control Register |
| +4 | MCR | R/W | Modem Control Register |
| +5 | LSR | R | Line Status Register |
| +6 | MSR | R/W | Modem Status Register |
| +7 | SCR | R/W | Scratch Register |
**LCR bits for 8N1 (8 data, no parity, 1 stop):**
- Bit 7 (DLAB): 0 (disable divisor latch access after setting baud)
- Bit 6: 0 (break disabled)
- Bit 5-3: 000 (no parity)
- Bit 2: 0 (1 stop bit)
- Bit 1-0: 11 (8 data bits)
- Result: `0x03`
**Baud rate divisor for 115200:**
- Divisor = 115200 / 115200 = 1
- Low byte: `0x01`, High byte: `0x00`
### Linker Script Memory Regions
```
Section       Virtual Address    Physical Address   Size (typical)
.text         0x00100000         0x00100000         Variable
.rodata       0x00100000 + N     0x00100000 + N     Variable (4KB aligned)
.data         0x00100000 + M     0x00100000 + M     Variable (4KB aligned)
.bss          0x00100000 + P     0x00100000 + P     Variable (4KB aligned)
__bss_start   —                  —                  Symbol at .bss start
__bss_end     —                  —                  Symbol at .bss end
_kernel_end   —                  —                  Symbol after all sections
```
---
## Interface Contracts
### Bootloader Entry Point (`boot.asm`)
**Entry State (from BIOS):**
- CS:IP = 0x0000:0x7C00 (execution begins here)
- DL = boot drive number (0x00 = floppy, 0x80 = hard disk)
- All other registers: undefined
- CPU mode: 16-bit real mode
- Interrupts: enabled
- A20 line: typically disabled
**Exit State (to kernel):**
- CPU mode: 32-bit protected mode
- CS = 0x08 (kernel code selector)
- DS = ES = FS = GS = SS = 0x10 (kernel data selector)
- ESP = 0x90000 (stack pointer)
- IDT: invalid (interrupts disabled)
- CR0.PE = 1, CR0.PG = 0 (protected mode, paging disabled)
- Control transferred to `_start` in `entry.asm`
### GDT Setup (`gdt.asm`)
**Function:** `setup_gdt`
**Parameters:** None
**Returns:** Nothing (GDTR loaded via LGDT)
**Side Effects:**
- GDTR loaded with pointer to GDT descriptor
- GDT contains 5 valid entries
**Error Conditions:** None (cannot fail in software)
### A20 Line Enable (`a20.asm`)
**Function:** `enable_a20`
**Parameters:** None
**Returns:** 
- AX = 1 if A20 enabled successfully
- AX = 0 if all methods failed (hangs with error message)
**Methods Attempted (in order):**
1. BIOS INT 15h AX=2401 (safest, may not exist on all systems)
2. Fast A20 via port 0x92 (System Control Port A)
3. Keyboard controller (8042) command sequence
**Test Mechanism:**
```
Address 0x0000:0x0500 = 0x00500 linear
Address 0xFFFF:0x0510 = 0x100510 linear → wraps to 0x00510 if A20 disabled
If writing to 0x00500 changes 0x00510 → A20 disabled (wraparound)
If addresses are independent → A20 enabled
```
### Kernel Load (`load_kernel.asm`)
**Function:** `load_kernel`
**Parameters:** 
- DL = boot drive number (saved from BIOS)
- KERNEL_LOAD_SEGMENT = 0x1000 (segment for real-mode addressing)
- KERNEL_SECTORS = 64 (32KB, adjust for kernel size)
**Returns:** 
- Kernel loaded at 0x10000 (will be copied to 0x100000 after protected mode)
- CF = 0 on success, CF = 1 on error (hangs with error message)
**Disk Address Packet Structure:**
```
Offset  Size  Field
0x00    1     Packet size (16 bytes)
0x01    1     Reserved (0)
0x02    2     Number of sectors to read
0x04    2     Transfer buffer offset
0x06    2     Transfer buffer segment
0x08    8     Starting LBA (sector 2 = after MBR)
```
### Kernel Entry (`entry.asm`)
**Function:** `_start` (global symbol)
**Entry State:**
- 32-bit protected mode
- CS = 0x08, all data segments = 0x10
- ESP = valid kernel stack
- Interrupts disabled
**Actions:**
1. Clear direction flag (CLD) for string operations
2. Zero BSS section from `__bss_start` to `__bss_end`
3. Call `kernel_main` (C function)
**Returns:** Never (kernel_main should not return; if it does, halt loop)
### VGA Driver (`vga.h/c`)
**Function:** `vga_init`
**Parameters:** None
**Returns:** Void
**Side Effects:**
- Clears screen
- Sets cursor to (0, 0)
- Initializes color attribute to light grey on black
**Function:** `vga_put_char_at`
**Parameters:**
- `c`: char (character to display)
- `color`: uint8_t (attribute byte)
- `x`: int (column, 0-79)
- `y`: int (row, 0-24)
**Returns:** Void
**Precondition:** x ∈ [0, 79], y ∈ [0, 24]
**Behavior:** Writes character with attribute at position (x, y)
**Function:** `vga_put_char`
**Parameters:**
- `c`: char
- `color`: uint8_t
**Returns:** Void
**Behavior:** 
- Handles '\n' (newline), '\r' (carriage return), '\t' (tab), '\b' (backspace)
- Scrolls screen if cursor reaches bottom
- Updates hardware cursor position
**Function:** `vga_write`
**Parameters:**
- `str`: const char* (null-terminated string)
- `color`: uint8_t
**Returns:** Void
**Behavior:** Calls `vga_put_char` for each character
### Serial Driver (`serial.h/c`)
**Function:** `serial_init`
**Parameters:** None
**Returns:** bool (true if serial port working, false if loopback test failed)
**Side Effects:**
- COM1 (0x3F8) configured for 115200 baud, 8N1
- FIFO enabled with 14-byte threshold
- Loopback test performed
**Function:** `serial_put_char`
**Parameters:**
- `c`: char
**Returns:** Void
**Behavior:** 
- Waits for transmit buffer empty (LSR bit 5)
- Writes character to THR
**Function:** `serial_write`
**Parameters:**
- `str`: const char*
**Returns:** Void
**Behavior:** Calls `serial_put_char` for each character
### kprintf (`kprintf.h/c`)
**Function:** `kprintf`
**Parameters:**
- `format`: const char* (format string)
- `...`: variadic arguments
**Returns:** int (number of characters printed)
**Supported Format Specifiers:**
- `%c`: char
- `%s`: const char* (null-terminated, prints "(null)" if NULL)
- `%d`, `%i`: int32_t (signed decimal)
- `%u`: uint32_t (unsigned decimal)
- `%x`: uint32_t (lowercase hex)
- `%X`: uint32_t (uppercase hex)
- `%p`: void* (pointer as hex with "0x" prefix)
- `%%`: literal '%'
**Output:** Both VGA console and serial port (COM1)
---
## Algorithm Specification
### Algorithm: Protected Mode Transition
**Input:** 
- GDT configured with 5 entries
- GDTR pointing to GDT descriptor
- A20 line enabled
- Kernel loaded in memory
**Output:**
- CPU in 32-bit protected mode
- Code executing at protected mode entry point
**Procedure:**
```
1.  cli                    ; Disable interrupts (IDT invalid in protected mode)
2.  lgdt [gdt_descriptor]  ; Load GDTR with GDT address and limit
3.  mov eax, cr0           ; Read CR0
4.  or eax, 1              ; Set PE bit (bit 0)
5.  mov cr0, eax           ; Write CR0 (protected mode NOW active)
6.  jmp 0x08:protected_mode_entry  ; Far jump: load CS=0x08, flush pipeline
    ; --- Now in 32-bit mode ---
7.  [BITS 32]
protected_mode_entry:
8.  mov ax, 0x10           ; Kernel data selector
9.  mov ds, ax             ; Load all data segment registers
10. mov es, ax
11. mov fs, ax
12. mov gs, ax
13. mov ss, ax
14. mov esp, 0x90000       ; Set up 32-bit stack
15. jmp 0x08:0x100000      ; Jump to kernel entry
```
**Invariants:**
- Interrupts remain disabled throughout transition
- GDT remains at same linear address (not overwritten)
- No memory accesses between step 5 and step 6 (pipeline hazard)
**Critical Timing:**
- Steps 1-5 must execute without interruption
- Step 6 MUST be a far jump to load new CS descriptor
- Missing step 6 causes triple fault (32-bit code with 16-bit CS descriptor)
### Algorithm: BSS Zeroing
**Input:**
- EDI = `__bss_start` address
- ECX = `__bss_end` - `__bss_start` (byte count)
- EAX = 0
**Output:** All bytes in BSS section set to 0
**Procedure:**
```
1.  mov edi, __bss_start   ; Destination pointer
2.  mov ecx, __bss_end
3.  sub ecx, edi           ; Calculate byte count
4.  xor eax, eax           ; Zero value
5.  rep stosb              ; Store ECX bytes of AL at [EDI], increment EDI
```
**Invariant:** After execution, all uninitialized global/static variables are zero
### Algorithm: Integer to String Conversion (for kprintf)
**Input:**
- value: uint32_t (unsigned integer to convert)
- base: int (radix: 10 or 16)
- uppercase: bool (for hex output)
**Output:** String representation in internal buffer
**Procedure:**
```
1.  IF value == 0:
        output '0'
        RETURN
2.  buffer[0..31] = local array
3.  i = 0
4.  WHILE value > 0:
        digit = value % base
        buffer[i] = digits[digit]  ; digits = "0123456789abcdef" or uppercase
        i++
        value = value / base
5.  WHILE i > 0:
        i--
        output buffer[i]  ; Print in reverse order (LSB first in buffer)
```
**Edge Cases:**
- value = 0 → output "0"
- base = 16 with uppercase → use "0123456789ABCDEF"
- Maximum digits: 32 (for 32-bit value in binary; 10 for decimal, 8 for hex)
### Algorithm: VGA Scroll
**Input:** None (triggered when cursor_y >= 25)
**Output:** Screen contents shifted up by one line, bottom line cleared
**Procedure:**
```
1.  FOR y = 0 TO 23:
        FOR x = 0 TO 79:
            src_index = (y + 1) * 80 + x
            dst_index = y * 80 + x
            VGA_BUFFER[dst_index] = VGA_BUFFER[src_index]
2.  FOR x = 0 TO 79:
        VGA_BUFFER[24 * 80 + x] = vga_entry(' ', current_color)
3.  cursor_y = 24
4.  cursor_x = 0
```
**Memory Access Pattern:** Sequential reads followed by sequential writes (cache-friendly)
---
## Error Handling Matrix
| Error | Detection Point | Detection Method | Recovery | User-Visible Message |
|-------|-----------------|------------------|----------|---------------------|
| A20 line disabled | enable_a20 | Wraparound test fails after all methods | Hang with error message | "ERROR: Could not enable A20 line" |
| Disk read failure | load_kernel | BIOS INT 13h sets CF | Hang with error message | "Disk read error!" |
| Triple fault (GDT bad) | After CR0.PE set | CPU reset (QEMU -no-reboot to observe) | None (CPU resets) | System restarts |
| GDT not loaded | After lgdt | Wrong descriptor causes fault | None (triple fault) | System restarts |
| Wrong segment selector | After mode switch | General Protection Fault (vector 13) | None (no IDT yet) | System restarts |
| Kernel not at 0x100000 | At kernel entry | Garbage execution | None | Random behavior |
| Serial loopback fail | serial_init | Written byte ≠ read byte | Return false, continue without serial | "Serial init failed!" (VGA) |
| NULL string in kprintf | kvprintf | Check pointer == NULL | Print "(null)" | "(null)" |
| BSS symbols undefined | Link time | Linker error | Build fails | N/A (build error) |
---
## Implementation Sequence with Checkpoints
### Phase 1: Boot Stub and Segment Setup (4-6 hours)
**Files:** `boot/boot.asm` (skeleton)
**Tasks:**
1. Create assembly file with `[BITS 16]`, `[ORG 0x7C00]`
2. Implement segment register initialization (DS, ES, SS = 0)
3. Set up stack at 0x7C00 (grows down)
4. Save boot drive number from DL
5. Implement `print_string` using BIOS INT 10h
6. Add boot message ("Booting MyOS...")
7. Add padding and boot signature (0xAA55)
**Checkpoint:** 
- Build produces 512-byte boot.bin
- Verify signature: `xxd boot.bin | grep "55 aa"` at offset 0x1FE
- Run: `qemu-system-i386 -fda boot.bin` → see boot message
### Phase 2: A20 Line Enable (3-4 hours)
**Files:** `boot/a20.asm`
**Tasks:**
1. Implement `test_a20` using wraparound check
2. Implement `kbd_wait_input` (wait for port 0x64 bit 1 = 0)
3. Implement `kbd_wait_output` (wait for port 0x64 bit 0 = 1)
4. Implement BIOS method (INT 15h AX=2401)
5. Implement Fast A20 method (port 0x92)
6. Implement keyboard controller method (port 0x64/0x60 sequence)
7. Add error handling and hang on failure
**Checkpoint:**
- Test with QEMU (A20 enabled by default)
- Verify test_a20 returns AX=1
- Force test_a20 to fail → see error message
### Phase 3: Disk Read and Kernel Load (3-4 hours)
**Files:** `boot/load_kernel.asm`
**Tasks:**
1. Define Disk Address Packet structure
2. Reset disk system (INT 13h AH=0)
3. Implement extended read (INT 13h AH=42h)
4. Load kernel to 0x10000 (temporary location)
5. Add error handling for disk read failures
**Checkpoint:**
- Create dummy kernel.bin (just 0x90 bytes)
- Verify kernel loaded at correct address in QEMU monitor: `xp /16x 0x10000`
### Phase 4: GDT Configuration (4-5 hours)
**Files:** `boot/gdt.asm`
**Tasks:**
1. Define null descriptor (8 bytes of 0)
2. Define kernel code descriptor (0x00CF9A000000FFFF)
3. Define kernel data descriptor (0x00CF92000000FFFF)
4. Define user code descriptor (0x00CFFA000000FFFF)
5. Define user data descriptor (0x00CFF2000000FFFF)
6. Define GDT descriptor (limit + base)
7. Define selector constants (0x08, 0x10, 0x18, 0x20)
**Checkpoint:**
- Manually decode GDT entries against Intel manual
- Verify GDT size: 5 × 8 = 40 bytes
- Verify GDT descriptor limit = 39
### Phase 5: Protected Mode Transition (2-3 hours)
**Files:** `boot/boot.asm`, `boot/protected.asm`
**Tasks:**
1. Disable interrupts (cli)
2. Call setup_gdt (lgdt)
3. Set CR0.PE bit
4. Execute far jump to 32-bit code section
5. Implement 32-bit entry point
6. Load all data segment registers with 0x10
7. Set up 32-bit stack (ESP = 0x90000)
**Checkpoint:**
- Add debug output after mode switch (write to VGA directly)
- Run: `qemu-system-i386 -fda os.img -d int -no-reboot`
- Check QEMU log for no interrupts/triple faults
### Phase 6: Kernel Entry and BSS Zeroing (2-3 hours)
**Files:** `kernel/entry.asm`
**Tasks:**
1. Create entry point `_start` with `[BITS 32]`
2. Clear direction flag (cld)
3. Implement BSS zeroing using rep stosb
4. Call kernel_main
5. Add halt loop if kernel_main returns
**Checkpoint:**
- Create minimal kernel_main that writes to VGA directly
- Verify BSS zeroing: declare test variable, check it's 0
### Phase 7: VGA Text Mode Driver (3-4 hours)
**Files:** `kernel/vga.h`, `kernel/vga.c`
**Tasks:**
1. Define VGA_BUFFER address (0xB8000)
2. Define color enum and attribute functions
3. Implement vga_init (clear screen)
4. Implement vga_put_char_at
5. Implement vga_put_char with special character handling
6. Implement scroll function
7. Implement hardware cursor update (ports 0x3D4/0x3D5)
**Checkpoint:**
- Test: `vga_write("Hello, World!\n", color)` displays correctly
- Test: Fill screen and verify scroll
- Test: Cursor position matches text position
### Phase 8: Serial Port Driver (2-3 hours)
**Files:** `kernel/serial.h`, `kernel/serial.c`
**Tasks:**
1. Define port offsets (DATA, IER, FCR, LCR, MCR, LSR)
2. Implement outb/inb inline functions
3. Implement serial_init with 115200 baud, 8N1
4. Implement loopback test
5. Implement serial_put_char (wait for empty, write)
6. Implement serial_write
**Checkpoint:**
- Run: `qemu-system-i386 -fda os.img -serial stdio`
- Verify serial output appears in terminal
- Test loopback failure detection
### Phase 9: kprintf Implementation (2-3 hours)
**Files:** `kernel/kprintf.h`, `kernel/kprintf.c`, `include/stdarg.h`, `include/stdint.h`
**Tasks:**
1. Define stdint.h types (int8_t, uint32_t, etc.)
2. Define stdarg.h using __builtin_va_list
3. Implement print_char (VGA + serial)
4. Implement print_string
5. Implement print_uint (decimal, hex)
6. Implement print_int (signed, handles negative)
7. Implement print_pointer (0x prefix + hex)
8. Implement kvprintf (format string parsing)
9. Implement kprintf (variadic wrapper)
**Checkpoint:**
- Test format specifiers: `%c %s %d %x %p %%`
- Test edge cases: NULL string, INT_MIN, 0
- Verify output on both VGA and serial
### Phase 10: Linker Script and Build System (3-4 hours)
**Files:** `kernel/linker.ld`, `Makefile`
**Tasks:**
1. Create linker script with ENTRY(_start)
2. Define memory layout (.text at 0x100000)
3. Define __bss_start and __bss_end symbols
4. Add 4KB alignment for sections
5. Create Makefile with NASM, GCC, LD rules
6. Implement boot.bin, kernel.bin, os.img targets
7. Add run and debug targets
8. Add clean target
**Checkpoint:**
- Run: `make clean && make`
- Run: `make run` → see full boot sequence with welcome message
- Run: `make debug` → GDB connects, can set breakpoints
---
## Test Specification
### Test: Boot Signature
**Function:** Boot sector validity
**Happy Path:** 
- Build boot.bin
- Verify bytes 510-511 are 0x55, 0xAA
- Command: `xxd boot.bin | tail -1` → ends with "55 aa"
**Failure Case:**
- Corrupt signature manually
- QEMU refuses to boot (no message, just hangs)
### Test: GDT Loading
**Function:** Protected mode transition
**Happy Path:**
- Boot completes without triple fault
- QEMU log (`-d int`) shows no exceptions before first kprintf
**Edge Case:**
- Modify GDT entry to wrong access byte
- System triple faults immediately after far jump
### Test: A20 Line
**Function:** Memory access above 1MB
**Happy Path:**
- test_a20 returns 1
- Kernel loaded at 0x100000 is accessible
**Failure Simulation:**
- Comment out all enable methods
- test_a20 returns 0, error message displayed
### Test: VGA Output
**Function:** Console display
**Happy Path:**
```c
vga_write("Test\n", vga_color_attr(VGA_COLOR_WHITE, VGA_COLOR_BLUE));
```
- "Test" appears on screen with white text on blue background
- Cursor moves to next line
**Edge Cases:**
- Write 80+ characters without newline → wraps to next line
- Write 25 lines → screen scrolls
- Write '\b' → cursor moves back, character erased
- Write '\t' → cursor aligns to next 4-column boundary
### Test: Serial Output
**Function:** Debug console
**Happy Path:**
- Run QEMU with `-serial stdio`
- kprintf output appears in terminal
- Characters match VGA output exactly
**Failure Case:**
- Disable serial_init
- Output only appears on VGA, not terminal
### Test: kprintf Format Specifiers
**Function:** Formatted output
**Test Cases:**
```c
kprintf("Char: %c\n", 'X');           // Output: "Char: X"
kprintf("String: %s\n", "hello");     // Output: "String: hello"
kprintf("Int: %d\n", 42);             // Output: "Int: 42"
kprintf("Neg: %d\n", -12345);         // Output: "Neg: -12345"
kprintf("Hex: %x\n", 0xDEAD);         // Output: "Hex: dead"
kprintf("HexUpper: %X\n", 0xBEEF);    // Output: "HexUpper: BEEF"
kprintf("Ptr: %p\n", (void*)0x100000);// Output: "Ptr: 0x00100000"
kprintf("Zero: %d\n", 0);             // Output: "Zero: 0"
kprintf("NULL: %s\n", NULL);          // Output: "NULL: (null)"
kprintf("Percent: %%\n");             // Output: "Percent: %"
```
### Test: BSS Zeroing
**Function:** Uninitialized variable initialization
**Test:**
```c
// In kernel/main.c
static int test_bss_var;  // Should be 0
void kernel_main(void) {
    kprintf("BSS test: %d\n", test_bss_var);
    // Must print 0
}
```
### Test: Full Boot Sequence
**Happy Path:**
1. Build: `make`
2. Run: `make run`
3. Expected output:
```
Booting MyOS...
Loading kernel...
========================================
  MyOS Kernel v0.1
  Built on [DATE] at [TIME]
========================================
Kernel loaded at: 0x00100000
Stack pointer:    0x00090000
Integer:  42 (0x2a)
Negative: -12345
Pointer:  0xdeadbeef
String:   Hello, World!
Char:     X
Kernel initialized successfully.
System halted.
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Boot to kernel_main | < 100ms | QEMU `-d cpu_reset` timing |
| VGA character write | < 1μs | Inline measurement (not critical) |
| Serial character transmit | ~86μs/char | 115200 baud = 1 bit / 86.8μs, 10 bits/char |
| Full boot sequence | < 500ms | Wall clock from QEMU start to "System halted" |
| GDT load | < 1μs | Single LGDT instruction |
| Protected mode switch | < 10μs | CR0 write + far jump |
---
## State Machine: Boot Sequence
```
[BIOS] ──loads MBR──> [REAL_MODE]
                          │
                          ▼
                    [SETUP_SEGMENTS]
                          │
                          ▼
                    [ENABLE_A20]
                          │
                    ┌─────┴─────┐
                    │ FAIL      │ SUCCESS
                    ▼           ▼
              [HALT_ERROR] [LOAD_KERNEL]
                                │
                          ┌─────┴─────┐
                          │ FAIL      │ SUCCESS
                          ▼           ▼
                    [HALT_ERROR] [SETUP_GDT]
                                    │
                                    ▼
                              [SET_CR0_PE]
                                    │
                                    ▼
                              [FAR_JUMP]
                                    │
                                    ▼
                          [PROTECTED_MODE]
                                    │
                                    ▼
                          [LOAD_SEGMENTS]
                                    │
                                    ▼
                          [SETUP_STACK]
                                    │
                                    ▼
                          [ZERO_BSS]
                                    │
                                    ▼
                          [CALL_KERNEL_MAIN]
                                    │
                                    ▼
                              [RUNNING]
```
**Illegal Transitions:**
- CR0.PE set without GDT loaded → Triple fault
- Far jump without CR0.PE → Stays in real mode
- 32-bit code with 16-bit CS → Garbage execution
- Interrupts enabled without IDT → Triple fault on first IRQ
---
## Hardware Soul: Critical Timing Analysis
### GDT Load and Mode Switch
```
Instruction         | Cycles (approx) | Cache Behavior
--------------------|-----------------|----------------
lgdt [gdt_desc]     | 20-50           | Cache miss (cold path)
mov eax, cr0        | 10-20           | Register only
or eax, 1           | 1               | Register only
mov cr0, eax        | 20-40           | Serializing operation
jmp 0x08:label      | 50-100          | Pipeline flush, cache miss on target
mov ax, 0x10        | 1               | Register only
mov ds, ax          | 10-20           | Segment load, TLB check
(total 4 segs)      | 40-80           |
mov esp, 0x90000    | 1               | Register only
─────────────────────────────────────────────────────────
TOTAL               | 150-300 cycles  | ~0.5-1μs at 1GHz
```
### VGA Write Path
```
Operation           | Access Type     | Timing
--------------------|-----------------|--------
Compute buffer idx  | CPU             | < 10 cycles
Write to 0xB8000    | MMIO (uncached) | ~100-200 cycles
Update cursor       | I/O ports       | ~500 cycles (2 outb)
─────────────────────────────────────────────────────────
Per character       |                 | ~700 cycles
```
**Note:** VGA buffer is memory-mapped I/O and bypasses cache. Every write goes directly to the video hardware.
### Serial Transmit
```
Baud rate: 115200
Bits per character: 10 (1 start + 8 data + 1 stop)
Time per character: 10 / 115200 = 86.8μs
Operation           | Timing
--------------------|--------
Wait for empty      | 0-86μs (buffered)
Write to THR        | < 1μs
Character shift out | 86.8μs (hardware)
```
**Throughput:** ~11,520 characters/second maximum
---
## Concurrency Model
**Single-threaded boot sequence.** No concurrency until interrupts are enabled in later milestones.
**Interrupt State:**
- Real mode: Interrupts enabled (BIOS IVT active)
- During mode switch: Interrupts disabled (cli before lgdt)
- Protected mode entry: Interrupts disabled (no IDT)
- kernel_main: Interrupts disabled (caller's responsibility to enable)
**No locks required** at this stage. All operations are sequential.
---
## Crash Recovery
**This module has no persistent state.** All memory is volatile.
**Failure Modes:**
| Failure | Symptom | Recovery |
|---------|---------|----------|
| Triple fault | CPU reset | Fix code, rebuild |
| A20 disabled | Kernel not loaded | Enable A20 |
| Disk read fail | Kernel garbage | Fix disk image |
**Debugging Aids:**
- QEMU `-d int` logs all interrupts
- QEMU `-no-reboot` pauses on triple fault
- QEMU `-s -S` enables GDB remote debugging
- Serial output provides persistent log
---
## Build System Specification
### Makefile Targets
```makefile
# Toolchain
AS = nasm
CC = gcc
LD = ld
# Flags
CFLAGS = -ffreestanding -fno-stack-protector -fno-pic -m32 \
         -Wall -Wextra -nostdlib -nostdinc -I include -O2 -g
ASFLAGS = -f elf32 -g -F dwarf
LDFLAGS = -m elf_i386 -nostdlib -T kernel/linker.ld
# Files
ASM_SOURCES = kernel/entry.asm
C_SOURCES = kernel/main.c kernel/vga.c kernel/serial.c kernel/kprintf.c
ASM_OBJECTS = $(ASM_SOURCES:.asm=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)
OBJECTS = $(ASM_OBJECTS) $(C_OBJECTS)
KERNEL_BIN = kernel.bin
BOOT_BIN = boot.bin
OS_IMAGE = os.img
.PHONY: all clean run debug
all: $(OS_IMAGE)
# Bootloader (raw binary, 512 bytes)
$(BOOT_BIN): boot/boot.asm boot/gdt.asm boot/a20.asm boot/load_kernel.asm
	$(AS) -f bin boot/boot.asm -o $(BOOT_BIN)
# Kernel objects
%.o: %.asm
	$(AS) $(ASFLAGS) $< -o $@
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
# Link kernel
$(KERNEL_BIN): $(OBJECTS)
	$(LD) $(LDFLAGS) -o $@ $(OBJECTS)
# Create disk image
$(OS_IMAGE): $(BOOT_BIN) $(KERNEL_BIN)
	dd if=/dev/zero of=$@ bs=512 count=2880
	dd if=$(BOOT_BIN) of=$@ bs=512 count=1 conv=notrunc
	dd if=$(KERNEL_BIN) of=$@ bs=512 seek=2 conv=notrunc
# Run in QEMU
run: $(OS_IMAGE)
	qemu-system-i386 -fda $(OS_IMAGE) -serial stdio
# Debug with GDB
debug: $(OS_IMAGE)
	qemu-system-i386 -fda $(OS_IMAGE) -serial stdio -s -S &
	gdb -ex "target remote localhost:1234" -ex "break kernel_main" -ex "continue"
clean:
	rm -f $(BOOT_BIN) $(KERNEL_BIN) $(OS_IMAGE) $(OBJECTS) kernel/*.o
```
---
## Synced Criteria
[[CRITERIA_JSON: {"module_id": "build-os-m1", "criteria": ["Bootloader code assembled as raw binary exactly 512 bytes with 0x55 signature at byte 510 and 0xAA at byte 511", "Bootloader entry point at 0x7C00 with [ORG 0x7C00] directive in NASM syntax", "Bootloader saves boot drive number from DL register before any BIOS calls", "Bootloader initializes segment registers DS, ES, SS to 0x0000 and sets SP to 0x7C00 (stack grows down)", "A20 line enable attempts three methods in order: BIOS INT 15h AX=2401, Fast A20 via port 0x92, keyboard controller via ports 0x64/0x60", "A20 test uses wraparound check comparing addresses 0x0000:0x0500 and 0xFFFF:0x0510", "A20 enable halts with error message if all three methods fail", "Disk read uses BIOS INT 13h AH=42h (extended read) with Disk Address Packet structure", "Disk Address Packet is 16 bytes: size(1), reserved(1), sector count(2), buffer offset(2), buffer segment(2), LBA(8)", "Kernel loaded to segment 0x1000 (linear address 0x10000) using extended disk read", "GDT contains exactly 5 entries each 8 bytes: null(0), kernel code(0x00CF9A000000FFFF), kernel data(0x00CF92000000FFFF), user code(0x00CFFA000000FFFF), user data(0x00CFF2000000FFFF)", "GDT descriptor is 6 bytes: 16-bit limit (39 for 5 entries) + 32-bit base address", "GDT loaded with lgdt instruction before CR0.PE is set", "Protected mode enabled by setting CR0 bit 0 (PE) to 1", "Far jump jmp 0x08:protected_mode_entry executed immediately after CR0.PE set", "All data segment registers (DS, ES, FS, GS, SS) loaded with selector 0x10 after protected mode entry", "32-bit stack pointer ESP initialized to 0x90000", "Kernel entry point _start zeroes BSS section from __bss_start to __bss_end using rep stosb", "Kernel entry point clears direction flag (cld) before BSS zeroing", "Kernel entry point calls kernel_main C function after BSS zeroing", "VGA text mode buffer accessed at physical address 0xB8000 as volatile uint16_t*", "VGA driver implements vga_entry(c, color) returning (c | (color << 8))", "VGA driver handles newline (cursor_x=0, cursor_y++), carriage return (cursor_x=0), tab (align to 4), backspace (cursor_x--)", "VGA driver scrolls screen when cursor_y >= 25 by copying all lines up and clearing bottom line", "VGA driver updates hardware cursor via ports 0x3D4/0x3D5 with cursor position = cursor_y * 80 + cursor_x", "Serial port COM1 initialized at base 0x3F8 with 115200 baud, 8 data bits, no parity, 1 stop bit (8N1)", "Serial init sets DLAB bit in LCR (0x80), writes divisor 1 to ports 0x3F8/0x3F9, then clears DLAB and sets 0x03 for 8N1", "Serial init enables FIFO with 0xC7 to port 0x3FA (enable + clear + 14-byte threshold)", "Serial init performs loopback test by writing 0xAE to port 0x3FA with MCR=0x1E and verifying readback", "Serial put_char waits for LSR bit 5 (transmit empty) before writing to THR", "kprintf supports format specifiers %c (char), %s (string, prints (null) if NULL), %d/%i (signed int), %u (unsigned), %x/%X (hex), %p (pointer with 0x prefix), %% (literal percent)", "kprintf outputs to both VGA console and serial port simultaneously", "kprintf integer conversion handles value 0 by outputting character '0'", "kprintf negative integer outputs '-' followed by absolute value", "stdarg.h implemented using __builtin_va_list, __builtin_va_start, __builtin_va_arg, __builtin_va_end", "stdint.h defines int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, uintptr_t, intptr_t", "Linker script places .text section at 0x100000 with ENTRY(_start)", "Linker script defines __bss_start and __bss_end symbols at .bss section boundaries", "Linker script uses 4KB (4K) alignment for all sections", "Build system produces boot.bin (512 bytes), kernel.bin (ELF32), and os.img (floppy image)", "os.img created by combining boot.bin at sector 0 and kernel.bin starting at sector 2", "QEMU runs os.img with qemu-system-i386 -fda os.img -serial stdio", "Kernel displays welcome message on both VGA console and serial output confirming successful boot"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-os-m2 -->
# Technical Design Document: Interrupts, Exceptions, and Keyboard
**Module ID:** `build-os-m2`
---
## Module Charter
The interrupt module transforms a passive kernel into a reactive system capable of responding to asynchronous hardware events and CPU-detected errors. It implements the Interrupt Descriptor Table (IDT) with 256 gate entries, remaps the dual 8259 PICs to route hardware IRQs to vectors 32-47 (avoiding conflicts with CPU exception vectors 0-31), provides assembly stubs that save and restore complete CPU state, dispatches to C handlers for exceptions and device drivers, manages the End-of-Interrupt (EOI) protocol with the PIC, implements a Programmable Interval Timer (PIT) driver for periodic scheduling ticks, and provides a PS/2 keyboard driver with scancode-to-ASCII translation and a circular input buffer.
**This module does NOT:** implement preemptive scheduling (that's the scheduler's job, triggered by timer ticks), handle page faults for virtual memory (Milestone 3), or implement system calls via INT 0x80 (Milestone 4). It provides the infrastructure for all interrupt-driven functionality.
**Upstream dependencies:** GDT and protected mode from Milestone 1 (segment selectors 0x08/0x10 required for IDT gates); kernel heap for IDT allocation.
**Downstream consumers:** Timer subsystem (IRQ0) for scheduling; keyboard subsystem (IRQ1) for input; page fault handler (exception 14) for virtual memory; system call interface (INT 0x80) for user-kernel transitions.
**Invariants:**
- IDT must be loaded (lidt) before interrupts are enabled (sti)
- PIC must be remapped before any IRQs are unmasked
- Every IRQ handler must send EOI before returning (or PIC blocks all future interrupts)
- All interrupt handlers must save and restore complete register state
- Error code handling differs between exceptions (8, 10-14 push error codes; others don't)
- Interrupt gates disable interrupts during handler execution (IF cleared automatically)
---
## File Structure
```
project/
├── 02_kernel/
│   ├── 10_idt.h              # IDT structure and function declarations
│   ├── 11_idt.c              # IDT initialization and gate management
│   ├── 12_idt_stubs.asm      # Assembly ISR stubs for all 256 vectors
│   ├── 13_interrupts.h       # Interrupt handling interface
│   ├── 14_interrupts.c       # Common interrupt dispatcher, exception handlers
│   ├── 15_pic.h              # PIC interface
│   ├── 16_pic.c              # PIC initialization, remapping, EOI
│   ├── 17_timer.h            # PIT timer interface
│   ├── 18_timer.c            # PIT initialization and tick counter
│   ├── 19_keyboard.h         # Keyboard driver interface
│   ├── 20_keyboard.c         # PS/2 keyboard implementation
│   └── 21_exception_names.c  # Exception message strings
└── 03_include/
    └── 03_cpu.h              # CPU register structures for ISRs
```
---
## Complete Data Model
### IDT Entry Structure (8 bytes per gate)

![IDT Entry Gate Descriptor](./diagrams/tdd-diag-m2-01.svg)

```
Byte 7    Byte 6    Byte 5    Byte 4    Byte 3    Byte 2    Byte 1    Byte 0
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│Offset  │ P│D│0│ │ Reserved│ P│DPL│S│  Type   │Reserved│Segment │Offset  │
│31:16   │  │ │ │ │         │  │   │ │         │        │Selector│15:0    │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Offset (32-bit, split):
  - Bytes 0-1:  Offset[15:0]  (low 16 bits of handler address)
  - Bytes 6-7:  Offset[31:16] (high 16 bits of handler address)
Segment Selector (16-bit, bytes 2-3):
  - Value: 0x0008 (kernel code segment selector from GDT)
Reserved (byte 4): Must be 0
Type/Attribute Byte (byte 5):
  - Bit 7 (P):     Present (1 = gate is valid)
  - Bit 6-5 (DPL): Descriptor Privilege Level
                   - 00 = Ring 0 only can invoke
                   - 11 = Any ring can invoke (for syscalls)
  - Bit 4 (S):     System (0 = system segment like interrupt gate)
  - Bit 3-0 (Type):
    - 0xE (1110) = 32-bit Interrupt Gate (IF cleared on entry)
    - 0xF (1111) = 32-bit Trap Gate (IF unchanged)
Flags (byte 6, bits 0-3): Reserved, must be 0
Reserved (byte 6, bits 4-7): 0
```
**IDT Gate Types Used:**
| Type | Value | Use Case | IF on Entry |
|------|-------|----------|-------------|
| Interrupt Gate | 0x8E | Hardware IRQs, exceptions | Cleared (interrupts disabled) |
| Trap Gate | 0x8F | Debug, breakpoint | Unchanged |
| User Interrupt | 0xEE | INT 0x80 syscalls | Cleared, DPL=3 |
### IDT Pointer Structure (for LIDT)
```c
typedef struct {
    uint16_t limit;    // Size of IDT - 1 (256 entries × 8 bytes - 1 = 2047)
    uint32_t base;     // Linear address of IDT array
} __attribute__((packed)) idt_ptr_t;
```
### CPU Exception Categories

![Interrupt Stack Frame](./diagrams/tdd-diag-m2-02.svg)

| Vector | Name | Error Code? | Cause | Fatal? |
|--------|------|-------------|-------|--------|
| 0 | Divide Error | No | DIV/IDIV by zero | Process kill |
| 1 | Debug | No | Debug trap | Continue |
| 2 | NMI | No | Hardware non-maskable | Handler |
| 3 | Breakpoint | No | INT 3 instruction | Continue |
| 4 | Overflow | No | INTO with OF set | Process kill |
| 5 | BOUND Range | No | BOUND check failed | Process kill |
| 6 | Invalid Opcode | No | UD2 or undefined opcode | Process kill |
| 7 | Device Not Available | No | FPU instruction without CR0.TS | Handler |
| 8 | **Double Fault** | **Yes** | Exception during exception | **HALT** |
| 9 | Coprocessor Segment Overrun | No | FPU operand wrap | Process kill |
| 10 | Invalid TSS | Yes | Task switch with bad TSS | HALT |
| 11 | Segment Not Present | Yes | Loading P=0 selector | Process kill |
| 12 | Stack-Segment Fault | Yes | SS limit violation | Process kill |
| 13 | General Protection | Yes | Privilege/bad selector | Process kill |
| 14 | **Page Fault** | **Yes** | Invalid page table entry | Handler/Kill |
| 16 | x87 FPU Error | No | FPU exception | Process kill |
| 17 | Alignment Check | Yes | Misaligned access + AC | Process kill |
| 18 | Machine Check | No | Hardware error | HALT |
| 19 | SIMD Exception | No | SSE/AVX exception | Process kill |
### Interrupt Stack Frame

![ISR Entry and Exit Flow](./diagrams/tdd-diag-m2-03.svg)

```
When interrupt occurs, CPU pushes (from high to low addresses):
┌──────────────────────┐  High addresses
│ SS                   │  (only if privilege change: Ring 3 → Ring 0)
│ ESP                  │  (only if privilege change)
├──────────────────────┤
│ EFLAGS               │  ← Always pushed
├──────────────────────┤
│ CS                   │  ← Always pushed
│ EIP (return address) │  ← Always pushed
├──────────────────────┤
│ Error Code           │  ← Only for exceptions 8, 10, 11, 12, 13, 14, 17
├──────────────────────┤
│ ... handler stack    │  Low addresses
└──────────────────────┘
Stack grows downward. ESP at handler entry points to:
  - Error code (if present) or
  - EIP (if no error code)
```
### Register Save Structure (for C handlers)
```c
// Passed to C interrupt handlers
typedef struct {
    // Segment registers (pushed by our stub)
    uint32_t gs;         // Offset 0x00
    uint32_t fs;         // Offset 0x04
    uint32_t es;         // Offset 0x08
    uint32_t ds;         // Offset 0x0C
    // General purpose (pusha order: EDI, ESI, EBP, ESP, EBX, EDX, ECX, EAX)
    uint32_t edi;        // Offset 0x10
    uint32_t esi;        // Offset 0x14
    uint32_t ebp;        // Offset 0x18
    uint32_t esp;        // Offset 0x1C (value BEFORE pusha)
    uint32_t ebx;        // Offset 0x20
    uint32_t edx;        // Offset 0x24
    uint32_t ecx;        // Offset 0x28
    uint32_t eax;        // Offset 0x2C
    // Interrupt info (pushed by our stub)
    uint32_t int_no;     // Offset 0x30 (interrupt vector number)
    uint32_t err_code;   // Offset 0x34 (error code, 0 if none pushed by CPU)
    // Pushed by CPU (for iret)
    uint32_t eip;        // Offset 0x38
    uint32_t cs;         // Offset 0x3C
    uint32_t eflags;     // Offset 0x40
    // Only present if privilege change occurred
    uint32_t useresp;    // Offset 0x44 (user stack pointer)
    uint32_t ss;         // Offset 0x48 (user stack segment)
} __attribute__((packed)) registers_t;
// Total size: 0x4C (76 bytes) for kernel-to-kernel interrupts
//             0x54 (84 bytes) for user-to-kernel interrupts
```
### PIC 8259 Register Map

![PIC Master/Slave Cascade](./diagrams/tdd-diag-m2-04.svg)

| Port | Master (0x20/0x21) | Slave (0xA0/0xA1) | Description |
|------|-------------------|-------------------|-------------|
| +0 (Cmd) | 0x20 | 0xA0 | Command/Status register |
| +1 (Data) | 0x21 | 0xA1 | Data register (mask/ICW2-4) |
**ICW (Initialization Command Word) Sequence:**
```
ICW1 (write to command port):
  Bit 7-5: 000 (required for 8086 mode)
  Bit 4:   1   (initialize)
  Bit 3:   0   (edge-triggered)
  Bit 2:   0   (8-byte interval for vectors)
  Bit 1:   0   (cascade mode, not single)
  Bit 0:   1   (need ICW4)
  Value:   0x11
ICW2 (write to data port): Vector offset
  Master:  0x20 (IRQ0 → vector 32)
  Slave:   0x28 (IRQ8 → vector 40)
ICW3 (write to data port): Cascade configuration
  Master:  0x04 (bit 2 set = IRQ2 has slave)
  Slave:   0x02 (slave ID = 2)
ICW4 (write to data port): Mode
  Bit 4:   0   (not special fully nested)
  Bit 3-2: 00  (non-buffered)
  Bit 1:   0   (normal EOI)
  Bit 0:   1   (8086 mode)
  Value:   0x01
```
**OCW (Operation Control Word):**
```
OCW1 (write to data port): Interrupt mask
  Bit N = 1: Mask (disable) IRQ N
  Bit N = 0: Unmask (enable) IRQ N
OCW2 (write to command port): EOI and priority
  0x20 = Non-specific EOI (clear highest ISR bit)
  0x60 = Specific EOI for IRQ 6 (example)
OCW3 (write to command port): Read register command
  0x0A = Read IRR on next read from command port
  0x0B = Read ISR on next read from command port
```
### PIT 8253/8254 Channel 0

![PIC Remapping Sequence](./diagrams/tdd-diag-m2-05.svg)

| Port | Name | Description |
|------|------|-------------|
| 0x40 | Channel 0 Data | Read/write divisor |
| 0x41 | Channel 1 Data | (unused, memory refresh on PC) |
| 0x42 | Channel 2 Data | Speaker tone |
| 0x43 | Mode Register | Command byte |
**Mode Register Byte:**
```
Bit 7-6: Channel select (00 = channel 0)
Bit 5-4: Access mode (11 = lobyte then hibyte)
Bit 3-1: Mode (011 = square wave generator)
Bit 0:   BCD (0 = binary, not BCD)
Value:   0x36
```
**Divisor Calculation:**
```
Base frequency: 1,193,182 Hz (≈1.193 MHz)
Output freq = 1,193,182 / divisor
For 100 Hz: divisor = 1,193,182 / 100 = 11,931 (0x2E9B)
For 1000 Hz: divisor = 1,193,182 / 1000 = 1,193 (0x04A9)
```
### PS/2 Keyboard Scancode Set 1

![EOI Decision Tree](./diagrams/tdd-diag-m2-06.svg)

| Scancode | Key | Make | Break |
|----------|-----|------|-------|
| 0x01 | Escape | 0x01 | 0x81 |
| 0x02-0x0B | 1-0 | 0x02-0x0B | +0x80 |
| 0x0E | Backspace | 0x0E | 0x8E |
| 0x0F | Tab | 0x0F | 0x8F |
| 0x1D | Left Ctrl | 0x1D | 0x9D |
| 0x2A | Left Shift | 0x2A | 0xAA |
| 0x36 | Right Shift | 0x36 | 0xB6 |
| 0x38 | Left Alt | 0x38 | 0xB8 |
| 0x3A | Caps Lock | 0x3A | 0xBA |
| 0x45 | Num Lock | 0x45 | 0xC5 |
| 0x46 | Scroll Lock | 0x46 | 0xC6 |
| 0x1E-0x26 | A-L | varies | +0x80 |
| 0x2C-0x32 | Z-M | varies | +0x80 |
| 0x39 | Space | 0x39 | 0xB9 |
**Break code = Make code | 0x80** (for most keys)
**Extended keys (start with 0xE0):**
- Arrow keys, Insert, Delete, Home, End, Page Up/Down
- Right Ctrl (0xE0 0x1D), Right Alt (0xE0 0x38)
### Keyboard Buffer Structure
```c
#define KB_BUFFER_SIZE 256
typedef struct {
    char buffer[KB_BUFFER_SIZE];     // Circular buffer storage
    volatile uint32_t read_pos;       // Index to read from
    volatile uint32_t write_pos;      // Index to write to
    volatile bool shift_held;         // Left or right shift
    volatile bool ctrl_held;          // Left or right ctrl
    volatile bool alt_held;           // Left or right alt
    volatile bool caps_lock;          // Caps lock state
    volatile bool num_lock;           // Num lock state
} keyboard_state_t;
// Buffer empty when: read_pos == write_pos
// Buffer full when: (write_pos + 1) % SIZE == read_pos
// Capacity: SIZE - 1 = 255 characters
```
---
## Interface Contracts
### IDT Initialization (`idt.h/c`)
```c
/**
 * Initialize the IDT with 256 entries.
 * Must be called before enabling interrupts.
 * 
 * Preconditions: GDT loaded, kernel heap available
 * Postconditions: IDT allocated, all gates set to default handler
 * Side effects: Allocates 2KB for IDT, loads IDTR via lidt
 */
void idt_init(void);
/**
 * Set an IDT gate entry.
 * 
 * @param num       Vector number (0-255)
 * @param handler   Address of handler function (assembly stub)
 * @param selector  Code segment selector (0x08 for kernel)
 * @param type      Gate type (0x8E for interrupt, 0xEE for user)
 * 
 * Preconditions: idt_init() called, num < 256
 * Postconditions: IDT entry updated, no TLB/IDTR reload needed
 */
void idt_set_gate(uint8_t num, uint32_t handler, uint16_t selector, uint8_t type);
/**
 * Load IDT (assembly wrapper for lidt).
 * 
 * @param idt_ptr  Address of idt_ptr_t structure
 */
extern void idt_load(uint32_t idt_ptr);
```
### Interrupt Dispatcher (`interrupts.h/c`)
```c
/**
 * Register a custom handler for an interrupt vector.
 * 
 * @param n       Vector number (0-255)
 * @param handler Function pointer to handler
 * 
 * Preconditions: idt_init() called
 * Postconditions: Handler called on next interrupt for this vector
 * 
 * Note: For IRQs (32-47), EOI is sent AFTER handler returns.
 */
void register_interrupt_handler(uint8_t n, isr_t handler);
/**
 * Main interrupt handler called from assembly stub.
 * 
 * @param regs  Pointer to saved register structure on stack
 * 
 * This function:
 * 1. Looks up custom handler for regs->int_no
 * 2. If found, calls it
 * 3. If not found and vector < 32, calls handle_exception()
 * 4. If vector >= 32 and < 48, sends EOI to PIC
 * 5. Returns to assembly stub for iret
 */
void interrupt_handler(registers_t *regs);
/**
 * Handle CPU exceptions (vectors 0-31).
 * Prints diagnostic and halts (except for debug/breakpoint).
 * 
 * @param regs  Pointer to saved register structure
 * 
 * Fatal exceptions: halt with cli/hlt loop
 * Non-fatal (3, 4): return to continue execution
 */
void handle_exception(registers_t *regs);
```
### PIC Driver (`pic.h/c`)
```c
/**
 * Initialize and remap the PIC.
 * 
 * Remaps:
 *   IRQ0-7  → vectors 32-39 (master)
 *   IRQ8-15 → vectors 40-47 (slave)
 * 
 * Preconditions: None (can call before IDT)
 * Postconditions: PICs initialized, all IRQs masked
 * Side effects: I/O port writes to 0x20, 0x21, 0xA0, 0xA1
 */
void pic_init(void);
/**
 * Send End of Interrupt to PIC.
 * 
 * @param irq  IRQ number (0-15), NOT vector number
 * 
 * Must be called at end of each IRQ handler.
 * For IRQ8-15, sends EOI to both slave and master.
 * 
 * Preconditions: PIC initialized, in IRQ handler
 * Postconditions: PIC can deliver next interrupt
 */
void pic_send_eoi(uint8_t irq);
/**
 * Mask (disable) an IRQ line.
 * 
 * @param irq  IRQ number (0-15)
 */
void pic_mask_irq(uint8_t irq);
/**
 * Unmask (enable) an IRQ line.
 * 
 * @param irq  IRQ number (0-15)
 */
void pic_unmask_irq(uint8_t irq);
```
### Timer Driver (`timer.h/c`)
```c
/**
 * Initialize PIT channel 0 at specified frequency.
 * 
 * @param freq  Desired frequency in Hz (1-1193182)
 * 
 * Calculates divisor and programs PIT.
 * Registers handler for IRQ0 (vector 32).
 * Unmasks IRQ0.
 * 
 * Preconditions: IDT loaded, PIC initialized
 * Postconditions: Timer interrupt fires at specified rate
 */
void timer_init(uint32_t freq);
/**
 * Get current tick count.
 * 
 * @return  Number of timer interrupts since boot
 */
uint64_t timer_get_ticks(void);
/**
 * Get seconds since boot.
 * 
 * @return  Approximate seconds (ticks / frequency)
 */
uint64_t timer_get_seconds(void);
/**
 * Sleep for specified milliseconds (busy-wait).
 * 
 * @param ms  Milliseconds to sleep
 * 
 * Uses hlt instruction to wait for next interrupt.
 */
void timer_sleep_ms(uint32_t ms);
```
### Keyboard Driver (`keyboard.h/c`)
```c
/**
 * Initialize PS/2 keyboard driver.
 * 
 * Registers handler for IRQ1 (vector 33).
 * Unmasks IRQ1.
 * Initializes modifier state to all false.
 * 
 * Preconditions: IDT loaded, PIC initialized
 * Postconditions: Keyboard interrupts enabled
 */
void keyboard_init(void);
/**
 * Check if character available in buffer.
 * 
 * @return  true if character ready, false otherwise
 */
bool keyboard_has_char(void);
/**
 * Read character from buffer (blocking).
 * 
 * @return  ASCII character
 * 
 * Blocks with hlt until character available.
 */
char keyboard_getchar(void);
/**
 * Read character from buffer (non-blocking).
 * 
 * @return  ASCII character, or 0 if none available
 */
char keyboard_try_getchar(void);
/**
 * Query modifier key state.
 */
bool keyboard_shift_held(void);
bool keyboard_ctrl_held(void);
bool keyboard_alt_held(void);
```
---
## Algorithm Specification
### Algorithm: IDT Gate Setting
```
INPUT: num (vector 0-255), handler (32-bit address), selector (16-bit), type (8-bit)
OUTPUT: IDT entry updated
PROCEDURE:
1. entry = &idt[num]
2. entry->offset_low = handler & 0xFFFF
3. entry->offset_high = (handler >> 16) & 0xFFFF
4. entry->selector = selector
5. entry->zero = 0
6. entry->type_attr = type
INVARIANT: IDT entry is valid 8-byte gate descriptor
POSTCONDITION: Interrupt 'num' will jump to 'handler'
```
### Algorithm: ISR Stub (No Error Code)
```nasm
; Macro for exceptions that don't push error code
ISR_NOERRCODE %1                    ; %1 = vector number
    cli                             ; Disable interrupts (redundant with int gate, but safe)
    push byte 0                     ; Push dummy error code (unifies stack frame)
    push byte %1                    ; Push vector number
    jmp isr_common_stub             ; Jump to common handler
%endmacro
```
### Algorithm: ISR Stub (With Error Code)
```nasm
; Macro for exceptions that DO push error code (8, 10-14, 17)
ISR_ERRCODE %1
    cli
    ; Error code already on stack from CPU
    push byte %1                    ; Push vector number
    jmp isr_common_stub
%endmacro
```
### Algorithm: Common ISR Stub
{{DIAGRAM:tdd-diag-m2-07}}
```nasm
isr_common_stub:
    ; Save all general-purpose registers
    pusha                           ; EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI
    ; Save segment registers
    mov ax, ds
    push ax
    mov ax, es
    push ax
    mov ax, fs
    push ax
    mov ax, gs
    push ax
    ; Load kernel data segment
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    ; Push stack pointer as argument
    push esp
    ; Call C handler
    call interrupt_handler
    ; Clean up argument
    add esp, 4
    ; Restore segment registers
    pop gs
    pop fs
    pop es
    pop ds
    ; Restore general-purpose registers
    popa
    ; Remove error code and vector number
    add esp, 8
    ; Return from interrupt
    iret
```
### Algorithm: PIC Remapping

![Keyboard Scancode Flow](./diagrams/tdd-diag-m2-08.svg)

```
INPUT: None
OUTPUT: IRQs remapped to vectors 32-47
PROCEDURE:
1. Save current masks: mask1 = inb(0x21), mask2 = inb(0xA1)
2. Start initialization (ICW1):
   outb(0x20, 0x11)  ; Master: init + need ICW4
   outb(0xA0, 0x11)  ; Slave: init + need ICW4
3. Set vector offsets (ICW2):
   outb(0x21, 0x20)  ; Master: IRQ0 → vector 32
   outb(0xA1, 0x28)  ; Slave: IRQ8 → vector 40
4. Configure cascade (ICW3):
   outb(0x21, 0x04)  ; Master: slave on IRQ2
   outb(0xA1, 0x02)  ; Slave: ID = 2
5. Set mode (ICW4):
   outb(0x21, 0x01)  ; Master: 8086 mode, normal EOI
   outb(0xA1, 0x01)  ; Slave: 8086 mode, normal EOI
6. Restore masks:
   outb(0x21, mask1)
   outb(0xA1, mask2)
INVARIANT: No IRQs fire until unmasked
POSTCONDITION: IRQ0-7 → vectors 32-39, IRQ8-15 → vectors 40-47
```
### Algorithm: EOI Sending
```
INPUT: irq (0-15)
OUTPUT: EOI sent to appropriate PIC(s)
PROCEDURE:
1. IF irq >= 8:
       outb(0xA0, 0x20)  ; Send EOI to slave
2. ALWAYS:
       outb(0x20, 0x20)  ; Send EOI to master
NOTE: Slave IRQs require EOI to BOTH PICs because
      slave's INT output is connected to master's IRQ2
```
### Algorithm: Timer Frequency Programming
```
INPUT: freq (desired frequency in Hz)
OUTPUT: PIT channel 0 programmed
PROCEDURE:
1. divisor = 1193182 / freq
2. IF divisor > 65535: divisor = 65535
3. IF divisor < 1: divisor = 1
4. outb(0x43, 0x36)       ; Channel 0, lobyte/hibyte, square wave
5. outb(0x40, divisor & 0xFF)     ; Low byte
6. outb(0x40, (divisor >> 8) & 0xFF)  ; High byte
INVARIANT: Timer fires at approximately 'freq' Hz
POSTCONDITION: IRQ0 triggers at specified rate
```
### Algorithm: Keyboard Scancode Processing

![Circular Keyboard Buffer](./diagrams/tdd-diag-m2-09.svg)

```
INPUT: scancode from port 0x60
OUTPUT: ASCII character in buffer (if valid)
PROCEDURE:
1. is_release = (scancode & 0x80) != 0
2. scancode = scancode & 0x7F  ; Remove release bit
3. SWITCH scancode:
   CASE 0x2A: left_shift = !is_release; RETURN
   CASE 0x36: right_shift = !is_release; RETURN
   CASE 0x1D: left_ctrl = !is_release; RETURN
   CASE 0x38: left_alt = !is_release; RETURN
   CASE 0x3A: IF !is_release:
                 caps_lock = !caps_lock
                 update_leds()
              RETURN
   ... (other modifiers)
4. IF is_release: RETURN  ; Ignore key release
5. ; Convert to ASCII
   shift = left_shift OR right_shift
   IF shift:
       ascii = scancode_to_ascii_shift[scancode]
   ELSE:
       ascii = scancode_to_ascii[scancode]
6. ; Apply caps lock
   IF caps_lock AND ('a' <= ascii <= 'z'):
       ascii = ascii - 32  ; To uppercase
   ELSE IF caps_lock AND shift AND ('A' <= ascii <= 'Z'):
       ascii = ascii + 32  ; Shift+Caps = lowercase
7. IF ascii != 0:
       buffer_put(ascii)
INVARIANT: Modifier state correctly tracked across press/release
POSTCONDITION: ASCII character in buffer if valid key pressed
```
### Algorithm: Circular Buffer Put
```
INPUT: character c
OUTPUT: c in buffer or dropped if full
PROCEDURE:
1. next_write = (write_pos + 1) % KB_BUFFER_SIZE
2. IF next_write == read_pos:
       RETURN  ; Buffer full, drop character
3. buffer[write_pos] = c
4. write_pos = next_write
INVARIANT: read_pos == write_pos means empty
POSTCONDITION: Character stored unless buffer full
```
### Algorithm: Circular Buffer Get
```
INPUT: None
OUTPUT: character c or block
PROCEDURE (blocking):
1. WHILE read_pos == write_pos:
       hlt  ; Wait for interrupt
2. c = buffer[read_pos]
3. read_pos = (read_pos + 1) % KB_BUFFER_SIZE
4. RETURN c
PROCEDURE (non-blocking):
1. IF read_pos == write_pos:
       RETURN 0
2. c = buffer[read_pos]
3. read_pos = (read_pos + 1) % KB_BUFFER_SIZE
4. RETURN c
```
---
## Error Handling Matrix
| Error | Detection Point | Detection Method | Recovery | User-Visible |
|-------|-----------------|------------------|----------|--------------|
| IDT allocation fails | idt_init() | kmalloc returns NULL | Halt with message | "PANIC: Failed to allocate IDT" |
| Invalid vector number | idt_set_gate() | num >= 256 | Ignore (no-op) | No |
| Handler causes page fault | In ISR | Exception 14 in ISR | Double fault handler | Register dump, halt |
| EOI not sent | After handler | PIC stops delivering | System hangs | No response to input |
| PIC remap conflicts | pic_init() | Vector overlap with exceptions | None (by design) | N/A (remap prevents) |
| Keyboard buffer overflow | buffer_put() | next_write == read_pos | Drop character | Character lost |
| Unknown scancode | keyboard_handler() | No table entry | Ignore | No character |
| Timer divisor overflow | timer_init() | divisor > 65535 | Clamp to 65535 | Slower timer rate |
| Extended scancode incomplete | keyboard_handler() | 0xE0 without second byte | Wait for next | Deferred handling |
| Double fault | Exception 8 | CPU pushes error code | Print diagnostic, halt | Full register dump |
---
## Implementation Sequence with Checkpoints
### Phase 1: IDT Structure and lidt (2-3 hours)
**Files:** `kernel/idt.h`, `kernel/idt.c`
**Tasks:**
1. Define `idt_entry_t` struct (8 bytes, packed)
2. Define `idt_ptr_t` struct (6 bytes, packed)
3. Declare `idt` array (256 entries)
4. Implement `idt_set_gate()` function
5. Implement `idt_init()` that zeros IDT and sets up pointer
6. Create `idt_load()` assembly stub
**Checkpoint:**
- Build succeeds
- `sizeof(idt_entry_t) == 8`
- IDT array address accessible
### Phase 2: ISR Assembly Stubs (4-5 hours)
**Files:** `kernel/idt_stubs.asm`
**Tasks:**
1. Create `ISR_NOERRCODE` macro for vectors without error code
2. Create `ISR_ERRCODE` macro for vectors with error code
3. Generate stubs for vectors 0-31 (exceptions)
4. Generate stubs for vectors 32-47 (IRQs)
5. Generate stubs for vectors 48-255 (unused)
6. Create `interrupt_stubs` array with addresses of all 256 stubs
7. Export symbols for C access
**Checkpoint:**
- Assemble without errors
- `nm kernel/idt_stubs.o | grep isr` shows 256 symbols
- `interrupt_stubs` array contains 256 addresses
### Phase 3: Common ISR Stub (3-4 hours)
**Files:** `kernel/idt_stubs.asm` (continued)
**Tasks:**
1. Implement `isr_common_stub`:
   - pusha (save registers)
   - push segment registers
   - load kernel DS
   - call C handler
   - restore segments
   - popa
   - add esp, 8 (remove error code + vector)
   - iret
2. Implement `irq_common_stub` (same, but EOI handled in C)
**Checkpoint:**
- Disassemble: `objdump -d kernel/idt_stubs.o`
- Verify pusha/popa pairs
- Verify iret at end
### Phase 4: Exception Handlers (3-4 hours)
**Files:** `kernel/interrupts.h`, `kernel/interrupts.c`, `kernel/exception_names.c`
**Tasks:**
1. Define `registers_t` struct matching assembly stack frame
2. Create exception message string array
3. Implement `handle_exception()`:
   - Print exception name
   - Print error code (if applicable)
   - Print register dump
   - For page fault, read CR2 and print faulting address
   - Halt for fatal exceptions
4. Implement `interrupt_handler()`:
   - Look up registered handler
   - Call or use default
   - Send EOI for IRQs
**Checkpoint:**
- Trigger divide by zero: `int x = 1 / 0;` (compile with -O0)
- See exception message with register dump
### Phase 5: PIC Remapping (3-4 hours)
**Files:** `kernel/pic.h`, `kernel/pic.c`
**Tasks:**
1. Define port constants (0x20, 0x21, 0xA0, 0xA1)
2. Implement `outb`/`inb` inline functions
3. Implement `pic_init()`:
   - Save masks
   - Send ICW1-ICW4 sequence
   - Restore masks
4. Implement `pic_send_eoi()`:
   - Check if IRQ >= 8
   - Send to slave if needed
   - Always send to master
**Checkpoint:**
- Call `pic_init()` before enabling interrupts
- Verify no immediate crashes (IRQs still masked)
### Phase 6: EOI Handling Integration (2-3 hours)
**Files:** `kernel/interrupts.c` (update)
**Tasks:**
1. Update `interrupt_handler()` to call `pic_send_eoi()` for vectors 32-47
2. Ensure EOI sent AFTER custom handler returns
3. Add debug output for unhandled IRQs
**Checkpoint:**
- Enable timer IRQ
- Verify system doesn't hang (EOI sent)
- Remove debug output after verification
### Phase 7: PIT Timer at 100Hz (2-3 hours)
**Files:** `kernel/timer.h`, `kernel/timer.c`
**Tasks:**
1. Define PIT port constants
2. Declare `tick_count` as volatile uint64_t
3. Implement `timer_handler()`:
   - Increment tick_count
   - (EOI handled by dispatcher)
4. Implement `timer_init()`:
   - Register handler for vector 32
   - Calculate divisor for 100Hz
   - Program PIT mode register
   - Program divisor (low byte, high byte)
   - Unmask IRQ0
5. Implement `timer_get_ticks()`, `timer_get_seconds()`, `timer_sleep_ms()`
**Checkpoint:**
- Call `timer_init(100)` after IDT and PIC init
- Enable interrupts with `sti`
- Print tick count in loop
- Verify count increases at ~100Hz
### Phase 8: Keyboard Scancode Handler (3-4 hours)
**Files:** `kernel/keyboard.h`, `kernel/keyboard.c`
**Tasks:**
1. Define keyboard state structure
2. Declare state as static globals
3. Implement `keyboard_handler()`:
   - Read scancode from port 0x60
   - Check for release (bit 7)
   - Update modifier state
   - Convert to ASCII (next phase)
4. Implement `keyboard_init()`:
   - Register handler for vector 33
   - Initialize state
   - Unmask IRQ1
**Checkpoint:**
- Type keys
- Print raw scancodes (hex)
- Verify make/break codes seen
### Phase 9: Scancode-to-ASCII Tables (2-3 hours)
**Files:** `kernel/keyboard.c` (continued)
**Tasks:**
1. Create `scancode_to_ascii[]` array (128 entries)
2. Create `scancode_to_ascii_shift[]` array
3. Fill in US keyboard layout
4. Implement shift logic in handler
5. Implement caps lock logic (XOR with shift)
6. Handle special keys (enter, backspace, tab)
**Checkpoint:**
- Type "Hello World!"
- Verify correct characters in buffer
- Test shift+a vs caps lock+a
### Phase 10: Circular Keyboard Buffer (2-3 hours)
**Files:** `kernel/keyboard.c` (continued)
**Tasks:**
1. Define KB_BUFFER_SIZE = 256
2. Declare buffer, read_pos, write_pos
3. Implement `buffer_put()`:
   - Check for overflow
   - Store character
   - Update write_pos
4. Implement `keyboard_has_char()`
5. Implement `keyboard_getchar()` (blocking with hlt)
6. Implement `keyboard_try_getchar()` (non-blocking)
**Checkpoint:**
- Type fast, verify no characters lost
- Type slow, verify blocking works
- Test buffer wrap-around
### Phase 11: Double Fault Handler (2-3 hours)
**Files:** `kernel/interrupts.c` (update)
**Tasks:**
1. Add special case for vector 8 in `handle_exception()`
2. Print "DOUBLE FAULT" banner
3. Print error code interpretation
4. Halt in tight loop with cli/hlt
**Checkpoint:**
- Intentionally cause double fault:
  - Load invalid IDT entry
  - Trigger exception
- Verify message before reset
### Phase 12: Integration Testing (3-4 hours)
**Files:** `kernel/main.c` (update)
**Tasks:**
1. Call all init functions in order:
   - idt_init()
   - pic_init()
   - register_interrupt_handler(14, page_fault_handler)
   - sti
   - timer_init(100)
   - keyboard_init()
2. Implement test loop:
   - Print tick count
   - Echo keyboard input
   - Exit on ESC
3. Test all exception handlers
4. Test interrupt latency (measure ticks)
**Checkpoint:**
- Full boot to prompt
- Keyboard input works
- Timer increments
- ESC exits cleanly
---
## Test Specification
### Test: IDT Load
**Function:** IDT structure validity
**Happy Path:**
```c
idt_init();
// Verify IDT pointer loaded
uint32_t idtr;
__asm__ volatile ("sidt %0" : "=m"(idtr));
// idtr should contain valid address
```
**Edge Case:** Call idt_set_gate with num=255 → should work
**Failure Case:** Call idt_set_gate with num=256 → should be no-op
### Test: Exception Handler - Divide Error
**Function:** Exception dispatch
**Test:**
```c
volatile int x = 1;
volatile int y = 0;
volatile int z = x / y;  // Should trigger exception 0
```
**Expected:** "EXCEPTION: Division By Zero" message, register dump, halt
### Test: Exception Handler - Page Fault
**Function:** CR2 reading
**Test:**
```c
volatile int *ptr = (volatile int *)0xDEADBEEF;
*ptr = 42;  // Page fault
```
**Expected:**
```
EXCEPTION: Page Fault
Faulting address: 0xdeadbeef
Error code: 0x2 (write, not present)
```
### Test: PIC Remap
**Function:** IRQ routing
**Test:**
1. Initialize PIC
2. Enable interrupts
3. Unmask IRQ0 (timer)
4. Verify no immediate crash (would indicate vector conflict)
**Expected:** System continues running
### Test: EOI Sending
**Function:** PIC state machine
**Test:**
1. Enable timer
2. Wait 1 second
3. Print tick count
**Expected:** Tick count ≈ 100 (for 100Hz timer)
**Failure Mode:** If EOI not sent, tick count stays at 1
### Test: Timer Frequency
**Function:** PIT programming
**Test:**
```c
timer_init(100);
uint64_t start = timer_get_ticks();
// busy wait ~1 second
for (volatile int i = 0; i < 100000000; i++);
uint64_t end = timer_get_ticks();
kprintf("Ticks: %d\n", (uint32_t)(end - start));
```
**Expected:** ~90-110 ticks (accounting for busy loop variance)
### Test: Keyboard Buffer
**Function:** Circular buffer correctness
**Test:**
```c
// Type "hello" quickly
for (int i = 0; i < 5; i++) {
    char c = keyboard_getchar();
    kprintf("%c", c);
}
```
**Expected:** "hello" printed back
### Test: Modifier Keys
**Function:** Shift and caps lock
**Test:**
1. Press and hold Shift
2. Press 'a'
3. Release Shift
4. Press Caps Lock
5. Press 'a'
**Expected:** 'A' then 'A'
### Test: Buffer Overflow
**Function:** Drop behavior
**Test:**
1. Fill buffer to capacity (hold a key down)
2. Verify no crash
3. Verify buffer still works after draining
**Expected:** Some characters dropped, system continues
### Test: Double Fault Prevention
**Function:** Exception in exception handler
**Test:**
1. Modify exception handler to cause page fault
2. Trigger any exception
**Expected:** Double fault message, then halt (NOT triple fault/reset)
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| ISR entry (IRQ to C handler) | <100 cycles | Counter before/after in stub |
| Exception handler (print + halt) | <10ms | Wall clock |
| Timer tick overhead | <500 cycles | Compare tick count to expected |
| Keyboard latency (IRQ to buffer) | <1ms | Timestamp on scancode read |
| EOI send | <100 cycles | Port write timing |
| Full interrupt round-trip | <2μs | IRQ to iret |
---
## State Machine: Interrupt Lifecycle

![Exception Handler Register Dump](./diagrams/tdd-diag-m2-10.svg)

```
[IDLE] ──IRQ arrives──> [PUSH_STATE]
                            │
                            ▼
                      [LOOKUP_IDT]
                            │
                    ┌───────┴───────┐
                    │ INVALID       │ VALID
                    ▼               ▼
              [TRIPLE_FAULT]  [LOAD_CS:EIP]
                                    │
                                    ▼
                              [ISR_STUB]
                                    │
                                    ▼
                              [SAVE_REGS]
                                    │
                                    ▼
                              [CALL_HANDLER]
                                    │
                            ┌───────┴───────┐
                            │ EXCEPTION     │ IRQ
                            ▼               ▼
                      [PRINT_DIAG]    [DEVICE_HANDLER]
                            │               │
                            ▼               ▼
                      [HALT]          [SEND_EOI]
                                            │
                                            ▼
                                      [RESTORE_REGS]
                                            │
                                            ▼
                                        [IRET]
                                            │
                                            ▼
                                        [IDLE]
```
**Illegal Transitions:**
- IRQ without IDT loaded → Triple fault
- ISR without saving all registers → Corruption
- IRQ handler without EOI → PIC blocks all future IRQs
- iret with wrong stack alignment → General protection fault
---
## Concurrency Model
{{DIAGRAM:tdd-diag-m2-11}}
**Single-threaded interrupt handling.** All ISRs run with interrupts disabled (interrupt gate type 0x8E clears IF on entry).
**No reentrancy concerns** because:
- Interrupt gates disable interrupts during handler execution
- Each CPU core handles one interrupt at a time
- No locks needed within handlers
**Shared State:**
| Variable | Access | Protection |
|----------|--------|------------|
| tick_count | Timer handler (write), other code (read) | volatile, atomic increment |
| keyboard buffer | Keyboard handler (write), getchar (read) | volatile indices, single producer/consumer |
| interrupt_handlers[] | register_handler (boot), dispatcher (runtime) | Only modified during init |
**No deadlock possible** - only one execution context per CPU.
---
## Crash Recovery
**No persistent state** in this module. All data is in volatile memory.
**Failure Modes:**
| Failure | Symptom | Debug Method |
|---------|---------|--------------|
| Triple fault | Immediate reset | QEMU `-d int -no-reboot` |
| No interrupts | System frozen | Check IF flag, IDT loaded |
| No keyboard | No input | Check IRQ1 masked, handler registered |
| Timer too fast/slow | Wrong tick rate | Verify divisor calculation |
**Debug Aids:**
- Serial output for early crash logging
- QEMU `-d int` logs all interrupts to file
- QEMU monitor: `info interrupts` shows IDT state
- GDB: Breakpoint on `interrupt_handler`
---
## Hardware Soul: Timing Analysis
### Interrupt Entry Path

![CPU Exception Vector Map](./diagrams/tdd-diag-m2-12.svg)

```
Event                          | Cycles (approx) | Notes
-------------------------------|-----------------|------------------
IRQ asserted by device         | —               | Hardware
PIC priority resolution        | ~50-100 ns      | Hardware
PIC sends INT to CPU           | —               | Hardware
CPU completes instruction      | 1-100+          | Depends on instruction
CPU pushes EFLAGS, CS, EIP     | ~10-20          | 3 memory writes
CPU looks up IDT entry         | ~20-50          | Cache miss likely
CPU validates privilege        | ~5-10           | Hardware check
CPU loads new CS:EIP           | ~10-20          | Memory read
Jump to ISR stub               | ~5              | Pipeline flush
─────────────────────────────────────────────────────────────────
Subtotal (CPU entry)           | ~60-200 cycles  |
```
### ISR Stub Execution
```
Instruction                    | Cycles | Notes
-------------------------------|--------|------------------
cli                            | 5-10   | Already disabled by gate
push byte 0                    | 1-2    | Dummy error code
push byte N                    | 1-2    | Vector number
pusha (8 registers)            | 8-16   | 8 memory writes
push DS, ES, FS, GS            | 4-8    | 4 memory writes
mov ax, 0x10 (×4)              | 4      | Register loads
mov ds, ax (×4)                | 12-20  | Segment loads are slow
push esp                       | 1-2    |
call handler                   | 2-5    | Function call overhead
─────────────────────────────────────────────────────────────────
Subtotal (stub entry)          | ~40-70 cycles
```
### C Handler Execution
```
Operation                      | Cycles | Notes
-------------------------------|--------|------------------
Function prologue              | 5-10   |
Array lookup (handler)         | 5-10   | Cache hit likely
Indirect call                  | 5-15   |
Handler execution              | 100+   | Device-dependent
Return                         | 5-10   |
─────────────────────────────────────────────────────────────────
Subtotal (C handler)           | ~120+ cycles
```
### ISR Exit Path
```
Instruction                    | Cycles | Notes
-------------------------------|--------|------------------
add esp, 4                     | 1-2    |
pop GS, FS, ES, DS             | 12-20  | Segment loads
popa (8 registers)             | 8-16   | 8 memory reads
add esp, 8                     | 1-2    | Remove error + vector
iret                           | 20-40  | Pop EIP, CS, EFLAGS
─────────────────────────────────────────────────────────────────
Subtotal (stub exit)           | ~45-80 cycles
```
### Total Interrupt Latency
```
Phase                | Cycles     | Time @ 1GHz
─────────────────────────────────────────────────
CPU entry            | 60-200     | 60-200 ns
Stub entry           | 40-70      | 40-70 ns
C handler            | 120+       | 120+ ns
Stub exit            | 45-80      | 45-80 ns
─────────────────────────────────────────────────
TOTAL                | 265-470+   | 265-470+ ns
```
**Key Insight:** Interrupt latency is dominated by C handler execution, not hardware. A slow handler (e.g., disk I/O) can take milliseconds.
---
## Build System Integration
Add to existing Makefile:
```makefile
# Add to C_SOURCES
C_SOURCES += kernel/idt.c kernel/interrupts.c kernel/pic.c \
             kernel/timer.c kernel/keyboard.c
# Add to ASM_SOURCES
ASM_SOURCES += kernel/idt_stubs.asm
# Add dependency for main.c
kernel/main.o: kernel/main.c kernel/idt.h kernel/interrupts.h \
               kernel/pic.h kernel/timer.h kernel/keyboard.h
```
---
## Synced Criteria
[[CRITERIA_JSON: {"module_id": "build-os-m2", "criteria": ["IDT structure defined as 8-byte packed entry with offset_low (16-bit), selector (16-bit), zero (8-bit), type_attr (8-bit), offset_high (16-bit)", "IDT pointer structure is 6 bytes packed: 16-bit limit (2047 for 256 entries) and 32-bit base address", "idt_init() allocates 256-entry IDT array (2048 bytes) and loads IDTR via lidt instruction", "idt_set_gate() sets all 5 fields of IDT entry: offset split into low/high, selector, zero=0, type_attr", "ISR stub macros defined: ISR_NOERRCODE for exceptions without error code, ISR_ERRCODE for exceptions 8,10-14,17", "ISR_NOERRCODE macro pushes dummy error code (byte 0) then vector number before jumping to common stub", "ISR_ERRCODE macro pushes only vector number (CPU already pushed error code) before jumping to common stub", "Common ISR stub executes pusha to save EAX,ECX,EDX,EBX,ESP,EBP,ESI,EDI in that order", "Common ISR stub saves segment registers DS,ES,FS,GS by pushing each after pusha", "Common ISR stub loads kernel data segment selector 0x10 into DS,ES,FS,GS before calling C handler", "Common ISR stub passes ESP as pointer argument to interrupt_handler C function", "Common ISR stub restores segment registers (pop GS,FS,ES,DS), executes popa, adds 8 to ESP, then iret", "interrupt_stubs array contains 256 function pointers to isr0 through isr255 for C registration", "registers_t structure matches assembly stack frame with gs,fs,es,ds at offsets 0x00-0x0C, general regs at 0x10-0x2C, int_no at 0x30, err_code at 0x31, eip/cs/eflags at 0x38-0x40", "interrupt_handler() looks up registered handler in interrupt_handlers array by regs->int_no", "If no handler registered and vector < 32, handle_exception() is called with register dump", "handle_exception() prints exception name from string array, error code, EIP, CS, EFLAGS, and all general registers", "Page fault handler (vector 14) reads CR2 register and prints faulting address plus error code bit decode", "Double fault handler (vector 8) prints diagnostic message and halts with cli/hlt loop to prevent triple fault", "PIC initialized with ICW1=0x11 to both master (0x20) and slave (0xA0) command ports", "PIC ICW2 sets master vector offset to 0x20 (IRQ0→32) and slave offset to 0x28 (IRQ8→40)", "PIC ICW3 sets master cascade to 0x04 (slave on IRQ2) and slave ID to 0x02", "PIC ICW4 sets both PICs to 0x01 for 8086 mode with normal EOI", "pic_send_eoi(irq) sends 0x20 to slave port 0xA0 if irq >= 8, then always sends 0x20 to master port 0x20", "pic_mask_irq(irq) reads current mask, sets bit for irq, writes back to appropriate data port (0x21 or 0xA1)", "pic_unmask_irq(irq) reads current mask, clears bit for irq, writes back to appropriate data port", "PIT channel 0 programmed with mode byte 0x36 (channel 0, lobyte/hibyte, square wave) to port 0x43", "PIT divisor calculated as 1193182 / freq, clamped to 1-65535 range", "PIT divisor written as low byte then high byte to port 0x40", "Timer handler increments volatile uint64_t tick_count and returns (EOI handled by dispatcher)", "timer_init(freq) registers timer handler for vector 32, programs PIT, and calls pic_unmask_irq(0)", "timer_get_ticks() returns current tick_count value", "timer_sleep_ms(ms) calculates target tick count and loops with hlt until reached", "Keyboard handler reads scancode from I/O port 0x60 on each IRQ1 (vector 33)", "Keyboard handler checks bit 7 of scancode for release (1=release, 0=press)", "Keyboard handler tracks modifier state: left_shift (0x2A), right_shift (0x36), left_ctrl (0x1D), left_alt (0x38)", "Keyboard handler toggles caps_lock (0x3A), num_lock (0x45), scroll_lock (0x46) on press only", "Keyboard handler uses scancode_to_ascii[] and scancode_to_ascii_shift[] lookup tables", "Keyboard handler applies caps lock logic: XOR with shift state for letter keys", "Keyboard buffer is 256-byte circular array with volatile read_pos and write_pos indices", "buffer_put() calculates next_write, checks for overflow (next == read), stores char, updates write_pos", "keyboard_has_char() returns true if read_pos != write_pos", "keyboard_getchar() loops with hlt while buffer empty, then reads char and advances read_pos", "keyboard_try_getchar() returns 0 immediately if buffer empty, otherwise reads char", "keyboard_init() registers keyboard handler for vector 33 and calls pic_unmask_irq(1)", "EOI for IRQs (vectors 32-47) sent by interrupt_handler() after custom handler returns", "All interrupt handlers use gate type 0x8E (interrupt gate, DPL=0, 32-bit) except syscall gate", "IDT loaded and PIC remapped before sti instruction enables interrupts", "System boots in QEMU and responds to keyboard input with correct character echo", "Timer tick counter visible and incrementing at approximately configured frequency"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-os-m3 -->
# Technical Design Document: Physical and Virtual Memory Management
**Module ID:** `build-os-m3`
---
## Module Charter
The memory management module provides the foundation for all kernel memory operations, transforming a flat physical address space into a virtualized, protected memory system. It implements a bitmap-based physical frame allocator that tracks 4KB frames across available RAM, a two-level x86 page table manager supporting identity mapping and higher-half kernel mapping, a kernel heap allocator with linked-list block management and corruption detection, and a page fault handler that reads CR2 to diagnose memory access violations. This module bridges the gap between hardware address translation (MMU, TLB) and software memory abstractions (kmalloc, page tables).
**This module does NOT:** implement per-process address spaces (Milestone 4), demand paging from disk, copy-on-write semantics, swap/page-out to disk, or user-space memory allocation (brk/sbrk/mmap syscalls). It provides the primitives that these features will build upon.
**Upstream dependencies:** GDT and protected mode from Milestone 1 (segment selectors); IDT from Milestone 2 (exception 14 for page faults); kernel heap for metadata allocation.
**Downstream consumers:** Process scheduler (per-process page directories); system call handler (user memory access); filesystem (buffer cache); all kernel subsystems via kmalloc/kfree.
**Invariants:**
- PMM bitmap accurately reflects physical frame state (1 = used, 0 = free)
- Every allocated frame has exactly one owner (no double-allocate, no leaks)
- Page directory is always 4KB-aligned
- TLB is flushed after any page table modification
- Identity mapping covers all code executing during paging enable
- Kernel heap block headers have valid magic bytes (0xA1 allocated, 0xF1 free)
- Page fault handler runs with interrupts disabled (exception gate)
---
## File Structure
```
project/
├── 02_kernel/
│   ├── 30_memory.h           # Memory management interface
│   ├── 31_pmm.h              # Physical memory manager interface
│   ├── 32_pmm.c              # PMM bitmap allocator implementation
│   ├── 33_vmm.h              # Virtual memory manager interface
│   ├── 34_vmm.c              # Page table management implementation
│   ├── 35_kheap.h            # Kernel heap interface
│   ├── 36_kheap.c            # Linked-list heap allocator implementation
│   ├── 37_pagefault.h        # Page fault handler interface
│   ├── 38_pagefault.c        # Page fault diagnostics implementation
│   └── 39_linker.ld          # Updated linker script with _kernel_end
└── 03_include/
    └── 04_memory_layout.h    # Memory region constants
```
---
## Complete Data Model
### Physical Memory Map

![Virtual Address Decomposition](./diagrams/tdd-diag-m3-01.svg)

| Address Range | Size | Purpose | PMM State |
|---------------|------|---------|-----------|
| 0x00000000 - 0x00000FFF | 4KB | Real Mode IVT + BDA | USED |
| 0x00001000 - 0x0007EFFF | ~506KB | Free conventional memory | FREE |
| 0x0007F000 - 0x0007FFFF | 4KB | Bootloader (0x7C00) | USED |
| 0x00080000 - 0x0009FFFF | 128KB | Free conventional memory | FREE |
| 0x000A0000 - 0x000BFFFF | 8KB | VGA video memory | USED |
| 0x000C0000 - 0x000FFFFF | 256KB | BIOS ROM | USED |
| 0x00100000 - 0x001XXXXX | Variable | Kernel binary | USED |
| 0x001XXXXX - 0x001YYYYY | ~128KB | PMM bitmap (for 4GB) | USED |
| 0x001YYYYY+ | — | Free extended memory | FREE |
### PMM Bitmap Structure
```c
// Bitmap for tracking 4KB frames
// One bit per frame: 1 = used, 0 = free
// For 4GB address space: 4GB / 4KB = 1,048,576 frames
// Bitmap size: 1,048,576 bits / 32 = 32,768 uint32_t = 131,072 bytes
typedef struct {
    uint32_t *bitmap;        // Pointer to bitmap array
    uint32_t  total_frames;  // Total number of frames
    uint32_t  free_frames;   // Number of free frames
    uint32_t  bitmap_size;   // Number of uint32_t elements
} pmm_state_t;
// Memory layout:
// bitmap[0] bits 0-31  → frames 0-31   (addresses 0x00000000-0x0003FFFF)
// bitmap[0] bit 0     → frame 0        (address 0x00000000-0x00000FFF)
// bitmap[1] bits 0-31 → frames 32-63   (addresses 0x00020000-0x0003FFFF)
// ...
// bitmap[n] bit b     → frame (n*32 + b)
```
**Bitmap Bit Manipulation:**
```c
#define INDEX_FROM_BIT(bit)  ((bit) / 32)
#define OFFSET_FROM_BIT(bit) ((bit) % 32)
#define SET_FRAME(bit)       (bitmap[INDEX_FROM_BIT(bit)] |=  (1 << OFFSET_FROM_BIT(bit)))
#define CLEAR_FRAME(bit)     (bitmap[INDEX_FROM_BIT(bit)] &= ~(1 << OFFSET_FROM_BIT(bit)))
#define TEST_FRAME(bit)      (bitmap[INDEX_FROM_BIT(bit)] &   (1 << OFFSET_FROM_BIT(bit)))
```
### Page Directory Entry (PDE) - 4 bytes

![Two-Level Page Table Walk](./diagrams/tdd-diag-m3-02.svg)

```
Bit  31-12: Page Table Base Address (20 bits, 4KB aligned)
Bit  11-9:  Available for OS use (3 bits)
Bit  8:     Global (G) - ignored for PDEs unless CR4.PGE=1
Bit  7:     Page Size (PS) - 0 = 4KB pages, 1 = 4MB pages
Bit  6:     Dirty (D) - 0 for PDEs
Bit  5:     Accessed (A) - set by CPU on read access
Bit  4:     Cache Disable (PCD)
Bit  3:     Write-Through (PWT)
Bit  2:     User/Supervisor (U/S) - 0 = kernel only, 1 = user accessible
Bit  1:     Read/Write (R/W) - 0 = read only, 1 = read/write
Bit  0:     Present (P) - 1 = page table exists
Example PDE pointing to page table at 0x00150000 with kernel access, writable:
  Address bits: 0x00150 → 0x00150000 (4KB aligned)
  Flags: P=1, R/W=1, U/S=0, PWT=0, PCD=0, A=0, PS=0, G=0
  Binary: 0000 0000 0001 0101 0000 0000 0000 0011 = 0x00150003
```
### Page Table Entry (PTE) - 4 bytes

![Page Directory Entry Bit Fields](./diagrams/tdd-diag-m3-03.svg)

```
Bit  31-12: Physical Frame Base Address (20 bits, 4KB aligned)
Bit  11-9:  Available for OS use (3 bits)
Bit  8:     Global (G) - not flushed on CR3 reload if CR4.PGE=1
Bit  7:     Page Attribute Table (PAT) - memory type
Bit  6:     Dirty (D) - set by CPU on write access
Bit  5:     Accessed (A) - set by CPU on any access
Bit  4:     Cache Disable (PCD)
Bit  3:     Write-Through (PWT)
Bit  2:     User/Supervisor (U/S)
Bit  1:     Read/Write (R/W)
Bit  0:     Present (P)
Example PTE mapping to frame 0x00200000, writable, user accessible:
  Address bits: 0x00200 → 0x00200000 (4KB aligned)
  Flags: P=1, R/W=1, U/S=1, PWT=0, PCD=0, A=0, D=0, PAT=0, G=0
  Binary: 0000 0000 0010 0000 0000 0000 0000 0111 = 0x00200007
```
### Page Flag Constants
```c
#define PAGE_PRESENT    (1 << 0)   // 0x001
#define PAGE_WRITABLE   (1 << 1)   // 0x002
#define PAGE_USER       (1 << 2)   // 0x004
#define PAGE_PWT        (1 << 3)   // 0x008 - Write-through caching
#define PAGE_PCD        (1 << 4)   // 0x010 - Cache disable
#define PAGE_ACCESSED   (1 << 5)   // 0x020
#define PAGE_DIRTY      (1 << 6)   // 0x040
#define PAGE_LARGE      (1 << 7)   // 0x080 - 4MB page (PDE only)
#define PAGE_GLOBAL     (1 << 8)   // 0x100
```
### Virtual Address Decomposition

![Page Table Entry Bit Fields](./diagrams/tdd-diag-m3-04.svg)

```
32-bit virtual address:
┌──────────────┬──────────────┬──────────────┐
│ PD Index     │ PT Index     │ Page Offset  │
│ (bits 31-22) │ (bits 21-12) │ (bits 11-0)  │
│ 10 bits      │ 10 bits      │ 12 bits      │
└──────────────┴──────────────┴──────────────┘
Example: Virtual address 0xC0101234
  PD Index:  (0xC0101234 >> 22) & 0x3FF = 0x300 (768)
  PT Index:  (0xC0101234 >> 12) & 0x3FF = 0x101 (257)
  Offset:     0xC0101234 & 0xFFF        = 0x234 (564)
Page directory has 1024 entries (4 bytes each = 4096 bytes)
Each page table has 1024 entries (4 bytes each = 4096 bytes)
Each entry covers 4KB of virtual address space
One PD covers 1024 × 1024 × 4KB = 4GB virtual address space
```
### Page Directory Pointer (for assembly access)
```c
// Current page directory physical address
// Updated when switching address spaces
static uint32_t *current_page_directory = NULL;
// For recursive mapping (optional advanced technique):
// Map PD at 0xFFC00000 to access page tables virtually
// PD entry 1023 points to PD itself
// Then 0xFFC00000 + (pd_index * 4096) = virtual address of page table
```
### Kernel Heap Block Header
{{DIAGRAM:tdd-diag-m3-05}}
```c
typedef struct block_header {
    uint32_t size;                  // Size of data area (excluding header)
    uint8_t  free;                  // 1 = free, 0 = allocated
    uint8_t  magic;                 // 0xA1 = allocated, 0xF1 = free
    uint8_t  _pad[2];               // Padding for alignment
    struct block_header *next;      // Next block in free list
} block_header_t;                   // Total: 12 bytes
// Memory layout of heap block:
// ┌────────────────────────────────────────┐
// │ block_header_t (12 bytes)              │
// ├────────────────────────────────────────┤
// │ Data area (size bytes, 8-byte aligned) │
// └────────────────────────────────────────┘
//
// User receives pointer to data area:
//   void *ptr = kmalloc(100);
//   ptr points to (block_header + 12)
//
// To get header from pointer:
//   block_header_t *hdr = (block_header_t *)((uint8_t *)ptr - 12);
```
### Heap State
```c
typedef struct {
    uint8_t  *heap_start;      // Virtual address of heap start
    uint8_t  *heap_end;        // Current end of heap
    uint8_t  *heap_max;        // Maximum heap address
    block_header_t *free_list; // Head of free block list
} kheap_state_t;
// Heap memory layout:
// 0xC0400000 ──────────────────── heap_start
//             │                  │
//             │ Initial pages    │ 1MB initially mapped
//             │ (256 pages)      │
//             │                  │
// 0xC0500000 ─┼──────────────────┤ heap_end (grows)
//             │                  │
//             │ Expansion area   │ Up to 16MB total
//             │ (on demand)      │
//             │                  │
// 0xC1400000 ──────────────────── heap_max
```
### Page Fault Error Code
```c
// Error code pushed by CPU on exception 14
#define PF_PRESENT  (1 << 0)  // 0 = page not present, 1 = protection violation
#define PF_WRITE    (1 << 1)  // 0 = read, 1 = write
#define PF_USER     (1 << 2)  // 0 = kernel mode, 1 = user mode
#define PF_RESERVED (1 << 3)  // Reserved bits set in page table
#define PF_FETCH    (1 << 4)  // Instruction fetch (NX bit violation)
```
### TLB (Translation Lookaside Buffer) Behavior
{{DIAGRAM:tdd-diag-m3-06}}
```
TLB Characteristics (32-bit x86):
- L1 TLB: ~64 entries (32 data + 32 instruction) on older CPUs
- L2 TLB: ~512-1536 entries on modern CPUs
- Each entry maps one 4KB page
- CR3 reload flushes all non-global entries
- invlpg flushes single entry
- Global pages (PAGE_GLOBAL) survive CR3 reload
TLB Miss Penalty:
- TLB hit: ~1-2 cycles (translation cached)
- TLB miss: ~30-50 cycles (page table walk)
  - Read PDE from memory (may be cache miss)
  - Read PTE from memory (may be cache miss)
  - Load translation into TLB
- Page fault: ~1000+ cycles (trap to kernel, handle, return)
```
---
## Interface Contracts
### PMM Initialization
```c
/**
 * Initialize the physical memory manager.
 * 
 * @param total_memory_kb  Total physical memory in kilobytes
 * 
 * Calculates total frames, allocates bitmap after kernel,
 * zeros bitmap, marks first 1MB as used, marks bitmap as used.
 * 
 * Preconditions: Kernel loaded, BSS zeroed
 * Postconditions: pmm_alloc_frame() returns valid frames
 * Side effects: Allocates bitmap at PAGE_ALIGN(&_kernel_end)
 */
void pmm_init(uint32_t total_memory_kb);
```
### PMM Frame Operations
```c
/**
 * Allocate a single physical frame.
 * 
 * @return Physical address of frame, or 0 if out of memory
 * 
 * Linear search through bitmap for first free frame.
 * Sets bit, decrements free_frames, returns address.
 * 
 * Time complexity: O(n) where n = total_frames
 * Space complexity: O(1)
 * 
 * Invariant: Returned frame is 4KB aligned
 */
void *pmm_alloc_frame(void);
/**
 * Free a physical frame.
 * 
 * @param frame  Physical address of frame to free
 * 
 * Validates frame number, checks for double-free,
 * clears bit, increments free_frames.
 * 
 * Preconditions: frame was returned by pmm_alloc_frame()
 * Postconditions: Frame available for reallocation
 * 
 * Error: Prints warning on double-free, does not crash
 */
void pmm_free_frame(void *frame);
/**
 * Mark a physical region as used.
 * 
 * @param physical_addr  Start address of region
 * @param size           Size of region in bytes
 * 
 * Used to mark kernel, bitmap, and other reserved regions.
 * Iterates through frames, sets bits, decrements free_frames.
 * 
 * Preconditions: pmm_init() called
 * Postconditions: pmm_alloc_frame() will not return these frames
 */
void pmm_mark_used(uint32_t physical_addr, uint32_t size);
/**
 * Get number of free frames.
 * 
 * @return Count of frames available for allocation
 */
uint32_t pmm_get_free_count(void);
/**
 * Get total number of frames.
 * 
 * @return Total frames in system (total_memory / 4KB)
 */
uint32_t pmm_get_total_count(void);
```
### VMM Initialization
```c
/**
 * Initialize virtual memory management.
 * 
 * Allocates page directory, zeros it, identity-maps first 4MB,
 * loads CR3, enables paging by setting CR0.PG.
 * 
 * Preconditions: pmm_init() called, IDT loaded (for page fault)
 * Postconditions: Paging enabled, all memory accesses go through MMU
 * Side effects: CR0.PG=1, CR3 loaded, TLB flushed
 * 
 * CRITICAL: Identity map must cover currently executing code
 *           or immediate page fault will occur!
 */
void vmm_init(void);
```
### VMM Page Mapping
```c
/**
 * Map a virtual page to a physical frame.
 * 
 * @param virtual_addr   Virtual address (will be page-aligned)
 * @param physical_addr  Physical address (will be page-aligned)
 * @param flags          Combination of PAGE_* constants
 * 
 * Extracts PD/PT indices, allocates page table if needed,
 * sets PTE with physical address and flags, flushes TLB entry.
 * 
 * Preconditions: vmm_init() called
 * Postconditions: Virtual address translates to physical address
 * Side effects: May allocate page table, invalidates TLB entry
 * 
 * Example:
 *   vmm_map_page(0xC0100000, 0x00100000, PAGE_PRESENT | PAGE_WRITABLE);
 */
void vmm_map_page(uint32_t virtual_addr, uint32_t physical_addr, 
                  uint32_t flags);
/**
 * Unmap a virtual page.
 * 
 * @param virtual_addr  Virtual address to unmap
 * 
 * Clears PTE to 0, flushes TLB entry.
 * Does NOT free the physical frame (caller's responsibility).
 * 
 * Preconditions: Page is currently mapped
 * Postconditions: Accessing virtual_addr causes page fault
 */
void vmm_unmap_page(uint32_t virtual_addr);
/**
 * Translate virtual address to physical address.
 * 
 * @param virtual_addr  Virtual address to translate
 * @return Physical address, or 0 if not mapped
 * 
 * Performs two-level page table walk.
 * Returns 0 if PDE not present or PTE not present.
 */
uint32_t vmm_get_physical(uint32_t virtual_addr);
```
### TLB Management
```c
/**
 * Flush TLB entry for a single page.
 * 
 * @param virtual_addr  Virtual address whose entry to flush
 * 
 * Executes invlpg instruction.
 * Must be called after modifying any PTE.
 */
void vmm_flush_tlb(uint32_t virtual_addr);
/**
 * Flush entire TLB (all non-global entries).
 * 
 * Reloads CR3 with current value.
 * Used when switching address spaces.
 */
void vmm_flush_tlb_all(void);
```
### Higher-Half Mapping
```c
/**
 * Map higher-half kernel region.
 * 
 * Maps 0xC0000000-0xC0400000 to physical 0x00000000-0x00400000.
 * Allows accessing kernel at both low and high addresses.
 * 
 * Preconditions: vmm_init() called, paging enabled
 * Postconditions: Kernel accessible at 0xC0100000+
 */
void vmm_map_higher_half(void);
```
### Kernel Heap
```c
/**
 * Initialize the kernel heap.
 * 
 * Sets up heap at KHEAP_START (0xC0400000), allocates initial
 * pages, creates first free block spanning initial heap.
 * 
 * Preconditions: vmm_init() called, paging enabled
 * Postconditions: kmalloc() returns valid pointers
 */
void kheap_init(void);
/**
 * Allocate memory from kernel heap.
 * 
 * @param size  Number of bytes to allocate
 * @return Pointer to allocated memory, or NULL if out of memory
 * 
 * Aligns size to 8 bytes, searches free list for suitable block,
 * splits block if remaining space >= header + 8, updates magic.
 * Expands heap if no suitable block found.
 * 
 * Time complexity: O(n) where n = blocks in free list
 * 
 * Example:
 *   char *buffer = (char *)kmalloc(1024);
 *   if (buffer == NULL) { handle_error(); }
 */
void *kmalloc(uint32_t size);
/**
 * Free memory back to kernel heap.
 * 
 * @param ptr  Pointer returned by kmalloc()
 * 
 * Validates magic byte, marks block as free, adds to free list.
 * Prints warning on invalid pointer or double-free.
 * 
 * Preconditions: ptr was returned by kmalloc(), not already freed
 * Postconditions: Memory available for future kmalloc()
 * 
 * Example:
 *   kfree(buffer);
 *   buffer = NULL;  // Good practice
 */
void kfree(void *ptr);
```
### Page Fault Handler
```c
/**
 * Handle page fault exception (vector 14).
 * 
 * @param regs  Pointer to register structure from ISR
 * 
 * Reads CR2 for faulting address, decodes error code,
 * prints diagnostic information, halts for now.
 * 
 * Future: Implement demand paging, copy-on-write, etc.
 * 
 * Preconditions: Registered as handler for exception 14
 * Postconditions: Diagnostic printed, system halted
 */
void page_fault_handler(registers_t *regs);
```
---
## Algorithm Specification
### Algorithm: PMM Bitmap Allocation
{{DIAGRAM:tdd-diag-m3-07}}
```
INPUT: None
OUTPUT: Physical address of free frame, or 0 if exhausted
INVARIANT: Bitmap accurately tracks frame state
PROCEDURE pmm_alloc_frame:
1. FOR i = 0 TO total_frames - 1:
       IF TEST_FRAME(i) == 0:           // Frame is free
           SET_FRAME(i)                  // Mark as used
           free_frames--
           RETURN (void *)(i * PAGE_SIZE)
2. RETURN NULL                          // Out of memory
OPTIMIZATION (future): Track last_allocated_frame index
                       Start search from there instead of 0
                       Reduces average search time for fragmented memory
```
### Algorithm: PMM Frame Free
```
INPUT: frame (physical address)
OUTPUT: None
INVARIANT: No double-free, frame within bounds
PROCEDURE pmm_free_frame(frame):
1. frame_num = (uint32_t)frame / PAGE_SIZE
2. IF frame_num >= total_frames:
       PRINT "WARNING: Invalid frame number"
       RETURN
3. IF TEST_FRAME(frame_num) == 0:
       PRINT "WARNING: Double free detected"
       RETURN
4. CLEAR_FRAME(frame_num)
5. free_frames++
POSTCONDITION: Frame available for reallocation
```
### Algorithm: VMM Map Page

![Identity Mapping Before Paging](./diagrams/tdd-diag-m3-08.svg)

```
INPUT: virtual_addr, physical_addr, flags
OUTPUT: Page mapped in page tables
INVARIANT: Page directory exists, PDE/PTE consistent
PROCEDURE vmm_map_page(virtual_addr, physical_addr, flags):
1. // Align addresses to page boundaries
   virtual_addr  = virtual_addr  & 0xFFFFF000
   physical_addr = physical_addr & 0xFFFFF000
2. // Extract indices from virtual address
   pd_index = (virtual_addr >> 22) & 0x3FF   // Bits 22-31
   pt_index = (virtual_addr >> 12) & 0x3FF   // Bits 12-21
3. // Get or create page table
   pde = &current_page_directory[pd_index]
   IF (*pde & PAGE_PRESENT) == 0:
       // Allocate new page table
       page_table = pmm_alloc_frame()
       IF page_table == NULL:
           PRINT "ERROR: Failed to allocate page table"
           RETURN
       // Zero the page table
       FOR i = 0 TO 1023:
           page_table[i] = 0
       // Set PDE to point to page table
       *pde = (uint32_t)page_table | PAGE_PRESENT | PAGE_WRITABLE | (flags & PAGE_USER)
   ELSE:
       // Get existing page table address from PDE
       page_table = (uint32_t *)(*pde & 0xFFFFF000)
4. // Set page table entry
   pte = &page_table[pt_index]
   *pte = physical_addr | flags | PAGE_PRESENT
5. // Flush TLB entry for this page
   invlpg(virtual_addr)
POSTCONDITION: MMU translates virtual_addr → physical_addr
```
### Algorithm: VMM Virtual to Physical
```
INPUT: virtual_addr
OUTPUT: Physical address, or 0 if not mapped
PROCEDURE vmm_get_physical(virtual_addr):
1. pd_index = (virtual_addr >> 22) & 0x3FF
2. pt_index = (virtual_addr >> 12) & 0x3FF
3. offset    = virtual_addr & 0xFFF
4. pde = current_page_directory[pd_index]
5. IF (pde & PAGE_PRESENT) == 0:
       RETURN 0  // Page table doesn't exist
6. page_table = (uint32_t *)(pde & 0xFFFFF000)
7. pte = page_table[pt_index]
8. IF (pte & PAGE_PRESENT) == 0:
       RETURN 0  // Page not mapped
9. frame = pte & 0xFFFFF000
10. RETURN frame | offset
```
### Algorithm: Identity Map Range
```
INPUT: start_addr, end_addr, flags
OUTPUT: Range identity-mapped (virtual = physical)
PROCEDURE identity_map_range(start_addr, end_addr, flags):
1. start = start_addr & 0xFFFFF000         // Page-align down
2. end   = (end_addr + 0xFFF) & 0xFFFFF000 // Page-align up
3. FOR addr = start TO end STEP PAGE_SIZE:
       vmm_map_page(addr, addr, flags)
POSTCONDITION: Accessing addr returns same addr on bus
```
### Algorithm: Enable Paging

![Paging Enable Sequence](./diagrams/tdd-diag-m3-09.svg)

```
INPUT: None (uses current_page_directory)
OUTPUT: Paging enabled
INVARIANT: Identity map covers executing code
PROCEDURE vmm_enable_paging:
1. // Load CR3 with page directory physical address
   CR3 = (uint32_t)current_page_directory
2. // Read current CR0
   cr0 = read_cr0()
3. // Set PG bit (bit 31)
   cr0 = cr0 | (1 << 31)
4. // Write CR0 - paging is NOW enabled
   write_cr0(cr0)
CRITICAL: Code must be identity-mapped before step 4!
          After step 4, ALL memory accesses go through page tables.
          If executing code not mapped, immediate page fault.
POSTCONDITION: CR0.PG = 1, MMU active
```
### Algorithm: Kernel Heap Block Split

![Higher-Half vs Identity Mapping](./diagrams/tdd-diag-m3-10.svg)

```
INPUT: block (pointer to block_header_t), size (requested size)
OUTPUT: Block split if large enough
PROCEDURE split_block(block, size):
1. // Check if block is large enough to split
   min_split_size = size + sizeof(block_header_t) + 8
   //                    requested + header        + minimum data
2. IF block->size < min_split_size:
       RETURN  // Don't split, use whole block
3. // Create new free block after allocated portion
   new_block = (block_header_t *)((uint8_t *)block + 
                                   sizeof(block_header_t) + size)
4. new_block->size  = block->size - size - sizeof(block_header_t)
5. new_block->free  = 1
6. new_block->magic = MAGIC_FREE  // 0xF1
7. new_block->next  = free_list
8. // Add new block to free list
   free_list = new_block
9. // Shrink original block
   block->size = size
POSTCONDITION: Two blocks where one was before
               First block: size = requested, free = 0
               Second block: size = remainder, free = 1
```
### Algorithm: Kernel Heap Allocation
```
INPUT: size (bytes to allocate)
OUTPUT: Pointer to allocated memory, or NULL
INVARIANT: All blocks have valid magic bytes
PROCEDURE kmalloc(size):
1. IF size == 0:
       RETURN NULL
2. // Align to 8 bytes, minimum 8
   size = (size + 7) & ~7
   IF size < 8: size = 8
3. // Search free list for suitable block
   block = free_list
   prev  = NULL
   WHILE block != NULL:
       IF block->magic != MAGIC_FREE:
           PRINT "ERROR: Heap corruption"
           RETURN NULL
       IF block->free AND block->size >= size:
           // Found suitable block
           GOTO found
       prev  = block
       block = block->next
4. // No block found - expand heap
   expand_heap(size + sizeof(block_header_t))
   block = free_list  // New block at head
   IF block == NULL:
       PRINT "kmalloc: Out of memory"
       RETURN NULL
LABEL found:
5. // Remove from free list
   IF prev != NULL:
       prev->next = block->next
   ELSE:
       free_list = block->next
6. // Split if large enough
   IF block->size >= size + sizeof(block_header_t) + 8:
       split_block(block, size)
7. // Mark as allocated
   block->free  = 0
   block->magic = MAGIC_ALLOC  // 0xA1
8. // Return pointer to data area
   RETURN (void *)((uint8_t *)block + sizeof(block_header_t))
```
### Algorithm: Kernel Heap Free
```
INPUT: ptr (pointer from kmalloc)
OUTPUT: Memory returned to free list
INVARIANT: Magic byte validated before any modification
PROCEDURE kfree(ptr):
1. IF ptr == NULL:
       RETURN
2. // Get block header
   block = (block_header_t *)((uint8_t *)ptr - sizeof(block_header_t))
3. // Validate magic
   IF block->magic != MAGIC_ALLOC:
       PRINT "kfree: Invalid pointer or double free"
       RETURN
4. // Mark as free
   block->free  = 1
   block->magic = MAGIC_FREE  // 0xF1
5. // Add to free list head
   block->next = free_list
   free_list   = block
// TODO: Coalesce adjacent free blocks
//       Check if next/prev blocks in memory are free
//       Merge to reduce fragmentation
```
### Algorithm: Heap Expansion
```
INPUT: min_size (minimum additional bytes needed)
OUTPUT: New free block added to heap
PROCEDURE expand_heap(min_size):
1. // Calculate pages needed
   total_needed = min_size + sizeof(block_header_t)
   pages_needed = (total_needed + PAGE_SIZE - 1) / PAGE_SIZE
2. // Check limit
   IF heap_end + pages_needed * PAGE_SIZE > heap_max:
       PRINT "WARNING: Heap expansion would exceed max"
       RETURN
3. // Allocate and map pages
   FOR i = 0 TO pages_needed - 1:
       frame = pmm_alloc_frame()
       IF frame == NULL:
           PRINT "WARNING: Out of memory for heap expansion"
           RETURN
       vmm_map_page(heap_end + i * PAGE_SIZE, 
                    (uint32_t)frame, 
                    PAGE_PRESENT | PAGE_WRITABLE)
4. // Create free block for expanded region
   new_block = (block_header_t *)heap_end
   new_block->size  = pages_needed * PAGE_SIZE - sizeof(block_header_t)
   new_block->free  = 1
   new_block->magic = MAGIC_FREE
   new_block->next  = free_list
   free_list = new_block
5. heap_end += pages_needed * PAGE_SIZE
POSTCONDITION: New free block available at old heap_end
```
### Algorithm: Page Fault Handler

![TLB Flush Methods](./diagrams/tdd-diag-m3-11.svg)

```
INPUT: regs (pointer to register save area)
OUTPUT: Diagnostic printed, system halted
PROCEDURE page_fault_handler(regs):
1. // Read faulting address from CR2
   faulting_addr = read_cr2()
2. // Decode error code
   err = regs->err_code
   present = (err & PF_PRESENT) != 0
   write   = (err & PF_WRITE)   != 0
   user    = (err & PF_USER)    != 0
3. // Print diagnostic
   PRINT "========== PAGE FAULT =========="
   PRINT "Faulting address: 0x%08x", faulting_addr
   PRINT "Error code: 0x%x", err
   PRINT "  Present: %d (%s)", present, 
         present ? "protection violation" : "page not present"
   PRINT "  Operation: %s", write ? "WRITE" : "READ"
   PRINT "  Mode: %s", user ? "USER" : "KERNEL"
   PRINT "EIP: 0x%08x", regs->eip
4. // Check if address is mapped
   phys = vmm_get_physical(faulting_addr)
   IF phys != 0:
       PRINT "Address IS mapped to physical 0x%08x", phys
       PRINT "This suggests a protection violation."
   ELSE:
       PRINT "Address is NOT mapped."
5. // Halt (future: kill process, demand page, etc.)
   PRINT "Halting..."
   WHILE true:
       cli
       hlt
```
---
## Error Handling Matrix
| Error | Detection Point | Detection Method | Recovery | User-Visible |
|-------|-----------------|------------------|----------|--------------|
| PMM allocation fails | pmm_alloc_frame() | Linear search returns NULL | Return NULL to caller | "WARNING: Out of physical memory" |
| PMM double-free | pmm_free_frame() | TEST_FRAME returns 0 (already free) | Print warning, return | "WARNING: Double free of frame 0x%x" |
| PMM invalid frame | pmm_free_frame() | frame_num >= total_frames | Print warning, return | "WARNING: Attempt to free invalid frame" |
| Page table allocation fails | vmm_map_page() | pmm_alloc_frame() returns NULL | Print error, return | "ERROR: Failed to allocate page table" |
| TLB not flushed | After PTE modify | Stale translation used | Must call vmm_flush_tlb() | Memory corruption (silent) |
| Paging enabled without identity map | vmm_init() | Immediate page fault | Triple fault, reset | System restarts |
| Heap corruption detected | kmalloc()/kfree() | magic != MAGIC_ALLOC/FREE | Print error, return NULL | "ERROR: Heap corruption detected" |
| Heap expansion fails | expand_heap() | pmm_alloc_frame() returns NULL | Print warning, return | "WARNING: Out of memory for heap expansion" |
| Invalid kfree pointer | kfree() | Header before ptr invalid | Print warning, return | "kfree: Invalid pointer or double free" |
| Page fault (kernel) | Exception 14 | CPU triggers | Print diagnostic, halt | Full fault info, system halted |
| Page fault (user) | Exception 14 | CPU triggers | Future: kill process | (Future implementation) |
---
## Implementation Sequence with Checkpoints
### Phase 1: PMM Bitmap Initialization (3-4 hours)
**Files:** `kernel/memory.h`, `kernel/pmm.h`, `kernel/pmm.c`
**Tasks:**
1. Define `PAGE_SIZE`, `PAGE_ALIGN`, `PAGE_ALIGN_DOWN` macros
2. Define bitmap manipulation macros (SET_FRAME, CLEAR_FRAME, TEST_FRAME)
3. Implement `pmm_init(total_memory_kb)`:
   - Calculate total_frames = total_memory_kb / 4
   - Calculate bitmap_size = (total_frames + 31) / 32
   - Place bitmap at `PAGE_ALIGN(&_kernel_end)`
   - Zero entire bitmap
4. Mark first 256 frames (1MB) as used
5. Mark bitmap frames as used
**Checkpoint:**
```c
pmm_init(512 * 1024);  // 512MB
kprintf("Total frames: %d\n", pmm_get_total_count());  // Should print ~131072
kprintf("Free frames: %d\n", pmm_get_free_count());    // Should print ~130000
```
### Phase 2: Frame Allocate/Free (3-4 hours)
**Files:** `kernel/pmm.c` (continued)
**Tasks:**
1. Implement `pmm_alloc_frame()`:
   - Linear search through bitmap
   - Return first free frame
2. Implement `pmm_free_frame()`:
   - Validate frame number
   - Check for double-free
   - Clear bit
3. Implement `pmm_mark_used()`:
   - Iterate through frame range
   - Set bits
**Checkpoint:**
```c
void *f1 = pmm_alloc_frame();
void *f2 = pmm_alloc_frame();
kprintf("Allocated: 0x%x, 0x%x\n", f1, f2);  // Should be different
pmm_free_frame(f1);
void *f3 = pmm_alloc_frame();
kprintf("Reallocated: 0x%x\n", f3);  // Should equal f1
```
### Phase 3: Page Directory Allocation (2-3 hours)
**Files:** `kernel/vmm.h`, `kernel/vmm.c`
**Tasks:**
1. Define `PAGE_*` flag constants
2. Declare `current_page_directory` global
3. Implement `vmm_init()` skeleton:
   - Allocate page directory frame
   - Zero all 1024 entries
   - Set `current_page_directory`
**Checkpoint:**
```c
// In vmm_init, before identity mapping:
kprintf("Page directory at: 0x%x\n", current_page_directory);
// Should print a valid physical address, 4KB aligned
```
### Phase 4: vmm_map_page Implementation (4-5 hours)
**Files:** `kernel/vmm.c` (continued)
**Tasks:**
1. Implement PD/PT index extraction macros
2. Implement `vmm_map_page()`:
   - Align addresses
   - Extract indices
   - Get or allocate page table
   - Set PTE
3. Implement inline `invlpg` wrapper
4. Implement `vmm_flush_tlb()`
**Checkpoint:**
```c
// Before enabling paging:
vmm_map_page(0x00100000, 0x00100000, PAGE_PRESENT | PAGE_WRITABLE);
uint32_t phys = vmm_get_physical(0x00100000);
kprintf("Translation: 0x00100000 -> 0x%x\n", phys);  // Should print 0x00100000
```
### Phase 5: Identity Mapping (2-3 hours)
**Files:** `kernel/vmm.c` (continued)
**Tasks:**
1. Implement `identity_map_range()` helper
2. In `vmm_init()`, call `identity_map_range(0, 0x400000, PAGE_WRITABLE)`
3. Verify VGA buffer (0xB8000) is mapped
**Checkpoint:**
```c
// After identity mapping, before enabling paging:
// Try writing to VGA through identity map
volatile uint16_t *vga = (volatile uint16_t *)0xB8000;
*vga = 0x0F00 | 'X';  // Should display 'X' on screen
```
### Phase 6: Paging Enable Sequence (3-4 hours)

![Page Fault Error Code Decode](./diagrams/tdd-diag-m3-12.svg)

**Files:** `kernel/vmm.c` (continued)
**Tasks:**
1. Implement inline CR3 read/write
2. Implement inline CR0 read/write
3. In `vmm_init()`, after identity map:
   - Load CR3
   - Set CR0.PG
4. Add diagnostic output after paging enabled
**Checkpoint:**
```c
// After vmm_init():
kprintf("Paging enabled!\n");
// If you see this, paging is working
// Try accessing various addresses
volatile int *test = (volatile int *)0xB8000;
*test = 0;  // Should not fault
```
### Phase 7: Higher-Half Mapping (3-4 hours)
**Files:** `kernel/vmm.c` (continued)
**Tasks:**
1. Implement `vmm_map_higher_half()`:
   - Loop from 0 to 0x400000
   - Map 0xC0000000+offset to offset
2. Update `vmm_init()` to call this
3. Test accessing kernel at 0xC0100000
**Checkpoint:**
```c
vmm_map_higher_half();
kprintf("Higher-half mapped\n");
// Access kernel at higher-half address
// (This requires linker script changes for full support)
```
### Phase 8: Page Fault Handler (3-4 hours)
**Files:** `kernel/pagefault.h`, `kernel/pagefault.c`
**Tasks:**
1. Define `PF_*` error code constants
2. Implement inline CR2 read
3. Implement `page_fault_handler()`:
   - Read CR2
   - Decode error code
   - Print diagnostic
   - Halt
4. Register handler for vector 14
**Checkpoint:**
```c
// Intentionally cause page fault:
volatile int *bad = (volatile int *)0xDEADBEEF;
*bad = 42;  // Should trigger handler with diagnostic
```
### Phase 9: Kernel Heap Initialization (4-5 hours)
**Files:** `kernel/kheap.h`, `kernel/kheap.c`
**Tasks:**
1. Define `KHEAP_START`, `KHEAP_INITIAL`, `KHEAP_MAX`
2. Define `block_header_t` structure
3. Define magic constants `MAGIC_ALLOC`, `MAGIC_FREE`
4. Implement `kheap_init()`:
   - Set heap_start, heap_end, heap_max
   - Allocate initial pages
   - Create initial free block
   - Set free_list
**Checkpoint:**
```c
kheap_init();
kprintf("Heap at 0x%x, size: %d KB\n", KHEAP_START, KHEAP_INITIAL / 1024);
// Should print valid address and size
```
### Phase 10: kmalloc Implementation (4-5 hours)
**Files:** `kernel/kheap.c` (continued)
**Tasks:**
1. Implement alignment helper `ALIGN8()`
2. Implement `find_free_block()`:
   - Search free_list
   - Return first suitable block
3. Implement `split_block()`:
   - Check minimum size
   - Create new free block
   - Update sizes
4. Implement `expand_heap()`:
   - Calculate pages needed
   - Allocate and map pages
   - Create free block
5. Implement `kmalloc()`:
   - Align size
   - Find or expand
   - Split if needed
   - Mark allocated
   - Return pointer
**Checkpoint:**
```c
char *buf1 = (char *)kmalloc(100);
char *buf2 = (char *)kmalloc(200);
kprintf("Allocated: 0x%x, 0x%x\n", buf1, buf2);
// Should print valid, different addresses
memset(buf1, 'A', 50);
buf1[50] = '\0';
kprintf("Buffer: %s\n", buf1);  // Should print "AAAA..."
```
### Phase 11: kfree Implementation (2-3 hours)
**Files:** `kernel/kheap.c` (continued)
**Tasks:**
1. Implement `kfree()`:
   - Get header from pointer
   - Validate magic
   - Mark free
   - Add to free_list
2. Add double-free detection
**Checkpoint:**
```c
char *buf = (char *)kmalloc(100);
kfree(buf);
char *buf2 = (char *)kmalloc(50);
kprintf("Reused: 0x%x\n", buf2);  // Should be same as buf or nearby
```
### Phase 12: Integration Testing (4-5 hours)
**Files:** `kernel/main.c` (update)
**Tasks:**
1. Update `kernel_main()` to call init functions in order:
   - pmm_init()
   - vmm_init()
   - kheap_init()
2. Add comprehensive tests:
   - PMM stress test (alloc/free many frames)
   - VMM translation test
   - Heap allocation test
   - Page fault test (intentional)
3. Add `_kernel_end` symbol to linker script
**Checkpoint:**
```bash
make clean && make
make run
# Should see:
# - PMM initialized with correct frame counts
# - Paging enabled message
# - Heap initialized message
# - All tests pass
# - System ready for input
```
---
## Test Specification
### Test: PMM Frame Allocation
**Function:** Basic allocation and deallocation
**Happy Path:**
```c
void *f1 = pmm_alloc_frame();
void *f2 = pmm_alloc_frame();
ASSERT(f1 != NULL);
ASSERT(f2 != NULL);
ASSERT(f1 != f2);
uint32_t free_before = pmm_get_free_count();
pmm_free_frame(f1);
ASSERT(pmm_get_free_count() == free_before + 1);
void *f3 = pmm_alloc_frame();
ASSERT(f3 == f1);  // Should reuse freed frame
```
**Edge Case:**
```c
// Allocate all frames
void *frames[1000];
int i;
for (i = 0; i < 1000; i++) {
    frames[i] = pmm_alloc_frame();
    if (frames[i] == NULL) break;
}
// Free them all
for (int j = 0; j < i; j++) {
    pmm_free_frame(frames[j]);
}
ASSERT(pmm_get_free_count() > 0);
```
**Failure Case:**
```c
pmm_free_frame((void *)0x12345000);  // Invalid frame
// Should print warning, not crash
```
### Test: VMM Page Mapping
**Function:** Virtual to physical translation
**Happy Path:**
```c
vmm_map_page(0xD0001000, 0x00200000, PAGE_PRESENT | PAGE_WRITABLE);
uint32_t phys = vmm_get_physical(0xD0001000);
ASSERT(phys == 0x00200000);
```
**Edge Case:**
```c
// Map same virtual address twice (remap)
vmm_map_page(0xD0001000, 0x00200000, PAGE_PRESENT | PAGE_WRITABLE);
vmm_map_page(0xD0001000, 0x00300000, PAGE_PRESENT | PAGE_WRITABLE);
phys = vmm_get_physical(0xD0001000);
ASSERT(phys == 0x00300000);  // Should use second mapping
```
**Unmapped Address:**
```c
uint32_t phys = vmm_get_physical(0xDEADBEEF);
ASSERT(phys == 0);  // Not mapped
```
### Test: Identity Mapping
**Function:** Low memory accessible at same address
**Happy Path:**
```c
// After vmm_init() with identity map
volatile uint16_t *vga = (volatile uint16_t *)0xB8000;
*vga = 0x0F00 | 'T';  // Write 'T' to screen
// Should not fault, should display 'T'
```
### Test: Paging Enable
**Function:** CR0.PG enables MMU
**Happy Path:**
```c
// After vmm_init()
uint32_t cr0;
__asm__ volatile ("mov %%cr0, %0" : "=r"(cr0));
ASSERT(cr0 & (1 << 31));  // PG bit set
```
### Test: TLB Flush
**Function:** invlpg invalidates cached translation
**Test:**
```c
vmm_map_page(0xD0000000, 0x00200000, PAGE_PRESENT | PAGE_WRITABLE);
volatile int *ptr = (volatile int *)0xD0000000;
*ptr = 42;  // Access to load TLB
vmm_unmap_page(0xD0000000);
// Without flush, stale TLB entry might still work
// With flush, access should fault
// (Test by temporarily disabling page fault handler)
```
### Test: Kernel Heap Allocation
**Function:** kmalloc returns usable memory
**Happy Path:**
```c
char *buf = (char *)kmalloc(100);
ASSERT(buf != NULL);
memset(buf, 'X', 99);
buf[99] = '\0';
ASSERT(strlen(buf) == 99);
```
**Edge Cases:**
```c
void *p1 = kmalloc(0);      ASSERT(p1 == NULL);
void *p2 = kmalloc(1);      ASSERT(p2 != NULL);  // Aligned to 8
void *p3 = kmalloc(7);      ASSERT(p3 != NULL);  // Aligned to 8
void *p4 = kmalloc(8);      ASSERT(p4 != NULL);
void *p5 = kmalloc(4096);   ASSERT(p5 != NULL);  // Full page
```
**Stress Test:**
```c
void *ptrs[100];
for (int i = 0; i < 100; i++) {
    ptrs[i] = kmalloc(100 + i);
    ASSERT(ptrs[i] != NULL);
}
for (int i = 0; i < 100; i++) {
    kfree(ptrs[i]);
}
// All should succeed without crash
```
### Test: Kernel Heap Free
**Function:** kfree returns memory to pool
**Happy Path:**
```c
void *p1 = kmalloc(100);
kfree(p1);
void *p2 = kmalloc(50);
ASSERT(p2 == p1);  // Should reuse freed block
```
**Double-Free Detection:**
```c
void *p = kmalloc(100);
kfree(p);
kfree(p);  // Should print warning, not crash
```
**Invalid Pointer:**
```c
kfree((void *)0xDEADBEEF);  // Should print warning, not crash
```
### Test: Page Fault Handler
**Function:** Diagnostic output on page fault
**Test:**
```c
// Trigger intentional page fault
volatile int *bad = (volatile int *)0xDEADBEEF;
*bad = 42;
// Should see:
// "========== PAGE FAULT =========="
// "Faulting address: 0xdeadbeef"
// "Error code: 0x2"
// "Present: 0 (page not present)"
// "Operation: WRITE"
// "Mode: KERNEL"
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Frame allocation | <1ms for 1000 frames | Timestamp before/after loop |
| Frame free | <100μs for 1000 frames | Timestamp before/after loop |
| Page mapping | <10μs per page | Inline measurement |
| Identity map 4MB | <50ms | Timestamp before/after |
| kmalloc (small) | <10μs | Inline measurement |
| kmalloc (large) | <1ms | Inline measurement |
| kfree | <5μs | Inline measurement |
| Page fault handler | <1ms to print | Serial output timing |
| TLB miss penalty | 30-50 cycles | Performance counter (advanced) |
---
## State Machine: Page Table Lifecycle

![Page Fault Handler Flow](./diagrams/tdd-diag-m3-13.svg)

```
[UNMAPPED] ──vmm_map_page──> [MAPPED]
     │                           │
     │                           │
     │                      vmm_unmap_page
     │                           │
     │                           ▼
     │<─────────────────── [UNMAPPED]
     │
     │  (Page accessed)
     │                           │
     │                      PTE.A set by CPU
     │                           │
     │                           ▼
     │                      [ACCESSED]
     │                           │
     │                      (Page written)
     │                           │
     │                      PTE.D set by CPU
     │                           │
     │                           ▼
     │                      [DIRTY]
     │
     │  (CR3 reload)
     │                           │
     │                      TLB flushed
     │                           │
     └────────────────────── [TLB_COLD]
                                │
                           (Access page)
                                │
                           TLB loaded
                                │
                                ▼
                           [TLB_HOT]
```
**Illegal Transitions:**
- Enable paging without identity map → Immediate page fault
- Access unmapped address → Page fault (exception 14)
- Write to read-only page → Page fault (protection violation)
- Modify PTE without TLB flush → Stale translation used
---
## Concurrency Model
**Single-threaded kernel execution.** All memory operations are sequential with interrupts disabled during critical sections.
**No locks required** because:
- Only one CPU core active
- Interrupts disabled during page table modification
- PMM bitmap accessed atomically (single-threaded)
- Heap free list accessed atomically (single-threaded)
**Future considerations for SMP:**
- PMM bitmap: atomic bit test-and-set
- Page tables: per-CPU page table locks or RCU
- Heap: per-CPU freelists or lock-free algorithms
---
## Crash Recovery
**No persistent state** in this module. All data is in volatile memory.
**Failure Modes:**
| Failure | Symptom | Debug Method |
|---------|---------|--------------|
| Triple fault on paging enable | Immediate reset | QEMU `-d int -no-reboot` |
| Page fault loop | Repeated faults | Check CR2 in handler |
| Heap corruption | Random crashes | Check magic bytes |
| Memory leak | Free frames → 0 | Print pmm_get_free_count() |
| Double free | Frame reused incorrectly | PMM warnings |
**Debug Aids:**
- QEMU `-d mmu` logs MMU operations
- QEMU monitor: `info mem`, `info tlb`
- Serial output for early logging
- Magic bytes detect heap corruption
- Page fault handler prints full diagnostic
---
## Hardware Soul: Memory Access Path

![Kernel Heap Block Header](./diagrams/tdd-diag-m3-14.svg)

### Complete Memory Access with Paging
```
Instruction: mov eax, [0xC0101234]
1. CPU extracts virtual address: 0xC0101234
2. TLB lookup:
   - Virtual page: 0xC0101000
   - TLB entry exists? 
     - YES: Get physical frame directly → Skip to step 6
     - NO: Continue with page table walk
3. Page directory access:
   - Read CR3 → PD physical address (e.g., 0x00100000)
   - PD index: 0xC0101234 >> 22 = 0x300 (768)
   - Read PDE at 0x00100000 + 768*4 = 0x00100C00
   - PDE value: 0x00151003 (PT at 0x00151000, present, writable)
   - Cache behavior: Likely cache miss (cold path)
4. Page table access:
   - PT index: (0xC0101234 >> 12) & 0x3FF = 0x101 (257)
   - Read PTE at 0x00151000 + 257*4 = 0x00151404
   - PTE value: 0x00100007 (frame 0x00100000, present, writable, user)
   - Cache behavior: May be cached if recently accessed
5. TLB update:
   - Load translation: 0xC0101000 → 0x00100000
   - Replace oldest TLB entry if full
6. Physical access:
   - Page offset: 0xC0101234 & 0xFFF = 0x234
   - Physical address: 0x00100000 + 0x234 = 0x00100234
   - Memory controller reads from RAM
   - Data returned to CPU
Timing:
  - TLB hit:    ~1-2 cycles    (translation cached)
  - TLB miss:   ~30-50 cycles  (PD + PT memory reads)
  - Page fault: ~1000+ cycles  (trap to kernel)
```
### Cache Line Analysis
```
Page directory (4KB):
  - 1024 entries × 4 bytes = 4096 bytes
  - 64 cache lines (64 bytes each)
  - Typically only 1-2 cache lines hot (current process)
Page table (4KB):
  - 1024 entries × 4 bytes = 4096 bytes
  - 64 cache lines
  - Access pattern: sequential for stack, random for heap
PMM bitmap (128KB for 4GB):
  - 2048 cache lines
  - Linear search touches many cache lines
  - Optimization: track last_allocated hint
Heap blocks:
  - Headers: 12 bytes each
  - Typically 5-6 headers per cache line
  - Free list traversal: sequential through list
```
---
## Build System Integration
Update `Makefile`:
```makefile
# Add to C_SOURCES
C_SOURCES += kernel/pmm.c kernel/vmm.c kernel/kheap.c kernel/pagefault.c
# Add dependencies
kernel/main.o: kernel/main.c kernel/memory.h kernel/pmm.h kernel/vmm.h \
               kernel/kheap.h kernel/pagefault.h
```
Update `kernel/linker.ld`:
```ld
/* Add at end of SECTIONS */
_kernel_end = .;
```
---
## Synced Criteria
[[CRITERIA_JSON: {"module_id": "build-os-m3", "criteria": ["PMM bitmap allocated at PAGE_ALIGN(&_kernel_end) with size (total_frames + 31) / 32 uint32_t elements", "PMM bitmap macros defined: INDEX_FROM_BIT(bit) = bit/32, OFFSET_FROM_BIT(bit) = bit%32, SET_FRAME, CLEAR_FRAME, TEST_FRAME", "pmm_init(total_memory_kb) calculates total_frames = total_memory_kb / 4, allocates bitmap, zeros it, marks first 256 frames (1MB) as used", "pmm_alloc_frame() performs linear search through bitmap for first free frame (TEST_FRAME returns 0), sets bit, decrements free_frames, returns physical address", "pmm_free_frame(frame) validates frame_num < total_frames, checks TEST_FRAME for double-free (prints warning if already free), clears bit, increments free_frames", "pmm_mark_used(addr, size) iterates from PAGE_ALIGN_DOWN(addr) to PAGE_ALIGN(addr+size), sets bits with free_frames decrement", "Page directory allocated as single 4KB frame via pmm_alloc_frame(), stored in current_page_directory global, zeroed to 1024 entries", "vmm_map_page(virt, phys, flags) extracts pd_index = (virt >> 22) & 0x3FF and pt_index = (virt >> 12) & 0x3FF", "vmm_map_page allocates new page table via pmm_alloc_frame() if PDE present bit is 0, zeros it, sets PDE with PT address ORed with PAGE_PRESENT | PAGE_WRITABLE | (flags & PAGE_USER)", "vmm_map_page sets PTE at page_table[pt_index] = (phys & 0xFFFFF000) | flags | PAGE_PRESENT", "vmm_unmap_page(virt) clears PTE to 0 and calls invlpg(virt) to flush TLB entry", "vmm_get_physical(virt) performs two-level walk: reads PDE, extracts PT address, reads PTE, extracts frame address, adds offset, returns 0 if not mapped", "vmm_flush_tlb(virt) executes __asm__ volatile (\"invlpg (%0)\" : : \"r\"(virt) : \"memory\")", "vmm_flush_tlb_all() reads CR3 into variable, writes same value back to flush all non-global TLB entries", "identity_map_range(start, end, flags) iterates addr from PAGE_ALIGN_DOWN(start) to PAGE_ALIGN(end) calling vmm_map_page(addr, addr, flags)", "vmm_init() allocates page directory, identity-maps 0x00000000-0x00400000 with PAGE_WRITABLE, loads CR3, sets CR0.PG bit 31", "Paging enabled by sequence: mov pd_addr to CR3, read CR0, or with (1<<31), write CR0", "vmm_map_higher_half() maps 0xC0000000-0xC0400000 to physical 0x00000000-0x0040000 using vmm_map_page loop", "page_fault_handler reads CR2 via __asm__ volatile (\"mov %%cr2, %0\" : \"=r\"(addr))", "page_fault_handler decodes error code bits: PF_PRESENT (bit 0), PF_WRITE (bit 1), PF_USER (bit 2), PF_RESERVED (bit 3), PF_FETCH (bit 4)", "page_fault_handler prints diagnostic with faulting address from CR2, error code hex, decoded bits (present/write/user), EIP, and vmm_get_physical result", "kheap_init() sets heap_start to KHEAP_START (0xC0400000), allocates KHEAP_INITIAL pages via pmm_alloc_frame/vmm_map_page, creates initial free block", "block_header_t structure: uint32_t size, uint8_t free, uint8_t magic, uint8_t _pad[2], block_header_t* next (12 bytes total)", "kmalloc(size) aligns size to 8 bytes with minimum 8, searches free_list for block with free=1 and size >= requested", "kmalloc splits block if remaining >= size + sizeof(block_header_t) + 8 by creating new free block at offset", "kmalloc calls expand_heap() if no suitable block found, which allocates pages via pmm_alloc_frame and vmm_map_page", "kmalloc sets block->free=0, block->magic=0xA1 (MAGIC_ALLOC), returns (void*)((uint8_t*)block + 12)", "kfree(ptr) calculates header = (block_header_t*)((uint8_t*)ptr - 12), validates magic == 0xA1, prints warning on mismatch", "kfree sets block->free=1, block->magic=0xF1 (MAGIC_FREE), adds to free_list head", "expand_heap(min_size) calculates pages_needed = (min_size + header_size + 4095) / 4096, checks against heap_max, allocates and maps pages", "VGA buffer at 0xB8000 remains accessible after paging enabled via identity mapping", "Linker script defines _kernel_end symbol at end of all sections for PMM bitmap placement", "System boots in QEMU with paging enabled, passes PMM/VMM/kheap tests, handles intentional page fault with diagnostic output"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-os-m4 -->
# Technical Design Document: Processes and Preemptive Scheduling
**Module ID:** `build-os-m4`
---
## Module Charter
The process management module transforms a single-threaded kernel into a preemptive multitasking operating system capable of running multiple processes concurrently on a single CPU core. It implements Process Control Blocks (PCBs) storing complete execution state, software context switching in assembly that saves and restores all registers atomically, a round-robin scheduler invoked by timer interrupts that selects the next ready process, a Task State Segment (TSS) that defines the kernel stack for Ring 3→Ring 0 transitions, user-mode process execution with isolated page directories, and a system call interface via INT 0x80 with DPL=3 allowing user code to request kernel services. This module provides the illusion of parallelism through rapid context switching while maintaining strict isolation between processes.
**This module does NOT:** implement priority-based scheduling, real-time scheduling guarantees, SMP/multi-core support, copy-on-write fork, demand paging of process memory, inter-process communication (pipes, shared memory), or signals. It provides the primitives that these features will build upon.
**Upstream dependencies:** GDT from Milestone 1 (segment selectors); IDT from Milestone 2 (timer interrupt, exception handlers); VMM from Milestone 3 (per-process page directories); kernel heap for PCB allocation.
**Downstream consumers:** Shell (process creation); filesystem (file descriptors per process); future IPC mechanisms; user programs via system calls.
**Invariants:**
- PCB field offsets MUST match assembly context switch macros exactly
- TSS.ESP0 MUST be updated before returning to user mode for any process
- Current process pointer is valid or NULL (never dangling)
- Circular ready queue is never empty (idle process always present)
- User process page directories have kernel mapped supervisor-only
- All context switches happen with interrupts disabled
- iret stack frame is correct: SS, ESP, EFLAGS, CS, EIP (for privilege change)
---
## File Structure
```
project/
├── 02_kernel/
│   ├── 40_process.h          # PCB structure and process management interface
│   ├── 41_process.c          # Process creation, termination, list management
│   ├── 42_context_switch.asm # Assembly context switch implementation
│   ├── 43_scheduler.h        # Scheduler interface
│   ├── 44_scheduler.c        # Round-robin scheduler implementation
│   ├── 45_tss.h              # TSS structure and interface
│   ├── 46_tss.c              # TSS initialization and ESP0 update
│   ├── 47_gdt.h              # Updated GDT interface (add TSS entry)
│   ├── 48_gdt.c              # Updated GDT implementation
│   ├── 49_gdt_flush.asm      # Assembly GDT reload
│   ├── 50_usermode.h         # User mode transition interface
│   ├── 51_usermode.c         # jump_to_user_mode implementation
│   ├── 52_syscall.h          # System call interface
│   ├── 53_syscall.c          # Syscall dispatch and implementations
│   ├── 54_syscall_entry.asm  # Assembly syscall entry stub
│   └── 55_test_processes.c   # Multi-process demonstration
└── 03_include/
    └── 05_process.h          # Shared process definitions
```
---
## Complete Data Model
### Process Control Block (PCB) Structure

![Process Control Block Layout](./diagrams/tdd-diag-m4-01.svg)

```c
// kernel/process.h
#ifndef PROCESS_H
#define PROCESS_H
#include <stdint.h>
#include <stdbool.h>
/* Process states */
typedef enum {
    PROCESS_STATE_READY,       // Ready to run, waiting for CPU
    PROCESS_STATE_RUNNING,     // Currently executing
    PROCESS_STATE_BLOCKED,     // Waiting for I/O or event
    PROCESS_STATE_TERMINATED,  // Finished execution
} process_state_t;
/* Process Control Block - MUST match assembly offsets exactly */
typedef struct process {
    /* === Identification (offsets 0x00-0x27) === */
    uint32_t pid;              // 0x00: Unique process ID
    char name[32];             // 0x04: Process name (null-terminated)
    /* === Scheduling state (offsets 0x28-0x2F) === */
    process_state_t state;     // 0x24: Current state (enum = int)
    uint32_t priority;         // 0x28: Priority (future use)
    uint32_t time_slice;       // 0x2C: Remaining time slice in ticks
    /* === Saved register state (offsets 0x30-0x5F) === */
    /* CRITICAL: These offsets must match context_switch.asm macros */
    uint32_t eax;              // 0x30: Saved EAX
    uint32_t ebx;              // 0x34: Saved EBX
    uint32_t ecx;              // 0x38: Saved ECX
    uint32_t edx;              // 0x3C: Saved EDX
    uint32_t esi;              // 0x40: Saved ESI
    uint32_t edi;              // 0x44: Saved EDI
    uint32_t ebp;              // 0x48: Saved EBP
    uint32_t esp;              // 0x4C: Saved ESP (kernel stack pointer)
    uint32_t eip;              // 0x50: Saved EIP (return address)
    uint32_t eflags;           // 0x54: Saved EFLAGS
    /* === Memory management (offsets 0x58-0x5F) === */
    uint32_t page_directory;   // 0x58: Physical address of page directory (CR3)
    uint32_t kernel_stack;     // 0x5C: Top of kernel stack (for TSS.ESP0)
    uint32_t user_stack;       // 0x60: User stack pointer (for user mode)
    /* === Entry point (offset 0x64) === */
    uint32_t entry_point;      // 0x64: Process entry point address
    /* === Linked list pointers (offsets 0x68-0x6F) === */
    struct process *next;      // 0x68: Next process in list
    struct process *prev;      // 0x6C: Previous process in list
} process_t;
// Total size: 112 bytes (0x70)
/* PCB field offsets for assembly - must match struct above exactly */
#define PCB_PID           0x00
#define PCB_NAME          0x04
#define PCB_STATE         0x24
#define PCB_PRIORITY      0x28
#define PCB_TIME_SLICE    0x2C
#define PCB_EAX           0x30
#define PCB_EBX           0x34
#define PCB_ECX           0x38
#define PCB_EDX           0x3C
#define PCB_ESI           0x40
#define PCB_EDI           0x44
#define PCB_EBP           0x48
#define PCB_ESP           0x4C
#define PCB_EIP           0x50
#define PCB_EFLAGS        0x54
#define PCB_PAGE_DIR      0x58
#define PCB_KERNEL_STACK  0x5C
#define PCB_USER_STACK    0x60
#define PCB_ENTRY_POINT   0x64
#define PCB_NEXT          0x68
#define PCB_PREV          0x6C
#define PCB_SIZE          0x70
/* Stack sizes */
#define KERNEL_STACK_SIZE 0x2000  // 8KB per process
#define USER_STACK_SIZE   0x10000 // 64KB per user process
/* Process functions */
void process_init(void);
process_t *process_create(const char *name, void (*entry_point)(void), bool user_mode);
void process_terminate(process_t *proc);
process_t *process_get_current(void);
process_t *process_get_by_pid(uint32_t pid);
/* External reference to current process */
extern process_t *current_process;
#endif /* PROCESS_H */
```
### PCB Memory Layout
{{DIAGRAM:tdd-diag-m4-02}}
| Offset | Size | Field | Purpose |
|--------|------|-------|---------|
| 0x00 | 4 | pid | Unique identifier for process |
| 0x04 | 32 | name | Human-readable name for debugging |
| 0x24 | 4 | state | READY/RUNNING/BLOCKED/TERMINATED |
| 0x28 | 4 | priority | Future: scheduling priority |
| 0x2C | 4 | time_slice | Ticks remaining before yield |
| 0x30 | 4 | eax | Saved general register |
| 0x34 | 4 | ebx | Saved general register |
| 0x38 | 4 | ecx | Saved general register |
| 0x3C | 4 | edx | Saved general register |
| 0x40 | 4 | esi | Saved index register |
| 0x44 | 4 | edi | Saved index register |
| 0x48 | 4 | ebp | Saved base pointer |
| 0x4C | 4 | esp | Saved stack pointer |
| 0x50 | 4 | eip | Saved instruction pointer |
| 0x54 | 4 | eflags | Saved flags register |
| 0x58 | 4 | page_directory | CR3 value for this process |
| 0x5C | 4 | kernel_stack | Top of kernel stack |
| 0x60 | 4 | user_stack | User mode stack pointer |
| 0x64 | 4 | entry_point | Initial EIP for process |
| 0x68 | 4 | next | Next PCB in list |
| 0x6C | 4 | prev | Previous PCB in list |
### Task State Segment (TSS) Structure
{{DIAGRAM:tdd-diag-m4-03}}
```c
// kernel/tss.h
#ifndef TSS_H
#define TSS_H
#include <stdint.h>
/* Task State Segment - 104 bytes fixed size */
typedef struct {
    uint16_t link;     uint16_t _pad0;   // 0x00: Previous task link
    uint32_t esp0;                       // 0x04: Stack pointer for ring 0 ★ CRITICAL
    uint16_t ss0;     uint16_t _pad1;   // 0x08: Stack segment for ring 0 (0x10)
    uint32_t esp1;                       // 0x0C: Stack pointer for ring 1 (unused)
    uint16_t ss1;     uint16_t _pad2;   // 0x10: Stack segment for ring 1
    uint32_t esp2;                       // 0x14: Stack pointer for ring 2 (unused)
    uint16_t ss2;     uint16_t _pad3;   // 0x18: Stack segment for ring 2
    uint32_t cr3;                        // 0x1C: Page directory (not used for soft switching)
    uint32_t eip;                        // 0x20: Instruction pointer
    uint32_t eflags;                     // 0x24: Flags register
    uint32_t eax;                        // 0x28: General registers
    uint32_t ecx;                        // 0x2C
    uint32_t edx;                        // 0x30
    uint32_t ebx;                        // 0x34
    uint32_t esp;                        // 0x38
    uint32_t ebp;                        // 0x3C
    uint32_t esi;                        // 0x40
    uint32_t edi;                        // 0x44
    uint16_t es;      uint16_t _pad4;   // 0x48: Segment registers
    uint16_t cs;      uint16_t _pad5;   // 0x4C
    uint16_t ss;      uint16_t _pad6;   // 0x50
    uint16_t ds;      uint16_t _pad7;   // 0x54
    uint16_t fs;      uint16_t _pad8;   // 0x58
    uint16_t gs;      uint16_t _pad9;   // 0x5C
    uint16_t ldtr;    uint16_t _pad10;  // 0x60: LDT selector
    uint16_t _pad11;                     // 0x62: Reserved
    uint16_t iomap_base;                 // 0x64: I/O permission bitmap offset
} __attribute__((packed)) tss_t;
// Total size: 104 bytes (0x68)
/* TSS functions */
void tss_init(void);
void tss_update_esp0(uint32_t esp0);
/* Global pointer for assembly access */
extern uint32_t tss_esp0_ptr;
#endif /* TSS_H */
```
**Critical TSS Fields:**
| Field | Offset | Value | Purpose |
|-------|--------|-------|---------|
| esp0 | 0x04 | Process kernel stack top | Where CPU pushes user state on Ring 3→0 |
| ss0 | 0x08 | 0x10 | Kernel data segment selector |
| iomap_base | 0x64 | 104 | No I/O bitmap (past end of TSS) |
### GDT TSS Entry Format
{{DIAGRAM:tdd-diag-m4-04}}
```c
// TSS descriptor in GDT (system segment, not code/data)
// 8 bytes, different format from code/data descriptors
/* TSS descriptor layout:
 * Byte 0-1: Limit[15:0]
 * Byte 2-3: Base[15:0]
 * Byte 4:   Base[23:16]
 * Byte 5:   Access byte (0x89 for 32-bit available TSS)
 *           - P=1 (present)
 *           - DPL=0 (ring 0 only)
 *           - S=0 (system segment)
 *           - Type=1001 (32-bit available TSS)
 * Byte 6:   Flags[7:4] | Limit[19:16]
 *           - G=0 (byte granularity)
 *           - 0
 *           - 0
 *           - AVL=0
 *           - Limit high nibble
 * Byte 7:   Base[31:24]
 */
// Example: TSS at 0x00152000, limit 103 (104-1)
// Base = 0x00152000
// Limit = 0x67 (103)
// 
// Entry = 0x00152067 00000089 00152000 (simplified)
// Actual bytes: 67 00 00 20 15 89 40 00
```
### Interrupt Stack Frame (for Context Switch)

![Ring 3 to Ring 0 Stack Transition](./diagrams/tdd-diag-m4-05.svg)

```
Stack layout after interrupt from user mode (Ring 3 → Ring 0):
High addresses
┌──────────────────────┐
│ SS (user)            │  Pushed by CPU (only on privilege change)
│ ESP (user)           │  Pushed by CPU (user stack pointer)
├──────────────────────┤
│ EFLAGS               │  Pushed by CPU
├──────────────────────┤
│ CS                   │  Pushed by CPU
│ EIP (return addr)    │  Pushed by CPU
├──────────────────────┤
│ Error Code           │  Pushed by CPU (for exceptions 8,10-14)
├──────────────────────┤
│ ... ISR pushes more registers ...
└──────────────────────┘  Low addresses
For kernel-to-kernel interrupts, SS:ESP not pushed (no privilege change).
```
### User Mode Entry Stack Frame (for iret)

![Round-Robin Circular Queue](./diagrams/tdd-diag-m4-06.svg)

```c
// Stack must be set up for iret to enter user mode:
// Push in reverse order (stack grows down):
// High addresses (setup by jump_to_user_mode)
┌──────────────────────┐
│ SS = 0x23            │  User data segment (0x20 | RPL=3)
│ ESP = user_stack     │  User stack pointer
├──────────────────────┤
│ EFLAGS = 0x202       │  IF=1, reserved bit 1
├──────────────────────┤
│ CS = 0x1B            │  User code segment (0x18 | RPL=3)
│ EIP = entry_point    │  Process entry point
└──────────────────────┘  Low addresses (ESP points here)
// After iret, CPU pops EIP, CS, EFLAGS, ESP, SS
// and begins executing at entry_point in Ring 3
```
### System Call Stack Frame

![Scheduler Tick Decision](./diagrams/tdd-diag-m4-07.svg)

```c
// User mode executes: int $0x80
// CPU transitions Ring 3 → Ring 0
// Stack changes from user stack to TSS.ESP0
// Kernel stack (TSS.ESP0) after CPU pushes:
┌──────────────────────┐
│ SS (user) = 0x23     │
│ ESP (user)           │
├──────────────────────┤
│ EFLAGS               │
├──────────────────────┤
│ CS (user) = 0x1B     │
│ EIP (after int 0x80) │
└──────────────────────┘  ← ESP at syscall_entry
// Our stub then pushes:
┌──────────────────────┐
│ ... (CPU pushed) ... │
├──────────────────────┤
│ GS, FS, ES, DS       │  Segment registers
├──────────────────────┤
│ EDI, ESI, EBP, ESP,  │  pusha
│ EBX, EDX, ECX, EAX   │
└──────────────────────┘  ← ESP passed to syscall_handler
// Register use for syscall:
// EAX = syscall number
// EBX = arg1, ECX = arg2, EDX = arg3
// Return value in EAX
```
### Scheduler Queue Structure

![User Mode Entry via iret](./diagrams/tdd-diag-m4-08.svg)

```c
// Circular doubly-linked list of processes
typedef struct {
    process_t *ready_queue;     // Head of ready queue (circular)
    process_t *current;         // Currently running process
    uint32_t tick_count;        // Total scheduler ticks
} scheduler_state_t;
// Circular list invariants:
// - Empty queue: ready_queue == NULL
// - Single process: proc->next == proc && proc->prev == proc
// - Multiple processes: circular chain via next/prev
// - ready_queue points to "next to run"
// 
// Example with 3 processes (A, B, C):
// 
//     ready_queue ──→ [A] ←──→ [B] ←──→ [C] ←──┐
//                     ↑                         │
//                     └─────────────────────────┘
// 
// A.next = B, B.next = C, C.next = A
// A.prev = C, B.prev = A, C.prev = B
```
### System Call Table
```c
// kernel/syscall.h
#define SYS_EXIT    0
#define SYS_READ    1
#define SYS_WRITE   2
#define SYS_YIELD   3
#define SYS_GETPID  4
typedef int (*syscall_func_t)(int arg1, int arg2, int arg3);
// Dispatch table
static syscall_func_t syscall_table[] = {
    [SYS_EXIT]   = (syscall_func_t)sys_exit,
    [SYS_READ]   = (syscall_func_t)sys_read,
    [SYS_WRITE]  = (syscall_func_t)sys_write,
    [SYS_YIELD]  = (syscall_func_t)sys_yield,
    [SYS_GETPID] = (syscall_func_t)sys_getpid,
};
```
---
## Interface Contracts
### Process Management
```c
/**
 * Initialize process management subsystem.
 * 
 * Sets process_list = NULL, current_process = NULL, next_pid = 1.
 * 
 * Preconditions: GDT loaded, IDT loaded, heap initialized
 * Postconditions: process_create() can be called
 */
void process_init(void);
/**
 * Create a new process.
 * 
 * @param name         Human-readable name (copied, max 31 chars)
 * @param entry_point  Function pointer to process code
 * @param user_mode    true = Ring 3 process, false = Ring 0 kernel thread
 * @return             Pointer to PCB, or NULL on allocation failure
 * 
 * Allocates PCB, kernel stack, and (for user mode) user stack + page directory.
 * Sets up initial stack frame for context switch or user mode entry.
 * Adds process to global process list.
 * Does NOT add to scheduler ready queue (call scheduler_add() separately).
 * 
 * Preconditions: process_init() called, kmalloc available
 * Postconditions: Process in READY state, can be added to scheduler
 */
process_t *process_create(const char *name, void (*entry_point)(void), bool user_mode);
/**
 * Terminate a process.
 * 
 * @param proc  Process to terminate (must not be NULL)
 * 
 * Sets state to TERMINATED, removes from process list.
 * Does NOT free memory (for simplicity; production OS would free).
 * If terminating current_process, caller must call scheduler_yield().
 * 
 * Preconditions: proc is valid, in process list
 * Postconditions: proc->state == TERMINATED, proc not in list
 */
void process_terminate(process_t *proc);
/**
 * Get currently running process.
 * 
 * @return Current process PCB, or NULL if none running
 */
process_t *process_get_current(void);
/**
 * Find process by PID.
 * 
 * @param pid  Process ID to search for
 * @return     PCB pointer, or NULL if not found
 */
process_t *process_get_by_pid(uint32_t pid);
```
### Context Switch (Assembly)
```nasm
; kernel/context_switch.asm
; void context_switch(process_t *old_proc, process_t *new_proc)
;
; Saves old_proc state, loads new_proc state, never returns normally.
; Instead, "returns" to new_proc->eip with new_proc->esp.
;
; Preconditions:
;   - Interrupts disabled
;   - old_proc may be NULL (first switch)
;   - new_proc must not be NULL
;   - PCB offsets must match struct process exactly
;
; Postconditions:
;   - old_proc->eax/ebx/ecx/edx/esi/edi/ebp/esp/eip/eflags saved
;   - CR3 loaded with new_proc->page_directory (if non-zero)
;   - TSS.ESP0 updated to new_proc->kernel_stack
;   - Execution resumes at new_proc->eip
;
; Destroys: All registers (by design)
[GLOBAL context_switch]
context_switch:
    ; Implementation in Assembly Specification section
```
### Scheduler Interface
```c
/**
 * Initialize scheduler.
 * 
 * Sets ready_queue = NULL, tick_count = 0.
 * 
 * Preconditions: process_init() called
 * Postconditions: scheduler_add() can be called
 */
void scheduler_init(void);
/**
 * Add process to ready queue.
 * 
 * @param proc  Process to add (must be in READY state)
 * 
 * Inserts into circular queue, updates next/prev pointers.
 * 
 * Preconditions: proc != NULL, proc->state == READY
 * Postconditions: proc in ready queue, will be scheduled
 */
void scheduler_add(process_t *proc);
/**
 * Remove process from ready queue.
 * 
 * @param proc  Process to remove
 * 
 * Unlinks from circular queue, updates neighbors' pointers.
 * 
 * Preconditions: proc in ready queue
 * Postconditions: proc not in ready queue
 */
void scheduler_remove(process_t *proc);
/**
 * Timer tick handler - called from IRQ0.
 * 
 * @param regs  Register save area from interrupt
 * 
 * Increments tick_count, decrements current->time_slice.
 * Calls scheduler_yield() if time_slice reaches 0.
 * 
 * Preconditions: Timer interrupt fired, IDT active
 * Postconditions: May switch to different process
 */
void scheduler_tick(registers_t *regs);
/**
 * Yield CPU to next ready process.
 * 
 * Selects next READY process from queue, performs context switch.
 * If no other process ready, resets time_slice and returns.
 * 
 * Preconditions: At least one process in system
 * Postconditions: Different process may be running
 */
void scheduler_yield(void);
/**
 * Block a process (remove from ready queue).
 * 
 * @param proc  Process to block
 * 
 * Sets state to BLOCKED, removes from ready queue.
 * If proc is current, yields to next process.
 */
void scheduler_block(process_t *proc);
/**
 * Unblock a process (add to ready queue).
 * 
 * @param proc  Process to unblock
 * 
 * Sets state to READY, adds to ready queue.
 */
void scheduler_unblock(process_t *proc);
```
### TSS Interface
```c
/**
 * Initialize Task State Segment.
 * 
 * Zeros TSS, sets ss0 = 0x10 (kernel data segment).
 * Adds TSS descriptor to GDT at index 5 (selector 0x28).
 * Loads TR with ltr instruction.
 * 
 * Preconditions: GDT initialized
 * Postconditions: TSS active, Ring 3→0 transitions use TSS.ESP0
 */
void tss_init(void);
/**
 * Update TSS.ESP0 for current process.
 * 
 * @param esp0  New kernel stack pointer for Ring 0
 * 
 * Sets tss.esp0 = esp0. Called before returning to user mode
 * and during context switch.
 * 
 * Preconditions: TSS initialized
 * Postconditions: Next Ring 3→0 uses this stack
 */
void tss_update_esp0(uint32_t esp0);
```
### User Mode Transition
```c
/**
 * Jump from kernel mode to user mode.
 * 
 * @param entry_point  Instruction pointer for user code
 * @param stack_ptr    User stack pointer (top of user stack)
 * 
 * Sets up stack for iret, loads user data segments,
 * executes iret to enter Ring 3.
 * 
 * NEVER RETURNS - execution continues in user mode.
 * 
 * Preconditions: 
 *   - TSS.ESP0 set to valid kernel stack
 *   - User page tables map entry_point and stack
 *   - Interrupts may be enabled (will be disabled during transition)
 * 
 * Postconditions:
 *   - CS = 0x1B (user code), DS/ES/FS/GS/SS = 0x23 (user data)
 *   - CPL = 3 (Ring 3)
 *   - Execution at entry_point
 */
void jump_to_user_mode(uint32_t entry_point, uint32_t stack_ptr);
```
### System Call Interface
```c
/**
 * Initialize system call handler.
 * 
 * Sets IDT entry 0x80 with:
 *   - Handler address: syscall_asm_entry
 *   - Selector: 0x08 (kernel code)
 *   - Type: 0xEE (DPL=3, trap gate, 32-bit)
 * 
 * DPL=3 allows user mode (Ring 3) to execute int $0x80.
 * 
 * Preconditions: IDT initialized
 * Postconditions: int $0x80 from user mode dispatches to syscall_handler
 */
void syscall_init(void);
/**
 * System call dispatcher (called from assembly).
 * 
 * @param regs  Register save area (regs->eax = syscall number)
 * 
 * Dispatches to syscall_table[regs->eax].
 * Arguments: regs->ebx (arg1), regs->ecx (arg2), regs->edx (arg3).
 * Sets regs->eax = return value.
 */
void syscall_handler(registers_t *regs);
/**
 * sys_exit - Terminate current process.
 * 
 * @param status  Exit status code
 * @return        Does not return
 */
int sys_exit(int status);
/**
 * sys_write - Write to file descriptor.
 * 
 * @param fd     File descriptor (1 = stdout, 2 = stderr)
 * @param buf    Buffer to write
 * @param count  Number of bytes
 * @return       Bytes written, or -1 on error
 */
int sys_write(int fd, const char *buf, int count);
/**
 * sys_read - Read from file descriptor.
 * 
 * @param fd     File descriptor (0 = stdin)
 * @param buf    Buffer to read into
 * @param count  Maximum bytes to read
 * @return       Bytes read, or -1 on error
 * 
 * Note: Currently returns -1 (not implemented).
 */
int sys_read(int fd, char *buf, int count);
/**
 * sys_yield - Voluntarily yield CPU.
 * 
 * @return  0 on success
 */
int sys_yield(void);
/**
 * sys_getpid - Get current process ID.
 * 
 * @return  PID of current process, or -1 if none
 */
int sys_getpid(void);
```
---
## Algorithm Specification
### Algorithm: Process Creation

![Process Page Directory Isolation](./diagrams/tdd-diag-m4-09.svg)

```
INPUT: name (string), entry_point (function pointer), user_mode (bool)
OUTPUT: Pointer to PCB, or NULL on failure
INVARIANT: Process added to global list, NOT in scheduler queue
PROCEDURE process_create(name, entry_point, user_mode):
1. // Allocate PCB
   proc = kmalloc(sizeof(process_t))
   IF proc == NULL:
       RETURN NULL
2. // Initialize identification
   proc->pid = next_pid++
   strncpy(proc->name, name, 31)
   proc->name[31] = '\0'
3. // Initialize scheduling state
   proc->state = PROCESS_STATE_READY
   proc->priority = 1
   proc->time_slice = 10  // 10 ticks = 100ms at 100Hz
4. // Set entry point
   proc->entry_point = (uint32_t)entry_point
   proc->eip = (uint32_t)entry_point
5. // Allocate kernel stack
   kernel_stack_mem = kmalloc(KERNEL_STACK_SIZE)  // 8KB
   IF kernel_stack_mem == NULL:
       kfree(proc)
       RETURN NULL
   proc->kernel_stack = (uint32_t)kernel_stack_mem + KERNEL_STACK_SIZE
6. IF user_mode:
       // Allocate user stack
       user_stack_mem = kmalloc(USER_STACK_SIZE)  // 64KB
       IF user_stack_mem == NULL:
           kfree(kernel_stack_mem)
           kfree(proc)
           RETURN NULL
       proc->user_stack = (uint32_t)user_stack_mem + USER_STACK_SIZE
       // Create page directory
       proc->page_directory = create_user_page_directory()
       IF proc->page_directory == 0:
           kfree(user_stack_mem)
           kfree(kernel_stack_mem)
           kfree(proc)
           RETURN NULL
       // Set up kernel stack for user mode entry
       // Push: SS, ESP, EFLAGS, CS, EIP, then general regs
       stack = (uint32_t *)proc->kernel_stack
       *--stack = 0x23                    // SS (user data)
       *--stack = proc->user_stack        // ESP (user stack)
       *--stack = 0x202                   // EFLAGS (IF=1)
       *--stack = 0x1B                    // CS (user code | RPL=3)
       *--stack = proc->eip               // EIP (entry point)
       // General registers (initial state)
       *--stack = 0  // EAX
       *--stack = 0  // ECX
       *--stack = 0  // EDX
       *--stack = 0  // EBX
       *--stack = 0  // ESP (placeholder)
       *--stack = proc->user_stack  // EBP
       *--stack = 0  // ESI
       *--stack = 0  // EDI
       proc->esp = (uint32_t)stack
       proc->ebp = proc->user_stack
       proc->eflags = 0x202
   ELSE:
       // Kernel mode process
       proc->page_directory = 0  // Use kernel's page directory
       proc->user_stack = 0
       // Set up stack to "return" to entry point
       stack = (uint32_t *)proc->kernel_stack
       // Push process_exit_handler as return address
       *--stack = (uint32_t)process_exit_handler
       *--stack = proc->eip  // "Return" to entry point
       // General registers
       *--stack = 0  // EAX
       *--stack = 0  // ECX
       *--stack = 0  // EDX
       *--stack = 0  // EBX
       *--stack = (uint32_t)(stack + 5)  // ESP
       *--stack = (uint32_t)(stack + 5)  // EBP
       *--stack = 0  // ESI
       *--stack = 0  // EDI
       proc->eflags = 0x202
       *--stack = proc->eflags
       *--stack = 0x08  // KERNEL_CODE_SEL
       *--stack = proc->eip
       proc->esp = (uint32_t)stack
       proc->ebp = (uint32_t)(stack + 5)
7. // Add to process list
   proc->next = process_list
   proc->prev = NULL
   IF process_list != NULL:
       process_list->prev = proc
   process_list = proc
8. RETURN proc
```
### Algorithm: Context Switch (Assembly)

![INT 0x80 System Call Flow](./diagrams/tdd-diag-m4-10.svg)

```nasm
; kernel/context_switch.asm
; void context_switch(process_t *old, process_t *new)
[BITS 32]
; PCB offsets (must match process.h)
OFFSET_EAX        equ 0x30
OFFSET_EBX        equ 0x34
OFFSET_ECX        equ 0x38
OFFSET_EDX        equ 0x3C
OFFSET_ESI        equ 0x40
OFFSET_EDI        equ 0x44
OFFSET_EBP        equ 0x48
OFFSET_ESP        equ 0x4C
OFFSET_EIP        equ 0x50
OFFSET_EFLAGS     equ 0x54
OFFSET_PAGE_DIR   equ 0x58
OFFSET_KSTACK     equ 0x5C
section .text
[GLOBAL context_switch]
[EXTERN tss_esp0_ptr]
context_switch:
    ; [esp+4] = old process (may be NULL)
    ; [esp+8] = new process (must not be NULL)
    mov eax, [esp+4]        ; old process
    mov edx, [esp+8]        ; new process
    ; ========================================
    ; SAVE OLD PROCESS STATE
    ; ========================================
    test eax, eax
    jz .load_new            ; Skip if old is NULL (first switch)
    ; Save callee-saved and caller-saved registers
    mov [eax+OFFSET_EBX], ebx
    mov [eax+OFFSET_ECX], ecx
    mov [eax+OFFSET_EDX], edx
    mov [eax+OFFSET_ESI], esi
    mov [eax+OFFSET_EDI], edi
    mov [eax+OFFSET_EBP], ebp
    ; Save ESP (stack pointer after our arguments)
    mov ecx, esp
    add ecx, 8              ; Point to return address
    mov [eax+OFFSET_ESP], ecx
    ; Save return address as EIP
    mov ecx, [esp]
    mov [eax+OFFSET_EIP], ecx
    ; Save EFLAGS
    pushfd
    pop ecx
    mov [eax+OFFSET_EFLAGS], ecx
    ; Save EAX (contains old PCB pointer, but save for completeness)
    mov ecx, [esp+4]        ; Get original first argument
    mov [eax+OFFSET_EAX], ecx
.load_new:
    ; ========================================
    ; LOAD NEW PROCESS STATE
    ; ========================================
    test edx, edx
    jz .panic               ; Cannot switch to NULL
    ; Switch page directory if needed
    mov ecx, [edx+OFFSET_PAGE_DIR]
    test ecx, ecx
    jz .skip_cr3            ; 0 means use kernel's PD
    mov cr3, ecx            ; Load CR3, flushes TLB
.skip_cr3:
    ; Update TSS.ESP0 for ring transitions
    mov ecx, [edx+OFFSET_KSTACK]
    mov [tss_esp0_ptr], ecx
    ; Restore callee-saved registers
    mov ebx, [edx+OFFSET_EBX]
    mov ecx, [edx+OFFSET_ECX]
    mov esi, [edx+OFFSET_ESI]
    mov edi, [edx+OFFSET_EDI]
    mov ebp, [edx+OFFSET_EBP]
    ; Restore EFLAGS
    push dword [edx+OFFSET_EFLAGS]
    popfd
    ; Restore EAX
    mov eax, [edx+OFFSET_EAX]
    ; Switch stacks and "return" to new EIP
    mov esp, [edx+OFFSET_ESP]
    ret                     ; Pops new->eip into EIP
.panic:
    ; No process to run - halt
    cli
.halt_loop:
    hlt
    jmp .halt_loop
```
### Algorithm: Round-Robin Scheduling

![System Call Dispatch Table](./diagrams/tdd-diag-m4-11.svg)

```
INPUT: None (called from timer interrupt or yield)
OUTPUT: Context switch to next process
INVARIANT: At least one process in system (idle process)
PROCEDURE scheduler_tick(regs):
1. tick_count++
2. current = process_get_current()
3. IF current == NULL:
       RETURN  // No process running yet
4. IF current->time_slice > 0:
       current->time_slice--
5. IF current->time_slice == 0:
       scheduler_yield()
PROCEDURE scheduler_yield():
1. current = process_get_current()
2. IF ready_queue == NULL:
       IF current != NULL:
           current->time_slice = 10
       RETURN  // No other process to run
3. // Select next process
   IF current == NULL:
       next = ready_queue
   ELSE:
       // Put current back in ready state
       current->state = PROCESS_STATE_READY
       current->time_slice = 10
       // Next in circular queue
       next = current->next
       // Skip non-ready processes
       WHILE next != current AND next->state != PROCESS_STATE_READY:
           next = next->next
       IF next == current OR next->state != PROCESS_STATE_READY:
           RETURN  // No ready process found
4. // Update ready queue head
   ready_queue = next
5. // Mark as running
   next->state = PROCESS_STATE_RUNNING
6. // Skip if same process
   IF current == next:
       RETURN
7. // Update current_process global
   current_process = next
8. // Update TSS.ESP0
   IF next->kernel_stack != 0:
       tss_update_esp0(next->kernel_stack)
9. // Perform context switch
   context_switch(current, next)
   // NEVER RETURNS HERE for old process
   // New process resumes from its saved EIP
```
### Algorithm: User Mode Entry

![IDT Gate DPL for User Syscalls](./diagrams/tdd-diag-m4-12.svg)

```
INPUT: entry_point (EIP for user code), stack_ptr (ESP for user stack)
OUTPUT: NEVER RETURNS - begins executing in Ring 3
INVARIANT: TSS.ESP0 set to valid kernel stack
PROCEDURE jump_to_user_mode(entry_point, stack_ptr):
1. // Get current EFLAGS and enable interrupts
   eflags = read_eflags()
   eflags = eflags | 0x200  // Set IF
2. // Update TSS.ESP0 to current kernel stack
   kernel_esp = read_esp()
   tss_update_esp0(kernel_esp)
3. // Disable interrupts during transition
   cli
4. // Load user data segments
   mov ax, 0x23  // User data selector (0x20 | RPL=3)
   mov ds, ax
   mov es, ax
   mov fs, ax
   mov gs, ax
5. // Push stack frame for iret
   push 0x23           // SS (user data segment | RPL=3)
   push stack_ptr      // ESP (user stack)
   push eflags         // EFLAGS
   push 0x1B           // CS (user code segment | RPL=3)
   push entry_point    // EIP
6. // Jump to user mode
   iret  // Pops EIP, CS, EFLAGS, ESP, SS
         // CPU transitions to Ring 3
         // Execution continues at entry_point
// NEVER REACHES HERE
```
### Algorithm: System Call Dispatch

![Kernel Stack per Process](./diagrams/tdd-diag-m4-13.svg)

```
INPUT: Registers from INT 0x80 (EAX=syscall#, EBX/ECX/EDX=args)
OUTPUT: Return value in EAX
PROCEDURE syscall_asm_entry:
1. // Save all registers
   pusha               // EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI
   push ds, es, fs, gs // Segment registers
2. // Load kernel data segment
   mov ax, 0x10
   mov ds, ax
   mov es, ax
3. // Call C handler
   push esp            // Pointer to register structure
   call syscall_handler
   add esp, 4
4. // Restore segment registers
   pop gs, fs, es, ds
5. // Restore general registers
   popa  // Note: EAX restored with return value
6. // Return to user mode
   iret
PROCEDURE syscall_handler(regs):
1. syscall_num = regs->eax
2. arg1 = regs->ebx
3. arg2 = regs->ecx
4. arg3 = regs->edx
5. IF syscall_num >= 0 AND syscall_num < NUM_SYSCALLS:
       func = syscall_table[syscall_num]
       IF func != NULL:
           result = func(arg1, arg2, arg3)
       ELSE:
           result = -1  // Invalid syscall
   ELSE:
       result = -1  // Invalid syscall number
6. regs->eax = result  // Return value
```
### Algorithm: User Page Directory Creation
```
INPUT: None
OUTPUT: Physical address of new page directory, or 0 on failure
INVARIANT: Kernel mapped supervisor-only in higher half
PROCEDURE create_user_page_directory():
1. // Allocate page directory frame
   pd_frame = pmm_alloc_frame()
   IF pd_frame == NULL:
       RETURN 0
2. pd = (uint32_t *)pd_frame
3. // Zero all entries
   FOR i = 0 TO 1023:
       pd[i] = 0
4. // Copy kernel mappings (higher half)
   // Kernel is at 0xC0000000+ (entries 768-1023)
   kernel_pd = current_page_directory
   FOR i = 768 TO 1023:
       pd[i] = kernel_pd[i]
       // Mark as supervisor-only for security
       pd[i] = pd[i] & ~PAGE_USER
5. // Identity map first 1MB (VGA, BIOS) - optional
   // User processes should NOT access VGA directly
   // They should use syscalls
6. RETURN (uint32_t)pd_frame
```
---
## Error Handling Matrix
| Error | Detection Point | Detection Method | Recovery | User-Visible |
|-------|-----------------|------------------|----------|--------------|
| PCB allocation fails | process_create() | kmalloc returns NULL | Return NULL | "ERROR: Failed to allocate PCB" |
| Kernel stack allocation fails | process_create() | kmalloc returns NULL | Free PCB, return NULL | "ERROR: Failed to allocate kernel stack" |
| User stack allocation fails | process_create() | kmalloc returns NULL | Free kernel stack + PCB | "ERROR: Failed to allocate user stack" |
| Page directory allocation fails | process_create() | pmm_alloc_frame returns 0 | Free all, return NULL | "ERROR: Failed to allocate page directory" |
| PCB offset mismatch | context_switch() | Garbage values after switch | System crash | Triple fault or random behavior |
| TSS.ESP0 not updated | User mode entry | Page fault on Ring 3→0 | Triple fault | System resets |
| NULL process in context switch | context_switch() | test edx, edx == 0 | Halt loop | "PANIC: No process to run" |
| Circular queue corruption | scheduler_yield() | Loop doesn't return to start | May hang or crash | System freeze |
| IDT syscall gate DPL=0 | syscall_init() | User int 0x80 causes GPF | GPF handler runs | "EXCEPTION: General Protection Fault" |
| Invalid syscall number | syscall_handler() | Number >= NUM_SYSCALLS | Return -1 | Syscall fails silently |
| NULL handler in syscall table | syscall_handler() | func == NULL | Return -1 | Syscall fails silently |
| Process exit without yield | process_terminate() | current_process == proc | Must call yield after | Process lingers |
| User access kernel memory | User process | Page fault (supervisor bit) | Page fault handler | "PAGE FAULT... protection violation" |
---
## Implementation Sequence with Checkpoints
### Phase 1: PCB Structure Definition (2-3 hours)
**Files:** `kernel/process.h`
**Tasks:**
1. Define `process_state_t` enum
2. Define `process_t` struct with ALL fields and exact offsets
3. Define PCB offset macros for assembly
4. Define stack size constants
5. Declare function prototypes
**Checkpoint:**
```c
// Compile and verify sizes
#include "process.h"
_Static_assert(sizeof(process_t) == 0x70, "PCB size mismatch");
_Static_assert(offsetof(process_t, eax) == 0x30, "EAX offset mismatch");
_Static_assert(offsetof(process_t, esp) == 0x4C, "ESP offset mismatch");
```
### Phase 2: Process Creation (4-5 hours)
**Files:** `kernel/process.c`
**Tasks:**
1. Implement `process_init()` to initialize globals
2. Implement `process_create()` for kernel processes:
   - Allocate PCB
   - Set up kernel stack
   - Initialize registers
3. Add to process list
4. Implement `process_terminate()` to mark terminated
5. Implement `process_get_current()` and `process_get_by_pid()`
**Checkpoint:**
```c
process_init();
process_t *p1 = process_create("Test1", test_func, false);
ASSERT(p1 != NULL);
ASSERT(p1->pid == 1);
ASSERT(p1->state == PROCESS_STATE_READY);
kprintf("Created process '%s' with PID %d\n", p1->name, p1->pid);
```
### Phase 3: Kernel Stack Setup (3-4 hours)
**Files:** `kernel/process.c`
**Tasks:**
1. Allocate kernel stack with kmalloc
2. Set up initial stack frame for kernel process:
   - Push return address (process_exit_handler)
   - Push entry point
   - Push general registers
   - Push EFLAGS, CS, EIP
3. Set `proc->esp` to point to stack top
4. Implement `process_exit_handler()` that terminates and yields
**Checkpoint:**
```c
// Verify stack setup
process_t *p = process_create("StackTest", test_func, false);
uint32_t *stack = (uint32_t *)p->esp;
kprintf("Stack top: 0x%x\n", p->esp);
kprintf("EIP on stack: 0x%x\n", stack[2]);  // Should be test_func address
```
### Phase 4: Context Switch Assembly (5-7 hours)
**Files:** `kernel/context_switch.asm`
**Tasks:**
1. Define PCB offset constants (must match header)
2. Implement save section:
   - Save EBX, ECX, EDX, ESI, EDI, EBP
   - Calculate and save ESP
   - Save return address as EIP
   - Save EFLAGS
3. Implement load section:
   - Load CR3 if page_directory != 0
   - Update TSS.ESP0
   - Restore registers
   - Switch stack and ret
**Checkpoint:**
```nasm
; Verify assembly compiles
nasm -f elf32 kernel/context_switch.asm -o kernel/context_switch.o
objdump -d kernel/context_switch.o
; Verify offsets match PCB_* macros
```
### Phase 5: TSS Structure and GDT Entry (3-4 hours)
**Files:** `kernel/tss.h`, `kernel/tss.c`, `kernel/gdt.h`, `kernel/gdt.c`
**Tasks:**
1. Define `tss_t` struct (104 bytes, packed)
2. Implement `tss_init()`:
   - Zero TSS
   - Set ss0 = 0x10
   - Set iomap_base = sizeof(tss_t)
3. Update GDT to add TSS entry at index 5:
   - Base = &tss
   - Limit = sizeof(tss_t) - 1
   - Access = 0x89 (32-bit available TSS)
4. Implement `gdt_flush()` assembly to reload GDT
5. Load TR with `ltr 0x28`
**Checkpoint:**
```c
tss_init();
kprintf("TSS at 0x%x, TR=0x28\n", &tss);
// Verify with QEMU monitor: info registers
```
### Phase 6: TSS.ESP0 Update Mechanism (2-3 hours)
**Files:** `kernel/tss.c`
**Tasks:**
1. Declare `tss_esp0_ptr` global (for assembly access)
2. Implement `tss_update_esp0()`:
   - Set `tss.esp0 = esp0`
   - Update `tss_esp0_ptr = &tss.esp0`
3. Export `tss_esp0_ptr` in assembly as `[EXTERN tss_esp0_ptr]`
**Checkpoint:**
```c
tss_update_esp0(0x90000);
ASSERT(tss.esp0 == 0x90000);
ASSERT(*tss_esp0_ptr == 0x90000);
```
### Phase 7: Round-Robin Scheduler Queue (3-4 hours)
**Files:** `kernel/scheduler.h`, `kernel/scheduler.c`
**Tasks:**
1. Implement `scheduler_init()`
2. Implement `scheduler_add()`:
   - Insert into circular queue
   - Handle empty queue case
   - Handle non-empty queue case
3. Implement `scheduler_remove()`:
   - Unlink from circular queue
   - Handle single-element case
   - Update ready_queue if needed
**Checkpoint:**
```c
scheduler_init();
process_t *a = process_create("A", func, false);
process_t *b = process_create("B", func, false);
scheduler_add(a);
scheduler_add(b);
ASSERT(a->next == b);
ASSERT(b->next == a);
ASSERT(a->prev == b);
ASSERT(b->prev == a);
```
### Phase 8: scheduler_tick from Timer IRQ (3-4 hours)
**Files:** `kernel/scheduler.c`, update `kernel/timer.c`
**Tasks:**
1. Update timer handler to call `scheduler_tick(regs)`
2. Implement `scheduler_tick()`:
   - Increment tick_count
   - Decrement time_slice
   - Call yield if slice == 0
**Checkpoint:**
```c
// In timer.c:
static void timer_handler(registers_t *regs) {
    tick_count++;
    scheduler_tick(regs);  // Add this
    // EOI sent by dispatcher
}
// After running, verify scheduler_tick is called
kprintf("Scheduler ticks: %d\n", scheduler.tick_count);
```
### Phase 9: scheduler_yield and Context Switch Call (2-3 hours)
**Files:** `kernel/scheduler.c`
**Tasks:**
1. Implement `scheduler_yield()`:
   - Select next process from queue
   - Skip non-ready processes
   - Update current_process global
   - Call tss_update_esp0()
   - Call context_switch()
2. Handle edge cases:
   - No other process ready
   - Same process selected
**Checkpoint:**
```c
// Create two processes, add to scheduler
process_t *a = process_create("A", proc_a, false);
process_t *b = process_create("B", proc_b, false);
scheduler_add(a);
scheduler_add(b);
// Start first process
current_process = a;
a->state = PROCESS_STATE_RUNNING;
// Manually trigger yield
scheduler_yield();
// Should switch to process B
```
### Phase 10: User Mode Entry via iret (4-5 hours)
**Files:** `kernel/usermode.h`, `kernel/usermode.c`
**Tasks:**
1. Implement inline EFLAGS read
2. Implement inline ESP read
3. Implement `jump_to_user_mode()`:
   - Get EFLAGS with IF=1
   - Update TSS.ESP0
   - Disable interrupts
   - Load user data segments (0x23)
   - Push: SS, ESP, EFLAGS, CS, EIP
   - Execute iret
4. Test with simple user function
**Checkpoint:**
```c
// Create simple user function
void user_test(void) {
    // Write 'U' to VGA directly (if mapped)
    volatile char *vga = (volatile char *)0xB8000;
    *vga = 'U';
    while(1);
}
// Jump to user mode
tss_update_esp0(0x90000);
jump_to_user_mode((uint32_t)user_test, 0xC0500000);
// Should see 'U' on screen, running in Ring 3
```
### Phase 11: User Process Page Directory Creation (3-4 hours)
**Files:** `kernel/process.c`
**Tasks:**
1. Implement `create_user_page_directory()`:
   - Allocate frame for PD
   - Zero all entries
   - Copy kernel entries (768-1023)
   - Clear PAGE_USER bit on kernel entries
2. Update `process_create()` to call this for user_mode=true
3. Allocate and map user stack in user address space
**Checkpoint:**
```c
uint32_t user_pd = create_user_page_directory();
ASSERT(user_pd != 0);
uint32_t *pd = (uint32_t *)user_pd;
// Verify kernel entries copied
ASSERT((pd[768] & PAGE_PRESENT) != 0);
// Verify user bit cleared
ASSERT((pd[768] & PAGE_USER) == 0);
```
### Phase 12: INT 0x80 Syscall Entry Stub (3-4 hours)
**Files:** `kernel/syscall_entry.asm`
**Tasks:**
1. Create `syscall_asm_entry` label
2. Push all general registers (pusha)
3. Push segment registers
4. Load kernel data segment
5. Push ESP and call C handler
6. Restore segments and registers
7. Execute iret
**Checkpoint:**
```nasm
; Verify assembly
nasm -f elf32 kernel/syscall_entry.asm -o kernel/syscall_entry.o
objdump -d kernel/syscall_entry.o
; Verify pusha/popa pair, iret at end
```
### Phase 13: Syscall Dispatch Table (2-3 hours)
**Files:** `kernel/syscall.h`, `kernel/syscall.c`
**Tasks:**
1. Define syscall number constants (SYS_EXIT, etc.)
2. Define syscall_func_t typedef
3. Create syscall_table[] array
4. Implement `syscall_init()`:
   - Set IDT gate 0x80 with type 0xEE (DPL=3)
5. Implement `syscall_handler()`:
   - Extract syscall number and args
   - Dispatch to table
   - Store return value
**Checkpoint:**
```c
syscall_init();
// Verify IDT entry
// Entry 0x80 should have DPL=3 (bits 5-6 of type_attr = 11)
// Type should be 0xE (trap gate)
```
### Phase 14: sys_write, sys_exit, sys_getpid (3-4 hours)
**Files:** `kernel/syscall.c`
**Tasks:**
1. Implement `sys_exit(status)`:
   - Call process_terminate()
   - Call scheduler_yield()
2. Implement `sys_write(fd, buf, count)`:
   - Check fd == 1 or fd == 2
   - Write to VGA using vga_put_char
   - Return count
3. Implement `sys_getpid()`:
   - Return current_process->pid
4. Implement `sys_yield()`:
   - Call scheduler_yield()
**Checkpoint:**
```c
// Test sys_write
char msg[] = "Hello from syscall!\n";
int result = sys_write(1, msg, strlen(msg));
ASSERT(result == strlen(msg));
// Test sys_getpid
int pid = sys_getpid();
ASSERT(pid == current_process->pid);
```
### Phase 15: Multi-Process Demo and Testing (4-5 hours)
**Files:** `kernel/test_processes.c`, update `kernel/main.c`
**Tasks:**
1. Create test processes (kernel mode):
   - Process A: prints 'A' to screen region
   - Process B: prints 'B' to different region
   - Process C: prints 'C' to different region
2. Create user mode test process:
   - Uses sys_write to print
   - Uses sys_getpid to show PID
   - Calls sys_exit
3. Update kernel_main() to:
   - Initialize all subsystems in order
   - Create test processes
   - Add to scheduler
   - Enable interrupts
   - Let scheduler run
**Checkpoint:**
```bash
make clean && make
make run
# Should see:
# - Processes A, B, C running concurrently
# - Characters appearing in different screen regions
# - User process printing via syscalls
# - System continues running until ESC pressed
```
---
## Test Specification
### Test: PCB Structure Layout
**Function:** Verify assembly offsets match C struct
**Test:**
```c
#include "process.h"
#include <stddef.h>
void test_pcb_offsets(void) {
    ASSERT(offsetof(process_t, pid) == PCB_PID);
    ASSERT(offsetof(process_t, state) == PCB_STATE);
    ASSERT(offsetof(process_t, eax) == PCB_EAX);
    ASSERT(offsetof(process_t, ebx) == PCB_EBX);
    ASSERT(offsetof(process_t, ecx) == PCB_ECX);
    ASSERT(offsetof(process_t, edx) == PCB_EDX);
    ASSERT(offsetof(process_t, esi) == PCB_ESI);
    ASSERT(offsetof(process_t, edi) == PCB_EDI);
    ASSERT(offsetof(process_t, ebp) == PCB_EBP);
    ASSERT(offsetof(process_t, esp) == PCB_ESP);
    ASSERT(offsetof(process_t, eip) == PCB_EIP);
    ASSERT(offsetof(process_t, eflags) == PCB_EFLAGS);
    ASSERT(offsetof(process_t, page_directory) == PCB_PAGE_DIR);
    ASSERT(offsetof(process_t, kernel_stack) == PCB_KSTACK);
    ASSERT(sizeof(process_t) == PCB_SIZE);
}
```
### Test: Process Creation
**Function:** Create and verify kernel process
**Happy Path:**
```c
void test_func(void) { while(1); }
process_t *p = process_create("Test", test_func, false);
ASSERT(p != NULL);
ASSERT(p->pid == 1);
ASSERT(p->state == PROCESS_STATE_READY);
ASSERT(p->entry_point == (uint32_t)test_func);
ASSERT(p->kernel_stack != 0);
ASSERT(p->page_directory == 0);  // Kernel process
```
**Edge Case - NULL name:**
```c
process_t *p = process_create(NULL, test_func, false);
// Should handle gracefully (crash or use default name)
```
**Failure - Out of memory:**
```c
// Exhaust heap, then try create
// Should return NULL, not crash
```
### Test: Scheduler Queue Operations
**Function:** Circular queue add/remove
**Test:**
```c
scheduler_init();
process_t *a = process_create("A", func, false);
process_t *b = process_create("B", func, false);
process_t *c = process_create("C", func, false);
scheduler_add(a);
ASSERT(ready_queue == a);
ASSERT(a->next == a);
ASSERT(a->prev == a);
scheduler_add(b);
ASSERT(ready_queue == b);  // New head
ASSERT(a->next == b);
ASSERT(b->next == a);
scheduler_remove(b);
ASSERT(ready_queue == a);
ASSERT(a->next == a);
```
### Test: Context Switch
**Function:** Save and restore process state
**Test:**
```c
volatile int a_count = 0;
volatile int b_count = 0;
void proc_a(void) {
    while (1) {
        a_count++;
        scheduler_yield();
    }
}
void proc_b(void) {
    while (1) {
        b_count++;
        scheduler_yield();
    }
}
process_t *pa = process_create("A", proc_a, false);
process_t *pb = process_create("B", proc_b, false);
scheduler_add(pa);
scheduler_add(pb);
current_process = pa;
pa->state = PROCESS_STATE_RUNNING;
// Run for some ticks
for (int i = 0; i < 100; i++) {
    // Simulate timer tick
    scheduler_tick(NULL);
}
// Both should have run
ASSERT(a_count > 0);
ASSERT(b_count > 0);
kprintf("A: %d, B: %d\n", a_count, b_count);
```
### Test: User Mode Entry
**Function:** Transition to Ring 3
**Test:**
```c
volatile bool user_reached = false;
void user_func(void) {
    user_reached = true;
    while(1);
}
// Set up user process
process_t *p = process_create("User", user_func, true);
ASSERT(p != NULL);
ASSERT(p->page_directory != 0);
// Jump to user mode
current_process = p;
tss_update_esp0(p->kernel_stack);
jump_to_user_mode(p->entry_point, p->user_stack);
// Should never reach here
// In QEMU, verify:
// - CS = 0x1B (CPL=3)
// - SS = 0x23
// - user_reached == true
```
### Test: TSS.ESP0 Update
**Function:** Correct kernel stack on Ring 3→0
**Test:**
```c
// Set TSS.ESP0
tss_update_esp0(0x90000);
ASSERT(tss.esp0 == 0x90000);
// Trigger syscall from user mode
// CPU should push to 0x90000 area
// Verify in QEMU with -d int
```
### Test: System Call Dispatch
**Function:** INT 0x80 routes to correct handler
**Test:**
```c
// From user mode or kernel:
int result;
// sys_getpid
__asm__ volatile (
    "int $0x80"
    : "=a"(result)
    : "a"(SYS_GETPID)
);
ASSERT(result == current_process->pid);
// sys_write
const char *msg = "Test";
__asm__ volatile (
    "int $0x80"
    : "=a"(result)
    : "a"(SYS_WRITE), "b"(1), "c"(msg), "d"(4)
);
ASSERT(result == 4);
```
### Test: Process Isolation
**Function:** User process cannot access kernel memory
**Test:**
```c
void malicious_user(void) {
    // Try to write to kernel memory
    volatile int *kernel = (volatile int *)0xC0100000;
    *kernel = 0xDEAD;  // Should page fault
    // If we reach here, isolation failed!
    sys_write(1, "FAIL", 4);
}
process_t *p = process_create("Malicious", malicious_user, true);
// Run process
// Should see page fault with:
//   "Faulting address: 0xC0100000"
//   "Protection violation"
```
### Test: Multi-Process Preemption
**Function:** Timer forces context switch
**Test:**
```c
volatile int counters[3] = {0, 0, 0};
void counter_proc(void) {
    int idx = /* somehow get index */;
    while (1) {
        counters[idx]++;
        // No yield - pure preemption
    }
}
// Create 3 processes, run for 1 second
// All counters should be > 0
// Counters should be roughly equal (fair scheduling)
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Context switch (full) | <1500 cycles (~1.5μs at 1GHz) | Performance counter before/after context_switch |
| Timer tick overhead | <500 cycles | Compare tick count to expected |
| System call (INT 0x80) | ~1000 cycles | Measure from int to iret |
| TLB flush on CR3 change | ~100-500 cycles | Depends on TLB size |
| User mode entry (iret) | ~50-100 cycles | Single instruction timing |
| PCB allocation | <10μs | kmalloc timing |
| Scheduler queue add/remove | <500 cycles | Pointer manipulation |
| TSS.ESP0 update | <50 cycles | Single memory write |
---
## State Machine: Process Lifecycle

![Multi-Process Demo Memory Layout](./diagrams/tdd-diag-m4-14.svg)

```
                    process_create()
[NULL] ────────────────────────────────> [READY]
                                              │
                                     scheduler_add()
                                              │
                                              ▼
              ┌─────────────────── scheduler_tick() ───────────────────┐
              │                           │                            │
              │                      time_slice=0                      │
              │                           │                            │
              ▼                           ▼                            │
         [RUNNING] <─────────────── scheduler_yield()                  │
              │                           │                            │
              │                    scheduler_yield()                   │
              │                           │                            │
              │                           ▼                            │
              │                       [READY] <────────────────────────┘
              │                           ▲
              │                           │
         sys_exit()              scheduler_unblock()
              │                           ▲
              │                           │
              ▼                      [BLOCKED]
        [TERMINATED] <────────── scheduler_block()
                                   (I/O wait, etc.)
ILLEGAL TRANSITIONS:
- READY → RUNNING without scheduler
- RUNNING → TERMINATED without sys_exit
- TERMINATED → any state
- NULL → RUNNING without READY
- BLOCKED → RUNNING without READY
```
---
## Concurrency Model
**Preemptive multitasking** via timer interrupt (IRQ0 at 100Hz).
**Synchronization Points:**
| Resource | Access Pattern | Protection |
|----------|---------------|------------|
| current_process | Read: frequent (every syscall), Write: context switch | Interrupts disabled during switch |
| ready_queue | Modify: scheduler only | Interrupts disabled during modify |
| process_list | Modify: create/terminate | Interrupts disabled |
| PCB fields | Read/Write: owning process only | Process isolation via page tables |
**Interrupt State During Critical Operations:**
- Context switch: Interrupts disabled (cli before, sti after iret)
- Scheduler operations: Interrupts disabled
- System call handler: Interrupts enabled (trap gate)
- User mode: Interrupts enabled (IF=1 in EFLAGS)
**No spinlocks or mutexes needed** because:
- Single CPU core
- Interrupts provide synchronization
- Each process has isolated address space
---
## Crash Recovery
**No persistent state** in this module. All data is in volatile memory.
**Failure Modes:**
| Failure | Symptom | Debug Method |
|---------|---------|--------------|
| PCB offset mismatch | Garbage after context switch | Verify sizeof and offsetof |
| TSS.ESP0 not updated | Triple fault on syscall | Check tss_update_esp0() called |
| Invalid IDT gate for syscall | GPF on int 0x80 | Verify DPL=3 (0xEE) |
| Circular queue broken | Scheduler hangs or loops forever | Print queue state |
| NULL process pointer | Triple fault | Check current_process before use |
| User accesses kernel | Page fault | Verify supervisor-only bits |
**Debug Aids:**
- QEMU `-d int` logs all interrupts
- QEMU `-d cpu` logs CPU state changes
- GDB: `break context_switch`, `info registers`
- Serial output from scheduler for trace
- Print ready_queue state on each switch
---
## Hardware Soul: Context Switch Analysis
{{DIAGRAM:tdd-diag-m4-15}}
### Complete Context Switch Timing
```
Event                          | Cycles | Notes
-------------------------------|--------|---------------------------
Timer interrupt fires          | —      | Hardware
CPU pushes SS,ESP,EFLAGS,CS,EIP| 20-40  | Memory writes
ISR stub entry                 | —      |
pusha (8 regs)                 | 8-16   | Memory writes
push 4 seg regs                | 8-16   | Memory writes
Load kernel DS                 | 12-20  | Segment loads slow
Call scheduler_tick            | 2-5    |
Scheduler logic                | 100-300| C code
Call context_switch            | 2-5    |
─────────────────────────────────────────────────────────
Save old registers             | 10-20  | Memory writes
Save ESP, EIP, EFLAGS          | 5-10   |
Load CR3 (if different)        | 50-200 | TLB flush!
Update TSS.ESP0                | 5-10   | Memory write
Restore new registers          | 10-20  | Memory reads
iret                           | 20-40  | Pop EIP,CS,EFLAGS,ESP,SS
─────────────────────────────────────────────────────────
TOTAL                          | 300-700| Without CR3 change
With CR3 change                | 400-900| TLB flush adds 100-200
Worst case (cold cache)        | 1000-1500|
```
### Cache Behavior During Context Switch
```
Before switch (Process A running):
- Process A's stack: HOT (in L1 cache)
- Process A's code: HOT
- Page tables for A: TLB entries present
- Kernel code: WARM
After switch (Process B running):
- Process B's stack: COLD (cache miss on first access)
- Process B's code: COLD or WARM
- Page tables for B: TLB cold if CR3 changed
- Kernel code: WARM (shared)
Time to "warm up" Process B:
- Stack: ~10-20 accesses (cache lines loaded)
- Code: ~50-100 instructions (instruction cache)
- TLB: ~10-20 page accesses
- Total: ~100-500 cycles of cache misses
```
### TLB Behavior on CR3 Change
```
TLB State Before:
  - 64 entries (typical 32-bit CPU)
  - Mappings for Process A's code, stack, heap
  - Kernel mappings (global if PAGE_GLOBAL set)
CR3 Reload Effect:
  - All non-global entries invalidated
  - Global entries preserved (if CR4.PGE=1)
  - Subsequent memory accesses cause TLB misses
TLB Warm-up:
  - Each TLB miss: ~30-50 cycles (page table walk)
  - Process typically touches 10-50 unique pages quickly
  - Total warm-up cost: ~300-2500 cycles
Optimization (future):
  - PCID (Process-Context Identifier) on newer CPUs
  - Allows multiple processes in TLB simultaneously
```
---
## Build System Integration
Update `Makefile`:
```makefile
# Add to C_SOURCES
C_SOURCES += kernel/process.c kernel/scheduler.c kernel/tss.c \
             kernel/gdt.c kernel/usermode.c kernel/syscall.c \
             kernel/test_processes.c
# Add to ASM_SOURCES
ASM_SOURCES += kernel/context_switch.asm kernel/syscall_entry.asm \
               kernel/gdt_flush.asm
# Update dependencies
kernel/main.o: kernel/main.c kernel/process.h kernel/scheduler.h \
               kernel/tss.h kernel/syscall.h kernel/usermode.h
# Add test target
test-processes: $(OS_IMAGE)
	qemu-system-i386 -fda $(OS_IMAGE) -serial stdio -d int
```
---
## Synced Criteria
[[CRITERIA_JSON: {"module_id": "build-os-m4", "criteria": ["Process control block (PCB) struct defined with uint32_t pid, char name[32], process_state_t state enum, uint32_t priority, uint32_t time_slice at documented byte offsets", "PCB saved register fields at exact offsets: eax(0x30), ebx(0x34), ecx(0x38), edx(0x3C), esi(0x40), edi(0x44), ebp(0x48), esp(0x4C), eip(0x50), eflags(0x54)", "PCB memory management fields: page_directory(0x58), kernel_stack(0x5C), user_stack(0x60), entry_point(0x64)", "PCB linked list pointers: next(0x68), prev(0x6C), total size 0x70 (112 bytes)", "PCB offset macros defined in both C header and assembly with identical values", "process_create(name, entry_point, user_mode) allocates PCB via kmalloc, assigns unique PID, initializes state to READY, time_slice to 10", "process_create allocates kernel stack (8KB) via kmalloc for each process", "process_create for user_mode=true allocates user stack (64KB) and creates new page directory via pmm_alloc_frame", "process_create sets up initial stack frame with EIP at entry point and EFLAGS with IF=1 (0x202)", "process_terminate sets state to TERMINATED and removes PCB from process list", "Task State Segment (TSS) struct is 104 bytes packed with esp0(0x04) and ss0(0x08) fields", "TSS initialized with ss0=0x10 (kernel data segment) and iomap_base=104", "TSS descriptor added to GDT at index 5 (selector 0x28) with access byte 0x89 (32-bit available TSS)", "Task Register (TR) loaded with TSS selector via ltr 0x28 instruction", "tss_update_esp0(esp0) writes to tss.esp0 field, exported tss_esp0_ptr global for assembly access", "context_switch(old, new) assembly saves EBX,ECX,EDX,ESI,EDI,EBP,ESP,EIP,EFLAGS to old PCB offsets", "context_switch loads CR3 with new->page_directory if non-zero (triggers TLB flush)", "context_switch updates TSS.ESP0 via tss_esp0_ptr before restoring new process registers", "context_switch restores registers from new PCB and executes ret to jump to new->eip", "Round-robin scheduler maintains circular doubly-linked ready_queue of PCBs", "scheduler_tick(regs) called from timer IRQ0 handler, decrements current->time_slice", "scheduler_yield() called when time_slice=0, selects next READY process from circular queue", "scheduler_yield() updates current_process global, calls tss_update_esp0, calls context_switch", "scheduler_add(proc) inserts process into circular ready queue with proc->next and proc->prev links", "scheduler_remove(proc) unlinks process from ready queue by updating neighbor pointers", "jump_to_user_mode(entry, stack) disables interrupts, loads user segments (0x23), pushes SS/ESP/EFLAGS/CS/EIP, executes iret", "iret stack for user entry: SS=0x23, ESP=user_stack, EFLAGS=0x202, CS=0x1B, EIP=entry_point", "System call interface uses INT 0x80 with syscall number in EAX, arguments in EBX/ECX/EDX", "IDT entry 0x80 configured with type_attr=0xEE (DPL=3, trap gate, 32-bit) allowing Ring 3 invocation", "syscall_handler(regs) reads EAX for syscall number, dispatches via syscall_table[] array", "sys_write(fd, buf, count) implemented for fd=1/2 writing to VGA via vga_put_char, returns count", "sys_exit(status) calls process_terminate and scheduler_yield, does not return", "sys_getpid() returns current_process->pid", "sys_yield() calls scheduler_yield() for cooperative multitasking", "System call assembly stub saves all registers with pusha, segment registers, calls handler, restores, irets", "System call return value stored in regs->eax for user process", "User process page directory copies kernel entries (768-1023) with PAGE_USER bit cleared for isolation", "User process page fault on kernel memory access demonstrates isolation (supervisor-only violation)", "At least 3 kernel-mode processes run concurrently with preemptive multitasking, each writing to different screen regions", "Timer interrupt at 100Hz provides 10ms time slices, scheduler_tick called on each IRQ0", "current_process global pointer updated atomically with interrupts disabled before context_switch", "Kernel stack per process is 8KB, user stack per user process is 64KB", "System boots in QEMU with multiple processes running and preempting correctly"]}]
<!-- END_TDD_MOD -->


# Project Structure: Build Your Own OS
## Directory Tree
```
project-root/
├── 01_boot/                         # Bootloader components (M1)
│   ├── 01_boot.asm                  # Stage 1 MBR bootloader (512 bytes)
│   ├── 02_gdt.asm                   # GDT definitions and descriptor
│   ├── 03_a20.asm                   # A20 line enable routines
│   ├── 04_load_kernel.asm           # Disk read routines
│   └── 05_protected.asm             # Protected mode transition
│
├── 02_kernel/                       # Kernel core (all milestones)
│   │
│   │  # === M1: Boot & Entry ===
│   ├── 01_entry.asm                 # Kernel entry point, BSS zeroing
│   ├── 02_main.c                    # C kernel entry (kernel_main)
│   ├── 03_vga.h                     # VGA text mode interface
│   ├── 04_vga.c                     # VGA implementation
│   ├── 05_serial.h                  # Serial port interface
│   ├── 06_serial.c                  # Serial implementation
│   ├── 07_kprintf.h                 # Kernel printf interface
│   ├── 08_kprintf.c                 # kprintf implementation
│   ├── 09_linker.ld                 # Linker script
│   │
│   │  # === M2: Interrupts ===
│   ├── 10_idt.h                     # IDT structure and function declarations
│   ├── 11_idt.c                     # IDT initialization and gate management
│   ├── 12_idt_stubs.asm             # Assembly ISR stubs for all 256 vectors
│   ├── 13_interrupts.h              # Interrupt handling interface
│   ├── 14_interrupts.c              # Common interrupt dispatcher, exception handlers
│   ├── 15_pic.h                     # PIC interface
│   ├── 16_pic.c                     # PIC initialization, remapping, EOI
│   ├── 17_timer.h                   # PIT timer interface
│   ├── 18_timer.c                   # PIT initialization and tick counter
│   ├── 19_keyboard.h                # Keyboard driver interface
│   ├── 20_keyboard.c                # PS/2 keyboard implementation
│   └── 21_exception_names.c         # Exception message strings
│
│   │  # === M3: Memory Management ===
│   ├── 30_memory.h                  # Memory management interface
│   ├── 31_pmm.h                     # Physical memory manager interface
│   ├── 32_pmm.c                     # PMM bitmap allocator implementation
│   ├── 33_vmm.h                     # Virtual memory manager interface
│   ├── 34_vmm.c                     # Page table management implementation
│   ├── 35_kheap.h                   # Kernel heap interface
│   ├── 36_kheap.c                   # Linked-list heap allocator implementation
│   ├── 37_pagefault.h               # Page fault handler interface
│   └── 38_pagefault.c               # Page fault diagnostics implementation
│
│   │  # === M4: Processes & Scheduling ===
│   ├── 40_process.h                 # PCB structure and process management interface
│   ├── 41_process.c                 # Process creation, termination, list management
│   ├── 42_context_switch.asm        # Assembly context switch implementation
│   ├── 43_scheduler.h               # Scheduler interface
│   ├── 44_scheduler.c               # Round-robin scheduler implementation
│   ├── 45_tss.h                     # TSS structure and interface
│   ├── 46_tss.c                     # TSS initialization and ESP0 update
│   ├── 47_gdt.h                     # Updated GDT interface (add TSS entry)
│   ├── 48_gdt.c                     # Updated GDT implementation
│   ├── 49_gdt_flush.asm             # Assembly GDT reload
│   ├── 50_usermode.h                # User mode transition interface
│   ├── 51_usermode.c                # jump_to_user_mode implementation
│   ├── 52_syscall.h                 # System call interface
│   ├── 53_syscall.c                 # Syscall dispatch and implementations
│   ├── 54_syscall_entry.asm         # Assembly syscall entry stub
│   └── 55_test_processes.c          # Multi-process demonstration
│
├── 03_include/                      # Header files
│   ├── 01_stdint.h                  # Fixed-width integer types
│   ├── 02_stdarg.h                  # Variable argument handling
│   ├── 03_cpu.h                     # CPU register structures for ISRs
│   ├── 04_memory_layout.h           # Memory region constants
│   └── 05_process.h                 # Shared process definitions
│
├── Makefile                         # Build system
├── boot.bin                         # Bootloader output (512 bytes)
├── kernel.bin                       # Kernel ELF binary
└── os.img                           # Bootable floppy image
```
## Creation Order
1. **Project Setup** (30 min)
   - Create directory structure (`01_boot/`, `02_kernel/`, `03_include/`)
   - Create `03_include/01_stdint.h`
   - Create `03_include/02_stdarg.h`
2. **Bootloader (M1)** (8-12 hours)
   - `01_boot/01_boot.asm` — MBR stub, segment setup
   - `01_boot/02_gdt.asm` — GDT definitions
   - `01_boot/03_a20.asm` — A20 enable routines
   - `01_boot/04_load_kernel.asm` — Disk read
   - `01_boot/05_protected.asm` — Mode switch
3. **Kernel Entry & Console (M1)** (6-8 hours)
   - `02_kernel/09_linker.ld` — Linker script with `_kernel_end`
   - `02_kernel/01_entry.asm` — Entry point, BSS zeroing
   - `02_kernel/03_vga.h`, `04_vga.c` — VGA driver
   - `02_kernel/05_serial.h`, `06_serial.c` — Serial driver
   - `02_kernel/07_kprintf.h`, `08_kprintf.c` — Formatted output
   - `02_kernel/02_main.c` — kernel_main with tests
4. **Build System (M1)** (2-3 hours)
   - `Makefile` — NASM, GCC, LD rules
   - Test: `make run` shows boot message
5. **IDT & ISR Stubs (M2)** (6-8 hours)
   - `02_kernel/10_idt.h`, `11_idt.c` — IDT structure and loading
   - `02_kernel/12_idt_stubs.asm` — All 256 ISR stubs
   - `02_kernel/13_interrupts.h`, `14_interrupts.c` — Dispatcher
6. **PIC & Timer (M2)** (4-6 hours)
   - `02_kernel/15_pic.h`, `16_pic.c` — PIC remapping
   - `02_kernel/17_timer.h`, `18_timer.c` — PIT at 100Hz
   - `02_kernel/21_exception_names.c` — Exception strings
7. **Keyboard (M2)** (4-5 hours)
   - `02_kernel/19_keyboard.h`, `20_keyboard.c` — Scancode handling, buffer
8. **Physical Memory (M3)** (4-5 hours)
   - `03_include/04_memory_layout.h` — Memory constants
   - `02_kernel/30_memory.h` — Interface header
   - `02_kernel/31_pmm.h`, `32_pmm.c` — Bitmap allocator
9. **Virtual Memory (M3)** (6-8 hours)
   - `02_kernel/33_vmm.h`, `34_vmm.c` — Page tables, identity map
   - `02_kernel/37_pagefault.h`, `38_pagefault.c` — Handler with CR2
10. **Kernel Heap (M3)** (4-5 hours)
    - `02_kernel/35_kheap.h`, `36_kheap.c` — Linked-list allocator
11. **Process Structures (M4)** (4-5 hours)
    - `03_include/05_process.h` — Shared definitions
    - `02_kernel/40_process.h` — PCB structure (verify offsets!)
    - `02_kernel/41_process.c` — Creation, termination
12. **Context Switch (M4)** (5-7 hours)
    - `02_kernel/42_context_switch.asm` — Save/restore, CR3, TSS update
13. **TSS & GDT Update (M4)** (3-4 hours)
    - `02_kernel/45_tss.h`, `46_tss.c` — TSS init, ESP0 update
    - `02_kernel/47_gdt.h`, `48_gdt.c` — Add TSS descriptor
    - `02_kernel/49_gdt_flush.asm` — Reload GDT
14. **Scheduler (M4)** (4-5 hours)
    - `02_kernel/43_scheduler.h`, `44_scheduler.c` — Round-robin
15. **User Mode (M4)** (3-4 hours)
    - `02_kernel/50_usermode.h`, `51_usermode.c` — iret transition
16. **System Calls (M4)** (4-5 hours)
    - `02_kernel/52_syscall.h`, `53_syscall.c` — Dispatch table
    - `02_kernel/54_syscall_entry.asm` — Entry stub
17. **Integration & Testing (M4)** (4-5 hours)
    - `02_kernel/55_test_processes.c` — Multi-process demo
    - Update `02_kernel/02_main.c` — Initialize all subsystems
## File Count Summary
- **Total files:** 55
- **Directories:** 3 main + build artifacts
- **Assembly files:** 7 (.asm)
- **C source files:** 32 (.c)
- **Header files:** 12 (.h)
- **Linker scripts:** 1 (.ld)
- **Build files:** 1 (Makefile)
- **Estimated lines of code:** ~8,000-10,000
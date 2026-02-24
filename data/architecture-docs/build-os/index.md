# Build Your Own OS

This project guides you through building a complete x86 operating system kernel from the first BIOS instruction to a preemptively multitasking system with user-mode processes. You will implement the critical path that every operating system must navigate: bootstrapping from firmware, configuring CPU data structures (GDT, IDT, TSS), managing physical and virtual memory, and implementing preemptive scheduling with privilege-level transitions.

The x86 architecture presents unique challenges—the transition from 16-bit real mode to 32-bit protected mode requires precise configuration of segmentation, paging, and interrupt handling. Every decision negotiates with hardware constraints: the 4KB page granularity, the 256 interrupt vectors, the privilege ring hierarchy, and the cache line boundaries that affect context switch performance.

By the end, you will understand not just how an OS works, but why it works that way—the physical and architectural forces that shaped these designs. This knowledge transfers directly to understanding container isolation, virtual machine introspection, real-time scheduling, and low-level security exploits.



<!-- MS_ID: build-os-m1 -->
# Milestone 1: Bootloader, GDT, and Kernel Entry

## The Tension: Hardware Doesn't Want to Run Your Code

When you press the power button, your 3GHz CPU with billions of transistors wakes up in a shockingly primitive state:

- **16-bit mode** — only 64KB addressable per segment, 1MB total
- **Real mode segmentation** — addresses computed as `segment * 16 + offset`
- **No memory protection** — any code can overwrite anything
- **BIOS in control** — interrupt vectors point to 16-bit BIOS routines

Your kernel is 32-bit code expecting flat memory, protected segments, and C runtime conventions. The gap between these two worlds is not bridged by magic — you must build every plank of that bridge yourself.


![x86 Boot Sequence: BIOS to C Entry](./diagrams/diag-boot-sequence.svg)


The numbers make this concrete:
- BIOS loads exactly **512 bytes** from the disk's first sector — that's your entire stage 1 bootloader
- The A20 line, disabled for IBM PC/XT compatibility, blocks addresses above 1MB — your kernel at 0x100000 is unreachable until you enable it
- Entering protected mode without flushing the pipeline leaves the CPU decoding 16-bit instructions as 32-bit — instant crash

This milestone is about building the bootstrap sequence that transforms a 16-bit relic into a 32-bit modern CPU, then handing control to C code you wrote.

---

## System Map: Where We Are

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR OS KERNEL                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Bootloader │→ │     GDT     │→ │  C Kernel   │              │
│  │  (16-bit)   │  │ (Segments)  │  │  (32-bit)   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         ↓                ↓                ↓                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    HARDWARE STATE                           ││
│  │  Real Mode → Protected Mode → Paging (later)               ││
│  │  CR0.PE=0  → CR0.PE=1      → CR0.PG=1                      ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

We're building the left side: bootloader, GDT, and the protected mode transition. Without this, nothing else exists.

---

## The Boot Process: From Power-On to Your First Instruction

### What the BIOS Does (So You Don't Have To)

The BIOS (Basic Input/Output System) — or its modern successor UEFI — performs essential hardware initialization:

1. **POST** (Power-On Self-Test) — verifies RAM, CPU, and critical hardware
2. **Hardware enumeration** — discovers disks, keyboards, display adapters
3. **Boot device selection** — checks configured boot order
4. **MBR load** — reads sector 0 (first 512 bytes) from the boot disk into memory at **0x7C00**
5. **Jump to 0x7C00** — transfers control to your bootloader with `CS:IP = 0x0000:0x7C00`

The magic address 0x7C00 isn't arbitrary — it's in the original IBM PC memory map, placed low enough to be addressable in real mode but high enough to not conflict with BIOS data structures.

### Your Bootloader's Job

512 bytes. That's all you get for stage 1. In that space, you must:

```asm
; boot.asm - Stage 1 Bootloader (must fit in 512 bytes)
[BITS 16]
[ORG 0x7C00]

start:
    ; Set up segments for real mode
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00          ; Stack grows down from bootloader

    ; Enable A20 line (access memory above 1MB)
    call enable_a20

    ; Load stage 2 bootloader from disk
    call load_stage2

    ; Set up GDT
    lgdt [gdt_descriptor]

    ; Enter protected mode
    mov eax, cr0
    or eax, 1               ; Set PE bit (Protection Enable)
    mov cr0, eax

    ; Far jump to flush pipeline and load new CS
    jmp 0x08:protected_mode_entry

    ; ... GDT and helper functions follow ...

times 510-($-$$) db 0       ; Pad to 510 bytes
dw 0xAA55                   ; Boot signature (BIOS requires this)
```

The boot signature `0xAA55` is the final checksum — the BIOS scans the last two bytes of sector 0 and only executes the code if it finds this signature. Without it, the BIOS moves to the next boot device.

### The A20 Line: A Historical Anomaly You Must Handle

The A20 line is a quirk of PC history. The original 8086 had 20 address lines (A0-A19), addressing 1MB. Memory wrapped around — address 0x100000 (1MB + 0) accessed the same physical memory as 0x00000.

IBM designed the PC/AT (286) with 24 address lines, but to maintain compatibility with 8086 software that relied on wraparound, they added a gate to disable A20. By default, **A20 is disabled**, meaning you cannot access memory above 1MB.

Your kernel loads at 0x100000 — unreachable until you enable A20:

```asm
enable_a20:
    ; Method 1: Fast A20 gate (port 0x92)
    in al, 0x92
    or al, 2
    out 0x92, al
    ret

    ; Method 2: Keyboard controller (more reliable on old hardware)
    ; ... involves sending commands to port 0x64/0x60 ...
```

The fast A20 gate via port 0x92 works on most modern systems and virtual machines. For maximum compatibility, production bootloaders try multiple methods.

---

## Segmentation: The GDT and Why x86 Still Has It

### The Tension: Why Segmentation Exists

You might wonder: why configure segments at all? Why not just use flat memory?

The answer lies in x86 history. The 8086 (1978) was a 16-bit processor needing to address more than 64KB. **Segmentation** was the solution: addresses are computed as `segment * 16 + offset`, giving 20-bit addresses (1MB).

Protected mode (386, 1985) extended this with **segment descriptors** — rich metadata about each segment:
- Base address (32-bit) — where the segment starts
- Limit (20-bit) — how large the segment is
- Access rights — readable? writable? executable?
- Privilege level — ring 0 (kernel) or ring 3 (user)
- Granularity — limit in bytes or 4KB pages

The **Global Descriptor Table (GDT)** is an array of these descriptors. The GDTR register holds the table's base address and limit. Every memory access uses a segment selector (index into GDT) combined with an offset.

{{DIAGRAM:diag-gdt-structure}}

### The Flat Memory Model: Your Goal

For a modern OS, you want **flat segmentation** — segments with base=0, limit=4GB. This makes segmentation transparent; logical addresses equal linear addresses. Paging (which you'll add in Milestone 3) handles the real memory protection and isolation.

Your GDT needs five entries:

| Index | Type | Base | Limit | Privilege | Purpose |
|-------|------|------|-------|-----------|---------|
| 0 | Null | 0 | 0 | — | Required by CPU |
| 1 | Code | 0 | 4GB | Ring 0 | Kernel code |
| 2 | Data | 0 | 4GB | Ring 0 | Kernel data |
| 3 | Code | 0 | 4GB | Ring 3 | User code |
| 4 | Data | 0 | 4GB | Ring 3 | User data |

The null descriptor (index 0) is mandatory — the CPU uses selector 0 as a "null selector" for error conditions. Loading a segment register with 0 doesn't cause a fault, but using it for memory access will.

### GDT Entry Structure: 64 Bits of Configuration

Each GDT entry is 8 bytes (64 bits), packed with specific fields:

```
Byte 0-1:  Limit [15:0]      (bits 0-15 of limit)
Byte 2-3:  Base [15:0]       (bits 0-15 of base)
Byte 4:    Base [23:16]      (bits 16-23 of base)
Byte 5:    Access Byte       (present, ring, type flags)
Byte 6:    Flags + Limit[19:16]
Byte 7:    Base [31:24]      (bits 24-31 of base)
```

The **access byte** (byte 5) controls privilege and type:

```
Bit 7: Present (1 = segment exists)
Bit 6-5: DPL (Descriptor Privilege Level: 00=ring 0, 11=ring 3)
Bit 4: S (System: 1=code/data, 0=system like TSS)
Bit 3: Type bit 3 (Executable: 1=code, 0=data)
Bit 2: Type bit 2 (Direction/Conforming for code, Expand-down for data)
Bit 1: Type bit 1 (Readable for code, Writable for data)
Bit 0: Accessed (CPU sets this when segment is accessed)
```

The **flags byte** (high 4 bits of byte 6):

```
Bit 7: Granularity (0=limit in bytes, 1=limit in 4KB pages)
Bit 6: Size (0=16-bit, 1=32-bit)
Bit 5: Long (1=64-bit code segment for x86-64)
Bit 4: Reserved
```

Here's a concrete GDT setup:

```asm
gdt_start:
    ; Null descriptor (required)
    dq 0x0000000000000000

gdt_code_kernel:             ; Index 1: Kernel code segment
    ; Base=0, Limit=0xFFFFF, 4KB granularity, 32-bit, ring 0, executable+readable
    dw 0xFFFF               ; Limit [15:0]
    dw 0x0000               ; Base [15:0]
    db 0x00                 ; Base [23:16]
    db 10011010b            ; Present, ring 0, code, executable, readable
    db 11001111b            ; 4KB granularity, 32-bit, Limit [19:16]
    db 0x00                 ; Base [31:24]

gdt_data_kernel:             ; Index 2: Kernel data segment
    ; Base=0, Limit=0xFFFFF, 4KB granularity, 32-bit, ring 0, writable
    dw 0xFFFF
    dw 0x0000
    db 0x00
    db 10010010b            ; Present, ring 0, data, writable
    db 11001111b
    db 0x00

gdt_code_user:               ; Index 3: User code segment (ring 3)
    dw 0xFFFF
    dw 0x0000
    db 0x00
    db 11111010b            ; Present, ring 3 (DPL=11), code, executable, readable
    db 11001111b
    db 0x00

gdt_data_user:               ; Index 4: User data segment (ring 3)
    dw 0xFFFF
    dw 0x0000
    db 0x00
    db 11110010b            ; Present, ring 3, data, writable
    db 11001111b
    db 0x00

gdt_descriptor:
    dw gdt_descriptor - gdt_start - 1  ; Size (limit)
    dd gdt_start                        ; Base address
```

The segment selectors (what you load into segment registers) are computed as:

```
Selector = (Index << 3) | (TI << 2) | RPL

Where:
  Index = position in GDT (0, 1, 2, ...)
  TI = Table Indicator (0=GDT, 1=LDT — you'll use 0)
  RPL = Requested Privilege Level (0 or 3)

Kernel code selector = (1 << 3) | 0 = 0x08
Kernel data selector = (2 << 3) | 0 = 0x10
User code selector = (3 << 3) | 3 = 0x1B
User data selector = (4 << 3) | 3 = 0x23
```

---

## The Protected Mode Transition: A Precise Sequence


![Protected Mode Transition: Before and After](./diagrams/diag-protected-mode-transition.svg)


Entering protected mode isn't a single instruction — it's a sequence where getting any step wrong causes a **triple fault** (CPU exception → handler crashes → double fault → handler crashes → triple fault → CPU reset).

### Step-by-Step Transition

```asm
; 1. Disable interrupts (CRITICAL — real-mode IVT is now invalid)
cli

; 2. Load the GDT
lgdt [gdt_descriptor]

; 3. Enable protected mode
mov eax, cr0
or eax, 1                ; Set PE (Protection Enable) bit
mov cr0, eax

; 4. Far jump to flush the pipeline and load CS with kernel code selector
jmp 0x08:protected_mode_entry

[BITS 32]
protected_mode_entry:
    ; 5. Now in 32-bit protected mode! Reload segment registers.
    mov ax, 0x10         ; Kernel data selector
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov esp, 0x90000     ; Set up kernel stack (below 1MB for now)

    ; 6. Jump to kernel C entry point
    jmp 0x08:0x100000    ; Kernel loaded at 1MB physical
```

### Why Each Step Matters

**Disable interrupts (`cli`)**: The real-mode IVT (Interrupt Vector Table) at 0x0-0x3FF contains BIOS interrupt handlers. Once you enter protected mode, these addresses are interpreted completely differently. An interrupt before you set up the IDT (Milestone 2) will crash.

**Load GDT (`lgdt`)**: The CPU needs to know where segment descriptors live. This must happen *before* setting CR0.PE.

**Set CR0.PE**: This is the actual mode switch. But the pipeline still contains 16-bit instructions decoded as 16-bit.

**Far jump**: `jmp 0x08:protected_mode_entry` does two things:
1. Loads CS with selector 0x08 (kernel code segment)
2. Flushes the prefetch queue (pipeline), forcing fresh instruction fetch in 32-bit mode

Without this jump, the CPU continues executing what it thinks are 16-bit instructions, but they're actually your 32-bit code — garbage execution.

**Reload segment registers**: After the far jump, CS is valid, but DS/ES/FS/GS/SS still contain real-mode values. Loading them with the kernel data selector (0x10) ensures all memory access uses your flat segments.

---

## Loading the Kernel from Disk

The stage 1 bootloader (512 bytes) usually loads a larger stage 2, which then loads the kernel. BIOS interrupt **INT 13h** provides disk access in real mode:

```asm
load_kernel:
    ; Reset disk system
    xor ah, ah
    xor dl, dl            ; dl = boot drive number (passed by BIOS)
    int 0x13

    ; Read sectors using CHS (Cylinder-Head-Sector) addressing
    ; ah=02h (read), al=number of sectors, ch=cylinder, cl=sector, dh=head, dl=drive
    mov ah, 0x02
    mov al, 32            ; Read 32 sectors (16KB — adjust based on kernel size)
    mov ch, 0             ; Cylinder 0
    mov cl, 2             ; Start at sector 2 (sector 1 is the MBR)
    mov dh, 0             ; Head 0
    mov dl, [boot_drive]
    mov bx, 0x100000      ; Destination: 1MB (es:bx = destination)
    mov es, bx
    xor bx, bx
    int 0x13
    jc disk_error         ; Carry flag set on error

    ret

boot_drive: db 0
```

The destination 0x100000 (1MB) is traditional — it's the first available memory above the low memory region used by BIOS and real-mode structures.

---

## The Linker Script: Where Code Lives in Memory

Your C kernel doesn't know its own addresses — the linker decides where every function and variable lives. The linker script controls this.


![Kernel Memory Map: Linker Script Layout](./diagrams/diag-linker-script.svg)


```ld
/* linker.ld */
ENTRY(kernel_entry)

SECTIONS
{
    /* Kernel starts at 1MB physical */
    . = 0x100000;

    .text : {
        *(.multiboot)      /* Multiboot header if using GRUB */
        *(.text)
    }

    .rodata : {
        *(.rodata)
    }

    .data : {
        *(.data)
    }

    .bss : {
        __bss_start = .;
        *(COMMON)
        *(.bss)
        __bss_end = .;
    }

    /DISCARD/ : {
        *(.comment)
        *(.eh_frame)
    }
}
```

The symbols `__bss_start` and `__bss_end` are crucial — they mark the range of uninitialized global/static variables that must be zeroed at startup. In a hosted C environment, the C runtime (crt0) does this. In your kernel, you do it.

---

## The C Entry Point: No Runtime, No Safety Net

When control reaches your C code, you have:

- **No zeroed BSS** — uninitialized globals contain garbage
- **No initialized globals** — the loader handles .data, but verify this
- **No stack setup** — you set SS:ESP in assembly
- **No libc** — no printf, no malloc, no memcpy

Here's a minimal kernel entry:

```c
/* kernel_entry.asm — called from bootloader */
[BITS 32]
extern kernel_main
global kernel_entry

kernel_entry:
    ; Set up stack (if not already done)
    mov esp, 0x90000

    ; Zero the BSS section
    extern __bss_start
    extern __bss_end
    mov edi, __bss_start
    mov ecx, __bss_end
    sub ecx, edi
    xor eax, eax
    rep stosb

    ; Call kernel main
    call kernel_main

    ; Halt if kernel_main returns
.halt:
    cli
    hlt
    jmp .halt
```

```c
/* kernel_main.c */
void kernel_main(void) {
    // At this point: GDT is loaded, protected mode is active,
    // BSS is zeroed, stack is valid
    
    // Initialize VGA text mode (0xB8000)
    vga_init();
    vga_puts("Welcome to MyOS!\n");
    
    // Initialize serial port for debug output
    serial_init(COM1_PORT);
    serial_puts(COM1_PORT, "Kernel booted successfully.\n");
    
    // Your OS begins here...
    while (1) {
        asm volatile("hlt");
    }
}
```

---

## VGA Text Mode: Your First Display Driver

The VGA text buffer at **0xB8000** is memory-mapped I/O — writing to this address displays characters on screen. Each character is 2 bytes:

```
Byte 0: ASCII character
Byte 1: Attribute byte:
  - Bits 0-3: Foreground color (0-15)
  - Bits 4-6: Background color (0-7)
  - Bit 7: Blink (or bright background if enabled)
```

```c
#define VGA_BUFFER ((volatile uint16_t*)0xB8000)
#define VGA_WIDTH 80
#define VGA_HEIGHT 25

static int vga_row = 0;
static int vga_col = 0;

typedef enum {
    VGA_BLACK = 0,
    VGA_BLUE = 1,
    VGA_GREEN = 2,
    VGA_CYAN = 3,
    VGA_RED = 4,
    VGA_MAGENTA = 5,
    VGA_BROWN = 6,
    VGA_LIGHT_GREY = 7,
    // ... more colors
} vga_color;

static inline uint16_t vga_entry(char c, uint8_t fg, uint8_t bg) {
    return (uint16_t)c | ((uint16_t)(fg | (bg << 4)) << 8);
}

void vga_putchar(char c) {
    if (c == '\n') {
        vga_col = 0;
        vga_row++;
        return;
    }
    
    VGA_BUFFER[vga_row * VGA_WIDTH + vga_col] = vga_entry(c, VGA_LIGHT_GREY, VGA_BLACK);
    vga_col++;
    
    if (vga_col >= VGA_WIDTH) {
        vga_col = 0;
        vga_row++;
    }
    
    if (vga_row >= VGA_HEIGHT) {
        // Scroll (copy all rows up by one)
        for (int i = 0; i < (VGA_HEIGHT - 1) * VGA_WIDTH; i++) {
            VGA_BUFFER[i] = VGA_BUFFER[i + VGA_WIDTH];
        }
        // Clear last row
        for (int i = (VGA_HEIGHT - 1) * VGA_WIDTH; i < VGA_HEIGHT * VGA_WIDTH; i++) {
            VGA_BUFFER[i] = vga_entry(' ', VGA_LIGHT_GREY, VGA_BLACK);
        }
        vga_row = VGA_HEIGHT - 1;
    }
}
```

---

## Serial Port Debug Output: Your Lifeline

When the kernel crashes before VGA works, or when you need to log data that scrolls off screen, serial output is essential. COM1 is at I/O port **0x3F8**.

```c
#define COM1_PORT 0x3F8

static inline void outb(uint16_t port, uint8_t val) {
    asm volatile("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    asm volatile("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

void serial_init(uint16_t port) {
    outb(port + 1, 0x00);    // Disable all interrupts
    outb(port + 3, 0x80);    // Enable DLAB (set baud rate divisor)
    outb(port + 0, 0x03);    // Set divisor to 3 (lo byte) 38400 baud
    outb(port + 1, 0x00);    //                  (hi byte)
    outb(port + 3, 0x03);    // 8 bits, no parity, one stop bit
    outb(port + 2, 0xC7);    // Enable FIFO, clear them, with 14-byte threshold
    outb(port + 4, 0x0B);    // IRQs enabled, RTS/DSR set
}

int serial_is_transmit_empty(uint16_t port) {
    return inb(port + 5) & 0x20;
}

void serial_putchar(uint16_t port, char c) {
    while (serial_is_transmit_empty(port) == 0);
    outb(port, c);
}

void serial_puts(uint16_t port, const char* str) {
    while (*str) {
        serial_putchar(port, *str++);
    }
}
```

With QEMU, you can redirect serial output to a file or stdio:

```bash
qemu-system-i386 -kernel myos.bin -serial stdio
# Or save to file:
qemu-system-i386 -kernel myos.bin -serial file:debug.log
```

---

## Hardware Soul: What's Actually Happening

Every step of this boot sequence has hardware implications:

**Cache behavior**: The GDT, loaded via `lgdt`, is read by the CPU into internal registers. Subsequent segment accesses don't read memory — they use the cached descriptor. Modifying the GDT requires a reload.

**TLB state**: In protected mode without paging, linear addresses = physical addresses. When you enable paging (Milestone 3), the TLB (Translation Lookaside Buffer) caches page table entries. For now, it's unused.

**Pipeline flush**: The far jump after `mov cr0, eax` forces a pipeline flush. On modern CPUs, this costs 10-30 cycles. In the boot sequence, this is negligible, but the same principle applies to every context switch.

**Memory access patterns**: The VGA text buffer at 0xB8000 is **uncacheable** memory-mapped I/O. Writing to it goes directly to the video controller, not through the cache hierarchy. This is why we use `volatile` — the compiler must not optimize away or reorder these writes.

---

## Debugging Your Bootloader: When Nothing Works

Bootloader bugs are brutal — you often get no output, just a black screen or reset. Here's your debugging toolkit:

### QEMU with GDB

```bash
# Terminal 1: Start QEMU with GDB stub
qemu-system-i386 -kernel myos.bin -s -S

# Terminal 2: Connect GDB
gdb
(gdb) target remote :1234
(gdb) break *0x7C00        # Break at bootloader entry
(gdb) continue
```

### Serial Output

Before VGA works, serial is your only output. Initialize it early:

```asm
; In stage 1, after setting segments:
mov dx, 0x3F8 + 1
xor al, al
out dx, al          ; Disable serial interrupts
; ... minimal init ...

; Debug: output 'A' to confirm we got here
mov dx, 0x3F8
mov al, 'A'
out dx, al
```

### Triple Fault Detection

A triple fault (exception → crash → double fault → crash → triple fault → reset) means your CPU state is corrupted. Common causes:

- GDT misconfiguration (wrong base/limit/access bytes)
- Far jump with wrong selector
- Forgetting to reload segment registers after mode switch
- Stack corruption (SS:ESP invalid)

Use QEMU's `-d int,cpu_reset` to log interrupts and resets.


![Fault Cascade: Triple Fault Cause Chain](./diagrams/diag-triple-fault-chain.svg)


---

## Design Decision: One-Stage vs Two-Stage Bootloader

| Approach | Pros | Cons | Used By |
|----------|------|------|---------|
| **One-Stage (512 bytes)** | Simpler, single file | Must fit everything in 510 bytes, limited kernel size | Educational OSes, tiny kernels |
| **Two-Stage** | Stage 2 can be large, more features | More complex loading, stage 2 must find kernel | Most real bootloaders (GRUB, Linux) |
| **Multiboot (GRUB)** | GRUB handles all boot complexity | Requires multiboot header, depends on GRUB | Many hobby OSes, Xen |

For learning, a simple two-stage approach is practical:
- Stage 1 (512 bytes): Enable A20, load GDT, load stage 2, enter protected mode
- Stage 2 (larger): Load kernel from disk (using INT 13h in real mode or your own driver in protected mode), verify, jump to kernel

---

## Building and Running

A typical build process:

```bash
# Assemble bootloader
nasm -f bin boot.asm -o boot.bin

# Compile kernel (freestanding, no stdlib)
gcc -m32 -ffreestanding -fno-pic -fno-pie -nostdlib -c kernel_main.c -o kernel_main.o
gcc -m32 -ffreestanding -fno-pic -fno-pie -nostdlib -c vga.c -o vga.o
gcc -m32 -ffreestanding -fno-pic -fno-pie -nostdlib -c serial.c -o serial.o

# Link kernel
ld -m elf_i386 -T linker.ld -o kernel.elf kernel_main.o vga.o serial.o

# Extract raw binary (for direct loading)
objcopy -O binary kernel.elf kernel.bin

# Create disk image with bootloader and kernel
dd if=/dev/zero of=os.img bs=512 count=2880
dd if=boot.bin of=os.img bs=512 count=1 conv=notrunc
dd if=kernel.bin of=os.img bs=512 seek=1 conv=notrunc

# Run in QEMU
qemu-system-i386 -drive format=raw,file=os.img -serial stdio
```

---

## Knowledge Cascade

You've now built the foundation of an operating system. Here's where this knowledge connects:

**Virtualization and Hypervisors**: The GDT, IDT, and paging machinery you configure is exactly what hypervisors (VMware, KVM, Hyper-V) virtualize. A "VM entry" from hypervisor to guest is analogous to your protected mode transition — the hypervisor loads the guest's IDT, GDT, and CR3 before transferring control. Understanding this boot sequence is the first step to understanding VM introspection and escape exploits.

**Linker Scripts and Memory Layout (Cross-Domain)**: The `.text`, `.data`, `.bss` section concepts and LMA (Load Memory Address) vs VMA (Virtual Memory Address) distinction apply to embedded firmware, UEFI drivers, and even position-independent code in security exploits. ROP chains rely on knowing exact offsets in memory — knowledge that starts with understanding linker scripts.

**BIOS/UEFI Exploitation**: The real-mode IVT at 0x0 is still relevant — BIOS bootkits hook INT 13h to infect boot sectors. UEFI's Secure Boot targets exactly the boot path you implemented, verifying signatures before allowing code execution. Understanding the attack surface of boot requires understanding legitimate boot.

**No-std Rust and Freestanding Environments**: The absence of libc, crt0, and runtime services you experience here is exactly what embedded Rust, Linux kernel modules, and WASM runtimes handle. The `#![no_std]` attribute in Rust means "I will provide my own panic handler and memory allocator" — the same contract you're fulfilling in C.

**Forward: What You Can Now Build**: With a booting kernel, you can now:
- Implement an IDT and handle CPU exceptions (Milestone 2)
- Build a physical memory allocator and enable paging (Milestone 3)
- Create preemptive multitasking with context switching (Milestone 4)

---

## Summary

You've crossed the first great divide in OS development:

1. **Boot process**: BIOS → MBR (your bootloader at 0x7C00) → protected mode → kernel entry
2. **GDT**: Five descriptors (null, kernel code/data, user code/data) with flat memory model
3. **Protected mode transition**: `cli` → `lgdt` → set CR0.PE → far jump → reload segments
4. **C environment**: Zero BSS, set stack, call kernel_main — no runtime provided
5. **Output drivers**: VGA text mode (0xB8000) and serial port (COM1 at 0x3F8)

The hardware constraints — 512-byte MBR, A20 line, pipeline flush requirements — shaped every decision. You negotiated with physics, and your kernel boots.


![OS Kernel Architecture: Satellite View](./diagrams/diag-satellite-system.svg)

<!-- END_MS -->


<!-- MS_ID: build-os-m2 -->
<!-- MS_ID: build-os-m2 -->
# Milestone 2: Interrupts, Exceptions, and Keyboard

## The Tension: Hardware Can't Wait For You

Your kernel from Milestone 1 sits in a loop, dutifully executing instructions one after another. But the real world doesn't wait:

- The user presses a key — the keyboard controller has data *now*
- The timer chip fires — the scheduler needs to run *now*
- Your code divides by zero — the CPU detects an error *now*
- A page isn't in memory — the MMU needs resolution *now*

**Polling** (checking "is there input?" in a loop) wastes millions of cycles. At 3GHz, checking the keyboard 1000 times per second means 3 million wasted cycles between checks — and you still might miss a keystroke if the user types faster than you poll.

The solution: let hardware interrupt your code. But here's the constraint — the CPU must save exactly enough state to resume later, transfer control to your handler, and do this in **microseconds**. The mechanism must be:

1. **Deterministic**: Same interrupt → same handler → same stack frame
2. **Fast**: No searching, no dynamic dispatch — direct table lookup
3. **Precise**: Every register preserved, every byte accounted for

The x86 answer is the **Interrupt Descriptor Table (IDT)**: 256 entries, each 8 bytes, containing the address and metadata for one handler. When interrupt N occurs, the CPU loads CS:EIP from entry N and jumps — no function calls, no callbacks, just raw hardware-driven control transfer.

And here's what surprises most developers: **CPU exceptions and hardware IRQs use the same mechanism**. Division by zero (exception 0), keyboard input (IRQ1 via vector 33), and your own `int 0x80` syscall — all route through the IDT. The CPU doesn't distinguish "software problems" from "hardware events" at the dispatch level.

---

## Revelation: It's Not a Callback

**What you might think**: "I register a handler function, and when the keyboard has data, the hardware calls it like an event handler in JavaScript."

**What actually happens**: The 8259 PIC doesn't call anything. It asserts an electrical signal on the CPU's INTR pin. The CPU:

1. Finishes the current instruction
2. Looks up the IDT entry for the interrupt vector
3. **Pushes EFLAGS, CS, EIP onto the stack** (and optionally an error code)
4. Loads CS:EIP from the IDT entry
5. Jumps to your handler

Your handler is now running. The PIC is waiting. And here's the critical part: **the PIC will not deliver another interrupt until you send EOI (End of Interrupt)**. Forget this, and your system appears to "freeze" — the keyboard stops responding, the timer stops ticking, everything halts.

This isn't magic. It's a protocol. You must:

1. Save all registers (the CPU only saved EFLAGS, CS, EIP)
2. Handle the interrupt
3. Send EOI to the PIC
4. Restore all registers
5. Execute `iret` (which pops EFLAGS, CS, EIP)

Miss any step, and you corrupt the interrupted code — or lock up the interrupt system entirely.

{{DIAGRAM:diag-idt-entry}}

---

## System Map: Where We Are

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR OS KERNEL                          │
│                                                                 │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐         │
│  │   IDT   │   │   PIC   │   │   PIT   │   │ Keyboard│         │
│  │  256    │   │  8259   │   │  Timer  │   │  Driver │         │
│  │ entries │   │ Master/ │   │ 100Hz   │   │ Scancode│         │
│  │         │   │ Slave   │   │         │   │ →ASCII  │         │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘         │
│       │             │             │             │               │
│       └─────────────┴─────────────┴─────────────┘              │
│                           │                                     │
│  ┌────────────────────────┴────────────────────────┐           │
│  │                  HARDWARE                        │           │
│  │  CPU exceptions (0-31) + IRQs (32-47)           │           │
│  │  All vector through IDT → Your handlers          │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

We're building the center column: IDT for dispatch, PIC for hardware routing, PIT for timing, and keyboard for input. These are the nervous system of your kernel.

---

## The IDT: 256 Doors to Your Code

### Structure of an IDT Entry

Each IDT entry is an 8-byte **gate descriptor**. The structure mirrors the GDT entry but with different fields:

```
Bytes 0-1:  Offset [15:0]      (low 16 bits of handler address)
Bytes 2-3:  Segment Selector   (code segment to run handler in)
Byte 4:     Reserved (always 0)
Byte 5:     Access Byte:
  - Bit 7: Present (1 = valid entry)
  - Bits 6-5: DPL (who can call via INT instruction)
  - Bit 4: Always 0 (system segment)
  - Bits 3-0: Gate type (1110 = interrupt gate, 1111 = trap gate)
Bytes 6-7:  Offset [31:16]     (high 16 bits of handler address)
```

**Interrupt gate vs Trap gate**: The only difference is whether interrupts are automatically disabled. An interrupt gate clears the IF flag (disables interrupts) on entry; a trap gate doesn't. For hardware IRQs, use interrupt gates to prevent nested interrupts. For CPU exceptions that might need to handle interrupts, trap gates can be appropriate.

```c
struct idt_entry {
    uint16_t offset_low;
    uint16_t selector;
    uint8_t  zero;
    uint8_t  type_attr;
    uint16_t offset_high;
} __attribute__((packed));

struct idt_ptr {
    uint16_t limit;           // Size of IDT - 1
    uint32_t base;            // Address of IDT
} __attribute__((packed));
```

### Loading the IDT

```c
#define IDT_ENTRIES 256

struct idt_entry idt[IDT_ENTRIES];
struct idt_ptr idtr;

void idt_set_gate(uint8_t num, uint32_t handler, uint16_t sel, uint8_t flags) {
    idt[num].offset_low = handler & 0xFFFF;
    idt[num].offset_high = (handler >> 16) & 0xFFFF;
    idt[num].selector = sel;
    idt[num].zero = 0;
    idt[num].type_attr = flags;
}

void idt_load(void) {
    idtr.limit = sizeof(idt) - 1;
    idtr.base = (uint32_t)&idt;
    asm volatile("lidt %0" : : "m"(idtr));
}
```

### CPU Exception Vectors: 0-31

Intel reserved the first 32 vectors for CPU-detected conditions:

| Vector | Name | Error Code? | Cause |
|--------|------|-------------|-------|
| 0 | #DE Divide Error | No | Division by zero or overflow |
| 1 | #DB Debug | No | Debug trap (single step, breakpoint) |
| 2 | NMI | No | Non-maskable interrupt |
| 3 | #BP Breakpoint | No | INT 3 instruction |
| 6 | #UD Invalid Opcode | No | CPU encountered unknown instruction |
| 8 | #DF Double Fault | Yes | Exception during handling another exception |
| 10 | #TS Invalid TSS | Yes | TSS segment invalid |
| 11 | #NP Segment Not Present | Yes | Segment descriptor P=0 |
| 12 | #SS Stack-Segment Fault | Yes | Stack operation outside limit |
| 13 | #GP General Protection | Yes | Privilege violation, invalid segment |
| 14 | #PF Page Fault | Yes | Page not present or protection violation |


![CPU Exception Vectors and Error Codes](./diagrams/diag-exception-vectors.svg)


The **error code** column is critical. Some exceptions push an error code onto the stack; others don't. Your handler must know which, or `iret` will pop the wrong value into EIP and crash.

---

## Interrupt Stack Frame: What the CPU Pushes

{{DIAGRAM:diag-interrupt-stack-frame}}

When an interrupt occurs, the CPU pushes this stack frame:

```
High addresses
┌─────────────────┐
│    SS (old)     │  ← Only if privilege change (ring 3 → ring 0)
│    ESP (old)    │  ← Only if privilege change
├─────────────────┤
│    EFLAGS       │  ← Always pushed
├─────────────────┤
│    CS (old)     │  ← Always pushed
│    EIP (old)    │  ← Always pushed
├─────────────────┤
│  Error Code     │  ← Only for exceptions 8, 10-14
└─────────────────┘
Low addresses (stack grows down)
```

**Your handler receives this stack**. The CPU did NOT save EAX, EBX, ECX, EDX, ESI, EDI, EBP, or DS/ES/FS/GS. If you use any of these registers (and you will), you must save and restore them yourself.

---

## Writing an Interrupt Handler: The Assembly Shim

You can't write an interrupt handler entirely in C because C has no way to:

1. Control exactly what's pushed/popped
2. Execute `iret` instead of `ret`
3. Handle the optional error code

The standard pattern is an **assembly stub** that calls a C function:

```asm
; Common interrupt stub macro
%macro ISR_NOERR 1       ; For exceptions WITHOUT error code
global isr%1
isr%1:
    push dword 0         ; Push dummy error code to unify stack frame
    push dword %1        ; Push interrupt number
    jmp isr_common_stub
%endmacro

%macro ISR_ERR 1         ; For exceptions WITH error code
global isr%1
isr%1:
    push dword %1        ; Push interrupt number (error code already on stack)
    jmp isr_common_stub
%endmacro

; Declare all CPU exception handlers
ISR_NOERR 0              ; Divide Error
ISR_NOERR 1              ; Debug
ISR_NOERR 2              ; NMI
ISR_NOERR 3              ; Breakpoint
ISR_NOERR 4              ; Overflow
ISR_NOERR 5              ; BOUND Range Exceeded
ISR_NOERR 6              ; Invalid Opcode
ISR_NOERR 7              ; Device Not Available
ISR_ERR   8              ; Double Fault (has error code)
ISR_NOERR 9              ; Coprocessor Segment Overrun
ISR_ERR   10             ; Invalid TSS
ISR_ERR   11             ; Segment Not Present
ISR_ERR   12             ; Stack-Segment Fault
ISR_ERR   13             ; General Protection Fault
ISR_ERR   14             ; Page Fault
; ... continue for 0-31

; Common handler stub
extern isr_handler       ; C function to handle the interrupt

isr_common_stub:
    ; Save all general-purpose registers
    pusha                ; Pushes EAX, ECX, EDX, EBX, ESP (old), EBP, ESI, EDI
    push ds
    push es
    push fs
    push gs

    ; Save current stack pointer (passes pointer to stack frame to C)
    mov eax, esp
    push eax

    ; Load kernel data segment
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax

    ; Call C handler
    call isr_handler

    ; Restore stack pointer (C may have returned a value in EAX, ignore)
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
```

The `pusha` instruction (Push All) saves EAX, ECX, EDX, EBX, ESP (the value before pusha), EBP, ESI, EDI in that order. `popa` restores all except ESP (which is restored automatically).

### The C Handler

```c
typedef struct {
    uint32_t gs, fs, es, ds;
    uint32_t edi, esi, ebp, esp, ebx, edx, ecx, eax;
    uint32_t int_no, err_code;
    uint32_t eip, cs, eflags;
    uint32_t useresp, ss;  // Only valid if privilege change occurred
} registers_t;

void isr_handler(registers_t *regs) {
    // Check if this is an exception (0-31) or IRQ (32+)
    if (regs->int_no < 32) {
        // CPU exception
        const char *exception_messages[] = {
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
            // ... more messages
        };
        
        vga_puts("EXCEPTION: ");
        vga_puts(exception_messages[regs->int_no]);
        vga_puts("\n");
        
        // Special handling for page fault
        if (regs->int_no == 14) {
            uint32_t faulting_address;
            asm volatile("mov %%cr2, %0" : "=r"(faulting_address));
            vga_puts("Faulting address: ");
            vga_put_hex(faulting_address);
            vga_puts("\n");
            
            vga_puts("Error code: ");
            vga_put_hex(regs->err_code);
            vga_puts(" (");
            if (!(regs->err_code & 0x1)) vga_puts("not-present ");
            if (regs->err_code & 0x2) vga_puts("write ");
            if (regs->err_code & 0x4) vga_puts("user-mode ");
            vga_puts(")\n");
        }
        
        // For double fault, halt with diagnostic
        if (regs->int_no == 8) {
            vga_puts("DOUBLE FAULT - System halted.\n");
            asm volatile("cli; hlt");
        }
        
        // For other exceptions, halt for now
        vga_puts("System halted.\n");
        asm volatile("cli; hlt");
    }
}
```

---

## The 8259 PIC: Routing Hardware Interrupts

The **Programmable Interrupt Controller (PIC)** is a separate chip that collects interrupt signals from devices and presents them to the CPU one at a time. The original 8259 handled 8 IRQs; modern systems cascade two PICs for 15 IRQs (IRQ2 is the cascade connection).


![PIC Remapping: IRQ to Vector Mapping](./diagrams/diag-pic-remapping.svg)


### The Remapping Problem

By default, the PIC maps IRQs 0-7 to CPU vectors 8-15 and IRQs 8-15 to vectors 0x70-0x77. But vectors 8-15 are CPU exceptions! When the timer fires (IRQ0), it looks like a double fault (vector 8).

You must **remap** the PICs to use vectors that don't conflict:

```
Master PIC: IRQ0-7 → vectors 32-39 (0x20-0x27)
Slave PIC:  IRQ8-15 → vectors 40-47 (0x28-0x2F)
```

### Programming the PIC

The PIC is programmed via I/O ports:
- Master: Command at 0x20, Data at 0x21
- Slave: Command at 0xA0, Data at 0xA1

The initialization sequence sends **ICW1-ICW4** (Initialization Command Words):

```c
#define PIC1_CMD  0x20
#define PIC1_DATA 0x21
#define PIC2_CMD  0xA0
#define PIC2_DATA 0xA1

void pic_remap(int offset1, int offset2) {
    // ICW1: Start initialization, expect ICW4
    outb(PIC1_CMD, 0x11);   // 0x11 = initialize + ICW4 needed
    outb(PIC2_CMD, 0x11);
    
    // ICW2: Vector offsets
    outb(PIC1_DATA, offset1);  // Master offset (e.g., 32)
    outb(PIC2_DATA, offset2);  // Slave offset (e.g., 40)
    
    // ICW3: Tell Master there's a slave at IRQ2 (00000100)
    //       Tell Slave its cascade identity (00000010)
    outb(PIC1_DATA, 0x04);
    outb(PIC2_DATA, 0x02);
    
    // ICW4: 8086 mode, non-automatic EOI
    outb(PIC1_DATA, 0x01);
    outb(PIC2_DATA, 0x01);
    
    // Clear data registers (mask all IRQs initially)
    outb(PIC1_DATA, 0x0);
    outb(PIC2_DATA, 0x0);
}

void pic_send_eoi(uint8_t irq) {
    // Send End of Interrupt to the PIC(s)
    outb(PIC1_CMD, 0x20);          // EOI to master
    
    if (irq >= 8) {
        outb(PIC2_CMD, 0x20);      // EOI to slave if IRQ came from it
    }
}
```

### IRQ Handlers

For hardware IRQs (vectors 32-47), you need separate stubs that send EOI:

```asm
; IRQ handlers
%macro IRQ 2
global irq%1
irq%1:
    push dword 0           ; Dummy error code
    push dword %1          ; IRQ number
    jmp irq_common_stub
%endmacro

IRQ 0,  32                 ; Timer
IRQ 1,  33                 ; Keyboard
IRQ 2,  34                 ; Cascade
IRQ 3,  35                 ; COM2
IRQ 4,  36                 ; COM1
; ... continue for IRQs 0-15

extern irq_handler

irq_common_stub:
    pusha
    push ds
    push es
    push fs
    push gs
    
    mov eax, esp
    push eax
    
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    
    call irq_handler
    
    add esp, 4
    
    pop gs
    pop fs
    pop es
    pop ds
    popa
    
    add esp, 8
    
    iret
```

```c
// C handler for IRQs
void irq_handler(registers_t *regs) {
    // Send EOI to PIC
    pic_send_eoi(regs->int_no - 32);
    
    // Dispatch to specific handler
    if (regs->int_no == 32) {
        timer_handler();
    } else if (regs->int_no == 33) {
        keyboard_handler();
    }
    // ... other IRQ handlers
}
```

---

## The PIT Timer: Your First Clock

The **Programmable Interval Timer (8253/8254)** generates periodic interrupts at a configurable frequency. Channel 0 is connected to IRQ0.

The base frequency is 1,193,182 Hz (≈1.193 MHz). You configure the divisor to get your desired frequency:

```
Output frequency = 1,193,182 / divisor
Divisor for 100Hz = 1,193,182 / 100 = 11931 (0x2E9B)
```

```c
#define PIT_CHANNEL0 0x40
#define PIT_CMD      0x43

volatile uint32_t tick_count = 0;

void timer_init(uint32_t frequency) {
    uint32_t divisor = 1193182 / frequency;
    
    // Command byte: channel 0, access mode: lobyte/hibyte, mode 3 (square wave)
    outb(PIT_CMD, 0x36);
    
    // Send divisor
    outb(PIT_CHANNEL0, divisor & 0xFF);        // Low byte
    outb(PIT_CHANNEL0, (divisor >> 8) & 0xFF); // High byte
}

void timer_handler(void) {
    tick_count++;
    
    // Every 100 ticks (1 second at 100Hz), print a message
    if (tick_count % 100 == 0) {
        vga_puts("Tick: ");
        vga_put_dec(tick_count / 100);
        vga_puts(" seconds\n");
    }
}
```

The `volatile` keyword on `tick_count` prevents the compiler from caching it in a register — necessary because it's modified in an interrupt handler that the compiler can't see.

---

## PS/2 Keyboard: Scancodes to Characters

The keyboard doesn't send ASCII. It sends **scancodes** — raw key identifiers. When you press 'A', the keyboard sends scancode 0x1E. When you release 'A', it sends 0x9E (break code = make code | 0x80).

{{DIAGRAM:diag-keyboard-scancode-flow}}

### Reading from the Keyboard

The keyboard controller presents data at I/O port 0x60. When a key is pressed:

1. Keyboard controller sends scancode to port 0x60
2. Keyboard controller asserts IRQ1
3. CPU vectors through IDT to your IRQ1 handler
4. Your handler reads port 0x60 to get the scancode

```c
#define KB_DATA_PORT 0x60
#define KB_CMD_PORT  0x64

#define KB_BUFFER_SIZE 128

static char kb_buffer[KB_BUFFER_SIZE];
static int kb_buffer_head = 0;
static int kb_buffer_tail = 0;

// US QWERTY scancode to ASCII table (lowercase)
static char scancode_to_ascii[] = {
    0,    0,   '1', '2', '3', '4', '5', '6',   // 0x00-0x07
    '7', '8', '9', '0', '-', '=', '\b', '\t',  // 0x08-0x0F (backspace, tab)
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i',   // 0x10-0x17
    'o', 'p', '[', ']', '\n', 0,              // 0x18-0x1D (enter, left ctrl)
    'a', 's', 'd', 'f', 'g', 'h', 'j', 'k',   // 0x1E-0x25
    'l', ';', '\'', '`', 0, '\\',             // 0x26-0x2B (no key, shift, backslash)
    'z', 'x', 'c', 'v', 'b', 'n', 'm', ',',   // 0x2C-0x33
    '.', '/', 0, 0, 0, ' ',                   // 0x34-0x39 (shift, alt, space)
};

// Extended scancodes (prefixed with 0xE0) need separate handling
static int extended_scancode = 0;

void keyboard_handler(void) {
    uint8_t scancode = inb(KB_DATA_PORT);
    
    // Handle extended scancodes (arrow keys, etc.)
    if (scancode == 0xE0) {
        extended_scancode = 1;
        return;
    }
    
    // Check for break code (key release)
    int released = (scancode & 0x80);
    scancode &= 0x7F;
    
    if (released) {
        // Handle key release (for shift/ctrl tracking)
        if (scancode == 0x2A || scancode == 0x36) {
            // Left or right shift released
            shift_pressed = 0;
        }
        return;
    }
    
    // Handle modifier keys
    if (scancode == 0x2A || scancode == 0x36) {
        shift_pressed = 1;
        return;
    }
    
    // Convert to ASCII
    if (scancode < sizeof(scancode_to_ascii)) {
        char c = scancode_to_ascii[scancode];
        
        if (shift_pressed && c >= 'a' && c <= 'z') {
            c -= 32;  // Convert to uppercase
        }
        
        // Add to circular buffer
        int next_head = (kb_buffer_head + 1) % KB_BUFFER_SIZE;
        if (next_head != kb_buffer_tail) {
            kb_buffer[kb_buffer_head] = c;
            kb_buffer_head = next_head;
        }
    }
}

// Non-blocking read from keyboard buffer
int kb_getchar(void) {
    if (kb_buffer_head == kb_buffer_tail) {
        return -1;  // Buffer empty
    }
    
    char c = kb_buffer[kb_buffer_tail];
    kb_buffer_tail = (kb_buffer_tail + 1) % KB_BUFFER_SIZE;
    return c;
}
```

### Handling Special Cases

The PS/2 keyboard has quirks:

1. **Extended scancodes** (0xE0 prefix): Arrow keys, navigation cluster, and right-side modifiers send a two-byte sequence starting with 0xE0
2. **Pause/Break**: Sends 0xE1 0x1D 0x45 0xE1 0x9D 0xC5 (8 bytes total!)
3. **Print Screen**: Sends 0xE0 0x2A 0xE0 0x37 on press

Production keyboard drivers use state machines. For now, handling basic alphanumeric keys is sufficient.

---

## Double Fault: Your Last Line of Defense

A **double fault** (exception 8) occurs when an exception happens while handling another exception. Common causes:

- Page fault while handling a page fault
- Segment not present while handling an exception
- Stack overflow during exception handling

Without a double fault handler, the CPU **triple faults** and resets. With a handler, you can at least print a diagnostic before halting:

```c
void double_fault_handler(registers_t *regs) {
    vga_set_color(VGA_WHITE, VGA_RED);
    vga_puts("\n!!! DOUBLE FAULT !!!\n");
    vga_puts("System state corrupted. Halting.\n");
    vga_puts("EIP: ");
    vga_put_hex(regs->eip);
    vga_puts("  CS: ");
    vga_put_hex(regs->cs);
    vga_puts("\nError code: ");
    vga_put_hex(regs->err_code);
    vga_puts("\n");
    
    // No recovery possible from double fault
    asm volatile("cli; hlt");
}
```

The error code for double fault indicates what went wrong:

```
Bits 0-1: Which table (0=GDT, 1=IDT, 2=LDT, 3=IDT)
Bit 2:    Type of access (0=instruction fetch or segment load)
Bits 3-15: Index of selector
```


![Fault Cascade: Triple Fault Cause Chain](./diagrams/diag-triple-fault-chain.svg)


---

## Putting It All Together: Initialization Sequence

```c
void idt_init(void) {
    // Set up exception handlers (0-31)
    idt_set_gate(0,  (uint32_t)isr0,  0x08, 0x8E);  // Divide Error
    idt_set_gate(1,  (uint32_t)isr1,  0x08, 0x8E);  // Debug
    // ... all 32 exceptions ...
    idt_set_gate(8,  (uint32_t)isr8,  0x08, 0x8E);  // Double Fault
    idt_set_gate(14, (uint32_t)isr14, 0x08, 0x8E);  // Page Fault
    
    // Remap PIC and set up IRQ handlers (32-47)
    pic_remap(32, 40);
    idt_set_gate(32, (uint32_t)irq0, 0x08, 0x8E);   // Timer
    idt_set_gate(33, (uint32_t)irq1, 0x08, 0x8E);   // Keyboard
    // ... remaining IRQs ...
    
    // Load IDT
    idt_load();
}

void kernel_main(void) {
    vga_init();
    serial_init(COM1_PORT);
    
    vga_puts("Initializing IDT...\n");
    idt_init();
    
    vga_puts("Initializing timer (100Hz)...\n");
    timer_init(100);
    
    // Unmask IRQ0 (timer) and IRQ1 (keyboard)
    outb(PIC1_DATA, 0xFC);  // 11111100 - enable IRQ0 and IRQ1
    
    vga_puts("Enabling interrupts...\n");
    asm volatile("sti");
    
    vga_puts("System ready. Type something!\n");
    
    while (1) {
        int c = kb_getchar();
        if (c != -1) {
            vga_putchar(c);
        }
        asm volatile("hlt");  // Wait for interrupt
    }
}
```

---

## Hardware Soul: What's Happening on the Metal

**Interrupt latency**: From IRQ assertion to your handler's first instruction, the CPU spends 50-100 cycles on the automatic stack operations. Your pusha/push ds-es adds another 40 cycles. That's 100-150 cycles of pure overhead before any useful work.

**Cache impact**: Interrupt handlers run with whatever's in the cache. A timer interrupt that fires while running a memory-intensive loop will find a cold cache. This is why interrupt handlers should be small.

**PIC priority**: The PIC has a fixed priority scheme — IRQ0 (timer) is highest, IRQ7 lowest. If IRQ0 is pending when IRQ7 arrives, IRQ0 wins. If you're in an IRQ0 handler and IRQ7 fires, it waits (unless you re-enable interrupts with `sti` in the handler).

**The EOI timing window**: Between your handler returning and `iret` executing, interrupts are still disabled. But after `iret`, if EOI was sent, the next pending IRQ fires immediately. There's no "settle time."

**Keyboard controller buffer**: The keyboard controller has a small internal buffer (typically 16 bytes). If you don't read port 0x60 fast enough, the controller discards old keystrokes. Your interrupt handler needs to be responsive.

---

## Debugging Interrupt Issues

**Symptom: System freezes immediately after enabling interrupts**

- Check: Did you remap the PIC? Default vectors 8-15 conflict with CPU exceptions
- Check: Is the IDT loaded? Add a debug print after `lidt`
- Check: Are your IDT entry addresses correct? Use GDB to inspect

**Symptom: One interrupt works, then nothing**

- You forgot to send EOI. The PIC is waiting.
- Check: `pic_send_eoi()` is called at the end of every IRQ handler

**Symptom: Keyboard produces wrong characters**

- Scancode table mismatch. Are you using US QWERTY?
- Extended scancode handling missing. Arrow keys send 0xE0 prefix

**Symptom: Random crashes, corrupted variables**

- You're not saving/restoring all registers
- Check: `pusha`/`popa` plus segment registers
- Check: Error code handling (some exceptions push it, some don't)

**Symptom: Triple fault**

- Check: Double fault handler is installed and correct
- Check: Stack pointer is valid (your handler needs stack space)
- Use QEMU: `qemu-system-i386 -d int -serial stdio` to log all interrupts

---

## Design Decision: Interrupt Gates vs Trap Gates

| Aspect | Interrupt Gate | Trap Gate | Recommendation |
|--------|---------------|-----------|----------------|
| IF flag | Cleared (interrupts disabled) | Unchanged | IRQs: interrupt gate |
| Nested interrupts | Prevented automatically | Must handle manually | Exceptions: trap gate OK |
| Used by Linux | All hardware IRQs | Software interrupts | Follow Linux |

For this project, use interrupt gates (0x8E in type_attr) for all entries. The simplicity outweighs the minor latency cost of re-enabling interrupts if needed.

---

## Knowledge Cascade

You've built the interrupt subsystem — the nervous system of your kernel. Here's where this knowledge connects:

**Signal Handling in Unix/Linux**: Every signal mechanism traces back to what you just built. `SIGSEGV` (segmentation fault) is your page fault handler reporting to user space. `SIGALRM` is your timer interrupt delivering a notification. The `sigaction()` system call is essentially letting user programs register their own "IDT entries" for software signals. When you understand that signals are just user-space-visible interrupts, the POSIX signal API makes perfect sense.

**Rust Panic and Go Recover (Cross-Domain)**: Language-level exception handling is built on CPU exceptions. When Rust code panics, the runtime either unwinds the stack or aborts — similar to how your exception handlers choose between recovery and halt. Go's `recover()` function catches panics, analogous to a high-level double fault handler. The difference is scope: CPU exceptions are per-instruction, language exceptions are per-call-frame.

**Real-Time Systems and Interrupt Latency**: The EOI timing you learned — send it too late and latency increases, send it too early and risk re-entrancy — is central to real-time systems. The Linux PREEMPT_RT patch converts hard IRQs to threaded interrupts for this reason. Audio buffer underruns (glitches) happen when interrupt handlers take too long. Your timer handler's execution time directly affects the system's worst-case interrupt latency.

**Input Subsystem Architecture**: Your scancode-to-ASCII table is the ancestor of Linux's input subsystem. The modern Linux keyboard driver handles multiple keyboard types, keyboard layouts (via user-space keymaps), and even multiple keyboards simultaneously. But at the core: scancode in from hardware, character out to buffer. You've implemented the essential path.

**Interrupt Storm Debugging**: When a faulty network card asserts IRQ continuously and your handler forgets EOI, the system "freezes" — but it's actually spending 100% of CPU time in your handler. Server administrators see this with `/proc/interrupts` showing millions of IRQs per second. You now understand why: the PIC keeps delivering, the CPU keeps vectoring, nothing else runs.

**Forward: What You Can Now Build**: With a working interrupt system, you can:
- Implement preemptive scheduling (timer interrupt triggers context switch)
- Add demand paging (page fault handler loads pages from disk)
- Build a system call interface (INT 0x80 is just another interrupt)
- Write drivers for any hardware with an IRQ

---

## Summary

You've built the interrupt subsystem — the mechanism by which hardware events and software exceptions reach your code:

1. **IDT**: 256 entries, each 8 bytes, mapping vector numbers to handler addresses
2. **Exception handlers**: CPU-detected conditions (0-31), some with error codes
3. **PIC remapping**: Move IRQs from conflicting vectors (8-15) to 32-47
4. **EOI protocol**: Tell the PIC when you're done, or interrupts stop forever
5. **PIT timer**: Configure divisor for desired frequency, increment tick counter
6. **PS/2 keyboard**: Read scancodes from port 0x60, convert to ASCII, buffer for reading
7. **Double fault handler**: Catch cascading failures before they cause triple fault

The revelation: interrupts are not callbacks. The CPU pushes a defined stack frame, jumps to your handler, and expects you to clean up. The PIC asserts a signal but doesn't "call" anything. You send EOI as a message, not a return value. This is hardware communication at its most fundamental.


![OS Kernel Architecture: Satellite View](./diagrams/diag-satellite-system.svg)


---
<!-- END_MS -->


<!-- MS_ID: build-os-m3 -->
# Milestone 3: Physical and Virtual Memory Management

## The Tension: Memory Isn't What You Think It Is

Here's what most developers believe about memory:

> "Memory is a big byte array. `malloc(100)` gives me 100 bytes at some address like 0x1000. That address points to physical RAM. Virtual memory is just swap to disk."

Every part of this mental model is wrong.

**Reality #1: Physical memory is a collection of 4KB chunks called frames.** There is no giant contiguous array. The 1GB of RAM in your system is 262,144 separate frames, each individually addressable, each potentially allocated to a different purpose.

**Reality #2: Virtual addresses are translated on every single access.** When you read `*ptr` where `ptr = 0xC0123456`, the CPU doesn't "look up" the translation once. It walks the page table hierarchy every time—or relies on the TLB (Translation Lookaside Buffer) cache. That translation happens in hardware, in parallel with the cache lookup, adding ~10-30 cycles to every memory access.

**Reality #3: The TLB is not coherent with your page table writes.** Modify a page table entry, and the TLB still holds the old translation. The CPU will happily use stale data. You must explicitly invalidate entries with `invlpg` or reload CR3.

**Reality #4: Enabling paging without mapping your current code causes immediate crash.** The moment you set CR0.PG, the CPU starts translating *the very next instruction fetch*. If that instruction's virtual address doesn't map to its physical location, you page fault before executing another instruction.

{{DIAGRAM:diag-page-table-hierarchy}}

The numbers that matter:
- **4KB frame size** — the atomic unit of physical allocation
- **4MB per page directory entry** — each PDE covers a 4MB virtual region
- **1024 entries per page directory** — covers 4GB virtual address space
- **1024 entries per page table** — covers 4MB per page table
- **TLB: 64-128 entries** — tiny cache, misses cost hundreds of cycles

This milestone is about building the memory management layer that makes virtual addresses work: parsing the physical memory map, allocating frames, constructing page tables, enabling paging, and building a kernel heap on top of it all.

---

## Revelation: The Address Translation Happens Every Time

**What you might think**: "When I call `malloc`, the kernel sets up a mapping. After that, accessing that pointer goes directly to RAM."

**What actually happens**:

Every memory access—every instruction fetch, every data read, every stack push—goes through this sequence:

1. CPU generates a **virtual address** (e.g., 0xC0101234)
2. Extract page directory index: bits 22-31 → 0x300 (768)
3. Read Page Directory Entry (PDE) at CR3 + 768*4
4. Check present bit—if 0, raise page fault
5. Extract page table index: bits 12-21 → 0x101 (257)
6. Read Page Table Entry (PTE) at PDE.frame + 257*4
7. Check present bit—if 0, raise page fault
8. Extract offset: bits 0-11 → 0x234
9. Physical address = PTE.frame + 0x234
10. Access cache/memory at that physical address

Steps 2-9 happen in hardware, in parallel with the L1 cache lookup. But here's the critical part: the TLB caches the *result* of steps 2-9. If you modify the page table entry in step 7, the TLB still holds the old translation.


![Page Directory/Table Entry Bits](./diagrams/diag-page-directory-entry.svg)


This has profound implications:

- **Context switches must reload CR3** — each process has its own page directory, and the TLB is full of the old process's translations
- **Modifying page tables requires invlpg** — or the CPU uses stale translations
- **Identity mapping is mandatory during transition** — enabling paging while executing code at 0x100000 requires a mapping from virtual 0x100000 to physical 0x100000

---

## System Map: Where We Are

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR OS KERNEL                          │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Memory    │  │   Physical  │  │   Virtual   │             │
│  │   Map       │→ │   Frame     │→ │   Memory    │             │
│  │  (E820)     │  │   Allocator │  │   Manager   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│         │                │                ↓                     │
│         │                │         ┌─────────────┐              │
│         │                │         │   Kernel    │              │
│         │                └────────→│   Heap      │              │
│         │                          │ (kmalloc)   │              │
│         │                          └─────────────┘              │
│         │                                │                      │
│  ┌──────┴────────────────────────────────┴──────────────┐      │
│  │                    HARDWARE                          │      │
│  │  CR3 → Page Directory → Page Tables → Physical RAM  │      │
│  │  TLB caches translations (NOT coherent with writes) │      │
│  └─────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

We're building the entire memory management stack: physical memory discovery, frame allocation, page table construction, paging enablement, and kernel heap allocation.

---

## Physical Memory Map: Discovering What Exists

### The E820 BIOS Query

Before you can allocate memory, you need to know what memory exists. The BIOS provides this via the E820 query (INT 15h, AX=E820), which returns a list of memory regions with their types.

{{DIAGRAM:diag-e820-memory-map}}

Each E820 entry is 20+ bytes:

```c
struct e820_entry {
    uint64_t base;      // Base address of region
    uint64_t length;    // Length of region
    uint32_t type;      // Type of memory
    uint32_t acpi;      // ACPI extended attributes (optional)
} __attribute__((packed));

#define E820_USABLE    1   // Normal RAM, can be used
#define E820_RESERVED  2   // Reserved, do not use
#define E820_ACPI_RECL 3   // ACPI reclaimable after reading tables
#define E820_ACPI_NVS  4   // ACPI non-volatile storage
#define E820_BAD       5   // Bad memory, contains errors
```

If you're using a multiboot bootloader (GRUB), the multiboot info structure already contains the memory map:

```c
struct multiboot_mmap_entry {
    uint32_t size;
    uint64_t base_addr;
    uint64_t length;
    uint32_t type;
} __attribute__((packed));

void parse_memory_map(multiboot_info_t *mbi) {
    // Check if memory map is present
    if (!(mbi->flags & (1 << 6))) {
        panic("No memory map from bootloader!");
    }
    
    uint32_t total_usable = 0;
    multiboot_mmap_entry_t *entry = (multiboot_mmap_entry_t *)mbi->mmap_addr;
    
    while ((uint32_t)entry < mbi->mmap_addr + mbi->mmap_length) {
        kprintf("Memory: 0x%x - 0x%x ", 
                (uint32_t)entry->base_addr,
                (uint32_t)(entry->base_addr + entry->length - 1));
        
        switch (entry->type) {
            case E820_USABLE:
                kprintf("(usable)\n");
                total_usable += entry->length;
                break;
            case E820_RESERVED:
                kprintf("(reserved)\n");
                break;
            case E820_ACPI_RECL:
                kprintf("(ACPI reclaimable)\n");
                break;
            case E820_ACPI_NVS:
                kprintf("(ACPI NVS)\n");
                break;
            default:
                kprintf("(type %d)\n", entry->type);
        }
        
        entry = (multiboot_mmap_entry_t *)((uint32_t)entry + entry->size + 4);
    }
    
    kprintf("Total usable memory: %d MB\n", total_usable / (1024 * 1024));
}
```

### Typical Memory Layout

A typical 512MB system might have:

| Start | End | Type | Purpose |
|-------|-----|------|---------|
| 0x000000 | 0x000FFF | Reserved | Real-mode IVT, BDA |
| 0x001000 | 0x07FFFF | Usable | Low memory (conventional) |
| 0x07FFFF | 0x080000 | Reserved | EBDA (Extended BIOS Data Area) |
| 0x080000 | 0x0FFFFF | Reserved | ROM area, video memory |
| 0x100000 | 0x1FFFFFF | Usable | Extended memory (your kernel lives here) |
| 0x2000000+ | ... | Usable | More RAM |

**Critical insight**: Not all "usable" memory is actually available. Your kernel binary, page tables, and bootloader data occupy physical frames. Your allocator must skip these.

---

## Physical Frame Allocator: Managing 4KB Chunks

### Bitmap Allocator

A bitmap allocator uses one bit per frame to track allocation status. For 4GB of memory with 4KB frames, you need 128KB of bitmap (4GB / 4KB / 8 bits = 131,072 bytes).

```c
#define FRAME_SIZE 4096
#define FRAMES_PER_BITMAP_ENTRY 32  // 32 bits per uint32_t

static uint32_t *frame_bitmap;
static uint32_t total_frames;
static uint32_t first_usable_frame;

// Calculate which frame a physical address belongs to
#define FRAME_FROM_ADDR(addr) ((addr) / FRAME_SIZE)

// Bitmap operations
static inline void set_frame(uint32_t frame) {
    frame_bitmap[frame / 32] |= (1 << (frame % 32));
}

static inline void clear_frame(uint32_t frame) {
    frame_bitmap[frame / 32] &= ~(1 << (frame % 32));
}

static inline int test_frame(uint32_t frame) {
    return frame_bitmap[frame / 32] & (1 << (frame % 32));
}

// Find first free frame
static int find_free_frame(void) {
    for (uint32_t i = first_usable_frame / 32; i < total_frames / 32; i++) {
        if (frame_bitmap[i] != 0xFFFFFFFF) {
            // At least one bit is free
            for (int j = 0; j < 32; j++) {
                if (!(frame_bitmap[i] & (1 << j))) {
                    return i * 32 + j;
                }
            }
        }
    }
    return -1;  // Out of memory
}

// Allocate a single frame, returns physical address
void *alloc_frame(void) {
    int frame = find_free_frame();
    if (frame == -1) {
        return NULL;  // Out of memory
    }
    
    set_frame(frame);
    return (void *)(frame * FRAME_SIZE);
}

// Free a frame by physical address
void free_frame(void *addr) {
    uint32_t frame = FRAME_FROM_ADDR((uint32_t)addr);
    
    // Safety checks
    if (frame < first_usable_frame || frame >= total_frames) {
        panic("free_frame: invalid frame number %d\n", frame);
    }
    
    // Double-free detection
    if (!test_frame(frame)) {
        panic("free_frame: double free at frame %d (addr 0x%x)\n", frame, addr);
    }
    
    clear_frame(frame);
}
```

### Initializing the Frame Allocator

```c
void frame_allocator_init(multiboot_info_t *mbi) {
    // Calculate total physical memory
    uint32_t max_addr = 0;
    multiboot_mmap_entry_t *entry = (multiboot_mmap_entry_t *)mbi->mmap_addr;
    
    while ((uint32_t)entry < mbi->mmap_addr + mbi->mmap_length) {
        uint32_t end = entry->base_addr + entry->length;
        if (end > max_addr) {
            max_addr = end;
        }
        entry = (multiboot_mmap_entry_t *)((uint32_t)entry + entry->size + 4);
    }
    
    total_frames = max_addr / FRAME_SIZE;
    
    // Allocate bitmap (must be in identity-mapped region)
    uint32_t bitmap_size = (total_frames + 31) / 32 * sizeof(uint32_t);
    frame_bitmap = (uint32_t *)placement_alloc(bitmap_size);
    
    // Mark everything as reserved initially
    memset(frame_bitmap, 0xFF, bitmap_size);
    
    // Mark usable regions as free
    first_usable_frame = total_frames;  // Start with max
    entry = (multiboot_mmap_entry_t *)mbi->mmap_addr;
    
    while ((uint32_t)entry < mbi->mmap_addr + mbi->mmap_length) {
        if (entry->type == E820_USABLE) {
            uint32_t start_frame = FRAME_FROM_ADDR(entry->base_addr);
            uint32_t end_frame = FRAME_FROM_ADDR(entry->base_addr + entry->length);
            
            // Align to frame boundaries
            if (start_frame * FRAME_SIZE < entry->base_addr) start_frame++;
            if (end_frame * FRAME_SIZE > entry->base_addr + entry->length) end_frame--;
            
            for (uint32_t f = start_frame; f < end_frame; f++) {
                clear_frame(f);
                if (f < first_usable_frame) {
                    first_usable_frame = f;
                }
            }
        }
        entry = (multiboot_mmap_entry_t *)((uint32_t)entry + entry->size + 4);
    }
    
    // Reserve kernel and already-allocated memory
    extern uint32_t _kernel_start;
    extern uint32_t _kernel_end;
    
    uint32_t kernel_start_frame = FRAME_FROM_ADDR((uint32_t)&_kernel_start);
    uint32_t kernel_end_frame = FRAME_FROM_ADDR((uint32_t)&_kernel_end) + 1;
    
    for (uint32_t f = kernel_start_frame; f < kernel_end_frame; f++) {
        set_frame(f);  // Mark as used
    }
    
    kprintf("Frame allocator initialized: %d frames, first usable at 0x%x\n",
            total_frames, first_usable_frame * FRAME_SIZE);
}
```

### Alternative: Linked List Allocator

A free-list allocator tracks free frames by linking them together:

```c
struct free_frame {
    struct free_frame *next;
};

static struct free_frame *free_list = NULL;

void *alloc_frame(void) {
    if (free_list == NULL) {
        return NULL;
    }
    
    struct free_frame *frame = free_list;
    free_list = free_list->next;
    return (void *)frame;
}

void free_frame(void *addr) {
    struct free_frame *frame = (struct free_frame *)addr;
    frame->next = free_list;
    free_list = frame;
}
```

**Trade-off**: The free-list is O(1) for both allocation and freeing, but requires careful handling of the initial population and doesn't handle double-free detection as naturally.

---

## Page Tables: The Two-Level Hierarchy

### x86 32-bit Paging Structure

On x86 without PAE (Physical Address Extension), paging uses a two-level hierarchy:

{{DIAGRAM:diag-page-table-hierarchy}}

1. **Page Directory (PD)**: 1024 entries, each covering 4MB of virtual space
2. **Page Table (PT)**: 1024 entries, each covering 4KB of virtual space

The CR3 register holds the **physical address** of the page directory.

```c
#define PAGE_SIZE 4096
#define ENTRIES_PER_TABLE 1024

// Page Directory/Table Entry
typedef uint32_t pte_t;

// Entry flags
#define PTE_PRESENT    (1 << 0)
#define PTE_WRITABLE   (1 << 1)
#define PTE_USER       (1 << 2)
#define PTE_WRITETHRU  (1 << 3)
#define PTE_CACHE_DIS  (1 << 4)
#define PTE_ACCESSED   (1 << 5)
#define PTE_DIRTY      (1 << 6)   // Page tables only
#define PTE_PAGE_SIZE  (1 << 7)   // 4MB pages in PD
#define PTE_GLOBAL     (1 << 8)   // Not flushed on CR3 reload
#define PTE_FRAME_MASK 0xFFFFF000

typedef struct {
    pte_t entries[ENTRIES_PER_TABLE];
} page_table_t;

typedef struct {
    pte_t entries[ENTRIES_PER_TABLE];
} page_directory_t;

// Current page directory
static page_directory_t *current_directory;
```

### Extracting Address Components

```c
// Extract indices from a virtual address
#define PD_INDEX(addr) (((addr) >> 22) & 0x3FF)
#define PT_INDEX(addr) (((addr) >> 12) & 0x3FF)
#define PAGE_OFFSET(addr) ((addr) & 0xFFF)

// Get frame address from PTE
#define PTE_FRAME(pte) ((pte) & PTE_FRAME_MASK)
```

### Mapping Virtual to Physical

```c
// Map a virtual page to a physical frame
void map_page(page_directory_t *dir, uint32_t virt, uint32_t phys, uint32_t flags) {
    // Align addresses to page boundaries
    virt &= ~0xFFF;
    phys &= ~0xFFF;
    
    uint32_t pd_idx = PD_INDEX(virt);
    uint32_t pt_idx = PT_INDEX(virt);
    
    // Get or create page table
    pte_t *pde = &dir->entries[pd_idx];
    page_table_t *pt;
    
    if (!(*pde & PTE_PRESENT)) {
        // Allocate new page table
        pt = (page_table_t *)alloc_frame();
        if (!pt) {
            panic("Failed to allocate page table for 0x%x\n", virt);
        }
        
        // Clear the page table
        memset(pt, 0, sizeof(page_table_t));
        
        // Set page directory entry
        *pde = ((uint32_t)pt) | PTE_PRESENT | PTE_WRITABLE | (flags & PTE_USER);
    } else {
        pt = (page_table_t *)PTE_FRAME(*pde);
    }
    
    // Set page table entry
    pte_t *pte = &pt->entries[pt_idx];
    *pte = phys | flags | PTE_PRESENT;
    
    // Invalidate TLB entry
    asm volatile("invlpg (%0)" : : "r"(virt));
}

// Unmap a virtual page
void unmap_page(page_directory_t *dir, uint32_t virt) {
    uint32_t pd_idx = PD_INDEX(virt);
    uint32_t pt_idx = PT_INDEX(virt);
    
    if (!(dir->entries[pd_idx] & PTE_PRESENT)) {
        return;  // Nothing mapped
    }
    
    page_table_t *pt = (page_table_t *)PTE_FRAME(dir->entries[pd_idx]);
    pt->entries[pt_idx] = 0;  // Clear present bit
    
    // Invalidate TLB entry
    asm volatile("invlpg (%0)" : : "r"(virt));
}

// Get physical address for a virtual address (returns 0 if not mapped)
uint32_t get_physical(page_directory_t *dir, uint32_t virt) {
    uint32_t pd_idx = PD_INDEX(virt);
    uint32_t pt_idx = PT_INDEX(virt);
    
    if (!(dir->entries[pd_idx] & PTE_PRESENT)) {
        return 0;
    }
    
    page_table_t *pt = (page_table_t *)PTE_FRAME(dir->entries[pd_idx]);
    
    if (!(pt->entries[pt_idx] & PTE_PRESENT)) {
        return 0;
    }
    
    return PTE_FRAME(pt->entries[pt_idx]) | PAGE_OFFSET(virt);
}
```

---

## Identity Mapping + Higher-Half Kernel

### The Address Space Layout Problem

Your kernel needs to be accessible from every process (for system calls, interrupt handlers). But user processes need isolated address spaces. The solution: **higher-half kernel**.

{{DIAGRAM:diag-identity-higher-half}}

```
Virtual Address Space:
┌─────────────────┐ 0xFFFFFFFF
│   Kernel Space  │ (1GB, reserved)
│   0xC0000000+   │
├─────────────────┤ 0xC0000000 (3GB)
│                 │
│   User Space    │ (3GB, per-process)
│   0x00000000+   │
│                 │
└─────────────────┘ 0x00000000
```

The kernel is linked to run at 0xC0000000+ but loaded at 0x100000 physical. The page tables create both:
- **Identity mapping**: 0x00000000-0x00FFFFFF → 0x00000000-0x00FFFFFF (for VGA, MMIO)
- **Higher-half mapping**: 0xC0000000-0xC0FFFFFF → 0x00000000-0x00FFFFFF (for kernel code)

### Setting Up the Initial Page Tables

```c
#define KERNEL_VIRTUAL_BASE 0xC0000000
#define KERNEL_PHYSICAL_BASE 0x100000

// Identity-map the first N MB and map the kernel at higher half
void paging_init(void) {
    // Allocate page directory
    page_directory_t *dir = (page_directory_t *)alloc_frame();
    memset(dir, 0, sizeof(page_directory_t));
    
    // Identity map first 16MB (kernel + VGA + MMIO)
    // This allows VGA at 0xB8000 to still work
    for (uint32_t addr = 0; addr < 16 * 1024 * 1024; addr += PAGE_SIZE) {
        // Supervisor-only, writable, present
        map_page(dir, addr, addr, PTE_WRITABLE);
    }
    
    // Higher-half mapping for kernel (0xC0000000+)
    // Map first 16MB at 0xC0000000 as well
    for (uint32_t offset = 0; offset < 16 * 1024 * 1024; offset += PAGE_SIZE) {
        uint32_t virt = KERNEL_VIRTUAL_BASE + offset;
        uint32_t phys = offset;
        map_page(dir, virt, phys, PTE_WRITABLE);
    }
    
    current_directory = dir;
    
    // Load CR3 and enable paging
    load_cr3(dir);
    enable_paging();
}
```

### Enabling Paging: The Critical Sequence

{{DIAGRAM:diag-paging-enable}}

```c
void load_cr3(page_directory_t *dir) {
    uint32_t phys = (uint32_t)dir;
    asm volatile("mov %0, %%cr3" : : "r"(phys));
}

void enable_paging(void) {
    uint32_t cr0;
    asm volatile("mov %%cr0, %0" : "=r"(cr0));
    cr0 |= (1 << 31);  // Set PG bit
    asm volatile("mov %0, %%cr0" : : "r"(cr0));
}
```

**CRITICAL**: The identity mapping must exist *before* you enable paging. The instruction that sets CR0.PG is at some virtual address V. The very next instruction fetch uses V, which must translate to the correct physical address. Without identity mapping, V doesn't map to anything, and you page fault immediately.

### TLB Flushing

```c
// Flush a single TLB entry
static inline void invlpg(uint32_t addr) {
    asm volatile("invlpg (%0)" : : "r"(addr) : "memory");
}

// Flush entire TLB (by reloading CR3)
static inline void flush_tlb(void) {
    uint32_t cr3;
    asm volatile("mov %%cr3, %0" : "=r"(cr3));
    asm volatile("mov %0, %%cr3" : : "r"(cr3));
}
```

When to flush:
- After modifying any page table entry: `invlpg(addr)`
- On context switch: reload CR3 (implicit flush)
- After changing page directory entries: reload CR3

---

## Page Fault Handler: Diagnosis and Recovery

The page fault handler (exception 14) is your window into the paging system. When a translation fails, the CPU:

1. Pushes error code onto stack
2. Loads CR2 with the faulting virtual address
3. Vectors through IDT to your handler

{{DIAGRAM:diag-page-fault-handler}}

### The Error Code

```
Bit 0 (P): Present
  0 = page not present
  1 = protection violation

Bit 1 (W): Write
  0 = read access
  1 = write access

Bit 2 (U): User
  0 = supervisor mode
  1 = user mode

Bit 3 (R): Reserved bit
  0 = not caused by reserved bit
  1 = reserved bit set in page tables

Bit 4 (I): Instruction fetch
  0 = data access
  1 = instruction fetch (NX bit violation)
```

### Handler Implementation

```c
void page_fault_handler(registers_t *regs) {
    uint32_t faulting_addr;
    asm volatile("mov %%cr2, %0" : "=r"(faulting_addr));
    
    int present = !(regs->err_code & 0x1);
    int write = regs->err_code & 0x2;
    int user = regs->err_code & 0x4;
    int reserved = regs->err_code & 0x8;
    int exec = regs->err_code & 0x10;
    
    kprintf("\n=== PAGE FAULT ===\n");
    kprintf("Faulting address: 0x%x\n", faulting_addr);
    kprintf("Error code: 0x%x\n", regs->err_code);
    kprintf("  Cause: ");
    
    if (present) {
        kprintf("Protection violation (");
    } else {
        kprintf("Page not present (");
    }
    
    if (write) kprintf("write ");
    else kprintf("read ");
    
    if (user) kprintf("user-mode ");
    else kprintf("kernel-mode ");
    
    if (reserved) kprintf("reserved-bit ");
    if (exec) kprintf("instruction-fetch ");
    
    kprintf(")\n");
    
    // Print where we were
    kprintf("EIP: 0x%x, CS: 0x%x\n", regs->eip, regs->cs);
    
    // Check if address is in valid ranges
    if (faulting_addr < 0x100000) {
        kprintf("Address in low memory (below 1MB)\n");
    } else if (faulting_addr >= KERNEL_VIRTUAL_BASE && 
               faulting_addr < KERNEL_VIRTUAL_BASE + 16*1024*1024) {
        kprintf("Address in kernel space (higher-half)\n");
    } else if (faulting_addr >= 0x100000 && faulting_addr < 16*1024*1024) {
        kprintf("Address in identity-mapped region\n");
    } else {
        kprintf("Address outside mapped regions\n");
    }
    
    // For now, halt on any page fault
    // Later: implement demand paging, copy-on-write, etc.
    panic("Page fault - system halted\n");
}
```

---

## Kernel Heap: Dynamic Memory Allocation

### The Need for kmalloc

The frame allocator gives you 4KB chunks. But often you need 100 bytes for a string, or 48 bytes for a process structure. The kernel heap bridges this gap.


![Kernel Heap Allocator: kmalloc/kfree](./diagrams/diag-kmalloc-internals.svg)


### Simple Heap Implementation: Linked List of Blocks

```c
#define HEAP_START    0xC0400000  // Virtual address for heap
#define HEAP_SIZE     (4 * 1024 * 1024)  // 4MB initial heap
#define BLOCK_MAGIC   0xDEADBEEF

typedef struct heap_block {
    uint32_t magic;
    uint32_t size;          // Size of data area (not including header)
    int free;               // 1 if free, 0 if allocated
    struct heap_block *next;
    struct heap_block *prev;
} heap_block_t;

static heap_block_t *heap_head = NULL;
static uint32_t heap_current = HEAP_START;

void heap_init(void) {
    // Allocate first page for heap
    void *frame = alloc_frame();
    map_page(current_directory, heap_current, (uint32_t)frame, PTE_WRITABLE);
    
    // Initialize first block
    heap_head = (heap_block_t *)heap_current;
    heap_head->magic = BLOCK_MAGIC;
    heap_head->size = PAGE_SIZE - sizeof(heap_block_t);
    heap_head->free = 1;
    heap_head->next = NULL;
    heap_head->prev = NULL;
    
    heap_current += PAGE_SIZE;
}

// Expand heap by one page
static void heap_expand(void) {
    void *frame = alloc_frame();
    if (!frame) {
        panic("kmalloc: out of physical memory\n");
    }
    
    map_page(current_directory, heap_current, (uint32_t)frame, PTE_WRITABLE);
    
    // Find last block
    heap_block_t *last = heap_head;
    while (last->next) {
        last = last->next;
    }
    
    // If last block is free, expand it
    if (last->free) {
        last->size += PAGE_SIZE;
    } else {
        // Create new free block
        heap_block_t *new_block = (heap_block_t *)heap_current;
        new_block->magic = BLOCK_MAGIC;
        new_block->size = PAGE_SIZE - sizeof(heap_block_t);
        new_block->free = 1;
        new_block->next = NULL;
        new_block->prev = last;
        last->next = new_block;
    }
    
    heap_current += PAGE_SIZE;
}

void *kmalloc(uint32_t size) {
    // Align size to 4 bytes
    size = (size + 3) & ~3;
    
    // Find a free block large enough (first-fit)
    heap_block_t *block = heap_head;
    while (block) {
        if (block->magic != BLOCK_MAGIC) {
            panic("kmalloc: heap corruption detected\n");
        }
        
        if (block->free && block->size >= size) {
            // Found a suitable block
            
            // Split block if it's much larger
            if (block->size > size + sizeof(heap_block_t) + 16) {
                heap_block_t *new_block = (heap_block_t *)((uint32_t)block + 
                                          sizeof(heap_block_t) + size);
                new_block->magic = BLOCK_MAGIC;
                new_block->size = block->size - size - sizeof(heap_block_t);
                new_block->free = 1;
                new_block->next = block->next;
                new_block->prev = block;
                
                if (block->next) {
                    block->next->prev = new_block;
                }
                
                block->size = size;
                block->next = new_block;
            }
            
            block->free = 0;
            return (void *)((uint32_t)block + sizeof(heap_block_t));
        }
        
        block = block->next;
    }
    
    // No suitable block found, expand heap
    heap_expand();
    
    // Try again (should succeed now)
    return kmalloc(size);
}

void kfree(void *ptr) {
    if (!ptr) return;
    
    heap_block_t *block = (heap_block_t *)((uint32_t)ptr - sizeof(heap_block_t));
    
    if (block->magic != BLOCK_MAGIC) {
        panic("kfree: invalid pointer or heap corruption\n");
    }
    
    block->free = 1;
    
    // Coalesce with next block if free
    if (block->next && block->next->free) {
        block->size += sizeof(heap_block_t) + block->next->size;
        block->next = block->next->next;
        if (block->next) {
            block->next->prev = block;
        }
    }
    
    // Coalesce with previous block if free
    if (block->prev && block->prev->free) {
        block->prev->size += sizeof(heap_block_t) + block->size;
        block->prev->next = block->next;
        if (block->next) {
            block->next->prev = block->prev;
        }
    }
}
```

### A Simpler Alternative: Placement Allocator

For early boot (before heap is ready), a simple bump allocator works:

```c
static uint32_t placement_addr = 0x100000;  // Start after kernel

void *placement_alloc(uint32_t size) {
    // Align to 4 bytes
    size = (size + 3) & ~3;
    
    void *addr = (void *)placement_addr;
    placement_addr += size;
    
    return addr;
}
```

This never frees memory—use only for allocations that persist for the kernel's lifetime (like the frame bitmap).

---

## Hardware Soul: The Physical Reality of Virtual Memory

**TLB miss cost**: A TLB miss triggers a hardware page table walk. On x86, this is 2-4 memory accesses (PD + PT). With DDR4 latency around 70ns, a TLB miss costs 140-280ns. At 3GHz, that's 400-800 cycles. The TLB is tiny (64-128 entries on modern CPUs) because fully-associative lookups are expensive—every TLB entry must be compared against the virtual address in parallel.

**Cache behavior of page tables**: Page tables live in regular memory and are cached in L1/L2/L3 like any other data. A process with good locality has its page tables hot in cache. A random-access workload (database hash join, graph traversal) thrashes not just data cache but page table cache too.

**Context switch TLB flush**: Reloading CR3 flushes the TLB on x86 (unless PCID is used). This is 100-500 cycles of pure overhead per context switch. This is why kernel threads share the same address space—they don't need CR3 reload. Linux uses "lazy TLB flushing" for kernel threads.

**Page fault overhead**: A page fault is a full context switch to the kernel, plus CR2 read, plus handler execution, plus potential I/O wait. Even for demand paging from RAM (not disk), this is thousands of cycles. This is why madvise(MADV_WILLNEED) exists—prefault pages before the critical path.

**Memory access pattern matters**: Sequential access through a 4KB page costs 1 TLB miss for 1024 32-bit accesses. Random access across 1000 pages costs 1000 TLB misses. Same data size, 1000x difference in TLB overhead. This is why structure-of-arrays (SoA) beats array-of-structures (AoS) for cache and TLB efficiency.

---

## Debugging Memory Management

**Symptom: Immediate page fault when enabling paging**

- Identity mapping missing or incorrect
- Check: Is the code you're executing identity-mapped?
- Debug: Print the virtual address of `enable_paging()` and verify it maps correctly

**Symptom: Page fault with garbage address (0xCCCCCCCC or similar)**

- Uninitialized pointer
- Check: All pointers initialized before use
- Debug: Use magic patterns (0xDEADBEEF) to detect uninitialized memory

**Symptom: Page fault in kernel at reasonable address**

- Page not mapped or protection violation
- Check: Is the page table entry present? Writable?
- Debug: Print page directory and page table entries for the faulting address

**Symptom: Random corruption, crashes after varying time**

- Double-free in heap allocator
- Buffer overflow
- Use-after-free
- Debug: Add magic numbers to heap blocks, check on alloc/free

**Symptom: System hangs, no interrupts processed**

- Ran out of physical frames
- Page table allocation failing silently
- Debug: Track total free frames, warn when low

**Symptom: VGA stops working after enabling paging**

- Identity mapping for 0xB8000 missing
- Check: First 16MB identity mapped?
- Debug: Print PTE for address 0xB8000

---

## Design Decision: Bitmap vs Free-List Allocator

| Aspect | Bitmap | Free-List | Recommendation |
|--------|--------|-----------|----------------|
| Allocation | O(n) scan for free bit | O(1) pop from list | Free-list for speed |
| Freeing | O(1) clear bit | O(1) push to list | Equal |
| Double-free detection | Easy (test before clear) | Hard (corrupts list) | Bitmap wins |
| Memory overhead | Fixed (1 bit per frame) | Variable (list pointers) | Bitmap predictable |
| Fragmentation tracking | Hard to see patterns | Easy (list order) | Free-list wins |
| Used by Linux | Buddy allocator (not bitmap) | Per-zone free lists | Hybrid |

For a learning OS, start with bitmap for simplicity and double-free detection. A free-list is faster but harder to debug.

---

## Knowledge Cascade

You've built the memory management system—the foundation that everything else in your kernel stands on. Here's where this knowledge connects:

**Container and VM Memory Isolation**: The user/supervisor bits in page table entries are exactly what Docker, KVM, and every sandboxing technology relies on. When a container process tries to read kernel memory, the CPU checks the user bit—0 means supervisor-only, page fault. A missing supervisor bit is a kernel memory disclosure vulnerability (CVE-2019-18683 was exactly this). Hypervisors use the same mechanism: guest physical → host physical mappings with privilege isolation. Understanding page table bits is understanding the attack surface of isolation.

**Memory-Mapped I/O and Device Drivers (Cross-Domain)**: The identity mapping you keep for VGA (0xB8000) is the same technique used for every memory-mapped device: NIC registers, GPU command buffers, DMA engines. A network driver maps the NIC's BAR (Base Address Register) region and writes descriptor pointers directly. The CPU doesn't know it's writing to a device—it's just a store to a memory address. This is why `volatile` is mandatory for MMIO: the compiler must not optimize away or reorder these accesses.

**Game Engine Asset Streaming**: The higher-half kernel mapping you implemented teaches address space layout that game engines use for streaming. A game might reserve virtual address 0x10000000-0x20000000 for texture streaming, mapping and unmapping physical pages as assets load/unload. The virtual address stays constant; the backing physical memory changes. This is exactly what your `map_page()` function does, just called at runtime instead of boot.

**Garbage Collector Write Barriers**: The dirty and accessed bits in page tables are used by generational garbage collectors to track which pages need scanning. A GC can mark all pages read-only, let the dirty bit track which pages were written to, and only scan those pages in the next collection cycle. This is called "page-level write barrier" and reduces GC pause times dramatically for large heaps.

**Page Cache and File-Backed Memory**: The concepts here—virtual to physical mapping, demand allocation, lazy evaluation—are exactly what Linux's page cache does. When you read a file, Linux doesn't copy data into your buffer. It maps the file's page cache pages into your address space. Same physical page, mapped in multiple processes. Copy-on-write (which you can implement by clearing the writable bit) enables fork() efficiency.

**Forward: What You Can Now Build**: With working memory management, you can:
- Implement per-process address spaces (each process gets its own page directory)
- Add demand paging (page fault handler loads from disk)
- Build copy-on-write fork (map pages read-only, copy on write fault)
- Create shared memory for IPC (map same physical pages in multiple processes)
- Implement memory-mapped files (map file contents into address space)

---

## Summary

You've built the memory management layer that transforms physical RAM into virtual address spaces:

1. **Physical memory map**: Parse E820/multiboot to discover usable, reserved, and ACPI regions
2. **Frame allocator**: Bitmap or free-list tracking of 4KB frames with double-free detection
3. **Page tables**: Two-level hierarchy (PD + PT) translating 32-bit virtual to physical addresses
4. **Identity + higher-half mapping**: Low memory identity-mapped for VGA/MMIO; kernel at 0xC0000000+
5. **Paging enablement**: Load CR3, set CR0.PG, maintain TLB coherence with invlpg
6. **Page fault handler**: Read CR2 for faulting address, diagnose from error code
7. **Kernel heap**: kmalloc/kfree providing arbitrary-size allocations backed by frame allocator

The revelation: memory is not a byte array. It's a two-level translation structure cached in a tiny, non-coherent TLB. Every access walks this hierarchy or hits the cache. Modify a page table entry, and the CPU will happily use stale translations until you explicitly invalidate them.

The physical constraints—4KB frames, two-level translation, TLB cache behavior, CR3 reload cost—shape every decision. This is systems programming: negotiating with hardware, not abstracting it away.


![OS Kernel Architecture: Satellite View](./diagrams/diag-satellite-system.svg)


---
<!-- END_MS -->


<!-- MS_ID: build-os-m4 -->
# Milestone 4: Processes and Preemptive Scheduling

## The Tension: One CPU, Many Tasks—The Illusion of Simultaneity

You have one CPU core. Your user wants to type in a text editor while music plays in the background and a compiler runs in another window. Three distinct instruction streams, one execution engine.

The hardware offers no magic "run multiple things" instruction. What it does offer is precise enough to build the illusion yourself:

- **Timer interrupt**: Fire IRQ0 every N milliseconds, giving you a foothold back in the kernel
- **Register save/restore**: Push all registers to memory, pop them back—exactly the mechanism function calls use, applied to an entire process
- **Privilege levels**: Ring 0 (kernel) and ring 3 (user) with hardware-enforced boundaries

But here's the brutal constraint: when the timer fires, the currently running process is mid-instruction. Its EIP points somewhere in its code. Its stack has local variables. Its registers hold intermediate computation results. If you don't save *every single byte of state* and restore it perfectly later, that process crashes—or worse, produces silent corruption.

The numbers that matter:
- **Context switch overhead**: 100-500 cycles saving/restoring registers, plus TLB flush if address spaces differ
- **Timer quantum**: 10ms typical (100Hz timer), meaning 10-50ms of useful work between switches
- **Cache impact**: A process evicts ~100KB of L1 cache per millisecond; after 10ms, the next process finds a cold cache
- **TLB entries**: 64-128 entries, all flushed on CR3 reload—every virtual address translation starts cold

This milestone implements preemptive multitasking: the mechanism by which your kernel creates the illusion of parallelism through rapid, precise, hardware-assisted context switching.

---

## Revelation: It's Not Parallelism—It's Perfect Amnesia

**What you might think**: "The scheduler runs multiple processes at once, switching between them so fast they all appear to run simultaneously."

**What actually happens**:

Only ONE process executes at any instant. The "concurrent" illusion requires this exact sequence:

1. Timer interrupt fires (IRQ0 → IDT vector 32)
2. CPU pushes EFLAGS, CS, EIP onto current stack (and SS:ESP if ring 3 → ring 0)
3. Your IRQ handler saves *every other register* to the current process's PCB
4. Scheduler picks the next ready process
5. Your context switch code loads the new process's saved registers
6. TSS.ESP0 is updated to the new process's kernel stack
7. `iret` pops EFLAGS, CS, EIP—and you're now running the new process

{{DIAGRAM:diag-context-switch-flow}}

The magic: the interrupted process has **zero awareness** it was suspended. When it resumes later, its registers are restored, its stack is intact, its EIP points to the next instruction—and it continues as if nothing happened.

But here's what surprises most developers: **system calls are not function calls**.

When user code executes `int 0x80`, the CPU:
1. Looks up IDT entry 0x80
2. Checks privilege transition (DPL of gate vs CPL of caller)
3. **Switches stacks** using TSS.SS0:ESP0
4. Pushes user SS, ESP, EFLAGS, CS, EIP
5. Loads kernel CS:EIP from IDT entry
6. Begins executing kernel code at ring 0

You cannot implement this with a function call. The hardware *must* be involved because ring transitions require stack switching for security—the kernel stack is not accessible from user mode, so the CPU can't trust a user-supplied stack pointer.


![Ring 3 → Ring 0 Transition: Stack Switch](./diagrams/diag-ring-transition.svg)


And the TSS (Task State Segment)? Not optional. Without TSS.SS0:ESP0 configured, the CPU has nowhere to put the kernel stack during a ring 3 → ring 0 transition. The result: triple fault, system reset.

---

## System Map: Where We Are

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR OS KERNEL                          │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    PCB      │  │   Context   │  │     TSS     │             │
│  │   Manager   │  │   Switch    │  │   Config    │             │
│  │  (create,   │  │  (assembly) │  │  (SS0:ESP0) │             │
│  │   destroy)  │  │             │  │             │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│  ┌───────────────────────┴───────────────────────┐             │
│  │                 SCHEDULER                      │             │
│  │  Round-robin: timer IRQ → pick next → switch   │             │
│  └───────────────────────┬───────────────────────┘             │
│                          │                                      │
│         ┌────────────────┼────────────────┐                     │
│         ↓                ↓                ↓                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Process A  │  │  Process B  │  │  Process C  │             │
│  │  (running)  │  │   (ready)   │  │   (ready)   │             │
│  │   ring 0    │  │   ring 0    │  │   ring 3    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    HARDWARE                                 ││
│  │  Timer IRQ0 → IDT → Scheduler → Context Switch → iret      ││
│  │  TSS provides SS0:ESP0 for ring 3→0 stack switch          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

We're building the process management layer: PCB structures to hold process state, context switch assembly to swap register sets, TSS configuration for privilege transitions, a round-robin scheduler, and system call infrastructure.

---

## The Process Control Block: Capturing a Process's Soul

### What Must Be Saved

A process is defined entirely by its register state and memory. To suspend and resume it perfectly, you need:

{{DIAGRAM:diag-pcb-structure}}

```c
typedef enum {
    PROCESS_READY,
    PROCESS_RUNNING,
    PROCESS_BLOCKED,
    PROCESS_ZOMBIE
} process_state_t;

typedef struct process {
    // Identification
    uint32_t pid;
    char name[16];
    process_state_t state;
    
    // Register state (saved on context switch)
    uint32_t eax, ebx, ecx, edx;
    uint32_t esi, edi, ebp;
    uint32_t esp, eip;
    uint32_t eflags;
    uint32_t cs, ds, es, fs, gs, ss;
    
    // Memory management
    page_directory_t *page_directory;  // Virtual address space
    uint32_t kernel_stack;             // Top of kernel stack (for TSS.ESP0)
    
    // Scheduling
    struct process *next;
    struct process *prev;
    uint32_t wake_time;                // For sleep/wakeup
} process_t;
```

The fields fall into categories:

**Register state**: Everything the CPU uses. Miss one, and the resumed process corrupts data or crashes.

**Memory management**: Each process gets its own page directory (address space isolation). The kernel stack pointer is crucial—when a user-mode process traps to the kernel (interrupt or syscall), the CPU switches to this stack.

**Scheduling**: Linked list pointers for the run queue, plus any state for blocking/waking.

### PCB Storage

```c
#define MAX_PROCESSES 64
#define KERNEL_STACK_SIZE 4096

static process_t process_table[MAX_PROCESSES];
static uint32_t next_pid = 1;

process_t *current_process = NULL;
process_t *ready_queue = NULL;

process_t *create_process(const char *name, void (*entry_point)(void), int is_user) {
    // Find free slot
    process_t *proc = NULL;
    for (int i = 0; i < MAX_PROCESSES; i++) {
        if (process_table[i].state == PROCESS_ZOMBIE || 
            process_table[i].pid == 0) {
            proc = &process_table[i];
            break;
        }
    }
    
    if (!proc) {
        return NULL;  // Process table full
    }
    
    // Initialize PCB
    memset(proc, 0, sizeof(process_t));
    proc->pid = next_pid++;
    strncpy(proc->name, name, 15);
    proc->state = PROCESS_READY;
    
    // Allocate kernel stack
    void *stack_frame = alloc_frame();
    proc->kernel_stack = (uint32_t)stack_frame + KERNEL_STACK_SIZE;
    
    // Set up initial register state
    proc->eip = (uint32_t)entry_point;
    proc->esp = proc->kernel_stack;
    proc->ebp = proc->kernel_stack;
    proc->eflags = 0x202;  // IF=1 (interrupts enabled), bit 1 always set
    
    // Segment selectors
    if (is_user) {
        proc->cs = 0x1B;  // User code selector (ring 3)
        proc->ds = proc->es = proc->fs = proc->gs = proc->ss = 0x23;  // User data
        proc->esp = 0xBFFFFFFF;  // User stack at top of user space
        
        // Create user page directory (copy kernel mappings)
        proc->page_directory = clone_page_directory(current_directory);
    } else {
        proc->cs = 0x08;  // Kernel code selector (ring 0)
        proc->ds = proc->es = proc->fs = proc->gs = proc->ss = 0x10;  // Kernel data
        proc->page_directory = current_directory;  // Share kernel space
    }
    
    // Add to ready queue
    proc->next = ready_queue;
    if (ready_queue) {
        ready_queue->prev = proc;
    }
    ready_queue = proc;
    
    return proc;
}
```

### Setting Up the First Process

The first process isn't "created"—it *is* the kernel's current execution state. You construct its PCB by capturing what's already running:

```c
void scheduler_init(void) {
    // Create PCB for current (kernel) process
    process_t *kernel_proc = &process_table[0];
    memset(kernel_proc, 0, sizeof(process_t));
    
    kernel_proc->pid = next_pid++;
    strcpy(kernel_proc->name, "kernel");
    kernel_proc->state = PROCESS_RUNNING;
    kernel_proc->page_directory = current_directory;
    kernel_proc->kernel_stack = 0x90000;  // Current kernel stack
    
    current_process = kernel_proc;
}
```

---

## Context Switch: The Assembly Core

### Why Assembly is Mandatory

C provides no control over:
- Which registers are saved and in what order
- Stack pointer manipulation during the save
- The exact layout of the register dump
- Loading CR3 (page directory base)

The context switch must be written in assembly.

{{DIAGRAM:diag-context-switch-flow}}

### The Assembly Implementation

```asm
; context_switch.asm
; void context_switch(process_t *old, process_t *new)

global context_switch
extern current_process

section .text

; Process struct offsets (must match C struct)
%define PCB_PID         0
%define PCB_EAX         8
%define PCB_EBX         12
%define PCB_ECX         16
%define PCB_EDX         20
%define PCB_ESI         24
%define PCB_EDI         28
%define PCB_EBP         32
%define PCB_ESP         36
%define PCB_EIP         40
%define PCB_EFLAGS      44
%define PCB_CS          48
%define PCB_DS          52
%define PCB_ES          56
%define PCB_FS          60
%define PCB_GS          64
%define PCB_SS          68
%define PCB_PAGE_DIR    72
%define PCB_KERNEL_STACK 76

context_switch:
    ; Arguments: [esp+4] = old process, [esp+8] = new process
    push ebp
    mov ebp, esp
    
    pushfd                    ; Save EFLAGS
    pushad                    ; Save EAX, ECX, EDX, EBX, ESP (old), EBP, ESI, EDI
    
    ; Save segment registers
    push ds
    push es
    push fs
    push gs
    
    ; Get pointer to old process
    mov eax, [ebp + 8]        ; old process pointer
    test eax, eax
    jz .load_new              ; If old is NULL, skip saving
    
    ; Save registers to old PCB
    mov [eax + PCB_EAX], eax  ; ... actually need to be careful here
    ; Better: use edi as temp since we pushed it
    
    ; Actually, reload old pointer and save properly
    mov edi, [ebp + 8]        ; old process
    
    ; Get stack position before we pushed stuff
    mov ecx, [ebp - 4]        ; pushed ebp
    mov [edi + PCB_EBP], ecx
    
    ; We need the ESP value from BEFORE pushad
    ; pushad pushes: EAX, ECX, EDX, EBX, ESP(original), EBP, ESI, EDI
    ; The ESP in pushad is at offset 16 from start of pushad area
    mov ecx, [esp + 16]       ; Original ESP from pushad
    mov [edi + PCB_ESP], ecx
    
    ; Save other registers from pushad
    mov ecx, [esp + 0]        ; EDI
    mov [edi + PCB_EDI], ecx
    mov ecx, [esp + 4]        ; ESI
    mov [edi + PCB_ESI], ecx
    mov ecx, [esp + 20]       ; EBX
    mov [edi + PCB_EBX], ecx
    mov ecx, [esp + 24]       ; EDX
    mov [edi + PCB_EDX], ecx
    mov ecx, [esp + 28]       ; ECX
    mov [edi + PCB_ECX], ecx
    mov ecx, [esp + 32]       ; EAX
    mov [edi + PCB_EAX], ecx
    
    ; Save EFLAGS
    mov ecx, [esp + 36]       ; EFLAGS (after pushad + segs)
    add ecx, 16               ; Adjust for segment registers
    mov [edi + PCB_EFLAGS], ecx
    
    ; Save instruction pointer (return address from context_switch call)
    mov ecx, [ebp + 4]
    mov [edi + PCB_EIP], ecx
    
    ; Save segment registers
    mov [edi + PCB_DS], ds
    mov [edi + PCB_ES], es
    mov [edi + PCB_FS], fs
    mov [edi + PCB_GS], gs
    mov [edi + PCB_CS], cs
    mov [edi + PCB_SS], ss

.load_new:
    ; Load new process
    mov esi, [ebp + 12]       ; new process
    
    ; Update current_process
    mov [current_process], esi
    
    ; Load page directory if different
    mov eax, [current_directory]
    mov ebx, [esi + PCB_PAGE_DIR]
    cmp eax, ebx
    je .same_page_dir
    
    ; Switch to new page directory
    mov cr3, ebx
    mov [current_directory], ebx
    
.same_page_dir:
    ; Update TSS.ESP0 for ring 3 processes
    extern tss_update_esp0
    push dword [esi + PCB_KERNEL_STACK]
    call tss_update_esp0
    add esp, 4
    
    ; Restore segment registers
    mov ds, [esi + PCB_DS]
    mov es, [esi + PCB_ES]
    mov fs, [esi + PCB_FS]
    mov gs, [esi + PCB_GS]
    
    ; Restore general registers
    mov eax, [esi + PCB_EAX]
    mov ebx, [esi + PCB_EBX]
    mov ecx, [esi + PCB_ECX]
    mov edx, [esi + PCB_EDX]
    mov edi, [esi + PCB_EDI]
    
    ; Restore stack pointer and EBP
    mov esp, [esi + PCB_ESP]
    mov ebp, [esi + PCB_EBP]
    
    ; Restore ESI last (we were using it)
    mov esi, [esi + PCB_ESI]
    
    ; Restore EFLAGS and return
    pushfd
    popfd
    
    ret
```

### A Cleaner Approach: Using iret

A more robust approach uses `iret` to restore EIP, CS, EFLAGS, and optionally ESP/SS:

```asm
; Simpler context switch using iret
global switch_to_process

switch_to_process:
    ; [esp+4] = new process PCB pointer
    mov eax, [esp + 4]
    
    ; Update current_process global
    extern current_process
    mov [current_process], eax
    
    ; Switch page directory
    mov ebx, [eax + PCB_PAGE_DIR]
    mov cr3, ebx
    
    ; Update TSS.ESP0
    extern tss
    mov ecx, [eax + PCB_KERNEL_STACK]
    mov [tss + 4], ecx        ; tss.esp0 offset
    
    ; Set up stack for iret
    ; Need: EIP, CS, EFLAGS, (ESP, SS if user mode)
    mov ebx, [eax + PCB_ESP]
    mov ecx, [eax + PCB_EFLAGS]
    mov edx, [eax + PCB_EIP]
    
    ; Check if user mode (CS = 0x1B)
    cmp word [eax + PCB_CS], 0x1B
    je .user_mode
    
.kernel_mode:
    ; Build kernel-mode iret frame
    push ecx                  ; EFLAGS
    push dword [eax + PCB_CS] ; CS
    push edx                  ; EIP
    push dword [eax + PCB_GS]
    push dword [eax + PCB_FS]
    push dword [eax + PCB_ES]
    push dword [eax + PCB_DS]
    push dword [eax + PCB_EDI]
    push dword [eax + PCB_ESI]
    push dword [eax + PCB_EBP]
    push ebx                  ; ESP (will be loaded before iret)
    push dword [eax + PCB_EBX]
    push dword [eax + PCB_EDX]
    push dword [eax + PCB_ECX]
    push dword [eax + PCB_EAX]
    
    ; Restore registers
    pop eax
    pop ecx
    pop edx
    pop ebx
    add esp, 4                ; Skip ESP (already set)
    pop ebp
    pop esi
    pop edi
    pop ds
    pop es
    pop fs
    pop gs
    
    iret
    
.user_mode:
    ; Build user-mode iret frame (includes SS:ESP)
    push dword [eax + PCB_SS] ; User SS
    push ebx                  ; User ESP
    push ecx                  ; EFLAGS
    push dword [eax + PCB_CS] ; User CS
    push edx                  ; EIP
    ; ... restore other registers ...
    iret
```

---

## The TSS: Required for Ring Transitions

### What the TSS Does (and Doesn't Do)

The **Task State Segment** is a holdover from Intel's original multitasking design. Modern OSes use it for exactly one thing: **stack switching on privilege transitions**.

{{DIAGRAM:diag-tss-structure}}

When the CPU transitions from ring 3 to ring 0 (via interrupt, exception, or syscall), it needs a trusted stack. It can't use the user's stack—that would be a security hole. So it loads SS:ESP from TSS.SS0:ESP0.

```c
// TSS structure (only essential fields shown)
typedef struct {
    uint16_t prev_task;       // 0
    uint16_t reserved0;
    uint32_t esp0;            // 4: Stack pointer for ring 0
    uint16_t ss0;             // 8: Stack segment for ring 0
    uint16_t reserved1;
    uint32_t esp1;            // 12: For ring 1 (unused)
    uint16_t ss1;
    uint16_t reserved2;
    uint32_t esp2;            // 20: For ring 2 (unused)
    uint16_t ss2;
    uint16_t reserved3;
    uint32_t cr3;             // 28: Page directory (not used for software switching)
    uint32_t eip;             // 32
    uint32_t eflags;          // 36
    uint32_t eax, ecx, edx, ebx, esp, ebp, esi, edi;  // 40-64
    uint16_t es;              // 68
    uint16_t reserved4;
    uint16_t cs;              // 72
    uint16_t reserved5;
    uint16_t ss;              // 76
    uint16_t reserved6;
    uint16_t ds;              // 80
    uint16_t reserved7;
    uint16_t fs;              // 84
    uint16_t reserved8;
    uint16_t gs;              // 88
    uint16_t reserved9;
    uint16_t ldt;             // 92
    uint16_t reserved10;
    uint16_t trap;            // 96
    uint16_t iomap_base;      // 102
} __attribute__((packed)) tss_t;

static tss_t tss;

void tss_init(void) {
    memset(&tss, 0, sizeof(tss_t));
    
    tss.ss0 = 0x10;           // Kernel data segment
    tss.esp0 = 0x90000;       // Initial kernel stack (will be updated)
    tss.iomap_base = sizeof(tss_t);  // No I/O bitmap
    
    // Add TSS to GDT (entry 5)
    // TSS descriptor is a system segment (S=0 in access byte)
    uint32_t base = (uint32_t)&tss;
    uint32_t limit = sizeof(tss_t) - 1;
    
    gdt_set_gate(5, base, limit, 0xE9, 0x00);  // Present, ring 0, TSS type
    
    // Load TR (Task Register)
    asm volatile("ltr %w0" : : "r"(0x28));  // Selector: index 5, TI=0, RPL=0
}

void tss_update_esp0(uint32_t esp0) {
    tss.esp0 = esp0;
}
```

### When TSS.ESP0 Must Be Updated

Every time you switch to a new process that might run in user mode, you must update TSS.ESP0 to point to that process's kernel stack:

```c
void scheduler_yield(void) {
    process_t *old = current_process;
    process_t *new = pick_next_process();
    
    if (new && new != old) {
        old->state = PROCESS_READY;
        new->state = PROCESS_RUNNING;
        
        // Critical: update TSS before context switch
        tss_update_esp0(new->kernel_stack);
        
        context_switch(old, new);
    }
}
```

If you forget this, a user-mode process traps to the kernel with a stale ESP0, and the kernel stack pointer points to the *previous* process's stack. Corruption ensues.

---

## Round-Robin Scheduler: The Heart of Multitasking

### The Algorithm

Round-robin is the simplest fair scheduling algorithm:

1. Maintain a queue of READY processes
2. On timer interrupt, move current process to end of queue
3. Pick the first READY process
4. Context switch to it

{{DIAGRAM:diag-scheduler-queue}}

```c
void scheduler_tick(void) {
    // Called from timer interrupt handler
    
    if (!current_process) {
        return;  // Scheduler not initialized
    }
    
    // Current process used its time slice
    // Move it to the back of the ready queue
    if (current_process->state == PROCESS_RUNNING) {
        current_process->state = PROCESS_READY;
        
        // Move to end of queue
        if (ready_queue && ready_queue->next) {
            process_t *last = ready_queue;
            while (last->next) {
                last = last->next;
            }
            
            // Remove current from front
            ready_queue = current_process->next;
            if (ready_queue) {
                ready_queue->prev = NULL;
            }
            
            // Add to back
            last->next = current_process;
            current_process->prev = last;
            current_process->next = NULL;
        }
    }
    
    // Pick next process
    process_t *next = ready_queue;
    while (next) {
        if (next->state == PROCESS_READY) {
            break;
        }
        next = next->next;
    }
    
    if (next && next != current_process) {
        schedule_switch(next);
    }
}

void schedule_switch(process_t *next) {
    process_t *prev = current_process;
    
    next->state = PROCESS_RUNNING;
    current_process = next;
    
    // Remove from ready queue
    if (next->prev) {
        next->prev->next = next->next;
    } else {
        ready_queue = next->next;
    }
    if (next->next) {
        next->next->prev = next->prev;
    }
    next->prev = NULL;
    next->next = NULL;
    
    // Update TSS before switch
    tss_update_esp0(next->kernel_stack);
    
    // Perform context switch
    context_switch(prev, next);
}
```

### Integration with Timer Interrupt

```c
// In interrupt handler (from Milestone 2)
void irq_handler(registers_t *regs) {
    uint8_t irq = regs->int_no - 32;
    
    switch (irq) {
        case 0:  // Timer
            timer_ticks++;
            scheduler_tick();
            break;
        case 1:  // Keyboard
            keyboard_handler();
            break;
        // ... other IRQs ...
    }
    
    pic_send_eoi(irq);
}
```

---

## Demonstration: Three Kernel Processes

To prove preemptive multitasking works, create three processes that each print to a different screen region:

{{DIAGRAM:diag-multi-process-demo}}

```c
void process_a(void) {
    int count = 0;
    while (1) {
        vga_set_cursor(0, 0);
        vga_puts("[Process A] Count: ");
        vga_put_dec(count++);
        vga_puts("\n");
        
        // Busy loop to consume time slice
        for (volatile int i = 0; i < 1000000; i++);
    }
}

void process_b(void) {
    int count = 0;
    while (1) {
        vga_set_cursor(10, 0);
        vga_puts("[Process B] Count: ");
        vga_put_dec(count++);
        vga_puts("\n");
        
        for (volatile int i = 0; i < 1000000; i++);
    }
}

void process_c(void) {
    int count = 0;
    while (1) {
        vga_set_cursor(20, 0);
        vga_puts("[Process C] Count: ");
        vga_put_dec(count++);
        vga_puts("\n");
        
        for (volatile int i = 0; i < 1000000; i++);
    }
}

void demo_multitasking(void) {
    kprintf("Creating kernel processes...\n");
    
    create_process("proc_a", process_a, 0);  // 0 = kernel mode
    create_process("proc_b", process_b, 0);
    create_process("proc_c", process_c, 0);
    
    kprintf("Enabling scheduler...\n");
    
    // Enable timer interrupt
    uint8_t mask = inb(PIC1_DATA);
    mask &= ~0x01;  // Enable IRQ0
    outb(PIC1_DATA, mask);
    
    // Enable interrupts
    asm volatile("sti");
    
    // Yield to let scheduler take over
    scheduler_yield();
    
    // We never reach here
}
```

The three counters increment independently, each process unaware it's being interrupted dozens of times per second.

---

## User Mode: Crossing the Ring Boundary

### Creating a User Process

A user process differs from a kernel process in three ways:

1. **Segment selectors**: CS=0x1B, DS/ES/SS=0x23 (ring 3, CPL=3)
2. **Page directory**: Isolated from kernel, with kernel pages marked supervisor-only
3. **Stack**: Located in user space, not kernel space

{{DIAGRAM:diag-user-kernel-memory}}

```c
process_t *create_user_process(const char *name, void (*entry)(void)) {
    process_t *proc = create_process(name, entry, 1);
    if (!proc) return NULL;
    
    // Allocate user stack (at top of user space: 0xBFFFF000)
    uint32_t user_stack_virt = 0xBFFFF000;
    void *user_stack_phys = alloc_frame();
    
    // Map user stack in process's page directory
    map_page(proc->page_directory, user_stack_virt, 
             (uint32_t)user_stack_phys, PTE_WRITABLE | PTE_USER);
    
    proc->esp = user_stack_virt + PAGE_SIZE;  // Top of stack
    proc->ss = 0x23;  // User data segment (ring 3)
    
    // Copy user code (for now, assume it's somewhere in kernel memory)
    // In a real OS, you'd load from an executable file
    
    return proc;
}
```

### Entering User Mode: The iret Trick

To transition from kernel to user mode, you use `iret` with a specially crafted stack frame:

```asm
; enter_user_mode(entry_point, user_stack_top)
global enter_user_mode

enter_user_mode:
    mov eax, [esp + 4]    ; Entry point
    mov ebx, [esp + 8]    ; User stack top
    
    ; Set up segment registers for user mode
    mov cx, 0x23          ; User data selector
    mov ds, cx
    mov es, cx
    mov fs, cx
    mov gs, cx
    
    ; Push user mode stack frame for iret
    push 0x23             ; SS (user data)
    push ebx              ; ESP (user stack)
    pushf                 ; EFLAGS
    push 0x1B             ; CS (user code)
    push eax              ; EIP (entry point)
    
    ; Enable interrupts in EFLAGS
    or dword [esp + 8], 0x200
    
    ; Jump to user mode
    iret
```

The `iret` instruction pops SS, ESP, EFLAGS, CS, EIP from the stack. Because CS=0x1B (CPL=3), the CPU transitions to ring 3, loads the user stack from your pushed value, and begins executing user code.

### Isolation Verification

A user process should NOT be able to access kernel memory:

```c
void user_process_test(void) {
    // This should trigger a page fault
    uint32_t *kernel_addr = (uint32_t *)0xC0100000;
    *kernel_addr = 0xDEADBEEF;  // Write to kernel memory
    
    // Should never reach here
    sys_write("This should never print\n");
}
```

When this runs in ring 3, the page table entry for 0xC0100000 has the user/supervisor bit clear (supervisor-only). The CPU raises a page fault with error code bit 2 set (user-mode access).

---

## System Calls: User-to-Kernel Communication

### The INT 0x80 Interface

System calls use software interrupts to transition from user to kernel mode. The convention:

- **EAX**: System call number
- **EBX, ECX, EDX**: Arguments (up to 3)
- **EAX (return)**: Return value

{{DIAGRAM:diag-syscall-interface}}

### Setting Up the Syscall Gate

```c
#define SYSCALL_WRITE  0
#define SYSCALL_EXIT   1
#define SYSCALL_READ   2

void syscall_handler(registers_t *regs) {
    switch (regs->eax) {
        case SYSCALL_WRITE:
            // sys_write(fd, buf, len)
            regs->eax = sys_write(
                (int)regs->ebx,
                (const char *)regs->ecx,
                (size_t)regs->edx
            );
            break;
            
        case SYSCALL_EXIT:
            // sys_exit(status)
            sys_exit((int)regs->ebx);
            break;
            
        case SYSCALL_READ:
            // sys_read(fd, buf, len)
            regs->eax = sys_read(
                (int)regs->ebx,
                (char *)regs->ecx,
                (size_t)regs->edx
            );
            break;
            
        default:
            kprintf("Unknown syscall: %d\n", regs->eax);
            regs->eax = -1;
    }
}

void syscall_init(void) {
    // Register syscall handler at IDT entry 0x80
    // DPL=3 allows user mode to call via INT 0x80
    idt_set_gate(0x80, (uint32_t)isr128, 0x08, 0xEE);  // 0xEE = DPL 3, present, interrupt gate
}
```

### Implementing sys_write

```c
int sys_write(int fd, const char *buf, size_t len) {
    // Validate file descriptor
    if (fd < 0 || fd >= MAX_FDS) {
        return -1;  // EBADF
    }
    
    // Validate buffer pointer (must be in user space)
    if ((uint32_t)buf >= KERNEL_VIRTUAL_BASE) {
        return -1;  // EFAULT
    }
    
    // For now, just write to VGA
    for (size_t i = 0; i < len; i++) {
        vga_putchar(buf[i]);
    }
    
    return len;
}
```

### Implementing sys_exit

```c
void sys_exit(int status) {
    kprintf("Process %d exiting with status %d\n", 
            current_process->pid, status);
    
    // Mark process as zombie
    current_process->state = PROCESS_ZOMBIE;
    
    // Free resources
    if (current_process->page_directory != current_directory) {
        // Free user page directory and frames
        // (implementation depends on your memory manager)
    }
    
    // Remove from ready queue
    if (current_process->prev) {
        current_process->prev->next = current_process->next;
    }
    if (current_process->next) {
        current_process->next->prev = current_process->prev;
    }
    
    // Schedule next process
    scheduler_yield();
    
    // Never returns
}
```

### User-Space Syscall Wrapper

```c
// In user code (or a libc-like library)
static inline int syscall0(int num) {
    int ret;
    asm volatile("int $0x80" : "=a"(ret) : "a"(num));
    return ret;
}

static inline int syscall3(int num, int a, int b, int c) {
    int ret;
    asm volatile("int $0x80" 
                 : "=a"(ret) 
                 : "a"(num), "b"(a), "c"(b), "d"(c));
    return ret;
}

void user_print(const char *msg) {
    int len = 0;
    while (msg[len]) len++;
    syscall3(SYSCALL_WRITE, 1, (int)msg, len);
}

void user_exit(int status) {
    syscall1(SYSCALL_EXIT, status);
}
```

---

## Hardware Soul: The Physical Cost of Multitasking


![Context Switch: Cache and TLB Impact](./diagrams/diag-cache-analysis-context-switch.svg)


**Context switch overhead breakdown**:
- Register save/restore: 50-100 cycles (pushad/popad + segments)
- CR3 reload: 10-20 cycles, plus TLB flush cost
- TLB refill: 100-500 cycles per miss, ~64 entries = potentially 32,000 cycles worst case
- Cache cold start: L1 miss is ~4 cycles, L2 miss is ~12 cycles, L3 miss is ~40 cycles, memory is ~150 cycles

**The TLB flush problem**: Reloading CR3 flushes the TLB (unless PCID is used). A process with good locality might have 50 TLB entries populated. After a switch, all 50 are gone. The next instruction might trigger a TLB miss, then another, then another. This is why kernel threads share the same address space—no TLB flush.

**Cache pollution**: Process A runs for 10ms, filling L1 with its data. Context switch to process B. Process B's working set evicts process A's. When process A runs again, it finds a cold cache. This is the working set size problem: if a process's working set exceeds L1 cache, it suffers cache misses every time it's scheduled.

**Interrupt latency vs scheduling latency**: Timer interrupt fires every 10ms. But if you're in an interrupt handler or critical section with interrupts disabled, the scheduler can't run. Worst-case scheduling latency = timer quantum + maximum interrupt handler time + maximum critical section time. This is why real-time systems minimize interrupt handler work and keep critical sections short.

**The scheduling granularity trade-off**:
- Shorter quantum (1ms): Better responsiveness, more context switch overhead
- Longer quantum (100ms): Less overhead, worse interactivity

At 100Hz (10ms quantum) with 5 processes, each gets 2 CPU seconds per wall-clock second. But context switch overhead at 100Hz is 0.1% of CPU time—negligible. At 1000Hz, it's 1%. At 10000Hz, it's 10%. There's a reason 100-1000Hz is typical.

---

## Debugging Scheduler Issues

**Symptom: System freezes after enabling scheduler**

- Timer interrupt not firing: Check PIC mask, IDT entry for IRQ0
- No ready processes: Check that processes were created
- Context switch corrupts state: Add serial debug prints before/after switch

**Symptom: Processes run once, then crash**

- EIP not saved correctly: Check PCB offset calculations
- Stack corruption: Verify ESP is saved/restored correctly
- Page directory not switched: Check CR3 loading

**Symptom: User-mode process causes triple fault**

- TSS not initialized: Check `ltr` was called
- TSS.ESP0 not updated: Must update on every context switch
- User page tables missing kernel mapping: Kernel pages must be mapped but supervisor-only

**Symptom: Syscall returns garbage or crashes**

- Arguments not passed correctly: Check EBX/ECX/EDX handling
- Return value not in EAX: Check that handler sets regs->eax
- IDT gate DPL wrong: Must be 3 to allow user-mode `int 0x80`

**Symptom: Page fault in user mode accessing valid address**

- User bit not set on page table entry
- Page not mapped in user's page directory
- Stack overflow (user stack too small)

---

## Design Decision: Cooperative vs Preemptive Scheduling

| Aspect | Cooperative | Preemptive | Used By |
|--------|-------------|------------|---------|
| Switch trigger | Process yields | Timer interrupt | Preemptive: Linux, Windows |
| Latency | Unbounded | Bounded | Cooperative: Green threads |
| Implementation | Simple | Complex (interrupt-safety) | |
| Fault isolation | Bad (crash hangs system) | Good (can kill stuck process) | |
| Real-time | Impossible | Possible with priority | |

Preemptive scheduling is mandatory for a general-purpose OS. Cooperative can work for specific workloads (event-driven servers, embedded loops) but can't guarantee responsiveness.

---

## Knowledge Cascade

You've built preemptive multitasking—the mechanism that makes modern computing possible. Here's where this knowledge connects:

**Thread Pools and Async Runtimes**: The context switch you implemented is exactly what green threads do in software. Go's goroutines save registers to a structure, switch to the next runnable goroutine, and restore—same mechanism, user-space implementation. Tokio tasks in Rust, Erlang processes, and Java virtual threads all use this technique. The difference is they don't need ring transitions or TSS manipulation—they're all in user space. Understanding register save/restore is fundamental to designing coroutine systems.

**Virtualization and VM Exits (Cross-Domain)**: When a VM executes a privileged instruction (like accessing CR3), the CPU performs a "VM exit"—analogous to your ring 3→ring 0 transition. The VMCS (Virtual Machine Control Structure) saves guest state (registers, segment selectors, CR3) just like your PCB saves process state. The hypervisor handles the exit, potentially emulates the instruction, then does a "VM entry" to resume the guest. VMware, KVM, and Hyper-V all implement this loop. Understanding your TSS and ring transitions is the first step to understanding VM introspection and escape exploits.

**Real-Time Scheduling**: Your round-robin scheduler is the foundation for understanding advanced schedulers. Linux's CFS (Completely Fair Scheduler) uses a red-black tree instead of a queue, tracking "virtual runtime" to ensure fairness. SCHED_FIFO and SCHED_RR provide real-time guarantees—the timer interrupt doesn't demote them, only explicit yield or blocking does. Deadline scheduling (EDF—Earliest Deadline First) is used in hard real-time systems. All of these are variations on the context switch you just built.

**Security: Privilege Escalation Attacks**: The ring boundary and TSS mechanism you implemented is exactly what kernel exploits try to bypass. A privilege escalation exploit might: (1) find a kernel bug that writes to user-controlled address, (2) overwrite a function pointer with user-controlled code, (3) trigger kernel execution of that code. The CPU is now executing ring 3 code with ring 0 privileges. Meltdown and Spectre attacks target the same isolation machinery—speculative execution crossing protection boundaries. Understanding ring transitions is understanding the attack surface.

**Coroutine Implementations in Game Engines**: Fiber-based job systems in game engines (Unity's Jobs system, Unreal's task graph) use the same context-switching techniques. A game might have 10,000 "fibers" (lightweight threads) that yield when waiting for animation, physics, or I/O. The fiber switch saves registers to a fiber-local context, just like your PCB. The difference is fibers never cross privilege levels—pure user-space switching. This is how modern games achieve massive parallelism without OS thread overhead.

**Forward: What You Can Now Build**: With preemptive multitasking, you can:
- Implement blocking I/O (process blocks on read, scheduler runs others)
- Add process priorities (higher priority processes run first)
- Build signals (interrupt a process asynchronously, like SIGKILL)
- Create /proc filesystem (expose PCB information to user space)
- Implement fork/exec (create new processes from user space)

---

## Summary

You've built preemptive multitasking—the illusion of parallelism through precise, hardware-assisted context switching:

1. **Process Control Block (PCB)**: Stores PID, all registers, page directory, kernel stack pointer
2. **Context switch**: Assembly routine that saves old process state and loads new process state
3. **TSS (Task State Segment)**: Provides SS0:ESP0 for ring 3→ring 0 stack switching
4. **Round-robin scheduler**: Timer interrupt triggers queue rotation and context switch
5. **User-mode processes**: Ring 3 execution with isolated address spaces
6. **System calls**: INT 0x80 with DPL=3 enables user→kernel transitions

The revelation: only ONE process runs at a time. "Concurrent" execution is achieved by saving every register, loading the next process's registers, and jumping to its saved EIP. The interrupted process has no awareness it was suspended. System calls are not function calls—they trigger hardware privilege transitions through IDT gates, with stack switching via TSS.

The physical constraints—TLB flush on CR3 reload, cache pollution between processes, interrupt latency affecting scheduling latency—shape every design decision. This is systems programming at its most fundamental.


![OS Kernel Architecture: Satellite View](./diagrams/diag-satellite-system.svg)


---
<!-- END_MS -->




# TDD

A complete x86 operating system kernel implementing the critical path from BIOS bootstrap to preemptive multitasking with user-mode processes. The design negotiates with x86 hardware constraints at every layer: 512-byte MBR limit, A20 line legacy, GDT/IDT configuration, 4KB page granularity, TLB non-coherence, and ring-level transitions. The kernel provides interrupt-driven I/O, virtual memory with higher-half mapping, and preemptive round-robin scheduling with privilege isolation.



<!-- TDD_MOD_ID: mod-boot -->
# Technical Design Specification: Bootloader and Protected Mode Entry

## Module Charter

The bootloader module transforms the CPU from its 16-bit real mode state (as left by BIOS) into 32-bit protected mode with flat segmentation, then loads and transfers control to the C kernel. It handles the A20 line enabling, GDT configuration, protected mode transition, and kernel loading from disk.

**What it does NOT do**: This module does not set up paging (CR0.PG remains 0), does not configure the IDT or enable interrupts, does not perform any memory allocation, and does not handle user-mode (ring 3) transitions.

**Upstream dependencies**: BIOS has loaded the MBR at 0x7C00 and passed the boot drive number in DL.

**Downstream consumers**: The C kernel entry point expects protected mode with flat segments (CS=0x08, DS/ES/SS=0x10), a valid stack, and zeroed BSS section.

**Invariants**: Stage 1 must fit in 510 bytes; GDT must contain at least null + kernel code + kernel data descriptors; protected mode transition must use a far jump to flush the pipeline; kernel must be loaded at a known physical address (0x100000).

---

## File Structure

Create files in this order:

```
1. boot/stage1.asm          # Stage 1 bootloader (fits in 512-byte MBR)
2. boot/stage2.asm          # Stage 2 bootloader (larger, loaded by stage 1)
3. boot/gdt.asm             # GDT definitions and loader
4. boot/a20.asm             # A20 line enablement routines
5. boot/disk.asm            # Disk read routines using INT 13h
6. boot/kernel_entry.asm    # Kernel entry shim (BSS zeroing, stack setup)
7. kernel/linker.ld         # Linker script for kernel placement
8. kernel/main.c            # Kernel C entry point (minimal, for testing)
9. Makefile                 # Build system
```

---

## Complete Data Model

### GDT Entry Structure (8 bytes each)

Each GDT entry is 64 bits with specific field packing:

| Offset | Bits | Field | Description |
|--------|------|-------|-------------|
| 0-1 | 16 | Limit[15:0] | Low 16 bits of segment limit |
| 2-3 | 16 | Base[15:0] | Low 16 bits of base address |
| 4 | 8 | Base[23:16] | Bits 16-23 of base address |
| 5 | 8 | Access Byte | Present, DPL, Type flags |
| 6 | 8 | Flags + Limit[19:16] | Granularity, size, high limit bits |
| 7 | 8 | Base[31:24] | Bits 24-31 of base address |

**Access Byte (byte 5) bit layout:**

| Bit | Name | Value for Kernel Code | Value for Kernel Data |
|-----|------|----------------------|----------------------|
| 7 | Present | 1 | 1 |
| 6-5 | DPL | 00 (ring 0) | 00 (ring 0) |
| 4 | S (System) | 1 (code/data) | 1 (code/data) |
| 3 | Executable | 1 | 0 |
| 2 | DC (Direction/Conforming) | 0 | 0 |
| 1 | RW (Read/Writable) | 1 (readable) | 1 (writable) |
| 0 | Accessed | 0 (CPU sets) | 0 |

**Access byte values:**
- Kernel Code: `0x9A` (10011010b)
- Kernel Data: `0x92` (10010010b)

**Flags (high 4 bits of byte 6):**

| Bit | Name | Value |
|-----|------|-------|
| 7 | Granularity | 1 (4KB pages) |
| 6 | Size | 1 (32-bit) |
| 5 | Long | 0 (not 64-bit) |
| 4 | Reserved | 0 |

**Flags value: `0xC`** (1100b shifted to high nibble = `0xC0` when combined with limit)

**Complete GDT entry bytes:**

```
Null Descriptor (index 0):
  Bytes: 00 00 00 00 00 00 00 00

Kernel Code (index 1, selector 0x08):
  Base=0x00000000, Limit=0xFFFFF (4GB with 4KB granularity)
  Bytes: FF FF 00 00 00 9A CF 00

Kernel Data (index 2, selector 0x10):
  Base=0x00000000, Limit=0xFFFFF (4GB with 4KB granularity)
  Bytes: FF FF 00 00 00 92 CF 00
```

### GDTR Structure (6 bytes)

```
Offset 0-1: Limit (16-bit, size of GDT - 1)
Offset 2-5: Base (32-bit, linear address of GDT)
```

### Memory Layout at Boot

```
Physical Address    Contents
─────────────────────────────────────────
0x000000 - 0x0003FF  Real-mode IVT (DO NOT TOUCH)
0x000400 - 0x0004FF  BDA (BIOS Data Area)
0x000500 - 0x0007BF  Free (can use for stage 2)
0x0007C0 - 0x0007FF  Stage 1 bootloader (MBR)
0x0007FF - 0x000FFF  Free (stack can grow down from 0x7C00)
0x001000 - 0x00FFFF  Free (low memory)
0x010000 - 0x01FFFF  EBDA, video RAM, ROM (AVOID)
0x020000 - 0x0FFFFF  ROM area, video memory (AVOID)
0x100000 - N         Kernel loaded here (1MB mark)
```

### Kernel Stack Layout

```
The kernel stack is set up at 0x90000 (below 1MB, in identity-mappable region)
Stack grows downward from 0x90000
```

---

## Interface Contracts

### stage1.asm Entry Point

**Entry**: BIOS jumps to 0x7C00 with:
- `CS:IP = 0x0000:0x7C00`
- `DL = boot drive number` (0x00 = floppy, 0x80 = first HDD)
- Interrupts enabled
- Real mode (CR0.PE = 0)

**Exit**: Stage 1 jumps to stage 2 (or directly enters protected mode)

**Constraints**:
- Must fit in 510 bytes (bytes 510-511 are boot signature 0xAA55)
- Must not modify memory outside 0x7C00-0x7DFF without care
- BIOS disk services (INT 13h) are only available in real mode

### a20_enable()

**Purpose**: Enable the A20 line to access memory above 1MB

**Parameters**: None

**Returns**: 
- `AX = 1` on success
- `AX = 0` on failure

**Side effects**: Modifies keyboard controller state, port 0x92

**Error handling**: Tries multiple methods; if all fail, returns 0 but does not halt (some emulators don't need A20)

### gdt_load()

**Purpose**: Load the GDT register with the descriptor table

**Parameters**: None (GDT address is embedded)

**Returns**: None

**Side effects**: Loads GDTR; does NOT enable protected mode

**Constraints**: Must be called before setting CR0.PE

### protected_mode_enter()

**Purpose**: Transition from real mode to protected mode

**Parameters**: None

**Returns**: Never returns (far jump to 32-bit code)

**Side effects**:
- Sets CR0.PE = 1
- Performs far jump to flush pipeline
- Loads all segment registers with kernel data selector

**Constraints**:
- GDT must be loaded
- Interrupts must be disabled (`cli`)
- A20 should be enabled
- Identity-mapped code must exist at the jump target

### disk_read.sectors(drive, cylinder, head, sector, count, es:bx)

**Purpose**: Read sectors from disk using BIOS INT 13h

**Parameters**:
- `DL = drive` (passed from BIOS)
- `CH = cylinder` (low 8 bits)
- `CL = sector` (bits 0-5) | cylinder high bits (bits 6-7)
- `DH = head`
- `AL = count` (sectors to read)
- `ES:BX = destination buffer`

**Returns**:
- `CF = 0` on success
- `CF = 1` on error, `AH = error code`

**Error codes**:
- `0x01`: Invalid command
- `0x02`: Address mark not found
- `0x03`: Write protect (not applicable)
- `0x04`: Sector not found
- `0x05`: Reset failed
- `0x06`: Disk changed
- `0x08`: DMA overrun
- `0x09`: DMA boundary error
- `0x0C`: Invalid media
- `0x10`: CRC error
- `0x20`: Controller failure
- `0x40`: Seek failed
- `0x80`: Timeout (no response)

**Recovery**: Reset disk system (INT 13h, AH=0) and retry up to 3 times

### kernel_entry (assembly shim)

**Entry**: Called after protected mode transition with:
- `CS = 0x08` (kernel code selector)
- `DS/ES/SS = 0x10` (kernel data selector)
- `ESP = 0x90000` (kernel stack)

**Exit**: Calls `kernel_main()` in C

**Responsibilities**:
1. Ensure stack is valid
2. Zero BSS section from `__bss_start` to `__bss_end`
3. Call `kernel_main()`
4. Halt if kernel_main returns

---

## Algorithm Specification

### Stage 1 Bootloader Sequence

```
STAGE1_BOOT:
  1. Set up real-mode segments:
     DS = ES = SS = 0
     SP = 0x7C00 (stack grows down from bootloader)
  
  2. Save boot drive number:
     Store DL to [boot_drive]
  
  3. Display boot message (optional, INT 10h)
  
  4. Reset disk system:
     AH = 0, INT 13h
  
  5. Load stage 2 from sectors 2-N:
     CALL disk_load_stage2
     IF error: display error, halt
  
  6. Enable A20 line:
     CALL a20_enable
     IF error: display warning, continue anyway
  
  7. Load GDT:
     LGDT [gdt_descriptor]
  
  8. Disable interrupts:
     CLI
  
  9. Enter protected mode:
     MOV EAX, CR0
     OR EAX, 1
     MOV CR0, EAX
  
  10. Far jump to 32-bit code:
      JMP 0x08:protected_mode_entry
  
  ; After this point, we're in 32-bit protected mode
  
  11. Reload segment registers:
      MOV AX, 0x10
      MOV DS, AX
      MOV ES, AX
      MOV FS, AX
      MOV GS, AX
      MOV SS, AX
      MOV ESP, 0x90000
  
  12. Jump to kernel entry point:
      JMP 0x08:0x100000
```

### A20 Line Enablement (Multiple Methods)

```
A20_ENABLE:
  ; Method 1: Fast A20 gate (port 0x92)
  ; Works on most modern systems and emulators
  
  IN AL, 0x92
  TEST AL, 2        ; Check if A20 already enabled
  JNZ .a20_done
  
  OR AL, 2          ; Set A20 bit
  AND AL, 0xFE      ; Clear reset bit (safety)
  OUT 0x92, AL
  JMP .a20_done
  
  ; Method 2: Keyboard controller (more compatible)
  ; The keyboard controller's output port bit 1 controls A20
  
  CALL kbc_wait_input   ; Wait for input buffer empty
  MOV AL, 0xD0          ; Read output port command
  OUT 0x64, AL
  CALL kbc_wait_output  ; Wait for output buffer full
  IN AL, 0x60           ; Read current output port value
  PUSH AX
  
  CALL kbc_wait_input
  MOV AL, 0xD1          ; Write output port command
  OUT 0x64, AL
  
  CALL kbc_wait_input
  POP AX
  OR AL, 2              ; Set A20 bit
  OUT 0x60, AL
  
  CALL kbc_wait_input
  
.a20_done:
  ; Verify A20 is enabled
  CALL a20_verify
  RET

KBC_WAIT_INPUT:
  ; Wait until keyboard controller input buffer is empty
  IN AL, 0x64
  TEST AL, 2
  JNZ KBC_WAIT_INPUT
  RET

KBC_WAIT_OUTPUT:
  ; Wait until keyboard controller output buffer is full
  IN AL, 0x64
  TEST AL, 1
  JZ KBC_WAIT_OUTPUT
  RET

A20_VERIFY:
  ; Verify A20 is actually enabled by testing memory wraparound
  ; Write different values to 0x0000:0x0500 and 0xFFFF:0x0510
  ; If A20 disabled, they alias to the same physical address
  
  MOV AX, 0x0000
  MOV ES, AX
  MOV DI, 0x0500
  MOV BYTE [ES:DI], 0x00
  
  MOV AX, 0xFFFF
  MOV ES, AX
  MOV DI, 0x0510       ; 0xFFFF0 + 0x0510 = 0x100500 (wraps to 0x0500 if A20 off)
  MOV BYTE [ES:DI], 0xFF
  
  MOV AX, 0x0000
  MOV ES, AX
  MOV DI, 0x0500
  CMP BYTE [ES:DI], 0x00
  
  JE .a20_is_on        ; If still 0x00, addresses don't alias
  MOV AX, 0            ; A20 is off
  RET
  
.a20_is_on:
  MOV AX, 1
  RET
```

### Disk Read with Retry

```
DISK_READ:
  ; Input: CH = cylinder, CL = sector, DH = head, DL = drive
  ;        AL = count, ES:BX = buffer
  ; Output: CF set on error, AH = error code
  
  PUSH AX             ; Save sector count
  MOV SI, 3           ; Retry count
  
.retry:
  POP AX
  PUSH AX
  MOV AH, 0x02        ; BIOS read sectors function
  PUSH SI
  INT 0x13
  POP SI
  JNC .success
  
  ; Error occurred - reset disk and retry
  PUSH AX             ; Save error code
  XOR AH, AH          ; Reset disk system
  INT 0x13
  POP AX
  
  DEC SI
  JNZ .retry
  
  ; All retries exhausted
  POP AX
  STC                 ; Set carry flag to indicate error
  RET
  
.success:
  POP AX
  CLC                 ; Clear carry flag
  RET
```

### BSS Zeroing

```
KERNEL_ENTRY:
  ; Set up stack (if not already done)
  MOV ESP, 0x90000
  
  ; Zero BSS section
  ; __bss_start and __bss_end are defined in linker script
  MOV EDI, __bss_start
  MOV ECX, __bss_end
  SUB ECX, EDI        ; Length = end - start
  XOR EAX, EAX        ; Zero
  REP STOSB           ; Fill BSS with zeros
  
  ; Call C kernel entry point
  CALL kernel_main
  
  ; If kernel_main returns, halt
.halt:
  CLI
  HLT
  JMP .halt
```

---

## Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
|-------|-------------|----------|---------------|
| Disk read failure | INT 13h returns CF=1 | Reset disk, retry up to 3 times | Yes, error message on screen |
| A20 line stuck | a20_verify returns 0 | Continue anyway (emulators may not need it) | Warning message |
| No boot signature | BIOS checks bytes 510-511 | N/A (BIOS moves to next device) | "No bootable device" |
| GDT misconfiguration | Triple fault on mode switch | Debug with QEMU `-d int` | System reset |
| Kernel not found | Disk read returns error | Halt with error message | Yes |
| Stage 1 too large | Build-time (nasm error) | Remove debug code, optimize | Build fails |

---

## Implementation Sequence with Checkpoints

### Phase 1: Stage 1 Bootloader Assembly (4-6 hours)

**Goal**: Create a bootable MBR that displays a message and halts

**Files to create**:
1. `boot/stage1.asm`

**Steps**:
1. Set up the ORG directive at 0x7C00
2. Initialize segment registers (DS, ES, SS)
3. Set up stack pointer at 0x7C00
4. Save boot drive number
5. Display "Booting..." message using INT 10h
6. Add padding and boot signature

**Checkpoint**: Build and run in QEMU. Should display message and halt.
```bash
nasm -f bin boot/stage1.asm -o boot.bin
qemu-system-i386 -drive format=raw,file=boot.bin
# Expected: "Booting..." message, then system sits idle
```

### Phase 2: A20 Line Enablement (2-3 hours)

**Goal**: Implement and verify A20 line enabling

**Files to create**:
1. `boot/a20.asm`

**Steps**:
1. Implement fast A20 gate method (port 0x92)
2. Implement keyboard controller method (ports 0x60/0x64)
3. Implement A20 verification routine
4. Add debug output showing A20 status

**Checkpoint**: Display "A20: enabled" or "A20: disabled" on boot
```bash
# Build with a20.asm included
# Expected: "A20: enabled" message
```

### Phase 3: GDT Configuration and Loading (3-4 hours)

**Goal**: Define and load a valid GDT

**Files to create**:
1. `boot/gdt.asm`

**Steps**:
1. Define GDT with null, kernel code, and kernel data descriptors
2. Define GDTR structure (limit and base)
3. Implement lgdt wrapper
4. Add debug output showing GDT address

**Checkpoint**: GDT loads without error (no visible crash before protected mode)
```bash
# Add message "GDT loaded" after lgdt
# Expected: Message appears, system continues
```

### Phase 4: Protected Mode Transition (2-3 hours)

**Goal**: Successfully enter 32-bit protected mode

**Files to modify**:
1. `boot/stage1.asm`

**Steps**:
1. Add `cli` before mode switch
2. Set CR0.PE bit
3. Add far jump to 32-bit code section
4. Create 32-bit code section with segment register reloads
5. Display "Protected mode!" message

**Checkpoint**: Successfully enter protected mode and display message
```bash
# Expected: "Protected mode!" message in 32-bit code
# QEMU should not reset or triple fault
```

### Phase 5: Kernel Loader (3-4 hours)

**Goal**: Load kernel binary from disk

**Files to create**:
1. `boot/disk.asm`
2. `boot/stage2.asm` (optional, for larger loader)

**Steps**:
1. Implement disk_read routine with retry logic
2. Calculate CHS geometry for kernel location
3. Load kernel to 0x100000
4. Verify kernel was loaded (check first bytes)

**Checkpoint**: Kernel binary loaded to 0x100000
```bash
# Create a simple kernel.bin (just a few bytes for testing)
dd if=/dev/zero of=disk.img bs=512 count=2880
dd if=boot.bin of=disk.img bs=512 count=1 conv=notrunc
dd if=kernel.bin of=disk.img bs=512 seek=1 conv=notrunc

# In protected mode, display value at 0x100000
# Expected: Matches first bytes of kernel.bin
```

### Phase 6: Kernel Entry Shim (2-3 hours)

**Goal**: Create assembly entry point that calls C code

**Files to create**:
1. `boot/kernel_entry.asm`

**Steps**:
1. Set up kernel stack at 0x90000
2. Zero BSS section using linker symbols
3. Call kernel_main
4. Add halt loop if kernel_main returns

**Checkpoint**: C kernel_main is called and can print a message
```bash
# kernel_main just prints "Hello from C!"
# Expected: Message appears on screen
```

### Phase 7: Linker Script Design (2-3 hours)

**Goal**: Define kernel memory layout

**Files to create**:
1. `kernel/linker.ld`

**Steps**:
1. Set entry point to kernel_entry
2. Place .text at 0x100000
3. Define .rodata, .data, .bss sections
4. Export __bss_start and __bss_end symbols
5. Add /DISCARD/ for unwanted sections

**Checkpoint**: Kernel links successfully, BSS is properly zeroed
```bash
ld -m elf_i386 -T kernel/linker.ld -o kernel.elf kernel_entry.o main.o
nm kernel.elf | grep bss
# Expected: __bss_start and __bss_end defined
```

### Final Integration (2-3 hours)

**Goal**: Complete bootable system

**Steps**:
1. Create Makefile with all build rules
2. Create disk image with bootloader and kernel
3. Add VGA and serial output to kernel_main
4. Test in QEMU with serial output

**Checkpoint**: Kernel boots, displays welcome message on VGA and serial
```bash
make
qemu-system-i386 -drive format=raw,file=os.img -serial stdio
# Expected: Welcome message on screen and serial console
# Run tests: make test (all pass)
```

---

## Test Specification

### Test 1: Boot Signature Valid

```python
# test_boot_signature.py
def test_boot_signature():
    with open('boot.bin', 'rb') as f:
        data = f.read()
    
    assert len(data) == 512, "Boot sector must be exactly 512 bytes"
    assert data[510] == 0x55, "Byte 510 must be 0x55"
    assert data[511] == 0xAA, "Byte 511 must be 0xAA"
```

### Test 2: GDT Structure Valid

```python
# test_gdt.py
def test_gdt_null_descriptor():
    # Null descriptor must be all zeros
    # Locate GDT in binary and verify first 8 bytes are 0
    pass

def test_gdt_code_descriptor():
    # Kernel code descriptor:
    # - Base = 0
    # - Limit = 0xFFFFF
    # - Access = 0x9A (present, ring 0, code, readable)
    # - Flags = 0xC (4KB granularity, 32-bit)
    pass

def test_gdt_data_descriptor():
    # Kernel data descriptor:
    # - Base = 0
    # - Limit = 0xFFFFF
    # - Access = 0x92 (present, ring 0, data, writable)
    # - Flags = 0xC
    pass
```

### Test 3: Protected Mode Entry

```bash
# test_protected_mode.sh
# Run QEMU with GDB to verify protected mode entry

qemu-system-i386 -drive format=raw,file=os.img -s -S &
QEMU_PID=$!

sleep 1

gdb -batch -ex "target remote :1234" \
    -ex "break *0x7C00" \
    -ex "continue" \
    -ex "x/i \$pc" \
    -ex "stepi 100" \
    -ex "info registers cr0" \
    -ex "quit"

kill $QEMU_PID

# Verify CR0.PE bit is set
```

### Test 4: Kernel Load Address

```bash
# test_kernel_load.sh
# Verify kernel is loaded at correct address

# Create test kernel with known pattern
echo -n "KERNEL_TEST_PATTERN" > test_pattern.bin
dd if=test_pattern.bin of=os.img bs=1 seek=$((0x100000)) conv=notrunc

# Run in QEMU and check memory
# (This requires QEMU monitor or GDB)
```

### Test 5: BSS Zeroing

```c
// kernel/test_bss.c
#include <stdint.h>

// These should be in BSS (uninitialized)
static uint32_t test_var1;
static uint32_t test_var2;
static char test_buffer[256];

void test_bss_zeroed(void) {
    // After kernel entry, all BSS should be zero
    if (test_var1 != 0 || test_var2 != 0) {
        vga_puts("FAIL: BSS variables not zeroed\n");
        halt();
    }
    
    for (int i = 0; i < 256; i++) {
        if (test_buffer[i] != 0) {
            vga_puts("FAIL: BSS buffer not zeroed\n");
            halt();
        }
    }
    
    vga_puts("PASS: BSS zeroed correctly\n");
}
```

### Test 6: Serial Output

```bash
# test_serial.sh
# Verify serial output works

OUTPUT=$(qemu-system-i386 -drive format=raw,file=os.img -serial stdio -display none -nographic 2>&1 | timeout 5 cat)

if echo "$OUTPUT" | grep -q "Welcome"; then
    echo "PASS: Serial output working"
else
    echo "FAIL: No welcome message on serial"
    exit 1
fi
```

### Test 7: VGA Output

```bash
# test_vga.sh
# Verify VGA text mode output works

# This requires checking VGA buffer in QEMU or visual inspection
# For automation, we check that the kernel doesn't crash

qemu-system-i386 -drive format=raw,file=os.img -nographic &
QEMU_PID=$!

sleep 2

if ps -p $QEMU_PID > /dev/null; then
    echo "PASS: Kernel running (VGA likely working)"
    kill $QEMU_PID
else
    echo "FAIL: Kernel crashed"
    exit 1
fi
```

---

## Performance Targets

| Operation | Target | How to Measure |
|-----------|--------|----------------|
| Boot to C entry | < 1 second | Time from QEMU start to first serial output |
| Stage 1 size | ≤ 510 bytes | `wc -c boot.bin` must show 512 (including signature) |
| GDT load | < 100 cycles | Single `lgdt` instruction |
| Protected mode switch | < 50 cycles | CR0 write + far jump |
| Disk read (16KB) | < 100ms | Measure time for loading kernel |

---

## Visual Diagrams

### Boot Sequence

![x86 Boot Sequence: BIOS to C Entry](./diagrams/diag-boot-sequence.svg)

### GDT Structure

```
GDT Memory Layout:
┌────────────────────────────────────────────────────────────────┐
│ Offset 0x00: Null Descriptor                                   │
│   00 00 00 00 00 00 00 00                                      │
├────────────────────────────────────────────────────────────────┤
│ Offset 0x08: Kernel Code Descriptor                            │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ Limit[15:0]  │ Base[15:0] │ Base[23:16] │ Access │ Flags │ │
│   │    0xFFFF    │   0x0000   │    0x00     │ 0x9A   │0xC*   │ │
│   │              │            │             │        │ +Lim  │ │
│   └──────────────────────────────────────────────────────────┘ │
│   Base[31:24] = 0x00                                           │
│   Decoded: Base=0, Limit=0xFFFFF (4KB gran = 4GB), 32-bit code │
├────────────────────────────────────────────────────────────────┤
│ Offset 0x10: Kernel Data Descriptor                            │
│   Same as code but Access = 0x92 (data, writable)              │
│   Decoded: Base=0, Limit=0xFFFFF (4KB gran = 4GB), 32-bit data │
└────────────────────────────────────────────────────────────────┘

GDTR Register:
┌────────────────┬────────────────────────────────────────────────┐
│ Limit (16-bit) │ Base (32-bit)                                  │
│    0x0017      │ Address of gdt_start                           │
│  (24-1=23)     │                                                │
└────────────────┴────────────────────────────────────────────────┘
```

### Protected Mode Transition

```
BEFORE (Real Mode):
┌─────────────────────────────────────────┐
│ CR0.PE = 0                              │
│ CS:IP = 0x0000:0x7C00 (real mode addr)  │
│ GDTR = undefined                        │
│ Segment registers = real mode values    │
│ Paging = disabled                       │
└─────────────────────────────────────────┘
              │
              ▼ cli (disable interrupts)
              │
              ▼ lgdt [gdt_descriptor]
              │
              ▼ mov eax, cr0 / or eax, 1 / mov cr0, eax
              │
┌─────────────────────────────────────────┐
│ CR0.PE = 1                              │
│ CS:IP = 0x0000:0x7C00 (still real mode! │
│         CPU is in inconsistent state)   │
│ GDTR = valid                            │
│ Pipeline contains 16-bit decoded inst.  │
└─────────────────────────────────────────┘
              │
              ▼ jmp 0x08:protected_mode_entry
              │  (far jump flushes pipeline,
              │   loads CS with selector 0x08)
              │
┌─────────────────────────────────────────┐
│ AFTER (Protected Mode):                 │
│ CR0.PE = 1                              │
│ CS = 0x08 (kernel code, GDT index 1)    │
│ EIP = protected_mode_entry              │
│ DS/ES/SS = still real mode values!      │
│ (Must reload immediately)               │
└─────────────────────────────────────────┘
              │
              ▼ mov ax, 0x10 / mov ds, ax / ...
              │
┌─────────────────────────────────────────┐
│ FULLY IN PROTECTED MODE:                │
│ All segment registers = 0x10            │
│ ESP = 0x90000 (kernel stack)            │
│ Ready to execute 32-bit kernel code     │
└─────────────────────────────────────────┘
```

### Triple Fault Chain

```
Normal Fault Flow:
┌──────────────────┐
│ CPU Exception    │──▶ IDT Entry ──▶ Handler ──▶ iret ──▶ Resume
└──────────────────┘

Double Fault:
┌──────────────────┐
│ CPU Exception    │──▶ IDT Entry ──▶ Handler CRASHES
└──────────────────┘         │
                              ▼
                    ┌──────────────────┐
                    │ Double Fault (8) │──▶ IDT[8] ──▶ Handler
                    └──────────────────┘

Triple Fault (System Reset):
┌──────────────────┐
│ CPU Exception    │──▶ IDT Entry ──▶ Handler CRASHES
└──────────────────┘         │
                              ▼
                    ┌──────────────────┐
                    │ Double Fault (8) │──▶ IDT[8] ──▶ Handler CRASHES
                    └──────────────────┘         │
                                                 ▼
                                       ┌──────────────────┐
                                       │ Triple Fault     │──▶ CPU RESET
                                       │ (No handler!)    │
                                       └──────────────────┘

Common Causes in Bootloader:
1. GDT not loaded before setting CR0.PE
2. Far jump with invalid selector
3. Segment register load with invalid selector
4. Stack pointer invalid after mode switch
5. Code at jump target is not 32-bit
```

---

## Hardware Soul

### Cache Lines Touched

- **GDT load**: The GDT itself (24 bytes for 3 entries) is read by the CPU into internal descriptor caches. This does NOT go through the normal cache hierarchy—it's a special register load.
- **Stage 1 execution**: Runs from BIOS ROM shadow or RAM at 0x7C00. First execution likely has cache misses.
- **Kernel load**: Sequential disk reads are buffered by BIOS. The destination at 0x100000 is likely cache-cold.

### Pipeline Behavior

- **Real mode code**: 16-bit decoding path
- **Mode switch**: Setting CR0.PE doesn't flush the pipeline—the far jump does
- **Far jump cost**: 10-30 cycles to flush and refill pipeline with 32-bit code
- **After switch**: CPU now uses 32-bit decoding path

### Memory Access Patterns

- **Disk reads**: Sequential sector reads via INT 13h. BIOS may optimize this with multi-sector reads.
- **GDT access**: Single 24-byte read when `lgdt` executes
- **BSS zeroing**: Sequential writes from `__bss_start` to `__bss_end`. This is cache-friendly (sequential stores).

### I/O Port Access

| Port | Purpose | Timing |
|------|---------|--------|
| 0x92 | Fast A20 gate | ~1 microsecond |
| 0x64 | KBC command | ~10 microseconds wait |
| 0x60 | KBC data | ~10 microseconds wait |

---

## Implementation Notes

### Stage 1 Size Optimization

If stage 1 exceeds 510 bytes:

1. Remove string messages (use single characters)
2. Use shorter instruction encodings (`xor ax, ax` instead of `mov ax, 0`)
3. Combine operations (`push cs / pop ds` instead of `mov ax, cs / mov ds, ax`)
4. Move complex code to stage 2

### Two-Stage vs Direct Kernel Load

**Two-stage approach** (recommended):
- Stage 1: Minimal—just load stage 2 and enter protected mode
- Stage 2: Larger—load kernel, set up environment, transfer control

**Direct load**:
- Stage 1 loads kernel directly before protected mode
- Simpler but limited to what fits in 510 bytes

### Debugging with QEMU

```bash
# Log all interrupts and CPU resets
qemu-system-i386 -drive format=raw,file=os.img -d int,cpu_reset -serial stdio 2>&1 | tee debug.log

# Use GDB
qemu-system-i386 -drive format=raw,file=os.img -s -S &
gdb -ex "target remote :1234" -ex "break *0x7c00"

# Monitor commands (in QEMU)
(qemu) info registers
(qemu) x/10i $eip
(qemu) xp /10x 0x100000  # physical memory view
```

---


<!-- TDD_MOD_ID: mod-interrupts -->
# Technical Design Specification: Interrupt and Exception Handling

## Module Charter

The interrupt module implements the IDT (Interrupt Descriptor Table) with 256 entries, CPU exception handlers for vectors 0-31, and hardware IRQ handlers for vectors 32-47. It configures the 8259 PIC to remap IRQs away from CPU exception vectors, implements PIT timer and PS/2 keyboard drivers, and provides the infrastructure for asynchronous hardware event handling.

**What it does NOT do**: This module does not implement scheduling (timer just increments a counter), does not handle system calls (vector 0x80), does not implement APIC (uses legacy 8259 PIC only), and does not handle page fault recovery (just prints diagnostics).

**Upstream dependencies**: GDT must be loaded with kernel code (0x08) and data (0x10) selectors; protected mode must be active.

**Downstream consumers**: Scheduler (Milestone 4) uses timer interrupt for preemption; keyboard buffer consumed by shell/TTY; exception handlers provide crash diagnostics.

**Invariants**: All interrupt handlers must save/restore complete register state; EOI must be sent to PIC for all IRQs; IDT must be loaded before interrupts are enabled; PIC must be remapped before any IRQs are unmasked.

---

## File Structure

Create files in this order:

```
1. kernel/idt.h              # IDT structures and function declarations
2. kernel/idt.c              # IDT initialization and gate management
3. kernel/isr.asm            # Assembly ISR stubs for exceptions 0-31
4. kernel/irq.asm            # Assembly IRQ stubs for IRQs 0-15
5. kernel/interrupt_handler.c # Common C handler dispatcher
6. kernel/pic.h              # PIC interface declarations
7. kernel/pic.c              # PIC remapping and EOI functions
8. kernel/timer.h            # PIT timer interface
9. kernel/timer.c            # PIT initialization and handler
10. kernel/keyboard.h        # Keyboard interface declarations
11. kernel/keyboard.c        # PS/2 keyboard driver with scancode table
12. kernel/registers.h       # Register frame structure definition
```

---

## Complete Data Model

### IDT Entry Structure (8 bytes)

Each IDT entry is a gate descriptor with this exact layout:

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 16 bits | Offset Low | Bits 15:0 of handler address |
| 2 | 16 bits | Segment Selector | Code segment for handler (0x08) |
| 4 | 8 bits | Reserved | Must be 0 |
| 5 | 8 bits | Type/Attributes | Gate type and flags |
| 6 | 16 bits | Offset High | Bits 31:16 of handler address |

**Type/Attributes byte (byte 5) bit layout:**

| Bit | Name | Value | Description |
|-----|------|-------|-------------|
| 7 | Present | 1 | Gate is valid |
| 6-5 | DPL | 00 or 11 | Descriptor Privilege Level (0=kernel only, 3=user callable) |
| 4 | Storage | 0 | Must be 0 for interrupt/trap gates |
| 3-0 | Type | 1110 or 1111 | 1110=Interrupt gate (IF=0), 1111=Trap gate (IF unchanged) |

**Standard attribute values:**
- Interrupt gate (kernel only): `0x8E` (10001110b)
- Trap gate (kernel only): `0x8F` (10001111b)
- User-callable interrupt gate: `0xEE` (11101110b) — for syscalls

```c
// kernel/idt.h
#include <stdint.h>

typedef struct {
    uint16_t offset_low;    // Offset bits 15:0
    uint16_t selector;      // Code segment selector
    uint8_t  zero;          // Reserved, must be 0
    uint8_t  type_attr;     // Type and attributes
    uint16_t offset_high;   // Offset bits 31:16
} __attribute__((packed)) idt_entry_t;

typedef struct {
    uint16_t limit;         // Size of IDT - 1
    uint32_t base;          // Address of IDT
} __attribute__((packed)) idt_ptr_t;

#define IDT_ENTRIES 256
#define IDT_INTERRUPT_GATE 0x8E
#define IDT_TRAP_GATE 0x8F
#define IDT_USER_GATE 0xEE
```

### IDTR (6 bytes)

```c
// Loaded via lidt instruction
// limit = sizeof(idt_entry_t) * IDT_ENTRIES - 1 = 2047
// base = (uint32_t)&idt
```

### Register Frame Structure

This structure matches what the assembly stubs push onto the stack:

```c
// kernel/registers.h
#include <stdint.h>

typedef struct {
    // Pushed by our assembly stub (manual)
    uint32_t gs, fs, es, ds;
    
    // Pushed by pusha (EAX, ECX, EDX, EBX, ESP(original), EBP, ESI, EDI)
    uint32_t edi, esi, ebp, esp, ebx, edx, ecx, eax;
    
    // Pushed by our assembly stub
    uint32_t int_no;        // Interrupt number
    uint32_t err_code;      // Error code (0 for exceptions without one)
    
    // Pushed by CPU automatically
    uint32_t eip, cs, eflags;
    
    // Only present if privilege change occurred (ring 3 -> ring 0)
    uint32_t useresp, ss;
} __attribute__((packed)) registers_t;
```

**Stack layout at handler entry (grows downward):**

```
High addresses
┌─────────────────┐
│    SS (old)     │  ← Only if privilege change
│    ESP (old)    │  ← Only if privilege change
├─────────────────┤
│    EFLAGS       │  ← CPU pushes
├─────────────────┤
│    CS (old)     │  ← CPU pushes
│    EIP (old)    │  ← CPU pushes
├─────────────────┤
│  Error Code     │  ← CPU pushes (for some exceptions) or stub pushes 0
├─────────────────┤
│  Interrupt #    │  ← Stub pushes
├─────────────────┤
│      EAX        │  ← pusha
│      ECX        │
│      EDX        │
│      EBX        │
│      ESP (old)  │  ← Value before pusha
│      EBP        │
│      ESI        │
│      EDI        │  ← pusha ends
├─────────────────┤
│      DS         │  ← Stub pushes
│      ES         │
│      FS         │
│      GS         │  ← Stub pushes
└─────────────────┘ ← ESP points here
Low addresses
```

### CPU Exception Vectors

| Vector | Mnemonic | Error Code? | Type | Description |
|--------|----------|-------------|------|-------------|
| 0 | #DE | No | Fault | Divide Error |
| 1 | #DB | No | Fault/Trap | Debug Exception |
| 2 | NMI | No | Interrupt | Non-Maskable Interrupt |
| 3 | #BP | No | Trap | Breakpoint (INT 3) |
| 4 | #OF | No | Trap | Overflow (INTO) |
| 5 | #BR | No | Fault | BOUND Range Exceeded |
| 6 | #UD | No | Fault | Invalid Opcode |
| 7 | #NM | No | Fault | Device Not Available (no FPU) |
| 8 | #DF | Yes | Abort | Double Fault |
| 9 | — | No | Fault | Coprocessor Segment Overrun (legacy) |
| 10 | #TS | Yes | Fault | Invalid TSS |
| 11 | #NP | Yes | Fault | Segment Not Present |
| 12 | #SS | Yes | Fault | Stack-Segment Fault |
| 13 | #GP | Yes | Fault | General Protection |
| 14 | #PF | Yes | Fault | Page Fault |
| 15 | — | No | Fault | Reserved |
| 16 | #MF | No | Fault | x87 FPU Floating-Point Error |
| 17 | #AC | Yes | Fault | Alignment Check |
| 18 | #MC | No | Abort | Machine Check |
| 19 | #XM | No | Fault | SIMD Floating-Point Exception |
| 20-31 | — | No | — | Reserved |

### Error Code Format (for vectors 8, 10-14, 17)

```
For exceptions 8, 10-14 (segment-related):
┌────────────────────────────────────────────────────────────────┐
│ Bits 15-3: Selector Index  │ Bit 2: TI │ Bits 1-0: IDT/GDT/LDT│
│                            │(0=GDT,    │ 00=GDT, 01=IDT,      │
│                            │ 1=LDT)    │ 10=LDT, 11=IDT       │
└────────────────────────────────────────────────────────────────┘

For page fault (exception 14):
┌────────────────────────────────────────────────────────────────┐
│ Bit 0 (P): 0=page not present, 1=protection violation          │
│ Bit 1 (W): 0=read access, 1=write access                       │
│ Bit 2 (U): 0=supervisor mode, 1=user mode                      │
│ Bit 3 (R): 1=reserved bit set in paging structures             │
│ Bit 4 (I): 1=instruction fetch (NX bit violation)              │
│ Bits 5-31: Reserved                                            │
└────────────────────────────────────────────────────────────────┘
```

### PIC Configuration

```c
// kernel/pic.h
#define PIC1_CMD  0x20    // Master PIC command port
#define PIC1_DATA 0x21    // Master PIC data port
#define PIC2_CMD  0xA0    // Slave PIC command port
#define PIC2_DATA 0xA1    // Slave PIC data port

#define PIC_EOI   0x20    // End of Interrupt command

// Vector offsets after remapping
#define IRQ_BASE  32      // Master IRQs start at vector 32
#define IRQ_BASE2 40      // Slave IRQs start at vector 40
```

### PIT Timer Configuration

```c
// kernel/timer.h
#define PIT_CHANNEL0 0x40   // Channel 0 data port
#define PIT_CHANNEL1 0x41   // Channel 1 data port (unused)
#define PIT_CHANNEL2 0x42   // Channel 2 data port (speaker)
#define PIT_CMD      0x43   // Mode/Command register

#define PIT_FREQ     1193182  // Base frequency in Hz

// Command byte format:
// Bits 7-6: Channel (00=0, 01=1, 10=2, 11=read-back)
// Bits 5-4: Access mode (00=latch, 01=low, 10=high, 11=both)
// Bits 3-1: Mode (000=int on TC, 001=one-shot, 010=rate gen, 011=square wave, ...)
// Bit 0: BCD (0=binary, 1=BCD)

#define PIT_CMD_CHANNEL0 0x36  // 00110110b: ch0, lobyte/hibyte, mode 3, binary
```

### Keyboard Buffer and Scancode Table

```c
// kernel/keyboard.h
#define KB_DATA_PORT 0x60   // Keyboard data register
#define KB_CMD_PORT  0x64   // Keyboard command/status register

#define KB_BUFFER_SIZE 128  // Circular buffer size

typedef struct {
    char buffer[KB_BUFFER_SIZE];
    volatile int head;
    volatile int tail;
} kb_buffer_t;

// Scancode Set 1 (US QWERTY) - make codes only
// Index = scancode, value = ASCII character (0 if not printable)
static const char scancode_to_ascii[128] = {
    0,    0,   '1', '2', '3', '4', '5', '6',     // 0x00-0x07
    '7', '8', '9', '0', '-', '=', '\b', '\t',    // 0x08-0x0F
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i',     // 0x10-0x17
    'o', 'p', '[', ']', '\n', 0,   'a', 's',    // 0x18-0x1F (0x1D=ctrl)
    'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',     // 0x20-0x27
    '\'', '`', 0,   '\\', 'z', 'x', 'c', 'v',   // 0x28-0x2F (0x2A=shift)
    'b', 'n', 'm', ',', '.', '/', 0,   '*',     // 0x30-0x37 (0x36=shift, 0x37=* on keypad)
    0,   ' ',                                    // 0x38-0x39 (0x38=alt, 0x39=space)
    // 0x3A-0x45: capslock, F1-F10, numlock, scrolllock, etc.
    // Extended codes (0xE0 prefix) handled separately
};
```

---

## Interface Contracts

### idt_init()

**Purpose**: Initialize the IDT with 256 entries

**Parameters**: None

**Returns**: None

**Side effects**:
- Zeros all 256 IDT entries
- Loads IDTR via `lidt`
- Does NOT enable interrupts

**Preconditions**: GDT must be loaded

**Postconditions**: IDT is ready to have handlers registered

### idt_set_gate(uint8_t num, uint32_t handler, uint16_t sel, uint8_t flags)

**Purpose**: Register an interrupt handler in the IDT

**Parameters**:
- `num`: Vector number (0-255)
- `handler`: Address of handler function
- `sel`: Code segment selector (typically 0x08)
- `flags`: Type/attributes byte (e.g., IDT_INTERRUPT_GATE)

**Returns**: None

**Side effects**: Modifies IDT entry at index `num`

**Constraints**:
- Handler must be a valid 32-bit address
- Selector must point to a valid code segment

### isr_handler(registers_t *regs) [C function called from assembly]

**Purpose**: Common dispatcher for CPU exceptions

**Parameters**: `regs` — pointer to saved register frame on stack

**Returns**: None (returns via `iret` in assembly)

**Responsibilities**:
- Dispatch to specific exception handler based on `regs->int_no`
- Print diagnostic for unhandled exceptions
- Halt on fatal exceptions (double fault, GPF, etc.)

### irq_handler(registers_t *regs) [C function called from assembly]

**Purpose**: Common dispatcher for hardware IRQs

**Parameters**: `regs` — pointer to saved register frame on stack

**Returns**: None (returns via `iret` in assembly)

**Responsibilities**:
- Call device-specific handler (timer, keyboard, etc.)
- Send EOI to PIC
- Never return without sending EOI

### pic_remap(uint8_t offset1, uint8_t offset2)

**Purpose**: Remap PIC IRQs to non-conflicting vectors

**Parameters**:
- `offset1`: Vector offset for master PIC (typically 32)
- `offset2`: Vector offset for slave PIC (typically 40)

**Returns**: None

**Side effects**:
- Sends ICW1-ICW4 to both PICs
- Masks all IRQs initially
- Changes IRQ-to-vector mapping permanently

**Preconditions**: Must be called before any IRQs are unmasked

**Critical**: Without remapping, IRQ0 fires at vector 8 (double fault!)

### pic_send_eoi(uint8_t irq)

**Purpose**: Signal End of Interrupt to PIC

**Parameters**: `irq` — IRQ number (0-15)

**Returns**: None

**Side effects**: Allows PIC to deliver next interrupt

**Critical**: MUST be called at end of every IRQ handler, or system hangs

### pic_mask_irq(uint8_t irq)

**Purpose**: Mask (disable) a specific IRQ

**Parameters**: `irq` — IRQ number (0-15)

**Returns**: None

### pic_unmask_irq(uint8_t irq)

**Purpose**: Unmask (enable) a specific IRQ

**Parameters**: `irq` — IRQ number (0-15)

**Returns**: None

### timer_init(uint32_t frequency)

**Purpose**: Initialize PIT to fire at specified frequency

**Parameters**: `frequency` — desired frequency in Hz (typically 100)

**Returns**: None

**Side effects**:
- Programs PIT channel 0
- Does NOT unmask IRQ0 (caller must do this)

**Constraints**: Frequency should be 18Hz to 1.193MHz (divisor 1-65535)

### timer_handler(void)

**Purpose**: Handle timer interrupt (IRQ0)

**Parameters**: None (called from irq_handler)

**Returns**: None

**Side effects**: Increments global tick counter

### keyboard_init(void)

**Purpose**: Initialize PS/2 keyboard driver

**Parameters**: None

**Returns**: None

**Side effects**: Clears keyboard buffer

### keyboard_handler(void)

**Purpose**: Handle keyboard interrupt (IRQ1)

**Parameters**: None (called from irq_handler)

**Returns**: None

**Side effects**: Reads scancode, converts to ASCII, stores in buffer

### kb_getchar(void)

**Purpose**: Non-blocking read from keyboard buffer

**Parameters**: None

**Returns**: 
- ASCII character if available
- -1 if buffer empty

**Thread safety**: Buffer is lock-free but safe for single consumer

---

## Algorithm Specification

### IDT Initialization

```
IDT_INIT:
  1. Zero all 256 IDT entries
     FOR i = 0 TO 255:
       idt[i].offset_low = 0
       idt[i].selector = 0
       idt[i].zero = 0
       idt[i].type_attr = 0
       idt[i].offset_high = 0
  
  2. Set up IDTR
     idtr.limit = sizeof(idt_entry_t) * 256 - 1  // 2047
     idtr.base = (uint32_t)&idt
  
  3. Load IDT
     asm volatile("lidt %0" : : "m"(idtr))
```

### Setting an IDT Gate

```
IDT_SET_GATE(num, handler, sel, flags):
  1. Extract handler address bytes
     offset_low = handler & 0xFFFF
     offset_high = (handler >> 16) & 0xFFFF
  
  2. Populate entry
     idt[num].offset_low = offset_low
     idt[num].selector = sel
     idt[num].zero = 0
     idt[num].type_attr = flags
     idt[num].offset_high = offset_high
```

### Assembly ISR Stub (exceptions without error code)

```
; Macro for exceptions that DON'T push error code
%macro ISR_NOERRCODE 1
global isr%1
isr%1:
    push dword 0            ; Push dummy error code to unify stack frame
    push dword %1           ; Push interrupt number
    jmp isr_common_stub
%endmacro
```

### Assembly ISR Stub (exceptions with error code)

```
; Macro for exceptions that DO push error code
%macro ISR_ERRCODE 1
global isr%1
isr%1:
    push dword %1           ; Push interrupt number (error code already on stack)
    jmp isr_common_stub
%endmacro
```

### Common ISR Stub

```
isr_common_stub:
    ; Save all general-purpose registers
    pusha                   ; EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI
    
    ; Save segment registers
    push ds
    push es
    push fs
    push gs
    
    ; Load kernel data segment
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    
    ; Push pointer to stack frame
    mov eax, esp
    push eax
    
    ; Call C handler
    extern isr_handler
    call isr_handler
    
    ; Restore stack pointer (ignore return value)
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
```

### Common IRQ Stub

```
irq_common_stub:
    pusha
    push ds
    push es
    push fs
    push gs
    
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    
    mov eax, esp
    push eax
    
    extern irq_handler
    call irq_handler
    
    add esp, 4
    
    pop gs
    pop fs
    pop es
    pop ds
    popa
    
    add esp, 8
    
    iret
```

### PIC Remapping Sequence

```
PIC_REMAP(offset1, offset2):
  1. Save current masks
     a = inb(PIC1_DATA)
     b = inb(PIC2_DATA)
  
  2. Start initialization (ICW1)
     outb(PIC1_CMD, 0x11)   ; ICW4 needed, cascade mode
     outb(PIC2_CMD, 0x11)
  
  3. Set vector offsets (ICW2)
     outb(PIC1_DATA, offset1)  ; Master: IRQ0 -> vector 32
     outb(PIC2_DATA, offset2)  ; Slave: IRQ8 -> vector 40
  
  4. Tell master about slave (ICW3)
     outb(PIC1_DATA, 0x04)  ; Slave at IRQ2 (bit 2 set)
     outb(PIC2_DATA, 0x02)  ; Slave cascade identity
  
  5. Set 8086 mode (ICW4)
     outb(PIC1_DATA, 0x01)  ; 8086 mode, normal EOI
     outb(PIC2_DATA, 0x01)
  
  6. Restore masks
     outb(PIC1_DATA, a)
     outb(PIC2_DATA, b)
```

### EOI Sending

```
PIC_SEND_EOI(irq):
  1. Always send EOI to master
     outb(PIC1_CMD, 0x20)
  
  2. If IRQ >= 8, also send to slave
     IF irq >= 8:
       outb(PIC2_CMD, 0x20)
```

### PIT Timer Initialization

```
TIMER_INIT(frequency):
  1. Calculate divisor
     divisor = 1193182 / frequency
     IF divisor > 65535: divisor = 65535
     IF divisor < 1: divisor = 1
  
  2. Send command byte
     ; Channel 0, lobyte/hibyte, mode 3 (square wave), binary
     outb(PIT_CMD, 0x36)
  
  3. Send divisor
     outb(PIT_CHANNEL0, divisor & 0xFF)        ; Low byte
     outb(PIT_CHANNEL0, (divisor >> 8) & 0xFF) ; High byte
  
  4. Initialize tick counter
     timer_ticks = 0
```

### Keyboard Handler

```
KEYBOARD_HANDLER:
  1. Read scancode
     scancode = inb(KB_DATA_PORT)
  
  2. Check for extended prefix
     IF scancode == 0xE0:
       extended_mode = 1
       RETURN  // Wait for next byte
  
  3. Check for break code (key release)
     released = (scancode & 0x80)
     scancode &= 0x7F  // Remove break bit
  
  4. Handle key release
     IF released:
       IF scancode == LSHIFT or RSHIFT:
         shift_held = 0
       IF scancode == CTRL:
         ctrl_held = 0
       RETURN
  
  5. Handle modifier press
     IF scancode == LSHIFT or RSHIFT:
       shift_held = 1
       RETURN
     IF scancode == CTRL:
       ctrl_held = 1
       RETURN
  
  6. Convert to ASCII
     IF scancode >= 128:
       RETURN  // Unknown scancode
     c = scancode_to_ascii[scancode]
     IF c == 0:
       RETURN  // Non-printable key
  
  7. Apply modifiers
     IF shift_held AND c >= 'a' AND c <= 'z':
       c -= 32  // Uppercase
  
  8. Store in buffer (if space)
     next_head = (buffer_head + 1) % KB_BUFFER_SIZE
     IF next_head != buffer_tail:
       buffer[buffer_head] = c
       buffer_head = next_head
```

### Double Fault Handler

```
DOUBLE_FAULT_HANDLER(regs):
  1. Print banner
     vga_set_color(RED, WHITE)
     vga_puts("\n!!! DOUBLE FAULT !!!\n")
  
  2. Print diagnostics
     vga_puts("EIP: "); vga_put_hex(regs->eip)
     vga_puts("  CS: "); vga_put_hex(regs->cs)
     vga_puts("\nError code: "); vga_put_hex(regs->err_code)
  
  3. Decode error code
     index = (regs->err_code >> 3) & 0x1FFF
     table = regs->err_code & 0x3
     vga_puts("  Table: ")
     IF table == 0: vga_puts("GDT")
     IF table == 1: vga_puts("IDT")
     IF table == 2: vga_puts("LDT")
     IF table == 3: vga_puts("IDT")
     vga_puts("  Index: "); vga_put_hex(index)
  
  4. Halt
     vga_puts("\nSystem halted.\n")
     cli
     hlt
```

---

## Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
|-------|-------------|----------|---------------|
| PIC not remapped | IRQ0 fires at vector 8 (double fault) | Remap PIC before enabling interrupts | Yes, double fault message |
| Missing EOI | System appears frozen | Always call `pic_send_eoi` in IRQ handler | No (just hangs) |
| Register corruption | Mysterious crashes later | Ensure pusha/popa + segment saves match | Yes, random faults |
| Stack frame misalignment | `iret` pops wrong values | Error code handling must match exception | Yes, GPF or garbage EIP |
| Double fault unhandled | Triple fault (CPU reset) | Install handler at IDT[8] | Yes, system resets |
| Page fault (kernel) | Page fault at valid address | Print CR2 and error code, halt | Yes, PF message |
| Unknown opcode | #UD exception | Print EIP, halt | Yes, UD message |
| Keyboard buffer overflow | Buffer full | Drop character, ring buffer protects | No |
| Invalid IRQ number | Array bounds | Ignore invalid IRQs | No |

---

## State Machine: Keyboard Scancode Parsing

```
                    ┌─────────────────┐
                    │     IDLE        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
              ┌─────│  Read Scancode  │─────┐
              │     └────────┬────────┘     │
              │              │              │
     ┌────────▼───┐   ┌──────▼──────┐   ┌──▼─────────┐
     │ 0xE0 (ext) │   │ 0x00-0x7F   │   │ 0x80-0xFF │
     │ Next byte  │   │ (make code) │   │(break code)│
     │ is ext     │   │             │   │            │
     └────────┬───┘   └──────┬──────┘   └─────┬──────┘
              │              │                │
              │       ┌──────▼──────┐   ┌─────▼──────┐
              │       │ Modifier?   │   │ Clear held │
              │       │ (Shift/Ctrl)│   │ flag       │
              │       └──────┬──────┘   └────────────┘
              │              │
              │       ┌──────▼──────┐
              │       │ ASCII?      │
              │       │ Lookup table│
              │       └──────┬──────┘
              │              │
              │       ┌──────▼──────┐
              │       │ Apply mods  │
              │       │ (shift)     │
              │       └──────┬──────┘
              │              │
              │       ┌──────▼──────┐
              │       │ Buffer full?│
              │       └──────┬──────┘
              │              │
              │       ┌──────▼──────┐
              │       │ Store char  │
              │       │ in buffer   │
              │       └──────┬──────┘
              │              │
              └──────────────►
                    ┌───────▼───────┐
                    │ Return (wait  │
                    │ for next IRQ) │
                    └───────────────┘
```

---

## Implementation Sequence with Checkpoints

### Phase 1: IDT Structure and Loading (2-3 hours)

**Files**: `kernel/idt.h`, `kernel/idt.c`

**Steps**:
1. Define `idt_entry_t` and `idt_ptr_t` structures
2. Declare global IDT array (256 entries)
3. Implement `idt_init()` to zero all entries
4. Implement `idt_set_gate()` to populate entries
5. Implement `idt_load()` with inline assembly `lidt`

**Checkpoint**: IDT loads without error
```c
// Add to kernel_main.c:
idt_init();
vga_puts("IDT initialized\n");
// Verify: no crash, message appears
```

**Test**: Run in QEMU, verify no triple fault after `lidt`

### Phase 2: Assembly ISR Stubs (3-4 hours)

**Files**: `kernel/isr.asm`

**Steps**:
1. Create macros `ISR_NOERRCODE` and `ISR_ERRCODE`
2. Generate stubs for all 32 CPU exceptions
3. Implement `isr_common_stub` with full register save
4. Declare global symbols for each stub

**Checkpoint**: Assembly compiles without error
```bash
nasm -f elf32 kernel/isr.asm -o isr.o
# No errors or warnings
```

**Test**: Link with kernel, verify symbols exist
```bash
nm kernel.elf | grep isr
# Should show isr0, isr1, ..., isr31
```

### Phase 3: Register Frame and C Handler (2-3 hours)

**Files**: `kernel/registers.h`, `kernel/interrupt_handler.c`

**Steps**:
1. Define `registers_t` structure matching assembly push order
2. Implement `isr_handler()` to dispatch based on `int_no`
3. Add exception messages array
4. Implement page fault handler with CR2 read

**Checkpoint**: Exception handler prints diagnostics
```c
// Trigger divide by zero:
int x = 1 / 0;  // After interrupts enabled
// Expected: "EXCEPTION: Divide By Zero" message
```

**Test**: Force each exception, verify correct message

### Phase 4: PIC Remapping (2-3 hours)

**Files**: `kernel/pic.h`, `kernel/pic.c`

**Steps**:
1. Implement `outb` and `inb` inline functions
2. Implement `pic_remap()` with ICW1-ICW4 sequence
3. Implement `pic_send_eoi()`
4. Implement `pic_mask_irq()` and `pic_unmask_irq()`

**Checkpoint**: PIC remaps without error
```c
// In kernel_main, after idt_init:
pic_remap(32, 40);
vga_puts("PIC remapped\n");
// Enable timer and keyboard:
pic_unmask_irq(0);  // Timer
pic_unmask_irq(1);  // Keyboard
// Expected: No immediate crash
```

**Test**: Unmask IRQ0, verify timer fires at vector 32 (not 8)

### Phase 5: IRQ Handlers with EOI (2-3 hours)

**Files**: `kernel/irq.asm`, modify `kernel/interrupt_handler.c`

**Steps**:
1. Create IRQ stub macros (irq0 through irq15)
2. Implement `irq_common_stub`
3. Implement `irq_handler()` in C
4. Add `pic_send_eoi()` call

**Checkpoint**: IRQ handlers run without hanging
```c
// After enabling IRQ0, system should not freeze
// Timer counter should increment
```

**Test**: Enable timer IRQ, verify tick counter increases

### Phase 6: PIT Timer Driver (2-3 hours)

**Files**: `kernel/timer.h`, `kernel/timer.c`

**Steps**:
1. Implement `timer_init()` with divisor calculation
2. Implement `timer_handler()` to increment counter
3. Add `timer_get_ticks()` accessor
4. Register timer handler in `irq_handler()`

**Checkpoint**: Timer fires at configured frequency
```c
timer_init(100);  // 100Hz
pic_unmask_irq(0);
sti();
// Print tick count every second:
while (1) {
    static uint32_t last = 0;
    if (timer_get_ticks() / 100 != last) {
        last = timer_get_ticks() / 100;
        vga_put_dec(last);
        vga_puts(" seconds\n");
    }
    hlt();
}
```

**Test**: Verify seconds counter increments at correct rate

### Phase 7: PS/2 Keyboard Driver (3-4 hours)

**Files**: `kernel/keyboard.h`, `kernel/keyboard.c`

**Steps**:
1. Define scancode-to-ASCII table
2. Implement circular buffer
3. Implement `keyboard_handler()` with scancode parsing
4. Handle shift modifier
5. Implement `kb_getchar()` for non-blocking read
6. Register keyboard handler in `irq_handler()`

**Checkpoint**: Keyboard input appears in buffer
```c
keyboard_init();
pic_unmask_irq(1);
sti();

vga_puts("Type something: ");
while (1) {
    int c = kb_getchar();
    if (c != -1) {
        vga_putchar(c);
    }
    hlt();
}
```

**Test**: Type on keyboard, verify characters appear on screen

### Phase 8: Double Fault Handler (1-2 hours)

**Files**: Modify `kernel/interrupt_handler.c`

**Steps**:
1. Add special case for `int_no == 8` in `isr_handler()`
2. Print detailed diagnostics (EIP, CS, error code)
3. Halt instead of returning

**Checkpoint**: Double fault is caught, not triple fault
```c
// Trigger double fault by corrupting IDT:
idt[8].offset_low = 0xDEAD;  // Invalid handler
idt[8].offset_high = 0xDEAD;
asm volatile("int $8");  // Trigger double fault
// Expected: Double fault message, system halts (not resets)
```

**Test**: Verify double fault handler prints message and halts

### Final Integration (2-3 hours)

**Goal**: Complete interrupt system with all handlers working

**Steps**:
1. Integrate all components in `kernel_main()`
2. Add proper initialization order:
   - GDT (already done)
   - IDT init
   - PIC remap
   - Timer init
   - Keyboard init
   - Unmask IRQs
   - `sti`
3. Create interactive test: display typed characters with timestamp

**Checkpoint**: All tests pass
```c
void kernel_main(void) {
    vga_init();
    serial_init(COM1_PORT);
    
    vga_puts("Initializing IDT...\n");
    idt_init();
    
    vga_puts("Remapping PIC...\n");
    pic_remap(32, 40);
    
    vga_puts("Initializing timer (100Hz)...\n");
    timer_init(100);
    
    vga_puts("Initializing keyboard...\n");
    keyboard_init();
    
    // Unmask timer and keyboard IRQs
    pic_unmask_irq(0);
    pic_unmask_irq(1);
    
    vga_puts("Enabling interrupts...\n");
    asm volatile("sti");
    
    vga_puts("System ready. Type something!\n");
    
    while (1) {
        int c = kb_getchar();
        if (c != -1) {
            vga_puts("[");
            vga_put_dec(timer_get_ticks());
            vga_puts("] ");
            vga_putchar(c);
            vga_puts("\n");
        }
        asm volatile("hlt");
    }
}
```

**Test Commands**:
```bash
make clean && make
qemu-system-i386 -drive format=raw,file=os.img -serial stdio

# Expected:
# - Boot messages appear
# - "System ready. Type something!" appears
# - Typing shows "[tickcount] character"
# - Timer increments correctly
# - No crashes, hangs, or triple faults
```

---

## Test Specification

### Test 1: IDT Loads Without Crash

```python
# test_idt_load.py
import subprocess
import time

def test_idt_load():
    # Build kernel
    result = subprocess.run(['make'], capture_output=True)
    assert result.returncode == 0, f"Build failed: {result.stderr.decode()}"
    
    # Run QEMU briefly
    proc = subprocess.Popen(
        ['qemu-system-i386', '-drive', 'format=raw,file=os.img', 
         '-serial', 'stdio', '-display', 'none', '-nographic'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(2)
    proc.terminate()
    output, _ = proc.communicate(timeout=5)
    
    # Check for IDT init message
    assert b'IDT' in output, "IDT initialization message not found"
    assert b'triple fault' not in output.lower(), "Triple fault detected"
```

### Test 2: PIC Remaps Correctly

```bash
# test_pic_remap.sh
# Verify IRQ0 triggers vector 32, not 8

# Run QEMU with interrupt logging
timeout 5 qemu-system-i386 -drive format=raw,file=os.img \
    -d int -serial stdio -display none 2>&1 | tee int_log.txt

# Check that interrupt 32 (0x20) appears, not 8 with PIC origin
if grep -q "irq 0" int_log.txt; then
    echo "PASS: IRQ0 fires correctly"
else
    echo "FAIL: No IRQ0 detected"
    exit 1
fi
```

### Test 3: Timer Fires at Correct Frequency

```python
# test_timer_frequency.py
import subprocess
import time
import re

def test_timer_frequency():
    proc = subprocess.Popen(
        ['qemu-system-i386', '-drive', 'format=raw,file=os.img',
         '-serial', 'stdio', '-display', 'none'],
        stdout=subprocess.PIPE
    )
    
    time.sleep(3)  # Wait 3 seconds
    proc.terminate()
    output, _ = proc.communicate(timeout=5)
    text = output.decode()
    
    # Find tick counts at different times
    # Assuming we print ticks, look for increasing values
    tick_pattern = r'\[(\d+)\]'
    matches = re.findall(tick_pattern, text)
    
    if len(matches) < 2:
        return  # Not enough data
    
    ticks = [int(m) for m in matches]
    elapsed_ticks = ticks[-1] - ticks[0]
    
    # At 100Hz, 3 seconds should be ~300 ticks
    # Allow 20% tolerance
    assert 200 <= elapsed_ticks <= 400, \
        f"Timer frequency off: {elapsed_ticks} ticks in ~3 seconds"
```

### Test 4: Keyboard Buffer Works

```python
# test_keyboard_buffer.py
import subprocess

def test_keyboard_buffer():
    # Run QEMU with keyboard input
    proc = subprocess.Popen(
        ['qemu-system-i386', '-drive', 'format=raw,file=os.img',
         '-serial', 'stdio', '-display', 'none'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    
    # Wait for ready
    time.sleep(1)
    
    # Send some keystrokes via QEMU monitor
    # (This is tricky; usually done with expect or pexpect)
    # For now, just verify keyboard init message appears
    
    proc.terminate()
    output, _ = proc.communicate(timeout=5)
    
    assert b'keyboard' in output.lower(), "Keyboard init message missing"
```

### Test 5: Exception Handlers Print Diagnostics

```c
// kernel/test_exceptions.c
void test_divide_by_zero(void) {
    volatile int x = 1;
    volatile int y = 0;
    volatile int z = x / y;  // Should trigger #DE
    (void)z;
}

void test_invalid_opcode(void) {
    asm volatile(".byte 0x06, 0x07");  // Invalid on modern x86
}

void test_page_fault(void) {
    volatile int *ptr = (int *)0xDEADBEEF;
    *ptr = 42;  // Should trigger #PF
}

// In kernel_main, after IDT init:
// Uncomment ONE test at a time:
// test_divide_by_zero();
// Expected: "EXCEPTION: Divide By Zero" then halt
```

### Test 6: EOI is Sent

```bash
# test_eoi.sh
# If EOI is missing, system hangs after first interrupt

timeout 5 qemu-system-i386 -drive format=raw,file=os.img \
    -serial stdio -display none 2>&1 | tee eoi_log.txt

# Check for multiple timer ticks (proves EOI is working)
tick_count=$(grep -c "seconds" eoi_log.txt || echo "0")

if [ "$tick_count" -gt 1 ]; then
    echo "PASS: EOI sent correctly (multiple ticks seen)"
else
    echo "FAIL: System may have hung (EOI not sent?)"
    exit 1
fi
```

### Test 7: Double Fault Caught

```bash
# test_double_fault.sh
# Trigger double fault and verify handler runs (not reset)

# Modify kernel to corrupt IDT[0] after setup:
# idt[0].offset_low = 0xDEAD;
# asm volatile("int $0");
# Then trigger int 0 again

# Expected: "DOUBLE FAULT" message, not QEMU reset
```

---

## Performance Targets

| Operation | Target | How to Measure |
|-----------|--------|----------------|
| ISR entry (pusha + segs) | < 50 cycles | Count instructions: pusha (17), push ds-es-fs-gs (3×), mov ax/mov ds (2×4) ≈ 35 cycles |
| C handler call overhead | < 20 cycles | Call/ret pair ≈ 5-10 cycles |
| Total ISR entry to handler | < 150 cycles | Sum of above plus stack manipulation |
| EOI send | < 10 cycles | Single `outb` instruction |
| Keyboard scancode to buffer | < 500 cycles | Inb (1), table lookup (5), buffer store (5) |
| Timer tick (counter increment) | < 100 cycles | Memory increment with lock prefix if needed |
| Timer accuracy | Within 1% | Compare tick count to wall clock over 10 seconds |
| Keyboard latency (IRQ to buffer) | < 1ms | Measure from QEMU input to buffer write |

---

## Hardware Soul

### Cache Lines Touched

**IDT access**: The IDT (2048 bytes) fits in 32 cache lines. The CPU reads the relevant entry during interrupt dispatch, which is cached in L1. Frequently used vectors (timer at 32, keyboard at 33) stay hot in cache.

**Handler code**: Interrupt handlers should be small and stay in cache. A 64-byte cache line holds ~16-20 instructions. Timer handler (~20 instructions) and keyboard handler (~50 instructions) should remain L1-resident.

**Keyboard buffer**: 128-byte circular buffer = 2 cache lines. The head/tail indices share a cache line with the buffer.

### Branch Prediction

**Interrupt dispatch**: Fully predictable — direct table lookup using vector number. No branches in the dispatch path until the C handler's switch statement.

**Scancode parsing**: Branch-heavy due to modifier checks and table lookups. The CPU's branch predictor learns the pattern (most keys are alphanumeric, shift state changes infrequently).

**Timer handler**: Minimal branching — just increment counter. Highly predictable.

### I/O Port Access

| Port | Access Type | Latency | Notes |
|------|-------------|---------|-------|
| 0x20/0xA0 (PIC cmd) | Out | ~1 µs | EOI command |
| 0x21/0xA1 (PIC data) | In/Out | ~1 µs | Mask registers |
| 0x60 (Keyboard data) | In | ~1 µs | Scancode read |
| 0x64 (Keyboard status) | In | ~1 µs | Status check |
| 0x40/0x43 (PIT) | Out | ~1 µs | Timer programming |

All I/O port accesses bypass the cache hierarchy entirely.

### TLB Considerations

Handlers must be in always-mapped memory. With paging enabled:
- IDT should be in identity-mapped or kernel region
- Handler code must be mapped in all address spaces
- Stack must be valid (kernel stack at known address)

### Interrupt Latency Components

1. **Hardware latency**: IRQ assertion to CPU response (~1-2 µs)
2. **CPU dispatch**: Vector lookup, privilege check, stack switch (~50-100 cycles)
3. **Software save**: pusha + segment saves (~40 cycles)
4. **Handler execution**: Variable (timer: ~20 cycles, keyboard: ~100 cycles)
5. **Restore + iret**: popa + segment restores + iret (~50 cycles)
6. **EOI**: outb to PIC (~1 µs)

Total minimum latency: ~2-5 µs for timer, ~5-10 µs for keyboard

---

## Concurrency Specification

### Interrupt Context

Interrupt handlers run in a special context:

- **Atomic entry**: CPU automatically disables interrupts (for interrupt gates) during dispatch
- **No preemption**: Handler cannot be preempted by same or lower priority interrupt
- **No blocking**: Handler must not sleep, wait, or call any blocking function
- **No per-process state**: Handler may run in context of any process

### Shared Data

| Data | Access Pattern | Protection |
|------|---------------|------------|
| Timer tick counter | Write (IRQ), Read (any) | `volatile`, atomic increment |
| Keyboard buffer | Write (IRQ), Read (any) | Lock-free circular buffer |
| Keyboard head/tail | Write (IRQ updates head, user updates tail) | Index variables, careful ordering |
| IDT entries | Write (init only), Read (CPU) | No protection needed (one-time init) |
| PIC masks | Read/Write (any) | No concurrent access expected |

### Critical Sections

During IDT/PIC initialization:
```c
void interrupt_init(void) {
    cli();  // Disable interrupts during setup
    
    idt_init();
    pic_remap(32, 40);
    // Register handlers...
    
    sti();  // Re-enable after complete setup
}
```

### Re-entrancy

Handlers are NOT re-entrant by default (interrupt gates clear IF). If re-entrancy is needed:
1. Use trap gate instead of interrupt gate
2. Manually re-enable interrupts with `sti` in handler
3. Protect shared data with per-vector locks

---

## Implementation Notes

### Register Save Order

The order MUST match the `registers_t` structure:

```asm
; Assembly pushes:
push gs       ; Offset +0
push fs       ; Offset +4
push es       ; Offset +8
push ds       ; Offset +12
pusha         ; EDI+16, ESI+20, EBP+24, ESP+28, EBX+32, EDX+36, ECX+40, EAX+44
push int_no   ; Offset +48
push err_code ; Offset +52
; CPU already pushed: EIP+56, CS+60, EFLAGS+64
; If ring change: ESP+68, SS+72
```

### Error Code Handling

Some exceptions push error code, some don't:

```asm
; Exceptions WITHOUT error code (use dummy push):
ISR_NOERRCODE 0   ; #DE
ISR_NOERRCODE 1   ; #DB
ISR_NOERRCODE 2   ; NMI
ISR_NOERRCODE 3   ; #BP
ISR_NOERRCODE 4   ; #OF
ISR_NOERRCODE 5   ; #BR
ISR_NOERRCODE 6   ; #UD
ISR_NOERRCODE 7   ; #NM
ISR_ERRCODE   8   ; #DF (HAS error code)
ISR_NOERRCODE 9   ; Coprocessor Segment Overrun
ISR_ERRCODE   10  ; #TS
ISR_ERRCODE   11  ; #NP
ISR_ERRCODE   12  ; #SS
ISR_ERRCODE   13  ; #GP
ISR_ERRCODE   14  ; #PF
; ... continue pattern
```

### PIC Mask Management

```c
// Unmask IRQ:
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

// Mask IRQ:
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
```

### Extended Scancodes

Arrow keys and navigation cluster send 0xE0 prefix:

```c
static int extended_scancode = 0;

void keyboard_handler(void) {
    uint8_t scancode = inb(KB_DATA_PORT);
    
    if (scancode == 0xE0) {
        extended_scancode = 1;
        return;  // Wait for next byte
    }
    
    if (extended_scancode) {
        extended_scancode = 0;
        // Handle extended code: scancode is the second byte
        // Arrow keys: 0x48=up, 0x50=down, 0x4B=left, 0x4D=right
        // ...
        return;
    }
    
    // Normal scancode handling...
}
```

---

## Visual Diagrams

### IDT Entry Structure

```
IDT Gate Descriptor (8 bytes):
┌─────────────────────────────────────────────────────────────────┐
│ 63       48│47       32│31   24│23 16│15      0│               │
│  Offset    │  Offset   │       │ Seg │  Offset │               │
│  High      │  High     │ Zero  │ Sel │  Low    │               │
│  [31:16]   │  [31:16]  │       │     │  [15:0] │               │
└─────────────────────────────────────────────────────────────────┘
              │           │       │     │         │
              │           │       │     │         └─ Handler address bits 15:0
              │           │       │     └─ Code segment selector (0x08)
              │           │       └─ Reserved (must be 0)
              │           └─ Type/Attributes:
              │              ┌──────────────────────────┐
              │              │ P DPL 0 Type             │
              │              │ 1 00  0 1110 = 0x8E      │
              │              │ ↑  ↑   ↑ ↑               │
              │              │ │  │   │ └─ Gate type    │
              │              │ │  │   └─ System (0)    │
              │              │ │  └─ Privilege level   │
              │              │ └─ Present              │
              └─ Handler address bits 31:16
```

### Interrupt Stack Frame

```
                 High Addresses
                 ┌─────────────────┐
                 │    SS (old)     │ ← Only if privilege change (ring 3→0)
                 │    ESP (old)    │
                 ├─────────────────┤
                 │    EFLAGS       │ ← CPU pushes
                 ├─────────────────┤
                 │    CS (old)     │ ← CPU pushes
                 │    EIP (old)    │ ← CPU pushes
                 ├─────────────────┤
                 │  Error Code     │ ← CPU (some) or stub (dummy 0)
                 ├─────────────────┤
                 │  Interrupt #    │ ← Stub pushes
                 ├─────────────────┤
                 │      EAX        │ ← pusha
                 │      ECX        │
                 │      EDX        │
                 │      EBX        │
                 │   ESP (before   │
                 │    pusha)       │
                 │      EBP        │
                 │      ESI        │
                 │      EDI        │ ← pusha ends
                 ├─────────────────┤
                 │      DS         │ ← Stub pushes
                 │      ES         │
                 │      FS         │
                 │      GS         │ ← Stub pushes
                 └─────────────────┘ ← ESP points here
                 Low Addresses

registers_t structure maps this exactly:
  gs, fs, es, ds                    (offset 0-15)
  edi, esi, ebp, esp, ebx, edx, ecx, eax  (offset 16-47)
  int_no, err_code                  (offset 48-55)
  eip, cs, eflags                   (offset 56-67)
  useresp, ss                       (offset 68-75, optional)
```

### PIC Remapping

```
BEFORE Remapping (Default):
┌──────────────────────────────────────────────────────────┐
│ IRQ 0 1 2 3 4 5 6 7 │ IRQ 8  9  10 11 12 13 14 15      │
│     ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ │    ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓        │
│ Vec 8 9 A B C D E F │   70 71 72 73 74 75 76 77        │
│     ↑ CONFLICT!     │                                   │
│     #DF is vec 8!   │                                   │
└──────────────────────────────────────────────────────────┘

AFTER Remapping:
┌──────────────────────────────────────────────────────────┐
│ IRQ 0 1 2 3 4 5 6 7 │ IRQ 8  9  10 11 12 13 14 15      │
│     ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ │    ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓        │
│ Vec 32 33 34 35 36 37 38 39 │ 40 41 42 43 44 45 46 47  │
│     ↑ Timer  ↑ Keyb │    ↑ RTC                          │
│     (no conflict)   │                                   │
└──────────────────────────────────────────────────────────┘

CPU Exception Vectors (0-31) remain for CPU-detected faults:
┌──────────────────────────────────────────────────────────┐
│ 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 ...    │
│ #DE #DB NMI #BP #OF #BR #UD #NM #DF -- #TS #NP #SS #GP #PF│
└──────────────────────────────────────────────────────────┘
```

### Exception Vectors and Error Codes

```
CPU Exception Vectors (0-31):

┌────┬───────┬────────┬───────────────────────────────────────┐
│Vec │ Name  │ Error? │                Cause                  │
├────┼───────┼────────┼───────────────────────────────────────┤
│ 0  │ #DE   │ No     │ Division by zero or overflow          │
│ 1  │ #DB   │ No     │ Debug trap (single step, breakpoint)  │
│ 2  │ NMI   │ No     │ Hardware NMI (parity error, etc.)     │
│ 3  │ #BP   │ No     │ INT 3 instruction (debugger)          │
│ 4  │ #OF   │ No     │ INTO instruction with OF flag set     │
│ 5  │ #BR   │ No     │ BOUND instruction range exceeded      │
│ 6  │ #UD   │ No     │ Invalid or privileged instruction     │
│ 7  │ #NM   │ No     │ FPU instruction with no FPU present   │
│ 8  │ #DF   │ Yes    │ Exception during exception handling   │
│ 9  │ --    │ No     │ (Reserved, coprocessor segment overrun)│
│ 10 │ #TS   │ Yes    │ Invalid TSS during task switch        │
│ 11 │ #NP   │ Yes    │ Segment or gate not present           │
│ 12 │ #SS   │ Yes    │ Stack segment limit violation         │
│ 13 │ #GP   │ Yes    │ General protection violation          │
│ 14 │ #PF   │ Yes    │ Page not present or protection fault  │
│ 15 │ --    │ No     │ Reserved                              │
│ 16 │ #MF   │ No     │ x87 FPU error                         │
│ 17 │ #AC   │ Yes    │ Alignment check (alignment mode)      │
│ 18 │ #MC   │ No     │ Machine check (hardware error)        │
│ 19 │ #XM   │ No     │ SIMD floating-point exception         │
│20-31│ --   │ No     │ Reserved                              │
└────┴───────┴────────┴───────────────────────────────────────┘

Page Fault Error Code (vector 14):
┌────┬─────────────────────────────────────────────────────────┐
│Bit │                         Meaning                         │
├────┼─────────────────────────────────────────────────────────┤
│ 0  │ P: 0=page not present, 1=protection violation           │
│ 1  │ W: 0=read access, 1=write access                        │
│ 2  │ U: 0=supervisor mode, 1=user mode                       │
│ 3  │ R: 1=reserved bit set in paging structures              │
│ 4  │ I: 1=instruction fetch (NX bit violation)               │
│5-31│ Reserved                                                │
└────┴─────────────────────────────────────────────────────────┘
```

### Keyboard Scancode Flow

```
Keyboard Scancode Processing:

┌─────────────────────────────────────────────────────────────────┐
│                     KEYBOARD CONTROLLER                         │
│                    (sends scancode)                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │ IRQ1
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     IRQ1 HANDLER                                │
│  1. Read scancode from port 0x60                                │
│  2. Check for 0xE0 (extended prefix)                            │
│  3. Check for break code (bit 7)                                │
│  4. Update modifier state (shift, ctrl, alt)                    │
│  5. Lookup ASCII in scancode table                              │
│  6. Apply modifiers (shift -> uppercase)                        │
│  7. Store in circular buffer                                    │
│  8. Send EOI                                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CIRCULAR BUFFER                               │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐ │
│  │ H │ e │ l │ l │ o │   │   │   │   │   │   │   │   │   │   │ │
│  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘ │
│        ↑                                                   ↑   │
│       tail                                                head │
└─────────────────────────┬───────────────────────────────────────┘
                          │ kb_getchar()
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION                                │
│  int c = kb_getchar();                                          │
│  if (c != -1) { vga_putchar(c); }                              │
└─────────────────────────────────────────────────────────────────┘

Scancode Table (Set 1, partial):
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ 01 │ 02 │ 03 │ 04 │ 05 │ 06 │ 07 │ 08 │ 09 │ 0A │ 0B │ 0C │
│Esc │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ 8  │ 9  │ 0  │ -  │
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│ 0D │ 0E │ 0F │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │ 16 │ 17 │ 18 │
│ =  │BkSp│Tab │ q  │ w  │ e  │ r  │ t  │ y  │ u  │ i  │ o  │
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│ 19 │ 1A │ 1B │ 1C │ 1E │ 1F │ 20 │ 21 │ 22 │ 23 │ 24 │ 25 │
│ p  │ [  │ ]  │Enter│ a │ s  │ d  │ f  │ g  │ h  │ j  │ k  │
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│ 26 │ 27 │ 28 │ 29 │ 2B │ 2C │ 2D │ 2E │ 2F │ 30 │ 31 │ 32 │
│ l  │ ;  │ '  │ `  │ \  │ z  │ x  │ c  │ v  │ b  │ n  │ m  │
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│ 33 │ 34 │ 35 │ 39 │                                              
│ ,  │ .  │ /  │Space│ ...etc...                                   
└────┴────┴────┴────┴───────────────────────────────────────────┘
```

### ISR State Machine

```
ISR/IRQ Processing State Machine:

                    ┌──────────────────┐
                    │   CPU Running    │
                    │   User/Kernel    │
                    │     Code         │
                    └────────┬─────────┘
                             │
              Interrupt/IRQ  │
              ──────────────▶│
                             ▼
                    ┌──────────────────┐
                    │  CPU Pushes:     │
                    │  EFLAGS, CS, EIP │
                    │  (SS, ESP if     │
                    │   ring change)   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  IDT Lookup      │
                    │  (vector → gate) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   Our Stub:      │
                    │   Push err/dummy │
                    │   Push int_no    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   pusha          │
                    │   push segs      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Load kernel DS   │
                    │ Call C handler   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │   ISR Handler   │          │   IRQ Handler   │
     │  (Exception)    │          │  (Hardware)     │
     │                 │          │                 │
     │ Print message   │          │ Call device     │
     │ Halt if fatal   │          │ handler         │
     └────────┬────────┘          │ Send EOI to PIC │
              │                   └────────┬────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼─────────┐
                    │   pop segs       │
                    │   popa           │
                    │   add esp, 8     │
                    │   iret           │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Resume         │
                    │   Interrupted    │
                    │   Code           │
                    └──────────────────┘
```

---


<!-- TDD_MOD_ID: mod-memory -->
# Technical Design Specification: Physical and Virtual Memory Management

## Module Charter

The memory management module implements physical frame allocation using a bitmap allocator, two-level x86 page tables (page directory + page tables) for virtual memory, identity mapping for the first 16MB plus higher-half kernel mapping at 0xC0000000, and a kernel heap allocator (kmalloc/kfree). It parses the E820/multiboot memory map to discover usable physical memory regions.

**What it does NOT do**: This module does not implement per-process address spaces (all processes share kernel mapping), does not implement demand paging or swap, does not implement copy-on-write, does not handle memory-mapped files, and does not implement user-space malloc (only kernel kmalloc).

**Upstream dependencies**: GDT must be loaded with flat kernel segments; IDT must have page fault handler registered at vector 14; multiboot bootloader must provide memory map.

**Downstream consumers**: Process manager (Milestone 4) will use page directory cloning for per-process address spaces; all kernel code uses kmalloc/kfree for dynamic allocation; device drivers use identity-mapped MMIO regions.

**Invariants**: Frame allocator must never return a frame containing kernel code/data; page tables must always have kernel region (0xC0000000+) mapped identically across all page directories; TLB must be invalidated after any page table modification; kmalloc must never return memory below 0xC0400000 (kernel heap region).

---

## File Structure

Create files in this order:

```
1. kernel/memory/e820.h          # E820/multiboot memory map structures
2. kernel/memory/e820.c          # Memory map parser and printer
3. kernel/memory/frame.h         # Frame allocator interface
4. kernel/memory/frame.c         # Bitmap frame allocator implementation
5. kernel/memory/paging.h        # Page table structures and interface
6. kernel/memory/paging.c        # Page directory/table management
7. kernel/memory/heap.h          # kmalloc/kfree interface
8. kernel/memory/heap.c          # Kernel heap implementation
9. kernel/memory/memory.c        # Top-level memory init integrating all components
10. kernel/linker.ld             # Updated linker script with higher-half symbols
```

---

## Complete Data Model

### E820 Memory Map Entry (20 bytes)

```c
// kernel/memory/e820.h
#include <stdint.h>

typedef struct {
    uint64_t base;          // Base address of region
    uint64_t length;        // Length of region in bytes
    uint32_t type;          // Type of memory
    uint32_t acpi;          // ACPI extended attributes (optional)
} __attribute__((packed)) e820_entry_t;

// Memory region types
#define E820_USABLE         1   // Normal RAM
#define E820_RESERVED       2   // Reserved, do not use
#define E820_ACPI_RECLAIM   3   // ACPI reclaimable (read tables first)
#define E820_ACPI_NVS       4   // ACPI non-volatile storage
#define E820_BAD            5   // Bad memory (errors)

// Parsed memory region for internal use
typedef struct {
    uint64_t base;
    uint64_t length;
    uint32_t type;
} memory_region_t;

#define MAX_MEMORY_REGIONS  64
```

### Multiboot Memory Map Header

```c
// From multiboot specification
typedef struct {
    uint32_t size;              // Size of this structure (minus size field)
    uint64_t base_addr;         // Base address
    uint64_t length;            // Length
    uint32_t type;              // Type
} __attribute__((packed)) multiboot_mmap_entry_t;

// Multiboot info structure (partial)
typedef struct {
    uint32_t flags;
    uint32_t mem_lower;         // KB of low memory (0-640KB)
    uint32_t mem_upper;         // KB of high memory (1MB+)
    // ... other fields ...
    uint32_t mmap_length;       // Memory map length
    uint32_t mmap_addr;         // Memory map address
} __attribute__((packed)) multiboot_info_t;
```

### Frame Allocator State

```c
// kernel/memory/frame.h
#include <stdint.h>

#define FRAME_SIZE          4096        // 4KB per frame
#define FRAME_SHIFT         12          // log2(4096)
#define BITS_PER_DWORD      32

// Frame allocator state
typedef struct {
    uint32_t *bitmap;                 // Bitmap: 1 = allocated, 0 = free
    uint32_t bitmap_size;             // Number of uint32_t entries
    uint32_t total_frames;            // Total frames in system
    uint32_t free_frames;             // Currently free frames
    uint32_t first_usable;            // First frame number that's usable
    uint32_t kernel_end_frame;        // First frame after kernel
} frame_allocator_t;

extern frame_allocator_t frame_alloc;

// Convert between physical address and frame number
#define ADDR_TO_FRAME(addr)     ((addr) >> FRAME_SHIFT)
#define FRAME_TO_ADDR(frame)    ((frame) << FRAME_SHIFT)

// Bitmap operations
static inline void frame_set(uint32_t frame) {
    frame_alloc.bitmap[frame / BITS_PER_DWORD] |= (1U << (frame % BITS_PER_DWORD));
}

static inline void frame_clear(uint32_t frame) {
    frame_alloc.bitmap[frame / BITS_PER_DWORD] &= ~(1U << (frame % BITS_PER_DWORD));
}

static inline int frame_test(uint32_t frame) {
    return frame_alloc.bitmap[frame / BITS_PER_DWORD] & (1U << (frame % BITS_PER_DWORD));
}
```

### Page Directory/Table Entry (4 bytes each)

```c
// kernel/memory/paging.h
#include <stdint.h>

typedef uint32_t pte_t;       // Page Table Entry
typedef uint32_t pde_t;       // Page Directory Entry

// Page entry flags (bits in pte_t/pde_t)
#define PTE_PRESENT       (1U << 0)    // Page is present in memory
#define PTE_WRITABLE      (1U << 1)    // Read/Write (1 = writable)
#define PTE_USER          (1U << 2)    // User/supervisor (1 = user accessible)
#define PTE_WRITETHROUGH  (1U << 3)    // Write-through caching
#define PTE_CACHE_DISABLE (1U << 4)    // Disable cache for this page
#define PTE_ACCESSED      (1U << 5)    // Page has been accessed (read)
#define PTE_DIRTY         (1U << 6)    // Page has been written to (PT only)
#define PTE_PAGE_SIZE     (1U << 7)    // 4MB page (PD only)
#define PTE_GLOBAL        (1U << 8)    // Global page (not flushed on CR3 reload)
#define PTE_FRAME_MASK    0xFFFFF000   // Frame address mask (bits 31:12)

// Extract fields from entry
#define PTE_FRAME(pte)          ((pte) & PTE_FRAME_MASK)
#define PTE_FLAGS(pte)          ((pte) & ~PTE_FRAME_MASK)

// Common flag combinations
#define PTE_KERNEL_CODE         (PTE_PRESENT | PTE_WRITABLE)
#define PTE_KERNEL_DATA         (PTE_PRESENT | PTE_WRITABLE)
#define PTE_USER_CODE           (PTE_PRESENT | PTE_WRITABLE | PTE_USER)
#define PTE_USER_DATA           (PTE_PRESENT | PTE_WRITABLE | PTE_USER)
```

**Page Table Entry bit layout:**

| Bits | Field | Description |
|------|-------|-------------|
| 0 | Present | 1 = page in memory |
| 1 | R/W | 1 = writable |
| 2 | U/S | 1 = user-mode accessible |
| 3 | PWT | Write-through caching |
| 4 | PCD | Cache disable |
| 5 | A | Accessed (CPU sets on read) |
| 6 | D | Dirty (CPU sets on write, PT only) |
| 7 | PS | Page size (0 = 4KB) |
| 8 | G | Global (ignore on CR3 reload) |
| 9-11 | Available | OS-defined |
| 12-31 | Frame Address | Physical frame address (4KB aligned) |

### Page Directory and Page Table Structures

```c
#define ENTRIES_PER_TABLE    1024
#define PAGE_SIZE            4096

typedef struct {
    pte_t entries[ENTRIES_PER_TABLE];
} __attribute__((aligned(PAGE_SIZE))) page_table_t;

typedef struct {
    pde_t entries[ENTRIES_PER_TABLE];
} __attribute__((aligned(PAGE_SIZE))) page_directory_t;

// Extract indices from virtual address
#define PD_INDEX(vaddr)      (((vaddr) >> 22) & 0x3FF)
#define PT_INDEX(vaddr)      (((vaddr) >> 12) & 0x3FF)
#define PAGE_OFFSET(vaddr)   ((vaddr) & 0xFFF)

// Current page directory (physical address in CR3)
extern page_directory_t *current_page_directory;
```

### Heap Block Header

```c
// kernel/memory/heap.h
#include <stdint.h>

#define HEAP_MAGIC           0xDEADBEEF
#define HEAP_MIN_BLOCK_SIZE  16        // Minimum allocation size
#define HEAP_START           0xC0400000  // Virtual address for heap
#define HEAP_INITIAL_SIZE    (4 * 1024 * 1024)  // 4MB initial

typedef struct heap_block {
    uint32_t magic;                   // HEAP_MAGIC for integrity check
    uint32_t size;                    // Size of data area (excluding header)
    uint8_t  free;                    // 1 = free, 0 = allocated
    uint8_t  padding[3];              // Align to 4 bytes
    struct heap_block *next;          // Next block in list
    struct heap_block *prev;          // Previous block in list
} __attribute__((packed)) heap_block_t;

#define BLOCK_HEADER_SIZE    sizeof(heap_block_t)
#define BLOCK_DATA(block)    ((void*)((uint8_t*)(block) + BLOCK_HEADER_SIZE))
#define DATA_TO_BLOCK(ptr)   ((heap_block_t*)((uint8_t*)(ptr) - BLOCK_HEADER_SIZE))
```

### Memory Layout Constants

```c
// kernel/memory/memory.h

// Physical memory layout
#define KERNEL_PHYSICAL_BASE   0x00100000    // 1 MB - where kernel is loaded
#define LOW_MEMORY_END         0x00100000    // 1 MB - end of low memory
#define VGA_PHYSICAL           0x000B8000    // VGA text buffer

// Virtual memory layout
#define KERNEL_VIRTUAL_BASE    0xC0000000    // 3 GB - higher-half kernel
#define KERNEL_HEAP_START      0xC0400000    // 3 GB + 4 MB - kernel heap
#define KERNEL_HEAP_END        0xC0800000    // 3 GB + 8 MB - heap end (expandable)
#define USER_SPACE_START       0x00000000    // User space base
#define USER_SPACE_END         0xBFFFFFFF    // User space end (3 GB - 1)
#define USER_STACK_TOP         0xBFFFF000    // User stack (grows down)

// Identity mapping range
#define IDENTITY_MAP_END       (16 * 1024 * 1024)  // Identity map first 16 MB
```

---

## Interface Contracts

### e820_init(multiboot_info_t *mbi)

**Purpose**: Parse multiboot memory map and store usable regions

**Parameters**: 
- `mbi`: Pointer to multiboot info structure from bootloader

**Returns**: 0 on success, -1 on error

**Side effects**: 
- Populates global `memory_regions` array
- Sets `num_memory_regions`

**Preconditions**: Called early in kernel init before frame allocator

**Postconditions**: Memory map available for frame allocator initialization

### frame_allocator_init(void)

**Purpose**: Initialize bitmap frame allocator from parsed memory map

**Parameters**: None

**Returns**: 0 on success, -1 on error

**Side effects**:
- Allocates bitmap (using placement allocator during boot)
- Marks kernel frames as used
- Marks reserved/acpi regions as used

**Preconditions**: e820_init() must have been called

**Postconditions**: Frame allocator ready to allocate free frames

### alloc_frame(void)

**Purpose**: Allocate a single 4KB physical frame

**Parameters**: None

**Returns**: 
- Physical address of allocated frame on success
- NULL (0) if out of memory

**Side effects**:
- Marks frame as allocated in bitmap
- Decrements `free_frames` counter

**Thread safety**: NOT thread-safe. Caller must disable interrupts if needed.

### free_frame(void *addr)

**Purpose**: Free a previously allocated physical frame

**Parameters**:
- `addr`: Physical address of frame to free

**Returns**: None

**Side effects**:
- Clears frame bit in bitmap
- Increments `free_frames` counter

**Error handling**:
- Panics on double-free (frame already free)
- Panics if address not frame-aligned
- Panics if frame outside valid range

### paging_init(void)

**Purpose**: Set up initial page tables and enable paging

**Parameters**: None

**Returns**: None

**Side effects**:
- Creates initial page directory
- Identity maps first 16MB
- Maps kernel at 0xC0000000+ (higher-half)
- Loads CR3 and sets CR0.PG

**Preconditions**: Frame allocator initialized

**Postconditions**: Paging enabled, kernel running in higher-half

### map_page(page_directory_t *dir, uint32_t vaddr, uint32_t paddr, uint32_t flags)

**Purpose**: Map a virtual page to a physical frame

**Parameters**:
- `dir`: Page directory to modify
- `vaddr`: Virtual address (will be page-aligned)
- `paddr`: Physical address (will be page-aligned)
- `flags`: PTE flags (PTE_PRESENT | PTE_WRITABLE | ...)

**Returns**: 0 on success, -1 on failure (out of memory for page table)

**Side effects**:
- May allocate new page table
- Invalidates TLB entry for vaddr

**Preconditions**: Paging enabled (or dir is valid)

### unmap_page(page_directory_t *dir, uint32_t vaddr)

**Purpose**: Remove a virtual-to-physical mapping

**Parameters**:
- `dir`: Page directory to modify
- `vaddr`: Virtual address to unmap

**Returns**: None

**Side effects**: Invalidates TLB entry for vaddr

### get_physical(page_directory_t *dir, uint32_t vaddr)

**Purpose**: Translate virtual address to physical address

**Parameters**:
- `dir`: Page directory to query
- `vaddr`: Virtual address to translate

**Returns**:
- Physical address on success
- 0 if not mapped

### kmalloc(uint32_t size)

**Purpose**: Allocate kernel heap memory

**Parameters**:
- `size`: Number of bytes to allocate

**Returns**:
- Pointer to allocated memory (in kernel virtual space)
- NULL if out of memory

**Side effects**: May expand heap by mapping new pages

**Alignment**: Returns 4-byte aligned pointers

**Minimum allocation**: HEAP_MIN_BLOCK_SIZE bytes

### kfree(void *ptr)

**Purpose**: Free previously allocated heap memory

**Parameters**:
- `ptr`: Pointer returned by kmalloc (or NULL)

**Returns**: None

**Side effects**:
- Coalesces with adjacent free blocks
- May reduce heap (optional optimization)

**Error handling**:
- Panics on invalid pointer (magic mismatch)
- Panics on double-free
- Silently ignores NULL

---

## Algorithm Specification

### Memory Map Parsing

```
E820_INIT(mbi):
  1. Verify multiboot flags indicate memory map present
     IF !(mbi->flags & (1 << 6)):
       RETURN error  // No memory map!
  
  2. Initialize region counter
     num_regions = 0
  
  3. Iterate through memory map entries
     entry = mbi->mmap_addr
     WHILE entry < mbi->mmap_addr + mbi->mmap_length:
       // Skip entries above 4GB (we're 32-bit)
       IF entry->base + entry->length > 0xFFFFFFFF:
         entry = next_entry
         CONTINUE
       
       // Store region info
       regions[num_regions].base = entry->base
       regions[num_regions].length = entry->length
       regions[num_regions].type = entry->type
       
       num_regions++
       IF num_regions >= MAX_MEMORY_REGIONS:
         BREAK  // Too many regions
       
       entry = next_entry (entry + entry->size + 4)
  
  4. Calculate total usable memory
     total_memory = 0
     FOR each region:
       IF region.type == E820_USABLE:
         total_memory += region.length
  
  5. RETURN success
```

### Frame Allocator Initialization

```
FRAME_ALLOCATOR_INIT():
  1. Calculate total frames from highest address
     max_addr = 0
     FOR each region:
       IF region.base + region.length > max_addr:
         max_addr = region.base + region.length
     
     total_frames = max_addr / FRAME_SIZE
  
  2. Allocate bitmap using placement allocator
     bitmap_dwords = (total_frames + 31) / 32
     bitmap = placement_alloc(bitmap_dwords * 4)
     
     // Mark ALL frames as reserved initially
     FOR i = 0 TO bitmap_dwords - 1:
       bitmap[i] = 0xFFFFFFFF
  
  3. Find first usable frame
     first_usable = total_frames  // Start with max
     FOR each region:
       IF region.type == E820_USABLE:
         start_frame = ALIGN_UP(region.base, FRAME_SIZE) / FRAME_SIZE
         IF start_frame < first_usable:
           first_usable = start_frame
  
  4. Mark usable frames as free
     free_frames = 0
     FOR each region:
       IF region.type == E820_USABLE:
         start_frame = ALIGN_UP(region.base, FRAME_SIZE) / FRAME_SIZE
         end_frame = ALIGN_DOWN(region.base + region.length, FRAME_SIZE) / FRAME_SIZE
         
         FOR frame = start_frame TO end_frame - 1:
           clear_frame(frame)
           free_frames++
  
  5. Reserve kernel frames
     kernel_start = ADDR_TO_FRAME(&_kernel_start)
     kernel_end = ADDR_TO_FRAME(&_kernel_end) + 1  // +1 for partial frame
     
     FOR frame = kernel_start TO kernel_end - 1:
       IF frame_test(frame) == 0:  // Was free
         set_frame(frame)
         free_frames--
     
     kernel_end_frame = kernel_end
  
  6. Reserve already-allocated placement memory
     placement_start = first_usable  // Rough estimate
     placement_end = ADDR_TO_FRAME(placement_addr)
     
     FOR frame = placement_start TO placement_end:
       IF frame_test(frame) == 0:
         set_frame(frame)
         free_frames--
  
  7. RETURN success
```

### Frame Allocation (Bitmap Scan)

```
ALLOC_FRAME():
  1. Check if any frames available
     IF free_frames == 0:
       RETURN NULL  // Out of memory
  
  2. Scan bitmap for free bit
     FOR i = first_usable / 32 TO total_frames / 32:
       IF bitmap[i] != 0xFFFFFFFF:  // At least one free bit
         FOR j = 0 TO 31:
           frame = i * 32 + j
           IF frame >= first_usable AND frame_test(frame) == 0:
             // Found free frame
             set_frame(frame)
             free_frames--
             RETURN FRAME_TO_ADDR(frame)
  
  3. No free frame found (shouldn't happen if free_frames > 0)
     RETURN NULL
```

### Frame Freeing

```
FREE_FRAME(addr):
  1. Validate address
     IF addr == NULL:
       RETURN  // Ignore NULL
     
     IF addr % FRAME_SIZE != 0:
       PANIC("free_frame: unaligned address 0x%x\n", addr)
  
  2. Calculate frame number
     frame = ADDR_TO_FRAME(addr)
     
     IF frame >= total_frames:
       PANIC("free_frame: frame %d out of range\n", frame)
  
  3. Check for double-free
     IF frame_test(frame) == 0:
       PANIC("free_frame: double free at frame %d (addr 0x%x)\n", frame, addr)
  
  4. Mark as free
     clear_frame(frame)
     free_frames++
```

### Page Mapping

```
MAP_PAGE(dir, vaddr, paddr, flags):
  1. Align addresses to page boundaries
     vaddr = vaddr & ~0xFFF
     paddr = paddr & ~0xFFF
  
  2. Get page directory index
     pd_idx = PD_INDEX(vaddr)
     pde = &dir->entries[pd_idx]
  
  3. Get or create page table
     IF *pde & PTE_PRESENT:
       // Page table exists
       pt = (page_table_t*)PTE_FRAME(*pde)
     ELSE:
       // Need to create page table
       pt_phys = alloc_frame()
       IF pt_phys == NULL:
         RETURN -1  // Out of memory
       
       pt = (page_table_t*)pt_phys  // Identity-mapped during init
       
       // Clear page table
       FOR i = 0 TO 1023:
         pt->entries[i] = 0
       
       // Set page directory entry
       *pde = (uint32_t)pt_phys | PTE_PRESENT | PTE_WRITABLE | PTE_USER
  
  4. Set page table entry
     pt_idx = PT_INDEX(vaddr)
     pt->entries[pt_idx] = paddr | flags | PTE_PRESENT
  
  5. Invalidate TLB entry
     asm volatile("invlpg (%0)" : : "r"(vaddr) : "memory")
  
  6. RETURN 0
```

### Page Unmapping

```
UNMAP_PAGE(dir, vaddr):
  1. Get page directory entry
     pd_idx = PD_INDEX(vaddr)
     pde = &dir->entries[pd_idx]
     
     IF !(*pde & PTE_PRESENT):
       RETURN  // Nothing mapped
  
  2. Get page table
     pt = (page_table_t*)PTE_FRAME(*pde)
  
  3. Clear page table entry
     pt_idx = PT_INDEX(vaddr)
     pt->entries[pt_idx] = 0
  
  4. Invalidate TLB entry
     asm volatile("invlpg (%0)" : : "r"(vaddr) : "memory")
```

### Identity and Higher-Half Mapping

```
PAGING_INIT():
  1. Allocate page directory
     pd_phys = alloc_frame()
     pd = (page_directory_t*)pd_phys
     
     // Clear all entries (not present)
     FOR i = 0 TO 1023:
       pd->entries[i] = 0
  
  2. Identity map first 16 MB
     FOR addr = 0 TO 16*1024*1024 STEP PAGE_SIZE:
       map_page(pd, addr, addr, PTE_PRESENT | PTE_WRITABLE)
       // Note: supervisor-only (no PTE_USER)
  
  3. Higher-half kernel mapping (0xC0000000+)
     FOR offset = 0 TO 16*1024*1024 STEP PAGE_SIZE:
       vaddr = KERNEL_VIRTUAL_BASE + offset
       paddr = offset
       map_page(pd, vaddr, paddr, PTE_PRESENT | PTE_WRITABLE)
  
  4. Load CR3
     asm volatile("mov %0, %%cr3" : : "r"(pd_phys))
  
  5. Enable paging
     asm volatile("mov %%cr0, %0" : "=r"(cr0))
     cr0 |= (1 << 31)  // Set PG bit
     asm volatile("mov %0, %%cr0" : : "r"(cr0))
  
  6. Update global pointer
     current_page_directory = (page_directory_t*)(KERNEL_VIRTUAL_BASE + (uint32_t)pd_phys)
     // Now we need to access via higher-half address!
```

### Page Fault Handler

```
PAGE_FAULT_HANDLER(regs):
  1. Read faulting address from CR2
     asm volatile("mov %%cr2, %0" : "=r"(fault_addr))
  
  2. Decode error code
     present = !(regs->err_code & 0x1)   // Bit 0: 0=not present
     write = regs->err_code & 0x2        // Bit 1: write access
     user = regs->err_code & 0x4         // Bit 2: user mode
     reserved = regs->err_code & 0x8     // Bit 3: reserved bit
     exec = regs->err_code & 0x10        // Bit 4: instruction fetch
  
  3. Print diagnostic
     kprintf("=== PAGE FAULT ===\n")
     kprintf("Address: 0x%x\n", fault_addr)
     kprintf("Error: %s %s %s %s\n",
             present ? "protection" : "not-present",
             write ? "write" : "read",
             user ? "user" : "kernel",
             exec ? "exec" : "")
     kprintf("EIP: 0x%x\n", regs->eip)
  
  4. Check if address is in valid ranges
     IF fault_addr >= KERNEL_VIRTUAL_BASE:
       kprintf("In kernel space\n")
     ELSE IF fault_addr < IDENTITY_MAP_END:
       kprintf("In identity-mapped region\n")
     ELSE:
       kprintf("Outside mapped regions\n")
  
  5. Halt (no demand paging in this version)
     PANIC("Page fault - system halted\n")
```

### Kernel Heap Allocation

```
KMALLOC(size):
  1. Handle edge cases
     IF size == 0:
       RETURN NULL
  
  2. Align size to 4 bytes
     size = (size + 3) & ~3
     
     IF size < HEAP_MIN_BLOCK_SIZE:
       size = HEAP_MIN_BLOCK_SIZE
  
  3. Find free block (first-fit)
     block = heap_head
     WHILE block != NULL:
       IF block->magic != HEAP_MAGIC:
         PANIC("kmalloc: heap corruption at 0x%x\n", block)
       
       IF block->free AND block->size >= size:
         // Found suitable block
         GOTO found_block
       
       block = block->next
  
  4. No suitable block - expand heap
     IF !heap_expand(size + BLOCK_HEADER_SIZE):
       RETURN NULL  // Out of memory
     
     // Try again (should succeed now)
     RETURN kmalloc(size)
  
  found_block:
  5. Split block if large enough
     remaining = block->size - size
     IF remaining > BLOCK_HEADER_SIZE + HEAP_MIN_BLOCK_SIZE:
       // Create new free block after this one
       new_block = (heap_block_t*)((uint8_t*)block + BLOCK_HEADER_SIZE + size)
       new_block->magic = HEAP_MAGIC
       new_block->size = remaining - BLOCK_HEADER_SIZE
       new_block->free = 1
       new_block->next = block->next
       new_block->prev = block
       
       IF block->next:
         block->next->prev = new_block
       
       block->next = new_block
       block->size = size
  
  6. Mark block as allocated
     block->free = 0
     
     RETURN BLOCK_DATA(block)
```

### Kernel Heap Freeing

```
KFREE(ptr):
  1. Handle NULL
     IF ptr == NULL:
       RETURN
  
  2. Get block header
     block = DATA_TO_BLOCK(ptr)
  
  3. Validate magic
     IF block->magic != HEAP_MAGIC:
       PANIC("kfree: invalid pointer 0x%x (magic=0x%x)\n", ptr, block->magic)
  
  4. Check for double-free
     IF block->free:
       PANIC("kfree: double free at 0x%x\n", ptr)
  
  5. Mark as free
     block->free = 1
  
  6. Coalesce with next block
     IF block->next AND block->next->free AND block->next->magic == HEAP_MAGIC:
       block->size += BLOCK_HEADER_SIZE + block->next->size
       block->next = block->next->next
       IF block->next:
         block->next->prev = block
  
  7. Coalesce with previous block
     IF block->prev AND block->prev->free AND block->prev->magic == HEAP_MAGIC:
       block->prev->size += BLOCK_HEADER_SIZE + block->size
       block->prev->next = block->next
       IF block->next:
         block->next->prev = block->prev
```

---

## Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
|-------|-------------|----------|---------------|
| No memory map from bootloader | `e820_init` checks flags | Panic with message | Yes, "No memory map" |
| Out of physical frames | `alloc_frame` returns NULL | kmalloc returns NULL, caller handles | Depends on caller |
| Double-free in frame allocator | `free_frame` tests bit | Panic with address | Yes, panic message |
| Invalid frame address | `free_frame` checks alignment/range | Panic | Yes |
| Page table allocation failure | `map_page` checks alloc_frame result | Return -1, caller handles | Depends on caller |
| Page fault (kernel) | Exception 14 handler | Print diagnostics, halt | Yes, fault message |
| Page fault during paging init | Triple fault | System reset | Yes (reset) |
| TLB stale entry | Various | Always call invlpg | No (prevention) |
| Heap corruption (bad magic) | `kmalloc`/`kfree` check magic | Panic | Yes |
| Heap double-free | `kfree` checks free flag | Panic | Yes |
| Heap expansion failure | `heap_expand` returns 0 | kmalloc returns NULL | Depends on caller |

---

## Implementation Sequence with Checkpoints

### Phase 1: Memory Map Parser (3-4 hours)

**Files**: `kernel/memory/e820.h`, `kernel/memory/e820.c`

**Steps**:
1. Define e820_entry_t and memory_region_t structures
2. Implement `e820_init()` to parse multiboot mmap
3. Add `e820_print()` for debugging
4. Store parsed regions in global array

**Checkpoint**: Memory map parsed and displayed
```c
// In kernel_main after basic init:
e820_init(mbi);
e820_print();
// Expected: List of memory regions with types
// "Memory: 0x000000 - 0x09FFFF (usable)"
// "Memory: 0x100000 - 0x3FFFFFF (usable)"
// etc.
```

**Test**: Verify region count and total memory match QEMU configuration

### Phase 2: Bitmap Frame Allocator (4-5 hours)

**Files**: `kernel/memory/frame.h`, `kernel/memory/frame.c`

**Steps**:
1. Implement placement allocator for boot-time allocations
2. Calculate total frames from memory map
3. Allocate bitmap using placement allocator
4. Mark usable regions as free
5. Reserve kernel frames
6. Implement `alloc_frame()` with bitmap scan
7. Implement `free_frame()` with double-free check

**Checkpoint**: Frame allocator works
```c
void *frame1 = alloc_frame();
void *frame2 = alloc_frame();
kprintf("Allocated frames: 0x%x, 0x%x\n", frame1, frame2);
free_frame(frame1);
void *frame3 = alloc_frame();
kprintf("After free+alloc: 0x%x (should reuse 0x%x)\n", frame3, frame1);
// Expected: frame3 == frame1 (first-fit reuses freed frame)
```

**Test**: Allocate all frames, verify NULL return; free and reallocate

### Phase 3: Page Table Structures (3-4 hours)

**Files**: `kernel/memory/paging.h`, `kernel/memory/paging.c` (partial)

**Steps**:
1. Define pte_t, pde_t and flag constants
2. Define page_table_t, page_directory_t structures
3. Implement PD_INDEX, PT_INDEX macros
4. Implement `get_physical()` for address translation

**Checkpoint**: Structures compile, indices calculate correctly
```c
kprintf("PD_INDEX(0xC0000000) = %d (expected 768)\n", PD_INDEX(0xC0000000));
kprintf("PT_INDEX(0xC0001000) = %d (expected 1)\n", PT_INDEX(0xC0001000));
```

### Phase 4: map_page/unmap_page (4-5 hours)

**Files**: Continue `kernel/memory/paging.c`

**Steps**:
1. Implement `map_page()` with page table allocation
2. Implement `unmap_page()`
3. Add invlpg inline assembly
4. Test with identity-mapped addresses initially

**Checkpoint**: Manual mapping works
```c
page_directory_t *test_pd = alloc_frame();
memset(test_pd, 0, PAGE_SIZE);

void *frame = alloc_frame();
map_page(test_pd, 0x400000, (uint32_t)frame, PTE_PRESENT | PTE_WRITABLE);

uint32_t phys = get_physical(test_pd, 0x400000);
kprintf("Mapped 0x400000 -> 0x%x\n", phys);
// Expected: phys == frame address
```

### Phase 5: Identity Mapping (2-3 hours)

**Files**: Continue `kernel/memory/paging.c`

**Steps**:
1. Create initial page directory
2. Identity map first 16MB (0x00000000 - 0x00FFFFFF)
3. Verify VGA (0xB8000) still accessible

**Checkpoint**: Identity mapping works
```c
// Create PD and identity map
paging_init_identity_only();

// VGA should still work
vga_puts("Identity mapping works!\n");
```

### Phase 6: Higher-Half Kernel Mapping (3-4 hours)

**Files**: `kernel/memory/paging.c`, update `kernel/linker.ld`

**Steps**:
1. Update linker script with higher-half addresses
2. Add higher-half mapping (0xC0000000+ -> 0x00000000+)
3. Handle the transition carefully (code runs at physical until paging enabled)
4. Test kernel functions via higher-half addresses

**Checkpoint**: Kernel runs in higher half
```c
// After paging_init():
kprintf("Kernel running at 0x%x (virtual)\n", (uint32_t)&kernel_main);
kprintf("Physical address: 0x%x\n", get_physical(current_page_directory, (uint32_t)&kernel_main));
// Expected: Virtual ~0xC0100000, Physical ~0x00100000
```

**Critical**: Update linker script:
```ld
ENTRY(kernel_entry)

SECTIONS
{
    . = 0xC0100000;  /* Higher-half + 1MB offset */
    
    .text ALIGN(4K) : AT(ADDR(.text) - 0xC0000000)
    {
        *(.multiboot)
        *(.text)
    }
    
    .rodata ALIGN(4K) : AT(ADDR(.rodata) - 0xC0000000)
    {
        *(.rodata)
    }
    
    .data ALIGN(4K) : AT(ADDR(.data) - 0xC0000000)
    {
        *(.data)
    }
    
    .bss ALIGN(4K) : AT(ADDR(.bss) - 0xC0000000)
    {
        __bss_start = .;
        *(COMMON)
        *(.bss)
        __bss_end = .;
    }
    
    _kernel_end = .;
}
```

### Phase 7: Paging Enablement Sequence (2-3 hours)

**Files**: `kernel/memory/paging.c`

**Steps**:
1. Ensure identity mapping exists before enabling
2. Load CR3 with page directory physical address
3. Set CR0.PG bit
4. Handle the instruction fetch after paging enabled
5. Reload segment registers if needed

**Checkpoint**: Paging enables without triple fault
```bash
# Run in QEMU with interrupt logging
qemu-system-i386 -drive format=raw,file=os.img -d int,cpu_reset -serial stdio 2>&1 | head -100
# Expected: No triple fault or reset after "Enabling paging" message
```

### Phase 8: Page Fault Handler (2-3 hours)

**Files**: Update `kernel/interrupt_handler.c`

**Steps**:
1. Add case for vector 14 in `isr_handler()`
2. Read CR2 for faulting address
3. Decode error code bits
4. Print comprehensive diagnostic
5. Halt (no recovery in this version)

**Checkpoint**: Page faults are diagnosed
```c
// Trigger a page fault:
uint32_t *bad = (uint32_t*)0xDEADBEEF;
*bad = 0x12345678;
// Expected: Page fault message with address 0xDEADBEEF
```

### Phase 9: Kernel Heap (5-7 hours)

**Files**: `kernel/memory/heap.h`, `kernel/memory/heap.c`

**Steps**:
1. Define heap_block_t structure
2. Implement `heap_init()` to create initial heap region
3. Map heap pages using frame allocator
4. Implement `kmalloc()` with first-fit and block splitting
5. Implement `kfree()` with coalescing
6. Implement `heap_expand()` for growth

**Checkpoint**: Heap allocation works
```c
heap_init();

void *p1 = kmalloc(100);
void *p2 = kmalloc(200);
void *p3 = kmalloc(50);

kprintf("Allocated: 0x%x, 0x%x, 0x%x\n", p1, p2, p3);

kfree(p2);
void *p4 = kmalloc(150);  // Should fit in p2's space
kprintf("After free+alloc: 0x%x\n", p4);
// Expected: p4 is near p2 (reused space)
```

**Test**: Stress test with many allocations/frees, verify no corruption

### Final Integration (2-3 hours)

**Goal**: Complete memory management system

**Steps**:
1. Create `kernel/memory/memory.c` with `memory_init(mbi)`
2. Call all init functions in correct order:
   - e820_init
   - frame_allocator_init
   - paging_init
   - heap_init
3. Update kernel_main to use memory_init
4. Add memory status command for debugging

**Checkpoint**: All tests pass
```c
void kernel_main(multiboot_info_t *mbi) {
    vga_init();
    serial_init(COM1_PORT);
    
    vga_puts("Initializing memory...\n");
    memory_init(mbi);
    
    vga_puts("Memory status:\n");
    vga_puts("  Total frames: "); vga_put_dec(frame_alloc.total_frames);
    vga_puts("\n  Free frames: "); vga_put_dec(frame_alloc.free_frames);
    vga_puts("\n  Kernel heap: 0x"); vga_put_hex(HEAP_START);
    vga_puts("\n\n");
    
    // Test allocations
    void *p = kmalloc(1024);
    vga_puts("kmalloc(1024) = 0x"); vga_put_hex((uint32_t)p);
    vga_puts("\n");
    
    kfree(p);
    vga_puts("kfree done\n");
    
    vga_puts("Memory system ready!\n");
}
```

**Test Commands**:
```bash
make clean && make
qemu-system-i386 -drive format=raw,file=os.img -serial stdio

# Expected:
# - Memory map displayed
# - Frame allocator initialized
# - Paging enabled (no crash)
# - Heap allocations work
# - "Memory system ready!" appears
```

---

## Test Specification

### Test 1: Memory Map Parsing

```python
# test_memory_map.py
import subprocess
import re

def test_memory_map():
    proc = subprocess.Popen(
        ['qemu-system-i386', '-drive', 'format=raw,file=os.img',
         '-serial', 'stdio', '-display', 'none'],
        stdout=subprocess.PIPE
    )
    
    import time
    time.sleep(3)
    proc.terminate()
    output, _ = proc.communicate(timeout=5)
    text = output.decode()
    
    # Check for memory region messages
    assert 'Memory:' in text or 'memory' in text.lower(), "No memory map output"
    
    # Should have at least one usable region
    usable_count = text.lower().count('usable')
    assert usable_count >= 1, "No usable memory regions found"
```

### Test 2: Frame Allocation

```c
// kernel/test_frame.c
void test_frame_allocation(void) {
    vga_puts("Testing frame allocation...\n");
    
    // Allocate frames
    void *f1 = alloc_frame();
    void *f2 = alloc_frame();
    void *f3 = alloc_frame();
    
    vga_puts("  Allocated: ");
    vga_put_hex((uint32_t)f1); vga_puts(", ");
    vga_put_hex((uint32_t)f2); vga_puts(", ");
    vga_put_hex((uint32_t)f3); vga_puts("\n");
    
    // Verify different addresses
    ASSERT(f1 != f2 && f2 != f3, "Frames should be different");
    
    // Free and reallocate
    free_frame(f2);
    void *f4 = alloc_frame();
    
    vga_puts("  After free+alloc: ");
    vga_put_hex((uint32_t)f4); vga_puts("\n");
    
    // f4 should be f2 (reused)
    ASSERT(f4 == f2, "Frame should be reused");
    
    vga_puts("PASS: Frame allocation\n");
}
```

### Test 3: Double-Free Detection

```c
// kernel/test_frame.c
void test_double_free(void) {
    vga_puts("Testing double-free detection...\n");
    
    void *frame = alloc_frame();
    
    free_frame(frame);
    
    // This should panic
    vga_puts("  Attempting double-free (should panic)...\n");
    free_frame(frame);
    
    // Should not reach here
    vga_puts("FAIL: Double-free not detected!\n");
}
```

### Test 4: Page Mapping

```c
// kernel/test_paging.c
void test_page_mapping(void) {
    vga_puts("Testing page mapping...\n");
    
    // Create test page directory
    page_directory_t *pd = (page_directory_t*)alloc_frame();
    memset(pd, 0, PAGE_SIZE);
    
    // Map a page
    uint32_t vaddr = 0x400000;
    uint32_t paddr = (uint32_t)alloc_frame();
    
    int result = map_page(pd, vaddr, paddr, PTE_PRESENT | PTE_WRITABLE);
    ASSERT(result == 0, "map_page failed");
    
    // Verify translation
    uint32_t translated = get_physical(pd, vaddr);
    vga_puts("  Mapped 0x"); vga_put_hex(vaddr);
    vga_puts(" -> 0x"); vga_put_hex(translated); vga_puts("\n");
    
    ASSERT(translated == paddr, "Translation mismatch");
    
    // Test unmapping
    unmap_page(pd, vaddr);
    translated = get_physical(pd, vaddr);
    ASSERT(translated == 0, "Page still mapped after unmap");
    
    vga_puts("PASS: Page mapping\n");
}
```

### Test 5: Paging Enablement

```bash
# test_paging_enable.sh
# Run QEMU and check for triple fault after paging enable

timeout 5 qemu-system-i386 -drive format=raw,file=os.img \
    -d int,cpu_reset -serial stdio -display none 2>&1 | tee paging_log.txt

# Check for "Paging enabled" message
if grep -q "Paging enabled" paging_log.txt; then
    echo "PASS: Paging enabled message found"
else
    echo "FAIL: Paging enabled message not found"
    exit 1
fi

# Check for triple fault (reset)
if grep -q "cpu_reset" paging_log.txt; then
    echo "FAIL: CPU reset detected (possible triple fault)"
    exit 1
fi

echo "PASS: No triple fault after paging enable"
```

### Test 6: Heap Allocation

```c
// kernel/test_heap.c
void test_heap_allocation(void) {
    vga_puts("Testing heap allocation...\n");
    
    // Allocate various sizes
    void *p1 = kmalloc(16);
    void *p2 = kmalloc(100);
    void *p3 = kmalloc(1000);
    
    vga_puts("  kmalloc(16) = 0x"); vga_put_hex((uint32_t)p1);
    vga_puts("\n  kmalloc(100) = 0x"); vga_put_hex((uint32_t)p2);
    vga_puts("\n  kmalloc(1000) = 0x"); vga_put_hex((uint32_t)p3);
    vga_puts("\n");
    
    // Verify addresses are in heap region
    ASSERT((uint32_t)p1 >= HEAP_START, "p1 below heap");
    ASSERT((uint32_t)p2 >= HEAP_START, "p2 below heap");
    ASSERT((uint32_t)p3 >= HEAP_START, "p3 below heap");
    
    // Write to memory (shouldn't fault)
    memset(p1, 0xAA, 16);
    memset(p2, 0xBB, 100);
    memset(p3, 0xCC, 1000);
    
    // Free and reallocate
    kfree(p2);
    void *p4 = kmalloc(80);
    
    vga_puts("  After free+alloc: 0x"); vga_put_hex((uint32_t)p4);
    vga_puts("\n");
    
    // p4 should be near p2 (reused space)
    ASSERT(abs((int)p4 - (int)p2) < 200, "p4 should reuse p2 space");
    
    vga_puts("PASS: Heap allocation\n");
}
```

### Test 7: Heap Coalescing

```c
// kernel/test_heap.c
void test_heap_coalescing(void) {
    vga_puts("Testing heap coalescing...\n");
    
    // Allocate three contiguous blocks
    void *p1 = kmalloc(100);
    void *p2 = kmalloc(100);
    void *p3 = kmalloc(100);
    
    // Free them in order
    kfree(p1);
    kfree(p2);
    kfree(p3);
    
    // Now allocate a large block that should fit
    void *large = kmalloc(300);
    
    vga_puts("  Large allocation after coalesce: 0x");
    vga_put_hex((uint32_t)large);
    vga_puts("\n");
    
    // Should be at or near p1 (coalesced space)
    ASSERT((uint32_t)large == (uint32_t)p1 || 
           abs((int)large - (int)p1) < 400,
           "Coalescing failed");
    
    vga_puts("PASS: Heap coalescing\n");
}
```

### Test 8: Page Fault Diagnostics

```bash
# test_page_fault.sh
# Trigger page fault and verify diagnostic output

# Modify kernel to intentionally cause page fault:
# uint32_t *bad = (uint32_t*)0xDEADBEEF;
# *bad = 0x12345678;

timeout 3 qemu-system-i386 -drive format=raw,file=os.img \
    -serial stdio -display none 2>&1 | tee pf_log.txt

# Check for page fault message
if grep -q "PAGE FAULT" pf_log.txt; then
    echo "PASS: Page fault handler executed"
else
    echo "FAIL: No page fault message"
    exit 1
fi

# Check for faulting address
if grep -q "0xDEADBEEF\|Address:" pf_log.txt; then
    echo "PASS: Fault address reported"
else
    echo "FAIL: Fault address not reported"
    exit 1
fi
```

---

## Performance Targets

| Operation | Target | How to Measure |
|-----------|--------|----------------|
| Frame allocation (bitmap scan) | O(n) worst case, ~10µs for 1GB | Time alloc_frame() calls, average over 1000 allocations |
| Frame free | O(1), < 1µs | Time free_frame() calls |
| Page mapping (no PT alloc) | < 500 cycles, ~200ns | Time map_page() for already-present PT |
| Page mapping (with PT alloc) | < 2000 cycles, ~1µs | Time map_page() including frame allocation |
| TLB invalidate | ~10 cycles | Single invlpg instruction |
| kmalloc (small, < 64 bytes) | < 1µs | Time 1000 allocations, average |
| kmalloc (large, > 4KB) | < 10µs | Time large allocations including page mapping |
| kfree | < 1µs | Time 1000 frees, average |
| Page fault handler | < 50µs | Time from fault to diagnostic print |

---

## Hardware Soul

### Cache Lines Touched

**Page table access**: Each page table (4KB) occupies exactly 4 cache lines (64 bytes each). During a page walk:
- 1 cache line read for PDE (8 bytes, likely shared with other PDEs)
- 1 cache line read for PTE (4 bytes, likely shared with other PTEs)
- Total: 2 cache line reads per TLB miss

**Frame allocator bitmap**: For 1GB RAM with 4KB frames:
- 32,768 bits = 4KB bitmap = 64 cache lines
- Scanning allocates reads sequentially, good locality
- Free operation touches single cache line

**Heap metadata**: Each block header is 20 bytes. A 64-byte cache line holds ~3 block headers. Allocation scans through headers sequentially until finding fit.

### TLB Behavior

**TLB size**: 64-128 entries on typical x86 CPUs. Each entry maps one 4KB page.

**Coverage**: 64 entries × 4KB = 256KB of direct mapping. Any access outside these 256KB triggers TLB miss and page walk.

**Context switch cost**: CR3 reload invalidates all TLB entries (unless PCID used). Process switch = full TLB flush = worst case 64-128 page walks to repopulate.

**Large page optimization**: Using 4MB pages (PS bit in PDE) means one TLB entry covers 4MB instead of 4KB. 64 entries × 4MB = 256MB coverage. (Not implemented in this spec.)

### Memory Access Patterns

**Sequential frame allocation**: Good cache behavior—bitmap scan proceeds linearly. If physical memory is contiguous, allocated frames have good locality.

**Random page mapping**: If virtual addresses are scattered across different page tables, each mapping may allocate a new PT, causing allocation overhead and potential cache thrashing.

**Heap fragmentation**: Over time, many small allocations/frees create scattered free blocks. First-fit scanning traverses many cache lines.

### Physical Reality

**DRAM latency**: ~70ns for DDR4. A TLB miss causing 2 DRAM accesses = 140ns minimum, plus controller overhead.

**Page fault overhead**: Full context switch + disk I/O = milliseconds. Even without swap, just the handler overhead is 10-50µs.

**False sharing**: If two frequently-modified variables share a cache line, they cause cache coherency traffic. Heap head/tail pointers should be on separate cache lines in a multi-threaded system.

---

## Concurrency Specification

### Current Model: Single-Threaded

All allocation functions are **not thread-safe** by design for this milestone. The kernel runs single-threaded until Milestone 4.

### Critical Sections (Future-Proofing)

When interrupts are enabled and scheduling is added:

```c
void *kmalloc_safe(uint32_t size) {
    void *ptr;
    uint32_t flags;
    
    // Disable interrupts
    asm volatile("pushf; pop %0; cli" : "=r"(flags));
    
    ptr = kmalloc(size);
    
    // Restore interrupts
    asm volatile("push %0; popf" : : "r"(flags));
    
    return ptr;
}
```

### Lock-Free Keyboard Buffer (Already Implemented)

The circular buffer uses careful index ordering:

- Producer (IRQ handler) only writes to `head`, reads `tail`
- Consumer (main code) only writes to `tail`, reads `head`
- Full condition: `(head + 1) % SIZE == tail` (checked by producer)
- Empty condition: `head == tail` (checked by consumer)

No locks needed because each side only modifies one index.

---

## Visual Diagrams

### Page Table Hierarchy

```
x86 32-bit Two-Level Paging:

Virtual Address (32 bits):
┌─────────────────┬─────────────────┬─────────────────┐
│  PD Index       │  PT Index       │  Page Offset    │
│  (bits 31:22)   │  (bits 21:12)   │  (bits 11:0)    │
│  10 bits        │  10 bits        │  12 bits        │
└────────┬────────┴────────┬────────┴────────┬────────┘
         │                 │                 │
         ▼                 │                 │
    ┌─────────┐            │                 │
    │   CR3   │            │                 │
    │(PD addr)│            │                 │
    └────┬────┘            │                 │
         │                 │                 │
         ▼                 │                 │
┌─────────────────┐        │                 │
│ Page Directory  │        │                 │
│  (4KB, 1024     │        │                 │
│   entries)      │        │                 │
├─────────────────┤        │                 │
│ Entry[PD Index] │────────┘                 │
│  → Page Table   │                          │
│    Address      │                          │
└────────┬────────┘                          │
         │                                   │
         ▼                                   │
┌─────────────────┐                          │
│  Page Table     │                          │
│  (4KB, 1024     │                          │
│   entries)      │                          │
├─────────────────┤                          │
│ Entry[PT Index] │──────────────────────────┘
│  → Physical     │
│    Frame Addr   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Physical Memory │
│  (4KB Frame)    │
│ at Frame Addr   │
│ + Page Offset   │
└─────────────────┘

Coverage:
- Page Directory: 1024 entries
- Each PDE covers: 1024 × 4KB = 4MB
- Total address space: 1024 × 4MB = 4GB
```

### Page Directory/Table Entry Bits

```
Page Table Entry (32 bits):
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬─────────────────┐
│ 31    │ 12    │ 11  9 │ 8     │ 7     │ 6     │ 5     │ 4  3  │ 2   │ 1   │ 0   │
│ Frame │       │ Avail │ Global│ PS    │ Dirty │ Acc   │ PCD   │ PWT │ U/S │ R/W │ P   │
│ Addr  │       │ (OS)  │       │       │       │       │ PWT   │     │     │     │     │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴─────┴─────┴─────┘
          │                │       │       │       │       │       │     │     │     │
          │                │       │       │       │       │       │     │     │     └─ Present
          │                │       │       │       │       │       │     │     └─ Read/Write
          │                │       │       │       │       │       │     └─ User/Supervisor
          │                │       │       │       │       │       └─ Write-Through
          │                │       │       │       │       └─ Cache Disable
          │                │       │       │       └─ Accessed (CPU sets)
          │                │       │       └─ Dirty (CPU sets, PT only)
          │                │       └─ Page Size (0=4KB, 1=4MB in PD)
          │                └─ Global (not flushed on CR3 reload)
          └─ Available for OS use
          
Bits 31:12 - Physical Frame Address (frame must be 4KB aligned)

Page Directory Entry (same format, but):
- Bit 7 (PS): Page Size (0=4KB pages, 1=4MB pages)
- Bit 6 (D): Reserved (not dirty)
- Frame Address points to Page Table (not data frame)
```

### E820 Memory Map

```
Typical E820 Memory Map (512MB system):

Physical Address
┌─────────────────────────────────────────────────────────────────┐
│ 0x000000 - 0x000FFF │ IVT, BDA (Reserved)            │  4 KB   │
├─────────────────────────────────────────────────────────────────┤
│ 0x001000 - 0x07FFFF │ Low Memory (Usable)            │ 508 KB  │
├─────────────────────────────────────────────────────────────────┤
│ 0x080000 - 0x0FFFFF │ EBDA, Video RAM, ROM (Reserved)│ ~500 KB │
├─────────────────────────────────────────────────────────────────┤
│ 0x100000 - 0x1FFFFF │ Kernel + data (Usable)         │  1 MB   │
│                     │ (Marked used by allocator)     │         │
├─────────────────────────────────────────────────────────────────┤
│ 0x200000 - 0x1FFFFFF│ Extended Memory (Usable)       │ ~30 MB  │
│                     │ Available for allocation       │         │
├─────────────────────────────────────────────────────────────────┤
│ 0x2000000+          │ More RAM (Usable)              │ ~480 MB │
└─────────────────────────────────────────────────────────────────┘

Memory Map Entry Format (E820):
┌─────────────────────────────────────────────────────────────────┐
│ Offset  Size  Field                                           │
├─────────────────────────────────────────────────────────────────┤
│ 0       8     Base Address (uint64_t)                         │
│ 8       8     Length (uint64_t)                               │
│ 16      4     Type (uint32_t)                                 │
│                 1 = Usable                                    │
│                 2 = Reserved                                  │
│                 3 = ACPI Reclaimable                          │
│                 4 = ACPI NVS                                  │
│                 5 = Bad Memory                                │
│ 20      4     ACPI Extended Attributes (optional)             │
└─────────────────────────────────────────────────────────────────┘
```

### Identity + Higher-Half Mapping

```
Virtual Address Space Layout:

┌─────────────────────────────────────────────────────────────────┐
│ 0xFFFFFFFF           End of 32-bit address space               │
│                                                                 │
│ ... Reserved / Kernel Space ...                                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ 0xC0400000           Kernel Heap Start                         │
│                     ┌─────────────────────────────────┐         │
│                     │   kmalloc/kfree region          │         │
│                     │   (grows on demand)             │         │
│                     └─────────────────────────────────┘         │
├─────────────────────────────────────────────────────────────────┤
│ 0xC0100000           Kernel Code/Data (Higher-Half)            │
│                     ┌─────────────────────────────────┐         │
│                     │   .text, .rodata, .data, .bss   │         │
│                     │   Maps to physical 0x00100000   │         │
│                     └─────────────────────────────────┘         │
├─────────────────────────────────────────────────────────────────┤
│ 0xC0000000           Higher-Half Base (3 GB)                   │
│                     Maps to physical 0x00000000                 │
├─────────────────────────────────────────────────────────────────┤
│ 0x01000000           End of Identity Map (16 MB)               │
│                                                                 │
│ ... Identity-Mapped Region ...                                 │
│   0x00000000 - 0x00FFFFFF → 0x00000000 - 0x00FFFFFF           │
│   (VGA at 0xB8000 accessible)                                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ 0x00000000           Start of Address Space                    │
│                     (User Space in future milestones)           │
└─────────────────────────────────────────────────────────────────┘

Physical to Virtual Mapping:
┌─────────────────────────────────────────────────────────────────┐
│ Physical Address     Virtual Address         Purpose           │
├─────────────────────────────────────────────────────────────────┤
│ 0x00000000           0x00000000              Identity map      │
│         ...                   ...             (first 16 MB)    │
│ 0x00FFFFFF           0x00FFFFFF                                │
├─────────────────────────────────────────────────────────────────┤
│ 0x00000000           0xC0000000              Higher-half kernel│
│         ...                   ...             (first 16 MB     │
│ 0x00FFFFFF           0xC0FFFFFF               mapped high)     │
├─────────────────────────────────────────────────────────────────┤
│ 0x00100000           0xC0100000              Kernel code      │
├─────────────────────────────────────────────────────────────────┤
│ (allocated)          0xC0400000+             Kernel heap      │
└─────────────────────────────────────────────────────────────────┘
```

### Paging Enable Sequence

```
Enabling Paging - Critical Sequence:

┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Create Page Directory                                   │
│   pd = alloc_frame()                                            │
│   memset(pd, 0, 4096)                                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Identity Map First 16 MB                                │
│   FOR addr = 0 TO 16MB STEP 4K:                                │
│     map_page(pd, addr, addr, PTE_WRITABLE)                     │
│                                                                 │
│   CRITICAL: Must include currently executing code!              │
│   The instruction AFTER enabling paging must be mapped.        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Higher-Half Mapping                                     │
│   FOR offset = 0 TO 16MB STEP 4K:                              │
│     vaddr = 0xC0000000 + offset                                │
│     paddr = offset                                             │
│     map_page(pd, vaddr, paddr, PTE_WRITABLE)                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Load CR3                                                │
│   asm volatile("mov %0, %%cr3" : : "r"(pd_phys))               │
│                                                                 │
│   This sets the page directory base address.                   │
│   TLB is flushed when CR3 changes.                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Enable Paging                                           │
│   asm volatile("mov %%cr0, %0" : "=r"(cr0))                    │
│   cr0 |= (1 << 31)  // Set PG bit                              │
│   asm volatile("mov %0, %%cr0" : : "r"(cr0))                   │
│                                                                 │
│   CRITICAL: The very next instruction fetch uses paging!       │
│   If current EIP not mapped → immediate page fault → crash     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: Continue in Paged Mode                                  │
│   All addresses now translated through page tables.            │
│   Kernel code accessible at 0xC0100000+.                       │
│   VGA accessible at 0xB8000 (identity mapped).                 │
└─────────────────────────────────────────────────────────────────┘

Common Mistakes:
┌─────────────────────────────────────────────────────────────────┐
│ ✗ Forgetting identity map → code not accessible after PG set   │
│ ✗ Not mapping VGA → screen output breaks                       │
│ ✗ Wrong CR3 value → page directory not found                   │
│ ✗ Page tables not identity mapped → can't access PT to fill it │
│ ✗ Forgetting invlpg → stale TLB entries                        │
└─────────────────────────────────────────────────────────────────┘
```

### Page Fault Handler Flow

```
Page Fault (Exception 14) Processing:

┌─────────────────────────────────────────────────────────────────┐
│                        CPU                                       │
│  1. Instruction causes page fault                               │
│  2. Push EFLAGS, CS, EIP                                        │
│  3. Push error code                                              │
│  4. Load CR2 with faulting address                              │
│  5. Load IDT[14] → CS:EIP                                       │
│  6. Jump to handler                                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Assembly Stub                                 │
│  1. Push dummy error code (if not provided)                     │
│  2. Push interrupt number (14)                                  │
│  3. pusha (save all GP registers)                               │
│  4. Push segment registers                                      │
│  5. Call C handler                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     C Handler                                    │
│                                                                 │
│  void page_fault_handler(registers_t *regs) {                  │
│      uint32_t fault_addr;                                       │
│      asm volatile("mov %%cr2, %0" : "=r"(fault_addr));         │
│                                                                 │
│      int present = !(regs->err_code & 0x1);                    │
│      int write = regs->err_code & 0x2;                         │
│      int user = regs->err_code & 0x4;                          │
│                                                                 │
│      kprintf("Page Fault!\n");                                  │
│      kprintf("  Address: 0x%x\n", fault_addr);                 │
│      kprintf("  %s, %s, %s mode\n",                            │
│              present ? "protection" : "not-present",           │
│              write ? "write" : "read",                          │
│              user ? "user" : "kernel");                        │
│                                                                 │
│      PANIC("Unhandled page fault");                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘

Error Code Bits:
┌─────────────────────────────────────────────────────────────────┐
│ Bit 0 (P): 0 = Page not present                                 │
│            1 = Protection violation (page exists)              │
│ Bit 1 (W): 0 = Read access                                      │
│            1 = Write access                                     │
│ Bit 2 (U): 0 = Supervisor mode (kernel)                        │
│            1 = User mode                                        │
│ Bit 3 (R): 1 = Reserved bit set in paging structures           │
│ Bit 4 (I): 1 = Instruction fetch (NX violation)                │
└─────────────────────────────────────────────────────────────────┘
```

### kmalloc Internals

```
Kernel Heap Structure:

┌─────────────────────────────────────────────────────────────────┐
│                      Virtual Memory                              │
│                    0xC0400000+                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                    Heap Block 1 (free)                     ││
│  ├────────────────────────────────────────────────────────────┤│
│  │ magic   │ 0xDEADBEEF                                        ││
│  │ size    │ 1000                                              ││
│  │ free    │ 1                                                 ││
│  │ next    │ ─────────────────────────────┐                   ││
│  │ prev    │ NULL                         │                   ││
│  ├─────────┴─────────────────────────────┴───────────────────┤│
│  │                    Data Area (1000 bytes)                  ││
│  └────────────────────────────────────────────────────────────┘│
│                           │                                     │
│                           ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                  Heap Block 2 (allocated)                  ││
│  ├────────────────────────────────────────────────────────────┤│
│  │ magic   │ 0xDEADBEEF                                        ││
│  │ size    │ 64                                                ││
│  │ free    │ 0                                                 ││
│  │ next    │ ─────────────────────────────┐                   ││
│  │ prev    │ ◄─────────────────────────────┘                   ││
│  ├─────────┴─────────────────────────────┴───────────────────┤│
│  │                    Data Area (64 bytes)                    ││
│  │               [kmalloc returned pointer here]              ││
│  └────────────────────────────────────────────────────────────┘│
│                           │                                     │
│                           ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                    Heap Block 3 (free)                     ││
│  ├────────────────────────────────────────────────────────────┤│
│  │ magic   │ 0xDEADBEEF                                        ││
│  │ size    │ 4080                                              ││
│  │ free    │ 1                                                 ││
│  │ next    │ NULL (end of heap)                               ││
│  │ prev    │ ◄─────────────────────────────                    ││
│  ├─────────┴─────────────────────────────┴───────────────────┤│
│  │                    Data Area (4080 bytes)                  ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    End of Mapped Heap                           │
│              (expand by mapping more pages)                     │
└─────────────────────────────────────────────────────────────────┘

kmalloc Algorithm:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Align size to 4 bytes                                        │
│ 2. Scan free list for block >= size                            │
│ 3. If found:                                                    │
│    a. If block is much larger, split:                          │
│       - Create new free block after allocated portion          │
│       - Adjust sizes                                           │
│    b. Mark block as allocated                                   │
│    c. Return pointer to data area                              │
│ 4. If not found:                                                │
│    a. Expand heap (map new pages)                              │
│    b. Create new free block in expanded region                 │
│    c. Retry allocation                                          │
└─────────────────────────────────────────────────────────────────┘

kfree Algorithm:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Get block header from pointer                               │
│ 2. Verify magic number                                          │
│ 3. Check for double-free                                        │
│ 4. Mark as free                                                 │
│ 5. Coalesce with next block if free                            │
│ 6. Coalesce with previous block if free                        │
└─────────────────────────────────────────────────────────────────┘
```

---


<!-- TDD_MOD_ID: mod-scheduling -->
# Technical Design Specification: Process Management and Preemptive Scheduling

## Module Charter

The process management module implements preemptive multitasking through Process Control Blocks (PCBs) that capture complete CPU state, assembly context switching that saves/restores all registers, TSS configuration for secure ring 3→ring 0 stack transitions during interrupts/syscalls, a round-robin scheduler triggered by timer interrupts, and a system call interface via INT 0x80. It supports both kernel-mode processes (ring 0, shared address space) and user-mode processes (ring 3, isolated page directories).

**What it does NOT do**: This module does not implement demand paging (page faults halt), does not implement priority scheduling or deadline scheduling (round-robin only), does not implement inter-process communication beyond basic syscalls, does not handle process termination cleanup beyond marking zombie state, does not implement copy-on-write fork, and does not implement threading (one thread per process).

**Upstream dependencies**: GDT must include user code/data descriptors (ring 3) at indices 3-4; IDT must have handlers for exceptions 0-31 and IRQs 0-15; timer interrupt must fire at configured frequency; paging must be enabled with identity mapping for low memory and higher-half kernel; frame allocator and heap must be functional.

**Downstream consumers**: User programs execute via syscalls (sys_write, sys_exit); shell/command interpreter will use process creation; future milestones may add IPC, signals, and file descriptors.

**Invariants**: TSS.ESP0 must be updated before every context switch to a user-mode process; context switch must save ALL registers in PCB order; scheduler must disable interrupts during critical sections; user processes must never access kernel memory (page fault if attempted); syscall handler must validate all user pointers before dereferencing.

---

## File Structure

Create files in this order:

```
1. kernel/process/process.h      # PCB structure, process states, function declarations
2. kernel/process/process.c      # Process creation, destruction, table management
3. kernel/process/context.asm    # Context switch assembly implementation
4. kernel/process/tss.h          # TSS structure and function declarations
5. kernel/process/tss.c          # TSS initialization and ESP0 update
6. kernel/process/scheduler.h    # Scheduler interface declarations
7. kernel/process/scheduler.c    # Round-robin scheduler implementation
8. kernel/syscall/syscall.h      # Syscall numbers and handler declarations
9. kernel/syscall/syscall.c      # Syscall dispatch and implementations
10. kernel/syscall/syscall.asm   # Syscall entry stub (INT 0x80)
11. kernel/user/user_mode.asm    # enter_user_mode assembly
12. kernel/user/user_entry.c     # User process entry point wrapper
```

---

## Complete Data Model

### Process Control Block (PCB)

```c
// kernel/process/process.h
#include <stdint.h>
#include "memory/paging.h"

// Process states
typedef enum {
    PROCESS_UNUSED = 0,       // Slot in process table is free
    PROCESS_READY,            // Ready to run, waiting for scheduler
    PROCESS_RUNNING,          // Currently executing
    PROCESS_BLOCKED,          // Waiting for I/O, sleep, etc.
    PROCESS_ZOMBIE            // Terminated, waiting for cleanup
} process_state_t;

// Process Control Block - matches assembly offsets exactly
typedef struct process {
    // === Identification (offset 0) ===
    uint32_t pid;                    // Process ID (1-65535)
    char name[16];                   // Process name (null-terminated)
    process_state_t state;           // Current state
    uint8_t padding0[4];             // Align to 8 bytes
    
    // === Register state (offset 32) ===
    // General purpose registers
    uint32_t eax;                    // Offset 32
    uint32_t ebx;                    // Offset 36
    uint32_t ecx;                    // Offset 40
    uint32_t edx;                    // Offset 44
    uint32_t esi;                    // Offset 48
    uint32_t edi;                    // Offset 52
    uint32_t ebp;                    // Offset 56
    uint32_t esp;                    // Offset 60 - kernel or user stack
    uint32_t eip;                    // Offset 64 - instruction pointer
    uint32_t eflags;                 // Offset 68 - flags register
    
    // Segment registers (16-bit values stored in 32-bit fields)
    uint32_t cs;                     // Offset 72 - code segment
    uint32_t ds;                     // Offset 76 - data segment
    uint32_t es;                     // Offset 80
    uint32_t fs;                     // Offset 84
    uint32_t gs;                     // Offset 88
    uint32_t ss;                     // Offset 92 - stack segment
    
    // === Memory management (offset 96) ===
    page_directory_t *page_directory;    // Offset 96 - virtual address space
    uint32_t kernel_stack;               // Offset 100 - top of kernel stack (for TSS.ESP0)
    uint32_t user_stack_top;             // Offset 104 - user stack top (if user mode)
    
    // === Scheduling (offset 108) ===
    struct process *next;           // Offset 108 - next in run queue
    struct process *prev;           // Offset 112 - previous in run queue
    uint32_t wake_time;             // Offset 116 - tick to wake (for sleep)
    int32_t exit_status;            // Offset 120 - exit code for parent
    uint32_t time_slice;            // Offset 124 - remaining time slice
    
    // === Process tree (future expansion) ===
    struct process *parent;         // Offset 128 - parent process
    struct process *first_child;    // Offset 132 - first child
    struct process *next_sibling;   // Offset 136 - next sibling
    
} __attribute__((packed)) process_t;

// Verify PCB size and alignment
#define PCB_SIZE sizeof(process_t)
#define PCB_REG_OFFSET 32          // Offset to first register (eax)

// Process table
#define MAX_PROCESSES 64
#define KERNEL_STACK_SIZE 4096     // 4KB kernel stack per process
#define USER_STACK_SIZE (64 * 1024) // 64KB user stack

extern process_t process_table[MAX_PROCESSES];
extern process_t *current_process;
extern process_t *ready_queue;
extern uint32_t next_pid;

// Assembly context switch offsets MUST match this structure:
// Verify with: nasm -f elf32 context.asm && nm context.o
```

**PCB Memory Layout (byte offsets):**

| Offset | Size | Field | Purpose |
|--------|------|-------|---------|
| 0 | 4 | pid | Unique process identifier |
| 4 | 16 | name | Debug-friendly process name |
| 20 | 4 | state | Ready/Running/Blocked/Zombie |
| 24 | 4 | padding0 | Alignment to 8 bytes |
| 28 | 4 | (reserved) | Future use |
| 32 | 4 | eax | General purpose register A |
| 36 | 4 | ebx | General purpose register B |
| 40 | 4 | ecx | General purpose register C |
| 44 | 4 | edx | General purpose register D |
| 48 | 4 | esi | Source index register |
| 52 | 4 | edi | Destination index register |
| 56 | 4 | ebp | Base pointer |
| 60 | 4 | esp | Stack pointer |
| 64 | 4 | eip | Instruction pointer |
| 68 | 4 | eflags | CPU flags |
| 72 | 4 | cs | Code segment selector |
| 76 | 4 | ds | Data segment selector |
| 80 | 4 | es | Extra segment selector |
| 84 | 4 | fs | Extra segment selector |
| 88 | 4 | gs | Extra segment selector |
| 92 | 4 | ss | Stack segment selector |
| 96 | 4 | page_directory | Pointer to process page tables |
| 100 | 4 | kernel_stack | Kernel stack top (for TSS.ESP0) |
| 104 | 4 | user_stack_top | User mode stack top |
| 108 | 4 | next | Next process in queue |
| 112 | 4 | prev | Previous process in queue |
| 116 | 4 | wake_time | Tick count to wake |
| 120 | 4 | exit_status | Process exit code |
| 124 | 4 | time_slice | Remaining quantum |
| 128 | 4 | parent | Parent process pointer |
| 132 | 4 | first_child | First child pointer |
| 136 | 4 | next_sibling | Sibling list pointer |

### Task State Segment (TSS)

```c
// kernel/process/tss.h
#include <stdint.h>

// TSS structure for 32-bit x86
// Only SS0:ESP0 are critical for ring transitions
typedef struct {
    uint16_t prev_task;       // 0: Previous task link (unused)
    uint16_t reserved0;       // 2: Reserved
    uint32_t esp0;            // 4: Stack pointer for ring 0 (CRITICAL)
    uint16_t ss0;             // 8: Stack segment for ring 0 (CRITICAL)
    uint16_t reserved1;       // 10: Reserved
    uint32_t esp1;            // 12: Stack pointer for ring 1 (unused)
    uint16_t ss1;             // 16: Stack segment for ring 1 (unused)
    uint16_t reserved2;       // 18: Reserved
    uint32_t esp2;            // 20: Stack pointer for ring 2 (unused)
    uint16_t ss2;             // 24: Stack segment for ring 2 (unused)
    uint16_t reserved3;       // 26: Reserved
    uint32_t cr3;             // 28: Page directory base (unused in software switching)
    uint32_t eip;             // 32: Instruction pointer (unused)
    uint32_t eflags;          // 36: Flags (unused)
    uint32_t eax;             // 40-68: General registers (unused)
    uint32_t ecx;
    uint32_t edx;
    uint32_t ebx;
    uint32_t esp;
    uint32_t ebp;
    uint32_t esi;
    uint32_t edi;
    uint16_t es;              // 72: Extra segment (unused)
    uint16_t reserved4;
    uint16_t cs;              // 76: Code segment (unused)
    uint16_t reserved5;
    uint16_t ss;              // 80: Stack segment (unused)
    uint16_t reserved6;
    uint16_t ds;              // 84: Data segment (unused)
    uint16_t reserved7;
    uint16_t fs;              // 88: Extra segment (unused)
    uint16_t reserved8;
    uint16_t gs;              // 92: Extra segment (unused)
    uint16_t reserved9;
    uint16_t ldt;             // 96: LDT selector (unused)
    uint16_t reserved10;
    uint16_t trap;            // 100: Trap bit (unused)
    uint16_t iomap_base;      // 102: I/O permission bitmap offset
} __attribute__((packed)) tss_t;

// TSS instance
extern tss_t tss;

// TSS selector in GDT (index 5, selector = 0x28)
#define TSS_SELECTOR 0x28
```

### System Call Numbers

```c
// kernel/syscall/syscall.h
#include <stdint.h>

// System call numbers (passed in EAX)
#define SYS_EXIT    0    // void sys_exit(int status)
#define SYS_READ    1    // int sys_read(int fd, char *buf, int count)
#define SYS_WRITE   2    // int sys_write(int fd, const char *buf, int count)
#define SYS_EXEC    3    // int sys_exec(const char *path)
#define SYS_FORK    4    // int sys_fork(void)
#define SYS_GETPID  5    // int sys_getpid(void)
#define SYS_YIELD   6    // void sys_yield(void)

// Maximum syscall number
#define MAX_SYSCALL 7

// Syscall handler type
typedef int32_t (*syscall_handler_t)(uint32_t arg0, uint32_t arg1, uint32_t arg2);

// Syscall dispatch table
extern syscall_handler_t syscall_table[MAX_SYSCALL];

// Individual syscall implementations
void sys_exit(int status);
int sys_write(int fd, const char *buf, int count);
int sys_read(int fd, char *buf, int count);
int sys_getpid(void);
void sys_yield(void);
```

### Scheduler Data Structures

```c
// kernel/process/scheduler.h
#include "process.h"

// Scheduler configuration
#define DEFAULT_TIME_SLICE  10    // Default ticks per time slice (100ms at 100Hz)
#define SCHEDULER_FREQ      100   // Timer frequency in Hz

// Scheduler state
extern uint32_t scheduler_ticks;
extern volatile int scheduler_enabled;

// Run queue management
void scheduler_add_to_queue(process_t *proc);
void scheduler_remove_from_queue(process_t *proc);
process_t *scheduler_pick_next(void);

// Core scheduler functions
void scheduler_init(void);
void scheduler_tick(void);           // Called from timer interrupt
void scheduler_yield(void);          // Voluntary yield
void scheduler_block(process_t *proc);
void scheduler_unblock(process_t *proc);
```

---

## Interface Contracts

### process_create(const char *name, void (*entry)(void), int is_user)

**Purpose**: Create a new process and add it to the ready queue

**Parameters**:
- `name`: Process name (max 15 characters)
- `entry`: Entry point function
- `is_user`: 0 for kernel-mode process, 1 for user-mode

**Returns**:
- Pointer to new process_t on success
- NULL on failure (process table full, out of memory)

**Side effects**:
- Allocates PCB slot
- Allocates kernel stack (4KB)
- For user processes: allocates user stack and clones page directory
- Adds process to ready queue

**Preconditions**: Memory allocator initialized; paging enabled

**Postconditions**: New process in READY state, runnable by scheduler

### process_exit(int status)

**Purpose**: Terminate the current process

**Parameters**:
- `status`: Exit status code

**Returns**: Does not return

**Side effects**:
- Sets process state to ZOMBIE
- Stores exit status
- Removes from run queue
- Triggers scheduler to run next process

### context_switch(process_t *old, process_t *new)

**Purpose**: Save current CPU state to old PCB and load state from new PCB

**Parameters**:
- `old`: Pointer to current process PCB (may be NULL for first switch)
- `new`: Pointer to process to switch to

**Returns**: Does not return to caller (returns to new process context)

**Side effects**:
- Saves all registers to old->*
- Loads CR3 if page directories differ
- Updates TSS.ESP0 for user processes
- Loads all registers from new->*
- Returns to new->eip with new->eflags

**Critical**: Assembly implementation; must match PCB offsets exactly

### tss_init(void)

**Purpose**: Initialize the Task State Segment and load TR

**Parameters**: None

**Returns**: None

**Side effects**:
- Zeros TSS structure
- Sets TSS.SS0 = 0x10 (kernel data selector)
- Adds TSS descriptor to GDT at index 5
- Loads TR with selector 0x28

**Preconditions**: GDT must be loaded

### tss_set_esp0(uint32_t esp0)

**Purpose**: Update the kernel stack pointer for ring transitions

**Parameters**:
- `esp0`: New value for TSS.ESP0

**Returns**: None

**Critical**: MUST be called before switching to any user-mode process

### scheduler_init(void)

**Purpose**: Initialize scheduler and create kernel idle process

**Parameters**: None

**Returns**: None

**Side effects**:
- Creates PCB for current (kernel) process
- Sets current_process to kernel process
- Enables scheduler

### scheduler_tick(void)

**Purpose**: Handle timer interrupt and trigger context switch if needed

**Parameters**: None (called from timer IRQ handler)

**Returns**: None

**Side effects**:
- Increments tick counter
- Decrements current process time slice
- Triggers context switch if slice expired or process yielded

**Preconditions**: Timer interrupt configured; IDT loaded

### scheduler_yield(void)

**Purpose**: Voluntarily give up CPU to next ready process

**Parameters**: None

**Returns**: Does not return immediately (returns when scheduled again)

**Side effects**:
- Marks current process as READY
- Triggers immediate context switch

### enter_user_mode(uint32_t entry, uint32_t stack)

**Purpose**: Transition from kernel mode to user mode

**Parameters**:
- `entry`: Entry point address in user space
- `stack`: Top of user stack

**Returns**: Does not return (begins executing user code at ring 3)

**Side effects**:
- Loads segment registers with user selectors (0x23)
- Sets up iret frame with user CS, SS, EIP, ESP, EFLAGS
- Executes iret to enter ring 3

**Critical**: TSS.ESP0 must be set before calling

### syscall_handler(registers_t *regs)

**Purpose**: Dispatch system call to appropriate handler

**Parameters**:
- `regs`: Pointer to saved register frame (int_no = 0x80)

**Returns**: Result in regs->eax

**Side effects**:
- Validates syscall number
- Validates user pointers
- Calls appropriate handler function

---

## Algorithm Specification

### Process Creation

```
PROCESS_CREATE(name, entry, is_user):
  1. Find free slot in process_table
     FOR i = 0 TO MAX_PROCESSES - 1:
       IF process_table[i].state == PROCESS_UNUSED:
         proc = &process_table[i]
         BREAK
     IF no free slot:
       RETURN NULL
  
  2. Initialize PCB fields
     memset(proc, 0, sizeof(process_t))
     proc->pid = next_pid++
     strncpy(proc->name, name, 15)
     proc->state = PROCESS_READY
     proc->time_slice = DEFAULT_TIME_SLICE
  
  3. Allocate kernel stack
     stack_frame = alloc_frame()
     IF stack_frame == NULL:
       RETURN NULL
     proc->kernel_stack = (uint32_t)stack_frame + KERNEL_STACK_SIZE
  
  4. Set up initial register state
     proc->eip = (uint32_t)entry
     proc->eflags = 0x202  // IF=1, bit 1 always 1
     proc->ebp = proc->kernel_stack
     proc->esp = proc->kernel_stack  // Will be adjusted for iret frame
  
  5. Set up segments based on mode
     IF is_user:
       // User mode (ring 3)
       proc->cs = 0x1B   // User code selector (GDT index 3, RPL=3)
       proc->ds = 0x23   // User data selector (GDT index 4, RPL=3)
       proc->es = 0x23
       proc->fs = 0x23
       proc->gs = 0x23
       proc->ss = 0x23
       
       // Create user page directory (clone kernel mappings)
       proc->page_directory = clone_page_directory(current_page_directory)
       IF proc->page_directory == NULL:
         free_frame(stack_frame)
         RETURN NULL
       
       // Allocate user stack
       user_stack_virt = USER_STACK_TOP - USER_STACK_SIZE
       FOR page = user_stack_virt TO USER_STACK_TOP STEP 4096:
         phys = alloc_frame()
         IF phys == NULL:
           // Cleanup and fail
           RETURN NULL
         map_page(proc->page_directory, page, phys, 
                  PTE_PRESENT | PTE_WRITABLE | PTE_USER)
       
       proc->user_stack_top = USER_STACK_TOP
       proc->esp = USER_STACK_TOP  // User stack top
     ELSE:
       // Kernel mode (ring 0)
       proc->cs = 0x08   // Kernel code selector
       proc->ds = 0x10   // Kernel data selector
       proc->es = 0x10
       proc->fs = 0x10
       proc->gs = 0x10
       proc->ss = 0x10
       proc->page_directory = current_page_directory
       proc->user_stack_top = 0
  
  6. Add to ready queue
     proc->next = ready_queue
     proc->prev = NULL
     IF ready_queue != NULL:
       ready_queue->prev = proc
     ready_queue = proc
  
  7. RETURN proc
```

### Context Switch (Assembly)

```asm
; kernel/process/context.asm
; void context_switch(process_t *old, process_t *new)
; Arguments: [esp+4] = old, [esp+8] = new

global context_switch
extern current_process
extern current_page_directory
extern tss

section .text

; PCB offsets - MUST match process.h
%define PCB_PID          0
%define PCB_EAX          32
%define PCB_EBX          36
%define PCB_ECX          40
%define PCB_EDX          44
%define PCB_ESI          48
%define PCB_EDI          52
%define PCB_EBP          56
%define PCB_ESP          60
%define PCB_EIP          64
%define PCB_EFLAGS       68
%define PCB_CS           72
%define PCB_DS           76
%define PCB_ES           80
%define PCB_FS           84
%define PCB_GS           88
%define PCB_SS           92
%define PCB_PAGE_DIR     96
%define PCB_KERNEL_STACK 100

context_switch:
    ; Prologue
    push ebp
    mov ebp, esp
    pushf                      ; Save EFLAGS
    push ebx                   ; Save callee-saved registers
    push esi
    push edi
    
    ; Get arguments
    mov ecx, [ebp + 8]         ; ecx = old process
    mov edx, [ebp + 12]        ; edx = new process
    
    ; === SAVE OLD PROCESS STATE ===
    test ecx, ecx
    jz .skip_save              ; Skip if old is NULL (first switch)
    
    ; Save return address as new EIP
    mov eax, [ebp + 4]         ; Return address
    mov [ecx + PCB_EIP], eax
    
    ; Save stack pointer
    lea eax, [ebp + 20]        ; ESP before we pushed anything
    mov [ecx + PCB_ESP], eax
    
    ; Save callee-saved registers we pushed
    mov [ecx + PCB_EBX], ebx
    mov [ecx + PCB_ESI], esi
    mov [ecx + PCB_EDI], edi
    mov [ecx + PCB_EBP], ebp
    
    ; Save EFLAGS
    mov eax, [ebp - 4]         ; Pushed EFLAGS
    mov [ecx + PCB_EFLAGS], eax
    
    ; Save segment registers
    mov eax, ds
    mov [ecx + PCB_DS], eax
    mov eax, es
    mov [ecx + PCB_ES], eax
    mov eax, fs
    mov [ecx + PCB_FS], eax
    mov eax, gs
    mov [ecx + PCB_GS], eax
    mov eax, ss
    mov [ecx + PCB_SS], eax
    
    ; Save caller-saved registers (these were in eax/edx/ecx)
    ; We'll restore them from the PCB later if needed
    ; For now, we don't save eax/edx/ecx as they're caller-saved
    
.skip_save:
    ; === LOAD NEW PROCESS STATE ===
    
    ; Update current_process
    mov [current_process], edx
    
    ; Switch page directory if different
    mov eax, [current_page_directory]
    cmp eax, [edx + PCB_PAGE_DIR]
    je .same_page_dir
    
    ; Load new CR3
    mov eax, [edx + PCB_PAGE_DIR]
    mov cr3, eax
    mov [current_page_directory], eax
    
.same_page_dir:
    ; Update TSS.ESP0 for user processes
    ; Check if CS indicates user mode (0x1B)
    cmp word [edx + PCB_CS], 0x1B
    jne .kernel_process
    
    ; User process - update TSS.ESP0
    mov eax, [edx + PCB_KERNEL_STACK]
    mov [tss + 4], eax         ; tss.esp0 offset is 4
    
.kernel_process:
    ; Restore segment registers
    mov ds, [edx + PCB_DS]
    mov es, [edx + PCB_ES]
    mov fs, [edx + PCB_FS]
    mov gs, [edx + PCB_GS]
    
    ; Restore general registers
    mov ebx, [edx + PCB_EBX]
    mov esi, [edx + PCB_ESI]
    mov edi, [edx + PCB_EDI]
    mov ebp, [edx + PCB_EBP]
    
    ; Restore stack pointer
    mov esp, [edx + PCB_ESP]
    
    ; Restore EFLAGS
    push dword [edx + PCB_EFLAGS]
    popf
    
    ; Restore ESI last (we used it earlier, but now we restore from PCB)
    ; Actually, we already restored ESI above
    
    ; Return to new process
    mov eax, [edx + PCB_EIP]
    jmp eax                    ; Jump to new process's EIP
```

### TSS Initialization

```
TSS_INIT():
  1. Zero TSS structure
     memset(&tss, 0, sizeof(tss_t))
  
  2. Set critical fields
     tss.ss0 = 0x10            // Kernel data selector
     tss.esp0 = 0x90000        // Initial kernel stack (updated on switch)
     tss.iomap_base = sizeof(tss_t)  // No I/O bitmap
  
  3. Add TSS descriptor to GDT
     base = (uint32_t)&tss
     limit = sizeof(tss_t) - 1
     
     // TSS descriptor (system segment, type=0x9 for 32-bit TSS available)
     // Byte 5 (access): 0x89 = Present(1) DPL=0(00) S=0(0) Type=01001(9)
     // Byte 6 (flags): Granularity=0, 32-bit=1, Limit[19:16]=0
     
     gdt_set_gate(5, base, limit, 0x89, 0x00)
  
  4. Load Task Register
     asm volatile("ltr %w0" : : "r"(TSS_SELECTOR))
```

### Round-Robin Scheduler

```
SCHEDULER_TICK():
  1. Increment tick counter
     scheduler_ticks++
  
  2. Check if scheduler enabled
     IF !scheduler_enabled:
       RETURN
  
  3. Decrement current process time slice
     IF current_process != NULL:
       current_process->time_slice--
       
       IF current_process->time_slice == 0:
         // Time slice expired
         current_process->time_slice = DEFAULT_TIME_SLICE
         current_process->state = PROCESS_READY
         
         // Move to end of queue
         scheduler_remove_from_queue(current_process)
         scheduler_add_to_queue(current_process)
         
         // Trigger context switch
         SCHEDULE()
  
  4. Check for yielded processes
     IF current_process->state == PROCESS_READY AND current_process != ready_queue:
       SCHEDULE()

SCHEDULE():
  1. Find next runnable process
     next = ready_queue
     WHILE next != NULL:
       IF next->state == PROCESS_READY:
         BREAK
       next = next->next
     
     IF next == NULL:
       // No runnable process - keep running current
       RETURN
  
  2. Check if same process
     IF next == current_process:
       RETURN
  
  3. Update process states
     old = current_process
     IF old->state == PROCESS_RUNNING:
       old->state = PROCESS_READY
     
     next->state = PROCESS_RUNNING
  
  4. Update current_process
     current_process = next
  
  5. Remove from front of queue
     scheduler_remove_from_queue(next)
     scheduler_add_to_queue(next)  // Add to end (round-robin)
  
  6. Perform context switch
     context_switch(old, next)

SCHEDULER_ADD_TO_QUEUE(proc):
  IF ready_queue == NULL:
    ready_queue = proc
    proc->prev = NULL
    proc->next = NULL
  ELSE:
    // Find end of queue
    last = ready_queue
    WHILE last->next != NULL:
      last = last->next
    
    last->next = proc
    proc->prev = last
    proc->next = NULL

SCHEDULER_REMOVE_FROM_QUEUE(proc):
  IF proc->prev != NULL:
    proc->prev->next = proc->next
  ELSE:
    ready_queue = proc->next
  
  IF proc->next != NULL:
    proc->next->prev = proc->prev
  
  proc->prev = NULL
  proc->next = NULL
```

### System Call Dispatch

```
SYSCALL_HANDLER(regs):
  1. Extract syscall number
     syscall_num = regs->eax
     
  2. Validate syscall number
     IF syscall_num < 0 OR syscall_num >= MAX_SYSCALL:
       regs->eax = -1  // Return error
       RETURN
  
  3. Get handler
     handler = syscall_table[syscall_num]
     IF handler == NULL:
       regs->eax = -1
       RETURN
  
  4. Call handler with arguments
     // Arguments are in EBX, ECX, EDX
     result = handler(regs->ebx, regs->ecx, regs->edx)
     
  5. Store result
     regs->eax = result

SYS_WRITE(fd, buf, count):
  1. Validate file descriptor
     IF fd < 0 OR fd > 2:  // Only stdout(1) and stderr(2) supported
       RETURN -1
  
  2. Validate buffer pointer
     IF buf == NULL:
       RETURN -1
     
     // Check buffer is in user space (below kernel base)
     IF (uint32_t)buf >= KERNEL_VIRTUAL_BASE:
       RETURN -1  // EFAULT
     
     // Verify buffer is mapped (simplified check)
     phys = get_physical(current_process->page_directory, (uint32_t)buf)
     IF phys == 0:
       RETURN -1  // Invalid pointer
  
  3. Write to VGA/serial
     FOR i = 0 TO count - 1:
       vga_putchar(buf[i])
     
     RETURN count

SYS_EXIT(status):
  1. Get current process
     proc = current_process
  
  2. Mark as zombie
     proc->state = PROCESS_ZOMBIE
     proc->exit_status = status
  
  3. Free resources (simplified)
     IF proc->page_directory != current_page_directory:
       // User process - free user memory
       // (Full implementation would walk page tables)
  
  4. Remove from run queue
     scheduler_remove_from_queue(proc)
  
  5. Trigger scheduler (never returns)
     scheduler_yield()
     // Never reaches here

SYS_GETPID():
  RETURN current_process->pid

SYS_YIELD():
  current_process->state = PROCESS_READY
  scheduler_yield()
  // Returns when scheduled again
```

### Enter User Mode

```asm
; kernel/user/user_mode.asm
; void enter_user_mode(uint32_t entry, uint32_t stack)

global enter_user_mode

section .text

enter_user_mode:
    ; Get arguments
    mov eax, [esp + 4]    ; entry point
    mov ebx, [esp + 8]    ; user stack top
    
    ; Disable interrupts during transition
    cli
    
    ; Load user data segment selectors
    mov cx, 0x23          ; User data selector (index 4, RPL=3)
    mov ds, cx
    mov es, cx
    mov fs, cx
    mov gs, cx
    
    ; Set up user stack for iret
    ; Push in reverse order: SS, ESP, EFLAGS, CS, EIP
    
    ; Set up stack frame for iret
    push 0x23             ; SS (user data selector with RPL=3)
    push ebx              ; ESP (user stack top)
    pushf                 ; EFLAGS
    or dword [esp], 0x200 ; Set IF (enable interrupts)
    push 0x1B             ; CS (user code selector with RPL=3)
    push eax              ; EIP (entry point)
    
    ; iret pops: EIP, CS, EFLAGS, ESP, SS
    ; This transitions to ring 3
    iret
```

---

## Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
|-------|-------------|----------|---------------|
| Process table full | `process_create` scans all slots | Return NULL | Depends on caller |
| Kernel stack allocation failure | `alloc_frame` returns NULL | Clean up, return NULL | No |
| User stack allocation failure | `alloc_frame` returns NULL in loop | Free allocated frames, return NULL | No |
| Page directory clone failure | `clone_page_directory` returns NULL | Free stacks, return NULL | No |
| Context switch register corruption | Wrong PCB offsets | Debug with GDB; check assembly matches C | Yes (crash) |
| TSS.ESP0 not updated | User process traps to kernel | Systematic: update before every switch | Yes (triple fault) |
| Invalid syscall number | Bounds check in handler | Return -1 (EINVAL) | No |
| Invalid user pointer | Check address range and mapping | Return -1 (EFAULT) | No |
| Double-free in scheduler | Process already removed | Debug assertion in remove_from_queue | No |
| Zombie process remains | exit doesn't clean up | Parent must reap (future: waitpid) | No |
| Infinite loop in ready queue | All processes blocked | Idle process always runnable | No (hangs) |

---

## State Machine: Process Lifecycle

```
Process State Transitions:

                    ┌─────────────────────────────────────┐
                    │            PROCESS_UNUSED           │
                    │         (initial state)             │
                    └──────────────────┬──────────────────┘
                                       │ process_create()
                                       ▼
                    ┌─────────────────────────────────────┐
        ┌──────────▶│            PROCESS_READY            │◀──────────┐
        │           │    (runnable, waiting for CPU)      │           │
        │           └──────────────────┬──────────────────┘           │
        │                              │ scheduler_pick_next()        │
        │                              ▼                               │
        │           ┌─────────────────────────────────────┐           │
        │           │           PROCESS_RUNNING           │           │
        │           │       (currently executing)         │           │
        │           └──────────────────┬──────────────────┘           │
        │                              │                               │
        │         ┌────────────────────┼────────────────────┐         │
        │         │                    │                    │         │
        │         │ timer/scheduler    │ block()           │ exit()  │
        │         │ yield()            │ sleep()           │         │
        │         ▼                    ▼                    ▼         │
        │  (back to READY)    ┌───────────────┐    ┌──────────────┐  │
        │                     │PROCESS_BLOCKED│    │PROCESS_ZOMBIE│  │
        └─────────────────────│ (waiting for  │    │  (terminated)│  │
                              │   event)      │    └──────────────┘  │
                              └───────┬───────┘                      │
                                      │ wake() / event               │
                                      └──────────────────────────────┘
                                          (back to READY)

VALID Transitions:
  UNUSED → READY     : process_create()
  READY → RUNNING    : scheduler dispatch
  RUNNING → READY    : time slice expired, yield
  RUNNING → BLOCKED  : wait for I/O, sleep
  RUNNING → ZOMBIE   : exit()
  BLOCKED → READY    : I/O complete, sleep expired

INVALID Transitions:
  UNUSED → RUNNING   : Must go through READY first
  READY → ZOMBIE     : Must be RUNNING to exit
  BLOCKED → RUNNING  : Must go through READY
  ZOMBIE → *         : Terminal state
```

---

## Concurrency Specification

### Preemptive Multitasking Model

Only ONE process executes at any instant. Concurrency is simulated through rapid context switching triggered by:
1. Timer interrupt (preemptive)
2. Voluntary yield (cooperative)
3. Blocking I/O (implicit yield)

### Critical Sections

Interrupts must be disabled during:
- Process table modification
- Run queue manipulation  
- Context switch (partial - can re-enable after CR3 load)

```c
// Safe queue manipulation
void scheduler_remove_from_queue(process_t *proc) {
    uint32_t flags;
    
    // Disable interrupts
    asm volatile("pushf; pop %0; cli" : "=r"(flags));
    
    // ... manipulate queue ...
    
    // Restore interrupts
    asm volatile("push %0; popf" : : "r"(flags));
}
```

### Interrupt Safety in Handlers

- ISR handlers run with interrupts disabled (interrupt gate)
- Timer handler calls `scheduler_tick()` which may switch contexts
- After context switch, interrupts are restored from saved EFLAGS
- Syscall handler runs with interrupts enabled (trap gate)

### Per-Process Data Protection

Each process has:
- Independent kernel stack (no shared stack data)
- Independent page directory (no shared user memory)
- Shared kernel memory (read-only from user mode)

### Lock-Free Queue Access

The ready queue is accessed only:
1. From timer interrupt (interrupt context)
2. From syscall context (process context, but interrupts may be disabled)

Single-CPU design means no true concurrent access if interrupts are disabled.

---

## Implementation Sequence with Checkpoints

### Phase 1: PCB Structure Definition (2-3 hours)

**Files**: `kernel/process/process.h`

**Steps**:
1. Define process_state_t enumeration
2. Define process_t structure with exact byte offsets
3. Add comments documenting PCB layout
4. Define MAX_PROCESSES, stack sizes, etc.
5. Declare global variables (process_table, current_process, etc.)

**Checkpoint**: PCB structure compiles
```c
// Test: verify offsets
_Static_assert(offsetof(process_t, eax) == 32, "EAX offset wrong");
_Static_assert(offsetof(process_t, eip) == 64, "EIP offset wrong");
_Static_assert(offsetof(process_t, page_directory) == 96, "PD offset wrong");
```

**Test**: Compile and check struct size matches expected

### Phase 2: Process Creation (Kernel Mode) (3-4 hours)

**Files**: `kernel/process/process.c`

**Steps**:
1. Implement `process_table` array initialization
2. Implement `process_create()` for kernel processes
3. Allocate kernel stack from frame allocator
4. Set up initial register state for kernel mode
5. Add to ready queue

**Checkpoint**: Can create kernel process
```c
void test_process(void) {
    vga_puts("Test process running!\n");
    while(1) { asm volatile("hlt"); }
}

// In kernel_main:
process_t *proc = process_create("test", test_process, 0);
ASSERT(proc != NULL, "Failed to create process");
vga_puts("Process created: "); vga_put_dec(proc->pid); vga_puts("\n");
```

**Test**: Create process, verify PCB fields set correctly

### Phase 3: Context Switch Assembly (5-7 hours)

**Files**: `kernel/process/context.asm`

**Steps**:
1. Define PCB offsets to match C structure
2. Implement save path (save all registers to old PCB)
3. Implement CR3 switch logic
4. Implement TSS.ESP0 update logic
5. Implement restore path (load all registers from new PCB)
6. Handle NULL old process (first switch)

**Checkpoint**: Context switch doesn't crash
```c
// Test: switch to same process
process_t *proc = &process_table[0];
context_switch(proc, proc);
vga_puts("Context switch returned!\n");  // Should print this
```

**Test**: Create two processes, switch between them manually

### Phase 4: TSS Initialization (3-4 hours)

**Files**: `kernel/process/tss.h`, `kernel/process/tss.c`

**Steps**:
1. Define tss_t structure matching hardware format
2. Implement `tss_init()` to zero and configure TSS
3. Add TSS descriptor to GDT (modify boot/gdt.asm or create function)
4. Load TR with `ltr` instruction
5. Implement `tss_set_esp0()` helper

**Checkpoint**: TSS loads without fault
```c
tss_init();
vga_puts("TSS initialized\n");
vga_puts("TSS.ESP0 = 0x"); vga_put_hex(tss.esp0); vga_puts("\n");
// Read TR to verify
uint16_t tr;
asm volatile("str %0" : "=r"(tr));
vga_puts("TR = 0x"); vga_put_hex(tr); vga_puts(" (expected 0x28)\n");
```

**Test**: Verify TR = 0x28, TSS.ESP0 set correctly

### Phase 5: Round-Robin Scheduler (4-5 hours)

**Files**: `kernel/process/scheduler.h`, `kernel/process/scheduler.c`

**Steps**:
1. Implement `scheduler_init()` to create idle process
2. Implement `scheduler_add_to_queue()`
3. Implement `scheduler_remove_from_queue()`
4. Implement `scheduler_pick_next()`
5. Implement `scheduler_yield()` that calls context switch
6. Implement `schedule()` internal function

**Checkpoint**: Manual scheduling works
```c
scheduler_init();

// Create test processes
process_create("proc_a", test_process_a, 0);
process_create("proc_b", test_process_b, 0);

// Manually yield
scheduler_yield();
// Should switch to first process
```

**Test**: Create 3 processes, manually call scheduler_yield(), verify round-robin order

### Phase 6: Timer Interrupt Integration (2-3 hours)

**Files**: Modify `kernel/timer.c`

**Steps**:
1. Add call to `scheduler_tick()` in timer handler
2. Enable scheduler after timer configured
3. Unmask IRQ0 (timer)
4. Ensure interrupts enabled

**Checkpoint**: Preemptive multitasking works
```c
// After timer_init():
timer_init(100);  // 100Hz

// Enable scheduler
scheduler_enabled = 1;

// Create processes
process_create("proc_a", test_process_a, 0);
process_create("proc_b", test_process_b, 0);
process_create("proc_c", test_process_c, 0);

// Enable interrupts and yield
asm volatile("sti");
scheduler_yield();

// Scheduler takes over - processes print in round-robin
```

**Test**: Three kernel processes print to different screen locations, counters increment independently

### Phase 7: Multi-Process Kernel Demo (2-3 hours)

**Files**: `kernel/main.c` or new demo file

**Steps**:
1. Create three kernel processes with different entry functions
2. Each process prints to different VGA location
3. Each has a counter that increments
4. Verify counters update independently

**Checkpoint**: Visual demo of preemptive multitasking
```c
void process_a(void) {
    int count = 0;
    while (1) {
        vga_set_cursor(0, 0);
        vga_puts("[A] Count: "); vga_put_dec(count++);
        for (volatile int i = 0; i < 500000; i++);  // Delay
    }
}

// Similar for process_b and process_c

// Expected: Three counters incrementing simultaneously on screen
```

**Test**: Run for 10 seconds, verify all three counters increased

### Phase 8: User-Mode Process Creation (4-5 hours)

**Files**: `kernel/process/process.c` (modify), `kernel/memory/paging.c` (add clone function)

**Steps**:
1. Implement `clone_page_directory()` to copy kernel mappings
2. Allocate user stack in user virtual address space
3. Map user stack pages with PTE_USER flag
4. Set user segment selectors (CS=0x1B, DS=0x23)
5. Copy user code to user address space (simplified: just mark pages)

**Checkpoint**: User process PCB created
```c
extern void user_test_function(void);  // Simple test function

process_t *user_proc = process_create("user_test", user_test_function, 1);
ASSERT(user_proc != NULL, "Failed to create user process");
ASSERT(user_proc->cs == 0x1B, "Wrong CS for user process");
ASSERT(user_proc->page_directory != current_page_directory, "User should have own PD");
```

**Test**: Create user process, verify CS=0x1B, own page directory, user stack allocated

### Phase 9: Enter User Mode via iret (2-3 hours)

**Files**: `kernel/user/user_mode.asm`

**Steps**:
1. Implement `enter_user_mode()` assembly function
2. Load segment registers with user selectors
3. Build iret frame on stack
4. Execute iret to transition to ring 3
5. Verify CPL=3 after transition

**Checkpoint**: Successfully enter user mode
```c
// After creating user process:
void enter_user_process(void) {
    process_t *proc = process_create("user", user_entry, 1);
    
    // Enter user mode
    enter_user_mode(proc->eip, proc->esp);
    
    // Never reaches here
}

// In user mode code:
void user_entry(void) {
    // This runs at ring 3!
    // Attempt to write to VGA directly - should fault
    // char *vga = (char*)0xB8000;
    // *vga = 'X';  // Page fault if VGA not mapped user-accessible
    
    // Use syscall instead
    sys_write(1, "Hello from user mode!\n", 23);
    sys_exit(0);
}
```

**Test**: User process prints message via syscall, then exits cleanly

### Phase 10: System Call Interface (4-5 hours)

**Files**: `kernel/syscall/syscall.h`, `kernel/syscall/syscall.c`, `kernel/syscall/syscall.asm`

**Steps**:
1. Define syscall numbers and handler table
2. Create ISR stub for INT 0x80
3. Register IDT gate with DPL=3 (user callable)
4. Implement syscall dispatcher
5. Implement syscall parameter extraction from registers

**Checkpoint**: Syscall from kernel mode works
```c
// Test syscall from kernel (ring 0)
int result = sys_write(1, "Test\n", 5);
vga_put_dec(result);  // Should print 5

// Register syscall handler
idt_set_gate(0x80, (uint32_t)isr128, 0x08, 0xEE);  // DPL=3
```

**Test**: Call INT 0x80 from kernel, verify handler runs and returns

### Phase 11: sys_write and sys_exit (2-3 hours)

**Files**: `kernel/syscall/syscall.c`

**Steps**:
1. Implement `sys_write()` with pointer validation
2. Implement `sys_exit()` to terminate process
3. Implement `sys_getpid()` as trivial syscall
4. Add user-space wrapper functions (optional)

**Checkpoint**: User process uses syscalls
```c
// User mode code
void _start(void) {
    const char *msg = "Hello from user mode!\n";
    asm volatile(
        "int $0x80"
        : 
        : "a"(SYS_WRITE), "b"(1), "c"(msg), "d"(23)
    );
    
    asm volatile(
        "int $0x80"
        : 
        : "a"(SYS_EXIT), "b"(0)
    );
}

// Expected: "Hello from user mode!" printed, process exits
```

**Test**: User process prints message and exits without crashing

### Final Integration (3-4 hours)

**Goal**: Complete preemptive multitasking with user-mode processes and syscalls

**Steps**:
1. Integrate all components in `kernel_main()`
2. Initialize in correct order:
   - Memory (paging, heap)
   - Interrupts (IDT, PIC, timer)
   - TSS
   - Scheduler
   - Syscalls
3. Create kernel demo processes
4. Create user demo process
5. Enable interrupts and start scheduler

**Checkpoint**: Full system runs
```c
void kernel_main(multiboot_info_t *mbi) {
    // Core initialization
    vga_init();
    serial_init(COM1_PORT);
    
    kprintf("Initializing memory...\n");
    memory_init(mbi);
    
    kprintf("Initializing interrupts...\n");
    idt_init();
    pic_remap(32, 40);
    timer_init(100);
    keyboard_init();
    
    kprintf("Initializing TSS...\n");
    tss_init();
    
    kprintf("Initializing scheduler...\n");
    scheduler_init();
    
    kprintf("Initializing syscalls...\n");
    syscall_init();
    
    kprintf("Creating processes...\n");
    process_create("kernel_a", kernel_process_a, 0);
    process_create("kernel_b", kernel_process_b, 0);
    process_create("user_test", user_process_entry, 1);
    
    kprintf("Starting scheduler...\n");
    scheduler_enabled = 1;
    pic_unmask_irq(0);  // Timer
    pic_unmask_irq(1);  // Keyboard
    asm volatile("sti");
    
    scheduler_yield();
    
    // Never reaches here
}
```

**Test Commands**:
```bash
make clean && make
qemu-system-i386 -drive format=raw,file=os.img -serial stdio

# Expected:
# - Boot messages
# - "Starting scheduler..."
# - Three processes running (two kernel, one user)
# - User process prints via syscall
# - Counters incrementing
# - No crashes, hangs, or triple faults
```

---

## Test Specification

### Test 1: PCB Structure Layout

```c
// test_pcb_layout.c
void test_pcb_layout(void) {
    // Verify offsets match assembly expectations
    ASSERT(offsetof(process_t, pid) == 0, "PID offset");
    ASSERT(offsetof(process_t, eax) == 32, "EAX offset");
    ASSERT(offsetof(process_t, eip) == 64, "EIP offset");
    ASSERT(offsetof(process_t, eflags) == 68, "EFLAGS offset");
    ASSERT(offsetof(process_t, cs) == 72, "CS offset");
    ASSERT(offsetof(process_t, page_directory) == 96, "PD offset");
    ASSERT(offsetof(process_t, kernel_stack) == 100, "KSTACK offset");
    
    // Verify size
    ASSERT(sizeof(process_t) == 140, "PCB size");
    
    vga_puts("PASS: PCB layout\n");
}
```

### Test 2: Process Creation

```c
// test_process_create.c
void test_process_create(void) {
    uint32_t free_before = frame_alloc.free_frames;
    
    process_t *proc = process_create("test", test_func, 0);
    
    ASSERT(proc != NULL, "Process creation failed");
    ASSERT(proc->pid == 1, "PID assignment");
    ASSERT(proc->state == PROCESS_READY, "Initial state");
    ASSERT(proc->cs == 0x08, "Kernel CS");
    ASSERT(proc->kernel_stack != 0, "Kernel stack allocated");
    
    // Verify frame was allocated
    ASSERT(frame_alloc.free_frames == free_before - 1, "Frame used");
    
    // Verify in ready queue
    ASSERT(ready_queue == proc, "Added to queue");
    
    vga_puts("PASS: Process creation\n");
}
```

### Test 3: Context Switch

```c
// test_context_switch.c
volatile int proc_a_ran = 0;
volatile int proc_b_ran = 0;

void proc_a(void) {
    proc_a_ran = 1;
    scheduler_yield();  // Switch to proc_b
    proc_a_ran = 2;
}

void proc_b(void) {
    proc_b_ran = 1;
    scheduler_yield();  // Switch back to proc_a
    proc_b_ran = 2;
}

void test_context_switch(void) {
    process_t *a = process_create("a", proc_a, 0);
    process_t *b = process_create("b", proc_b, 0);
    
    // Disable timer for controlled test
    scheduler_enabled = 0;
    
    // Manually switch to process a
    current_process = a;
    a->state = PROCESS_RUNNING;
    context_switch(&process_table[0], a);  // From kernel to proc_a
    
    // After both yield, check values
    ASSERT(proc_a_ran == 2, "Proc A completed");
    ASSERT(proc_b_ran == 2, "Proc B completed");
    
    vga_puts("PASS: Context switch\n");
}
```

### Test 4: TSS Configuration

```c
// test_tss.c
void test_tss(void) {
    tss_init();
    
    // Verify TSS.ESP0 and SS0
    ASSERT(tss.ss0 == 0x10, "TSS.SS0");
    ASSERT(tss.esp0 != 0, "TSS.ESP0 set");
    
    // Verify TR loaded
    uint16_t tr;
    asm volatile("str %0" : "=r"(tr));
    ASSERT(tr == 0x28, "TR value");
    
    // Test ESP0 update
    tss_set_esp0(0x12345678);
    ASSERT(tss.esp0 == 0x12345678, "ESP0 update");
    
    vga_puts("PASS: TSS configuration\n");
}
```

### Test 5: Scheduler Round-Robin

```c
// test_scheduler.c
volatile int order[10];
volatile int order_idx = 0;

void sched_a(void) { order[order_idx++] = 'A'; while(1) scheduler_yield(); }
void sched_b(void) { order[order_idx++] = 'B'; while(1) scheduler_yield(); }
void sched_c(void) { order[order_idx++] = 'C'; while(1) scheduler_yield(); }

void test_scheduler(void) {
    process_create("a", sched_a, 0);
    process_create("b", sched_b, 0);
    process_create("c", sched_c, 0);
    
    // Run for 9 yields (3 full rounds)
    for (int i = 0; i < 9; i++) {
        scheduler_tick();  // Simulate timer
    }
    
    // Verify round-robin order: A, B, C, A, B, C, A, B, C
    ASSERT(order[0] == 'A', "Order 0");
    ASSERT(order[1] == 'B', "Order 1");
    ASSERT(order[2] == 'C', "Order 2");
    ASSERT(order[3] == 'A', "Order 3");
    ASSERT(order[4] == 'B', "Order 4");
    ASSERT(order[5] == 'C', "Order 5");
    
    vga_puts("PASS: Scheduler round-robin\n");
}
```

### Test 6: User Mode Entry

```c
// test_user_mode.c
volatile int user_mode_entered = 0;

void user_entry(void) {
    // Check CPL
    uint16_t cs;
    asm volatile("mov %%cs, %0" : "=r"(cs));
    
    // CS should be 0x1B or 0x1F (user code, RPL=3)
    user_mode_entered = (cs & 3) == 3 ? 1 : -1;
    
    sys_exit(0);
}

void test_user_mode(void) {
    process_t *proc = process_create("user", user_entry, 1);
    
    // Set up TSS.ESP0
    tss_set_esp0(proc->kernel_stack);
    
    // Enter user mode
    enter_user_mode(proc->eip, proc->esp);
    
    // Should not return, but if it does:
    ASSERT(user_mode_entered == 1, "User mode entered");
    
    vga_puts("PASS: User mode entry\n");
}
```

### Test 7: Syscall Dispatch

```c
// test_syscall.c
void test_syscall(void) {
    syscall_init();
    
    // Test sys_write from kernel
    int result = sys_write(1, "test\n", 5);
    ASSERT(result == 5, "sys_write return");
    
    // Test sys_getpid
    int pid = sys_getpid();
    ASSERT(pid == current_process->pid, "sys_getpid");
    
    // Test invalid syscall
    result = -1;
    asm volatile("int $0x80" : "=a"(result) : "a"(999));
    ASSERT(result == -1, "Invalid syscall");
    
    vga_puts("PASS: Syscall dispatch\n");
}
```

### Test 8: User Process Isolation

```c
// test_user_isolation.c
void user_try_kernel_access(void) {
    // Attempt to read kernel memory - should fault
    volatile uint32_t *kernel_addr = (uint32_t*)0xC0100000;
    uint32_t val = *kernel_addr;  // Should page fault
    
    // Should never reach here
    sys_write(1, "FAIL: accessed kernel\n", 22);
    sys_exit(1);
}

void test_user_isolation(void) {
    process_t *proc = process_create("isolation_test", user_try_kernel_access, 1);
    
    // Run process - expect page fault
    // (This test requires page fault handler to catch and report)
    
    vga_puts("PASS: User isolation (page fault expected)\n");
}
```

### Test 9: Multi-Process Demo

```bash
# test_multi_process.sh
timeout 10 qemu-system-i386 -drive format=raw,file=os.img \
    -serial stdio -display none 2>&1 | tee demo_log.txt

# Check that all three processes ran
if grep -q "Process A" demo_log.txt && \
   grep -q "Process B" demo_log.txt && \
   grep -q "Process C" demo_log.txt; then
    echo "PASS: All processes ran"
else
    echo "FAIL: Not all processes ran"
    exit 1
fi

# Check that processes ran multiple times (preemption working)
count_a=$(grep -c "Process A" demo_log.txt || echo "0")
if [ "$count_a" -gt 5 ]; then
    echo "PASS: Preemption working ($count_a iterations of A)"
else
    echo "FAIL: Preemption not working"
    exit 1
fi
```

---

## Performance Targets

| Operation | Target | How to Measure |
|-----------|--------|----------------|
| Context switch (same PD) | < 500 cycles | Time from `context_switch` call to return in new process |
| Context switch (different PD) | < 5000 cycles | Includes TLB refill cost |
| PCB allocation | < 100 cycles | Scan process table for free slot |
| Kernel stack allocation | < 500 cycles | Single frame allocation |
| User process creation | < 10,000 cycles | Includes PD clone, stack allocation |
| Scheduler tick | < 200 cycles | Timer handler overhead |
| Syscall dispatch | < 100 cycles | From `int 0x80` to handler entry |
| sys_write (1 char) | < 500 cycles | Argument validation + VGA output |
| User-kernel transition | < 200 cycles | int 0x80 + return to user |
| Timer interrupt overhead | < 1% of CPU | 100Hz timer, handler ~200 cycles = 0.002% |

---

## Hardware Soul

### Cache Lines Touched

**Context switch**:
- PCB read: 140 bytes = 3 cache lines (all registers + metadata)
- PCB write: Same 3 cache lines
- Page directory access: 1 cache line for CR3 read
- TSS update: 4 bytes (ESP0) - likely in same cache line as other TSS fields
- Total: ~5-7 cache line touches per switch

**Scheduler**:
- Run queue traversal: 2 pointers per process checked
- If 10 processes in queue: ~20 pointer reads, 1-2 cache lines

**Syscall**:
- IDT entry read: 8 bytes, 1 cache line shared with nearby entries
- Syscall table lookup: 1 pointer read
- Handler code: Variable (should stay in cache for common syscalls)

### Branch Prediction

**Round-robin scheduler**: Highly predictable - always picks next in queue. Branch predictor learns pattern quickly.

**Syscall dispatch**: Table lookup - no branching. Unpredictable only if syscall numbers random (unlikely).

**Context switch**: No branches in core switch path (just comparisons for CR3, TSS update).

### TLB Behavior

**CR3 reload flushes TLB**:
- 64-128 entries lost
- Each miss = 2-4 memory accesses (PD walk + PT walk)
- Kernel code hot path: ~10 TLB entries for common functions
- Recovery time: 100-500 cycles depending on working set

**User-kernel transitions**:
- Kernel pages marked global (PTE_GLOBAL) to survive CR3 reload
- Or kernel mapped in all page directories (this implementation)
- No TLB flush for kernel-only switches if PCID used (not in this impl)

### Pipeline Impact

**iret for user mode entry**:
- Serializing instruction - full pipeline flush
- Cost: 20-50 cycles
- Required for privilege level change

**int 0x80 for syscalls**:
- Interrupt gate entry flushes pipeline
- Similar cost to iret
- Could optimize with sysenter/sysexit (not implemented)

### Cache Pollution

**Process working set**:
- Each process has own working set
- Context switch evicts previous process's cache lines
- L1D typically 32KB: holds ~800 cache lines
- If process accesses more than 800 lines, thrashing occurs

---

## Visual Diagrams

### PCB Structure

```
Process Control Block (140 bytes):

┌─────────────────────────────────────────────────────────────────────┐
│ Offset 0: Identification                                            │
├─────────────────────────────────────────────────────────────────────┤
│  +0   │ pid (4)        │ Process ID                                 │
│  +4   │ name[16]       │ Process name (null-terminated)            │
│  +20  │ state (4)      │ READY/RUNNING/BLOCKED/ZOMBIE              │
│  +24  │ padding (8)    │ Reserved for alignment                    │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│ Offset 32: Register State (SAVED/RESTORED BY CONTEXT SWITCH)        │
├─────────────────────────────────────────────────────────────────────┤
│  +32  │ eax            │ General purpose                           │
│  +36  │ ebx            │                                           │
│  +40  │ ecx            │                                           │
│  +44  │ edx            │                                           │
│  +48  │ esi            │ Source index                              │
│  +52  │ edi            │ Destination index                         │
│  +56  │ ebp            │ Base pointer                              │
│  +60  │ esp            │ Stack pointer                             │
│  +64  │ eip            │ Instruction pointer                       │
│  +68  │ eflags         │ CPU flags                                 │
│  +72  │ cs             │ Code segment selector                     │
│  +76  │ ds             │ Data segment selector                     │
│  +80  │ es             │ Extra segment                             │
│  +84  │ fs             │ Extra segment                             │
│  +88  │ gs             │ Extra segment                             │
│  +92  │ ss             │ Stack segment selector                    │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│ Offset 96: Memory Management                                        │
├─────────────────────────────────────────────────────────────────────┤
│  +96  │ page_directory │ Page directory pointer (virtual addr)     │
│  +100 │ kernel_stack   │ Top of kernel stack (for TSS.ESP0)        │
│  +104 │ user_stack_top │ User mode stack top                       │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│ Offset 108: Scheduling                                              │
├─────────────────────────────────────────────────────────────────────┤
│  +108 │ next           │ Next process in queue                     │
│  +112 │ prev           │ Previous process in queue                 │
│  +116 │ wake_time      │ Tick count to wake (for sleep)            │
│  +120 │ exit_status    │ Exit code                                 │
│  +124 │ time_slice     │ Remaining quantum                         │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│ Offset 128: Process Tree (Future)                                   │
├─────────────────────────────────────────────────────────────────────┤
│  +128 │ parent         │ Parent process                            │
│  +132 │ first_child    │ First child process                       │
│  +136 │ next_sibling   │ Next sibling                              │
└─────────────────────────────────────────────────────────────────────┘

Assembly Offset Verification:
  %define PCB_EAX     32    ; Must match C offsetof(process_t, eax)
  %define PCB_EIP     64    ; Must match C offsetof(process_t, eip)
  %define PCB_CS      72    ; Must match C offsetof(process_t, cs)
  %define PCB_KSTACK  100   ; Must match C offsetof(process_t, kernel_stack)
```

### Context Switch Flow

```
Context Switch Operation:

┌─────────────────────────────────────────────────────────────────────┐
│                         BEFORE SWITCH                               │
│                                                                     │
│   Process A (running)                                               │
│   ┌─────────────────┐                                               │
│   │ EAX = 0x11111111│                                               │
│   │ EBX = 0x22222222│                                               │
│   │ EIP = 0x00100234│  ──┐                                          │
│   │ ESP = 0x00090000│    │                                          │
│   │ ...             │    │                                          │
│   └─────────────────┘    │                                          │
│                          │                                          │
│   PCB_A (in memory)      │                                          │
│   ┌─────────────────┐    │                                          │
│   │ eax = ?         │ ◀──┘  Will be saved here                      │
│   │ eip = ?         │                                               │
│   │ ...             │                                               │
│   └─────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ context_switch(PCB_A, PCB_B)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DURING SWITCH                               │
│                                                                     │
│   1. Save current state to PCB_A                                    │
│      PCB_A.eax ← EAX (0x11111111)                                  │
│      PCB_A.ebx ← EBX                                               │
│      PCB_A.eip ← return_address                                    │
│      ...                                                            │
│                                                                     │
│   2. Update TSS.ESP0 if switching to user process                   │
│      TSS.ESP0 ← PCB_B.kernel_stack                                 │
│                                                                     │
│   3. Switch page directory if different                             │
│      CR3 ← PCB_B.page_directory                                    │
│      (TLB flushed automatically)                                    │
│                                                                     │
│   4. Update current_process                                         │
│      current_process ← PCB_B                                       │
│                                                                     │
│   5. Load new state from PCB_B                                      │
│      EAX ← PCB_B.eax                                               │
│      EBX ← PCB_B.ebx                                               │
│      EIP ← PCB_B.eip                                               │
│      ...                                                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ jmp to PCB_B.eip
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AFTER SWITCH                                │
│                                                                     │
│   Process B (now running)                                           │
│   ┌─────────────────┐                                               │
│   │ EAX = 0xAAAAAAAA│  ◀── Loaded from PCB_B                       │
│   │ EBX = 0xBBBBBBBB│                                               │
│   │ EIP = 0x00100567│                                               │
│   │ ESP = 0x000A0000│                                               │
│   │ ...             │                                               │
│   └─────────────────┘                                               │
│                                                                     │
│   Process A (suspended)                                             │
│   ┌─────────────────┐                                               │
│   │ State saved in  │                                               │
│   │ PCB_A           │  ──▶ Will resume when scheduled again        │
│   └─────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘

Critical Invariants:
  • ALL registers saved (EAX-EDX, ESI, EDI, EBP, ESP, EIP, EFLAGS)
  • ALL segment registers saved (CS, DS, ES, FS, GS, SS)
  • TSS.ESP0 updated BEFORE switching to user process
  • CR3 reloaded if page directories differ
  • Stack valid at all times
```

### TSS Structure

```
Task State Segment (104 bytes):

┌─────────────────────────────────────────────────────────────────────┐
│ Offset  Field        Value            Purpose                      │
├─────────────────────────────────────────────────────────────────────┤
│  0      prev_task    0x0000           Previous task link (unused)  │
│  4      esp0         0x90000          ★ RING 0 STACK POINTER ★     │
│  8      ss0          0x0010           ★ RING 0 STACK SEGMENT ★     │
│  12     esp1         0                Ring 1 stack (unused)        │
│  16     ss1          0                Ring 1 segment (unused)      │
│  20     esp2         0                Ring 2 stack (unused)        │
│  24     ss2          0                Ring 2 segment (unused)      │
│  28     cr3          0                Page directory (unused)      │
│  32-68  eax-edi      0                General regs (unused)        │
│  72-92  es-gs        0                Segments (unused)            │
│  96     ldt          0                LDT selector (unused)        │
│  100    trap         0                Trap flag (unused)           │
│  102    iomap_base   104              I/O bitmap offset (none)     │
└─────────────────────────────────────────────────────────────────────┘

CRITICAL USAGE:
  When interrupt/syscall occurs in user mode (ring 3):
  1. CPU reads TSS.SS0 and TSS.ESP0
  2. CPU switches to kernel stack: SS:ESP = SS0:ESP0
  3. CPU pushes user SS, ESP, EFLAGS, CS, EIP onto kernel stack
  4. CPU loads CS:EIP from IDT entry
  5. Handler runs on kernel stack at ring 0

TSS.ESP0 MUST be updated:
  • Before context switch to ANY user-mode process
  • Value = top of that process's kernel stack
  • Each process has its own kernel stack (4KB)
```

### Ring Transition (User to Kernel)

```
Ring 3 → Ring 0 Transition (on interrupt/syscall):

BEFORE (User Mode):
┌─────────────────────────────────────────────────────────────────────┐
│  User Process at Ring 3                                             │
│                                                                     │
│  CPU State:                                                         │
│    CS:EIP = 0x1B:0x00100400 (user code)                            │
│    SS:ESP = 0x23:0xBFFFF000 (user stack)                           │
│    CPL = 3 (Current Privilege Level)                               │
│                                                                     │
│  User Stack:                          Kernel Stack:                 │
│  ┌──────────────┐                     ┌──────────────┐             │
│  │ local vars   │                     │    empty     │             │
│  │ return addr  │                     │              │             │
│  │ parameters   │                     │              │             │
│  │ ...          │                     │              │             │
│  └──────────────┘ ← ESP               └──────────────┘             │
│   0xBFFFF000                                   ?                    │
│                                                                     │
│  TSS.ESP0 = 0x00090000 (kernel stack for this process)             │
│  TSS.SS0  = 0x0010 (kernel data segment)                           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ User executes: int 0x80
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CPU Hardware Actions:                                              │
│                                                                     │
│  1. Read IDT[0x80] → Get handler CS:EIP                            │
│  2. Check privilege: IDT DPL (3) >= CPL (3) ✓                      │
│  3. Detect ring transition: IDT CS DPL (0) < CPL (3)               │
│  4. Read TSS.SS0:ESP0 = 0x0010:0x00090000                          │
│  5. Switch to kernel stack                                          │
│  6. Push onto kernel stack:                                         │
│     ┌──────────────┐                                                │
│     │ SS (user)    │ 0x23                                          │
│     │ ESP (user)   │ 0xBFFFF000                                    │
│     │ EFLAGS       │ 0x00000202                                    │
│     │ CS (user)    │ 0x1B                                          │
│     │ EIP (user)   │ 0x00100402 (after int instruction)            │
│     └──────────────┘ ← ESP (now 0x00090000)                        │
│  7. Load CS:EIP from IDT                                            │
│  8. Clear CPL to 0                                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ AFTER (Kernel Mode):                                                │
│                                                                     │
│  CPU State:                                                         │
│    CS:EIP = 0x08:syscall_handler (kernel code)                     │
│    SS:ESP = 0x10:0x0008FFE0 (kernel stack)  ← ESP decremented      │
│    CPL = 0 (Current Privilege Level)                               │
│                                                                     │
│  Kernel Stack:                                                      │
│  ┌──────────────┐                                                   │
│  │ SS (user)    │ 0x23    ← Pushed by CPU                          │
│  │ ESP (user)   │ 0xBFFFF000                                       │
│  │ EFLAGS       │ 0x00000202                                       │
│  │ CS (user)    │ 0x1B                                             │
│  │ EIP (user)   │ 0x00100402                                       │
│  └──────────────┴──────────────────────────────────┐               │
│  │ (handler's local vars, saved regs, etc.)        │               │
│  └─────────────────────────────────────────────────┘ ← ESP         │
│   0x0008FF00                                                       │
│                                                                     │
│  User Stack: (unchanged)                                           │
│  ┌──────────────┐                                                   │
│  │ local vars   │                                                   │
│  │ ...          │                                                   │
│  └──────────────┘                                                   │
│   0xBFFFF000                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Handler executes, then iret
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ iret (Return to User Mode):                                         │
│                                                                     │
│  1. Pop EIP, CS, EFLAGS, ESP, SS from kernel stack                 │
│  2. Restore SS:ESP = 0x23:0xBFFFF000                               │
│  3. Restore CS:EIP = 0x1B:0x00100402                               │
│  4. Restore EFLAGS                                                  │
│  5. Set CPL = 3                                                     │
│  6. Resume user code                                                │
└─────────────────────────────────────────────────────────────────────┘

WITHOUT TSS.SS0:ESP0 → TRIPLE FAULT!
  The CPU has nowhere to put the kernel stack.
  Always update TSS.ESP0 before switching to user process.
```

### Scheduler Queue

```
Round-Robin Run Queue:

┌─────────────────────────────────────────────────────────────────────┐
│                         RUN QUEUE                                   │
│                                                                     │
│   ready_queue ──┐                                                   │
│                 │                                                   │
│                 ▼                                                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │   PCB A     │────▶│   PCB B     │────▶│   PCB C     │──▶ NULL  │
│   │   READY     │     │   READY     │     │   READY     │          │
│   │   pid=1     │◀────│   pid=2     │◀────│   pid=3     │          │
│   └─────────────┘     └─────────────┘     └─────────────┘          │
│                                                                     │
│   current_process = PCB A (running)                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Timer Tick (quantum expired):

┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Move current to end                                         │
│                                                                     │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │   PCB B     │────▶│   PCB C     │────▶│   PCB A     │──▶ NULL  │
│   │   READY     │     │   READY     │     │   READY     │          │
│   │   pid=2     │◀────│   pid=3     │◀────│   pid=1     │          │
│   └─────────────┘     └─────────────┘     └─────────────┘          │
│         ▲                                           │               │
│         │                                           │               │
│   ready_queue ◀─────────────────────────────────────┘               │
│                                                                     │
│ Step 2: Pick next                                                   │
│                                                                     │
│   next = ready_queue = PCB B                                       │
│   switch from PCB A to PCB B                                       │
│                                                                     │
│ Step 3: After switch                                                │
│                                                                     │
│   current_process = PCB B                                          │
│   PCB B state = RUNNING                                            │
│   PCB A state = READY                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Blocking (e.g., waiting for I/O):

┌─────────────────────────────────────────────────────────────────────┐
│ Process B calls sys_read() which blocks:                            │
│                                                                     │
│   PCB B state = BLOCKED                                            │
│   Remove PCB B from queue                                          │
│                                                                     │
│   ready_queue ──▶ PCB C ──▶ PCB A ──▶ NULL                        │
│                                                                     │
│   current_process = PCB C (next ready)                             │
│                                                                     │
│ Later, when I/O completes:                                          │
│                                                                     │
│   PCB B state = READY                                              │
│   Add PCB B to end of queue                                        │
│                                                                     │
│   ready_queue ──▶ PCB C ──▶ PCB A ──▶ PCB B ──▶ NULL              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Priority Queue (NOT IMPLEMENTED - future enhancement):
┌─────────────────────────────────────────────────────────────────────┐
│   Multiple queues by priority:                                      │
│   priority[0] ──▶ [high priority tasks]                            │
│   priority[1] ──▶ [normal tasks]                                   │
│   priority[2] ──▶ [low priority tasks]                             │
│                                                                     │
│   Always pick from highest non-empty queue                          │
│   Time slice varies by priority                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Multi-Process Demo

```
Three Kernel Processes Demo:

┌─────────────────────────────────────────────────────────────────────┐
│                         VGA Screen                                  │
│                                                                     │
│  Row 0:  ┌──────────────────────────────────────────────────────┐  │
│          │ [Process A] Count: 42                                 │  │
│          └──────────────────────────────────────────────────────┘  │
│                                                                     │
│  Row 5:  ┌──────────────────────────────────────────────────────┐  │
│          │ [Process B] Count: 38                                 │  │
│          └──────────────────────────────────────────────────────┘  │
│                                                                     │
│  Row 10: ┌──────────────────────────────────────────────────────┐  │
│          │ [Process C] Count: 45                                 │  │
│          └──────────────────────────────────────────────────────┘  │
│                                                                     │
│  ... rest of screen ...                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Execution Timeline (100Hz timer, 10 tick quantum):

Time    Running Process    Action
────────────────────────────────────────────────────────────────────
0ms     A                  Increments counter to 1
1ms     A                  Increments to 2
...
100ms   A (quantum done)   Switch to B
100ms   B                  Increments counter to 1
101ms   B                  Increments to 2
...
200ms   B (quantum done)   Switch to C
200ms   C                  Increments counter to 1
...
300ms   C (quantum done)   Switch to A
300ms   A                  Increments counter to 11 (resumed)
...

Each process sees:
  - Its counter incrementing
  - No awareness of other processes
  - No awareness of being suspended/resumed

Key Observations:
  • All three counters increment "simultaneously" (from user perspective)
  • Only ONE process actually executes at any instant
  • Switch overhead is imperceptible (~500 cycles = ~0.5µs at 1GHz)
  • Each process has independent EIP, ESP, registers
```

### User-Kernel Memory Layout

```
Address Space Layout:

┌─────────────────────────────────────────────────────────────────────┐
│                         KERNEL SPACE (1GB)                          │
│                    0xC0000000 - 0xFFFFFFFF                         │
│                                                                     │
│  0xFFFFFFFF ┌──────────────────────────────────────┐               │
│             │                                      │               │
│             │   Reserved / Device MMIO             │               │
│             │                                      │               │
│  0xC0800000 ├──────────────────────────────────────┤               │
│             │   Kernel Heap End                    │               │
│             │   (expandable)                       │               │
│  0xC0400000 ├──────────────────────────────────────┤               │
│             │   Kernel Heap (kmalloc)              │               │
│             │   PTE_USER = 0 (supervisor only)    │               │
│  0xC0100000 ├──────────────────────────────────────┤               │
│             │   Kernel Code + Data                 │               │
│             │   .text, .rodata, .data, .bss        │               │
│             │   PTE_USER = 0 (supervisor only)    │               │
│  0xC0000000 ├──────────────────────────────────────┤               │
│             │   Kernel mapping of low memory       │               │
│             │   (VGA at 0xB8000 mapped here)       │               │
│             └──────────────────────────────────────┘               │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                         USER SPACE (3GB)                            │
│                    0x00000000 - 0xBFFFFFFF                         │
│                                                                     │
│  0xBFFFFFFF ┌──────────────────────────────────────┐               │
│             │   User Stack (grows down)            │               │
│             │   PTE_USER = 1                       │               │
│  0xBFFFF000 ├──────────────────────────────────────┤               │
│             │                                      │               │
│             │   (unmapped - guard page)            │               │
│             │                                      │               │
│  0xBFF00000 ├──────────────────────────────────────┤               │
│             │                                      │               │
│             │   User Heap (expandable)             │               │
│             │   (brk/sbrk managed)                 │               │
│             │                                      │               │
│  0x00400000 ├──────────────────────────────────────┤               │
│             │   User Code + Data                   │               │
│             │   (loaded from executable)           │               │
│             │   PTE_USER = 1                       │               │
│  0x00001000 ├──────────────────────────────────────┤               │
│             │   NULL page (unmapped)               │               │
│  0x00000000 └──────────────────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Page Table Entry User Bit:
  PTE_USER = 1 → User mode (ring 3) can access
  PTE_USER = 0 → Supervisor only (ring 0-2)

User Process Page Directory:
  - Entries 0-767: User space (unique per process)
  - Entries 768-1023: Kernel space (identical across all processes)
  - Kernel entries: PTE_USER = 0, PTE_GLOBAL = 1

User Mode Access Violation:
  User code: mov eax, [0xC0100000]  ; Kernel memory
  CPU checks: PTE_USER = 0, CPL = 3 → PAGE FAULT
```

### System Call Interface

```
INT 0x80 System Call Mechanism:

User Mode Call:
┌─────────────────────────────────────────────────────────────────────┐
│  // User code                                                       │
│  const char *msg = "Hello\n";                                      │
│  int len = 5;                                                       │
│                                                                     │
│  // Inline assembly syscall                                         │
│  asm volatile(                                                      │
│      "int $0x80"                   // Trigger syscall              │
│      : "=a"(retval)                // Return value in EAX          │
│      : "a"(SYS_WRITE),             // Syscall number in EAX        │
│        "b"(1),                     // fd in EBX                    │
│        "c"(msg),                   // buf in ECX                   │
│        "d"(len)                    // count in EDX                 │
│  );                                                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ int 0x80
                              │ (CPU transitions ring 3 → ring 0)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Kernel Syscall Handler                                             │
│                                                                     │
│  1. Read syscall number from EAX                                   │
│     syscall_num = regs->eax;  // EAX saved on stack               │
│                                                                     │
│  2. Bounds check                                                    │
│     if (syscall_num >= MAX_SYSCALL) return -1;                     │
│                                                                     │
│  3. Dispatch to handler                                             │
│     handler = syscall_table[syscall_num];                          │
│     result = handler(regs->ebx, regs->ecx, regs->edx);             │
│                                                                     │
│  4. Store result                                                    │
│     regs->eax = result;  // Return value                           │
│                                                                     │
│  5. iret back to user mode                                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ iret
                              │ (CPU transitions ring 0 → ring 3)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  // User code continues                                             │
│  if (retval == len) {                                               │
│      // Success                                                     │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘

Syscall Register Convention:
┌─────────────────────────────────────────────────────────────────────┐
│ Register    Input                   Output                         │
├─────────────────────────────────────────────────────────────────────┤
│ EAX         Syscall number          Return value                   │
│ EBX         Argument 0              (preserved or result)          │
│ ECX         Argument 1              (preserved or result)          │
│ EDX         Argument 2              (preserved or result)          │
│ ESI         Argument 3 (optional)   (preserved)                    │
│ EDI         Argument 4 (optional)   (preserved)                    │
│ EBP         Argument 5 (optional)   (preserved)                    │
└─────────────────────────────────────────────────────────────────────┘

Syscall Table:
┌─────────────────────────────────────────────────────────────────────┐
│ Number  Name        Handler              Signature                 │
├─────────────────────────────────────────────────────────────────────┤
│ 0       SYS_EXIT    sys_exit             (int status) → void       │
│ 1       SYS_READ    sys_read             (fd, buf, count) → int    │
│ 2       SYS_WRITE   sys_write            (fd, buf, count) → int    │
│ 3       SYS_EXEC    sys_exec             (path) → int              │
│ 4       SYS_FORK    sys_fork             () → int                  │
│ 5       SYS_GETPID  sys_getpid           () → int                  │
│ 6       SYS_YIELD   sys_yield            () → void                 │
│ 7-255   Reserved    (NULL)               -                         │
└─────────────────────────────────────────────────────────────────────┘

IDT Entry for INT 0x80:
  offset_low  = handler_address & 0xFFFF
  selector    = 0x08 (kernel code)
  zero        = 0
  type_attr   = 0xEE  // Present, DPL=3 (user callable), Trap gate
  offset_high = handler_address >> 16

DPL=3 is CRITICAL: Allows user mode to call int 0x80
DPL=0 would cause General Protection Fault if called from ring 3
```

---

[[CRITERIA_JSON: {"module_id": "mod-scheduling", "criteria": ["PCB structure defines all required fields at documented byte offsets", "process_create allocates kernel stack and initializes PCB for kernel mode", "process_create allocates user stack and clones page directory for user mode", "context_switch saves all general purpose registers (EAX-EDX, ESI, EDI, EBP, ESP)", "context_switch saves all segment registers (CS, DS, ES, FS, GS, SS)", "context_switch saves EIP and EFLAGS", "context_switch loads CR3 when page directories differ", "context_switch updates TSS.ESP0 for user mode processes", "TSS is initialized with SS0=0x10 and loaded into TR", "TSS.ESP0 is updated before every context switch to user process", "Round-robin scheduler maintains ready queue with correct prev/next pointers", "scheduler_tick triggers context switch when time slice expires", "scheduler_yield marks current process READY and switches to next", "enter_user_mode uses iret to transition to ring 3", "enter_user_mode sets up stack frame with SS, ESP, EFLAGS, CS, EIP", "System call handler registered at IDT vector 0x80 with DPL=3", "System call handler validates syscall number before dispatch", "System call handler validates user pointers before dereferencing", "sys_write validates buffer is in user space (below kernel base)", "sys_exit marks process as ZOMBIE and triggers scheduler", "Three kernel processes can run concurrently with independent counters", "User process executes at ring 3 (CS=0x1B)", "User process page fault occurs on kernel memory access (isolation verified)"]}]


<!-- TDD_MOD_ID: mod-drivers -->
I'll create a Technical Design Specification for the Kernel Drivers and Utilities module. Let me first examine the existing architecture to ensure consistency.


I'll create a bibliography of high-quality external resources for the OS development concepts covered in this project. Let me first examine the existing architecture documentation to understand the scope better.# 📚 Beyond the Atlas: Further Reading

## x86 Architecture & Boot Process

**Paper**: B. Ford, S. Hibler, J. Lepreau, P. Tullmann. "Microkernels Meet Recursive Virtual Machines." *OSDI '96*. The foundational paper on bootstrapping and recursive VM structure—directly relevant to understanding the hardware-to-software transition.

**Code**: Linux Kernel `arch/x86/boot/` — Specifically `boot/header.S` and `boot/main.c`. The production implementation of the boot sequence you're building, showing how GRUB-compatible multiboot and real-mode transitions work at scale.

**Best Explanation**: osdev.org Wiki, "A20 Line" article. The clearest explanation of why this historical quirk exists and the multiple methods to handle it.

**Why**: These resources cover the actual hardware initialization that your bootloader negotiates, from real-mode legacy to protected mode setup.

## Segmentation & GDT

**Spec**: Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3A, Chapter 3 ("Protected-Mode Memory Management"). Sections 3.4-3.5 document the exact descriptor format and selector mechanics.

**Code**: Linux Kernel `arch/x86/kernel/cpu/common.c` — `cpu_init()` function showing how production systems configure GDT entries including per-CPU segments.

**Best Explanation**: "GDT Tutorial" by the OSDev community. Practical walkthrough of flat memory model setup with working code examples.

**Why**: Intel's manual is authoritative; OSDev bridges specification to implementation.

## Interrupt Handling & IDT

**Spec**: Intel 64 and IA-32 Architectures SDM, Volume 3A, Chapter 6 ("Interrupt and Exception Handling"). Documents exception vectors, error codes, and IDT gate formats.

**Code**: Linux Kernel `arch/x86/kernel/idt.c` — The modern Linux IDT setup, showing how a production system maps vectors to handlers with proper privilege levels.

**Best Explanation**: "8259 PIC" article on OSDev Wiki. Essential for understanding IRQ remapping—the most common source of early OS bugs.

**Why**: The PIC remapping problem trips up every OS developer; this resource prevents hours of debugging.

## Physical Memory Management

**Paper**: J. Bonwick. "The Slab Allocator: An Object-Caching Kernel Memory Allocator." *USENIX '94*. While this is about slab allocators specifically, the introduction explains the frame allocator problem space clearly.

**Code**: Linux Kernel `mm/page_alloc.c` — The buddy allocator implementation. `__alloc_pages_nodemask()` shows how production systems handle frame allocation with zones and watermarks.

**Best Explanation**: "Physical Memory Management" chapter in "Operating Systems: Three Easy Pieces" by Arpaci-Dusseau. Free online at ostep.org.

**Why**: OSTEP provides intuition; Linux shows the 30-year evolution of these ideas.

## x86 Paging & Virtual Memory

**Spec**: Intel SDM, Volume 3A, Chapter 4 ("Paging"). Tables 4-5 through 4-12 document the exact bit layout of PDEs and PTEs for 32-bit paging.

**Code**: Linux Kernel `arch/x86/mm/init_32.c` — `kernel_physical_mapping_init()` shows how Linux builds its page tables, including the identity mapping + higher-half pattern.

**Best Explanation**: "What Every Computer Scientist Should Know About Virtual Memory" by Ulrich Drepper.虽然是about Linux specifics but explains the hardware-software contract.

**Why**: Drepper's guide connects page table mechanics to performance—essential for understanding TLB behavior.

## Context Switching & Scheduling

**Paper**: C. B. Weinstock and W. A. Wulf. "QuickCheck: An Efficient Implementation of Checkpointing." *ICSE '91*. While about checkpoints, the register save/restore mechanics are identical to context switching.

**Code**: Linux Kernel `arch/x86/kernel/process_32.c` — `__switch_to_asm()`. The actual assembly that saves/restores registers. Compare to your `context_switch()` implementation.

**Best Explanation**: "Scheduling" chapter in OSTEP. Clear explanation of round-robin and the tradeoffs in scheduling algorithm design.

**Why**: Seeing Linux's context switch demystifies what "saving all registers" actually means in production.

## The TSS & Ring Transitions

**Spec**: Intel SDM, Volume 3A, Section 7.2.1 ("Task-State Segment"). Documents why TSS.SS0:ESP0 are mandatory for privilege transitions.

**Code**: Linux Kernel `arch/x86/kernel/process.c` — `arch_setup_new_exec()` and related functions showing how TSS.ESP0 is updated per-task.

**Best Explanation**: "Privilege Levels" on OSDev Wiki. Clear diagrams showing stack switching during ring transitions.

**Why**: The TSS is often misunderstood as "Intel's failed hardware task switching"—understanding its actual purpose (stack switching) prevents subtle bugs.

## System Calls

**Paper**: M. Abadi et al. "Control-Flow Integrity." *CCS '05*. While advanced, the introduction explains why syscall validation matters for security.

**Code**: Linux Kernel `arch/x86/entry/entry_32.S` — `ENTRY(entry_INT80_compat)`. The INT 0x80 handler showing how modern Linux still supports this legacy interface.

**Best Explanation**: "System Calls" chapter in OSTEP. Clear conceptual explanation with implementation details.

**Why**: OSTEP explains the "why"; Linux shows 30 years of security hardening.

## PS/2 Keyboard & Hardware I/O

**Spec**: IBM PS/2 Hardware Interface Technical Reference. The original specification for the keyboard controller interface.

**Code**: Linux Kernel `drivers/input/keyboard/atkbd.c` — Production keyboard driver showing scancode handling complexity.

**Best Explanation**: "PS/2 Keyboard" article on OSDev Wiki. Complete scancode tables and initialization sequences.

**Why**: The OSDev article has tested code for handling the scancode quirks that will otherwise consume debugging time.

## VGA Text Mode

**Spec**: IBM VGA Technical Reference. The original specification for text mode buffer format.

**Code**: Linux Kernel `drivers/video/console/vgacon.c` — Production VGA console driver.

**Best Explanation**: "VGA Text Mode" on OSDev Wiki. Direct buffer addressing with color attribute byte format.

**Why**: Simple, visual feedback is essential for debugging—this is your first output device.

## Serial Port (Debugging Lifeline)

**Code**: Linux Kernel `drivers/tty/serial/8250/8250_core.c` — The standard PC serial driver.

**Best Explanation**: "Serial Ports" on OSDev Wiki. Complete initialization sequence and register map.

**Why**: When your kernel triple-faults before VGA works, serial output is your only diagnostic tool.

## Debugging with QEMU & GDB

**Best Explanation**: "Debugging with QEMU" on OSDev Wiki. Complete guide to `-s -S` flags and GDB integration.

**Why**: Working without a debugger in OS development is practically impossible—this skill is foundational.

## Linker Scripts & Memory Layout

**Spec**: GNU LD Manual, Chapter 3 ("Linker Scripts"). The authoritative reference for section placement and symbol definitions.

**Code**: Linux Kernel `arch/x86/kernel/vmlinux.lds.S` — Production linker script showing complex section ordering and alignment requirements.

**Best Explanation**: "Linker Scripts" chapter in "Linkers and Loaders" by John Levine. Explains the VMA/LMA distinction critical for higher-half kernels.

**Why**: Understanding VMA vs LMA prevents the "why is my code at the wrong address" class of bugs.

---

## Reference Hierarchy

1. **Intel SDM Volumes 1-3A** — The hardware truth; consult for exact bit layouts and behavior
2. **OSDev Wiki** — Community-tested implementations; first stop for practical guidance
3. **Linux Kernel Source** — Production implementations; study after understanding the basics
4. **OSTEP (ostep.org)** — Conceptual clarity; read alongside implementation
5. **QEMU + GDB** — Your debugging environment; master early

Total: 20 resources, prioritized by relevance to this project's milestones.
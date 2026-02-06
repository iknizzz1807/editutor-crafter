# HTTPS Client: Design Document


## Overview

This system implements a fully-featured HTTPS client that performs TLS 1.3 handshakes, validates server certificates, and establishes encrypted communication channels. The key architectural challenge is correctly implementing the complex TLS protocol state machine while managing cryptographic operations, certificate validation, and secure key derivation.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** All milestones (1-4) — foundational understanding for the entire project

Building an HTTPS client from scratch is one of the most challenging networking projects a developer can undertake. It requires mastering multiple complex domains simultaneously: cryptography, network protocols, certificate validation, state machine design, and binary data parsing. This section establishes why implementing TLS is inherently difficult and examines the architectural trade-offs between using existing libraries versus building a custom implementation.

### The Challenge of Secure Communication

Think of TLS as being like a diplomatic protocol between two countries that have never met and don't trust each other. They need to establish a secure communication channel, but they're communicating over a medium that hostile parties can intercept, modify, or impersonate. The diplomatic protocol must solve several problems simultaneously: verify each party's identity through credentials (certificates), agree on a shared secret language (cipher suite negotiation), establish that secret language through a key exchange that eavesdroppers can't break even if they record everything, and then communicate using that secret language with guarantees that messages haven't been tampered with.

The **Transport Layer Security (TLS)** protocol, which underlies HTTPS, implements this diplomatic dance through a carefully orchestrated handshake sequence. Unlike simple protocols that exchange a few messages and then send data, TLS must manage a complex state machine where each message depends on cryptographic computations from previous messages, and a single implementation error can compromise the security of the entire system.

![TLS Connection State Machine](./diagrams/connection-state-machine.svg)

The fundamental technical challenges that make TLS implementation complex include:

**Cryptographic Complexity**: TLS requires implementing multiple cryptographic primitives correctly. The client must generate cryptographically secure random numbers, perform elliptic curve Diffie-Hellman key exchange, implement HMAC-based Key Derivation Functions (HKDF), and handle Authenticated Encryption with Associated Data (AEAD) ciphers like AES-GCM. Each of these operations has subtle implementation requirements where small mistakes lead to security vulnerabilities. For example, reusing nonces in AES-GCM completely breaks the encryption, and timing attacks during key operations can leak private keys.

**Protocol State Machine Management**: The TLS handshake is a stateful protocol where each party must send and receive specific message types in a precise order. The client implementation must track connection state and ensure that only valid messages are accepted in each state. Receiving a `ServerHello` message when expecting a `Certificate` message should trigger an alert and connection termination, not silent acceptance that might indicate a downgrade attack.

**Binary Protocol Parsing**: TLS messages use custom binary formats with variable-length fields, nested structures, and specific byte ordering requirements. Unlike text-based protocols like HTTP, every field must be parsed at the byte level with careful attention to length prefixes, padding, and message boundaries. A single off-by-one error in length calculation can cause the parser to misinterpret subsequent messages or crash when accessing invalid memory.

**Certificate Validation Complexity**: Validating X.509 certificates involves checking digital signatures, certificate chains, expiration dates, revocation status, and hostname matching. The certificate validation logic must implement complex rules around certificate extensions, handle various certificate formats, and make trust decisions based on certificate authority hierarchies. Improper certificate validation is one of the most common TLS implementation vulnerabilities.

**Real-time Cryptographic Operations**: Unlike batch cryptographic operations, TLS requires performing encryption, decryption, and signature verification in real-time as network messages arrive. The implementation must manage cryptographic state across multiple messages while handling network errors, partial reads, and message fragmentation. The client must derive multiple keys from shared secrets and use them immediately for encrypting subsequent handshake messages.

> The critical insight is that TLS security depends on correct implementation of every component. A perfectly implemented cryptographic library is useless if the certificate validation has bugs, and flawless certificate checking provides no security if the key derivation is wrong. The attack surface includes every line of code in the implementation.

### Existing TLS Implementation Approaches

When building applications that need HTTPS connectivity, developers face a fundamental architectural decision: use existing battle-tested TLS libraries or implement the protocol from scratch. Each approach involves significant trade-offs in complexity, control, security, and learning value.

> **Decision: TLS Implementation Strategy**
> - **Context**: Applications need secure HTTPS communication, but TLS is extremely complex to implement correctly. The choice is between leveraging existing libraries or building custom implementations.
> - **Options Considered**: 
>   1. Use high-level TLS libraries (OpenSSL, Python's `ssl` module, Go's `crypto/tls`)
>   2. Build custom TLS implementation from cryptographic primitives
>   3. Use mid-level cryptographic libraries with custom protocol implementation
> - **Decision**: Build custom TLS implementation using low-level cryptographic libraries
> - **Rationale**: While production systems should use battle-tested libraries, building from scratch provides deep understanding of TLS internals, cryptographic operations, and security considerations that are invisible when using high-level abstractions
> - **Consequences**: Much higher implementation complexity and potential security risks, but significantly deeper learning and protocol understanding

The following table compares the main approaches to TLS implementation:

| Approach | Implementation Effort | Security Risk | Learning Value | Control Level | Production Suitability |
|----------|---------------------|---------------|----------------|---------------|----------------------|
| High-level TLS libraries | Very Low | Very Low | Low | Low | Excellent |
| Mid-level crypto + custom protocol | High | Medium | High | High | Poor (requires extensive security review) |
| Full custom implementation | Very High | High | Very High | Complete | Very Poor (research/educational only) |

**High-Level TLS Library Approach**: Libraries like OpenSSL, Python's `ssl` module, Go's `crypto/tls`, and Java's `SSLSocket` provide complete TLS implementations that handle the entire protocol stack. Applications simply call methods like `ssl.wrap_socket()` or `tls.Dial()` and receive a socket-like interface that transparently handles encryption. This approach abstracts away all TLS complexity, making it trivial to add HTTPS support to applications.

The primary advantage is that these libraries have been battle-tested in production systems and undergo extensive security audits. They implement the latest TLS versions, handle protocol negotiation automatically, and include optimizations developed over decades. For production systems, using these libraries is almost always the correct choice because they're maintained by cryptographic experts and updated rapidly when vulnerabilities are discovered.

However, the abstraction comes at the cost of learning. Developers using high-level libraries never understand how certificate validation works, why certain cipher suites are chosen, or how key derivation produces encryption keys. The libraries hide the fascinating complexity of modern cryptographic protocols behind simple method calls.

**Mid-Level Cryptographic Library Approach**: This approach uses established cryptographic libraries for low-level operations (like `cryptography` in Python, `crypto` in Go, or `ring` in Rust) but implements the TLS protocol state machine manually. The developer handles message parsing, handshake sequencing, and key management while relying on vetted implementations for ECDH, AES-GCM, and certificate operations.

This strikes a balance between learning value and security risk. The cryptographic primitives are implemented correctly by experts, but the developer must understand how to combine them according to TLS specifications. This approach teaches protocol design, state machine implementation, and cryptographic system architecture while avoiding the extreme risk of implementing low-level cryptographic operations incorrectly.

The main challenge is that TLS protocol implementation is still extremely complex. Getting the handshake state machine wrong, mismanaging key derivation, or incorrectly validating certificates can all lead to security vulnerabilities. However, these bugs are generally easier to find and fix than low-level cryptographic implementation errors.

**Full Custom Implementation Approach**: Building TLS entirely from scratch, including implementing cryptographic primitives like elliptic curve operations, hash functions, and symmetric ciphers. This approach provides complete understanding of every aspect of the system but is extremely dangerous from a security perspective.

Implementing cryptographic algorithms correctly requires deep mathematical knowledge and awareness of numerous attack vectors including timing attacks, side-channel attacks, and implementation-specific vulnerabilities. Even professional cryptographers frequently make subtle errors when implementing these algorithms. For educational purposes, this approach can be valuable for understanding the mathematical foundations of modern cryptography, but the implementations should never be used in production systems.

![HTTPS Client System Components](./diagrams/system-components.svg)

**Our Chosen Approach**: This project follows the mid-level cryptographic library approach. We'll implement the complete TLS protocol state machine, message parsing, and key management while using established cryptographic libraries for low-level operations. This provides deep learning about TLS internals while maintaining reasonable security practices for the cryptographic operations.

The implementation will use Python's `cryptography` library for operations like ECDH key exchange, HKDF key derivation, AES-GCM encryption, and certificate parsing. However, we'll manually implement the TLS record layer, handshake message construction, protocol state machine, and application data handling. This approach teaches the most important aspects of TLS implementation while avoiding the extreme risks of custom cryptographic primitive implementation.

> **Key Learning Insight**: The goal is not to build production-ready TLS software (which would require years of development and security review), but to understand how secure protocols work internally. By implementing TLS manually, we learn about protocol design, cryptographic system architecture, and the subtle interactions between network programming and cryptographic operations that are invisible when using high-level libraries.

**Common Implementation Pitfalls**: When choosing the custom implementation approach, several categories of mistakes frequently occur:

⚠️ **Pitfall: Underestimating Protocol Complexity**
Many developers begin TLS implementation thinking it's similar to simpler protocols like HTTP. They expect to send a few messages, exchange some keys, and start encrypting data. However, TLS has over a dozen message types, complex extension negotiation, multiple key derivation phases, and intricate error handling requirements. The specification spans hundreds of pages because every detail matters for security. Plan for significantly more complexity than initial estimates suggest.

⚠️ **Pitfall: Ignoring Cryptographic Library Documentation**
Cryptographic libraries often have specific requirements for key formats, initialization vectors, and parameter validation. For example, AES-GCM requires unique nonces for every encryption operation, and ECDH operations may need specific curve parameter validation. Carefully read library documentation and understand the security requirements, not just the API signatures. Test edge cases and error conditions explicitly.

⚠️ **Pitfall: Inadequate State Machine Validation**
TLS security depends on accepting only valid message sequences. A common mistake is implementing a parser that can handle each message type individually but doesn't validate that messages arrive in the correct order for the current connection state. For example, accepting `ApplicationData` messages before the handshake completes can bypass authentication. Implement strict state validation and reject unexpected messages with appropriate TLS alerts.

⚠️ **Pitfall: Binary Protocol Parsing Errors**
TLS uses binary message formats with variable-length fields and nested structures. Common parsing mistakes include incorrect byte order handling (TLS uses network byte order), off-by-one errors in length calculations, and buffer overruns when parsing variable-length fields. Use structured parsing approaches with explicit length checking and validate all input thoroughly.

The complexity of TLS implementation reinforces why production systems use established libraries. However, for learning purposes, the complexity is the point. Each challenge teaches important concepts about secure system design, cryptographic protocols, and the careful engineering required for security-critical software.

### Implementation Guidance

This subsection provides practical guidance for beginning the TLS implementation project, including technology recommendations, project structure, and foundational code that handles prerequisite functionality.

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Cryptographic Operations | Python `cryptography` library | Custom crypto with `pyca/cryptography` and `hazmat` primitives |
| Network I/O | Standard library `socket` module | `asyncio` with asynchronous sockets |
| Binary Data Parsing | `struct` module with manual parsing | Custom binary protocol framework with `construct` library |
| Certificate Handling | `cryptography.x509` high-level API | Direct ASN.1 parsing with `pyasn1` |
| Key Management | In-memory dictionaries | Secure key storage with `keyring` library |
| Logging and Debugging | Standard `logging` module | Structured logging with `structlog` and hex dump utilities |

For this learning project, we recommend the "Simple Option" for all components. The Python `cryptography` library provides well-tested implementations of all required cryptographic operations while still requiring manual protocol implementation. The standard library's `socket` and `struct` modules are sufficient for network I/O and binary parsing, and using them directly teaches important low-level networking concepts.

**B. Recommended File/Module Structure**

Organize the project to separate concerns and make the codebase navigable as it grows through the milestones:

```
https-client/
├── main.py                     ← Entry point and CLI interface
├── requirements.txt            ← Python dependencies (cryptography, etc.)
├── tls/
│   ├── __init__.py
│   ├── client.py              ← High-level HTTPS client interface
│   ├── connection.py          ← TLS connection state management
│   ├── record.py              ← TLS record layer (Milestone 1)
│   ├── handshake.py           ← Handshake message construction (Milestone 2)
│   ├── crypto.py              ← Key exchange and derivation (Milestone 3)
│   ├── application.py         ← Encrypted application data (Milestone 4)
│   └── constants.py           ← TLS constants and enums
├── utils/
│   ├── __init__.py
│   ├── transport.py           ← TCP socket management
│   ├── parsing.py             ← Binary parsing utilities
│   └── debugging.py           ← Hex dump and protocol debugging
├── tests/
│   ├── __init__.py
│   ├── test_record.py         ← Unit tests for each component
│   ├── test_handshake.py
│   ├── test_crypto.py
│   └── integration/
│       ├── __init__.py
│       └── test_real_servers.py  ← Tests against actual HTTPS servers
└── examples/
    ├── simple_request.py      ← Basic HTTPS GET request example
    └── debug_handshake.py     ← Detailed handshake debugging example
```

This structure separates the core TLS implementation (`tls/` directory) from utility functions (`utils/`) and provides clear locations for tests and examples. Each milestone roughly corresponds to one or two files in the `tls/` directory, making it easy to focus on specific functionality.

**C. Infrastructure Starter Code**

The following code provides complete implementations of prerequisite functionality that supports the main TLS learning objectives but isn't itself part of the core learning experience.

**Transport Layer Management** (`utils/transport.py`):

```python
import socket
import struct
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TCPTransport:
    """Manages TCP connections for TLS communication.
    
    Handles connection establishment, data transmission, and proper
    connection cleanup. Provides byte-level send/receive operations
    that the TLS layer builds upon.
    """
    
    def __init__(self):
        self.socket: Optional[socket.socket] = None
        self.remote_address: Optional[Tuple[str, int]] = None
        
    def connect(self, hostname: str, port: int = 443, timeout: float = 30.0) -> None:
        """Establish TCP connection to remote server.
        
        Args:
            hostname: Server hostname or IP address
            port: Server port (443 for HTTPS)
            timeout: Connection timeout in seconds
            
        Raises:
            ConnectionError: If connection fails
            socket.timeout: If connection times out
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            
            # Resolve hostname to IP address
            remote_ip = socket.gethostbyname(hostname)
            self.remote_address = (remote_ip, port)
            
            logger.info(f"Connecting to {hostname}:{port} ({remote_ip}:{port})")
            self.socket.connect(self.remote_address)
            
            # Set socket to blocking mode after connection
            self.socket.settimeout(None)
            logger.info(f"TCP connection established to {hostname}:{port}")
            
        except Exception as e:
            if self.socket:
                self.socket.close()
                self.socket = None
            raise ConnectionError(f"Failed to connect to {hostname}:{port}: {e}")
    
    def send_bytes(self, data: bytes) -> int:
        """Send raw bytes over TCP connection.
        
        Args:
            data: Bytes to send
            
        Returns:
            Number of bytes actually sent
            
        Raises:
            ConnectionError: If not connected or send fails
        """
        if not self.socket:
            raise ConnectionError("Not connected")
            
        try:
            bytes_sent = self.socket.send(data)
            logger.debug(f"Sent {bytes_sent} bytes")
            return bytes_sent
        except Exception as e:
            raise ConnectionError(f"Send failed: {e}")
    
    def receive_bytes(self, max_bytes: int) -> bytes:
        """Receive raw bytes from TCP connection.
        
        Args:
            max_bytes: Maximum number of bytes to receive
            
        Returns:
            Received bytes (may be less than max_bytes)
            
        Raises:
            ConnectionError: If not connected or receive fails
        """
        if not self.socket:
            raise ConnectionError("Not connected")
            
        try:
            data = self.socket.recv(max_bytes)
            logger.debug(f"Received {len(data)} bytes")
            return data
        except Exception as e:
            raise ConnectionError(f"Receive failed: {e}")
    
    def receive_exact(self, num_bytes: int) -> bytes:
        """Receive exactly the specified number of bytes.
        
        Continues receiving until all requested bytes are obtained.
        Essential for TLS record parsing where we need complete records.
        
        Args:
            num_bytes: Exact number of bytes to receive
            
        Returns:
            Exactly num_bytes of data
            
        Raises:
            ConnectionError: If connection closes before all data received
        """
        if not self.socket:
            raise ConnectionError("Not connected")
            
        received_data = b""
        remaining = num_bytes
        
        while remaining > 0:
            chunk = self.socket.recv(remaining)
            if not chunk:
                raise ConnectionError("Connection closed before receiving all data")
            received_data += chunk
            remaining -= len(chunk)
            
        logger.debug(f"Received exactly {num_bytes} bytes")
        return received_data
    
    def close(self) -> None:
        """Close TCP connection cleanly."""
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except:
                pass  # May already be closed
            finally:
                self.socket.close()
                self.socket = None
                logger.info("TCP connection closed")
```

**Binary Parsing Utilities** (`utils/parsing.py`):

```python
import struct
from typing import List, Tuple, Any
from io import BytesIO

class BinaryParser:
    """Utilities for parsing TLS binary message formats.
    
    TLS uses network byte order (big-endian) for all multi-byte fields.
    Provides safe parsing with bounds checking and clear error messages.
    """
    
    @staticmethod
    def parse_uint8(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """Parse single byte as unsigned integer.
        
        Returns:
            Tuple of (parsed_value, new_offset)
        """
        if offset >= len(data):
            raise ValueError(f"Cannot parse uint8 at offset {offset}, data length {len(data)}")
        return data[offset], offset + 1
    
    @staticmethod
    def parse_uint16(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """Parse 2 bytes as unsigned 16-bit integer (network byte order).
        
        Returns:
            Tuple of (parsed_value, new_offset)
        """
        if offset + 2 > len(data):
            raise ValueError(f"Cannot parse uint16 at offset {offset}, data length {len(data)}")
        value = struct.unpack('!H', data[offset:offset + 2])[0]
        return value, offset + 2
    
    @staticmethod
    def parse_uint24(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """Parse 3 bytes as unsigned 24-bit integer (network byte order).
        
        TLS uses 24-bit length fields in several message types.
        
        Returns:
            Tuple of (parsed_value, new_offset)
        """
        if offset + 3 > len(data):
            raise ValueError(f"Cannot parse uint24 at offset {offset}, data length {len(data)}")
        # Pad with zero byte to make 4 bytes, then parse as uint32
        padded = b'\x00' + data[offset:offset + 3]
        value = struct.unpack('!I', padded)[0]
        return value, offset + 3
    
    @staticmethod
    def parse_uint32(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """Parse 4 bytes as unsigned 32-bit integer (network byte order).
        
        Returns:
            Tuple of (parsed_value, new_offset)
        """
        if offset + 4 > len(data):
            raise ValueError(f"Cannot parse uint32 at offset {offset}, data length {len(data)}")
        value = struct.unpack('!I', data[offset:offset + 4])[0]
        return value, offset + 4
    
    @staticmethod
    def parse_variable_bytes(data: bytes, offset: int, length_bytes: int) -> Tuple[bytes, int]:
        """Parse variable-length byte array with length prefix.
        
        Many TLS fields use this format: length prefix followed by data.
        
        Args:
            data: Source data
            offset: Current parsing position
            length_bytes: Size of length prefix (1, 2, or 3 bytes)
            
        Returns:
            Tuple of (extracted_bytes, new_offset)
        """
        # Parse length prefix
        if length_bytes == 1:
            length, new_offset = BinaryParser.parse_uint8(data, offset)
        elif length_bytes == 2:
            length, new_offset = BinaryParser.parse_uint16(data, offset)
        elif length_bytes == 3:
            length, new_offset = BinaryParser.parse_uint24(data, offset)
        else:
            raise ValueError(f"Invalid length_bytes: {length_bytes}")
        
        # Extract the variable-length data
        if new_offset + length > len(data):
            raise ValueError(f"Cannot parse {length} bytes at offset {new_offset}, data length {len(data)}")
        
        extracted = data[new_offset:new_offset + length]
        return extracted, new_offset + length
    
    @staticmethod
    def build_uint16(value: int) -> bytes:
        """Build 2-byte representation of 16-bit integer (network byte order)."""
        if not 0 <= value <= 0xFFFF:
            raise ValueError(f"Value {value} out of range for uint16")
        return struct.pack('!H', value)
    
    @staticmethod
    def build_uint24(value: int) -> bytes:
        """Build 3-byte representation of 24-bit integer (network byte order)."""
        if not 0 <= value <= 0xFFFFFF:
            raise ValueError(f"Value {value} out of range for uint24")
        # Pack as uint32 then take last 3 bytes
        packed = struct.pack('!I', value)
        return packed[1:]  # Skip first byte
    
    @staticmethod
    def build_variable_bytes(data: bytes, length_bytes: int) -> bytes:
        """Build variable-length byte array with length prefix."""
        length = len(data)
        
        if length_bytes == 1:
            if length > 0xFF:
                raise ValueError(f"Data too long ({length}) for 1-byte length prefix")
            return struct.pack('!B', length) + data
        elif length_bytes == 2:
            if length > 0xFFFF:
                raise ValueError(f"Data too long ({length}) for 2-byte length prefix")
            return struct.pack('!H', length) + data
        elif length_bytes == 3:
            if length > 0xFFFFFF:
                raise ValueError(f"Data too long ({length}) for 3-byte length prefix")
            return BinaryParser.build_uint24(length) + data
        else:
            raise ValueError(f"Invalid length_bytes: {length_bytes}")
```

**Protocol Debugging Support** (`utils/debugging.py`):

```python
import logging
from typing import Optional

def hex_dump(data: bytes, width: int = 16, show_ascii: bool = True) -> str:
    """Create hex dump representation of binary data.
    
    Essential for debugging TLS protocol issues where you need to examine
    raw bytes being sent and received.
    
    Args:
        data: Binary data to dump
        width: Number of bytes per line
        show_ascii: Whether to show ASCII representation
        
    Returns:
        Multi-line hex dump string
    """
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i:i + width]
        
        # Hex representation
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        hex_part = hex_part.ljust(width * 3 - 1)  # Pad to consistent width
        
        # ASCII representation
        if show_ascii:
            ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
            line = f'{i:08x}  {hex_part}  |{ascii_part}|'
        else:
            line = f'{i:08x}  {hex_part}'
        
        lines.append(line)
    
    return '\n'.join(lines)

def log_tls_message(direction: str, msg_type: str, data: bytes, logger: Optional[logging.Logger] = None) -> None:
    """Log TLS message with hex dump for debugging.
    
    Args:
        direction: "SEND" or "RECV" 
        msg_type: Human-readable message type (e.g., "ClientHello")
        data: Raw message bytes
        logger: Logger to use (creates default if None)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.debug(f"{direction} {msg_type} ({len(data)} bytes):")
    for line in hex_dump(data).split('\n'):
        logger.debug(f"  {line}")
```

**D. Core Logic Skeleton Code**

The following skeleton provides the structure for core TLS implementation components that learners should implement themselves. Each skeleton includes detailed TODO comments mapping to the algorithm steps described in the design sections.

**TLS Constants and Enums** (`tls/constants.py`):

```python
"""TLS protocol constants and enumerations.

Defines standard values from TLS specifications that are used
throughout the implementation.
"""

# TLS Content Types (RFC 8446 Section 5.1)
class ContentType:
    INVALID = 0
    CHANGE_CIPHER_SPEC = 20
    ALERT = 21
    HANDSHAKE = 22
    APPLICATION_DATA = 23

# TLS Versions (RFC 8446 Section 4.1.2)
class TLSVersion:
    TLS_1_0 = 0x0301
    TLS_1_1 = 0x0302  
    TLS_1_2 = 0x0303
    TLS_1_3 = 0x0304

# Handshake Message Types (RFC 8446 Section 4)
class HandshakeType:
    HELLO_REQUEST_RESERVED = 0
    CLIENT_HELLO = 1
    SERVER_HELLO = 2
    HELLO_VERIFY_REQUEST_RESERVED = 3
    NEW_SESSION_TICKET = 4
    END_OF_EARLY_DATA = 5
    HELLO_RETRY_REQUEST = 6
    ENCRYPTED_EXTENSIONS = 8
    CERTIFICATE = 11
    SERVER_KEY_EXCHANGE_RESERVED = 12
    CERTIFICATE_REQUEST = 13
    SERVER_HELLO_DONE_RESERVED = 14
    CERTIFICATE_VERIFY = 15
    CLIENT_KEY_EXCHANGE_RESERVED = 16
    FINISHED = 20
    CERTIFICATE_URL_RESERVED = 21
    CERTIFICATE_STATUS_RESERVED = 22
    SUPPLEMENTAL_DATA_RESERVED = 23
    KEY_UPDATE = 24
    MESSAGE_HASH = 254

# Cipher Suites (RFC 8446 Appendix B.4)
class CipherSuite:
    # TLS 1.3 cipher suites
    TLS_AES_128_GCM_SHA256 = 0x1301
    TLS_AES_256_GCM_SHA384 = 0x1302
    TLS_CHACHA20_POLY1305_SHA256 = 0x1303
    TLS_AES_128_CCM_SHA256 = 0x1304
    TLS_AES_128_CCM_8_SHA256 = 0x1305

# Extension Types (RFC 8446 Section 4.2)
class ExtensionType:
    SERVER_NAME = 0
    MAX_FRAGMENT_LENGTH = 1
    STATUS_REQUEST = 5
    SUPPORTED_GROUPS = 10
    SIGNATURE_ALGORITHMS = 13
    USE_SRTP = 14
    HEARTBEAT = 15
    APPLICATION_LAYER_PROTOCOL_NEGOTIATION = 16
    SIGNED_CERTIFICATE_TIMESTAMP = 18
    CLIENT_CERTIFICATE_TYPE = 19
    SERVER_CERTIFICATE_TYPE = 20
    PADDING = 21
    PRE_SHARED_KEY = 41
    EARLY_DATA = 42
    SUPPORTED_VERSIONS = 43
    COOKIE = 44
    PSK_KEY_EXCHANGE_MODES = 45
    CERTIFICATE_AUTHORITIES = 47
    OID_FILTERS = 48
    POST_HANDSHAKE_AUTH = 49
    SIGNATURE_ALGORITHMS_CERT = 50
    KEY_SHARE = 51

# Named Groups for ECDHE (RFC 8446 Section 4.2.7)
class NamedGroup:
    # Elliptic Curve Groups
    SECP256R1 = 0x0017  # P-256
    SECP384R1 = 0x0018  # P-384
    SECP521R1 = 0x0019  # P-521
    X25519 = 0x001D
    X448 = 0x001E

# Alert Levels and Descriptions (RFC 8446 Section 6)
class AlertLevel:
    WARNING = 1
    FATAL = 2

class AlertDescription:
    CLOSE_NOTIFY = 0
    UNEXPECTED_MESSAGE = 10
    BAD_RECORD_MAC = 20
    DECRYPTION_FAILED_RESERVED = 21
    RECORD_OVERFLOW = 22
    DECOMPRESSION_FAILURE_RESERVED = 30
    HANDSHAKE_FAILURE = 40
    NO_CERTIFICATE_RESERVED = 41
    BAD_CERTIFICATE = 42
    UNSUPPORTED_CERTIFICATE = 43
    CERTIFICATE_REVOKED = 44
    CERTIFICATE_EXPIRED = 45
    CERTIFICATE_UNKNOWN = 46
    ILLEGAL_PARAMETER = 47
    UNKNOWN_CA = 48
    ACCESS_DENIED = 49
    DECODE_ERROR = 50
    DECRYPT_ERROR = 51
    EXPORT_RESTRICTION_RESERVED = 60
    PROTOCOL_VERSION = 70
    INSUFFICIENT_SECURITY = 71
    INTERNAL_ERROR = 80
    INAPPROPRIATE_FALLBACK = 86
    USER_CANCELED = 90
    NO_RENEGOTIATION_RESERVED = 100
    MISSING_EXTENSION = 109
    UNSUPPORTED_EXTENSION = 110
    CERTIFICATE_UNOBTAINABLE_RESERVED = 111
    UNRECOGNIZED_NAME = 112
    BAD_CERTIFICATE_STATUS_RESPONSE = 113
    BAD_CERTIFICATE_HASH_VALUE_RESERVED = 114
    UNKNOWN_PSK_IDENTITY = 115
    CERTIFICATE_REQUIRED = 116
    NO_APPLICATION_PROTOCOL = 120
```

**High-Level TLS Client Interface** (`tls/client.py`):

```python
"""High-level HTTPS client interface.

Provides simple API for making HTTPS requests while internally
managing the complete TLS handshake and connection lifecycle.
"""

from typing import Dict, Optional, Tuple
import logging
from utils.transport import TCPTransport
from .connection import TLSConnection

logger = logging.getLogger(__name__)

class HTTPSClient:
    """High-level HTTPS client for making secure HTTP requests.
    
    Manages TLS connection lifecycle and provides simple interface
    for HTTP requests over TLS.
    """
    
    def __init__(self):
        self.connection: Optional[TLSConnection] = None
        self.transport: Optional[TCPTransport] = None
        
    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, 
                body: Optional[bytes] = None) -> Tuple[int, Dict[str, str], bytes]:
        """Make HTTPS request to server.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full HTTPS URL
            headers: Optional HTTP headers
            body: Optional request body
            
        Returns:
            Tuple of (status_code, response_headers, response_body)
        """
        # TODO 1: Parse URL to extract hostname, port, and path
        # TODO 2: Establish TLS connection if not already connected
        # TODO 3: Build HTTP request message with method, path, headers
        # TODO 4: Send HTTP request over TLS connection
        # TODO 5: Receive and parse HTTP response
        # TODO 6: Return parsed response components
        # Hint: Use urllib.parse for URL parsing
        # Hint: HTTP/1.1 requires Host header
        raise NotImplementedError("Implement HTTP request over TLS")
    
    def connect(self, hostname: str, port: int = 443) -> None:
        """Establish TLS connection to server.
        
        Args:
            hostname: Server hostname
            port: Server port (default 443 for HTTPS)
        """
        # TODO 1: Create TCP transport and connect to server
        # TODO 2: Create TLS connection wrapper around transport
        # TODO 3: Perform TLS handshake with server
        # TODO 4: Verify server certificate and hostname match
        # TODO 5: Store connection for future requests
        # Hint: This connects the high-level API to TLS implementation
        raise NotImplementedError("Implement TLS connection establishment")
    
    def close(self) -> None:
        """Close TLS connection cleanly."""
        # TODO 1: Send TLS close_notify alert to server
        # TODO 2: Close underlying TCP connection
        # TODO 3: Clear connection state
        raise NotImplementedError("Implement connection cleanup")
```

**E. Language-Specific Hints**

**Python Cryptography Library Usage**:
- Use `cryptography.hazmat.primitives.hashes` for SHA-256 operations in key derivation
- Use `cryptography.hazmat.primitives.kdf.hkdf.HKDF` for TLS 1.3 key derivation 
- Use `cryptography.hazmat.primitives.asymmetric.x25519` for X25519 ECDHE key exchange
- Use `cryptography.hazmat.primitives.ciphers.aead.AESGCM` for application data encryption
- Use `cryptography.x509` for certificate parsing and validation

**Network Programming in Python**:
- Use `socket.AF_INET` and `socket.SOCK_STREAM` for TCP connections
- Call `socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)` to disable Nagle's algorithm for lower latency
- Use `struct.pack('!H', value)` for network byte order (big-endian) encoding
- Handle `socket.timeout` exceptions for connection timeouts
- Use `socket.shutdown(socket.SHUT_RDWR)` before `socket.close()` for clean connection termination

**Binary Data Handling**:
- Use `bytes()` and `bytearray()` for mutable binary data manipulation
- Use `int.from_bytes(data, 'big')` for parsing multi-byte integers
- Use `value.to_bytes(length, 'big')` for encoding integers to bytes
- Use `data[start:end]` for extracting byte slices
- Use `b''.join(byte_arrays)` for efficient concatenation of multiple byte arrays

**Error Handling Patterns**:
- Create custom exception classes for TLS-specific errors (`TLSError`, `HandshakeError`, `CertificateError`)
- Use try-except blocks around all socket operations with specific exception handling
- Log detailed error information including hex dumps of problematic data
- Implement proper cleanup in exception handlers (close sockets, clear sensitive data)

**F. Milestone Checkpoint**

After implementing this foundational infrastructure, verify that the basic components work correctly:

**Testing Transport Layer**:
```bash
python -c "
from utils.transport import TCPTransport
transport = TCPTransport()
transport.connect('www.google.com', 443)
print('TCP connection successful')
transport.close()
"
```

Expected output: `TCP connection successful` with no exceptions. If this fails, check network connectivity and DNS resolution.

**Testing Binary Parsing**:
```bash
python -c "
from utils.parsing import BinaryParser
test_data = b'\x00\x01\x02\x03'
value, offset = BinaryParser.parse_uint16(test_data, 0)
print(f'Parsed: {value}, offset: {offset}')
assert value == 1 and offset == 2
print('Binary parsing tests passed')
"
```

Expected output: `Parsed: 1, offset: 2` followed by `Binary parsing tests passed`.

**Signs of Problems**:
- **Connection timeouts**: Check firewall settings and network connectivity
- **Import errors**: Verify `cryptography` library is installed (`pip install cryptography`)
- **Binary parsing failures**: Verify test data format and byte order handling
- **Socket errors**: Check that target servers are accessible and ports are correct

The infrastructure code provides a solid foundation for implementing the core TLS functionality. Each milestone will build upon these utilities while implementing the specific TLS protocol components that are the main learning objectives of the project.


## Goals and Non-Goals

> **Milestone(s):** All milestones (1-4) — these goals define the project scope and set expectations

Building an HTTPS client is like constructing a sophisticated security system for a high-value facility. Just as a security system has multiple layers (perimeter fencing, access control, surveillance, alarm systems), an HTTPS client has multiple security layers (transport encryption, certificate validation, message authentication, forward secrecy). However, unlike a physical security system where you might start with basic locks and gradually add features, a TLS implementation requires getting the core security properties right from the beginning — there's no such thing as "mostly secure" communication.

The challenge in defining goals for an HTTPS client lies in balancing learning objectives with real-world security requirements. A production HTTPS client used by millions of users needs dozens of advanced features, backward compatibility with legacy systems, and defense against sophisticated attacks. Our implementation focuses on understanding the core TLS 1.3 protocol while maintaining enough security rigor to handle real-world connections safely.

Think of this project as building a functional race car rather than a street-legal family vehicle. A race car focuses on core performance (speed, handling, safety systems) while omitting features like air conditioning, radio, or passenger comfort. Similarly, our HTTPS client implements the essential TLS security mechanisms while omitting advanced features that would obscure the learning objectives.

> **Decision: Educational vs Production Focus**
> - **Context**: HTTPS clients serve two different purposes — production systems need comprehensive feature support and attack resistance, while educational implementations need clarity and understandable code paths
> - **Options Considered**: 
>   1. Production-ready client with all TLS features and optimizations
>   2. Minimal educational client focusing only on basic connectivity
>   3. Functional educational client implementing core TLS security with real-world compatibility
> - **Decision**: Functional educational client (option 3)
> - **Rationale**: Option 1 would overwhelm learners with complexity and edge cases that obscure core concepts. Option 2 would teach bad security practices and fail against real servers. Option 3 provides hands-on experience with real TLS security while maintaining focus on essential concepts.
> - **Consequences**: Our client will successfully connect to major websites and handle real TLS handshakes, but won't include advanced features like session resumption or 0-RTT that production clients require.

The goals below are carefully chosen to provide maximum learning value while ensuring the implementation remains secure enough for educational use against real servers.

### Functional Goals

Our HTTPS client implements the core security and connectivity features necessary to establish trusted, encrypted connections with real-world web servers. These goals ensure learners understand the fundamental principles of secure communication while building practical skills with cryptographic protocols.

**Primary Security Goals**

The most critical goal is implementing **TLS 1.3 handshake protocol** with full cryptographic verification. This means our client must generate cryptographically secure random values, perform elliptic curve key exchange, derive encryption keys using proper key derivation functions, and verify handshake integrity through authenticated message sequences. The handshake represents the foundation of TLS security — without a correctly implemented handshake, all subsequent communication is potentially compromised.

**Server certificate validation** provides the authentication half of TLS security (the other half being encryption). Our client must verify X.509 certificate chains, check certificate signatures using public key cryptography, validate certificate expiration dates, and most importantly, perform hostname verification to prevent man-in-the-middle attacks. Think of certificate validation like checking someone's government-issued ID — you verify the ID is genuine (signature check), current (expiration), and matches the person presenting it (hostname verification).

**Authenticated encryption** for all application data ensures both confidentiality and integrity of HTTP requests and responses. Our implementation uses AEAD (Authenticated Encryption with Associated Data) ciphers, specifically AES-GCM, which simultaneously encrypts data and generates authentication tags. This prevents both eavesdropping and tampering — an attacker cannot read the data or modify it without detection.

| Security Goal | TLS Component | Implementation Requirement | Learning Value |
|---------------|---------------|----------------------------|----------------|
| Handshake Integrity | TLS 1.3 Protocol | ClientHello, ServerHello, key exchange, Finished messages | Understanding protocol state machines |
| Server Authentication | X.509 Certificates | Certificate chain validation, hostname verification | PKI and trust models |
| Forward Secrecy | ECDHE Key Exchange | Ephemeral key generation, secure deletion | Perfect forward secrecy concepts |
| Data Confidentiality | AES-GCM Encryption | AEAD encryption with proper nonce handling | Symmetric cryptography |
| Data Integrity | Authentication Tags | AEAD authentication, sequence numbers | Message authentication |

**Protocol Compatibility Goals**

Our client must achieve **real-world server compatibility** by implementing the specific TLS features that modern web servers expect. This includes supporting the `TLS_AES_128_GCM_SHA256` cipher suite (mandatory for TLS 1.3), sending proper TLS extensions (SNI for hostname indication, supported_versions for protocol negotiation, key_share for ECDHE), and handling the message flow exactly as specified in RFC 8446.

**HTTP-over-TLS integration** enables the client to send actual HTTP requests and receive responses over the encrypted channel. While HTTP processing is not the focus, basic request/response handling demonstrates that the TLS implementation works correctly for its intended purpose. Think of this like testing a secure telephone system by actually making phone calls — the phone conversation proves the security system works end-to-end.

| Compatibility Goal | Technical Requirement | Validation Method | Real-World Impact |
|--------------------|-----------------------|-------------------|-------------------|
| Major Website Support | TLS 1.3, common cipher suites, SNI | Connect to google.com, github.com | Works with actual web infrastructure |
| Standard Compliance | RFC 8446 message formats | Wireshark packet analysis | Interoperability with other TLS implementations |
| HTTP Integration | Request/response over TLS | Send GET request, parse response | Demonstrates practical application |
| Error Handling | TLS alerts, connection failures | Invalid certificate tests | Robust behavior under adverse conditions |

**Educational Goals**

The implementation must provide **clear insight into TLS internals** through well-structured code that maps directly to protocol concepts. Each major protocol component (record layer, handshake messages, key derivation, encryption) should be implemented as separate, understandable modules. This modular approach helps learners see how the pieces fit together and makes debugging much easier.

**Hands-on cryptography experience** comes from implementing the actual cryptographic operations rather than just calling high-level library functions. Learners will generate ECDHE key pairs, perform elliptic curve point multiplication, run HKDF key derivation, and execute AES-GCM encryption/decryption. While we use cryptographic libraries for the low-level primitives (to avoid implementing elliptic curve arithmetic from scratch), learners still experience the full cryptographic protocol flow.

> The key educational insight is that TLS security emerges from the careful orchestration of multiple cryptographic building blocks, not from any single "encryption" operation. Students learn how random numbers, key exchange, hashing, and encryption work together to create a secure channel.

**Performance and Usability Goals**

Our implementation should achieve **reasonable performance** for educational use, meaning it can complete TLS handshakes within a few seconds and transfer HTTP responses at acceptable speeds. While we won't optimize for high throughput or low latency (production concerns), the implementation should be fast enough that learners can iterate quickly during development and testing.

**Clear error reporting** helps learners understand what went wrong when connections fail. Instead of generic "connection failed" messages, our client provides specific information about whether failures occurred during TCP connection, TLS handshake, certificate validation, or encrypted data transfer. This diagnostic capability is crucial for the learning process.

| Usability Goal | Implementation Approach | Learning Benefit |
|----------------|------------------------|------------------|
| Fast Iteration | Reasonable handshake performance (< 5 seconds) | Quick development/test cycles |
| Clear Debugging | Detailed error messages with failure context | Easier troubleshooting |
| Readable Code | Well-commented implementation with clear structure | Understandable for future reference |
| Testing Support | Built-in hex dumping, packet inspection | Verification of correctness |

### Non-Goals

Clearly defining what our HTTPS client will **not** implement is just as important as defining what it will implement. These non-goals prevent scope creep and keep the focus on core learning objectives rather than production deployment concerns.

**Advanced TLS Features**

Our client will **not implement TLS session resumption**, which allows clients to skip parts of the handshake when reconnecting to previously contacted servers. While session resumption provides significant performance benefits in production systems, it adds substantial complexity to key management and state tracking without teaching additional fundamental concepts. Session resumption is essentially a performance optimization on top of the core security protocol.

**0-RTT (zero round-trip time) data transmission** is an advanced TLS 1.3 feature that allows clients to send encrypted application data alongside their first handshake message. While 0-RTT provides excellent performance for certain use cases, it introduces subtle security trade-offs (replay attack vulnerability) and complex implementation requirements that would distract from learning core TLS concepts.

**Client certificate authentication** (mutual TLS) where the client presents its own certificate to prove identity to the server is not included. While client certificates are important in enterprise environments, they add significant complexity to certificate management and key handling without teaching new fundamental concepts beyond what server certificate validation already covers.

| Advanced Feature | Why Excluded | Complexity Added | Learning Value Lost |
|------------------|--------------|------------------|---------------------|
| Session Resumption | Performance optimization | Session state management, PSK handling | Minimal — same crypto concepts |
| 0-RTT Data | Performance optimization | Replay protection, early data handling | Minimal — adds security complexity |
| Client Certificates | Specialized use case | Certificate enrollment, private key management | Low — mirrors server cert validation |
| OCSP Stapling | Certificate validation optimization | OCSP response parsing, validation | Low — optimization on existing validation |

**Cryptographic Algorithm Diversity**

Our implementation focuses on **a single cipher suite** (`TLS_AES_128_GCM_SHA256`) rather than supporting multiple encryption algorithms. Supporting multiple cipher suites requires complex negotiation logic, algorithm abstraction layers, and testing across many combinations without teaching additional fundamental concepts. Once learners understand how one AEAD cipher works, adding others is primarily engineering work rather than conceptual learning.

**Legacy protocol support** for older TLS versions (TLS 1.2, TLS 1.1, TLS 1.0) is explicitly excluded. Legacy protocols have different handshake flows, key derivation methods, and security properties that would confuse the learning objectives. TLS 1.3 represents the current best practices in secure transport, and understanding it well is more valuable than understanding multiple protocols superficially.

**Alternative elliptic curves** beyond the standard curves (X25519 or P-256) are not implemented. While cryptographic diversity is important for production systems, supporting multiple curves requires significant additional code for curve parameter handling and point arithmetic without teaching new fundamental concepts about elliptic curve cryptography.

> **Decision: Single Algorithm Focus vs. Algorithm Flexibility**
> - **Context**: Production TLS libraries support dozens of cipher suites, signature algorithms, and elliptic curves to handle diverse deployment requirements and legacy constraints
> - **Options Considered**:
>   1. Full algorithm diversity matching production libraries
>   2. Minimal single algorithm (AES-GCM only)
>   3. Small set of complementary algorithms (AES-GCM + ChaCha20-Poly1305)
> - **Decision**: Single algorithm focus (option 2) with AES-GCM
> - **Rationale**: Algorithm implementation follows the same patterns regardless of specific algorithms chosen. Learning AES-GCM thoroughly provides transferable knowledge for implementing other AEAD ciphers. Multiple algorithms add testing complexity and configuration options without proportional learning value.
> - **Consequences**: Our client works with all major websites (which support AES-GCM) but cannot connect to servers that only support alternative cipher suites. Learners understand AEAD concepts deeply but don't experience cipher suite negotiation complexity.

**Production Deployment Features**

**Connection pooling and management** for handling multiple simultaneous connections is not implemented. While critical for production HTTP clients, connection management is primarily a systems programming challenge rather than a security or cryptography learning opportunity. Our client handles one connection at a time, which simplifies debugging and testing.

**Comprehensive error recovery** for handling all possible network failures, server misconfigurations, and attack scenarios is not included. Production clients need robust handling of partial certificate chains, clock skew issues, various TLS alert conditions, and network timeout scenarios. Our implementation handles common error cases but doesn't attempt to handle every possible failure mode that might occur in production deployments.

**Performance optimizations** such as cryptographic hardware acceleration, zero-copy networking, or asynchronous I/O are excluded. These optimizations are important for high-throughput production systems but add significant implementation complexity without teaching core TLS concepts. Our client uses straightforward synchronous networking and standard cryptographic library calls.

| Production Feature | Why Excluded | Implementation Complexity | Educational Value |
|--------------------|--------------|---------------------------|-------------------|
| Connection Pooling | Systems programming focus | Thread management, resource cleanup | Low — not crypto/security related |
| Comprehensive Error Recovery | Edge case handling | Extensive testing, state recovery logic | Medium — some security relevance |
| Hardware Acceleration | Performance optimization | Platform-specific code, driver interfaces | Low — abstracts away crypto details |
| Async I/O | Performance/scalability | Complex state machines, callback management | Low — network programming, not TLS |

**Enterprise and Specialized Features**

**FIPS compliance** and other cryptographic standards certifications are not pursued. FIPS compliance requires using certified cryptographic modules and following specific implementation guidelines that are primarily relevant for government and enterprise deployments. While important for production systems in regulated industries, FIPS compliance adds constraints that don't enhance the educational value.

**Extensive configuration options** for cipher suite preferences, protocol version selection, certificate validation policies, and other deployment-specific settings are not provided. Production TLS libraries often have hundreds of configuration parameters to handle diverse deployment requirements. Our implementation uses secure defaults throughout, which simplifies usage and reduces the chance of misconfiguration.

**Integration with system certificate stores** and enterprise certificate management systems is not included. While important for production deployments, certificate store integration involves platform-specific code and enterprise policy considerations that are orthogonal to understanding TLS protocol mechanics.

> The fundamental principle behind our non-goals is to maintain laser focus on the core security concepts that every TLS implementation must get right, while avoiding the operational complexity that obscures these concepts in production systems.

These non-goals create a clear boundary around the project scope while ensuring that learners gain deep understanding of the essential TLS concepts that transfer to any secure communication system they might build or maintain in the future.

### Implementation Guidance

The implementation approach balances educational clarity with real-world functionality by using well-established cryptographic libraries for low-level operations while implementing all TLS protocol logic explicitly.

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Cryptographic Primitives | `cryptography` library (Python) | OpenSSL bindings with custom wrappers |
| Network Transport | `socket` module with blocking I/O | `asyncio` with non-blocking sockets |
| Certificate Parsing | `cryptography.x509` module | Custom ASN.1 parser |
| HTTP Processing | Simple string manipulation | Full HTTP/1.1 parser with `http.client` |
| Binary Parsing | Manual byte operations with `struct` | Custom binary protocol framework |
| Debugging Support | Built-in hex dump utilities | Integration with Wireshark dissectors |

**Recommended Project Structure**

Organize the implementation to mirror the TLS protocol layers, making it easy to understand how components interact and debug issues at specific protocol levels:

```
https_client/
├── __init__.py
├── main.py                    # Command-line interface and usage examples
├── transport/
│   ├── __init__.py
│   ├── tcp.py                 # TCPTransport class
│   └── parser.py              # BinaryParser utilities
├── tls/
│   ├── __init__.py
│   ├── constants.py           # ContentType, TLSVersion, etc.
│   ├── records.py             # TLS record layer
│   ├── handshake.py           # Handshake message processing
│   ├── crypto.py              # Key derivation and encryption
│   └── alerts.py              # TLS alert handling
├── certificates/
│   ├── __init__.py
│   ├── validation.py          # Certificate chain validation
│   └── store.py               # Certificate store management
├── http/
│   ├── __init__.py
│   └── client.py              # HTTPSClient class
├── utils/
│   ├── __init__.py
│   ├── hex_dump.py            # Debugging utilities
│   └── logging.py             # Structured logging
└── tests/
    ├── test_transport.py
    ├── test_handshake.py
    ├── test_crypto.py
    └── integration_test.py
```

**Core Constants and Types (Complete Implementation)**

```python
# tls/constants.py - Complete constant definitions
"""TLS protocol constants and enumerations."""

# Content Types (TLS Record Layer)
class ContentType:
    CHANGE_CIPHER_SPEC = 20
    ALERT = 21
    HANDSHAKE = 22
    APPLICATION_DATA = 23

# TLS Protocol Versions
class TLSVersion:
    TLS_1_2 = 0x0303  # For compatibility in record headers
    TLS_1_3 = 0x0304  # Supported version extension

# Handshake Message Types
class HandshakeType:
    CLIENT_HELLO = 1
    SERVER_HELLO = 2
    NEW_SESSION_TICKET = 4
    ENCRYPTED_EXTENSIONS = 8
    CERTIFICATE = 11
    CERTIFICATE_VERIFY = 15
    FINISHED = 20

# Cipher Suites (TLS 1.3)
class CipherSuite:
    TLS_AES_128_GCM_SHA256 = 0x1301
    TLS_AES_256_GCM_SHA384 = 0x1302
    TLS_CHACHA20_POLY1305_SHA256 = 0x1303

# TLS Extensions
class ExtensionType:
    SERVER_NAME = 0
    SUPPORTED_GROUPS = 10
    SIGNATURE_ALGORITHMS = 13
    SUPPORTED_VERSIONS = 43
    COOKIE = 44
    PSK_KEY_EXCHANGE_MODES = 45
    KEY_SHARE = 51

# Named Groups (Elliptic Curves)
class NamedGroup:
    SECP256R1 = 0x0017  # P-256
    X25519 = 0x001D     # Recommended curve

# Alert Levels and Descriptions
class AlertLevel:
    WARNING = 1
    FATAL = 2

class AlertDescription:
    CLOSE_NOTIFY = 0
    UNEXPECTED_MESSAGE = 10
    BAD_RECORD_MAC = 20
    HANDSHAKE_FAILURE = 40
    BAD_CERTIFICATE = 42
    CERTIFICATE_EXPIRED = 45
    CERTIFICATE_UNKNOWN = 46
    INTERNAL_ERROR = 80
```

**Binary Parsing Infrastructure (Complete Implementation)**

```python
# transport/parser.py - Complete binary parsing utilities
"""Binary parsing utilities for TLS protocol messages."""

import struct
from typing import Tuple, Optional

class BinaryParser:
    """Utilities for parsing binary TLS protocol data."""
    
    @staticmethod
    def parse_uint8(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse single byte as unsigned 8-bit integer."""
        if offset >= len(data):
            raise ValueError(f"Not enough data: need 1 byte at offset {offset}")
        value = data[offset]
        return value, offset + 1
    
    @staticmethod
    def parse_uint16(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse 2 bytes as unsigned 16-bit integer in network byte order."""
        if offset + 2 > len(data):
            raise ValueError(f"Not enough data: need 2 bytes at offset {offset}")
        value = struct.unpack("!H", data[offset:offset + 2])[0]
        return value, offset + 2
    
    @staticmethod
    def parse_uint24(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse 3 bytes as unsigned 24-bit integer in network byte order."""
        if offset + 3 > len(data):
            raise ValueError(f"Not enough data: need 3 bytes at offset {offset}")
        # Pad with zero byte for 32-bit parsing
        value = struct.unpack("!I", b"\x00" + data[offset:offset + 3])[0]
        return value, offset + 3
    
    @staticmethod
    def parse_uint32(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse 4 bytes as unsigned 32-bit integer in network byte order."""
        if offset + 4 > len(data):
            raise ValueError(f"Not enough data: need 4 bytes at offset {offset}")
        value = struct.unpack("!I", data[offset:offset + 4])[0]
        return value, offset + 4
    
    @staticmethod
    def parse_variable_bytes(data: bytes, offset: int, length_bytes: int) -> Tuple[bytes, int]:
        """Parse variable-length byte array with length prefix."""
        if length_bytes == 1:
            length, new_offset = BinaryParser.parse_uint8(data, offset)
        elif length_bytes == 2:
            length, new_offset = BinaryParser.parse_uint16(data, offset)
        elif length_bytes == 3:
            length, new_offset = BinaryParser.parse_uint24(data, offset)
        else:
            raise ValueError(f"Unsupported length field size: {length_bytes}")
        
        if new_offset + length > len(data):
            raise ValueError(f"Not enough data: need {length} bytes at offset {new_offset}")
        
        payload = data[new_offset:new_offset + length]
        return payload, new_offset + length
    
    @staticmethod
    def encode_uint8(value: int) -> bytes:
        """Encode unsigned 8-bit integer as bytes."""
        return struct.pack("!B", value)
    
    @staticmethod
    def encode_uint16(value: int) -> bytes:
        """Encode unsigned 16-bit integer as bytes in network byte order."""
        return struct.pack("!H", value)
    
    @staticmethod
    def encode_uint24(value: int) -> bytes:
        """Encode unsigned 24-bit integer as bytes in network byte order."""
        return struct.pack("!I", value)[1:]  # Remove first byte
    
    @staticmethod
    def encode_uint32(value: int) -> bytes:
        """Encode unsigned 32-bit integer as bytes in network byte order."""
        return struct.pack("!I", value)
    
    @staticmethod
    def encode_variable_bytes(data: bytes, length_bytes: int) -> bytes:
        """Encode variable-length byte array with length prefix."""
        length = len(data)
        if length_bytes == 1:
            return BinaryParser.encode_uint8(length) + data
        elif length_bytes == 2:
            return BinaryParser.encode_uint16(length) + data
        elif length_bytes == 3:
            return BinaryParser.encode_uint24(length) + data
        else:
            raise ValueError(f"Unsupported length field size: {length_bytes}")
```

**Core Classes (Skeleton Only)**

```python
# transport/tcp.py - TCP transport skeleton
class TCPTransport:
    """Manages TCP socket connections for TLS communication."""
    
    def __init__(self):
        self.socket: Optional[socket.socket] = None
        self.remote_address: Optional[Tuple[str, int]] = None
    
    def connect(self, hostname: str, port: int, timeout: float) -> None:
        """Establish TCP connection to remote server."""
        # TODO 1: Create socket with appropriate family (IPv4/IPv6)
        # TODO 2: Set socket timeout for connection establishment
        # TODO 3: Resolve hostname to IP address
        # TODO 4: Connect to remote address and port
        # TODO 5: Store socket and address for later use
        # Hint: Use socket.getaddrinfo() for hostname resolution
        pass
    
    def send_bytes(self, data: bytes) -> int:
        """Send raw bytes over TCP connection."""
        # TODO 1: Check that socket is connected
        # TODO 2: Send all data using socket.sendall()
        # TODO 3: Return number of bytes sent
        # TODO 4: Handle socket errors appropriately
        pass
    
    def receive_bytes(self, max_bytes: int) -> bytes:
        """Receive raw bytes from TCP connection."""
        # TODO 1: Check that socket is connected
        # TODO 2: Receive up to max_bytes using socket.recv()
        # TODO 3: Handle partial receives and socket errors
        # TODO 4: Return received data (may be less than max_bytes)
        pass
    
    def receive_exact(self, num_bytes: int) -> bytes:
        """Receive exactly specified number of bytes."""
        # TODO 1: Loop until exactly num_bytes received
        # TODO 2: Call receive_bytes() repeatedly if needed
        # TODO 3: Accumulate data until target length reached
        # TODO 4: Handle connection closure before target reached
        # Hint: This is critical for TLS record parsing
        pass

# http/client.py - HTTPS client skeleton  
class HTTPSClient:
    """High-level HTTPS client with TLS and HTTP support."""
    
    def __init__(self):
        self.connection: Optional[TLSConnection] = None
        self.transport: Optional[TCPTransport] = None
    
    def request(self, method: str, url: str, headers: Dict, body: bytes) -> Tuple[int, Dict, bytes]:
        """Make HTTPS request to server."""
        # TODO 1: Parse URL to extract hostname, port, path
        # TODO 2: Establish TLS connection if not already connected
        # TODO 3: Format HTTP request with method, path, headers, body
        # TODO 4: Send request over TLS connection
        # TODO 5: Receive and parse HTTP response
        # TODO 6: Return status code, headers dict, and response body
        pass
```

**Debugging and Development Support**

```python
# utils/hex_dump.py - Complete debugging utilities
"""Debugging utilities for binary protocol analysis."""

def hex_dump(data: bytes, width: int = 16, show_ascii: bool = True) -> str:
    """Create hex dump representation of binary data."""
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i:i + width]
        
        # Hex representation
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        hex_part = hex_part.ljust(width * 3 - 1)  # Pad to consistent width
        
        # ASCII representation
        ascii_part = ''
        if show_ascii:
            ascii_part = ' |' + ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk) + '|'
        
        lines.append(f'{i:08x}  {hex_part}{ascii_part}')
    
    return '\n'.join(lines)

def log_tls_record(record_type: int, data: bytes) -> None:
    """Log TLS record with appropriate formatting."""
    type_names = {20: "ChangeCipherSpec", 21: "Alert", 22: "Handshake", 23: "ApplicationData"}
    type_name = type_names.get(record_type, f"Unknown({record_type})")
    
    print(f"=== TLS Record: {type_name} ({len(data)} bytes) ===")
    print(hex_dump(data))
    print("=" * 50)
```

**Milestone Checkpoints**

After implementing each milestone, verify functionality with these specific tests:

**Milestone 1 Checkpoint - TCP and Record Layer**
```bash
# Test TCP connection establishment
python -c "
from transport.tcp import TCPTransport
t = TCPTransport()
t.connect('httpbin.org', 443, 5.0)
print('TCP connection successful')
"

# Test record layer parsing with real TLS data
# Expected: Should parse record type, version, length correctly
```

**Milestone 2 Checkpoint - ClientHello**
```bash
# Test ClientHello construction and sending
python -c "
from tls.handshake import ClientHello
hello = ClientHello('httpbin.org')
data = hello.encode()
print(f'ClientHello: {len(data)} bytes')
print('Extensions:', hello.extensions.keys())
"

# Expected output: ~200-300 byte ClientHello with SNI, supported_versions, key_share extensions
```

**Debugging Tips**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| "Connection refused" | Target port closed or firewall | `telnet hostname 443` to verify port | Check hostname and port number |
| "Partial TLS record received" | Fragmentation across TCP segments | Check `receive_exact()` implementation | Implement proper reassembly loop |
| "Invalid record type" | Parsing at wrong byte boundary | Hex dump received data | Verify record header parsing logic |
| "Handshake failure" | Malformed ClientHello | Wireshark capture of ClientHello | Compare with RFC 8446 message format |


## High-Level Architecture

> **Milestone(s):** All milestones (1-4) — this architecture supports the complete HTTPS client implementation from TCP connection through encrypted communication

Building an HTTPS client is like constructing a sophisticated embassy in a foreign country. Just as an embassy has multiple layers of security, communication protocols, and administrative processes working together to enable secure diplomatic exchanges, an HTTPS client requires multiple architectural layers that each handle specific aspects of secure communication. The outermost layer handles basic transportation (like the embassy's mail system), while inner layers manage increasingly complex security protocols (like diplomatic encryption and authentication), all orchestrated by a central coordination system that ensures messages flow correctly between layers.

The fundamental architectural challenge is organizing these layers so that each component has clear responsibilities while maintaining the complex state required for TLS communication. Unlike a simple HTTP client that just sends requests and receives responses, an HTTPS client must coordinate cryptographic handshakes, manage certificate validation, derive encryption keys, and maintain security state across multiple message exchanges.

### System Components

The HTTPS client architecture follows a strict layered design where each layer provides services to the layer above while depending only on services from layers below. This separation of concerns is critical because TLS involves intricate interactions between network communication, cryptographic operations, and application-level data processing.

Think of this layered architecture like a diplomatic protocol stack. At the bottom, you have basic transportation mechanisms (TCP sockets) that simply move bytes between locations. Above that, you have formatting protocols (TLS records) that package messages in standard diplomatic pouches. Higher up, you have negotiation protocols (TLS handshake) that establish secure communication channels. At the top, you have the actual diplomatic exchanges (HTTP requests and responses) that accomplish the real work.

| Component | Layer | Primary Responsibility | Key Dependencies |
|-----------|-------|----------------------|------------------|
| `HTTPSClient` | Application | High-level HTTPS request/response interface | `TLSConnection`, HTTP message formatting |
| `TLSConnection` | TLS Session | Complete TLS connection state and coordination | `HandshakeEngine`, `CryptoEngine`, `RecordLayer` |
| `HandshakeEngine` | TLS Handshake | TLS handshake message construction and parsing | `CryptoEngine`, certificate validation |
| `CryptoEngine` | Cryptographic Operations | Key derivation, encryption, decryption, signing | Platform crypto libraries, random number generation |
| `CertificateValidator` | Certificate Processing | X.509 certificate chain validation and hostname verification | X.509 parsing, trust store access |
| `RecordLayer` | TLS Framing | TLS record parsing, fragmentation, message classification | `BinaryParser`, message type constants |
| `TCPTransport` | Network Transport | Raw TCP socket communication and connection management | System socket API, network error handling |
| `BinaryParser` | Data Parsing | Binary message parsing utilities | None (static utility functions) |

The `HTTPSClient` serves as the primary interface that application code interacts with. It presents a familiar HTTP-like API while internally coordinating the complex TLS operations required for secure communication. This component maintains the illusion of simple request/response semantics while hiding the underlying handshake complexity, key management, and encryption operations.

The `TLSConnection` component acts as the central coordinator for all TLS-related state and operations. It maintains the connection's security context, coordinates handshake progression, and manages the transition from handshake to application data phases. This component implements the TLS state machine and ensures that operations occur in the correct sequence according to the TLS specification.

The `HandshakeEngine` specializes in constructing and parsing TLS handshake messages. It understands the complex binary formats of messages like ClientHello and ServerHello, manages extension processing, and coordinates with the crypto engine to perform key exchange operations. This component encapsulates the intricate details of TLS negotiation while providing a clean interface for the connection coordinator.

The `CryptoEngine` handles all cryptographic operations including random number generation, key derivation, digital signatures, and symmetric encryption/decryption. It abstracts the underlying cryptographic libraries and ensures that all operations use cryptographically secure implementations. This component also manages the complex HKDF-based key derivation required for TLS 1.3.

The `CertificateValidator` implements X.509 certificate chain validation according to RFC 5280 standards. It verifies digital signatures, checks certificate validity periods, validates certificate chains back to trusted root authorities, and performs hostname verification to prevent man-in-the-middle attacks. This component is critical for establishing trust in the server's identity.

The `RecordLayer` manages the TLS record format that frames all TLS communication. It handles record parsing, reassembly of fragmented records, message type classification, and the transition between plaintext and encrypted record processing. This component provides a clean message-oriented interface above the byte-oriented TCP transport.

The `TCPTransport` encapsulates all TCP socket operations including connection establishment, data transmission, and connection teardown. It handles network errors, partial reads/writes, and provides a reliable byte stream interface to higher layers. This component isolates network-specific concerns from the TLS protocol logic.

> **Decision: Layered Architecture with Clear Separation of Concerns**
> - **Context**: TLS involves complex interactions between networking, cryptography, certificate processing, and application data handling. These concerns could be tightly coupled or cleanly separated.
> - **Options Considered**: 
>   1. Monolithic design with all functionality in a single large class
>   2. Layered architecture with strict dependency hierarchy  
>   3. Event-driven architecture with loosely coupled components
> - **Decision**: Layered architecture with clear component boundaries and strict dependency hierarchy
> - **Rationale**: TLS has natural layering (transport → records → handshake → application), and each layer has distinct responsibilities. Strict layering makes testing easier since each component can be tested in isolation. The complexity is manageable when each layer focuses on its specific concerns.
> - **Consequences**: Enables independent testing and development of each component. Makes the codebase easier to understand and maintain. May introduce some performance overhead due to abstraction layers, but the benefits of maintainability outweigh the costs.

| Architecture Option | Pros | Cons | Chosen? |
|---------------------|------|------|---------|
| Monolithic Design | Simple dependencies, potentially faster | Very difficult to test and debug, tight coupling | No |
| Layered Architecture | Clear separation of concerns, testable components, matches TLS structure | Some abstraction overhead, more complex initialization | **Yes** |
| Event-Driven Architecture | Highly flexible, good for complex state machines | Difficult to reason about flow, debugging challenges | No |

The component interactions follow a strict request/response pattern within each layer, but the overall flow is more complex due to the stateful nature of TLS connections. During handshake phases, components must coordinate to build and parse complex message sequences. During application data phases, the flow simplifies to a more traditional request/response pattern.

![HTTPS Client System Components](./diagrams/system-components.svg)

### Recommended Module Structure

The module structure reflects the layered architecture while grouping related functionality to minimize cross-module dependencies. The organization follows Python packaging conventions while ensuring that import relationships match the architectural dependency hierarchy.

Think of the module structure like organizing a large engineering firm. You group teams by expertise area (networking, cryptography, certificates) while ensuring that teams only depend on services from teams lower in the organizational hierarchy. This prevents circular dependencies and ensures that changes in lower-level modules don't unexpectedly break higher-level functionality.

```
https_client/
├── __init__.py                    ← Public API exports
├── client.py                      ← HTTPSClient implementation
├── connection/
│   ├── __init__.py               ← TLS connection coordination
│   ├── tls_connection.py         ← TLSConnection main implementation
│   ├── state_machine.py          ← Connection state management
│   └── session.py                ← Session resumption support (future)
├── handshake/
│   ├── __init__.py               ← Handshake message processing
│   ├── engine.py                 ← HandshakeEngine implementation
│   ├── messages.py               ← ClientHello, ServerHello, etc.
│   ├── extensions.py             ← TLS extension handling
│   └── finished.py               ← Finished message verification
├── crypto/
│   ├── __init__.py               ← Cryptographic operations
│   ├── engine.py                 ← CryptoEngine implementation
│   ├── key_derivation.py         ← HKDF-based key derivation
│   ├── ecdhe.py                  ← ECDHE key exchange
│   ├── aead.py                   ← AES-GCM encryption/decryption
│   └── random.py                 ← Cryptographically secure random numbers
├── certificates/
│   ├── __init__.py               ← Certificate validation
│   ├── validator.py              ← CertificateValidator implementation
│   ├── x509_parser.py            ← X.509 certificate parsing
│   ├── chain_validation.py       ← Certificate chain verification
│   └── hostname_verification.py  ← Hostname matching logic
├── record/
│   ├── __init__.py               ← TLS record layer
│   ├── layer.py                  ← RecordLayer implementation
│   ├── types.py                  ← Record type constants and parsing
│   └── fragmentation.py          ← Record fragmentation handling
├── transport/
│   ├── __init__.py               ← Network transport
│   ├── tcp.py                    ← TCPTransport implementation
│   └── timeouts.py               ← Connection timeout handling
├── utils/
│   ├── __init__.py               ← Utility functions
│   ├── binary_parser.py          ← BinaryParser implementation
│   ├── constants.py              ← TLS constants and enums
│   └── hex_dump.py               ← Binary data debugging utilities
└── tests/
    ├── __init__.py
    ├── unit/                     ← Unit tests for each component
    │   ├── test_tcp_transport.py
    │   ├── test_record_layer.py
    │   ├── test_handshake_engine.py
    │   ├── test_crypto_engine.py
    │   └── test_certificate_validator.py
    ├── integration/              ← Integration tests
    │   ├── test_handshake_flow.py
    │   └── test_full_connection.py
    └── fixtures/                 ← Test certificates and data
        ├── test_certificates/
        └── handshake_vectors/
```

The top-level `client.py` module contains the `HTTPSClient` class that serves as the primary API entry point. This module should have minimal implementation logic, primarily coordinating between the connection layer and HTTP message formatting. It presents a clean, HTTP-like interface that hides the TLS complexity from application code.

The `connection/` package manages TLS connection state and coordinates the overall TLS protocol flow. The `tls_connection.py` module contains the main `TLSConnection` class that implements the connection state machine and coordinates handshake and application data phases. The `state_machine.py` module provides explicit state management to ensure TLS operations occur in the correct sequence.

The `handshake/` package encapsulates all TLS handshake message construction and parsing logic. The `engine.py` module contains the main `HandshakeEngine` class, while `messages.py` provides classes for specific handshake message types like `ClientHello` and `ServerHello`. The `extensions.py` module handles TLS extensions including SNI, supported versions, and key share extensions. The `finished.py` module implements the complex Finished message verification that authenticates the entire handshake transcript.

The `crypto/` package isolates all cryptographic operations behind clean interfaces. The `engine.py` module provides the main `CryptoEngine` class that coordinates cryptographic operations. Specialized modules handle specific operations: `key_derivation.py` implements HKDF-based key derivation, `ecdhe.py` handles elliptic curve key exchange, `aead.py` provides AES-GCM encryption/decryption, and `random.py` ensures cryptographically secure random number generation.

The `certificates/` package manages X.509 certificate validation and trust establishment. The `validator.py` module contains the main `CertificateValidator` class, while specialized modules handle certificate parsing, chain validation, and hostname verification. This separation allows for clean unit testing of each validation step.

The `record/` package implements the TLS record layer that frames all TLS communication. The `layer.py` module provides the main `RecordLayer` class, while `types.py` defines constants and parsing logic for different record types. The `fragmentation.py` module handles the complex case where TLS records span multiple TCP segments or where multiple records arrive in a single TCP segment.

The `transport/` package abstracts network communication details. The `tcp.py` module contains the `TCPTransport` class that handles TCP socket operations, connection management, and network error handling. The `timeouts.py` module provides timeout management for various network operations.

The `utils/` package contains utility functions and constants used throughout the codebase. The `binary_parser.py` module provides the `BinaryParser` class with static methods for parsing binary TLS messages. The `constants.py` module defines all TLS constants and enumerations in a centralized location. The `hex_dump.py` module provides debugging utilities for examining binary data.

> **Decision: Package Organization by Functional Area Rather Than Layer**
> - **Context**: The codebase could be organized by architectural layer (layer1/, layer2/) or by functional area (crypto/, certificates/, handshake/). Each approach has different benefits for code navigation and maintenance.
> - **Options Considered**:
>   1. Layer-based organization (transport/, record/, handshake/, application/)
>   2. Functional area organization (crypto/, certificates/, handshake/, connection/)
>   3. Single flat package with all modules
> - **Decision**: Functional area organization with clear sub-packages for each major concern
> - **Rationale**: Developers typically work on related functionality together (e.g., all certificate validation code, or all cryptographic operations). Functional organization makes it easier to find related code and understand component boundaries. The import hierarchy still enforces architectural layering even with functional packaging.
> - **Consequences**: Makes the codebase easier to navigate for developers familiar with TLS concepts. Enables cleaner unit testing since related functionality is grouped together. May require more careful import management to maintain proper dependency hierarchy.

The dependency hierarchy must be carefully managed to prevent circular imports. Lower-level packages (transport, utils) should never import from higher-level packages (connection, client). The import flow should follow this pattern:

| Package Level | Packages | Can Import From | Cannot Import From |
|---------------|----------|-----------------|-------------------|
| Level 1 (Base) | `utils`, `transport` | Standard library only | Any project packages |
| Level 2 (Protocol) | `record`, `crypto` | Level 1 packages, standard library | Level 3+ packages |
| Level 3 (TLS Logic) | `handshake`, `certificates` | Level 1-2 packages, standard library | Level 4+ packages |
| Level 4 (Session) | `connection` | Level 1-3 packages, standard library | Level 5+ packages |
| Level 5 (Application) | `client` (top-level) | All lower levels, standard library | None |

This import hierarchy ensures that the architectural layering is enforced at the code level. Each package can depend on services from lower levels but cannot create circular dependencies that would make the code difficult to test and maintain.

### Common Pitfalls

⚠️ **Pitfall: Circular Dependencies Between Components**
Many developers create circular dependencies between architectural layers, especially between the handshake engine and crypto engine, or between the connection and handshake components. This happens because these components need to coordinate closely during TLS operations. However, circular dependencies make unit testing extremely difficult and can cause import errors in Python. The solution is to use dependency injection where higher-level components pass callback functions or interfaces to lower-level components, allowing coordination without circular imports.

⚠️ **Pitfall: Mixing State Management Across Components**
A common mistake is allowing multiple components to directly modify TLS connection state, leading to inconsistent state and difficult-to-debug issues. For example, both the handshake engine and record layer might try to update cipher state independently. This violates the single responsibility principle and makes state transitions unpredictable. Instead, designate the `TLSConnection` component as the single owner of connection state, with other components requesting state changes through well-defined interfaces.

⚠️ **Pitfall: Insufficient Error Boundary Isolation**
Developers often let errors from one component propagate uncontrolled through other components, making it difficult to implement proper error recovery. For instance, a certificate validation failure might bubble up as a generic "connection failed" error without specific details about what went wrong. Each architectural layer should catch errors from lower layers, add appropriate context, and re-raise them with layer-specific error types that higher layers can handle appropriately.

⚠️ **Pitfall: Bypassing Architectural Layers for Performance**
When developers encounter performance bottlenecks, they sometimes create direct dependencies between non-adjacent layers to avoid abstraction overhead. For example, directly calling crypto functions from the record layer instead of going through the crypto engine interface. While this might provide minor performance improvements, it breaks the architectural integrity and makes the code much harder to maintain and test. The performance gains are usually negligible compared to the maintenance costs.

### Implementation Guidance

The HTTPS client architecture requires careful coordination of multiple complex subsystems. The recommended approach is to build and test each layer independently before integrating them into the complete system.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|------------------|
| TCP Transport | Python `socket` module with basic error handling | `asyncio` with connection pooling and advanced timeout management |
| Binary Parsing | Manual `struct.unpack()` with error checking | Custom binary parser with automatic length validation and bounds checking |
| Cryptographic Operations | Python `cryptography` library with direct API calls | Abstracted crypto engine with pluggable backend support |
| Certificate Validation | Basic X.509 parsing with manual chain validation | Full RFC 5280 implementation with policy validation |
| Random Number Generation | `os.urandom()` for all random bytes | Hardware RNG with fallback to software implementation |
| Error Handling | Exception propagation with basic logging | Structured error types with detailed context and recovery hints |

#### Recommended Project Structure

```
https_client_project/
├── README.md                      ← Project documentation and setup instructions
├── requirements.txt               ← Python dependencies
├── setup.py                       ← Package installation configuration
├── https_client/                  ← Main package (as described above)
│   ├── __init__.py
│   ├── client.py
│   └── [all subpackages...]
├── examples/                      ← Usage examples and demonstrations
│   ├── simple_request.py          ← Basic HTTPS GET request
│   ├── certificate_inspection.py  ← Certificate validation demo
│   └── debugging_connection.py    ← Debug output examples
├── tests/                         ← Test suite (as described above)
├── docs/                          ← Additional documentation
│   ├── tls_primer.md             ← TLS concepts explanation
│   ├── debugging_guide.md        ← Troubleshooting help
│   └── api_reference.md          ← Complete API documentation
└── scripts/                      ← Development and testing utilities
    ├── test_server.py            ← Local TLS test server
    ├── generate_certs.py         ← Test certificate generation
    └── benchmark.py              ← Performance testing
```

#### Infrastructure Starter Code

The following infrastructure components provide complete, working implementations for non-core functionality that supports the main learning objectives:

**Binary Parser Utilities** (`utils/binary_parser.py`):
```python
import struct
from typing import Tuple, List, Optional

class BinaryParser:
    """Static utility methods for parsing binary TLS data structures."""
    
    @staticmethod
    def parse_uint8(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse single byte as unsigned 8-bit integer."""
        if offset >= len(data):
            raise ValueError(f"Cannot parse uint8 at offset {offset}, data length {len(data)}")
        return data[offset], offset + 1
    
    @staticmethod  
    def parse_uint16(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse 2 bytes as unsigned 16-bit integer in network byte order."""
        if offset + 2 > len(data):
            raise ValueError(f"Cannot parse uint16 at offset {offset}, need 2 bytes, data length {len(data)}")
        value = struct.unpack('!H', data[offset:offset+2])[0]
        return value, offset + 2
    
    @staticmethod
    def parse_uint24(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse 3 bytes as unsigned 24-bit integer in network byte order."""
        if offset + 3 > len(data):
            raise ValueError(f"Cannot parse uint24 at offset {offset}, need 3 bytes, data length {len(data)}")
        # Parse as 32-bit with leading zero byte
        padded = b'\x00' + data[offset:offset+3]
        value = struct.unpack('!I', padded)[0]
        return value, offset + 3
    
    @staticmethod
    def parse_variable_bytes(data: bytes, offset: int, length_bytes: int) -> Tuple[bytes, int]:
        """Parse variable-length byte array with length prefix."""
        if length_bytes == 1:
            length, new_offset = BinaryParser.parse_uint8(data, offset)
        elif length_bytes == 2:
            length, new_offset = BinaryParser.parse_uint16(data, offset)
        elif length_bytes == 3:
            length, new_offset = BinaryParser.parse_uint24(data, offset)
        else:
            raise ValueError(f"Unsupported length field size: {length_bytes}")
        
        if new_offset + length > len(data):
            raise ValueError(f"Cannot parse {length} bytes at offset {new_offset}, data length {len(data)}")
        
        return data[new_offset:new_offset + length], new_offset + length
    
    @staticmethod
    def encode_uint16(value: int) -> bytes:
        """Encode 16-bit unsigned integer in network byte order."""
        return struct.pack('!H', value)
    
    @staticmethod
    def encode_uint24(value: int) -> bytes:
        """Encode 24-bit unsigned integer in network byte order."""
        return struct.pack('!I', value)[1:]  # Remove leading byte
    
    @staticmethod
    def hex_dump(data: bytes, width: int = 16, show_ascii: bool = True) -> str:
        """Create hex dump representation of binary data for debugging."""
        lines = []
        for i in range(0, len(data), width):
            chunk = data[i:i + width]
            hex_str = ' '.join(f'{b:02x}' for b in chunk)
            if show_ascii:
                ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
                lines.append(f'{i:08x}  {hex_str:<{width*3}}  {ascii_str}')
            else:
                lines.append(f'{i:08x}  {hex_str}')
        return '\n'.join(lines)
```

**TLS Constants and Enumerations** (`utils/constants.py`):
```python
"""TLS protocol constants and enumerations."""

# Content Types (RFC 8446 Section 5.1)
class ContentType:
    CHANGE_CIPHER_SPEC = 20
    ALERT = 21
    HANDSHAKE = 22
    APPLICATION_DATA = 23

# TLS Versions
class TLSVersion:
    TLS_1_0 = 0x0301
    TLS_1_1 = 0x0302
    TLS_1_2 = 0x0303
    TLS_1_3 = 0x0304

# Handshake Message Types (RFC 8446 Section 4)
class HandshakeType:
    CLIENT_HELLO = 1
    SERVER_HELLO = 2
    NEW_SESSION_TICKET = 4
    END_OF_EARLY_DATA = 5
    ENCRYPTED_EXTENSIONS = 8
    CERTIFICATE = 11
    CERTIFICATE_REQUEST = 13
    CERTIFICATE_VERIFY = 15
    FINISHED = 20
    KEY_UPDATE = 24
    MESSAGE_HASH = 254

# Cipher Suites (RFC 8446 Appendix B.4)
class CipherSuite:
    TLS_AES_128_GCM_SHA256 = 0x1301
    TLS_AES_256_GCM_SHA384 = 0x1302
    TLS_CHACHA20_POLY1305_SHA256 = 0x1303

# Extension Types (IANA TLS Extensions Registry)
class ExtensionType:
    SERVER_NAME = 0
    MAX_FRAGMENT_LENGTH = 1
    STATUS_REQUEST = 5
    SUPPORTED_GROUPS = 10
    SIGNATURE_ALGORITHMS = 13
    USE_SRTP = 14
    HEARTBEAT = 15
    APPLICATION_LAYER_PROTOCOL_NEGOTIATION = 16
    SIGNED_CERTIFICATE_TIMESTAMP = 18
    CLIENT_CERTIFICATE_TYPE = 19
    SERVER_CERTIFICATE_TYPE = 20
    PADDING = 21
    PRE_SHARED_KEY = 41
    EARLY_DATA = 42
    SUPPORTED_VERSIONS = 43
    COOKIE = 44
    PSK_KEY_EXCHANGE_MODES = 45
    CERTIFICATE_AUTHORITIES = 47
    OID_FILTERS = 48
    POST_HANDSHAKE_AUTH = 49
    SIGNATURE_ALGORITHMS_CERT = 50
    KEY_SHARE = 51

# Named Groups for ECDHE (RFC 8446 Section 4.2.7)
class NamedGroup:
    # Elliptic Curve Groups
    SECP256R1 = 0x0017
    SECP384R1 = 0x0018
    SECP521R1 = 0x0019
    X25519 = 0x001D
    X448 = 0x001E

# Alert Levels and Descriptions (RFC 8446 Section 6)
class AlertLevel:
    WARNING = 1
    FATAL = 2

class AlertDescription:
    CLOSE_NOTIFY = 0
    UNEXPECTED_MESSAGE = 10
    BAD_RECORD_MAC = 20
    RECORD_OVERFLOW = 22
    HANDSHAKE_FAILURE = 40
    BAD_CERTIFICATE = 42
    UNSUPPORTED_CERTIFICATE = 43
    CERTIFICATE_REVOKED = 44
    CERTIFICATE_EXPIRED = 45
    CERTIFICATE_UNKNOWN = 46
    ILLEGAL_PARAMETER = 47
    UNKNOWN_CA = 48
    ACCESS_DENIED = 49
    DECODE_ERROR = 50
    DECRYPT_ERROR = 51
    PROTOCOL_VERSION = 70
    INSUFFICIENT_SECURITY = 71
    INTERNAL_ERROR = 80
    INAPPROPRIATE_FALLBACK = 86
    USER_CANCELED = 90
    MISSING_EXTENSION = 109
    UNSUPPORTED_EXTENSION = 110
    UNRECOGNIZED_NAME = 112
    BAD_CERTIFICATE_STATUS_RESPONSE = 113
    UNKNOWN_PSK_IDENTITY = 115
    CERTIFICATE_REQUIRED = 116
    NO_APPLICATION_PROTOCOL = 120
```

#### Core Logic Skeletons

The following skeletons provide the structure for core components that learners should implement:

**HTTPSClient Main Interface** (`client.py`):
```python
from typing import Dict, Tuple, Optional
from .connection.tls_connection import TLSConnection
from .transport.tcp import TCPTransport

class HTTPSClient:
    """High-level HTTPS client interface providing HTTP-like request/response API."""
    
    def __init__(self):
        self.connection: Optional[TLSConnection] = None
        self.transport: Optional[TCPTransport] = None
    
    def request(self, method: str, url: str, headers: Optional[Dict[str, str]] = None, 
                body: Optional[bytes] = None) -> Tuple[int, Dict[str, str], bytes]:
        """Make an HTTPS request and return status, headers, and body."""
        # TODO 1: Parse URL to extract hostname, port, and path
        # TODO 2: Establish TCP connection using TCPTransport
        # TODO 3: Perform TLS handshake using TLSConnection
        # TODO 4: Format HTTP request with method, path, headers, and body
        # TODO 5: Send HTTP request over encrypted TLS connection
        # TODO 6: Receive and parse HTTP response from TLS connection
        # TODO 7: Extract status code, response headers, and response body
        # TODO 8: Return parsed response as tuple
        # Hint: Use urllib.parse for URL parsing
        # Hint: HTTP/1.1 requires Host header matching the URL hostname
        pass
    
    def close(self) -> None:
        """Close the HTTPS connection and clean up resources."""
        # TODO 1: Send TLS close_notify alert to server
        # TODO 2: Close TLS connection and clear cryptographic state
        # TODO 3: Close TCP transport connection
        # TODO 4: Reset connection and transport to None
        pass
```

**TCP Transport Layer** (`transport/tcp.py`):
```python
import socket
from typing import Optional, Tuple

class TCPTransport:
    """TCP socket transport providing reliable byte stream for TLS."""
    
    def __init__(self):
        self.socket: Optional[socket.socket] = None
        self.remote_address: Optional[Tuple[str, int]] = None
    
    def connect(self, hostname: str, port: int, timeout: float = 10.0) -> None:
        """Establish TCP connection to remote server."""
        # TODO 1: Create TCP socket with appropriate socket family (AF_INET/AF_INET6)
        # TODO 2: Set socket timeout for connection establishment
        # TODO 3: Resolve hostname to IP address using socket.getaddrinfo()
        # TODO 4: Connect to resolved address and store connection info
        # TODO 5: Configure socket for TLS use (disable Nagle's algorithm)
        # Hint: Use socket.IPPROTO_TCP and socket.TCP_NODELAY
        # Hint: Handle both IPv4 and IPv6 addresses from getaddrinfo()
        pass
    
    def send_bytes(self, data: bytes) -> int:
        """Send raw bytes over TCP connection."""
        # TODO 1: Check if socket is connected, raise error if not
        # TODO 2: Handle partial sends with loop until all data sent
        # TODO 3: Handle network errors and convert to appropriate exceptions
        # TODO 4: Return total number of bytes successfully sent
        # Hint: socket.send() may not send all data in one call
        # Hint: Handle EAGAIN/EWOULDBLOCK for non-blocking sockets
        pass
    
    def receive_bytes(self, max_bytes: int) -> bytes:
        """Receive raw bytes from TCP connection."""
        # TODO 1: Check if socket is connected, raise error if not
        # TODO 2: Receive up to max_bytes from socket
        # TODO 3: Handle connection closed by peer (empty bytes returned)
        # TODO 4: Handle network errors and timeouts appropriately
        # TODO 5: Return received bytes (may be less than max_bytes)
        # Hint: Empty bytes means peer closed connection
        pass
    
    def receive_exact(self, num_bytes: int) -> bytes:
        """Receive exactly the specified number of bytes."""
        # TODO 1: Initialize buffer to accumulate received bytes
        # TODO 2: Loop calling receive_bytes() until exactly num_bytes received
        # TODO 3: Handle case where peer closes connection before all bytes received
        # TODO 4: Combine partial receives into complete message
        # TODO 5: Return exactly num_bytes or raise exception
        # Hint: This is critical for TLS record parsing
        pass
```

#### Milestone Checkpoints

**Milestone 1 Checkpoint - TCP and Record Layer:**
After implementing `TCPTransport` and basic `RecordLayer`, you should be able to:
- Connect to a real HTTPS server on port 443: `transport.connect("www.google.com", 443)`
- Send raw bytes and receive the server's response (likely a TLS alert since you haven't performed handshake)
- Parse the server's TLS records and identify record types
- Expected behavior: Server should send back a TLS alert record indicating handshake failure

**Milestone 2 Checkpoint - ClientHello:**
After implementing `HandshakeEngine` and ClientHello construction:
- Send a properly formatted ClientHello to a real server
- Receive and parse the server's ServerHello response
- Verify that cipher suite negotiation selected a supported algorithm
- Expected behavior: Server responds with ServerHello, Certificate, and ServerHelloDone messages

**Milestone 3 Checkpoint - Key Exchange:**
After implementing ECDHE key exchange and key derivation:
- Generate ephemeral key pairs for ECDHE
- Compute shared secret from server's key share
- Derive traffic keys using HKDF
- Expected behavior: Successfully derive matching keys that will decrypt server's Finished message

**Milestone 4 Checkpoint - Encrypted Communication:**
After implementing AEAD encryption and HTTP processing:
- Send encrypted HTTP GET request over TLS connection
- Receive and decrypt HTTP response from server
- Verify response contains expected HTML content
- Test command: `python -c "from https_client import HTTPSClient; print(HTTPSClient().request('GET', 'https://www.google.com/'))"`


## Data Model

> **Milestone(s):** All milestones (1-4) — these data structures form the foundation for TLS record processing, handshake message construction, and cryptographic state management

Think of the data model as the blueprint for a secure diplomatic communication system. Just as diplomats need standardized message formats, authentication credentials, and secure channels to communicate between nations, our HTTPS client needs precisely defined data structures to represent TLS records, handshake messages, and cryptographic state. Each data structure serves a specific role in the complex choreography of establishing and maintaining secure connections.

The TLS protocol operates through a layered approach where binary data flows through multiple stages of parsing, validation, and transformation. At the lowest level, we have the **TLS record format** that provides framing and message classification. Above that, we have **handshake message types** that negotiate connection parameters and exchange cryptographic material. Finally, we maintain **cryptographic state** that tracks keys, cipher suites, and connection status throughout the TLS session lifecycle.

Understanding these data structures is crucial because TLS is fundamentally a binary protocol where every byte has precise meaning. Unlike text-based protocols like HTTP, TLS messages follow rigid binary formats with specific field lengths, byte ordering, and encoding rules. A single misaligned byte or incorrect length calculation can cause handshake failures or security vulnerabilities.

### TLS Record Format

The TLS record layer serves as the foundation for all TLS communication, functioning like the envelope system for secure mail. Just as every letter needs an envelope with sender information, recipient details, and content type markings, every piece of TLS data gets wrapped in a record header that identifies its purpose and boundaries. This consistent framing allows the receiver to properly classify and route different types of TLS messages.

The record layer provides several critical functions. First, it establishes **message boundaries** in the TCP stream, since TCP is a byte-stream protocol that doesn't preserve message boundaries. Second, it provides **message classification** so the receiver knows whether incoming data represents handshake negotiations, alert notifications, or encrypted application data. Third, it handles **fragmentation** when large messages need to be split across multiple records, and **reassembly** when records span multiple TCP segments.

> **Decision: Fixed-Size Record Header with Variable-Length Payload**
> - **Context**: TLS needs to frame messages over TCP while supporting different content types and variable-size payloads
> - **Options Considered**: 
>   1. Fixed-size records with padding
>   2. Variable-length headers with embedded type information
>   3. Fixed-size headers with variable-length payloads
> - **Decision**: Fixed-size 5-byte header with variable-length payload up to 16KB
> - **Rationale**: Provides efficient parsing (header size is known), supports large messages (16KB limit), and minimizes overhead for small messages
> - **Consequences**: Simple parsing logic but requires fragmentation handling for messages exceeding 16KB

The TLS record structure follows a precise binary format that must be implemented exactly as specified in RFC 8446. Each record begins with a 5-byte header followed by a variable-length payload:

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Content Type | uint8 | 1 byte | Identifies the type of data in this record (handshake, alert, application data, etc.) |
| Legacy Version | uint16 | 2 bytes | Always set to TLS 1.2 (0x0303) for compatibility, regardless of actual TLS version |
| Length | uint16 | 2 bytes | Length of the payload in bytes, maximum 16384 (16KB) |
| Payload | bytes | variable | The actual message data, format depends on content type |

The **content type** field uses specific constants to identify different categories of TLS messages. This classification system allows the TLS implementation to route records to appropriate processing handlers:

| Content Type | Value | Purpose | Handler |
|--------------|-------|---------|---------|
| `CHANGE_CIPHER_SPEC` | 20 | Legacy cipher change notification (unused in TLS 1.3) | Compatibility processing |
| `ALERT` | 21 | Error notifications and connection closure alerts | Alert processing engine |
| `HANDSHAKE` | 22 | Handshake negotiation messages (ClientHello, ServerHello, etc.) | Handshake state machine |
| `APPLICATION_DATA` | 23 | Encrypted application data after handshake completion | Application data decryption |

The **legacy version** field represents an interesting architectural decision in TLS 1.3. Rather than updating this field to reflect the actual TLS version, the protocol designers chose to maintain backward compatibility by always setting it to TLS 1.2 (0x0303). The actual version negotiation occurs within handshake messages using the Supported Versions extension. This approach prevents middleboxes from blocking connections based on version numbers they don't recognize.

The **length field** uses network byte order (big-endian) and has a maximum value of 16,384 bytes (16KB). This limit serves multiple purposes: it prevents excessive memory allocation attacks, ensures reasonable processing latency, and aligns with typical network packet sizes. When a TLS message exceeds this limit, it must be fragmented across multiple records.

> The critical insight here is that TLS records are the atomic unit of TLS processing. Even though TCP provides a reliable byte stream, the application must reconstruct record boundaries to properly process TLS messages. This means handling partial record reads and multi-record messages correctly.

**Record Fragmentation and Reassembly**

Large TLS messages, particularly Certificate messages containing long certificate chains, often exceed the 16KB record limit and require fragmentation. The fragmentation process works at the TLS layer, not the TCP layer, meaning a single logical handshake message might be split across multiple TLS records.

The fragmentation algorithm follows these steps:
1. Determine the total message size including handshake message headers
2. Calculate how many 16KB records are needed to contain the entire message
3. Create the first record with the handshake message header and as much payload as fits
4. Create subsequent records containing only payload data (no repeated headers)
5. Mark the final record to indicate message completion

Reassembly works in reverse, collecting record payloads until a complete message is reconstructed. The implementation must maintain a reassembly buffer and track expected message lengths to detect completion.

⚠️ **Pitfall: Partial Record Reads**
TCP `recv()` calls can return fewer bytes than requested, potentially splitting a TLS record across multiple `recv()` operations. Many implementations incorrectly assume that a single `recv()` call will return a complete record. The fix is to implement a `receive_exact()` function that calls `recv()` in a loop until the requested number of bytes is received or an error occurs.

⚠️ **Pitfall: Length Field Validation**
Failing to validate the length field can lead to buffer overflows or excessive memory allocation. Always check that the length field is within the valid range (0-16384) before allocating buffers or attempting to read payload data.

### Handshake Message Types

Handshake messages represent the negotiation phase of TLS connections, functioning like a diplomatic protocol where two parties exchange credentials, agree on communication parameters, and verify each other's identity. Unlike the simple record layer format, handshake messages have complex internal structures with nested fields, variable-length arrays, and cryptographic material.

Each handshake message follows a consistent header format, regardless of the specific message type. This standardization allows implementations to parse the message type and length before processing the type-specific payload:

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Message Type | uint8 | 1 byte | Identifies the specific handshake message (ClientHello, ServerHello, etc.) |
| Length | uint24 | 3 bytes | Length of the message payload in bytes (excluding this 4-byte header) |
| Payload | bytes | variable | Message-specific data structure, format depends on message type |

The use of a 24-bit length field (uint24) is unusual in modern protocols but reflects TLS's heritage and need to handle large certificate chains. This allows handshake messages up to 16MB in size, far larger than typical record payloads.

> **Decision: 24-bit Length Fields for Handshake Messages**
> - **Context**: Handshake messages, particularly Certificate messages, can be very large due to certificate chains
> - **Options Considered**:
>   1. 16-bit length fields (limited to 64KB)
>   2. 32-bit length fields (potential DoS via memory exhaustion)
>   3. 24-bit length fields (up to 16MB)
> - **Decision**: 24-bit length fields for handshake message headers
> - **Rationale**: Supports large certificate chains while preventing excessive memory allocation attacks
> - **Consequences**: Requires custom 24-bit parsing logic but provides appropriate size limits

**ClientHello Message Structure**

The ClientHello message initiates the TLS handshake and represents the client's opening negotiation position. Think of it as a diplomatic proposal that lists the client's capabilities, preferences, and requirements for the secure communication channel.

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Legacy Version | uint16 | 2 bytes | Set to TLS 1.2 (0x0303) for compatibility |
| Random | bytes[32] | 32 bytes | Cryptographically secure random bytes for replay protection |
| Legacy Session ID | uint8 + bytes | variable | Session ID length (1 byte) followed by session ID data |
| Cipher Suites | uint16 + CipherSuite[] | variable | Number of cipher suites (2 bytes) followed by cipher suite list |
| Legacy Compression Methods | uint8 + bytes | variable | Compression method count (1 byte) and methods (always just null compression) |
| Extensions | uint16 + Extension[] | variable | Extensions length (2 bytes) followed by extension data |

The **random field** contains 32 bytes of cryptographically secure random data that serves multiple security purposes. It provides entropy for key derivation, prevents replay attacks, and ensures that each handshake produces unique keys even with identical parameters. The random bytes must be generated using a cryptographically secure random number generator (CSPRNG) and should never be reused across connections.

The **legacy session ID** field exists for backward compatibility with TLS 1.2 session resumption mechanisms. In TLS 1.3, this field should contain 32 random bytes to provide "session ID compatibility mode" that prevents certain middlebox compatibility issues. The session ID length is encoded as a single byte (0-32), followed by that many session ID bytes.

The **cipher suites** field contains the client's ordered list of supported cipher suites. Each cipher suite is identified by a 16-bit value that specifies a complete cryptographic package including key exchange method, authentication algorithm, encryption cipher, and hash function. For TLS 1.3, clients should include `TLS_AES_128_GCM_SHA256` (0x1301) as a minimum requirement.

**ServerHello Message Structure**

The ServerHello represents the server's response to the client's negotiation proposal, selecting specific parameters from the client's offered options:

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Legacy Version | uint16 | 2 bytes | Set to TLS 1.2 (0x0303) for compatibility |
| Random | bytes[32] | 32 bytes | Server's cryptographically secure random bytes |
| Legacy Session ID Echo | uint8 + bytes | variable | Echoes the client's session ID for compatibility |
| Cipher Suite | uint16 | 2 bytes | Server's selected cipher suite from client's list |
| Legacy Compression Method | uint8 | 1 byte | Selected compression method (always 0 for null compression) |
| Extensions | uint16 + Extension[] | variable | Server's extensions in response to client extensions |

The server's random field follows the same requirements as the client's random: 32 cryptographically secure random bytes that contribute to key derivation and prevent replay attacks.

**Extension Format**

TLS extensions provide a mechanism for adding new features to the protocol without breaking backward compatibility. Each extension follows a standardized Type-Length-Value (TLV) format:

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Extension Type | uint16 | 2 bytes | Identifies the specific extension (SNI, Supported Versions, etc.) |
| Extension Length | uint16 | 2 bytes | Length of the extension data in bytes |
| Extension Data | bytes | variable | Extension-specific data structure |

Critical extensions for TLS 1.3 include:

| Extension Type | Value | Purpose | Required |
|----------------|-------|---------|----------|
| `SERVER_NAME` | 0 | Server Name Indication (SNI) for virtual hosting | Yes |
| `SUPPORTED_VERSIONS` | 43 | Negotiates actual TLS version (overrides legacy version field) | Yes |
| `KEY_SHARE` | 51 | Contains client's ECDHE key share for key exchange | Yes |

⚠️ **Pitfall: Extension Ordering**
Some TLS implementations and middleboxes are sensitive to extension ordering. While the TLS specification doesn't mandate a specific order, implementations should follow common conventions: SNI first, then Supported Versions, then Key Share, followed by other extensions.

⚠️ **Pitfall: Length Field Calculations**
Extension length fields must exactly match the extension data size. Off-by-one errors in length calculations cause parsing failures. Always calculate lengths programmatically rather than hardcoding them.

### Cryptographic State

The cryptographic state represents the security context of a TLS connection, functioning like the key management system for a high-security facility. Just as a security system tracks which keys are active, when they expire, and what areas they protect, the TLS cryptographic state manages encryption keys, cipher configurations, and security parameters throughout the connection lifecycle.

This state management is particularly complex in TLS 1.3 because the protocol uses multiple sets of keys for different purposes and connection phases. Understanding this state model is crucial for implementing secure key derivation, proper key rotation, and correct encryption/decryption operations.

> **Decision: Separate Key Sets for Different TLS Phases**
> - **Context**: TLS 1.3 uses different keys for handshake protection versus application data protection
> - **Options Considered**:
>   1. Single key set used for all encryption
>   2. Separate handshake and application data key sets
>   3. Multiple key sets with forward secrecy guarantees
> - **Decision**: Separate key sets for handshake traffic and application data traffic
> - **Rationale**: Provides forward secrecy, allows key rotation, and isolates handshake security from application data security
> - **Consequences**: Requires complex key derivation logic but provides stronger security guarantees

**Connection State Structure**

The overall connection state tracks the current phase of the TLS handshake and maintains all cryptographic material needed for secure communication:

| Field | Type | Description |
|-------|------|-------------|
| Connection Phase | enum | Current handshake state (START, WAIT_SH, WAIT_EE, CONNECTED, etc.) |
| TLS Version | uint16 | Negotiated TLS version (should be 0x0304 for TLS 1.3) |
| Cipher Suite | uint16 | Negotiated cipher suite identifier |
| Client Random | bytes[32] | Client's random bytes from ClientHello |
| Server Random | bytes[32] | Server's random bytes from ServerHello |
| Handshake Context | bytes | Running hash of all handshake messages for Finished verification |
| Key Schedule State | KeySchedule | Current key derivation state and derived keys |
| Sequence Numbers | SequenceNumbers | Separate sequence numbers for reading and writing |
| Certificate Chain | Certificate[] | Server's certificate chain for validation |

**Key Schedule State**

TLS 1.3 uses a sophisticated key derivation system based on HKDF (HMAC-based Key Derivation Function) that produces different keys for different purposes. The key schedule follows a tree structure where each level derives keys from the previous level:

| Field | Type | Description |
|-------|------|-------------|
| Early Secret | bytes[32] | Root of the key derivation tree (derived from PSK or zero) |
| Handshake Secret | bytes[32] | Derived from Early Secret and ECDHE shared secret |
| Master Secret | bytes[32] | Final secret derived from Handshake Secret |
| Client Handshake Traffic Secret | bytes[32] | Protects client-to-server handshake messages |
| Server Handshake Traffic Secret | bytes[32] | Protects server-to-client handshake messages |
| Client Application Traffic Secret | bytes[32] | Protects client-to-server application data |
| Server Application Traffic Secret | bytes[32] | Protects server-to-client application data |

**Traffic Key Material**

Each traffic secret gets expanded into the actual cryptographic material used for AEAD encryption:

| Field | Type | Description |
|-------|------|-------------|
| Traffic Key | bytes[16] | AES-128 key for encryption/decryption (for AES-128-GCM) |
| Traffic IV | bytes[12] | Base IV for nonce construction (for AES-128-GCM) |

The relationship between traffic secrets and traffic keys follows this pattern:
1. Traffic Secret (32 bytes) serves as the root key material
2. Traffic Key is derived by HKDF-Expand-Label(Traffic Secret, "key", "", key_length)
3. Traffic IV is derived by HKDF-Expand-Label(Traffic Secret, "iv", "", iv_length)

**Sequence Number Management**

AEAD ciphers require unique nonces for each encryption operation to maintain security. TLS 1.3 constructs these nonces by XORing the traffic IV with the sequence number:

| Field | Type | Description |
|-------|------|-------------|
| Read Sequence Number | uint64 | Sequence number for incoming records (starts at 0) |
| Write Sequence Number | uint64 | Sequence number for outgoing records (starts at 0) |

The sequence numbers increment independently for reading and writing, and they reset when traffic keys are updated. The nonce construction algorithm works as follows:
1. Convert the sequence number to a 12-byte big-endian integer (padded with leading zeros)
2. XOR this 12-byte value with the 12-byte traffic IV
3. Use the result as the AEAD nonce for encryption/decryption

**Cipher Suite Configuration**

The negotiated cipher suite determines the specific cryptographic algorithms used throughout the connection:

| Field | Type | Description |
|-------|------|-------------|
| Key Exchange | enum | Key exchange method (always ECDHE for TLS 1.3) |
| Authentication | enum | Authentication method (RSA, ECDSA, EdDSA) |
| Encryption | enum | Symmetric encryption algorithm (AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305) |
| Hash | enum | Hash function for HKDF and signatures (SHA-256, SHA-384) |

For `TLS_AES_128_GCM_SHA256`, this translates to:
- Key Exchange: ECDHE (Elliptic Curve Diffie-Hellman Ephemeral)
- Authentication: Determined by server certificate (RSA, ECDSA, or EdDSA)
- Encryption: AES-128 in Galois/Counter Mode (128-bit key, 96-bit IV)
- Hash: SHA-256 for all hash operations

⚠️ **Pitfall: Key Material Lifetime**
Traffic keys must be securely erased from memory when no longer needed. Implementations should zero out key material immediately after deriving new keys or closing connections to prevent key recovery from memory dumps.

⚠️ **Pitfall: Sequence Number Overflow**
AEAD sequence numbers must never repeat for the same key. TLS implementations must either terminate connections or rotate keys before the 64-bit sequence number overflows. For AES-GCM, the practical limit is 2^32 records due to security considerations.

**State Machine Integration**

The cryptographic state closely integrates with the TLS handshake state machine. Different connection states have access to different key material:

| Connection State | Available Keys | Encryption Capability |
|------------------|----------------|----------------------|
| START | None | No encryption |
| WAIT_SH | None | No encryption |
| WAIT_EE | Handshake traffic keys | Can encrypt/decrypt handshake messages |
| WAIT_CERT_CR | Handshake traffic keys | Can encrypt/decrypt handshake messages |
| WAIT_CV | Handshake traffic keys | Can encrypt/decrypt handshake messages |
| WAIT_FINISHED | Handshake traffic keys | Can encrypt/decrypt handshake messages |
| CONNECTED | Application traffic keys | Can encrypt/decrypt application data |

This state-dependent key availability ensures that encryption operations only occur with appropriate key material and that handshake protection remains separate from application data protection.

![TLS Data Structure Relationships](./diagrams/data-structures.svg)

### Implementation Guidance

The data model implementation requires careful attention to binary parsing, memory management, and cryptographic material handling. Python provides excellent libraries for these tasks while maintaining readability for learning purposes.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Binary Parsing | `struct` module with manual parsing | Custom binary parser class with error handling |
| Random Generation | `secrets.token_bytes()` for cryptographic randomness | Hardware RNG integration with `secrets` fallback |
| Cryptographic Operations | `cryptography` library for HKDF and AEAD | Low-level `cryptography.hazmat` for maximum control |
| State Management | Simple Python classes with `@dataclass` | State machine library with formal validation |

**Recommended File Structure:**
```
https_client/
  data_model/
    __init__.py              ← exports all data structures
    tls_record.py            ← TLS record format and parsing
    handshake_messages.py    ← handshake message structures
    crypto_state.py          ← cryptographic state management
    constants.py             ← TLS constants and enums
    binary_parser.py         ← binary parsing utilities
  tests/
    test_data_model.py       ← unit tests for data structures
```

**TLS Constants and Enums (constants.py):**
```python
"""TLS protocol constants and enumeration values."""

# Content Types (RFC 8446 Section 6.1)
class ContentType:
    CHANGE_CIPHER_SPEC = 20
    ALERT = 21
    HANDSHAKE = 22
    APPLICATION_DATA = 23

# TLS Versions
class TLSVersion:
    TLS_1_2 = 0x0303  # Used for legacy compatibility
    TLS_1_3 = 0x0304  # Actual TLS 1.3 version

# Handshake Message Types (RFC 8446 Section 4)
class HandshakeType:
    CLIENT_HELLO = 1
    SERVER_HELLO = 2
    NEW_SESSION_TICKET = 4
    END_OF_EARLY_DATA = 5
    ENCRYPTED_EXTENSIONS = 8
    CERTIFICATE = 11
    CERTIFICATE_REQUEST = 13
    CERTIFICATE_VERIFY = 15
    FINISHED = 20
    KEY_UPDATE = 24
    MESSAGE_HASH = 254

# Cipher Suites (RFC 8446 Appendix B.4)
class CipherSuite:
    TLS_AES_128_GCM_SHA256 = 0x1301
    TLS_AES_256_GCM_SHA384 = 0x1302
    TLS_CHACHA20_POLY1305_SHA256 = 0x1303

# Extension Types (IANA TLS ExtensionType Values)
class ExtensionType:
    SERVER_NAME = 0
    MAX_FRAGMENT_LENGTH = 1
    STATUS_REQUEST = 5
    SUPPORTED_GROUPS = 10
    SIGNATURE_ALGORITHMS = 13
    USE_SRTP = 14
    HEARTBEAT = 15
    APPLICATION_LAYER_PROTOCOL_NEGOTIATION = 16
    SIGNED_CERTIFICATE_TIMESTAMP = 18
    CLIENT_CERTIFICATE_TYPE = 19
    SERVER_CERTIFICATE_TYPE = 20
    PADDING = 21
    PRE_SHARED_KEY = 41
    EARLY_DATA = 42
    SUPPORTED_VERSIONS = 43
    COOKIE = 44
    PSK_KEY_EXCHANGE_MODES = 45
    CERTIFICATE_AUTHORITIES = 47
    OID_FILTERS = 48
    POST_HANDSHAKE_AUTH = 49
    SIGNATURE_ALGORITHMS_CERT = 50
    KEY_SHARE = 51

# Named Groups for ECDHE (RFC 8446 Section 4.2.7)
class NamedGroup:
    SECP256R1 = 0x0017
    SECP384R1 = 0x0018
    SECP521R1 = 0x0019
    X25519 = 0x001D
    X448 = 0x001E

# Alert Levels and Descriptions (RFC 8446 Section 6)
class AlertLevel:
    WARNING = 1
    FATAL = 2

class AlertDescription:
    CLOSE_NOTIFY = 0
    UNEXPECTED_MESSAGE = 10
    BAD_RECORD_MAC = 20
    DECRYPTION_FAILED = 21
    RECORD_OVERFLOW = 22
    DECOMPRESSION_FAILURE = 30
    HANDSHAKE_FAILURE = 40
    NO_CERTIFICATE = 41
    BAD_CERTIFICATE = 42
    UNSUPPORTED_CERTIFICATE = 43
    CERTIFICATE_REVOKED = 44
    CERTIFICATE_EXPIRED = 45
    CERTIFICATE_UNKNOWN = 46
    ILLEGAL_PARAMETER = 47
    UNKNOWN_CA = 48
    ACCESS_DENIED = 49
    DECODE_ERROR = 50
    DECRYPT_ERROR = 51
    EXPORT_RESTRICTION = 60
    PROTOCOL_VERSION = 70
    INSUFFICIENT_SECURITY = 71
    INTERNAL_ERROR = 80
    USER_CANCELED = 90
    NO_RENEGOTIATION = 100
    MISSING_EXTENSION = 109
```

**Binary Parsing Utilities (binary_parser.py):**
```python
"""Binary parsing utilities for TLS protocol messages."""

import struct
from typing import Tuple


class BinaryParser:
    """Static utility methods for parsing binary TLS data."""
    
    @staticmethod
    def parse_uint8(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse single byte as unsigned 8-bit integer.
        
        Args:
            data: Byte buffer to parse from
            offset: Starting position in buffer
            
        Returns:
            Tuple of (parsed_value, new_offset)
            
        Raises:
            ValueError: If not enough data available
        """
        if len(data) < offset + 1:
            raise ValueError(f"Not enough data for uint8 at offset {offset}")
        
        value = struct.unpack_from('!B', data, offset)[0]
        return value, offset + 1
    
    @staticmethod
    def parse_uint16(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse 2 bytes as unsigned 16-bit integer in network byte order.
        
        Args:
            data: Byte buffer to parse from
            offset: Starting position in buffer
            
        Returns:
            Tuple of (parsed_value, new_offset)
            
        Raises:
            ValueError: If not enough data available
        """
        if len(data) < offset + 2:
            raise ValueError(f"Not enough data for uint16 at offset {offset}")
        
        value = struct.unpack_from('!H', data, offset)[0]
        return value, offset + 2
    
    @staticmethod
    def parse_uint24(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse 3 bytes as unsigned 24-bit integer in network byte order.
        
        Args:
            data: Byte buffer to parse from
            offset: Starting position in buffer
            
        Returns:
            Tuple of (parsed_value, new_offset)
            
        Raises:
            ValueError: If not enough data available
        """
        if len(data) < offset + 3:
            raise ValueError(f"Not enough data for uint24 at offset {offset}")
        
        # Parse as 4-byte integer with leading zero byte
        padded = b'\x00' + data[offset:offset+3]
        value = struct.unpack('!I', padded)[0]
        return value, offset + 3
    
    @staticmethod
    def parse_variable_bytes(data: bytes, offset: int, length_bytes: int) -> Tuple[bytes, int]:
        """Parse variable-length byte array with length prefix.
        
        Args:
            data: Byte buffer to parse from
            offset: Starting position in buffer
            length_bytes: Number of bytes used for length field (1, 2, or 3)
            
        Returns:
            Tuple of (parsed_bytes, new_offset)
            
        Raises:
            ValueError: If not enough data available or invalid length_bytes
        """
        if length_bytes == 1:
            length, new_offset = BinaryParser.parse_uint8(data, offset)
        elif length_bytes == 2:
            length, new_offset = BinaryParser.parse_uint16(data, offset)
        elif length_bytes == 3:
            length, new_offset = BinaryParser.parse_uint24(data, offset)
        else:
            raise ValueError(f"Invalid length_bytes: {length_bytes}")
        
        if len(data) < new_offset + length:
            raise ValueError(f"Not enough data for {length} bytes at offset {new_offset}")
        
        value = data[new_offset:new_offset + length]
        return value, new_offset + length
    
    @staticmethod
    def encode_uint8(value: int) -> bytes:
        """Encode unsigned 8-bit integer as bytes."""
        return struct.pack('!B', value)
    
    @staticmethod
    def encode_uint16(value: int) -> bytes:
        """Encode unsigned 16-bit integer as bytes in network byte order."""
        return struct.pack('!H', value)
    
    @staticmethod
    def encode_uint24(value: int) -> bytes:
        """Encode unsigned 24-bit integer as bytes in network byte order."""
        if value >= 2**24:
            raise ValueError(f"Value {value} too large for 24-bit integer")
        
        # Encode as 4-byte integer and take last 3 bytes
        encoded = struct.pack('!I', value)
        return encoded[1:]  # Skip first byte
    
    @staticmethod
    def encode_variable_bytes(value: bytes, length_bytes: int) -> bytes:
        """Encode variable-length byte array with length prefix."""
        length = len(value)
        
        if length_bytes == 1:
            if length >= 256:
                raise ValueError(f"Length {length} too large for 1-byte length field")
            return BinaryParser.encode_uint8(length) + value
        elif length_bytes == 2:
            if length >= 65536:
                raise ValueError(f"Length {length} too large for 2-byte length field")
            return BinaryParser.encode_uint16(length) + value
        elif length_bytes == 3:
            if length >= 16777216:
                raise ValueError(f"Length {length} too large for 3-byte length field")
            return BinaryParser.encode_uint24(length) + value
        else:
            raise ValueError(f"Invalid length_bytes: {length_bytes}")
    
    @staticmethod
    def hex_dump(data: bytes, width: int = 16, show_ascii: bool = True) -> str:
        """Create hex dump representation of binary data.
        
        Args:
            data: Binary data to dump
            width: Number of bytes per line
            show_ascii: Whether to show ASCII representation
            
        Returns:
            Formatted hex dump string
        """
        lines = []
        for i in range(0, len(data), width):
            chunk = data[i:i + width]
            hex_part = ' '.join(f'{b:02x}' for b in chunk)
            hex_part = hex_part.ljust(width * 3 - 1)  # Pad to consistent width
            
            if show_ascii:
                ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
                lines.append(f'{i:08x}: {hex_part} |{ascii_part}|')
            else:
                lines.append(f'{i:08x}: {hex_part}')
        
        return '\n'.join(lines)
```

**TLS Record Structure (skeleton for tls_record.py):**
```python
"""TLS record layer data structures and parsing."""

from dataclasses import dataclass
from typing import Optional, List
from .constants import ContentType
from .binary_parser import BinaryParser


@dataclass
class TLSRecord:
    """Represents a single TLS record with header and payload."""
    content_type: int
    legacy_version: int
    length: int
    payload: bytes
    
    def to_bytes(self) -> bytes:
        """Serialize TLS record to binary format.
        
        Returns:
            Binary representation of the TLS record
        """
        # TODO 1: Encode content_type as 1-byte unsigned integer
        # TODO 2: Encode legacy_version as 2-byte unsigned integer in network byte order  
        # TODO 3: Encode length as 2-byte unsigned integer in network byte order
        # TODO 4: Concatenate header bytes with payload bytes
        # TODO 5: Validate that payload length matches length field
        pass
    
    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> tuple['TLSRecord', int]:
        """Parse TLS record from binary data.
        
        Args:
            data: Binary data containing TLS record
            offset: Starting position in data
            
        Returns:
            Tuple of (parsed_record, new_offset)
        """
        # TODO 1: Parse content_type using BinaryParser.parse_uint8()
        # TODO 2: Parse legacy_version using BinaryParser.parse_uint16()
        # TODO 3: Parse length using BinaryParser.parse_uint16()
        # TODO 4: Validate length is within acceptable range (0-16384)
        # TODO 5: Extract payload bytes using parsed length
        # TODO 6: Create and return TLSRecord instance
        pass
    
    def is_handshake(self) -> bool:
        """Check if this record contains handshake data."""
        return self.content_type == ContentType.HANDSHAKE
    
    def is_application_data(self) -> bool:
        """Check if this record contains application data."""
        return self.content_type == ContentType.APPLICATION_DATA
    
    def is_alert(self) -> bool:
        """Check if this record contains alert data."""
        return self.content_type == ContentType.ALERT


class RecordLayer:
    """Handles TLS record parsing, fragmentation, and message classification."""
    
    def __init__(self):
        self._reassembly_buffer = bytearray()
        self._expected_length: Optional[int] = None
    
    def parse_records(self, data: bytes) -> List[TLSRecord]:
        """Parse one or more TLS records from binary data.
        
        Args:
            data: Binary data potentially containing multiple TLS records
            
        Returns:
            List of parsed TLS records
        """
        # TODO 1: Add incoming data to reassembly buffer
        # TODO 2: Try to parse complete records from buffer
        # TODO 3: Handle partial records that span multiple TCP segments
        # TODO 4: Remove parsed record data from buffer
        # TODO 5: Return list of complete records
        # Hint: Use TLSRecord.from_bytes() in a loop, handling incomplete data gracefully
        pass
    
    def fragment_large_message(self, content_type: int, message: bytes) -> List[TLSRecord]:
        """Fragment large message across multiple TLS records if needed.
        
        Args:
            content_type: TLS content type for the records
            message: Message data to fragment
            
        Returns:
            List of TLS records containing the fragmented message
        """
        # TODO 1: Check if message fits in single record (≤ 16384 bytes)
        # TODO 2: If single record, create one TLSRecord and return
        # TODO 3: If fragmentation needed, split message into 16KB chunks
        # TODO 4: Create TLSRecord for each chunk with same content_type
        # TODO 5: Return list of fragmented records
        pass
```

**Handshake Message Structures (skeleton for handshake_messages.py):**
```python
"""TLS handshake message data structures."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import secrets
from .constants import HandshakeType, TLSVersion, CipherSuite, ExtensionType
from .binary_parser import BinaryParser


@dataclass
class Extension:
    """Base class for TLS extensions."""
    extension_type: int
    extension_data: bytes
    
    def to_bytes(self) -> bytes:
        """Serialize extension to binary format."""
        # TODO 1: Encode extension_type as 2-byte unsigned integer
        # TODO 2: Encode length of extension_data as 2-byte unsigned integer
        # TODO 3: Concatenate type, length, and data
        pass


@dataclass
class ClientHello:
    """TLS ClientHello handshake message."""
    legacy_version: int = TLSVersion.TLS_1_2
    random: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    legacy_session_id: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    cipher_suites: List[int] = field(default_factory=lambda: [CipherSuite.TLS_AES_128_GCM_SHA256])
    legacy_compression_methods: List[int] = field(default_factory=lambda: [0])  # Null compression
    extensions: List[Extension] = field(default_factory=list)
    
    def add_server_name_extension(self, hostname: str) -> None:
        """Add Server Name Indication (SNI) extension.
        
        Args:
            hostname: Server hostname for SNI
        """
        # TODO 1: Encode hostname as UTF-8 bytes
        # TODO 2: Create server name list with name_type=0 (hostname) and hostname bytes
        # TODO 3: Encode server name list with proper length prefixes
        # TODO 4: Create Extension with type=SERVER_NAME and encoded data
        # TODO 5: Add extension to self.extensions list
        pass
    
    def add_supported_versions_extension(self, versions: List[int]) -> None:
        """Add Supported Versions extension.
        
        Args:
            versions: List of supported TLS versions
        """
        # TODO 1: Create version list with 1-byte length prefix
        # TODO 2: Encode each version as 2-byte unsigned integer
        # TODO 3: Create Extension with type=SUPPORTED_VERSIONS
        # TODO 4: Add extension to self.extensions list
        pass
    
    def add_key_share_extension(self, key_shares: List[Tuple[int, bytes]]) -> None:
        """Add Key Share extension with client's ECDHE key shares.
        
        Args:
            key_shares: List of (named_group, key_exchange_data) tuples
        """
        # TODO 1: Calculate total length of all key share entries
        # TODO 2: Encode key share list length as 2-byte unsigned integer
        # TODO 3: For each key share: encode named_group (2 bytes), key length (2 bytes), key data
        # TODO 4: Create Extension with type=KEY_SHARE and encoded key share list
        # TODO 5: Add extension to self.extensions list
        pass
    
    def to_bytes(self) -> bytes:
        """Serialize ClientHello to binary handshake message format."""
        # TODO 1: Encode legacy_version as 2-byte unsigned integer
        # TODO 2: Add 32-byte random field
        # TODO 3: Encode legacy_session_id with 1-byte length prefix
        # TODO 4: Encode cipher_suites with 2-byte length prefix
        # TODO 5: Encode legacy_compression_methods with 1-byte length prefix  
        # TODO 6: Serialize all extensions and encode with 2-byte total length
        # TODO 7: Concatenate all fields to create complete ClientHello payload
        # TODO 8: Prepend handshake header: message_type=CLIENT_HELLO, length (3 bytes)
        pass


@dataclass  
class ServerHello:
    """TLS ServerHello handshake message."""
    legacy_version: int
    random: bytes
    legacy_session_id_echo: bytes
    cipher_suite: int
    legacy_compression_method: int
    extensions: List[Extension]
    
    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple['ServerHello', int]:
        """Parse ServerHello from binary handshake message.
        
        Args:
            data: Binary data containing ServerHello message
            offset: Starting position in data
            
        Returns:
            Tuple of (parsed_serverhello, new_offset)
        """
        # TODO 1: Parse handshake message header (type=SERVER_HELLO, length)
        # TODO 2: Parse legacy_version using BinaryParser.parse_uint16()
        # TODO 3: Extract 32-byte random field
        # TODO 4: Parse legacy_session_id_echo with 1-byte length prefix
        # TODO 5: Parse cipher_suite using BinaryParser.parse_uint16()
        # TODO 6: Parse legacy_compression_method using BinaryParser.parse_uint8()
        # TODO 7: Parse extensions list with 2-byte total length prefix
        # TODO 8: Create and return ServerHello instance
        pass
    
    def get_extension(self, extension_type: int) -> Optional[Extension]:
        """Get extension by type, or None if not present."""
        for ext in self.extensions:
            if ext.extension_type == extension_type:
                return ext
        return None
```

**Cryptographic State Management (skeleton for crypto_state.py):**
```python
"""TLS cryptographic state management."""

from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum


class ConnectionState(Enum):
    """TLS connection states for handshake state machine."""
    START = "start"
    WAIT_SH = "wait_server_hello"  
    WAIT_EE = "wait_encrypted_extensions"
    WAIT_CERT_CR = "wait_cert_cert_request"
    WAIT_CV = "wait_cert_verify"
    WAIT_FINISHED = "wait_finished"
    CONNECTED = "connected"
    CLOSED = "closed"


@dataclass
class TrafficKeys:
    """Traffic key material for AEAD encryption."""
    key: bytes          # 16 bytes for AES-128-GCM
    iv: bytes           # 12 bytes for AES-128-GCM
    
    def derive_nonce(self, sequence_number: int) -> bytes:
        """Derive AEAD nonce from IV and sequence number.
        
        Args:
            sequence_number: Record sequence number
            
        Returns:
            12-byte nonce for AEAD encryption
        """
        # TODO 1: Convert sequence_number to 12-byte big-endian integer (pad with leading zeros)
        # TODO 2: XOR the sequence number bytes with the IV bytes
        # TODO 3: Return resulting 12-byte nonce
        pass


@dataclass
class KeyScheduleState:
    """TLS 1.3 key derivation state using HKDF."""
    early_secret: Optional[bytes] = None
    handshake_secret: Optional[bytes] = None  
    master_secret: Optional[bytes] = None
    client_handshake_traffic_secret: Optional[bytes] = None
    server_handshake_traffic_secret: Optional[bytes] = None
    client_application_traffic_secret: Optional[bytes] = None
    server_application_traffic_secret: Optional[bytes] = None
    
    def derive_handshake_secrets(self, ecdhe_shared_secret: bytes) -> None:
        """Derive handshake traffic secrets from ECDHE shared secret.
        
        Args:
            ecdhe_shared_secret: Shared secret from ECDHE key exchange
        """
        # TODO 1: If early_secret is None, derive it from zero salt using HKDF-Extract
        # TODO 2: Derive handshake_secret using HKDF-Extract(early_secret, ecdhe_shared_secret)
        # TODO 3: Create handshake context hash from ClientHello and ServerHello messages
        # TODO 4: Derive client_handshake_traffic_secret using HKDF-Expand-Label
        # TODO 5: Derive server_handshake_traffic_secret using HKDF-Expand-Label
        # Hint: Use cryptography library's HKDF implementation
        pass
    
    def derive_application_secrets(self, handshake_context: bytes) -> None:
        """Derive application traffic secrets from handshake context.
        
        Args:
            handshake_context: Hash of complete handshake transcript
        """
        # TODO 1: Derive master_secret using HKDF-Extract(handshake_secret, zero_salt)
        # TODO 2: Derive client_application_traffic_secret using HKDF-Expand-Label
        # TODO 3: Derive server_application_traffic_secret using HKDF-Expand-Label
        # TODO 4: Securely erase handshake secrets (zero out bytes)
        pass


@dataclass
class CryptographicState:
    """Complete TLS connection cryptographic state."""
    connection_state: ConnectionState = ConnectionState.START
    tls_version: int = 0x0304  # TLS 1.3
    cipher_suite: int = 0x1301  # TLS_AES_128_GCM_SHA256
    client_random: Optional[bytes] = None
    server_random: Optional[bytes] = None
    handshake_context: bytearray = None
    key_schedule: KeyScheduleState = None
    read_sequence_number: int = 0
    write_sequence_number: int = 0
    client_handshake_keys: Optional[TrafficKeys] = None
    server_handshake_keys: Optional[TrafficKeys] = None
    client_application_keys: Optional[TrafficKeys] = None
    server_application_keys: Optional[TrafficKeys] = None
    
    def __post_init__(self):
        if self.handshake_context is None:
            self.handshake_context = bytearray()
        if self.key_schedule is None:
            self.key_schedule = KeyScheduleState()
    
    def update_handshake_context(self, message: bytes) -> None:
        """Add handshake message to running transcript hash.
        
        Args:
            message: Complete handshake message including header
        """
        self.handshake_context.extend(message)
    
    def get_current_handshake_context_hash(self) -> bytes:
        """Get SHA-256 hash of current handshake transcript."""
        # TODO 1: Import hashlib for SHA-256
        # TODO 2: Create SHA-256 hasher
        # TODO 3: Hash all bytes in handshake_context
        # TODO 4: Return digest bytes
        pass
    
    def can_encrypt_handshake(self) -> bool:
        """Check if handshake encryption is available."""
        return (self.connection_state.value in ['wait_cert_cert_request', 'wait_cert_verify', 'wait_finished'] and
                self.client_handshake_keys is not None and 
                self.server_handshake_keys is not None)
    
    def can_encrypt_application_data(self) -> bool:
        """Check if application data encryption is available."""
        return (self.connection_state == ConnectionState.CONNECTED and
                self.client_application_keys is not None and
                self.server_application_keys is not None)
```

**Milestone Checkpoints:**

After implementing the data model structures, verify correct behavior with these checkpoints:

1. **Binary Parsing Validation:**
   - Create test TLS records with known binary representations
   - Verify parsing and encoding round-trip produces identical data
   - Test edge cases like maximum length records (16KB) and minimum length records

2. **ClientHello Construction:**
   - Generate ClientHello with required extensions (SNI, Supported Versions, Key Share)
   - Verify random field contains 32 cryptographically secure bytes
   - Check that extension ordering matches common implementations

3. **Cryptographic State Management:**  
   - Initialize CryptographicState and verify default values
   - Test state transitions match TLS 1.3 handshake flow
   - Validate that key material is properly isolated between handshake and application phases

4. **Error Handling:**
   - Test parsing with insufficient data (should raise appropriate exceptions)
   - Verify length field validation prevents buffer overflows
   - Confirm that invalid extension types are handled gracefully

**Common Implementation Issues:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|----------------|-----|
| "struct.error: unpack requires a buffer" | Insufficient data for parsing | Check data length before parsing | Add bounds checking in parse methods |
| Extension parsing fails | Incorrect length calculation | Print hex dump of extension data | Recalculate extension lengths programmatically |  
| Random bytes are predictable | Using `random` instead of `secrets` | Check random generation code | Use `secrets.token_bytes()` for cryptographic randomness |
| Handshake context hash mismatches | Missing messages in transcript | Log all messages added to context | Ensure all handshake messages are included in order |


## TCP Transport and Record Layer

> **Milestone(s):** Milestone 1 — TCP Socket & Record Layer

Think of the TCP transport and TLS record layer as the foundation and framing system of a secure building. The TCP connection is like the concrete foundation — it provides a reliable, ordered stream of bytes between two points, but offers no privacy or security. The TLS record layer is like the security framework built on top of that foundation — it takes the raw byte stream and organizes it into classified, structured messages that can later be encrypted and authenticated. Just as a building's framing system divides space into rooms with specific purposes, the TLS record layer divides the byte stream into different types of messages (handshake negotiations, alerts, application data) while handling the practical concerns of message boundaries and fragmentation.

The TCP transport layer serves as the reliable delivery mechanism for all TLS communication. Unlike UDP, which provides no delivery guarantees, TCP ensures that bytes arrive in order and without loss, making it the ideal foundation for the complex, stateful TLS protocol. However, TCP's stream-oriented nature creates challenges — it delivers bytes as they arrive from the network, which may not align with TLS message boundaries. A single TLS record might arrive split across multiple TCP segments, or multiple small TLS records might arrive bundled in a single TCP segment.

The TLS record layer addresses these challenges by implementing a structured framing protocol on top of TCP's byte stream. Every TLS message, whether it's a handshake message, alert, or application data, is wrapped in a standardized record format that includes the message type, protocol version, length, and payload. This framing allows the receiving side to parse incoming bytes into discrete messages and route them to the appropriate handler based on their type.

![TLS Record Layer Processing Flow](./diagrams/record-layer-processing.svg)

### TCP Connection Management

The `TCPTransport` component encapsulates all TCP socket operations and provides a clean interface for the higher-level TLS components. Think of it as the postal service for your secure communication — it knows how to establish connections to remote servers, send bytes reliably, and receive bytes as they arrive from the network, but it doesn't understand or care about the content of those bytes.

**TCP connection establishment** follows the standard three-way handshake process, but with specific considerations for TLS usage. The client creates a socket, configures appropriate timeout values, and connects to the remote server on port 443 (the standard HTTPS port). The connection process must handle various failure modes including DNS resolution failures, network unreachability, connection timeouts, and connection refused errors.

> **Decision: Synchronous TCP Operations**
> - **Context**: Network operations can be implemented synchronously (blocking) or asynchronously (non-blocking with callbacks/promises)
> - **Options Considered**:
>   - Synchronous blocking sockets with timeouts
>   - Asynchronous non-blocking sockets with event loops
>   - Thread-per-connection with blocking sockets
> - **Decision**: Synchronous blocking sockets with configurable timeouts
> - **Rationale**: Simplifies the implementation significantly while still allowing timeout control. TLS handshake is inherently sequential, reducing benefits of async. Learning focus is on TLS protocol, not async programming patterns.
> - **Consequences**: Enables simpler error handling and debugging. Limits concurrency to one connection per thread. Sufficient for educational implementation.

The socket configuration requires several important settings for reliable TLS operation:

| Configuration Option | Value | Purpose |
|---------------------|--------|---------|
| Socket Family | `AF_INET` or `AF_INET6` | IPv4 or IPv6 connectivity |
| Socket Type | `SOCK_STREAM` | Reliable, ordered byte stream |
| TCP_NODELAY | `True` | Disable Nagle's algorithm for low latency |
| SO_REUSEADDR | `True` | Allow socket reuse for testing |
| Connect Timeout | 10-30 seconds | Prevent indefinite connection attempts |
| Send Timeout | 5-10 seconds | Prevent indefinite send blocking |
| Receive Timeout | 5-10 seconds | Prevent indefinite receive blocking |

**Byte transmission** through the TCP transport involves careful handling of partial sends and receives. TCP sockets may accept fewer bytes than requested in a send operation, requiring the sender to track how many bytes were actually transmitted and retry sending the remaining bytes. Similarly, receive operations may return fewer bytes than requested, even when more data is available.

The `send_bytes` method implements a loop that continues sending until all requested bytes have been transmitted:

1. Attempt to send the remaining bytes using the socket's send operation
2. Check the return value to determine how many bytes were actually sent
3. If fewer bytes were sent than requested, adjust the data pointer and length
4. Repeat until all bytes have been transmitted or an error occurs
5. Handle EAGAIN/EWOULDBLOCK errors by retrying the send operation
6. Return the total number of bytes successfully transmitted

**Byte reception** is more complex because the application must handle both exact-length reads and variable-length reads. The `receive_bytes` method performs a single receive operation and returns whatever data is immediately available, up to the maximum requested length. This is used for initial record header parsing where the application needs to peek at the incoming data.

The `receive_exact` method implements a different pattern — it continues reading until exactly the specified number of bytes have been received:

1. Initialize a buffer to accumulate received bytes
2. Calculate the number of bytes still needed to reach the target length
3. Attempt to receive the remaining bytes from the socket
4. Append any received data to the accumulation buffer
5. If the socket returns zero bytes, the connection has been closed by the peer
6. If fewer bytes are received than needed, adjust the target and continue reading
7. Return the complete buffer only when exactly the requested number of bytes have been received

> The critical insight here is that TCP provides a byte stream, not a message stream. The application must implement its own message framing and boundary detection on top of TCP's byte delivery guarantees.

**Error handling** in the TCP transport layer must distinguish between recoverable and fatal errors. Network operations can fail in various ways, and the appropriate response depends on the type of failure:

| Error Type | Examples | Recovery Strategy |
|------------|----------|------------------|
| Temporary Network | EAGAIN, EWOULDBLOCK | Retry the operation after a brief delay |
| Connection Lost | ECONNRESET, EPIPE | Close socket and notify higher layers |
| Timeout | ETIMEDOUT | Close socket and report timeout to application |
| Host Unreachable | EHOSTUNREACH, ENETUNREACH | Immediate failure, no retry |
| DNS Failure | Name resolution errors | Immediate failure, possible retry with different hostname |

Connection lifecycle management involves tracking the socket state and providing clean shutdown procedures. The transport maintains a connection state that progresses through distinct phases:

1. **Disconnected**: No socket exists, ready to establish new connection
2. **Connecting**: Socket created, connection attempt in progress
3. **Connected**: Three-way handshake complete, ready for data transfer
4. **Closing**: Close initiated, waiting for graceful shutdown
5. **Closed**: Socket closed, resources released

### TLS Record Layer Implementation

The TLS record layer transforms TCP's unstructured byte stream into classified, length-delimited messages that higher layers can process independently. Think of it as a sophisticated mail sorting facility — raw bytes arrive continuously from the TCP connection, and the record layer organizes them into individual messages with clear type labels, length information, and payload boundaries.

**TLS record structure** follows a fixed format that enables efficient parsing and message classification. Every TLS record begins with a five-byte header containing the content type, protocol version, and payload length, followed by the actual message payload:

| Field | Size (bytes) | Description |
|-------|--------------|-------------|
| `content_type` | 1 | Message classification (handshake, alert, application data) |
| `legacy_version` | 2 | TLS version (0x0303 for TLS 1.2 compatibility) |
| `length` | 2 | Payload length in bytes (maximum 16,384) |
| `payload` | variable | Actual message content |

The record format uses network byte order (big-endian) for all multi-byte fields, ensuring consistent interpretation across different computer architectures. The length field limits individual records to 16,384 bytes (16 KB), which prevents excessive memory allocation and enables efficient buffering strategies.

**Content type classification** routes incoming records to the appropriate processing components based on their purpose within the TLS protocol:

| Content Type | Value | Purpose | Handler |
|--------------|--------|---------|---------|
| `CHANGE_CIPHER_SPEC` | 20 | Legacy cipher change notification | Compatibility handler |
| `ALERT` | 21 | Error conditions and warnings | Alert processor |
| `HANDSHAKE` | 22 | Negotiation and key exchange | Handshake engine |
| `APPLICATION_DATA` | 23 | Encrypted application payload | Application data handler |

During the initial handshake phase, most traffic consists of handshake records containing negotiation messages. Once the handshake completes and encryption is established, application data records carry the encrypted HTTP requests and responses. Alert records can arrive at any time to signal error conditions or normal connection closure.

**Record parsing** implements a multi-stage process that handles the inherent mismatch between TCP's stream delivery and TLS's record boundaries. The parser maintains an internal buffer to accumulate bytes and implements a state machine to track parsing progress:

1. **Header Collection**: Accumulate bytes until a complete 5-byte record header is available
2. **Header Validation**: Parse content type, version, and length fields from the header
3. **Length Validation**: Verify the payload length is within acceptable bounds (1 to 16,384 bytes)
4. **Payload Collection**: Accumulate additional bytes until the complete payload is available
5. **Record Assembly**: Combine header and payload into a complete `TLSRecord` structure
6. **Message Classification**: Route the completed record to the appropriate handler based on content type

The parsing state machine handles partial data gracefully — if only part of a record header or payload has arrived, the parser maintains its current state and resumes processing when additional bytes become available.

> **Decision: Buffer Management Strategy**
> - **Context**: Record parsing requires buffering partial data between TCP receives, with trade-offs between memory usage and parsing efficiency
> - **Options Considered**:
>   - Single growing buffer that accumulates all data
>   - Fixed-size circular buffer with overflow handling
>   - Separate header and payload buffers
> - **Decision**: Separate header and payload buffers with size limits
> - **Rationale**: Header buffer can be small (5 bytes) and reused. Payload buffer allocated based on record length field prevents unbounded memory growth. Separation enables early record classification.
> - **Consequences**: Enables efficient memory usage and early error detection. Requires careful buffer state management. Provides clear separation of concerns.

**Fragmentation handling** addresses the reality that large TLS records may be split across multiple TCP segments, while small records may be bundled together in a single TCP receive operation. The record layer must reassemble fragmented records and separate bundled records without losing data or message boundaries.

For fragmented records, the parser implements a continuation mechanism:

1. Parse the record header to determine the expected payload length
2. Allocate a payload buffer of exactly the required size
3. Copy payload bytes from each TCP segment as they arrive
4. Track the number of payload bytes received versus the expected length
5. Continue reading from TCP until the payload buffer is completely filled
6. Process the complete record only when all payload bytes have been received

For bundled records, the parser processes available bytes in a loop:

1. Check if enough bytes are available for a complete record header
2. Parse the header and determine the payload length requirement
3. Check if enough additional bytes are available for the complete payload
4. If a complete record is available, parse it and add it to the output queue
5. Advance the buffer position past the processed record
6. Repeat until insufficient bytes remain for another complete record
7. Preserve any remaining bytes for the next parsing iteration

**Message classification and routing** directs completed records to the appropriate handler based on their content type and the current connection state. During the handshake phase, only handshake and alert records are valid — application data records would indicate a protocol violation. After handshake completion, both application data and alert records are acceptable.

The record layer maintains a dispatch table that maps content types to handler functions:

| Content Type | Handler Function | Valid States |
|--------------|------------------|--------------|
| `HANDSHAKE` | `process_handshake_record` | All states except CLOSED |
| `ALERT` | `process_alert_record` | All states except CLOSED |
| `APPLICATION_DATA` | `process_application_data` | CONNECTED only |
| `CHANGE_CIPHER_SPEC` | `process_legacy_change_cipher` | Handshake states only |

Invalid content types or records received in inappropriate states trigger alert generation and connection termination. This strict enforcement prevents protocol violations that could compromise security.

**Buffer management** balances memory efficiency with parsing performance. The record layer maintains several buffers for different purposes:

| Buffer Type | Size | Purpose |
|-------------|------|---------|
| Receive Buffer | 8,192 bytes | Accumulate bytes from TCP |
| Header Buffer | 5 bytes | Store partial record headers |
| Payload Buffer | Variable (up to 16,384) | Store record payloads |
| Output Queue | Multiple records | Buffer completed records for processing |

The receive buffer uses a sliding window approach — processed bytes are discarded by shifting remaining bytes to the beginning of the buffer, making space for new TCP data. This prevents unbounded buffer growth while maintaining parsing state across multiple receive operations.

### Common Pitfalls

Understanding the common mistakes in TCP transport and record layer implementation helps developers avoid subtle bugs that can cause intermittent failures or security vulnerabilities.

⚠️ **Pitfall: Assuming Complete TCP Operations**

Many developers assume that TCP send and receive operations always transfer the complete requested amount of data. In reality, both operations may transfer fewer bytes than requested, even when no error occurs.

For send operations, the TCP stack may accept only part of the data if internal buffers are full. The application must check the return value and retry sending the remaining bytes. Failing to implement this retry loop can result in truncated messages that cause parsing errors or protocol violations on the receiving side.

For receive operations, TCP may return fewer bytes than requested even when more data is available in the network buffers. This commonly occurs when data arrives in multiple IP packets or when the TCP stack optimizes buffer management. Applications must implement proper buffering to accumulate partial data until complete messages are available.

**Detection**: Messages appear truncated or parsing fails with "unexpected end of data" errors. Network traces show complete data transmission but application logs indicate incomplete reception.

**Fix**: Implement retry loops for both send and receive operations. Track bytes transferred and continue operations until the complete request is satisfied or a genuine error occurs.

⚠️ **Pitfall: Incorrect Endianness Handling**

TLS records use network byte order (big-endian) for all multi-byte fields, but many CPU architectures use little-endian byte ordering internally. Failing to convert between network and host byte order causes length fields and version numbers to be interpreted incorrectly.

This mistake often manifests as wildly incorrect record lengths (like 16 million bytes instead of 64 bytes) because the bytes are interpreted in the wrong order. The record parser may attempt to allocate enormous buffers or may reject valid records as malformed.

**Detection**: Record length fields appear unreasonably large (> 16,384 bytes). Version fields show impossible values. Memory allocation errors when attempting to allocate payload buffers.

**Fix**: Use proper byte order conversion functions for all multi-byte field parsing. Convert from network byte order to host byte order when reading, and from host byte order to network byte order when writing.

⚠️ **Pitfall: Ignoring Record Length Limits**

TLS records have a maximum payload length of 16,384 bytes (16 KB). Accepting records with larger length fields can enable denial-of-service attacks where malicious servers claim to send gigabyte-sized records, causing the client to allocate excessive memory or enter infinite read loops.

Even within the valid length range, allocating buffers based on untrusted length fields without validation can exhaust available memory if many large records arrive simultaneously.

**Detection**: Memory usage grows unexpectedly during connection establishment. Application becomes unresponsive when connecting to certain servers. Out-of-memory errors during record parsing.

**Fix**: Validate record length fields against the TLS specification limits before allocating payload buffers. Implement reasonable memory usage limits for the total amount of buffered record data.

⚠️ **Pitfall: Mixing Record Boundaries with TCP Boundaries**

TCP delivers a byte stream with no inherent message boundaries — the boundaries between TCP segments have no relationship to TLS record boundaries. A common mistake is assuming that each TCP receive operation returns exactly one TLS record, or that TLS records never span multiple TCP segments.

This assumption works in simple test scenarios where small records fit in single TCP segments, but fails in production environments with larger records, network fragmentation, or varying TCP segment sizes.

**Detection**: Parsing works correctly with small test messages but fails with larger payloads. Errors occur sporadically and seem related to network conditions or server implementations.

**Fix**: Implement proper buffering that accumulates bytes from multiple TCP receives until complete TLS records are available. Parse records from the accumulated buffer rather than directly from TCP receive operations.

⚠️ **Pitfall: Inadequate Error Context in Network Operations**

Network operations can fail in numerous ways, but many implementations provide insufficient error context to diagnose the root cause. Generic error messages like "connection failed" don't distinguish between DNS failures, network unreachability, connection timeouts, or connection refusal.

Different network errors require different recovery strategies — DNS failures might be retried with alternative hostnames, while connection refused errors indicate the server is not accepting connections on the specified port.

**Detection**: Application reports network failures but provides insufficient information for debugging. Users cannot determine whether problems are client-side, network-related, or server-side.

**Fix**: Capture and report specific error codes and error messages from network operations. Include relevant context like hostname, IP address, port number, and operation type in error reports.

⚠️ **Pitfall: Blocking Operations Without Timeouts**

Network operations can block indefinitely if the remote server stops responding or if network connectivity is lost. Without proper timeouts, applications may hang forever waiting for data that will never arrive.

This is particularly problematic during connection establishment, where a server that accepts the TCP connection but never responds to TLS handshake messages can cause the client to wait indefinitely.

**Detection**: Application becomes unresponsive when connecting to certain servers. Operations appear to hang without completing or reporting errors.

**Fix**: Configure appropriate timeouts for all network operations, including connection establishment, send operations, and receive operations. Choose timeout values that balance responsiveness with tolerance for slow networks.

### Implementation Guidance

The TCP transport and record layer implementation requires careful attention to binary data handling, network programming fundamentals, and buffer management. These components form the foundation for all higher-level TLS operations.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Socket Library | Python `socket` module | `asyncio` with async sockets |
| Binary Parsing | Manual byte manipulation with `struct` | Custom binary parser with type safety |
| Buffer Management | Python `bytearray` and slicing | Memory-mapped buffers or zero-copy techniques |
| Error Handling | Exception-based with try/catch | Result types with explicit error handling |

**Recommended File Structure:**

```
https_client/
├── transport/
│   ├── __init__.py
│   ├── tcp_transport.py         ← TCP connection management
│   └── tcp_transport_test.py     ← TCP transport tests
├── record_layer/
│   ├── __init__.py
│   ├── record_layer.py          ← TLS record parsing and classification
│   ├── binary_parser.py         ← Binary data parsing utilities
│   └── record_layer_test.py     ← Record layer tests
├── common/
│   ├── __init__.py
│   ├── constants.py             ← TLS constants and enumerations
│   └── exceptions.py            ← Custom exception types
└── main.py                      ← Main application entry point
```

**Infrastructure Starter Code:**

```python
# common/constants.py - Complete TLS constants definition
"""TLS protocol constants and enumerations."""

class ContentType:
    """TLS record content types."""
    CHANGE_CIPHER_SPEC = 20
    ALERT = 21
    HANDSHAKE = 22
    APPLICATION_DATA = 23

class TLSVersion:
    """TLS protocol versions."""
    TLS_1_3 = 0x0304
    TLS_1_2 = 0x0303  # Legacy version for compatibility

class HandshakeType:
    """TLS handshake message types."""
    CLIENT_HELLO = 1
    SERVER_HELLO = 2
    ENCRYPTED_EXTENSIONS = 8
    CERTIFICATE = 11
    CERTIFICATE_VERIFY = 15
    FINISHED = 20

# Record layer limits
MAX_RECORD_PAYLOAD_LENGTH = 16384  # 16 KB maximum
RECORD_HEADER_LENGTH = 5

# common/exceptions.py - Complete exception hierarchy
"""Custom exceptions for HTTPS client."""

class TLSError(Exception):
    """Base class for all TLS-related errors."""
    pass

class TransportError(TLSError):
    """Errors related to TCP transport."""
    pass

class RecordLayerError(TLSError):
    """Errors in TLS record parsing or validation."""
    pass

class HandshakeError(TLSError):
    """Errors during TLS handshake processing."""
    pass

class ConnectionTimeoutError(TransportError):
    """Connection operation timed out."""
    pass

class InvalidRecordError(RecordLayerError):
    """Received malformed or invalid TLS record."""
    pass
```

**Binary Parser Infrastructure:**

```python
# record_layer/binary_parser.py - Complete binary parsing utilities
"""Utilities for parsing binary TLS data structures."""

import struct
from typing import Tuple

class BinaryParser:
    """Static utility methods for parsing binary TLS data."""
    
    @staticmethod
    def parse_uint8(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse single byte as unsigned 8-bit integer.
        
        Returns:
            Tuple of (parsed_value, new_offset)
        """
        if offset >= len(data):
            raise ValueError("Insufficient data for uint8")
        value = data[offset]
        return value, offset + 1
    
    @staticmethod
    def parse_uint16(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse 2 bytes as unsigned 16-bit integer in network byte order.
        
        Returns:
            Tuple of (parsed_value, new_offset)
        """
        if offset + 2 > len(data):
            raise ValueError("Insufficient data for uint16")
        value = struct.unpack('!H', data[offset:offset+2])[0]
        return value, offset + 2
    
    @staticmethod
    def parse_uint24(data: bytes, offset: int) -> Tuple[int, int]:
        """Parse 3 bytes as unsigned 24-bit integer in network byte order.
        
        Returns:
            Tuple of (parsed_value, new_offset)
        """
        if offset + 3 > len(data):
            raise ValueError("Insufficient data for uint24")
        # Parse as 4-byte integer with leading zero
        padded = b'\x00' + data[offset:offset+3]
        value = struct.unpack('!I', padded)[0]
        return value, offset + 3
    
    @staticmethod
    def parse_variable_bytes(data: bytes, offset: int, length_bytes: int) -> Tuple[bytes, int]:
        """Parse variable-length byte array with length prefix.
        
        Args:
            data: Source byte array
            offset: Starting position
            length_bytes: Number of bytes used for length field (1, 2, or 3)
        
        Returns:
            Tuple of (parsed_bytes, new_offset)
        """
        if length_bytes == 1:
            length, new_offset = BinaryParser.parse_uint8(data, offset)
        elif length_bytes == 2:
            length, new_offset = BinaryParser.parse_uint16(data, offset)
        elif length_bytes == 3:
            length, new_offset = BinaryParser.parse_uint24(data, offset)
        else:
            raise ValueError(f"Invalid length_bytes: {length_bytes}")
        
        if new_offset + length > len(data):
            raise ValueError("Insufficient data for variable bytes")
        
        value = data[new_offset:new_offset + length]
        return value, new_offset + length
    
    @staticmethod
    def hex_dump(data: bytes, width: int = 16, show_ascii: bool = True) -> str:
        """Create hex dump representation of binary data for debugging.
        
        Args:
            data: Binary data to dump
            width: Number of bytes per line
            show_ascii: Whether to include ASCII representation
        
        Returns:
            Multi-line hex dump string
        """
        lines = []
        for i in range(0, len(data), width):
            chunk = data[i:i+width]
            hex_part = ' '.join(f'{b:02x}' for b in chunk)
            hex_part = hex_part.ljust(width * 3 - 1)  # Pad to consistent width
            
            line = f'{i:04x}: {hex_part}'
            
            if show_ascii:
                ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
                line += f' |{ascii_part}|'
            
            lines.append(line)
        
        return '\n'.join(lines)
```

**Core Logic Skeleton - TCP Transport:**

```python
# transport/tcp_transport.py - Core implementation skeleton
"""TCP transport layer for TLS connections."""

import socket
from typing import Optional, Tuple
from ..common.exceptions import TransportError, ConnectionTimeoutError

class TCPTransport:
    """Manages TCP socket connection for TLS communication."""
    
    def __init__(self):
        self.socket: Optional[socket.socket] = None
        self.remote_address: Optional[Tuple[str, int]] = None
    
    def connect(self, hostname: str, port: int, timeout: float = 10.0) -> None:
        """Establish TCP connection to remote server.
        
        Args:
            hostname: Target server hostname or IP address
            port: Target server port (typically 443 for HTTPS)
            timeout: Connection timeout in seconds
        """
        # TODO 1: Create socket with AF_INET family and SOCK_STREAM type
        # TODO 2: Configure socket options (TCP_NODELAY, SO_REUSEADDR, timeouts)
        # TODO 3: Resolve hostname to IP address using socket.getaddrinfo()
        # TODO 4: Attempt connection using socket.connect() with timeout handling
        # TODO 5: Store successful connection details in self.socket and self.remote_address
        # TODO 6: Handle connection errors and wrap in appropriate exception types
        # Hint: Use socket.settimeout() for connection timeout control
        # Hint: socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) disables Nagle
        pass
    
    def send_bytes(self, data: bytes) -> int:
        """Send raw bytes over TCP connection.
        
        Args:
            data: Bytes to transmit
            
        Returns:
            Number of bytes actually sent
        """
        # TODO 1: Check that socket is connected (self.socket is not None)
        # TODO 2: Initialize bytes_sent counter and data_remaining pointer
        # TODO 3: Loop until all data is sent or error occurs
        # TODO 4: Call socket.send() with remaining data portion
        # TODO 5: Check return value - if 0, connection is closed
        # TODO 6: Update bytes_sent counter and advance data_remaining pointer
        # TODO 7: Handle EAGAIN/EWOULDBLOCK by retrying after brief delay
        # TODO 8: Handle other socket errors by raising TransportError
        # Hint: Use socket.error to catch socket-specific exceptions
        # Hint: Track total bytes sent across all iterations
        pass
    
    def receive_bytes(self, max_bytes: int) -> bytes:
        """Receive raw bytes from TCP connection.
        
        Args:
            max_bytes: Maximum bytes to receive in single operation
            
        Returns:
            Bytes received (may be less than max_bytes)
        """
        # TODO 1: Check that socket is connected
        # TODO 2: Call socket.recv() with max_bytes parameter
        # TODO 3: Check return value - if empty bytes, connection is closed
        # TODO 4: Handle socket errors and convert to TransportError
        # TODO 5: Return received bytes (may be less than requested)
        # Hint: Empty return value indicates graceful connection closure
        # Hint: Don't loop here - return whatever data is immediately available
        pass
    
    def receive_exact(self, num_bytes: int) -> bytes:
        """Receive exactly the specified number of bytes.
        
        Args:
            num_bytes: Exact number of bytes to receive
            
        Returns:
            Bytes received (exactly num_bytes length)
        """
        # TODO 1: Initialize accumulator buffer (bytearray or list)
        # TODO 2: Loop until exactly num_bytes have been received
        # TODO 3: Calculate bytes_remaining for this iteration
        # TODO 4: Call receive_bytes() to get available data
        # TODO 5: If no data received, connection is closed - raise error
        # TODO 6: Append received data to accumulator buffer
        # TODO 7: Continue until accumulator contains exactly num_bytes
        # TODO 8: Return complete buffer as bytes object
        # Hint: Use len(accumulator) to track progress toward target
        # Hint: May require multiple receive_bytes() calls
        pass
```

**Core Logic Skeleton - Record Layer:**

```python
# record_layer/record_layer.py - Core implementation skeleton
"""TLS record layer parsing and message classification."""

from typing import List, Optional
from dataclasses import dataclass
from ..common.constants import ContentType, MAX_RECORD_PAYLOAD_LENGTH, RECORD_HEADER_LENGTH
from ..common.exceptions import InvalidRecordError
from .binary_parser import BinaryParser

@dataclass
class TLSRecord:
    """Represents a parsed TLS record."""
    content_type: int
    legacy_version: int
    length: int
    payload: bytes

class RecordLayer:
    """Handles TLS record framing, parsing, and message classification."""
    
    def __init__(self):
        self.receive_buffer = bytearray()
        self.pending_records: List[TLSRecord] = []
    
    def add_tcp_data(self, data: bytes) -> None:
        """Add newly received TCP data to the processing buffer.
        
        Args:
            data: Raw bytes from TCP transport
        """
        # TODO 1: Append new data to self.receive_buffer
        # TODO 2: Call self._parse_available_records() to process any complete records
        # Hint: bytearray.extend() efficiently appends bytes
        pass
    
    def get_next_record(self) -> Optional[TLSRecord]:
        """Get the next complete record if available.
        
        Returns:
            Complete TLSRecord or None if no records ready
        """
        # TODO 1: Check if any records are pending in self.pending_records
        # TODO 2: If pending records exist, remove and return the first one
        # TODO 3: If no pending records, return None
        # Hint: Use list.pop(0) to remove first element
        pass
    
    def _parse_available_records(self) -> None:
        """Parse any complete records from the receive buffer."""
        # TODO 1: Loop while buffer contains enough data for record header
        # TODO 2: Parse record header (content_type, version, length) at current position
        # TODO 3: Validate content_type is a known value
        # TODO 4: Validate length is within acceptable bounds (1 to MAX_RECORD_PAYLOAD_LENGTH)
        # TODO 5: Check if complete payload is available in buffer
        # TODO 6: If complete record available, extract payload and create TLSRecord
        # TODO 7: Add completed record to self.pending_records
        # TODO 8: Remove processed bytes from beginning of receive_buffer
        # TODO 9: Continue loop to process any additional complete records
        # TODO 10: Stop when insufficient data remains for another complete record
        # Hint: Use BinaryParser methods for header field parsing
        # Hint: buffer[start:end] syntax extracts payload bytes
        # Hint: del buffer[:end] removes processed bytes from beginning
        pass
    
    def _validate_content_type(self, content_type: int) -> None:
        """Validate that content type is recognized."""
        # TODO 1: Check content_type against known values in ContentType class
        # TODO 2: Raise InvalidRecordError if content_type is not recognized
        # Hint: Use hasattr() or check against specific values
        pass
```

**Milestone Checkpoint:**

After implementing the TCP transport and record layer:

1. **Test TCP Connection**: Create a simple script that connects to `www.google.com:443` and verifies the connection succeeds:
   ```python
   transport = TCPTransport()
   transport.connect("www.google.com", 443, timeout=10.0)
   print("Connection successful!")
   ```

2. **Test Byte Operations**: Verify send and receive operations work correctly by sending a simple HTTP request (this won't complete the TLS handshake, but should demonstrate TCP communication):
   ```python
   request = b"GET / HTTP/1.1\r\nHost: www.google.com\r\n\r\n"
   bytes_sent = transport.send_bytes(request)
   response_data = transport.receive_bytes(1024)
   print(f"Sent {bytes_sent} bytes, received {len(response_data)} bytes")
   ```

3. **Test Record Parsing**: Create test TLS records and verify the record layer parses them correctly:
   ```python
   # Handshake record with dummy payload
   test_record = bytes([22, 3, 3, 0, 10]) + b"0123456789"
   record_layer = RecordLayer()
   record_layer.add_tcp_data(test_record)
   parsed = record_layer.get_next_record()
   assert parsed.content_type == 22
   assert parsed.length == 10
   assert parsed.payload == b"0123456789"
   ```

4. **Test Fragmentation Handling**: Verify the record layer correctly handles records split across multiple TCP segments:
   ```python
   # Send record in two fragments
   record_layer = RecordLayer()
   record_layer.add_tcp_data(bytes([22, 3, 3, 0, 10, 1, 2]))  # Header + 2 payload bytes
   assert record_layer.get_next_record() is None  # Should be incomplete
   record_layer.add_tcp_data(b"34567890")  # Complete the payload
   parsed = record_layer.get_next_record()
   assert parsed is not None
   assert len(parsed.payload) == 10
   ```

**Expected Behavior**: TCP connections should establish successfully to port 443 on various HTTPS servers. The record layer should parse well-formed TLS records and handle fragmentation correctly. Error conditions (invalid record lengths, unknown content types, connection failures) should raise appropriate exceptions with descriptive error messages.

**Debugging Tips:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| Connection hangs indefinitely | No timeout configured | Check if socket.settimeout() is called | Add timeout to connect() operation |
| "Connection refused" errors | Wrong port or server not running | Verify port 443 and test with different servers | Check hostname/port, try well-known sites |
| Record length shows huge values | Endianness error in parsing | Check if network byte order conversion is used | Use struct.unpack('!H', ...) for uint16 |
| Parsing fails with "insufficient data" | Not handling partial TCP receives | Check if receive_exact() loops until complete | Implement retry loop in receive_exact() |
| Records mixed up or corrupted | Buffer management error | Check receive_buffer slicing and removal | Ensure del buffer[:pos] removes processed bytes |
| Memory usage grows continuously | Buffer not being cleared | Check if processed data is removed from buffers | Remove parsed records from accumulator |


## ClientHello Message Construction

> **Milestone(s):** Milestone 2 — ClientHello

Think of the ClientHello message as introducing yourself at a diplomatic reception. Just as a diplomat presents their credentials, supported languages, and areas of expertise to begin negotiations, the ClientHello message presents the client's capabilities, supported encryption methods, and essential information to begin the TLS handshake. However, unlike a simple introduction, this message must be precisely formatted according to strict protocol requirements, contain cryptographically secure random data, and include extensions that enable modern security features.

The ClientHello message serves as the opening move in the TLS handshake chess game. It must communicate the client's capabilities while maintaining compatibility with older TLS versions, include proper security parameters, and set up the foundation for secure key exchange. This milestone transforms our basic TCP connection into the beginning of a secure TLS negotiation.

![TLS 1.3 Handshake Message Flow](./diagrams/tls-handshake-sequence.svg)

### ClientHello Message Format

The ClientHello message follows a precisely defined binary format that balances backward compatibility with TLS 1.2 while enabling TLS 1.3 features. Think of this format as a diplomatic passport — it contains legacy fields required for border control (TLS 1.2 compatibility) and modern security features (TLS 1.3 extensions) in a single document.

**ClientHello Binary Structure**

The ClientHello message consists of several required fields followed by optional extensions. Each field serves a specific purpose in the handshake negotiation process.

| Field Name | Type | Length | Description |
|------------|------|--------|-------------|
| `legacy_version` | uint16 | 2 bytes | Must be 0x0303 (TLS 1.2) for compatibility, actual version negotiated via extensions |
| `random` | bytes | 32 bytes | Cryptographically secure random bytes for key derivation and replay protection |
| `legacy_session_id` | variable | 1-33 bytes | Legacy field for TLS 1.2 compatibility, length prefix + session ID bytes |
| `cipher_suites` | variable | 2+ bytes | List of supported cipher suites, length prefix + cipher suite identifiers |
| `legacy_compression_methods` | variable | 1+ bytes | Must contain single null byte (no compression) for TLS 1.3 |
| `extensions` | variable | 2+ bytes | TLS extensions containing actual negotiation parameters |

**Random Field Generation**

The 32-byte random field serves multiple critical security functions. It provides entropy for key derivation, prevents replay attacks, and ensures each handshake produces unique encryption keys even with identical parameters.

> **Critical Security Requirement**: The random field MUST use a cryptographically secure random number generator. Using predictable values like timestamps or weak PRNGs completely undermines TLS security.

The random field structure contains:
- 32 bytes of cryptographically secure random data
- No timestamp requirement (unlike older TLS versions)
- Uniform distribution across all 256 possible byte values
- Sufficient entropy to resist brute force attacks

**Legacy Session ID Handling**

The legacy session ID field maintains compatibility with TLS 1.2 middleboxes that expect session resumption parameters. For TLS 1.3, this field serves no cryptographic purpose but prevents certain network appliances from dropping connections.

> **Decision: Include Legacy Session ID for Middlebox Compatibility**
> - **Context**: Some network middleboxes and firewalls expect TLS connections to include session IDs and may drop connections without them
> - **Options Considered**: 
>   - Empty session ID (cleaner but risks compatibility issues)
>   - Random 32-byte session ID (maximum compatibility)
>   - Fixed session ID (predictable but functional)
> - **Decision**: Generate random 32-byte session ID for maximum compatibility
> - **Rationale**: The compatibility benefits outweigh the slight increase in message size, and random values prevent tracking
> - **Consequences**: Improves middlebox traversal but increases ClientHello size by 33 bytes

| Session ID Approach | Pros | Cons | Compatibility Risk |
|---------------------|------|------|--------------------|
| Empty (0 bytes) | Minimal message size | Some middleboxes may drop | High |
| Random 32 bytes | Maximum compatibility | Larger message | Low |
| Fixed value | Predictable behavior | Enables tracking | Medium |

**Message Length Calculations**

Proper length field calculation is critical for message parsing. Each variable-length field includes a length prefix that allows parsers to correctly extract field boundaries.

The total ClientHello message size calculation follows this pattern:
1. Fixed fields: 2 (version) + 32 (random) = 34 bytes
2. Legacy session ID: 1 (length) + session_id_length bytes
3. Cipher suites: 2 (length) + (2 × number_of_cipher_suites) bytes
4. Compression methods: 1 (length) + 1 (null compression) = 2 bytes
5. Extensions: 2 (total length) + sum of individual extension lengths

### Cipher Suite Negotiation

Cipher suite negotiation determines the cryptographic algorithms used for the TLS connection. Think of cipher suites as selecting a complete security toolkit — each suite specifies the key exchange method, authentication algorithm, encryption cipher, and hash function as a coordinated package.

**TLS 1.3 Cipher Suite Architecture**

TLS 1.3 simplified cipher suites by separating key exchange (handled by extensions) from symmetric encryption. Each TLS 1.3 cipher suite specifies only the AEAD encryption algorithm and hash function.

| Cipher Suite | Value | AEAD Algorithm | Hash Function | Key Length | Security Level |
|-------------|-------|----------------|---------------|------------|----------------|
| `TLS_AES_128_GCM_SHA256` | 0x1301 | AES-128-GCM | SHA-256 | 128 bits | Standard |
| `TLS_AES_256_GCM_SHA384` | 0x1302 | AES-256-GCM | SHA-384 | 256 bits | High |
| `TLS_CHACHA20_POLY1305_SHA256` | 0x1303 | ChaCha20-Poly1305 | SHA-256 | 256 bits | High |

**Cipher Suite Preference Ordering**

The order of cipher suites in the ClientHello indicates client preference. Servers typically select the first cipher suite they support from the client's list, making ordering a critical security decision.

> **Decision: Prioritize AES-GCM Over ChaCha20-Poly1305**
> - **Context**: Modern processors include AES hardware acceleration (AES-NI) that significantly improves AES-GCM performance
> - **Options Considered**:
>   - AES-128-GCM first (hardware acceleration, standard security)
>   - ChaCha20-Poly1305 first (constant-time software implementation)
>   - AES-256-GCM first (maximum security, slower)
> - **Decision**: Order as AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305
> - **Rationale**: Hardware-accelerated AES provides better performance on most modern systems while maintaining strong security
> - **Consequences**: Optimizes for common deployment scenarios but may be suboptimal on systems without AES-NI

The recommended cipher suite ordering prioritizes:
1. Hardware acceleration availability (AES-NI support)
2. Security level (128-bit minimum, 256-bit preferred)
3. Performance characteristics (GCM vs Poly1305)
4. Compatibility with common servers

**Cipher Suite Binary Encoding**

Each cipher suite is encoded as a 2-byte identifier in network byte order. The cipher suite list includes a 2-byte length prefix followed by the cipher suite identifiers.

Example encoding for three cipher suites:
1. Length prefix: 0x0006 (6 bytes total)
2. TLS_AES_128_GCM_SHA256: 0x1301
3. TLS_AES_256_GCM_SHA384: 0x1302
4. TLS_CHACHA20_POLY1305_SHA256: 0x1303

### TLS Extensions

TLS extensions provide a backward-compatible mechanism for adding new features to the TLS protocol. Think of extensions as optional modules that can be plugged into the base TLS framework — they allow clients and servers to negotiate additional capabilities without breaking compatibility with implementations that don't support them.

Extensions transform the basic TLS handshake from a simple protocol negotiation into a sophisticated capability exchange system. Each extension serves a specific purpose and follows a standardized format that allows for graceful degradation.

**Extension Binary Format**

All TLS extensions follow a consistent binary structure that enables parsers to skip unknown extensions gracefully.

| Field | Type | Length | Description |
|-------|------|--------|-------------|
| `extension_type` | uint16 | 2 bytes | Identifies the extension type (e.g., SNI, key share) |
| `extension_length` | uint16 | 2 bytes | Length of extension data in bytes |
| `extension_data` | bytes | variable | Extension-specific data payload |

**Server Name Indication (SNI) Extension**

The SNI extension solves the problem of virtual hosting for HTTPS servers. Just as HTTP includes a Host header to specify which website to serve, SNI tells the TLS server which certificate to present during the handshake.

> **Critical Functionality**: SNI is essential for accessing websites hosted on shared IP addresses, which includes most modern web hosting scenarios including CDNs and cloud platforms.

SNI Extension Structure:

| Field | Type | Length | Description |
|-------|------|--------|-------------|
| `extension_type` | uint16 | 2 bytes | 0x0000 (SERVER_NAME) |
| `extension_length` | uint16 | 2 bytes | Length of entire extension data |
| `server_name_list_length` | uint16 | 2 bytes | Length of server name list |
| `name_type` | uint8 | 1 byte | 0x00 (hostname type) |
| `hostname_length` | uint16 | 2 bytes | Length of hostname in bytes |
| `hostname` | bytes | variable | ASCII hostname (e.g., "example.com") |

The SNI extension construction process:
1. Convert hostname to ASCII bytes (no Unicode normalization required)
2. Calculate hostname length and encode as 2-byte length prefix
3. Wrap in server name list structure with type indicator
4. Add extension type and total length fields

**Supported Versions Extension**

The supported versions extension enables TLS version negotiation while maintaining backward compatibility. This extension allows clients to advertise support for newer TLS versions without breaking older servers that don't recognize them.

Supported Versions Extension Structure:

| Field | Type | Length | Description |
|-------|------|--------|-------------|
| `extension_type` | uint16 | 2 bytes | 0x002B (SUPPORTED_VERSIONS) |
| `extension_length` | uint16 | 2 bytes | Length of extension data |
| `supported_versions_length` | uint8 | 1 byte | Length of supported versions list |
| `supported_versions` | uint16[] | variable | List of supported TLS versions |

For TLS 1.3 support, the supported versions list should include:
- 0x0304 (TLS 1.3) — primary target version
- 0x0303 (TLS 1.2) — fallback compatibility version

**Key Share Extension**

The key share extension enables the client to send its ephemeral public key in the ClientHello, allowing TLS 1.3's 1-RTT handshake optimization. This extension transforms the handshake from a 2-RTT negotiation (negotiate curve, then exchange keys) into a 1-RTT exchange (send key immediately).

Key Share Extension Structure:

| Field | Type | Length | Description |
|-------|------|--------|-------------|
| `extension_type` | uint16 | 2 bytes | 0x0033 (KEY_SHARE) |
| `extension_length` | uint16 | 2 bytes | Length of extension data |
| `key_share_list_length` | uint16 | 2 bytes | Length of key share entries |
| `named_group` | uint16 | 2 bytes | Elliptic curve identifier (e.g., X25519) |
| `key_exchange_length` | uint16 | 2 bytes | Length of public key data |
| `key_exchange` | bytes | variable | Client's ephemeral public key |

> **Decision: Use X25519 for Key Exchange**
> - **Context**: Multiple elliptic curves are available for ECDHE key exchange, each with different security and performance characteristics
> - **Options Considered**:
>   - X25519 (modern, fast, constant-time)
>   - P-256 (NIST standard, widely supported)
>   - P-384 (higher security level, slower)
> - **Decision**: Use X25519 as primary choice with P-256 fallback
> - **Rationale**: X25519 provides excellent security with superior performance and resistance to timing attacks
> - **Consequences**: Optimizes for modern security practices but requires fallback for legacy compatibility

| Curve Option | Security Level | Performance | Timing Attack Resistance | Server Support |
|--------------|----------------|-------------|--------------------------|----------------|
| X25519 | High | Excellent | Excellent | Good |
| P-256 | High | Good | Moderate | Excellent |
| P-384 | Very High | Fair | Moderate | Good |

**Extension Ordering Considerations**

Extension order in the ClientHello can affect compatibility with some TLS implementations. While the TLS specification states that extension order should not matter, some servers or middleboxes may have implementation bugs that depend on specific ordering.

Recommended extension ordering:
1. Server Name Indication (0x0000) — enables proper certificate selection
2. Supported Versions (0x002B) — establishes protocol version early
3. Key Share (0x0033) — provides key exchange material

### Common Pitfalls

⚠️ **Pitfall: Incorrect Length Field Calculations**

Many implementations fail due to incorrect length field calculations in variable-length structures. TLS uses nested length fields where outer structures must include the lengths of all inner content.

**Why it's wrong**: Incorrect length fields cause parsers to read beyond message boundaries or stop parsing prematurely, resulting in handshake failures or security vulnerabilities.

**How to fix**: Calculate lengths bottom-up, starting with innermost structures:
1. Calculate content length first
2. Add length prefix sizes to get total field length
3. Sum all field lengths for outer structure length
4. Verify total message length matches sum of all parts

**Example calculation for SNI extension**:
- Hostname "example.com" = 11 bytes
- Plus 2-byte hostname length = 13 bytes
- Plus 1-byte name type = 14 bytes  
- Plus 2-byte server name list length = 16 bytes
- Plus 4-byte extension header = 20 bytes total

⚠️ **Pitfall: Using Predictable Random Values**

Some implementations use weak random number generators or include predictable values like timestamps in the random field.

**Why it's wrong**: Predictable random values enable replay attacks, reduce key derivation entropy, and can allow attackers to predict future encryption keys.

**How to fix**: Always use cryptographically secure random number generators:
- Python: `secrets.token_bytes(32)`
- Go: `crypto/rand.Read()`
- Rust: `getrandom` crate
- C: `/dev/urandom` or `getentropy()`

⚠️ **Pitfall: Extension Length Miscalculation**

Extension lengths must include all nested structure lengths, but many implementations forget to account for length prefix bytes within the extension data.

**Why it's wrong**: Incorrect extension lengths cause servers to reject the ClientHello or parse extensions incorrectly, leading to handshake failures.

**How to fix**: For each extension, calculate the total extension_data length including all internal length prefixes, type bytes, and payload data. The extension_length field must exactly match the extension_data size.

⚠️ **Pitfall: Incompatible Legacy Version Values**

Setting the legacy_version field to TLS 1.3 (0x0304) can cause middlebox interference and connection failures.

**Why it's wrong**: Many network middleboxes don't recognize TLS 1.3 version numbers and may block or reset connections that advertise it in the legacy version field.

**How to fix**: Always set legacy_version to 0x0303 (TLS 1.2) and use the supported_versions extension to advertise TLS 1.3 support. This ensures maximum compatibility while enabling modern features.

⚠️ **Pitfall: Missing Compression Methods Field**

Some implementations forget to include the legacy_compression_methods field or set it incorrectly.

**Why it's wrong**: TLS 1.3 requires the compression methods field to contain exactly one byte with value 0x00 (no compression). Missing or incorrect values cause handshake failures.

**How to fix**: Always include compression methods as:
- 1-byte length: 0x01
- 1-byte value: 0x00 (no compression)

⚠️ **Pitfall: Network Byte Order Confusion**

Multi-byte integer fields must be encoded in network byte order (big-endian), but many implementations accidentally use host byte order.

**Why it's wrong**: Little-endian systems that don't convert to network byte order will send incorrect values, causing parsing failures on the server side.

**How to fix**: Always convert multi-byte integers to network byte order before serialization:
- Use explicit byte manipulation or
- Use language-specific network byte order functions
- Test on both big-endian and little-endian systems

### Implementation Guidance

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Random Generation | `secrets` module | `cryptography` library with OS entropy |
| Binary Serialization | `struct.pack()` with manual formatting | Custom binary writer class |
| Extension Management | List of tuples | Dedicated Extension class hierarchy |
| Length Calculation | Manual arithmetic | Automatic length tracking wrapper |

**File Structure**

This implementation extends the existing TLS client structure with handshake message construction:

```
https-client/
  src/
    transport.py          ← TCPTransport from previous milestone
    record_layer.py       ← RecordLayer from previous milestone
    handshake.py          ← HandshakeEngine (new)
    crypto_engine.py      ← CryptoEngine placeholder
    client.py             ← HTTPSClient main interface
  tests/
    test_handshake.py     ← ClientHello construction tests
```

**Handshake Engine Infrastructure Code**

```python
import secrets
import struct
from typing import List, Tuple, Optional
from enum import IntEnum

class TLSVersion:
    TLS_1_2 = 0x0303
    TLS_1_3 = 0x0304

class HandshakeType:
    CLIENT_HELLO = 1
    SERVER_HELLO = 2

class ExtensionType:
    SERVER_NAME = 0
    SUPPORTED_VERSIONS = 43
    KEY_SHARE = 51

class CipherSuite:
    TLS_AES_128_GCM_SHA256 = 0x1301
    TLS_AES_256_GCM_SHA384 = 0x1302
    TLS_CHACHA20_POLY1305_SHA256 = 0x1303

class NamedGroup:
    X25519 = 0x001D
    P_256 = 0x0017

class Extension:
    def __init__(self, extension_type: int, extension_data: bytes):
        self.extension_type = extension_type
        self.extension_data = extension_data
    
    def to_bytes(self) -> bytes:
        """Serialize extension to binary format."""
        return struct.pack('!HH', self.extension_type, len(self.extension_data)) + self.extension_data

class ClientHello:
    def __init__(self):
        self.legacy_version = TLSVersion.TLS_1_2
        self.random = secrets.token_bytes(32)
        self.legacy_session_id = secrets.token_bytes(32)  # For middlebox compatibility
        self.cipher_suites = [
            CipherSuite.TLS_AES_128_GCM_SHA256,
            CipherSuite.TLS_AES_256_GCM_SHA384,
            CipherSuite.TLS_CHACHA20_POLY1305_SHA256
        ]
        self.legacy_compression_methods = [0]  # No compression
        self.extensions = []
    
    def to_bytes(self) -> bytes:
        """Serialize ClientHello to binary format for transmission."""
        # TODO 1: Pack legacy_version (2 bytes, network byte order)
        # TODO 2: Add 32 bytes of random data
        # TODO 3: Add legacy_session_id with 1-byte length prefix + session ID bytes
        # TODO 4: Add cipher_suites with 2-byte length prefix + cipher suite list
        # TODO 5: Add legacy_compression_methods with 1-byte length prefix + methods
        # TODO 6: Serialize all extensions and add with 2-byte total length prefix
        # TODO 7: Wrap in handshake message header (type + 3-byte length + payload)
        # Hint: Use struct.pack('!H', value) for network byte order 16-bit integers
        # Hint: Use struct.pack('!HHH', value) for 24-bit length (split into 0 + 16-bit)
        pass

class HandshakeEngine:
    def __init__(self):
        self.handshake_messages = []  # For transcript hash calculation
    
    def create_client_hello(self, hostname: str) -> ClientHello:
        """Create ClientHello message with required extensions."""
        client_hello = ClientHello()
        
        # TODO 1: Add SNI extension with provided hostname
        # TODO 2: Add supported_versions extension with TLS 1.3 and 1.2
        # TODO 3: Add key_share extension with X25519 public key
        # TODO 4: Return completed ClientHello message
        # Hint: Call helper methods add_server_name_extension(), etc.
        pass
    
    def add_server_name_extension(self, client_hello: ClientHello, hostname: str) -> None:
        """Add Server Name Indication extension to ClientHello."""
        # TODO 1: Convert hostname to ASCII bytes
        # TODO 2: Create server name list entry: type(1) + length(2) + hostname
        # TODO 3: Wrap in server name list: total_length(2) + entries
        # TODO 4: Create Extension object and add to client_hello.extensions
        # Hint: Name type is 0x00 for hostname
        # Hint: Calculate lengths bottom-up (hostname -> entry -> list -> extension)
        pass
    
    def add_supported_versions_extension(self, client_hello: ClientHello, versions: List[int]) -> None:
        """Add Supported Versions extension to ClientHello."""
        # TODO 1: Calculate total length of versions list (1 byte length + 2 bytes per version)
        # TODO 2: Pack version list length as single byte
        # TODO 3: Pack each version as 2-byte network byte order integer
        # TODO 4: Create Extension object and add to client_hello.extensions
        # Hint: versions should be [TLS_1_3, TLS_1_2] for maximum compatibility
        pass
    
    def add_key_share_extension(self, client_hello: ClientHello, key_shares: List[Tuple[int, bytes]]) -> None:
        """Add Key Share extension to ClientHello."""
        # TODO 1: For each key share, pack named_group(2) + key_length(2) + key_data
        # TODO 2: Calculate total key share list length
        # TODO 3: Pack key share list length as 2 bytes + all key share entries
        # TODO 4: Create Extension object and add to client_hello.extensions
        # Hint: key_shares format is [(named_group, public_key_bytes), ...]
        # Hint: Generate X25519 key pair using cryptography library for real implementation
        pass
    
    def update_handshake_context(self, message: bytes) -> None:
        """Add handshake message to transcript for hash calculation."""
        self.handshake_messages.append(message)
```

**Core Logic Skeleton for ClientHello Construction**

```python
def create_sni_extension_data(hostname: str) -> bytes:
    """Create SNI extension data with proper nested length fields."""
    # TODO 1: Convert hostname to ASCII bytes and get length
    # TODO 2: Pack server name entry: name_type(1) + hostname_length(2) + hostname
    # TODO 3: Calculate server name list length (includes all entries)
    # TODO 4: Pack complete structure: list_length(2) + server_name_entries
    # TODO 5: Return complete extension data (without extension type/length header)
    # 
    # Binary structure:
    # server_name_list_length (2 bytes) - total length of following data
    # name_type (1 byte) - 0x00 for hostname
    # hostname_length (2 bytes) - length of hostname in bytes  
    # hostname (variable) - ASCII hostname string
    pass

def serialize_cipher_suites(cipher_suites: List[int]) -> bytes:
    """Serialize cipher suite list with length prefix."""
    # TODO 1: Calculate total length (2 bytes per cipher suite)
    # TODO 2: Pack length as 2-byte network byte order integer
    # TODO 3: Pack each cipher suite as 2-byte network byte order integer
    # TODO 4: Return length prefix + all cipher suite bytes
    # Hint: Total length = len(cipher_suites) * 2
    pass

def generate_x25519_key_share() -> Tuple[int, bytes]:
    """Generate X25519 key pair and return (named_group, public_key)."""
    # TODO 1: Import X25519PrivateKey from cryptography.hazmat.primitives.asymmetric
    # TODO 2: Generate private key using X25519PrivateKey.generate()
    # TODO 3: Extract public key using private_key.public_key()
    # TODO 4: Serialize public key using public_key.public_bytes(Raw, Raw)
    # TODO 5: Return (NamedGroup.X25519, public_key_bytes)
    # Note: Store private key for later key derivation in Milestone 3
    pass
```

**Language-Specific Hints**

- Use `struct.pack('!H', value)` for network byte order 16-bit integers
- Use `struct.pack('!I', value)` for 32-bit integers, but TLS uses 24-bit lengths
- For 24-bit lengths, use `struct.pack('!I', value)[1:]` to get last 3 bytes
- The `secrets.token_bytes(n)` function provides cryptographically secure random bytes
- Use `hostname.encode('ascii')` to convert hostnames to bytes for SNI
- Install `cryptography` library for X25519 key generation: `pip install cryptography`
- Network byte order is big-endian: most significant byte first

**Milestone Checkpoint**

After implementing ClientHello construction, verify the implementation:

**Unit Test Validation:**
```bash
python -m pytest tests/test_handshake.py::test_client_hello_construction -v
```

**Expected Test Scenarios:**
- ClientHello with SNI extension produces correct binary format
- Random field contains 32 cryptographically secure bytes
- Cipher suite list is properly ordered and encoded
- Extensions have correct type identifiers and lengths
- Total message length matches sum of all components

**Manual Verification Steps:**
1. Create ClientHello for hostname "example.com"
2. Serialize to binary format using `to_bytes()`
3. Verify message starts with handshake type (0x01) and correct length
4. Check that SNI extension contains "example.com" in correct format
5. Confirm supported versions extension includes TLS 1.3 (0x0304)

**Signs of Correct Implementation:**
- ClientHello binary output matches expected TLS message format
- Extension parsing tools (like Wireshark) can decode the message correctly
- Length fields allow complete message parsing without errors
- Random fields are different on each program execution

**Common Debug Issues:**
- Message length too short → Missing extension data or incorrect length calculation
- Parser fails on extensions → Extension length fields don't match data size  
- SNI hostname garbled → Hostname length miscalculation or encoding issues
- Cipher suites rejected → Wrong byte order or unsupported cipher suite values


## ECDHE Key Exchange and Derivation

> **Milestone(s):** Milestone 3 — Key Exchange

Think of the ECDHE key exchange as the moment when two strangers at a crowded party figure out how to share a secret without anyone else overhearing. They can't whisper directly because everyone is listening, but they can use a mathematical magic trick: each person contributes part of a puzzle that, when combined, creates a shared secret that only they know. Even if everyone watches the entire exchange, the secret remains hidden. This is the essence of ECDHE (Elliptic Curve Diffie-Hellman Ephemeral) — a way to establish a shared secret over an insecure channel while providing forward secrecy.

After the ClientHello and ServerHello exchange establishes the basic parameters of the TLS connection, the real cryptographic work begins. The client and server must perform three critical operations: exchange cryptographic key material to derive shared secrets, validate the server's identity through certificate verification, and confirm the integrity of the entire handshake process. This milestone transforms the connection from a simple parameter negotiation into a cryptographically secured channel ready for encrypted communication.

![TLS 1.3 Handshake Message Flow](./diagrams/tls-handshake-sequence.svg)

The key exchange process involves several sophisticated cryptographic operations that must be performed in precise sequence. First, the client and server use ECDHE to establish a shared secret without transmitting the secret itself over the network. Next, this shared secret feeds into a key derivation function that produces all the encryption keys needed for the session. Simultaneously, the client must validate the server's certificate to ensure it's communicating with the intended server, not an imposter. Finally, both parties compute and verify Finished messages that cryptographically confirm the integrity of the entire handshake process.

### ECDHE Key Exchange

Think of ECDHE key exchange as two people mixing paint colors to create a shared secret color. Each person starts with their own private color (the private key) and a publicly known base color (the generator point on the elliptic curve). They each mix their private color with the base to create a public mixture (the public key), then exchange these public mixtures. Each person then adds their original private color to the mixture they received, resulting in the same final color for both — but an eavesdropper who only saw the public mixtures cannot recreate this final secret color.

The mathematical foundation of ECDHE relies on the difficulty of the elliptic curve discrete logarithm problem. Given an elliptic curve point P and another point Q = kP (where k is a scalar and P is multiplied by k), it's computationally infeasible to determine k even when P and Q are publicly known. This one-way mathematical function enables secure key exchange over insecure channels.

| ECDHE Operation | Input | Output | Purpose |
|-----------------|-------|---------|----------|
| Key Pair Generation | Random private scalar (32 bytes) | Private scalar + Public point (32-65 bytes) | Create ephemeral keys for this session |
| Public Key Exchange | Local private key + Remote public key | Shared secret (32 bytes) | Compute common secret without transmission |
| Curve Point Validation | Remote public key point | Valid/Invalid | Ensure received key is mathematically valid |
| Shared Secret Derivation | Private scalar × Public point | Raw shared secret | Extract usable key material |

> **Decision: X25519 Curve Selection**
> - **Context**: TLS 1.3 supports multiple elliptic curves for ECDHE, including P-256, P-384, and X25519
> - **Options Considered**: 
>   1. P-256 (NIST curve with widespread support)
>   2. X25519 (Curve25519 with optimized implementation)
>   3. Multiple curve support with negotiation
> - **Decision**: Implement X25519 as the primary curve with fallback to P-256
> - **Rationale**: X25519 provides equivalent security to P-256 but with faster computation, simpler implementation (no point compression issues), and resistance to timing attacks
> - **Consequences**: Simplified key exchange implementation but requires handling servers that don't support X25519

The ECDHE key exchange process begins when the client generates an ephemeral key pair during ClientHello construction. The client creates a random 32-byte private scalar and computes the corresponding public key by performing scalar multiplication on the curve's generator point. This public key gets included in the ClientHello's KeyShare extension, advertising the client's willingness to perform ECDHE with the specified curve.

When the server receives the ClientHello, it extracts the client's public key from the KeyShare extension and generates its own ephemeral key pair using the same curve. The server then computes the ECDHE shared secret by multiplying its private scalar with the client's public point. The server includes its own public key in the ServerHello's KeyShare extension and sends it back to the client.

Upon receiving the ServerHello, the client extracts the server's public key and performs the corresponding shared secret computation by multiplying its private scalar with the server's public point. Due to the mathematical properties of elliptic curve cryptography, both client and server arrive at the same shared secret despite using different private keys.

| Key Exchange Step | Client Action | Server Action | Data Transmitted |
|-------------------|---------------|---------------|-------------------|
| 1. Client Key Generation | Generate private scalar, compute public point | None | ClientHello with KeyShare |
| 2. Server Key Generation | None | Generate private scalar, compute public point | ServerHello with KeyShare |
| 3. Shared Secret Computation | Private scalar × Server public point | Private scalar × Client public point | None (computed locally) |
| 4. Secret Validation | Verify non-zero result | Verify non-zero result | None |

The shared secret computation must include several validation steps to prevent cryptographic attacks. First, both parties must validate that the received public key represents a valid point on the specified elliptic curve. For X25519, this involves checking that the received 32-byte value is a valid curve point encoding. Second, the computed shared secret must be non-zero — a zero result indicates either an invalid key or a potential attack.

> The critical security property of ECDHE is forward secrecy: even if long-term keys (like certificate private keys) are compromised in the future, past session keys remain secure because the ephemeral keys are discarded after use.

### TLS 1.3 Key Derivation

Think of TLS 1.3 key derivation as a sophisticated key-cutting machine that takes a single master blank (the ECDHE shared secret) and precisely cuts multiple specialized keys for different purposes — one for encrypting client messages, another for server messages, separate keys for handshake vs. application data, and additional keys for authentication. Each key is cut using a different template (salt and info parameters), ensuring they're all unique and suitable for their specific cryptographic purpose.

TLS 1.3 uses HKDF (HMAC-based Key Derivation Function) to derive all cryptographic key material from the ECDHE shared secret. HKDF provides cryptographic separation between different keys and ensures that even if one derived key is compromised, other keys remain secure. The derivation process follows a carefully designed schedule that produces keys at precisely the right moments in the handshake.

![TLS 1.3 Key Derivation Process](./diagrams/key-derivation-flow.svg)

The key derivation process operates in three distinct phases, each producing keys for different stages of the TLS connection. The Early Secret phase (not used in our basic implementation) handles pre-shared key scenarios. The Handshake Secret phase derives keys for encrypting handshake messages after the key exchange. The Master Secret phase derives keys for encrypting application data after handshake completion.

| Derivation Phase | Input Secret | Derived Keys | Usage Period |
|------------------|--------------|--------------|--------------|
| Early Secret | PSK or zero | Early traffic keys | 0-RTT data (advanced feature) |
| Handshake Secret | ECDHE shared secret | Client/Server handshake traffic keys | Encrypted handshake messages |
| Master Secret | Previous secret | Client/Server application traffic keys | Application data encryption |
| Resumption Secret | Master secret | PSK for future sessions | Session resumption (advanced feature) |

> **Decision: HKDF with SHA-256**
> - **Context**: TLS 1.3 key derivation requires a cryptographically strong key derivation function
> - **Options Considered**:
>   1. HKDF-SHA256 (standard for TLS_AES_128_GCM_SHA256)
>   2. HKDF-SHA384 (for stronger cipher suites)
>   3. Custom key derivation function
> - **Decision**: Implement HKDF-SHA256 to match our chosen cipher suite
> - **Rationale**: HKDF provides proven security properties with extract-then-expand design, and SHA-256 offers sufficient security for AES-128-GCM
> - **Consequences**: Simplified implementation focused on single hash algorithm, but requires HKDF library or implementation

The HKDF key derivation process consists of two phases: Extract and Expand. The Extract phase takes the input key material (like the ECDHE shared secret) and a salt value to produce a pseudorandom key of fixed length. The Expand phase takes this pseudorandom key and generates the desired output key material of arbitrary length using an info parameter that provides cryptographic separation between different derived keys.

The handshake secret derivation begins immediately after computing the ECDHE shared secret. The process starts by deriving the Handshake Secret using HKDF-Extract with the shared secret as input key material and a salt derived from the Early Secret (which is zero for basic implementations). From this Handshake Secret, the system derives separate client and server handshake traffic secrets using HKDF-Expand with different info parameters.

| HKDF Operation | Salt | Input Key Material | Info Parameter | Output Length | Purpose |
|----------------|------|-------------------|----------------|---------------|---------|
| Handshake Secret | Derived-Secret(Early-Secret) | ECDHE shared secret | "derived" | 32 bytes | Base secret for handshake keys |
| Client Handshake Traffic | None | Handshake Secret | "c hs traffic" + context | 32 bytes | Client handshake encryption |
| Server Handshake Traffic | None | Handshake Secret | "s hs traffic" + context | 32 bytes | Server handshake encryption |
| Master Secret | Derived-Secret(Handshake-Secret) | Zero input | "derived" | 32 bytes | Base secret for application keys |

Each traffic secret then expands into the specific key material needed for AEAD encryption: a symmetric encryption key and an initialization vector (IV). For AES-128-GCM, the key length is 16 bytes and the IV length is 12 bytes. These values derive from the traffic secret using HKDF-Expand with appropriate info parameters that identify the specific purpose of each derived value.

The handshake context hash plays a crucial role in key derivation security. This hash represents a SHA-256 digest of all handshake messages exchanged so far, providing cryptographic binding between the derived keys and the specific handshake conversation. Any tampering with handshake messages results in different derived keys, causing authentication failures that detect the attack.

> The key derivation schedule ensures that keys for different purposes are cryptographically independent. Even if an attacker compromises the client handshake traffic key, they cannot derive the server handshake traffic key or any application traffic keys.

### Certificate Validation

Think of certificate validation as checking someone's credentials at a high-security facility. Just as a security guard examines an ID card to verify the photo matches the person, checks the expiration date, validates the issuing authority's seal, and confirms the person's name matches the visitor list, certificate validation involves multiple verification steps. The guard doesn't just trust the ID — they follow a chain of trust back to a known authority they recognize.

X.509 certificate validation forms the foundation of TLS security by ensuring the client connects to the authentic server, not an imposter. The validation process involves multiple cryptographic and policy checks that together establish the server's identity and the certificate's validity. Without proper certificate validation, all the cryptographic sophistication of TLS becomes meaningless because an attacker could simply present their own certificate for the target domain.

![Certificate Validation Process](./diagrams/certificate-validation-flow.svg)

The certificate validation process examines multiple aspects of the server's certificate and any intermediate certificates in the chain. Cryptographic validation ensures the certificate signatures are mathematically correct and trace back to a trusted root. Policy validation ensures the certificate is currently valid, hasn't expired, and is authorized for the intended use. Hostname validation ensures the certificate covers the specific domain being accessed.

| Validation Check | Input | Verification | Failure Impact |
|------------------|-------|--------------|---------------|
| Signature Verification | Certificate + Issuer public key | RSA/ECDSA signature validation | Certificate could be forged |
| Chain of Trust | Certificate chain + Root CAs | Path from server cert to trusted root | Unknown or untrusted issuer |
| Expiration Check | Certificate dates + Current time | Not before ≤ now ≤ not after | Certificate expired or not yet valid |
| Hostname Validation | Certificate SAN/CN + Server hostname | String/wildcard matching | Certificate for wrong domain |
| Key Usage Validation | Certificate extensions | Digital signature + Key encipherment | Certificate not authorized for TLS |

> **Decision: Basic Certificate Validation**
> - **Context**: Full certificate validation involves CRL checking, OCSP, policy constraints, and complex chain building
> - **Options Considered**:
>   1. Full validation with CRL/OCSP checking
>   2. Basic validation (signature, expiration, hostname)
>   3. Trust-on-first-use (TOFU) approach
> - **Decision**: Implement basic validation without revocation checking
> - **Rationale**: Basic validation catches most common attacks while keeping implementation complexity manageable for learning purposes
> - **Consequences**: Simplified implementation but misses revoked certificates (acceptable for educational project)

The certificate chain validation process begins with the server certificate and works backwards through intermediate certificates until reaching a trusted root certificate. Each certificate in the chain must be signed by the private key corresponding to the next certificate's public key. The chain terminates when a certificate is found in the client's trusted root certificate store.

Signature verification requires extracting the public key from the issuing certificate and using it to verify the cryptographic signature on the issued certificate. For RSA signatures, this involves RSA public key verification of a PKCS#1 v1.5 or PSS signature. For ECDSA signatures, this involves elliptic curve signature verification using the issuer's ECDSA public key.

The certificate parsing process extracts key information from the X.509 certificate structure. The Subject field identifies the certificate holder, the Issuer field identifies who signed the certificate, and the Validity field specifies the time period when the certificate is valid. The Subject Alternative Name (SAN) extension lists all hostnames and IP addresses the certificate covers.

| Certificate Field | ASN.1 Location | Content | Validation Rule |
|-------------------|----------------|---------|-----------------|
| Subject | tbsCertificate.subject | Distinguished name of certificate holder | Must not be empty for end-entity certs |
| Issuer | tbsCertificate.issuer | Distinguished name of signing authority | Must match subject of issuer certificate |
| Validity | tbsCertificate.validity | notBefore and notAfter dates | Current time must be within range |
| Subject Alternative Name | extensions.subjectAltName | DNS names, IP addresses | Must include target hostname |
| Key Usage | extensions.keyUsage | Permitted key operations | Must allow digital signature |

Hostname validation ensures the certificate authorizes the specific hostname being accessed. The validation process first checks the Subject Alternative Name extension for DNS name entries. If SAN is present, the Common Name field in the Subject is ignored. The hostname matching supports exact matches and wildcard matches (*.example.com matches any single subdomain of example.com).

The certificate's public key extraction requires parsing the SubjectPublicKeyInfo structure to determine the key algorithm (RSA, ECDSA, etc.) and extract the actual public key bytes. This public key will be used later to verify the server's digital signature in the CertificateVerify message, proving the server possesses the corresponding private key.

> Certificate validation must be performed before trusting any cryptographic operations with the server. An invalid certificate indicates either a configuration error or a potential man-in-the-middle attack.

### Finished Message Verification

Think of the Finished message as the final handshake in a diplomatic treaty signing. After all the negotiations, document exchanges, and identity verifications, both parties create a cryptographic seal over everything that was discussed. Each party computes a unique signature that can only be created by someone who witnessed the entire conversation and possesses the correct cryptographic keys. When both parties successfully verify each other's seals, they know the treaty negotiation was authentic and complete.

The Finished message provides cryptographic authentication for the entire TLS handshake process. It ensures that both client and server participated in the same handshake conversation and that no messages were tampered with by attackers. The Finished message uses a keyed hash (HMAC) computed over a hash of all previous handshake messages, creating a compact proof of handshake integrity.

The TLS 1.3 Finished message serves multiple security purposes simultaneously. It provides authentication, confirming that the peer possesses the correct traffic keys derived from the shared secret. It provides integrity protection, detecting any tampering with handshake messages. It provides replay protection by incorporating the unique handshake context. And it provides confirmation that the handshake is complete and both parties can begin sending application data.

| Finished Message Component | Input | Computation | Security Property |
|----------------------------|--------|-------------|------------------|
| Handshake Context | All handshake messages | SHA-256 hash | Integrity of message sequence |
| Finished Key | Traffic secret + "finished" label | HKDF-Expand | Cryptographic authentication |
| Verify Data | Finished key + handshake hash | HMAC-SHA256 | Proof of key possession |
| Message Authentication | Entire Finished message | AEAD encryption | Transport security |

The handshake context calculation requires maintaining a running hash of all handshake messages from ClientHello through the message immediately before the Finished message. For the client's Finished message, this includes ClientHello, ServerHello, EncryptedExtensions, Certificate, CertificateVerify, and the server's Finished message. The context must include the 4-byte handshake message headers (type and length) along with the message bodies.

> **Decision: Incremental Hash Computation**
> - **Context**: Handshake context requires hashing potentially large amounts of data
> - **Options Considered**:
>   1. Store all handshake messages and hash at the end
>   2. Maintain running hash state updated with each message
>   3. Hash messages on-demand when needed
> - **Decision**: Maintain running SHA-256 hash state updated incrementally
> - **Rationale**: Reduces memory usage by avoiding storage of all handshake data while ensuring accurate context computation
> - **Consequences**: Simpler memory management but requires careful hash state management and message ordering

The Finished key derivation uses HKDF-Expand to derive a 32-byte key from the appropriate traffic secret. For the server's Finished message, the server handshake traffic secret provides the input. For the client's Finished message, the client handshake traffic secret provides the input. The derivation uses "finished" as the info parameter with an empty context.

The verify data computation takes the Finished key and the handshake context hash to produce a 32-byte authentication value using HMAC-SHA256. This value proves that the sender possesses the correct traffic secret (derived from the shared ECDHE secret) and participated in the same handshake conversation represented by the context hash.

| Finished Message Step | Client Action | Server Action | Verification |
|-----------------------|---------------|---------------|--------------|
| 1. Context Hash | Hash(ClientHello...Server Finished) | Hash(ClientHello...CertificateVerify) | Must match peer's computation |
| 2. Finished Key | HKDF-Expand(client_handshake_secret) | HKDF-Expand(server_handshake_secret) | Derived from shared secret |
| 3. Verify Data | HMAC(finished_key, context_hash) | HMAC(finished_key, context_hash) | Proves key possession |
| 4. Message Construction | Handshake(finished, verify_data) | Handshake(finished, verify_data) | Encrypted with handshake keys |

The Finished message verification process requires the recipient to compute the same verify data value and compare it with the received value. Any difference indicates either a key derivation mismatch (suggesting ECDHE failure), handshake message tampering (suggesting man-in-the-middle attack), or implementation errors. The comparison must use constant-time comparison to prevent timing attacks that could leak information about the correct value.

After both Finished messages are successfully exchanged and verified, the handshake is cryptographically complete. The client and server can now derive application traffic keys and begin encrypting application data. The successful Finished message exchange represents mutual authentication: the client has verified the server's certificate and Finished message, while the server has verified the client's Finished message (and optionally a client certificate if client authentication was requested).

⚠️ **Pitfall: Handshake Context Ordering**
The handshake context hash must include messages in the exact order they were transmitted, including the 4-byte handshake headers. A common mistake is forgetting to include the handshake message type and length fields in the hash, or including messages in the wrong order. This results in context hash mismatches that cause Finished message verification failures. Always hash the complete handshake message including headers as soon as it's sent or received.

⚠️ **Pitfall: Key Derivation Sequence**
The key derivation must follow the exact TLS 1.3 schedule with correct salt and info parameters at each stage. A common mistake is using the wrong secret as input to HKDF-Extract or using incorrect info strings for HKDF-Expand. Each derived secret depends on previous secrets in a specific sequence, and any error propagates to all subsequent keys. Carefully follow the key schedule diagram and test key derivation with known test vectors.

⚠️ **Pitfall: Certificate Validation Bypass**
It's tempting to implement certificate validation as optional or to skip hostname verification during development. However, this creates a security vulnerability that's easy to forget to fix before deployment. Always implement certificate validation as a required step that fails closed. If you need to test against servers with invalid certificates, use an explicit bypass flag rather than making validation optional.

### Implementation Guidance

This section provides practical guidance for implementing ECDHE key exchange, TLS 1.3 key derivation, certificate validation, and Finished message verification using Python's cryptographic libraries.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Elliptic Curve Operations | `cryptography.hazmat.primitives.asymmetric.x25519` | Custom curve implementation with multiple curve support |
| Key Derivation | `cryptography.hazmat.primitives.kdf.hkdf.HKDF` | Manual HMAC-based implementation for learning |
| Certificate Parsing | `cryptography.x509` module | Manual ASN.1 parsing with `pyasn1` |
| Hash Functions | `hashlib.sha256` | `cryptography.hazmat.primitives.hashes` |
| HMAC Computation | `hmac.new(key, msg, hashlib.sha256)` | `cryptography.hazmat.primitives.hmac` |

#### Recommended File Structure

```
https_client/
  crypto_engine.py          ← Key derivation, ECDHE, encryption
  certificate_validator.py  ← X.509 certificate validation
  handshake_engine.py      ← Handshake message construction
  tls_connection.py        ← Connection state management
  test_vectors/            ← Known test values for validation
    key_derivation_tests.py
    certificate_tests.py
```

#### Infrastructure Starter Code

Complete HKDF implementation ready to use:

```python
import hashlib
import hmac
from typing import bytes

class HKDF:
    """HMAC-based Key Derivation Function (RFC 5869)"""
    
    def __init__(self, hash_func=hashlib.sha256):
        self.hash_func = hash_func
        self.hash_len = hash_func().digest_size
    
    def extract(self, salt: bytes, ikm: bytes) -> bytes:
        """HKDF-Extract: extract pseudorandom key from input key material"""
        if salt is None or len(salt) == 0:
            salt = b'\x00' * self.hash_len
        return hmac.new(salt, ikm, self.hash_func).digest()
    
    def expand(self, prk: bytes, info: bytes = b'', length: int = 32) -> bytes:
        """HKDF-Expand: expand pseudorandom key to desired length"""
        if length > 255 * self.hash_len:
            raise ValueError("Cannot expand to more than 255 * hash_len bytes")
        
        n = (length + self.hash_len - 1) // self.hash_len
        t = b''
        okm = b''
        
        for i in range(n):
            t = hmac.new(prk, t + info + bytes([i + 1]), self.hash_func).digest()
            okm += t
            
        return okm[:length]
    
    def derive(self, salt: bytes, ikm: bytes, info: bytes = b'', length: int = 32) -> bytes:
        """HKDF: extract then expand in one operation"""
        prk = self.extract(salt, ikm)
        return self.expand(prk, info, length)
```

X25519 key exchange utilities:

```python
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization
import os

class X25519KeyExchange:
    """Handles X25519 ECDHE key exchange operations"""
    
    def __init__(self):
        self.private_key = None
        self.public_key = None
    
    def generate_key_pair(self):
        """Generate ephemeral X25519 key pair"""
        self.private_key = x25519.X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
    
    def get_public_key_bytes(self) -> bytes:
        """Get public key as 32-byte value for transmission"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
    
    def compute_shared_secret(self, peer_public_key_bytes: bytes) -> bytes:
        """Compute ECDHE shared secret from peer's public key"""
        peer_public_key = x25519.X25519PublicKey.from_public_bytes(peer_public_key_bytes)
        shared_secret = self.private_key.exchange(peer_public_key)
        if shared_secret == b'\x00' * 32:
            raise ValueError("Computed shared secret is all zeros - invalid key exchange")
        return shared_secret
```

#### Core Logic Skeleton Code

Key derivation implementation for CryptoEngine:

```python
class CryptoEngine:
    def __init__(self):
        self.hkdf = HKDF()
        self.key_schedule = KeyScheduleState()
        self.handshake_context = hashlib.sha256()
        
    def derive_handshake_secrets(self, ecdhe_shared_secret: bytes) -> None:
        """Derive handshake traffic secrets from ECDHE shared secret"""
        # TODO 1: Compute Early Secret (zero for basic implementation)
        # TODO 2: Derive salt for handshake secret using HKDF-Expand-Label(Early-Secret, "derived", "", 32)
        # TODO 3: Extract handshake secret using HKDF-Extract(salt, ecdhe_shared_secret)
        # TODO 4: Get current handshake context hash from self.get_current_handshake_context_hash()
        # TODO 5: Derive client handshake traffic secret using HKDF-Expand-Label(handshake_secret, "c hs traffic", context_hash, 32)
        # TODO 6: Derive server handshake traffic secret using HKDF-Expand-Label(handshake_secret, "s hs traffic", context_hash, 32)
        # TODO 7: Store secrets in self.key_schedule for later use
        # Hint: Use _hkdf_expand_label helper method for TLS 1.3 label format
    
    def derive_application_secrets(self) -> None:
        """Derive application traffic secrets from handshake secret"""
        # TODO 1: Derive salt using HKDF-Expand-Label(handshake_secret, "derived", "", 32)
        # TODO 2: Extract master secret using HKDF-Extract(salt, zero_input)
        # TODO 3: Get current handshake context hash
        # TODO 4: Derive client application traffic secret
        # TODO 5: Derive server application traffic secret
        # TODO 6: Store secrets in key schedule
    
    def derive_traffic_keys(self, traffic_secret: bytes) -> TrafficKeys:
        """Derive encryption key and IV from traffic secret"""
        # TODO 1: Derive 16-byte AES key using HKDF-Expand-Label(traffic_secret, "key", "", 16)
        # TODO 2: Derive 12-byte IV using HKDF-Expand-Label(traffic_secret, "iv", "", 12)
        # TODO 3: Return TrafficKeys(key=aes_key, iv=aes_iv)
        
    def compute_finished_verify_data(self, traffic_secret: bytes, handshake_context_hash: bytes) -> bytes:
        """Compute verify data for Finished message"""
        # TODO 1: Derive finished key using HKDF-Expand-Label(traffic_secret, "finished", "", 32)
        # TODO 2: Compute HMAC-SHA256(finished_key, handshake_context_hash)
        # TODO 3: Return 32-byte verify data
        
    def _hkdf_expand_label(self, secret: bytes, label: str, context: bytes, length: int) -> bytes:
        """TLS 1.3 HKDF-Expand-Label function"""
        # TODO 1: Construct HkdfLabel structure with length, "tls13 " + label, and context
        # TODO 2: Encode as TLS wire format: uint16(length) + uint8(label_len) + label + uint8(context_len) + context
        # TODO 3: Call self.hkdf.expand(secret, hkdf_label_bytes, length)
```

Certificate validation implementation:

```python
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, ec
import datetime

class CertificateValidator:
    def __init__(self, trusted_roots: List[x509.Certificate] = None):
        self.trusted_roots = trusted_roots or self._load_default_roots()
    
    def validate_certificate_chain(self, cert_chain: List[bytes], hostname: str) -> bool:
        """Validate complete certificate chain"""
        # TODO 1: Parse each certificate bytes using x509.load_der_x509_certificate()
        # TODO 2: Build certificate chain from server cert through intermediates
        # TODO 3: Verify signature chain from server cert to trusted root
        # TODO 4: Check certificate validity periods against current time
        # TODO 5: Validate hostname matches certificate SAN or CN
        # TODO 6: Check key usage extensions allow digital signature
        # Return True if all validations pass, False otherwise
        
    def verify_certificate_signature(self, cert: x509.Certificate, issuer_cert: x509.Certificate) -> bool:
        """Verify certificate signature using issuer's public key"""
        # TODO 1: Extract issuer's public key from issuer_cert.public_key()
        # TODO 2: Get signature algorithm from cert.signature_algorithm_oid
        # TODO 3: Verify signature using appropriate method (RSA or ECDSA)
        # TODO 4: Handle signature verification exceptions and return boolean result
        # Hint: Use issuer_public_key.verify(cert.signature, cert.tbs_certificate_bytes, padding, hash_algo)
        
    def validate_hostname(self, cert: x509.Certificate, hostname: str) -> bool:
        """Validate hostname against certificate SAN extension"""
        # TODO 1: Extract Subject Alternative Name extension from certificate
        # TODO 2: Get list of DNS names from SAN extension
        # TODO 3: Check for exact match with hostname
        # TODO 4: Check for wildcard match (*.domain.com matches sub.domain.com)
        # TODO 5: Fall back to Common Name if no SAN extension exists
        # Return True if hostname matches, False otherwise
```

Finished message verification:

```python
def verify_finished_message(self, finished_payload: bytes, expected_traffic_secret: bytes) -> bool:
    """Verify received Finished message"""
    # TODO 1: Get current handshake context hash (all messages except this Finished)
    # TODO 2: Compute expected verify data using compute_finished_verify_data()
    # TODO 3: Compare received payload with expected verify data using constant-time comparison
    # TODO 4: Return True if values match, False otherwise
    # Hint: Use hmac.compare_digest() for constant-time comparison

def construct_finished_message(self, our_traffic_secret: bytes) -> bytes:
    """Construct our Finished message"""
    # TODO 1: Get current handshake context hash
    # TODO 2: Compute verify data using our traffic secret
    # TODO 3: Construct handshake message with type=FINISHED and verify data payload
    # TODO 4: Update handshake context with this message
    # TODO 5: Return complete handshake message bytes
```

#### Milestone Checkpoint

After implementing ECDHE key exchange and key derivation:

1. **Key Exchange Test**: Verify X25519 key exchange produces correct shared secrets
   ```python
   # Test with known vectors - both parties should compute same shared secret
   python -m pytest test_vectors/key_derivation_tests.py::test_x25519_known_vectors
   ```

2. **Key Derivation Test**: Verify TLS 1.3 key schedule produces expected keys
   ```python
   # Test against RFC 8448 test vectors
   python -m pytest test_vectors/key_derivation_tests.py::test_tls13_key_schedule
   ```

3. **Certificate Validation Test**: Test against real certificates
   ```python
   # Validate against known good certificates
   python -m pytest test_vectors/certificate_tests.py::test_real_certificates
   ```

4. **Integration Test**: Complete handshake through Finished messages
   ```bash
   python https_client.py --host www.google.com --port 443 --debug
   # Should see: "Handshake completed successfully"
   # Should NOT see: Certificate validation errors, key derivation failures, or Finished verification errors
   ```

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|------------------|-----|
| "Invalid shared secret" error | Wrong curve or malformed public key | Check key exchange logs, validate public key format | Ensure X25519 public keys are exactly 32 bytes |
| Key derivation produces wrong keys | Incorrect HKDF labels or context | Compare with RFC 8448 test vectors | Verify "tls13 " prefix in labels, correct context inclusion |
| Certificate validation always fails | Missing root certificates or wrong hostname | Check certificate chain, verify SAN extension | Load system root certificates, check hostname matching logic |
| Finished message verification fails | Wrong handshake context or key derivation | Log handshake context hash and traffic secrets | Verify all handshake messages included in context hash |
| Handshake hangs after ServerHello | Server waiting for client certificate | Check server's CertificateRequest message | Implement client certificate handling or send empty Certificate message |


## Encrypted Application Data

> **Milestone(s):** Milestone 4 — Encrypted Communication

Think of encrypted application data as the secure vault where your valuable conversations finally take place. After all the elaborate security theater of the TLS handshake—the introductions, credential verification, and secret key exchange—you finally have a private, authenticated channel where you can conduct actual business. It's like having gone through airport security, identity verification, and entering a soundproof diplomatic meeting room where you can finally discuss sensitive matters, knowing that every word is both private and tamper-evident.

The transition from handshake to application data represents a fundamental shift in the TLS protocol. During the handshake, messages were primarily unencrypted and focused on negotiation. Now, every byte of application data must be encrypted using **Authenticated Encryption with Associated Data (AEAD)**, providing both confidentiality and integrity. This isn't just basic encryption—it's a sophisticated cryptographic construction that ensures data cannot be read, modified, or replayed by attackers.

The challenge of implementing encrypted application data lies in correctly managing the complex interplay between encryption algorithms, sequence numbers, record framing, and connection lifecycle. Unlike simple file encryption where you encrypt once and decrypt once, TLS requires managing an ongoing stream of encrypted records, each with unique nonces, proper sequencing, and robust error handling.

> **Design Insight:** The beauty of TLS 1.3's application data phase is its simplicity compared to earlier versions. There's no more cipher spec changes or renegotiation complexity—just a clean, authenticated encryption pipeline that processes application data records uniformly.

### AEAD Encryption with AES-GCM

**Authenticated Encryption with Associated Data (AEAD)** represents one of modern cryptography's most important advances. Think of AEAD as a sophisticated electronic seal that not only locks your message in an unbreakable box (confidentiality) but also provides a tamper-evident seal that reveals any attempt to modify the contents (authenticity and integrity). Traditional encryption schemes require separate steps for encryption and authentication, creating opportunities for implementation errors. AEAD combines these operations into a single, foolproof primitive.

TLS 1.3 exclusively uses AEAD cipher suites, with **AES-GCM (Galois/Counter Mode)** being the most common. AES-GCM provides excellent performance characteristics because it can be parallelized and benefits from hardware acceleration on modern processors. The GCM mode combines AES in counter mode for encryption with the Galois field multiplication for authentication, producing both ciphertext and an authentication tag in a single operation.

> **Decision: AES-GCM for Application Data Encryption**
> - **Context**: TLS 1.3 requires AEAD encryption for all application data, and we need to choose which AEAD algorithm to implement first
> - **Options Considered**: AES-GCM, ChaCha20-Poly1305, AES-CCM
> - **Decision**: Implement AES-128-GCM as the primary AEAD cipher
> - **Rationale**: AES-GCM offers the best balance of security, performance, and implementation availability. It has widespread hardware acceleration support, excellent performance characteristics, and is the most commonly supported cipher suite across TLS implementations
> - **Consequences**: Enables compatibility with the vast majority of TLS servers while providing strong security guarantees. The GCM mode's parallelizable nature offers good performance, though we're limited to AES block cipher characteristics

| Cipher Option | Security Level | Performance | Hardware Support | Implementation Complexity |
|---------------|----------------|-------------|------------------|---------------------------|
| AES-128-GCM | High | Excellent | Widespread | Moderate |
| ChaCha20-Poly1305 | High | Good | Limited | Higher |
| AES-256-GCM | Very High | Good | Widespread | Moderate |

The **AES-GCM encryption process** follows a precise sequence that must be implemented correctly to maintain security properties. The process requires three key inputs: the plaintext data to encrypt, the encryption key derived from the key schedule, and a unique nonce that must never be reused with the same key. The nonce construction is particularly critical—TLS 1.3 uses an XOR of the traffic secret's IV with the record sequence number to ensure uniqueness.

The **authentication tag** produced by AES-GCM serves as the cryptographic proof that the data hasn't been tampered with. This 16-byte tag is appended to the ciphertext within the TLS record. During decryption, the receiver recomputes the authentication tag and compares it with the received tag. If they don't match, the record is rejected entirely—there's no partial acceptance or error correction.

| AEAD Operation Component | Input | Output | Purpose |
|-------------------------|--------|---------|----------|
| Encryption Key | 16 bytes from traffic secret | Used for AES encryption | Provides confidentiality |
| Initialization Vector (IV) | 12 bytes from traffic secret | Base for nonce construction | Ensures nonce uniqueness |
| Nonce | IV XOR sequence number | 12-byte unique value | Prevents replay attacks |
| Additional Data | TLS record header | Authenticated but not encrypted | Binds encryption to record context |
| Authentication Tag | Computed during encryption | 16-byte integrity proof | Guarantees authenticity and integrity |

The **record encryption algorithm** processes application data through the following steps:

1. The application provides plaintext data that needs to be transmitted securely over the TLS connection
2. The system retrieves the current client application traffic key and IV from the cryptographic state
3. A unique nonce is constructed by XORing the 12-byte IV with the current sequence number, properly padded to 12 bytes
4. The TLS record header is constructed with content type `APPLICATION_DATA`, TLS version, and the length of the encrypted payload plus authentication tag
5. AES-GCM encryption is performed using the traffic key, constructed nonce, plaintext data, and the record header as additional authenticated data
6. The resulting ciphertext and 16-byte authentication tag are concatenated to form the record payload
7. The complete TLS record (header + encrypted payload + tag) is transmitted over the TCP connection
8. The sequence number is incremented to ensure the next record uses a different nonce

The **record decryption algorithm** reverses this process with careful error handling:

1. A complete TLS record is received and parsed to extract the encrypted payload and authentication tag
2. The current server application traffic key and IV are retrieved from the cryptographic state
3. The expected nonce is reconstructed using the current receive sequence number
4. AES-GCM decryption is attempted using the traffic key, reconstructed nonce, received ciphertext, and record header as additional data
5. The authentication tag is verified as part of the decryption process—if verification fails, the entire record is rejected
6. Upon successful verification, the plaintext application data is extracted and returned to the application layer
7. The receive sequence number is incremented to maintain synchronization with the sender

### Sequence Number Handling

**Sequence numbers** in TLS serve as the foundation for replay protection and nonce uniqueness. Think of sequence numbers as timestamps on a diplomatic cable—each message gets a strictly increasing number that prevents an adversary from recording messages and replaying them later, or from reordering messages to cause confusion. Without proper sequence number handling, an attacker could capture encrypted records and replay them to cause duplicate transactions or disrupt the communication flow.

TLS maintains **separate sequence numbers** for sending and receiving directions. Each direction starts at zero when the connection transitions to application data and increments by one for each record processed. This bidirectional approach allows full-duplex communication where both client and server can send records simultaneously without interference.

The **nonce construction mechanism** represents one of TLS 1.3's most elegant security features. Rather than transmitting nonces explicitly (which would consume bandwidth and create synchronization challenges), TLS constructs nonces deterministically from the traffic secret's IV and the sequence number. This approach ensures that every record uses a unique nonce while maintaining perfect synchronization between sender and receiver.

| Sequence Number Property | Client Send | Client Receive | Server Send | Server Receive |
|-------------------------|-------------|----------------|-------------|----------------|
| Initial Value | 0 | 0 | 0 | 0 |
| Increment Trigger | Each record sent | Each record received | Each record sent | Each record received |
| Maximum Value | 2^64 - 1 | 2^64 - 1 | 2^64 - 1 | 2^64 - 1 |
| Overflow Behavior | Connection must close | Connection must close | Connection must close | Connection must close |
| Nonce Construction | IV XOR send_seq | IV XOR recv_seq | IV XOR send_seq | IV XOR recv_seq |

The **nonce derivation process** requires careful attention to byte ordering and padding. The sequence number is a 64-bit integer that must be converted to a 12-byte value for XORing with the IV. The sequence number is represented in network byte order (big-endian) and right-padded with zeros to reach 12 bytes. This padding ensures that the most significant bytes of the nonce change most slowly, which has favorable properties for the underlying AES-GCM algorithm.

```
Nonce Construction Example:
Traffic IV:     0x404142434445464748494a4b
Sequence #3:    0x000000000000000000000003
Nonce Result:   0x404142434445464748494a48
```

**Sequence number overflow** represents a serious security concern that requires proactive handling. When sequence numbers approach their maximum value (2^64 - 1), the connection must be closed and a new TLS handshake performed. Allowing sequence number wraparound would result in nonce reuse, which catastrophically breaks AES-GCM's security guarantees. In practice, sequence number overflow is extremely rare—even at gigabit speeds with minimum-sized records, it would take centuries to exhaust the sequence number space.

The **replay protection mechanism** operates through the combination of sequence numbers and authentication tags. Each record's authentication tag is computed over both the plaintext data and the sequence number (indirectly, through the nonce). This means that even if an attacker captures a valid encrypted record, they cannot replay it successfully because the receiver's sequence number will have advanced, causing nonce mismatch and authentication failure.

| Replay Attack Scenario | Attack Vector | TLS 1.3 Protection | Result |
|------------------------|---------------|-------------------|---------|
| Record Duplication | Attacker captures and retransmits record | Sequence number mismatch causes nonce mismatch | Authentication failure |
| Record Reordering | Attacker delays and reorders records | Each record authenticated with specific sequence number | Authentication failure |
| Cross-Connection Replay | Attacker replays record from different connection | Different traffic keys and IVs | Authentication failure |
| Delayed Replay | Attacker saves records for later replay | Connection state prevents old records | Authentication failure |

### HTTP Request/Response Processing

**HTTP over TLS** represents the practical culmination of all the complex cryptographic machinery we've built. Think of this layer as the moment when all the security infrastructure becomes invisible to the application—you can now send HTTP requests and receive responses exactly as if you were using plain HTTP, but with the confidence that every byte is encrypted and authenticated. The application layer shouldn't need to worry about record boundaries, encryption details, or sequence numbers.

The **HTTP request construction** process builds a standard HTTP/1.1 request message that will be encrypted and transmitted over the TLS channel. HTTP requests consist of three main parts: the request line specifying the method and URL, headers providing metadata, and an optional body containing data. Each component must be formatted according to HTTP specifications, with proper line endings (CRLF) and header formatting.

| HTTP Request Component | Format | Example | Purpose |
|------------------------|--------|---------|---------|
| Request Line | `METHOD /path HTTP/1.1\r\n` | `GET /index.html HTTP/1.1\r\n` | Specifies operation and resource |
| Host Header | `Host: hostname\r\n` | `Host: example.com\r\n` | Identifies target server (required) |
| Additional Headers | `Header-Name: value\r\n` | `User-Agent: HTTPS-Client/1.0\r\n` | Provides request metadata |
| Header Terminator | `\r\n` | `\r\n` | Separates headers from body |
| Message Body | Raw bytes | `{"key": "value"}` | Contains request payload |

The **request sending process** involves several layers of encapsulation and processing:

1. The application constructs an HTTP request message with appropriate method, path, headers, and body content
2. The complete HTTP request is converted to bytes using UTF-8 encoding for headers and appropriate encoding for the body
3. The HTTP request bytes are passed to the TLS layer as application data that needs to be encrypted and transmitted
4. The TLS layer splits the request into one or more application data records, each no larger than the maximum record size
5. Each record is encrypted using AES-GCM with the current client application traffic keys and sequence number
6. The encrypted records are transmitted over the established TCP connection in order
7. Sequence numbers are incremented for each record to ensure unique nonces for subsequent records

**Large request handling** requires careful attention to record size limits and fragmentation. TLS records have a maximum payload size of 16,384 bytes, so HTTP requests larger than this limit must be split across multiple records. This fragmentation is transparent to the HTTP layer—the receiver will decrypt and reassemble the records to reconstruct the original HTTP request.

The **response receiving process** handles the complexity of reassembling potentially fragmented HTTP responses from multiple TLS records:

1. Encrypted TLS records are received over the TCP connection and parsed by the record layer
2. Each APPLICATION_DATA record is decrypted using AES-GCM with the server application traffic keys
3. The decrypted application data from multiple records is concatenated to reconstruct the complete HTTP response
4. The HTTP response is parsed to extract the status line, headers, and body according to HTTP specifications
5. Content-Length or Transfer-Encoding headers are used to determine response completeness
6. The parsed response components are returned to the application for processing

**HTTP response parsing** must handle the variable-length nature of HTTP messages correctly. Unlike TLS records which have explicit length fields, HTTP responses use either Content-Length headers or chunked transfer encoding to indicate message boundaries. The parser must accumulate data until it has received the complete response.

| HTTP Response Component | Format | Parsing Strategy | Completion Signal |
|------------------------|--------|------------------|-------------------|
| Status Line | `HTTP/1.1 200 OK\r\n` | Read until first CRLF | Always single line |
| Headers | `Header: value\r\n` pairs | Read until blank line | Double CRLF sequence |
| Fixed-Length Body | Raw bytes | Read Content-Length bytes | Byte count reached |
| Chunked Body | Hex size + data chunks | Parse chunk headers | Zero-length chunk |

The **connection persistence** decision affects how HTTP requests and responses are managed over the TLS connection. HTTP/1.1 supports persistent connections where multiple request-response cycles can occur over a single TLS connection. This approach amortizes the expensive TLS handshake cost across multiple HTTP transactions.

> **Decision: HTTP/1.1 with Connection Persistence**
> - **Context**: Need to decide how many HTTP requests to support over a single TLS connection
> - **Options Considered**: Single request per connection, HTTP/1.1 persistent connections, HTTP/2 multiplexing
> - **Decision**: Implement HTTP/1.1 with persistent connection support
> - **Rationale**: Provides good performance benefits by reusing TLS connections while maintaining implementation simplicity. HTTP/2 would require substantial additional complexity for stream multiplexing and flow control
> - **Consequences**: Enables multiple requests over one TLS handshake, reducing latency and server load. Requires proper connection lifecycle management and error handling

### Connection Closure

**TLS connection closure** represents the final act in the secure communication protocol. Think of it as the proper diplomatic conclusion to a sensitive meeting—both parties must formally acknowledge the end of the session and ensure that all sensitive materials are properly secured. Improper connection closure can leave security vulnerabilities or cause application errors, so TLS defines a specific closure protocol that ensures clean termination.

The **close_notify alert** serves as TLS's formal goodbye message. Unlike abrupt TCP connection termination, TLS requires explicit notification when either party wants to close the connection. This alert prevents **truncation attacks** where an adversary terminates the connection prematurely to cut off data transmission at a favorable point.

| TLS Alert Component | Value | Purpose | Security Implication |
|-------------------|--------|---------|---------------------|
| Alert Level | Warning (1) | Indicates non-fatal alert | Connection can continue after handling |
| Alert Description | close_notify (0) | Specifies connection closure | Prevents truncation attacks |
| Record Type | ALERT (21) | TLS record type for alert messages | Distinguishes from application data |
| Processing | Must send and receive | Both parties must participate | Ensures mutual closure acknowledgment |

The **graceful closure process** follows a specific sequence that ensures both parties agree to terminate the connection:

1. The initiating party (client or server) decides to close the TLS connection due to application completion or error conditions
2. The initiator constructs a close_notify alert message with alert level Warning and description close_notify
3. The alert message is encrypted using the current traffic keys and transmitted as a TLS ALERT record
4. The initiator marks its send direction as closed and stops sending new application data records
5. The peer receives and decrypts the close_notify alert, recognizing the closure request
6. The peer sends its own close_notify alert back to acknowledge the closure and mark its send direction as closed
7. Both parties have now exchanged close_notify alerts and can safely close the underlying TCP connection
8. The TCP connection is terminated using standard TCP FIN/ACK sequence

**Unidirectional closure** represents an important TLS feature where one party can close its sending direction while continuing to receive data. This capability supports HTTP patterns where the client sends a complete request and then closes its send side, while the server continues to stream a potentially large response.

The **alert processing mechanism** must distinguish between different alert types and handle them appropriately. While close_notify alerts trigger normal connection closure, other alert types indicate error conditions that require different handling strategies.

| Alert Type | Level | Description | Action Required |
|------------|-------|-------------|-----------------|
| close_notify | Warning | Normal connection closure | Send close_notify response, close connection |
| unexpected_message | Fatal | Received inappropriate message | Log error, close connection immediately |
| bad_record_mac | Fatal | Authentication tag verification failed | Log error, close connection immediately |
| handshake_failure | Fatal | Handshake negotiation failed | Log error, close connection immediately |
| internal_error | Fatal | Implementation encountered error | Log error, close connection immediately |

**Connection state cleanup** ensures that cryptographic materials and connection resources are properly disposed of when the connection closes. This cleanup prevents memory leaks and ensures that sensitive key material doesn't persist longer than necessary.

The connection closure cleanup process includes:

1. Clearing all traffic keys and IVs from memory to prevent key material from persisting after connection termination
2. Resetting sequence numbers and cryptographic state to prevent state confusion if connection objects are reused
3. Closing the underlying TCP socket to release network resources and prevent further data transmission
4. Releasing any buffered data or partially processed records to free memory resources
5. Updating connection status to indicate closure and prevent further operations on the closed connection
6. Logging closure reason and statistics for debugging and monitoring purposes

**Error-triggered closure** handles abnormal termination scenarios where the connection must be closed due to protocol violations, cryptographic failures, or implementation errors. Unlike graceful closure, error closure may not allow for the full close_notify exchange.

| Error Scenario | Detection Point | Alert Sent | TCP Behavior |
|----------------|-----------------|------------|--------------|
| Decryption Failure | Record processing | bad_record_mac (Fatal) | Immediate close |
| Protocol Violation | Message parsing | unexpected_message (Fatal) | Immediate close |
| Internal Error | Any component | internal_error (Fatal) | Immediate close |
| Timeout | Connection management | None (may not be possible) | TCP reset |

> ⚠️ **Pitfall: Ignoring Close Notify Alerts**
> A common implementation mistake is treating TLS connections like raw TCP sockets and simply closing the TCP connection without proper TLS closure. This creates security vulnerabilities because the peer cannot distinguish between intentional connection closure and network-level attacks that terminate connections prematurely. Always send close_notify alerts before closing the underlying TCP connection, and always verify that the peer sends close_notify before considering the connection cleanly closed.

### Common Pitfalls

⚠️ **Pitfall: Nonce Reuse in AES-GCM**
One of the most catastrophic mistakes in AEAD implementation is reusing nonces with the same encryption key. This completely breaks the security guarantees of AES-GCM, potentially allowing attackers to recover plaintext or forge authenticated messages. The vulnerability occurs when sequence numbers are reset incorrectly, when multiple connections share the same key material, or when implementation bugs cause sequence number management errors. Always ensure that sequence numbers are properly maintained per connection direction, never reset sequence numbers without deriving new keys, and implement overflow detection to close connections before sequence number wraparound.

⚠️ **Pitfall: Incomplete HTTP Response Handling**
HTTP responses can span multiple TLS records, and implementations often fail to correctly reassemble fragmented responses. This leads to partial data processing, premature connection closure, or application errors when large responses are received. The issue is particularly common with chunked transfer encoding where the response size isn't known in advance. Always accumulate decrypted application data across multiple records, parse HTTP responses according to Content-Length or chunked encoding rules, and maintain receive buffers until complete responses are assembled.

⚠️ **Pitfall: Ignoring Authentication Tag Failures**
When AES-GCM decryption fails due to authentication tag mismatch, implementations sometimes log the error but continue processing the connection or attempt error recovery. This is a serious security violation—authentication failures always indicate either transmission corruption or active attacks, and the connection must be immediately terminated. Never ignore authentication failures, always send fatal alerts when decryption fails, and close the connection immediately without attempting to process the corrupted data.

⚠️ **Pitfall: Sequence Number Synchronization Errors**
TLS requires perfect synchronization of sequence numbers between sender and receiver. Implementation bugs that cause sequence number mismatches lead to all subsequent records being rejected due to nonce mismatches. Common causes include incorrect error recovery that skips sequence number increments, race conditions in multithreaded implementations, or failure to maintain separate sequence numbers for send and receive directions. Always increment sequence numbers exactly once per record processed, maintain separate counters for each direction, and implement connection recovery by closing and re-establishing connections when synchronization is lost.

⚠️ **Pitfall: Improper Connection Closure Handling**
Many implementations either ignore close_notify alerts or fail to send them during connection closure. This creates truncation attack vulnerabilities and prevents clean connection termination. The problem is compounded when implementations mix TLS closure with TCP closure incorrectly, leading to connection state confusion. Always send close_notify alerts before closing connections, wait for peer close_notify before considering closure complete, and ensure that close_notify alerts are properly encrypted and authenticated like other TLS records.

![AEAD Encryption/Decryption Process](./diagrams/encryption-decryption-flow.svg)

![TLS Record Layer Processing Flow](./diagrams/record-layer-processing.svg)

![TLS 1.3 Key Derivation Process](./diagrams/key-derivation-flow.svg)

### Implementation Guidance

The encrypted application data layer represents the culmination of the TLS implementation, where all previous work comes together to provide a secure communication channel. This section bridges the gap between cryptographic theory and practical HTTP communication.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| AEAD Implementation | `cryptography.hazmat` primitives | Hardware-accelerated AES-NI |
| HTTP Processing | Manual string parsing | `http.client` integration |
| Sequence Numbers | Simple integer counters | Atomic operations for threading |
| Connection Management | Blocking I/O | Asynchronous `asyncio` |

**B. Core Infrastructure (Complete Implementation):**

```python
import struct
import secrets
from typing import Tuple, Optional, Dict, List
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes

class SequenceNumbers:
    """Manages TLS sequence numbers for nonce generation and replay protection."""
    
    def __init__(self):
        self.send_seq = 0
        self.recv_seq = 0
        self.max_seq = (1 << 64) - 1
    
    def get_send_nonce(self, iv: bytes) -> bytes:
        """Generate nonce for outgoing records using current send sequence number."""
        if self.send_seq > self.max_seq:
            raise ValueError("Send sequence number overflow - connection must close")
        
        # Convert sequence number to 12-byte big-endian value
        seq_bytes = struct.pack('>Q', self.send_seq).rjust(12, b'\x00')
        
        # XOR with IV to create unique nonce
        nonce = bytes(a ^ b for a, b in zip(iv, seq_bytes))
        
        self.send_seq += 1
        return nonce
    
    def get_recv_nonce(self, iv: bytes) -> bytes:
        """Generate nonce for incoming records using current receive sequence number."""
        if self.recv_seq > self.max_seq:
            raise ValueError("Receive sequence number overflow - connection must close")
        
        seq_bytes = struct.pack('>Q', self.recv_seq).rjust(12, b'\x00')
        nonce = bytes(a ^ b for a, b in zip(iv, seq_bytes))
        
        self.recv_seq += 1
        return nonce

class AEADCipher:
    """Handles AES-GCM encryption and decryption for TLS application data."""
    
    def __init__(self, key: bytes):
        if len(key) not in [16, 24, 32]:
            raise ValueError(f"Invalid AES key length: {len(key)}")
        self.cipher = AESGCM(key)
    
    def encrypt(self, plaintext: bytes, nonce: bytes, additional_data: bytes) -> bytes:
        """Encrypt plaintext with authentication using AES-GCM."""
        return self.cipher.encrypt(nonce, plaintext, additional_data)
    
    def decrypt(self, ciphertext: bytes, nonce: bytes, additional_data: bytes) -> bytes:
        """Decrypt and verify ciphertext using AES-GCM."""
        return self.cipher.decrypt(nonce, ciphertext, additional_data)

class HTTPProcessor:
    """Handles HTTP request construction and response parsing over TLS."""
    
    @staticmethod
    def build_request(method: str, path: str, hostname: str, 
                     headers: Optional[Dict[str, str]] = None, 
                     body: Optional[bytes] = None) -> bytes:
        """Construct HTTP/1.1 request message."""
        if headers is None:
            headers = {}
        
        # Ensure required headers are present
        headers['Host'] = hostname
        headers.setdefault('User-Agent', 'Python-HTTPS-Client/1.0')
        headers.setdefault('Connection', 'keep-alive')
        
        if body:
            headers['Content-Length'] = str(len(body))
        
        # Build request line
        request_parts = [f"{method} {path} HTTP/1.1\r\n"]
        
        # Add headers
        for name, value in headers.items():
            request_parts.append(f"{name}: {value}\r\n")
        
        # End headers with blank line
        request_parts.append("\r\n")
        
        # Combine all parts
        request_bytes = ''.join(request_parts).encode('utf-8')
        if body:
            request_bytes += body
        
        return request_bytes
    
    @staticmethod
    def parse_response(data: bytes) -> Tuple[int, Dict[str, str], bytes]:
        """Parse HTTP response into status code, headers, and body."""
        # Split headers from body
        if b'\r\n\r\n' not in data:
            raise ValueError("Incomplete HTTP response - missing header/body separator")
        
        header_data, body_data = data.split(b'\r\n\r\n', 1)
        header_lines = header_data.decode('utf-8').split('\r\n')
        
        # Parse status line
        status_line = header_lines[0]
        if not status_line.startswith('HTTP/'):
            raise ValueError(f"Invalid HTTP status line: {status_line}")
        
        status_code = int(status_line.split(' ')[1])
        
        # Parse headers
        headers = {}
        for line in header_lines[1:]:
            if ':' not in line:
                continue
            name, value = line.split(':', 1)
            headers[name.strip().lower()] = value.strip()
        
        return status_code, headers, body_data
```

**C. Core Logic Skeleton (Implementation Required):**

```python
class EncryptedDataHandler:
    """Handles encrypted application data processing for TLS connections."""
    
    def __init__(self, crypto_state: 'CryptographicState'):
        self.crypto_state = crypto_state
        self.sequence_numbers = SequenceNumbers()
        self.send_cipher = None
        self.recv_cipher = None
        self.response_buffer = b''
    
    def initialize_ciphers(self) -> None:
        """Initialize AEAD ciphers using derived traffic keys."""
        # TODO 1: Extract client and server application traffic keys from crypto_state.traffic_keys
        # TODO 2: Create AEADCipher instances for send and receive directions
        # TODO 3: Store cipher instances for use in encrypt_record and decrypt_record
        # Hint: Client sends with client_application_traffic_key, receives with server_application_traffic_key
        pass
    
    def encrypt_application_data(self, plaintext: bytes) -> bytes:
        """Encrypt application data into TLS APPLICATION_DATA record."""
        # TODO 1: Get current client application traffic IV from crypto_state
        # TODO 2: Generate unique nonce using sequence_numbers.get_send_nonce()
        # TODO 3: Construct TLS record header with APPLICATION_DATA type and payload length
        # TODO 4: Encrypt plaintext using send_cipher with nonce and record header as AAD
        # TODO 5: Combine record header with encrypted payload to create complete TLS record
        # TODO 6: Return complete encrypted record ready for TCP transmission
        # Hint: Record header format is type(1) + version(2) + length(2) bytes
        pass
    
    def decrypt_application_data(self, encrypted_record: bytes) -> bytes:
        """Decrypt TLS APPLICATION_DATA record to extract plaintext."""
        # TODO 1: Parse TLS record header to extract content type, version, and payload length
        # TODO 2: Verify content type is APPLICATION_DATA (23)
        # TODO 3: Extract encrypted payload from record (everything after 5-byte header)
        # TODO 4: Get current server application traffic IV from crypto_state
        # TODO 5: Generate expected nonce using sequence_numbers.get_recv_nonce()
        # TODO 6: Decrypt payload using recv_cipher with reconstructed nonce and header as AAD
        # TODO 7: Return decrypted plaintext application data
        # Hint: Handle authentication failures by raising exceptions - never ignore them
        pass
    
    def send_http_request(self, method: str, path: str, hostname: str, 
                         headers: Optional[Dict[str, str]] = None, 
                         body: Optional[bytes] = None) -> None:
        """Send HTTP request over encrypted TLS connection."""
        # TODO 1: Use HTTPProcessor.build_request() to construct HTTP request message
        # TODO 2: Split large requests into multiple TLS records if needed (16KB max payload)
        # TODO 3: Encrypt each record using encrypt_application_data()
        # TODO 4: Send encrypted records over TCP transport
        # TODO 5: Handle any encryption or transmission errors appropriately
        # Hint: Large requests may need fragmentation across multiple records
        pass
    
    def receive_http_response(self, timeout: float = 30.0) -> Tuple[int, Dict[str, str], bytes]:
        """Receive and parse HTTP response from encrypted TLS connection."""
        # TODO 1: Receive TLS records from TCP transport until complete HTTP response assembled
        # TODO 2: Decrypt each APPLICATION_DATA record using decrypt_application_data()
        # TODO 3: Accumulate decrypted data in response_buffer
        # TODO 4: Parse accumulated data to determine when complete response received
        # TODO 5: Use HTTPProcessor.parse_response() to extract status, headers, and body
        # TODO 6: Handle Content-Length and chunked transfer encoding correctly
        # TODO 7: Return parsed response components to application
        # Hint: HTTP responses may span multiple TLS records - must reassemble completely
        pass
    
    def send_close_notify(self) -> None:
        """Send TLS close_notify alert to initiate connection closure."""
        # TODO 1: Construct close_notify alert with alert level Warning(1) and description close_notify(0)
        # TODO 2: Create TLS ALERT record containing the alert message
        # TODO 3: Encrypt alert record using current traffic keys (alerts are encrypted in TLS 1.3)
        # TODO 4: Send encrypted alert record over TCP transport
        # TODO 5: Mark send direction as closed to prevent further application data
        # Hint: Alert record format is similar to application data but uses ALERT content type
        pass
    
    def handle_close_notify(self, alert_record: bytes) -> None:
        """Process received close_notify alert and respond appropriately."""
        # TODO 1: Decrypt and parse received ALERT record
        # TODO 2: Verify alert level is Warning and description is close_notify
        # TODO 3: Mark receive direction as closed
        # TODO 4: Send close_notify response if we haven't already sent one
        # TODO 5: Prepare for connection closure after mutual close_notify exchange
        # Hint: Both parties must send close_notify for clean closure
        pass
```

**D. Language-Specific Hints:**

- Use `secrets.randbits()` for cryptographically secure sequence number initialization if needed
- The `cryptography` library's AESGCM class handles nonce length validation automatically
- Use `struct.pack('>Q', seq_num)` for big-endian sequence number encoding
- HTTP header parsing with `str.split(':', 1)` prevents issues with colons in header values
- Use `bytes.join()` for efficient concatenation of multiple byte strings
- Handle `cryptography.exceptions.InvalidTag` for AES-GCM authentication failures

**E. Milestone Checkpoint:**

After implementing encrypted application data:
1. **Test AEAD Encryption**: Verify that records can be encrypted and decrypted correctly
   ```python
   # Should encrypt and decrypt successfully
   plaintext = b"Hello, TLS World!"
   encrypted = handler.encrypt_application_data(plaintext)
   decrypted = handler.decrypt_application_data(encrypted)
   assert decrypted == plaintext
   ```

2. **Test HTTP Request**: Send a simple GET request to a test server
   ```bash
   # Should complete without errors and return valid HTTP response
   python -c "
   client = HTTPSClient()
   client.connect('httpbin.org', 443)
   status, headers, body = client.request('GET', '/get')
   print(f'Status: {status}')
   print(f'Response length: {len(body)} bytes')
   "
   ```

3. **Test Connection Closure**: Verify clean connection termination
   - Connection should send close_notify before TCP FIN
   - Both client and server should exchange close_notify alerts
   - No error messages about truncated connections

**F. Debugging Tips:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| "Authentication tag verification failed" | Sequence number mismatch | Check sequence counter synchronization | Reset connection, verify nonce construction |
| HTTP response truncated | Missing data across records | Inspect record reassembly logic | Buffer all records before parsing |
| Connection hangs during response | Incomplete HTTP parsing | Check Content-Length handling | Implement proper HTTP message completion detection |
| Nonce reuse detected | Sequence number reset bug | Examine sequence counter management | Fix increment logic, add overflow checks |
| Close notify ignored | Missing alert processing | Check alert record handling | Implement proper alert parsing and response |


## Component Interactions and Data Flow

> **Milestone(s):** All milestones (1-4) — understanding component interactions is essential for implementing the complete TLS handshake and application data flow

Think of the HTTPS client component interactions as a carefully choreographed diplomatic summit. Each component has a specific role, knows exactly when to speak and when to listen, and follows a precise protocol. The `TCPTransport` is like the embassy building that provides the secure meeting space, the `RecordLayer` is the interpreter who formats all messages into the proper diplomatic language, the `HandshakeEngine` handles the formal introductions and credential exchanges, the `CryptoEngine` manages the secret keys and encryption, and the `HTTPSClient` orchestrates the entire interaction. Just as diplomats must follow a strict sequence of presentations, credential verification, and formal agreements before conducting actual business, TLS components must complete their handshake dance before any HTTP data can flow securely.

The complexity of TLS component interaction lies in the stateful nature of the protocol. Unlike HTTP where each request is independent, TLS builds up cryptographic context progressively. Each handshake message depends on all previous messages, keys are derived from accumulated shared secrets, and the connection state machine determines what messages are valid at each step. Understanding this flow is crucial because any deviation from the prescribed sequence will cause the handshake to fail, often with cryptic error messages.

### TLS Handshake Sequence

The TLS handshake sequence represents the most intricate component interaction in the entire HTTPS client. Think of it as a formal dance where each partner must know their steps perfectly, respond to their partner's moves correctly, and maintain perfect timing throughout. Each step builds upon the previous ones, creating layers of cryptographic binding that ultimately establish mutual trust and shared encryption keys.

![TLS 1.3 Handshake Message Flow](./diagrams/tls-handshake-sequence.svg)

The handshake begins with client-side components working together to construct the ClientHello message. The `HTTPSClient` initiates the connection by calling the `HandshakeEngine` to begin the TLS negotiation. The `HandshakeEngine` coordinates with the `CryptoEngine` to generate the 32 bytes of cryptographically secure random data that will become part of the handshake transcript. Simultaneously, the `X25519KeyExchange` component generates an ephemeral key pair for forward secrecy, providing the public key bytes that will be included in the key share extension.

| Component | Handshake Step | Primary Responsibility | Data Produced | Next Component |
|-----------|----------------|------------------------|---------------|----------------|
| `HTTPSClient` | Initiation | Trigger handshake start | Connection parameters | `HandshakeEngine` |
| `CryptoEngine` | Random Generation | Generate client random | 32 random bytes | `HandshakeEngine` |
| `X25519KeyExchange` | Key Generation | Generate ephemeral keys | Public key bytes | `HandshakeEngine` |
| `HandshakeEngine` | Message Construction | Build ClientHello | Complete ClientHello | `RecordLayer` |
| `RecordLayer` | Message Framing | Wrap in TLS record | TLS record bytes | `TCPTransport` |
| `TCPTransport` | Network Transmission | Send over TCP | - | Server |

The `HandshakeEngine` constructs the ClientHello by assembling all required fields into the proper binary format. It includes the legacy version field for backward compatibility, the freshly generated client random, an empty legacy session ID for TLS 1.2 compatibility, the list of supported cipher suites (including `TLS_AES_128_GCM_SHA256`), and the extension list. The extension construction requires careful coordination: the Server Name Indication (SNI) extension includes the target hostname, the supported versions extension indicates TLS 1.3 support, and the key share extension contains the X25519 public key generated earlier.

Once the ClientHello is fully constructed, the `HandshakeEngine` passes it to the `RecordLayer` for proper framing. The `RecordLayer` wraps the handshake message in a TLS record with content type `HANDSHAKE`, version field set to TLS 1.2 for compatibility, and the correct length field. The resulting TLS record is then passed to the `TCPTransport` for transmission over the established TCP connection.

> **Critical Insight**: The handshake context begins accumulating immediately. The `CryptoEngine` must update its handshake context hash with the ClientHello message bytes as soon as they are constructed, even before transmission. This hash will be essential for deriving keys and computing the Finished message later.

After sending the ClientHello, the client components enter a coordinated waiting state. The `TCPTransport` receives incoming bytes from the server, which flow up through the `RecordLayer` for parsing and classification. The `RecordLayer` maintains an internal buffer to handle fragmented TCP segments and partially received records. When a complete TLS record is received, it extracts the payload and classifies the message type.

The server's response typically consists of multiple handshake messages: ServerHello, EncryptedExtensions, Certificate, CertificateVerify, and Finished. Each message triggers specific component interactions:

**ServerHello Processing:**
The `RecordLayer` receives the ServerHello record and passes the payload to the `HandshakeEngine` for parsing. The `HandshakeEngine` extracts the server's chosen cipher suite, server random bytes, and extensions. The key share extension contains the server's X25519 public key, which is immediately passed to the `X25519KeyExchange` component for shared secret computation. The 32-byte shared secret flows to the `CryptoEngine` for key derivation.

| ServerHello Field | Processing Component | Action Taken | Next Step |
|-------------------|---------------------|--------------|-----------|
| Server Random | `CryptoEngine` | Store for key derivation | Add to handshake context |
| Cipher Suite | `HandshakeEngine` | Validate selection | Configure AEAD cipher |
| Key Share | `X25519KeyExchange` | Compute shared secret | Pass to `CryptoEngine` |
| Extensions | `HandshakeEngine` | Parse and validate | Store extension data |

**Key Derivation Cascade:**
The `CryptoEngine` performs the HKDF-based key derivation as soon as the ECDHE shared secret is available. This triggers a cascade of cryptographic computations: the shared secret combines with accumulated salt to produce the handshake secret, which then generates client and server handshake traffic secrets. These traffic secrets immediately produce the actual encryption keys and initialization vectors needed for subsequent message protection.

The key derivation must complete before any encrypted handshake messages arrive because the EncryptedExtensions, Certificate, CertificateVerify, and Finished messages are all encrypted with the handshake traffic keys. The `EncryptedDataHandler` uses the newly derived server handshake traffic keys to decrypt these incoming messages.

**Certificate Validation Coordination:**
When the encrypted Certificate message arrives, the component interaction becomes particularly complex. The `EncryptedDataHandler` first decrypts the record using the server handshake traffic keys, then passes the plaintext to the `HandshakeEngine` for Certificate message parsing. The `HandshakeEngine` extracts the certificate chain (typically 2-3 certificates) and forwards them to the `CertificateValidator`.

The `CertificateValidator` performs multiple validation steps that require coordination with external cryptographic libraries. It verifies each certificate's signature using the issuer's public key, checks expiration dates against the current system time, and validates the end-entity certificate's Subject Alternative Name extension against the target hostname. This validation can fail at multiple points, requiring careful error propagation back through the component chain.

**Finished Message Exchange:**
The Finished message represents the culmination of handshake coordination. When the server's encrypted Finished message arrives, the `EncryptedDataHandler` decrypts it and passes the verify data to the `CryptoEngine` for validation. The `CryptoEngine` computes the expected verify data using the server handshake traffic secret and the complete handshake context hash, then compares it with the received value. Only if this verification succeeds can the client proceed to send its own Finished message.

The client's Finished message construction requires the same level of coordination in reverse. The `CryptoEngine` computes the client's verify data using the client handshake traffic secret, the `HandshakeEngine` constructs the Finished message, the `EncryptedDataHandler` encrypts it with the client handshake traffic keys, and the `RecordLayer` frames it for transmission.

### Connection State Machine

The TLS connection state machine governs all component interactions by defining valid message sequences and enforcing protocol correctness. Think of it as the traffic control system at a busy airport - it ensures that each aircraft (message) follows the correct sequence, lands on the right runway (gets processed by the right component), and doesn't interfere with other operations. Every component must respect the current connection state when deciding how to process incoming messages or what actions are valid.

![TLS Connection State Machine](./diagrams/connection-state-machine.svg)

The state machine begins in the `START` state when the `HTTPSClient` initiates a new connection. In this state, only the ClientHello transmission is valid. Once the ClientHello is sent, the connection transitions to `WAIT_SH` (Wait Server Hello), and all components must coordinate to handle only ServerHello messages while rejecting any other handshake message types.

| Current State | Valid Incoming Messages | Component Responsibilities | Next State | Invalid Messages |
|---------------|------------------------|---------------------------|------------|------------------|
| `START` | None (client sends first) | Send ClientHello | `WAIT_SH` | Any incoming message |
| `WAIT_SH` | ServerHello | Parse server parameters, derive keys | `WAIT_EE` | ClientHello, Certificate, Finished |
| `WAIT_EE` | EncryptedExtensions | Parse server extensions | `WAIT_CERT_CR` | ServerHello, Finished |
| `WAIT_CERT_CR` | Certificate, CertificateRequest | Validate certificate chain | `WAIT_CV` | ServerHello, EncryptedExtensions |
| `WAIT_CV` | CertificateVerify | Verify certificate signature | `WAIT_FINISHED` | Certificate, ClientHello |
| `WAIT_FINISHED` | Finished | Verify handshake integrity | `CONNECTED` | Any non-Finished handshake |
| `CONNECTED` | ApplicationData, Alert | Process application data | `CONNECTED` or `CLOSED` | Handshake messages |
| `CLOSED` | None | Connection terminated | `CLOSED` | Any message |

The state machine enforces strict message ordering through component coordination. When the connection is in `WAIT_EE` state, the `RecordLayer` will receive and decrypt incoming records, but the `HandshakeEngine` will reject any handshake message that is not EncryptedExtensions. This prevents protocol downgrade attacks and ensures that all components maintain consistent expectations about the connection state.

**State Transition Coordination:**
Each state transition requires multiple components to update their internal state simultaneously. When transitioning from `WAIT_SH` to `WAIT_EE`, the `CryptoEngine` must complete key derivation, the `EncryptedDataHandler` must initialize its AEAD ciphers with the new handshake traffic keys, and the `HandshakeEngine` must switch from expecting plaintext to encrypted handshake messages.

| State Transition | Component Updates Required | Validation Checks | Error Handling |
|------------------|---------------------------|-------------------|----------------|
| `START` → `WAIT_SH` | `TCPTransport`: Mark handshake started | ClientHello sent successfully | Reset connection on send failure |
| `WAIT_SH` → `WAIT_EE` | `CryptoEngine`: Derive handshake secrets<br>`EncryptedDataHandler`: Initialize AEAD | Server cipher suite supported | Abort on key derivation failure |
| `WAIT_EE` → `WAIT_CERT_CR` | `HandshakeEngine`: Prepare for certificate | Valid EncryptedExtensions | Alert on malformed extensions |
| `WAIT_CERT_CR` → `WAIT_CV` | `CertificateValidator`: Begin validation | Certificate chain present | Alert on validation failure |
| `WAIT_CV` → `WAIT_FINISHED` | `CryptoEngine`: Verify signature | CertificateVerify valid | Alert on signature verification failure |
| `WAIT_FINISHED` → `CONNECTED` | `CryptoEngine`: Derive application secrets<br>`EncryptedDataHandler`: Switch to application keys | Finished verify data matches | Alert on Finished validation failure |

**Error State Handling:**
The state machine must handle both network-level errors and protocol violations gracefully. When the `TCPTransport` detects a connection failure, all components must transition immediately to the `CLOSED` state and clean up their resources. When the `HandshakeEngine` receives an unexpected message type for the current state, it must generate a `unexpected_message` alert and transition to `CLOSED`.

> **Decision: Centralized State Management**
> - **Context**: Components need consistent view of connection state for coordinated message processing
> - **Options Considered**: 
>   1. Distributed state (each component tracks its own state)
>   2. Centralized state in `TLSConnection`
>   3. Event-driven state updates
> - **Decision**: Centralized state management in `TLSConnection` class
> - **Rationale**: Ensures atomic state transitions, simplifies debugging, prevents state inconsistencies between components
> - **Consequences**: Single source of truth for connection state, but requires careful synchronization for concurrent access

**Application Data State:**
Once the connection reaches the `CONNECTED` state, the component interaction pattern changes dramatically. The `RecordLayer` now primarily handles ApplicationData records rather than handshake messages, the `EncryptedDataHandler` uses application traffic keys instead of handshake traffic keys, and the `HTTPProcessor` becomes the primary consumer of decrypted data.

In the `CONNECTED` state, the sequence number handling becomes critical for security. The `EncryptedDataHandler` maintains separate send and receive sequence numbers, incrementing them for each record processed. These sequence numbers are combined with the traffic key initialization vectors to create unique nonces for each AEAD operation, preventing nonce reuse attacks.

**Connection Termination:**
The transition to `CLOSED` state can occur through several paths: graceful shutdown via close_notify alerts, error conditions that trigger fatal alerts, or network-level failures detected by the `TCPTransport`. Each termination path requires different component cleanup procedures:

- **Graceful shutdown**: `HTTPSClient` sends close_notify, waits for peer's close_notify, then closes TCP connection
- **Fatal alert**: `HandshakeEngine` sends appropriate alert, immediately transitions to `CLOSED`, closes TCP connection
- **Network failure**: `TCPTransport` detects connection loss, notifies all components, transitions to `CLOSED`

### Common Pitfalls

⚠️ **Pitfall: State Machine Synchronization Issues**
Novice implementations often have different components tracking connection state independently, leading to inconsistent behavior. For example, the `CryptoEngine` might still be using handshake traffic keys while the `HandshakeEngine` has already transitioned to expecting application data. This causes decryption failures and connection termination. The fix is to implement centralized state management where the `TLSConnection` class maintains the authoritative connection state and notifies all components atomically when state transitions occur.

⚠️ **Pitfall: Premature Key Derivation**
A common mistake is triggering key derivation before all required input data is available. The handshake secret derivation requires the ECDHE shared secret, but attempting this before parsing the server's key share extension will fail. Similarly, application secret derivation requires the complete handshake context hash, but computing this before all handshake messages are processed produces incorrect keys. The fix is to implement clear dependency tracking where each derivation step explicitly checks for required inputs before proceeding.

⚠️ **Pitfall: Handshake Context Update Ordering**
The handshake context hash must include messages in exactly the order they appear on the wire, but component processing can happen asynchronously. If the `CryptoEngine` updates its handshake context before the `HandshakeEngine` completes message validation, the hash might include invalid messages. The fix is to separate message validation from handshake context updates, ensuring the hash is updated only after all validation passes.

⚠️ **Pitfall: Sequence Number Desynchronization**
AEAD encryption requires unique nonces, which TLS generates by combining the IV with sequence numbers. If send and receive sequence numbers get out of sync between client and server (due to lost records, duplicate records, or implementation bugs), all subsequent decryption will fail. The fix is to implement careful sequence number tracking with bounds checking and explicit synchronization during handshake completion.

⚠️ **Pitfall: Component Initialization Dependencies**
Components often have initialization dependencies that aren't obvious from the API. For example, the `EncryptedDataHandler` requires the `CryptoEngine` to complete key derivation before it can initialize its AEAD ciphers, but this dependency isn't enforced at the type level. Attempting encryption before key derivation causes runtime panics. The fix is to implement explicit initialization phases with dependency checking, ensuring components can't be used before their prerequisites are satisfied.

### Implementation Guidance

The component interaction implementation requires careful orchestration of state changes, message passing, and error propagation. The primary challenge is maintaining consistency across all components while handling the asynchronous nature of network communication and cryptographic operations.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| State Machine | Enum with explicit transitions | Actor-based state management |
| Message Passing | Direct method calls | Event-driven message queue |
| Error Handling | Exception-based propagation | Result/Option types with explicit error flow |
| Logging | Standard library logging | Structured logging with correlation IDs |
| Testing | Unit tests per component | State machine property testing |

**B. Recommended File/Module Structure:**

```
https_client/
  tls/
    connection.py              ← TLS connection orchestration
    state_machine.py           ← Connection state management
    handshake_coordinator.py   ← Handshake message flow coordination
    __init__.py
  crypto/
    key_exchange.py           ← X25519 key exchange
    key_derivation.py         ← HKDF key derivation  
    cipher.py                 ← AEAD encryption/decryption
    __init__.py
  record/
    layer.py                  ← TLS record processing
    types.py                  ← Record and message type constants
    __init__.py
  transport/
    tcp.py                    ← TCP socket management
    __init__.py
  application/
    http_processor.py         ← HTTP request/response handling
    __init__.py
```

**C. Infrastructure Starter Code:**

```python
# tls/state_machine.py - Complete connection state management
from enum import Enum
from typing import Optional, Callable, Dict, Any
import threading

class ConnectionState(Enum):
    START = "START"
    WAIT_SH = "WAIT_SH"  
    WAIT_EE = "WAIT_EE"
    WAIT_CERT_CR = "WAIT_CERT_CR"
    WAIT_CV = "WAIT_CV"
    WAIT_FINISHED = "WAIT_FINISHED"
    CONNECTED = "CONNECTED"
    CLOSED = "CLOSED"

class TLSStateMachine:
    def __init__(self):
        self.current_state = ConnectionState.START
        self.state_lock = threading.Lock()
        self.state_callbacks: Dict[ConnectionState, List[Callable]] = {}
        
    def transition_to(self, new_state: ConnectionState, context: Dict[str, Any] = None):
        """Thread-safe state transition with callback notification."""
        with self.state_lock:
            old_state = self.current_state
            if not self._is_valid_transition(old_state, new_state):
                raise ValueError(f"Invalid state transition: {old_state} -> {new_state}")
            
            self.current_state = new_state
            self._notify_callbacks(new_state, context or {})
    
    def _is_valid_transition(self, from_state: ConnectionState, to_state: ConnectionState) -> bool:
        valid_transitions = {
            ConnectionState.START: [ConnectionState.WAIT_SH, ConnectionState.CLOSED],
            ConnectionState.WAIT_SH: [ConnectionState.WAIT_EE, ConnectionState.CLOSED],
            ConnectionState.WAIT_EE: [ConnectionState.WAIT_CERT_CR, ConnectionState.CLOSED],
            ConnectionState.WAIT_CERT_CR: [ConnectionState.WAIT_CV, ConnectionState.CLOSED],
            ConnectionState.WAIT_CV: [ConnectionState.WAIT_FINISHED, ConnectionState.CLOSED],
            ConnectionState.WAIT_FINISHED: [ConnectionState.CONNECTED, ConnectionState.CLOSED],
            ConnectionState.CONNECTED: [ConnectionState.CLOSED],
            ConnectionState.CLOSED: []
        }
        return to_state in valid_transitions.get(from_state, [])

    def register_callback(self, state: ConnectionState, callback: Callable):
        """Register callback to be called when entering specified state."""
        if state not in self.state_callbacks:
            self.state_callbacks[state] = []
        self.state_callbacks[state].append(callback)
    
    def _notify_callbacks(self, state: ConnectionState, context: Dict[str, Any]):
        for callback in self.state_callbacks.get(state, []):
            try:
                callback(context)
            except Exception as e:
                # Log but don't propagate callback errors
                print(f"State callback error for {state}: {e}")

# tls/handshake_coordinator.py - Complete handshake orchestration
import hashlib
from typing import Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class HandshakeContext:
    messages: List[bytes]
    hash_func: hashlib.sha256
    
    def __post_init__(self):
        self.hash_func = hashlib.sha256()
    
    def add_message(self, message: bytes):
        """Add handshake message to transcript and update hash."""
        self.messages.append(message)
        self.hash_func.update(message)
    
    def get_hash(self) -> bytes:
        """Get current handshake context hash."""
        return self.hash_func.copy().digest()

class HandshakeCoordinator:
    def __init__(self, state_machine: TLSStateMachine):
        self.state_machine = state_machine
        self.handshake_context = HandshakeContext()
        self.server_random: Optional[bytes] = None
        self.client_random: Optional[bytes] = None
        
        # Register state transition callbacks
        self.state_machine.register_callback(
            ConnectionState.WAIT_SH, 
            self._on_wait_server_hello
        )
        self.state_machine.register_callback(
            ConnectionState.WAIT_FINISHED,
            self._on_wait_finished
        )
    
    def _on_wait_server_hello(self, context: Dict[str, Any]):
        """Called when transitioning to WAIT_SH state."""
        # Enable ServerHello message processing
        pass
    
    def _on_wait_finished(self, context: Dict[str, Any]):
        """Called when transitioning to WAIT_FINISHED state.""" 
        # Prepare for Finished message validation
        pass
```

**D. Core Logic Skeleton Code:**

```python
# tls/connection.py - TLS connection orchestration skeleton
class TLSConnection:
    def __init__(self, transport: TCPTransport):
        self.transport = transport
        self.state_machine = TLSStateMachine()
        self.handshake_coordinator = HandshakeCoordinator(self.state_machine)
        self.crypto_engine: Optional[CryptoEngine] = None
        self.record_layer: Optional[RecordLayer] = None
        
    def perform_handshake(self, hostname: str, port: int) -> bool:
        """Execute complete TLS handshake with proper component coordination."""
        # TODO 1: Initialize all components (CryptoEngine, RecordLayer, etc.)
        # TODO 2: Generate and send ClientHello message
        # TODO 3: Process ServerHello and derive handshake secrets
        # TODO 4: Process encrypted handshake messages (EE, Cert, CV, Finished)
        # TODO 5: Send client Finished message
        # TODO 6: Derive application traffic secrets
        # TODO 7: Transition to CONNECTED state
        # Hint: Each step should check current state before proceeding
        # Hint: Use state_machine.transition_to() for state changes
        pass
    
    def send_application_data(self, data: bytes) -> int:
        """Send application data over established TLS connection."""
        # TODO 1: Verify connection is in CONNECTED state
        # TODO 2: Encrypt data using current application traffic keys
        # TODO 3: Frame as ApplicationData TLS record  
        # TODO 4: Send via transport layer
        # TODO 5: Increment send sequence number
        # Hint: Use EncryptedDataHandler for encryption
        pass
    
    def receive_application_data(self, max_bytes: int) -> bytes:
        """Receive and decrypt application data.""" 
        # TODO 1: Verify connection is in CONNECTED state
        # TODO 2: Receive TLS record via RecordLayer
        # TODO 3: Verify record type is ApplicationData
        # TODO 4: Decrypt using current application traffic keys
        # TODO 5: Increment receive sequence number
        # TODO 6: Return plaintext data
        # Hint: Handle partial records and reassembly
        pass

    def _process_handshake_message(self, record: TLSRecord) -> bool:
        """Process incoming handshake message based on current state."""
        # TODO 1: Validate record type is HANDSHAKE
        # TODO 2: Decrypt if connection state requires it (post-ServerHello)
        # TODO 3: Parse handshake message type from payload
        # TODO 4: Dispatch to appropriate handler based on current state
        # TODO 5: Update handshake context with message bytes
        # TODO 6: Trigger state transition if message processing succeeds
        # Hint: Use pattern matching on (current_state, message_type) 
        pass

    def _validate_state_transition(self, from_state: ConnectionState, 
                                 to_state: ConnectionState, 
                                 trigger_message: bytes) -> bool:
        """Validate that state transition is valid given current context."""
        # TODO 1: Check state machine allows transition
        # TODO 2: Validate trigger message is appropriate for transition
        # TODO 3: Verify all required component updates can succeed
        # TODO 4: Check no pending operations that would conflict
        # Hint: Different message types trigger different transitions
        pass
```

**E. Language-Specific Hints:**

- Use `threading.Lock` for state machine synchronization to prevent race conditions
- Leverage `dataclasses` for clean message structure definitions with automatic serialization
- Use `enum.Enum` for type-safe state and message type constants
- Consider `asyncio` for advanced implementations to handle concurrent connections
- Use `typing.Optional` and `typing.Union` for clear API contracts about nullable values
- Implement `__enter__` and `__exit__` for automatic connection cleanup in `TLSConnection`

**F. Milestone Checkpoints:**

After implementing the component coordination:

**Checkpoint 1 - State Machine Validation:**
```bash
python -m pytest tests/test_state_machine.py -v
```
Expected output: All state transition tests pass, invalid transitions raise appropriate exceptions.

**Checkpoint 2 - Handshake Coordination:**
```bash
python test_handshake_flow.py --hostname httpbin.org --port 443
```
Expected behavior: ClientHello sent successfully, ServerHello processed, handshake context accumulates correctly, connection reaches CONNECTED state.

**Checkpoint 3 - Full Integration:**
```bash
python test_https_request.py --url https://httpbin.org/get
```
Expected output: Complete TLS handshake, encrypted HTTP request sent, decrypted HTTP response received, clean connection termination.

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| State machine stuck in WAIT_SH | ClientHello not sent or malformed | Check TCP traffic, verify ClientHello format | Fix message construction, ensure TCP send succeeds |
| Decryption failures after ServerHello | Key derivation incomplete or incorrect | Log key derivation inputs and outputs | Ensure ECDHE shared secret computed before key derivation |
| Unexpected message alerts | Component state inconsistency | Log current state before processing each message | Implement atomic state transitions |
| Connection hangs during handshake | Missing state transition after message processing | Add state transition logging | Ensure each successful message handler calls transition_to() |
| Sequence number errors | Send/receive counters out of sync | Log sequence numbers for each record | Reset counters during handshake completion, check increment logic |


## Error Handling and Edge Cases

> **Milestone(s):** All milestones (1-4) — robust error handling is critical throughout the TLS implementation from TCP connection establishment through encrypted communication

Think of error handling in a TLS implementation as building a comprehensive air traffic control system. Just as air traffic controllers must handle everything from routine weather delays to emergency landings, your HTTPS client must gracefully manage a vast spectrum of failures — from simple network timeouts to sophisticated cryptographic attacks. Every component in the TLS stack can fail in multiple ways, and the failure of any single component can cascade through the entire system if not properly contained and handled.

Unlike simpler network protocols, TLS error handling is particularly complex because the protocol operates across multiple layers simultaneously. A single TCP packet loss can manifest as a record layer parsing error, which might trigger a handshake failure, which could indicate either a benign network issue or an active man-in-the-middle attack. Your error handling system must be sophisticated enough to distinguish between these scenarios and respond appropriately — sometimes by retrying, sometimes by alerting the user, and sometimes by immediately terminating the connection for security reasons.

The TLS protocol itself provides a structured alert system for communicating errors between client and server, but this represents only a fraction of the error scenarios your implementation must handle. Network-level errors, cryptographic failures, certificate validation problems, and implementation bugs can all manifest in subtle ways that require careful detection and recovery strategies.

![TLS Connection State Machine](./diagrams/connection-state-machine.svg)

### Network-Level Errors

Network-level errors form the foundation layer of TLS error handling, much like how plumbing problems in a building affect everything built on top of the water system. These errors occur at the TCP transport layer and can manifest in numerous ways that your TLS implementation must detect and handle gracefully. Unlike application-level errors, network errors often provide incomplete or misleading information about their root cause, requiring your error handling logic to make intelligent decisions based on symptoms and context.

The most fundamental challenge in network error handling is distinguishing between transient issues that should trigger retry logic and permanent failures that require immediate connection termination. A timeout during record reception might indicate network congestion that will resolve in seconds, or it could signal that an attacker has injected malformed data to confuse your parser. Your error handling strategy must account for both scenarios while maintaining security properties.

**TCP Connection Establishment Failures**

TCP connection failures occur before any TLS processing begins, but they set the stage for how your entire error handling system responds to subsequent problems. These failures typically manifest during the initial `connect()` system call or during the three-way handshake process. Your implementation must distinguish between different types of connection failures because they indicate vastly different underlying problems.

| Failure Mode | Typical Cause | Detection Method | Recovery Strategy | Security Implications |
|--------------|---------------|------------------|-------------------|----------------------|
| Connection Refused | Server not listening on port 443 | `socket.connect()` raises `ConnectionRefusedError` | Immediate failure, no retry | None - legitimate server unavailability |
| DNS Resolution Failure | Hostname doesn't exist or DNS server unreachable | `getaddrinfo()` fails before socket creation | Retry with exponential backoff, try alternative DNS | Possible DNS hijacking - validate with alternative resolution |
| Network Unreachable | Routing problem or firewall blocking | `socket.connect()` raises `OSError` with `ENETUNREACH` | Retry with different network interface if available | None - network infrastructure problem |
| Connection Timeout | Server overloaded or network path congestion | `socket.connect()` times out after specified duration | Retry with longer timeout, then exponential backoff | Possible DoS attack on server - monitor retry patterns |
| Connection Reset | Server actively rejecting connections | `socket.connect()` raises `ConnectionResetError` | Short retry delay, then immediate failure | Possible server misconfiguration or active filtering |

The `TCPTransport.connect()` method must implement sophisticated timeout handling because TLS handshakes can be computationally expensive for servers, leading to longer connection establishment times than typical HTTP connections. A connection timeout of 30 seconds might be appropriate for HTTPS clients, compared to 5-10 seconds for regular HTTP traffic.

```python
def connect(hostname: str, port: int, timeout: float) -> None:
    """Establish TCP connection with comprehensive error handling and retry logic."""
    # Implementation details in Implementation Guidance section
```

**Partial Data Reception and Fragmentation**

Once a TCP connection is established, your implementation must handle the fundamental mismatch between TCP's stream-oriented nature and TLS's record-oriented protocol. TCP provides a reliable byte stream but makes no guarantees about message boundaries — a single `recv()` call might return one complete TLS record, half of a record, or parts of multiple records. Your record layer implementation must buffer incoming data and carefully parse record boundaries while handling various fragmentation scenarios.

The most insidious network-level error occurs when TCP delivers partial records that appear syntactically valid but are semantically incomplete. For example, you might receive a complete TLS record header indicating a 2048-byte payload, but only 1500 bytes of the payload arrive before the connection stalls. Your implementation must detect this scenario and either wait for the remaining data or timeout with an appropriate error.

| Fragmentation Scenario | Detection Method | Handling Strategy | Potential Security Risk |
|-------------------------|------------------|-------------------|-------------------------|
| Record header incomplete | `len(buffer) < RECORD_HEADER_LENGTH` when parsing | Buffer data until complete header available | Header field manipulation in partial header |
| Record payload incomplete | Payload length from header exceeds available buffer data | Buffer data until `length` bytes accumulated | Large length field causing memory exhaustion |
| Multiple records in single TCP segment | Buffer contains more data after parsing complete record | Continue parsing additional records from buffer | Record boundary confusion attacks |
| Record spanning multiple TCP segments | Record header indicates payload extending beyond current buffer | Accumulate TCP segments until complete record assembled | Reassembly state exhaustion attacks |
| Zero-byte TCP reads | `recv()` returns empty bytes without connection close | Detect peer shutdown vs temporary unavailability | Connection state desynchronization |

The `RecordLayer` component must implement stateful buffering that accumulates TCP data while maintaining strict bounds checking to prevent memory exhaustion attacks. An attacker could send a TLS record header claiming a payload length of several gigabytes, potentially causing your implementation to allocate excessive memory while waiting for data that never arrives.

**Timeout Handling Strategies**

Timeout handling in TLS clients requires balancing responsiveness with tolerance for legitimate delays. TLS handshakes involve multiple cryptographic operations that can be computationally expensive, especially on resource-constrained servers or when using larger key sizes. However, excessively long timeouts create opportunities for resource exhaustion attacks where an attacker initiates many connections and lets them hang indefinitely.

Your timeout strategy must differentiate between different phases of the TLS connection lifecycle because each phase has different performance characteristics and attack vectors:

| Connection Phase | Typical Timeout | Rationale | Attack Vector Addressed |
|------------------|-----------------|-----------|-------------------------|
| TCP Connection | 30 seconds | Network routing can be slow, legitimate servers may be under load | Connection exhaustion - attacker opens many connections |
| ClientHello Response | 60 seconds | Server must select certificates and generate key shares | Computational DoS - server overwhelmed with handshakes |
| Certificate Chain Download | 45 seconds | Large certificate chains take time to transmit | Bandwidth exhaustion - server sends enormous certificate chains |
| Handshake Completion | 30 seconds | Final cryptographic verifications are typically fast | Implementation DoS - server stalls during final verification |
| Application Data | 120 seconds | User-generated content can involve substantial processing | Application-layer DoS - server processing expensive requests |

The `TCPTransport.receive_exact()` method must implement timeout logic that can be adjusted based on the current connection state and expected operation type. During the handshake phase, shorter timeouts help detect active attacks, while during application data transfer, longer timeouts accommodate legitimate server processing delays.

**Common Network-Level Pitfalls**

⚠️ **Pitfall: Blocking Forever on Partial Reads**
Many implementations fail to handle the scenario where `recv()` returns fewer bytes than requested but the connection remains open. The implementation continues calling `recv()` indefinitely, waiting for data that may never arrive. This occurs because developers assume `recv()` will either return the requested number of bytes or indicate connection closure, but TCP allows for partial reads during normal operation.

The fix requires implementing timeout-based partial read handling where each `recv()` call has an individual timeout, and the cumulative time spent reading a single record is bounded. If partial data arrives but stops before completing a record, the connection should be terminated with a timeout error.

⚠️ **Pitfall: Ignoring TCP Reset During Handshake**
Some implementations treat connection reset errors during the TLS handshake as benign network problems and attempt automatic retry. However, a TCP reset immediately after sending ClientHello often indicates that the server rejected the TLS parameters, detected a protocol violation, or identified the connection as potentially malicious.

The correct approach is to distinguish between resets that occur during connection establishment (retry appropriate) versus resets that occur after sending TLS data (investigation required, retry potentially harmful). Logging the exact timing and context of TCP resets helps identify patterns that might indicate attacks or misconfigurations.

⚠️ **Pitfall: Buffer Overflow in Record Assembly**
When implementing record fragmentation handling, developers often allocate buffers based on the length field in TLS record headers without validating that the length is reasonable. An attacker can send a record header claiming a payload length of 4GB, causing the implementation to allocate massive amounts of memory or crash with an out-of-memory error.

The fix requires enforcing the TLS specification's maximum record payload length of 16,384 bytes and immediately terminating connections that violate this limit. Any record claiming a payload larger than `MAX_RECORD_PAYLOAD_LENGTH` should trigger connection termination with an appropriate TLS alert.

### TLS Alert Processing

TLS alerts serve as the protocol's built-in error reporting mechanism, similar to how HTTP status codes communicate request outcomes. However, TLS alerts are far more critical because they often indicate security-relevant events that require immediate action. Unlike HTTP errors, which typically allow for user retry or alternative approaches, many TLS alerts signal conditions that make continued communication impossible or dangerous.

The TLS alert system operates on two levels: warning alerts that indicate problems but allow communication to continue, and fatal alerts that require immediate connection termination. Your implementation must correctly interpret each alert type and respond appropriately, as incorrect alert handling can lead to security vulnerabilities or poor user experience.

![Certificate Validation Process](./diagrams/certificate-validation-flow.svg)

**Alert Message Structure and Classification**

TLS alerts consist of two bytes: an alert level (warning or fatal) and an alert description that specifies the exact error condition. Your implementation must parse these alerts correctly and map each combination to appropriate handling logic. The distinction between warning and fatal alerts is crucial because warning alerts require different response strategies than fatal alerts.

| Alert Level | Numeric Value | Meaning | Required Action |
|-------------|---------------|---------|-----------------|
| Warning | 1 | Non-fatal condition that allows communication to continue | Log alert, continue processing, monitor for patterns |
| Fatal | 2 | Critical error requiring immediate connection termination | Terminate connection, clear secrets, report error to application |

The most commonly encountered TLS alerts in HTTPS client implementations include certificate-related errors, protocol version mismatches, and cryptographic failures. Each alert type provides specific information about what went wrong, enabling your error handling logic to take appropriate corrective or protective action.

| Alert Description | Numeric Value | Typical Cause | Client Response Strategy |
|-------------------|---------------|---------------|-------------------------|
| `close_notify` | 0 | Peer cleanly closing connection | Respond with close_notify, terminate gracefully |
| `unexpected_message` | 10 | Wrong handshake message for current state | Fatal error - terminate connection immediately |
| `bad_record_mac` | 20 | AEAD decryption failure or tampering | Fatal error - possible attack, terminate connection |
| `handshake_failure` | 40 | Server cannot find acceptable cipher suite | Check supported cipher suites, possibly retry |
| `bad_certificate` | 42 | Certificate parsing or validation failure | Examine certificate, report to user for decision |
| `unsupported_certificate` | 43 | Certificate type not supported by server | Try different certificate or report error |
| `certificate_revoked` | 44 | Certificate has been revoked by CA | Terminate connection, warn user of revocation |
| `certificate_expired` | 45 | Certificate validity period has passed | Check system clock, warn user, allow override |
| `certificate_unknown` | 46 | Certificate has unknown critical extension | Terminate connection, certificate unusable |
| `unknown_ca` | 48 | Certificate chain doesn't lead to trusted root | Check trust store, allow user to add exception |
| `decode_error` | 50 | Message parsing failure | Fatal error - possible implementation bug |
| `protocol_version` | 70 | Unsupported or deprecated TLS version | Retry with different supported version |
| `internal_error` | 80 | Server internal error unrelated to protocol | Retry after delay, may be transient |

**Alert Generation and Transmission**

Your HTTPS client implementation must generate appropriate alerts when it detects error conditions, not just process alerts received from servers. Proper alert generation serves both debugging and security purposes — it helps servers understand why connections fail and ensures that protocol violations are communicated clearly rather than causing mysterious connection drops.

The `TLSConnection` component should maintain alert generation logic that maps internal error conditions to appropriate TLS alerts. For example, if certificate validation fails due to hostname mismatch, your implementation should send a `bad_certificate` alert before terminating the connection. This provides valuable feedback to server administrators and makes debugging easier.

```python
def send_alert(level: int, description: int) -> None:
    """Send TLS alert and take appropriate connection action."""
    # Implementation details in Implementation Guidance section
```

**Alert-Based Connection Termination**

When your implementation receives a fatal alert or generates one locally, the connection termination process must follow specific steps to ensure cryptographic material is properly cleared and no sensitive data remains in memory. Simply closing the TCP socket is insufficient — all derived keys, sequence numbers, and handshake state must be explicitly zeroed to prevent potential information leakage.

The connection termination sequence triggered by fatal alerts should:

1. Immediately stop processing any additional incoming data from the TCP connection
2. Clear all cryptographic state including traffic keys, IVs, and sequence numbers from memory
3. Reset the connection state machine to prevent further TLS operations
4. Close the underlying TCP socket with appropriate error reporting
5. Notify the application layer with specific error information derived from the alert

For warning alerts, the handling strategy depends on the specific alert type and your implementation's security policy. Some warning alerts, like `close_notify`, require specific responses but don't indicate security problems. Others, like repeated certificate warnings, might indicate ongoing attacks and should trigger connection termination despite being classified as warnings.

**Close Notify Handling**

The `close_notify` alert deserves special attention because it implements TLS's equivalent of a graceful connection shutdown. Unlike TCP FIN packets, which can be sent unilaterally, TLS close_notify alerts should be acknowledged to prevent truncation attacks where an attacker cuts off communication by injecting connection close messages.

When your implementation receives a `close_notify` alert, the proper response sequence involves:

1. Acknowledging the close by sending your own `close_notify` alert
2. Waiting briefly for the peer's acknowledgment of your close_notify
3. Closing the TCP connection cleanly
4. Clearing all cryptographic state associated with the connection

Failure to implement proper close_notify handling can lead to truncation attacks where attackers terminate connections at convenient points to cause application-level failures or extract information about the data being transmitted.

**Common Alert Processing Pitfalls**

⚠️ **Pitfall: Continuing After Fatal Alerts**
Some implementations incorrectly treat fatal alerts as recoverable errors and attempt to continue communication after receiving them. This violates the TLS specification and can lead to security vulnerabilities because the peer has already terminated its side of the connection and cleared its cryptographic state.

The fix requires immediately terminating the connection upon receiving any fatal alert, regardless of whether the alert seems recoverable. Fatal alerts indicate that the peer considers the connection compromised and will not process any further messages.

⚠️ **Pitfall: Not Clearing Secrets After Alerts**
When alert processing terminates a connection, implementations sometimes forget to clear derived cryptographic material from memory. This can leave sensitive key material accessible to other processes or attackers with memory access capabilities.

The correct approach requires implementing explicit secret clearing routines that overwrite key material with zeros when any connection termination occurs. This should happen regardless of whether termination was caused by alerts, network errors, or normal connection closure.

⚠️ **Pitfall: Alert Amplification Attacks**
Some implementations automatically retry connections after receiving certain alerts without implementing rate limiting or exponential backoff. An attacker can exploit this by sending alerts that trigger automatic retries, causing the client to generate excessive connection attempts that overwhelm the server.

The fix requires implementing alert-aware retry logic that treats repeated alerts of the same type as indicators of persistent problems rather than transient errors. After receiving the same alert multiple times from the same server, the implementation should stop retrying and report the error to the user.

### Cryptographic Errors

Cryptographic errors in TLS implementations represent some of the most serious failure modes because they directly compromise the security properties that TLS is designed to provide. Unlike network errors, which typically result in connection failures, cryptographic errors can succeed partially, leading to connections that appear to work but provide no actual security. Your error handling must be particularly careful in this area because the difference between proper cryptographic error handling and improper handling often determines whether your implementation is vulnerable to sophisticated attacks.

Think of cryptographic error handling as quality control in a pharmaceutical manufacturing process. Just as a single contaminated batch of medicine can cause harm even if it appears identical to proper medicine, a single cryptographic verification that passes when it should fail can compromise the security of the entire communication channel. Your implementation must apply rigorous verification at every cryptographic step and fail immediately when any verification step produces unexpected results.

**Certificate Validation Failures**

Certificate validation represents the most complex area of cryptographic error handling in TLS because it involves multiple verification steps that can fail independently. A certificate chain might pass signature verification but fail hostname validation, or pass both but fail due to expiration. Your implementation must perform all validation steps and handle the various failure modes appropriately.

The certificate validation process in your `CertificateValidator` component must implement defense-in-depth verification where each validation step is performed independently and any single failure causes the entire validation to fail. This prevents subtle attacks where one type of certificate problem masks another more serious issue.

| Validation Step | Failure Modes | Security Impact | Error Handling Strategy |
|-----------------|---------------|-----------------|------------------------|
| Certificate Parsing | Malformed ASN.1, unsupported extensions | DoS via parsing crashes | Immediate failure, detailed logging |
| Signature Verification | Invalid signatures, wrong algorithms | Complete security bypass | Immediate failure, no retry |
| Chain Building | Missing intermediate certificates | Connection failure | Request missing certificates if possible |
| Trust Anchor Validation | Unknown CA, self-signed certificates | Complete security bypass | User decision required for self-signed |
| Expiration Checking | Expired certificates, not-yet-valid | Time-based security bypass | Check system clock, user override possible |
| Hostname Validation | Name mismatch, wildcard abuse | Identity spoofing attacks | Immediate failure, clear error message |
| Revocation Checking | CRL unavailable, OCSP timeout | Revoked certificate acceptance | Configurable policy: fail-open or fail-closed |

The `validate_certificate_chain()` method must implement comprehensive error reporting that distinguishes between different types of validation failures because they require different responses. A hostname mismatch indicates a potential man-in-the-middle attack and should never be automatically retried, while a certificate chain with missing intermediate certificates might be resolvable by fetching the missing certificates from the server.

```python
def validate_certificate_chain(cert_chain: List[bytes], hostname: str) -> bool:
    """Perform comprehensive certificate validation with detailed error reporting."""
    # Implementation details in Implementation Guidance section
```

**Key Exchange and Derivation Errors**

ECDHE key exchange errors can be subtle because the mathematical operations involved might produce results that appear valid but are cryptographically weak or compromised. Your implementation must validate not only that key exchange operations complete successfully, but also that the results have the expected cryptographic properties.

The most dangerous key exchange errors involve accepting weak or invalid public keys from the server. An attacker might send a public key that lies on a weak subgroup of the elliptic curve, causing the shared secret to be predictable or have reduced entropy. Your `X25519KeyExchange` component must validate incoming public keys to ensure they represent valid points on the expected curve.

| Key Exchange Error | Detection Method | Security Impact | Handling Strategy |
|-------------------|------------------|-----------------|-------------------|
| Invalid peer public key | Point validation on curve | Weak shared secret | Immediate connection termination |
| Weak shared secret | Entropy analysis of result | Predictable encryption keys | Immediate connection termination |
| Key derivation failure | HKDF output validation | Corrupted traffic keys | Immediate connection termination |
| Sequence number overflow | Counter bounds checking | Nonce reuse vulnerability | Force connection renegotiation |
| Traffic key corruption | Decryption failure pattern | Complete security bypass | Immediate connection termination |

The `compute_shared_secret()` method must implement point validation to ensure the peer's public key represents a valid point on the X25519 curve. Invalid points can lead to weak shared secrets that compromise the security of all subsequent communication.

**AEAD Encryption and Decryption Failures**

AEAD (Authenticated Encryption with Associated Data) failures in your `AEADCipher` component represent active attacks or implementation bugs, both of which require immediate connection termination. Unlike other cryptographic operations where some failures might be recoverable, AEAD failures indicate that either the data has been tampered with in transit or your implementation has a serious bug in key handling or nonce generation.

The most critical AEAD error occurs when decryption fails due to authentication tag mismatch. This indicates that either an attacker has modified the ciphertext in transit, or your implementation has a bug in nonce handling that causes nonce reuse. Either scenario compromises security and requires immediate connection termination.

| AEAD Error Type | Probable Cause | Security Implication | Required Response |
|-----------------|---------------|---------------------|-------------------|
| Authentication tag mismatch | Data tampering or nonce reuse | Active attack or critical bug | Immediate connection termination |
| Nonce reuse detection | Sequence number handling bug | Complete security bypass | Immediate connection termination |
| Key corruption | Memory corruption or race condition | Unpredictable decryption behavior | Immediate connection termination |
| Plaintext length validation | Padding oracle or implementation bug | Information disclosure | Connection termination, bug investigation |

The `decrypt()` method in your AEAD implementation must never return partially decrypted data when authentication fails. Some implementations mistakenly return the decrypted plaintext even when the authentication tag doesn't match, creating padding oracle vulnerabilities that allow attackers to decrypt arbitrary ciphertext.

**Sequence Number and Replay Protection Errors**

TLS sequence numbers serve dual purposes: they ensure that AEAD nonces are never reused, and they provide replay protection by detecting when attackers attempt to retransmit captured messages. Your `SequenceNumbers` component must implement rigorous bounds checking and overflow detection to maintain both security properties.

Sequence number overflow represents a particularly serious error because it leads directly to nonce reuse, which completely breaks AEAD security. When sequence numbers approach their maximum value, your implementation must either renegotiate the connection to establish new keys or terminate the connection entirely.

```python
def derive_nonce(sequence_number: int) -> bytes:
    """Generate AEAD nonce with overflow protection and replay detection."""
    # Implementation details in Implementation Guidance section
```

**Common Cryptographic Error Pitfalls**

⚠️ **Pitfall: Accepting Invalid Certificate Chains Silently**
Many implementations perform certificate validation but fail to properly handle validation errors, either by ignoring certain types of failures or by providing overly permissive error recovery. For example, an implementation might ignore hostname validation failures if signature validation passes, creating a vulnerability to attacks using valid certificates for wrong hostnames.

The fix requires implementing strict validation where any single validation step failure causes the entire certificate validation to fail. The implementation should provide detailed error reporting to help users understand why validation failed, but should never automatically bypass validation failures.

⚠️ **Pitfall: Timing Attacks in Cryptographic Verification**
Some implementations inadvertently leak information about cryptographic secrets through timing variations in verification operations. For example, string comparison operations that fail fast on the first mismatched byte can allow attackers to guess secrets one byte at a time by measuring response times.

The correct approach requires implementing constant-time comparison operations for all cryptographic verification, including certificate validation, signature verification, and authentication tag checking. These operations should take the same amount of time regardless of where differences occur in the compared values.

⚠️ **Pitfall: Not Clearing Cryptographic State on Errors**
When cryptographic operations fail, some implementations leave sensitive key material in memory longer than necessary, creating opportunities for information disclosure through memory dumps or other side channels. This is particularly problematic when errors occur during key derivation, as partially computed keys might leak information about the final keys.

The fix requires implementing explicit memory clearing routines that execute whenever cryptographic operations fail, regardless of the error type. All intermediate values, derived keys, and cryptographic state should be overwritten with zeros immediately when errors occur, before any error reporting or connection termination logic runs.

### Implementation Guidance

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Error Logging | Python `logging` module with structured JSON output | ELK stack integration with correlation IDs |
| Timeout Management | `socket.settimeout()` with try/catch blocks | `asyncio` with sophisticated timeout hierarchies |
| Memory Management | Manual `memset` equivalent for secret clearing | `mlock()`/`munlock()` for preventing swap |
| Alert Processing | Simple dictionary mapping alert codes to handlers | State machine with alert context tracking |
| Certificate Validation | `cryptography` library with basic validation | Full RFC 5280 compliance with policy engines |

**Recommended File Structure:**
```
https_client/
  transport/
    tcp_transport.py           ← TCP connection and timeout handling
    connection_errors.py       ← Network-level error definitions
  tls/
    alert_processor.py         ← TLS alert handling and generation
    crypto_errors.py           ← Cryptographic error definitions
    certificate_validator.py   ← Certificate validation with error handling
  utils/
    secure_memory.py           ← Memory clearing and secret management
    error_recovery.py          ← Retry logic and exponential backoff
    debug_logging.py           ← Structured logging for error analysis
```

**Complete TCP Transport Error Handling:**

```python
import socket
import time
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

class ConnectionErrorType(Enum):
    REFUSED = "connection_refused"
    TIMEOUT = "connection_timeout"
    NETWORK_UNREACHABLE = "network_unreachable"
    DNS_FAILURE = "dns_failure"
    RESET = "connection_reset"
    PARTIAL_READ_TIMEOUT = "partial_read_timeout"

@dataclass
class ConnectionError:
    error_type: ConnectionErrorType
    original_exception: Exception
    retry_recommended: bool
    security_implications: str

class TCPTransport:
    def __init__(self):
        self.socket: Optional[socket.socket] = None
        self.remote_address: Optional[Tuple[str, int]] = None
        self._receive_buffer = bytearray()
        self._logger = logging.getLogger(__name__)
    
    def connect(self, hostname: str, port: int, timeout: float) -> None:
        """Establish TCP connection with comprehensive error handling."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # TODO: Implement DNS resolution with error handling
                # TODO: Create socket with appropriate options (SO_KEEPALIVE, etc.)
                # TODO: Set socket timeout for connection attempt
                # TODO: Attempt connection with timeout
                # TODO: Configure socket options for TLS (TCP_NODELAY, etc.)
                # TODO: Log successful connection with timing metrics
                break
                
            except socket.gaierror as e:
                # DNS resolution failure
                error = ConnectionError(
                    ConnectionErrorType.DNS_FAILURE,
                    e, 
                    attempt < max_retries - 1,
                    "Possible DNS hijacking - validate with alternative resolution"
                )
                if not error.retry_recommended:
                    raise
                
            except ConnectionRefusedError as e:
                # Server not listening
                error = ConnectionError(
                    ConnectionErrorType.REFUSED,
                    e,
                    False,  # Don't retry refused connections
                    "None - legitimate server unavailability"
                )
                raise
                
            except socket.timeout as e:
                # Connection timeout
                error = ConnectionError(
                    ConnectionErrorType.TIMEOUT,
                    e,
                    attempt < max_retries - 1,
                    "Possible DoS attack on server - monitor retry patterns"
                )
                if not error.retry_recommended:
                    raise
                    
            # TODO: Add handling for other socket errors (ENETUNREACH, etc.)
            # TODO: Implement exponential backoff delay between retries
            # TODO: Log retry attempts with context
    
    def receive_exact(self, num_bytes: int) -> bytes:
        """Receive exactly the specified number of bytes with timeout handling."""
        # TODO: Implement buffered receive that accumulates data until num_bytes available
        # TODO: Handle partial reads by continuing to call recv() until complete
        # TODO: Implement per-read timeout to prevent hanging on partial data
        # TODO: Detect connection closure during multi-part reads
        # TODO: Validate that exactly num_bytes was received before returning
        pass
        
    def send_bytes(self, data: bytes) -> int:
        """Send raw bytes with comprehensive error handling."""
        # TODO: Handle partial sends by tracking bytes sent and continuing
        # TODO: Implement send timeout to prevent hanging on blocked sends
        # TODO: Handle connection reset during send operations
        # TODO: Return actual number of bytes sent for caller verification
        pass

    def close(self) -> None:
        """Close connection and clean up resources."""
        # TODO: Shutdown socket for both read and write
        # TODO: Close socket and handle any exceptions
        # TODO: Clear internal state and buffers
        # TODO: Log connection closure with timing and data transfer metrics
        pass
```

**TLS Alert Processing System:**

```python
from enum import IntEnum
from typing import Dict, Callable, Optional
import struct

class AlertLevel(IntEnum):
    WARNING = 1
    FATAL = 2

class AlertDescription(IntEnum):
    close_notify = 0
    unexpected_message = 10
    bad_record_mac = 20
    handshake_failure = 40
    bad_certificate = 42
    unsupported_certificate = 43
    certificate_revoked = 44
    certificate_expired = 45
    certificate_unknown = 46
    unknown_ca = 48
    decode_error = 50
    protocol_version = 70
    internal_error = 80

class AlertProcessor:
    def __init__(self, connection_state):
        self.connection_state = connection_state
        self._alert_handlers: Dict[AlertDescription, Callable] = {
            AlertDescription.close_notify: self._handle_close_notify,
            AlertDescription.bad_certificate: self._handle_certificate_error,
            AlertDescription.certificate_expired: self._handle_certificate_expired,
            # TODO: Add handlers for all alert types
        }
        self._logger = logging.getLogger(__name__)
    
    def process_alert(self, alert_record: bytes) -> bool:
        """Process incoming TLS alert and return whether connection should continue."""
        # TODO: Parse alert record to extract level and description
        # TODO: Validate alert record format and length
        # TODO: Log alert with full context (connection state, timing, etc.)
        # TODO: Look up appropriate handler for alert description
        # TODO: Execute handler and return its recommendation for connection continuation
        # TODO: Clear cryptographic state if alert is fatal
        pass
    
    def send_alert(self, level: AlertLevel, description: AlertDescription) -> None:
        """Send TLS alert to peer and take appropriate local action."""
        # TODO: Construct alert record with proper TLS record framing
        # TODO: Send alert record over TCP transport
        # TODO: Log outgoing alert with context
        # TODO: If alert is fatal, clear cryptographic state and close connection
        # TODO: Update connection state machine based on alert type
        pass
    
    def _handle_close_notify(self, alert_level: AlertLevel) -> bool:
        """Handle close_notify alert with proper acknowledgment."""
        # TODO: Send acknowledgment close_notify alert
        # TODO: Wait briefly for peer acknowledgment
        # TODO: Close TCP connection cleanly
        # TODO: Clear all cryptographic state
        # TODO: Return False to indicate connection should terminate
        pass
    
    def _handle_certificate_error(self, alert_level: AlertLevel) -> bool:
        """Handle certificate-related alerts."""
        # TODO: Log detailed certificate error information
        # TODO: Clear any cached certificate validation state
        # TODO: Return False for fatal certificate errors
        # TODO: For warnings, return based on security policy
        pass
```

**Cryptographic Error Handling:**

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import x25519
import os
import logging

class CryptoError(Exception):
    def __init__(self, message: str, security_impact: str, should_terminate: bool = True):
        super().__init__(message)
        self.security_impact = security_impact
        self.should_terminate = should_terminate

class SecureMemory:
    """Utilities for secure memory management and secret clearing."""
    
    @staticmethod
    def clear_bytes(data: bytearray) -> None:
        """Securely clear sensitive data from memory."""
        # TODO: Overwrite data with zeros
        # TODO: Use os.urandom() to overwrite with random data as second pass
        # TODO: Force memory sync to ensure clearing reaches physical memory
        pass

class X25519KeyExchange:
    def __init__(self):
        self.private_key: Optional[x25519.X25519PrivateKey] = None
        self.public_key: Optional[x25519.X25519PublicKey] = None
        self._logger = logging.getLogger(__name__)
    
    def compute_shared_secret(self, peer_public_key_bytes: bytes) -> bytes:
        """Compute ECDHE shared secret with comprehensive validation."""
        try:
            # TODO: Validate peer_public_key_bytes length (must be 32 bytes for X25519)
            # TODO: Create X25519PublicKey object from peer bytes
            # TODO: Validate that peer public key represents valid curve point
            # TODO: Perform ECDHE computation using private key
            # TODO: Validate shared secret is not all zeros or other weak values
            # TODO: Return shared secret for key derivation
            pass
            
        except ValueError as e:
            # Invalid public key format or weak point
            raise CryptoError(
                f"Invalid peer public key: {e}",
                "Weak shared secret could compromise all derived keys",
                should_terminate=True
            )
        except Exception as e:
            # Unexpected error during key exchange
            self._clear_key_material()
            raise CryptoError(
                f"Key exchange computation failed: {e}",
                "Unknown cryptographic failure - assume compromise",
                should_terminate=True
            )
    
    def _clear_key_material(self) -> None:
        """Clear all key material from memory."""
        # TODO: Clear private key if it exists
        # TODO: Clear public key if it exists  
        # TODO: Overwrite any intermediate computation values
        pass

class CertificateValidator:
    def __init__(self, trusted_roots: List[bytes]):
        self.trusted_roots = trusted_roots
        self._logger = logging.getLogger(__name__)
    
    def validate_certificate_chain(self, cert_chain: List[bytes], hostname: str) -> bool:
        """Comprehensive certificate validation with detailed error reporting."""
        try:
            # TODO: Parse each certificate in chain using cryptography library
            # TODO: Verify signature chain from leaf to root
            # TODO: Check certificate validity periods against current time
            # TODO: Validate hostname against certificate SAN extension
            # TODO: Check certificate revocation status if configured
            # TODO: Verify all certificates have required key usage extensions
            # TODO: Return True only if ALL validation steps pass
            pass
            
        except Exception as e:
            # Log detailed validation failure information
            self._logger.error(f"Certificate validation failed: {e}")
            self._logger.error(f"Hostname: {hostname}")
            self._logger.error(f"Chain length: {len(cert_chain)}")
            return False
```

**Milestone Checkpoints:**

After implementing each component, verify behavior with these checkpoints:

**Milestone 1 - Network Error Handling:**
```bash
# Test connection timeout handling
python test_transport.py --test connection_timeout --timeout 1
# Expected: ConnectionError with TIMEOUT type after 1 second

# Test partial read handling  
python test_transport.py --test partial_reads --fragment-size 100
# Expected: Complete records assembled from multiple TCP segments
```

**Milestone 2-4 - Cryptographic Error Handling:**
```bash
# Test certificate validation
python test_crypto.py --test invalid_certificate
# Expected: CryptoError with detailed validation failure information

# Test alert processing
python test_alerts.py --test close_notify
# Expected: Clean connection termination with acknowledgment
```

**Debugging Guide:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|-------------|-----------|-----|
| Connection hangs indefinitely | Missing timeout on receive_exact() | Check if recv() calls have timeouts | Add per-operation timeouts with bounds |
| Memory usage grows continuously | Not clearing secrets on errors | Monitor memory allocation patterns | Implement explicit secret clearing |
| Intermittent decryption failures | Sequence number overflow | Check sequence number bounds | Force renegotiation before overflow |
| Certificate errors for valid sites | System clock incorrect | Compare local time with HTTP Date headers | Sync system clock or adjust validation |


## Testing Strategy

> **Milestone(s):** All milestones (1-4) — comprehensive testing is essential for validating TLS implementation correctness, security, and interoperability

Testing a cryptographic protocol implementation is like conducting a multi-stage security audit of a high-security facility. Just as you would test every lock, alarm system, and access control independently before trusting the entire facility, you must validate each component of your TLS implementation in isolation and then verify their coordinated behavior. The stakes are particularly high because subtle bugs in cryptographic code can create security vulnerabilities that attackers can exploit years later.

The testing strategy for an HTTPS client requires a layered approach that mirrors the protocol's architecture. At the lowest level, you must verify that binary parsing and network communication work correctly. Moving up the stack, you need to validate cryptographic operations, certificate handling, and protocol state transitions. Finally, you must test the complete system against real-world servers to ensure interoperability.

Unlike testing typical application code, TLS testing requires both positive and negative test cases that verify not just correct behavior, but also proper handling of malicious or malformed inputs. An attacker might send crafted messages designed to cause buffer overflows, trigger timing attacks, or bypass certificate validation. Your test suite must defend against these scenarios.

### Milestone Validation Checkpoints

Think of milestone validation as a series of increasingly complex flight tests for a new aircraft. You start by testing individual components on the ground, then progress to taxi tests, short flights, and finally full operational scenarios. Each milestone builds confidence that the system will perform correctly under real-world conditions.

The validation approach follows a bottom-up strategy that tests each layer of the TLS stack independently before integrating them. This isolation is crucial because TLS bugs often manifest as subtle behavioral differences that become impossible to debug when multiple components interact simultaneously.

#### Milestone 1: TCP Socket & Record Layer Validation

The first milestone focuses on the fundamental communication primitives that everything else depends upon. Think of this as testing the postal system before sending important letters — you need confidence that messages arrive intact and in order.

**TCP Connection Establishment Tests**

The `TCPTransport` component must reliably establish connections to remote servers and handle various network conditions. Your test suite should verify behavior across different scenarios:

| Test Scenario | Expected Behavior | Validation Method | Common Failure Modes |
|---------------|-------------------|-------------------|----------------------|
| Successful connection to port 443 | `connect()` returns without error, socket state is connected | Verify `transport.socket` is not None, attempt to send/receive data | Connection timeout due to firewall rules |
| Connection to invalid hostname | `connect()` raises appropriate exception with hostname resolution details | Catch exception, verify error message contains hostname | Generic socket error without context |
| Connection timeout | `connect()` respects timeout parameter and raises timeout exception | Set short timeout, connect to blackhole IP, measure elapsed time | Indefinite blocking instead of timeout |
| Port 443 blocked by firewall | Connection attempt fails with connection refused or timeout | Verify exception type matches expected network error | Misleading error messages |

**TLS Record Layer Parsing Tests**

The `RecordLayer` component must correctly parse TLS records from TCP byte streams, handling fragmentation and reassembly. This is where many implementations fail because they assume complete records arrive in single TCP segments.

Your validation should test the record parsing pipeline with carefully crafted input data:

| Test Case | Input Data | Expected Output | Why This Matters |
|-----------|------------|-----------------|------------------|
| Complete handshake record | 5-byte header + payload matching length field | Correctly parsed `TLSRecord` with accurate `content_type`, `length`, `payload` | Baseline functionality check |
| Record split across TCP segments | Header in first segment, payload in second | Single complete record after reassembly | Fragmentation handling |
| Multiple records in single segment | Two complete records concatenated | Two separate `TLSRecord` objects | Prevents records from being merged |
| Partial record at end of segment | Complete record + partial header | Complete record parsed, partial data buffered for next segment | Prevents data corruption |
| Maximum size record (16384 bytes) | Record with maximum allowed payload | Successful parsing without buffer overflow | Large message handling |
| Oversized record (16385+ bytes) | Record exceeding maximum payload length | Parsing error or alert generation | Prevents buffer overflow attacks |

The critical validation point is testing `RecordLayer.add_tcp_data()` and `RecordLayer.get_next_record()` with carefully controlled input sequences. You should simulate various network conditions by feeding data to the record layer in different chunk sizes and timing patterns.

**Binary Parser Validation**

The `BinaryParser` utility functions form the foundation for all TLS message parsing. These functions must handle network byte order conversion and bounds checking correctly:

```
Validation Steps:
1. Test parse_uint16() with values 0x0000, 0x1234, 0xFFFF to verify big-endian conversion
2. Test parse_uint24() with maximum value 0xFFFFFF to verify 24-bit handling
3. Test parse_variable_bytes() with length prefixes of 1, 2, and 3 bytes
4. Test boundary conditions: parsing at end of buffer, insufficient data remaining
5. Verify all functions return correct (value, new_offset) tuples for chaining
```

**Milestone 1 Success Criteria**

Your implementation passes Milestone 1 when it can:
- Establish TCP connections to `google.com:443`, `cloudflare.com:443`, and `github.com:443`
- Parse TLS records from real server responses (capture with Wireshark, feed to your parser)
- Handle record fragmentation by splitting test data across multiple `add_tcp_data()` calls
- Process multiple record types (handshake, alert, application data) from the same connection

#### Milestone 2: ClientHello Message Construction Validation

The second milestone validates your ability to construct properly formatted TLS handshake messages that real servers will accept. Think of this as ensuring your diplomatic credentials are properly formatted and contain all required information.

**ClientHello Message Format Validation**

The `HandshakeEngine` must construct ClientHello messages that comply with TLS specifications and include all necessary extensions for modern servers:

| Component | Validation Test | Expected Result | Common Implementation Bugs |
|-----------|----------------|-----------------|---------------------------|
| Random field generation | Generate 100 ClientHello messages, verify all random fields are different | Zero duplicate random values | Using timestamp or weak PRNG |
| Cipher suite ordering | Include TLS_AES_128_GCM_SHA256 as first cipher suite | Cipher suite list starts with 0x1301 | Wrong byte order or missing required suite |
| Extension format | Add SNI extension with hostname "example.com" | Extension type 0x0000, length field, hostname in correct format | Incorrect length calculations |
| Message length calculation | Construct complete ClientHello, verify length field matches payload | Length field matches actual message bytes | Off-by-one errors in length calculation |

**Server Name Indication (SNI) Extension Testing**

Modern HTTPS requires proper SNI extension formatting to support virtual hosting. Your validation must ensure the extension includes the target hostname in the correct format:

```
SNI Extension Validation Steps:
1. Call add_server_name_extension("www.example.com")
2. Serialize ClientHello to bytes using to_bytes()
3. Parse the extension list and locate extension type 0x0000
4. Verify the extension contains server_name_list with name_type=0 (hostname)
5. Verify hostname bytes match "www.example.com" in UTF-8 encoding
6. Confirm extension length field correctly spans the entire extension payload
```

**Key Share Extension Validation**

The key share extension advertises the client's ephemeral public key for ECDHE key exchange. This extension must be properly formatted or the handshake will fail:

| Test Aspect | Validation Method | Success Criteria | Failure Symptoms |
|-------------|-------------------|------------------|------------------|
| Named group support | Add key share with X25519 (0x001D) | Extension includes group 0x001D with 32-byte key share | Server responds with unsupported_group alert |
| Public key format | Generate X25519 key pair, extract public key bytes | Key share contains exactly 32 bytes of valid curve point | Handshake failure or invalid key share alert |
| Extension ordering | Add multiple extensions, verify order preservation | Extensions appear in same order as added | Some servers sensitive to extension ordering |

**Wire Format Compatibility Testing**

The ultimate validation for Milestone 2 is testing against real servers. Your ClientHello must be accepted by production TLS servers:

```
Real Server Testing Process:
1. Connect to www.google.com:443 via TCP
2. Send your constructed ClientHello wrapped in TLS record
3. Receive ServerHello response (don't worry about processing it yet)
4. Verify server responds with ServerHello (not alert)
5. Repeat test with www.cloudflare.com:443 and github.com:443
6. If any server sends alert, analyze the alert description for format issues
```

**Milestone 2 Success Criteria**

Your implementation passes Milestone 2 when:
- ClientHello messages are accepted by at least 3 different production servers
- SNI extension correctly specifies target hostname
- Key share extension includes valid X25519 public key
- Message serialization produces byte-for-byte identical output on repeated runs (except random field)

#### Milestone 3: ECDHE Key Exchange and Certificate Validation

The third milestone validates the most complex cryptographic operations in the TLS handshake. Think of this as verifying that your secure communication channel can actually establish shared secrets and verify the identity of the remote party.

**ECDHE Key Exchange Validation**

The `X25519KeyExchange` component must generate valid key pairs, compute shared secrets, and integrate with the key derivation process:

| Operation | Test Vectors | Validation Method | Security Considerations |
|-----------|-------------|-------------------|------------------------|
| Key pair generation | Generate 1000 key pairs | Verify all public keys are different, private keys are 32 bytes | Weak randomness creates predictable keys |
| Shared secret computation | Use RFC 7748 test vectors | Compute shared secret, compare with expected output | Implementation bugs cause different shared secrets |
| Invalid public key rejection | Try all-zero, all-FF, invalid curve points | Key exchange should reject invalid inputs | Accepting invalid keys can compromise security |
| Shared secret validation | Exchange with known good implementation | Both sides compute identical shared secret | Subtle implementation differences |

**Key Derivation Function (HKDF) Testing**

The `HKDF` implementation must produce cryptographically strong keys that match the TLS 1.3 specification:

```
HKDF Test Vector Validation:
1. Use test vectors from RFC 5869 to validate HKDF-Extract and HKDF-Expand
2. Test TLS 1.3 specific derivation with known handshake context
3. Verify derive_handshake_secrets() produces correct client/server traffic secrets
4. Test derive_application_secrets() with complete handshake transcript
5. Validate traffic key derivation produces different keys for client/server directions
```

The key derivation process must be tested with known test vectors to ensure compatibility with other TLS implementations. Even small bugs in key derivation will cause authentication failures that are difficult to debug.

**Certificate Validation Testing**

The `CertificateValidator` must correctly verify X.509 certificate chains and validate hostname matching. This is critical for preventing man-in-the-middle attacks:

| Validation Scenario | Test Setup | Expected Result | Security Impact |
|--------------------|------------|-----------------|-----------------|
| Valid certificate chain | Use real certificate from www.google.com | Validation succeeds, hostname matches | Baseline functionality |
| Expired certificate | Create certificate with past expiration date | Validation fails with expiration error | Prevents use of expired certificates |
| Wrong hostname | Valid certificate for different hostname | Validation fails with hostname mismatch | Prevents MITM with valid but wrong certificate |
| Self-signed certificate | Certificate signed by itself, not trusted CA | Validation fails with trust chain error | Prevents acceptance of untrusted certificates |
| Invalid signature | Certificate with corrupted signature | Validation fails with signature error | Prevents forged certificates |

**Finished Message Validation**

The Finished message provides cryptographic proof that both parties possess the correct traffic keys and have received the same handshake messages:

```
Finished Message Testing Process:
1. Complete handshake through Certificate and CertificateVerify messages
2. Compute handshake context hash of all received messages
3. Calculate expected verify data using compute_finished_verify_data()
4. Compare received Finished message verify data with computed value
5. Verify byte-for-byte match (any difference indicates handshake failure)
```

**Milestone 3 Success Criteria**

Your implementation passes Milestone 3 when it can:
- Complete full TLS handshake with real servers (receive their Finished message)
- Validate certificates from major websites (Google, Cloudflare, GitHub)
- Derive identical traffic keys as reference implementations (test with known vectors)
- Generate and verify Finished messages that servers accept

#### Milestone 4: Encrypted Application Data Validation

The final milestone validates end-to-end encrypted communication, ensuring that your HTTPS client can securely exchange application data with real servers. Think of this as the final certification that your secure communication system works in production environments.

**AEAD Encryption/Decryption Testing**

The `AEADCipher` and sequence number handling must provide confidentiality and integrity for all application data:

| Test Scenario | Input Data | Validation Steps | Expected Outcome |
|---------------|------------|------------------|------------------|
| Basic encryption/decryption | "Hello, World!" plaintext | Encrypt with sequence 0, decrypt result | Recovered plaintext matches original |
| Sequence number progression | Multiple messages in order | Verify nonces increment, no nonce reuse | Each message uses unique nonce |
| Authentication tag verification | Tampered ciphertext | Attempt decryption of modified data | Decryption fails with authentication error |
| Large message handling | 16KB application data record | Encrypt and decrypt maximum size record | Successful processing without buffer issues |

**HTTP Request/Response Processing**

The `HTTPProcessor` must construct valid HTTP requests and parse server responses correctly:

```
HTTP Processing Validation:
1. Build GET request for "/" with Host header
2. Send over encrypted TLS connection
3. Receive encrypted HTTP response
4. Decrypt application data and parse HTTP response
5. Verify response contains expected status line, headers, and body
6. Test with different HTTP methods (GET, POST) and various content types
```

**Connection Lifecycle Management**

Complete connection lifecycle testing verifies that your client properly establishes, uses, and terminates TLS connections:

| Lifecycle Phase | Validation Test | Success Criteria | Common Issues |
|-----------------|----------------|------------------|---------------|
| Handshake completion | Full handshake with real server | Receive server Finished, transition to CONNECTED state | State machine transitions |
| Application data exchange | Send HTTP request, receive response | Successful encryption/decryption of both directions | Sequence number desynchronization |
| Graceful closure | Send close_notify alert | Server responds with close_notify, connection closes | Truncation attack vulnerability |
| Error recovery | Handle server alerts | Appropriate error handling, connection cleanup | Resource leaks |

**Real-World Server Testing**

The ultimate validation is testing against diverse production servers with different TLS configurations:

```
Production Server Test Suite:
1. www.google.com - Test against Google's server configuration
2. www.cloudflare.com - Test Cloudflare's edge server implementation
3. github.com - Test GitHub's certificate and TLS setup
4. badssl.com/good/ - Test against known-good configuration
5. expired.badssl.com - Verify proper handling of expired certificates
6. wrong.host.badssl.com - Verify hostname validation rejects mismatches
```

Each server may have slightly different TLS implementations, certificate chains, and supported features. Your client must interoperate correctly with all of them.

**Milestone 4 Success Criteria**

Your implementation passes Milestone 4 when it can:
- Complete HTTPS requests to major websites and receive correct responses
- Handle various HTTP response types (redirects, different content types, large responses)
- Properly encrypt and decrypt application data with sequence number handling
- Gracefully close connections and handle server-initiated closure

### Integration Testing Approach

Think of integration testing as conducting full-scale exercises that simulate real-world conditions. While unit tests verify individual components work correctly, integration tests ensure that components coordinate properly and handle the complex interactions that occur during actual TLS sessions.

Integration testing for TLS requires a multi-layered approach that validates both positive scenarios (successful handshakes and data exchange) and negative scenarios (handling various failure modes). The testing must also account for the stateful nature of TLS connections and the precise timing requirements of the handshake protocol.

#### Test Vector Validation

Cryptographic protocols require testing against known test vectors to ensure compatibility with the broader ecosystem. Test vectors provide predetermined inputs and expected outputs that allow you to validate your implementation against reference implementations.

**TLS 1.3 Handshake Test Vectors**

The TLS 1.3 specification includes comprehensive test vectors that cover the complete handshake process. Your testing should validate each cryptographic operation against these vectors:

| Test Vector Category | Source | Validation Target | Implementation Component |
|---------------------|--------|-------------------|-------------------------|
| HKDF test vectors | RFC 5869 | Key derivation functions | `HKDF.extract()`, `HKDF.expand()` |
| X25519 test vectors | RFC 7748 | ECDHE key exchange | `X25519KeyExchange.compute_shared_secret()` |
| AES-GCM test vectors | NIST test vectors | AEAD encryption/decryption | `AEADCipher.encrypt()`, `AEADCipher.decrypt()` |
| TLS 1.3 handshake vectors | TLS 1.3 specification | Complete handshake flow | `HandshakeEngine`, `CryptoEngine` coordination |

**Message Format Compatibility Testing**

Your TLS implementation must produce wire-format messages that other implementations can process. This requires byte-level compatibility testing:

```
Wire Format Validation Process:
1. Capture real TLS handshake using Wireshark from a known-good client
2. Extract each handshake message (ClientHello, ServerHello, etc.)
3. Parse captured messages with your implementation
4. Generate equivalent messages with your code
5. Compare generated messages byte-for-byte with captured originals
6. Focus on extension ordering, length calculations, and padding
```

The goal is not perfect byte-for-byte reproduction (some fields like random values will differ), but rather ensuring that your messages are semantically equivalent and will be accepted by real servers.

#### Real Server Testing Matrix

Testing against real servers provides the ultimate validation that your implementation works in production environments. However, different servers have varying configurations, supported features, and edge case behaviors that your client must handle robustly.

**Server Diversity Testing**

Your test matrix should include servers with different characteristics to ensure broad compatibility:

| Server Category | Example Servers | Unique Characteristics | Testing Focus |
|----------------|-----------------|----------------------|---------------|
| CDN providers | Cloudflare, Amazon CloudFront | Edge server implementations, certificate rotation | Connection establishment, certificate validation |
| Major websites | Google, Facebook, GitHub | High-security configurations, HSTS enforcement | Modern TLS features, strict certificate validation |
| Banking/Financial | Bank websites, payment processors | Conservative TLS configurations, extended validation certificates | Older TLS versions, complex certificate chains |
| Government | .gov websites | FIPS compliance requirements, specific cipher suite preferences | Restricted cryptographic algorithms |
| Test servers | badssl.com, howsmyssl.com | Deliberately misconfigured scenarios | Error handling, edge case behavior |

**Automated Integration Test Suite**

Your integration testing should be automated to run continuously as you develop and refactor your implementation:

```
Automated Test Pipeline:
1. Network connectivity test - verify test environment can reach target servers
2. Handshake completion test - complete TLS handshake with each target server
3. HTTP request test - send GET request, verify response received
4. Certificate validation test - verify certificate chain validation works
5. Error condition test - test with invalid certificates, expired certs, etc.
6. Performance test - measure handshake time and throughput
7. Memory usage test - verify no memory leaks during repeated connections
```

The automated suite should run after every significant code change to catch regressions early. Integration test failures often indicate subtle bugs that only manifest under specific conditions.

**Certificate Validation Testing**

Certificate validation is one of the most critical security functions in your HTTPS client. Your testing must cover various certificate scenarios that attackers might exploit:

| Certificate Scenario | Test Server | Expected Behavior | Security Rationale |
|---------------------|-------------|-------------------|-------------------|
| Valid certificate | badssl.com/good/ | Connection succeeds, certificate validation passes | Baseline functionality |
| Expired certificate | expired.badssl.com | Connection fails, certificate validation rejects expired cert | Prevents use of expired certificates |
| Wrong hostname | wrong.host.badssl.com | Connection fails, hostname mismatch detected | Prevents MITM attacks with valid but wrong certificates |
| Self-signed certificate | self-signed.badssl.com | Connection fails, untrusted certificate chain | Prevents acceptance of attacker-controlled certificates |
| Incomplete chain | incomplete-chain.badssl.com | Connection fails or succeeds with proper chain building | Tests certificate chain construction logic |

> **Critical Security Insight**: Certificate validation testing must include both positive and negative test cases. A client that accepts all certificates is worse than no TLS at all because it provides a false sense of security.

**Error Handling Integration Tests**

Real-world network conditions include various failure modes that your client must handle gracefully. Your integration testing should simulate these conditions:

```
Network Error Simulation:
1. Connection timeout - use iptables to drop packets to specific servers
2. Connection reset - close TCP connection during TLS handshake
3. Partial data - simulate network congestion causing fragmented records
4. TLS alerts - connect to servers configured to send specific alert messages
5. Certificate errors - test with revoked certificates, unknown CA, etc.
```

The goal is to verify that your error handling code actually works under realistic failure conditions, not just in unit tests with mocked failures.

#### Performance and Scalability Testing

While correctness is the primary concern for a learning project, understanding the performance characteristics of your TLS implementation provides valuable insights into the protocol's computational requirements and bottlenecks.

**Handshake Performance Measurement**

TLS handshakes involve significant cryptographic computation that affects user experience. Your testing should measure handshake performance across different scenarios:

| Performance Metric | Measurement Method | Typical Values | Performance Factors |
|-------------------|-------------------|----------------|-------------------|
| Handshake latency | Time from ClientHello to application data ready | 100-500ms including network RTT | Network latency, certificate validation time |
| CPU time per handshake | Process CPU time for cryptographic operations | 1-10ms | Key exchange computation, certificate verification |
| Memory usage per connection | Peak memory during handshake and data exchange | 10-50KB per connection | Certificate chain storage, cryptographic state |
| Concurrent connections | Number of simultaneous TLS connections supported | Limited by memory and file descriptors | Connection state storage, OS limits |

**Cryptographic Operation Profiling**

Understanding which cryptographic operations consume the most resources helps identify optimization opportunities:

```
Profiling Test Process:
1. Instrument each cryptographic function with timing measurements
2. Run 100 complete handshakes against a test server
3. Collect timing data for: key generation, shared secret computation, HKDF operations, AES-GCM encryption/decryption, certificate verification
4. Identify bottlenecks and compare with optimized cryptographic libraries
5. Measure impact of different cipher suites and key sizes
```

This profiling helps understand the computational cost of security and can guide decisions about caching, connection reuse, and cipher suite preferences in production systems.

### Implementation Guidance

The testing strategy for your HTTPS client requires a combination of unit tests, integration tests, and real-world validation. The following guidance provides concrete tools and approaches for implementing comprehensive testing.

**Technology Recommendations**

| Testing Category | Simple Option | Advanced Option |
|------------------|---------------|-----------------|
| Unit Testing Framework | Python unittest (built-in) | pytest with fixtures and parameterization |
| Network Testing | Manual server connections | Docker containers with nginx/OpenSSL for controlled testing |
| Binary Data Validation | Manual hex comparison | Scapy for packet crafting and validation |
| Performance Testing | Simple timing with time.time() | cProfile for detailed performance analysis |
| Test Vector Management | Hardcoded test data in test files | JSON test vector files with automated loading |

**Test Project Structure**

Organize your tests to mirror the component structure and make it easy to run targeted test suites:

```
https-client/
├── src/
│   ├── transport/
│   │   ├── tcp_transport.py
│   │   └── record_layer.py
│   ├── handshake/
│   │   ├── client_hello.py
│   │   └── handshake_engine.py
│   ├── crypto/
│   │   ├── key_exchange.py
│   │   ├── hkdf.py
│   │   └── aead_cipher.py
│   └── https_client.py
├── tests/
│   ├── unit/
│   │   ├── test_tcp_transport.py
│   │   ├── test_record_layer.py
│   │   ├── test_client_hello.py
│   │   ├── test_key_exchange.py
│   │   └── test_certificate_validator.py
│   ├── integration/
│   │   ├── test_handshake_flow.py
│   │   ├── test_real_servers.py
│   │   └── test_error_conditions.py
│   ├── vectors/
│   │   ├── tls13_test_vectors.json
│   │   ├── hkdf_test_vectors.json
│   │   └── certificate_chains.pem
│   └── helpers/
│       ├── test_servers.py
│       ├── packet_capture.py
│       └── crypto_test_utils.py
```

**Unit Test Infrastructure**

Here's complete infrastructure code for testing binary parsing and record layer functionality:

```python
import unittest
import socket
from typing import List, Tuple
from unittest.mock import Mock, patch
import os
import json

class TLSTestCase(unittest.TestCase):
    """Base test case with TLS-specific test utilities."""
    
    def setUp(self):
        """Set up test fixtures for TLS testing."""
        self.test_vectors_dir = os.path.join(os.path.dirname(__file__), '..', 'vectors')
        self.sample_records = self._load_sample_records()
    
    def _load_sample_records(self) -> List[bytes]:
        """Load sample TLS records for testing."""
        # ClientHello record from real capture
        client_hello = bytes.fromhex(
            "16 03 01 00 f4"  # TLS record header
            "01 00 00 f0"     # ClientHello header
            "03 03"           # TLS version
            "00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f"  # Random (16 bytes)
            "10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f"  # Random (remaining 16)
            "00"              # Session ID length
            "00 02 13 01"     # Cipher suites (TLS_AES_128_GCM_SHA256)
            "01 00"           # Compression methods
        )
        
        # ServerHello record
        server_hello = bytes.fromhex(
            "16 03 03 00 5a"  # TLS record header
            "02 00 00 56"     # ServerHello header
            "03 03"           # TLS version
        )
        
        return [client_hello, server_hello]
    
    def assert_bytes_equal(self, expected: bytes, actual: bytes, message: str = ""):
        """Compare bytes with helpful hex dump on failure."""
        if expected != actual:
            expected_hex = expected.hex(' ')
            actual_hex = actual.hex(' ')
            self.fail(f"{message}\nExpected: {expected_hex}\nActual:   {actual_hex}")
    
    def create_fragmented_data(self, complete_data: bytes, fragment_sizes: List[int]) -> List[bytes]:
        """Split complete data into fragments for testing reassembly."""
        fragments = []
        offset = 0
        for size in fragment_sizes:
            if offset >= len(complete_data):
                break
            end_offset = min(offset + size, len(complete_data))
            fragments.append(complete_data[offset:end_offset])
            offset = end_offset
        return fragments

class MockTCPTransport:
    """Mock TCP transport for testing without actual network connections."""
    
    def __init__(self, response_data: bytes = b""):
        self.sent_data = bytearray()
        self.response_data = response_data
        self.response_offset = 0
        self.connected = False
        
    def connect(self, hostname: str, port: int, timeout: float) -> None:
        if hostname == "fail.test":
            raise ConnectionRefusedError("Connection refused")
        self.connected = True
    
    def send_bytes(self, data: bytes) -> int:
        if not self.connected:
            raise OSError("Not connected")
        self.sent_data.extend(data)
        return len(data)
    
    def receive_bytes(self, max_bytes: int) -> bytes:
        if not self.connected:
            raise OSError("Not connected")
        
        if self.response_offset >= len(self.response_data):
            return b""  # No more data
            
        end_offset = min(self.response_offset + max_bytes, len(self.response_data))
        data = self.response_data[self.response_offset:end_offset]
        self.response_offset = end_offset
        return data
    
    def get_sent_data(self) -> bytes:
        """Get all data that was sent through this transport."""
        return bytes(self.sent_data)

class CryptoTestVectors:
    """Utility class for loading and managing cryptographic test vectors."""
    
    @staticmethod
    def load_hkdf_vectors() -> List[dict]:
        """Load HKDF test vectors from RFC 5869."""
        return [
            {
                "IKM": bytes.fromhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b"),
                "salt": bytes.fromhex("000102030405060708090a0b0c"),
                "info": bytes.fromhex("f0f1f2f3f4f5f6f7f8f9"),
                "L": 42,
                "PRK": bytes.fromhex("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"),
                "OKM": bytes.fromhex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865")
            }
        ]
    
    @staticmethod
    def load_x25519_vectors() -> List[dict]:
        """Load X25519 test vectors from RFC 7748."""
        return [
            {
                "private": bytes.fromhex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a"),
                "public": bytes.fromhex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a"),
                "peer_public": bytes.fromhex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f"),
                "shared_secret": bytes.fromhex("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742")
            }
        ]
```

**Integration Test Framework**

Complete integration test infrastructure for testing against real servers:

```python
import unittest
import time
import ssl
import socket
from typing import Dict, List, Tuple
import threading
import queue

class IntegrationTestSuite(unittest.TestCase):
    """Integration tests against real TLS servers."""
    
    # Test servers with different characteristics
    TEST_SERVERS = [
        {"hostname": "www.google.com", "port": 443, "category": "major_website"},
        {"hostname": "github.com", "port": 443, "category": "developer_platform"},
        {"hostname": "badssl.com", "port": 443, "category": "test_server"},
        {"hostname": "expired.badssl.com", "port": 443, "category": "invalid_cert", "should_fail": True},
        {"hostname": "wrong.host.badssl.com", "port": 443, "category": "hostname_mismatch", "should_fail": True}
    ]
    
    def setUp(self):
        """Set up integration test environment."""
        self.client = None  # Will be initialized with your HTTPSClient
        self.test_results = []
        
    def test_handshake_with_real_servers(self):
        """Test TLS handshake completion with real servers."""
        for server in self.TEST_SERVERS:
            with self.subTest(hostname=server["hostname"]):
                should_fail = server.get("should_fail", False)
                
                try:
                    # TODO: Initialize your HTTPSClient here
                    # client = HTTPSClient()
                    # success = client.perform_handshake(server["hostname"], server["port"])
                    
                    success = self._test_handshake(server["hostname"], server["port"])
                    
                    if should_fail:
                        self.assertFalse(success, f"Expected handshake to fail for {server['hostname']}")
                    else:
                        self.assertTrue(success, f"Handshake failed for {server['hostname']}")
                        
                except Exception as e:
                    if should_fail:
                        # Expected failure
                        pass
                    else:
                        self.fail(f"Unexpected exception with {server['hostname']}: {e}")
    
    def _test_handshake(self, hostname: str, port: int) -> bool:
        """Test handshake with a specific server."""
        # TODO: Replace with your actual implementation
        # This is a placeholder that uses Python's ssl module for comparison
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    return True
        except Exception:
            return False
    
    def test_http_requests(self):
        """Test complete HTTP request/response cycle."""
        test_requests = [
            {"hostname": "httpbin.org", "path": "/get", "method": "GET"},
            {"hostname": "httpbin.org", "path": "/post", "method": "POST", "body": b'{"test": true}'},
        ]
        
        for req in test_requests:
            with self.subTest(url=f"{req['hostname']}{req['path']}"):
                # TODO: Implement with your HTTPSClient
                # client = HTTPSClient()
                # client.perform_handshake(req["hostname"], 443)
                # status, headers, body = client.request(req["method"], req["path"], {}, req.get("body", b""))
                # self.assertEqual(status, 200)
                pass
    
    def test_certificate_validation_scenarios(self):
        """Test certificate validation with various certificate conditions."""
        cert_tests = [
            {"hostname": "badssl.com", "expected_valid": True},
            {"hostname": "expired.badssl.com", "expected_valid": False, "expected_error": "certificate_expired"},
            {"hostname": "wrong.host.badssl.com", "expected_valid": False, "expected_error": "hostname_mismatch"},
            {"hostname": "self-signed.badssl.com", "expected_valid": False, "expected_error": "unknown_ca"},
        ]
        
        for test in cert_tests:
            with self.subTest(hostname=test["hostname"]):
                # TODO: Test certificate validation
                # validator = CertificateValidator()
                # result = validator.validate_certificate_chain(cert_chain, test["hostname"])
                # self.assertEqual(result, test["expected_valid"])
                pass

class PerformanceTestSuite(unittest.TestCase):
    """Performance and scalability tests."""
    
    def test_handshake_performance(self):
        """Measure TLS handshake performance."""
        hostname = "www.google.com"
        port = 443
        num_trials = 10
        
        handshake_times = []
        
        for i in range(num_trials):
            start_time = time.time()
            
            # TODO: Perform handshake with your implementation
            # client = HTTPSClient()
            # client.perform_handshake(hostname, port)
            
            end_time = time.time()
            handshake_times.append(end_time - start_time)
        
        avg_time = sum(handshake_times) / len(handshake_times)
        max_time = max(handshake_times)
        
        # Performance assertions (adjust based on your environment)
        self.assertLess(avg_time, 2.0, "Average handshake time too slow")
        self.assertLess(max_time, 5.0, "Maximum handshake time too slow")
        
        print(f"Handshake performance: avg={avg_time:.3f}s, max={max_time:.3f}s")
    
    def test_concurrent_connections(self):
        """Test handling multiple concurrent TLS connections."""
        hostname = "httpbin.org"
        port = 443
        num_connections = 10
        
        results = queue.Queue()
        
        def worker():
            try:
                # TODO: Create and test connection
                # client = HTTPSClient()
                # success = client.perform_handshake(hostname, port)
                # results.put(("success", success))
                results.put(("success", True))
            except Exception as e:
                results.put(("error", str(e)))
        
        # Start concurrent connections
        threads = []
        for i in range(num_connections):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all connections to complete
        for t in threads:
            t.join(timeout=30)
        
        # Collect results
        successes = 0
        errors = []
        
        while not results.empty():
            result_type, result_value = results.get()
            if result_type == "success" and result_value:
                successes += 1
            else:
                errors.append(result_value)
        
        # Verify most connections succeeded
        success_rate = successes / num_connections
        self.assertGreaterEqual(success_rate, 0.8, f"Success rate too low: {success_rate}")
        
        if errors:
            print(f"Connection errors: {errors}")
```

**Milestone Validation Scripts**

Create specific validation scripts for each milestone:

```python
#!/usr/bin/env python3
"""
Milestone validation script - run after completing each milestone
"""

import sys
import importlib
from typing import Dict, List, Callable

class MilestoneValidator:
    """Validates completion of project milestones."""
    
    def __init__(self):
        self.validators: Dict[int, List[Callable]] = {
            1: [self.validate_tcp_connection, self.validate_record_parsing],
            2: [self.validate_client_hello, self.validate_extensions],
            3: [self.validate_key_exchange, self.validate_certificate_validation],
            4: [self.validate_application_data, self.validate_http_requests]
        }
    
    def validate_milestone(self, milestone_num: int) -> bool:
        """Validate a specific milestone completion."""
        print(f"\n=== Validating Milestone {milestone_num} ===")
        
        if milestone_num not in self.validators:
            print(f"Unknown milestone: {milestone_num}")
            return False
        
        validators = self.validators[milestone_num]
        all_passed = True
        
        for validator in validators:
            try:
                result = validator()
                status = "PASS" if result else "FAIL"
                print(f"  {validator.__name__}: {status}")
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"  {validator.__name__}: ERROR - {e}")
                all_passed = False
        
        overall_status = "PASS" if all_passed else "FAIL"
        print(f"\nMilestone {milestone_num}: {overall_status}")
        return all_passed
    
    def validate_tcp_connection(self) -> bool:
        """Validate TCP connection capability."""
        # TODO: Test your TCPTransport implementation
        # transport = TCPTransport()
        # transport.connect("www.google.com", 443, 10.0)
        # return transport.socket is not None
        return True  # Placeholder
    
    def validate_record_parsing(self) -> bool:
        """Validate TLS record layer parsing."""
        # TODO: Test your RecordLayer implementation
        # record_layer = RecordLayer()
        # # Add test data and verify parsing
        return True  # Placeholder
    
    def validate_client_hello(self) -> bool:
        """Validate ClientHello message construction."""
        # TODO: Test your ClientHello implementation
        return True  # Placeholder
    
    def validate_extensions(self) -> bool:
        """Validate TLS extension handling."""
        # TODO: Test SNI, key share extensions
        return True  # Placeholder
    
    def validate_key_exchange(self) -> bool:
        """Validate ECDHE key exchange."""
        # TODO: Test X25519 key exchange
        return True  # Placeholder
    
    def validate_certificate_validation(self) -> bool:
        """Validate certificate validation logic."""
        # TODO: Test certificate validation
        return True  # Placeholder
    
    def validate_application_data(self) -> bool:
        """Validate encrypted application data handling."""
        # TODO: Test AEAD encryption/decryption
        return True  # Placeholder
    
    def validate_http_requests(self) -> bool:
        """Validate HTTP request/response processing."""
        # TODO: Test complete HTTPS request
        return True  # Placeholder

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_milestone.py <milestone_number>")
        sys.exit(1)
    
    try:
        milestone = int(sys.argv[1])
    except ValueError:
        print("Milestone number must be an integer")
        sys.exit(1)
    
    validator = MilestoneValidator()
    success = validator.validate_milestone(milestone)
    sys.exit(0 if success else 1)
```

**Test Execution Commands**

Create a simple test runner that executes the appropriate tests for each development phase:

```bash
#!/bin/bash
# test_runner.sh - Execute tests for HTTPS client implementation

set -e

echo "HTTPS Client Test Suite"
echo "======================"

# Unit tests
echo "Running unit tests..."
python -m pytest tests/unit/ -v

# Integration tests (if network available)
if ping -c 1 google.com &> /dev/null; then
    echo "Running integration tests..."
    python -m pytest tests/integration/ -v
else
    echo "Skipping integration tests (no network connectivity)"
fi

# Milestone validation
if [ $# -eq 1 ]; then
    echo "Validating milestone $1..."
    python tests/validate_milestone.py $1
fi

echo "Test execution complete!"
```

This comprehensive testing strategy ensures that your HTTPS client implementation is correct, secure, and interoperable with real-world TLS servers. The layered approach catches bugs early and provides confidence that your implementation will work reliably in production environments.


## Debugging Guide

> **Milestone(s):** All milestones (1-4) — debugging skills are essential throughout TLS implementation from TCP connection establishment through encrypted communication

Think of debugging a TLS implementation as being a detective investigating a complex case where the evidence is scattered across network packets, cryptographic computations, and state machine transitions. Unlike debugging typical application code, TLS debugging requires understanding multiple layers simultaneously — the network transport, binary protocol parsing, cryptographic operations, and security state management. Each bug can manifest at one layer while having its root cause in a completely different layer, making systematic investigation crucial.

The complexity of TLS debugging stems from the protocol's layered security model. A single malformed byte in a ClientHello message can cascade into handshake failures, while subtle timing issues in key derivation can cause intermittent decryption errors that appear completely random. The binary nature of TLS messages means that small mistakes often produce completely garbled output rather than obviously incorrect results, requiring careful examination of wire-format data.

Furthermore, TLS debugging involves security-sensitive operations where logging too much information can create security vulnerabilities, while logging too little makes diagnosis impossible. The state-dependent nature of the protocol means that the same message can be valid or invalid depending on the current connection state, requiring deep understanding of the TLS state machine to interpret observations correctly.

### Common Implementation Bugs

Understanding typical TLS implementation bugs is crucial because they follow predictable patterns. The layered nature of TLS means bugs often manifest far from their root cause, requiring systematic investigation techniques to trace symptoms back to their origins.

#### Binary Protocol Parsing Errors

Binary protocol parsing represents the most common source of TLS implementation bugs because even single-byte mistakes can cause catastrophic failures. The network byte order requirements and variable-length field encoding create numerous opportunities for subtle errors.

> **Decision: Systematic Binary Parsing Validation**
> - **Context**: Binary parsing errors are difficult to debug because they often cause failures in downstream components rather than at the parsing site itself
> - **Options Considered**: Ad-hoc debugging, comprehensive parsing validation, binary diff tools
> - **Decision**: Implement systematic validation with hex dump analysis and boundary checking
> - **Rationale**: Early detection of parsing errors prevents cascading failures and provides clear diagnostic information
> - **Consequences**: Requires more upfront validation code but dramatically reduces debugging time for protocol-level issues

| Symptom | Root Cause | Diagnosis Steps | Fix |
|---------|------------|-----------------|-----|
| ClientHello appears truncated in server logs | Incorrect length field calculation in message construction | 1. Hex dump the raw ClientHello bytes 2. Manually calculate expected length fields 3. Compare with actual message structure | Verify all length calculations use correct field sizes and include all nested structures |
| Server rejects ClientHello with "decode error" | Endianness bug in multi-byte field parsing | 1. Capture ClientHello with Wireshark 2. Compare byte order in sent vs expected format 3. Check `parse_uint16` and `parse_uint24` implementations | Ensure all multi-byte integers use network byte order (big-endian) |
| Extension parsing fails with "malformed extension" | Off-by-one error in extension boundary calculation | 1. Print extension start/end offsets during parsing 2. Verify total extension length matches sum of individual extensions 3. Check for missing length field updates | Add boundary validation after parsing each extension |
| Record layer reports "invalid record type" | Incorrect content type byte in record header | 1. Hex dump first 5 bytes of each sent record 2. Verify content type constants match RFC values 3. Check record construction logic | Use correct `ContentType` constants: `HANDSHAKE` (22), `APPLICATION_DATA` (23) |
| "Record too large" errors during handshake | Length field includes header length when it shouldn't | 1. Check if record length calculation includes the 5-byte header 2. Verify `MAX_RECORD_PAYLOAD_LENGTH` enforcement 3. Compare with specification requirements | Record length field must contain only payload length, not including header |

⚠️ **Pitfall: Assuming ASCII Text Debugging**  
Many developers try to print TLS binary data as strings, which produces garbage output. TLS records contain arbitrary binary data that must be examined as hexadecimal bytes. Always use `hex_dump()` functions to visualize binary protocol data, showing both hex representation and ASCII interpretation where meaningful.

#### Cryptographic Operation Failures

Cryptographic bugs in TLS implementations are particularly insidious because they often produce plausible-looking but incorrect results. The deterministic nature of cryptographic operations means that bugs typically cause consistent failures, but the complexity of key derivation chains makes it difficult to isolate the failing component.

| Symptom | Root Cause | Diagnosis Steps | Fix |
|---------|------------|-----------------|-----|
| Decryption fails with "authentication tag mismatch" | Incorrect nonce construction from sequence number | 1. Log IV, sequence number, and computed nonce 2. Verify nonce XOR operation against specification 3. Check sequence number increment timing | Ensure nonce = IV XOR sequence_number with proper 64-bit big-endian encoding |
| Finished message verification fails | Wrong handshake context hash used in computation | 1. Print all handshake messages added to context 2. Verify SHA-256 hash computation includes all messages 3. Compare hash with test vectors | Include ALL handshake messages in order, without TLS record headers |
| Server rejects key share with "invalid key" | Malformed X25519 public key format | 1. Verify public key is exactly 32 bytes 2. Check key generation uses proper curve parameters 3. Validate key encoding format | X25519 public keys must be 32 raw bytes, not ASN.1 or other encoding |
| HKDF key derivation produces wrong results | Incorrect info string or salt values | 1. Log all HKDF inputs: salt, IKM, info, length 2. Compare with test vectors from RFC 8446 3. Verify UTF-8 encoding of info strings | Use exact info strings from specification, verify salt handling |
| Certificate signature verification fails | Wrong hash algorithm used for signature | 1. Check certificate's signature algorithm identifier 2. Verify hash function matches algorithm (SHA-256 for RSA-PSS-SHA256) 3. Validate signature format | Match hash algorithm to certificate's declared signature algorithm |

⚠️ **Pitfall: Reusing Cryptographic State**  
Cryptographic objects like `AEADCipher` instances often maintain internal state that becomes invalid after certain operations. Creating new cipher instances for each record rather than reusing them prevents subtle state corruption bugs that cause intermittent failures.

#### State Machine Transition Errors

TLS state machine bugs are challenging because they depend on the precise sequence of messages and events. The same message can be valid in one state and invalid in another, making diagnosis require careful tracking of state transitions.

| Symptom | Root Cause | Diagnosis Steps | Fix |
|---------|------------|-----------------|-----|
| "Unexpected message" errors during handshake | Incorrect state transitions or message ordering | 1. Log all state transitions with timestamps 2. Trace message sequence against expected flow 3. Verify state machine transition table | Implement strict state validation before processing each message |
| Handshake hangs waiting for server response | Client sent malformed message, server closed connection | 1. Capture network traffic to see if server sent alert 2. Check for TCP connection closure 3. Validate last client message format | Add timeout handling and alert processing to detect server rejection |
| Application data encryption fails after handshake | Traffic keys not derived or not activated properly | 1. Verify `derive_application_secrets()` was called 2. Check that key derivation completed before attempting encryption 3. Validate traffic key lengths | Ensure key derivation completes before transitioning to `CONNECTED` state |
| Connection state becomes inconsistent | Race condition between message processing threads | 1. Add thread safety logging around state transitions 2. Verify all state changes use proper locking 3. Check for atomic state update requirements | Use `state_lock` consistently around all state machine operations |
| Alerts not handled properly | Missing alert processing logic or incorrect alert interpretation | 1. Log all received alert messages with level and description 2. Check alert handler registration 3. Verify proper connection cleanup after fatal alerts | Implement complete alert processing for all alert types |

#### Network Transport and Timing Issues

Network-level bugs often manifest as seemingly random failures because they depend on timing, packet fragmentation, and network conditions that vary between test runs.

| Symptom | Root Cause | Diagnosis Steps | Fix |
|---------|------------|-----------------|-----|
| Intermittent "connection reset" errors | TCP connection closed due to timeout or resource limits | 1. Check server logs for connection limits 2. Verify proper TCP keep-alive handling 3. Monitor connection timing patterns | Implement proper connection pooling and timeout handling |
| Partial record reception causes parsing failures | Not handling TCP stream fragmentation correctly | 1. Log received byte counts and expected record sizes 2. Check if `receive_exact()` handles partial reads 3. Verify record boundary detection | Always use `receive_exact()` for fixed-size fields, handle partial TCP reads |
| Handshake succeeds but application data fails | Firewall or middlebox interference with encrypted data | 1. Test with different networks/paths 2. Check for consistent MTU and fragmentation 3. Verify record size limits | Use smaller record sizes, implement MTU discovery |
| DNS resolution affects certificate validation | SNI hostname doesn't match certificate due to DNS variations | 1. Log exact hostname used in SNI vs certificate validation 2. Check for case sensitivity issues 3. Verify FQDN vs short name usage | Normalize hostnames consistently across SNI and certificate validation |

### Debugging Tools and Techniques

Effective TLS debugging requires specialized tools and systematic techniques because traditional application debugging approaches are insufficient for binary protocol analysis and cryptographic validation.

#### Wireshark Analysis for TLS Traffic

Wireshark provides the most comprehensive view of TLS communication by capturing and analyzing the actual network traffic. Understanding how to use Wireshark effectively for TLS debugging is essential for diagnosing protocol-level issues.

Think of Wireshark as having X-ray vision into your TLS communication — it shows you exactly what bytes are transmitted over the network, how they're interpreted as TLS messages, and where protocol violations occur. Unlike application logs that show your program's interpretation of events, Wireshark shows the ground truth of what actually happened on the network.

The key insight is that Wireshark can decode TLS messages automatically, highlighting protocol violations and providing detailed breakdowns of message structure. This allows you to compare what your implementation sends against what the specification requires, and to see how the other endpoint responds to your messages.

| Analysis Technique | Purpose | Steps | Key Insights |
|-------------------|---------|-------|--------------|
| TLS Handshake Flow Analysis | Verify complete handshake message sequence | 1. Filter traffic with `tls.handshake.type` 2. Follow TCP stream for specific connection 3. Verify message ordering against RFC 8446 | Identifies missing messages, incorrect ordering, or premature connection closure |
| Record Layer Examination | Debug record framing and length issues | 1. Expand TLS record layer in packet details 2. Verify content type, version, length fields 3. Check payload size against declared length | Reveals record construction bugs, fragmentation issues, length calculation errors |
| Extension Analysis | Validate ClientHello/ServerHello extensions | 1. Navigate to Handshake → Extensions in packet details 2. Verify extension types and data format 3. Check for required vs optional extensions | Shows extension encoding problems, missing required extensions, malformed data |
| Alert Message Investigation | Diagnose handshake failures and errors | 1. Filter for `tls.alert` messages 2. Examine alert level (warning/fatal) and description 3. Correlate with preceding messages | Provides server's reason for rejecting connection, guides debugging focus |
| Encrypted Record Analysis | Monitor post-handshake data flow | 1. Look for Application Data records after handshake 2. Verify record sizes and frequency 3. Check for proper connection closure | Confirms successful encryption establishment, reveals data flow issues |

**Wireshark Filter Commands for TLS Debugging:**

Understanding the right Wireshark filters dramatically improves debugging efficiency by focusing on relevant traffic and message types.

| Filter Purpose | Filter Expression | What It Shows |
|---------------|------------------|---------------|
| All TLS traffic to specific host | `tls and ip.addr == 192.168.1.100` | Complete TLS conversation with target server |
| Handshake messages only | `tls.handshake.type` | ClientHello, ServerHello, Certificate, Finished messages |
| TLS alerts and errors | `tls.alert` | Alert messages indicating handshake failures or warnings |
| Application data records | `tls.record.content_type == 23` | Encrypted application data flow after handshake completion |
| Certificate-related messages | `tls.handshake.type == 11 or tls.handshake.type == 15` | Certificate and CertificateVerify messages for validation debugging |

> The critical insight for Wireshark analysis is that TLS bugs often appear as subtle deviations from expected patterns rather than obvious errors. Learning to recognize normal TLS handshake patterns makes anomalies immediately visible.

#### Logging Strategies for TLS Components

Effective logging in TLS implementations requires careful balance between security and diagnostics. Too much logging can expose sensitive cryptographic material, while too little makes debugging impossible.

The mental model for TLS logging is like creating a medical chart that tracks a patient's vital signs over time — you need enough information to diagnose problems without recording sensitive personal information that could be misused. Each TLS component should log its key decision points and state changes while protecting cryptographic secrets.

**Component-Specific Logging Approaches:**

| Component | What to Log | What NOT to Log | Format Example |
|-----------|------------|-----------------|----------------|
| `TCPTransport` | Connection events, byte counts, network errors | Raw application data, detailed packet contents | `TCP: Connected to example.com:443, sent 245 bytes, recv 1337 bytes` |
| `RecordLayer` | Record types, lengths, sequence numbers | Decrypted payload contents, authentication tags | `Record: HANDSHAKE len=89 seq=3` |
| `HandshakeEngine` | Message types, extensions, state transitions | Private keys, shared secrets, finished verify data | `Handshake: Sent ClientHello with 3 extensions, transitioning to WAIT_SH` |
| `CryptoEngine` | Algorithm selections, key derivation stages | Actual key material, nonces, intermediate secrets | `Crypto: Derived handshake traffic secrets, cipher=AES-128-GCM` |
| `CertificateValidator` | Validation steps, hostname matching results | Certificate private keys, signature values | `CertVal: Chain length=2, hostname match=true, expiry check=passed` |

**Structured Logging Format for TLS Events:**

Implementing consistent logging format across all TLS components enables easier debugging and automated analysis of log files.

```
[TIMESTAMP] [COMPONENT] [LEVEL] [CONNECTION_ID] [STATE] MESSAGE
2024-01-15T10:30:45.123Z TLS_CLIENT INFO conn_001 WAIT_SH Received ServerHello, cipher=TLS_AES_128_GCM_SHA256
2024-01-15T10:30:45.145Z CRYPTO_ENGINE DEBUG conn_001 KEY_DERIVE HKDF extract completed, expanding handshake secrets
2024-01-15T10:30:45.167Z CERT_VALIDATOR INFO conn_001 VALIDATING Certificate chain validation passed for example.com
```

⚠️ **Pitfall: Logging Sensitive Cryptographic Material**  
Never log actual key bytes, nonces, authentication tags, or other sensitive cryptographic values. Instead, log key derivation events, algorithm choices, and validation results. Use key fingerprints (first 8 hex characters) if you must reference specific keys for correlation.

#### State Inspection Techniques

TLS debugging often requires examining the internal state of components at specific points in the protocol flow. Effective state inspection provides snapshots of cryptographic state, connection parameters, and component interactions.

**Connection State Monitoring:**

| State Category | Key Information | Inspection Method | Debugging Value |
|---------------|-----------------|-------------------|-----------------|
| Protocol State | Current state, valid transitions, pending operations | State machine logger with transition history | Identifies illegal state transitions, timing issues |
| Cryptographic State | Active cipher suite, key derivation stage, sequence numbers | Crypto engine status dump (without key material) | Confirms proper key lifecycle, detects sequence number issues |
| Transport State | TCP connection status, buffer states, pending I/O | Transport layer diagnostics | Reveals network-level problems affecting TLS |
| Certificate State | Validation results, trust anchor usage, hostname status | Certificate validator report | Isolates certificate-related handshake failures |

**Real-time State Inspection Commands:**

Implementing debug commands that can be triggered during execution provides valuable insights into component state without requiring debugger attachment.

| Command | Purpose | Output Format | Use Cases |
|---------|---------|---------------|-----------|
| `dump_connection_state()` | Show current TLS connection status | JSON object with all non-sensitive state | Diagnosing handshake hangs, state machine issues |
| `show_handshake_context()` | Display handshake message history | List of message types and lengths (not contents) | Verifying handshake message sequence correctness |
| `crypto_status()` | Report cryptographic operation status | Algorithm selections, key derivation progress | Debugging encryption/decryption failures |
| `validate_state_consistency()` | Cross-check component state consistency | Boolean result with inconsistency details | Detecting race conditions, component synchronization issues |

#### Binary Data Analysis Techniques

TLS protocols involve extensive binary data manipulation, requiring specialized techniques for examining byte-level message structure and identifying encoding problems.

**Hex Dump Analysis for TLS Messages:**

The `hex_dump()` function becomes essential for TLS debugging because it provides both hexadecimal and ASCII views of binary data, making protocol structure visible.

| Data Type | Hex Dump Focus Areas | Analysis Technique | Common Issues |
|-----------|---------------------|-------------------|---------------|
| TLS Records | First 5 bytes (header), content type patterns | Compare header values against constants | Wrong content type, length calculation errors |
| Handshake Messages | Message type byte, length fields, extension boundaries | Validate length consistency, extension parsing | Malformed extensions, incorrect length encoding |
| Cryptographic Data | Key lengths, random value entropy, tag positions | Verify expected data sizes, check for patterns | Wrong key sizes, predictable randoms, tag corruption |
| Certificate Data | ASN.1 structure markers, length encodings | Follow ASN.1 parsing rules, validate DER encoding | Certificate parsing errors, encoding violations |

**Binary Protocol Validation Workflow:**

1. **Capture Raw Bytes**: Use hex dump to visualize the actual transmitted data
2. **Parse Structure**: Manually walk through the binary format following RFC specifications  
3. **Validate Fields**: Check each field against expected values and formats
4. **Cross-Reference**: Compare with known-good examples or test vectors
5. **Identify Deviations**: Highlight any differences from specification requirements

> The key insight for binary data debugging is that TLS implementations must handle data exactly as specified — even single-bit errors can cause complete protocol failures. Systematic byte-level analysis is often the only way to identify subtle encoding bugs.

### Implementation Guidance

Understanding TLS debugging requires both theoretical knowledge and practical tools. The following implementation guidance provides concrete debugging infrastructure and techniques for building robust TLS implementations.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Binary Data Inspection | Built-in hex dump with ASCII view | Wireshark integration with custom dissectors |
| Logging Framework | Standard library logging with structured format | Structured logging with JSON output and log aggregation |
| State Inspection | Print statements with state dumps | Interactive debugging interface with real-time state viewing |
| Protocol Analysis | Manual hex analysis with calculators | Automated protocol compliance checking against test vectors |
| Cryptographic Debugging | Algorithm selection logging without key material | Cryptographic test vector validation with known inputs/outputs |

#### Debugging Infrastructure Starter Code

**Complete Hex Dump Utility:**

```python
def hex_dump(data: bytes, width: int = 16, show_ascii: bool = True) -> str:
    """Create detailed hex dump representation of binary data for TLS debugging.
    
    Args:
        data: Binary data to format
        width: Bytes per line (typically 16)
        show_ascii: Whether to include ASCII representation
    
    Returns:
        Formatted hex dump string with offsets, hex bytes, and ASCII view
    """
    lines = []
    for offset in range(0, len(data), width):
        chunk = data[offset:offset + width]
        hex_part = ' '.join(f'{byte:02x}' for byte in chunk)
        hex_part = hex_part.ljust(width * 3 - 1)  # Pad for alignment
        
        ascii_part = ''
        if show_ascii:
            ascii_chars = []
            for byte in chunk:
                if 32 <= byte <= 126:  # Printable ASCII
                    ascii_chars.append(chr(byte))
                else:
                    ascii_chars.append('.')
            ascii_part = f" |{''.join(ascii_chars).ljust(width)}|"
        
        lines.append(f"{offset:08x}  {hex_part}{ascii_part}")
    
    return '\n'.join(lines)

class TLSLogger:
    """Centralized logging for TLS components with security-aware formatting."""
    
    def __init__(self, component_name: str, connection_id: str = None):
        self.component = component_name
        self.connection_id = connection_id or "unknown"
        self.logger = logging.getLogger(f"tls.{component_name}")
    
    def log_binary_data(self, description: str, data: bytes, max_bytes: int = 64):
        """Log binary data with hex dump, limiting size for security."""
        if len(data) > max_bytes:
            truncated_data = data[:max_bytes]
            self.logger.debug(f"{description} (truncated to {max_bytes} bytes):\n{hex_dump(truncated_data)}")
        else:
            self.logger.debug(f"{description}:\n{hex_dump(data)}")
    
    def log_state_transition(self, from_state: str, to_state: str, trigger: str):
        """Log state machine transitions for debugging state issues."""
        self.logger.info(f"[{self.connection_id}] State: {from_state} -> {to_state} (trigger: {trigger})")
    
    def log_crypto_operation(self, operation: str, algorithm: str, success: bool, details: str = ""):
        """Log cryptographic operations without exposing sensitive material."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"[{self.connection_id}] Crypto: {operation} using {algorithm} - {status} {details}")
```

**TLS State Inspection Framework:**

```python
class TLSStateInspector:
    """Provides non-invasive inspection of TLS connection state for debugging."""
    
    def __init__(self, tls_connection: 'TLSConnection'):
        self.connection = tls_connection
        self.inspection_history = []
    
    def dump_connection_state(self) -> Dict[str, Any]:
        """Create comprehensive state snapshot without sensitive data."""
        state = {
            'timestamp': time.time(),
            'connection_id': self.connection.connection_id,
            'transport_state': self._get_transport_state(),
            'protocol_state': self._get_protocol_state(),
            'crypto_state': self._get_crypto_state_safe(),
            'record_layer_state': self._get_record_layer_state()
        }
        self.inspection_history.append(state)
        return state
    
    def _get_transport_state(self) -> Dict[str, Any]:
        """Get TCP transport state information."""
        transport = self.connection.transport
        return {
            'connected': transport.socket is not None,
            'remote_address': transport.remote_address,
            'bytes_sent': getattr(transport, 'bytes_sent', 0),
            'bytes_received': getattr(transport, 'bytes_received', 0)
        }
    
    def _get_protocol_state(self) -> Dict[str, Any]:
        """Get TLS protocol state without sensitive information."""
        state_machine = self.connection.state_machine
        return {
            'current_state': state_machine.current_state.name if state_machine.current_state else 'UNKNOWN',
            'handshake_complete': state_machine.current_state == ConnectionState.CONNECTED,
            'last_transition_time': getattr(state_machine, 'last_transition_time', None)
        }
    
    def _get_crypto_state_safe(self) -> Dict[str, Any]:
        """Get cryptographic state information without exposing key material."""
        crypto = self.connection.crypto_engine
        if not crypto:
            return {'initialized': False}
        
        return {
            'initialized': True,
            'cipher_suite': crypto.cipher_suite,
            'key_derivation_complete': hasattr(crypto, 'key_schedule') and crypto.key_schedule is not None,
            'handshake_context_length': len(crypto.handshake_context.digest()) if crypto.handshake_context else 0
        }
    
    def validate_state_consistency(self) -> List[str]:
        """Cross-validate component states for consistency issues."""
        issues = []
        
        # Check transport/protocol state alignment
        if self.connection.state_machine.current_state == ConnectionState.CONNECTED:
            if not self.connection.transport.socket:
                issues.append("Protocol state CONNECTED but no TCP socket")
        
        # Check crypto/protocol state alignment  
        if self.connection.state_machine.current_state in [ConnectionState.CONNECTED]:
            if not self.connection.crypto_engine or not hasattr(self.connection.crypto_engine, 'key_schedule'):
                issues.append("Protocol state requires crypto but key derivation incomplete")
        
        return issues
```

#### Core Debugging Logic Skeleton

**TLS Protocol Analyzer:**

```python
class TLSProtocolAnalyzer:
    """Analyzes TLS protocol compliance and identifies common implementation bugs."""
    
    def __init__(self):
        self.logger = TLSLogger("protocol_analyzer")
        self.message_history = []
        self.known_issues = []
    
    def analyze_handshake_sequence(self, messages: List[bytes]) -> List[str]:
        """Analyze handshake message sequence for protocol compliance.
        
        Returns list of identified issues or empty list if sequence is valid.
        """
        issues = []
        
        # TODO 1: Validate message sequence follows TLS 1.3 specification
        # Expected: ClientHello -> ServerHello -> EncryptedExtensions -> Certificate -> CertificateVerify -> Finished
        # Hint: Use HandshakeType constants to identify message types
        
        # TODO 2: Check for required extensions in ClientHello and ServerHello
        # Required in ClientHello: supported_versions, server_name, key_share
        # Required in ServerHello: supported_versions, key_share
        
        # TODO 3: Verify message ordering constraints
        # Certificate must come before CertificateVerify
        # Finished must be last message in handshake flight
        
        # TODO 4: Validate extension consistency between ClientHello and ServerHello
        # Server must not echo unsupported extensions
        # Key share groups must match between client offer and server selection
        
        # TODO 5: Check for security violations
        # No downgrade to earlier TLS versions
        # Strong cipher suite selection (no weak or deprecated algorithms)
        
        return issues
    
    def analyze_record_structure(self, record_data: bytes) -> Dict[str, Any]:
        """Analyze TLS record structure for format compliance."""
        analysis = {
            'valid': True,
            'issues': [],
            'record_info': {}
        }
        
        # TODO 1: Parse record header and validate field values
        # Check content type is valid (20, 21, 22, or 23)
        # Verify legacy version field (should be 0x0303 for TLS 1.3)
        # Validate length field is within allowed range
        
        # TODO 2: Verify record payload size constraints
        # Payload must not exceed MAX_RECORD_PAYLOAD_LENGTH (16384 bytes)
        # Empty records only allowed for specific content types
        
        # TODO 3: Check for record fragmentation issues
        # Handshake messages may span multiple records
        # Application data records should not be unnecessarily fragmented
        
        # TODO 4: Validate record type transitions
        # ChangeCipherSpec records deprecated in TLS 1.3
        # Application data only after handshake completion
        
        return analysis
```

#### Milestone Checkpoints

**Milestone 1 Debugging Checkpoint:**
After implementing TCP transport and record layer:
- Run `python -m pytest tests/test_record_layer.py -v` - should show all record parsing tests passing
- Expected output: Record layer correctly parses all test vectors, handles fragmentation
- Manual verification: Connect to `example.com:443`, should establish TCP connection and parse initial ServerHello record
- Debug signs: If connection hangs, check firewall settings; if parsing fails, verify endianness handling

**Milestone 2 Debugging Checkpoint:**  
After implementing ClientHello construction:
- Run `python debug_clienthello.py --server example.com` - should show well-formed ClientHello analysis
- Expected output: ClientHello hex dump shows correct structure, extensions properly formatted
- Manual verification: Wireshark capture should show ClientHello with SNI, supported_versions, key_share extensions
- Debug signs: If server sends decode_error alert, check extension length calculations; if no response, verify TCP connection

**Milestone 3 Debugging Checkpoint:**
After implementing key exchange:
- Run `python test_key_derivation.py --test-vectors` - should validate against RFC 8446 test vectors
- Expected output: All HKDF derivations match expected values, handshake context hashes correct
- Manual verification: ServerHello processing completes, certificate validation passes
- Debug signs: If finished verification fails, check handshake context message ordering; if decryption fails, verify key derivation sequence

**Milestone 4 Debugging Checkpoint:**
After implementing encrypted communication:
- Run `python test_https_request.py --url https://example.com/` - should complete full HTTPS request/response
- Expected output: HTTP/1.1 200 OK response with expected content
- Manual verification: Wireshark shows encrypted Application Data records after handshake
- Debug signs: If AEAD decryption fails, check nonce construction; if HTTP parsing fails, verify record reassembly

#### Language-Specific Debugging Hints

**Python TLS Debugging Techniques:**

- Use `cryptography` library's debug logging: `logging.getLogger('cryptography').setLevel(logging.DEBUG)`
- Enable socket debugging: `socket.socket.settrace()` for detailed network I/O tracking  
- Memory inspection: `sys.getsizeof()` to track cryptographic state memory usage
- Exception chaining: Use `raise NewException() from original_exception` to preserve debugging context
- Timing analysis: `time.perf_counter()` for measuring cryptographic operation performance

**Common Python TLS Implementation Gotchas:**

| Issue | Symptom | Fix |
|-------|---------|-----|
| Bytes vs string confusion | `TypeError: a bytes-like object is required` | Always use `bytes` for cryptographic data, `str.encode('utf-8')` for text |
| Mutable default arguments | State corruption between connections | Use `None` defaults, initialize mutable objects in function body |
| Integer overflow in length calculations | Parsing errors with large messages | Use appropriate integer types, validate ranges before arithmetic |
| GIL contention in crypto operations | Poor performance with concurrent connections | Use `cryptography` library's C extensions, avoid pure Python crypto |
| Memory leaks with cryptographic objects | Gradually increasing memory usage | Explicitly clear sensitive data, use context managers for key material |


## Future Extensions

> **Milestone(s):** Beyond Milestone 4 — these extensions build upon the complete HTTPS client implementation to add advanced TLS features and performance optimizations

Think of the current HTTPS client implementation as a well-built house with solid foundations, secure walls, and functional utilities. Future extensions are like adding smart home features, solar panels, and luxury amenities — they enhance the core functionality without changing the fundamental architecture. These extensions transform a functional TLS client into an enterprise-grade communication library capable of handling high-performance workloads and advanced security requirements.

The extensions fall into two primary categories: advanced TLS protocol features that enhance security and compatibility, and performance optimizations that improve throughput, reduce latency, and enable scalable deployment. Each extension builds upon the existing component architecture while introducing new complexity in cryptographic operations, state management, and resource optimization.

### Advanced TLS Features

Advanced TLS features extend the protocol capabilities beyond basic handshake and encryption, adding sophisticated mechanisms for performance optimization, enhanced security, and improved user experience. These features require deep understanding of TLS internals and careful implementation to maintain security properties while adding functionality.

#### Session Resumption and Session Tickets

Session resumption is like having a VIP pass to a secure facility — instead of going through the complete security checkpoint every time, you present your pre-validated credentials and gain immediate access. This mechanism dramatically reduces handshake latency and computational overhead by reusing previously established cryptographic state.

**TLS 1.3 Session Resumption Architecture**

TLS 1.3 session resumption operates through a fundamentally different mechanism than TLS 1.2, using Pre-Shared Keys (PSK) derived from previous sessions rather than cached session state. The server provides a session ticket during the initial handshake, which the client can use to resume the session in future connections.

| Component | Purpose | Security Properties |
|-----------|---------|-------------------|
| `SessionCache` | Store session tickets and associated PSKs | Forward secrecy through time-bounded storage |
| `TicketProcessor` | Generate and validate session tickets | Cryptographic binding to original handshake |
| `PSKManager` | Manage pre-shared key derivation and usage | Key isolation between sessions |
| `ResumedHandshakeEngine` | Handle PSK-based handshake flow | Reduced attack surface through shortened handshake |

The session resumption process involves several cryptographic operations that must maintain the security properties of full handshakes while providing performance benefits:

> **Decision: PSK-Based Resumption over Session ID Caching**
> - **Context**: TLS 1.3 moved away from server-side session caching to client-side ticket storage
> - **Options Considered**: 
>   1. Implement TLS 1.2-style session ID caching with server-side state
>   2. Use TLS 1.3 PSK-based resumption with encrypted session tickets
>   3. Support both mechanisms for backward compatibility
> - **Decision**: Implement PSK-based resumption as the primary mechanism
> - **Rationale**: PSK resumption provides better scalability (no server-side state), stronger forward secrecy properties, and aligns with TLS 1.3 design principles. Session tickets are cryptographically bound to the original handshake and can include expiration and usage limits.
> - **Consequences**: Requires implementing ticket encryption/decryption, PSK key derivation, and modified handshake flow. Provides significant performance benefits for repeated connections to the same server.

**Session Ticket Structure and Cryptographic Binding**

Session tickets must securely encode session state while preventing tampering, replay attacks, and unauthorized access. The ticket structure includes both cryptographic material and metadata necessary for session resumption.

| Field | Type | Description |
|-------|------|-------------|
| `ticket_lifetime` | uint32 | Ticket validity period in seconds |
| `ticket_age_add` | uint32 | Random value for age calculation obfuscation |
| `ticket_nonce` | bytes | Unique value for PSK derivation |
| `ticket` | bytes | Encrypted session state including resumption secret |
| `extensions` | List[Extension] | Additional ticket parameters |

The session resumption process follows a specific sequence that maintains security while reducing handshake round trips:

1. During the initial handshake, the server generates a resumption secret using HKDF-Expand-Label from the master secret
2. The server creates an encrypted session ticket containing the resumption secret and relevant session parameters
3. The client stores the ticket and associated metadata in its session cache
4. For subsequent connections, the client includes the ticket in a PSK extension within its ClientHello
5. The server decrypts the ticket, validates its authenticity and expiration, and derives the PSK from the resumption secret
6. Both parties use the PSK as input to a modified key derivation process, skipping the ECDHE exchange
7. The handshake completes with only ClientHello, ServerHello, and Finished messages

**Session Cache Management and Security Considerations**

The session cache requires careful design to balance performance benefits with security requirements. The cache must handle ticket expiration, storage limits, and potential security threats while providing fast lookup and insertion operations.

| Security Property | Implementation Requirement | Attack Mitigation |
|-------------------|---------------------------|------------------|
| Forward Secrecy | Time-bounded ticket storage with automatic expiration | Limits exposure window if tickets are compromised |
| Replay Protection | Ticket age validation and anti-replay mechanisms | Prevents reuse of captured tickets |
| Ticket Binding | Cryptographic binding to server identity and parameters | Prevents ticket misuse across different servers |
| Storage Limits | Bounded cache size with LRU eviction policy | Prevents memory exhaustion attacks |

⚠️ **Pitfall: Insecure Session Ticket Storage**

A common implementation mistake is storing session tickets in plaintext or in persistent storage without proper protection. This creates a significant security vulnerability where ticket compromise can lead to session impersonation and traffic decryption. Session tickets should be stored in memory only, with secure cleanup on process termination, and should never be written to disk in plaintext form. Additionally, tickets must be validated for expiration and binding to the expected server before use.

#### Zero Round-Trip Time (0-RTT) Data

0-RTT data transmission is like having express lanes at airport security — trusted travelers with proper credentials can skip the full security process and proceed directly to their destination. This mechanism allows clients to send encrypted application data in their first message to the server, reducing connection establishment latency to zero round trips.

**0-RTT Security Model and Trade-offs**

The 0-RTT feature provides significant performance benefits but introduces security trade-offs that must be carefully managed. The fundamental challenge is that 0-RTT data lacks the forward secrecy properties of 1-RTT data, as it's encrypted with keys derived from the previous session.

| Security Property | 0-RTT Behavior | 1-RTT Comparison | Mitigation Strategy |
|-------------------|----------------|------------------|-------------------|
| Forward Secrecy | Limited — uses previous session keys | Full forward secrecy | Time-bounded 0-RTT window |
| Replay Protection | Vulnerable to replay attacks | Strong replay protection | Application-level idempotency |
| Key Freshness | Uses derived PSK from previous session | Fresh ECDHE keys | Mandatory key update after 0-RTT |
| Server State | Requires anti-replay mechanisms | Stateless operation | Bloom filters or database tracking |

The 0-RTT implementation requires modifications to multiple components of the TLS client to handle early data transmission and the associated security considerations:

> **Decision: Conservative 0-RTT Implementation with Application Control**
> - **Context**: 0-RTT provides performance benefits but introduces replay vulnerability
> - **Options Considered**:
>   1. Automatic 0-RTT for all resumption scenarios
>   2. Application-controlled 0-RTT with explicit opt-in
>   3. No 0-RTT support to avoid security complexity
> - **Decision**: Implement application-controlled 0-RTT with clear security warnings
> - **Rationale**: Applications understand their data semantics and can make informed decisions about replay safety. Automatic 0-RTT could lead to security vulnerabilities in applications that aren't designed for replay tolerance. Explicit control provides performance benefits while maintaining security.
> - **Consequences**: Requires additional API surface for 0-RTT control, documentation of replay risks, and application-level consideration of idempotency requirements.

**0-RTT Data Handling and State Management**

The 0-RTT implementation requires careful coordination between the handshake engine, record layer, and application data handler to manage the transition from early data to normal traffic encryption.

| Component | 0-RTT Responsibility | Implementation Complexity |
|-----------|---------------------|--------------------------|
| `EarlyDataHandler` | Encrypt/decrypt 0-RTT application data | Medium — special key derivation |
| `HandshakeEngine` | Negotiate 0-RTT capability and handle rejection | High — state machine complexity |
| `RecordLayer` | Distinguish early data from handshake messages | Medium — record type handling |
| `CryptoEngine` | Derive early traffic keys from PSK | Low — HKDF expansion |

The 0-RTT handshake flow introduces additional complexity in the state machine, as the client must handle the possibility of 0-RTT rejection by the server:

1. Client includes `early_data` extension in ClientHello when resuming with stored ticket
2. Client immediately sends encrypted application data using early traffic keys derived from PSK
3. Client continues with normal handshake flow while early data transmission proceeds
4. Server indicates 0-RTT acceptance or rejection in EncryptedExtensions
5. If rejected, client must be prepared to retransmit the data using 1-RTT keys
6. If accepted, early data transitions seamlessly to normal application data flow

#### Additional Cipher Suites and Cryptographic Algorithms

Supporting additional cipher suites is like adding more languages to your diplomatic toolkit — each cipher suite provides different security properties, performance characteristics, and compatibility with various environments. Expanding cipher suite support enhances interoperability while allowing optimization for specific deployment scenarios.

**Modern Cipher Suite Architecture**

TLS 1.3 simplified cipher suite selection by separating authentication from bulk encryption, but implementations can benefit from supporting additional AEAD algorithms and hash functions for different security and performance requirements.

| Cipher Suite | Security Level | Performance | Use Case |
|--------------|---------------|-------------|----------|
| `TLS_AES_128_GCM_SHA256` | Standard | High | General purpose, hardware acceleration |
| `TLS_AES_256_GCM_SHA384` | High | Medium | High security requirements |
| `TLS_CHACHA20_POLY1305_SHA256` | Standard | High | Mobile devices, software-only encryption |
| `TLS_AES_128_CCM_SHA256` | Standard | Medium | IoT devices, constrained environments |

The cipher suite extension requires implementing additional AEAD algorithms and integrating them into the existing cryptographic framework:

| Component Extension | Implementation Requirement | Integration Point |
|-------------------|---------------------------|------------------|
| `AEADCipherFactory` | Support multiple cipher algorithms | `CryptoEngine.create_cipher()` |
| `HashFunctionProvider` | Multiple hash algorithms for different suites | `KeyScheduleState` derivation |
| `CipherSuiteNegotiator` | Preference ordering and compatibility checking | `ClientHello` construction |
| `HardwareAcceleration` | Platform-specific optimizations | `AEADCipher` implementations |

**ChaCha20-Poly1305 Implementation Considerations**

ChaCha20-Poly1305 provides an important alternative to AES-GCM, particularly for environments without hardware AES acceleration. This cipher suite offers excellent software performance and resistance to timing attacks.

The ChaCha20-Poly1305 implementation requires understanding its unique nonce construction and authentication mechanism:

1. ChaCha20 uses a 96-bit nonce combined with a 32-bit counter, different from AES-GCM's nonce structure
2. Poly1305 authentication operates over the ciphertext and additional authenticated data with specific padding
3. The algorithm provides strong security properties with simpler implementation than AES-GCM
4. Nonce construction must prevent reuse while maintaining compatibility with TLS sequence number handling

> The critical insight for ChaCha20-Poly1305 integration is that while the AEAD interface remains consistent, the internal nonce handling and key schedule requirements differ significantly from AES-GCM. The implementation must abstract these differences while maintaining the security properties of each algorithm.

### Performance Optimizations

Performance optimizations transform the functional HTTPS client into a high-performance communication library capable of handling production workloads. These optimizations address the computational, memory, and network bottlenecks that limit scalability in real-world deployments.

#### Connection Pooling and Multiplexing

Connection pooling is like having a fleet of pre-warmed vehicles ready for immediate use — instead of going through the time-consuming process of starting, warming up, and preparing a vehicle for each trip, you maintain a pool of ready-to-use connections that can be immediately assigned to new requests. This eliminates the handshake overhead for subsequent requests to the same server.

**Connection Pool Architecture and Lifecycle Management**

A robust connection pool must handle connection lifecycle, health monitoring, and resource management while providing thread-safe access for concurrent operations. The pool serves as an intermediary between application requests and the underlying TLS connections.

| Component | Responsibility | Complexity Factors |
|-----------|----------------|-------------------|
| `ConnectionPool` | Maintain pool of active connections | Thread safety, resource limits |
| `ConnectionHealthMonitor` | Detect and remove stale connections | Network timeouts, TLS alerts |
| `PoolingStrategy` | Determine connection allocation and eviction | LRU vs LFU, server affinity |
| `ConnectionFactory` | Create new connections when needed | Handshake management, error handling |

The connection pool implementation must address several challenging aspects of HTTP/TLS connection management:

> **Decision: Per-Host Connection Pooling with HTTP/1.1 Keep-Alive**
> - **Context**: HTTP requests to the same server can reuse TLS connections to amortize handshake costs
> - **Options Considered**:
>   1. Global connection pool shared across all hosts
>   2. Per-host connection pools with separate limits
>   3. Single-use connections with no pooling
> - **Decision**: Implement per-host connection pools with configurable limits
> - **Rationale**: Per-host pooling provides better resource control and prevents one busy host from consuming all connections. It aligns with browser connection management patterns and allows tuning based on server capabilities. HTTP/1.1 keep-alive provides good compatibility without HTTP/2 complexity.
> - **Consequences**: Requires host-based connection tracking, more complex pool management, but provides better performance isolation and resource control.

**Connection Pool State Management and Thread Safety**

The connection pool must handle concurrent access from multiple threads while maintaining connection state consistency and resource limits. The implementation requires careful synchronization to prevent race conditions while maximizing performance.

| Synchronization Challenge | Solution Approach | Performance Impact |
|-------------------------|------------------|-------------------|
| Pool access contention | Fine-grained locking per host pool | Low — minimal lock scope |
| Connection state updates | Atomic state transitions with CAS | Low — lock-free for common operations |
| Resource limit enforcement | Semaphores for connection counting | Medium — blocking under high load |
| Health check coordination | Background thread with periodic checks | Low — asynchronous operation |

The connection pool lifecycle management involves several critical operations that must maintain consistency:

1. **Connection Acquisition**: Check pool for available connection, validate health, create new if needed
2. **Connection Health Monitoring**: Periodic checks for connection liveness, TLS alert handling
3. **Connection Return**: Validate connection state, update last-use timestamp, return to pool
4. **Connection Eviction**: Remove idle or unhealthy connections based on policy
5. **Pool Shutdown**: Gracefully close all connections with proper TLS termination

⚠️ **Pitfall: Connection State Leakage Between Requests**

A critical security issue in connection pooling is failing to properly reset connection state between requests. If cryptographic state, authentication credentials, or request context leak between pooled connections, it can lead to information disclosure or privilege escalation. Each connection returned from the pool must be validated to ensure it's in a clean state, with no residual data from previous requests. Additionally, connections that have experienced errors or alerts should be discarded rather than returned to the pool.

#### Asynchronous I/O and Non-Blocking Operations

Asynchronous I/O is like having a highly efficient restaurant kitchen where chefs can work on multiple dishes simultaneously — instead of preparing each dish from start to finish before beginning the next, they can start multiple dishes, work on whichever ingredients are ready, and efficiently utilize their time and equipment. This approach dramatically improves throughput by allowing a single thread to manage many concurrent connections.

**Async TLS Integration Architecture**

Integrating asynchronous operations into the TLS implementation requires careful redesign of the synchronous components to support non-blocking operations while maintaining security properties and protocol correctness.

| Async Integration Point | Synchronous Challenge | Async Solution |
|------------------------|----------------------|----------------|
| TCP Socket Operations | Blocking read/write calls | Event-driven I/O with callbacks |
| Cryptographic Operations | CPU-intensive computations | Thread pool for crypto work |
| Certificate Validation | Network CRL/OCSP fetching | Async HTTP client for validation |
| Handshake State Machine | Sequential message processing | Coroutine-based state management |

The asynchronous TLS client architecture requires a fundamental shift from the sequential processing model to an event-driven approach:

> **Decision: Async/Await Pattern with Cooperative Multitasking**
> - **Context**: High-performance applications need to handle thousands of concurrent TLS connections
> - **Options Considered**:
>   1. Thread-per-connection with blocking I/O
>   2. Event loop with callbacks for all operations
>   3. Async/await with coroutines for connection handling
> - **Decision**: Implement async/await pattern with cooperative multitasking
> - **Rationale**: Async/await provides the readability benefits of sequential code while enabling high concurrency. It's easier to reason about than callback-based approaches and provides better error handling than pure event loops. The pattern is well-supported in modern languages.
> - **Consequences**: Requires async versions of all I/O operations, more complex error propagation, but enables handling thousands of concurrent connections with minimal resource usage.

**Event Loop Integration and Resource Management**

The async implementation must integrate cleanly with the application's event loop while managing resources efficiently and handling backpressure appropriately.

| Resource Management Aspect | Implementation Approach | Scalability Consideration |
|---------------------------|------------------------|---------------------------|
| Memory Buffers | Pool reusable buffers for record processing | Prevents allocation overhead under load |
| Crypto Context | Cache expensive cryptographic contexts | Reduces initialization costs |
| Timer Management | Integrated timeout handling with event loop | Prevents connection leaks from stalled handshakes |
| Backpressure Handling | Flow control when buffers fill | Maintains system stability under overload |

The async TLS handshake implementation requires careful orchestration of network I/O, cryptographic operations, and state management:

1. **Async Connection Establishment**: Non-blocking TCP connection with timeout handling
2. **Async Record Processing**: Event-driven record assembly from partial reads
3. **Async Crypto Operations**: Offload expensive operations to thread pool
4. **Async Certificate Validation**: Non-blocking OCSP and CRL checking
5. **Async Error Handling**: Proper error propagation through async call chains

#### Cryptographic Hardware Acceleration

Cryptographic hardware acceleration is like having specialized tools for specific jobs — instead of using a general-purpose hammer for everything, you use the right tool that's optimized for each task. Modern processors provide specialized instructions for cryptographic operations that can dramatically improve performance while maintaining security.

**Hardware Acceleration Integration Strategy**

The hardware acceleration implementation must gracefully fall back to software implementations when hardware support is unavailable while maximizing performance when acceleration is present.

| Acceleration Target | Hardware Feature | Performance Benefit |
|--------------------|------------------|-------------------|
| AES Operations | AES-NI instructions | 10-20x improvement for AES-GCM |
| Hash Functions | SHA extensions | 3-5x improvement for SHA-256 |
| Elliptic Curves | Intel ADX/BMI2 | 2-3x improvement for P-256 |
| Random Generation | RDRAND/RDSEED | Faster entropy collection |

The acceleration integration requires runtime detection and dynamic dispatch to the appropriate implementation:

| Component | Acceleration Integration | Fallback Strategy |
|-----------|------------------------|------------------|
| `AccelDetector` | Runtime CPU feature detection | Feature flags with capability testing |
| `CryptoProvider` | Dynamic dispatch to best implementation | Graceful degradation to software |
| `PerformanceProfiler` | Benchmark different implementations | Adaptive selection based on workload |
| `SecurityValidator` | Ensure acceleration maintains security properties | Constant-time operation validation |

**AES-NI Integration and Constant-Time Guarantees**

AES-NI provides dramatic performance improvements for AES operations but requires careful integration to maintain the security properties of software implementations. The hardware acceleration must preserve constant-time operation characteristics to prevent timing-based side-channel attacks.

The AES-NI integration involves several implementation considerations:

1. **Feature Detection**: Runtime detection of AES-NI availability using CPU feature flags
2. **Key Schedule Acceleration**: Hardware-accelerated key expansion for improved performance
3. **Block Cipher Operations**: Direct hardware execution of AES encryption/decryption
4. **GCM Mode Integration**: Optimized GHASH operations using carry-less multiplication
5. **Constant-Time Validation**: Ensuring hardware operations maintain timing consistency

> The critical security consideration for hardware acceleration is that while performance improves dramatically, the implementation must maintain the same security guarantees as software implementations. This includes constant-time operation, proper key handling, and resistance to side-channel attacks. Hardware acceleration should be transparent to the application layer while providing security-preserving performance benefits.

⚠️ **Pitfall: Side-Channel Vulnerabilities in Accelerated Code**

Hardware acceleration can introduce subtle side-channel vulnerabilities if not implemented carefully. While hardware instructions like AES-NI are designed to be constant-time, the surrounding code for key handling, buffer management, and control flow can leak timing information. Additionally, some acceleration features may have different power consumption or electromagnetic emanation patterns that could be exploited in specialized attack scenarios. All accelerated code paths must be validated for constant-time behavior and side-channel resistance.

### Implementation Guidance

The implementation of advanced TLS features and performance optimizations requires careful architectural planning and incremental development. These extensions build upon the core HTTPS client while adding significant complexity in cryptographic operations, resource management, and concurrent programming.

**Technology Recommendations**

| Feature Category | Simple Option | Advanced Option |
|-----------------|---------------|-----------------|
| Session Storage | In-memory dictionary with expiration | Redis/Memcached with encryption |
| Async I/O | asyncio with simple event loop | uvloop with optimized performance |
| Connection Pooling | Basic per-host pools | Advanced pool with health monitoring |
| Hardware Acceleration | Software-only crypto | Intel IPP or OpenSSL with hardware support |
| 0-RTT Storage | Memory-only ticket storage | Persistent storage with secure encryption |

**Recommended File/Module Structure**

The extensions require additional modules while maintaining clean separation from the core TLS implementation:

```
https_client/
  core/                           ← existing core implementation
    tls_connection.py
    handshake_engine.py
    crypto_engine.py
  extensions/
    session_resumption.py         ← PSK and ticket handling
    early_data.py                ← 0-RTT implementation  
    cipher_suites.py             ← additional AEAD algorithms
  performance/
    connection_pool.py           ← connection pooling logic
    async_client.py              ← async/await TLS client
    hardware_accel.py            ← CPU feature detection and acceleration
  utils/
    crypto_providers.py          ← pluggable crypto backend interface
    performance_monitor.py       ← metrics and profiling
```

**Session Resumption Infrastructure**

Complete session cache implementation with PSK management and ticket handling:

```python
import time
import secrets
import threading
from typing import Dict, Optional, Tuple
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

class SessionTicket:
    """Represents a TLS session ticket with associated metadata."""
    def __init__(self, ticket_data: bytes, resumption_secret: bytes, 
                 cipher_suite: int, server_name: str, issued_time: float,
                 lifetime_seconds: int):
        self.ticket_data = ticket_data
        self.resumption_secret = resumption_secret
        self.cipher_suite = cipher_suite
        self.server_name = server_name
        self.issued_time = issued_time
        self.lifetime_seconds = lifetime_seconds
        self.usage_count = 0
    
    def is_expired(self) -> bool:
        """Check if ticket has exceeded its lifetime."""
        return time.time() > (self.issued_time + self.lifetime_seconds)
    
    def derive_psk(self, ticket_nonce: bytes) -> bytes:
        """Derive PSK from resumption secret and ticket nonce."""
        hkdf = HKDFExpand(
            algorithm=hashes.SHA256(),
            length=32,
            info=b"resumption" + ticket_nonce
        )
        return hkdf.derive(self.resumption_secret)

class SessionCache:
    """Thread-safe session ticket cache with automatic cleanup."""
    def __init__(self, max_tickets: int = 1000, cleanup_interval: int = 300):
        self.max_tickets = max_tickets
        self.cleanup_interval = cleanup_interval
        self.tickets: Dict[str, SessionTicket] = {}
        self.lock = threading.RWLock()
        self.last_cleanup = time.time()
    
    def store_ticket(self, server_name: str, ticket: SessionTicket) -> None:
        """Store session ticket with automatic cleanup."""
        with self.lock.writer():
            # Cleanup expired tickets if needed
            current_time = time.time()
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_expired_tickets()
                self.last_cleanup = current_time
            
            # Enforce size limit with LRU eviction
            if len(self.tickets) >= self.max_tickets:
                oldest_key = min(self.tickets.keys(), 
                               key=lambda k: self.tickets[k].issued_time)
                del self.tickets[oldest_key]
            
            self.tickets[server_name] = ticket
    
    def get_ticket(self, server_name: str) -> Optional[SessionTicket]:
        """Retrieve valid session ticket for server."""
        with self.lock.reader():
            ticket = self.tickets.get(server_name)
            if ticket and not ticket.is_expired():
                ticket.usage_count += 1
                return ticket
            return None
    
    def _cleanup_expired_tickets(self) -> None:
        """Remove expired tickets from cache."""
        expired_keys = [key for key, ticket in self.tickets.items() 
                       if ticket.is_expired()]
        for key in expired_keys:
            del self.tickets[key]
```

**Connection Pool Infrastructure**

Complete connection pool implementation with health monitoring and thread safety:

```python
import threading
import time
import queue
from typing import Dict, Optional, Set
from dataclasses import dataclass
from .tls_connection import TLSConnection

@dataclass
class PooledConnection:
    """Wrapper for pooled TLS connections with metadata."""
    connection: TLSConnection
    created_time: float
    last_used_time: float
    usage_count: int
    is_healthy: bool = True

class ConnectionPool:
    """Thread-safe connection pool with per-host management."""
    
    def __init__(self, max_connections_per_host: int = 10, 
                 max_idle_time: int = 300, health_check_interval: int = 60):
        self.max_connections_per_host = max_connections_per_host
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        
        # Per-host connection queues
        self.pools: Dict[str, queue.Queue] = {}
        self.pool_locks: Dict[str, threading.Lock] = {}
        self.connection_counts: Dict[str, int] = {}
        
        # Global state management
        self.global_lock = threading.Lock()
        self.active_connections: Set[TLSConnection] = set()
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop, daemon=True)
        self.shutdown_event = threading.Event()
        self.health_monitor_thread.start()
    
    def get_connection(self, hostname: str, port: int, timeout: float = 5.0) -> TLSConnection:
        """Get connection from pool or create new one."""
        # TODO 1: Get or create per-host pool and lock
        # TODO 2: Try to get existing connection from pool
        # TODO 3: Validate connection health and return if good
        # TODO 4: If no pool connection available, check connection limit
        # TODO 5: Create new connection if under limit
        # TODO 6: Update connection tracking and return connection
        # Hint: Use queue.Queue.get(block=False) to avoid blocking
        pass
    
    def return_connection(self, hostname: str, connection: TLSConnection) -> None:
        """Return connection to pool or close if unhealthy."""
        # TODO 1: Validate connection is still healthy
        # TODO 2: Update last_used_time on pooled connection
        # TODO 3: Return to appropriate host pool if healthy
        # TODO 4: Close and remove from tracking if unhealthy
        # Hint: Check for TLS alerts or socket errors to determine health
        pass
    
    def _health_monitor_loop(self) -> None:
        """Background thread to monitor connection health."""
        while not self.shutdown_event.wait(self.health_check_interval):
            # TODO 1: Iterate through all host pools
            # TODO 2: Check each pooled connection for idle timeout
            # TODO 3: Perform lightweight health check (socket status)
            # TODO 4: Remove unhealthy or expired connections
            # TODO 5: Update pool statistics and connection counts
            pass
    
    def close_all(self) -> None:
        """Close all connections and shutdown pool."""
        self.shutdown_event.set()
        with self.global_lock:
            for connection in self.active_connections:
                try:
                    connection.close()
                except:
                    pass  # Ignore errors during shutdown
            self.active_connections.clear()
            self.pools.clear()
```

**Async TLS Client Foundation**

Core async implementation with proper resource management:

```python
import asyncio
import ssl
from typing import Optional, Tuple, Dict, Any
from .tls_connection import TLSConnection
from .connection_pool import ConnectionPool

class AsyncTLSConnection:
    """Asynchronous TLS connection wrapper."""
    
    def __init__(self, hostname: str, port: int):
        self.hostname = hostname
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.tls_connection: Optional[TLSConnection] = None
        self.connection_lock = asyncio.Lock()
    
    async def connect(self, timeout: float = 10.0) -> None:
        """Establish async TLS connection with timeout."""
        # TODO 1: Create asyncio TCP connection with timeout
        # TODO 2: Wrap streams with TLS connection handler
        # TODO 3: Perform TLS handshake in thread pool
        # TODO 4: Set up async read/write interfaces
        # TODO 5: Configure connection for async operation
        # Hint: Use asyncio.wait_for() for timeout handling
        pass
    
    async def send_request(self, data: bytes) -> None:
        """Send data over async TLS connection."""
        async with self.connection_lock:
            # TODO 1: Encrypt data using TLS connection
            # TODO 2: Write to async stream with backpressure handling
            # TODO 3: Await drain to ensure data is sent
            # TODO 4: Handle partial writes and connection errors
            # Hint: writer.write() is not coroutine, but drain() is
            pass
    
    async def receive_response(self, max_bytes: int = 16384) -> bytes:
        """Receive data from async TLS connection."""
        async with self.connection_lock:
            # TODO 1: Read from async stream with timeout
            # TODO 2: Accumulate TLS records until complete
            # TODO 3: Decrypt using TLS connection
            # TODO 4: Return decrypted application data
            # TODO 5: Handle connection errors and TLS alerts
            # Hint: Use reader.read() for non-blocking reads
            pass

class AsyncHTTPSClient:
    """High-level async HTTPS client with connection pooling."""
    
    def __init__(self, max_connections: int = 100):
        self.connection_pool = ConnectionPool(max_connections)
        self.session_stats = {"requests": 0, "connections": 0}
    
    async def request(self, method: str, url: str, headers: Dict[str, str] = None,
                     body: bytes = b"", timeout: float = 30.0) -> Tuple[int, Dict[str, str], bytes]:
        """Make async HTTPS request with automatic connection management."""
        # TODO 1: Parse URL to extract hostname, port, path
        # TODO 2: Get or create async connection from pool
        # TODO 3: Build HTTP request with proper headers
        # TODO 4: Send request and await response
        # TODO 5: Parse HTTP response and return structured data
        # TODO 6: Return connection to pool for reuse
        # Hint: Use asyncio.timeout() for request timeout
        pass
```

**Milestone Checkpoints**

After implementing advanced features, validate functionality with these checkpoints:

1. **Session Resumption Validation**:
   - Connect to a TLS server twice with the same client
   - Verify second connection uses PSK instead of full handshake
   - Check that session ticket is properly stored and retrieved
   - Validate that resumption provides performance improvement

2. **Connection Pool Testing**:
   - Make multiple requests to same server rapidly
   - Verify connections are reused from pool
   - Test pool limits and eviction under load
   - Validate proper connection cleanup on idle timeout

3. **Async Client Performance**:
   - Create 100+ concurrent connections to test server
   - Measure throughput compared to synchronous client
   - Verify proper resource cleanup and no connection leaks
   - Test graceful handling of connection errors

**Hardware Acceleration Detection**

Runtime CPU feature detection for crypto acceleration:

```python
import platform
import subprocess
from typing import Dict, Set, Optional

class CPUFeatureDetector:
    """Detect available CPU cryptographic acceleration features."""
    
    def __init__(self):
        self.features: Dict[str, bool] = {}
        self._detect_features()
    
    def _detect_features(self) -> None:
        """Detect available CPU features for crypto acceleration."""
        if platform.machine().lower() in ['x86_64', 'amd64']:
            self._detect_x86_features()
        elif platform.machine().lower() in ['aarch64', 'arm64']:
            self._detect_arm_features()
    
    def _detect_x86_features(self) -> None:
        """Detect x86/x64 cryptographic features."""
        # TODO 1: Check for AES-NI support using CPUID
        # TODO 2: Detect SHA extensions availability
        # TODO 3: Check for RDRAND/RDSEED instructions
        # TODO 4: Detect carry-less multiplication (PCLMULQDQ)
        # TODO 5: Test actual performance of detected features
        # Hint: Use ctypes to call CPUID or parse /proc/cpuinfo
        pass
    
    def has_aes_acceleration(self) -> bool:
        """Check if AES hardware acceleration is available."""
        return self.features.get('aes_ni', False)
    
    def has_sha_acceleration(self) -> bool:
        """Check if SHA hardware acceleration is available.""" 
        return self.features.get('sha_ext', False)
```

The advanced features implementation requires careful testing, performance validation, and security review. Each extension should be implemented incrementally with comprehensive testing to ensure it maintains the security properties of the base TLS implementation while providing the expected functionality and performance benefits.


## Glossary

> **Milestone(s):** All milestones (1-4) — this glossary provides essential terminology and concept definitions used throughout the HTTPS client implementation

Think of this glossary as your navigation map through the complex landscape of TLS and cryptographic terminology. Just as explorers need to understand the local language and landmarks before venturing into uncharted territory, developers implementing TLS need to master the specialized vocabulary that describes security protocols, cryptographic operations, and network communication patterns. Each term in this glossary represents a critical concept that appears repeatedly throughout the implementation, and understanding these definitions will help you navigate the technical discussions in each milestone with confidence.

The terminology in TLS spans multiple domains: network protocols, cryptographic algorithms, security concepts, and software engineering patterns. This intersection of domains means that a single operation, like encrypting application data, involves concepts from all these areas working together. Understanding each piece of terminology in isolation helps you see how they combine to create the complete security system.

### TLS Protocol Core Terms

**TLS (Transport Layer Security)** represents the modern security protocol that creates an encrypted, authenticated communication channel between client and server. Think of TLS as a diplomatic protocol that establishes trust between two parties who have never met, negotiates the rules for their communication, and ensures that their subsequent conversation remains private and tamper-proof. TLS evolved from SSL (Secure Sockets Layer) and has gone through multiple versions, with TLS 1.3 being the latest major revision that simplifies the handshake process and improves security properties.

**HTTPS (HTTP over TLS)** is the combination of the HTTP application protocol with TLS transport security. Like sending a letter inside a locked diplomatic pouch, HTTPS wraps ordinary web communication inside TLS encryption. The client connects to port 443 (instead of HTTP's port 80), performs a TLS handshake to establish encryption, then sends normal HTTP requests and receives HTTP responses over the encrypted channel. This provides confidentiality, integrity, and authenticity for web communication.

**Handshake messages** represent the negotiation phase where client and server agree on security parameters and establish encryption keys. Think of the handshake as a formal diplomatic protocol with specific steps that must be followed in order: introductions (ClientHello/ServerHello), credentials exchange (Certificate), key agreement (KeyShare), and final verification (Finished). Each handshake message has a specific format, serves a particular purpose in the security protocol, and contributes to the overall cryptographic binding between client and server.

**Record layer** provides the fundamental framing and message classification system for TLS. Like the postal system that handles envelopes regardless of their contents, the record layer packages all TLS communication into standardized records with headers indicating content type, protocol version, and payload length. Records can contain handshake messages, application data, alerts, or change cipher spec messages. The record layer handles fragmentation when messages exceed the maximum record size and reassembly when records arrive across multiple TCP segments.

**Cryptographic state** encompasses all the security-related information that defines a TLS connection's current security context. This includes the negotiated cipher suite, derived encryption keys, initialization vectors, sequence numbers, handshake context hash, and connection state. Think of cryptographic state as the security configuration that both client and server maintain in parallel, ensuring they remain synchronized throughout the communication process. Any mismatch in cryptographic state between client and server will result in communication failure.

### Key Exchange and Cryptography Terms

**ECDHE (Elliptic Curve Diffie-Hellman Ephemeral)** provides forward secrecy by using temporary key pairs that exist only for the duration of a single TLS handshake. Think of ECDHE as two people independently choosing secret numbers, performing mathematical operations to create public values they can share openly, then using those public values to compute a shared secret that only they know. The "ephemeral" aspect means these keys are discarded after use, so even if long-term keys are later compromised, past communications remain secure.

**X25519** represents a specific elliptic curve designed for ECDH key exchange that balances security, performance, and implementation simplicity. Unlike traditional curves like P-256 that require careful implementation to avoid side-channel attacks, X25519 is designed to be fast and secure even with straightforward implementations. The curve produces 32-byte public keys and 32-byte shared secrets, making it efficient for network transmission and cryptographic operations.

**Forward secrecy** ensures that compromise of long-term cryptographic material does not enable decryption of past communication sessions. Think of forward secrecy as burning the bridge behind you - even if someone later captures your long-term keys, they cannot use them to decrypt traffic they recorded from previous sessions. ECDHE provides forward secrecy by using ephemeral keys that are discarded after each handshake, ensuring each session has unique encryption keys derived from temporary material.

**HKDF (HMAC-based Key Derivation Function)** transforms shared secrets into cryptographically strong keys suitable for encryption and authentication. Like a sophisticated recipe that turns basic ingredients into a gourmet meal, HKDF takes the raw shared secret from ECDHE and systematically derives all the different keys needed for TLS: handshake traffic secrets, application traffic secrets, encryption keys, and initialization vectors. HKDF ensures that all derived keys are cryptographically independent and suitable for their specific purposes.

**Key derivation** represents the systematic process of generating multiple cryptographic keys from a single shared secret using cryptographically sound methods. In TLS 1.3, key derivation follows a specific sequence: early secret, handshake secret, master secret, and finally the traffic secrets that produce actual encryption keys and IVs. Each step in the derivation process incorporates additional context information, ensuring that keys are bound to the specific handshake and cannot be used in different contexts.

### Authentication and Certificate Terms

**X.509** defines the standard format for digital certificates used to bind public keys to identities. Think of an X.509 certificate as a digital passport issued by a trusted authority, containing the holder's public key, identity information (like hostname), validity period, and the issuer's digital signature. Just as passport authorities vouch for the identity of passport holders, certificate authorities vouch for the binding between public keys and the entities that claim to own them.

**Certificate chain** represents the sequence of certificates from a server's identity certificate up to a trusted root certificate authority. Like a chain of trust in the physical world where you trust someone because you trust the person who vouched for them, certificate chains allow clients to verify server identity by following the chain of digital signatures from the server certificate to a root CA certificate the client already trusts. Each certificate in the chain is signed by the issuer's private key, creating a cryptographic link.

**Subject Alternative Name (SAN)** provides the X.509 extension that lists all hostnames and IP addresses covered by a certificate. Since the internet evolved to use many different hostnames for the same service (like www.example.com, example.com, api.example.com), the SAN extension allows a single certificate to cover multiple names. During hostname validation, the client checks that the hostname it connected to appears in either the certificate's subject field or the SAN extension.

**Certificate validation** encompasses the complete process of verifying that a server's certificate is valid, trusted, and applicable to the current connection. This includes checking that signatures are valid and trace back to a trusted root, that the certificate has not expired, that the hostname matches the certificate's subject or SAN fields, and that the certificate has not been revoked. Certificate validation is critical for preventing man-in-the-middle attacks where attackers present fraudulent certificates.

### Encryption and Data Protection Terms

**AEAD (Authenticated Encryption with Associated Data)** provides encryption that simultaneously ensures confidentiality and authenticity of data. Think of AEAD as a tamper-evident security envelope that not only hides the contents but also proves whether the envelope has been opened or modified. AEAD algorithms like AES-GCM take plaintext, a secret key, a unique nonce, and optional associated data, producing ciphertext with an integrated authentication tag that proves the data hasn't been tampered with.

**AES-GCM (Advanced Encryption Standard in Galois/Counter Mode)** combines AES block cipher encryption with Galois field authentication to provide AEAD functionality. AES provides confidentiality by transforming plaintext into ciphertext using a secret key, while GCM mode adds counter-based encryption and polynomial-based authentication. The result is an encryption scheme that is both fast (especially with hardware acceleration) and secure, providing both privacy and integrity protection in a single operation.

**Nonce (Number Used Once)** represents a unique value that must never be repeated with the same encryption key. Think of a nonce as a unique serial number for each encryption operation - just as no two products should have the same serial number, no two encryption operations should use the same nonce with the same key. In TLS, nonces are constructed by XORing a per-connection IV with an incrementing sequence number, ensuring uniqueness across all records in a connection.

**Authentication tag** provides the cryptographic proof that data has not been tampered with during transmission. Generated by AEAD encryption algorithms, the authentication tag is computed over both the ciphertext and any associated data using the same secret key used for encryption. Recipients can verify the tag to detect any modification, truncation, or substitution of the encrypted data. If tag verification fails, the entire record must be rejected as potentially compromised.

**Sequence numbers** provide replay protection by ensuring that each TLS record has a unique identifier that increments with each message. Think of sequence numbers as timestamps that ensure messages can only be used once and in the correct order. The sequence number is incorporated into nonce generation for encryption and is implicitly verified during decryption. If records arrive out of order or are replayed, the sequence number mismatch will be detected.

### Network and Protocol Terms

**Network byte order** refers to the big-endian byte ordering convention used in network protocols where the most significant byte comes first. Think of network byte order as reading numbers left-to-right just like text - the leftmost digit is most significant. This standardization ensures that all network participants interpret multi-byte integers the same way, regardless of their underlying CPU architecture. TLS follows network byte order for all length fields and numeric values in message headers.

**Binary parsing** represents the process of extracting structured data from byte streams according to predefined formats. Like reading a precisely formatted document where each piece of information appears at a specific location with a specific length, binary parsing interprets raw bytes as integers, strings, and complex data structures. TLS message parsing requires careful attention to field lengths, byte ordering, and variable-length encodings to correctly reconstruct message contents.

**Fragmentation** occurs when large messages must be split across multiple TLS records due to size limitations. Think of fragmentation like sending a long letter across multiple postcards - each postcard can only hold a limited amount of text, so the complete message must be reconstructed by combining multiple pieces in the correct order. TLS records have a maximum payload size of 16,384 bytes, so larger handshake messages or application data must be fragmented and later reassembled.

**Record framing** provides the standardized format for packaging all TLS communication into discrete messages with headers indicating content type, version, and length. Like postal envelopes that provide addressing and handling information regardless of their contents, TLS records provide a uniform wrapper for handshake messages, application data, alerts, and control messages. The 5-byte record header allows recipients to determine how to process each record's payload.

### State Management and Control Flow Terms

**Connection state machine** represents the formal model that defines valid TLS connection states and the events that trigger transitions between states. Think of the state machine as a railroad switching system where trains (messages) can only move along valid tracks (state transitions) to reach their destinations (protocol states). Each state represents a specific phase of the TLS protocol, and messages can only be sent or received when the connection is in appropriate states.

**Handshake context** maintains the cryptographic binding of all handshake messages through a running hash that incorporates each message as it's sent or received. Think of the handshake context as a tamper-evident seal that covers the entire negotiation process - any modification to any handshake message will change the final hash value, allowing both parties to detect tampering. The handshake context hash is used to compute Finished message verify data and bind the handshake to derived keys.

**State transition** represents the atomic movement from one connection state to another triggered by message processing or error conditions. Like stepping stones across a river where you can only move to adjacent stones, state transitions in TLS must follow valid paths defined by the protocol. Each transition may trigger actions like key derivation, message transmission, or error handling, and all components must remain synchronized during state changes.

**Component coordination** ensures that all parts of the TLS implementation remain synchronized and operate according to the current connection state. Think of component coordination like conducting an orchestra where different sections (TCP transport, crypto engine, certificate validator) must play their parts at the right times and in harmony with each other. Poor coordination leads to race conditions, inconsistent state, and protocol violations.

### Error Handling and Security Terms

**TLS alerts** provide the protocol's error reporting and notification mechanism, allowing endpoints to signal various conditions ranging from protocol errors to successful connection closure. Think of TLS alerts as standardized messages in a diplomatic protocol that allow parties to communicate about procedural issues without breaking the overall communication framework. Alerts are classified as warning or fatal, with fatal alerts requiring immediate connection termination.

**Fatal alerts** indicate errors that compromise the security or integrity of the TLS connection and require immediate termination. Like fire alarms that demand immediate evacuation, fatal alerts signal conditions where continuing the connection would be unsafe. Examples include certificate validation failures, decryption errors, and protocol violations. When a fatal alert is sent or received, both endpoints must close the connection and discard all associated state.

**Warning alerts** signal non-fatal conditions that allow the connection to continue but may indicate potential issues. Like yellow warning lights that indicate caution but don't require immediate shutdown, warning alerts inform the peer about conditions that might affect communication quality or security. The most common warning alert is close_notify, which provides graceful connection closure without implying any security problems.

**Close_notify** provides the standardized mechanism for gracefully terminating TLS connections without creating vulnerability to truncation attacks. Think of close_notify as saying "goodbye" in a way that confirms the conversation is ending intentionally rather than being cut off by an attacker. Both endpoints should send close_notify when they finish sending data, and recipients should acknowledge close_notify before closing the underlying TCP connection.

### Testing and Debugging Terms

**Test vectors** represent predetermined inputs and expected outputs for cryptographic operations that allow implementations to verify correctness against known good results. Think of test vectors as answer keys for cryptographic homework - they provide specific inputs and the exact outputs that a correct implementation should produce. Test vectors are essential for validating that cryptographic operations, key derivation, and message construction produce standard-compliant results.

**Integration testing** validates that TLS components work correctly together and can successfully communicate with real-world servers and clients. Unlike unit testing that checks individual components in isolation, integration testing verifies that the complete system can perform actual TLS handshakes, exchange application data, and handle error conditions correctly. This includes testing against major web servers and other TLS implementations to ensure interoperability.

**Wire format** describes the exact byte-level representation of protocol messages as they appear on the network. Think of wire format as the physical blueprint that shows exactly how abstract data structures are encoded into bytes for transmission. Understanding wire format is crucial for debugging protocol issues, analyzing network captures, and ensuring that message construction produces standard-compliant output.

**Hex dump** provides a human-readable representation of binary data showing both hexadecimal byte values and ASCII character interpretation. Like a translator that converts binary data into a format humans can analyze, hex dumps allow developers to examine message contents, verify parsing correctness, and debug protocol issues. Hex dumps typically display 16 bytes per line with both hex values and ASCII representation.

### Performance and Optimization Terms

**Session resumption** allows TLS connections to reuse cryptographic state from previous sessions, reducing handshake overhead and improving connection establishment performance. Think of session resumption as a VIP pass that lets you skip the full security checkpoint because you've already been verified recently. TLS 1.3 uses PSK-based resumption where the server provides encrypted session tickets that clients can present in subsequent connections to resume with the same security parameters.

**0-RTT data (Zero Round-Trip Time)** enables clients to send application data immediately with the first message to the server when resuming sessions, eliminating the latency of the handshake process. Like having a pre-approved security clearance that lets you enter immediately without waiting for verification, 0-RTT allows applications to achieve optimal performance by sending data before the handshake completes. However, 0-RTT data has specific security limitations and replay attack vulnerabilities.

**Connection pooling** maintains reusable TLS connections to amortize the cost of handshake operations across multiple application requests. Think of connection pooling like maintaining a fleet of vehicles ready for use rather than buying a new car for each trip. Connection pools manage the lifecycle of TLS connections, handle connection validation, and provide efficient allocation and cleanup of network resources.

**Hardware acceleration** leverages CPU-specific instructions to optimize cryptographic operations, particularly AES encryption and SHA hashing. Think of hardware acceleration like using specialized tools for specific jobs - just as a purpose-built tool works faster and more efficiently than improvised solutions, CPU instructions designed for cryptography can perform these operations much faster than software implementations. Modern CPUs include AES-NI and other cryptographic instruction sets.

### Advanced TLS Features Terms

**PSK (Pre-Shared Key)** enables TLS connections to authenticate using previously established shared secrets rather than certificate-based authentication. Think of PSK as using a secret password that both parties already know, like a diplomatic code word established in previous meetings. In TLS 1.3, PSKs are derived from previous sessions and enable both session resumption and 0-RTT data transmission. PSKs provide perfect forward secrecy when combined with ephemeral key exchange.

**Session tickets** are encrypted tokens that contain session state information, allowing servers to avoid maintaining per-client state while still enabling session resumption. Think of session tickets as claim tickets at a coat check - the server gives you an encrypted token containing your session information, and you can present this token later to resume your session. The server encrypts tickets with its own key, so it can decrypt and validate them when clients present them for resumption.

**ChaCha20-Poly1305** provides an alternative AEAD cipher suite that uses stream cipher encryption with polynomial authentication, particularly beneficial on platforms without AES hardware acceleration. Think of ChaCha20-Poly1305 as an alternative route to the same destination - it provides the same security properties as AES-GCM but uses different mathematical operations that may perform better on certain hardware platforms, especially mobile devices and embedded systems.

**Event loop** represents the programming construct used in asynchronous I/O systems to handle multiple concurrent operations without blocking threads. Think of an event loop like a skilled restaurant host managing multiple tables - constantly checking which tables need attention, handling requests as they arrive, and ensuring smooth operation without getting stuck waiting at any single table. Event loops enable efficient handling of many simultaneous TLS connections.

This comprehensive terminology foundation prepares you for the technical discussions throughout the HTTPS client implementation. Each term represents a critical concept that appears repeatedly across different milestones, and understanding these definitions will help you navigate the complexity of TLS protocol implementation with confidence.

### Implementation Guidance

Understanding TLS terminology is essential for successful implementation, but connecting abstract concepts to concrete code requires bridging the gap between protocol specifications and programming constructs. This section provides practical guidance for how these terms translate into implementation decisions and coding patterns.

#### Term-to-Implementation Mapping

The terminology above maps directly to implementation concepts you'll encounter throughout the project. Here's how to connect abstract terms to concrete programming tasks:

| Term Category | Implementation Focus | Key Programming Concepts |
|---------------|---------------------|-------------------------|
| Protocol Terms | State machines, message parsing | Enums for states, struct definitions for messages |
| Cryptographic Terms | Library integration, key management | Cryptography library bindings, secure memory handling |
| Network Terms | Binary protocol handling | Byte array manipulation, endianness conversion |
| State Management | Concurrency, synchronization | Locks, atomic operations, state validation |
| Error Handling | Exception hierarchies, recovery | Error types, retry logic, cleanup procedures |

#### Terminology in Code Comments and Documentation

When implementing the HTTPS client, use this terminology consistently in your code comments and documentation. This creates a clear mapping between the protocol specification and your implementation:

```python
# TLS Protocol Implementation - Core Terminology Usage

class TLSConnection:
    """
    Manages the complete TLS connection lifecycle including handshake state machine,
    cryptographic state, and application data processing.
    
    Key terminology:
    - Connection state machine: tracks protocol progression through handshake phases
    - Cryptographic state: maintains keys, IVs, and security parameters
    - Record layer: handles TLS record framing and message classification
    """
    
    def perform_handshake(self, hostname: str, port: int) -> bool:
        """
        Execute complete TLS handshake sequence.
        
        Handshake terminology:
        - ClientHello: initial client message with cipher suites and extensions
        - ECDHE: ephemeral key exchange providing forward secrecy
        - Certificate validation: X.509 chain verification and hostname matching
        - Key derivation: HKDF-based generation of traffic keys from shared secret
        - Finished messages: handshake authentication via verify data
        """
        # Implementation follows the terminology established above
```

#### Debugging with Terminology

When debugging TLS implementations, use the established terminology to describe issues precisely. This creates clear communication with other developers and helps identify problems faster:

```python
def analyze_handshake_failure(self, error_context: Dict) -> str:
    """
    Analyze handshake failures using precise TLS terminology.
    
    Common failure patterns mapped to terminology:
    - "Certificate validation failed": X.509 chain verification or hostname mismatch
    - "Key derivation error": HKDF operation failed or sequence number mismatch  
    - "Record layer parsing failed": binary parsing error or fragmentation issue
    - "State transition invalid": attempted operation not valid in current state
    - "AEAD decryption failed": authentication tag verification failed
    """
    # Use terminology consistently in error analysis and reporting
```

#### Language-Specific Term Implementation

Different programming languages have various ways to represent TLS concepts. Here's how Python specifically implements key terminology:

```python
# Python-Specific Implementation of TLS Terminology

from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import struct

# Connection State Machine - implementing state terminology
class ConnectionState(IntEnum):
    START = 1           # Initial state before handshake
    WAIT_SH = 2        # Waiting for ServerHello
    WAIT_EE = 3        # Waiting for EncryptedExtensions  
    WAIT_CERT_CR = 4   # Waiting for Certificate
    WAIT_CV = 5        # Waiting for CertificateVerify
    WAIT_FINISHED = 6  # Waiting for server Finished message
    CONNECTED = 7      # Handshake complete, application data flow
    CLOSED = 8         # Connection terminated

# Record Layer - implementing framing terminology
class TLSRecord:
    """
    TLS record implementing record layer terminology:
    - Content type classification (handshake, application data, alert)
    - Protocol version identification  
    - Length-prefixed payload framing
    - Binary parsing for wire format compatibility
    """
    
    def __init__(self, content_type: int, legacy_version: int, 
                 length: int, payload: bytes):
        self.content_type = content_type      # Record type classification
        self.legacy_version = legacy_version  # Protocol version (always 0x0303)
        self.length = length                  # Payload length in network byte order
        self.payload = payload                # Record payload data

# Cryptographic State - implementing security terminology        
class CryptographicState:
    """
    Maintains cryptographic state terminology:
    - Cipher suite negotiation results
    - Key derivation state from HKDF operations
    - Handshake context hash for message binding
    - Traffic keys for AEAD encryption
    - Sequence numbers for replay protection
    """
    
    def __init__(self):
        self.cipher_suite: Optional[int] = None
        self.client_random: bytes = b''
        self.server_random: bytes = b''
        self.handshake_context: List[bytes] = []
        self.traffic_keys: Optional[TrafficKeys] = None
        self.sequence_numbers: SequenceNumbers = SequenceNumbers()
```

#### Milestone Terminology Checkpoints

Each project milestone introduces specific terminology that you should master:

**Milestone 1 (TCP & Record Layer)**: Focus on network and framing terminology - binary parsing, network byte order, record layer, fragmentation handling.

**Milestone 2 (ClientHello)**: Master handshake and negotiation terminology - cipher suites, extensions (SNI, supported versions, key share), protocol version negotiation.

**Milestone 3 (Key Exchange)**: Understand cryptographic terminology - ECDHE, forward secrecy, HKDF, key derivation, certificate validation, X.509, handshake context.

**Milestone 4 (Encrypted Communication)**: Apply encryption and security terminology - AEAD, AES-GCM, nonce construction, sequence numbers, authentication tags, close_notify.

#### Terminology Consistency Validation

Validate your implementation's terminology usage by ensuring consistent naming throughout your codebase:

```python
# Terminology Validation Checklist
def validate_terminology_consistency():
    """
    Ensure consistent use of TLS terminology throughout implementation.
    
    Check for:
    - Consistent state machine state names matching TLS specifications
    - Cryptographic operation names following IANA registry terminology  
    - Message type constants using standard TLS naming conventions
    - Error messages incorporating precise TLS terminology
    - Comments and documentation using established protocol terms
    """
    
    # Example validation points:
    assert hasattr(ConnectionState, 'WAIT_FINISHED')  # Not 'WAITING_FOR_FINISHED'
    assert hasattr(ContentType, 'HANDSHAKE')          # Not 'HANDSHAKE_MESSAGE'
    assert hasattr(CipherSuite, 'TLS_AES_128_GCM_SHA256')  # Exact IANA name
```

This systematic approach to terminology ensures that your implementation remains consistent with TLS specifications and creates clear communication between your code and the broader TLS development community.

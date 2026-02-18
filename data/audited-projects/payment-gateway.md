# AUDIT & FIX: payment-gateway

## CRITIQUE
- **Audit Finding 1 (PCI-DSS/Tokenization):** Confirmed critical. The project claims PCI DSS compliance as a learning outcome and skill, but no milestone addresses tokenization of card data, encryption at rest, or minimizing cardholder data scope. This is a glaring omission for an 'expert' project claiming PCI compliance.
- **Audit Finding 2 (Settlement/Payout):** Valid but less critical for a gateway *integration* project. However, the project never clarifies whether this is building a gateway from scratch or integrating with Stripe. The description says 'Payment processing with PCI compliance' but milestones heavily reference Stripe APIs. This ambiguity is a fundamental design flaw.
- **Milestone Sequencing Issue:** M2 references 3DS and payment processing but assumes M1's idempotency layer is complete. This is correct sequencing, but M4 (Webhook Reconciliation) should logically follow immediately after M2 since webhooks are the *primary* mechanism for payment status updates in async flows (3DS, bank transfers). Putting refunds/disputes (M3) before webhook handling means M3 has no reliable way to know payment status.
- **Estimated Hours Discrepancy:** 4 milestones × 11 hours = 44 hours, but project says 45. Minor but sloppy.
- **Missing Distributed Locking Detail:** M1 mentions idempotency but the acceptance criteria never specify *how* distributed locking is achieved (Redis SETNX, DB advisory locks, etc.). For an expert project, this is insufficiently rigorous.
- **4xx Retry Logic Absence:** M4 pitfalls mention out-of-order events but never address the critical distinction between retriable (5xx, network timeout) and non-retriable (4xx) webhook processing failures.
- **Security Gap:** No mention of TLS requirements, API key rotation, or rate limiting on the payment API itself. An expert payment project without rate limiting is dangerous.
- **Race Condition Coverage:** Learning outcomes mention 'race conditions' but no milestone AC specifically tests for or verifies concurrent payment confirmation scenarios.
- **Client Secret Exposure:** M1 pitfall mentions client secret but no AC verifies that client secrets are scoped to authenticated sessions only.

## FIXED YAML
```yaml
id: payment-gateway
name: Payment Gateway
description: >-
  Payment processing integration with idempotent transaction handling,
  webhook reconciliation, PCI-compliant tokenization, and settlement tracking.
difficulty: expert
estimated_hours: 55
essence: >-
  Distributed transaction state management across unreliable networks with
  cryptographic verification, idempotent request handling to prevent duplicate
  charges, asynchronous webhook reconciliation ensuring eventual consistency
  between payment provider events and internal ledger states, and PCI-compliant
  tokenization to minimize cardholder data exposure.
why_important: >-
  Building this teaches critical production payment system architecture patterns
  that power e-commerce platforms, including preventing duplicate charges that
  could cost businesses millions, implementing security standards required by
  financial regulations, and designing fault-tolerant reconciliation systems
  that ensure financial consistency across distributed services.
learning_outcomes:
  - Implement idempotency key validation with distributed locking to prevent duplicate payment processing
  - Design payment intent state machines handling pending, requires_action, succeeded, and failed transitions
  - Build 3D Secure authentication flows with challenge redirects and verification callbacks
  - Implement HMAC-SHA256 webhook signature verification with timing-safe comparison
  - Design asynchronous webhook processing queues with retry logic and exponential backoff
  - Implement partial refund calculations with balance reconciliation and dispute tracking
  - Build PCI DSS-compliant architectures using tokenization to eliminate cardholder data from your systems
  - Debug payment flow race conditions and handle network failures with proper rollback mechanisms
  - Implement settlement reconciliation comparing internal ledger state against provider payout reports
skills:
  - Idempotent API Design
  - Payment State Machines
  - HMAC Signature Verification
  - Webhook Processing
  - PCI DSS Compliance & Tokenization
  - Transaction Reconciliation
  - 3D Secure (3DS2)
  - Distributed Locking
  - Settlement Tracking
tags:
  - api-integration
  - expert
  - fintech
  - idempotency
  - payments
  - security
  - service
  - stripe
  - transactions
  - webhooks
  - pci-dss
architecture_doc: architecture-docs/payment-gateway/index.md
languages:
  recommended:
    - Python
    - Go
    - Java
  also_possible: []
resources:
  - name: Stripe Payment Intents API
    url: https://docs.stripe.com/payments/payment-intents
    type: documentation
  - name: Stripe Idempotent Requests Guide
    url: https://stripe.com/blog/idempotency
    type: article
  - name: PCI Security Standards Council
    url: https://www.pcisecuritystandards.org/standards/
    type: documentation
  - name: Stripe 3D Secure Documentation
    url: https://docs.stripe.com/payments/3d-secure
    type: documentation
  - name: Stripe Webhook Signature Verification
    url: https://docs.stripe.com/webhooks/signature
    type: tutorial
  - name: PCI DSS Tokenization Guidelines
    url: https://listings.pcisecuritystandards.org/documents/Tokenization_Guidelines_Info_Supplement.pdf
    type: documentation
prerequisites:
  - type: project
    id: http-server-basic
    name: HTTP Server (Basic)
  - type: skill
    name: Database transactions
  - type: skill
    name: Basic cryptography (hashing, HMAC)
milestones:
  - id: payment-gateway-m1
    name: Payment Intent & Idempotency
    description: >-
      Implement payment intents with idempotency keys and distributed locking
      to prevent duplicate charges.
    acceptance_criteria:
      - POST /payment-intents creates a payment intent with a unique ID, amount (integer cents), currency, and status 'created'
      - Client-provided Idempotency-Key header causes identical request to return the original response with HTTP 200, not create a duplicate
      - Idempotency key reused with different parameters (amount, currency) returns HTTP 422 Unprocessable Entity with a clear error message
      - Payment intent status transitions follow a strict state machine (created -> processing -> requires_action -> succeeded | failed | cancelled) with no backward transitions
      - Concurrent requests with the same idempotency key are serialized via distributed lock (e.g., Redis SETNX or DB advisory lock) so only one processes
      - Stale payment intents in 'created' status for more than 24 hours are automatically cancelled by a background job
      - All monetary amounts are stored as integers in the smallest currency unit (e.g., cents) - never as floating-point
    pitfalls:
      - Idempotency key reused with different params must error, not silently return the old response - this prevents silent data corruption
      - Client secret must be scoped to the authenticated user session only; never expose it in server logs or error responses
      - Amount in cents prevents floating-point errors ($10.00 = 1000 cents); using floats for money is an unrecoverable design mistake
      - Status transitions must be validated atomically - a race between two concurrent confirmations must not double-process
      - Idempotency keys should have a TTL (e.g., 24-48 hours) to prevent unbounded storage growth
      - Distributed lock must have a TTL to prevent deadlocks if the holder crashes
    concepts:
      - Idempotency key generation, storage, and TTL-based expiration
      - Payment intent state machine with explicit valid transitions
      - Request deduplication using unique constraints and distributed locks
      - Atomic database transactions for payment operations
      - Optimistic vs pessimistic concurrency control for payment state
    skills:
      - Idempotency patterns
      - State machines
      - Distributed locking
      - Atomic operations
    deliverables:
      - Payment intent creation API with unique ID, amount, currency, and metadata fields
      - Idempotency key middleware that checks for existing responses before processing
      - Distributed lock acquisition around idempotency key processing to serialize concurrent duplicates
      - State machine enforcing valid transitions with rejection of invalid state changes
      - Background job that cancels stale uncompleted payment intents after configurable TTL
    estimated_hours: 11

  - id: payment-gateway-m2
    name: Payment Processing & 3DS
    description: >-
      Implement payment confirmation with 3D Secure authentication flow
      and payment method tokenization.
    acceptance_criteria:
      - POST /payment-intents/{id}/confirm transitions intent from 'created' to 'processing' and submits the charge to the payment provider
      - When the issuer requires 3DS, status transitions to 'requires_action' with a redirect URL returned to the client
      - 3DS callback endpoint receives authentication result and transitions to 'succeeded' or 'failed' accordingly
      - Payment methods are tokenized via the provider (e.g., Stripe PaymentMethod) - raw card numbers never touch your server
      - Authorization holds are tracked with an expiration timestamp (typically 7 days); capture must occur before expiry
      - Currency of the confirm request is validated against the original intent currency; mismatches return HTTP 400
      - Multiple payment method types are supported (card, bank_transfer) with type-specific validation
    pitfalls:
      - 3DS redirects must include return_url for the user to come back; omitting it strands the user
      - Payment can succeed at the provider but your database update can fail - always treat provider state as authoritative and reconcile via webhooks
      - Auth-capture gap - authorization expires (typically 7 days for cards); failing to capture before expiry loses the authorization
      - Currency mismatch between intent creation and capture causes hard errors at the provider; validate early
      - Never log or store raw card numbers, CVV, or full track data - this violates PCI DSS and is a compliance breach
    concepts:
      - 3D Secure 2 authentication challenge-response protocol
      - Synchronous vs asynchronous payment confirmation flows
      - Payment method tokenization via provider SDKs
      - Authorization hold mechanics and settlement timing
      - PCI scope reduction through client-side tokenization
    skills:
      - 3DS flow
      - Payment processing
      - Tokenization
      - Error handling
    deliverables:
      - Payment confirmation endpoint that submits charges to the provider
      - 3D Secure redirect flow with return_url handling and callback processing
      - Payment method attachment using provider-generated tokens (never raw card data)
      - Authorization tracking with expiration monitoring and capture-before-expiry alerts
    estimated_hours: 11

  - id: payment-gateway-m3
    name: Webhook Reconciliation
    description: >-
      Implement reliable webhook processing with HMAC signature verification,
      idempotent handling, and periodic reconciliation against provider state.
    acceptance_criteria:
      - POST /webhooks endpoint receives payment event notifications from the provider
      - HMAC-SHA256 signature is verified using timing-safe comparison before processing any event; invalid signatures return HTTP 400
      - Webhook timestamp is validated to be within a 5-minute tolerance window to prevent replay attacks
      - Each webhook event is processed idempotently - redelivered events with the same event ID produce no duplicate side effects
      - Local payment intent status is updated based on webhook event data (e.g., payment_intent.succeeded, payment_intent.payment_failed)
      - Out-of-order events are handled correctly by comparing event timestamps and ignoring stale state transitions
      - Failed webhook processing is retried with exponential backoff (base 2s, max 5 retries) via a background job queue
      - Periodic reconciliation job (every 15 minutes) queries the provider API and corrects any local state drift
    pitfalls:
      - Never skip signature verification, even in development - this creates habits that lead to production security holes
      - Events can arrive out of order (e.g., payment_failed before payment_intent.created) - use event timestamps and state machine validation
      - The provider may retry webhooks for days - your handler must be idempotent with event ID deduplication
      - Webhook endpoint must return 2xx quickly (within 5 seconds for Stripe); offload heavy processing to a background queue
      - Periodic reconciliation catches silently dropped webhooks - never rely solely on webhooks for financial state
    concepts:
      - HMAC-SHA256 signature verification for webhook authenticity
      - Timing-safe string comparison to prevent timing attacks
      - Event ordering and causality preservation using timestamps
      - At-least-once delivery semantics and deduplication
      - Background job processing with retry and exponential backoff
      - Reconciliation patterns comparing local vs provider state
    skills:
      - Webhook security
      - HMAC verification
      - Event processing
      - Reconciliation
      - Idempotent handlers
    deliverables:
      - Webhook endpoint that accepts provider POST requests and returns 200 after queuing
      - HMAC-SHA256 signature verification with timing-safe comparison and timestamp tolerance
      - Event deduplication using event ID tracking to prevent duplicate processing
      - Background event processor that updates local payment state from webhook data
      - Periodic reconciliation job that queries provider API and corrects local state drift
    estimated_hours: 11

  - id: payment-gateway-m4
    name: Refunds & Disputes
    description: >-
      Implement refund processing, partial refunds, and chargeback dispute handling
      with evidence submission.
    acceptance_criteria:
      - POST /payment-intents/{id}/refund processes a full refund against a succeeded payment, transitioning to 'refunded'
      - Partial refund specifies an amount less than the original; total refunded amount is tracked and cannot exceed the original charge
      - Refund status is tracked through pending, succeeded, and failed stages via webhook updates from the provider
      - Chargeback webhook events create a dispute record with reason code, deadline, and status tracking
      - Dispute evidence submission API accepts text and file evidence before the provider deadline
      - Dispute deadline alerts are generated 48 hours before the evidence submission window closes
      - All refund and dispute operations are recorded in an immutable audit log with timestamps and actor identity
    pitfalls:
      - Refunds can take 5-10 business days to appear on customer statements - communicate this clearly
      - Dispute evidence deadline is strict and non-negotiable - automate alerts well in advance
      - Dispute fee is charged even if you win the dispute - prevention (good product, clear billing descriptor) is far cheaper
      - Partial refunds require careful tracking of total_refunded vs original_amount to prevent over-refunding
      - Refunding a disputed payment can result in double loss - check dispute status before allowing refunds
    concepts:
      - Refund state transitions and reversibility windows
      - Chargeback lifecycle and evidence submission workflows
      - Ledger-based accounting for partial refund tracking
      - Dispute reason codes (fraud, product not received, etc.) and response strategies
      - Audit trail requirements for financial operations
    skills:
      - Refund workflows
      - Dispute handling
      - Financial reconciliation
      - Audit logging
    deliverables:
      - Full and partial refund API with total-refunded tracking and over-refund prevention
      - Refund status tracking updated via webhook events from the provider
      - Dispute record creation from chargeback webhook events with deadline tracking
      - Evidence submission API for responding to disputes before the deadline
      - Automated deadline alerts generated 48 hours before evidence window closes
    estimated_hours: 11

  - id: payment-gateway-m5
    name: PCI Compliance & Settlement
    description: >-
      Implement PCI DSS-compliant architecture patterns, data encryption,
      and settlement reconciliation against provider payout reports.
    acceptance_criteria:
      - System architecture is documented showing that raw card data (PAN, CVV) never enters your server - only provider-generated tokens are stored
      - All stored payment data (tokens, customer info, transaction records) is encrypted at rest using AES-256
      - API endpoints enforce TLS 1.2+ and reject plaintext HTTP connections
      - Access to payment data is restricted by role-based access control with audit logging of all access
      - Settlement reconciliation job compares internal transaction records against provider payout/settlement reports
      - Discrepancies between internal records and provider settlements are flagged with automated alerts
      - PCI DSS self-assessment questionnaire (SAQ-A or SAQ-A-EP) scope is documented based on the architecture
    pitfalls:
      - Logging raw card data (even accidentally in error logs) is a PCI violation - scrub all logs
      - Encryption at rest is necessary but not sufficient - key management (rotation, access control) is equally critical
      - Settlement amounts may differ from charge amounts due to provider fees, currency conversion, and chargebacks
      - Provider payout timing varies (T+2 for Stripe, variable for others) - reconciliation windows must account for this
      - SAQ scope depends on how card data flows through your system - client-side tokenization (Stripe.js/Elements) qualifies for SAQ-A which is simplest
    concepts:
      - PCI DSS compliance levels and SAQ types (A, A-EP, D)
      - Tokenization as a scope-reduction strategy
      - Encryption at rest with AES-256 and key management
      - Settlement and payout reconciliation workflows
      - Role-based access control for payment data
    skills:
      - PCI DSS compliance architecture
      - Encryption at rest
      - Key management
      - Settlement reconciliation
      - Access control
    deliverables:
      - Architecture document proving raw card data never enters the system (SAQ-A eligible)
      - Encryption at rest for all stored payment data with key rotation mechanism
      - TLS enforcement middleware rejecting non-HTTPS connections
      - Role-based access control for payment data endpoints with audit logging
      - Settlement reconciliation job comparing internal ledger against provider payout reports
      - Discrepancy alerting system for settlement mismatches
    estimated_hours: 11
```
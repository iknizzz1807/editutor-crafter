# AUDIT & FIX: notification-service

## CRITIQUE
- **Audit Finding Confirmed (Missing Message Queue - CRITICAL):** The essence explicitly mentions 'queue-based decoupling' and 'at-least-once delivery semantics' but NO milestone implements a message broker. There is no RabbitMQ, Kafka, Redis Pub/Sub, or any queue. Without this, the service is a synchronous notification sender, not the asynchronous, reliable system described. This is a fundamental architectural gap.
- **Missing Retry & Dead Letter Queue:** The learning outcomes mention 'retry mechanisms with exponential backoff and dead-letter queues for failed deliveries' but no milestone has ACs for retry logic, backoff, or DLQ processing.
- **Missing Rate Limiting:** The learning outcomes mention 'rate limiting and circuit breakers for third-party API integrations' but no AC addresses this.
- **M1 Fallback AC Without Queue:** M1 AC says 'Fallback channels are attempted in order when the primary channel fails.' Without a queue, this means synchronous fallback in the request path, which is architecturally wrong for a notification service. Fallback should be handled by the queue consumer.
- **M4 Tracking Pixel Privacy:** The pitfall correctly identifies Apple Mail Privacy Protection inflating open rates, but the ACs don't mention any mitigation or acknowledgment that open tracking is inherently unreliable.
- **M3 Legal Compliance Gap:** Pitfalls mention CAN-SPAM and GDPR but the ACs don't require distinguishing transactional from marketing notifications or preventing unsubscription from transactional messages.
- **Uniform Hours:** All four milestones are 11 hours each. A template system (M2) is significantly less complex than building a multi-channel routing system with fallback (M1) or a delivery tracking pipeline (M4).
- **M2 XSS Concern Misplaced:** Pitfall mentions 'Template variables need escaping for XSS prevention in HTML.' For email HTML, XSS is not the right threat model (email clients don't execute JavaScript). The real concern is HTML injection that could alter email rendering or phishing link injection. For push/SMS, this is irrelevant.
- **Missing Idempotency:** The concepts mention 'Idempotency keys for delivery guarantees' but no AC requires idempotent processing of notification requests.

## FIXED YAML
```yaml
id: notification-service
name: Notification Service
description: >-
  Build an asynchronous, multi-channel notification service with queue-based
  decoupling, pluggable delivery channels, template rendering, user preferences,
  retry logic, and delivery tracking.
difficulty: expert
estimated_hours: "45-60"
essence: >-
  Asynchronous message routing across heterogeneous delivery channels with
  queue-based decoupling, webhook-driven status reconciliation, template
  interpolation, preference-based filtering, retry with exponential backoff
  and dead-letter queues to guarantee at-least-once delivery semantics
  across unreliable third-party APIs.
why_important: >-
  Building this teaches you distributed systems patterns essential for
  production infrastructure, including message queues, event-driven
  architecture, third-party API integration, retry strategies, and
  observability for asynchronous workflows that form the backbone of
  modern SaaS applications.
learning_outcomes:
  - Design queue-based architecture with RabbitMQ or equivalent for reliable async notification processing
  - Implement a plugin-based channel abstraction with strategy pattern for multi-provider routing
  - Build template engines with variable interpolation and localization support
  - Implement retry with exponential backoff and dead-letter queues for failed deliveries
  - Design user preference systems with opt-in/opt-out controls and regulatory compliance
  - Build webhook-based delivery status tracking with idempotent event processing
  - Implement rate limiting and circuit breakers for third-party API integrations
  - Track delivery metrics including delivery rates, bounce rates, and open/click data
skills:
  - Message Queue Architecture
  - Event-Driven Systems
  - Third-Party API Integration
  - Template Rendering Engines
  - Asynchronous Processing
  - Distributed Systems Patterns
  - Webhook Processing
  - Rate Limiting & Circuit Breakers
  - Retry & Dead Letter Queue Patterns
tags:
  - concurrency
  - email
  - expert
  - messaging
  - notifications
  - push
  - queues
  - service
  - sms
  - webhooks
architecture_doc: architecture-docs/notification-service/index.md
languages:
  recommended:
    - Go
    - Python
    - Java
  also_possible:
    - Node.js
    - Rust
resources:
  - name: RabbitMQ Tutorials
    url: https://www.rabbitmq.com/tutorials
    type: tutorial
  - name: Firebase Cloud Messaging
    url: https://firebase.google.com/docs/cloud-messaging
    type: documentation
  - name: Twilio SMS API
    url: https://www.twilio.com/docs/messaging/api
    type: documentation
  - name: SendGrid API Reference
    url: https://www.twilio.com/docs/sendgrid/api-reference
    type: documentation
prerequisites:
  - type: skill
    name: Message queue/pub-sub basics
  - type: project
    id: rest-api-design
    name: Production REST API
milestones:
  - id: notification-service-m1
    name: Queue Infrastructure & Channel Abstraction
    description: >-
      Set up the message queue infrastructure for async processing and
      implement pluggable notification channels with routing and fallback.
    acceptance_criteria:
      - "Message broker (RabbitMQ, Redis Streams, or equivalent) is configured with a notification queue and a dead-letter queue"
      - "API endpoint POST /notifications accepts notification requests, assigns an idempotency key, and enqueues them; returns 202 Accepted with a notification ID"
      - "Duplicate notification requests with the same idempotency key are detected and ignored (at-least-once but not double-delivered)"
      - "Queue consumer dequeues notifications and routes them to the appropriate channel based on notification type and user preference"
      - "Channel interface defines a common contract (send, validate_recipient, format_payload) implemented by email (SMTP/SendGrid), SMS (Twilio), and push (FCM/APNs) backends"
      - "Channel-specific formatting transforms the notification payload into the format required by each backend (HTML for email, 160-char segments for SMS, JSON for push)"
      - "Fallback channels are attempted in configured order when the primary channel fails; fallback attempts are logged"
      - "Rate limiting per channel type prevents exceeding third-party API rate limits (e.g., SendGrid 100/sec)"
    pitfalls:
      - "Processing notifications synchronously in the API request path defeats the purpose of queue-based decoupling"
      - "Provider fallback must track failure counts to activate circuit breaker and avoid cascading fallback failures"
      - "SMS costs real money; only route truly urgent notifications to SMS; classify notification urgency"
      - "Push tokens can become invalid; handle token refresh errors and mark invalid tokens for cleanup"
      - "Not configuring dead-letter queue means permanently failed messages are silently lost"
    concepts:
      - Message queue producer/consumer pattern
      - Dead-letter queue for undeliverable messages
      - Adapter pattern for channel implementations
      - Idempotency keys for duplicate detection
      - Rate limiting per external API
    skills:
      - Message queue configuration and management
      - Channel abstraction design
      - Provider fallback logic
      - Rate limiting for external APIs
    deliverables:
      - Message broker setup with notification queue and dead-letter queue
      - Notification ingestion API returning 202 Accepted
      - Channel interface with email, SMS, and push implementations
      - Queue consumer with routing, fallback, and rate limiting
      - Idempotency key deduplication
    estimated_hours: "12-16"

  - id: notification-service-m2
    name: Template System & Retry Logic
    description: >-
      Implement notification templates with localization, and build retry
      with exponential backoff for transient delivery failures.
    acceptance_criteria:
      - "Templates are stored with named placeholders (e.g., {{user_name}}) and versioned; each template has an ID and version number"
      - "Variable substitution replaces all placeholders with data values at send time; unresolved placeholders cause a rendering error (not silent empty strings)"
      - "Localization selects the correct language variant based on user locale; fallback chain resolves gracefully (e.g., fr-CA → fr → en)"
      - "HTML email templates use inline styles (not CSS classes) since most email clients strip <style> tags"
      - "SMS templates are validated to fit within 160 characters (single segment) or explicitly document multi-segment cost"
      - "Failed deliveries due to transient errors (timeout, 503, rate limit) are retried with exponential backoff (base 2s, max 5 retries)"
      - "After max retries are exhausted, the notification is moved to the dead-letter queue with failure reason metadata"
      - "Permanent failures (invalid recipient, hard bounce) are NOT retried; they are immediately moved to DLQ"
    pitfalls:
      - "SMS templates exceeding 160 characters are split into multiple segments, each billed separately"
      - "HTML email rendered with CSS classes will look broken in Gmail, Outlook, etc.; always inline styles"
      - Template variable injection: user-controlled data in HTML email templates can inject malicious HTML/links; escape user data contextually
      - "Locale detection should never fail silently; always fall back to a default locale"
      - "Retrying permanent failures wastes resources and annoys third-party providers; classify errors correctly"
      - "Exponential backoff without jitter causes retry storms when many notifications fail simultaneously"
    concepts:
      - Template rendering with Mustache/Handlebars syntax
      - Locale fallback chain resolution
      - Exponential backoff with jitter
      - Transient vs permanent failure classification
      - Dead-letter queue processing
    skills:
      - Template engine implementation
      - Internationalization (i18n)
      - Retry logic with backoff
      - Error classification
    deliverables:
      - Template storage with versioning and CRUD management
      - Variable substitution engine with unresolved placeholder detection
      - Locale-aware template selection with fallback chain
      - Retry middleware with exponential backoff and jitter
      - Dead-letter queue routing for exhausted retries and permanent failures
    estimated_hours: "10-14"

  - id: notification-service-m3
    name: User Preferences & Compliance
    description: >-
      Implement user notification preferences with granular controls,
      one-click unsubscribe, and regulatory compliance.
    acceptance_criteria:
      - "Preferences are stored per user and per notification category, controlling which channels receive which notification types"
      - "Notifications are classified as transactional or marketing; transactional notifications (receipts, password resets, security alerts) cannot be unsubscribed per CAN-SPAM/GDPR"
      - "Marketing notifications include a one-click unsubscribe mechanism via List-Unsubscribe header (RFC 8058) and a tokenized unsubscribe URL"
      - "Unsubscribe tokens are HMAC-signed to prevent enumeration attacks; invalid tokens return 400 Bad Request"
      - "Unsubscribe requests take effect immediately; the preference store is updated synchronously before returning success"
      - "Quiet hours settings suppress non-urgent notifications during the user's configured do-not-disturb window, using the USER's timezone (not server timezone)"
      - "Preference changes are audit-logged with timestamp, old value, new value, and source (user action, unsubscribe link, admin)"
    pitfalls:
      - "Allowing unsubscribe from transactional emails violates regulations and breaks critical user flows"
      - "Unsigned unsubscribe tokens allow attackers to enumerate and unsubscribe arbitrary users"
      - "Quiet hours must use the user's timezone; using server timezone sends 3 AM notifications to users in different zones"
      - "GDPR requires honoring unsubscribe requests without delay; implement synchronously, not in a background job"
      - "Not audit-logging preference changes makes compliance investigations impossible"
    concepts:
      - Transactional vs marketing notification classification
      - HMAC-signed unsubscribe tokens
      - List-Unsubscribe header (RFC 8058)
      - Timezone-aware quiet hours
      - Consent audit logging
    skills:
      - Preference management
      - GDPR/CAN-SPAM compliance
      - HMAC token generation and validation
      - Timezone handling
    deliverables:
      - Per-user, per-category preference storage with CRUD API
      - Transactional vs marketing classification preventing transactional unsubscribe
      - HMAC-signed one-click unsubscribe URLs with List-Unsubscribe header
      - Quiet hours enforcement with user timezone support
      - Audit log for all preference changes
    estimated_hours: "10-14"

  - id: notification-service-m4
    name: Delivery Tracking & Analytics
    description: >-
      Track delivery status through the notification lifecycle, process
      webhook events from providers, and build analytics for delivery metrics.
    acceptance_criteria:
      - Each notification's delivery status is tracked through states: queued → sent → delivered → opened → clicked → bounced → failed
      - "Webhook endpoint receives delivery events (delivered, bounced, complained) from email/SMS providers; events are processed idempotently using event ID"
      - "Webhook payload signatures are verified using provider-specific HMAC validation before processing"
      - "Hard bounces update the recipient's reachability status; subsequent notifications to unreachable recipients are suppressed"
      - "Spam complaints trigger immediate unsubscribe from marketing notifications for the complaining user"
      - "Delivery metrics (delivery rate, bounce rate, failure rate) are calculated per channel and per notification category"
      - "Metrics are exposed via an API endpoint (GET /analytics/delivery) with time range filtering"
      - "Alert fires when the failure rate for any channel exceeds a configurable threshold (e.g., >10% in 5 minutes)"
    pitfalls:
      - Email open tracking via 1x1 pixel is unreliable: image blocking, Apple Mail Privacy Protection prefetching pixels inflate counts
      - "Do not track opens/clicks on transactional emails (password resets, security alerts) for user privacy"
      - "Webhook endpoints must be idempotent; providers may send the same event multiple times"
      - "Not verifying webhook signatures allows attackers to inject fake delivery events"
      - "Spam complaints (feedback loops) must trigger immediate unsubscribe; ignoring them damages sender reputation"
      - "Open rate metrics should be presented with a disclaimer about unreliability due to pixel blocking"
    concepts:
      - Webhook payload signature verification (HMAC)
      - Idempotent event processing
      - Bounce classification (hard vs soft)
      - Feedback loop processing for spam complaints
      - Time-series metrics aggregation
    skills:
      - Webhook processing and verification
      - Event-driven status tracking
      - Metrics aggregation
      - Alerting on threshold breaches
    deliverables:
      - Delivery status tracker updating state through the notification lifecycle
      - Webhook endpoint processing provider delivery events with signature verification
      - Bounce handler classifying hard/soft bounces and updating recipient reachability
      - Spam complaint handler triggering automatic unsubscribe
      - Analytics API exposing delivery metrics with time range filtering
      - Alert mechanism for channel failure rate threshold breaches
    estimated_hours: "12-16"
```
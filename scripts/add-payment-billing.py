#!/usr/bin/env python3
"""
Add Payment & Billing projects - essential for SaaS and e-commerce.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

payment_billing_projects = {
    "payment-gateway": {
        "name": "Payment Gateway Integration",
        "description": "Build a payment processing system handling credit cards, idempotency, PCI compliance considerations, refunds, and webhook reconciliation.",
        "why_expert": "Payment bugs cost real money and trust. Understanding payment flow internals helps prevent double-charges, handle edge cases, and integrate any payment provider.",
        "difficulty": "expert",
        "tags": ["payments", "fintech", "security", "api-integration", "idempotency"],
        "estimated_hours": 45,
        "prerequisites": ["build-http-server"],
        "milestones": [
            {
                "name": "Payment Intent & Idempotency",
                "description": "Implement payment intents with idempotency keys to prevent duplicate charges",
                "skills": ["Idempotency patterns", "State machines", "Atomic operations"],
                "hints": {
                    "level1": "Create a payment intent before charging - it represents customer's intention to pay",
                    "level2": "Idempotency key: same key = same response. Store request hash with response.",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import secrets
import hashlib
import time
import json

class PaymentStatus(Enum):
    CREATED = "created"
    REQUIRES_PAYMENT_METHOD = "requires_payment_method"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"

@dataclass
class PaymentIntent:
    id: str
    amount: int              # In smallest currency unit (cents)
    currency: str
    status: PaymentStatus
    customer_id: Optional[str]
    payment_method_id: Optional[str]
    description: str
    metadata: dict
    created_at: float
    updated_at: float
    idempotency_key: Optional[str]
    client_secret: str       # For frontend confirmation
    error_message: Optional[str] = None
    charge_id: Optional[str] = None

@dataclass
class IdempotencyRecord:
    key: str
    request_hash: str
    response: dict
    status_code: int
    created_at: float
    expires_at: float

class IdempotencyManager:
    def __init__(self, ttl: int = 86400):  # 24 hour TTL
        self.records: dict[str, IdempotencyRecord] = {}
        self.ttl = ttl

    def _hash_request(self, method: str, path: str, body: dict) -> str:
        data = f"{method}|{path}|{json.dumps(body, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()

    def check(self, idempotency_key: str, method: str,
              path: str, body: dict) -> Optional[tuple[dict, int]]:
        '''Check if we've seen this request before'''
        record = self.records.get(idempotency_key)
        if not record:
            return None

        # Check expiry
        if time.time() > record.expires_at:
            del self.records[idempotency_key]
            return None

        # Verify request matches (prevent key reuse with different request)
        request_hash = self._hash_request(method, path, body)
        if request_hash != record.request_hash:
            raise ValueError(
                "Idempotency key already used with different request parameters"
            )

        return record.response, record.status_code

    def store(self, idempotency_key: str, method: str, path: str,
              body: dict, response: dict, status_code: int):
        '''Store response for idempotency key'''
        self.records[idempotency_key] = IdempotencyRecord(
            key=idempotency_key,
            request_hash=self._hash_request(method, path, body),
            response=response,
            status_code=status_code,
            created_at=time.time(),
            expires_at=time.time() + self.ttl
        )

class PaymentService:
    def __init__(self):
        self.intents: dict[str, PaymentIntent] = {}
        self.idempotency = IdempotencyManager()

    def create_payment_intent(self, amount: int, currency: str,
                              customer_id: str = None,
                              description: str = "",
                              metadata: dict = None,
                              idempotency_key: str = None) -> PaymentIntent:
        # Check idempotency
        if idempotency_key:
            existing = self.idempotency.check(
                idempotency_key, "POST", "/payment_intents",
                {"amount": amount, "currency": currency}
            )
            if existing:
                return self.intents[existing[0]["id"]]

        # Create new payment intent
        intent_id = f"pi_{secrets.token_urlsafe(16)}"
        client_secret = f"{intent_id}_secret_{secrets.token_urlsafe(16)}"

        now = time.time()
        intent = PaymentIntent(
            id=intent_id,
            amount=amount,
            currency=currency.lower(),
            status=PaymentStatus.REQUIRES_PAYMENT_METHOD,
            customer_id=customer_id,
            payment_method_id=None,
            description=description,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            idempotency_key=idempotency_key,
            client_secret=client_secret
        )

        self.intents[intent_id] = intent

        # Store for idempotency
        if idempotency_key:
            self.idempotency.store(
                idempotency_key, "POST", "/payment_intents",
                {"amount": amount, "currency": currency},
                {"id": intent_id, "client_secret": client_secret},
                200
            )

        return intent

    def attach_payment_method(self, intent_id: str,
                              payment_method_id: str) -> PaymentIntent:
        intent = self.intents.get(intent_id)
        if not intent:
            raise ValueError("Payment intent not found")

        if intent.status not in [PaymentStatus.REQUIRES_PAYMENT_METHOD,
                                 PaymentStatus.REQUIRES_CONFIRMATION]:
            raise ValueError(f"Cannot modify intent in status: {intent.status}")

        intent.payment_method_id = payment_method_id
        intent.status = PaymentStatus.REQUIRES_CONFIRMATION
        intent.updated_at = time.time()

        return intent
```
"""
                },
                "pitfalls": [
                    "Idempotency key reused with different params should error, not return old response",
                    "Client secret must be kept secret - only share with authenticated user",
                    "Amount in cents prevents floating point errors ($10.00 = 1000 cents)",
                    "Status transitions must be validated - can't go backwards"
                ]
            },
            {
                "name": "Payment Processing & 3DS",
                "description": "Implement payment confirmation with 3D Secure authentication flow",
                "skills": ["3DS flow", "Payment processing", "Error handling"],
                "hints": {
                    "level1": "3DS (3D Secure) is required for Strong Customer Authentication (SCA) in EU",
                    "level2": "Payment can be synchronous (immediate) or async (3DS redirect needed)",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class ThreeDSStatus(Enum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    CHALLENGE = "challenge"      # User must complete challenge
    SUCCEEDED = "succeeded"
    FAILED = "failed"

@dataclass
class ThreeDSResult:
    status: ThreeDSStatus
    redirect_url: Optional[str] = None  # For challenge flow
    version: str = "2.0"

@dataclass
class PaymentMethodDetails:
    id: str
    card_brand: str           # visa, mastercard, amex
    last4: str
    exp_month: int
    exp_year: int
    fingerprint: str          # Unique card identifier
    three_ds_supported: bool

class PaymentProcessor:
    def __init__(self, payment_service: PaymentService):
        self.service = payment_service
        self.charges: dict[str, dict] = {}

    def confirm_payment(self, intent_id: str,
                        return_url: str = None) -> dict:
        '''
        Confirm payment intent. May require 3DS.
        Returns: {status, next_action?, charge_id?}
        '''
        intent = self.service.intents.get(intent_id)
        if not intent:
            raise ValueError("Payment intent not found")

        if intent.status != PaymentStatus.REQUIRES_CONFIRMATION:
            raise ValueError(f"Cannot confirm intent in status: {intent.status}")

        # Check if 3DS is required (based on card, amount, region)
        three_ds = self._check_3ds_requirement(intent)

        if three_ds.status == ThreeDSStatus.CHALLENGE:
            # Need user interaction
            intent.status = PaymentStatus.PROCESSING
            return {
                "status": "requires_action",
                "next_action": {
                    "type": "redirect_to_url",
                    "redirect_to_url": {
                        "url": three_ds.redirect_url,
                        "return_url": return_url
                    }
                }
            }

        if three_ds.status == ThreeDSStatus.FAILED:
            intent.status = PaymentStatus.FAILED
            intent.error_message = "3D Secure authentication failed"
            return {"status": "failed", "error": intent.error_message}

        # Process payment
        return self._process_charge(intent)

    def handle_3ds_callback(self, intent_id: str,
                            three_ds_result: str) -> dict:
        '''Handle return from 3DS challenge'''
        intent = self.service.intents.get(intent_id)
        if not intent:
            raise ValueError("Payment intent not found")

        # Verify 3DS result (in production: verify with card network)
        if three_ds_result == "authenticated":
            return self._process_charge(intent)
        else:
            intent.status = PaymentStatus.FAILED
            intent.error_message = "3D Secure authentication failed"
            return {"status": "failed", "error": intent.error_message}

    def _check_3ds_requirement(self, intent: PaymentIntent) -> ThreeDSResult:
        '''Determine if 3DS is required'''
        # In production: check card issuer requirements, SCA rules, etc.

        # SCA required for EU cards over certain threshold
        if intent.amount > 3000:  # €30.00
            return ThreeDSResult(
                status=ThreeDSStatus.CHALLENGE,
                redirect_url=f"https://3ds.example.com/challenge/{intent.id}"
            )

        return ThreeDSResult(status=ThreeDSStatus.NOT_REQUIRED)

    def _process_charge(self, intent: PaymentIntent) -> dict:
        '''Actually charge the card'''
        intent.status = PaymentStatus.PROCESSING

        try:
            # In production: call payment processor API (Stripe, Adyen, etc.)
            charge_id = f"ch_{secrets.token_urlsafe(16)}"

            # Simulate processing
            success = True  # In reality: check processor response

            if success:
                intent.status = PaymentStatus.SUCCEEDED
                intent.charge_id = charge_id
                intent.updated_at = time.time()

                self.charges[charge_id] = {
                    "id": charge_id,
                    "amount": intent.amount,
                    "currency": intent.currency,
                    "payment_intent": intent.id,
                    "status": "succeeded",
                    "created_at": time.time()
                }

                return {
                    "status": "succeeded",
                    "charge_id": charge_id
                }
            else:
                intent.status = PaymentStatus.FAILED
                intent.error_message = "Card declined"
                return {"status": "failed", "error": "card_declined"}

        except Exception as e:
            intent.status = PaymentStatus.FAILED
            intent.error_message = str(e)
            return {"status": "failed", "error": str(e)}

    def capture_payment(self, intent_id: str, amount: int = None) -> dict:
        '''
        Capture a previously authorized payment.
        Used for auth-then-capture flows (hotels, rentals).
        '''
        intent = self.service.intents.get(intent_id)
        if not intent or intent.status != PaymentStatus.SUCCEEDED:
            raise ValueError("Invalid intent for capture")

        capture_amount = amount or intent.amount
        if capture_amount > intent.amount:
            raise ValueError("Capture amount exceeds authorized amount")

        # In production: call processor capture API
        return {
            "status": "captured",
            "amount": capture_amount,
            "charge_id": intent.charge_id
        }
```
"""
                },
                "pitfalls": [
                    "3DS redirects must include return_url for user to come back",
                    "Payment can succeed but webhook fail - always reconcile",
                    "Auth-capture: authorization expires (usually 7 days) - capture before expiry",
                    "Currency mismatch between intent and capture causes errors"
                ]
            },
            {
                "name": "Refunds & Disputes",
                "description": "Implement refund processing, partial refunds, and dispute handling",
                "skills": ["Refund workflows", "Dispute handling", "Financial reconciliation"],
                "hints": {
                    "level1": "Refunds can be full or partial; total refunds can't exceed original charge",
                    "level2": "Disputes (chargebacks) require evidence submission within deadline",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import time

class RefundStatus(Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

class DisputeStatus(Enum):
    WARNING_NEEDS_RESPONSE = "warning_needs_response"
    WARNING_UNDER_REVIEW = "warning_under_review"
    WARNING_CLOSED = "warning_closed"
    NEEDS_RESPONSE = "needs_response"
    UNDER_REVIEW = "under_review"
    CHARGE_REFUNDED = "charge_refunded"
    WON = "won"
    LOST = "lost"

class DisputeReason(Enum):
    DUPLICATE = "duplicate"
    FRAUDULENT = "fraudulent"
    SUBSCRIPTION_CANCELED = "subscription_canceled"
    PRODUCT_NOT_RECEIVED = "product_not_received"
    PRODUCT_UNACCEPTABLE = "product_unacceptable"
    UNRECOGNIZED = "unrecognized"
    CREDIT_NOT_PROCESSED = "credit_not_processed"
    GENERAL = "general"

@dataclass
class Refund:
    id: str
    charge_id: str
    payment_intent_id: str
    amount: int
    currency: str
    status: RefundStatus
    reason: Optional[str]
    created_at: float
    metadata: dict

@dataclass
class Dispute:
    id: str
    charge_id: str
    amount: int
    currency: str
    status: DisputeStatus
    reason: DisputeReason
    evidence_due_by: float
    evidence: Optional[dict]
    created_at: float

class RefundService:
    def __init__(self, payment_service: PaymentService,
                 processor: PaymentProcessor):
        self.payments = payment_service
        self.processor = processor
        self.refunds: dict[str, Refund] = {}
        self.charge_refunds: dict[str, list[str]] = {}  # charge_id -> refund_ids

    def create_refund(self, charge_id: str = None,
                      payment_intent_id: str = None,
                      amount: int = None,
                      reason: str = None) -> Refund:
        # Find the charge
        if payment_intent_id:
            intent = self.payments.intents.get(payment_intent_id)
            if not intent or not intent.charge_id:
                raise ValueError("Payment intent has no charge")
            charge_id = intent.charge_id

        charge = self.processor.charges.get(charge_id)
        if not charge:
            raise ValueError("Charge not found")

        # Calculate refundable amount
        existing_refunds = self.charge_refunds.get(charge_id, [])
        total_refunded = sum(
            self.refunds[rid].amount for rid in existing_refunds
            if self.refunds[rid].status == RefundStatus.SUCCEEDED
        )

        refundable = charge["amount"] - total_refunded
        refund_amount = amount or refundable

        if refund_amount > refundable:
            raise ValueError(
                f"Refund amount ({refund_amount}) exceeds "
                f"refundable amount ({refundable})"
            )

        if refund_amount <= 0:
            raise ValueError("Charge already fully refunded")

        # Create refund
        refund_id = f"re_{secrets.token_urlsafe(16)}"
        refund = Refund(
            id=refund_id,
            charge_id=charge_id,
            payment_intent_id=charge.get("payment_intent"),
            amount=refund_amount,
            currency=charge["currency"],
            status=RefundStatus.PENDING,
            reason=reason,
            created_at=time.time(),
            metadata={}
        )

        self.refunds[refund_id] = refund

        # Track refunds per charge
        if charge_id not in self.charge_refunds:
            self.charge_refunds[charge_id] = []
        self.charge_refunds[charge_id].append(refund_id)

        # Process refund (in production: async with webhook confirmation)
        self._process_refund(refund)

        # Update payment intent status
        intent = self.payments.intents.get(charge.get("payment_intent"))
        if intent:
            if total_refunded + refund_amount >= charge["amount"]:
                intent.status = PaymentStatus.REFUNDED
            else:
                intent.status = PaymentStatus.PARTIALLY_REFUNDED

        return refund

    def _process_refund(self, refund: Refund):
        # In production: call processor refund API
        try:
            # Simulate processing
            refund.status = RefundStatus.SUCCEEDED
        except Exception as e:
            refund.status = RefundStatus.FAILED

class DisputeService:
    def __init__(self, processor: PaymentProcessor):
        self.processor = processor
        self.disputes: dict[str, Dispute] = {}

    def handle_dispute_webhook(self, event_type: str, data: dict):
        '''Handle dispute webhook from payment processor'''
        if event_type == "charge.dispute.created":
            dispute = Dispute(
                id=data["id"],
                charge_id=data["charge"],
                amount=data["amount"],
                currency=data["currency"],
                status=DisputeStatus.NEEDS_RESPONSE,
                reason=DisputeReason(data["reason"]),
                evidence_due_by=data["evidence_due_by"],
                evidence=None,
                created_at=time.time()
            )
            self.disputes[dispute.id] = dispute

            # Alert team immediately
            self._alert_dispute(dispute)

        elif event_type == "charge.dispute.closed":
            dispute = self.disputes.get(data["id"])
            if dispute:
                dispute.status = DisputeStatus(data["status"])

    def submit_evidence(self, dispute_id: str, evidence: dict) -> Dispute:
        '''
        Submit evidence for a dispute.
        Evidence includes: receipts, shipping proof, correspondence, etc.
        '''
        dispute = self.disputes.get(dispute_id)
        if not dispute:
            raise ValueError("Dispute not found")

        if time.time() > dispute.evidence_due_by:
            raise ValueError("Evidence submission deadline passed")

        if dispute.status not in [DisputeStatus.NEEDS_RESPONSE,
                                   DisputeStatus.WARNING_NEEDS_RESPONSE]:
            raise ValueError("Dispute not accepting evidence")

        # Validate required evidence based on reason
        required_fields = self._get_required_evidence(dispute.reason)
        missing = [f for f in required_fields if f not in evidence]
        if missing:
            raise ValueError(f"Missing required evidence: {missing}")

        dispute.evidence = evidence
        dispute.status = DisputeStatus.UNDER_REVIEW

        # In production: submit to processor API
        return dispute

    def _get_required_evidence(self, reason: DisputeReason) -> list[str]:
        evidence_requirements = {
            DisputeReason.PRODUCT_NOT_RECEIVED: [
                "shipping_carrier", "shipping_tracking_number",
                "shipping_date"
            ],
            DisputeReason.FRAUDULENT: [
                "customer_email_address", "customer_ip_address",
                "billing_address"
            ],
            DisputeReason.DUPLICATE: [
                "duplicate_charge_explanation",
                "original_transaction"
            ],
        }
        return evidence_requirements.get(reason, [])

    def _alert_dispute(self, dispute: Dispute):
        print(f"ALERT: New dispute {dispute.id} for ${dispute.amount/100:.2f}")
        print(f"Reason: {dispute.reason.value}")
        print(f"Evidence due: {dispute.evidence_due_by}")
```
"""
                },
                "pitfalls": [
                    "Refund can take 5-10 business days to appear on statement",
                    "Dispute evidence deadline is strict - automate alerts",
                    "Dispute fee charged even if you win - prevention is key",
                    "Partial refunds: track total refunded vs original amount carefully"
                ]
            },
            {
                "name": "Webhook Reconciliation",
                "description": "Implement reliable webhook processing with signature verification and reconciliation",
                "skills": ["Webhook security", "Event processing", "Reconciliation"],
                "hints": {
                    "level1": "Always verify webhook signature before processing",
                    "level2": "Webhooks can arrive out of order or duplicated - handle idempotently",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Callable
import hmac
import hashlib
import time
import json

@dataclass
class WebhookEvent:
    id: str
    type: str
    data: dict
    created_at: float
    processed_at: float = None
    attempts: int = 0

class PaymentWebhookHandler:
    def __init__(self, webhook_secret: str,
                 payment_service: PaymentService,
                 refund_service: RefundService,
                 dispute_service: DisputeService):
        self.secret = webhook_secret
        self.payments = payment_service
        self.refunds = refund_service
        self.disputes = dispute_service
        self.processed_events: set[str] = set()
        self.handlers: dict[str, Callable] = {
            "payment_intent.succeeded": self._handle_payment_succeeded,
            "payment_intent.payment_failed": self._handle_payment_failed,
            "charge.refunded": self._handle_refund,
            "charge.dispute.created": self._handle_dispute_created,
            "charge.dispute.closed": self._handle_dispute_closed,
        }

    def verify_signature(self, payload: bytes, signature: str,
                         timestamp: str) -> bool:
        '''Verify Stripe-style webhook signature'''
        # Signature format: t=timestamp,v1=signature
        expected = hmac.new(
            self.secret.encode(),
            f"{timestamp}.{payload.decode()}".encode(),
            hashlib.sha256
        ).hexdigest()

        # Constant-time comparison
        return hmac.compare_digest(f"v1={expected}", signature.split(",")[1])

    def handle_webhook(self, payload: bytes, signature: str) -> dict:
        '''Process incoming webhook'''
        # Parse signature header
        parts = dict(p.split("=") for p in signature.split(","))
        timestamp = parts.get("t", "")

        # Verify signature
        if not self.verify_signature(payload, signature, timestamp):
            raise ValueError("Invalid webhook signature")

        # Check timestamp (prevent replay attacks)
        if abs(time.time() - int(timestamp)) > 300:  # 5 minute tolerance
            raise ValueError("Webhook timestamp too old")

        # Parse event
        event_data = json.loads(payload)
        event_id = event_data["id"]
        event_type = event_data["type"]

        # Idempotency check
        if event_id in self.processed_events:
            return {"status": "already_processed"}

        event = WebhookEvent(
            id=event_id,
            type=event_type,
            data=event_data["data"]["object"],
            created_at=event_data["created"]
        )

        # Route to handler
        handler = self.handlers.get(event_type)
        if handler:
            try:
                handler(event)
                event.processed_at = time.time()
                self.processed_events.add(event_id)
                return {"status": "processed"}
            except Exception as e:
                event.attempts += 1
                # Log error, will be retried by processor
                return {"status": "error", "error": str(e)}
        else:
            # Unknown event type - acknowledge to prevent retries
            return {"status": "ignored", "reason": "unknown_event_type"}

    def _handle_payment_succeeded(self, event: WebhookEvent):
        '''Reconcile successful payment'''
        intent_id = event.data["id"]
        intent = self.payments.intents.get(intent_id)

        if intent:
            # Update our records to match processor state
            intent.status = PaymentStatus.SUCCEEDED
            intent.charge_id = event.data.get("latest_charge")
            intent.updated_at = time.time()

            # Trigger fulfillment (send email, provision service, etc.)
            self._trigger_fulfillment(intent)
        else:
            # Payment succeeded but we don't have the intent
            # This can happen if webhook arrives before API response
            # Log for investigation
            print(f"WARNING: Unknown payment intent succeeded: {intent_id}")

    def _handle_payment_failed(self, event: WebhookEvent):
        '''Handle failed payment'''
        intent_id = event.data["id"]
        intent = self.payments.intents.get(intent_id)

        if intent:
            intent.status = PaymentStatus.FAILED
            intent.error_message = event.data.get("last_payment_error", {}).get("message")
            intent.updated_at = time.time()

    def _handle_refund(self, event: WebhookEvent):
        '''Reconcile refund'''
        charge_id = event.data["id"]
        # Check if refund already recorded
        existing = self.refunds.charge_refunds.get(charge_id, [])

        for refund_data in event.data.get("refunds", {}).get("data", []):
            if refund_data["id"] not in [self.refunds.refunds[r].id for r in existing]:
                # Refund happened outside our system (admin dashboard, etc.)
                # Record it
                print(f"Recording external refund: {refund_data['id']}")

    def _handle_dispute_created(self, event: WebhookEvent):
        self.disputes.handle_dispute_webhook("charge.dispute.created", event.data)

    def _handle_dispute_closed(self, event: WebhookEvent):
        self.disputes.handle_dispute_webhook("charge.dispute.closed", event.data)

    def _trigger_fulfillment(self, intent: PaymentIntent):
        '''Trigger order fulfillment after successful payment'''
        # In production: queue job to send confirmation email,
        # update order status, provision service, etc.
        print(f"Fulfilling order for payment: {intent.id}")

    def reconcile(self):
        '''
        Periodic reconciliation to catch missed webhooks.
        Compare our records with processor's records.
        '''
        # In production: fetch recent payments from processor API
        # Compare status with local records
        # Update any mismatches
        pass
```
"""
                },
                "pitfalls": [
                    "Webhook signature verification is critical - never skip",
                    "Events can arrive out of order - check timestamps and states",
                    "Processor may retry webhooks - always be idempotent",
                    "Run periodic reconciliation - webhooks can fail silently"
                ]
            }
        ]
    },

    "subscription-billing": {
        "name": "Subscription & Billing System",
        "description": "Build a complete subscription management system with plans, billing cycles, proration, trials, and usage-based billing.",
        "why_expert": "Subscription logic is complex with edge cases (upgrades, downgrades, trials, cancellations). Building one teaches financial calculations and state management.",
        "difficulty": "expert",
        "tags": ["billing", "subscriptions", "saas", "fintech", "recurring-payments"],
        "estimated_hours": 50,
        "prerequisites": ["payment-gateway"],
        "milestones": [
            {
                "name": "Plans & Pricing",
                "description": "Implement flexible pricing plans with tiers, features, and currencies",
                "skills": ["Pricing models", "Feature flags", "Multi-currency"],
                "hints": {
                    "level1": "Plans define pricing; subscriptions track who's on what plan",
                    "level2": "Support multiple billing intervals (monthly, yearly) with different prices",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from decimal import Decimal
import time

class BillingInterval(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

class PricingModel(Enum):
    FLAT_RATE = "flat_rate"           # Fixed price
    PER_SEAT = "per_seat"             # Price × number of users
    TIERED = "tiered"                 # Different rates at volume tiers
    VOLUME = "volume"                 # Single rate based on total volume
    USAGE_BASED = "usage_based"       # Pay for what you use

@dataclass
class PriceTier:
    up_to: Optional[int]  # None = unlimited
    unit_amount: int      # Price per unit in cents
    flat_amount: int = 0  # Optional flat fee for this tier

@dataclass
class Price:
    id: str
    plan_id: str
    currency: str
    unit_amount: int                          # Base price in cents
    billing_interval: BillingInterval
    interval_count: int = 1                   # Every N intervals
    pricing_model: PricingModel = PricingModel.FLAT_RATE
    tiers: list[PriceTier] = field(default_factory=list)
    trial_period_days: int = 0
    metadata: dict = field(default_factory=dict)

@dataclass
class Feature:
    id: str
    name: str
    description: str
    value_type: str = "boolean"  # boolean, number, unlimited
    default_value: any = False

@dataclass
class Plan:
    id: str
    name: str
    description: str
    prices: dict[str, Price]   # currency -> Price
    features: dict[str, any]   # feature_id -> value
    active: bool = True
    metadata: dict = field(default_factory=dict)

class PricingEngine:
    def __init__(self):
        self.plans: dict[str, Plan] = {}
        self.features: dict[str, Feature] = {}

    def create_plan(self, plan_id: str, name: str, description: str,
                    base_price: int, currency: str = "usd",
                    interval: BillingInterval = BillingInterval.MONTHLY,
                    features: dict = None,
                    trial_days: int = 0) -> Plan:
        price = Price(
            id=f"price_{plan_id}_{currency}",
            plan_id=plan_id,
            currency=currency,
            unit_amount=base_price,
            billing_interval=interval,
            trial_period_days=trial_days
        )

        plan = Plan(
            id=plan_id,
            name=name,
            description=description,
            prices={currency: price},
            features=features or {}
        )

        self.plans[plan_id] = plan
        return plan

    def add_tiered_pricing(self, plan_id: str, currency: str,
                           tiers: list[dict], model: PricingModel):
        '''Add tiered or volume pricing to a plan'''
        plan = self.plans.get(plan_id)
        if not plan:
            raise ValueError("Plan not found")

        price = plan.prices.get(currency)
        if not price:
            raise ValueError("Price not found for currency")

        price.pricing_model = model
        price.tiers = [
            PriceTier(
                up_to=t.get("up_to"),
                unit_amount=t["unit_amount"],
                flat_amount=t.get("flat_amount", 0)
            )
            for t in tiers
        ]

    def calculate_price(self, plan_id: str, currency: str,
                        quantity: int = 1) -> dict:
        '''Calculate price for given quantity'''
        plan = self.plans.get(plan_id)
        if not plan:
            raise ValueError("Plan not found")

        price = plan.prices.get(currency)
        if not price:
            raise ValueError("Price not found for currency")

        if price.pricing_model == PricingModel.FLAT_RATE:
            return {
                "subtotal": price.unit_amount,
                "quantity": 1,
                "unit_amount": price.unit_amount
            }

        elif price.pricing_model == PricingModel.PER_SEAT:
            return {
                "subtotal": price.unit_amount * quantity,
                "quantity": quantity,
                "unit_amount": price.unit_amount
            }

        elif price.pricing_model == PricingModel.TIERED:
            # Each tier applies to units within that tier
            total = 0
            remaining = quantity
            breakdown = []

            for tier in sorted(price.tiers, key=lambda t: t.up_to or float('inf')):
                if remaining <= 0:
                    break

                tier_max = tier.up_to or float('inf')
                prev_max = breakdown[-1]["up_to"] if breakdown else 0
                tier_quantity = min(remaining, tier_max - prev_max)

                tier_amount = tier.flat_amount + (tier.unit_amount * tier_quantity)
                total += tier_amount
                remaining -= tier_quantity

                breakdown.append({
                    "up_to": tier.up_to,
                    "quantity": tier_quantity,
                    "unit_amount": tier.unit_amount,
                    "amount": tier_amount
                })

            return {
                "subtotal": total,
                "quantity": quantity,
                "breakdown": breakdown
            }

        elif price.pricing_model == PricingModel.VOLUME:
            # Single rate based on total quantity (find applicable tier)
            applicable_tier = None
            for tier in sorted(price.tiers, key=lambda t: t.up_to or float('inf')):
                if tier.up_to is None or quantity <= tier.up_to:
                    applicable_tier = tier
                    break

            if not applicable_tier:
                applicable_tier = price.tiers[-1]

            total = applicable_tier.flat_amount + (applicable_tier.unit_amount * quantity)
            return {
                "subtotal": total,
                "quantity": quantity,
                "unit_amount": applicable_tier.unit_amount,
                "tier_up_to": applicable_tier.up_to
            }

        raise ValueError(f"Unknown pricing model: {price.pricing_model}")

    def compare_plans(self, plan_id_1: str, plan_id_2: str) -> dict:
        '''Compare features between two plans'''
        plan1 = self.plans.get(plan_id_1)
        plan2 = self.plans.get(plan_id_2)

        if not plan1 or not plan2:
            raise ValueError("Plan not found")

        comparison = {}
        all_features = set(plan1.features.keys()) | set(plan2.features.keys())

        for feature_id in all_features:
            comparison[feature_id] = {
                plan_id_1: plan1.features.get(feature_id),
                plan_id_2: plan2.features.get(feature_id)
            }

        return comparison
```
"""
                },
                "pitfalls": [
                    "Tiered vs Volume pricing: tiered charges each tier separately; volume uses single rate",
                    "Currency precision: always use smallest unit (cents) to avoid float errors",
                    "Plan changes: archive old plans rather than deleting (existing subscribers)",
                    "Feature access: check plan features, not subscription status alone"
                ]
            },
            {
                "name": "Subscription Lifecycle",
                "description": "Implement subscription creation, activation, renewal, and cancellation",
                "skills": ["State machines", "Billing cycles", "Grace periods"],
                "hints": {
                    "level1": "Subscription has lifecycle: trialing -> active -> past_due -> canceled",
                    "level2": "Track current_period_start/end for billing; anchor date for consistent billing",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime, timedelta
import calendar

class SubscriptionStatus(Enum):
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"      # Payment failed, retrying
    UNPAID = "unpaid"          # Payment failed, no more retries
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"   # Initial payment failed
    INCOMPLETE_EXPIRED = "incomplete_expired"

class CancellationReason(Enum):
    CUSTOMER_REQUEST = "customer_request"
    PAYMENT_FAILURE = "payment_failure"
    FRAUD = "fraud"

@dataclass
class Subscription:
    id: str
    customer_id: str
    plan_id: str
    price_id: str
    status: SubscriptionStatus
    quantity: int
    current_period_start: float
    current_period_end: float
    trial_start: Optional[float]
    trial_end: Optional[float]
    canceled_at: Optional[float]
    cancel_at_period_end: bool
    ended_at: Optional[float]
    billing_cycle_anchor: float
    created_at: float
    metadata: dict = field(default_factory=dict)

class SubscriptionManager:
    def __init__(self, pricing: PricingEngine,
                 payment_service):  # PaymentService from payment project
        self.pricing = pricing
        self.payments = payment_service
        self.subscriptions: dict[str, Subscription] = {}

    def create_subscription(self, customer_id: str, plan_id: str,
                            currency: str = "usd",
                            quantity: int = 1,
                            trial_from_plan: bool = True,
                            billing_cycle_anchor: int = None) -> Subscription:
        plan = self.pricing.plans.get(plan_id)
        if not plan:
            raise ValueError("Plan not found")

        price = plan.prices.get(currency)
        if not price:
            raise ValueError("Price not found")

        now = time.time()
        sub_id = f"sub_{secrets.token_urlsafe(16)}"

        # Determine trial period
        trial_end = None
        if trial_from_plan and price.trial_period_days > 0:
            trial_end = now + (price.trial_period_days * 86400)

        # Set billing anchor (day of month for monthly, etc.)
        anchor = billing_cycle_anchor or now

        # Calculate initial period
        if trial_end:
            period_start = now
            period_end = trial_end
            status = SubscriptionStatus.TRIALING
        else:
            period_start = now
            period_end = self._calculate_period_end(
                now, price.billing_interval, price.interval_count
            )
            status = SubscriptionStatus.INCOMPLETE

        subscription = Subscription(
            id=sub_id,
            customer_id=customer_id,
            plan_id=plan_id,
            price_id=price.id,
            status=status,
            quantity=quantity,
            current_period_start=period_start,
            current_period_end=period_end,
            trial_start=now if trial_end else None,
            trial_end=trial_end,
            canceled_at=None,
            cancel_at_period_end=False,
            ended_at=None,
            billing_cycle_anchor=anchor,
            created_at=now
        )

        self.subscriptions[sub_id] = subscription

        # If no trial, create initial payment
        if not trial_end:
            self._create_initial_invoice(subscription)

        return subscription

    def _calculate_period_end(self, start: float,
                              interval: BillingInterval,
                              interval_count: int) -> float:
        dt = datetime.fromtimestamp(start)

        if interval == BillingInterval.DAILY:
            end = dt + timedelta(days=interval_count)
        elif interval == BillingInterval.WEEKLY:
            end = dt + timedelta(weeks=interval_count)
        elif interval == BillingInterval.MONTHLY:
            # Add months, preserving day of month (or last day if overflow)
            month = dt.month + interval_count
            year = dt.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            day = min(dt.day, calendar.monthrange(year, month)[1])
            end = dt.replace(year=year, month=month, day=day)
        elif interval == BillingInterval.YEARLY:
            try:
                end = dt.replace(year=dt.year + interval_count)
            except ValueError:
                # Feb 29 -> Feb 28
                end = dt.replace(year=dt.year + interval_count, day=28)

        return end.timestamp()

    def cancel_subscription(self, subscription_id: str,
                            at_period_end: bool = True,
                            reason: CancellationReason = CancellationReason.CUSTOMER_REQUEST) -> Subscription:
        sub = self.subscriptions.get(subscription_id)
        if not sub:
            raise ValueError("Subscription not found")

        if sub.status in [SubscriptionStatus.CANCELED,
                          SubscriptionStatus.INCOMPLETE_EXPIRED]:
            raise ValueError("Subscription already canceled")

        sub.canceled_at = time.time()

        if at_period_end:
            # Cancel at end of billing period (most common)
            sub.cancel_at_period_end = True
        else:
            # Immediate cancellation
            sub.status = SubscriptionStatus.CANCELED
            sub.ended_at = time.time()

        return sub

    def reactivate_subscription(self, subscription_id: str) -> Subscription:
        '''Reactivate a subscription scheduled for cancellation'''
        sub = self.subscriptions.get(subscription_id)
        if not sub:
            raise ValueError("Subscription not found")

        if not sub.cancel_at_period_end:
            raise ValueError("Subscription not scheduled for cancellation")

        if sub.status == SubscriptionStatus.CANCELED:
            raise ValueError("Cannot reactivate fully canceled subscription")

        sub.cancel_at_period_end = False
        sub.canceled_at = None

        return sub

    def renew_subscription(self, subscription_id: str) -> Subscription:
        '''Process subscription renewal (called by billing job)'''
        sub = self.subscriptions.get(subscription_id)
        if not sub:
            raise ValueError("Subscription not found")

        # Check if should be canceled
        if sub.cancel_at_period_end:
            sub.status = SubscriptionStatus.CANCELED
            sub.ended_at = time.time()
            return sub

        # Create invoice for new period
        plan = self.pricing.plans[sub.plan_id]
        price = plan.prices[sub.price_id.split("_")[1]]  # Extract currency

        # Update period
        sub.current_period_start = sub.current_period_end
        sub.current_period_end = self._calculate_period_end(
            sub.current_period_start,
            price.billing_interval,
            price.interval_count
        )

        # Create payment
        self._create_renewal_invoice(sub)

        return sub

    def _create_initial_invoice(self, sub: Subscription):
        plan = self.pricing.plans[sub.plan_id]
        price_info = self.pricing.calculate_price(
            sub.plan_id, "usd", sub.quantity
        )

        # In production: create invoice, attempt payment
        # If payment succeeds: sub.status = ACTIVE
        # If payment fails: sub.status = INCOMPLETE

    def _create_renewal_invoice(self, sub: Subscription):
        # Similar to initial, but with retry logic for failures
        pass
```
"""
                },
                "pitfalls": [
                    "Billing anchor: ensures consistent billing date each month",
                    "Past due vs unpaid: past_due = still retrying; unpaid = gave up",
                    "Cancel at period end: most user-friendly, they paid for the period",
                    "Month overflow: Jan 31 + 1 month = Feb 28, not Mar 3"
                ]
            },
            {
                "name": "Proration & Plan Changes",
                "description": "Implement upgrade/downgrade with prorated charges and credits",
                "skills": ["Proration calculation", "Credits", "Mid-cycle changes"],
                "hints": {
                    "level1": "Proration: charge/credit proportionally for unused time",
                    "level2": "Upgrade immediately charges difference; downgrade creates credit for next invoice",
                    "level3": """
```python
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP

class ProrationBehavior(Enum):
    CREATE_PRORATIONS = "create_prorations"   # Immediate adjustment
    NONE = "none"                              # No proration
    ALWAYS_INVOICE = "always_invoice"          # Invoice immediately

@dataclass
class ProrationItem:
    description: str
    amount: int           # Positive = charge, negative = credit
    quantity: int
    period_start: float
    period_end: float

class ProrationCalculator:
    def calculate_proration(self, old_price: int, new_price: int,
                            remaining_days: int, total_days: int,
                            old_quantity: int = 1,
                            new_quantity: int = 1) -> list[ProrationItem]:
        '''
        Calculate proration items for plan change.

        Returns list of line items:
        - Credit for unused time on old plan
        - Charge for remaining time on new plan
        '''
        items = []

        # Credit for unused time on old plan
        daily_old = Decimal(old_price * old_quantity) / total_days
        credit = int((daily_old * remaining_days).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP
        ))

        if credit > 0:
            items.append(ProrationItem(
                description="Unused time on previous plan",
                amount=-credit,  # Negative = credit
                quantity=old_quantity,
                period_start=0,  # Will be filled in
                period_end=0
            ))

        # Charge for remaining time on new plan
        daily_new = Decimal(new_price * new_quantity) / total_days
        charge = int((daily_new * remaining_days).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP
        ))

        if charge > 0:
            items.append(ProrationItem(
                description="Remaining time on new plan",
                amount=charge,  # Positive = charge
                quantity=new_quantity,
                period_start=0,
                period_end=0
            ))

        return items

class PlanChangeManager:
    def __init__(self, subscription_manager: SubscriptionManager,
                 pricing: PricingEngine):
        self.subscriptions = subscription_manager
        self.pricing = pricing
        self.proration_calc = ProrationCalculator()
        self.customer_credits: dict[str, int] = {}  # customer_id -> credit balance

    def preview_change(self, subscription_id: str, new_plan_id: str,
                       new_quantity: int = None) -> dict:
        '''Preview what charges/credits would result from plan change'''
        sub = self.subscriptions.subscriptions.get(subscription_id)
        if not sub:
            raise ValueError("Subscription not found")

        old_plan = self.pricing.plans[sub.plan_id]
        new_plan = self.pricing.plans.get(new_plan_id)
        if not new_plan:
            raise ValueError("New plan not found")

        currency = "usd"  # Simplified
        old_price_info = self.pricing.calculate_price(
            sub.plan_id, currency, sub.quantity
        )
        new_quantity = new_quantity or sub.quantity
        new_price_info = self.pricing.calculate_price(
            new_plan_id, currency, new_quantity
        )

        # Calculate remaining time in period
        now = time.time()
        total_seconds = sub.current_period_end - sub.current_period_start
        remaining_seconds = sub.current_period_end - now
        total_days = int(total_seconds / 86400)
        remaining_days = int(remaining_seconds / 86400)

        items = self.proration_calc.calculate_proration(
            old_price_info["subtotal"],
            new_price_info["subtotal"],
            remaining_days,
            total_days,
            sub.quantity,
            new_quantity
        )

        total_due = sum(item.amount for item in items)

        return {
            "items": [
                {
                    "description": item.description,
                    "amount": item.amount
                }
                for item in items
            ],
            "total_due": total_due,
            "is_upgrade": new_price_info["subtotal"] > old_price_info["subtotal"],
            "immediate_charge": total_due if total_due > 0 else 0,
            "credit_applied": abs(total_due) if total_due < 0 else 0
        }

    def change_plan(self, subscription_id: str, new_plan_id: str,
                    new_quantity: int = None,
                    proration_behavior: ProrationBehavior = ProrationBehavior.CREATE_PRORATIONS) -> Subscription:
        '''Execute plan change with proration'''
        sub = self.subscriptions.subscriptions.get(subscription_id)
        if not sub:
            raise ValueError("Subscription not found")

        preview = self.preview_change(subscription_id, new_plan_id, new_quantity)

        if proration_behavior == ProrationBehavior.CREATE_PRORATIONS:
            if preview["total_due"] > 0:
                # Upgrade: charge immediately
                # In production: create payment intent and process
                print(f"Charging ${preview['total_due']/100:.2f} for upgrade")
            elif preview["total_due"] < 0:
                # Downgrade: apply credit
                credit = abs(preview["total_due"])
                self.customer_credits[sub.customer_id] = (
                    self.customer_credits.get(sub.customer_id, 0) + credit
                )
                print(f"Applied ${credit/100:.2f} credit for downgrade")

        # Update subscription
        new_plan = self.pricing.plans[new_plan_id]
        new_price = new_plan.prices.get("usd")

        sub.plan_id = new_plan_id
        sub.price_id = new_price.id
        sub.quantity = new_quantity or sub.quantity

        return sub

    def change_quantity(self, subscription_id: str,
                        new_quantity: int) -> Subscription:
        '''Change subscription quantity (seats)'''
        sub = self.subscriptions.subscriptions.get(subscription_id)
        if not sub:
            raise ValueError("Subscription not found")

        return self.change_plan(subscription_id, sub.plan_id, new_quantity)

    def apply_credit_to_invoice(self, customer_id: str,
                                invoice_amount: int) -> tuple[int, int]:
        '''Apply customer credit balance to invoice'''
        credit = self.customer_credits.get(customer_id, 0)
        if credit <= 0:
            return invoice_amount, 0

        credit_applied = min(credit, invoice_amount)
        remaining_due = invoice_amount - credit_applied
        self.customer_credits[customer_id] = credit - credit_applied

        return remaining_due, credit_applied
```
"""
                },
                "pitfalls": [
                    "Proration direction: upgrade charges extra; downgrade credits",
                    "Round proration carefully - small errors accumulate over many customers",
                    "Credits must be applied automatically on next invoice",
                    "Quantity changes are proration too - not just plan changes"
                ]
            },
            {
                "name": "Usage-Based Billing",
                "description": "Implement metered billing with usage tracking, aggregation, and reporting",
                "skills": ["Usage metering", "Aggregation", "Rate limiting"],
                "hints": {
                    "level1": "Track usage events in real-time; aggregate for billing at period end",
                    "level2": "Usage can be: sum, max, last, unique count over billing period",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import defaultdict
import time

class AggregationType(Enum):
    SUM = "sum"              # Total usage (API calls, storage GB)
    MAX = "max"              # Peak usage (concurrent users)
    LAST = "last"            # Final value (current storage)
    UNIQUE_COUNT = "unique"  # Unique values (active users)

@dataclass
class Meter:
    id: str
    name: str
    event_name: str                    # e.g., "api_call", "storage_byte"
    aggregation: AggregationType
    unit_label: str                    # e.g., "API calls", "GB"
    filter_expression: Optional[str]   # Filter which events count

@dataclass
class UsageRecord:
    meter_id: str
    subscription_id: str
    timestamp: float
    quantity: int
    properties: dict                   # Additional metadata

@dataclass
class UsageSummary:
    meter_id: str
    subscription_id: str
    period_start: float
    period_end: float
    total_quantity: int
    billable_quantity: int
    unit_amount: int
    total_amount: int

class UsageTracker:
    def __init__(self):
        self.meters: dict[str, Meter] = {}
        self.usage_records: list[UsageRecord] = []
        # Indexed for fast lookup
        self.usage_by_sub: dict[str, dict[str, list[UsageRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )  # subscription_id -> meter_id -> records

    def create_meter(self, meter_id: str, name: str, event_name: str,
                     aggregation: AggregationType,
                     unit_label: str = "units") -> Meter:
        meter = Meter(
            id=meter_id,
            name=name,
            event_name=event_name,
            aggregation=aggregation,
            unit_label=unit_label,
            filter_expression=None
        )
        self.meters[meter_id] = meter
        return meter

    def record_usage(self, subscription_id: str, meter_id: str,
                     quantity: int = 1,
                     timestamp: float = None,
                     properties: dict = None,
                     idempotency_key: str = None):
        '''Record a usage event'''
        meter = self.meters.get(meter_id)
        if not meter:
            raise ValueError("Meter not found")

        record = UsageRecord(
            meter_id=meter_id,
            subscription_id=subscription_id,
            timestamp=timestamp or time.time(),
            quantity=quantity,
            properties=properties or {}
        )

        # In production: check idempotency key to prevent duplicates
        self.usage_records.append(record)
        self.usage_by_sub[subscription_id][meter_id].append(record)

    def get_usage_summary(self, subscription_id: str, meter_id: str,
                          period_start: float,
                          period_end: float) -> UsageSummary:
        '''Aggregate usage for a billing period'''
        meter = self.meters.get(meter_id)
        if not meter:
            raise ValueError("Meter not found")

        records = [
            r for r in self.usage_by_sub[subscription_id][meter_id]
            if period_start <= r.timestamp < period_end
        ]

        if not records:
            total = 0
        elif meter.aggregation == AggregationType.SUM:
            total = sum(r.quantity for r in records)
        elif meter.aggregation == AggregationType.MAX:
            total = max(r.quantity for r in records)
        elif meter.aggregation == AggregationType.LAST:
            latest = max(records, key=lambda r: r.timestamp)
            total = latest.quantity
        elif meter.aggregation == AggregationType.UNIQUE_COUNT:
            # Count unique values of a property (e.g., user_id)
            unique_values = set()
            for r in records:
                unique_values.add(r.properties.get("unique_key", r.quantity))
            total = len(unique_values)

        return UsageSummary(
            meter_id=meter_id,
            subscription_id=subscription_id,
            period_start=period_start,
            period_end=period_end,
            total_quantity=total,
            billable_quantity=total,  # May differ with included amounts
            unit_amount=0,  # Set by billing
            total_amount=0
        )

class UsageBilling:
    def __init__(self, usage_tracker: UsageTracker, pricing: PricingEngine):
        self.usage = usage_tracker
        self.pricing = pricing

    def calculate_usage_charges(self, subscription_id: str,
                                period_start: float,
                                period_end: float) -> list[dict]:
        '''Calculate usage charges for all meters'''
        charges = []

        for meter_id in self.usage.meters:
            summary = self.usage.get_usage_summary(
                subscription_id, meter_id, period_start, period_end
            )

            if summary.total_quantity == 0:
                continue

            # In production: get meter-specific pricing from plan
            # This is simplified
            unit_price = 10  # $0.10 per unit

            charge = {
                "meter_id": meter_id,
                "meter_name": self.usage.meters[meter_id].name,
                "quantity": summary.total_quantity,
                "unit_amount": unit_price,
                "total_amount": summary.total_quantity * unit_price,
                "unit_label": self.usage.meters[meter_id].unit_label
            }
            charges.append(charge)

        return charges

    def get_current_usage(self, subscription_id: str) -> dict:
        '''Get real-time usage for customer dashboard'''
        # In production: might need current period from subscription
        now = time.time()
        period_start = now - (30 * 86400)  # Last 30 days approximation

        usage = {}
        for meter_id, meter in self.usage.meters.items():
            summary = self.usage.get_usage_summary(
                subscription_id, meter_id, period_start, now
            )
            usage[meter_id] = {
                "name": meter.name,
                "quantity": summary.total_quantity,
                "unit_label": meter.unit_label
            }

        return usage

# Example usage tracking middleware
class UsageMiddleware:
    def __init__(self, tracker: UsageTracker):
        self.tracker = tracker

    def track_api_call(self, subscription_id: str, endpoint: str):
        '''Track an API call'''
        self.tracker.record_usage(
            subscription_id=subscription_id,
            meter_id="api_calls",
            quantity=1,
            properties={"endpoint": endpoint}
        )

    def track_storage(self, subscription_id: str, bytes_used: int):
        '''Track storage usage (last value aggregation)'''
        self.tracker.record_usage(
            subscription_id=subscription_id,
            meter_id="storage",
            quantity=bytes_used,  # Total storage, not delta
        )

    def track_active_user(self, subscription_id: str, user_id: str):
        '''Track unique active user'''
        self.tracker.record_usage(
            subscription_id=subscription_id,
            meter_id="active_users",
            quantity=1,
            properties={"unique_key": user_id}
        )
```
"""
                },
                "pitfalls": [
                    "Idempotency: same event reported twice shouldn't be double-counted",
                    "Clock skew: use server timestamps, not client timestamps",
                    "Aggregation type matters: sum vs max give very different results",
                    "Usage limits: alert before hitting limit, not after"
                ]
            }
        ]
    }
}

# Load and update
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

if 'expert_projects' not in data:
    data['expert_projects'] = {}

for project_id, project in payment_billing_projects.items():
    data['expert_projects'][project_id] = project
    print(f"Added: {project_id} - {project['name']}")

with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nAdded {len(payment_billing_projects)} Payment & Billing projects")

#!/usr/bin/env python3
"""
Add Communication projects - multi-channel notifications and webhooks.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

communication_projects = {
    "notification-service": {
        "name": "Multi-Channel Notification Service",
        "description": "Build a unified notification system supporting email, SMS, push notifications, and in-app messages with templates, preferences, and delivery tracking.",
        "why_expert": "Every app needs notifications. Understanding delivery patterns, rate limiting, and preference management helps build user-friendly communication systems.",
        "difficulty": "expert",
        "tags": ["notifications", "messaging", "email", "push", "sms"],
        "estimated_hours": 45,
        "prerequisites": ["build-message-queue"],
        "milestones": [
            {
                "name": "Channel Abstraction & Routing",
                "description": "Implement pluggable channels (email, SMS, push) with intelligent routing",
                "skills": ["Channel abstraction", "Provider fallback", "Routing logic"],
                "hints": {
                    "level1": "Each channel (email, SMS, push) is a plugin implementing common interface",
                    "level2": "Router decides which channel(s) based on notification type and user preferences",
                    "level3": """
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import time

class Channel(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEBHOOK = "webhook"

class Priority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class Notification:
    id: str
    user_id: str
    type: str                     # e.g., "order_shipped", "password_reset"
    title: str
    body: str
    data: dict = field(default_factory=dict)
    channels: list[Channel] = field(default_factory=list)
    priority: Priority = Priority.NORMAL
    created_at: float = field(default_factory=time.time)
    scheduled_for: Optional[float] = None
    expires_at: Optional[float] = None

@dataclass
class DeliveryResult:
    channel: Channel
    success: bool
    provider: str
    message_id: Optional[str] = None
    error: Optional[str] = None
    delivered_at: Optional[float] = None

class NotificationChannel(ABC):
    @abstractmethod
    def send(self, notification: Notification, recipient: dict) -> DeliveryResult:
        pass

    @abstractmethod
    def supports_batch(self) -> bool:
        pass

class EmailChannel(NotificationChannel):
    def __init__(self, providers: list):
        self.providers = providers  # Ordered by preference
        self.current_provider_idx = 0

    def send(self, notification: Notification, recipient: dict) -> DeliveryResult:
        email = recipient.get("email")
        if not email:
            return DeliveryResult(
                channel=Channel.EMAIL,
                success=False,
                provider="none",
                error="No email address"
            )

        # Try providers with fallback
        for i, provider in enumerate(self.providers[self.current_provider_idx:]):
            try:
                result = provider.send_email(
                    to=email,
                    subject=notification.title,
                    body=notification.body,
                    html=notification.data.get("html_body")
                )
                return DeliveryResult(
                    channel=Channel.EMAIL,
                    success=True,
                    provider=provider.name,
                    message_id=result.message_id,
                    delivered_at=time.time()
                )
            except Exception as e:
                if i < len(self.providers) - 1:
                    continue  # Try next provider
                return DeliveryResult(
                    channel=Channel.EMAIL,
                    success=False,
                    provider=provider.name,
                    error=str(e)
                )

    def supports_batch(self) -> bool:
        return True

class SMSChannel(NotificationChannel):
    def __init__(self, provider):
        self.provider = provider

    def send(self, notification: Notification, recipient: dict) -> DeliveryResult:
        phone = recipient.get("phone")
        if not phone:
            return DeliveryResult(
                channel=Channel.SMS,
                success=False,
                provider="none",
                error="No phone number"
            )

        try:
            # SMS is expensive - keep messages short
            message = notification.body[:160]  # SMS character limit

            result = self.provider.send_sms(to=phone, body=message)
            return DeliveryResult(
                channel=Channel.SMS,
                success=True,
                provider=self.provider.name,
                message_id=result.sid
            )
        except Exception as e:
            return DeliveryResult(
                channel=Channel.SMS,
                success=False,
                provider=self.provider.name,
                error=str(e)
            )

    def supports_batch(self) -> bool:
        return False

class PushChannel(NotificationChannel):
    def __init__(self, fcm_client, apns_client):
        self.fcm = fcm_client      # Firebase for Android
        self.apns = apns_client    # Apple Push for iOS

    def send(self, notification: Notification, recipient: dict) -> DeliveryResult:
        device_tokens = recipient.get("device_tokens", [])
        if not device_tokens:
            return DeliveryResult(
                channel=Channel.PUSH,
                success=False,
                provider="none",
                error="No device tokens"
            )

        errors = []
        for token in device_tokens:
            try:
                platform = token.get("platform", "android")
                push_token = token.get("token")

                if platform == "ios":
                    self.apns.send(
                        token=push_token,
                        title=notification.title,
                        body=notification.body,
                        data=notification.data
                    )
                else:
                    self.fcm.send(
                        token=push_token,
                        title=notification.title,
                        body=notification.body,
                        data=notification.data
                    )
            except Exception as e:
                errors.append(str(e))

        if len(errors) == len(device_tokens):
            return DeliveryResult(
                channel=Channel.PUSH,
                success=False,
                provider="fcm/apns",
                error="; ".join(errors)
            )

        return DeliveryResult(
            channel=Channel.PUSH,
            success=True,
            provider="fcm/apns",
            delivered_at=time.time()
        )

    def supports_batch(self) -> bool:
        return True

class NotificationRouter:
    def __init__(self):
        self.channels: dict[Channel, NotificationChannel] = {}
        self.type_channel_map: dict[str, list[Channel]] = {}
        self.priority_channel_map: dict[Priority, list[Channel]] = {
            Priority.URGENT: [Channel.SMS, Channel.PUSH, Channel.EMAIL],
            Priority.HIGH: [Channel.PUSH, Channel.EMAIL],
            Priority.NORMAL: [Channel.EMAIL, Channel.IN_APP],
            Priority.LOW: [Channel.IN_APP]
        }

    def register_channel(self, channel: Channel, implementation: NotificationChannel):
        self.channels[channel] = implementation

    def configure_notification_type(self, notification_type: str, channels: list[Channel]):
        self.type_channel_map[notification_type] = channels

    def route(self, notification: Notification, user_preferences: dict) -> list[Channel]:
        '''Determine which channels to use for a notification'''

        # 1. Check if channels explicitly specified
        if notification.channels:
            return notification.channels

        # 2. Check notification type configuration
        if notification.type in self.type_channel_map:
            channels = self.type_channel_map[notification.type]
        else:
            # 3. Fall back to priority-based routing
            channels = self.priority_channel_map.get(
                notification.priority,
                [Channel.EMAIL]
            )

        # 4. Filter by user preferences
        filtered = []
        for channel in channels:
            pref_key = f"{channel.value}_enabled"
            if user_preferences.get(pref_key, True):  # Default enabled
                filtered.append(channel)

        # 5. Check quiet hours (don't SMS/Push during sleep)
        if self._is_quiet_hours(user_preferences):
            filtered = [c for c in filtered if c not in [Channel.SMS, Channel.PUSH]]

        return filtered or [Channel.IN_APP]  # Always fallback to in-app

    def _is_quiet_hours(self, preferences: dict) -> bool:
        quiet_start = preferences.get("quiet_hours_start")  # e.g., 22 (10 PM)
        quiet_end = preferences.get("quiet_hours_end")      # e.g., 8 (8 AM)

        if not quiet_start or not quiet_end:
            return False

        from datetime import datetime
        hour = datetime.now().hour

        if quiet_start < quiet_end:
            return quiet_start <= hour < quiet_end
        else:  # Crosses midnight
            return hour >= quiet_start or hour < quiet_end
```
"""
                },
                "pitfalls": [
                    "Provider fallback must track failures to avoid cascading errors",
                    "SMS costs money - only use for truly urgent notifications",
                    "Push token can become invalid - handle token refresh errors",
                    "Quiet hours must consider user's timezone, not server's"
                ]
            },
            {
                "name": "Template System",
                "description": "Implement notification templates with localization and personalization",
                "skills": ["Template engines", "i18n", "Content personalization"],
                "hints": {
                    "level1": "Templates separate content from delivery logic - easier to update",
                    "level2": "Support different content per channel (email HTML vs SMS text)",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from jinja2 import Environment, BaseLoader, TemplateNotFound
import json

@dataclass
class NotificationTemplate:
    id: str
    name: str
    type: str                    # Notification type this template is for
    channel_content: dict        # channel -> {subject, body, html_body}
    variables: list[str]         # Required variables
    default_locale: str = "en"
    localized_content: dict = field(default_factory=dict)  # locale -> channel_content

class TemplateStore:
    def __init__(self):
        self.templates: dict[str, NotificationTemplate] = {}

    def create_template(self, template_id: str, name: str, notification_type: str,
                        channel_content: dict, variables: list[str]) -> NotificationTemplate:
        template = NotificationTemplate(
            id=template_id,
            name=name,
            type=notification_type,
            channel_content=channel_content,
            variables=variables
        )
        self.templates[template_id] = template
        return template

    def add_localization(self, template_id: str, locale: str, channel_content: dict):
        template = self.templates.get(template_id)
        if not template:
            raise ValueError("Template not found")
        template.localized_content[locale] = channel_content

    def get_template(self, template_id: str, locale: str = None) -> dict:
        template = self.templates.get(template_id)
        if not template:
            raise ValueError("Template not found")

        if locale and locale in template.localized_content:
            return template.localized_content[locale]
        return template.channel_content

class TemplateRenderer:
    def __init__(self, template_store: TemplateStore):
        self.store = template_store
        self.jinja_env = Environment(loader=BaseLoader())

        # Add custom filters
        self.jinja_env.filters['currency'] = self._currency_filter
        self.jinja_env.filters['date'] = self._date_filter
        self.jinja_env.filters['truncate_sms'] = lambda s: s[:160] if s else ''

    def render(self, template_id: str, channel: Channel,
               variables: dict, locale: str = "en") -> dict:
        '''Render template for a specific channel'''
        template_content = self.store.get_template(template_id, locale)
        channel_content = template_content.get(channel.value, {})

        if not channel_content:
            # Fallback to email content
            channel_content = template_content.get("email", {})

        rendered = {}
        for key in ['subject', 'body', 'html_body']:
            if key in channel_content:
                template = self.jinja_env.from_string(channel_content[key])
                rendered[key] = template.render(**variables)

        return rendered

    def validate_variables(self, template_id: str, variables: dict) -> list[str]:
        '''Check if all required variables are provided'''
        template = self.store.templates.get(template_id)
        if not template:
            raise ValueError("Template not found")

        missing = [v for v in template.variables if v not in variables]
        return missing

    def _currency_filter(self, amount: int, currency: str = "USD") -> str:
        '''Format amount (in cents) as currency'''
        symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
        symbol = symbols.get(currency, currency)
        return f"{symbol}{amount/100:.2f}"

    def _date_filter(self, timestamp: float, format: str = "%B %d, %Y") -> str:
        '''Format timestamp as date'''
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime(format)

# Example template setup
def setup_templates(store: TemplateStore):
    # Order shipped notification
    store.create_template(
        template_id="order_shipped",
        name="Order Shipped",
        notification_type="order_shipped",
        channel_content={
            "email": {
                "subject": "Your order #{{ order_id }} has shipped!",
                "body": "Hi {{ user_name }},\\n\\nGreat news! Your order has shipped.\\n\\nTracking: {{ tracking_number }}\\nCarrier: {{ carrier }}\\n\\nEstimated delivery: {{ delivery_date | date }}",
                "html_body": '''
                    <h1>Your order is on its way!</h1>
                    <p>Hi {{ user_name }},</p>
                    <p>Great news! Your order <strong>#{{ order_id }}</strong> has shipped.</p>
                    <div class="tracking-info">
                        <p><strong>Tracking Number:</strong> {{ tracking_number }}</p>
                        <p><strong>Carrier:</strong> {{ carrier }}</p>
                        <p><strong>Estimated Delivery:</strong> {{ delivery_date | date }}</p>
                    </div>
                    <a href="{{ tracking_url }}" class="button">Track Package</a>
                '''
            },
            "sms": {
                "body": "Order #{{ order_id }} shipped! Track: {{ tracking_url | truncate_sms }}"
            },
            "push": {
                "subject": "Order Shipped",
                "body": "Your order #{{ order_id }} is on the way!"
            }
        },
        variables=["user_name", "order_id", "tracking_number", "carrier",
                   "delivery_date", "tracking_url"]
    )

    # Add Vietnamese localization
    store.add_localization("order_shipped", "vi", {
        "email": {
            "subject": "Đơn hàng #{{ order_id }} đã được gửi!",
            "body": "Xin chào {{ user_name }},\\n\\nĐơn hàng của bạn đã được gửi.\\n\\nMã vận đơn: {{ tracking_number }}"
        },
        "sms": {
            "body": "Đơn hàng #{{ order_id }} đã gửi! Theo dõi: {{ tracking_url | truncate_sms }}"
        }
    })
```
"""
                },
                "pitfalls": [
                    "SMS templates must fit in 160 chars (or pay for multiple segments)",
                    "HTML email needs inline styles - CSS classes often stripped",
                    "Template variables need escaping for XSS prevention in HTML",
                    "Locale detection should fall back gracefully (vi -> vi-VN -> en)"
                ]
            },
            {
                "name": "User Preferences & Unsubscribe",
                "description": "Implement user notification preferences with granular controls and one-click unsubscribe",
                "skills": ["Preference management", "GDPR compliance", "Unsubscribe handling"],
                "hints": {
                    "level1": "Users need control: what notifications, which channels, when",
                    "level2": "CAN-SPAM/GDPR: must include unsubscribe link in marketing emails",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import secrets
import time
import hashlib
import hmac

class NotificationCategory(Enum):
    TRANSACTIONAL = "transactional"  # Order confirmations, receipts
    SECURITY = "security"             # Password reset, login alerts
    MARKETING = "marketing"           # Promotions, newsletters
    PRODUCT = "product"               # Feature updates, tips
    SOCIAL = "social"                 # Comments, mentions

@dataclass
class UserPreferences:
    user_id: str
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = True
    in_app_enabled: bool = True

    # Category-specific settings
    category_settings: dict[str, dict] = field(default_factory=dict)
    # e.g., {"marketing": {"email": False, "push": True}}

    # Quiet hours
    quiet_hours_enabled: bool = False
    quiet_hours_start: int = 22  # 10 PM
    quiet_hours_end: int = 8     # 8 AM
    timezone: str = "UTC"

    # Frequency limits
    max_emails_per_day: Optional[int] = None
    max_sms_per_week: Optional[int] = None

    # Unsubscribe tracking
    unsubscribed_types: set[str] = field(default_factory=set)
    global_unsubscribe: bool = False

    updated_at: float = field(default_factory=time.time)

class PreferenceManager:
    def __init__(self, secret_key: bytes):
        self.preferences: dict[str, UserPreferences] = {}
        self.secret_key = secret_key
        # Track sends for frequency limiting
        self.send_counts: dict[str, dict] = {}  # user_id -> {channel: count}

    def get_preferences(self, user_id: str) -> UserPreferences:
        if user_id not in self.preferences:
            self.preferences[user_id] = UserPreferences(user_id=user_id)
        return self.preferences[user_id]

    def update_preferences(self, user_id: str, updates: dict) -> UserPreferences:
        prefs = self.get_preferences(user_id)

        for key, value in updates.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)

        prefs.updated_at = time.time()
        return prefs

    def set_category_preference(self, user_id: str, category: NotificationCategory,
                                 channel: Channel, enabled: bool):
        prefs = self.get_preferences(user_id)

        if category.value not in prefs.category_settings:
            prefs.category_settings[category.value] = {}

        prefs.category_settings[category.value][channel.value] = enabled
        prefs.updated_at = time.time()

    def can_send(self, user_id: str, notification_type: str,
                 channel: Channel, category: NotificationCategory) -> tuple[bool, str]:
        '''Check if notification can be sent to user'''
        prefs = self.get_preferences(user_id)

        # Global unsubscribe
        if prefs.global_unsubscribe and category != NotificationCategory.TRANSACTIONAL:
            return False, "global_unsubscribe"

        # Type-specific unsubscribe
        if notification_type in prefs.unsubscribed_types:
            return False, f"unsubscribed_from_{notification_type}"

        # Channel disabled globally
        channel_enabled = getattr(prefs, f"{channel.value}_enabled", True)
        if not channel_enabled:
            return False, f"{channel.value}_disabled"

        # Category-specific setting
        cat_settings = prefs.category_settings.get(category.value, {})
        if channel.value in cat_settings and not cat_settings[channel.value]:
            return False, f"category_{category.value}_disabled_for_{channel.value}"

        # Frequency limits
        if not self._check_frequency_limit(user_id, channel, prefs):
            return False, "frequency_limit_exceeded"

        # Transactional and security always allowed (after unsubscribe checks)
        if category in [NotificationCategory.TRANSACTIONAL, NotificationCategory.SECURITY]:
            return True, "allowed_transactional"

        return True, "allowed"

    def _check_frequency_limit(self, user_id: str, channel: Channel,
                                prefs: UserPreferences) -> bool:
        counts = self.send_counts.get(user_id, {})

        if channel == Channel.EMAIL and prefs.max_emails_per_day:
            today_count = counts.get("email_today", 0)
            return today_count < prefs.max_emails_per_day

        if channel == Channel.SMS and prefs.max_sms_per_week:
            week_count = counts.get("sms_week", 0)
            return week_count < prefs.max_sms_per_week

        return True

    def record_send(self, user_id: str, channel: Channel):
        if user_id not in self.send_counts:
            self.send_counts[user_id] = {}

        if channel == Channel.EMAIL:
            self.send_counts[user_id]["email_today"] = (
                self.send_counts[user_id].get("email_today", 0) + 1
            )
        elif channel == Channel.SMS:
            self.send_counts[user_id]["sms_week"] = (
                self.send_counts[user_id].get("sms_week", 0) + 1
            )

    # Unsubscribe link generation
    def generate_unsubscribe_token(self, user_id: str,
                                    notification_type: str = None) -> str:
        '''Generate signed unsubscribe token'''
        timestamp = str(int(time.time()))
        data = f"{user_id}|{notification_type or 'all'}|{timestamp}"

        signature = hmac.new(
            self.secret_key,
            data.encode(),
            hashlib.sha256
        ).hexdigest()[:16]

        token = f"{data}|{signature}"
        # Base64 encode for URL safety
        import base64
        return base64.urlsafe_b64encode(token.encode()).decode()

    def process_unsubscribe(self, token: str) -> dict:
        '''Process unsubscribe from token'''
        import base64
        try:
            decoded = base64.urlsafe_b64decode(token.encode()).decode()
            parts = decoded.split("|")
            if len(parts) != 4:
                return {"success": False, "error": "invalid_token"}

            user_id, notification_type, timestamp, signature = parts

            # Verify signature
            data = f"{user_id}|{notification_type}|{timestamp}"
            expected = hmac.new(
                self.secret_key,
                data.encode(),
                hashlib.sha256
            ).hexdigest()[:16]

            if not hmac.compare_digest(signature, expected):
                return {"success": False, "error": "invalid_signature"}

            # Check token age (30 days max)
            if time.time() - int(timestamp) > 30 * 86400:
                return {"success": False, "error": "token_expired"}

            # Process unsubscribe
            prefs = self.get_preferences(user_id)
            if notification_type == "all":
                prefs.global_unsubscribe = True
            else:
                prefs.unsubscribed_types.add(notification_type)

            return {
                "success": True,
                "user_id": user_id,
                "unsubscribed_from": notification_type
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_unsubscribe_url(self, user_id: str, notification_type: str,
                                  base_url: str) -> str:
        token = self.generate_unsubscribe_token(user_id, notification_type)
        return f"{base_url}/unsubscribe?token={token}"
```
"""
                },
                "pitfalls": [
                    "Transactional emails (receipts, password reset) can't be unsubscribed per CAN-SPAM",
                    "One-click unsubscribe required for marketing - RFC 8058",
                    "Unsubscribe token must be signed to prevent enumeration attacks",
                    "GDPR: must honor unsubscribe within 10 days (do it immediately)"
                ]
            },
            {
                "name": "Delivery Tracking & Analytics",
                "description": "Implement delivery status tracking, open/click tracking, and analytics",
                "skills": ["Event tracking", "Pixel tracking", "Analytics"],
                "hints": {
                    "level1": "Track: sent, delivered, bounced, opened, clicked",
                    "level2": "Email opens tracked via tracking pixel; clicks via redirect",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import defaultdict
import time
import secrets
import hashlib

class DeliveryStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    CLICKED = "clicked"
    BOUNCED = "bounced"
    COMPLAINED = "complained"  # Spam report
    FAILED = "failed"
    UNSUBSCRIBED = "unsubscribed"

@dataclass
class DeliveryEvent:
    notification_id: str
    channel: Channel
    status: DeliveryStatus
    timestamp: float
    metadata: dict = field(default_factory=dict)

@dataclass
class NotificationDelivery:
    notification_id: str
    user_id: str
    channel: Channel
    status: DeliveryStatus
    provider_message_id: Optional[str]
    events: list[DeliveryEvent] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

class DeliveryTracker:
    def __init__(self, tracking_domain: str, secret_key: bytes):
        self.deliveries: dict[str, NotificationDelivery] = {}
        self.tracking_domain = tracking_domain
        self.secret_key = secret_key
        # Tracking ID -> notification mapping
        self.tracking_ids: dict[str, str] = {}

    def create_delivery(self, notification_id: str, user_id: str,
                        channel: Channel) -> NotificationDelivery:
        delivery = NotificationDelivery(
            notification_id=notification_id,
            user_id=user_id,
            channel=channel,
            status=DeliveryStatus.PENDING,
            provider_message_id=None
        )
        self.deliveries[notification_id] = delivery
        return delivery

    def update_status(self, notification_id: str, status: DeliveryStatus,
                      metadata: dict = None):
        delivery = self.deliveries.get(notification_id)
        if not delivery:
            return

        event = DeliveryEvent(
            notification_id=notification_id,
            channel=delivery.channel,
            status=status,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        delivery.events.append(event)
        delivery.status = status

    def handle_provider_webhook(self, provider: str, event_type: str,
                                 data: dict):
        '''Handle delivery webhooks from email/SMS providers'''
        # Map provider event types to our statuses
        event_mapping = {
            "sendgrid": {
                "delivered": DeliveryStatus.DELIVERED,
                "open": DeliveryStatus.OPENED,
                "click": DeliveryStatus.CLICKED,
                "bounce": DeliveryStatus.BOUNCED,
                "spamreport": DeliveryStatus.COMPLAINED
            },
            "twilio": {
                "delivered": DeliveryStatus.DELIVERED,
                "failed": DeliveryStatus.FAILED,
                "undelivered": DeliveryStatus.BOUNCED
            }
        }

        mapping = event_mapping.get(provider, {})
        status = mapping.get(event_type)

        if status:
            message_id = data.get("message_id") or data.get("MessageSid")
            # Find notification by provider message ID
            for nid, delivery in self.deliveries.items():
                if delivery.provider_message_id == message_id:
                    self.update_status(nid, status, {"provider_event": event_type})
                    break

    # Email tracking pixel and link tracking
    def generate_tracking_pixel_url(self, notification_id: str) -> str:
        '''Generate unique tracking pixel URL for email opens'''
        tracking_id = secrets.token_urlsafe(16)
        self.tracking_ids[tracking_id] = notification_id

        # Sign to prevent forgery
        signature = hmac.new(
            self.secret_key,
            tracking_id.encode(),
            hashlib.sha256
        ).hexdigest()[:8]

        return f"https://{self.tracking_domain}/track/open/{tracking_id}/{signature}.gif"

    def generate_tracking_link(self, notification_id: str,
                                original_url: str, link_id: str = None) -> str:
        '''Generate tracking redirect URL for link clicks'''
        tracking_id = secrets.token_urlsafe(16)
        link_id = link_id or hashlib.md5(original_url.encode()).hexdigest()[:8]

        # Store mapping
        self.tracking_ids[tracking_id] = {
            "notification_id": notification_id,
            "original_url": original_url,
            "link_id": link_id
        }

        return f"https://{self.tracking_domain}/track/click/{tracking_id}"

    def handle_tracking_pixel(self, tracking_id: str, signature: str,
                               ip: str, user_agent: str) -> bytes:
        '''Handle tracking pixel request - return 1x1 transparent GIF'''
        # Verify signature
        expected = hmac.new(
            self.secret_key,
            tracking_id.encode(),
            hashlib.sha256
        ).hexdigest()[:8]

        if hmac.compare_digest(signature, expected):
            notification_id = self.tracking_ids.get(tracking_id)
            if notification_id:
                self.update_status(notification_id, DeliveryStatus.OPENED, {
                    "ip": ip,
                    "user_agent": user_agent
                })

        # Return 1x1 transparent GIF
        return bytes([
            0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00,
            0x01, 0x00, 0x80, 0x00, 0x00, 0xff, 0xff, 0xff,
            0x00, 0x00, 0x00, 0x21, 0xf9, 0x04, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44,
            0x01, 0x00, 0x3b
        ])

    def handle_tracking_click(self, tracking_id: str,
                               ip: str, user_agent: str) -> Optional[str]:
        '''Handle click tracking - return redirect URL'''
        data = self.tracking_ids.get(tracking_id)
        if not data or not isinstance(data, dict):
            return None

        self.update_status(data["notification_id"], DeliveryStatus.CLICKED, {
            "ip": ip,
            "user_agent": user_agent,
            "link_id": data["link_id"]
        })

        return data["original_url"]

class NotificationAnalytics:
    def __init__(self, tracker: DeliveryTracker):
        self.tracker = tracker

    def get_delivery_stats(self, start_time: float, end_time: float) -> dict:
        '''Get delivery statistics for time period'''
        stats = defaultdict(lambda: defaultdict(int))

        for delivery in self.tracker.deliveries.values():
            if start_time <= delivery.created_at <= end_time:
                channel = delivery.channel.value
                stats[channel]["total"] += 1
                stats[channel][delivery.status.value] += 1

        # Calculate rates
        result = {}
        for channel, counts in stats.items():
            total = counts["total"]
            result[channel] = {
                "total": total,
                "delivered": counts.get("delivered", 0),
                "delivery_rate": counts.get("delivered", 0) / total if total else 0,
                "opened": counts.get("opened", 0),
                "open_rate": counts.get("opened", 0) / counts.get("delivered", 1),
                "clicked": counts.get("clicked", 0),
                "click_rate": counts.get("clicked", 0) / counts.get("opened", 1),
                "bounced": counts.get("bounced", 0),
                "bounce_rate": counts.get("bounced", 0) / total if total else 0,
                "complained": counts.get("complained", 0)
            }

        return result

    def get_notification_type_performance(self, notification_type: str) -> dict:
        '''Analyze performance of specific notification type'''
        # Would join with notification data to get type
        pass
```
"""
                },
                "pitfalls": [
                    "Email opens tracked via pixel are unreliable (image blocking)",
                    "Apple Mail Privacy Protection prefetches pixels - inflates open rates",
                    "Don't track transactional emails for privacy (password resets)",
                    "Spam complaints must trigger immediate unsubscribe"
                ]
            }
        ]
    },

    "webhook-delivery": {
        "name": "Webhook Delivery System",
        "description": "Build a reliable webhook delivery system with signature verification, retry logic, circuit breakers, and delivery guarantees.",
        "why_expert": "Webhooks are the backbone of modern integrations. Building one teaches async delivery patterns, failure handling, and at-least-once guarantees.",
        "difficulty": "advanced",
        "tags": ["webhooks", "async", "reliability", "integration", "events"],
        "estimated_hours": 35,
        "prerequisites": ["build-message-queue"],
        "milestones": [
            {
                "name": "Webhook Registration & Security",
                "description": "Implement webhook endpoint registration with signature verification",
                "skills": ["HMAC signatures", "Endpoint validation", "Secret management"],
                "hints": {
                    "level1": "Each webhook endpoint gets a unique signing secret",
                    "level2": "Verify endpoint ownership via challenge before activating",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import secrets
import hmac
import hashlib
import time
import httpx

class WebhookStatus(Enum):
    PENDING_VERIFICATION = "pending_verification"
    ACTIVE = "active"
    DISABLED = "disabled"
    FAILED = "failed"  # Too many delivery failures

@dataclass
class WebhookEndpoint:
    id: str
    url: str
    secret: str
    events: list[str]               # Event types to subscribe to
    status: WebhookStatus
    created_at: float
    verified_at: Optional[float] = None
    description: str = ""
    metadata: dict = field(default_factory=dict)

    # Health tracking
    consecutive_failures: int = 0
    last_failure_at: Optional[float] = None
    total_deliveries: int = 0
    successful_deliveries: int = 0

class WebhookRegistry:
    def __init__(self, verify_timeout: int = 30):
        self.endpoints: dict[str, WebhookEndpoint] = {}
        self.verify_timeout = verify_timeout

    def register_endpoint(self, url: str, events: list[str],
                          description: str = "") -> WebhookEndpoint:
        '''Register new webhook endpoint'''
        # Validate URL
        if not url.startswith("https://"):
            raise ValueError("Webhook URLs must use HTTPS")

        endpoint_id = secrets.token_urlsafe(16)
        signing_secret = f"whsec_{secrets.token_urlsafe(32)}"

        endpoint = WebhookEndpoint(
            id=endpoint_id,
            url=url,
            secret=signing_secret,
            events=events,
            status=WebhookStatus.PENDING_VERIFICATION,
            created_at=time.time(),
            description=description
        )

        self.endpoints[endpoint_id] = endpoint

        # Initiate verification
        self._send_verification_challenge(endpoint)

        return endpoint

    def _send_verification_challenge(self, endpoint: WebhookEndpoint):
        '''Send challenge to verify endpoint ownership'''
        challenge = secrets.token_urlsafe(32)

        # Store challenge for later verification
        endpoint.metadata["verification_challenge"] = challenge
        endpoint.metadata["verification_expires"] = time.time() + 3600  # 1 hour

        payload = {
            "type": "webhook.verification",
            "challenge": challenge
        }

        try:
            # Endpoint must respond with challenge value
            response = httpx.post(
                endpoint.url,
                json=payload,
                headers=self._build_headers(endpoint, payload),
                timeout=self.verify_timeout
            )

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("challenge") == challenge:
                    endpoint.status = WebhookStatus.ACTIVE
                    endpoint.verified_at = time.time()
                    del endpoint.metadata["verification_challenge"]
        except Exception as e:
            endpoint.metadata["verification_error"] = str(e)

    def _build_headers(self, endpoint: WebhookEndpoint, payload: dict) -> dict:
        '''Build headers with signature'''
        timestamp = str(int(time.time()))
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)

        # Create signature
        signature_payload = f"{timestamp}.{payload_str}"
        signature = hmac.new(
            endpoint.secret.encode(),
            signature_payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "Content-Type": "application/json",
            "X-Webhook-ID": endpoint.id,
            "X-Webhook-Timestamp": timestamp,
            "X-Webhook-Signature": f"v1={signature}",
            "User-Agent": "WebhookDelivery/1.0"
        }

    def update_events(self, endpoint_id: str, events: list[str]) -> WebhookEndpoint:
        endpoint = self.endpoints.get(endpoint_id)
        if not endpoint:
            raise ValueError("Endpoint not found")

        endpoint.events = events
        return endpoint

    def disable_endpoint(self, endpoint_id: str):
        endpoint = self.endpoints.get(endpoint_id)
        if endpoint:
            endpoint.status = WebhookStatus.DISABLED

    def get_endpoints_for_event(self, event_type: str) -> list[WebhookEndpoint]:
        '''Get all active endpoints subscribed to event type'''
        return [
            e for e in self.endpoints.values()
            if e.status == WebhookStatus.ACTIVE
            and (event_type in e.events or "*" in e.events)
        ]

# Signature verification helper for receivers
class WebhookSignatureVerifier:
    def __init__(self, secret: str, tolerance: int = 300):
        self.secret = secret
        self.tolerance = tolerance  # Timestamp tolerance in seconds

    def verify(self, payload: bytes, timestamp: str, signature: str) -> bool:
        '''Verify webhook signature'''
        # Check timestamp to prevent replay attacks
        try:
            ts = int(timestamp)
            if abs(time.time() - ts) > self.tolerance:
                return False
        except ValueError:
            return False

        # Verify signature
        expected_payload = f"{timestamp}.{payload.decode()}"
        expected_sig = hmac.new(
            self.secret.encode(),
            expected_payload.encode(),
            hashlib.sha256
        ).hexdigest()

        # Extract signature value
        if signature.startswith("v1="):
            signature = signature[3:]

        return hmac.compare_digest(expected_sig, signature)
```
"""
                },
                "pitfalls": [
                    "HTTPS only - never deliver webhooks over HTTP",
                    "Timestamp tolerance prevents replay but allows clock skew",
                    "Signing secret rotation: support multiple active secrets temporarily",
                    "URL validation: block private IPs to prevent SSRF"
                ]
            },
            {
                "name": "Delivery Queue & Retry Logic",
                "description": "Implement reliable delivery with exponential backoff and dead letter queue",
                "skills": ["Retry strategies", "Exponential backoff", "Dead letter queues"],
                "hints": {
                    "level1": "Use message queue for delivery; retry with exponential backoff on failure",
                    "level2": "After N retries, move to dead letter queue for manual review",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
import time
import random
import httpx
import asyncio

class DeliveryStatus(Enum):
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"

@dataclass
class WebhookDelivery:
    id: str
    endpoint_id: str
    event_type: str
    payload: dict
    status: DeliveryStatus
    created_at: float
    scheduled_for: float         # When to attempt delivery
    attempts: int = 0
    max_attempts: int = 5
    last_attempt_at: Optional[float] = None
    last_error: Optional[str] = None
    last_response_code: Optional[int] = None
    delivered_at: Optional[float] = None

class RetryPolicy:
    def __init__(self, base_delay: int = 60,
                 max_delay: int = 3600,
                 jitter: float = 0.1):
        self.base_delay = base_delay   # 1 minute
        self.max_delay = max_delay     # 1 hour
        self.jitter = jitter           # 10% random jitter

    def get_next_delay(self, attempt: int) -> int:
        '''Calculate delay with exponential backoff and jitter'''
        # Exponential: 1m, 2m, 4m, 8m, 16m...
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)

        # Add jitter to prevent thundering herd
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        return int(delay)

class WebhookDeliveryQueue:
    def __init__(self, registry: WebhookRegistry,
                 retry_policy: RetryPolicy = None):
        self.registry = registry
        self.retry_policy = retry_policy or RetryPolicy()
        self.pending: list[WebhookDelivery] = []
        self.dead_letter: list[WebhookDelivery] = []
        self.delivered: list[WebhookDelivery] = []

    def enqueue(self, endpoint_id: str, event_type: str,
                payload: dict) -> WebhookDelivery:
        '''Add webhook delivery to queue'''
        delivery = WebhookDelivery(
            id=secrets.token_urlsafe(16),
            endpoint_id=endpoint_id,
            event_type=event_type,
            payload=payload,
            status=DeliveryStatus.PENDING,
            created_at=time.time(),
            scheduled_for=time.time()  # Immediate
        )

        self.pending.append(delivery)
        return delivery

    def enqueue_for_event(self, event_type: str, payload: dict) -> list[WebhookDelivery]:
        '''Fan out event to all subscribed endpoints'''
        endpoints = self.registry.get_endpoints_for_event(event_type)
        deliveries = []

        for endpoint in endpoints:
            delivery = self.enqueue(endpoint.id, event_type, payload)
            deliveries.append(delivery)

        return deliveries

    async def process_queue(self):
        '''Process pending deliveries'''
        now = time.time()
        ready = [d for d in self.pending if d.scheduled_for <= now]

        for delivery in ready:
            self.pending.remove(delivery)
            await self._attempt_delivery(delivery)

    async def _attempt_delivery(self, delivery: WebhookDelivery):
        '''Attempt single delivery'''
        endpoint = self.registry.endpoints.get(delivery.endpoint_id)
        if not endpoint or endpoint.status != WebhookStatus.ACTIVE:
            delivery.status = DeliveryStatus.DEAD_LETTER
            delivery.last_error = "Endpoint not active"
            self.dead_letter.append(delivery)
            return

        delivery.status = DeliveryStatus.IN_FLIGHT
        delivery.attempts += 1
        delivery.last_attempt_at = time.time()

        try:
            # Build signed request
            headers = self.registry._build_headers(endpoint, delivery.payload)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint.url,
                    json=delivery.payload,
                    headers=headers,
                    timeout=30.0
                )

            delivery.last_response_code = response.status_code

            # 2xx = success
            if 200 <= response.status_code < 300:
                delivery.status = DeliveryStatus.DELIVERED
                delivery.delivered_at = time.time()
                self.delivered.append(delivery)

                # Update endpoint health
                endpoint.total_deliveries += 1
                endpoint.successful_deliveries += 1
                endpoint.consecutive_failures = 0
                return

            # 4xx (except 429) = don't retry, bad request
            if 400 <= response.status_code < 500 and response.status_code != 429:
                delivery.status = DeliveryStatus.DEAD_LETTER
                delivery.last_error = f"HTTP {response.status_code}"
                self.dead_letter.append(delivery)
                return

            # 5xx or 429 = retry
            delivery.last_error = f"HTTP {response.status_code}"

        except httpx.TimeoutException:
            delivery.last_error = "Timeout"
        except httpx.RequestError as e:
            delivery.last_error = str(e)

        # Failed - check retry
        endpoint.total_deliveries += 1
        endpoint.consecutive_failures += 1
        endpoint.last_failure_at = time.time()

        if delivery.attempts >= delivery.max_attempts:
            delivery.status = DeliveryStatus.DEAD_LETTER
            self.dead_letter.append(delivery)

            # Disable endpoint if too many failures
            if endpoint.consecutive_failures >= 10:
                endpoint.status = WebhookStatus.FAILED
        else:
            # Schedule retry
            delay = self.retry_policy.get_next_delay(delivery.attempts)
            delivery.status = DeliveryStatus.PENDING
            delivery.scheduled_for = time.time() + delay
            self.pending.append(delivery)

    def retry_dead_letter(self, delivery_id: str) -> Optional[WebhookDelivery]:
        '''Manually retry a dead letter delivery'''
        for delivery in self.dead_letter:
            if delivery.id == delivery_id:
                self.dead_letter.remove(delivery)
                delivery.status = DeliveryStatus.PENDING
                delivery.scheduled_for = time.time()
                delivery.attempts = 0  # Reset attempt count
                self.pending.append(delivery)
                return delivery
        return None

    def get_delivery_status(self, delivery_id: str) -> Optional[WebhookDelivery]:
        for queue in [self.pending, self.delivered, self.dead_letter]:
            for delivery in queue:
                if delivery.id == delivery_id:
                    return delivery
        return None
```
"""
                },
                "pitfalls": [
                    "Jitter prevents thundering herd when many webhooks fail simultaneously",
                    "4xx errors (except 429) shouldn't retry - request is malformed",
                    "Dead letter queue needs alerting and manual review process",
                    "Circuit breaker: disable endpoint after consecutive failures"
                ]
            },
            {
                "name": "Circuit Breaker & Rate Limiting",
                "description": "Implement circuit breaker to protect failing endpoints and rate limiting",
                "skills": ["Circuit breaker pattern", "Rate limiting", "Health checks"],
                "hints": {
                    "level1": "Circuit breaker prevents hammering failing endpoints",
                    "level2": "States: closed (normal), open (skip), half-open (test)",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import time
import threading

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, skip requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreaker:
    endpoint_id: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = 0

    # Thresholds
    failure_threshold: int = 5     # Failures to open
    success_threshold: int = 3     # Successes to close
    timeout: int = 60              # Seconds before half-open

class CircuitBreakerManager:
    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}
        self.lock = threading.Lock()

    def get_breaker(self, endpoint_id: str) -> CircuitBreaker:
        with self.lock:
            if endpoint_id not in self.breakers:
                self.breakers[endpoint_id] = CircuitBreaker(
                    endpoint_id=endpoint_id,
                    last_state_change=time.time()
                )
            return self.breakers[endpoint_id]

    def can_execute(self, endpoint_id: str) -> bool:
        '''Check if request should proceed'''
        breaker = self.get_breaker(endpoint_id)

        if breaker.state == CircuitState.CLOSED:
            return True

        if breaker.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - breaker.last_state_change >= breaker.timeout:
                self._transition(breaker, CircuitState.HALF_OPEN)
                return True  # Allow test request
            return False

        if breaker.state == CircuitState.HALF_OPEN:
            return True  # Allow test requests

        return False

    def record_success(self, endpoint_id: str):
        '''Record successful delivery'''
        breaker = self.get_breaker(endpoint_id)

        with self.lock:
            if breaker.state == CircuitState.HALF_OPEN:
                breaker.success_count += 1
                if breaker.success_count >= breaker.success_threshold:
                    self._transition(breaker, CircuitState.CLOSED)
            elif breaker.state == CircuitState.CLOSED:
                breaker.failure_count = 0  # Reset on success

    def record_failure(self, endpoint_id: str):
        '''Record failed delivery'''
        breaker = self.get_breaker(endpoint_id)

        with self.lock:
            breaker.failure_count += 1
            breaker.last_failure_time = time.time()

            if breaker.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition(breaker, CircuitState.OPEN)
            elif breaker.state == CircuitState.CLOSED:
                if breaker.failure_count >= breaker.failure_threshold:
                    self._transition(breaker, CircuitState.OPEN)

    def _transition(self, breaker: CircuitBreaker, new_state: CircuitState):
        breaker.state = new_state
        breaker.last_state_change = time.time()

        if new_state == CircuitState.CLOSED:
            breaker.failure_count = 0
            breaker.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            breaker.success_count = 0

class RateLimiter:
    '''Per-endpoint rate limiting using sliding window'''

    def __init__(self, requests_per_second: int = 10,
                 requests_per_minute: int = 100):
        self.rps = requests_per_second
        self.rpm = requests_per_minute
        self.second_windows: dict[str, list[float]] = {}
        self.minute_windows: dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def can_send(self, endpoint_id: str) -> tuple[bool, Optional[float]]:
        '''Check if can send, returns (allowed, retry_after)'''
        now = time.time()

        with self.lock:
            # Check per-second limit
            second_window = self.second_windows.get(endpoint_id, [])
            second_window = [t for t in second_window if now - t < 1]
            self.second_windows[endpoint_id] = second_window

            if len(second_window) >= self.rps:
                retry_after = 1 - (now - second_window[0])
                return False, retry_after

            # Check per-minute limit
            minute_window = self.minute_windows.get(endpoint_id, [])
            minute_window = [t for t in minute_window if now - t < 60]
            self.minute_windows[endpoint_id] = minute_window

            if len(minute_window) >= self.rpm:
                retry_after = 60 - (now - minute_window[0])
                return False, retry_after

            return True, None

    def record_send(self, endpoint_id: str):
        '''Record a send'''
        now = time.time()
        with self.lock:
            if endpoint_id not in self.second_windows:
                self.second_windows[endpoint_id] = []
            if endpoint_id not in self.minute_windows:
                self.minute_windows[endpoint_id] = []

            self.second_windows[endpoint_id].append(now)
            self.minute_windows[endpoint_id].append(now)

class WebhookDeliveryWithProtection:
    '''Delivery queue with circuit breaker and rate limiting'''

    def __init__(self, queue: WebhookDeliveryQueue):
        self.queue = queue
        self.circuit_breaker = CircuitBreakerManager()
        self.rate_limiter = RateLimiter()

    async def attempt_delivery(self, delivery: WebhookDelivery) -> bool:
        endpoint_id = delivery.endpoint_id

        # Check circuit breaker
        if not self.circuit_breaker.can_execute(endpoint_id):
            # Reschedule for later
            delivery.scheduled_for = time.time() + 60
            return False

        # Check rate limit
        allowed, retry_after = self.rate_limiter.can_send(endpoint_id)
        if not allowed:
            delivery.scheduled_for = time.time() + (retry_after or 1)
            return False

        # Attempt delivery
        self.rate_limiter.record_send(endpoint_id)
        success = await self.queue._attempt_delivery(delivery)

        if success:
            self.circuit_breaker.record_success(endpoint_id)
        else:
            self.circuit_breaker.record_failure(endpoint_id)

        return success
```
"""
                },
                "pitfalls": [
                    "Half-open: only allow limited test requests, not full traffic",
                    "Circuit breaker timeout should increase on repeated failures",
                    "Rate limiting should respect Retry-After headers from receiver",
                    "Alert when circuit opens - endpoint owner needs to know"
                ]
            },
            {
                "name": "Event Log & Replay",
                "description": "Implement event logging for debugging and replay capability",
                "skills": ["Event sourcing", "Log retention", "Event replay"],
                "hints": {
                    "level1": "Log all webhook events for debugging and audit",
                    "level2": "Allow replaying failed webhooks from a specific time",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional, Iterator
import time
import json

@dataclass
class WebhookEventLog:
    id: str
    event_type: str
    payload: dict
    timestamp: float
    deliveries: list[dict] = field(default_factory=list)
    # Each delivery: {endpoint_id, delivery_id, status, attempts, delivered_at}

class EventLogStore:
    def __init__(self, retention_days: int = 30):
        self.events: list[WebhookEventLog] = []
        self.retention_days = retention_days
        self.index_by_type: dict[str, list[int]] = {}  # event_type -> indices

    def log_event(self, event_type: str, payload: dict) -> WebhookEventLog:
        '''Log a new event'''
        event = WebhookEventLog(
            id=secrets.token_urlsafe(16),
            event_type=event_type,
            payload=payload,
            timestamp=time.time()
        )

        idx = len(self.events)
        self.events.append(event)

        # Index by type
        if event_type not in self.index_by_type:
            self.index_by_type[event_type] = []
        self.index_by_type[event_type].append(idx)

        return event

    def record_delivery(self, event_id: str, endpoint_id: str,
                        delivery_id: str, status: str,
                        attempts: int = 0, delivered_at: float = None):
        '''Record delivery attempt for an event'''
        for event in self.events:
            if event.id == event_id:
                event.deliveries.append({
                    "endpoint_id": endpoint_id,
                    "delivery_id": delivery_id,
                    "status": status,
                    "attempts": attempts,
                    "delivered_at": delivered_at
                })
                return

    def query_events(self, event_type: str = None,
                     start_time: float = None,
                     end_time: float = None,
                     limit: int = 100,
                     offset: int = 0) -> list[WebhookEventLog]:
        '''Query events with filters'''
        results = self.events

        if event_type:
            indices = self.index_by_type.get(event_type, [])
            results = [self.events[i] for i in indices]

        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        # Sort by timestamp descending (newest first)
        results = sorted(results, key=lambda e: e.timestamp, reverse=True)

        return results[offset:offset + limit]

    def get_failed_deliveries(self, endpoint_id: str,
                              start_time: float = None) -> list[WebhookEventLog]:
        '''Get events with failed deliveries to specific endpoint'''
        results = []

        for event in self.events:
            if start_time and event.timestamp < start_time:
                continue

            for delivery in event.deliveries:
                if (delivery["endpoint_id"] == endpoint_id and
                    delivery["status"] in ["failed", "dead_letter"]):
                    results.append(event)
                    break

        return results

    def cleanup_old_events(self):
        '''Remove events older than retention period'''
        cutoff = time.time() - (self.retention_days * 86400)
        self.events = [e for e in self.events if e.timestamp >= cutoff]
        # Rebuild indices
        self._rebuild_indices()

    def _rebuild_indices(self):
        self.index_by_type = {}
        for i, event in enumerate(self.events):
            if event.event_type not in self.index_by_type:
                self.index_by_type[event.event_type] = []
            self.index_by_type[event.event_type].append(i)

class WebhookReplayService:
    def __init__(self, event_store: EventLogStore,
                 delivery_queue: WebhookDeliveryQueue):
        self.store = event_store
        self.queue = delivery_queue

    def replay_event(self, event_id: str,
                     endpoint_ids: list[str] = None) -> list[WebhookDelivery]:
        '''Replay a specific event'''
        event = None
        for e in self.store.events:
            if e.id == event_id:
                event = e
                break

        if not event:
            raise ValueError("Event not found")

        deliveries = []

        if endpoint_ids:
            # Replay to specific endpoints
            for endpoint_id in endpoint_ids:
                delivery = self.queue.enqueue(
                    endpoint_id, event.event_type, event.payload
                )
                deliveries.append(delivery)
        else:
            # Replay to all original endpoints that failed
            for d in event.deliveries:
                if d["status"] in ["failed", "dead_letter"]:
                    delivery = self.queue.enqueue(
                        d["endpoint_id"], event.event_type, event.payload
                    )
                    deliveries.append(delivery)

        return deliveries

    def replay_range(self, endpoint_id: str,
                     start_time: float, end_time: float = None) -> list[WebhookDelivery]:
        '''Replay all events in a time range to an endpoint'''
        end_time = end_time or time.time()
        events = self.store.query_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Safety limit
        )

        deliveries = []
        for event in events:
            # Check if endpoint is subscribed to this event type
            endpoint = self.queue.registry.endpoints.get(endpoint_id)
            if endpoint and (event.event_type in endpoint.events or "*" in endpoint.events):
                delivery = self.queue.enqueue(
                    endpoint_id, event.event_type, event.payload
                )
                deliveries.append(delivery)

        return deliveries

    def export_events(self, start_time: float, end_time: float,
                      event_types: list[str] = None) -> Iterator[dict]:
        '''Export events as JSON for external processing'''
        events = self.store.query_events(
            start_time=start_time,
            end_time=end_time,
            limit=100000
        )

        for event in events:
            if event_types and event.event_type not in event_types:
                continue

            yield {
                "id": event.id,
                "type": event.event_type,
                "timestamp": event.timestamp,
                "payload": event.payload
            }
```
"""
                },
                "pitfalls": [
                    "Event log can grow huge - implement retention and archival",
                    "Replay should use new delivery IDs to track separately",
                    "Bulk replay can overwhelm endpoints - respect rate limits",
                    "Payload might be stale on replay - warn users"
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

for project_id, project in communication_projects.items():
    data['expert_projects'][project_id] = project
    print(f"Added: {project_id} - {project['name']}")

with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nAdded {len(communication_projects)} Communication projects")

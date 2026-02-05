# Project Athena: Design Document for Intent-Based Chatbot


## Overview

This document outlines the architecture for an intent-based chatbot that interprets user messages to understand goals (intents) and extract key information (entities), managing multi-turn conversations without using Large Language Models (LLMs). The key architectural challenge is designing a modular, maintainable system that accurately interprets user input while gracefully handling ambiguity, missing information, and conversational context.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## 1. Context and Problem Statement

> **Milestone(s):** All Milestones (Foundational Context)

Building a conversational agent that can understand and respond to human language is a fundamental challenge in artificial intelligence. While modern Large Language Models (LLMs) offer impressive capabilities, they require substantial computational resources, extensive training data, and can be opaque in their decision-making. This project addresses the core problem: **How can we build a practical, maintainable chatbot that understands user goals from natural language without relying on LLMs?**

The challenge involves translating unstructured text into structured actions. When a user says "I want to book a table for four tomorrow at 7 PM," the system must recognize the booking intent, extract the entities (party size: 4, date: tomorrow, time: 7 PM), and manage the conversation until all required information is collected. This process—**intent classification**, **entity extraction**, and **dialog management**—forms the backbone of traditional conversational AI systems.

### The Concierge Mental Model

Imagine you're a hotel concierge at a busy front desk. Your job involves:

1. **Quickly categorizing requests** - Is the guest asking about restaurant reservations, tour bookings, or room service?
2. **Extracting key details** - For a reservation, you need to know: how many people, what date, what time, any special requirements.
3. **Managing the conversation flow** - If the guest says "I'd like to book a table," you know to ask follow-up questions: "For how many people?" "What time?" "Any dietary restrictions?"
4. **Providing appropriate responses** - Once you have all the information, you confirm and execute: "I've booked a table for four at 7 PM tomorrow."

This concierge doesn't need to understand every nuance of language. They rely on patterns, checklists, and clear procedures. Our chatbot follows the same mental model: it's a **procedural problem-solver** rather than a creative conversationalist. This approach makes the system:

- **Predictable** - Follows defined paths for known intents
- **Maintainable** - New intents can be added by defining patterns and required information
- **Explainable** - Decisions are based on explicit rules and confidence scores
- **Resource-efficient** - Runs on modest hardware without GPU acceleration

The concierge model also clarifies what our system *isn't*: it's not a general-purpose AI that can discuss philosophy, write poetry, or reason about novel situations. It's a specialized tool for structured interactions within defined domains.

### Defining Intent and Entity Understanding

At the core of our system are two fundamental concepts that transform natural language into actionable data:

**Intent** represents the user's goal or purpose—*what* they want to accomplish. Intents are categorical labels applied to whole utterances. For example:
- `book_restaurant` - User wants to reserve a table
- `check_weather` - User wants weather information  
- `cancel_order` - User wants to cancel a previous order
- `greeting` - User is saying hello

Each intent defines a distinct "conversational pathway" with its own required information, validation rules, and response templates. The system must handle both **single-intent** scenarios (clear, unambiguous requests) and **multi-intent** scenarios where a user combines requests ("book a table and tell me the weather").

**Entity** represents the specific pieces of information mentioned—*the details* needed to fulfill the intent. Entities are extracted from the text itself:
- **Named Entities**: Predefined categories like people (`"John Smith"`), locations (`"Paris"`), dates (`"tomorrow"`), organizations
- **Custom Entities**: Domain-specific values like `cuisine_type` (`"Italian"`), `movie_genre` (`"sci-fi"`), `product_code` (`"SKU-123"`)
- **Structured Entities**: Pattern-based values like phone numbers, email addresses, credit card numbers

The relationship between intents and entities is defined through **slots**—named parameters that an intent requires. For a `book_restaurant` intent, slots might include: `party_size`, `date`, `time`, `restaurant_name`. The entity extractor finds values in the user's message, and the slot filler maps them to the appropriate slots.

> **Key Insight:** Intents answer "what does the user want to do?" while entities answer "with what specific details?" The chatbot's job is to correctly identify both and ensure all required slots are filled before taking action.

This separation creates a clear architecture: the Intent Classifier acts as a router, the Entity Extractor gathers details, and the Dialog Manager ensures completeness before response generation.

### Comparison of Existing Approaches

Several architectural patterns exist for building conversational agents, each with different trade-offs in complexity, maintainability, and capability. Understanding these alternatives clarifies why we've chosen our specific hybrid approach.

**Architecture Decision Record: Hybrid ML + Rules Approach**

> **Decision: Hybrid ML for Classification, Rules for Extraction**
> - **Context**: We need a system that balances accuracy with explainability, works with limited training data, and handles both common patterns and edge cases efficiently. Pure ML approaches risk being opaque and data-hungry; pure rule-based approaches become unmaintainable.
> - **Options Considered**: 
>   1. Pure rule-based system (regex patterns, keyword matching)
>   2. Pure machine learning (end-to-end neural models)
>   3. Hybrid approach (ML for intent classification, rules + patterns for entity extraction)
> - **Decision**: Adopt a hybrid approach: ML-based intent classification with TF-IDF + classifier, and rule-based entity extraction with regex patterns + spaCy NER.
> - **Rationale**: 
>   - Intent classification benefits from ML's ability to generalize across phrasings ("book a table," "reserve seats," "make a dinner reservation").
>   - Entity extraction often requires precise pattern matching (dates, phone numbers) or knowledge of linguistic structure (noun phrases for names) that rules handle reliably.
>   - The hybrid approach provides clear separation of concerns: statistical understanding of intent, deterministic extraction of details.
>   - This architecture aligns with our concierge mental model—the concierge understands the request's category (ML) but uses checklists for details (rules).
> - **Consequences**: 
>   - Positive: More maintainable (rules can be edited directly), better handling of structured entities, interpretable decisions.
>   - Negative: Requires maintaining both training data and rule sets, potential redundancy between ML and rule components.

The following table compares the three major approaches:

| Approach | How It Works | Pros | Cons | Best For |
|----------|--------------|------|------|----------|
| **Pure Rule-Based** | Pattern matching with regular expressions, keyword lists, and hand-crafted grammar rules | • Highly interpretable<br>• No training data needed<br>• Precise control over behavior<br>• Fast execution | • Brittle to phrasing variations<br>• Manual maintenance scales poorly<br>• Cannot generalize to unseen phrasings<br>• Limited natural language understanding | Small, fixed domains with predictable phrasing (IVR systems, command interfaces) |
| **Pure Machine Learning** | End-to-end neural networks (transformers, RNNs) trained on large datasets of conversations | • Handles diverse phrasing naturally<br>• Can learn complex patterns<br>• Reduces manual rule writing<br>• State-of-the-art performance with sufficient data | • Requires large, diverse training datasets<br>• Computationally expensive<br>• Black-box decisions (hard to debug)<br>• Can hallucinate or produce unsafe outputs | Large-scale systems with abundant training data, where maximum accuracy is paramount |
| **Hybrid (Our Approach)** | ML for intent classification, rule-based patterns for entity extraction, state machine for dialog | • Balances generalization and precision<br>• Explainable entity extraction<br>• Maintainable with moderate training data<br>• Clear separation of concerns | • Requires both ML and rule expertise<br>• Potential redundancy between components<br>• Manual tuning needed for edge cases | Practical business applications with 10-100 intents, where maintainability and reliability matter more than cutting-edge performance |

Our system specifically avoids LLMs (like GPT-3/4) for several reasons aligned with our project goals:
1. **Resource Constraints**: LLMs require significant GPU memory and computational power, making them impractical for beginner projects or resource-constrained deployments.
2. **Determinism**: LLM outputs can be stochastic and unpredictable, while our concierge model requires reliable, repeatable behavior.
3. **Transparency**: With our approach, we can trace exactly why a particular intent was chosen (confidence scores) or why an entity was extracted (matching pattern).
4. **Data Efficiency**: We can achieve good performance with tens of examples per intent, not millions of tokens of pretraining data.
5. **Learning Value**: Building components from scratch teaches fundamental NLP concepts that are abstracted away by LLM APIs.

The system diagram below shows how our hybrid approach structures the components:

![System Component Diagram](./diagrams/sys-component-diagram.svg)

**Common Pitfalls in Approach Selection**

⚠️ **Pitfall: Over-reliance on pure regex for intent classification**
- **Description**: Trying to match every possible phrasing of an intent with regular expressions (e.g., `/book|reserve|make.*table|dinner.*reservation/`).
- **Why it's wrong**: This approach fails to generalize. Users will phrase requests in ways the regex author didn't anticipate ("Can we get a table for tonight?", "I'd like to make a booking"). The system becomes brittle and requires constant tweaking.
- **Fix**: Use ML-based classification for intents, reserving regex for structured entity patterns (dates, times, product codes) where patterns are well-defined.

⚠️ **Pitfall: Using ML for everything because "it's more advanced"**
- **Description**: Applying neural networks to extract entities like phone numbers or email addresses.
- **Why it's wrong**: ML models can make mistakes on pattern-based entities that regex handles perfectly. Training requires labeled examples of these entities in context, which is unnecessary work for problems already solved by rules.
- **Fix**: Follow the hybrid approach: ML for fuzzy classification problems, rules for deterministic extraction problems. Use spaCy's NER for linguistic entities (person names, locations) and custom regex for domain-specific patterns.

⚠️ **Pitfall: Treating the chatbot as a general conversation partner**
- **Description**: Expecting the system to handle open-ended discussion, humor, or creative tasks.
- **Why it's wrong**: Our concierge model is designed for goal-oriented conversations with clear intents. Without LLM capabilities, the system lacks world knowledge and generative capacity for open domains.
- **Fix**: Clearly scope the chatbot's domain and capabilities. Design fallback responses that gracefully guide users back to supported intents ("I can help you with restaurant bookings, weather checks, and order cancellations. What would you like to do?").

### Implementation Guidance

While this section is primarily conceptual, establishing the right mental models and architectural understanding is critical before writing code. Here are practical steps to translate these concepts into implementation:

**A. Technology Recommendations Table**

| Component | Simple Option | Advanced Option | Rationale |
|-----------|---------------|-----------------|-----------|
| Intent Classification | `scikit-learn` with `TfidfVectorizer` and `MultinomialNB`/`SVM` | Custom neural network with `TensorFlow`/`PyTorch` | Scikit-learn provides production-ready implementations with minimal code; neural nets offer better accuracy but more complexity |
| Entity Extraction (Rules) | Python `re` module for regex patterns | Advanced pattern engines like `parsley` or `pyparsing` | Built-in `re` module is sufficient for most patterns; advanced engines help with complex grammatical structures |
| Entity Extraction (NER) | `spaCy` small English model (`en_core_web_sm`) | `spaCy` transformer model or custom CRF | The small model balances accuracy and speed; transformer models are more accurate but slower |
| Dialog Management | Custom state machine with Python classes | `Rasa` dialog management or statechart libraries | Custom implementation teaches fundamentals; frameworks handle edge cases but abstract core concepts |
| Response Generation | Python f-strings or `str.format()` | Template engines like `Jinja2` or dedicated NLG libraries | Simple string formatting is transparent; template engines support conditionals and inheritance |
| Session Storage | In-memory dictionary with session IDs | Redis or SQLite for persistence | In-memory is simplest for learning; persistent storage needed for production deployment |

**B. Recommended Initial Project Structure**

Before writing any components, establish this directory structure:

```
project-athena/
├── data/                          # Training data and configurations
│   ├── intents/                   # Intent training examples
│   │   ├── greeting.json          # Examples for greeting intent
│   │   ├── book_restaurant.json   # Examples for booking intent
│   │   └── check_weather.json     # Examples for weather intent
│   ├── entities/                  # Entity patterns and lookup lists
│   │   ├── patterns.json          # Regex patterns for dates, times, etc.
│   │   └── lookup_tables/         # CSV/JSON files for entity values
│   └── responses/                 # Response templates
│       ├── greeting.yaml          # Response variations for greeting
│       └── book_restaurant.yaml   # Templates for booking flow
├── src/                           # Source code
│   ├── classifier/                # Intent classification component
│   │   ├── __init__.py
│   │   ├── vectorizer.py          # TF-IDF vectorization
│   │   ├── model.py               # Classification model
│   │   └── train.py               # Training script
│   ├── entities/                  # Entity extraction component
│   │   ├── __init__.py
│   │   ├── extractor.py           # Main entity extraction logic
│   │   ├── patterns.py            # Regex pattern definitions
│   │   └── normalizer.py          # Entity value normalization
│   ├── dialog/                    # Dialog management component
│   │   ├── __init__.py
│   │   ├── manager.py             # Dialog state machine
│   │   ├── state.py               # DialogState class definition
│   │   └── slots.py               # Slot filling logic
│   ├── nlg/                       # Natural language generation
│   │   ├── __init__.py
│   │   ├── generator.py           # Response template rendering
│   │   └── templates.py           # Template loading and management
│   └── utils/                     # Shared utilities
│       ├── __init__.py
│       ├── preprocessor.py        # Text cleaning and normalization
│       └── session_store.py       # Session management
├── tests/                         # Test suites
│   ├── test_classifier.py
│   ├── test_entities.py
│   ├── test_dialog.py
│   └── test_integration.py
├── models/                        # Trained model files (generated)
│   └── intent_classifier.pkl      # Serialized classifier
├── requirements.txt               # Python dependencies
├── train.py                       # Main training script
└── chatbot.py                     # Main chatbot application
```

**C. Core Data Type Definitions (Starter Code)**

These foundational types will be used throughout the system:

```python
"""
File: src/utils/types.py
Core data type definitions for the chatbot system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime

class Intent:
    """Represents a user intent with its required slots."""
    
    def __init__(self, name: str, required_slots: List[str], optional_slots: List[str] = None):
        self.name = name  # e.g., "book_restaurant"
        self.required_slots = required_slots  # e.g., ["date", "time", "party_size"]
        self.optional_slots = optional_slots or []  # e.g., ["cuisine", "special_requests"]
        
    def __repr__(self):
        return f"Intent(name={self.name}, required={self.required_slots})"

@dataclass
class Entity:
    """Represents an extracted entity from user text."""
    
    type: str           # Entity category, e.g., "date", "person", "location"
    value: str          # Raw extracted text, e.g., "tomorrow", "John Smith"
    start: int          # Start index in original text
    end: int            # End index in original text
    confidence: float = 1.0  # Confidence score (1.0 for rule-based, <1.0 for ML)
    normalized: Any = None   # Normalized value, e.g., datetime object for dates
    
    def __repr__(self):
        return f"Entity(type={self.type}, value='{self.value}', normalized={self.normalized})"

@dataclass  
class UserUtterance:
    """Container for processed user input with analysis results."""
    
    raw_text: str                     # Original user message
    cleaned_text: str                 # Preprocessed text (lowercase, punctuation removed)
    intent: Optional[Intent] = None   # Detected intent (if any)
    confidence: float = 0.0           # Confidence score for intent
    entities: List[Entity] = field(default_factory=list)  # Extracted entities
    slots: Dict[str, Any] = field(default_factory=dict)   # Filled slots from entities
    
    def __repr__(self):
        return f"UserUtterance(text='{self.raw_text[:30]}...', intent={self.intent})"

class DialogState:
    """Tracks conversation state across multiple turns."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_intent: Optional[Intent] = None
        self.filled_slots: Dict[str, Any] = {}  # slot_name -> value
        self.missing_slots: List[str] = []      # Required slots not yet filled
        self.confirmation_required: bool = False
        self.last_activity: datetime = datetime.now()
        self.conversation_history: List[Dict] = []  # List of turn records
        
    def update_activity(self):
        """Update the last activity timestamp for session expiration."""
        self.last_activity = datetime.now()
        
    def is_expired(self, timeout_seconds: int = 300) -> bool:
        """Check if session has expired due to inactivity."""
        delta = datetime.now() - self.last_activity
        return delta.total_seconds() > timeout_seconds
        
    def reset(self):
        """Clear conversation context and return to initial state."""
        self.current_intent = None
        self.filled_slots.clear()
        self.missing_slots.clear()
        self.confirmation_required = False
        
    def __repr__(self):
        return (f"DialogState(session={self.session_id[:8]}..., "
                f"intent={self.current_intent}, "
                f"filled={len(self.filled_slots)} slots)")
```

**D. Key Implementation Principles**

Before diving into specific components, adhere to these principles:

1. **Separation of Concerns**: Each component (classifier, extractor, dialog manager, response generator) should have a single responsibility and communicate through well-defined interfaces.
2. **Data Immutability**: Process user utterances through a pipeline where each stage adds information without modifying previous stages' outputs.
3. **Graceful Degradation**: When components fail or return low-confidence results, provide helpful fallback responses rather than crashing or giving nonsense answers.
4. **Test-Driven Development**: Write tests for each component's behavior before implementing, especially for edge cases in entity extraction and dialog flows.

**E. Language-Specific Hints (Python)**

- Use `dataclasses` for simple data containers (like `Entity`, `DialogState`) to reduce boilerplate code.
- For JSON serialization of your data types, implement `to_dict()` and `from_dict()` methods or use libraries like `marshmallow`.
- Use Python's `enum.Enum` for fixed sets of values (like intent names or dialog states) to avoid string typos.
- For session storage, consider using `contextvars` for request-scoped data if your chatbot will handle multiple concurrent users.
- Install spaCy models with `python -m spacy download en_core_web_sm` and load them once at application startup.

---


## 2. Goals and Non-Goals

> **Milestone(s):** All Milestones (Foundational Requirements)

This section establishes the functional and non-functional requirements for the intent-based chatbot system, clearly defining the boundaries of what the system must accomplish and what lies outside its scope. Think of this as the **project's contract** — a clear agreement about what will be delivered to avoid misunderstandings later. For a junior developer, understanding these boundaries is crucial because it helps focus implementation effort on what truly matters while preventing endless feature creep.

### Functional Goals (What it MUST do)

The functional goals describe the core capabilities the chatbot must exhibit to be considered successful. These map directly to the four project milestones and represent the essential behaviors a user would expect when interacting with a basic but functional conversational assistant.

| Goal Category | Specific Capability | Description | Milestone Reference |
|---------------|-------------------|-------------|---------------------|
| **Input Understanding** | Intent Classification | The system must analyze the user's raw text input and assign it to one of several predefined intent categories (e.g., `book_flight`, `check_weather`, `set_reminder`). This classification must include a confidence score. | Milestone 1 |
| | Entity Extraction | The system must identify and extract specific pieces of structured information (entities) from the user's text, such as person names, dates, locations, times, and custom domain-specific values. | Milestone 2 |
| | Slot Filling | The system must map extracted entities to named parameters (slots) required by the detected intent. For example, a `book_flight` intent with required slots `destination` and `date` must have those slots populated from extracted entities. | Milestone 2 |
| **Conversation Management** | Multi-Turn Dialog | The system must maintain context across multiple message exchanges, allowing users to provide information piecemeal without restating their entire goal in each message. | Milestone 3 |
| | Slot Elicitation | When required slots are missing, the system must proactively ask the user for that specific information with clear, contextual prompts (e.g., "What date would you like to travel?"). | Milestone 3 |
| | Context Reset | The system must recognize a command (like "start over" or "cancel") to completely clear the current conversation context and return to an initial, idle state. | Milestone 3 |
| | Session Expiration | The system must automatically discard stale conversation contexts after a period of inactivity to prevent resource leaks and confusion from outdated information. | Milestone 3 |
| **Output Generation** | Template-Based Responses | The system must generate natural language responses by filling predefined template structures with context-specific values from filled slots and dialog state. | Milestone 4 |
| | Response Variation | For a given intent and state combination, the system must have multiple response templates and randomly select among them to make the conversation feel less robotic. | Milestone 4 |
| | Graceful Fallback | When the system cannot confidently understand user input (unknown intent or low confidence), it must provide a helpful, non-technical response that guides the user toward rephrasing or choosing a supported action. | Milestone 4 |
| **Data & Configuration** | Trainable Classifier | The system must allow for the intent classifier to be trained on new examples without modifying core code, using a simple, human-readable data format (e.g., YAML, JSON). | Milestone 1 |
| | Configurable Thresholds | Key operational parameters like confidence thresholds for intent classification and session timeout durations must be externally configurable, not hard-coded. | Milestone 1, 3 |
| | Custom Entity Patterns | The system must support the definition of custom, rule-based patterns (regular expressions or keyword lists) for extracting domain-specific entities not covered by general-purpose NER models. | Milestone 2 |

> **Key Insight:** The functional goals are **not** about implementing every possible feature, but about creating a **complete, end-to-end pipeline** from raw user input to contextually appropriate response. Each milestone builds upon the previous one, creating a vertical slice of functionality that works together.

**Detailed Behavioral Specifications:**

1. **For Intent Classification (Milestone 1):**
   - The system must accept a string of text (the `UserUtterance.raw_text`).
   - It must transform this text into a numerical representation using TF-IDF vectorization, based on a model trained from example utterances.
   - It must apply a statistical classification algorithm (Naive Bayes or SVM) to predict the most likely intent label.
   - It must output not just the label but also a probability score representing the classifier's confidence.
   - It must compare this confidence score against a configurable threshold (e.g., 0.7). If below threshold, the intent must be reported as `unknown` rather than the low-confidence label.
   - It must achieve at least 80% accuracy on a held-out test set of labeled examples that were not used during training.

2. **For Entity Extraction and Slot Filling (Milestone 2):**
   - The system must apply both rule-based patterns (for structured entities like dates, times, email addresses) and a statistical Named Entity Recognition (NER) model (for entities like person names, organizations, locations).
   - When multiple entities of the same type are found, the system must capture all of them.
   - Each extracted entity must be normalized to a canonical format where applicable (e.g., the date "tomorrow" → `2024-05-21`, time "2pm" → `14:00`).
   - The system must map extracted entities to the required and optional slots defined for the detected intent. For example, if the intent `book_restaurant` requires slots `cuisine` and `time`, and the user says "Italian food at 7," the system must map `Italian` → `cuisine` and `7pm` → `time`.
   - The system must identify which required slots remain empty after processing the current utterance.

3. **For Dialog Management (Milestone 3):**
   - The system must create and maintain a `DialogState` object for each unique conversation session, identified by a `session_id`.
   - This state must persist across HTTP requests or message turns within the same logical conversation.
   - The dialog manager must implement a state machine with at least the following states: `IDLE` (no active intent), `COLLECTING_SLOTS` (intent identified, gathering required information), `CONFIRMING` (asking user to verify extracted slot values), and `READY_TO_EXECUTE` (all information gathered, ready to trigger business logic).
   - On each user turn, the dialog manager must update the `DialogState.last_activity` timestamp.
   - A background process or check must be able to identify and clean up `DialogState` objects where `is_expired(timeout_seconds)` returns `True`.

4. **For Response Generation (Milestone 4):**
   - The system must store response templates as strings with named placeholders (e.g., `"Great! I've booked a flight to {destination} on {date}."`).
   - It must replace these placeholders with actual values from the `DialogState.filled_slots` dictionary.
   - It must select randomly from a list of alternative templates for the same dialog state to provide variety.
   - For unknown intents or ambiguous inputs, it must use a dedicated set of fallback templates (e.g., `"I'm not sure I understand. You can ask me to book a flight, check weather, or set a reminder."`).

### Non-Functional Goals (Quality Attributes)

Non-functional goals describe *how* the system should behave rather than *what* it does. These quality attributes are critical for the system's usability, maintainability, and robustness in a real-world setting.

| Quality Attribute | Target | Rationale & Measurement Approach |
|-------------------|--------|----------------------------------|
| **Accuracy** | ≥80% intent classification accuracy on held-out test data. | The primary measure of understanding correctness. Measured by training on 80% of labeled data, testing on 20%, and calculating (correct predictions / total predictions). |
| **Latency** | ≤200ms end-to-end response time per user message (excluding network). | Users expect conversational pace. Measured from receipt of `UserUtterance` to generation of final response text, averaged over many requests. |
| **Maintainability** | Clear separation of concerns, with each component having a single responsibility. Configuration (intents, entities, responses) should be external to code. | Enables developers to update business logic (new intents, response wording) without touching core algorithms. Measured by subjective code review and ease of adding a new intent. |
| **Extensibility** | New intent types can be added by providing only new training examples and configuration, not code changes. New entity types require only new pattern definitions or NER model updates. | Supports organic growth of the chatbot's capabilities. Measured by the time it takes a developer to add support for a completely new intent (e.g., `cancel_order`). |
| **Robustness** | The system must not crash on malformed input, unexpected Unicode characters, or extremely long messages. It should degrade gracefully (return `unknown` intent) rather than fail. | Real users provide noisy, unpredictable input. Measured through fuzz testing with random text inputs and monitoring for unhandled exceptions. |
| **Diagnosability** | All component transitions, classification decisions, and entity extractions should be logged with sufficient detail to trace a conversation's processing path. | Critical for debugging misunderstandings and improving training data. Measured by the ability to reconstruct why a specific response was given from logs alone. |
| **Resource Efficiency** | The system should operate within 512MB of RAM for typical workloads, loading ML models (like spaCy) once at startup. | Allows deployment on modest infrastructure (e.g., basic cloud instances, edge devices). Measured via memory profiling under simulated conversation load. |
| **Consistency** | Responses for the same intent and slot values should be semantically consistent, even with template variation. The chatbot's "personality" (tone, formality) should not shift randomly. | Builds user trust and avoids confusion. Measured by human evaluation of response sets for logical coherence. |

> **Design Principle:** **Worse is Better.** This project prioritizes **simplicity, correctness, and consistency** over handling every possible edge case with perfect sophistication. A system that correctly handles 80% of inputs predictably is more valuable than a complex system that handles 95% but fails mysteriously on the rest. This principle guides many of the architectural trade-offs.

**Architecture Decision Record: Prioritizing Accuracy Over Coverage**

> **Decision: Accept Higher Unknown Rate for Fewer Misclassifications**
> - **Context**: In intent classification, there is a fundamental trade-off between classifying everything (low unknown rate) and ensuring classifications are correct (high accuracy). Misclassifications (e.g., thinking "What's the weather?" is a request to book a flight) create frustrating user experiences.
> - **Options Considered**:
>     1. **Always choose the highest-probability intent** regardless of confidence. This minimizes `unknown` responses but leads to more incorrect actions.
>     2. **Use a dynamic, intent-specific threshold** based on class distribution. More complex, requires more tuning data.
>     3. **Use a fixed confidence threshold** (e.g., 0.7), below which the input is labeled `unknown`.
> - **Decision**: Option 3 — Fixed confidence threshold.
> - **Rationale**: For a beginner project and a system where **correctness is more important than coverage**, a simple, understandable rule is best. A fixed threshold is easy to explain, configure, and debug. While it may reject some valid but ambiguous inputs (increasing the `unknown` rate), it dramatically reduces catastrophic misclassifications. The fallback handler can then politely ask for clarification.
> - **Consequences**: The system will occasionally respond with "I don't understand" to inputs that a human could interpret, but it will rarely perform completely wrong actions. This builds user trust. The threshold becomes a key tuning parameter for system behavior.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Always choose top intent | Maximizes coverage, never says "I don't know" | High rate of incorrect actions, poor user experience | ❌ |
| Dynamic per-intent thresholds | Potentially better precision/recall balance per intent | Complex to implement and tune, requires lots of data, opaque to debug | ❌ |
| Fixed confidence threshold | Simple, predictable, easy to configure, reduces major errors | Some valid low-confidence inputs are rejected, requires fallback handling | ✅ |

### Explicit Non-Goals (What it does NOT do)

Clearly defining what the system will **not** do is equally important to prevent scope creep, manage expectations, and allow for focused development. For a junior developer, this list provides clear guardrails.

| Non-Goal Category | Specific Limitation | Reasoning & Alternative Approach |
|-------------------|---------------------|----------------------------------|
| **Language Understanding** | **General Conversation or "Chat"**: The system is not a general-purpose conversational partner. It cannot discuss arbitrary topics, tell jokes (unless explicitly programmed as an intent), or maintain open-ended dialogue. | **Scope Control**: Building a general chatbot requires vastly more complex technology (LLMs). This project focuses on **task-oriented dialogue** for specific, predefined goals. |
| | **Deep Semantic Understanding**: The system does not build a rich semantic model of the world. It doesn't understand that "New York" is a city in the United States or that "tomorrow" is relative to today's date beyond the simple normalization rule. | **Simplicity**: We rely on pattern matching and statistical correlations in text, not true understanding. Date normalization is done via libraries, not reasoning. |
| **Input Modality** | **Voice Input**: The system accepts only text input. It does not include speech-to-text capabilities. | **Focus on Core Logic**: Audio processing is a separate, complex domain. Text input allows focusing on dialog management and NLU. Voice could be added later via a separate service (e.g., Google Speech-to-Text). |
| | **Multimedia Input**: The system does not process images, videos, or files. | **Scope**: The project is about intent and entity extraction from *text*. |
| **Output Modality** | **Rich Media Responses**: Responses are plain text. The system does not generate images, audio, or structured UI elements (cards, buttons). | **Simplicity**: The response generator's concern is natural language. A front-end client could enrich text responses with UI, but that's outside the chatbot core. |
| **Learning & Adaptation** | **Online Learning**: The system does not learn from user interactions in real-time. The intent classifier and entity models are static after initial training and require an explicit retraining cycle with new data. | **Reliability & Safety**: Online learning can lead to unpredictable behavior and corruption from adversarial inputs. Offline retraining provides control and validation. |
| | **Personalization**: The system does not remember individual user preferences or history across different conversation sessions. | **Privacy & Complexity**: Session state is ephemeral and not linked to user identity. Adding personalization requires a persistent user profile store and significant design changes. |
| **System Integration** | **Executing Actual Actions**: The system identifies intents and slots but does not actually book flights, query weather APIs, or set calendar reminders. It prepares the structured request. | **Separation of Concerns**: The chatbot is a **dialogue interface**. It should output a structured command (e.g., `{"intent": "book_flight", "slots": {"destination": "NYC", "date": "2024-05-22"}}`) that a separate backend service executes. |
| **Scalability & Operations** | **High Availability / Fault Tolerance**: The reference implementation is not designed as a distributed, always-available service. It may use in-memory state stores that are lost on restart. | **Beginner-Friendly**: Production deployment concerns (databases, replication, load balancing) are out of scope for the initial learning objectives. The architecture should, however, allow swapping in persistent stores later. |
| | **Multi-User Concurrency**: While the system should handle multiple independent conversations, sophisticated locking or throughput optimization for thousands of simultaneous users is not a goal. | **Focus on Algorithmics**: The core challenge is natural language understanding, not massive-scale systems engineering. |

> **Critical Boundary:** The most important non-goal is **action execution**. The chatbot's job ends at **understanding and structuring the request**. It outputs a clear, structured representation of the user's goal (intent + filled slots). A separate "action executor" service (which is outside this project) would take that structure and actually call the flight booking API, weather service, etc. This separation keeps the chatbot focused and testable.

**Common Pitfall: Misinterpreting Non-Goals as Future Goals**

⚠️ **Pitfall: Treating the "Action Execution" boundary as optional.**
- **Description:** A developer might be tempted to directly call a weather API inside the dialog manager when a `check_weather` intent is detected with a filled `location` slot.
- **Why it's wrong:** This tightly couples the dialog logic to specific external services, making the system brittle and hard to test. It also mixes concerns — the dialog manager should manage conversation state, not perform business operations.
- **How to fix:** Adhere strictly to the separation. The dialog manager, upon reaching a `READY_TO_EXECUTE` state, should pass the completed `DialogState` (or just the intent and slots) to a designated output channel (e.g., a message queue, a callback function). A separate, independently deployable service listens to that channel and performs the actual action. In a simple learning project, this could be as straightforward as printing the structured command to the console.

### Implementation Guidance

#### A. Technology Recommendations Table

| Component | Simple Option (Beginner-Friendly) | Advanced Option (More Robust) |
|-----------|-----------------------------------|-------------------------------|
| **Intent Classifier** | `scikit-learn` Pipeline with `TfidfVectorizer` and `MultinomialNB`/`SVC` | Same, but with `GridSearchCV` for hyperparameter tuning or `fasttext` for subword embeddings |
| **Entity Extraction** | `spaCy` (small model) for NER + `regex` library for patterns | `spaCy` with custom model training, or `Stanza`/`flair` for more entity types |
| **Dialog State Storage** | In-memory dictionary keyed by `session_id` (Python `dict`) | External store: `Redis` (fast, persistent), `SQLite` (simple file-based) |
| **Response Templates** | Python f-strings or `str.format()` with a dictionary of templates | Jinja2 templating engine for more complex logic and inheritance |
| **Web Framework (if needed)** | `Flask` or `FastAPI` for simple HTTP endpoints | Same, but with async support (`FastAPI`) for better concurrency |
| **Configuration** | Python `.py` files or `.env` for thresholds, timeouts | YAML/JSON config files loaded via `pyyaml` or `json` |

#### B. Recommended File/Module Structure

For the goals defined, here's how the project directory should be organized to support clean separation of concerns and configuration:

```
intent_bot_project/
├── config/
│   ├── __init__.py
│   ├── settings.py           # Central config: thresholds, timeout_seconds, file paths
│   ├── intents.yaml          # Intent definitions: name, required_slots, optional_slots
│   └── responses.yaml        # Response templates per intent/state
├── data/
│   ├── training/
│   │   └── nlu.yml           # Training examples in Rasa-style format or simple CSV
│   └── models/               # Serialized sklearn model, spaCy pipeline cache
├── src/
│   ├── __init__.py
│   ├── nlu/                  # Natural Language Understanding
│   │   ├── __init__.py
│   │   ├── intent_classifier.py  # IntentClassifier class (Milestone 1)
│   │   └── entity_extractor.py   # EntityExtractor class (Milestone 2)
│   ├── dialog/               # Dialog Management
│   │   ├── __init__.py
│   │   ├── dialog_manager.py     # DialogManager & DialogState classes (Milestone 3)
│   │   └── session_store.py      # Abstract base & in-memory impl for session storage
│   ├── nlg/                  # Natural Language Generation
│   │   ├── __init__.py
│   │   └── response_generator.py # ResponseGenerator class (Milestone 4)
│   └── bot.py                # Main orchestrator class tying components together
├── tests/                    # Unit and integration tests
├── scripts/
│   ├── train_intent_classifier.py
│   └── evaluate_model.py
├── requirements.txt
└── README.md
```

#### C. Infrastructure Starter Code

**Core Configuration Loader (`config/settings.py`):**
```python
import os
from typing import Dict, Any
import yaml

class Settings:
    """Central configuration for the chatbot."""
    
    def __init__(self):
        # Default values (can be overridden by environment variables or config files)
        self.INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_THRESHOLD", "0.7"))
        self.SESSION_TIMEOUT_SECONDS = int(os.getenv("SESSION_TIMEOUT", "300"))  # 5 minutes
        self.MODEL_PATH = os.getenv("MODEL_PATH", "./data/models/intent_model.pkl")
        self.SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
        
        # Load intent and response configurations
        self.intents = self._load_yaml("./config/intents.yaml")
        self.responses = self._load_yaml("./config/responses.yaml")
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load and parse a YAML configuration file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {path} not found. Using empty dict.")
            return {}

# Global settings instance
settings = Settings()
```

#### D. Core Logic Skeleton Code

**Main Bot Orchestrator (`src/bot.py`):**
```python
from typing import Optional, Dict, Any
from src.nlu.intent_classifier import IntentClassifier
from src.nlu.entity_extractor import EntityExtractor
from src.dialog.dialog_manager import DialogManager
from src.nlg.response_generator import ResponseGenerator
from config.settings import settings

class ChatBot:
    """
    Main orchestrator that ties all components together.
    Follows the processing pipeline: Classify → Extract → Manage Dialog → Generate Response.
    """
    
    def __init__(self):
        self.intent_classifier = IntentClassifier(model_path=settings.MODEL_PATH)
        self.entity_extractor = EntityExtractor(spacy_model=settings.SPACY_MODEL)
        self.dialog_manager = DialogManager(timeout_seconds=settings.SESSION_TIMEOUT_SECONDS)
        self.response_generator = ResponseGenerator(templates=settings.responses)
    
    def process_message(self, user_input: str, session_id: str) -> str:
        """
        Process a single user message and return the bot's response.
        
        Args:
            user_input: Raw text from the user
            session_id: Unique identifier for the conversation session
            
        Returns:
            Bot's response text
        """
        # TODO 1: Retrieve or create DialogState for the given session_id
        # TODO 2: Update the DialogState's last_activity timestamp
        # TODO 3: Clean/preprocess the user_input (lowercase, remove extra spaces)
        # TODO 4: Use intent_classifier to predict intent and confidence
        # TODO 5: If confidence < threshold, set intent to None (unknown)
        # TODO 6: Use entity_extractor to extract entities from the cleaned text
        # TODO 7: Pass intent, entities, and current DialogState to dialog_manager.process_turn()
        #   This updates the DialogState (fills slots, changes state, etc.)
        # TODO 8: Pass the updated DialogState to response_generator.generate()
        # TODO 9: Return the generated response string
        # TODO 10: Handle any exceptions (e.g., classifier not trained) with appropriate fallback
        pass
```

**Intent Configuration Structure (`config/intents.yaml`):**
```yaml
intents:
  greet:
    required_slots: []
    optional_slots: []
  book_flight:
    required_slots: ["destination", "departure_date"]
    optional_slots: ["return_date", "passenger_count"]
  check_weather:
    required_slots: ["location"]
    optional_slots: ["date"]
  set_reminder:
    required_slots: ["reminder_text", "reminder_time"]
    optional_slots: []
  goodbye:
    required_slots: []
    optional_slots: []
```

#### E. Python-Specific Hints

1. **TF-IDF Vectorizer Persistence**: Use `joblib` or `pickle` to save and load your trained `scikit-learn` pipeline. Remember to fit the vectorizer only on training data, not on live user inputs.
   ```python
   import joblib
   # Save
   joblib.dump(pipeline, 'intent_model.pkl')
   # Load
   pipeline = joblib.load('intent_model.pkl')
   ```

2. **spaCy Model Loading**: The first run requires downloading the model: `python -m spacy download en_core_web_sm`. In code, load it once and reuse:
   ```python
   import spacy
   nlp = spacy.load("en_core_web_sm")
   ```

3. **Session Expiration Cleanup**: Use a simple periodic task or check on each access. For a simple implementation:
   ```python
   class SimpleSessionStore:
       def __init__(self, timeout_seconds: int):
           self.sessions: Dict[str, DialogState] = {}
           self.timeout = timeout_seconds
       
       def get(self, session_id: str) -> Optional[DialogState]:
           state = self.sessions.get(session_id)
           if state and state.is_expired(self.timeout):
               del self.sessions[session_id]
               return None
           return state
   ```

#### F. Milestone Checkpoint

After completing the design for this section (before any coding), you should be able to:

1. **Define Success Criteria**: Write down 3-5 concrete, testable acceptance criteria for each milestone based on the functional goals above.
   *Example for Milestone 1:* "Given the training data in `data/training/nlu.yml`, when I run `scripts/train_intent_classifier.py`, a model file is created at `data/models/intent_model.pkl`."

2. **Create Configuration Files**: Create the `config/intents.yaml` file with at least 3 intents, each with required/optional slots. Create `config/responses.yaml` with at least one template for each intent.

3. **Outline the Pipeline**: Draw a simple flowchart on paper showing the data flow from user input to bot response, labeling each step with the component responsible and the data structures passed between them.

#### G. Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Chatbot responds "I don't understand" to everything | Intent classifier confidence threshold too high, or model not trained/loaded | Check logs for confidence scores. Verify model file exists and is loaded correctly. | Lower `INTENT_CONFIDENCE_THRESHOLD` in config. Ensure training script ran successfully. |
| Entities not being extracted | spaCy model not installed or regex patterns incorrect | Print the output of `entity_extractor.extract("test with New York and tomorrow")`. Check for import errors. | Run `python -m spacy download en_core_web_sm`. Verify regex patterns match your test strings. |
| Conversation context lost between messages | Session ID not being passed consistently, or session store not implemented | Print `session_id` at start of `process_message`. Check if `DialogManager.get_state` returns a new state each time. | Ensure your client sends the same session ID for a conversation. Implement a session store that persists between requests. |
| Response has `{placeholder}` in output | Template variable substitution failed | Check that the slot name in the template matches a key in `DialogState.filled_slots`. Print both structures. | Ensure template uses same slot names as intent definition. Add default values for missing slots in template rendering. |


## 3. High-Level Architecture

> **Milestone(s):** All Milestones (Architectural Foundation)

This section provides a bird's-eye view of the intent-based chatbot system. We'll break down the architecture into four core components, explain their responsibilities and interactions, and establish the project's organizational structure. The primary challenge is designing a **modular pipeline** where each component has a single, well-defined responsibility, allowing for independent development, testing, and replacement. Think of this as an **assembly line for conversations**: user input enters at one end, passes through specialized stations that each perform a specific transformation, and exits as a coherent response.

The architecture follows a **linear processing pipeline with stateful conversation management**. Data flows sequentially through the components for each user turn, but the `DialogManager` introduces a feedback loop by maintaining context across turns. This design separates the stateless understanding components (classifier, extractor) from the stateful orchestration component, simplifying scalability and testing.

### Component Responsibilities and Data Flow

The system decomposes into four primary components, each embodying a distinct phase of conversational understanding and response generation. The following diagram illustrates their relationships and the primary data flow:

![System Component Diagram](./diagrams/sys-component-diagram.svg)

**Core Component Responsibilities:**

| Component | Primary Analogy | Core Responsibility | Key Inputs | Key Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **Intent Classifier** | The Sorting Hat | Categorizes the user's raw text into a predefined intent (goal). | Raw user text (`str`) | `Intent` object with a confidence score. |
| **Entity Extractor & Slot Filler** | The Form Filler | Identifies and extracts specific pieces of structured information (entities) from the text and maps them to the intent's required parameters (slots). | Raw user text, detected `Intent` | List of `Entity` objects, updated `UserUtterance.slots` dictionary. |
| **Dialog Manager** | The Conversation Conductor | Maintains the conversation state across multiple turns, orchestrates slot filling, and determines the next system action (e.g., ask for missing info, confirm, respond). | `UserUtterance` (with intent & slots), current `DialogState` | Updated `DialogState`, next `Action` (e.g., `PROMPT_FOR_SLOT`, `GENERATE_RESPONSE`). |
| **Response Generator** | The Mad Libs Assembler | Constructs a natural language response by filling predefined templates with values from the dialog state and extracted entities. | `DialogState`, `Action` | Natural language response text (`str`). |

**End-to-End Data Flow (Single Turn):**

The processing of a single user message follows a deterministic sequence. The following steps describe the "happy path" where an intent is confidently recognized and all required information is present.

1.  **Input Reception:** The system receives a raw text string from the user (e.g., "Book a table for two at 7pm tonight").
2.  **Intent Classification:** The `Intent Classifier` takes the raw text.
    1.  It cleans and vectorizes the text (e.g., using TF-IDF).
    2.  It predicts the most likely intent label and a confidence score.
    3.  If the confidence is above `INTENT_CONFIDENCE_THRESHOLD` (e.g., 0.7), the intent is accepted. Otherwise, the utterance is tagged with an `"unknown"` intent.
    4.  It creates an `Intent` object (e.g., `name="book_restaurant"`) and attaches it to a new `UserUtterance` object.
3.  **Entity Extraction & Slot Filling:** The `Entity Extractor` processes the same raw text and the detected intent.
    1.  It runs rule-based patterns (regex) and/or a statistical NER model (spaCy) to identify entity spans (e.g., `"two"` → `{"type": "number", "value": "2"}`; `"7pm tonight"` → `{"type": "datetime", "value": "2023-10-27 19:00:00"}`).
    2.  The `Slot Filler` maps these entities to the slot names defined for the intent (e.g., `number` → `party_size`, `datetime` → `reservation_time`). It updates the `UserUtterance.slots` dictionary.
4.  **Dialog State Management:** The `Dialog Manager` receives the enriched `UserUtterance`.
    1.  It retrieves or creates the `DialogState` for the user's session (using `session_id`).
    2.  It updates the state: sets the `current_intent`, merges new `filled_slots` from the utterance, and recalculates `missing_slots` by comparing `filled_slots` with the intent's `required_slots`.
    3.  Based on the updated state, it decides the next action. If slots are missing, the action is `PROMPT_FOR_SLOT`. If all required slots are filled, the action is `GENERATE_RESPONSE`.
5.  **Response Generation:** The `Response Generator` is invoked with the `DialogState` and the chosen `Action`.
    1.  It selects an appropriate response template based on the intent and action (e.g., a `book_restaurant` confirmation template vs. a prompt template asking for `party_size`).
    2.  It performs variable substitution, replacing placeholders like `{party_size}` and `{reservation_time}` with values from `DialogState.filled_slots`.
    3.  It returns the finalized natural language string (e.g., "Great! I've booked a table for 2 at 7:00 PM.").
6.  **Output:** The generated response is sent back to the user, completing the turn.

**State Persistence Across Turns:**
The `DialogState` object is the glue that enables multi-turn conversations. It is persisted in a session store (in-memory for this project) between user turns. When a follow-up message arrives (e.g., "Make it for 4 people instead"), the `DialogManager` loads the existing state. The new utterance is processed, and its extracted slots (e.g., `party_size: 4`) are merged into the existing `filled_slots`, allowing the conversation to continue contextually without the user restating their entire request.

> **Design Insight:** The clean separation between the `Intent Classifier` and `Entity Extractor` is a key architectural feature. It allows you to improve entity recognition (e.g., by adding a new regex pattern) without retraining the intent model, and vice-versa. This modularity significantly eases maintenance and iteration.

**ADR: Modular Pipeline vs. Monolithic Model**

> **Decision: Choose a modular pipeline of specialized components over a single end-to-end neural model.**
>
> - **Context:** We need a system that is interpretable, maintainable by beginners, and trainable with small datasets. While modern LLMs can handle intent, entity, and dialog in one model, they require massive data and computational resources.
> - **Options Considered:**
>     1.  **Monolithic End-to-End Model:** A single neural network (e.g., a fine-tuned transformer) that takes text and outputs intent, entities, and dialog actions.
>     2.  **Modular Pipeline:** Separate, specialized components (classifier, extractor, manager) connected in a processing chain.
> - **Decision:** We chose the **Modular Pipeline**.
> - **Rationale:** This approach aligns perfectly with our project constraints and learning goals. (1) **Interpretability:** It's easy to debug—you can inspect the output of each component. (2) **Data Efficiency:** Each component (like the TF-IDF classifier) can be trained effectively with just a few dozen examples per intent. (3) **Incremental Development:** It maps directly to our milestones, allowing for step-by-step implementation and testing. (4) **Educational Value:** It teaches fundamental NLP concepts (text classification, NER, state machines) in isolation.
> - **Consequences:** We gain transparency and ease of development but must explicitly design the interfaces (data structures) between components. We also accept that errors can cascade (e.g., a misclassified intent will misguide entity extraction), necessitating robust error handling in the `DialogManager`.

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **Monolithic Model** | Potentially higher accuracy with enough data; fewer handoffs. | Opaque "black box"; requires large datasets & GPU; hard to debug for beginners. | No |
| **Modular Pipeline** | Highly interpretable; data-efficient; matches milestone structure; easy to test components in isolation. | Requires careful interface design; errors can propagate. | **Yes** |

### Recommended File and Module Structure

A well-organized codebase is critical for managing the complexity of a multi-component system. The recommended structure groups files by **component responsibility** and separates **configuration**, **core logic**, **data models**, and **utilities**. This follows the principle of separation of concerns and makes the project navigable.

**Project Root Layout:**
```
intent_bot_project/
├── bot/                          # Core chatbot package
│   ├── __init__.py
│   ├── data_models.py            # Core type definitions: UserUtterance, Intent, Entity, DialogState
│   ├── intent_classifier.py      # Milestone 1: TF-IDF vectorizer, classifier, threshold logic
│   ├── entity_extractor.py       # Milestone 2: NER, regex patterns, slot filling
│   ├── dialog_manager.py         # Milestone 3: State machine, session management
│   ├── response_generator.py     # Milestone 4: Template loading, variable substitution
│   └── pipeline.py               # Orchestrates the component chain for a single turn
├── training/                     # Training data & model persistence
│   ├── intents.yml               # Intent definitions and example utterances
│   ├── patterns/                 # Regex patterns for entity extraction
│   │   ├── datetime_patterns.json
│   │   └── ...
│   └── models/                   # Saved sklearn/spaCy models
│       ├── intent_classifier.pkl
│       └── tfidf_vectorizer.pkl
├── config.py                     # Constants: INTENT_CONFIDENCE_THRESHOLD, SESSION_TIMEOUT_SECONDS
├── sessions.py                   # In-memory session store (simple dict)
├── app.py                        # Main application entry point (CLI or web server)
└── tests/                        # Comprehensive test suite
    ├── test_intent_classifier.py
    ├── test_entity_extractor.py
    ├── test_dialog_manager.py
    ├── test_response_generator.py
    └── test_integration.py
```

**Module Dependencies and Imports:**
The dependency flow is linear and acyclic, which prevents circular imports and enforces architectural boundaries. The following table outlines the key imports:

| Module | Imports From | Purpose |
| :--- | :--- | :--- |
| `bot.data_models` | (None - pure data classes) | Defines the common language (`UserUtterance`, `DialogState`) for the entire system. |
| `bot.intent_classifier` | `bot.data_models`, `config` | Uses the `Intent` type and the confidence threshold constant. |
| `bot.entity_extractor` | `bot.data_models` | Creates `Entity` objects and populates `UserUtterance.slots`. |
| `bot.dialog_manager` | `bot.data_models`, `sessions`, `config` | Manages `DialogState` objects, uses the session store, and checks timeout. |
| `bot.response_generator` | `bot.data_models` | Reads values from `DialogState` to fill templates. |
| `bot.pipeline` | All other `bot.*` modules | Composes the components into the full processing pipeline. |
| `app.py` | `bot.pipeline`, `sessions` | Glues everything together to handle user interactions. |

This structure ensures that:
1.  **Data models are the central contract.** All components agree on the shape of `UserUtterance` and `DialogState`.
2.  **Components are isolated.** You can refactor the `IntentClassifier` internals without touching the `DialogManager`.
3.  **Configuration is centralized.** Changing the confidence threshold only requires an edit in `config.py`.
4.  **Testing is straightforward.** Each component can be tested in isolation by mocking its dependencies.

### Implementation Guidance

This section provides concrete Python code to set up the project skeleton and define the core data contracts. The goal is to provide a foundation you can build upon in subsequent milestones.

**A. Technology Recommendations Table**

| Component | Simple Option (Beginner-Friendly) | Advanced Option (More Robust) |
| :--- | :--- | :--- |
| **Text Vectorization & Intent Model** | `scikit-learn` (`TfidfVectorizer`, `MultinomialNB` or `SGDClassifier`) | `spaCy` (text preprocessing + own classifier) or `sentence-transformers` (dense embeddings) |
| **Entity Extraction** | `spaCy` (small pre-trained model) + `regex` library | Custom CRF model or fine-tuned `spaCy` NER pipeline |
| **Session Storage** | In-memory Python `dict` with `session_id` keys | External store (Redis, SQLite) for persistence across server restarts |
| **Response Templating** | Python `str.format()` or simple dictionary lookup | Jinja2 templating engine for complex logic |
| **Application Interface** | Command-line interface (CLI) | Web server (FastAPI/Flask) with REST/WebSocket endpoints |

**B. Recommended File/Module Structure (Detailed Starter)**

We'll create the foundational files. First, ensure your project has a virtual environment and install core dependencies:
```bash
pip install scikit-learn spacy regex
python -m spacy download en_core_web_sm  # Small English model for entity extraction
```

Now, create the following files with the provided starter code.

**File: `bot/data_models.py`**
This file defines the **single source of truth** for all data flowing through your system. Using Python's `dataclasses` or Pydantic models is recommended for clarity and validation.

```python
"""
Core data models for the intent-based chatbot.
These classes define the structure of information passed between components.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Intent:
    """Represents a user's goal or purpose."""
    name: str
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)

    # Example usage: Intent("book_restaurant", ["party_size", "reservation_time"], ["location"])


@dataclass
class Entity:
    """Represents a specific piece of information extracted from text."""
    type: str          # e.g., "person", "date", "number"
    value: str         # The raw text extracted
    start: int         # Start character index in the original text
    end: int           # End character index (exclusive)
    confidence: float = 1.0  # Useful for ML-based extraction
    normalized: Any = None   # Canonical form, e.g., datetime object for "tomorrow at 5pm"

    # Example: Entity("number", "two", start=15, end=18, normalized=2)


@dataclass
class UserUtterance:
    """Represents a processed user message with understanding results."""
    raw_text: str
    cleaned_text: str = ""  # After preprocessing (lowercase, remove punctuation)
    intent: Optional[Intent] = None
    confidence: float = 0.0
    entities: List[Entity] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)  # Map slot name -> extracted value

    def update_slots_from_entities(self, slot_mapping: Dict[str, str]):
        """
        Helper to fill the slots dict using extracted entities.
        slot_mapping maps entity types to slot names (e.g., {"number": "party_size"}).
        """
        for entity in self.entities:
            if entity.type in slot_mapping:
                slot_name = slot_mapping[entity.type]
                self.slots[slot_name] = entity.normalized or entity.value


@dataclass
class DialogState:
    """Tracks the state of an ongoing conversation for a single user/session."""
    session_id: str
    current_intent: Optional[Intent] = None
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    confirmation_required: bool = False  # Whether we need user confirmation before proceeding
    last_activity: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict] = field(default_factory=list)  # Log of turns

    def update_activity(self):
        """Updates last_activity timestamp to now. Call on every user interaction."""
        self.last_activity = datetime.now()

    def is_expired(self, timeout_seconds: int) -> bool:
        """Checks if the session has expired due to inactivity."""
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed > timeout_seconds

    def reset(self):
        """Clears conversation context and returns to initial state."""
        self.current_intent = None
        self.filled_slots.clear()
        self.missing_slots.clear()
        self.confirmation_required = False
        self.update_activity()
        self.conversation_history.append({"action": "reset", "timestamp": self.last_activity})
```

**File: `config.py`**
Centralize your tunable parameters here. This makes experimentation and configuration management easy.

```python
"""
Configuration constants for the chatbot.
"""

# Intent Classification
INTENT_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to accept a predicted intent

# Session Management
SESSION_TIMEOUT_SECONDS = 300      # 5 minutes of inactivity before session reset

# File Paths (Adjust as needed)
INTENTS_DATA_PATH = "training/intents.yml"
PATTERNS_DIR = "training/patterns/"
MODELS_DIR = "training/models/"
```

**File: `sessions.py`**
A simple, in-memory session store. In a production system, this would be replaced with a persistent database.

```python
"""
In-memory session store for DialogState objects.
"""
from typing import Dict
from bot.data_models import DialogState
import config

# Global in-memory store. Key: session_id, Value: DialogState
_session_store: Dict[str, DialogState] = {}


def get_session(session_id: str) -> DialogState:
    """
    Retrieve or create a DialogState for the given session_id.
    Also performs session expiration cleanup.
    """
    global _session_store

    # Cleanup expired sessions (simple, inefficient for large stores)
    expired_ids = [
        sid for sid, state in _session_store.items()
        if state.is_expired(config.SESSION_TIMEOUT_SECONDS)
    ]
    for sid in expired_ids:
        del _session_store[sid]

    # Return existing or create new
    if session_id not in _session_store:
        _session_store[session_id] = DialogState(session_id=session_id)
    return _session_store[session_id]


def save_session(state: DialogState):
    """Updates the session store with the current state."""
    _session_store[state.session_id] = state
```

**File: `bot/pipeline.py`**
This is the orchestrator. It wires the components together in the correct order. We'll provide the skeleton; you will fill in the component logic in later milestones.

```python
"""
Main processing pipeline that orchestrates the flow between components.
"""
from typing import Tuple
from bot.data_models import UserUtterance, DialogState
from bot.intent_classifier import IntentClassifier
from bot.entity_extractor import EntityExtractor
from bot.dialog_manager import DialogManager
from bot.response_generator import ResponseGenerator
import sessions


class ChatbotPipeline:
    """
    The core pipeline that processes a user message and returns a response.
    This class follows the steps outlined in the Component Responsibilities section.
    """

    def __init__(self):
        # TODO: Initialize your components here.
        # These will be populated in later milestones.
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.dialog_manager = DialogManager()
        self.response_generator = ResponseGenerator()

    def process_message(self, session_id: str, user_message: str) -> Tuple[str, DialogState]:
        """
        The main entry point for processing a user message.
        Returns: (response_text, updated_dialog_state)
        """
        # Step 1: Create initial UserUtterance
        utterance = UserUtterance(raw_text=user_message)

        # Step 2: Classify Intent (Milestone 1)
        # TODO: Call intent_classifier.classify(utterance)
        # The classifier should set utterance.intent and utterance.confidence
        utterance = self.intent_classifier.classify(utterance)

        # Step 3: Extract Entities & Fill Slots (Milestone 2)
        # TODO: Call entity_extractor.extract(utterance)
        # The extractor should populate utterance.entities and utterance.slots
        utterance = self.entity_extractor.extract(utterance)

        # Step 4: Manage Dialog State (Milestone 3)
        # TODO: Retrieve session state, update it with the new utterance, decide next action
        dialog_state = sessions.get_session(session_id)
        next_action = self.dialog_manager.handle_utterance(utterance, dialog_state)
        sessions.save_session(dialog_state)

        # Step 5: Generate Response (Milestone 4)
        # TODO: Call response_generator.generate(dialog_state, next_action)
        response_text = self.response_generator.generate(dialog_state, next_action)

        return response_text, dialog_state
```

**File: `app.py` (CLI Version)**
A simple command-line interface to test your pipeline as you build it.

```python
"""
Simple CLI application for testing the chatbot pipeline.
"""
import uuid
from bot.pipeline import ChatbotPipeline

def main():
    pipeline = ChatbotPipeline()
    session_id = str(uuid.uuid4())[:8]  # Generate a short session ID
    print("Chatbot CLI Started. Type 'quit' to exit, 'reset' to clear session.")
    print(f"Session ID: {session_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'reset':
            session_id = str(uuid.uuid4())[:8]
            print(f"[Session reset. New Session ID: {session_id}]")
            continue

        # Process the message
        response, state = pipeline.process_message(session_id, user_input)
        print(f"Bot: {response}")
        # Optional: Print debug info about dialog state
        # print(f"[Debug] Intent: {state.current_intent.name if state.current_intent else None}")
        # print(f"[Debug] Filled Slots: {state.filled_slots}")

    print("Goodbye!")

if __name__ == "__main__":
    main()
```

**C. Infrastructure Starter Code**

The code above for `data_models.py`, `config.py`, `sessions.py`, and the skeleton `pipeline.py` and `app.py` provides a complete, runnable foundation. You can copy these files directly into your project structure.

**D. Core Logic Skeleton Code**

The `ChatbotPipeline.process_message` method in `pipeline.py` already outlines the core sequence with TODOs. Each TODO corresponds to a milestone. As you complete each milestone, you will implement the corresponding component (`IntentClassifier.classify`, `EntityExtractor.extract`, etc.) and remove the TODOs.

**E. Language-Specific Hints (Python)**

*   **Data Classes:** Use `@dataclass` from the `dataclasses` module (as shown) for clean, boilerplate-free model definitions. They automatically generate `__init__`, `__repr__`, and comparison methods.
*   **Type Hints:** Always use type hints (`str`, `List[Entity]`, `Optional[Intent]`). This serves as documentation and helps catch errors with tools like `mypy`.
*   **Path Management:** Use `pathlib.Path` for file operations. It's more readable and cross-platform than string concatenation.
*   **Virtual Environment:** Always develop inside a virtual environment (`venv` or `conda`) to manage dependencies.

**F. Milestone Checkpoint (Architecture Setup)**

After creating the files above, you should be able to run the skeleton application:
```bash
python app.py
```
**Expected Output:** The program should start, print a session ID, and prompt you with "You:". It will not understand any input yet (all components are stubs), but it should not crash. Typing "quit" should exit cleanly.

**Signs of Success:**
*   No `ModuleNotFoundError` when running `app.py`.
*   The `bot/` directory exists with `data_models.py` and `pipeline.py`.
*   You can import the data models in a Python shell: `from bot.data_models import Intent, UserUtterance`.

**If something is wrong:**
1.  Check your project structure matches the layout exactly.
2.  Ensure all `__init__.py` files are present (they can be empty).
3.  Verify your Python interpreter is using the virtual environment where you installed `scikit-learn` and `spacy`.

**G. Debugging Tips**

| Symptom | Likely Cause | How to Diagnose | Fix |
| :--- | :--- | :--- | :--- |
| `ImportError: cannot import name 'Intent'` | Circular import or incorrect module path. | Check `bot/__init__.py` exists. Check import statements in `pipeline.py`. | Ensure imports follow the structure: `from bot.data_models import Intent`. |
| App crashes immediately on start | Missing dependencies or syntax error in starter code. | Look at the traceback. Run `python -m py_compile bot/data_models.py` to check for syntax errors. | Install required packages (`pip install scikit-learn spacy`). Fix any typos in the provided code. |
| Changes to `config.py` don't take effect | Python module caching. | Print the constant value at runtime. | Restart your Python interpreter/application. |

---


## 4. Data Model

> **Milestone(s):** Milestone 1, Milestone 2, Milestone 3, Milestone 4 (Foundational Data Structures)

The data model defines the foundational vocabulary and relationships that enable our chatbot to understand, process, and respond to conversations. Think of it as the chatbot's **mental filing cabinet** — a structured system where raw user input gets analyzed, categorized, and stored in labeled folders (intents) with sticky notes (entities) attached, all tracked within a conversation folder (dialog state). This systematic organization transforms chaotic natural language into structured, actionable data that the system's components can process.

### Core Type Definitions and Relationships

The chatbot operates on five core data types that flow through the processing pipeline. Each type serves a distinct purpose in the conversational lifecycle, from raw input to structured understanding.

![Data Model Relationship Diagram](./diagrams/data-model-diagram.svg)

#### UserUtterance: The Raw Input Container
The `UserUtterance` represents a single user message after it enters the system. Think of it as an **incoming mail envelope** that gets opened, analyzed, and annotated. Initially, it contains just the raw text, but as it progresses through the pipeline, it accumulates analysis results: the cleaned text, the detected intent (with confidence), extracted entities, and finally, a structured representation of the information in named slots.

| Field | Type | Description |
|-------|------|-------------|
| `raw_text` | `str` | The exact text as received from the user, preserving original casing, punctuation, and any typos. Useful for entity extraction where character positions matter. |
| `cleaned_text` | `str` | The normalized version of the text after preprocessing (lowercasing, removing special characters, etc.). Used primarily for intent classification. |
| `intent` | `Optional[Intent]` | The intent object representing the user's goal, as identified by the intent classifier. Will be `None` if classification confidence is below the threshold. |
| `confidence` | `float` | The classifier's confidence score (0.0 to 1.0) for the predicted intent. Used to determine if the prediction is reliable enough to proceed. |
| `entities` | `List[Entity]` | A list of structured pieces of information extracted from the text, such as dates, names, or numbers. Each entity captures its type, value, and position in the original text. |
| `slots` | `Dict[str, Any]` | A dictionary mapping slot names (parameters the intent requires) to their values, populated from extracted entities. This is the final, structured representation of the information in the utterance. |

The `UserUtterance` has a key helper method:
- `update_slots_from_entities(slot_mapping)`: This method takes a mapping dictionary (which defines how entity types correspond to slot names for the current intent) and populates the `slots` dictionary. For example, if the slot mapping says `{"date": "departure_date"}`, and an entity of type `date` with value "tomorrow" is found, it will set `slots["departure_date"] = "tomorrow"`.

#### Intent: The Goal Blueprint
An `Intent` represents a category of user goals, like booking a flight or checking the weather. Think of it as a **form template** that defines what information (slots) needs to be collected to fulfill that goal. Each intent has a unique name and specifies which slots are mandatory versus optional.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | A unique, descriptive identifier for the intent (e.g., `"book_flight"`, `"check_weather"`). This is the label the classifier predicts. |
| `required_slots` | `List[str]` | A list of slot names that must be filled for the intent to be actionable. The dialog manager will prompt the user for any missing required slots. |
| `optional_slots` | `List[str]` | A list of slot names that provide additional context but are not strictly necessary to execute the intent (e.g., `"seat_preference"` for flight booking). |

The `Intent` object is typically instantiated from a configuration file or database when the system starts. For example, `Intent("book_flight", ["destination", "departure_date"], ["seat_preference"])` defines a flight booking intent that requires a destination and date, with an optional seat preference.

#### Entity: The Information Fragment
An `Entity` represents a specific piece of information extracted from the user's text. Think of it as a **highlighted phrase with metadata** — the system identifies a span of text, classifies its type, and may normalize it to a standard format. This structured extraction is crucial for converting free text into machine-readable data.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `str` | The category of the entity (e.g., `"date"`, `"person"`, `"location"`, `"custom_product_code"`). Determines which slot this entity can fill. |
| `value` | `str` | The raw text substring that was identified as the entity, exactly as it appears in the `raw_text`. |
| `start` | `int` | The character index in `raw_text` where the entity begins (inclusive). Used for debugging and to avoid overlapping extractions. |
| `end` | `int` | The character index in `raw_text` where the entity ends (exclusive). Together with `start`, defines the exact text span. |
| `confidence` | `float` | The extractor's confidence in this identification (0.0 to 1.0). Useful for ranking multiple possible extractions. |
| `normalized` | `Any` | A canonical, structured representation of the entity value. For a date, this might be a `datetime.date` object; for a number, a float. May be `None` if no normalization is applied. |

Entities are produced by the entity extractor and later consumed by the slot filler, which maps them to the slots defined in the intent.

#### DialogState: The Conversation Memory
The `DialogState` is the **conversational scratchpad** that persists across multiple turns. It tracks the current context of the interaction: what the user is trying to do, what information has been gathered so far, and what's still missing. Without this state, every user message would be treated as an isolated, new request.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | A unique identifier for the conversation session, typically generated when a user starts interacting. Used to retrieve the correct state from the session store. |
| `current_intent` | `Optional[Intent]` | The intent currently being processed in this conversation. May be `None` if the conversation is idle or has been reset. |
| `filled_slots` | `Dict[str, Any]` | A dictionary of slot names to values that have been collected so far in the conversation. This accumulates across multiple turns. |
| `missing_slots` | `List[str]` | A list of required slot names (from the current intent) that have not yet been filled. The dialog manager uses this to determine what to ask next. |
| `confirmation_required` | `bool` | A flag indicating whether the system needs to confirm a slot value with the user before proceeding (e.g., "Did you mean Boston, Massachusetts?"). |
| `last_activity` | `datetime` | The timestamp of the last user interaction in this session. Used to implement inactivity timeouts and clean up stale sessions. |
| `conversation_history` | `List[Dict]` | A log of previous turns in the conversation, typically storing processed `UserUtterance` objects or summaries. Useful for debugging, context recovery, and potentially for more advanced dialog management. |

The `DialogState` includes several important methods:
- `update_activity()`: Called whenever there's user interaction to refresh the `last_activity` timestamp to the current time.
- `is_expired(timeout_seconds)`: Compares the current time with `last_activity` plus the timeout period (using the constant `SESSION_TIMEOUT_SECONDS`). Returns `True` if the session has been inactive for too long.
- `reset()`: Clears the conversation context by setting `current_intent` to `None`, emptying `filled_slots` and `missing_slots`, and resetting flags. Returns the dialog to its initial state.

#### ResponseTemplate: The Response Blueprint
While not explicitly listed in the naming conventions as a core type (it's referenced in the diagram), the `ResponseTemplate` is an important data structure for the response generator. Think of it as a **Mad Libs sentence** with blanks to fill in. It contains the response text with placeholder variables and metadata about which slots are needed to complete it.

| Field | Type | Description |
|-------|------|-------------|
| `template_string` | `str` | The response text with placeholders (e.g., "Flight booked to {destination} on {departure_date}"). Placeholders are typically in `{slot_name}` format. |
| `required_slots` | `List[str]` | The list of slot names that must be present in the dialog state for this template to be usable. Ensures we don't try to fill a template with missing data. |
| `variation_group` | `str` | An identifier grouping multiple alternative templates for the same intent-state combination, allowing for natural variation in responses. |

### Training Data Format for Intents and Entities

The training data is the **textbook from which the chatbot learns**. It consists of annotated examples that teach the system how to recognize user intents and extract relevant entities. This data must be carefully structured and representative of real user inputs.

#### Intent Training Data Format
For the intent classifier (Milestone 1), training data consists of labeled examples: user utterances paired with their correct intent. The data should be stored in a simple, readable format like CSV or JSON.

**CSV Format Example:**
```
text,intent
"I want to book a flight to Paris","book_flight"
"Can you help me find a flight?","book_flight"
"What's the weather in Tokyo?","check_weather"
"Will it rain tomorrow?","check_weather"
"Is it sunny in London?","check_weather"
"I need to cancel my reservation","cancel_booking"
```

**JSON Format (More Structured):**
```json
{
  "intents": [
    {
      "name": "book_flight",
      "examples": [
        "I want to book a flight to Paris",
        "Can you help me find a flight?",
        "Looking for tickets to New York"
      ]
    },
    {
      "name": "check_weather",
      "examples": [
        "What's the weather in Tokyo?",
        "Will it rain tomorrow?",
        "Is it sunny in London?"
      ]
    }
  ]
}
```

**Key Requirements for Intent Training Data:**
1. **Minimum examples per intent**: At least 10-15 diverse examples per intent to provide adequate coverage of different phrasings.
2. **Balanced distribution**: Roughly similar numbers of examples per intent to avoid classifier bias toward more frequent intents.
3. **Natural variation**: Include different sentence structures, synonyms, and colloquial phrasings that real users might employ.
4. **Representative vocabulary**: Ensure examples contain the key words that are characteristic of each intent.

> **Design Insight:** The training data's quality directly limits the classifier's performance. A common pitfall is writing all examples in a similar, formal style, which fails to capture the diversity of natural language. Have multiple people write examples or collect real user queries if possible.

#### Entity Training Data Format
For training a machine-learning-based entity extractor (like spaCy's NER), data needs to be annotated with character-level spans. This is more complex than intent data because it requires marking the exact position of entities within the text.

**spaCy JSONL Format (One JSON object per line):**
```json
{
  "text": "Book a flight to Paris on March 15th",
  "entities": [
    {"start": 18, "end": 23, "label": "DESTINATION"},
    {"start": 27, "end": 37, "label": "DATE"}
  ]
}
{
  "text": "I want to go to Tokyo tomorrow",
  "entities": [
    {"start": 15, "end": 20, "label": "DESTINATION"},
    {"start": 21, "end": 29, "label": "DATE"}
  ]
}
```

**Simplified Format for Rule-Based Entities:**
For rule-based extraction (regex patterns, keyword lists), training data might simply be a configuration file defining the patterns:

```json
{
  "entity_patterns": {
    "DATE": [
      {"pattern": "\\btoday\\b", "normalized": "TODAY"},
      {"pattern": "\\btomorrow\\b", "normalized": "TOMORROW"},
      {"pattern": "\\d{1,2}/\\d{1,2}/\\d{4}", "normalizer": "parse_date"}
    ],
    "DESTINATION": {
      "keyword_list": ["Paris", "Tokyo", "London", "New York"],
      "fuzzy_match": true
    }
  }
}
```

**Key Requirements for Entity Training Data:**
1. **Consistent annotation boundaries**: Decide whether "New York City" should be one entity or two ("New York" + "City") and apply consistently.
2. **Overlap handling policy**: Define how to handle overlapping entities (e.g., "Friday morning" where "Friday" is a DATE and "morning" is a TIME).
3. **Negative examples**: Include sentences without entities to help the model learn what not to extract.
4. **Normalization examples**: Provide examples of how raw text should be converted to canonical forms (e.g., "tomorrow" → actual date, "NYC" → "New York City").

> **Architecture Decision Record: Flat vs. Hierarchical Entity Types**
> - **Context**: We need to categorize extracted entities to map them to intent slots. Some entities could be grouped (e.g., "city", "country", "airport" all map to a `location` slot).
> - **Options Considered**:
>   1. **Flat entity types**: Each distinct category gets its own type (`city`, `country`, `airport`). More precise but requires more training data.
>   2. **Hierarchical types**: Broad categories (`location`) with subtypes. More flexible but adds complexity to the mapping logic.
> - **Decision**: Use flat entity types for rule-based extraction (simpler) but allow a mapping layer that can group multiple entity types to a single slot.
> - **Rationale**: Flat types are easier to implement and understand for beginners. The slot mapping dictionary can handle grouping (e.g., map both `city` and `airport` to `destination` slot). This keeps the entity extraction simple while maintaining flexibility in slot filling.
> - **Consequences**: The entity extractor may produce fine-grained types that need to be mapped, but this mapping is centralized in one configuration spot.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Flat entity types | Simple implementation, clear boundaries, easier debugging | May require more patterns, less generalization | **Yes** (with slot mapping) |
| Hierarchical types | Better generalization, fewer patterns needed | More complex implementation, ambiguous boundaries | No |

#### Relationship Between Data Types
The data types form a clear dependency chain that mirrors the chatbot's processing pipeline:

1. **Training Phase**: 
   - `Intent` definitions (name + required slots) are created.
   - Training examples (text + intent labels) are used to train the classifier.
   - Entity patterns or annotated data are used to configure the entity extractor.

2. **Runtime Phase**:
   - User input creates a `UserUtterance` with `raw_text`.
   - The classifier matches it to an `Intent` (stored in `UserUtterance.intent`).
   - The entity extractor identifies `Entity` objects from the text (stored in `UserUtterance.entities`).
   - Entities are mapped to slots using the intent's slot definitions, populating `UserUtterance.slots`.
   - The `DialogState` is updated with the new information, tracking progress toward completing the intent.
   - When ready, a `ResponseTemplate` is selected and filled with values from `DialogState.filled_slots`.

This clear separation of concerns allows each component to focus on its specific task while passing structured data to the next component in the pipeline.

### Implementation Guidance

#### A. Technology Recommendations Table
| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Data Model Representation | Python dataclasses with type hints | Pydantic models with validation |
| Data Persistence (DialogState) | In-memory dictionary with session expiry | Redis or SQLite for persistent storage |
| Training Data Storage | JSON/CSV files in `data/` directory | SQL database with versioning |

#### B. Recommended File/Module Structure
```
intent_based_chatbot/
├── data/                          # Training data
│   ├── intents.json              # Intent definitions with examples
│   ├── entities.json             # Entity patterns and rules
│   └── responses.json            # Response templates
├── models/                       # Core data structures
│   ├── __init__.py
│   ├── utterance.py              # UserUtterance class
│   ├── intent.py                 # Intent class
│   ├── entity.py                 # Entity class
│   ├── dialog_state.py           # DialogState class
│   └── response_template.py      # ResponseTemplate class
├── classifiers/                  # Milestone 1
├── extractors/                   # Milestone 2
├── managers/                     # Milestone 3
├── generators/                   # Milestone 4
└── config.py                     # Constants like SESSION_TIMEOUT_SECONDS
```

#### C. Infrastructure Starter Code
Create the foundational data model classes that all other components will use. These are complete implementations that learners can import directly.

**models/intent.py**
```python
from dataclasses import dataclass
from typing import List

@dataclass
class Intent:
    """Represents a user intent with required and optional slots."""
    name: str
    required_slots: List[str]
    optional_slots: List[str]
    
    def __post_init__(self):
        """Validate that slot lists are actually lists."""
        if not isinstance(self.required_slots, list):
            self.required_slots = list(self.required_slots)
        if not isinstance(self.optional_slots, list):
            self.optional_slots = list(self.optional_slots)
```

**models/entity.py**
```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Entity:
    """Represents an entity extracted from user text."""
    type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0
    normalized: Any = None
    
    def __post_init__(self):
        """Validate entity span positions."""
        if self.start < 0:
            raise ValueError(f"Start position cannot be negative: {self.start}")
        if self.end <= self.start:
            raise ValueError(f"End position {self.end} must be greater than start {self.start}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0: {self.confidence}")
```

#### D. Core Logic Skeleton Code
Here are the key methods that learners should implement themselves, with TODO comments guiding the implementation.

**models/utterance.py**
```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from .intent import Intent
from .entity import Entity

@dataclass
class UserUtterance:
    """Represents a user message and its analysis results."""
    raw_text: str
    cleaned_text: str = ""
    intent: Optional[Intent] = None
    confidence: float = 0.0
    entities: List[Entity] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    
    def update_slots_from_entities(self, slot_mapping: Dict[str, str]) -> None:
        """
        Fill the slots dictionary using extracted entities and a mapping.
        
        Args:
            slot_mapping: Dictionary mapping entity types to slot names.
                Example: {"date": "departure_date", "location": "destination"}
        
        TODO 1: Initialize an empty dictionary for normalized slot values
        TODO 2: Iterate through each entity in self.entities
        TODO 3: Check if the entity's type exists in slot_mapping
        TODO 4: If it does, get the corresponding slot name from the mapping
        TODO 5: Use the entity's normalized value if available, otherwise use raw value
        TODO 6: Store the value in the slots dictionary with the slot name as key
        TODO 7: Handle the case where multiple entities map to the same slot 
                (e.g., keep the one with highest confidence)
        TODO 8: Update self.slots with the collected slot values
        """
```

**models/dialog_state.py**
```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from .intent import Intent

@dataclass
class DialogState:
    """Tracks the state of a conversation across multiple turns."""
    session_id: str
    current_intent: Optional[Intent] = None
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    confirmation_required: bool = False
    last_activity: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict] = field(default_factory=list)
    
    def update_activity(self) -> None:
        """Update the last activity timestamp to the current time."""
        # TODO 1: Set self.last_activity to the current datetime
        pass
    
    def is_expired(self, timeout_seconds: int) -> bool:
        """
        Check if the session has expired due to inactivity.
        
        Args:
            timeout_seconds: Number of seconds of inactivity before expiry
            
        Returns:
            True if session has expired, False otherwise
            
        TODO 1: Calculate the expiry time by adding timeout_seconds to last_activity
        TODO 2: Get the current datetime
        TODO 3: Return True if current time is past the expiry time, False otherwise
        """
        return False
    
    def reset(self) -> None:
        """Clear conversation context and return to initial state."""
        # TODO 1: Set current_intent to None
        # TODO 2: Clear the filled_slots dictionary
        # TODO 3: Clear the missing_slots list
        # TODO 4: Set confirmation_required to False
        # TODO 5: Update the activity timestamp (optional: keep history or clear it?)
        # TODO 6: Consider whether to clear conversation_history or keep it for analytics
        pass
```

#### E. Language-Specific Hints
- **Use Python dataclasses**: They automatically generate `__init__`, `__repr__`, and comparison methods, reducing boilerplate.
- **Type hints are crucial**: They improve code readability and enable static type checking with tools like `mypy`.
- **Default factories**: Use `field(default_factory=list)` instead of `=[]` to avoid mutable default argument issues.
- **Datetime handling**: Use `datetime.now()` for timestamps, not `time.time()`, for timezone-aware operations.
- **JSON serialization**: Add `asdict()` method or implement `to_dict()` methods for serialization to store dialog states.

#### F. Milestone Checkpoint
After implementing the data model:
1. **Create test instances**: Write a script that creates sample `Intent`, `Entity`, `UserUtterance`, and `DialogState` objects.
2. **Verify relationships**: Ensure an `Intent` can be assigned to a `UserUtterance`, and `Entity` objects can be added to it.
3. **Test helper methods**: Call `update_slots_from_entities` with a simple mapping and verify slots are populated correctly.
4. **Check session expiry**: Create a `DialogState`, wait 1 second, and verify `is_expired(1)` returns `True`.
5. **Run validation**: Execute the script to ensure no runtime errors and print the objects to verify their structure.

#### G. Debugging Tips
| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| `ValueError: Start position cannot be negative` | Entity extraction returned invalid span | Print entity start/end values and the original text length | Check the entity extractor's span calculation logic |
| Slots not being populated | Entity types don't match slot mapping keys | Print entity types and slot mapping dictionary | Ensure slot mapping includes all possible entity types for the intent |
| Session always expired | `last_activity` not updating | Print `last_activity` before and after `update_activity()` | Ensure `update_activity()` is called on every user interaction |
| Dialog state reset loses needed info | `reset()` clears too much | Check what `reset()` clears vs. what should be preserved | Consider keeping `conversation_history` while resetting intent and slots |


## 5.1 Component: Intent Classifier

> **Milestone(s):** Milestone 1 (Intent Classification)

The Intent Classifier is the "brain" of the chatbot's understanding system. Its primary responsibility is to analyze a user's raw text input and determine the user's underlying goal or purpose, expressed as a categorical label called an **intent**. This classification is the first and most critical step in the processing pipeline, as all subsequent actions—entity extraction, dialog management, and response generation—depend on correctly identifying the user's intent. The component must be robust, handling variations in phrasing, ambiguous statements, and inputs that fall outside its known categories.

### Mental Model: The Sorting Hat

Think of the Intent Classifier as the **Sorting Hat** from the Harry Potter series. The Hat's job is to listen to a student's thoughts and qualities (the user's utterance) and decide which of the four Hogwarts houses (the set of known intents) they best belong to. It doesn't just pick the first house that comes to mind; it considers the evidence, weighs the student's attributes against each house's values, and makes a reasoned judgment. Sometimes, the student's traits are so clear that the Hat shouts the house name immediately (high confidence). Other times, the student is a difficult case, and the Hat takes longer, considering multiple possibilities before making a choice (lower confidence). Crucially, if a student's qualities don't align with any of the four houses (an "out-of-domain" input), the Hat must honestly admit it doesn't have a good match, rather than forcing a bad fit. This mental model emphasizes the classifier's role in **probabilistic categorization** and the importance of a **confidence threshold** to protect against overconfident, incorrect classifications.

### Interface and Behavior

The Intent Classifier is a self-contained module with a clear interface. Its core function is to transform a string of text into a structured prediction containing an `Intent` object and a confidence score. Internally, it consists of two primary phases: a **training pipeline** that builds a model from labeled examples, and a **prediction pipeline** that applies the model to new inputs.

**Public Interface:**

The classifier exposes a minimal set of methods for training and prediction, abstracting away the complexities of vectorization and model selection.

| Method | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `train` | `training_data: List[Tuple[str, str]]` | `None` | Accepts a list of `(utterance, intent_label)` pairs. Preprocesses the text, performs TF-IDF vectorization, and trains a supervised classification model (e.g., Naive Bayes, SVM). Stores the fitted vectorizer and model for use in `predict`. |
| `predict` | `utterance: str` | `Tuple[Optional[Intent], float]` | The core prediction method. Takes a raw user utterance, preprocesses it, vectorizes it using the fitted TF-IDF vectorizer, and runs it through the trained model. Returns a tuple containing: 1) the predicted `Intent` object (or `None` if below threshold), and 2) the confidence score (the highest probability from the model). |
| `predict_proba` | `utterance: str` | `Dict[str, float]` | Returns a dictionary mapping every known intent label to the model's predicted probability for that intent. Useful for debugging and multi-intent analysis. |

**Internal State & Data Structures:**

The classifier maintains several internal objects after training.

| Internal Object | Type (Example) | Purpose |
| :--- | :--- | :--- |
| `vectorizer` | `TfidfVectorizer` (from scikit-learn) | Transforms raw text into a numerical TF-IDF feature vector. It "learns" the vocabulary and IDF weights from the training data. |
| `model` | `MultinomialNB` or `LinearSVC` (from scikit-learn) | The core statistical model that maps the TF-IDF feature vector to an intent label. |
| `intent_labels` | `List[str]` | The list of intent names the classifier was trained on, in the order the model expects. |
| `label_to_intent_map` | `Dict[str, Intent]` | A mapping from intent label strings to the full `Intent` objects (which include slot definitions). This allows `predict` to return a rich `Intent` object, not just a string. |

**Prediction Algorithm:**

The following numbered steps detail the algorithm executed by the `predict` method for a given user utterance:

1.  **Text Cleaning:** The raw input text is normalized. This typically involves converting to lowercase, removing extra whitespace, and stripping punctuation. More advanced cleaning (removing stop words, lemmatization) can be applied but is often deferred to the TF-IDF vectorizer's configuration.
2.  **Vectorization:** The cleaned text is transformed into a numerical feature vector using the pre-fitted `vectorizer`. This applies the TF-IDF formula, weighting words by their importance relative to the training corpus.
3.  **Probability Estimation:** The feature vector is passed to the trained classification `model`. The model outputs a probability distribution over all known intents (for probabilistic models like Naive Bayes) or a decision function score (for models like SVM, which must be calibrated to produce probabilities).
4.  **Confidence Assessment:** The highest probability score from the distribution is identified. This score represents the model's confidence that the input belongs to the corresponding intent.
5.  **Threshold Check:** The highest confidence score is compared against the `INTENT_CONFIDENCE_THRESHOLD` (e.g., 0.7). This is the critical gating mechanism.
    *   If `confidence >= INTENT_CONFIDENCE_THRESHOLD`: The prediction is accepted. Proceed to step 6.
    *   If `confidence < INTENT_CONFIDENCE_THRESHOLD`: The prediction is rejected. The method returns `(None, confidence)` to signal an **unknown intent**.
6.  **Intent Object Retrieval:** The intent label with the highest confidence is used as a key to look up the full `Intent` object from the `label_to_intent_map`.
7.  **Return Result:** The method returns the tuple `(Intent, confidence)`.

This flow is visualized in the provided diagram:
![Intent Classification and Fallback Flowchart](./diagrams/intent-classification-flowchart.svg)

### ADR: Choosing TF-IDF over Word Embeddings

> **Decision: Use TF-IDF + Traditional ML for Intent Classification**

*   **Context:** We are building a beginner-friendly, intent-based chatbot without LLMs. The intent classifier must be accurate, fast to train with limited data, simple to understand and debug, and have minimal external dependencies. The training data is expected to be modest (10s-100s of examples per intent) and potentially imbalanced.

*   **Options Considered:**
    1.  **TF-IDF + Traditional ML Classifier (Naive Bayes, SVM, Logistic Regression):** Represent text as a bag-of-words weighted by Term Frequency-Inverse Document Frequency. Train a fast, lightweight statistical model.
    2.  **Pre-trained Word Embeddings (Word2Vec, GloVe) + Neural Network:** Use dense, pre-trained vector representations of words. Average or pool word vectors for the sentence, then feed into a shallow neural network (e.g., a few Dense layers).
    3.  **Fine-tuned Transformer (BERT, DistilBERT):** Use a small, pre-trained transformer model and fine-tune it on our intent classification dataset. This represents the modern, high-performance approach.

*   **Decision:** We selected **Option 1: TF-IDF + Traditional ML Classifier** (specifically, `TfidfVectorizer` with `LinearSVC` or `MultinomialNB` from scikit-learn).

*   **Rationale:** The decision is driven by the project's **beginner** difficulty level and specific constraints.
    1.  **Simplicity and Transparency:** TF-IDF and models like Naive Bayes are mathematically straightforward and easier for a beginner to grasp, implement, and debug. Feature importance can be inspected (e.g., seeing which words are most indicative of an intent).
    2.  **Data Efficiency:** Traditional ML models often perform well with small to medium-sized datasets (10-100 examples per class). They are less prone to overfitting on tiny datasets compared to neural networks, which require careful regularization.
    3.  **Computational Cost:** Training and inference are extremely fast, requiring only CPU and minimal memory. This eliminates the need for GPU acceleration and complex dependency management (PyTorch/TensorFlow).
    4.  **Dependency Minimization:** Scikit-learn is a single, well-maintained dependency. Pre-trained word embeddings or transformer models add significant library overhead and require managing model download/caching.
    5.  **Adequate Performance:** For many closed-domain, task-oriented chatbots (booking, FAQs, simple commands), a well-tuned TF-IDF + SVM model can achieve >90% accuracy, satisfying our 80% acceptance criteria.

*   **Consequences:**
    *   **Positive:** The system will be quick to train, easy to explain, and run anywhere. Developers can focus on curating good training examples and tuning the confidence threshold.
    *   **Negative:** The model lacks deep semantic understanding. It will struggle with synonyms not present in the training data (e.g., "make a reservation" vs. "book a table") and complex paraphrasing. This is an acceptable trade-off for the project's scope.

**Comparison of Vectorization Options:**

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **TF-IDF** | Simple, fast, interpretable, works well with small data, no external dependencies. | Bag-of-words limitation (no word order), cannot handle unseen words well, may produce high-dimensional sparse vectors. | **Yes** |
| **Pre-trained Word Embeddings** | Captures semantic meaning, handles synonyms better, dense lower-dimensional vectors. | Requires managing embedding files, less interpretable, may not be domain-specific, averaging word vectors loses nuance. | No |
| **Fine-tuned Transformer** | State-of-the-art accuracy, excellent semantic understanding, handles complex language. | Very heavy dependency (transformers lib), slow training/inference, requires GPU for efficiency,极易在小数据上过拟合. | No |

### Common Pitfalls and Mitigations

Building the intent classifier seems straightforward, but several subtle mistakes can drastically reduce its reliability.

⚠️ **Pitfall: Insufficient and Unbalanced Training Data**
*   **Description:** Providing only 2-3 example sentences per intent, or having one intent with 100 examples and another with only 5.
*   **Why it's wrong:** Machine learning models learn patterns from data. Too few examples prevent the model from learning the full range of ways a user might express an intent. Imbalanced data biases the model towards the majority class, causing it to over-predict the common intents.
*   **Fix:** Adhere strictly to the acceptance criteria of **at least 10 diverse examples per intent**. Aim for variety in vocabulary, sentence length, and structure. If imbalance is unavoidable, use scikit-learn's `class_weight='balanced'` parameter in classifiers like `SVC` or `LogisticRegression` to penalize mistakes on the minority class more heavily.

⚠️ **Pitfall: Ignoring the Confidence Threshold**
*   **Description:** Always returning the top predicted intent, even if the model's confidence is only 0.51 (barely above random guessing).
*   **Why it's wrong:** This leads to frequent misclassifications on ambiguous or out-of-domain inputs (e.g., a user asking "what's the weather?" to a food-ordering bot might be incorrectly classified as `order_food` with low confidence). It destroys user trust.
*   **Fix:** **Always implement the threshold check.** Use the `INTENT_CONFIDENCE_THRESHOLD` constant. Return an `unknown` intent for low-confidence predictions, which the dialog manager can handle with a fallback response ("I'm not sure I understand").

⚠️ **Pitfall: Data Leakage in Vectorization**
*   **Description:** Fitting the TF-IDF vectorizer on the entire dataset (including the test set) before splitting into train and test, or using test data during the preprocessing/cleaning step.
*   **Why it's wrong:** This allows information from the test set to "leak" into the training process (via the IDF calculation), artificially inflating accuracy metrics and giving a false sense of model performance. The model will not generalize to real, unseen data.
*   **Fix:** Follow a strict pipeline: 1) Split data into training and test sets. 2) Fit the `TfidfVectorizer` **only on the training set**. 3) Transform both the training and test sets using that fitted vectorizer. In scikit-learn, use a `Pipeline` object to encapsulate vectorizer and model, and then evaluate only with `cross_val_score` or a held-out test set.

⚠️ **Pitfall: Overfitting on Small Datasets**
*   **Description:** Using a very complex model (like a non-linear SVM kernel or a deep neural network) on a tiny dataset, resulting in near-perfect training accuracy but terrible test accuracy.
*   **Why it's wrong:** The model memorizes the noise and specific examples in the training data rather than learning generalizable patterns.
*   **Fix:** Start with simple, high-bias models (`MultinomialNB` is very resilient to overfitting). If using `SVC`, begin with a linear kernel (`kernel='linear'`). Use techniques like cross-validation on the training set to evaluate real performance before the final test.

⚠️ **Pitfall: Not Updating the Label-to-Intent Map**
*   **Description:** After training the model, the `predict` method returns a string label but cannot reconstruct the full `Intent` object with its `required_slots`.
*   **Why it's wrong:** The downstream dialog manager needs the `Intent` object to know which slots are required for fulfillment. A missing map breaks the pipeline.
*   **Fix:** During training, build and store a `label_to_intent_map` dictionary. During prediction, use the predicted label to fetch the corresponding `Intent` object from this map before returning it.

### Implementation Guidance

**A. Technology Recommendations Table:**

| Component | Simple Option (Beginner-Friendly) | Advanced Option (For Extension) |
| :--- | :--- | :--- |
| **Text Vectorization** | `TfidfVectorizer` (scikit-learn) | `CountVectorizer` with n-grams, or `HashingVectorizer` for fixed memory. |
| **Classification Model** | `MultinomialNB` (Naive Bayes) | `LinearSVC` (Support Vector Classifier) or `LogisticRegression` (often higher accuracy). |
| **Probability Calibration** | Use `model.predict_proba()` (Naive Bayes). | For `LinearSVC`, wrap in `CalibratedClassifierCV` to get meaningful probabilities. |
| **Pipeline & Utilities** | `sklearn.pipeline.Pipeline`, `train_test_split`, `classification_report` | `GridSearchCV` for hyperparameter tuning. |

**B. Recommended File/Module Structure:**

Place the intent classifier within a clear module structure that separates data, models, and the core processing logic.

```
intent_based_chatbot/
├── data/
│   ├── intents.yaml (or .json)           # Intent definitions with examples
│   └── training_data.pkl                 # Serialized processed training data (optional)
├── models/
│   ├── __init__.py
│   ├── intent_classifier.py              # This component's main code
│   └── trained_intent_model.joblib       # Saved model after training
├── core/
│   ├── __init__.py
│   ├── intents.py                        # Intent class definition
│   └── dialog_state.py                   # DialogState class definition
├── utils/
│   ├── __init__.py
│   └── text_preprocessor.py              # Shared text cleaning functions
└── train_intent_classifier.py            # Script to train and save the model
```

**C. Infrastructure Starter Code (Text Preprocessor):**

A shared utility for consistent text cleaning across training and prediction.

```python
# utils/text_preprocessor.py
import re
import string

class TextPreprocessor:
    """Simple text cleaner for intent classification."""

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize input text for consistent processing.
        Converts to lowercase, removes extra whitespace, and strips punctuation.
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Optional: Remove punctuation (can be too aggressive for some use cases)
        # text = text.translate(str.maketrans('', '', string.punctuation))

        return text

    # Optional: Add more methods like removing stop words or lemmatization later.
```

**D. Core Logic Skeleton Code (Intent Classifier):**

The main classifier class with TODOs that map directly to the algorithm steps.

```python
# models/intent_classifier.py
import joblib
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from core.intents import Intent
from utils.text_preprocessor import TextPreprocessor

class IntentClassifier:
    """Classifies user utterances into predefined intents using TF-IDF and a probabilistic model."""

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the classifier.
        Args:
            confidence_threshold: Minimum probability to accept a prediction.
        """
        self.confidence_threshold = confidence_threshold
        self.pipeline = None  # Will hold the scikit-learn Pipeline (vectorizer + model)
        self.label_to_intent_map = {}  # Maps intent label string -> Intent object
        self.preprocessor = TextPreprocessor()

    def train(self, training_data: List[Tuple[str, str]], intent_objects: Dict[str, Intent]):
        """
        Train the intent classification model.
        Args:
            training_data: List of (utterance, intent_label) pairs.
            intent_objects: Dictionary mapping intent_label to its full Intent object.
        """
        # TODO 1: Store the label_to_intent_map for later use in prediction.
        #   self.label_to_intent_map = intent_objects

        # TODO 2: Separate the training data into features (X) and labels (y).
        #   X = [utterance for utterance, label in training_data]
        #   y = [label for utterance, label in training_data]

        # TODO 3: Preprocess the feature texts using the preprocessor.
        #   X_cleaned = [self.preprocessor.normalize(utt) for utt in X]

        # TODO 4: Define the model pipeline. A good starting point:
        #   vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        #   classifier = MultinomialNB()
        #   self.pipeline = Pipeline([('tfidf', vectorizer), ('clf', classifier)])

        # TODO 5: Fit the pipeline on the cleaned training data (X_cleaned, y).
        #   self.pipeline.fit(X_cleaned, y)

        # TODO 6 (Optional): Evaluate on a held-out validation set and print metrics.
        pass

    def predict(self, utterance: str) -> Tuple[Optional[Intent], float]:
        """
        Predict the intent for a given user utterance.
        Returns a tuple of (Intent object or None, confidence score).
        """
        # TODO 1: Preprocess the input utterance.
        #   cleaned_utt = self.preprocessor.normalize(utterance)

        # TODO 2: Use the trained pipeline to predict probabilities for ALL intents.
        #   probas = self.pipeline.predict_proba([cleaned_utt])[0]  # Returns a 2D array, take first row.

        # TODO 3: Find the index of the highest probability (confidence).
        #   max_idx = probas.argmax()
        #   confidence = probas[max_idx]

        # TODO 4: Get the intent label corresponding to the highest probability.
        #   The pipeline's classifier has a `classes_` attribute listing labels in order.
        #   predicted_label = self.pipeline.classes_[max_idx]

        # TODO 5: Check if confidence meets the threshold.
        #   if confidence >= self.confidence_threshold:
        #       intent_obj = self.label_to_intent_map.get(predicted_label)
        #       return intent_obj, confidence
        #   else:
        #       return None, confidence  # Unknown intent

        # Placeholder return for skeleton
        return None, 0.0

    def predict_proba(self, utterance: str) -> Dict[str, float]:
        """Return probability distribution over all intents for debugging."""
        # TODO 1: Preprocess utterance.
        # TODO 2: Get probability array from pipeline.
        # TODO 3: Create a dict mapping intent label -> probability using pipeline.classes_.
        #   return {label: prob for label, prob in zip(self.pipeline.classes_, probas)}
        return {}

    def save(self, filepath: str):
        """Save the trained pipeline and label map to disk."""
        if self.pipeline is not None:
            model_data = {
                'pipeline': self.pipeline,
                'label_map': self.label_to_intent_map,
                'threshold': self.confidence_threshold
            }
            joblib.dump(model_data, filepath)

    def load(self, filepath: str):
        """Load a trained pipeline and label map from disk."""
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.label_to_intent_map = model_data['label_map']
        self.confidence_threshold = model_data.get('threshold', 0.7)
```

**E. Language-Specific Hints (Python):**

*   **Scikit-learn Pipelines:** Use `Pipeline` to chain the vectorizer and classifier. This ensures the vectorizer is fitted only on training data during `cross_val_score` or `GridSearchCV`.
*   **Probability Calibration:** `LinearSVC` does not natively output probabilities. Wrap it with `CalibratedClassifierCV` to get reliable `predict_proba`.
    ```python
    from sklearn.svm import LinearSVC
    model = CalibratedClassifierCV(LinearSVC(class_weight='balanced'))
    ```
*   **Saving Models:** Use `joblib` (from scikit-learn) instead of `pickle` for efficiently storing large numpy arrays inside scikit-learn models.
*   **Handling Text:** The `TfidfVectorizer` has built-in tokenization and lowercasing. You can configure it via parameters like `lowercase=True`, `stop_words='english'`, and `ngram_range=(1,2)` to capture phrases.

**F. Milestone Checkpoint (Milestone 1):**

After implementing the `IntentClassifier`:

1.  **Run Training:** Execute your training script (`train_intent_classifier.py`). You should see output indicating the model has been fitted, possibly with a cross-validation accuracy score.
2.  **Test Prediction:** Write a small script or use an interactive Python shell to load the model and test predictions.
    ```python
    from models.intent_classifier import IntentClassifier
    classifier = IntentClassifier()
    classifier.load('models/trained_intent_model.joblib')

    test_phrases = ["I'd like to book a table", "What's the weather like?", "Tell me a joke"]
    for phrase in test_phrases:
        intent, conf = classifier.predict(phrase)
        print(f"'{phrase}' -> {intent.name if intent else 'UNKNOWN'} (conf: {conf:.2f})")
    ```
3.  **Expected Behavior:**
    *   Clear, in-domain phrases (e.g., "book a table" for a `reservation` intent) should return the correct intent with high confidence (>0.9).
    *   Gibberish or clearly out-of-domain phrases (e.g., "asdfghjkl") should return `None` (unknown intent) with low confidence.
    *   Ambiguous or edge-case phrases should be handled according to the threshold.
4.  **Signs of Trouble:**
    *   **All predictions are UNKNOWN:** Your confidence threshold is likely set too high. Temporarily lower it or check your model's probability outputs.
    *   **Same intent for every input:** The model is severely biased. Check for class imbalance and ensure your training data is correctly loaded and shuffled.
    *   **Error during prediction:** Likely a mismatch between the vocabulary learned during training and the words in your test input (if you preprocess differently). Ensure the exact same preprocessing is used in `train` and `predict`.

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
| :--- | :--- | :--- | :--- |
| Accuracy is very low (< 50%) on test set. | 1. Data leakage (vectorizer fitted on test data).<br>2. Too little/no training data.<br>3. Extreme class imbalance. | Print the size of your training set. Check your train/test split code. Use `classification_report` to see per-class metrics. | Ensure vectorizer is fit only on training fold. Collect more examples. Use `class_weight='balanced'`. |
| Model predicts the same intent for every input. | One intent class dominates the training set. The model learns to always predict the majority class. | Check the distribution of labels in `y_train` (e.g., using `collections.Counter`). | Balance the dataset (collect more examples for minority classes, use downsampling, or set `class_weight`). |
| Confidence scores are all very close (e.g., 0.33, 0.33, 0.34). | The model is uncertain and not discriminative. Possibly due to overly similar training examples or a weak model. | Inspect the TF-IDF features. Are they diverse? Try using n-grams (`ngram_range=(1,2)`) to capture phrases. | Improve training data variety. Try a stronger model like `LinearSVC`. Increase `max_features` in the vectorizer. |
| `predict_proba` returns 1.0 for one intent and 0.0 for all others, always. | This is normal for `MultinomialNB` with default settings; it tends to produce extreme probabilities. | This is expected behavior for Naive Bayes. It doesn't mean the model is 100% certain in a human sense. | Consider using a different model (LogisticRegression) or calibrating the probabilities. The threshold still works. |
| Error: `Vocabulary wasn't fitted.` | The `predict` method is called before `train`, or the pipeline wasn't fitted properly. | Check that `self.pipeline` is not `None` before calling `predict_proba`. | Ensure the training method completes successfully and the pipeline object is assigned to `self.pipeline`. |


## 5.2 Component: Entity Extractor and Slot Filler

> **Milestone(s):** Milestone 2 (Entity Extraction)

The Entity Extractor and Slot Filler is the chatbot's "detail-oriented assistant" responsible for identifying specific pieces of information within user messages and mapping them to structured parameters that the dialog system can act upon. While the Intent Classifier determines *what* the user wants (e.g., "book a flight"), this component determines *which specific details* are provided (e.g., "to Paris on Friday").

### Mental Model: The Form Filler

Imagine a concierge helping you fill out a complex paper form. The form has various fields like "Destination," "Travel Dates," and "Passenger Name." As you speak to the concierge, they listen carefully and:
1. **Spot key details** in your speech ("I need to go to Paris next Friday with Anna")
2. **Extract those details** and write them in the correct boxes on the form
3. **Interpret ambiguous terms** ("next Friday" becomes "2024-10-25")
4. **Notice missing information** and ask clarifying questions when needed
5. **Handle corrections** if you provide updated information

This component operates similarly: it scans the text for structured information, normalizes it to standard formats, and populates the intent's parameter slots. When slots remain empty, it enables the Dialog Manager to ask targeted questions to complete the form.

### Interface and Behavior

The Entity Extractor and Slot Filler operates as a pipeline with three sequential stages:

1. **Entity Extraction**: Scans `UserUtterance.raw_text` to identify spans of text representing entities
2. **Entity Normalization**: Converts extracted text values to canonical, structured formats
3. **Slot Filling**: Maps normalized entities to the appropriate slots based on the detected intent

The component's interface consists of two primary methods that operate on the `UserUtterance` and `DialogState` objects:

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `extract_entities` | `utterance: UserUtterance` | `List[Entity]` | Processes the raw text to identify all entities using both rule-based patterns and ML-based NER |
| `fill_slots` | `utterance: UserUtterance`, `intent: Intent`, `state: DialogState` | `Dict[str, Any]` | Maps extracted entities to intent slots, updates dialog state, and identifies missing required slots |

**Entity Extraction Pipeline** (detailed algorithm):
1. **Text Preprocessing**: Clean and normalize the input text (lowercasing, removing extra whitespace)
2. **Rule-Based Pattern Matching**: Apply regular expressions for structured entities (dates, times, phone numbers, etc.)
3. **Machine Learning NER**: Use spaCy's pre-trained model to identify named entities (persons, organizations, locations)
4. **Conflict Resolution**: Handle overlapping entity spans (rule-based takes precedence over ML when confidence is high)
5. **Entity Assembly**: Combine all detected entities into a single list, removing duplicates

**Slot Filling Logic** (detailed algorithm):
1. **Slot Mapping Resolution**: Determine which entity types map to which slots for the current intent
2. **Value Assignment**: For each extracted entity, assign its normalized value to the corresponding slot
3. **Multi-Entity Handling**: When multiple entities of the same type exist, store them as a list
4. **Missing Slot Detection**: Compare filled slots against `intent.required_slots` to identify what's still needed
5. **State Update**: Update `DialogState.filled_slots` and `DialogState.missing_slots` accordingly

**Core Data Structure Enhancements**:

The `Entity` type represents each extracted piece of information:

| Field Name | Type | Description |
|------------|------|-------------|
| `type` | `str` | The category of entity (e.g., "DATE", "PERSON", "LOCATION", "CUSTOM_TIME") |
| `value` | `str` | The raw text extracted from the utterance |
| `start` | `int` | Character index where the entity begins in the original text |
| `end` | `int` | Character index where the entity ends (exclusive) |
| `confidence` | `float` | How certain the extractor is about this entity (0.0-1.0) |
| `normalized` | `Any` | The canonical representation (e.g., `datetime` object for dates, `int` for numbers) |

**Entity Type Catalog** (common types):

| Entity Type | Examples | Normalized Format | Extraction Method |
|-------------|----------|-------------------|-------------------|
| `DATE` | "tomorrow", "October 25th", "12/31/2024" | `datetime.date` | Rule-based + spaCy |
| `TIME` | "3 PM", "14:30", "noon" | `datetime.time` | Rule-based |
| `DATETIME` | "next Friday at 5 PM" | `datetime.datetime` | Rule-based |
| `PERSON` | "John Smith", "Dr. Adams" | `str` (name) | spaCy NER |
| `LOCATION` | "Paris", "New York City" | `str` (standardized name) | spaCy NER + gazetteer |
| `ORGANIZATION` | "Google", "NYU" | `str` | spaCy NER |
| `MONEY` | "$25.50", "50 euros" | `float` (in base currency) | Rule-based |
| `PERCENT` | "25%", "100 percent" | `float` (0.0-1.0) | Rule-based |
| `QUANTITY` | "5 people", "2 kilograms" | `tuple(float, str)` | Rule-based |
| `PHONE` | "(555) 123-4567", "+1-800-555-1212" | `str` (E.164 format) | Rule-based |
| `EMAIL` | "user@example.com" | `str` | Rule-based |
| `URL` | "https://example.com" | `str` | Rule-based |

### ADR: Hybrid Rule-Based and ML NER Approach

> **Decision: Hybrid Extraction Strategy**
> - **Context**: We need to extract both common named entities (people, places) and domain-specific structured data (times, durations, custom formats) from user input. Pure ML approaches struggle with structured patterns, while pure regex approaches miss semantic understanding.
> - **Options Considered**:
>   1. **Pure Rule-Based**: Handcrafted regex patterns for all entity types
>   2. **Pure ML-Based**: Train or use pre-trained NER models for all extraction
>   3. **Hybrid Approach**: Combine rule-based for structured data with ML for named entities
> - **Decision**: Hybrid approach using rule-based patterns for structured entities (dates, times, numbers) and spaCy's pre-trained NER model for named entities (people, organizations, locations)
> - **Rationale**: 
>   - **Accuracy on structured patterns**: Regular expressions provide 100% precision for well-defined patterns like phone numbers and dates
>   - **Generalization for named entities**: ML models handle variations in names and locations that would require enormous regex complexity
>   - **Development efficiency**: Leveraging spaCy's pre-trained model saves weeks of training data collection and model tuning
>   - **Maintainability**: Rules for structured data are transparent and easy to debug, while ML handles the fuzzy cases
> - **Consequences**:
>   - **Positive**: High accuracy for both structured and unstructured entities, reasonable development time
>   - **Negative**: Requires both regex expertise and ML library integration, potential conflicts between overlapping rule-based and ML entities
>   - **Mitigation**: Implement conflict resolution strategy (rule-based wins for high-confidence matches)

**Option Comparison Table**:

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| **Pure Rule-Based** | - Complete control over patterns<br>- No external dependencies<br>- Transparent and debuggable | - Cannot handle unseen variations<br>- Exponential regex complexity for names/locations<br>- Poor generalization | ❌ |
| **Pure ML-Based** | - Handles variations and ambiguity well<br>- Single unified approach<br>- Can learn from data | - Poor at structured patterns (dates, times)<br>- Requires large training dataset<br>- Black-box decisions hard to debug | ❌ |
| **Hybrid Approach** | - Best of both worlds<br>- Leverages pre-trained models where effective<br>- Rules where patterns are clear | - Conflict resolution needed<br>- Two systems to maintain<br>- Slightly higher complexity | ✅ |

### Common Pitfalls and Mitigations

⚠️ **Pitfall: Greedy Regex Matches Incorrect Text**
- **Description**: A regex pattern like `\d+` intended to match "3 PM" might incorrectly match "room 205" as the time "205"
- **Why it's wrong**: Incorrect entity extraction leads to wrong slot values and confusing dialog responses
- **Fix**: Use word boundaries (`\b`) and context-aware patterns, validate extracted values (e.g., time should be 0-23), and implement confidence scoring based on pattern specificity

⚠️ **Pitfall: Overlapping Entity Spans Create Conflicts**
- **Description**: Both regex and spaCy might identify "Friday" as an entity—regex as a `DATE`, spaCy as a `PERSON` (if it's someone's name)
- **Why it's wrong**: Duplicate or conflicting entities cause slot filling ambiguity
- **Fix**: Implement a conflict resolution strategy: 
  1. Rule-based entities with high confidence (>0.9) take precedence
  2. Longer spans win over shorter ones for same text
  3. Maintain provenance metadata to track extraction source

⚠️ **Pitfall: Date Parsing Ambiguity (MM/DD vs DD/MM)**
- **Description**: "01/02/2024" could be January 2nd or February 1st depending on locale
- **Why it's wrong**: Critical errors in bookings or scheduling
- **Fix**: 
  1. Use explicit date libraries (dateparser) that accept format hints
  2. Maintain user locale preference in `DialogState`
  3. When ambiguous, ask for clarification ("Did you mean January 2nd or February 1st?")
  4. Prefer unambiguous formats in prompts ("Please use YYYY-MM-DD format")

⚠️ **Pitfall: Missing spaCy Model Causes Runtime Error**
- **Description**: Code assumes `import spacy; nlp = spacy.load("en_core_web_sm")` works, but model isn't downloaded
- **Why it's wrong**: Entire entity extraction fails on first run
- **Fix**: 
  1. Add runtime check with clear error message
  2. Include installation instructions in README
  3. Consider lazy loading with fallback to rule-only mode
  4. Use `python -m spacy download en_core_web_sm` in setup script

⚠️ **Pitfall: Slot Filling Overwrites User Corrections**
- **Description**: User says "Change destination to London" but system overwrites all previously filled slots
- **Why it's wrong**: Loses conversation context and previously confirmed information
- **Fix**:
  1. Implement intent-specific update policies (some intents replace, others add to existing)
  2. Track slot provenance (when/from which utterance each slot was filled)
  3. Use `DialogState.conversation_history` to detect correction patterns

### Implementation Guidance

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Rule-Based Extraction** | Python `re` module with carefully crafted patterns | `regex` library (supports advanced features), `dateparser` for dates |
| **ML-Based NER** | spaCy's pre-trained `en_core_web_sm` model (small, fast) | spaCy `en_core_web_lg` (more accurate), fine-tuned custom model |
| **Entity Normalization** | Custom conversion functions for each entity type | `dateutil.parser`, `phonenumbers` library, custom normalizers |
| **Slot Mapping** | Hardcoded dictionary mapping entity types to slot names | Configurable YAML/JSON mapping files for different intents |

#### B. Recommended File/Module Structure

```
intent_based_chatbot/
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
├── data/
│   ├── intents/                     # Intent definitions
│   │   ├── booking.yaml             # Intent with slot mappings
│   │   └── inquiry.yaml
│   └── entities/                    # Entity patterns
│       ├── patterns.json            # Regex patterns by entity type
│       └── custom_entities.py       # Custom entity definitions
├── src/
│   ├── entities/                    # Entity extraction module
│   │   ├── __init__.py
│   │   ├── extractor.py            # Main EntityExtractor class
│   │   ├── patterns.py             # Regex pattern definitions
│   │   ├── normalizers.py          # Entity normalization functions
│   │   └── conflict_resolver.py    # Overlap resolution logic
│   ├── intents/                     # Intent classification (Milestone 1)
│   ├── dialog/                      # Dialog management (Milestone 3)
│   └── responses/                   # Response generation (Milestone 4)
└── tests/
    ├── test_entity_extractor.py
    └── test_slot_filler.py
```

#### C. Infrastructure Starter Code

**Complete Entity Type Definitions** (save as `src/entities/types.py`):

```python
"""
Entity type definitions and data structures.
"""
from dataclasses import dataclass
from datetime import date, time, datetime
from typing import Any, Optional


@dataclass
class Entity:
    """Represents an extracted entity from user text."""
    type: str
    value: str  # Raw text extracted
    start: int  # Start index in original text
    end: int    # End index (exclusive)
    confidence: float = 1.0  # 0.0-1.0 confidence score
    normalized: Optional[Any] = None  # Canonical representation
    
    def __post_init__(self):
        """Validate entity data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span indices: start={self.start}, end={self.end}")


class EntityNormalizer:
    """Converts raw entity values to canonical formats."""
    
    @staticmethod
    def normalize_date(date_str: str) -> Optional[date]:
        """
        Convert date string to datetime.date object.
        Example: "tomorrow" -> date(2024, 10, 26)
        """
        try:
            # Simple implementation - extend with dateparser for production
            from datetime import datetime as dt
            # Handle common formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]:
                try:
                    return dt.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            return None
        except Exception as e:
            print(f"Date normalization failed for '{date_str}': {e}")
            return None
    
    @staticmethod
    def normalize_time(time_str: str) -> Optional[time]:
        """
        Convert time string to datetime.time object.
        Example: "3:30 PM" -> time(15, 30)
        """
        # Implementation for common time formats
        # This is a simplified version - use a library like dateutil in production
        try:
            time_str_lower = time_str.lower().strip()
            if "pm" in time_str_lower:
                hour_offset = 12
                time_str_lower = time_str_lower.replace("pm", "").strip()
            elif "am" in time_str_lower:
                hour_offset = 0
                time_str_lower = time_str_lower.replace("am", "").strip()
            else:
                hour_offset = 0
            
            if ":" in time_str_lower:
                hour_str, minute_str = time_str_lower.split(":")
                hour = int(hour_str) + hour_offset
                minute = int(minute_str)
            else:
                hour = int(time_str_lower) + hour_offset
                minute = 0
            
            # Handle 12 AM/PM edge cases
            if hour == 12 and hour_offset == 0:  # 12 AM
                hour = 0
            elif hour == 12 and hour_offset == 12:  # 12 PM
                hour = 12
            elif hour >= 24:  # Handle 24-hour wrap
                hour -= 12
            
            return time(hour % 24, minute)
        except Exception as e:
            print(f"Time normalization failed for '{time_str}': {e}")
            return None
    
    @staticmethod
    def normalize_quantity(quantity_str: str) -> Optional[tuple]:
        """
        Convert quantity string to (value, unit) tuple.
        Example: "5 people" -> (5.0, "people")
        """
        try:
            import re
            match = re.match(r"(\d+(?:\.\d+)?)\s*(\w+)", quantity_str)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                return (value, unit)
            return None
        except Exception as e:
            print(f"Quantity normalization failed for '{quantity_str}': {e}")
            return None
```

#### D. Core Logic Skeleton Code

**Main Entity Extractor Class** (save as `src/entities/extractor.py`):

```python
"""
Entity extraction and slot filling implementation.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .types import Entity, EntityNormalizer
# Import your Intent and UserUtterance types from the appropriate modules
# from ..intents.classifier import Intent
# from ..dialog.models import UserUtterance, DialogState


class EntityExtractor:
    """
    Extracts entities from text using hybrid rule-based and ML approach.
    """
    
    def __init__(self):
        """Initialize the entity extractor with patterns and ML model."""
        # TODO 1: Load regex patterns from file or define them here
        self.patterns = self._load_patterns()
        
        # TODO 2: Initialize spaCy model (lazy load to avoid startup cost)
        self.nlp = None  # Will be loaded on first use
    
    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for different entity types."""
        # TODO 3: Load from JSON/YAML file or define inline
        patterns = {
            "DATE": [
                r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # MM/DD/YYYY or DD/MM/YYYY
                r"\b\d{4}-\d{2}-\d{2}\b",      # YYYY-MM-DD
                r"\b(?:tomorrow|today|yesterday)\b",
            ],
            "TIME": [
                r"\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b",
                r"\b(?:noon|midnight)\b",
            ],
            "PHONE": [
                r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            ],
            "EMAIL": [
                r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
            ],
            "QUANTITY": [
                r"\b\d+\s+\w+\b",  # "5 people", "2 hours"
            ],
        }
        return patterns
    
    def _load_spacy_model(self):
        """Lazy load spaCy model to avoid startup delay."""
        # TODO 4: Import spacy and load the model
        # import spacy
        # self.nlp = spacy.load("en_core_web_sm")
        pass
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract all entities from the given text.
        
        Args:
            text: The input text to process
            
        Returns:
            List of Entity objects found in the text
        """
        entities = []
        
        # TODO 5: Apply rule-based patterns for each entity type
        for entity_type, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = Entity(
                        type=entity_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,  # Rule-based high confidence
                    )
                    entities.append(entity)
        
        # TODO 6: Apply spaCy NER for named entities
        if self.nlp is None:
            self._load_spacy_model()
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Map spaCy entity labels to our entity types
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    entity = Entity(
                        type=entity_type,
                        value=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=ent._.score if hasattr(ent._, 'score') else 0.8,
                    )
                    entities.append(entity)
        
        # TODO 7: Resolve overlapping entities (rule-based takes precedence)
        entities = self._resolve_overlaps(entities)
        
        # TODO 8: Normalize entity values (convert to canonical formats)
        entities = self._normalize_entities(entities)
        
        return entities
    
    def _map_spacy_label(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to our entity type system."""
        mapping = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION", 
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "PERCENT": "PERCENT",
        }
        return mapping.get(spacy_label)
    
    def _resolve_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """
        Resolve overlapping entity spans.
        Rule: Rule-based entities with confidence > 0.9 win over ML entities.
        """
        # TODO 9: Sort entities by start position, then by length
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        
        filtered = []
        for entity in entities:
            # Check if this entity overlaps with any already accepted entity
            overlap = False
            for accepted in filtered:
                if (entity.start < accepted.end and entity.end > accepted.start):
                    # Overlap detected - apply conflict resolution rules
                    # TODO 10: Implement conflict resolution logic
                    # Rule 1: Higher confidence wins
                    # Rule 2: Rule-based (confidence 0.9) wins over ML
                    # Rule 3: Longer span wins for same confidence
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(entity)
        
        return filtered
    
    def _normalize_entities(self, entities: List[Entity]) -> List[Entity]:
        """Normalize entity values to canonical formats."""
        normalizer = EntityNormalizer()
        
        for entity in entities:
            if entity.type == "DATE" and entity.normalized is None:
                entity.normalized = normalizer.normalize_date(entity.value)
            elif entity.type == "TIME" and entity.normalized is None:
                entity.normalized = normalizer.normalize_time(entity.value)
            elif entity.type == "QUANTITY" and entity.normalized is None:
                entity.normalized = normalizer.normalize_quantity(entity.value)
            # TODO 11: Add normalization for other entity types
        
        return entities


class SlotFiller:
    """
    Maps extracted entities to intent slots and manages slot state.
    """
    
    def __init__(self, slot_mappings: Dict[str, Dict[str, str]]):
        """
        Initialize with slot mappings.
        
        Args:
            slot_mappings: Dict mapping intent names to entity_type->slot_name mappings
                Example: {"book_flight": {"LOCATION": "destination", "DATE": "travel_date"}}
        """
        self.slot_mappings = slot_mappings
    
    def fill_slots(
        self,
        utterance,  # UserUtterance type
        intent,     # Intent type  
        state       # DialogState type
    ) -> Dict[str, Any]:
        """
        Fill slots based on extracted entities and update dialog state.
        
        Args:
            utterance: The UserUtterance containing extracted entities
            intent: The detected Intent with required/optional slots
            state: Current DialogState to update
            
        Returns:
            Dictionary of filled slots (slot_name -> value)
        """
        filled_slots = state.filled_slots.copy() if state.filled_slots else {}
        
        # TODO 12: Get entity-to-slot mapping for this intent
        intent_mapping = self.slot_mappings.get(intent.name, {})
        
        # TODO 13: Group entities by type for multi-entity handling
        entities_by_type = {}
        for entity in utterance.entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)
        
        # TODO 14: For each entity type that maps to a slot
        for entity_type, entities in entities_by_type.items():
            slot_name = intent_mapping.get(entity_type)
            if slot_name:
                if len(entities) == 1:
                    # Single entity - use normalized value if available
                    value = entities[0].normalized or entities[0].value
                    filled_slots[slot_name] = value
                else:
                    # Multiple entities of same type - store as list
                    values = [e.normalized or e.value for e in entities]
                    filled_slots[slot_name] = values
        
        # TODO 15: Update the utterance's slots dictionary
        utterance.slots = filled_slots.copy()
        
        # TODO 16: Update dialog state
        state.filled_slots = filled_slots
        
        # TODO 17: Identify missing required slots
        state.missing_slots = [
            slot for slot in intent.required_slots 
            if slot not in filled_slots or filled_slots[slot] is None
        ]
        
        # TODO 18: Update activity timestamp
        state.update_activity()
        
        return filled_slots
```

#### E. Language-Specific Hints

1. **spaCy Installation**: Remember to run `python -m spacy download en_core_web_sm` before using the ML NER features. Wrap this in a try-except with helpful error messages.

2. **Regex Patterns**: Use raw strings (`r"pattern"`) for regex patterns to avoid escaping issues. The `regex` library (pip install regex) offers advanced features like fuzzy matching.

3. **Date Parsing**: For production use, prefer `dateparser` library over custom date parsing: `pip install dateparser`. It handles "tomorrow", "next Friday", "in 2 weeks" etc.

4. **Performance**: Cache spaCy model loading (singleton pattern) to avoid reloading on every extraction. Process multiple texts in batch using `nlp.pipe()` if needed.

5. **Testing Entity Extraction**: Write unit tests with example sentences covering edge cases: ambiguous dates, overlapping entities, and entities at text boundaries.

#### F. Milestone Checkpoint

After implementing the Entity Extractor and Slot Filler:

**Run the verification test**:
```bash
python -m pytest tests/test_entity_extractor.py -v
```

**Expected behavior**:
1. Entity extraction should identify "Paris" as a LOCATION and "next Friday" as a DATE in the sentence "I want to fly to Paris next Friday"
2. Slot filling should map these to "destination" and "travel_date" slots for a "book_flight" intent
3. The system should correctly identify when required slots are missing
4. Date normalization should convert "next Friday" to an actual date object

**Signs of correct implementation**:
- Test passes showing at least 80% entity extraction accuracy on sample sentences
- Console shows clear entity extraction results with confidence scores
- Slot values are stored in standardized formats (not raw strings)

**Common issues to check**:
- If spaCy model fails to load, check installation with `python -c "import spacy; print(spacy.__version__)"`
- If regex patterns don't match, test them individually at https://regex101.com/
- If slot mapping doesn't work, verify the intent mapping dictionary structure

#### G. Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **No entities extracted** | spaCy model not loaded or regex patterns too strict | Print intermediate results, check if `self.nlp` is `None` | Run `python -m spacy download en_core_web_sm`, simplify patterns |
| **Wrong entity type assigned** | Incorrect spaCy label mapping or regex pattern matching wrong text | Print entity details including raw text and confidence | Adjust label mapping in `_map_spacy_label()`, make regex patterns more specific |
| **Slot filling overwrites previous values** | `fill_slots` creates new dict instead of updating existing | Check if `filled_slots = {}` instead of copying from state | Start with `filled_slots = state.filled_slots.copy()` |
| **Date normalization returns None** | Date format not recognized or ambiguous | Print the raw date string and attempted parsing formats | Add more date formats, use `dateparser` library for robust parsing |
| **Multiple entities of same type lost** | Slot filler only keeps the last one | Check if slot values are being overwritten instead of appended | Implement logic to handle multiple entities as lists |
| **High memory usage** | spaCy model loaded multiple times | Check if `__init__` creates new model instance each time | Implement singleton pattern or lazy loading |


## 5.3 Component: Dialog Manager

> **Milestone(s):** Milestone 3 (Dialog Management)

The **Dialog Manager** is the central, stateful orchestrator of the chatbot system. It is the component that transforms a series of disconnected user messages into a coherent, goal-oriented conversation. While the Intent Classifier and Entity Extractor understand the *what* and the *who/what/when* of a single message, the Dialog Manager understands the *flow* of the entire interaction. It remembers what the user has already said, identifies what information is still missing, and decides what the system should do or say next to advance the conversation toward fulfilling the user's goal.

### Mental Model: The Conversation Conductor

Think of the Dialog Manager as a **Conversation Conductor** leading an orchestra. The user is the soloist, providing musical phrases (utterances). The other components (Intent Classifier, Entity Extractor) are the different sections of the orchestra (strings, woodwinds), each interpreting their part of the score.

The conductor's job is multifaceted:
1.  **Listening & Interpreting:** The conductor listens to the soloist (user) and uses the orchestra (other components) to understand the musical theme (intent) and the specific notes (entities) being played.
2.  **Maintaining the Score:** The conductor keeps the full musical score (the dialog state) in front of them. This score tracks which movement (intent) they are in, which bars (slots) have been completed, and which are yet to be played.
3.  **Directing the Flow:** If the soloist skips a bar, the conductor doesn't stop the performance. Instead, they cue the soloist with a specific gesture (a prompt) to play the missing part. The conductor ensures the performance progresses smoothly from introduction to finale, even if there are repetitions or corrections.
4.  **Resetting for a New Piece:** When the piece ends or the soloist wants to start a new one, the conductor turns the page back to the beginning (resets the state), ready for a fresh performance.

This mental model emphasizes the Dialog Manager's role in **statefulness, orchestration, and flow control**. It does not generate the music (responses) itself—that's the role of the Response Generator. Instead, it decides *which* piece of music needs to be played next, based on the current state of the performance.

### Interface and State Machine

The Dialog Manager's primary interface is a single method that processes a new user utterance within the context of a specific conversation session. Its internal logic is best modeled as a finite state machine (FSM), where the state is stored in the `DialogState` object.

#### Core Interface

| Method Name | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `process_utterance` | `session_id: str`, `user_input: str` | `DialogState` | The central method of the Dialog Manager. Takes a raw user message and a session identifier, orchestrates the full processing pipeline (intent classification, entity extraction, slot filling), updates the conversation state, and returns the updated `DialogState`. The `DialogState` contains all information needed by the Response Generator to formulate the next system message. |

#### Dialog State Machine

The behavior of `process_utterance` is governed by the state of the conversation. The following table defines the states, events, transitions, and actions of the Dialog Manager's state machine. This machine is conceptually "run" on each call to `process_utterance`.

![Dialog Manager State Machine](./diagrams/dialog-state-machine.svg)

| Current State | Trigger Event | Next State | Actions Taken by Dialog Manager |
| :--- | :--- | :--- | :--- |
| **IDLE** | `intent_detected` (new intent with confidence > threshold) | **INTENT_IDENTIFIED** | 1. Set `current_intent` to the detected intent.<br>2. Initialize `filled_slots` as empty dict, `missing_slots` as `intent.required_slots`.<br>3. Call `SlotFiller.fill_slots` with the initial utterance.<br>4. Update `missing_slots` based on what was filled.<br>5. If `missing_slots` is empty, transition to **READY_TO_EXECUTE**. |
| **IDLE** | `intent_detected` (intent is `None` or confidence < threshold) | **IDLE** | 1. Keep state in **IDLE**.<br>2. Set `current_intent` to `None`.<br>3. The Response Generator will use this to produce a "I didn't understand" fallback. |
| **INTENT_IDENTIFIED** | `slot_received` (new utterance provides one or more entities) | **COLLECTING_SLOTS** | 1. Call `SlotFiller.fill_slots` with the new utterance.<br>2. Update `filled_slots` and `missing_slots`.<br>3. If `missing_slots` is empty, transition to **READY_TO_EXECUTE**. If still missing slots, remain in **COLLECTING_SLOTS**. |
| **INTENT_IDENTIFIED** | `reset` (e.g., user says "start over") | **IDLE** | 1. Call `DialogState.reset()`. |
| **COLLECTING_SLOTS** | `slot_received` | **COLLECTING_SLOTS** | 1. Same actions as transition from INTENT_IDENTIFIED on `slot_received`.<br>2. Loop, prompting for next missing slot until all are filled. |
| **COLLECTING_SLOTS** | `reset` | **IDLE** | 1. Call `DialogState.reset()`. |
| **COLLECTING_SLOTS** | All required slots filled (`missing_slots` empty) | **READY_TO_EXECUTE** | 1. Set `confirmation_required = True` if the intent or any critical slot value warrants it (configurable).<br>2. If confirmation is required, transition to **CONFIRMING**, else stay in **READY_TO_EXECUTE** (and the Response Generator will produce the final action response). |
| **READY_TO_EXECUTE** | (Automatic) Confirmation not required | **IDLE** | 1. The Response Generator produces the final response for the action.<br>2. After response is sent, the Dialog Manager automatically calls `DialogState.reset()` to return to **IDLE**. |
| **READY_TO_EXECUTE** | Confirmation required | **CONFIRMING** | 1. The Dialog Manager sets `confirmation_required = True` and stays in this state until the user confirms or denies. |
| **CONFIRMING** | `confirmation_yes` (user says "yes", "correct", etc.) | **IDLE** | 1. The Response Generator produces the final "action taken" response.<br>2. After response, call `DialogState.reset()`. |
| **CONFIRMING** | `confirmation_no` (user says "no", "that's wrong") | **COLLECTING_SLOTS** | 1. The Dialog Manager clears `filled_slots` for the slots the user indicated were wrong, or resets all slots.<br>2. Sets `missing_slots` accordingly and re-enters slot collection. |
| **CONFIRMING** | `reset` | **IDLE** | 1. Call `DialogState.reset()`. |
| *Any State* | `timeout` (`DialogState.is_expired()` returns `True`) | **IDLE** | 1. Call `DialogState.reset()` silently. The next user message will start a fresh session. |

#### Data Structure: `DialogState`

This object is the physical manifestation of the conversation state. It is persisted (in memory or otherwise) keyed by `session_id`.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `session_id` | `str` | A unique identifier for the conversation session (e.g., generated from user ID, channel ID, or a random UUID). |
| `current_intent` | `Optional[Intent]` | The intent currently being pursued in this conversation. `None` means no active intent (state is **IDLE**). |
| `filled_slots` | `Dict[str, Any]` | A mapping from slot names (e.g., `"location"`, `"date"`) to their confirmed values. Values are normalized (e.g., a `datetime` object for a date). |
| `missing_slots` | `List[str]` | The list of required slot names (from `current_intent.required_slots`) that have not yet been filled. This is the Dialog Manager's "to-do list". |
| `confirmation_required` | `bool` | A flag indicating whether the system needs explicit user confirmation before executing the action associated with the intent. Set automatically based on rules (e.g., high-stakes actions) or manually by the slot filler if a low-confidence entity was used. |
| `last_activity` | `datetime` | The timestamp of the last user interaction for this session. Used with `SESSION_TIMEOUT_SECONDS` to expire stale sessions. |
| `conversation_history` | `List[Dict]` | A log of interactions in this session. Each entry is a dictionary capturing a turn, typically containing the user utterance, the system's response, the intent, and entities detected. Useful for debugging, analytics, and potentially for more advanced context tracking. |

The `DialogState` also exposes the helper methods `update_activity()`, `is_expired(timeout_seconds)`, and `reset()` as defined in the naming conventions.

### ADR: In-Memory Session Store vs. Persistent Storage

> **Decision: Use an In-Memory Session Store with Simple Time-Based Expiry**

*   **Context:** The Dialog Manager needs to maintain `DialogState` for each active conversation session. This state must be accessible with very low latency for every user turn. The system is designed as a single-process, beginner-friendly application without initial requirements for horizontal scaling or persistence across server restarts.

*   **Options Considered:**
    1.  **In-Memory Dictionary:** Store `DialogState` objects in a Python dictionary within the Dialog Manager class, keyed by `session_id`. Implement a background cleanup thread or a lazy cleanup on access to remove expired sessions using `is_expired()`.
    2.  **External Key-Value Store:** Use an external, persistent store like Redis or a simple SQLite database to store serialized `DialogState` objects. This would allow state to survive server restarts and facilitate multiple server instances.

*   **Decision:** Use an **In-Memory Dictionary** for session storage.

*   **Rationale:** The primary drivers are simplicity and alignment with the project's **beginner** difficulty level.
    1.  **Implementation Simplicity:** A `Dict[str, DialogState]` requires no additional dependencies, configuration, or serialization logic. It can be implemented in a few lines of code.
    2.  **Performance:** In-memory access is extremely fast, with O(1) lookup time, which is ideal for the interactive, turn-based nature of a chatbot.
    3.  **Minimal Operational Overhead:** There is no separate service to install, run, or monitor. This reduces the cognitive load and setup steps for a learner.
    4.  **Sufficient for Learning Goals:** The core learning objective of Milestone 3 is to understand state machines and dialog flow, not distributed systems or persistent storage. An in-memory solution perfectly supports building and testing these concepts.

*   **Consequences:**
    *   **Positive:** The system is self-contained and easy to run. Development and debugging are straightforward.
    *   **Negative (Trade-offs):**
        *   **State Loss on Restart:** All conversation states are lost if the server process stops. This leads to a poor user experience but is acceptable for a learning project.
        *   **Scalability Limitation:** The system cannot be scaled horizontally (multiple server instances) because session state is not shared. All requests for a given session must be routed to the same server process.
        *   **Memory Bound:** The server's memory usage grows with the number of active sessions. A memory leak (e.g., not expiring sessions) could crash the server.

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **In-Memory Dictionary** | • Extremely simple to implement (few lines)<br>• No external dependencies<br>• Maximum read/write performance | • State lost on server restart<br>• Not scalable beyond a single process<br>• Risk of memory leaks if sessions not expired | **Yes** |
| **External Store (e.g., Redis)** | • State persists across restarts<br>• Enables horizontal scaling<br>• Built-in expiration mechanisms | • Adds operational complexity (run Redis)<br>• Introduces network latency<br>• Requires serialization/deserialization logic | No |

### Common Pitfalls and Mitigations

⚠️ **Pitfall 1: Memory Leak from Never-Expiring Sessions**
*   **Description:** The developer stores `DialogState` objects in a dictionary but never removes them, even after the conversation is finished or the user leaves. Over time, the dictionary grows indefinitely, consuming all available memory and crashing the server.
*   **Why it's Wrong:** This is a classic resource leak. It makes the system unreliable and unsuitable for any long-running deployment.
*   **Fix:** **Always implement session expiration.** On every access to a session (`process_utterance`), call `state.is_expired(SESSION_TIMEOUT_SECONDS)`. If expired, call `state.reset()` and treat it as a new session. Optionally, run a periodic background task that iterates through all sessions and deletes expired ones.

⚠️ **Pitfall 2: Losing Context on Intent Change**
*   **Description:** The user is in the middle of providing details for a `book_flight` intent (e.g., "to Paris") and then abruptly asks a new question, "What's the weather like?". The chatbot correctly identifies the new `get_weather` intent but then responds "Where are you?" (asking for the `location` slot), forgetting that the user just said "Paris".
*   **Why it's Wrong:** This feels jarring and unintelligent. The chatbot fails to leverage available context, forcing the user to repeat information.
*   **Fix:** Implement **context carryover rules** in the `SlotFiller`. When a new intent is detected, before resetting the state, check if any entities extracted from the utterance (like "Paris") map to slots in the *new* intent. If they do, pre-fill those slots. The logic should be: 1) Detect new intent, 2) Extract entities from the utterance, 3) Fill slots for the *new* intent with any matching entities, 4) Reset state and set the new intent.

⚠️ **Pitfall 3: Infinite Loop in Slot Filling**
*   **Description:** The chatbot asks for a `date` slot. The user provides an answer, but the Entity Extractor fails to recognize it (e.g., "next Thursday"). The `missing_slots` list still contains `"date"`, so the Dialog Manager prompts for the date again, creating an endless loop of "What date?" -> "next Thursday" -> "What date?".
*   **Why it's Wrong:** The conversation is stuck. The system cannot recover from its own failure to understand valid user input.
*   **Fix:** **Implement re-prompt limits and fallback strategies.** Keep a counter for each missing slot. After 2 failed attempts to fill it, change the prompt (e.g., "Could you please phrase the date differently, like 'March 15th'?"). After one more failure, consider clearing the intent and responding with a fallback like "Let's start over, or you can ask me something else."

⚠️ **Pitfall 4: Not Handling 'Cancel' or 'Start Over'**
*   **Description:** The user gets stuck or changes their mind mid-conversation and types "cancel", "never mind", or "start over". The chatbot either ignores it (if it doesn't match any intent) or, worse, tries to interpret it as a value for the current missing slot.
*   **Why it's Wrong:** It violates user autonomy and creates frustration. A fundamental principle of conversational design is to always provide a clear exit or reset path.
*   **Fix:** **Implement a global intent/command recognizer.** Before running the standard intent classification, check the user's cleaned text against a list of global commands (`["cancel", "stop", "start over", "reset", "new request"]`). If a match is found, call `DialogState.reset()` for that session and generate a confirmation response (e.g., "Okay, let's start fresh. What can I help you with?").

### Implementation Guidance

**A. Technology Recommendations Table**

| Component | Simple Option (Recommended) | Advanced Option |
| :--- | :--- | :--- |
| **Session Store** | Python `dict` in memory | **Redis** with `redis-py` library for persistence & scaling |
| **State Machine** | Explicit `if-elif` logic on `DialogState` attributes | **Transitions** library (a lightweight FSM library) |
| **Session Cleanup** | Lazy expiration check on each `process_utterance` call | **Background thread** with `threading.Timer` or `sched` |

**B. Recommended File/Module Structure**
```
project-athena/
├── core/
│   ├── __init__.py
│   ├── intents.py                # Intent definitions
│   ├── dialog_state.py           # DialogState class definition
│   ├── dialog_manager.py         # DialogManager class (this component)
│   └── response_templates.py     # ResponseTemplate definitions
├── nlp/
│   ├── intent_classifier.py
│   └── entity_extractor.py
├── data/                         # Training data
└── app.py                        # Main application entry point (e.g., Flask server)
```

**C. Infrastructure Starter Code: `core/dialog_state.py`**
```python
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import uuid

from .intents import Intent  # Assume Intent is defined elsewhere

@dataclass
class DialogState:
    """Represents the state of a single conversation session."""
    session_id: str
    current_intent: Optional[Intent] = None
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    confirmation_required: bool = False
    last_activity: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict] = field(default_factory=list)

    def update_activity(self) -> None:
        """Update the last activity timestamp to now."""
        self.last_activity = datetime.now()

    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if the session has expired due to inactivity."""
        if not self.last_activity:
            return True
        expiry_time = self.last_activity + timedelta(seconds=timeout_seconds)
        return datetime.now() > expiry_time

    def reset(self) -> None:
        """Clear conversation context and return to initial state."""
        self.current_intent = None
        self.filled_slots.clear()
        self.missing_slots.clear()
        self.confirmation_required = False
        self.update_activity()
        # Optional: Archive or clear history
        # self.conversation_history.clear()
```

**D. Core Logic Skeleton Code: `core/dialog_manager.py`**
```python
import logging
from typing import Dict, Optional
from datetime import datetime

from .dialog_state import DialogState
from ..nlp.intent_classifier import IntentClassifier
from ..nlp.entity_extractor import EntityExtractor, SlotFiller

# Constants
SESSION_TIMEOUT_SECONDS = 300

class DialogManager:
    """Orchestrates conversations, tracks context, and manages dialog flow."""

    def __init__(self,
                 intent_classifier: IntentClassifier,
                 entity_extractor: EntityExtractor,
                 slot_filler: SlotFiller):
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor
        self.slot_filler = slot_filler
        self.sessions: Dict[str, DialogState] = {}
        self.logger = logging.getLogger(__name__)

    def _get_or_create_state(self, session_id: str) -> DialogState:
        """Retrieve existing session or create a new one. Handles expiration."""
        # TODO 1: Check if session_id exists in self.sessions
        # TODO 2: If it exists, get the DialogState object and check if it's expired using .is_expired(SESSION_TIMEOUT_SECONDS)
        # TODO 3: If expired, call state.reset() and update the last_activity
        # TODO 4: If it doesn't exist, create a new DialogState with the given session_id and store it in self.sessions
        # TODO 5: Return the (possibly new or reset) DialogState object
        pass

    def _handle_global_commands(self, text: str, state: DialogState) -> bool:
        """Check for global commands like 'reset' or 'cancel'. Returns True if handled."""
        reset_keywords = ["cancel", "reset", "start over", "new", "never mind"]
        cleaned = text.strip().lower()
        # TODO 1: Check if the cleaned text is in reset_keywords or starts with any of them
        # TODO 2: If yes, call state.reset(), log the action, and return True
        # TODO 3: Otherwise, return False
        pass

    def process_utterance(self, session_id: str, user_input: str) -> DialogState:
        """
        The main entry point. Processes a user message within a session.
        Returns the updated DialogState.
        """
        # Step A: Get session state (handles creation/expiry)
        state = self._get_or_create_state(session_id)
        state.update_activity()

        # Step B: Check for global commands (e.g., "reset")
        if self._handle_global_commands(user_input, state):
            return state  # State has been reset, ready for new intent

        # Step C: Classify Intent
        # TODO 1: Use self.intent_classifier to predict intent and confidence from user_input
        # TODO 2: If confidence is below INTENT_CONFIDENCE_THRESHOLD, set predicted_intent to None
        # TODO 3: Log the prediction result

        # Step D: Intent Handling Logic (State Machine Core)
        # TODO 4: Check state.current_intent.
        #   - If it's None (IDLE) and we have a new predicted_intent:
        #        * Set state.current_intent = predicted_intent
        #        * Set state.missing_slots = predicted_intent.required_slots copy
        #        * Proceed to slot filling (Step E)
        #   - If state.current_intent exists (we are in an active dialog):
        #        * Check if predicted_intent is different. If yes, consider intent switching (optional advanced feature).
        #        * For now, assume we stay on the current intent. Proceed to slot filling (Step E).

        # Step E: Extract Entities & Fill Slots
        # TODO 5: Use self.entity_extractor.extract_entities(user_input) to get entities
        # TODO 6: Use self.slot_filler.fill_slots(...) to map entities to the current intent's slots and update state.filled_slots
        # TODO 7: Update state.missing_slots by removing any slots that were just filled

        # Step F: Determine Next Action & Update State
        # TODO 8: Check if state.missing_slots is empty.
        #   - If empty: All required slots are filled. Set state.confirmation_required = True if needed (e.g., based on intent or low-confidence slot values).
        #   - If not empty: The next action is to prompt for the first missing slot.

        # Step G: Log this turn in conversation_history
        # TODO 9: Append a dictionary to state.conversation_history with relevant info: timestamp, user_input, detected intent, extracted entities, etc.

        self.logger.debug(f"State for session {session_id}: {state}")
        return state
```

**E. Language-Specific Hints**

*   **Use `defaultdict` or `dict.setdefault` for convenience** when managing the `sessions` dictionary, but our skeleton uses explicit checks for clarity.
*   **Thread Safety:** If you plan to serve multiple users concurrently (e.g., with a web server like Flask), the `self.sessions` dictionary is a shared resource. Use `threading.Lock` to prevent race conditions when creating or accessing sessions.
    ```python
    from threading import Lock
    class DialogManager:
        def __init__(self, ...):
            self._session_lock = Lock()
        def _get_or_create_state(self, session_id: str):
            with self._session_lock:
                # ... session access logic ...
    ```
*   **Serialization for Debugging:** The `DialogState` is a dataclass. You can easily print it or log it using `dataclasses.asdict(state)` for a clean dictionary representation.

**F. Milestone Checkpoint**

After implementing the `DialogManager` and integrating it with your existing `IntentClassifier` and `EntityExtractor`:

1.  **Run a test script** that simulates a multi-turn conversation:
    ```python
    # test_dialog.py
    dm = DialogManager(classifier, extractor, slot_filler)
    session = "test_user_1"

    state = dm.process_utterance(session, "I want to book a flight")
    print(f"State1 - Intent: {state.current_intent.name}, Missing Slots: {state.missing_slots}")
    # Should show: Intent: book_flight, Missing Slots: ['destination', 'date']

    state = dm.process_utterance(session, "to Paris")
    print(f"State2 - Filled: {state.filled_slots}, Missing: {state.missing_slots}")
    # Should show: Filled: {'destination': 'Paris'}, Missing: ['date']

    state = dm.process_utterance(session, "cancel")
    print(f"State3 - Intent after cancel: {state.current_intent}")
    # Should show: Intent after cancel: None (state reset)
    ```

2.  **Expected Behavior:** The state should persist across calls with the same `session_id`, slots should be filled as entities are provided, and the `missing_slots` list should shrink accordingly. The "cancel" command should reset the state.

3.  **Signs of Trouble:**
    *   **State doesn't persist:** Check that `_get_or_create_state` is correctly storing and returning the same `DialogState` object from `self.sessions`.
    *   **Slots aren't filled:** Ensure the `SlotFiller.fill_slots` method is being called correctly and that its logic for mapping entity types to slot names is correct for your intent definition.
    *   **Infinite prompt loop:** Verify that `state.missing_slots` is being updated after slot filling. If an entity isn't extracted, the slot should remain missing, but you should have a re-prompt limit (Pitfall 3).

---


## 5.4 Component: Response Generator
> **Milestone(s):** Milestone 4 (Response Generation)

The **Response Generator** is the chatbot's "voice." It is the final stage in the processing pipeline, responsible for transforming the structured internal representation of a dialog state—including a detected intent and any filled slots—into a fluent, natural language response that the user can understand. Its core challenge is to produce human-like, contextually appropriate text without relying on an LLM, while maintaining consistency, handling dynamic content, and gracefully managing situations where a perfect response cannot be generated.

### Mental Model: The Mad Libs Assembler

Think of the Response Generator as an automated **Mad Libs assembler**. In the classic word game, you are given a story template with blanks (e.g., "I went to the ____ and saw a ____."). You provide words (nouns, verbs) to fill those blanks, resulting in a unique, often humorous, sentence. The template provides the grammatical structure and overall meaning, while the fill-in words provide the specific, dynamic content.

Similarly, our Response Generator holds a library of pre-written sentence templates for each possible conversational situation (e.g., greeting, confirming a booking, asking for a missing date). Each template contains placeholders (slots) marked by a special syntax (e.g., `{location}`). When it's time to respond, the generator:
1.  **Selects** an appropriate template based on the current dialog state (intent and which slots are filled).
2.  **Fetches** the required data (slot values) from the `DialogState`.
3.  **Fills** the placeholders with that data.
4.  **Presents** the completed, grammatically correct sentence to the user.

This model ensures control over the bot's tone and output quality while allowing for dynamic, personalized responses. The "Mad Libs" analogy underscores the separation of concerns: the `DialogManager` decides *what* to say (which template to use), and the `ResponseGenerator` handles *how* to say it (assembling the final text).

### Interface and Template Design

The Response Generator's interface is simple: given the current state of the conversation, produce a text response. Its internal complexity lies in template management, selection logic, and content insertion.

#### Core Interface Method

The primary method is a single function that orchestrates the entire response generation process.

| Method Name | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `generate_response` | `state: DialogState` | `str` (the response text) | The main public method. Given the current dialog state, selects the appropriate response template, fills its variables with slot values from the state, and returns the completed response string. This method also handles fallback scenarios (e.g., unknown intent, low confidence) by selecting a generic fallback template. |

#### Response Template Data Structure

Templates are defined using the `ResponseTemplate` type. A collection of these templates forms the response "knowledge base" for the chatbot.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `template_string` | `str` | The raw template text containing placeholders in a defined syntax (e.g., `"I've booked a table for {party_size} at {restaurant} on {date}."`). |
| `required_slots` | `List[str]` | A list of slot names that **must** be present in the `DialogState.filled_slots` for this template to be usable. These names correspond to the placeholders in `template_string`. |
| `variation_group` | `str` | An identifier used to group multiple templates that express the same semantic meaning in different ways (e.g., `"greeting_variants"`). This allows for random selection among alternatives to make the conversation feel less robotic. |

#### Template Selection and Resolution Algorithm

The logic for choosing and populating a response follows a deterministic, prioritized flow. This algorithm is executed by the `generate_response` method.

1.  **Determine Response Intent:** Inspect the `DialogState`. The primary driver is the `current_intent`. However, special states like `confirmation_required=True` or an empty intent (`None`) trigger different template categories (e.g., confirmation prompts, fallback messages).
2.  **Filter Candidate Templates:** From the global template registry, filter all `ResponseTemplate` objects where the `required_slots` are a **subset** of the keys in `DialogState.filled_slots`. This ensures we only consider templates for which we have the necessary data to fill all placeholders.
3.  **Group by Variation Group:** Among the filtered candidates, group templates by their `variation_group`.
4.  **Select Final Template:**
    *   If the primary intent's group has templates, **randomly select one** from that group. This introduces natural variation.
    *   If no template is found for the primary intent (e.g., missing slots for all variants), check for a more generic template for that intent (perhaps with fewer required slots) or a template designed for asking clarifying questions (e.g., `"ask_for_{missing_slot}"`).
    *   If the `current_intent` is `None` (unknown intent) or confidence is below `INTENT_CONFIDENCE_THRESHOLD`, select a template from the `"fallback"` variation group.
5.  **Perform Variable Substitution:** For each slot name in the selected template's `required_slots`, retrieve its value from `DialogState.filled_slots`. Replace every occurrence of the placeholder `{slot_name}` in the `template_string` with the string representation of the slot value.
6.  **Return Final Text:** The fully substituted string is the final response to be sent to the user.

#### Template Registry Structure

Templates are typically loaded once at system startup from a structured file (e.g., YAML, JSON). This separation of content from logic is crucial for maintainability, allowing non-developers to edit bot responses.

**Example YAML Structure:**
```yaml
response_templates:
  - variation_group: "greeting"
    required_slots: []
    template_string: "Hello! How can I help you today?"
  - variation_group: "greeting"
    required_slots: []
    template_string: "Hi there! What can I do for you?"
  - variation_group: "book_restaurant_complete"
    required_slots: ["restaurant", "date", "party_size", "time"]
    template_string: "Great! I've reserved a table for {party_size} at {restaurant} on {date} at {time}."
  - variation_group: "ask_for_slot"
    required_slots: []
    template_string: "What {slot_name} do you have in mind?"
  - variation_group: "fallback"
    required_slots: []
    template_string: "I'm not sure I understand. Could you rephrase that?"
```

### ADR: String Templates vs. Templating Engine

> **Decision: Use Python's `str.format()` for Simple String Templates Over a Full-Fledged Templating Engine (Jinja2)**
>
> - **Context**: We need a reliable, lightweight method to insert dynamic values (slot data) into pre-defined response text. The primary requirements are string substitution and minimal logic (no loops, conditionals). The system is built by beginners and must remain simple to understand and debug.
> - **Options Considered**:
>     1.  **Manual String Concatenation/`f-strings`**: Building strings by directly concatenating literals and variables.
>     2.  **`str.format()` or `string.Template`**: Using Python's built-in string formatting methods with placeholder syntax.
>     3.  **Dedicated Templating Engine (Jinja2)**: Using a third-party library designed for complex template rendering.
> - **Decision**: Use `str.format()` with a dictionary of slot values. The template syntax will be `"{slot_name}"`.
> - **Rationale**:
>     *   **Simplicity & Learning Curve**: `str.format()` is a built-in, well-understood Python feature. Beginners can grasp the `{key}` substitution model quickly. Jinja2 introduces a new syntax and concepts (filters, inheritance) that are overkill for our use case.
>     *   **Sufficiency**: Our needs are strictly substitutional. We do not require the loops, conditionals, or template inheritance that justify Jinja2's complexity.
>     *   **Safety**: `str.format()` with a restricted dictionary of known keys (slot values) is safe from unintended code execution, unlike approaches that might use `eval()`.
>     *   **Performance**: Built-in string operations are extremely fast and have no external dependencies.
> - **Consequences**:
>     *   **Positive**: The system is easier to implement, test, and explain. There are no additional library dependencies to manage.
>     *   **Negative**: If future requirements evolve to need conditional phrasing within a template (e.g., "I found {count} result" vs "I found {count} results"), we would need to implement that logic outside the template or re-evaluate this decision. For now, such cases can be handled by having multiple, separate templates.

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **Manual Concatenation** | Ultimate control, no syntax to learn. | Error-prone, difficult to read and maintain, breaks easily with grammar changes. | ❌ |
| **`str.format()`** | Simple, built-in, safe, performant. Clear `{key}` syntax. | Limited logic (no conditionals/loops in-template). | ✅ |
| **Jinja2** | Extremely powerful, supports complex logic, inheritance, filters. | Heavy overkill for our needs, steeper learning curve, external dependency. | ❌ |

### Common Pitfalls and Mitigations

⚠️ **Pitfall 1: Missing Slot Values Causing KeyErrors**
*   **Description**: A template requires a `{date}` slot, but the `DialogState.filled_slots` dictionary does not contain a `"date"` key. When `str.format()` tries to substitute it, a `KeyError` is raised, crashing the response generation.
*   **Why it's wrong**: The chatbot fails to respond, leading to a poor user experience and an unhandled exception in the application.
*   **Mitigation**: The template **selection** algorithm (Step 2 & 4 in the algorithm) must be robust. A template should only be selected if *all* its `required_slots` are present in the state's `filled_slots`. This is a pre-condition check. Additionally, wrap the final `format()` call in a try-except block as a safety net, falling back to a generic error template.

⚠️ **Pitfall 2: Non-String Slot Values Breaking Grammar**
*   **Description**: A slot value is a Python `datetime` object or an integer. Directly substituting it into a template like `"Booked for {party_size}"` results in `"Booked for 4"` which is fine, but `"Booked for {date}"` might become `"Booked for 2024-12-05 19:30:00"`, which is not a natural way to express a date.
*   **Why it's wrong**: The response is technically correct but feels robotic and poorly formatted. It violates the "natural language" goal.
*   **Mitigation**: Implement a **slot value formatting** step before substitution. The `EntityNormalizer` (from Milestone 2) should store a canonical object, but the `ResponseGenerator` should have a helper function to convert common types (`datetime`, `int`, `float`) into user-friendly strings (e.g., `"Thursday, December 5th at 7:30 PM"`). This is part of **entity normalization**.

⚠️ **Pitfall 3: Too Few Template Variations Leading to Repetition**
*   **Description**: Every time the user says "hello," the bot responds with the exact same phrase: `"Hello! How can I help you?"` This quickly makes the bot feel mechanical and un-engaging.
*   **Why it's wrong**: A key aspect of natural conversation is variation. Repetition breaks the illusion of a conversational partner.
*   **Mitigation**: Design the template system with `variation_group` from the start. For every common response scenario, author 3-5 alternative phrasings. The random selection among a group provides necessary variation with minimal complexity.

⚠️ **Pitfall 4: Inconsistent Tone Across Templates**
*   **Description**: One template uses casual language (`"Hey! What's up?"`) while another is very formal (`"Greetings. Please state your inquiry."`). This creates a jarring, inconsistent personality for the chatbot.
*   **Why it's wrong**: Brand and user experience rely on consistency. An inconsistent tone feels unprofessional and confusing.
*   **Mitigation**: Define a clear **persona guide** for the chatbot (e.g., "helpful and professional concierge") before writing any templates. Review all templates against this guide. Consider creating a "tone checklist" (e.g., use contractions, avoid slang, be proactive) to apply during template authoring.

⚠️ **Pitfall 5: Unescaped User Input in Responses (Indirect XSS)**
*   **Description**: While our system doesn't directly echo *raw* user input, a slot value originating from user text (e.g., a `restaurant` name they typed) is inserted into a response template. If a user entered malicious HTML/JavaScript as a restaurant name (`<script>alert('xss')</script>`), and the response is served in a web context without escaping, it could execute.
*   **Why it's wrong**: This is a security vulnerability (Cross-Site Scripting).
*   **Mitigation**: The response generator's output is typically consumed by a backend service. The **rendering layer** (e.g., the web server or frontend that displays the message) is ultimately responsible for proper output encoding. However, as a defensive practice, the `ResponseGenerator` could apply basic HTML escaping (`html.escape`) to any string slot value that originated from user text before substitution, if the final output medium is HTML. This should be documented clearly.

### Implementation Guidance

**A. Technology Recommendations Table**

| Component | Simple Option | Advanced Option |
| :--- | :--- | :--- |
| Template Storage & Loading | Python dictionaries defined directly in code or in a `.py` file. | YAML/JSON configuration files for easy editing without code changes. |
| Template Rendering Engine | Python's built-in `str.format()` method. | `string.Template` class for even simpler `$slot` syntax with stricter control. |
| Variation Selection | `random.choice()` from the Python standard library. | Weighted random selection (`random.choices` with weights) to favor more common phrasings. |

**B. Recommended File/Module Structure**
```
project-athena/
  data/
    responses.yaml          # All response templates (YAML format)
  src/
    core/
      models.py             # Contains Intent, Entity, DialogState, ResponseTemplate
    components/
      intent_classifier.py
      entity_extractor.py
      dialog_manager.py
      response_generator.py # <-- This component
    main.py                 # Application entry point
  tests/
    test_response_generator.py
```

**C. Infrastructure Starter Code (Complete)**

This is a helper for safely loading templates from a YAML file. It provides a simple registry.

```python
# src/utils/template_loader.py
import yaml
import random
from typing import List, Dict
from core.models import ResponseTemplate

class TemplateRegistry:
    """A simple registry to load and manage response templates."""
    
    def __init__(self, template_file_path: str):
        self.templates_by_group: Dict[str, List[ResponseTemplate]] = {}
        self._load_templates(template_file_path)
    
    def _load_templates(self, file_path: str) -> None:
        """Load templates from a YAML file into the registry."""
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            # If file doesn't exist, start with an empty registry
            data = {'response_templates': []}
        
        for template_data in data.get('response_templates', []):
            template = ResponseTemplate(
                template_string=template_data['template_string'],
                required_slots=template_data.get('required_slots', []),
                variation_group=template_data['variation_group']
            )
            # Group templates by their variation_group for easy lookup
            group = template.variation_group
            self.templates_by_group.setdefault(group, []).append(template)
    
    def get_templates_for_group(self, group: str) -> List[ResponseTemplate]:
        """Return all templates belonging to a specific variation group."""
        return self.templates_by_group.get(group, [])
    
    def get_random_template_for_group(self, group: str) -> ResponseTemplate:
        """Randomly select one template from a variation group."""
        templates = self.get_templates_for_group(group)
        if not templates:
            raise ValueError(f"No templates found for group: {group}")
        return random.choice(templates)
```

**D. Core Logic Skeleton Code**

This is the main `ResponseGenerator` class. Fill in the TODOs following the algorithm described in the "Template Selection and Resolution" section.

```python
# src/components/response_generator.py
import random
from typing import Optional
from core.models import DialogState, ResponseTemplate
from utils.template_loader import TemplateRegistry

class ResponseGenerator:
    """Generates natural language responses based on dialog state and templates."""
    
    def __init__(self, template_file_path: str):
        self.template_registry = TemplateRegistry(template_file_path)
        # Fallback groups for various error scenarios
        self.FALLBACK_GROUP = "fallback"
        self.CONFIRMATION_GROUP = "confirmation"
        self.ASK_SLOT_GROUP_PREFIX = "ask_for_"  # e.g., "ask_for_date"
    
    def generate_response(self, state: DialogState) -> str:
        """
        Main entry point. Generates a response string based on the current dialog state.
        
        Args:
            state: The current DialogState object.
            
        Returns:
            A fully formed natural language response string.
        """
        # TODO 1: Handle global commands or reset state.
        #   If state.current_intent is None and state.filled_slots is empty,
        #   we might be in an initial state. Consider selecting a greeting template.
        
        # TODO 2: Determine the primary intent and slot filling status.
        #   Extract state.current_intent.name and state.missing_slots.
        #   Check state.confirmation_required flag.
        
        # TODO 3: Select the appropriate template group.
        #   Logic:
        #   a) If state.confirmation_required is True -> use self.CONFIRMATION_GROUP.
        #   b) Else if state.missing_slots is not empty -> construct a group name
        #      like f"{self.ASK_SLOT_GROUP_PREFIX}{state.missing_slots[0]}".
        #      (We ask for the first missing slot).
        #   c) Else if state.current_intent is not None -> use the intent name
        #      as the group (e.g., "book_restaurant").
        #   d) Else -> use self.FALLBACK_GROUP (unknown intent).
        
        # TODO 4: Get candidate templates for the selected group.
        #   Use self.template_registry.get_templates_for_group(group_name).
        
        # TODO 5: Filter candidates based on required_slots.
        #   A template is viable only if ALL its required_slots are keys in
        #   state.filled_slots. Keep only viable templates.
        
        # TODO 6: If no viable templates exist for the primary group, fall back.
        #   Example: If asking for a slot but no specific template exists,
        #   fall back to a generic "ask_for_slot" template. Ultimately, use
        #   self.FALLBACK_GROUP.
        
        # TODO 7: Randomly select one template from the filtered list.
        #   Use random.choice(). If the filtered list is empty, select a random
        #   template from the fallback group.
        
        # TODO 8: Prepare slot values for substitution.
        #   For each required_slot in the selected template, get its value from
        #   state.filled_slots. Convert non-string values (like datetime) to
        #   user-friendly strings using a helper method (e.g., _format_slot_value).
        
        # TODO 9: Perform the substitution.
        #   Use the template's template_string and the prepared slot values dict.
        #   Use str.format() or a try/except block to catch any KeyErrors and
        #   fall back to a safe response.
        
        # TODO 10: Return the final response string.
        pass
    
    def _format_slot_value(self, value: any) -> str:
        """
        Converts a slot value (which might be a datetime, int, etc.) into a
        user-friendly string for insertion into a response template.
        
        Args:
            value: The raw slot value.
            
        Returns:
            A formatted string representation.
        """
        # TODO: Implement formatting for common types.
        #   - For datetime: use strftime to format as "Weekday, Month Day at HH:MM AM/PM".
        #   - For numbers: ensure they are converted to string.
        #   - For lists: join with commas and an 'and' before the last item.
        #   - Default: use str(value).
        pass
```

**E. Language-Specific Hints (Python)**
*   **YAML Loading**: Use `PyYAML` (`pip install pyyaml`). The `yaml.safe_load()` function prevents arbitrary code execution, which is safer than `yaml.load()`.
*   **String Formatting with Dictionaries**: `str.format(**slot_dict)` is perfect when your dictionary keys match the placeholder names exactly.
*   **Random Selection**: `random.choice(list)` is simple. For reproducible debugging, you can seed the random generator (`random.seed(42)`).
*   **Date Formatting**: Use `value.strftime("%A, %B %d at %I:%M %p")` for a friendly date string.
*   **Error Handling in `format()`**: Wrap the substitution in a try-except: `try: return template.format(**slots); except KeyError: return fallback_text`.

**F. Milestone Checkpoint**
After implementing the `ResponseGenerator`:
1.  **Create a Test Template File** (`data/responses.yaml`) with at least 3 variation groups (e.g., `greeting`, `book_restaurant_complete`, `fallback`).
2.  **Write a Simple Test Script** in `tests/test_response_generator.py` that:
    *   Instantiates the `ResponseGenerator` with your YAML file path.
    *   Creates a mock `DialogState` with a `current_intent` and some `filled_slots`.
    *   Calls `generate_response(state)` and prints the result.
3.  **Expected Behavior**: The printed response should be a coherent sentence with the slot values correctly inserted into the appropriate places in the template. If you change the slot values, the response should update accordingly. If you provide a state with a missing slot required by the primary template, the generator should fall back to an appropriate "ask for slot" or generic template.
4.  **Signs of Success**: No crashes. Responses are grammatically correct and contextually appropriate.
5.  **Common Debugging Issues**:
    *   **`KeyError` during `format()`**: Check that your template's `required_slots` list matches the placeholders in `template_string` and that those keys exist in `state.filled_slots` during selection.
    *   **No template found**: Verify the `variation_group` name in your YAML file matches the group name your code is trying to look up (e.g., intent name).
    *   **Incorrect date/number formatting**: Ensure your `_format_slot_value` method handles the data types you are using in your slots.


## 6. Interactions and Data Flow

> **Milestone(s):** Milestone 1, Milestone 2, Milestone 3, Milestone 4 (End-to-End Integration)

This section traces the complete journey of a user's message through the chatbot's processing pipeline—from raw text input to a meaningful response. Understanding these component hand-offs is critical for debugging, extending the system, and appreciating how modularity enables maintainability. We'll examine two fundamental conversational patterns: the **happy path** where a single message contains everything needed, and the **multi-turn dialogue** where information is gathered across multiple exchanges.

Think of the chatbot's data flow as an assembly line in a **custom workshop**. Raw materials (user text) enter the line, pass through specialized stations (components) where they're measured, shaped, and assembled, and emerge as a finished product (response). Each station adds specific value, and the conveyor belt (the `DialogManager`) coordinates the entire process, ensuring the right pieces move to the right places at the right times.

The system's component architecture, as shown in the diagram below, defines the flow:

![System Component Diagram](./diagrams/sys-component-diagram.svg)

### Happy Path: Complete Intent Fulfillment

The **happy path** represents the ideal conversational exchange: the user states their request clearly and completely in a single message, providing all required information. The chatbot understands the intent, extracts all necessary entities, and generates an appropriate, complete response. This scenario demonstrates the baseline capability of the system when no ambiguity or missing information exists.

**Mental Model: The Efficient Assembly Line**
Imagine a perfectly prepared customer walking into a coffee shop and stating: "I'd like a large latte with oat milk to go, please." The barista (the system) immediately understands the order (intent: `order_coffee`), notes all details (size: large, type: latte, milk: oat, format: to-go), and begins preparation. No follow-up questions are needed.

The step-by-step processing flow for a happy path utterance like "Book a table for two people tomorrow at 7 PM at The Bistro" proceeds as follows:

1.  **Input Reception & Session Management**: The `DialogManager.process_utterance(session_id, user_input)` method is invoked. The manager calls `_get_or_create_state(session_id)` to retrieve the existing `DialogState` for the user or create a new one. It updates the state's activity timestamp via `DialogState.update_activity()` and checks for expiration.

2.  **Global Command Check**: The manager passes the input text to `_handle_global_commands(text, state)`. Since the input is not a command like "reset" or "start over", this returns `False`, indicating normal processing should continue.

3.  **Intent Classification**: The raw text is sent to the `IntentClassifier`. The classifier performs text cleaning, TF-IDF vectorization, and probability prediction. It identifies the top intent as `book_restaurant` with a confidence of 0.92, which exceeds the `INTENT_CONFIDENCE_THRESHOLD` (0.7). A `UserUtterance` object is created with `intent` set to this `Intent` and `confidence` set to 0.92.

4.  **Entity Extraction**: The `UserUtterance.raw_text` is passed to the `EntityExtractor.extract_entities()` method. Using its hybrid approach, it identifies:
    *   A `number` entity with value "two" (normalized to integer 2) for party size.
    *   A `date` entity for "tomorrow" (normalized to a Python `date` object).
    *   A `time` entity for "7 PM" (normalized to a Python `time` object).
    *   A `location` entity for "The Bistro".
    These are stored as `Entity` objects in `UserUtterance.entities`.

5.  **Slot Filling & State Update**: The `SlotFiller.fill_slots(utterance, intent, state)` method is called. It uses the intent's `slot_mappings` (e.g., map entity type `number` to slot `party_size`) to populate `UserUtterance.slots`. The `DialogManager` then updates the `DialogState`:
    *   `current_intent` is set to the `book_restaurant` intent.
    *   `filled_slots` is updated with the extracted values: `{"party_size": 2, "date": <date_obj>, "time": <time_obj>, "restaurant_name": "The Bistro"}`.
    *   `missing_slots` is calculated by comparing `filled_slots` with the intent's `required_slots`. Since all required slots (`party_size`, `date`, `time`, `restaurant_name`) are now filled, `missing_slots` becomes an empty list.
    *   The `UserUtterance` (with its intent and entities) is appended to `conversation_history`.

6.  **Response Generation**: With all slots filled and no missing information, the `DialogManager` calls `generate_response(state)`. The `ResponseGenerator` inspects the `DialogState`:
    *   `current_intent.name` = `"book_restaurant"`
    *   `missing_slots` = `[]`
    *   This maps to the `intent_complete` variation group for the `book_restaurant` intent.
    *   `TemplateRegistry.get_random_template_for_group("book_restaurant_intent_complete")` selects a template like `"Great! I've booked a table for {party_size} at {restaurant_name} on {date} at {time}."`
    *   The generator calls `_format_slot_value()` for each slot (e.g., converting the date object to "2024-05-20") and performs **variable substitution**.
    *   The final response, `"Great! I've booked a table for 2 at The Bistro on 2024-05-20 at 19:00."`, is returned.

7.  **Output & State Finalization**: The `DialogManager` returns the updated `DialogState` (now ready for a new conversation) and the generated response text to the caller (e.g., a web server), which delivers it to the user. The session remains active, awaiting the next user input.

This sequence is summarized in the following table of component interactions:

| Processing Step | Component Involved | Input | Output | Key Action |
| :--- | :--- | :--- | :--- | :--- |
| **1. Session Init** | `DialogManager` | `session_id`, `"Book a table for two..."` | `DialogState` (active) | Retrieve/create state, update activity timestamp. |
| **2. Command Filter** | `DialogManager` | Raw text, `DialogState` | `False` (not a command) | Check for "reset", "cancel", etc. |
| **3. Intent Detection** | `IntentClassifier` | Raw text | `UserUtterance` with `intent=book_restaurant`, `confidence=0.92` | Vectorize, predict, apply confidence threshold. |
| **4. Detail Extraction** | `EntityExtractor` | Raw text | `UserUtterance.entities` list | Run regex patterns and spaCy NER, normalize values. |
| **5. Context Assembly** | `SlotFiller` & `DialogManager` | `UserUtterance`, `Intent`, `DialogState` | Updated `DialogState` with `filled_slots` complete | Map entities to slots, update state, clear `missing_slots`. |
| **6. Response Build** | `ResponseGenerator` | `DialogState` | `"Great! I've booked..."` | Select template, substitute slot values, format. |
| **7. Delivery** | `DialogManager` | Response string | (Returns state and response) | Pass response to output channel. |

### Multi-Turn: Slot Filling Dialogue

Real-world conversations are rarely complete in one turn. The **multi-turn dialogue** scenario showcases the chatbot's ability to manage **conversational context**—remembering the goal across exchanges and proactively asking for missing information. This is where the stateful nature of the `DialogManager` becomes essential.

**Mental Model: The Patient Interviewer**
Imagine a doctor's initial consultation. The patient says, "My head hurts." The doctor (the system) identifies the core issue (`report_symptom`) but needs more details to help. They ask a series of focused follow-up questions: "How long has it been hurting?" "Rate the pain from 1 to 10." The doctor remembers the original complaint (`head`) while collecting new details (`duration`, `severity`), only acting once the picture is complete.

Let's walk through a detailed example where a user wants to set a reminder but provides information incrementally.

**Turn 1: User provides intent but incomplete details.**

*   **User Input:** `"Set a reminder for tomorrow."`
*   **Processing Steps:**
    1.  **Intent Classification:** The classifier identifies `set_reminder` with high confidence.
    2.  **Entity Extraction:** Only a `date` entity for "tomorrow" is extracted. The `time` and `reminder_text` entities are missing.
    3.  **Slot Filling & State Update:** The `SlotFiller` maps the date entity to the `date` slot. The `DialogManager` updates the `DialogState`:
        *   `current_intent` = `set_reminder` (which has `required_slots = ["date", "time", "reminder_text"]`)
        *   `filled_slots` = `{"date": <date_obj>}`
        *   `missing_slots` = `["time", "reminder_text"]` (calculated by comparing `required_slots` with keys in `filled_slots`)
    4.  **Response Generation:** The `ResponseGenerator` sees a `current_intent` is set and `missing_slots` is not empty. It selects a template from the `prompt_for_slot` variation group. The system is designed to prompt for one missing slot at a time, typically starting with the first in the list. It chooses a template like `"For what time tomorrow?"` (intelligently incorporating the filled `date` slot for naturalness).
    5.  **Output:** The user sees: `"For what time tomorrow?"` The `DialogState` persists, holding the partial context.

**Turn 2: User provides the requested slot value.**

*   **User Input:** `"At 3 PM"`
*   **Processing Steps (Key Difference - Context Carryover):**
    1.  **Intent Classification (with Context):** The input "At 3 PM" is ambiguous alone. The `DialogManager` passes the *existing* `DialogState` to the classifier as context. The classifier, seeing an active intent in the state, may **skip full classification** and inherit the `current_intent` (`set_reminder`), or it may re-classify and get a low-confidence result, defaulting to the context's intent. This is **context carryover**.
    2.  **Entity Extraction:** A `time` entity for "3 PM" is extracted and normalized.
    3.  **Slot Filling & State Update:** The `SlotFiller` maps the new entity to the `time` slot. The `DialogManager` updates the `DialogState`:
        *   `filled_slots` = `{"date": <date_obj>, "time": <time_obj>}`
        *   `missing_slots` = `["reminder_text"]` (recalculated)
    4.  **Response Generation:** With `missing_slots` still not empty, the generator prompts for the next slot: `"What should the reminder say?"`
    5.  **Output:** User sees: `"What should the reminder say?"`

**Turn 3: User provides the final piece and changes mind.**

*   **User Input:** `"Actually, cancel that."`
*   **Processing Steps (Handling a Global Command Mid-Stream):**
    1.  **Global Command Check:** `_handle_global_commands("Actually, cancel that.", state)` detects the "cancel" keyword. It calls `DialogState.reset()` on the current state, clearing `current_intent`, `filled_slots`, and `missing_slots`.
    2.  **Early Exit:** Because a global command was processed, the normal intent/entity pipeline is bypassed.
    3.  **Response Generation:** The reset state has no intent. The generator selects a generic `fallback` or `acknowledge_reset` template: `"Okay, I've cancelled the reminder. What would you like to do next?"`
    4.  **Output:** User sees the cancellation acknowledgement, and the system returns to an idle state.

The flow of this multi-turn conversation is illustrated in the sequence diagram:

![Sequence Diagram: Multi-Turn Slot Filling](./diagrams/conversation-sequence-diagram.svg)

The `DialogManager`'s internal state machine (referenced in Section 5.3) governs these transitions. The table below maps our example to the state machine transitions:

| Turn | User Input | Dialog State (Before → After) | State Machine Event | Action Taken |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `"Set a reminder..."` | `IDLE` → `COLLECTING_SLOTS` | `intent_detected` | Identify `set_reminder`, extract partial slots, `prompt_for_slot` (time). |
| 2 | `"At 3 PM"` | `COLLECTING_SLOTS` → `COLLECTING_SLOTS` | `slot_received` | Fill `time` slot, `prompt_for_slot` (reminder_text). |
| 3 | `"Actually, cancel that."` | `COLLECTING_SLOTS` → `IDLE` | `reset` | `DialogState.reset()`, generate reset acknowledgement. |

**Key Design Insight: Maintaining Conversational Coherence**
> The system's ability to engage in coherent multi-turn dialogues hinges on two principles: **1) Stateful Context Tracking**, where the `DialogState` object is the single source of truth for the conversation's progress; and **2) Incremental Slot Filling**, where the manager treats each new utterance as a potential contribution to an ongoing task rather than always as a fresh start. This allows users to interact naturally, using pronouns ("it"), ellipsis ("At 3 PM"), and corrections ("Actually..."), much like in human conversation.

### Implementation Guidance

This section provides concrete code structure and skeletons to implement the data flow described above, focusing on the `DialogManager` as the orchestrator.

#### A. Technology Recommendations Table

| Component | Simple Option (Beginner-Friendly) | Advanced Option (Scalable) |
| :--- | :--- | :--- |
| **Session Store** | Python `dict` in memory (per-process) | Redis with expiry for production-scale, multi-server deployment. |
| **State Persistence** | Pickle `DialogState` objects in memory. | Serialize to JSON for debugging/logging; use a database for crash recovery. |
| **Pipeline Orchestration** | Sequential function calls within `DialogManager.process_utterance`. | Async/await for I/O-bound components (e.g., calling external APIs for entity resolution). |
| **Context Carryover Logic** | Simple rule: if `state.current_intent` exists and new utterance is short/low-confidence, inherit intent. | Train a binary classifier to decide if a new utterance is a follow-up or a new intent. |

#### B. Recommended File/Module Structure

```
intent_based_chatbot/
├── main.py                          # Application entry point (e.g., Flask/CLI)
├── core/
│   ├── __init__.py
│   ├── data_models.py              # `Intent`, `Entity`, `UserUtterance`, `DialogState`, `ResponseTemplate`
│   ├── dialog_manager.py           # `DialogManager` class (orchestrator)
│   ├── intent_classifier.py        # `IntentClassifier` class
│   ├── entity_extractor.py         # `EntityExtractor` and `SlotFiller` classes
│   └── response_generator.py       # `ResponseGenerator` and `TemplateRegistry` classes
├── config/
│   ├── intents.yml                 # Intent definitions with required slots
│   ├── training_data/              # Example utterances per intent
│   ├── entity_patterns.json        # Regex patterns for entity extraction
│   └── response_templates.yml      # Template definitions
├── models/                         # Serialized ML models (TF-IDF vectorizer, classifier)
│   └── intent_classifier.pkl
├── tests/                          # Unit and integration tests
│   ├── test_dialog_flow.py
│   └── ...
└── utils/
    ├── session_store.py            # Abstract interface for session storage
    └── helpers.py                  # Text cleaning, date normalization, etc.
```

#### C. Infrastructure Starter Code: In-Memory Session Store

This simple, thread-safe session store is sufficient for learning and small-scale deployment.

```python
# core/session_store.py
import threading
import time
from typing import Dict, Optional
from core.data_models import DialogState

class InMemorySessionStore:
    """Simple thread-safe store for DialogState sessions."""
    def __init__(self, timeout_seconds: int = 300):
        self._store: Dict[str, DialogState] = {}
        self._lock = threading.RLock()
        self.timeout_seconds = timeout_seconds

    def get(self, session_id: str) -> Optional[DialogState]:
        """Retrieve a session if it exists and is not expired."""
        with self._lock:
            state = self._store.get(session_id)
            if state and state.is_expired(self.timeout_seconds):
                del self._store[session_id]
                return None
            return state

    def set(self, session_id: str, state: DialogState) -> None:
        """Store or update a session."""
        with self._lock:
            self._store[session_id] = state

    def delete(self, session_id: str) -> None:
        """Explicitly delete a session."""
        with self._lock:
            self._store.pop(session_id, None)

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        with self._lock:
            expired_keys = [
                sid for sid, state in self._store.items()
                if state.is_expired(self.timeout_seconds)
            ]
            for key in expired_keys:
                del self._store[key]
            return len(expired_keys)
```

#### D. Core Logic Skeleton: DialogManager.process_utterance

This is the heart of the interaction flow. The TODO comments map directly to the numbered steps in the "Happy Path" and "Multi-Turn" walkthroughs.

```python
# core/dialog_manager.py
import logging
from typing import Tuple, Optional
from core.data_models import DialogState, UserUtterance
from core.intent_classifier import IntentClassifier
from core.entity_extractor import EntityExtractor, SlotFiller
from core.response_generator import ResponseGenerator
from core.session_store import InMemorySessionStore

class DialogManager:
    def __init__(self,
                 intent_classifier: IntentClassifier,
                 entity_extractor: EntityExtractor,
                 slot_filler: SlotFiller,
                 response_generator: ResponseGenerator,
                 session_store: InMemorySessionStore):
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor
        self.slot_filler = slot_filler
        self.response_generator = response_generator
        self.session_store = session_store
        self.logger = logging.getLogger(__name__)

    def process_utterance(self, session_id: str, user_input: str) -> Tuple[DialogState, str]:
        """
        Main orchestrator method. Processes a user message and returns updated state and response.
        
        Args:
            session_id: Unique identifier for the conversation session.
            user_input: Raw text from the user.
            
        Returns:
            Tuple of (updated DialogState, response_text).
        """
        # TODO 1: Session Management - Retrieve or create the DialogState for this session_id.
        #   Call `self._get_or_create_state(session_id)`.
        #   Update the state's last_activity timestamp using `state.update_activity()`.

        # TODO 2: Global Command Handling - Check if the input is a global command (e.g., 'reset').
        #   Call `self._handle_global_commands(user_input, state)`.
        #   If it returns True, skip to step 8 (response generation) using the potentially reset state.

        # TODO 3: Intent Classification - Determine the user's intent from the input.
        #   Pass `user_input` and optionally the current `state` (for context) to `self.intent_classifier.predict`.
        #   Receive a UserUtterance object with `intent` and `confidence` populated.
        #   If confidence is below INTENT_CONFIDENCE_THRESHOLD, set utterance.intent to None (unknown intent).

        # TODO 4: Context Carryover Logic - Decide whether to inherit intent from context.
        #   Condition: If `utterance.intent is None` AND `state.current_intent is not None` AND `len(user_input.split()) < 5` (short follow-up).
        #   Action: Set `utterance.intent = state.current_intent` and log this decision.

        # TODO 5: Entity Extraction - Extract details from the text.
        #   If `utterance.intent` is not None, call `self.entity_extractor.extract_entities(user_input)`.
        #   Attach the returned list of Entity objects to `utterance.entities`.

        # TODO 6: Slot Filling & State Update - Integrate new information into the conversation context.
        #   If `utterance.intent` is not None:
        #       a. Call `self.slot_filler.fill_slots(utterance, utterance.intent, state)`.
        #       b. This method updates `utterance.slots` and modifies `state.filled_slots`.
        #   Update `state.current_intent` to `utterance.intent` (may be the same or new).
        #   Recalculate `state.missing_slots` by comparing `state.current_intent.required_slots` with `state.filled_slots.keys()`.
        #   Append the processed `utterance` to `state.conversation_history`.

        # TODO 7: Session Persistence - Save the updated state back to the session store.
        #   Call `self.session_store.set(session_id, state)`.

        # TODO 8: Response Generation - Construct a natural language reply based on the updated state.
        #   Call `self.response_generator.generate_response(state)`.
        #   This method selects a template based on `state.current_intent` and `state.missing_slots`.

        # TODO 9: Return Results - Return the updated state and the generated response string.
        #   Return (state, response_text)

        pass  # Remove this line when implementing

    def _get_or_create_state(self, session_id: str) -> DialogState:
        """Internal helper to retrieve or create a session state."""
        # TODO 1: Try to get the state from `self.session_store.get(session_id)`.
        # TODO 2: If state is None (expired or new session), create a new DialogState with the given session_id.
        # TODO 3: Return the state.
        pass

    def _handle_global_commands(self, text: str, state: DialogState) -> bool:
        """Check for global commands like reset/cancel. Returns True if a command was processed."""
        # TODO 1: Normalize the text (lowercase, strip).
        # TODO 2: Check for keywords: 'reset', 'start over', 'cancel', 'exit'.
        # TODO 3: If a command is found:
        #   - Call `state.reset()` to clear the conversation context.
        #   - Log the command.
        #   - Return True.
        # TODO 4: Otherwise, return False.
        pass
```

#### E. Language-Specific Hints (Python)

*   **Thread Safety:** The provided `InMemorySessionStore` uses `threading.RLock`. If using an async framework (e.g., FastAPI), consider using `asyncio.Lock` and `dict`.
*   **State Serialization:** For debugging, you can add a `to_dict()` method to `DialogState` using `dataclasses.asdict()` if it's a dataclass, or implement custom JSON serialization for non-serializable fields (like `datetime` objects).
*   **Logging:** Use structured logging (`logging.getLogger(__name__)`) within each component. Log key decision points: intent classified (with confidence), entities extracted, slot updates, and template selection.
*   **Performance:** For production, consider caching the loaded spaCy model and TF-IDF vectorizer to avoid reloading on every request.

#### F. Milestone Checkpoint: End-to-End Flow

After implementing the `DialogManager` and integrating all components:

1.  **Run the Integration Test:**
    ```bash
    python -m pytest tests/test_dialog_flow.py -v
    ```
2.  **Expected Behavior:** Tests should pass for both happy path and multi-turn scenarios. Look for log output showing the sequence of component calls.
3.  **Manual Verification:** Write a simple script or use the provided CLI in `main.py`:
    ```python
    # test_script.py
    from core.dialog_manager import DialogManager
    # ... initialize all components and the manager
    session = "test_user_123"
    state, resp1 = manager.process_utterance(session, "Set a reminder for tomorrow.")
    print(f"Bot: {resp1}")  # Should prompt for time
    state, resp2 = manager.process_utterance(session, "At 3 PM")
    print(f"Bot: {resp2}")  # Should prompt for reminder text
    state, resp3 = manager.process_utterance(session, "Call mom")
    print(f"Bot: {resp3}")  # Should finalize or acknowledge?
    ```
    Verify that the state is correctly maintained across turns and that the bot's responses are contextually appropriate.
4.  **Signs of Trouble:**
    *   **State doesn't persist:** The session store is not being saved/retrieved correctly. Check the `session_id` is consistent and the `set()`/`get()` methods work.
    *   **Intent is forgotten on second turn:** Context carryover logic is missing or flawed. Check step 4 in `process_utterance`.
    *   **Bot asks for the same slot repeatedly:** The `missing_slots` list is not being updated correctly after a slot is filled. Debug the `SlotFiller.fill_slots` logic and the `missing_slots` recalculation.

---


## 7. Error Handling and Edge Cases

> **Milestone(s):** Milestone 1, Milestone 2, Milestone 3, Milestone 4 (Cross-Cutting Reliability)

A robust chatbot must gracefully handle the messy reality of human conversation. Unlike deterministic systems, users will provide ambiguous, incomplete, contradictory, or unexpected input. This section documents the anticipated failure modes and edge cases in the intent-based chatbot system, along with concrete strategies for detection, recovery, and graceful degradation. Proper error handling transforms a fragile prototype into a reliable system that maintains user trust even when it cannot provide a perfect answer.

The core philosophy is **defense in depth**: each component in the `Modular Pipeline` has its own validation and fallback mechanisms, with the `Dialog Manager` serving as the final safety net that orchestrates recovery. We'll examine failures at each layer—from low-confidence intent predictions to expired conversational context—and define clear pathways to either resolve the issue or guide the user toward a successful interaction.

### Common Failure Modes and Recovery

Think of the chatbot as a **help desk agent**. Even the best agent occasionally misunderstands a request, needs to ask for clarification, or must escalate when a problem is outside their expertise. The system's failure recovery mechanisms are its playbook for these situations: standardized procedures to get the conversation back on track.

The following table catalogs the most common failure modes, how they are detected by the system, and the specific recovery actions taken. Each failure is mapped to the component where detection primarily occurs.

| Failure Mode | Primary Detection Component | Detection Mechanism | Recovery Action |
|--------------|----------------------------|----------------------|-----------------|
| **Low-confidence intent prediction** | `IntentClassifier` | `confidence` score from `predict()` compared against `INTENT_CONFIDENCE_THRESHOLD` (0.7) | Classify utterance as `"unknown"` intent; `Dialog Manager` triggers fallback response asking for rephrasing. |
| **Ambiguous input (multiple intents with similar confidence)** | `IntentClassifier` | Top two predicted intents have confidence difference < 0.1 (configurable ambiguity threshold) | `Dialog Manager` enters a disambiguation state, asking the user to clarify which intent they meant (e.g., "Did you want to book a flight or check flight status?"). |
| **Missing required slot for detected intent** | `Dialog Manager` | After `SlotFiller.fill_slots()`, `DialogState.missing_slots` list is non-empty. | `Dialog Manager` transitions to `COLLECTING_SLOTS` state and generates a specific prompt for the next missing slot (e.g., "What date would you like to book?"). |
| **Overlapping or conflicting entity spans** | `EntityExtractor` | `_resolve_overlaps()` method detects multiple `Entity` objects with overlapping `start`/`end` indices in the same text. | Apply a **priority rule**: keep the entity with higher `confidence`, or if tied, keep the longer span. Discard the lower-priority entity. Log the conflict for analysis. |
| **Unparseable or ambiguous entity value** | `EntityExtractor._normalize_entities()` | Normalization logic fails to convert raw text (e.g., "next Tuesday" has unclear date reference without context). | Store the raw text in `Entity.normalized` and flag `confidence` as low (e.g., 0.5). `Dialog Manager` may later ask for confirmation (e.g., "Did you mean Tuesday, March 12th?"). |
| **Session expiration due to inactivity** | `InMemorySessionStore.get()` | `DialogState.is_expired(SESSION_TIMEOUT_SECONDS)` returns `True` on retrieval attempt. | Discard the expired `DialogState`. Create a new session with fresh state. Response informs user the conversation has timed out and asks them to start over. |
| **Unexpected global command mid-conversation** | `DialogManager._handle_global_commands()` | User input matches a reserved keyword (e.g., "reset", "start over", "help") regardless of current dialog state. | Immediately execute the command (e.g., `state.reset()`), bypassing normal intent classification. Provide a confirmation response (e.g., "Okay, let's start fresh."). |
| **Template variable substitution error** | `ResponseGenerator.generate_response()` | Placeholder in `ResponseTemplate.template_string` (e.g., `{location}`) does not exist in `DialogState.filled_slots`. | Fall back to a generic template for the intent/state, or use a safe default (e.g., "that location"). Log the error for template debugging. |
| **Unhandled exception in any component** | Wrapper function in `DialogManager.process_utterance()` | `try/except` block catches exceptions (e.g., `spacy` model not loaded, file I/O error). | Return a generic system error response (e.g., "I'm having trouble processing that right now."), log the full exception with `session_id` for debugging, and maintain session state if possible. |

> **Key Design Insight:** Recovery strategies follow an **escalation ladder**. First, attempt to resolve the issue within the current component (e.g., entity overlap resolution). If impossible, escalate to the `Dialog Manager` to engage the user (e.g., ask for clarification). Finally, if the system is fundamentally stuck, reset the conversation gracefully. This minimizes user frustration.

Let's examine three critical failure modes in greater detail with walk-through examples:

**1. Low-Confidence Intent Prediction:** The `IntentClassifier.predict()` method always returns a `confidence` score between 0 and 1. When the highest confidence is below the `INTENT_CONFIDENCE_THRESHOLD` (0.7), the classifier labels the intent as `"unknown"`. The `Dialog Manager`, upon seeing an `"unknown"` intent, will not proceed with entity extraction or slot filling. Instead, it will invoke a fallback response from the `Response Generator`, such as "I'm not sure I understand. Could you rephrase that?" This prevents the system from taking incorrect actions based on shaky guesses.

**2. Missing Required Slot Scenario:** Consider the `book_flight` intent with `required_slots: ["destination", "date"]`. A user says, "I want to fly to Paris." The `IntentClassifier` correctly identifies `book_flight`. The `EntityExtractor` finds an entity of type `"location"` with value `"Paris"`, which the `SlotFiller` maps to the `"destination"` slot. The `DialogState` now has `filled_slots: {"destination": "Paris"}`, and `missing_slots: ["date"]`. The `Dialog Manager`, in the `COLLECTING_SLOTS` state, triggers a response template that asks for the missing information: "When would you like to travel?"

**3. Session Expiration Recovery:** A user starts a booking conversation but leaves for 10 minutes (`SESSION_TIMEOUT_SECONDS` is 300). When they return and say "actually, make it tomorrow," the `InMemorySessionStore.get()` retrieves their `DialogState` and immediately checks `is_expired(300)`. Finding the `last_activity` timestamp is older than 5 minutes, the store discards the state and returns `None`. The `Dialog Manager` creates a new `DialogState` for the session. The user's utterance is processed as a fresh input, likely resulting in an `"unknown"` intent because "actually, make it tomorrow" lacks context. The bot responds with a timeout message: "Our conversation has timed out. Please start over with your request." This prevents the confusion of trying to continue a stale conversation.

### Edge Case Handling Strategies

Beyond common failures, certain edge cases require proactive design to ensure smooth interaction. These scenarios test the system's boundaries and assumptions about user behavior. Handling them well is the hallmark of a polished conversational experience.

#### 1. Context Carryover and Context Switching

**Mental Model:** **Conversational Thread vs. New Topic** – Imagine you're speaking with someone who suddenly changes the subject. You must decide whether to follow the new topic or remind them of the previous unfinished discussion.

**Edge Case:** A user provides follow-up information without restating the intent (e.g., after being asked for a date, they say "tomorrow"). Conversely, a user abruptly changes the topic mid-conversation (e.g., while booking a flight, they ask "what's the weather?").

**Strategy:**
- **Context Carryover:** The `DialogManager` maintains the `current_intent` in the `DialogState`. If the `IntentClassifier.predict()` is called with a `state` that has a `current_intent` and the new utterance is short (e.g., fewer than 4 words) and classified with low confidence, the system assumes the utterance is a continuation. It bypasses intent re-classification and uses the existing `current_intent`. This allows for natural follow-ups like "tomorrow" or "for two people."
- **Context Switching:** The `_handle_global_commands()` method always runs first. If the user input contains strong indicators of a new intent (e.g., high-confidence prediction of a different intent) or explicit reset commands, the `Dialog Manager` will call `state.reset()` and process the new input afresh. A mid-conversation question like "what's the weather?" should trigger a high-confidence `weather_inquiry` intent, causing the system to reset and handle the new request, potentially after a brief confirmation ("I'll help with the weather. This will cancel your ongoing booking. Proceed?").

#### 2. Entity Ambiguity and Normalization Conflicts

**Mental Model:** **The Interpreter for Ambiguous Phrases** – Like an interpreter hearing "next Tuesday" on a Monday, the system must resolve relative references based on a known reference point (today's date) and flag any remaining ambiguity.

**Edge Case:** Date expressions like "01/02/2023" (MM/DD vs DD/MM), relative times like "next week", or plural entities like "two adults and one child" need to be normalized to a standard format. Different `EntityExtractor` patterns or the spaCy model might also produce conflicting interpretations.

**Strategy:**
- **Normalization with Reference Date:** The `EntityNormalizer` for date types uses a reference date (typically the current date) to resolve "next Tuesday" to an absolute date. It logs the assumption made.
- **Format Preference Configuration:** A configuration setting (e.g., `DATE_ORDER = "MDY"`) dictates the interpretation of numeric dates. If ambiguity remains (e.g., "01/02"), the system defaults to the configured order but sets a low `confidence` score (e.g., 0.6).
- **Confidence-Based Confirmation:** When an extracted entity has a `confidence` below a threshold (e.g., 0.8) after normalization, the `Dialog Manager` can set `confirmation_required = True` in the `DialogState`. The next response will ask for verification: "Just to confirm, you meant January 2nd, 2023, right?"

#### 3. Multi-Intent and Compound Requests

**Mental Model:** **The Multi-Tasking Assistant** – A user might ask two things at once, like "book a flight to Paris and also rent a car." A human assistant would acknowledge both requests and handle them sequentially.

**Edge Case:** A single user utterance contains multiple distinct intents (e.g., "I want to book a flight and check my account balance").

**Strategy:**
- **Limited Multi-Intent Handling:** The current architecture is designed for a single primary intent per turn. To handle compound requests partially, we implement a **priority-based selection**. The `IntentClassifier` picks the intent with the highest confidence, and the system proceeds with that one. However, the `Response Generator` can include an acknowledgment of the secondary intent: "I'll help you book a flight first. After that, you can ask about your account balance." This is achieved by checking the top two intent confidence scores; if both are above threshold and close, the response template can include a tailored acknowledgment.
- **Future Extension Hook:** The `UserUtterance` data structure can be extended with a `secondary_intents` list to formally support multi-intent parsing in a future iteration.

#### 4. Handling Sarcasm, Negation, and Complex Language

**Mental Model:** **The Literal Interpreter with Safeguards** – The system primarily understands literal statements. It uses simple keyword triggers to detect common non-literal constructs that could lead to serious misunderstanding.

**Edge Case:** User employs sarcasm ("Oh great, another delay"), negation ("don't book that flight"), or complex conditional statements ("if it's rainy, book a hotel near the airport").

**Strategy:**
- **Negation Detection:** The `IntentClassifier` includes a pre-processing step that scans for negation keywords ("don't", "cancel", "not", "stop") near intent-related keywords. If detected, the system may route to a specific `cancel_intent` or respond with a clarifying question ("It sounds like you want to cancel. Is that right?").
- **Fallback for Complex Language:** Utterances with complex clauses typically result in low-confidence classification. The system will trigger the `"unknown"` intent fallback, prompting the user to rephrase more simply. This is an acceptable limitation for a non-LLM system.
- **Sentiment Awareness (Basic):** A simple sentiment lexicon (positive/negative words) can be used to detect user frustration. If negative sentiment words are present in an `"unknown"` intent utterance, the fallback response can be more apologetic: "I'm sorry you're having trouble. Could you try saying that in a different way?"

#### 5. Slot Overwriting and Correction

**Mental Model:** **The Editable Form** – When filling out a form, you might go back and change a field. The system must allow corrections without restarting the entire process.

**Edge Case:** A user provides a value for a slot that is already filled (e.g., first says "to Paris", then later says "actually, to London").

**Strategy:**
- **Implicit Overwrite:** The `SlotFiller.fill_slots()` method writes entities into `DialogState.filled_slots` unconditionally. If a slot key already exists, the new value overwrites the old one. This simple approach handles natural corrections.
- **Explicit Correction Command:** The `_handle_global_commands()` can recognize phrases like "change destination to London" as a special command. It would clear the specific slot (`destination`) from `filled_slots`, add it back to `missing_slots`, then process "London" as a new entity to fill it. This provides a more intuitive correction flow.

#### 6. No Input or Empty Input

**Edge Case:** The user sends a message with only whitespace, or the input is an empty string.

**Strategy:**
- **Pre-processing Filter:** Before calling `IntentClassifier.predict()`, the input text is stripped of whitespace. If the resulting string is empty, the `Dialog Manager` short-circuits the pipeline and returns a gentle prompt: "I didn't receive any message. Please type your request."

> **Architecture Decision Record (ADR): Graceful Degradation Over Silent Failure**
> - **Context:** The system must remain responsive and helpful even when internal components fail or produce low-quality outputs.
> - **Options Considered:**
>     1. **Silent Fallback to Default:** On error, use a generic response without informing the user of the issue.
>     2. **Technical Error Messages:** Expose internal error details (e.g., exception traces) to the user.
>     3. **Graceful Degradation with User Guidance:** Inform the user that something went wrong in a friendly way, suggest corrective actions, and log the technical details internally.
> - **Decision:** Implement **Graceful Degradation with User Guidance**.
> - **Rationale:** Silent failures erode user trust (they think the system ignored them). Technical messages confuse non-technical users. A friendly message acknowledges the problem, maintains the conversational tone, and provides a path forward (e.g., asking to rephrase), while detailed logs allow developers to debug.
> - **Consequences:** Requires designing a set of user-friendly error responses for different failure points. Adds slight complexity to ensure errors are caught and mapped to appropriate responses.

### Implementation Guidance

**A. Technology Recommendations Table:**

| Component | Simple Option (Beginner-Friendly) | Advanced Option (More Robust) |
|-----------|-----------------------------------|--------------------------------|
| Error Detection & Logging | Python's built-in `logging` module with different levels (INFO, WARNING, ERROR). | Structured logging with `structlog` or `loguru` for JSON-formatted logs, easier to query and analyze. |
| Session Expiration | `InMemorySessionStore` with periodic cleanup in a background thread (e.g., using `threading.Timer`). | External session store like Redis with built-in TTL (time-to-live) for automatic expiration. |
| Input Sanitization | Basic string trimming and length checks before processing. | More comprehensive sanitization to prevent injection attacks if templates are ever exposed to unsafe input (e.g., using `html.escape` for web interfaces). |

**B. Recommended File/Module Structure:**

Add error handling utilities and constants to the existing structure.
```
intent_based_chatbot/
│
├── core/
│   ├── __init__.py
│   ├── intent_classifier.py      # Contains confidence threshold check
│   ├── entity_extractor.py       # Contains overlap resolution & normalization
│   ├── dialog_manager.py         # Main orchestrator with try/except wrapper
│   ├── response_generator.py     # Handles template substitution errors
│   └── session_store.py          # Contains expiration logic
│
├── models/                       # Data models (UserUtterance, DialogState, etc.)
│
├── utils/
│   ├── __init__.py
│   ├── error_handlers.py         # Centralized error handling decorators/helpers
│   └── validators.py             # Input validation functions
│
├── config/
│   └── constants.py              # INTENT_CONFIDENCE_THRESHOLD, SESSION_TIMEOUT_SECONDS, etc.
│
└── templates/                    # Response templates, including error templates
    └── responses.yaml
```

**C. Infrastructure Starter Code (Complete Error Handler):**

Create a decorator to catch exceptions in component methods and log them appropriately.

```python
# utils/error_handlers.py
import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

def component_error_handler(func: Callable) -> Callable:
    """
    Decorator to catch exceptions in pipeline components, log them, and return a safe default.
    Used on critical methods like predict(), extract_entities(), process_utterance().
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the full exception with context
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            # Return a safe default based on the function's expected return type
            # This is a simplistic fallback; adjust per function.
            if func.__name__ == "predict":
                # Return an unknown intent UserUtterance
                from models import UserUtterance, Intent
                return UserUtterance(
                    raw_text=kwargs.get('text', ''),
                    cleaned_text=kwargs.get('text', ''),
                    intent=Intent(name="unknown", required_slots=[], optional_slots=[]),
                    confidence=0.0,
                    entities=[],
                    slots={}
                )
            elif func.__name__ == "extract_entities":
                return []  # Empty entity list on failure
            elif func.__name__ == "generate_response":
                return "I'm experiencing a technical issue. Please try again in a moment."
            else:
                raise  # Re-raise if we don't have a fallback
    return wrapper
```

**D. Core Logic Skeleton Code (Dialog Manager with Error Handling):**

Enhance the `DialogManager.process_utterance()` method with robust error handling and edge case checks.

```python
# core/dialog_manager.py
from models import DialogState
from utils.error_handlers import component_error_handler
import logging

logger = logging.getLogger(__name__)

class DialogManager:
    # ... other methods ...

    @component_error_handler
    def process_utterance(self, session_id: str, user_input: str) -> tuple[DialogState, str]:
        """
        Main orchestrator method with integrated error handling.
        Returns the updated dialog state and the response text.
        """
        # Step 1: Validate and clean input
        cleaned_input = user_input.strip()
        if not cleaned_input:
            return self._get_or_create_state(session_id), "I didn't receive any message. Please type your request."

        # Step 2: Retrieve or create session state
        state = self._get_or_create_state(session_id)

        # Step 3: Check for global commands (e.g., reset, help)
        if self._handle_global_commands(cleaned_input, state):
            # Command handled; state has been updated (e.g., reset)
            response = self.response_generator.generate_response(state)
            return state, response

        # Step 4: Intent Classification (with context for carryover)
        # TODO 1: Determine if context carryover should be attempted.
        #   Hint: Check if state.current_intent exists and if cleaned_input is short (e.g., <4 words).
        # TODO 2: If context carryover is appropriate, skip intent classification and use state.current_intent.
        #   Otherwise, call self.intent_classifier.predict(cleaned_input, state).
        # TODO 3: If the predicted intent is "unknown" or confidence is below threshold, generate a fallback response and return early.

        # Step 5: Entity Extraction and Slot Filling
        # TODO 4: Extract entities from cleaned_input using self.entity_extractor.extract_entities().
        # TODO 5: Fill slots in the utterance and update state using self.slot_filler.fill_slots().

        # Step 6: Update Dialog State and Determine Next Action
        # TODO 6: Update state.current_intent, state.filled_slots, state.missing_slots.
        # TODO 7: Check if any required slots are missing. If so, set state to COLLECTING_SLOTS.
        # TODO 8: If all required slots are filled, set state to READY_TO_EXECUTE and prepare for action (e.g., calling an API).

        # Step 7: Generate Response
        # TODO 9: Call self.response_generator.generate_response(state) to get the final response text.

        # Step 8: Update session activity timestamp and store state
        state.update_activity()
        self.session_store.set(session_id, state)

        return state, response

    def _handle_global_commands(self, text: str, state: DialogState) -> bool:
        """Check for and process global commands like 'reset'. Returns True if a command was handled."""
        # TODO 1: Normalize text to lowercase.
        # TODO 2: Check for commands like 'reset', 'start over', 'cancel', 'help'.
        # TODO 3: If 'reset' or similar, call state.reset() and return True.
        # TODO 4: If 'help', set a flag in state to trigger a help response and return True.
        # TODO 5: For other commands, return False.
        pass
```

**E. Language-Specific Hints (Python):**

- Use `try...except` blocks around external library calls (e.g., `spacy` model loading, file reading for templates).
- The `logging` module can be configured to write to a file with different log levels. Use `logging.error("msg", exc_info=True)` to include the exception traceback.
- For session expiration, use `datetime.utcnow()` for timestamps to avoid timezone issues.
- When using `@component_error_handler`, ensure the fallback return types match the original method's return type exactly to prevent downstream errors.

**F. Milestone Checkpoint for Error Handling:**

After implementing the error handling in each component, run a comprehensive test that simulates failures.

1. **Test Low-Confidence Intent:**
   - Input: A gibberish or out-of-domain sentence.
   - Expected Output: A fallback response like "I'm not sure I understand."
   - Command: Run your test suite or a manual test script that calls `intent_classifier.predict("asdfghjkl")` and verifies the returned intent name is `"unknown"` and confidence is below threshold.

2. **Test Session Expiration:**
   - Set `SESSION_TIMEOUT_SECONDS` to 10 seconds for testing.
   - Start a conversation, wait 15 seconds, send another message.
   - Expected Output: A timeout message and a fresh session.
   - Verify that the old `DialogState` is no longer in the `InMemorySessionStore._store`.

3. **Test Exception Handling:**
   - Temporarily rename your template YAML file to cause a `FileNotFoundError` in `TemplateRegistry._load_templates()`.
   - Start the bot. Expected: The system should log an error but still start, using a hard-coded default fallback response.

**G. Debugging Tips Table:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Every user input returns "I'm not sure I understand." | `INTENT_CONFIDENCE_THRESHOLD` set too high, or training data is insufficient. | Check the confidence scores logged by the `IntentClassifier`. | Lower the threshold temporarily, or add more varied training examples. |
| Session state seems to reset unexpectedly between requests. | Session ID is not being passed correctly, or `SESSION_TIMEOUT_SECONDS` is very low. | Log the `session_id` at the start of `process_utterance`. Check the timestamp comparison logic in `is_expired()`. | Ensure the client sends a consistent session ID (e.g., from a cookie). Verify timeout calculation uses same timezone (prefer UTC). |
| Entity extraction picks wrong part of text (e.g., matches "May" as a month in "Maybe tomorrow"). | Regex patterns are too greedy or not specific enough. | Log the raw text and all entities found before overlap resolution. | Refine regex patterns with word boundaries (`\b`) and context. Use more specific patterns or add negative lookahead/lookbehind. |
| Bot asks for the same slot repeatedly in a loop. | Slot filling logic is not updating `DialogState.filled_slots` correctly, or the entity type is not mapped to the slot. | Log the `filled_slots` and `missing_slots` after each `fill_slots()` call. Check the `slot_mappings` dictionary. | Ensure the entity type extracted matches a key in the slot mapping for the current intent. Debug the `fill_slots` method step-by-step. |
| Template error: "KeyError: 'location'" in response generation. | A response template uses a placeholder for a slot that hasn't been filled. | Check which template is being selected and its `required_slots` vs. the current `filled_slots`. | Add a default value for the placeholder in the template (e.g., `{location:that place}`), or ensure the slot is filled before reaching response generation. |


## 8. Testing Strategy

> **Milestone(s):** Milestone 1, Milestone 2, Milestone 3, Milestone 4 (Quality Assurance)
>
> Testing a conversational AI system presents unique challenges: understanding is probabilistic, user input is unpredictable, and system state evolves across multiple turns. This section outlines a **tiered testing strategy** that builds confidence from the ground up—starting with isolated component behavior, progressing through integration of the processing pipeline, and culminating in complete conversational flows. Each milestone includes **verification checkpoints** with concrete acceptance criteria to validate functional requirements.

### Testing Pyramid for Chatbot Systems

Think of testing a chatbot like **inspecting a multi-layered cake**. You examine each ingredient (unit tests), verify the layers bond together (integration tests), and finally taste the complete slice (system tests) to ensure the overall experience meets expectations. This layered approach ensures defects are caught early, when they're cheapest to fix, and that each component works correctly before being assembled into the larger system.

The chatbot's **modular pipeline architecture** naturally aligns with the testing pyramid. Each component has well-defined inputs and outputs, making them ideal for isolated testing. The pyramid consists of three primary layers:

| Layer | Scope | What It Validates | Tools/Techniques | Frequency |
|-------|-------|-------------------|------------------|-----------|
| **Unit Tests** | Individual functions/classes | Component correctness in isolation (e.g., TF-IDF vectorization, regex pattern matching, state transitions) | `pytest`, `unittest`, mocks | On every code change |
| **Integration Tests** | Interactions between 2+ components | Data flow and handoffs (e.g., intent → entity extraction → slot filling, session persistence across turns) | `pytest` with fixtures, component wiring | On feature completion |
| **System/Conversation Tests** | Entire pipeline with simulated user | End-to-end conversational flows, error handling, and multi-turn behavior | Scripted dialog flows, automated conversation runners | Before releases |

**1. Unit Testing: Verifying Individual Cogs in the Machine**

Unit tests focus on the smallest testable units—individual functions, methods, or classes. Each component's internal logic should be thoroughly validated with a comprehensive test suite.

**Intent Classifier Unit Tests** should verify:
- Vectorization produces correct TF-IDF features for sample texts
- Training produces a model that can predict known intents with high confidence
- The `predict` method returns a `UserUtterance` with the correct `intent` field and `confidence` score
- Low-confidence predictions (below `INTENT_CONFIDENCE_THRESHOLD`) are labeled as "unknown"
- The classifier handles edge cases like empty strings, very long inputs, and inputs with special characters

**Entity Extractor Unit Tests** should verify:
- Regex patterns correctly match and extract entities (dates, times, numbers)
- spaCy NER model identifies named entities (PERSON, DATE, GPE) when available
- The `_resolve_overlaps` method properly handles conflicting entity spans (e.g., "New York" vs "York")
- The `_normalize_entities` method converts extracted text to canonical formats (e.g., "tomorrow" → "2024-05-20")
- The `extract_entities` method returns a list of `Entity` objects with correct `type`, `value`, `start`, and `end` fields

**Dialog Manager Unit Tests** should verify:
- `DialogState.update_activity()` correctly updates the timestamp
- `DialogState.is_expired()` returns `True` after timeout period
- `DialogState.reset()` clears all context fields appropriately
- `DialogManager._handle_global_commands()` detects "reset" and "help" commands
- State transitions follow the expected state machine logic

**Response Generator Unit Tests** should verify:
- Template placeholders (`{slot_name}`) are correctly replaced with slot values
- `_format_slot_value()` formats different data types (dates, numbers, strings) appropriately
- `get_random_template_for_group()` returns different templates from the same variation group
- Missing slot values in templates are handled gracefully (either skipped or replaced with defaults)
- Fallback responses are generated when no matching template exists

> **Testing Principle:** Each unit test should be **fast, isolated, and deterministic**. Use mocking to replace external dependencies (like file I/O, spaCy model loading, or random number generation) with predictable behavior.

**2. Integration Testing: Ensuring Components Work Together**

Integration tests validate that data flows correctly between components. They catch issues with incompatible data formats, incorrect assumptions about interfaces, and timing problems.

**Key Integration Test Scenarios:**

| Test Scenario | Components Involved | What to Verify |
|---------------|---------------------|----------------|
| **Intent → Entity Extraction** | `IntentClassifier`, `EntityExtractor` | When intent is identified, the correct entity types are extracted based on intent requirements |
| **Entity → Slot Filling** | `EntityExtractor`, `SlotFiller`, `DialogManager` | Extracted entities are correctly mapped to slot names in the dialog state |
| **Multi-turn Context Preservation** | `DialogManager`, `InMemorySessionStore` | Dialog state persists across multiple `process_utterance` calls for the same `session_id` |
| **Full Pipeline Happy Path** | All components | A complete user message yields correct intent, entities, slot filling, and appropriate response |
| **Error Propagation** | All components | Low confidence or extraction errors trigger appropriate fallback responses |

**3. System/Conversation Testing: Validating Complete Dialog Flows**

System tests simulate real conversations between a user and the chatbot. These tests validate the end-to-end experience and are particularly important for verifying multi-turn interactions.

**Conversation Test Format:**
Each test is defined as a sequence of **turns**, where each turn has:
- **User Input**: The simulated user message
- **Expected System State**: The expected `DialogState` after processing (intent, filled slots, etc.)
- **Expected Response Pattern**: A regex pattern or key phrases the response should contain

**Example Conversation Test Table:**

| Turn | User Input | Expected Intent | Expected Filled Slots | Response Should Contain |
|------|------------|-----------------|----------------------|-------------------------|
| 1 | "Book a flight to Paris" | `book_flight` | `{"destination": "Paris"}` | "When would you like to travel?" |
| 2 | "Tomorrow" | `book_flight` (context) | `{"destination": "Paris", "date": "2024-05-20"}` | "From which city will you depart?" |
| 3 | "New York" | `book_flight` (context) | All required slots filled | Confirmation with all details |

**ADR: Automated Conversation Testing vs Manual Testing**

> **Decision: Scripted Conversation Testing with Validation Rules**
>
> - **Context**: We need to verify complex multi-turn dialog flows work correctly, but manually testing all possible conversation paths is time-consuming and error-prone.
> - **Options Considered**:
>   1. **Manual testing only**: Testers manually have conversations with the chatbot and verify responses.
>   2. **Record-and-replay**: Record successful conversations and replay them to detect regressions.
>   3. **Scripted conversation tests**: Write code that simulates user inputs and validates system responses against expectations.
> - **Decision**: Implement **scripted conversation tests** using a lightweight test framework.
> - **Rationale**: Scripted tests are reproducible, can be run automatically in CI/CD, catch regressions early, and document expected behavior. Manual testing is too slow for frequent regression testing, and record-and-replay lacks validation logic.
> - **Consequences**: Requires writing test scripts for each conversation flow, but provides high confidence in system behavior. Tests must be updated when dialog flows change.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Manual testing only | Flexible, human judgment for nuanced responses | Time-consuming, not reproducible, hard to scale | ❌ |
| Record-and-replay | Easy to create tests from real conversations | No validation logic, breaks on minor UI/text changes | ❌ |
| Scripted conversation tests | Reproducible, automatable, documents expected behavior | Requires writing test code, must update when flows change | ✅ |

### Milestone Verification Checkpoints

Each milestone in the project has specific acceptance criteria. The verification checkpoints below translate those criteria into concrete test scenarios and validation steps.

#### Milestone 1: Intent Classification

**Acceptance Test Suite** for intent classification focuses on accuracy, confidence handling, and unknown intent detection.

**1. Training Data Validation Tests:**
- Verify training data contains at least 10 examples per intent
- Check for class imbalance (no intent with 3x more examples than others)
- Validate data format (CSV/YAML with columns: text, intent)

**2. Classifier Accuracy Test:**
- Split data into 80% training, 20% testing
- Train classifier on training set
- Predict on test set and calculate accuracy
- **Checkpoint**: Accuracy ≥ 80% on held-out test set

**3. Confidence Threshold Verification:**
| Test Input | Expected Intent | Confidence Should Be | Notes |
|------------|-----------------|----------------------|-------|
| Clear examples from training data | Correct intent | > `INTENT_CONFIDENCE_THRESHOLD` | Verify high confidence |
| Ambiguous/out-of-domain phrases | "unknown" | < `INTENT_CONFIDENCE_THRESHOLD` | Verify fallback |
| Edge cases (empty string, single word) | "unknown" or appropriate intent | Any | Ensure no crashes |

**4. Multi-Intent Detection Test** (if implemented):
- Test inputs that match multiple intents (e.g., "book a flight and check weather" might match both `book_flight` and `check_weather`)
- Verify system either picks the highest confidence intent or handles multi-intent appropriately

**Verification Checklist for Milestone 1:**
- [ ] Training data passes validation (≥10 examples per intent, balanced)
- [ ] Classifier achieves ≥80% accuracy on test set
- [ ] Low-confidence predictions (<0.7) are labeled "unknown"
- [ ] Clear predictions have confidence >0.9
- [ ] Model handles edge cases without crashing
- [ ] `UserUtterance` objects are correctly populated with `intent` and `confidence`

#### Milestone 2: Entity Extraction

**Acceptance Test Suite** for entity extraction validates pattern matching, NER integration, slot filling, and normalization.

**1. Pattern-Based Entity Extraction Tests:**
Create a test table for each entity type with regex patterns:

| Entity Type | Test Input | Expected Value | Normalized Format | Notes |
|-------------|------------|----------------|-------------------|-------|
| `date` | "tomorrow" | "tomorrow" | "2024-05-20" | Relative date |
| `date` | "May 20, 2024" | "May 20, 2024" | "2024-05-20" | Absolute date |
| `time` | "3 PM" | "3 PM" | "15:00" | 12-hour format |
| `number` | "two tickets" | "two" | 2 | Word number |
| `duration` | "for 3 days" | "3 days" | "P3D" | ISO 8601 |

**2. NER Integration Tests:**
- Verify spaCy model loads successfully
- Test mapping of spaCy labels to our entity types:
  - `PERSON` → `person`
  - `GPE` → `location`  
  - `DATE` → `date`
  - `TIME` → `time`

**3. Slot Filling Integration Tests:**
| Test Scenario | Input Text | Intent | Expected Slots Filled |
|---------------|------------|--------|----------------------|
| Single entity | "Book flight to Paris" | `book_flight` | `{"destination": "Paris"}` |
| Multiple entities | "Fly from New York to London tomorrow" | `book_flight` | `{"origin": "New York", "destination": "London", "date": "2024-05-20"}` |
| Missing required entity | "Book a flight" | `book_flight` | `{}` (no slots filled) |

**4. Overlap Resolution Tests:**
Test that overlapping entities are resolved correctly:
- "New York City" should produce one `location` entity, not three separate ones
- When regex and NER both extract the same entity, duplicates should be removed

**Verification Checklist for Milestone 2:**
- [ ] Regex patterns extract all test cases correctly
- [ ] spaCy NER model loads and extracts named entities
- [ ] Entity normalization converts values to canonical formats
- [ ] Slot filling correctly maps entities to intent parameters
- [ ] Missing required slots are identified in `DialogState.missing_slots`
- [ ] Overlapping entities are resolved (no duplicates, longest span wins)

#### Milestone 3: Dialog Management

**Acceptance Test Suite** for dialog management focuses on state persistence, multi-turn conversations, and context management.

**1. Session Management Tests:**
- Create a session, add data, retrieve it → data should persist
- Update session activity timestamp, wait beyond timeout → session should expire
- Call `reset()` command → session should return to initial state
- Multiple simultaneous sessions should not interfere

**2. Multi-Turn Conversation Flow Tests:**
Script complete conversations and verify state at each turn:

```python
# Pseudo-test structure
def test_multi_turn_booking():
    # Turn 1: User provides partial information
    state1 = dialog_manager.process_utterance("session1", "Book flight to Paris")
    assert state1.current_intent.name == "book_flight"
    assert state1.filled_slots == {"destination": "Paris"}
    assert "date" in state1.missing_slots
    
    # Turn 2: User provides missing slot
    state2 = dialog_manager.process_utterance("session1", "Tomorrow")
    assert state2.filled_slots["date"] == "2024-05-20"
    assert len(state2.missing_slots) == 0  # All required slots filled
```

**3. Context Carryover Tests:**
- Short follow-ups like "tomorrow" or "from New York" should inherit intent from previous turn
- After slot confirmation ("No, I meant Tuesday"), the corrected value should replace the old one
- Changing intent mid-conversation ("Actually, check weather instead") should reset relevant slots

**4. Global Command Tests:**
| Command | Expected Behavior |
|---------|------------------|
| "reset", "start over" | `DialogState.reset()` called, fresh session |
| "help" | System provides help message without breaking flow |
| "cancel" | Current intent abandoned, returns to IDLE state |

**Verification Checklist for Milestone 3:**
- [ ] Dialog state persists across multiple turns for same session
- [ ] Context carryover works for short follow-up messages
- [ ] Missing slots trigger appropriate prompts
- [ ] Global commands (`reset`, `help`, `cancel`) work correctly
- [ ] Sessions expire after `SESSION_TIMEOUT_SECONDS` of inactivity
- [ ] State machine transitions follow expected logic

#### Milestone 4: Response Generation

**Acceptance Test Suite** for response generation validates template substitution, variation, and fallback handling.

**1. Template Substitution Tests:**
| Template | Slots | Expected Output |
|----------|-------|----------------|
| "Booked {count} tickets for {destination}" | `{"count": 2, "destination": "Paris"}` | "Booked 2 tickets for Paris" |
| "Flight on {date} at {time}" | `{"date": "2024-05-20", "time": "15:00"}` | "Flight on May 20, 2024 at 3:00 PM" |
| "Meeting with {person}" | `{"person": "John Smith"}` | "Meeting with John Smith" |

**2. Response Variation Tests:**
- For a given intent/slot combination, call `generate_response()` multiple times
- Verify different templates from the same variation group are returned
- All variations should be grammatically correct and appropriate

**3. Fallback Response Tests:**
| Scenario | Expected Response Characteristics |
|----------|-----------------------------------|
| Unknown intent | Generic "I didn't understand" message |
| Low confidence | Polite request for clarification |
| Missing template | Appropriate default response |
| Empty/malformed user input | Friendly prompt to rephrase |

**4. XSS/Security Tests:**
- Attempt to inject HTML/JavaScript via slot values
- Verify templates escape user input appropriately (or that slot values are trusted)

**Verification Checklist for Milestone 4:**
- [ ] Template placeholders are correctly replaced with slot values
- [ ] Multiple response variations exist for common scenarios
- [ ] Fallback responses handle all error scenarios gracefully
- [ ] Response tone is consistent across all templates
- [ ] User input in slots is properly escaped/rendered safely
- [ ] Empty/missing slot values are handled (skip placeholder or use default)

**Integration Verification Across All Milestones:**

After completing all four milestones, run **end-to-end conversation tests** that exercise the complete pipeline:

1. **Happy Path Complete Conversation**: Single turn with all information provided
2. **Multi-Turn Slot Collection**: User provides information across multiple messages
3. **Error Recovery Flow**: Handle unknown intents, then successful redirection
4. **Session Timeout and Renewal**: Conversation expires, new session starts fresh
5. **Mixed Initiative**: User changes topic mid-conversation, system adapts

> **Testing Insight**: The most valuable tests for a chatbot are **integration tests that simulate real conversation patterns**. While unit tests ensure components work correctly in isolation, it's the interaction between components that determines the user experience. Focus testing effort on the most common user journeys and error scenarios.

### Implementation Guidance

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Testing Framework | `pytest` (lightweight, popular) | `pytest` with custom plugins |
| Mocking | `unittest.mock` (standard library) | `pytest-mock` (integration with pytest) |
| Test Data Management | JSON/YAML files | Parameterized fixtures in pytest |
| Conversation Testing | Custom test runner with scripted dialogs | `pytest-bdd` for behavior-driven tests |
| Coverage Reporting | `pytest-cov` with HTML reports | Integrated with CI/CD pipeline |

**B. Recommended File/Module Structure:**

```
project_root/
├── tests/                           # All test files
│   ├── __init__.py
│   ├── conftest.py                  # Shared pytest fixtures
│   ├── unit/                        # Unit tests
│   │   ├── test_intent_classifier.py
│   │   ├── test_entity_extractor.py
│   │   ├── test_dialog_manager.py
│   │   └── test_response_generator.py
│   ├── integration/                 # Integration tests  
│   │   ├── test_intent_to_entities.py
│   │   ├── test_slot_filling.py
│   │   └── test_session_persistence.py
│   └── system/                      # System/Conversation tests
│       ├── test_conversation_flows.py
│       ├── test_error_handling.py
│       └── test_multi_turn_dialogs.py
├── test_data/                       # Test datasets
│   ├── training_examples.csv
│   ├── test_utterances.json
│   └── conversation_scripts.yaml
└── src/                             # Main source code
    └── chatbot/                     # Package with all components
```

**C. Infrastructure Starter Code (Test Fixtures):**

Create a `tests/conftest.py` file with shared fixtures:

```python
# tests/conftest.py
import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from src.chatbot.intent_classifier import IntentClassifier
from src.chatbot.entity_extractor import EntityExtractor
from src.chatbot.dialog_manager import DialogManager
from src.chatbot.response_generator import ResponseGenerator
from src.chatbot.session_store import InMemorySessionStore

@pytest.fixture
def sample_training_data():
    """Load sample training data for intent classifier tests."""
    data_path = Path(__file__).parent / "test_data" / "training_examples.csv"
    # TODO: Load and return training data
    return []  # Placeholder

@pytest.fixture
def mock_spacy_model():
    """Create a mock spaCy model for entity extraction tests."""
    mock_nlp = Mock()
    mock_doc = Mock()
    
    # Mock entity spans
    mock_ent1 = Mock()
    mock_ent1.text = "New York"
    mock_ent1.label_ = "GPE"
    mock_ent1.start_char = 10
    mock_ent1.end_char = 18
    
    mock_ent2 = Mock()
    mock_ent2.text = "tomorrow"
    mock_ent2.label_ = "DATE"
    mock_ent2.start_char = 20
    mock_ent2.end_char = 28
    
    mock_doc.ents = [mock_ent1, mock_ent2]
    mock_nlp.return_value = mock_doc
    
    return mock_nlp

@pytest.fixture
def intent_classifier(sample_training_data):
    """Create an IntentClassifier instance with test data."""
    classifier = IntentClassifier()
    # TODO: Train classifier with sample_training_data
    return classifier

@pytest.fixture
def entity_extractor(mock_spacy_model):
    """Create an EntityExtractor with mocked spaCy model."""
    extractor = EntityExtractor()
    extractor.nlp = mock_spacy_model  # Inject mock
    # TODO: Load test regex patterns
    return extractor

@pytest.fixture
def session_store():
    """Create a fresh InMemorySessionStore for each test."""
    return InMemorySessionStore(timeout_seconds=300)

@pytest.fixture
def dialog_manager(session_store, intent_classifier, entity_extractor):
    """Create a DialogManager with all dependencies."""
    # TODO: Create and wire up all components
    return DialogManager(
        session_store=session_store,
        intent_classifier=intent_classifier,
        entity_extractor=entity_extractor
    )
```

**D. Core Test Skeleton Code (Example Unit Test):**

```python
# tests/unit/test_intent_classifier.py
import pytest
from src.chatbot.intent_classifier import IntentClassifier
from src.chatbot.data_model import UserUtterance

class TestIntentClassifier:
    
    def test_predict_with_high_confidence(self, intent_classifier):
        """Test that clear inputs get high confidence predictions."""
        # TODO 1: Create test utterance that should be clearly classified
        test_text = "book a flight to paris"
        
        # TODO 2: Call classifier.predict()
        result = intent_classifier.predict(test_text)
        
        # TODO 3: Verify result is a UserUtterance object
        assert isinstance(result, UserUtterance)
        
        # TODO 4: Verify intent is not None
        assert result.intent is not None
        
        # TODO 5: Verify confidence exceeds threshold
        assert result.confidence >= 0.7
        
        # TODO 6: Verify the specific intent name if known
        # (e.g., assert result.intent.name == "book_flight")
    
    def test_predict_with_low_confidence(self, intent_classifier):
        """Test that ambiguous inputs get low confidence and 'unknown' intent."""
        # TODO 1: Create gibberish or out-of-domain test utterance
        test_text = "asdfkjh lkjhqw er"
        
        # TODO 2: Call classifier.predict()
        result = intent_classifier.predict(test_text)
        
        # TODO 3: Verify confidence is below threshold
        assert result.confidence < 0.7
        
        # TODO 4: Verify intent is None or marked as "unknown"
        # (Implementation specific: either intent=None or intent.name="unknown")
    
    def test_empty_input(self, intent_classifier):
        """Test that empty string doesn't crash."""
        # TODO 1: Call classifier.predict() with empty string
        result = intent_classifier.predict("")
        
        # TODO 2: Verify no exception is raised
        # TODO 3: Verify reasonable handling (unknown intent or None)
    
    def test_special_characters(self, intent_classifier):
        """Test that special characters are handled."""
        # TODO 1: Test with various special characters
        test_cases = [
            "book flight!!!",
            "book @flight",
            "book & flight to <paris>",
            "book 'flight' to \"paris\""
        ]
        
        for text in test_cases:
            # TODO 2: Call classifier.predict() for each
            result = intent_classifier.predict(text)
            
            # TODO 3: Verify no crashes and reasonable output
            assert isinstance(result, UserUtterance)
```

**E. Language-Specific Hints (Python):**

- Use `pytest.mark.parametrize` for data-driven tests with multiple input/output pairs:
  ```python
  @pytest.mark.parametrize("input_text,expected_intent", [
      ("book flight", "book_flight"),
      ("check weather", "check_weather"),
      ("set alarm", "set_alarm"),
  ])
  def test_intent_mapping(self, input_text, expected_intent, intent_classifier):
      result = intent_classifier.predict(input_text)
      assert result.intent.name == expected_intent
  ```

- Use `unittest.mock.patch` to mock external dependencies (like spaCy model loading, file I/O):
  ```python
  from unittest.mock import patch
  
  def test_entity_extractor_loads_spacy(self):
      with patch('src.chatbot.entity_extractor.spacy.load') as mock_load:
          extractor = EntityExtractor()
          # Verify spacy.load was called
          mock_load.assert_called_once_with('en_core_web_sm')
  ```

- For testing time-sensitive functionality (like session expiration), use `freezegun` to mock time:
  ```python
  from freezegun import freeze_time
  
  def test_session_expiration(session_store):
      session_id = "test_session"
      state = DialogState(session_id=session_id)
      session_store.set(session_id, state)
      
      # Advance time by more than timeout
      with freeze_time("2024-01-01 12:00:00"):
          session_store.set(session_id, state)
      
      with freeze_time("2024-01-01 12:06:00"):  # 6 minutes later (> 5 min timeout)
          retrieved = session_store.get(session_id)
          assert retrieved is None  # Should have expired
  ```

**F. Milestone Checkpoint Verification:**

**After Milestone 1:**
```bash
# Run intent classifier tests
pytest tests/unit/test_intent_classifier.py -v

# Expected: All tests pass, including:
# - Accuracy test shows ≥80% accuracy
# - Confidence threshold test shows unknowns for <0.7 confidence
# - Edge case tests pass without crashes

# Manual verification:
python -m src.chatbot.demo_intent  # Should show predictions for sample inputs
```

**After Milestone 2:**
```bash
# Run entity extraction tests
pytest tests/unit/test_entity_extractor.py tests/integration/test_slot_filling.py -v

# Expected: All tests pass, including:
# - Regex patterns extract expected entities
# - NER integration works (with mock or real model)
# - Slot filling correctly maps entities

# Manual verification:
python -m src.chatbot.demo_entities  # Should extract entities from sample sentences
```

**After Milestone 3:**
```bash
# Run dialog management tests
pytest tests/unit/test_dialog_manager.py tests/integration/test_session_persistence.py -v

# Expected: All tests pass, including:
# - Multi-turn conversations maintain state
# - Sessions expire correctly
# - Global commands work

# Manual verification:
python -m src.chatbot.demo_dialog  # Should have a simple interactive conversation
```

**After Milestone 4:**
```bash
# Run all tests
pytest tests/ -v

# Expected: All unit, integration, and system tests pass
# Conversation tests validate complete flows

# Manual verification:
python -m src.chatbot.demo_full  # Complete chatbot with all components
```

**G. Debugging Tips for Testing:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **Test passes locally but fails in CI** | Environment differences (Python version, dependencies) | Check CI logs for error details, compare dependency versions | Use `requirements.txt` with exact versions, test in Docker container |
| **Intent classifier accuracy < 80%** | Insufficient training data, class imbalance, poor feature extraction | Print confusion matrix, analyze misclassified examples | Add more training examples, balance classes, try different vectorizer parameters |
| **Entities not extracted correctly** | Regex patterns too strict/loose, spaCy model not loaded | Print debug output of raw extraction before normalization | Test regex patterns individually, verify spaCy model download |
| **Session state lost between turns** | New `session_id` generated each time, session store not being used | Log `session_id` values, check if same ID used across turns | Ensure session ID persistence (e.g., from cookie or user ID) |
| **Template substitution errors** | Slot names mismatch between template and dialog state | Print template string and available slots before substitution | Verify slot naming consistency, add fallback for missing slots |
| **Random test failures** | Non-deterministic code (random, time-dependent) | Use fixed random seeds, mock time functions | Set random seeds in tests, use `freezegun` for time-based tests |


## 9. Debugging Guide

> **Milestone(s):** Milestone 1, Milestone 2, Milestone 3, Milestone 4 (Operational Excellence)

Building an intent-based chatbot involves multiple interconnected components where subtle bugs can create confusing failures. Unlike traditional software with straightforward inputs and outputs, conversational systems process ambiguous natural language through a pipeline where errors propagate and compound. This guide provides a systematic approach to diagnosing and fixing common implementation issues, transforming frustrating debugging sessions into structured investigations.

Think of debugging this system as **being a detective investigating a crime scene**. Each component leaves forensic evidence in the form of log messages, intermediate data structures, and state changes. Your job is to follow the evidence trail from the symptom (what the user experiences) back to the root cause (which component failed and why). This section provides both a symptom-cause-fix reference manual and investigative techniques for when you encounter novel bugs.

### Common Bug Table: Symptom → Cause → Fix

The following table catalogs the most frequent implementation errors, organized by the symptom a developer or user would observe. Each entry includes the underlying cause, how to confirm the diagnosis, and the specific fix to apply.

| Symptom (What You Observe) | Root Cause | How to Diagnose | Fix |
| :--- | :--- | :--- | :--- |
| **"Unknown intent" for clear, well-phrased inputs** | The `IntentClassifier` returns confidence below `INTENT_CONFIDENCE_THRESHOLD` (0.7). This often happens with insufficient or imbalanced training data, or when the TF-IDF vocabulary doesn't cover the user's words. | Log the confidence scores for all intents after `IntentClassifier.predict()`. Check if the correct intent appears in top predictions but with low probability (e.g., 0.65). Inspect training data coverage for that intent. | Add more diverse training examples (10-15 per intent). Ensure training examples include synonym variations. Consider lowering `INTENT_CONFIDENCE_THRESHOLD` temporarily during development, but address data quality. |
| **Intent misclassification (wrong intent detected)** | Overlapping vocabulary between intents causes TF-IDF vectors to be similar. For example, "book a flight" and "book a hotel" share "book a". Classifier picks the wrong intent because of insufficient distinguishing features. | Examine the TF-IDF features (most weighted terms) for the competing intents. Check if training examples for different intents are too similar. | Add distinctive phrases to training data. For `book_flight`, add examples mentioning "airplane", "airline", "departure". For `book_hotel`, add "check-in", "suite", "reservation". Use n-grams in vectorizer (bigrams like "book flight" vs "book hotel"). |
| **No entities extracted from text that clearly contains them** | The `EntityExtractor` failed to match patterns or the spaCy model wasn't loaded. For regex patterns, the pattern might be too specific or have incorrect syntax. For spaCy, the model label might not map to your entity types. | Log the raw text sent to `EntityExtractor.extract_entities()`. Check if `EntityExtractor._load_spacy_model()` was called (look for "Loading spaCy model" log). Test regex patterns independently with a regex tester. | Verify spaCy model installation: `python -m spacy download en_core_web_sm`. Update `EntityExtractor._map_spacy_label()` to map spaCy labels (like "GPE", "PERSON") to your types ("location", "person"). Debug regex patterns: print pattern and test string. |
| **Entities extracted but slots remain empty** | The `SlotFiller.fill_slots()` mapping doesn't match entity types to slot names, or the `UserUtterance.update_slots_from_entities()` helper wasn't called. The mapping dictionary might have incorrect keys. | Log the extracted entities (type and value) and the `slot_mappings` dictionary. Check if entity type appears as a key in `slot_mappings` for the current intent. | Verify `slot_mappings` structure: for each intent, map entity types to slot names (e.g., {"person": "guest_name"}). Ensure `SlotFiller.fill_slots()` calls `utterance.update_slots_from_entities()` with correct mapping. |
| **Dialog loses context between turns** | The `DialogState` is not being persisted in the `InMemorySessionStore`, or the `session_id` changes between requests. Each new request creates a new state instead of retrieving the existing one. | Log the `session_id` at the start of `DialogManager.process_utterance()`. Check if `InMemorySessionStore.get()` returns `None` for expected session. Verify session expiration hasn't occurred (`SESSION_TIMEOUT_SECONDS`). | Ensure client sends consistent `session_id` (e.g., via cookies or user ID). Check `DialogManager._get_or_create_state()` retrieves existing state via `session_store.get(session_id)`. Consider implementing session stickiness in your application layer. |
| **Bot asks for same slot repeatedly in a loop** | The `DialogState.missing_slots` list isn't being updated after receiving an entity, or the entity isn't being mapped to the correct slot. The state machine remains in `COLLECTING_SLOTS` state because required slots remain unfilled. | Log the `DialogState` after each turn: `current_intent`, `filled_slots`, `missing_slots`. Check if the extracted entity type matches any `required_slots` for the intent. | Verify `SlotFiller.fill_slots()` updates `state.filled_slots` and recalculates `state.missing_slots`. Ensure `Intent.required_slots` list matches slot names used in `slot_mappings`. |
| **Response contains `{slot_name}` placeholder instead of value** | The `ResponseGenerator` template substitution failed. The placeholder name doesn't match any key in `DialogState.filled_slots`, or the slot value is `None`. The `_format_slot_value()` method might be missing for complex types. | Log the template string selected and the `filled_slots` dictionary. Check if slot names in template (e.g., `{guest_name}`) exist as keys in `filled_slots`. | Ensure `ResponseGenerator._format_slot_value()` handles all data types (str, int, date, list). Verify template placeholders use exact slot names. Use `template_string.format(**filled_slots)` with caution (ensure all placeholders have corresponding slots). |
| **Bot responds with wrong template (off-topic response)** | The `ResponseGenerator` selected template from incorrect variation group. The `DialogState` might have incorrect `current_intent` or state machine state, causing wrong template lookup. | Log the variation group used for template selection (e.g., `intent_book_hotel_slots_complete`). Check `DialogState.current_intent.name` and which slots are filled/missing. | Verify template registry organization: templates grouped by `variation_group` matching intent and state. Ensure `ResponseGenerator` logic correctly maps dialog state to variation group (e.g., if missing slots, use prompt template group). |
| **Overlapping entities extracted (duplicate or conflicting)** | Multiple regex patterns match same text span, or spaCy and regex both extract same entity. The `EntityExtractor._resolve_overlaps()` strategy (e.g., "keep longest") may produce unexpected results. | Log all entities before and after `_resolve_overlaps()`. Check entity spans (start, end) for overlaps. | Adjust overlap resolution strategy. Common approach: keep higher confidence entity, or keep more specific type. Modify `_resolve_overlaps()` to prioritize ML entities over regex when confidence > threshold. |
| **Session memory leak (RAM usage grows indefinitely)** | `InMemorySessionStore` never removes expired sessions. The cleanup mechanism (e.g., periodic task) isn't implemented or isn't running. | Log session count periodically. Check if `InMemorySessionStore._store` dictionary size increases monotonically. | Implement session expiration cleanup: periodically iterate through store and remove states where `state.is_expired(SESSION_TIMEOUT_SECONDS)` returns `True`. Use background thread or cleanup in `get()`/`set()` methods. |
| **Date entities parsed incorrectly (01/02 ambiguity)** | Date normalization assumes MM/DD format but user uses DD/MM, or vice versa. The `EntityNormalizer` lacks locale awareness or ambiguous date handling. | Log raw entity text and normalized value. Test with ambiguous dates like "01/02/2023". | Implement explicit date format handling: assume ISO format (YYYY-MM-DD) when possible, or add context (e.g., "Jan 2"). Use library like `dateutil.parser` with `dayfirst` parameter. Add user confirmation for ambiguous dates. |
| **Multi-intent input classified as single intent** | The `IntentClassifier` only returns the highest-confidence intent, but user expressed multiple goals (e.g., "book flight and hotel"). System design assumes single intent per utterance. | Log all intents with confidence scores. Check if multiple intents have confidence above threshold. | For advanced handling, implement multi-intent detection: if two intents have confidence > threshold and are not mutually exclusive, create composite intent. For MVP, design conversation to handle one intent at a time and prompt user to focus. |
| **Global commands like "reset" don't work** | `DialogManager._handle_global_commands()` returns `False` or isn't being called early in processing pipeline. The command detection regex/pattern is too restrictive. | Log text input and check if `_handle_global_commands()` is called. Test command detection with various phrasings ("start over", "cancel", "new conversation"). | Expand global command patterns (regex for "reset", "start over", "new chat", "cancel"). Ensure `_handle_global_commands()` is called BEFORE intent classification and returns `True` when command matched, triggering `state.reset()`. |
| **Confidence scores always 1.0 or 0.0** | The classifier's `predict_proba()` method might not be implemented correctly, or the training data has perfect separation. Some models like `SVC` with `probability=True` need proper configuration. | Check classifier implementation: use `model.predict_proba()` not `model.predict()`. Verify training data has some overlap between intents. | For SVM, ensure `probability=True` is set during initialization. For Naive Bayes, use `predict_proba()`. Add slight noise to training data or more varied examples to prevent perfect separation. |

### Debugging Techniques and Logging

Effective debugging requires more than just a lookup table—it demands a systematic investigative methodology. When encountering a novel bug, follow this **five-step investigative process**:

#### Step 1: Reproduce and Isolate
First, create a minimal, reproducible test case. Isolate whether the issue is in a specific component or in their integration.

> **Investigation Tip**: Create a standalone test script that bypasses the full pipeline and tests the suspect component directly with the problematic input. For example, if the bot misclassifies "book a flight tomorrow", create a script that calls `IntentClassifier.predict("book a flight tomorrow")` and prints the confidence scores for all intents.

#### Step 2: Instrument with Strategic Logging
Add logging statements at key boundaries in the processing pipeline. Each component should log its inputs, outputs, and significant internal decisions. Use different log levels:
- `DEBUG`: Verbose internal details (TF-IDF vector values, individual pattern matches)
- `INFO`: Component boundaries (input received, output produced)
- `WARNING`: Recoverable issues (low confidence, missing but optional slots)
- `ERROR`: Failures that break processing (model not loaded, invalid state)

**Recommended Log Points:**
| Component | What to Log | Log Level |
| :--- | :--- | :--- |
| `IntentClassifier` | Input text, top 3 intents with confidence scores, final predicted intent | INFO |
| `EntityExtractor` | Input text, all extracted entities (type, value, confidence), overlap resolution decisions | DEBUG |
| `SlotFiller` | Extracted entities, slot mappings, filled slots dictionary | INFO |
| `DialogManager` | Session ID, current state (intent, filled slots), state transitions, global commands detected | INFO |
| `ResponseGenerator` | Selected template, variation group, filled slots used for substitution | DEBUG |
| `InMemorySessionStore` | Session retrieval/creation, expiration cleanup counts | INFO |

#### Step 3: Visualize the Data Flow
Create a mental map of how data transforms through the pipeline. For a given input, trace the journey:

1. **Raw text** → `UserUtterance.raw_text`
2. **Cleaned text** → `UserUtterance.cleaned_text` (lowercase, punctuation removed)
3. **Intent classification** → `UserUtterance.intent` and `.confidence`
4. **Entity extraction** → `UserUtterance.entities` list
5. **Slot filling** → `UserUtterance.slots` dictionary
6. **Dialog state update** → `DialogState` modifications
7. **Response generation** → final response string

At each step, ask: "Does this transformation make sense given the input?" Use the `conversation-sequence-diagram` as a reference for normal flow.

#### Step 4: Inspect Intermediate State
When a bug occurs, capture and examine the complete state of the system. Create a debugging endpoint or function that dumps:

- Complete `DialogState` object (JSON serialized)
- Recent conversation history
- All extracted entities with confidence scores
- Confidence distribution across all intents

> **Design Insight**: Build a `DebugView` class that can be toggled via configuration. In development mode, it attaches detailed diagnostic information to responses (as hidden metadata or separate debug panel). In production, it's disabled for security and performance.

#### Step 5: Hypothesis Testing
Form a hypothesis about the root cause, then design an experiment to test it. For example:

**Hypothesis**: "The date entity isn't being extracted because the regex pattern doesn't match 'next Tuesday'."

**Experiment**: Test the date pattern directly:
```python
import re
pattern = r'\b(tomorrow|today|next week)\b'  # simplified example
test = "book for next Tuesday"
print(bool(re.search(pattern, test.lower())))
```

If the experiment confirms the hypothesis, you've found the bug. If not, refine your hypothesis and test again.

#### Structured Logging Implementation

Implement logging consistently across all components using Python's `logging` module. Configure it to output structured JSON for easier parsing by log aggregators:

```python
import logging
import json

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        
    def log_event(self, level, event_type, component, data):
        """Log structured event with consistent schema."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "event_type": event_type,
            "component": component,
            "session_id": data.get("session_id", "unknown"),
            "data": data
        }
        if level == "DEBUG":
            self.logger.debug(json.dumps(log_entry))
        elif level == "INFO":
            self.logger.info(json.dumps(log_entry))
        # ... other levels
```

Use this logger throughout your components with consistent event types:
- `intent_classification`: With data `{"input": text, "top_intents": [...]}`
- `entity_extraction`: With data `{"input": text, "entities": [...]}`
- `slot_filling`: With data `{"entities": [...], "filled_slots": {...}}`
- `state_transition`: With data `{"from_state": old, "to_state": new, "trigger": event}`

#### Interactive Debugging Session

For complex issues, use an interactive debugger (pdb for Python) with breakpoints at critical junctures:

1. **Set breakpoint** in `DialogManager.process_utterance()` to examine the complete flow
2. **Step through** each component's processing
3. **Inspect variables** at each stage to see where transformations diverge from expectations

Create a test harness that simulates a conversation turn-by-turn, pausing after each turn for manual inspection.

#### Common Investigation Patterns

When facing specific symptom categories, follow these investigative patterns:

| Symptom Category | First Component to Check | Key Questions to Ask |
| :--- | :--- | :--- |
| **Wrong understanding** | `IntentClassifier` | 1. What were the confidence scores?<br>2. Does training data cover this phrasing?<br>3. Are stopwords dominating the TF-IDF vector? |
| **Missing information** | `EntityExtractor` | 1. Did regex patterns fire?<br>2. Was spaCy model loaded?<br>3. Do patterns match the exact text? |
| **Context problems** | `DialogManager` & `InMemorySessionStore` | 1. Is session ID consistent?<br>2. Did state persist between turns?<br>3. Has session expired? |
| **Bad responses** | `ResponseGenerator` | 1. Which template was selected?<br>2. Were all slot placeholders filled?<br>3. Does variation group match state? |

#### Debugging Checklist

Before concluding your investigation, verify these common oversight points:

- [ ] **Training data balance**: Each intent has ≥10 diverse examples
- [ ] **spaCy model**: Downloaded and loaded (`python -m spacy download en_core_web_sm`)
- [ ] **Session storage**: Sessions persist between requests with same `session_id`
- [ ] **Threshold tuning**: `INTENT_CONFIDENCE_THRESHOLD` is appropriate for your data quality
- [ ] **Pattern testing**: All regex patterns tested with edge cases
- [ ] **Slot mappings**: Every entity type maps to a slot for each intent
- [ ] **Template coverage**: Every intent/state combination has at least one template
- [ ] **Error handling**: Low confidence, missing slots, and expired sessions have graceful fallbacks

Remember that debugging conversational AI is iterative. The first fix might not completely solve the issue but will reveal deeper insights. Document your findings and add tests to prevent regression of the same bug class.

### Implementation Guidance

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
| :--- | :--- | :--- |
| Logging | Python `logging` module with JSON formatter | `structlog` for structured logging, ELK stack for aggregation |
| Debug Interface | Command-line debug flag printing state | Web-based debug dashboard with real-time state inspection |
| Session Inspection | `print()` statements in development | Redis explorer for persistent session store |
| Testing | `pytest` with fixture for chatbot instance | Property-based testing with `hypothesis` for input variations |

**Recommended Debug Utilities File Structure:**

```
project-root/
  chatbot/
    core/                    # Main components (as previously defined)
    debug/                   # Debugging utilities
      __init__.py
      state_inspector.py     # Functions to dump and visualize DialogState
      conversation_replayer.py # Record and replay conversation flows
      logging_config.py      # Structured logging setup
    tests/
      debug_helpers/         # Debug-focused test utilities
        test_harness.py      # Simulates multi-turn conversations
        anomaly_detector.py  # Flags unexpected behavior patterns
```

**Debugging Helper Class (Complete Implementation):**

```python
# chatbot/debug/state_inspector.py
import json
from datetime import datetime
from typing import Dict, Any, Optional
from ..core.data_models import DialogState, UserUtterance

class StateInspector:
    """Utility for inspecting and debugging chatbot state."""
    
    @staticmethod
    def dump_state(state: DialogState, include_history: bool = True) -> Dict[str, Any]:
        """Serialize DialogState to dictionary for debugging."""
        dump = {
            "session_id": state.session_id,
            "current_intent": state.current_intent.name if state.current_intent else None,
            "filled_slots": state.filled_slots,
            "missing_slots": state.missing_slots,
            "confirmation_required": state.confirmation_required,
            "last_activity": state.last_activity.isoformat() if state.last_activity else None,
            "is_expired": state.is_expired(300) if state.last_activity else True,
        }
        
        if include_history and state.conversation_history:
            dump["conversation_history"] = state.conversation_history[-5:]  # Last 5 turns
        
        return dump
    
    @staticmethod
    def format_state_for_display(state: DialogState) -> str:
        """Create human-readable state summary."""
        lines = []
        lines.append(f"=== Session: {state.session_id} ===")
        lines.append(f"Intent: {state.current_intent.name if state.current_intent else 'None'}")
        lines.append(f"Filled slots: {len(state.filled_slots)}")
        for slot, value in state.filled_slots.items():
            lines.append(f"  - {slot}: {value}")
        lines.append(f"Missing slots: {', '.join(state.missing_slots) if state.missing_slots else 'None'}")
        lines.append(f"Confirmation needed: {state.confirmation_required}")
        if state.last_activity:
            lines.append(f"Last active: {state.last_activity.strftime('%H:%M:%S')}")
        return "\n".join(lines)
    
    @staticmethod
    def analyze_utterance(utterance: UserUtterance) -> Dict[str, Any]:
        """Analyze a UserUtterance for debugging."""
        return {
            "raw_text": utterance.raw_text,
            "cleaned_text": utterance.cleaned_text,
            "intent": utterance.intent.name if utterance.intent else None,
            "confidence": utterance.confidence,
            "entities": [
                {"type": e.type, "value": e.value, "confidence": e.confidence}
                for e in utterance.entities
            ],
            "slots": utterance.slots
        }
```

**Debug-Enhanced Dialog Manager Skeleton:**

```python
# chatbot/core/dialog_manager.py (partial)
import logging
from typing import Tuple, Optional
from .data_models import DialogState
from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .slot_filler import SlotFiller
from .response_generator import ResponseGenerator
from .session_store import InMemorySessionStore

class DialogManager:
    """Orchestrator with built-in debugging support."""
    
    def __init__(self, debug_mode: bool = False):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.slot_filler = SlotFiller()
        self.response_generator = ResponseGenerator()
        self.session_store = InMemorySessionStore()
        self.logger = logging.getLogger(__name__)
        self.debug_mode = debug_mode
        self.debug_info = {}  # Stores debug data for last request
        
    def process_utterance(self, session_id: str, user_input: str) -> Tuple[DialogState, str]:
        """
        Main orchestrator method to process a user message.
        Returns updated dialog state and response text.
        """
        if self.debug_mode:
            self.debug_info = {"session_id": session_id, "input": user_input, "steps": []}
        
        # Step 1: Get or create dialog state
        state = self._get_or_create_state(session_id)
        if self.debug_mode:
            self.debug_info["steps"].append(("get_state", state.session_id))
        
        # Step 2: Check for global commands (reset, cancel, etc.)
        if self._handle_global_commands(user_input, state):
            if self.debug_mode:
                self.debug_info["steps"].append(("global_command", "handled"))
            response = self.response_generator.generate_response(state)
            return state, response
        
        # Step 3: Classify intent (with optional context)
        utterance = self.intent_classifier.predict(user_input, state)
        if self.debug_mode:
            self.debug_info["steps"].append(("intent_prediction", {
                "intent": utterance.intent.name if utterance.intent else None,
                "confidence": utterance.confidence
            }))
        
        # TODO: Add confidence threshold check here
        # If confidence < INTENT_CONFIDENCE_THRESHOLD, set intent to None
        
        # Step 4: Extract entities
        entities = self.entity_extractor.extract_entities(utterance.cleaned_text)
        utterance.entities = entities
        if self.debug_mode:
            self.debug_info["steps"].append(("entity_extraction", [
                {"type": e.type, "value": e.value} for e in entities
            ]))
        
        # Step 5: Fill slots from entities
        filled_slots = self.slot_filler.fill_slots(utterance, state.current_intent, state)
        if self.debug_mode:
            self.debug_info["steps"].append(("slot_filling", filled_slots))
        
        # Step 6: Update dialog state based on new information
        self._update_dialog_state(state, utterance, filled_slots)
        
        # Step 7: Generate response based on updated state
        response = self.response_generator.generate_response(state)
        
        # Step 8: Persist updated state
        self.session_store.set(session_id, state)
        
        if self.debug_mode:
            self.debug_info["final_state"] = state
            self.debug_info["response"] = response
            self._log_debug_info()
        
        return state, response
    
    def _log_debug_info(self):
        """Log detailed debug information if in debug mode."""
        self.logger.debug("=== Debug Info ===")
        for step, data in self.debug_info.get("steps", []):
            self.logger.debug(f"Step {step}: {data}")
        self.logger.debug(f"Final state: {self.debug_info.get('final_state')}")
        self.logger.debug(f"Response: {self.debug_info.get('response')}")
    
    def get_debug_info(self) -> Dict:
        """Retrieve debug information from last processed utterance."""
        return self.debug_info.copy()
    
    # TODO: Implement _get_or_create_state helper
    # TODO: Implement _handle_global_commands helper  
    # TODO: Implement _update_dialog_state helper
```

**Debugging Test Harness Skeleton:**

```python
# tests/debug_helpers/test_harness.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chatbot.core.dialog_manager import DialogManager
from chatbot.debug.state_inspector import StateInspector

class ChatbotTestHarness:
    """Simulates conversations for debugging."""
    
    def __init__(self):
        self.dm = DialogManager(debug_mode=True)
        self.conversation_log = []
    
    def simulate_turn(self, session_id: str, user_input: str) -> str:
        """Process one user turn and return response."""
        state, response = self.dm.process_utterance(session_id, user_input)
        
        # Log this turn
        turn = {
            "input": user_input,
            "response": response,
            "state_after": StateInspector.dump_state(state),
            "debug_info": self.dm.get_debug_info()
        }
        self.conversation_log.append(turn)
        
        return response
    
    def run_conversation(self, session_id: str, script: list):
        """Run a complete conversation script."""
        print(f"=== Starting conversation {session_id} ===")
        for user_input in script:
            print(f"User: {user_input}")
            response = self.simulate_turn(session_id, user_input)
            print(f"Bot: {response}")
            print()
    
    def diagnose_issue(self, session_id: str, problematic_input: str):
        """Run diagnostics on problematic input."""
        print(f"=== Diagnosing: '{problematic_input}' ===")
        response = self.simulate_turn(session_id, problematic_input)
        
        # Get debug info
        debug_info = self.dm.get_debug_info()
        
        # Analyze intent classification
        intent_step = next((s for s in debug_info.get("steps", []) if s[0] == "intent_prediction"), None)
        if intent_step:
            intent_data = intent_step[1]
            print(f"Intent: {intent_data.get('intent')} (confidence: {intent_data.get('confidence'):.2f})")
            if intent_data.get('confidence', 0) < 0.7:
                print("WARNING: Low confidence - may be classified as 'unknown'")
        
        # Analyze entity extraction
        entity_step = next((s for s in debug_info.get("steps", []) if s[0] == "entity_extraction"), None)
        if entity_step:
            entities = entity_step[1]
            if entities:
                print(f"Entities extracted: {len(entities)}")
                for e in entities:
                    print(f"  - {e['type']}: '{e['value']}'")
            else:
                print("No entities extracted - check patterns and spaCy model")
        
        return response

# Example usage
if __name__ == "__main__":
    harness = ChatbotTestHarness()
    
    # Test 1: Simple classification
    print("Test 1: Intent classification")
    harness.diagnose_issue("test_session", "book a flight to New York")
    
    # Test 2: Problematic input
    print("\n\nTest 2: Problematic input")
    harness.diagnose_issue("test_session", "I want to go somewhere nice tomorrow")
```

**Debugging Tips for Python:**

1. **Use `pdb` interactively**: Add `import pdb; pdb.set_trace()` at suspicious points
2. **Pretty-print complex structures**: Use `pprint.pprint(state.__dict__)` for visual inspection
3. **Check spaCy model loading**: Ensure `nlp = spacy.load("en_core_web_sm")` succeeds
4. **Test TF-IDF vectorizer independently**: Create a small script to test vectorization with your training data
5. **Monitor memory usage**: Use `tracemalloc` to detect session store memory leaks
6. **Validate regex patterns**: Use online testers or Python's `re.debug` flag to understand pattern matching
7. **Create minimal reproduction cases**: Isolate the bug by creating the smallest possible test that reproduces it

By combining these debugging tools with the systematic investigative approach, you'll be able to efficiently diagnose and fix even the most elusive chatbot bugs.


## 10. Future Extensions

> **Milestone(s):** Future Enhancements (Post-Milestone 4)

The modular pipeline architecture of Project Athena is intentionally designed for evolution. While the current implementation provides a solid foundation for intent-based conversational AI without LLMs, its component-based nature creates clear extension points for enhancing capabilities, improving accuracy, and adapting to new use cases. This section explores strategic enhancements that could be built upon the existing architecture, organized by the component they would extend or the new capability they would introduce.

### Ideas for Enhancement and Scaling

The chatbot system can be thought of as a **modular conversation factory**—a production line with standardized interfaces where you can swap out machinery, add new processing stages, or even open new production facilities. Each component's clean separation allows independent upgrades without requiring a complete system redesign.

#### 1. Enhanced Natural Language Understanding (NLU)

The current TF-IDF and rule-based NER approach provides excellent baseline performance but has inherent limitations in semantic understanding and contextual awareness.

> **Decision: Gradual NLU Enhancement Strategy**
> - **Context**: The current NLU pipeline (TF-IDF + regex/rule-based NER) works well for clear, structured inputs but struggles with complex paraphrasing, synonyms, and contextual disambiguation.
> - **Options Considered**:
>     1. **Fine-tuned Transformer Models**: Replace TF-IDF with a lightweight transformer (e.g., DistilBERT) fine-tuned on domain-specific data
>     2. **LLM-as-a-Service Fallback**: Keep current pipeline but call an LLM API (e.g., GPT-3.5) for low-confidence or ambiguous cases
>     3. **Ensemble Classifiers**: Combine predictions from TF-IDF, word embeddings, and simple neural networks
> - **Decision**: Implement LLM fallback as an enhancement to the existing pipeline, not a replacement
> - **Rationale**: This provides immediate improvement for edge cases without requiring complete retraining infrastructure or large computational resources. The modular design allows the LLM to be "plugged in" as an optional component that only activates when the primary classifier is uncertain.
> - **Consequences**: Adds API dependency and latency for fallback cases but dramatically improves handling of linguistic variation without abandoning the efficient, deterministic core.

| Enhancement | Implementation Approach | Benefits | Challenges |
|-------------|------------------------|----------|------------|
| **LLM Fallback for Low-Confidence Predictions** | Add `LLMIntentClassifier` that implements the same `predict()` interface, configured as a secondary classifier in `DialogManager` when primary confidence < threshold | Handles paraphrasing, synonyms, and nuanced language; reduces unknown intent rate | API costs, latency (200-500ms), dependency on external service |
| **Context-Aware Intent Classification** | Modify `IntentClassifier.predict()` to accept `DialogState` context and include recent conversation history in feature vector | Better handles elliptical sentences ("same time tomorrow") and follow-up questions | Increases feature dimensionality; requires contextual training examples |
| **Semantic Slot Filling** | Replace regex patterns with transformer-based NER (e.g., spaCy's `transformer` pipeline) for entity extraction | Extracts entities based on semantic understanding, not just patterns; handles novel entity mentions | Requires GPU for inference, larger model sizes, more training data |
| **Synonym and Phrasal Variation Expansion** | Add preprocessing layer that expands training data with synonyms using WordNet or word embeddings | Improves classifier generalization without collecting more real examples | May introduce noise if synonyms are inappropriate for domain |

**Implementation Pathway**: The LLM fallback can be implemented as a decorator/wrapper around the existing `IntentClassifier`:

```python
class LLMFallbackClassifier:
    def __init__(self, primary_classifier, llm_client, confidence_threshold=0.7):
        self.primary = primary_classifier
        self.llm = llm_client
        self.threshold = confidence_threshold
    
    def predict(self, text, state=None):
        # First try primary classifier
        utterance = self.primary.predict(text, state)
        
        # If low confidence, call LLM
        if utterance.confidence < self.threshold:
            return self._call_llm_fallback(text, state)
        
        return utterance
```

#### 2. Knowledge Base Integration

Currently, the chatbot understands intent and extracts entities but has no factual knowledge about the world. Adding a knowledge base transforms it from a **form-filling assistant** to an **information-providing expert**.

**Mental Model**: The **Reference Librarian**—someone who knows both what you're asking for (intent) and where to find the authoritative information (knowledge base lookup).

| Knowledge Integration Pattern | Architecture Impact | Example Use Case |
|-------------------------------|---------------------|------------------|
| **Embedded FAQ Lookup** | Add `KnowledgeRetriever` component between `DialogManager` and `ResponseGenerator` that queries local vector database | Customer support: "What's your return policy?" → retrieves policy text |
| **External API Integration** | Extend `ResponseGenerator` to include API call placeholders in templates that fetch live data | Weather bot: "Weather in Boston" → calls weather API, inserts data into response |
| **Structured Database Query** | Add `QueryPlanner` component that converts filled slots into database queries (SQL, GraphQL) | Inventory bot: "Do you have red shoes in size 9?" → queries product database |
| **Document Retrieval** | Implement `DocumentRetriever` using TF-IDF or embeddings to find relevant document passages | HR bot: "How do I request vacation?" → retrieves relevant HR policy section |

**Data Flow Enhancement**: The enhanced pipeline would include a knowledge retrieval step:
1. `DialogManager` identifies intent and fills slots
2. If intent requires knowledge, passes `DialogState` to `KnowledgeRetriever`
3. `KnowledgeRetriever` queries appropriate source, returns facts
4. `ResponseGenerator` receives facts as additional template variables

> **Key Insight**: The knowledge base should be consulted *after* intent and entity extraction, not as part of NLU. This maintains separation of concerns: NLU understands *what the user wants*, knowledge base provides *what the system knows*.

#### 3. Multi-Modal and Voice Interfaces

The current text-only interface can be extended to support voice input/output and visual elements, transforming the chatbot into a true multi-modal assistant.

| Interface Extension | Required Components | Integration Point |
|---------------------|---------------------|-------------------|
| **Voice Input (Speech-to-Text)** | STT service (e.g., Whisper, Google Speech-to-Text) | Pre-processor before `DialogManager.process_utterance()` |
| **Voice Output (Text-to-Speech)** | TTS service (e.g., Amazon Polly, Google Text-to-Speech) | Post-processor after `ResponseGenerator.generate_response()` |
| **Rich UI Elements** | Markdown/HTML renderer with cards, buttons, carousels | Enhanced `ResponseTemplate` with structured payloads |
| **File/Image Analysis** | Vision API for image analysis, document parsers | New `MultiModalExtractor` component alongside `EntityExtractor` |

**Architecture Consideration**: Voice interfaces require session-aware handling of audio stream segmentation and turn-taking detection. The `InMemorySessionStore` would need enhancement to track audio session state, timeouts, and partial transcriptions.

```python
class VoiceEnabledDialogManager(DialogManager):
    def process_audio_chunk(self, session_id, audio_chunk):
        # Accumulate audio in session state
        state = self._get_or_create_state(session_id)
        state.audio_buffer.append(audio_chunk)
        
        # Detect speech endpoint (silence or pause)
        if self._detected_end_of_utterance(state.audio_buffer):
            # Convert speech to text
            text = self.stt_service.transcribe(state.audio_buffer)
            # Process as text utterance
            return self.process_utterance(session_id, text)
```

#### 4. Advanced Dialog Management

Beyond simple slot filling, more sophisticated conversation patterns can be implemented by enhancing the `DialogManager`'s state machine and adding new components.

| Advanced Dialog Capability | State Machine Extension | New Components |
|----------------------------|-------------------------|----------------|
| **Multi-Intent Conversations** | Nested state machines allowing parallel intent tracking | `MultiIntentTracker`, `ConversationThreadManager` |
| **Proactive Suggestions** | System-initiated turn capability in state machine | `SuggestionEngine`, `OpportunityDetector` |
| **Contextual Memory** | Long-term memory storage beyond current session | `UserProfileStore`, `ConversationMemory` |
| **Emotional Intelligence** | Sentiment analysis influencing response selection | `SentimentAnalyzer`, `EmpathyScorer` |

**Example: Proactive Suggestions Enhancement**:
1. Add `suggestion_state` to `DialogState` tracking when suggestions are appropriate
2. Implement `SuggestionEngine.analyze(state)` that runs after each turn
3. If suggestion conditions met, `DialogManager` can initiate a system turn
4. Response templates include suggestion variations triggered by specific state conditions

#### 5. Scalability and Production Deployment

The current in-memory session store and file-based templates work for development but need enhancement for production-scale deployment.

| Scalability Enhancement | Implementation Strategy | Benefit |
|-------------------------|-------------------------|---------|
| **Distributed Session Store** | Replace `InMemorySessionStore` with Redis or database backend | Horizontal scaling, persistence across restarts |
| **Model Versioning and A/B Testing** | Add `ModelRegistry` component with canary deployment support | Safe rollout of classifier improvements |
| **Pipeline Performance Monitoring** | Add telemetry to each component, export metrics to Prometheus | Identify bottlenecks, track accuracy over time |
| **Containerized Deployment** | Package components as separate microservices with gRPC/protobuf interfaces | Independent scaling of NLU vs dialog management |

> **Architecture Evolution Path**: Start with monolithic application for simplicity, then extract components to microservices based on scaling needs:
> 1. **Phase 1**: Single process with all components (current architecture)
> 2. **Phase 2**: Separate NLU service (intent + entity) from dialog service
> 3. **Phase 3**: Add specialized services for knowledge retrieval, voice processing, etc.

#### 6. Personalization and Adaptation

A static chatbot becomes more engaging when it adapts to individual users and learns from interactions over time.

| Personalization Feature | Data Requirements | Implementation Approach |
|-------------------------|-------------------|-------------------------|
| **User-Specific Slot Defaults** | Store user preferences in profile | Enhance `SlotFiller` to check `UserProfile` before prompting |
| **Adaptive Response Style** | Track user sentiment and response engagement | `ResponseGenerator` selects template variation based on user's preferred communication style |
| **Conversation-Based Learning** | Log misunderstood utterances for retraining | `FeedbackCollector` component identifies misclassifications, adds to training data |
| **Domain Adaptation** | Continuous evaluation on new utterances | `ModelEvaluator` component detects performance drift, triggers retraining |

**Implementation Consideration**: Personalization requires careful privacy design. The `DialogState` should be extended with `user_id` and consent flags, with clear separation between ephemeral conversation state and persistent user profiles.

### Implementation Guidance

While the previous sections are speculative designs, here's concrete guidance for implementing the first and most impactful enhancement: LLM fallback handling.

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| LLM Integration | OpenAI GPT-3.5 Turbo API | Self-hosted Llama 2 via Hugging Face |
| Fallback Orchestrator | Decorator pattern wrapping IntentClassifier | Configurable fallback chain with multiple fallback strategies |
| Caching Layer | Python dictionary cache for repeated queries | Redis cache with TTL for common ambiguous phrases |
| Monitoring | Log LLM usage and latency | OpenTelemetry metrics + cost tracking dashboard |

**B. Recommended File/Module Structure:**

```
project-athena/
  chatbot/                 # Existing core
    core/                  # Existing components
    enhancement/           # New enhancement modules
      llm_fallback/
        __init__.py
        llm_classifier.py      # LLMFallbackClassifier implementation
        prompt_templates.py    # LLM prompt templates for intent classification
        cache.py               # Caching layer for LLM responses
        config.py              # API keys, model parameters
    services/              # External service integrations
      openai_client.py     # Wrapper for OpenAI API
      redis_client.py      # (Optional) Redis client for caching
```

**C. Infrastructure Starter Code (LLM Client Wrapper):**

```python
# services/openai_client.py
import openai
from typing import Dict, Any, Optional
import json
import logging

class OpenAIClient:
    """Wrapper for OpenAI API calls with retry logic and error handling."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def classify_intent(self, text: str, possible_intents: list, 
                        context: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to classify intent from text.
        
        Args:
            text: User input text
            possible_intents: List of intent names the system knows
            context: Optional conversation context
            
        Returns:
            Dictionary with 'intent_name', 'confidence', and 'reasoning'
        """
        prompt = self._build_classification_prompt(
            text, possible_intents, context
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an intent classifier for a chatbot."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                "intent_name": result.get("intent"),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
            # Return a safe fallback
            return {
                "intent_name": "unknown",
                "confidence": 0.0,
                "reasoning": f"LLM error: {str(e)}"
            }
    
    def _build_classification_prompt(self, text: str, intents: list, 
                                     context: Optional[str]) -> str:
        """Build prompt for intent classification."""
        intent_list = "\n".join([f"- {intent}" for intent in intents])
        
        prompt = f"""Classify the user's intent from their message.

Available intents:
{intent_list}

User message: "{text}"

{("Conversation context: " + context if context else "")}

Return JSON with:
1. "intent": the intent name from the list above, or "unknown" if none match
2. "confidence": number between 0 and 1
3. "reasoning": brief explanation of your choice

JSON:"""
        return prompt
```

**D. Core Logic Skeleton Code (LLM Fallback Classifier):**

```python
# enhancement/llm_fallback/llm_classifier.py
from typing import Optional
from chatbot.core.intent_classifier import IntentClassifier
from chatbot.core.data_models import UserUtterance, DialogState
from services.openai_client import OpenAIClient
import logging

class LLMFallbackClassifier:
    """
    Decorator/wrapper that adds LLM fallback to an existing intent classifier.
    
    This follows the same interface as IntentClassifier but calls an LLM
    when the primary classifier's confidence is below threshold.
    """
    
    def __init__(self, primary_classifier: IntentClassifier, 
                 llm_client: OpenAIClient,
                 confidence_threshold: float = 0.7,
                 use_cache: bool = True):
        self.primary = primary_classifier
        self.llm = llm_client
        self.threshold = confidence_threshold
        self.use_cache = use_cache
        self.cache = {}  # Simple in-memory cache
        self.logger = logging.getLogger(__name__)
    
    def predict(self, text: str, 
                state: Optional[DialogState] = None) -> UserUtterance:
        """
        Predict intent with primary classifier, fall back to LLM if low confidence.
        
        Args:
            text: User input text
            state: Current dialog state for context
            
        Returns:
            UserUtterance with intent and confidence
        """
        # TODO 1: First, try the primary classifier
        # Call self.primary.predict() with the same arguments
        
        # TODO 2: Check if primary classifier's confidence meets threshold
        # If utterance.confidence >= self.threshold, return the utterance
        
        # TODO 3: If below threshold, check cache if we've seen this text before
        # Hash the text (and maybe context) as cache key
        # If in cache and use_cache is True, return cached result
        
        # TODO 4: Prepare context string from dialog state
        # Extract relevant context: current intent, filled slots, last few turns
        
        # TODO 5: Get list of possible intents from primary classifier
        # The primary classifier should have access to all known intents
        
        # TODO 6: Call LLM client to classify intent
        # Use self.llm.classify_intent(text, possible_intents, context)
        
        # TODO 7: Convert LLM response to UserUtterance
        # Create new UserUtterance with raw_text=text, 
        # intent=Intent(name=llm_result['intent_name']),
        # confidence=llm_result['confidence']
        
        # TODO 8: Cache the result if caching is enabled
        # Store in self.cache with appropriate TTL/logic
        
        # TODO 9: Log the fallback occurrence for monitoring
        # Log confidence scores, which classifier was used
        
        # TODO 10: Return the final UserUtterance
        pass
    
    def _get_context_string(self, state: Optional[DialogState]) -> str:
        """
        Extract relevant context from dialog state for LLM prompt.
        
        Args:
            state: Current dialog state
            
        Returns:
            String representation of relevant context
        """
        if not state:
            return ""
        
        # TODO 1: Build context from current intent
        # If state.current_intent exists, include it
        
        # TODO 2: Add filled slots that might be relevant
        # Include key slot values that might help disambiguation
        
        # TODO 3: Add recent conversation history
        # Last 2-3 turns from state.conversation_history
        
        # TODO 4: Format as a coherent string for LLM prompt
        # Example: "Current intent: book_flight. Filled slots: destination=Paris."
        
        pass
```

**E. Language-Specific Hints:**

- **Caching Strategy**: Use `functools.lru_cache` for simple caching or `redis-py` for distributed caching in production
- **Async/Await**: If using asynchronous web frameworks (FastAPI, etc.), make the LLM calls async with `asyncio.to_thread` or `aiohttp` for the API calls
- **Error Handling**: Implement exponential backoff for LLM API calls using `tenacity` library
- **Configuration Management**: Store API keys and model parameters in environment variables or config files, not in code

**F. Milestone Checkpoint for LLM Fallback Enhancement:**

1. **Integration Test**: Create a test that sends ambiguous phrases and verifies LLM fallback activates:
   ```bash
   python -m pytest tests/test_llm_fallback.py -v
   ```
   
2. **Expected Behavior**: When primary classifier confidence < 0.7, system should log "Using LLM fallback for: [text]" and return a classification from the LLM

3. **Manual Verification**: 
   - Set debug mode to True in DialogManager
   - Send ambiguous input like "I want to arrange travel" (when trained on "book_flight")
   - Check logs to see fallback triggering
   - Response should still be appropriate despite ambiguous phrasing

4. **Monitoring**: Implement basic logging to track:
   - Fallback rate (percentage of queries using LLM)
   - LLM latency percentiles
   - Cache hit rate

**G. Debugging Tips for LLM Enhancements:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| LLM always returns "unknown" | Poor prompt engineering or intent list formatting | Examine the exact prompt sent to LLM in logs | Improve prompt, ensure intent list is clear |
| High latency on every request | No caching, LLM called for every utterance | Check cache hit rate metrics | Implement caching layer for common phrases |
| LLM returns intent not in list | Intent list not properly constrained in prompt | Verify prompt includes "or 'unknown'" instruction | Add validation to only accept intents from known list |
| Confidence scores inconsistent | LLM confidence not calibrated to primary classifier | Compare distributions of both classifiers | Apply scaling/normalization to LLM confidence scores |
| Cost escalating quickly | LLM called too frequently | Check fallback threshold (may be too aggressive) | Increase confidence threshold or add rate limiting |

---


## 11. Glossary

> **Milestone(s):** All Milestones (Foundational Terminology)

This section defines the key terms and concepts used throughout the Project Athena design document. A shared vocabulary is essential for clear communication about the chatbot's architecture, especially as the system involves specialized terminology from natural language processing, dialog systems, and software engineering. These definitions serve as an authoritative reference that bridges the conceptual models, component designs, and implementation details discussed in previous sections.

### Term Definitions and References

The following table provides comprehensive definitions for all significant terms used in the document, organized alphabetically. Each entry includes a clear definition, relevant mental models or analogies when helpful, and references to sections where the concept is discussed in depth.

| Term | Definition | Reference Section |
|------|------------|-------------------|
| **Assembly Line** | A mental model describing the **modular pipeline** architecture where user input flows sequentially through specialized processing stages (intent classification → entity extraction → dialog management → response generation), similar to how a product moves through different workstations in a factory. Each component transforms the input in a specific way before passing it to the next stage. | 3. High-Level Architecture, 6. Interactions and Data Flow |
| **Confidence Threshold** | A numerical cutoff value (typically 0.5-0.8) used by the **intent classifier** to distinguish reliable predictions from uncertain ones. If the classifier's highest probability score falls below this threshold, the input is labeled as **unknown intent** rather than assigned a potentially incorrect intent label. The constant `INTENT_CONFIDENCE_THRESHOLD` (0.7) defines this boundary. | 5.1 Component: Intent Classifier, Milestone 1 |
| **Conflict Resolution** | The strategy employed by the **entity extractor** when multiple extracted entities overlap in text spans (e.g., "New York" as both a location and part of "New York Times"). The system uses priority rules (longest match, most specific type, highest confidence) to select which entity to keep, ensuring consistent **slot filling**. | 5.2 Component: Entity Extractor and Slot Filler |
| **Context-Aware Classification** | An enhancement to intent classification where the **dialog state** (previous intent, filled slots) influences the classification of the current utterance. This helps interpret short follow-up messages like "tomorrow" or "the blue one" by narrowing the possible intent space based on conversation history. | 5.1 Component: Intent Classifier, 10. Future Extensions |
| **Context Carryover** | The ability of the **dialog manager** to maintain and utilize conversation context across multiple turns. When a user provides a follow-up message without restating their full intent (e.g., "What about tomorrow?" after asking about weather), the system carries over the previous **intent** and **filled slots** to interpret the incomplete input correctly. | 5.3 Component: Dialog Manager, 6. Interactions and Data Flow |
| **Conversation History** | A chronological record of exchanges within a **session**, stored as part of the **dialog state**. Each entry typically includes the user's raw input, the system's response, extracted **entities**, and the identified **intent**. This history enables context tracking, debugging, and potential learning from past interactions. | 4. Data Model, 5.3 Component: Dialog Manager |
| **Debugging** | The process of identifying, diagnosing, and fixing defects in the chatbot system. Given the interconnected nature of the **modular pipeline**, debugging often involves examining intermediate processing results (intent probabilities, extracted entities) and the evolving **dialog state** to pinpoint where understanding breaks down. | 9. Debugging Guide |
| **Dialog State** | The complete representation of a conversation's current context at a given moment. Defined by the `DialogState` data type, it tracks the **current intent**, **filled slots**, **missing slots**, confirmation status, activity timestamp, and **conversation history**. This state is persisted by the **session store** and updated by the **dialog manager** after each user turn. | 4. Data Model, 5.3 Component: Dialog Manager |
| **Entity** | A specific piece of structured information extracted from user text, such as a person's name, date, location, or product quantity. Defined by the `Entity` data type, each entity includes its type, raw text value, position in the input, confidence score, and normalized canonical value. Entities are the raw materials that become **slot** values. | 4. Data Model, 5.2 Component: Entity Extractor and Slot Filler |
| **Entity Extraction** | The process of identifying and extracting **entities** from natural language text. In Project Athena, this involves a **hybrid extraction** approach combining rule-based **regex patterns** (for structured formats like dates and times) with machine learning-based **NER** (for open-ended categories like person names and locations). | 5.2 Component: Entity Extractor and Slot Filler, Milestone 2 |
| **Entity Normalization** | The conversion of extracted entity text into canonical, standardized data formats suitable for programmatic use. For example, converting "tomorrow," "next Tuesday," and "04/09/2024" into ISO-8601 date strings, or "five" and "5" into the integer 5. Performed by the `EntityExtractor._normalize_entities()` method. | 5.2 Component: Entity Extractor and Slot Filler |
| **Fallback Response** | A generic, helpful response generated when the system cannot confidently understand the user's input (low intent confidence) or when no specific **response template** matches the current **dialog state**. Fallback responses typically ask for clarification or rephrasing, maintaining engagement even during failures. | 5.4 Component: Response Generator, 7. Error Handling and Edge Cases |
| **Global Command** | A user utterance that interrupts the normal dialog flow to perform system-level actions, regardless of the current conversation context. Examples include "reset," "start over," "help," or "cancel." Handled by the `DialogManager._handle_global_commands()` method before intent classification. | 5.3 Component: Dialog Manager, 6. Interactions and Data Flow |
| **Happy Path** | The ideal, simplest conversation scenario where a user provides a complete request in a single turn, including all required information. The system successfully identifies the **intent**, extracts all necessary **entities**, fills all required **slots**, and generates an appropriate response without needing clarification. | 6. Interactions and Data Flow |
| **Hybrid Extraction** | An entity extraction strategy that combines multiple techniques—typically rule-based **regex patterns** for well-structured entities (phone numbers, times) and statistical **NER** models for semantic entities (person names, organizations). This approach balances precision on predictable formats with generalization for open-ended categories. | 5.2 Component: Entity Extractor and Slot Filler |
| **Intent** | The user's goal, purpose, or desired action behind an utterance, represented as a categorical label. Defined by the `Intent` data type, each intent has a name (e.g., "book_flight"), a list of **required slots** (e.g., "destination", "date"), and optional slots. Intents are the primary way the system categorizes what the user wants to do. | 4. Data Model, 5.1 Component: Intent Classifier |
| **LLM Fallback** | A future extension where a Large Language Model (like GPT) serves as a backup classifier when the primary **intent classifier** produces low-confidence predictions. The `LLMFallbackClassifier` would use the LLM to re-analyze ambiguous input, potentially improving accuracy on edge cases without replacing the entire pipeline. | 10. Future Extensions |
| **Logging** | The practice of recording system events, state changes, and processing results to persistent storage (files, databases) for monitoring, auditing, and **debugging**. **Structured logging** formats these records in machine-parsable ways (JSON) to enable automated analysis and alerting. | 9. Debugging Guide |
| **Mental Model** | An intuitive analogy or conceptual framework used to explain a component's purpose and behavior before delving into technical details. Examples include "The Sorting Hat" for the intent classifier, "The Form Filler" for entity extraction, "The Conversation Conductor" for dialog management, and "The Mad Libs Assembler" for response generation. | Throughout component design sections (5.1-5.4) |
| **Modular Conversation Factory** | A metaphor describing the chatbot's extensible architecture, where each component (intent classifier, entity extractor, etc.) is a specialized "machine" that can be upgraded, replaced, or reconfigured independently. This design enables iterative improvement and adaptation to new domains without rebuilding the entire system. | 3. High-Level Architecture, 10. Future Extensions |
| **Modular Pipeline** | The core architectural pattern of Project Athena, consisting of independent, single-responsibility components connected in a sequential processing chain. Each component receives input from the previous stage, performs its specialized transformation, and passes output to the next stage. This separation of concerns improves testability, maintainability, and evolvability. | 3. High-Level Architecture |
| **Multi-Modal** | Supporting multiple input and output modalities beyond text, such as voice (speech-to-text and text-to-speech), vision (image understanding), or rich UI elements (buttons, cards). A future extension could involve a `VoiceEnabledDialogManager` that processes audio chunks and synthesizes spoken responses. | 10. Future Extensions |
| **Multi-Turn Dialogue** | A conversation spanning multiple exchanges between user and bot, typically required when the user initially provides incomplete information. The system must identify **missing slots**, prompt for them, and maintain context across turns until all required information is collected. Contrasts with **happy path** single-turn completion. | 6. Interactions and Data Flow |
| **Named Entity Recognition (NER)** | A subfield of natural language processing focused on identifying and classifying **entities** in text into predefined categories such as person names, organizations, locations, dates, and numerical expressions. In Project Athena, NER is performed using spaCy's pre-trained models as part of the **hybrid extraction** approach. | 5.2 Component: Entity Extractor and Slot Filler, Milestone 2 |
| **Orchestrator** | Another term for the **dialog manager**, emphasizing its role as the central coordinator that invokes other components (classifier, extractor, response generator) in the correct sequence, manages the **dialog state**, and determines the next system action based on the current conversation context. | 5.3 Component: Dialog Manager |
| **Placeholder** | Syntax markers within **response templates** that indicate where dynamic content should be inserted. In Project Athena, placeholders use curly brace notation like `{destination}` or `{date}`. During **variable substitution**, these placeholders are replaced with actual **slot** values from the **dialog state**. | 5.4 Component: Response Generator |
| **Proactive Suggestions** | A future enhancement where the chatbot, based on **conversation history** and inferred user goals, initiates turns to suggest relevant actions or information before the user explicitly requests them. This transforms the system from purely reactive to partially proactive, creating more natural and helpful interactions. | 10. Future Extensions |
| **Regex Patterns** | Regular expressions used for rule-based **entity extraction** of structured entities with predictable formats. Examples include patterns for phone numbers (`\d{3}-\d{3}-\d{4}`), times (`(\d{1,2}):(\d{2})\s*(am\|pm)?`), and product codes. Stored in the `EntityExtractor.patterns` dictionary and applied before ML-based NER. | 5.2 Component: Entity Extractor and Slot Filler |
| **Reference Librarian** | A mental model for a potential knowledge base integration extension, where the chatbot can query structured databases or document collections to answer factual questions. Like a librarian who knows where information is stored and how to retrieve it, this component would augment the core pipeline with external knowledge. | 10. Future Extensions |
| **Response Generator** | The component responsible for constructing natural language replies to the user. It selects appropriate **response templates** based on the current **dialog state**, performs **variable substitution** to insert **slot** values, and adds variation to avoid repetitive phrasing. The final step in the **modular pipeline**. | 5.4 Component: Response Generator, Milestone 4 |
| **Response Template** | A pre-written text pattern with **placeholders** for dynamic content, defined by the `ResponseTemplate` data type. Templates are organized into **variation groups** to provide multiple phrasings for the same scenario. The template's `required_slots` list ensures all necessary values are available before the template can be used. | 4. Data Model, 5.4 Component: Response Generator |
| **Semantic Slot Filling** | An advanced form of **slot filling** that extracts entities based on meaning and context rather than rigid patterns. For example, understanding that "the one with better reviews" refers to a product entity in a comparison context. Typically requires more sophisticated ML models and is noted as a future extension. | 10. Future Extensions |
| **Session** | A unique conversation instance between a specific user (or user device) and the chatbot, identified by a `session_id`. The **dialog state** for that session persists across multiple turns and is managed by the **session store**. Sessions may expire after `SESSION_TIMEOUT_SECONDS` of inactivity to free resources. | 5.3 Component: Dialog Manager |
| **Session Store** | The component responsible for persisting **dialog state** between user requests. The `InMemorySessionStore` provides a simple implementation using a dictionary, while production systems might use Redis or databases. Offers `get()` and `set()` methods for state retrieval and storage. | 5.3 Component: Dialog Manager |
| **Slot** | A named parameter that an **intent** requires or optionally accepts. Slots are filled with values extracted from **entities** (e.g., the `destination` slot gets the value "Paris" from a LOCATION entity). The **dialog state** tracks which slots are filled (`filled_slots`) and which are still **missing slots**. | 4. Data Model, 5.2 Component: Entity Extractor and Slot Filler |
| **Slot Filling** | The process of mapping extracted **entities** to the **slots** defined by the current **intent**. Performed by the `SlotFiller.fill_slots()` method, which uses a `slot_mappings` dictionary to specify which entity types correspond to which slot names (e.g., "LOCATION" → "destination"). | 5.2 Component: Entity Extractor and Slot Filler, Milestone 2 |
| **State Inspection** | The practice of examining the internal state of a component (particularly the **dialog state**) for **debugging** purposes. The `StateInspector` utility provides methods like `dump_state()` and `format_state_for_display()` to serialize and visualize state in human- and machine-readable formats. | 9. Debugging Guide |
| **State Machine** | A computational model used by the **dialog manager** to define discrete conversation states (e.g., `IDLE`, `COLLECTING_SLOTS`, `CONFIRMING`) and the transitions between them triggered by events (intent detected, slot received). This model provides structure to the conversation flow and ensures all necessary steps are completed. | 5.3 Component: Dialog Manager |
| **Structured Logging** | A logging approach where log entries follow a consistent, machine-parsable format (typically JSON) with well-defined fields. The `StructuredLogger` wrapper facilitates this by ensuring all log messages include relevant context like `session_id`, `intent`, and `timestamp`, enabling automated log analysis and monitoring. | 9. Debugging Guide |
| **Template** | Short for **response template**. | 5.4 Component: Response Generator |
| **Term Frequency-Inverse Document Frequency (TF-IDF)** | A numerical statistic and text vectorization method used by the **intent classifier** to convert raw text into numerical feature vectors. TF-IDF reflects how important a word is to a document in a collection, weighting frequent terms in the current document but penalizing terms that appear too commonly across all documents. Chosen over word embeddings for its simplicity and effectiveness on small datasets. | 5.1 Component: Intent Classifier |
| **Test Harness** | A framework for automated testing of chatbot components and complete conversation flows. The `ChatbotTestHarness` provides methods like `simulate_turn()` and `run_conversation()` to script interactions and verify expected behaviors, outputs, and state changes, forming a crucial part of the testing pyramid. | 8. Testing Strategy, 9. Debugging Guide |
| **Unknown Intent** | A special intent label assigned when the **intent classifier** cannot confidently match the user's input to any known intent (confidence below `INTENT_CONFIDENCE_THRESHOLD`). Triggers **fallback responses** asking for clarification, preventing the system from guessing incorrectly and executing the wrong action. | 5.1 Component: Intent Classifier, 7. Error Handling and Edge Cases |
| **Variable Substitution** | The process of replacing **placeholder** markers in **response templates** with actual values from the **dialog state**. Performed by the `ResponseGenerator.generate_response()` method, which extracts values from `filled_slots`, formats them via `_format_slot_value()`, and inserts them into the template string. | 5.4 Component: Response Generator |
| **Variation Group** | An identifier that groups multiple **response templates** offering alternative phrasings for the same conversational scenario. The `TemplateRegistry.get_random_template_for_group()` method randomly selects one template from the group, ensuring natural variation in responses and avoiding robotic repetition. | 5.4 Component: Response Generator |

### Implementation Guidance

While a glossary doesn't typically require implementation code, this is an appropriate place to provide a utility that helps enforce consistent terminology usage—a terminology validation script that checks for proper naming conventions in code and documentation.

**A. Technology Recommendations Table:**
| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Terminology Validation | Custom Python script using regex patterns | Integration with IDE linters (e.g., VS Code, PyCharm plugins) |

**B. Recommended File/Module Structure:**
```
project-athena/
  docs/
    glossary.md                    # This document
  scripts/
    validate_terminology.py        # Terminology validation script
  src/
    ...                            # Main source code
```

**C. Terminology Validation Script:**
This complete Python script can be run during development to ensure all code and documentation follows the naming conventions defined in this glossary.

```python
#!/usr/bin/env python3
"""
Terminology Validator for Project Athena

Scans Python source files and documentation for adherence to the project's
naming conventions and glossary terms. Helps maintain consistency across
the codebase.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Define required terminology patterns from the naming conventions
REQUIRED_TERMS: Set[str] = {
    # Types/Structs/Interfaces
    "Intent", "Entity", "UserUtterance", "DialogState", "ResponseTemplate",
    "EntityNormalizer", "EntityExtractor", "SlotFiller", "InMemorySessionStore",
    "StructuredLogger", "StateInspector", "ChatbotTestHarness", "DialogManager",
    "LLMFallbackClassifier", "OpenAIClient", "VoiceEnabledDialogManager",
    
    # Functions/Methods (partial list - include key ones)
    "predict", "extract_entities", "fill_slots", "process_utterance",
    "generate_response", "get", "set", "dump_state", "simulate_turn",
    
    # Constants
    "INTENT_CONFIDENCE_THRESHOLD", "SESSION_TIMEOUT_SECONDS",
    
    # Terminology (conceptual)
    "intent", "entity", "slot", "dialog state", "TF-IDF", "NER",
    "modular pipeline", "entity extraction", "entity normalization",
    "slot filling", "regex patterns", "conflict resolution",
    "hybrid extraction", "session", "state machine", "context carryover",
    "response generator", "template", "placeholder", "variable substitution",
    "variation group", "fallback response", "happy path", "multi-turn dialogue",
    "session store", "orchestrator", "assembly line", "debugging", "logging",
    "structured logging", "test harness", "state inspection",
    "modular conversation factory", "reference librarian", "LLM fallback",
    "multi-modal", "proactive suggestions", "semantic slot filling",
    "context-aware classification"
}

# Common incorrect variations that should be flagged
COMMON_MISTAKES: Dict[str, str] = {
    "IntentClassification": "Intent Classifier (or intent classification as concept)",
    "EntityExtraction": "entity extraction (lowercase for concept)",
    "DialogManagement": "dialog management (lowercase for concept)",
    "Slot": "slot (lowercase when referring to concept, uppercase for type)",
    "Utterance": "UserUtterance (type) or utterance (concept)",
    "TFIDF": "TF-IDF (with hyphen)",
    "NamedEntityRecognition": "NER (acronym) or named entity recognition (lowercase)",
    "Regex": "regex patterns (lowercase) or regex (lowercase acronym)",
}

class TerminologyValidator:
    """Validates that source code and docs use correct terminology."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.violations: List[Tuple[str, int, str]] = []  # (file, line_no, message)
        
    def scan_file(self, file_path: Path) -> None:
        """Scan a single file for terminology violations."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                self._check_line(file_path, line_num, line)
                
        except UnicodeDecodeError:
            # Skip binary files
            pass
    
    def _check_line(self, file_path: Path, line_num: int, line: str) -> None:
        """Check a single line for terminology issues."""
        
        # Check for common mistakes
        for wrong, correct in COMMON_MISTAKES.items():
            if re.search(r'\b' + re.escape(wrong) + r'\b', line):
                self.violations.append((
                    str(file_path),
                    line_num,
                    f"Terminology: Use '{correct}' instead of '{wrong}'"
                ))
        
        # In Python files, check for type names that should be used
        if file_path.suffix == '.py':
            # Look for class definitions that should match our types
            class_match = re.match(r'^\s*class\s+(\w+)', line)
            if class_match:
                class_name = class_match.group(1)
                # If it's similar to a required type but not exact
                if class_name.lower() in [t.lower() for t in REQUIRED_TERMS if t[0].isupper()]:
                    if class_name not in REQUIRED_TERMS:
                        similar = [t for t in REQUIRED_TERMS 
                                  if t[0].isupper() and t.lower() == class_name.lower()]
                        if similar:
                            self.violations.append((
                                str(file_path),
                                line_num,
                                f"Type name: Should be '{similar[0]}' not '{class_name}'"
                            ))
    
    def scan_directory(self, directory: Path, extensions: List[str] = None) -> None:
        """Recursively scan directory for files with given extensions."""
        if extensions is None:
            extensions = ['.py', '.md', '.txt']
            
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in extensions:
                self.scan_file(path)
    
    def print_violations(self) -> None:
        """Print all found violations in a readable format."""
        if not self.violations:
            print("✓ All files follow terminology conventions!")
            return
            
        print(f"Found {len(self.violations)} terminology violation(s):\n")
        
        # Group by file for better readability
        violations_by_file: Dict[str, List[Tuple[int, str]]] = {}
        for file_path, line_num, message in self.violations:
            violations_by_file.setdefault(file_path, []).append((line_num, message))
        
        for file_path, file_violations in violations_by_file.items():
            print(f"File: {file_path}")
            for line_num, message in sorted(file_violations):
                print(f"  Line {line_num}: {message}")
            print()
    
    def validate_glossary_coverage(self) -> None:
        """Check if all required terms appear somewhere in the codebase."""
        print("Checking glossary term coverage in codebase...")
        
        # This is a simplified check - in practice you'd want to
        # search for each term in all source files
        print("Note: Full term coverage validation requires more sophisticated search.")
        print("Current validation focuses on common mistakes and type names.")


def main():
    """Main entry point for the terminology validator."""
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path.cwd()
    
    if not root_dir.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        sys.exit(1)
    
    validator = TerminologyValidator(root_dir)
    
    # Scan Python and Markdown files
    validator.scan_directory(root_dir, ['.py', '.md'])
    
    # Print results
    validator.print_violations()
    
    # Check glossary coverage
    validator.validate_glossary_coverage()
    
    # Exit with error code if violations found
    if validator.violations:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
```

**D. Core Terminology Constants Module:**
While not required by the design, maintaining a central module with terminology constants can help ensure consistency. Here's a skeleton for such a module:

```python
# src/terminology.py
"""
Central definitions of Project Athena terminology constants.

This module provides string constants for all key terms to ensure
consistent usage throughout the codebase.
"""

# === CORE CONCEPT CONSTANTS ===
# Use these when referring to concepts in logs, documentation, or UI messages

INTENT_CONCEPT = "intent"
ENTITY_CONCEPT = "entity"
SLOT_CONCEPT = "slot"
DIALOG_STATE_CONCEPT = "dialog state"
SESSION_CONCEPT = "session"

# === TYPE NAME CONSTANTS ===
# Use these when you need to reference type names as strings
# (e.g., in serialization, error messages)

INTENT_TYPE = "Intent"
ENTITY_TYPE = "Entity"
USER_UTTERANCE_TYPE = "UserUtterance"
DIALOG_STATE_TYPE = "DialogState"
RESPONSE_TEMPLATE_TYPE = "ResponseTemplate"

# === METHOD GROUP CONSTANTS ===
# For categorizing methods in documentation

CLASSIFICATION_METHODS = "classification methods"
EXTRACTION_METHODS = "extraction methods"
DIALOG_METHODS = "dialog management methods"
RESPONSE_METHODS = "response generation methods"

# === ERROR CATEGORIES ===
# Standard error types for consistent error handling

ERROR_INTENT_UNKNOWN = "intent_unknown"
ERROR_ENTITY_MISSING = "entity_missing"
ERROR_SLOT_CONFLICT = "slot_conflict"
ERROR_SESSION_EXPIRED = "session_expired"

def validate_term(term: str) -> bool:
    """
    Validate that a term follows Project Athena conventions.
    
    Args:
        term: The term to validate
        
    Returns:
        True if term is valid according to conventions
        
    Example:
        >>> validate_term("Intent")
        True
        >>> validate_term("intentClassification")  # Wrong case
        False
    """
    # TODO 1: Check if term matches any required term (case-insensitive)
    # TODO 2: For type names, verify PascalCase
    # TODO 3: For concept names, verify lowercase_with_underscores or spaced lowercase
    # TODO 4: For constants, verify UPPER_CASE_WITH_UNDERSCORES
    # TODO 5: Return True if valid, False otherwise
    pass


def get_term_variations(term: str) -> List[str]:
    """
    Get all valid variations of a term (different cases, spacings).
    
    Useful for fuzzy matching or generating user-friendly messages.
    
    Args:
        term: The base term
        
    Returns:
        List of valid variations
        
    Example:
        >>> get_term_variations("dialog state")
        ["dialog state", "DialogState", "DIALOG_STATE"]
    """
    # TODO 1: For concept terms (lowercase with spaces), generate:
    #         - Original (e.g., "dialog state")
    #         - Type name (PascalCase, e.g., "DialogState")
    #         - Constant name (UPPER_CASE, e.g., "DIALOG_STATE")
    # TODO 2: For type names (PascalCase), generate:
    #         - Original (e.g., "DialogState")
    #         - Concept name (lowercase with spaces, e.g., "dialog state")
    #         - Constant name (UPPER_CASE, e.g., "DIALOG_STATE")
    # TODO 3: Return list of all valid variations
    pass
```

**E. Language-Specific Hints:**
- **Python Enums for State Machines:** Consider using Python's `enum.Enum` class to define dialog states (e.g., `DialogStateEnum.IDLE`, `DialogStateEnum.COLLECTING_SLOTS`) for type safety and autocompletion.
- **Type Hints for Glossary Terms:** Use `typing.TypeAlias` to create aliases for complex types, making code more readable: `DialogStateDict = Dict[str, Any]`.
- **Docstring Terminology:** Include `.. glossary::` directives in Sphinx documentation to automatically link glossary terms to their definitions.

**F. Milestone Checkpoint:**
After implementing the chatbot, run the terminology validator to ensure consistency:
```bash
python scripts/validate_terminology.py src/
```
Expected output should show "✓ All files follow terminology conventions!" or list specific violations to fix. This checkpoint helps maintain code quality and ensures the implementation matches the design vocabulary.

**G. Debugging Tips:**
| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Validator reports many "Terminology" violations | Inconsistent naming in code vs. glossary | Run validator on a small file first, examine patterns | Update code to use terms from `REQUIRED_TERMS` list |
| Validator misses obvious naming issues | Regex patterns don't match all cases | Test validator on known bad examples | Expand pattern matching in `_check_line()` method |
| Constant naming conflicts (e.g., `Intent` vs `INTENT`) | Case sensitivity in terminology | Check if term exists in multiple cases in glossary | Standardize to one case per concept type |

This glossary and associated validation tools ensure that as Project Athena evolves, the entire team speaks the same language—literally and figuratively—about the chatbot's architecture and behavior.

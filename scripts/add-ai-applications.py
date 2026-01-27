#!/usr/bin/env python3
"""
Add AI Applications projects - modern LLM-powered applications.
Focus on production-ready AI systems that enterprises need.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

# New AI Application projects
ai_projects = {
    "chatbot-intent": {
        "id": "chatbot-intent",
        "name": "Intent-Based Chatbot",
        "description": "Build a chatbot with intent classification and entity extraction. Foundation for conversational AI without LLMs.",
        "difficulty": "beginner",
        "estimated_hours": "15-25",
        "prerequisites": ["Python basics", "Text processing", "Basic ML concepts"],
        "languages": {
            "recommended": ["Python"],
            "also_possible": ["JavaScript", "Go"]
        },
        "resources": [
            {"name": "Rasa Open Source", "url": "https://rasa.com/docs/rasa/", "type": "documentation"},
            {"name": "Intent Classification Tutorial", "url": "https://www.tensorflow.org/tutorials/text/text_classification_rnn", "type": "tutorial"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Intent Classification",
                "description": "Build intent classifier from training examples.",
                "acceptance_criteria": [
                    "Load training data (intent -> examples)",
                    "Vectorize text (TF-IDF or embeddings)",
                    "Train classifier (Naive Bayes or SVM)",
                    "Predict intent with confidence score",
                    "Handle unknown/low-confidence intents"
                ],
                "hints": {
                    "level1": "Start with TF-IDF + Naive Bayes. Simple but effective baseline.",
                    "level2": "Use scikit-learn Pipeline for clean preprocessing + classification.",
                    "level3": """from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

class IntentClassifier:
    def __init__(self, confidence_threshold=0.5):
        self.threshold = confidence_threshold
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', MultinomialNB())
        ])
        self.intents = []

    def train(self, texts, intents):
        self.intents = list(set(intents))
        self.pipeline.fit(texts, intents)

    def predict(self, text):
        probs = self.pipeline.predict_proba([text])[0]
        max_idx = probs.argmax()
        confidence = probs[max_idx]

        if confidence < self.threshold:
            return {"intent": "unknown", "confidence": confidence}

        return {
            "intent": self.pipeline.classes_[max_idx],
            "confidence": float(confidence)
        }

# Training data
training_data = [
    ("What's the weather like?", "weather"),
    ("Is it going to rain?", "weather"),
    ("Set an alarm for 7am", "alarm"),
    ("Wake me up at 6", "alarm"),
    # ... more examples
]
texts, intents = zip(*training_data)
clf = IntentClassifier()
clf.train(texts, intents)"""
                },
                "pitfalls": [
                    "Too few training examples per intent",
                    "Imbalanced classes skew predictions",
                    "Not handling out-of-domain inputs",
                    "Overfitting on small datasets"
                ],
                "concepts": ["Text classification", "TF-IDF", "Naive Bayes", "Confidence thresholds"],
                "estimated_hours": "4-6"
            },
            {
                "id": 2,
                "name": "Entity Extraction",
                "description": "Extract entities (names, dates, numbers) from user input.",
                "acceptance_criteria": [
                    "Define entity types for your domain",
                    "Rule-based extraction (regex, patterns)",
                    "Named Entity Recognition (NER)",
                    "Slot filling for intents",
                    "Handle multiple entities in one message"
                ],
                "hints": {
                    "level1": "Combine regex for structured data + spaCy NER for names/dates.",
                    "level2": "Create entity templates per intent. 'set alarm for {time}' -> extract time.",
                    "level3": """import re
import spacy
from dateutil import parser as date_parser

class EntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.patterns = {
            'time': r'\\b(\\d{1,2}(?::\\d{2})?\\s*(?:am|pm)?|\\d{1,2}\\s*o\'?clock)\\b',
            'duration': r'\\b(\\d+)\\s*(minutes?|hours?|seconds?)\\b',
            'number': r'\\b(\\d+(?:\\.\\d+)?)\\b'
        }

    def extract(self, text, intent=None):
        entities = {}

        # spaCy NER for standard entities
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ('PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'MONEY'):
                entities[ent.label_.lower()] = ent.text

        # Custom patterns
        for entity_type, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities[entity_type] = match.group(0)

        # Parse dates/times
        if 'time' in entities:
            try:
                entities['parsed_time'] = date_parser.parse(entities['time'])
            except:
                pass

        return entities

# Slot filling
class SlotFiller:
    def __init__(self):
        self.intent_slots = {
            'set_alarm': ['time'],
            'book_restaurant': ['date', 'time', 'party_size', 'restaurant'],
            'send_email': ['recipient', 'subject']
        }

    def fill_slots(self, intent, entities):
        required = self.intent_slots.get(intent, [])
        filled = {slot: entities.get(slot) for slot in required}
        missing = [s for s in required if not filled.get(s)]
        return {'slots': filled, 'missing': missing}"""
                },
                "pitfalls": [
                    "Regex too greedy matches wrong text",
                    "spaCy model not loaded (need python -m spacy download)",
                    "Date parsing ambiguity (01/02 = Jan 2 or Feb 1?)",
                    "Overlapping entity matches"
                ],
                "concepts": ["Named Entity Recognition", "Regex patterns", "Slot filling", "spaCy"],
                "estimated_hours": "4-6"
            },
            {
                "id": 3,
                "name": "Dialog Management",
                "description": "Manage multi-turn conversations with context.",
                "acceptance_criteria": [
                    "Track conversation state",
                    "Handle follow-up questions",
                    "Ask clarifying questions for missing slots",
                    "Support conversation reset",
                    "Context timeout/expiry"
                ],
                "hints": {
                    "level1": "Store conversation state in dict keyed by session_id.",
                    "level2": "State machine: INIT -> COLLECTING_SLOTS -> CONFIRMING -> EXECUTING",
                    "level3": """from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import time

class DialogState(Enum):
    INIT = "init"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    COMPLETE = "complete"

@dataclass
class ConversationContext:
    session_id: str
    state: DialogState = DialogState.INIT
    current_intent: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    history: list = field(default_factory=list)
    last_active: float = field(default_factory=time.time)

    def is_expired(self, timeout=300):
        return time.time() - self.last_active > timeout

class DialogManager:
    def __init__(self, intent_classifier, entity_extractor, slot_filler):
        self.classifier = intent_classifier
        self.extractor = entity_extractor
        self.filler = slot_filler
        self.contexts: Dict[str, ConversationContext] = {}

    def process(self, session_id: str, user_input: str) -> str:
        ctx = self._get_or_create_context(session_id)
        ctx.last_active = time.time()
        ctx.history.append({'role': 'user', 'text': user_input})

        # Intent classification
        if ctx.state == DialogState.INIT:
            result = self.classifier.predict(user_input)
            ctx.current_intent = result['intent']
            ctx.state = DialogState.COLLECTING

        # Entity extraction
        entities = self.extractor.extract(user_input, ctx.current_intent)
        ctx.slots.update(entities)

        # Check missing slots
        slot_result = self.filler.fill_slots(ctx.current_intent, ctx.slots)

        if slot_result['missing']:
            response = self._ask_for_slot(slot_result['missing'][0])
        else:
            ctx.state = DialogState.CONFIRMING
            response = self._confirm_action(ctx)

        ctx.history.append({'role': 'assistant', 'text': response})
        return response

    def _ask_for_slot(self, slot_name):
        prompts = {
            'time': "What time would you like?",
            'date': "What date?",
            'recipient': "Who should I send it to?"
        }
        return prompts.get(slot_name, f"Please provide {slot_name}")"""
                },
                "pitfalls": [
                    "Memory leak from never-expiring sessions",
                    "Losing context on intent change",
                    "Infinite loops in dialog flow",
                    "Not handling 'cancel' or 'start over'"
                ],
                "concepts": ["State machines", "Session management", "Multi-turn dialog", "Context tracking"],
                "estimated_hours": "5-8"
            },
            {
                "id": 4,
                "name": "Response Generation",
                "description": "Generate natural language responses with templates.",
                "acceptance_criteria": [
                    "Template-based responses",
                    "Variable substitution",
                    "Response variations (avoid repetition)",
                    "Personality/tone consistency",
                    "Error message handling"
                ],
                "hints": {
                    "level1": "Use Jinja2 templates with random.choice for variations.",
                    "level2": "Group responses by intent and state. Add personality traits.",
                    "level3": """import random
from jinja2 import Template
from dataclasses import dataclass

@dataclass
class ResponseTemplate:
    templates: list

    def render(self, **kwargs):
        template = random.choice(self.templates)
        return Template(template).render(**kwargs)

class ResponseGenerator:
    def __init__(self):
        self.responses = {
            ('weather', 'success'): ResponseTemplate([
                "It's currently {{temp}}°F and {{condition}} in {{location}}.",
                "The weather in {{location}}: {{temp}}°F, {{condition}}.",
                "{{location}} is {{condition}} right now, {{temp}}°F."
            ]),
            ('alarm', 'success'): ResponseTemplate([
                "Done! I've set an alarm for {{time}}.",
                "Alarm set for {{time}}. I'll wake you up!",
                "Got it - {{time}} alarm is ready."
            ]),
            ('alarm', 'confirm'): ResponseTemplate([
                "Just to confirm: set an alarm for {{time}}?",
                "You want me to wake you at {{time}}, correct?"
            ]),
            ('error', 'not_understood'): ResponseTemplate([
                "I'm not sure I understand. Could you rephrase?",
                "Sorry, I didn't catch that. Can you try again?",
                "I'm having trouble understanding. What would you like?"
            ]),
            ('error', 'missing_slot'): ResponseTemplate([
                "I need a bit more info. {{question}}",
                "{{question}}"
            ])
        }

        # Personality traits
        self.personality = {
            'greeting_prefix': ["Sure!", "Absolutely!", "Of course!", ""],
            'filler_words': ["just", "actually", ""],
        }

    def generate(self, intent, state, **context):
        key = (intent, state)
        if key not in self.responses:
            key = ('error', 'not_understood')

        response = self.responses[key].render(**context)

        # Add personality
        if random.random() < 0.3:
            prefix = random.choice(self.personality['greeting_prefix'])
            if prefix:
                response = f"{prefix} {response}"

        return response"""
                },
                "pitfalls": [
                    "Templates not escaping user input (XSS)",
                    "Missing template variables cause errors",
                    "Too few variations feel robotic",
                    "Inconsistent tone across responses"
                ],
                "concepts": ["Template engines", "NLG basics", "Response variation", "Personality design"],
                "estimated_hours": "3-5"
            }
        ]
    },

    "rag-system": {
        "id": "rag-system",
        "name": "RAG System (Retrieval Augmented Generation)",
        "description": "Build a production RAG pipeline: document ingestion, chunking, embedding, vector search, and LLM generation with retrieved context.",
        "difficulty": "intermediate",
        "estimated_hours": "30-50",
        "prerequisites": ["Python", "Basic ML concepts", "REST APIs", "Database basics"],
        "languages": {
            "recommended": ["Python"],
            "also_possible": ["TypeScript", "Go"]
        },
        "resources": [
            {"name": "LangChain RAG Tutorial", "url": "https://python.langchain.com/docs/tutorials/rag/", "type": "tutorial"},
            {"name": "OpenAI Embeddings", "url": "https://platform.openai.com/docs/guides/embeddings", "type": "documentation"},
            {"name": "Pinecone Vector DB", "url": "https://docs.pinecone.io/", "type": "documentation"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Document Ingestion & Chunking",
                "description": "Load documents and split into optimal chunks for retrieval.",
                "acceptance_criteria": [
                    "Load PDF, Markdown, HTML documents",
                    "Extract text with metadata preservation",
                    "Implement chunking strategies (fixed, semantic, recursive)",
                    "Handle chunk overlap for context continuity",
                    "Track source document and position"
                ],
                "hints": {
                    "level1": "Start with fixed-size chunks (500-1000 tokens) with 100 token overlap.",
                    "level2": "Semantic chunking: split on paragraphs/sections, then enforce max size.",
                    "level3": """from dataclasses import dataclass
from typing import List, Optional
import tiktoken
from pypdf import PdfReader
import markdown
from bs4 import BeautifulSoup

@dataclass
class Chunk:
    text: str
    metadata: dict  # source, page, position, etc.
    token_count: int

class DocumentLoader:
    def load(self, path: str) -> str:
        if path.endswith('.pdf'):
            return self._load_pdf(path)
        elif path.endswith('.md'):
            return self._load_markdown(path)
        elif path.endswith('.html'):
            return self._load_html(path)
        else:
            return open(path).read()

    def _load_pdf(self, path):
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            pages.append({'text': text, 'page': i + 1})
        return pages

class TextChunker:
    def __init__(self, chunk_size=500, overlap=100, model="gpt-4"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.encoding_for_model(model)

    def chunk_text(self, text: str, metadata: dict = None) -> List[Chunk]:
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            chunks.append(Chunk(
                text=chunk_text,
                metadata={**(metadata or {}), 'start_token': start},
                token_count=len(chunk_tokens)
            ))

            start = end - self.overlap

        return chunks

    def semantic_chunk(self, text: str, metadata: dict = None) -> List[Chunk]:
        # Split on semantic boundaries first
        paragraphs = text.split('\\n\\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self.encoder.encode(para))

            if current_tokens + para_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(self._make_chunk(current_chunk, metadata))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append(self._make_chunk(current_chunk, metadata))

        return chunks"""
                },
                "pitfalls": [
                    "Chunks too small lose context",
                    "Chunks too large exceed LLM context window",
                    "PDF extraction loses formatting/tables",
                    "Not handling encoding issues (UTF-8)"
                ],
                "concepts": ["Document parsing", "Text chunking", "Tokenization", "Metadata tracking"],
                "estimated_hours": "6-10"
            },
            {
                "id": 2,
                "name": "Embedding Generation",
                "description": "Convert text chunks to vector embeddings for semantic search.",
                "acceptance_criteria": [
                    "Generate embeddings using OpenAI/local model",
                    "Batch processing for efficiency",
                    "Handle rate limits and retries",
                    "Cache embeddings to avoid recomputation",
                    "Support multiple embedding models"
                ],
                "hints": {
                    "level1": "Use OpenAI text-embedding-3-small for good quality/cost ratio.",
                    "level2": "Batch requests (max 2048 texts per call). Implement exponential backoff.",
                    "level3": """import openai
from tenacity import retry, wait_exponential, stop_after_attempt
import numpy as np
from typing import List
import hashlib
import pickle
from pathlib import Path

class EmbeddingService:
    def __init__(self, model="text-embedding-3-small", cache_dir=".cache/embeddings"):
        self.model = model
        self.client = openai.OpenAI()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = 1536  # OpenAI small model

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()

    def _get_cached(self, text: str) -> Optional[np.ndarray]:
        cache_path = self.cache_dir / f"{self._cache_key(text)}.pkl"
        if cache_path.exists():
            return pickle.load(open(cache_path, 'rb'))
        return None

    def _set_cached(self, text: str, embedding: np.ndarray):
        cache_path = self.cache_dir / f"{self._cache_key(text)}.pkl"
        pickle.dump(embedding, open(cache_path, 'wb'))

    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [np.array(e.embedding) for e in response.data]

    def embed(self, texts: List[str], batch_size=100) -> List[np.ndarray]:
        results = [None] * len(texts)
        to_embed = []  # (index, text) pairs

        # Check cache first
        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                results[i] = cached
            else:
                to_embed.append((i, text))

        # Batch embed uncached
        for batch_start in range(0, len(to_embed), batch_size):
            batch = to_embed[batch_start:batch_start + batch_size]
            indices, batch_texts = zip(*batch) if batch else ([], [])

            embeddings = self._embed_batch(list(batch_texts))

            for idx, text, emb in zip(indices, batch_texts, embeddings):
                self._set_cached(text, emb)
                results[idx] = emb

        return results

# Local embedding alternative with sentence-transformers
class LocalEmbedding:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        return self.model.encode(texts, convert_to_numpy=True)"""
                },
                "pitfalls": [
                    "Rate limits cause failures without retry logic",
                    "Large batches exceed API limits",
                    "Not normalizing embeddings for cosine similarity",
                    "Embedding dimension mismatch between models"
                ],
                "concepts": ["Text embeddings", "API rate limiting", "Caching strategies", "Batch processing"],
                "estimated_hours": "4-6"
            },
            {
                "id": 3,
                "name": "Vector Store & Retrieval",
                "description": "Store embeddings and perform similarity search.",
                "acceptance_criteria": [
                    "Store vectors with metadata",
                    "K-nearest neighbor search",
                    "Hybrid search (vector + keyword)",
                    "Filtering by metadata",
                    "Support Pinecone/Chroma/pgvector"
                ],
                "hints": {
                    "level1": "Start with Chroma (local, no setup). Graduate to Pinecone for scale.",
                    "level2": "Hybrid search: combine BM25 keyword score with vector similarity.",
                    "level3": """import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi

class VectorStore:
    def __init__(self, collection_name="documents", persist_dir=".chroma"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._bm25 = None
        self._doc_texts = []

    def add(self, chunks: List[Chunk], embeddings: List[np.ndarray]):
        ids = [f"chunk_{i}_{hash(c.text)}" for i, c in enumerate(chunks)]

        self.collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )

        # Update BM25 index
        self._doc_texts.extend([c.text for c in chunks])
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        tokenized = [doc.lower().split() for doc in self._doc_texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query_embedding: np.ndarray, query_text: str,
               k: int = 5, filter: Dict = None, hybrid_alpha: float = 0.5) -> List[Dict]:

        # Vector search
        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k * 2,  # Get more for reranking
            where=filter
        )

        if not self._bm25 or hybrid_alpha == 1.0:
            return self._format_results(vector_results)

        # BM25 search
        tokenized_query = query_text.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Combine scores
        combined = []
        for i, (doc_id, doc, metadata, distance) in enumerate(zip(
            vector_results['ids'][0],
            vector_results['documents'][0],
            vector_results['metadatas'][0],
            vector_results['distances'][0]
        )):
            vector_score = 1 - distance  # Convert distance to similarity

            # Find BM25 score for this doc
            try:
                bm25_idx = self._doc_texts.index(doc)
                bm25_score = bm25_scores[bm25_idx] / max(bm25_scores)
            except ValueError:
                bm25_score = 0

            combined_score = hybrid_alpha * vector_score + (1 - hybrid_alpha) * bm25_score
            combined.append({
                'id': doc_id,
                'text': doc,
                'metadata': metadata,
                'score': combined_score
            })

        # Sort by combined score and return top k
        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined[:k]"""
                },
                "pitfalls": [
                    "Using wrong distance metric (cosine vs L2)",
                    "Not normalizing scores for hybrid search",
                    "Memory issues with large collections",
                    "Stale BM25 index after updates"
                ],
                "concepts": ["Vector databases", "ANN search", "BM25", "Hybrid retrieval"],
                "estimated_hours": "6-8"
            },
            {
                "id": 4,
                "name": "LLM Integration & Prompting",
                "description": "Generate answers using retrieved context with LLM.",
                "acceptance_criteria": [
                    "Build effective RAG prompt template",
                    "Handle context window limits",
                    "Stream responses",
                    "Handle LLM errors gracefully",
                    "Support multiple LLM providers"
                ],
                "hints": {
                    "level1": "Simple template: 'Answer based on context: {context}\\n\\nQuestion: {question}'",
                    "level2": "Add instructions: cite sources, say 'I don't know' if not in context.",
                    "level3": """from openai import OpenAI
from anthropic import Anthropic
from typing import Generator, List
import tiktoken

class LLMService:
    def __init__(self, provider="openai", model="gpt-4-turbo-preview"):
        self.provider = provider
        self.model = model
        if provider == "openai":
            self.client = OpenAI()
            self.encoder = tiktoken.encoding_for_model(model)
        elif provider == "anthropic":
            self.client = Anthropic()

        self.max_context = 128000  # GPT-4 Turbo

    def _build_rag_prompt(self, question: str, contexts: List[Dict]) -> str:
        context_text = "\\n\\n---\\n\\n".join([
            f"[Source: {c['metadata'].get('source', 'unknown')}]\\n{c['text']}"
            for c in contexts
        ])

        return f'''You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context_text}

INSTRUCTIONS:
- Answer the question based ONLY on the context above
- If the answer is not in the context, say "I don't have enough information to answer this"
- Cite your sources using [Source: filename] format
- Be concise but complete

QUESTION: {question}

ANSWER:'''

    def _truncate_contexts(self, contexts: List[Dict], max_tokens: int) -> List[Dict]:
        result = []
        total_tokens = 0

        for ctx in contexts:
            ctx_tokens = len(self.encoder.encode(ctx['text']))
            if total_tokens + ctx_tokens > max_tokens:
                break
            result.append(ctx)
            total_tokens += ctx_tokens

        return result

    def generate(self, question: str, contexts: List[Dict],
                 stream: bool = True) -> Generator[str, None, None]:
        # Reserve tokens for question and response
        max_context_tokens = self.max_context - 4000
        contexts = self._truncate_contexts(contexts, max_context_tokens)

        prompt = self._build_rag_prompt(question, contexts)

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=stream,
                temperature=0.1
            )

            if stream:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                yield response.choices[0].message.content

class RAGPipeline:
    def __init__(self, embedding_service, vector_store, llm_service):
        self.embedder = embedding_service
        self.store = vector_store
        self.llm = llm_service

    def query(self, question: str, k: int = 5) -> Generator[str, None, None]:
        # Embed question
        query_embedding = self.embedder.embed([question])[0]

        # Retrieve relevant chunks
        contexts = self.store.search(query_embedding, question, k=k)

        # Generate response
        yield from self.llm.generate(question, contexts)"""
                },
                "pitfalls": [
                    "Exceeding context window with too many chunks",
                    "LLM hallucinating despite 'only use context' instruction",
                    "Not handling streaming errors mid-response",
                    "Prompt injection from retrieved content"
                ],
                "concepts": ["Prompt engineering", "Context window management", "Streaming responses", "LLM providers"],
                "estimated_hours": "6-8"
            },
            {
                "id": 5,
                "name": "Evaluation & Optimization",
                "description": "Measure and improve RAG quality.",
                "acceptance_criteria": [
                    "Implement retrieval metrics (recall, MRR)",
                    "Implement generation metrics (faithfulness, relevance)",
                    "Create evaluation dataset",
                    "A/B test different configurations",
                    "Implement reranking"
                ],
                "hints": {
                    "level1": "Create golden dataset: questions + expected chunks + expected answers.",
                    "level2": "Use LLM-as-judge for generation quality (faithfulness, relevance).",
                    "level3": """from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class EvalSample:
    question: str
    expected_chunks: List[str]  # Ground truth relevant chunks
    expected_answer: str

class RAGEvaluator:
    def __init__(self, rag_pipeline, llm_judge):
        self.rag = rag_pipeline
        self.judge = llm_judge

    def evaluate_retrieval(self, samples: List[EvalSample], k: int = 5) -> Dict:
        recalls = []
        mrrs = []

        for sample in samples:
            query_emb = self.rag.embedder.embed([sample.question])[0]
            retrieved = self.rag.store.search(query_emb, sample.question, k=k)
            retrieved_texts = [r['text'] for r in retrieved]

            # Recall@k
            hits = sum(1 for exp in sample.expected_chunks
                      if any(exp in ret for ret in retrieved_texts))
            recall = hits / len(sample.expected_chunks)
            recalls.append(recall)

            # MRR
            for i, ret in enumerate(retrieved_texts):
                if any(exp in ret for exp in sample.expected_chunks):
                    mrrs.append(1 / (i + 1))
                    break
            else:
                mrrs.append(0)

        return {
            'recall@k': np.mean(recalls),
            'mrr': np.mean(mrrs)
        }

    def evaluate_generation(self, samples: List[EvalSample]) -> Dict:
        faithfulness_scores = []
        relevance_scores = []

        for sample in samples:
            # Get RAG response
            response = ''.join(self.rag.query(sample.question))
            query_emb = self.rag.embedder.embed([sample.question])[0]
            contexts = self.rag.store.search(query_emb, sample.question, k=5)

            # Faithfulness: is response grounded in context?
            faithfulness = self.judge.evaluate(
                prompt=f'''Rate if the response is fully supported by the context.
Context: {[c["text"] for c in contexts]}
Response: {response}
Score 1-5 (5=fully grounded):''',
                parse_fn=lambda x: int(x.strip()) / 5
            )
            faithfulness_scores.append(faithfulness)

            # Relevance: does response answer the question?
            relevance = self.judge.evaluate(
                prompt=f'''Rate if the response answers the question.
Question: {sample.question}
Response: {response}
Score 1-5 (5=fully answers):''',
                parse_fn=lambda x: int(x.strip()) / 5
            )
            relevance_scores.append(relevance)

        return {
            'faithfulness': np.mean(faithfulness_scores),
            'relevance': np.mean(relevance_scores)
        }

# Reranking for better retrieval
class Reranker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        pairs = [(query, doc['text']) for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)

        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        return documents[:top_k]"""
                },
                "pitfalls": [
                    "Evaluation set too small to be statistically significant",
                    "LLM judge bias toward verbose responses",
                    "Not versioning evaluation datasets",
                    "Overfitting to evaluation set"
                ],
                "concepts": ["Retrieval metrics", "Generation evaluation", "LLM-as-judge", "Cross-encoder reranking"],
                "estimated_hours": "8-12"
            }
        ]
    },

    "semantic-search": {
        "id": "semantic-search",
        "name": "Semantic Search Engine",
        "description": "Build a semantic search engine that understands meaning, not just keywords. Core technology behind modern search.",
        "difficulty": "intermediate",
        "estimated_hours": "25-40",
        "prerequisites": ["Python", "Embeddings basics", "Database fundamentals"],
        "languages": {
            "recommended": ["Python", "Go"],
            "also_possible": ["TypeScript", "Rust"]
        },
        "resources": [
            {"name": "Sentence Transformers", "url": "https://www.sbert.net/", "type": "documentation"},
            {"name": "FAISS Library", "url": "https://github.com/facebookresearch/faiss", "type": "documentation"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Embedding Index",
                "description": "Build efficient vector index for similarity search.",
                "acceptance_criteria": [
                    "Index millions of vectors efficiently",
                    "Sub-second query latency",
                    "Support incremental updates",
                    "Implement HNSW or IVF index",
                    "Memory-mapped for large datasets"
                ],
                "hints": {
                    "level1": "FAISS IVF for large scale, HNSW for accuracy. Start with Flat for small data.",
                    "level2": "IVF: cluster vectors, search only nearby clusters. HNSW: navigable small world graph.",
                    "level3": """import faiss
import numpy as np
from typing import List, Tuple

class VectorIndex:
    def __init__(self, dimension: int, index_type: str = "hnsw"):
        self.dimension = dimension
        self.index_type = index_type
        self._build_index()
        self.id_map = {}  # internal_id -> external_id
        self.next_id = 0

    def _build_index(self):
        if self.index_type == "flat":
            # Exact search, O(n)
            self.index = faiss.IndexFlatIP(self.dimension)

        elif self.index_type == "ivf":
            # Inverted file index, faster for large datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
            self.needs_training = True

        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World - best accuracy/speed tradeoff
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 connections per node
            self.index.hnsw.efConstruction = 200  # Build-time quality
            self.index.hnsw.efSearch = 50  # Search-time quality

    def add(self, vectors: np.ndarray, ids: List[str]):
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        # Train if needed (IVF)
        if hasattr(self, 'needs_training') and self.needs_training:
            self.index.train(vectors)
            self.needs_training = False

        # Map external IDs to internal
        for ext_id in ids:
            self.id_map[self.next_id] = ext_id
            self.next_id += 1

        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        query = query.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        distances, indices = self.index.search(query, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # -1 means not found
                ext_id = self.id_map.get(idx, str(idx))
                results.append((ext_id, float(dist)))

        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        np.save(f"{path}_idmap.npy", self.id_map)

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        self.id_map = np.load(f"{path}_idmap.npy", allow_pickle=True).item()"""
                },
                "pitfalls": [
                    "Not normalizing vectors for cosine similarity",
                    "IVF requires training before adding",
                    "Memory explosion with large HNSW M parameter",
                    "ID mapping gets out of sync"
                ],
                "concepts": ["HNSW algorithm", "IVF indexing", "Approximate nearest neighbors", "Index persistence"],
                "estimated_hours": "6-10"
            },
            {
                "id": 2,
                "name": "Query Processing",
                "description": "Parse and enhance queries for better results.",
                "acceptance_criteria": [
                    "Query expansion (synonyms, related terms)",
                    "Query understanding (intent, entities)",
                    "Multi-vector queries",
                    "Negative queries ('without X')",
                    "Query caching"
                ],
                "hints": {
                    "level1": "Expand queries with WordNet synonyms or use LLM to generate variations.",
                    "level2": "Multi-vector: average embeddings of query expansion. Weight original higher.",
                    "level3": """from functools import lru_cache
import numpy as np
from nltk.corpus import wordnet

class QueryProcessor:
    def __init__(self, embedding_service):
        self.embedder = embedding_service

    @lru_cache(maxsize=10000)
    def _get_synonyms(self, word: str) -> List[str]:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)[:5]

    def expand_query(self, query: str) -> List[str]:
        words = query.lower().split()
        expanded = [query]

        for word in words:
            synonyms = self._get_synonyms(word)
            for syn in synonyms[:2]:
                expanded.append(query.replace(word, syn))

        return list(set(expanded))

    def process(self, query: str) -> np.ndarray:
        # Parse negative queries
        positive_query, negative_terms = self._parse_negatives(query)

        # Expand query
        expansions = self.expand_query(positive_query)

        # Embed all expansions
        embeddings = self.embedder.embed(expansions)

        # Weighted average (original query weighted 2x)
        weights = [2.0] + [1.0] * (len(embeddings) - 1)
        weights = np.array(weights) / sum(weights)
        query_vector = np.average(embeddings, axis=0, weights=weights)

        # Subtract negative term vectors
        if negative_terms:
            neg_embeddings = self.embedder.embed(negative_terms)
            neg_vector = np.mean(neg_embeddings, axis=0)
            query_vector = query_vector - 0.5 * neg_vector

        # Normalize
        query_vector = query_vector / np.linalg.norm(query_vector)

        return query_vector

    def _parse_negatives(self, query: str) -> Tuple[str, List[str]]:
        # "python tutorial -video -beginner"
        parts = query.split()
        positive = []
        negative = []

        for part in parts:
            if part.startswith('-'):
                negative.append(part[1:])
            else:
                positive.append(part)

        return ' '.join(positive), negative"""
                },
                "pitfalls": [
                    "Over-expansion dilutes original intent",
                    "Synonym quality varies by domain",
                    "Negative subtraction can flip meaning",
                    "Cache invalidation on embedding model change"
                ],
                "concepts": ["Query expansion", "Semantic understanding", "Vector arithmetic", "Query caching"],
                "estimated_hours": "5-8"
            },
            {
                "id": 3,
                "name": "Ranking & Relevance",
                "description": "Combine signals for optimal result ranking.",
                "acceptance_criteria": [
                    "Multi-stage ranking (retrieve -> rerank)",
                    "Combine semantic + lexical scores",
                    "Personalization signals",
                    "Freshness/recency boost",
                    "Click-through rate learning"
                ],
                "hints": {
                    "level1": "First stage: fast vector search. Second stage: cross-encoder reranking on top-100.",
                    "level2": "Linear combination: score = α*semantic + β*bm25 + γ*freshness + δ*popularity",
                    "level3": """from sentence_transformers import CrossEncoder
from datetime import datetime
import math

class RankingService:
    def __init__(self):
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Feature weights (learned or tuned)
        self.weights = {
            'semantic': 0.4,
            'lexical': 0.2,
            'rerank': 0.25,
            'freshness': 0.1,
            'popularity': 0.05
        }

    def rank(self, query: str, candidates: List[Dict],
             user_context: Dict = None) -> List[Dict]:

        # Stage 1: Score all candidates
        for doc in candidates:
            doc['scores'] = {}
            doc['scores']['semantic'] = doc.get('vector_score', 0)
            doc['scores']['lexical'] = doc.get('bm25_score', 0)
            doc['scores']['freshness'] = self._freshness_score(doc)
            doc['scores']['popularity'] = self._popularity_score(doc)

        # Stage 2: Rerank top candidates with cross-encoder
        top_candidates = sorted(candidates,
                                key=lambda x: x['scores']['semantic'],
                                reverse=True)[:100]

        pairs = [(query, doc['text'][:512]) for doc in top_candidates]
        rerank_scores = self.cross_encoder.predict(pairs)

        for doc, score in zip(top_candidates, rerank_scores):
            doc['scores']['rerank'] = float(score)

        # Combine scores
        for doc in top_candidates:
            doc['final_score'] = sum(
                self.weights[k] * doc['scores'].get(k, 0)
                for k in self.weights
            )

            # Personalization boost
            if user_context:
                doc['final_score'] *= self._personalization_boost(doc, user_context)

        return sorted(top_candidates, key=lambda x: x['final_score'], reverse=True)

    def _freshness_score(self, doc: Dict) -> float:
        if 'timestamp' not in doc:
            return 0.5
        age_days = (datetime.now() - doc['timestamp']).days
        # Exponential decay, half-life of 30 days
        return math.exp(-age_days / 30)

    def _popularity_score(self, doc: Dict) -> float:
        clicks = doc.get('click_count', 0)
        impressions = doc.get('impression_count', 1)
        # Smoothed CTR
        return (clicks + 1) / (impressions + 10)

    def _personalization_boost(self, doc: Dict, user: Dict) -> float:
        boost = 1.0
        # Boost docs from user's preferred categories
        if doc.get('category') in user.get('preferred_categories', []):
            boost *= 1.2
        # Boost recent user interests
        if any(tag in user.get('recent_tags', []) for tag in doc.get('tags', [])):
            boost *= 1.1
        return boost"""
                },
                "pitfalls": [
                    "Cross-encoder too slow for all results",
                    "Weight tuning is data-dependent",
                    "Popularity bias creates filter bubbles",
                    "Freshness can demote evergreen content"
                ],
                "concepts": ["Multi-stage ranking", "Learning to rank", "Feature engineering", "Personalization"],
                "estimated_hours": "6-10"
            },
            {
                "id": 4,
                "name": "Search API & UI",
                "description": "Build production search API with instant results.",
                "acceptance_criteria": [
                    "RESTful search API",
                    "Typeahead/autocomplete",
                    "Faceted search (filters)",
                    "Highlighting matched terms",
                    "Search analytics"
                ],
                "hints": {
                    "level1": "Autocomplete: prefix trie + popular queries. Update on each keystroke.",
                    "level2": "Facets: pre-compute counts per category. Update with filters.",
                    "level3": """from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict] = {}
    page: int = 1
    page_size: int = 10

class SearchResult(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    highlights: List[str]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    facets: Dict[str, List[Dict]]
    query_time_ms: float

@app.post("/search")
async def search(request: SearchRequest) -> SearchResponse:
    start = time.time()

    # Process query
    query_vector = query_processor.process(request.query)

    # Search with filters
    candidates = await vector_store.search_async(
        query_vector,
        k=request.page_size * 10,  # Over-fetch for filtering
        filters=request.filters
    )

    # Rank
    ranked = ranking_service.rank(request.query, candidates)

    # Paginate
    start_idx = (request.page - 1) * request.page_size
    page_results = ranked[start_idx:start_idx + request.page_size]

    # Build response with highlights
    results = [
        SearchResult(
            id=doc['id'],
            title=doc['title'],
            snippet=highlight_snippet(doc['text'], request.query),
            score=doc['final_score'],
            highlights=find_highlights(doc['text'], request.query)
        )
        for doc in page_results
    ]

    # Compute facets
    facets = compute_facets(ranked, request.filters)

    return SearchResponse(
        results=results,
        total=len(ranked),
        facets=facets,
        query_time_ms=(time.time() - start) * 1000
    )

@app.get("/autocomplete")
async def autocomplete(q: str = Query(min_length=2)) -> List[str]:
    # Prefix search in popular queries
    suggestions = await trie.search_prefix(q.lower(), limit=10)
    return suggestions

def highlight_snippet(text: str, query: str, context_chars: int = 150) -> str:
    query_terms = query.lower().split()
    text_lower = text.lower()

    # Find best matching window
    best_pos = 0
    best_count = 0

    for i in range(0, len(text) - context_chars, 50):
        window = text_lower[i:i + context_chars]
        count = sum(1 for term in query_terms if term in window)
        if count > best_count:
            best_count = count
            best_pos = i

    snippet = text[best_pos:best_pos + context_chars]

    # Highlight terms
    for term in query_terms:
        snippet = re.sub(
            f'({re.escape(term)})',
            r'<mark>\\1</mark>',
            snippet,
            flags=re.IGNORECASE
        )

    return snippet"""
                },
                "pitfalls": [
                    "Autocomplete latency > 100ms feels slow",
                    "Facet counts expensive to compute",
                    "Highlighting breaks on HTML entities",
                    "Not tracking null results for improvement"
                ],
                "concepts": ["Search API design", "Autocomplete", "Faceted navigation", "Result highlighting"],
                "estimated_hours": "8-12"
            }
        ]
    },

    "ai-agent-framework": {
        "id": "ai-agent-framework",
        "name": "AI Agent Framework",
        "description": "Build a framework for autonomous AI agents that can use tools, plan, and execute multi-step tasks.",
        "difficulty": "advanced",
        "estimated_hours": "50-80",
        "prerequisites": ["LLM APIs", "Python async", "System design"],
        "languages": {
            "recommended": ["Python"],
            "also_possible": ["TypeScript"]
        },
        "resources": [
            {"name": "LangChain Agents", "url": "https://python.langchain.com/docs/modules/agents/", "type": "documentation"},
            {"name": "AutoGPT", "url": "https://github.com/Significant-Gravitas/AutoGPT", "type": "reference"},
            {"name": "ReAct Paper", "url": "https://arxiv.org/abs/2210.03629", "type": "paper"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Tool System",
                "description": "Build extensible tool system for agent capabilities.",
                "acceptance_criteria": [
                    "Define tool interface (name, description, parameters, execute)",
                    "Tool discovery and registration",
                    "Parameter validation",
                    "Error handling and retries",
                    "Tool result formatting for LLM"
                ],
                "hints": {
                    "level1": "Tools are functions with JSON schema for parameters. LLM chooses which to call.",
                    "level2": "Use Pydantic for parameter validation. Return structured results.",
                    "level3": """from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, Type
import inspect

class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: str = None

class Tool(ABC):
    name: str
    description: str
    parameters_schema: Type[BaseModel]

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

    def to_openai_function(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema.model_json_schema()
        }

class WebSearchParams(BaseModel):
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results")

class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for current information"
    parameters_schema = WebSearchParams

    def __init__(self, api_key: str):
        self.api_key = api_key

    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        try:
            results = search_api.search(query, num_results)
            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

class CodeExecutionParams(BaseModel):
    code: str = Field(description="Python code to execute")
    timeout: int = Field(default=30, description="Timeout in seconds")

class CodeExecutionTool(Tool):
    name = "execute_code"
    description = "Execute Python code in a sandboxed environment"
    parameters_schema = CodeExecutionParams

    def execute(self, code: str, timeout: int = 30) -> ToolResult:
        try:
            # Run in sandbox (Docker, subprocess, etc.)
            result = sandbox.run(code, timeout=timeout)
            return ToolResult(success=True, data={
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_value': result.return_value
            })
        except TimeoutError:
            return ToolResult(success=False, error="Code execution timed out")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self.tools.get(name)

    def list_functions(self) -> List[Dict]:
        return [t.to_openai_function() for t in self.tools.values()]

    def execute(self, name: str, params: Dict) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

        # Validate parameters
        try:
            validated = tool.parameters_schema(**params)
            return tool.execute(**validated.model_dump())
        except Exception as e:
            return ToolResult(success=False, error=f"Parameter error: {e}")"""
                },
                "pitfalls": [
                    "Tool descriptions too vague for LLM to choose correctly",
                    "Missing parameter validation causes runtime errors",
                    "Tool execution without timeout hangs forever",
                    "Sensitive tools without permission checks"
                ],
                "concepts": ["Tool abstraction", "JSON Schema", "Sandboxed execution", "Registry pattern"],
                "estimated_hours": "8-12"
            },
            {
                "id": 2,
                "name": "ReAct Loop (Reasoning + Acting)",
                "description": "Implement the core agent loop: think, act, observe, repeat.",
                "acceptance_criteria": [
                    "Thought-Action-Observation cycle",
                    "Parse LLM output for actions",
                    "Handle action failures gracefully",
                    "Maximum iteration limits",
                    "Structured output parsing"
                ],
                "hints": {
                    "level1": "Prompt: 'Think step by step. Use Action: tool_name(params). Wait for Observation.'",
                    "level2": "Use function calling API for reliable action parsing. Fallback to regex.",
                    "level3": """from openai import OpenAI
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class AgentState(Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"

@dataclass
class AgentStep:
    thought: Optional[str]
    action: Optional[str]
    action_input: Optional[Dict]
    observation: Optional[str]

class ReActAgent:
    def __init__(self, tools: ToolRegistry, model: str = "gpt-4-turbo-preview"):
        self.tools = tools
        self.client = OpenAI()
        self.model = model
        self.max_iterations = 10

    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task}
        ]

        steps: List[AgentStep] = []

        for i in range(self.max_iterations):
            # Get LLM response with function calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[{"type": "function", "function": f}
                       for f in self.tools.list_functions()],
                tool_choice="auto"
            )

            message = response.choices[0].message

            # Check if finished (no tool calls)
            if not message.tool_calls:
                return message.content

            # Execute tool calls
            messages.append(message)

            for tool_call in message.tool_calls:
                step = AgentStep(
                    thought=message.content,
                    action=tool_call.function.name,
                    action_input=json.loads(tool_call.function.arguments),
                    observation=None
                )

                # Execute tool
                result = self.tools.execute(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )

                step.observation = str(result.data if result.success else result.error)
                steps.append(step)

                # Add observation to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": step.observation
                })

        return "Max iterations reached without completing task"

    def _system_prompt(self) -> str:
        tool_descriptions = "\\n".join([
            f"- {t.name}: {t.description}"
            for t in self.tools.tools.values()
        ])

        return f'''You are a helpful AI assistant that can use tools to accomplish tasks.

Available tools:
{tool_descriptions}

Think step by step about how to accomplish the task. Use tools when needed.
When you have the final answer, respond without using any tools.'''"""
                },
                "pitfalls": [
                    "Infinite loops when agent doesn't know when to stop",
                    "LLM hallucinates non-existent tools",
                    "Action parsing fails on malformed output",
                    "Not handling tool timeouts"
                ],
                "concepts": ["ReAct framework", "Function calling", "Agent loops", "Structured outputs"],
                "estimated_hours": "10-15"
            },
            {
                "id": 3,
                "name": "Planning & Task Decomposition",
                "description": "Add planning layer for complex multi-step tasks.",
                "acceptance_criteria": [
                    "Decompose complex tasks into subtasks",
                    "Create execution plan",
                    "Handle dependencies between steps",
                    "Replan on failure",
                    "Parallel execution where possible"
                ],
                "hints": {
                    "level1": "First ask LLM to create plan, then execute each step. Simpler than full planning.",
                    "level2": "DAG of tasks with dependencies. Topological sort for execution order.",
                    "level3": """from dataclasses import dataclass, field
from typing import List, Set
import asyncio
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PlanTask:
    id: str
    description: str
    dependencies: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = None

class TaskPlanner:
    def __init__(self, llm_client):
        self.llm = llm_client

    def create_plan(self, objective: str) -> List[PlanTask]:
        prompt = f'''Create a plan to accomplish this objective:
{objective}

Return a JSON list of tasks with:
- id: unique task identifier
- description: what to do
- dependencies: list of task ids this depends on

Example:
[
  {{"id": "1", "description": "Search for information about X", "dependencies": []}},
  {{"id": "2", "description": "Analyze the search results", "dependencies": ["1"]}},
  {{"id": "3", "description": "Write summary based on analysis", "dependencies": ["2"]}}
]'''

        response = self.llm.generate(prompt)
        tasks_data = json.loads(response)

        return [
            PlanTask(
                id=t['id'],
                description=t['description'],
                dependencies=set(t.get('dependencies', []))
            )
            for t in tasks_data
        ]

class PlanExecutor:
    def __init__(self, agent: ReActAgent):
        self.agent = agent

    async def execute_plan(self, tasks: List[PlanTask]) -> Dict[str, Any]:
        task_map = {t.id: t for t in tasks}
        results = {}

        while not all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in tasks):
            # Find ready tasks (dependencies met)
            ready = [
                t for t in tasks
                if t.status == TaskStatus.PENDING and
                all(task_map[dep].status == TaskStatus.COMPLETED
                    for dep in t.dependencies)
            ]

            if not ready:
                # Check for deadlock
                pending = [t for t in tasks if t.status == TaskStatus.PENDING]
                if pending:
                    raise RuntimeError("Deadlock detected in task dependencies")
                break

            # Execute ready tasks in parallel
            async def run_task(task: PlanTask):
                task.status = TaskStatus.RUNNING
                try:
                    # Include context from dependencies
                    context = {dep: results[dep] for dep in task.dependencies}
                    prompt = f"Previous results: {context}\\n\\nTask: {task.description}"

                    result = await asyncio.to_thread(self.agent.run, prompt)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    results[task.id] = result
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)

            await asyncio.gather(*[run_task(t) for t in ready])

        return results

    async def replan_on_failure(self, tasks: List[PlanTask], objective: str) -> List[PlanTask]:
        failed = [t for t in tasks if t.status == TaskStatus.FAILED]
        completed = [t for t in tasks if t.status == TaskStatus.COMPLETED]

        prompt = f'''The original plan failed. Here's what happened:

Completed tasks: {[(t.id, t.description, t.result) for t in completed]}
Failed tasks: {[(t.id, t.description, t.error) for t in failed]}

Create a new plan to accomplish: {objective}
Consider what already succeeded and avoid the failures.'''

        return self.planner.create_plan(prompt)"""
                },
                "pitfalls": [
                    "Circular dependencies in task graph",
                    "Over-decomposition creates too many steps",
                    "Context lost between subtasks",
                    "Replanning loops infinitely"
                ],
                "concepts": ["Task decomposition", "DAG execution", "Async parallel execution", "Replanning"],
                "estimated_hours": "12-18"
            },
            {
                "id": 4,
                "name": "Memory & Context Management",
                "description": "Give agents short-term and long-term memory.",
                "acceptance_criteria": [
                    "Conversation history (short-term)",
                    "Working memory for current task",
                    "Long-term memory (vector store)",
                    "Memory retrieval for context",
                    "Memory summarization and compression"
                ],
                "hints": {
                    "level1": "Short-term: sliding window of last N messages. Long-term: embed and store in vector DB.",
                    "level2": "Summarize old messages to save context. Retrieve relevant memories for current task.",
                    "level3": """from datetime import datetime
from typing import List, Optional
import numpy as np

@dataclass
class Memory:
    content: str
    type: str  # 'conversation', 'task_result', 'fact', 'reflection'
    timestamp: datetime
    importance: float = 0.5
    embedding: Optional[np.ndarray] = None

class AgentMemory:
    def __init__(self, embedding_service, vector_store, llm):
        self.embedder = embedding_service
        self.store = vector_store
        self.llm = llm
        self.working_memory: List[Memory] = []
        self.max_working_memory = 10

    def add(self, content: str, memory_type: str = 'conversation', importance: float = 0.5):
        embedding = self.embedder.embed([content])[0]

        memory = Memory(
            content=content,
            type=memory_type,
            timestamp=datetime.now(),
            importance=importance,
            embedding=embedding
        )

        # Add to working memory
        self.working_memory.append(memory)
        if len(self.working_memory) > self.max_working_memory:
            # Move oldest to long-term
            old = self.working_memory.pop(0)
            self._store_long_term(old)

        # High importance goes straight to long-term
        if importance > 0.8:
            self._store_long_term(memory)

    def _store_long_term(self, memory: Memory):
        self.store.add(
            ids=[f"mem_{memory.timestamp.isoformat()}"],
            embeddings=[memory.embedding],
            metadatas=[{
                'content': memory.content,
                'type': memory.type,
                'timestamp': memory.timestamp.isoformat(),
                'importance': memory.importance
            }]
        )

    def retrieve(self, query: str, k: int = 5) -> List[Memory]:
        query_embedding = self.embedder.embed([query])[0]
        results = self.store.search(query_embedding, query, k=k)

        return [
            Memory(
                content=r['metadata']['content'],
                type=r['metadata']['type'],
                timestamp=datetime.fromisoformat(r['metadata']['timestamp']),
                importance=r['metadata']['importance']
            )
            for r in results
        ]

    def get_context(self, current_task: str, max_tokens: int = 2000) -> str:
        # Recent working memory
        recent = [m.content for m in self.working_memory[-5:]]

        # Relevant long-term memories
        relevant = self.retrieve(current_task, k=10)
        relevant_content = [m.content for m in relevant]

        context = f'''RECENT CONTEXT:
{chr(10).join(recent)}

RELEVANT MEMORIES:
{chr(10).join(relevant_content)}'''

        # Summarize if too long
        if len(context) > max_tokens * 4:  # Rough char estimate
            context = self._summarize(context, max_tokens)

        return context

    def _summarize(self, content: str, max_tokens: int) -> str:
        return self.llm.generate(
            f"Summarize the following context in about {max_tokens} tokens, "
            f"preserving the most important information:\\n\\n{content}"
        )

    def reflect(self):
        '''Generate insights from recent memories'''
        recent = [m.content for m in self.working_memory]

        reflection = self.llm.generate(
            f"Based on these recent events, what are key insights or patterns?\\n\\n"
            f"{chr(10).join(recent)}"
        )

        self.add(reflection, memory_type='reflection', importance=0.9)
        return reflection"""
                },
                "pitfalls": [
                    "Memory grows unbounded without cleanup",
                    "Retrieved memories not relevant to task",
                    "Summarization loses critical details",
                    "Importance scoring is subjective"
                ],
                "concepts": ["Working vs long-term memory", "Memory retrieval", "Context compression", "Reflection"],
                "estimated_hours": "10-15"
            },
            {
                "id": 5,
                "name": "Multi-Agent Collaboration",
                "description": "Build systems where multiple agents work together.",
                "acceptance_criteria": [
                    "Define agent roles and capabilities",
                    "Message passing between agents",
                    "Coordinator/orchestrator agent",
                    "Shared context and state",
                    "Conflict resolution"
                ],
                "hints": {
                    "level1": "Start with simple delegation: main agent calls specialist agents as tools.",
                    "level2": "Pub/sub messaging: agents subscribe to topics, broadcast results.",
                    "level3": """from abc import ABC
from typing import Dict, List, Any
import asyncio
from dataclasses import dataclass

@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str  # or 'broadcast'
    content: str
    metadata: Dict = None

class SpecialistAgent:
    def __init__(self, name: str, role: str, tools: ToolRegistry):
        self.name = name
        self.role = role
        self.tools = tools
        self.inbox: asyncio.Queue = asyncio.Queue()

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        # Add role context
        prompt = f"You are a {self.role} specialist.\\n\\nTask: {message.content}"

        result = self.react_agent.run(prompt)

        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            content=result
        )

class OrchestratorAgent:
    def __init__(self, specialists: Dict[str, SpecialistAgent]):
        self.specialists = specialists
        self.message_bus: asyncio.Queue = asyncio.Queue()

    async def run(self, task: str) -> str:
        # Plan which specialists to involve
        plan = self._create_delegation_plan(task)

        results = {}

        for step in plan:
            if step['type'] == 'delegate':
                specialist = self.specialists[step['agent']]

                message = AgentMessage(
                    from_agent='orchestrator',
                    to_agent=step['agent'],
                    content=step['task']
                )

                response = await specialist.process_message(message)
                results[step['agent']] = response.content

            elif step['type'] == 'parallel_delegate':
                tasks = [
                    self.specialists[agent].process_message(
                        AgentMessage('orchestrator', agent, subtask)
                    )
                    for agent, subtask in step['assignments'].items()
                ]
                responses = await asyncio.gather(*tasks)
                for resp in responses:
                    results[resp.from_agent] = resp.content

            elif step['type'] == 'synthesize':
                # Combine results from specialists
                synthesis_prompt = f'''Combine these specialist results:
{json.dumps(results, indent=2)}

Original task: {task}'''

                return self.llm.generate(synthesis_prompt)

        return str(results)

    def _create_delegation_plan(self, task: str) -> List[Dict]:
        specialist_info = {
            name: agent.role
            for name, agent in self.specialists.items()
        }

        prompt = f'''You are an orchestrator with these specialists:
{json.dumps(specialist_info, indent=2)}

Create a delegation plan for: {task}

Return JSON list of steps:
{{"type": "delegate", "agent": "agent_name", "task": "specific task"}}
{{"type": "parallel_delegate", "assignments": {{"agent1": "task1", "agent2": "task2"}}}}
{{"type": "synthesize"}}'''

        return json.loads(self.llm.generate(prompt))

# Example usage
specialists = {
    'researcher': SpecialistAgent('researcher', 'research and information gathering', research_tools),
    'coder': SpecialistAgent('coder', 'writing and analyzing code', coding_tools),
    'writer': SpecialistAgent('writer', 'writing and editing text', writing_tools)
}

orchestrator = OrchestratorAgent(specialists)
result = await orchestrator.run("Research best practices for API design and create a style guide")"""
                },
                "pitfalls": [
                    "Agents talk past each other without shared context",
                    "Delegation loops (A delegates to B delegates to A)",
                    "Orchestrator becomes bottleneck",
                    "No timeout for unresponsive agents"
                ],
                "concepts": ["Multi-agent systems", "Message passing", "Orchestration patterns", "Role specialization"],
                "estimated_hours": "15-20"
            }
        ]
    },

    "llm-eval-framework": {
        "id": "llm-eval-framework",
        "name": "LLM Evaluation Framework",
        "description": "Build comprehensive evaluation system for LLM applications. Critical for production AI.",
        "difficulty": "advanced",
        "estimated_hours": "40-60",
        "prerequisites": ["LLM APIs", "Statistics", "Python"],
        "languages": {
            "recommended": ["Python"],
            "also_possible": ["TypeScript"]
        },
        "resources": [
            {"name": "Anthropic Eval Best Practices", "url": "https://docs.anthropic.com/en/docs/build-with-claude/develop-tests", "type": "documentation"},
            {"name": "LangSmith", "url": "https://docs.smith.langchain.com/", "type": "documentation"},
            {"name": "Braintrust", "url": "https://www.braintrustdata.com/docs", "type": "documentation"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Dataset Management",
                "description": "Create and version evaluation datasets.",
                "acceptance_criteria": [
                    "Define test case schema",
                    "Version datasets with git-like tracking",
                    "Import from CSV/JSON",
                    "Split into train/test/validation",
                    "Handle golden examples"
                ],
                "hints": {
                    "level1": "Test case = input + expected output + metadata. Store as JSON lines.",
                    "level2": "Hash dataset content for versioning. Track lineage of derived datasets.",
                    "level3": """from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import hashlib
import json
from pathlib import Path
from datetime import datetime

@dataclass
class TestCase:
    id: str
    input: str
    expected_output: Optional[str] = None
    context: Optional[Dict] = None
    tags: List[str] = None
    metadata: Dict = None

    def to_dict(self):
        return asdict(self)

@dataclass
class Dataset:
    name: str
    version: str
    cases: List[TestCase]
    created_at: datetime
    parent_version: Optional[str] = None

    @property
    def hash(self) -> str:
        content = json.dumps([c.to_dict() for c in self.cases], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

class DatasetManager:
    def __init__(self, storage_path: str = ".evals/datasets"):
        self.storage = Path(storage_path)
        self.storage.mkdir(parents=True, exist_ok=True)

    def create(self, name: str, cases: List[TestCase]) -> Dataset:
        dataset = Dataset(
            name=name,
            version=f"v1_{datetime.now().strftime('%Y%m%d')}",
            cases=cases,
            created_at=datetime.now()
        )
        self._save(dataset)
        return dataset

    def add_cases(self, dataset: Dataset, new_cases: List[TestCase]) -> Dataset:
        new_dataset = Dataset(
            name=dataset.name,
            version=f"v{len(self._get_versions(dataset.name)) + 1}_{datetime.now().strftime('%Y%m%d')}",
            cases=dataset.cases + new_cases,
            created_at=datetime.now(),
            parent_version=dataset.version
        )
        self._save(new_dataset)
        return new_dataset

    def _save(self, dataset: Dataset):
        path = self.storage / dataset.name / f"{dataset.version}.jsonl"
        path.parent.mkdir(exist_ok=True)

        with open(path, 'w') as f:
            for case in dataset.cases:
                f.write(json.dumps(case.to_dict()) + '\\n')

        # Save metadata
        meta_path = self.storage / dataset.name / f"{dataset.version}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'name': dataset.name,
                'version': dataset.version,
                'hash': dataset.hash,
                'case_count': len(dataset.cases),
                'created_at': dataset.created_at.isoformat(),
                'parent_version': dataset.parent_version
            }, f, indent=2)

    def load(self, name: str, version: str = None) -> Dataset:
        if version is None:
            version = self._get_latest_version(name)

        path = self.storage / name / f"{version}.jsonl"
        cases = []
        with open(path) as f:
            for line in f:
                cases.append(TestCase(**json.loads(line)))

        meta_path = self.storage / name / f"{version}.meta.json"
        meta = json.loads(meta_path.read_text())

        return Dataset(
            name=name,
            version=version,
            cases=cases,
            created_at=datetime.fromisoformat(meta['created_at']),
            parent_version=meta.get('parent_version')
        )

    def split(self, dataset: Dataset, train_ratio=0.8, seed=42) -> tuple:
        import random
        random.seed(seed)
        cases = dataset.cases.copy()
        random.shuffle(cases)

        split_idx = int(len(cases) * train_ratio)
        train_cases = cases[:split_idx]
        test_cases = cases[split_idx:]

        return (
            Dataset(f"{dataset.name}_train", dataset.version, train_cases, datetime.now()),
            Dataset(f"{dataset.name}_test", dataset.version, test_cases, datetime.now())
        )"""
                },
                "pitfalls": [
                    "Test set leaking into training data",
                    "Version conflicts with concurrent edits",
                    "Large datasets slow to load",
                    "Not tracking dataset provenance"
                ],
                "concepts": ["Dataset versioning", "Train/test splits", "Golden examples", "Data lineage"],
                "estimated_hours": "6-10"
            },
            {
                "id": 2,
                "name": "Evaluation Metrics",
                "description": "Implement metrics for different evaluation types.",
                "acceptance_criteria": [
                    "Exact match and fuzzy match",
                    "Semantic similarity",
                    "LLM-as-judge grading",
                    "Custom metric functions",
                    "Aggregate scoring"
                ],
                "hints": {
                    "level1": "Start with exact match. Add fuzzy match (Levenshtein). Then semantic similarity.",
                    "level2": "LLM-as-judge: ask GPT-4 to rate on criteria. Parse structured score.",
                    "level3": """from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
from difflib import SequenceMatcher

class Metric(ABC):
    name: str

    @abstractmethod
    def score(self, expected: Any, actual: Any, context: Dict = None) -> float:
        '''Return score between 0 and 1'''
        pass

class ExactMatch(Metric):
    name = "exact_match"

    def __init__(self, case_sensitive: bool = False, strip: bool = True):
        self.case_sensitive = case_sensitive
        self.strip = strip

    def score(self, expected: str, actual: str, context: Dict = None) -> float:
        if self.strip:
            expected, actual = expected.strip(), actual.strip()
        if not self.case_sensitive:
            expected, actual = expected.lower(), actual.lower()
        return 1.0 if expected == actual else 0.0

class FuzzyMatch(Metric):
    name = "fuzzy_match"

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def score(self, expected: str, actual: str, context: Dict = None) -> float:
        ratio = SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
        return ratio

class SemanticSimilarity(Metric):
    name = "semantic_similarity"

    def __init__(self, embedding_service):
        self.embedder = embedding_service

    def score(self, expected: str, actual: str, context: Dict = None) -> float:
        embeddings = self.embedder.embed([expected, actual])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)

class LLMJudge(Metric):
    name = "llm_judge"

    def __init__(self, llm_client, criteria: str, rubric: Dict[int, str] = None):
        self.llm = llm_client
        self.criteria = criteria
        self.rubric = rubric or {
            5: "Excellent - fully meets criteria",
            4: "Good - mostly meets criteria with minor issues",
            3: "Acceptable - meets basic criteria",
            2: "Poor - partially meets criteria",
            1: "Unacceptable - does not meet criteria"
        }

    def score(self, expected: str, actual: str, context: Dict = None) -> float:
        rubric_text = "\\n".join([f"{k}: {v}" for k, v in self.rubric.items()])

        prompt = f'''Evaluate the following response based on this criteria:
{self.criteria}

Expected answer: {expected}
Actual response: {actual}

Scoring rubric:
{rubric_text}

Provide your score (1-5) and brief justification.
Format: SCORE: [number]
REASON: [explanation]'''

        response = self.llm.generate(prompt)

        # Parse score
        import re
        match = re.search(r'SCORE:\\s*(\\d)', response)
        if match:
            score = int(match.group(1))
            return score / 5.0
        return 0.5  # Default if parsing fails

class MultiCriteriaJudge(Metric):
    name = "multi_criteria"

    def __init__(self, llm_client, criteria: List[Dict]):
        self.llm = llm_client
        self.criteria = criteria  # [{'name': 'accuracy', 'weight': 0.5, 'description': '...'}]

    def score(self, expected: str, actual: str, context: Dict = None) -> float:
        scores = {}

        for criterion in self.criteria:
            judge = LLMJudge(self.llm, criterion['description'])
            scores[criterion['name']] = judge.score(expected, actual, context)

        # Weighted average
        total_weight = sum(c['weight'] for c in self.criteria)
        weighted_sum = sum(
            scores[c['name']] * c['weight']
            for c in self.criteria
        )

        return weighted_sum / total_weight"""
                },
                "pitfalls": [
                    "Semantic similarity not calibrated (what's 'good' score?)",
                    "LLM judge inconsistent across runs",
                    "Custom metrics not between 0-1",
                    "Not handling edge cases (empty strings, None)"
                ],
                "concepts": ["Evaluation metrics", "Fuzzy matching", "Semantic similarity", "LLM-as-judge"],
                "estimated_hours": "8-12"
            },
            {
                "id": 3,
                "name": "Evaluation Runner",
                "description": "Run evaluations efficiently with caching and parallelism.",
                "acceptance_criteria": [
                    "Batch evaluation runs",
                    "Parallel LLM calls",
                    "Result caching to avoid re-running",
                    "Progress tracking",
                    "Resume from failures"
                ],
                "hints": {
                    "level1": "asyncio.gather for parallel LLM calls. Cache results by (input_hash, model, prompt_version).",
                    "level2": "Checkpoint results periodically. Resume by loading checkpoint and skipping completed.",
                    "level3": """import asyncio
from dataclasses import dataclass, field
from typing import Callable, List, Dict
import hashlib
import json
from tqdm.asyncio import tqdm

@dataclass
class EvalResult:
    case_id: str
    input: str
    expected: str
    actual: str
    scores: Dict[str, float]
    latency_ms: float
    error: str = None

@dataclass
class EvalRun:
    id: str
    dataset_name: str
    model: str
    timestamp: datetime
    results: List[EvalResult] = field(default_factory=list)
    aggregate_scores: Dict[str, float] = None

class EvalRunner:
    def __init__(self,
                 model_fn: Callable[[str], str],
                 metrics: List[Metric],
                 cache_dir: str = ".evals/cache",
                 max_parallel: int = 10):
        self.model_fn = model_fn
        self.metrics = metrics
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_parallel)

    def _cache_key(self, input_text: str, model_id: str) -> str:
        content = f"{model_id}:{input_text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[str]:
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())['output']
        return None

    def _set_cached(self, cache_key: str, output: str):
        cache_path = self.cache_dir / f"{cache_key}.json"
        cache_path.write_text(json.dumps({'output': output}))

    async def _evaluate_case(self, case: TestCase, model_id: str) -> EvalResult:
        async with self.semaphore:
            cache_key = self._cache_key(case.input, model_id)
            cached = self._get_cached(cache_key)

            start = time.time()

            if cached:
                actual = cached
            else:
                try:
                    actual = await asyncio.to_thread(self.model_fn, case.input)
                    self._set_cached(cache_key, actual)
                except Exception as e:
                    return EvalResult(
                        case_id=case.id,
                        input=case.input,
                        expected=case.expected_output,
                        actual=None,
                        scores={},
                        latency_ms=(time.time() - start) * 1000,
                        error=str(e)
                    )

            latency = (time.time() - start) * 1000

            # Calculate scores
            scores = {}
            for metric in self.metrics:
                try:
                    scores[metric.name] = metric.score(
                        case.expected_output, actual, case.context
                    )
                except Exception as e:
                    scores[metric.name] = None

            return EvalResult(
                case_id=case.id,
                input=case.input,
                expected=case.expected_output,
                actual=actual,
                scores=scores,
                latency_ms=latency
            )

    async def run(self, dataset: Dataset, model_id: str,
                  checkpoint_every: int = 50) -> EvalRun:
        run = EvalRun(
            id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataset_name=dataset.name,
            model=model_id,
            timestamp=datetime.now()
        )

        # Check for existing checkpoint
        checkpoint_path = self.cache_dir / f"{run.id}_checkpoint.json"
        completed_ids = set()
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            run.results = [EvalResult(**r) for r in checkpoint['results']]
            completed_ids = {r.case_id for r in run.results}

        # Filter to incomplete cases
        pending = [c for c in dataset.cases if c.id not in completed_ids]

        # Run with progress bar
        tasks = [self._evaluate_case(c, model_id) for c in pending]

        for i, result in enumerate(await tqdm.gather(*tasks)):
            run.results.append(result)

            # Checkpoint periodically
            if (i + 1) % checkpoint_every == 0:
                self._save_checkpoint(run, checkpoint_path)

        # Calculate aggregates
        run.aggregate_scores = self._aggregate(run.results)

        return run

    def _aggregate(self, results: List[EvalResult]) -> Dict[str, float]:
        aggregates = {}

        for metric in self.metrics:
            scores = [r.scores.get(metric.name) for r in results
                     if r.scores.get(metric.name) is not None]
            if scores:
                aggregates[f"{metric.name}_mean"] = np.mean(scores)
                aggregates[f"{metric.name}_std"] = np.std(scores)
                aggregates[f"{metric.name}_p50"] = np.percentile(scores, 50)
                aggregates[f"{metric.name}_p95"] = np.percentile(scores, 95)

        return aggregates"""
                },
                "pitfalls": [
                    "Rate limiting without backoff crashes run",
                    "Cache invalidation when prompt changes",
                    "Memory issues with large result sets",
                    "Lost progress on crash without checkpoints"
                ],
                "concepts": ["Async evaluation", "Result caching", "Checkpointing", "Progress tracking"],
                "estimated_hours": "10-15"
            },
            {
                "id": 4,
                "name": "Reporting & Analysis",
                "description": "Generate insights from evaluation results.",
                "acceptance_criteria": [
                    "Score breakdown by tags/categories",
                    "Regression detection vs baseline",
                    "Failure analysis (common patterns)",
                    "Export to HTML/PDF reports",
                    "CI/CD integration (pass/fail)"
                ],
                "hints": {
                    "level1": "Group results by tags. Calculate per-group metrics. Flag if below threshold.",
                    "level2": "Compare runs: significant difference = > 2 std devs. Use bootstrap for confidence.",
                    "level3": """import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from jinja2 import Template

class EvalAnalyzer:
    def __init__(self):
        self.significance_threshold = 0.05

    def to_dataframe(self, run: EvalRun) -> pd.DataFrame:
        rows = []
        for result in run.results:
            row = {
                'case_id': result.case_id,
                'latency_ms': result.latency_ms,
                'error': result.error is not None,
                **result.scores
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def breakdown_by_tag(self, run: EvalRun, dataset: Dataset) -> Dict:
        df = self.to_dataframe(run)

        # Add tags from dataset
        case_tags = {c.id: c.tags or [] for c in dataset.cases}

        breakdowns = {}
        all_tags = set()
        for tags in case_tags.values():
            all_tags.update(tags)

        for tag in all_tags:
            tag_cases = [cid for cid, tags in case_tags.items() if tag in tags]
            tag_df = df[df['case_id'].isin(tag_cases)]

            breakdowns[tag] = {
                'count': len(tag_df),
                'metrics': {
                    col: tag_df[col].mean()
                    for col in df.columns
                    if col not in ('case_id', 'latency_ms', 'error')
                }
            }

        return breakdowns

    def compare_runs(self, baseline: EvalRun, current: EvalRun) -> Dict:
        baseline_df = self.to_dataframe(baseline)
        current_df = self.to_dataframe(current)

        comparisons = {}
        metric_cols = [c for c in baseline_df.columns
                      if c not in ('case_id', 'latency_ms', 'error')]

        for metric in metric_cols:
            baseline_scores = baseline_df[metric].dropna()
            current_scores = current_df[metric].dropna()

            # Statistical test
            t_stat, p_value = stats.ttest_ind(baseline_scores, current_scores)

            delta = current_scores.mean() - baseline_scores.mean()
            delta_pct = (delta / baseline_scores.mean()) * 100 if baseline_scores.mean() != 0 else 0

            comparisons[metric] = {
                'baseline_mean': baseline_scores.mean(),
                'current_mean': current_scores.mean(),
                'delta': delta,
                'delta_pct': delta_pct,
                'p_value': p_value,
                'significant': p_value < self.significance_threshold,
                'regression': delta < 0 and p_value < self.significance_threshold
            }

        return comparisons

    def failure_analysis(self, run: EvalRun, threshold: float = 0.5) -> Dict:
        failures = [r for r in run.results
                   if any(s < threshold for s in r.scores.values() if s is not None)]

        # Cluster by input patterns (simple: first 50 chars)
        patterns = {}
        for f in failures:
            pattern = f.input[:50]
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(f)

        # Sort by frequency
        sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)

        return {
            'total_failures': len(failures),
            'failure_rate': len(failures) / len(run.results),
            'top_patterns': sorted_patterns[:10]
        }

    def generate_report(self, run: EvalRun, baseline: EvalRun = None) -> str:
        template = Template('''
<!DOCTYPE html>
<html>
<head>
    <title>Eval Report: {{ run.id }}</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        .metric { padding: 10px; margin: 10px; border: 1px solid #ccc; }
        .regression { background: #ffcccc; }
        .improvement { background: #ccffcc; }
    </style>
</head>
<body>
    <h1>Evaluation Report</h1>
    <p>Run ID: {{ run.id }}</p>
    <p>Dataset: {{ run.dataset_name }}</p>
    <p>Model: {{ run.model }}</p>

    <h2>Aggregate Scores</h2>
    {% for metric, value in run.aggregate_scores.items() %}
    <div class="metric">{{ metric }}: {{ "%.3f"|format(value) }}</div>
    {% endfor %}

    {% if comparison %}
    <h2>Comparison to Baseline</h2>
    {% for metric, comp in comparison.items() %}
    <div class="metric {{ 'regression' if comp.regression else 'improvement' if comp.delta > 0 else '' }}">
        {{ metric }}: {{ "%.3f"|format(comp.current_mean) }}
        ({{ "%+.1f"|format(comp.delta_pct) }}% vs baseline)
        {% if comp.significant %}*{% endif %}
    </div>
    {% endfor %}
    {% endif %}
</body>
</html>
        ''')

        comparison = self.compare_runs(baseline, run) if baseline else None

        return template.render(run=run, comparison=comparison)"""
                },
                "pitfalls": [
                    "Small sample sizes give unreliable comparisons",
                    "Multiple comparisons without correction",
                    "Ignoring variance in aggregate scores",
                    "Report generation fails silently"
                ],
                "concepts": ["Statistical testing", "Regression detection", "Failure analysis", "Reporting"],
                "estimated_hours": "8-12"
            }
        ]
    },

    "recommendation-engine": {
        "id": "recommendation-engine",
        "name": "Recommendation Engine",
        "description": "Build a production recommendation system with collaborative filtering, content-based, and hybrid approaches.",
        "difficulty": "intermediate",
        "estimated_hours": "35-50",
        "prerequisites": ["Python", "Linear algebra basics", "Database fundamentals"],
        "languages": {
            "recommended": ["Python"],
            "also_possible": ["Go", "Scala"]
        },
        "resources": [
            {"name": "Surprise Library", "url": "https://surpriselib.com/", "type": "documentation"},
            {"name": "Netflix Recommendation Paper", "url": "https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/The-BellKor-Solution.pdf", "type": "paper"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Collaborative Filtering",
                "description": "Implement user-based and item-based collaborative filtering.",
                "acceptance_criteria": [
                    "User-item rating matrix",
                    "User similarity calculation",
                    "Item similarity calculation",
                    "K-nearest neighbors prediction",
                    "Handle cold start problem"
                ],
                "hints": {
                    "level1": "Cosine similarity between user rating vectors. Predict = weighted average of neighbors.",
                    "level2": "Item-based is more stable than user-based (items change less than users).",
                    "level3": """import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilter:
    def __init__(self, k_neighbors: int = 20):
        self.k = k_neighbors
        self.user_matrix = None
        self.item_matrix = None
        self.user_similarity = None
        self.item_similarity = None

    def fit(self, ratings: List[tuple]):  # [(user_id, item_id, rating), ...]
        # Build sparse matrix
        users = list(set(r[0] for r in ratings))
        items = list(set(r[1] for r in ratings))

        self.user_idx = {u: i for i, u in enumerate(users)}
        self.item_idx = {it: i for i, it in enumerate(items)}
        self.idx_user = {i: u for u, i in self.user_idx.items()}
        self.idx_item = {i: it for it, i in self.item_idx.items()}

        # User-item matrix
        rows, cols, data = [], [], []
        for user, item, rating in ratings:
            rows.append(self.user_idx[user])
            cols.append(self.item_idx[item])
            data.append(rating)

        self.user_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(users), len(items))
        )
        self.item_matrix = self.user_matrix.T

        # Precompute similarities
        self.user_similarity = cosine_similarity(self.user_matrix)
        self.item_similarity = cosine_similarity(self.item_matrix)

    def predict_user_based(self, user_id, item_id) -> float:
        if user_id not in self.user_idx or item_id not in self.item_idx:
            return self._global_mean()

        u_idx = self.user_idx[user_id]
        i_idx = self.item_idx[item_id]

        # Find users who rated this item
        item_raters = self.user_matrix[:, i_idx].nonzero()[0]

        if len(item_raters) == 0:
            return self._user_mean(u_idx)

        # Get similarities and ratings
        similarities = self.user_similarity[u_idx, item_raters]
        ratings = np.array(self.user_matrix[item_raters, i_idx].todense()).flatten()

        # Top-k neighbors
        top_k = np.argsort(similarities)[-self.k:]

        sim_sum = np.sum(np.abs(similarities[top_k]))
        if sim_sum == 0:
            return self._user_mean(u_idx)

        weighted_sum = np.sum(similarities[top_k] * ratings[top_k])
        return weighted_sum / sim_sum

    def predict_item_based(self, user_id, item_id) -> float:
        if user_id not in self.user_idx or item_id not in self.item_idx:
            return self._global_mean()

        u_idx = self.user_idx[user_id]
        i_idx = self.item_idx[item_id]

        # Find items rated by this user
        user_items = self.user_matrix[u_idx, :].nonzero()[1]

        if len(user_items) == 0:
            return self._item_mean(i_idx)

        # Get similarities and ratings
        similarities = self.item_similarity[i_idx, user_items]
        ratings = np.array(self.user_matrix[u_idx, user_items].todense()).flatten()

        # Top-k similar items
        top_k = np.argsort(similarities)[-self.k:]

        sim_sum = np.sum(np.abs(similarities[top_k]))
        if sim_sum == 0:
            return self._item_mean(i_idx)

        weighted_sum = np.sum(similarities[top_k] * ratings[top_k])
        return weighted_sum / sim_sum

    def recommend(self, user_id, n: int = 10) -> List[tuple]:
        if user_id not in self.user_idx:
            return self._popular_items(n)

        u_idx = self.user_idx[user_id]
        rated_items = set(self.user_matrix[u_idx, :].nonzero()[1])

        predictions = []
        for i_idx in range(self.user_matrix.shape[1]):
            if i_idx not in rated_items:
                item_id = self.idx_item[i_idx]
                pred = self.predict_item_based(user_id, item_id)
                predictions.append((item_id, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]"""
                },
                "pitfalls": [
                    "Sparse matrix operations are memory-intensive",
                    "Precomputing all similarities doesn't scale",
                    "Cold start: new users/items have no data",
                    "Popularity bias in recommendations"
                ],
                "concepts": ["Collaborative filtering", "Similarity metrics", "K-NN", "Sparse matrices"],
                "estimated_hours": "8-12"
            },
            {
                "id": 2,
                "name": "Matrix Factorization",
                "description": "Implement SVD and ALS for latent factor models.",
                "acceptance_criteria": [
                    "SVD decomposition of rating matrix",
                    "ALS (Alternating Least Squares)",
                    "Regularization to prevent overfitting",
                    "Implicit feedback handling",
                    "Hyperparameter tuning"
                ],
                "hints": {
                    "level1": "Rating ≈ User_vector · Item_vector. Learn vectors via gradient descent.",
                    "level2": "ALS: fix users, optimize items. Then fix items, optimize users. Repeat.",
                    "level3": """import numpy as np

class MatrixFactorization:
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01,
                 regularization: float = 0.02, n_epochs: int = 20):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs

    def fit(self, ratings: List[tuple]):
        users = list(set(r[0] for r in ratings))
        items = list(set(r[1] for r in ratings))

        self.user_idx = {u: i for i, u in enumerate(users)}
        self.item_idx = {it: i for i, it in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        # Initialize factors randomly
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Biases
        self.global_mean = np.mean([r[2] for r in ratings])
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # SGD training
        for epoch in range(self.n_epochs):
            np.random.shuffle(ratings)
            total_error = 0

            for user, item, rating in ratings:
                u_idx = self.user_idx[user]
                i_idx = self.item_idx[item]

                # Prediction
                pred = (self.global_mean +
                       self.user_bias[u_idx] +
                       self.item_bias[i_idx] +
                       np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))

                error = rating - pred
                total_error += error ** 2

                # Update biases
                self.user_bias[u_idx] += self.lr * (error - self.reg * self.user_bias[u_idx])
                self.item_bias[i_idx] += self.lr * (error - self.reg * self.item_bias[i_idx])

                # Update factors
                user_factor = self.user_factors[u_idx].copy()
                self.user_factors[u_idx] += self.lr * (
                    error * self.item_factors[i_idx] - self.reg * user_factor
                )
                self.item_factors[i_idx] += self.lr * (
                    error * user_factor - self.reg * self.item_factors[i_idx]
                )

            rmse = np.sqrt(total_error / len(ratings))
            print(f"Epoch {epoch + 1}: RMSE = {rmse:.4f}")

    def predict(self, user_id, item_id) -> float:
        if user_id not in self.user_idx or item_id not in self.item_idx:
            return self.global_mean

        u_idx = self.user_idx[user_id]
        i_idx = self.item_idx[item_id]

        return (self.global_mean +
               self.user_bias[u_idx] +
               self.item_bias[i_idx] +
               np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))

    def recommend(self, user_id, n: int = 10, exclude_rated: set = None) -> List[tuple]:
        if user_id not in self.user_idx:
            return []

        u_idx = self.user_idx[user_id]
        exclude_rated = exclude_rated or set()

        # Score all items
        scores = (self.global_mean +
                 self.user_bias[u_idx] +
                 self.item_bias +
                 np.dot(self.item_factors, self.user_factors[u_idx]))

        # Filter and sort
        items_scores = [
            (self.idx_item[i], scores[i])
            for i in range(len(scores))
            if self.idx_item.get(i) not in exclude_rated
        ]
        items_scores.sort(key=lambda x: x[1], reverse=True)

        return items_scores[:n]"""
                },
                "pitfalls": [
                    "Learning rate too high causes divergence",
                    "Overfitting without regularization",
                    "Implicit data needs different loss function",
                    "Factor initialization affects convergence"
                ],
                "concepts": ["Matrix factorization", "SVD", "Stochastic gradient descent", "Regularization"],
                "estimated_hours": "8-12"
            },
            {
                "id": 3,
                "name": "Content-Based Filtering",
                "description": "Recommend based on item features and user preferences.",
                "acceptance_criteria": [
                    "Extract item features (text, categories, tags)",
                    "Build user preference profile",
                    "Content similarity matching",
                    "Combine with collaborative signals",
                    "Explain recommendations"
                ],
                "hints": {
                    "level1": "TF-IDF on item descriptions. User profile = weighted avg of liked item vectors.",
                    "level2": "For explainability, track which features contributed most to recommendation.",
                    "level3": """from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.item_features = None
        self.item_ids = []

    def fit(self, items: List[Dict]):
        # items = [{'id': '1', 'title': '...', 'description': '...', 'tags': [...]}]
        self.item_ids = [item['id'] for item in items]
        self.item_idx = {id: i for i, id in enumerate(self.item_ids)}

        # Combine text features
        texts = []
        for item in items:
            text = f"{item.get('title', '')} {item.get('description', '')} {' '.join(item.get('tags', []))}"
            texts.append(text)

        # TF-IDF matrix
        self.item_features = self.tfidf.fit_transform(texts)

        # Precompute item similarity
        self.item_similarity = cosine_similarity(self.item_features)

    def build_user_profile(self, user_ratings: List[tuple]) -> np.ndarray:
        # user_ratings = [(item_id, rating), ...]
        if not user_ratings:
            return np.zeros(self.item_features.shape[1])

        # Weighted average of item vectors
        profile = np.zeros(self.item_features.shape[1])
        total_weight = 0

        for item_id, rating in user_ratings:
            if item_id in self.item_idx:
                idx = self.item_idx[item_id]
                weight = rating - 3  # Center around neutral rating
                profile += weight * self.item_features[idx].toarray().flatten()
                total_weight += abs(weight)

        if total_weight > 0:
            profile /= total_weight

        return profile

    def recommend(self, user_ratings: List[tuple], n: int = 10) -> List[Dict]:
        user_profile = self.build_user_profile(user_ratings)

        # Similarity to user profile
        scores = cosine_similarity([user_profile], self.item_features)[0]

        # Exclude already rated
        rated_items = {r[0] for r in user_ratings}

        recommendations = []
        for idx in np.argsort(scores)[::-1]:
            item_id = self.item_ids[idx]
            if item_id not in rated_items:
                # Get explanation
                explanation = self._explain(idx, user_profile)
                recommendations.append({
                    'item_id': item_id,
                    'score': float(scores[idx]),
                    'explanation': explanation
                })
                if len(recommendations) >= n:
                    break

        return recommendations

    def _explain(self, item_idx: int, user_profile: np.ndarray, top_k: int = 3) -> str:
        item_vector = self.item_features[item_idx].toarray().flatten()

        # Find overlapping important features
        feature_names = self.tfidf.get_feature_names_out()

        contributions = item_vector * user_profile
        top_indices = np.argsort(contributions)[::-1][:top_k]

        top_features = [feature_names[i] for i in top_indices if contributions[i] > 0]

        if top_features:
            return f"Recommended because you liked items with: {', '.join(top_features)}"
        return "Recommended based on your preferences"

    def similar_items(self, item_id: str, n: int = 10) -> List[tuple]:
        if item_id not in self.item_idx:
            return []

        idx = self.item_idx[item_id]
        similarities = self.item_similarity[idx]

        similar = []
        for i in np.argsort(similarities)[::-1][1:n+1]:  # Skip self
            similar.append((self.item_ids[i], float(similarities[i])))

        return similar"""
                },
                "pitfalls": [
                    "Feature extraction misses important signals",
                    "User profile dominated by few items",
                    "Filter bubble: only recommend similar items",
                    "Explanations don't match user perception"
                ],
                "concepts": ["Content-based filtering", "TF-IDF", "User profiling", "Explainability"],
                "estimated_hours": "6-10"
            },
            {
                "id": 4,
                "name": "Production System",
                "description": "Build production-ready recommendation service.",
                "acceptance_criteria": [
                    "Real-time recommendation API",
                    "Batch candidate generation",
                    "A/B testing framework",
                    "Metrics and monitoring",
                    "Model versioning"
                ],
                "hints": {
                    "level1": "Two-stage: batch generates candidates, real-time ranks them.",
                    "level2": "Track impressions, clicks, conversions. Calculate CTR, conversion rate.",
                    "level3": """from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import random

app = FastAPI()

class RecommendationRequest(BaseModel):
    user_id: str
    context: Optional[Dict] = {}
    n: int = 10
    experiment_id: Optional[str] = None

class RecommendationResponse(BaseModel):
    items: List[Dict]
    model_version: str
    experiment_group: str

class RecommendationService:
    def __init__(self):
        self.models = {}  # version -> model
        self.candidate_cache = {}  # user_id -> candidates
        self.experiments = {}  # experiment_id -> config

    def load_model(self, version: str, model):
        self.models[version] = model

    async def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        # Determine experiment group
        experiment_group = self._assign_experiment_group(
            request.user_id, request.experiment_id
        )
        model_version = self._get_model_for_group(experiment_group)

        # Get candidates (from cache or generate)
        candidates = await self._get_candidates(request.user_id)

        # Rank candidates
        model = self.models[model_version]
        ranked = model.rank(request.user_id, candidates, request.context)

        # Apply business rules
        final = self._apply_business_rules(ranked, request.context)

        return RecommendationResponse(
            items=final[:request.n],
            model_version=model_version,
            experiment_group=experiment_group
        )

    def _assign_experiment_group(self, user_id: str, experiment_id: str) -> str:
        if not experiment_id or experiment_id not in self.experiments:
            return "control"

        config = self.experiments[experiment_id]
        # Deterministic assignment based on user_id
        hash_val = hash(f"{user_id}_{experiment_id}") % 100

        cumulative = 0
        for group, percentage in config['groups'].items():
            cumulative += percentage
            if hash_val < cumulative:
                return group

        return "control"

# Metrics tracking
class MetricsTracker:
    def __init__(self):
        self.events = []

    def track_impression(self, user_id: str, items: List[str],
                        experiment_group: str, model_version: str):
        self.events.append({
            'type': 'impression',
            'user_id': user_id,
            'items': items,
            'experiment_group': experiment_group,
            'model_version': model_version,
            'timestamp': datetime.now()
        })

    def track_click(self, user_id: str, item_id: str, position: int):
        self.events.append({
            'type': 'click',
            'user_id': user_id,
            'item_id': item_id,
            'position': position,
            'timestamp': datetime.now()
        })

    def calculate_metrics(self, experiment_id: str) -> Dict:
        impressions = [e for e in self.events if e['type'] == 'impression']
        clicks = [e for e in self.events if e['type'] == 'click']

        by_group = {}
        for group in set(e.get('experiment_group') for e in impressions):
            group_impressions = [e for e in impressions if e.get('experiment_group') == group]
            group_users = set(e['user_id'] for e in group_impressions)
            group_clicks = [e for e in clicks if e['user_id'] in group_users]

            by_group[group] = {
                'impressions': len(group_impressions),
                'clicks': len(group_clicks),
                'ctr': len(group_clicks) / max(1, len(group_impressions)),
                'unique_users': len(group_users)
            }

        return by_group"""
                },
                "pitfalls": [
                    "Cold cache causes latency spikes",
                    "A/B test assignment not deterministic",
                    "Metrics delayed lose temporal correlation",
                    "Model drift not detected"
                ],
                "concepts": ["Two-stage recommendation", "A/B testing", "Metrics tracking", "Model serving"],
                "estimated_hours": "12-16"
            }
        ]
    }
}

# Load YAML
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# Update AI/ML domain with new projects
for domain in data['domains']:
    if domain['id'] == 'ai-ml':
        # Add to beginner
        if 'beginner' not in domain['projects']:
            domain['projects']['beginner'] = []
        domain['projects']['beginner'].append({
            'id': 'chatbot-intent',
            'name': 'Intent-Based Chatbot',
            'detailed': True
        })

        # Add to intermediate
        if 'intermediate' not in domain['projects']:
            domain['projects']['intermediate'] = []
        existing_ids = [p['id'] for p in domain['projects']['intermediate']]
        for proj_id in ['rag-system', 'semantic-search', 'recommendation-engine']:
            if proj_id not in existing_ids:
                domain['projects']['intermediate'].append({
                    'id': proj_id,
                    'name': ai_projects[proj_id]['name'],
                    'detailed': True
                })

        # Add to advanced
        if 'advanced' not in domain['projects']:
            domain['projects']['advanced'] = []
        existing_ids = [p['id'] for p in domain['projects']['advanced']]
        for proj_id in ['ai-agent-framework', 'llm-eval-framework']:
            if proj_id not in existing_ids:
                domain['projects']['advanced'].append({
                    'id': proj_id,
                    'name': ai_projects[proj_id]['name'],
                    'detailed': True
                })
        break

# Add expert_projects
expert_projects = data.get('expert_projects', {})
for proj_id, proj_data in ai_projects.items():
    expert_projects[proj_id] = proj_data
    print(f"Added: {proj_id}")

data['expert_projects'] = expert_projects

# Save
with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nTotal expert_projects: {len(expert_projects)}")

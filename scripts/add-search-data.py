#!/usr/bin/env python3
"""
Add Search & Data Processing projects - search engines, ETL, and CDC.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

search_data_projects = {
    "search-engine": {
        "name": "Full-Text Search Engine",
        "description": "Build a search engine with inverted indexes, TF-IDF ranking, fuzzy matching, and query parsing like Elasticsearch/Meilisearch.",
        "why_expert": "Search is everywhere. Understanding inverted indexes, ranking algorithms, and query optimization helps debug search issues and build better search UX.",
        "difficulty": "expert",
        "tags": ["search", "indexing", "information-retrieval", "ranking", "nlp"],
        "estimated_hours": 55,
        "prerequisites": [],
        "milestones": [
            {
                "name": "Inverted Index",
                "description": "Implement an inverted index with tokenization and normalization",
                "skills": ["Inverted indexes", "Tokenization", "Text normalization"],
                "hints": {
                    "level1": "Inverted index: term -> [doc_id, positions] for fast lookup",
                    "level2": "Tokenization splits text; normalization lowercases, removes accents, stems",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import re
import unicodedata

@dataclass
class Posting:
    doc_id: str
    positions: list[int]      # Word positions in document
    term_frequency: int       # Count of term in doc

@dataclass
class Document:
    id: str
    content: str
    fields: dict = field(default_factory=dict)  # title, body, tags, etc.
    metadata: dict = field(default_factory=dict)

class Tokenizer:
    def __init__(self, min_length: int = 2, stopwords: set = None):
        self.min_length = min_length
        self.stopwords = stopwords or {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'is', 'it', 'this', 'that', 'with'
        }

    def tokenize(self, text: str) -> list[tuple[str, int]]:
        '''Tokenize text, return (token, position) pairs'''
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))

        # Lowercase
        text = text.lower()

        # Split on non-alphanumeric
        tokens = []
        position = 0

        for match in re.finditer(r'[a-z0-9]+', text):
            word = match.group()
            if len(word) >= self.min_length and word not in self.stopwords:
                tokens.append((word, position))
            position += 1

        return tokens

    def stem(self, word: str) -> str:
        '''Simple Porter-like stemming'''
        # Very simplified - real implementation uses Porter/Snowball
        suffixes = ['ing', 'ed', 'ly', 'es', 's']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

class InvertedIndex:
    def __init__(self, tokenizer: Tokenizer = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.index: dict[str, list[Posting]] = defaultdict(list)
        self.documents: dict[str, Document] = {}
        self.doc_lengths: dict[str, int] = {}  # For BM25

    def add_document(self, doc: Document):
        '''Index a document'''
        self.documents[doc.id] = doc

        # Combine all fields for indexing
        full_text = doc.content
        for field_name, field_value in doc.fields.items():
            if isinstance(field_value, str):
                full_text += " " + field_value

        tokens = self.tokenizer.tokenize(full_text)
        self.doc_lengths[doc.id] = len(tokens)

        # Build term -> positions mapping for this doc
        term_positions: dict[str, list[int]] = defaultdict(list)
        for token, position in tokens:
            stemmed = self.tokenizer.stem(token)
            term_positions[stemmed].append(position)

        # Add to inverted index
        for term, positions in term_positions.items():
            posting = Posting(
                doc_id=doc.id,
                positions=positions,
                term_frequency=len(positions)
            )

            # Insert in sorted order by doc_id for efficient merging
            postings = self.index[term]
            # Binary search insert
            lo, hi = 0, len(postings)
            while lo < hi:
                mid = (lo + hi) // 2
                if postings[mid].doc_id < doc.id:
                    lo = mid + 1
                else:
                    hi = mid
            postings.insert(lo, posting)

    def remove_document(self, doc_id: str):
        '''Remove document from index'''
        if doc_id not in self.documents:
            return

        del self.documents[doc_id]
        del self.doc_lengths[doc_id]

        # Remove from all posting lists
        for term in list(self.index.keys()):
            self.index[term] = [p for p in self.index[term] if p.doc_id != doc_id]
            if not self.index[term]:
                del self.index[term]

    def get_postings(self, term: str) -> list[Posting]:
        '''Get postings for a term'''
        stemmed = self.tokenizer.stem(term.lower())
        return self.index.get(stemmed, [])

    def search(self, query: str) -> list[tuple[str, list[Posting]]]:
        '''Basic AND search - returns docs containing all terms'''
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        # Get postings for each term
        term_postings = []
        for token, _ in tokens:
            stemmed = self.tokenizer.stem(token)
            postings = self.index.get(stemmed, [])
            if not postings:
                return []  # AND semantics: no matches
            term_postings.append((stemmed, postings))

        # Intersect posting lists (find docs containing all terms)
        # Start with smallest list for efficiency
        term_postings.sort(key=lambda x: len(x[1]))

        result_docs = {p.doc_id for p in term_postings[0][1]}
        for _, postings in term_postings[1:]:
            result_docs &= {p.doc_id for p in postings}

        # Collect matching postings
        results = []
        for doc_id in result_docs:
            doc_postings = []
            for term, postings in term_postings:
                for p in postings:
                    if p.doc_id == doc_id:
                        doc_postings.append(p)
                        break
            results.append((doc_id, doc_postings))

        return results
```
"""
                },
                "pitfalls": [
                    "Stemming can over-stem (running -> run -> r) - use proven algorithms",
                    "Stopwords list should be configurable per language",
                    "Unicode normalization crucial for international text",
                    "Index updates are expensive - use batch updates when possible"
                ]
            },
            {
                "name": "TF-IDF & BM25 Ranking",
                "description": "Implement relevance ranking with TF-IDF and BM25 algorithms",
                "skills": ["TF-IDF", "BM25", "Relevance scoring"],
                "hints": {
                    "level1": "TF-IDF: terms appearing often in doc but rarely overall are important",
                    "level2": "BM25 improves on TF-IDF with document length normalization",
                    "level3": """
```python
import math
from dataclasses import dataclass

@dataclass
class SearchResult:
    doc_id: str
    score: float
    highlights: dict = None  # term -> [positions]

class Scorer:
    def __init__(self, index: InvertedIndex):
        self.index = index

    def tf(self, term_freq: int) -> float:
        '''Term frequency component'''
        return 1 + math.log(term_freq) if term_freq > 0 else 0

    def idf(self, term: str) -> float:
        '''Inverse document frequency'''
        N = len(self.index.documents)
        df = len(self.index.get_postings(term))
        if df == 0:
            return 0
        return math.log(N / df)

    def tf_idf_score(self, doc_id: str, terms: list[str]) -> float:
        '''Calculate TF-IDF score for document'''
        score = 0.0
        for term in terms:
            postings = self.index.get_postings(term)
            for posting in postings:
                if posting.doc_id == doc_id:
                    score += self.tf(posting.term_frequency) * self.idf(term)
                    break
        return score

class BM25Scorer(Scorer):
    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75):
        super().__init__(index)
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Length normalization

        # Precompute average document length
        total_length = sum(index.doc_lengths.values())
        self.avg_doc_length = total_length / len(index.documents) if index.documents else 0

    def bm25_idf(self, term: str) -> float:
        '''BM25 IDF variant'''
        N = len(self.index.documents)
        df = len(self.index.get_postings(term))
        if df == 0:
            return 0
        # Robertson-Sparck Jones IDF
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, doc_id: str, terms: list[str]) -> float:
        '''Calculate BM25 score'''
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        if doc_length == 0:
            return 0

        score = 0.0
        for term in terms:
            idf = self.bm25_idf(term)

            # Find term frequency in this doc
            tf = 0
            for posting in self.index.get_postings(term):
                if posting.doc_id == doc_id:
                    tf = posting.term_frequency
                    break

            if tf == 0:
                continue

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        '''Search with BM25 ranking'''
        tokens = self.index.tokenizer.tokenize(query)
        terms = [self.index.tokenizer.stem(t[0]) for t in tokens]

        # Get candidate documents (OR semantics for ranking)
        candidate_docs = set()
        for term in terms:
            for posting in self.index.get_postings(term):
                candidate_docs.add(posting.doc_id)

        # Score all candidates
        results = []
        for doc_id in candidate_docs:
            score = self.score(doc_id, terms)
            if score > 0:
                # Build highlights
                highlights = {}
                for term in terms:
                    for posting in self.index.get_postings(term):
                        if posting.doc_id == doc_id:
                            highlights[term] = posting.positions[:5]  # First 5 positions
                            break

                results.append(SearchResult(
                    doc_id=doc_id,
                    score=score,
                    highlights=highlights
                ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

class FieldWeightedScorer(BM25Scorer):
    '''BM25 with field boosting (title matches worth more than body)'''

    def __init__(self, index: InvertedIndex, field_weights: dict = None):
        super().__init__(index)
        self.field_weights = field_weights or {
            'title': 3.0,
            'body': 1.0,
            'tags': 2.0
        }

    def score(self, doc_id: str, terms: list[str]) -> float:
        '''Score with field weighting'''
        base_score = super().score(doc_id, terms)

        # Boost based on field matches
        doc = self.index.documents.get(doc_id)
        if not doc:
            return base_score

        boost = 1.0
        for field_name, weight in self.field_weights.items():
            field_value = doc.fields.get(field_name, '')
            if isinstance(field_value, str):
                field_tokens = set(
                    self.index.tokenizer.stem(t[0])
                    for t in self.index.tokenizer.tokenize(field_value)
                )
                # Check how many query terms match this field
                matches = sum(1 for term in terms if term in field_tokens)
                if matches > 0:
                    boost += (weight - 1.0) * (matches / len(terms))

        return base_score * boost
```
"""
                },
                "pitfalls": [
                    "BM25 k1 and b parameters need tuning for your data",
                    "Very short documents can get artificially high scores",
                    "Precompute IDF values for performance",
                    "Field boosting can be gamed - normalize carefully"
                ]
            },
            {
                "name": "Fuzzy Matching & Autocomplete",
                "description": "Implement typo tolerance with Levenshtein distance and prefix-based autocomplete",
                "skills": ["Edit distance", "Prefix trees", "Fuzzy search"],
                "hints": {
                    "level1": "Levenshtein distance counts edits needed to transform one string to another",
                    "level2": "For autocomplete, use trie (prefix tree) for O(prefix_length) lookup",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional

class TrieNode:
    def __init__(self):
        self.children: dict[str, 'TrieNode'] = {}
        self.is_end: bool = False
        self.frequency: int = 0  # For ranking suggestions
        self.doc_ids: set[str] = set()

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, doc_id: str = None, frequency: int = 1):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.frequency += frequency
        if doc_id:
            node.doc_ids.add(doc_id)

    def search_prefix(self, prefix: str, limit: int = 10) -> list[tuple[str, int, set]]:
        '''Find all words with given prefix'''
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]

        # DFS to find all words
        results = []
        self._collect_words(node, prefix, results, limit * 3)

        # Sort by frequency and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _collect_words(self, node: TrieNode, prefix: str,
                       results: list, limit: int):
        if len(results) >= limit:
            return

        if node.is_end:
            results.append((prefix, node.frequency, node.doc_ids))

        for char, child in node.children.items():
            self._collect_words(child, prefix + char, results, limit)

class FuzzyMatcher:
    def __init__(self, max_distance: int = 2):
        self.max_distance = max_distance

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        '''Calculate Levenshtein edit distance'''
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def damerau_levenshtein(self, s1: str, s2: str) -> int:
        '''Damerau-Levenshtein: includes transpositions'''
        len1, len2 = len(s1), len(s2)

        # Create distance matrix
        d = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            d[i][0] = i
        for j in range(len2 + 1):
            d[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1

                d[i][j] = min(
                    d[i-1][j] + 1,      # Deletion
                    d[i][j-1] + 1,      # Insertion
                    d[i-1][j-1] + cost  # Substitution
                )

                # Transposition
                if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                    d[i][j] = min(d[i][j], d[i-2][j-2] + cost)

        return d[len1][len2]

    def find_fuzzy_matches(self, query: str, vocabulary: list[str],
                           max_distance: int = None) -> list[tuple[str, int]]:
        '''Find words within edit distance'''
        max_dist = max_distance or self.max_distance
        matches = []

        for word in vocabulary:
            distance = self.damerau_levenshtein(query.lower(), word.lower())
            if distance <= max_dist:
                matches.append((word, distance))

        # Sort by distance then alphabetically
        matches.sort(key=lambda x: (x[1], x[0]))
        return matches

class FuzzySearcher:
    def __init__(self, index: InvertedIndex, max_typos: int = 2):
        self.index = index
        self.trie = Trie()
        self.fuzzy = FuzzyMatcher(max_typos)
        self.vocabulary = set()

        # Build trie from index vocabulary
        for term in index.index.keys():
            self.trie.insert(term)
            self.vocabulary.add(term)

    def autocomplete(self, prefix: str, limit: int = 10) -> list[dict]:
        '''Autocomplete suggestions'''
        suggestions = self.trie.search_prefix(prefix, limit)
        return [
            {'term': term, 'frequency': freq, 'doc_count': len(docs)}
            for term, freq, docs in suggestions
        ]

    def expand_query(self, query: str) -> list[str]:
        '''Expand query with fuzzy matches'''
        tokens = self.index.tokenizer.tokenize(query)
        expanded_terms = []

        for token, _ in tokens:
            stemmed = self.index.tokenizer.stem(token)

            # Exact match first
            if stemmed in self.vocabulary:
                expanded_terms.append(stemmed)
                continue

            # Fuzzy match
            matches = self.fuzzy.find_fuzzy_matches(
                stemmed, list(self.vocabulary), max_distance=2
            )

            if matches:
                # Add top fuzzy match
                expanded_terms.append(matches[0][0])
            else:
                expanded_terms.append(stemmed)

        return expanded_terms

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        '''Search with typo tolerance'''
        expanded_terms = self.expand_query(query)

        # Use BM25 scorer with expanded terms
        scorer = BM25Scorer(self.index)

        # Get candidates
        candidate_docs = set()
        for term in expanded_terms:
            for posting in self.index.get_postings(term):
                candidate_docs.add(posting.doc_id)

        results = []
        for doc_id in candidate_docs:
            score = scorer.score(doc_id, expanded_terms)
            if score > 0:
                results.append(SearchResult(doc_id=doc_id, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
```
"""
                },
                "pitfalls": [
                    "Edit distance is O(n*m) - prefilter candidates first",
                    "Max 2 typos is usually enough; more causes too many false matches",
                    "Transpositions are common typos - Damerau-Levenshtein handles them",
                    "Short words need fewer allowed edits (1 typo in 'cat' is too much)"
                ]
            },
            {
                "name": "Query Parser & Filters",
                "description": "Implement query parsing with boolean operators, phrases, and field filters",
                "skills": ["Query parsing", "Boolean logic", "Filter expressions"],
                "hints": {
                    "level1": "Parse: 'title:python AND (error OR exception) -deprecated'",
                    "level2": "Build AST from query, then evaluate against documents",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Union
from enum import Enum
import re

class QueryOperator(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

@dataclass
class TermQuery:
    term: str
    field: str = None        # None = search all fields
    is_phrase: bool = False  # "exact phrase"
    is_prefix: bool = False  # python*

@dataclass
class BooleanQuery:
    operator: QueryOperator
    operands: list  # List of Query objects

@dataclass
class FilterQuery:
    field: str
    operator: str  # =, !=, >, <, >=, <=, in
    value: any

Query = Union[TermQuery, BooleanQuery, FilterQuery]

class QueryParser:
    def __init__(self):
        self.default_operator = QueryOperator.AND

    def parse(self, query_string: str) -> Query:
        '''Parse query string into AST'''
        tokens = self._tokenize(query_string)
        return self._parse_or(tokens, 0)[0]

    def _tokenize(self, query: str) -> list[str]:
        '''Tokenize query into terms and operators'''
        tokens = []
        i = 0

        while i < len(query):
            # Skip whitespace
            if query[i].isspace():
                i += 1
                continue

            # Quoted phrase
            if query[i] == '"':
                j = query.find('"', i + 1)
                if j == -1:
                    j = len(query)
                tokens.append(('PHRASE', query[i+1:j]))
                i = j + 1
                continue

            # Parentheses
            if query[i] in '()':
                tokens.append((query[i], query[i]))
                i += 1
                continue

            # Operators and terms
            word_match = re.match(r'[\w:*\-\.]+', query[i:])
            if word_match:
                word = word_match.group()
                if word.upper() in ('AND', 'OR', 'NOT'):
                    tokens.append((word.upper(), word.upper()))
                elif ':' in word:
                    field, value = word.split(':', 1)
                    tokens.append(('FIELD', (field, value)))
                elif word.startswith('-'):
                    tokens.append(('NOT_TERM', word[1:]))
                elif word.endswith('*'):
                    tokens.append(('PREFIX', word[:-1]))
                else:
                    tokens.append(('TERM', word))
                i += len(word)
                continue

            i += 1

        return tokens

    def _parse_or(self, tokens: list, pos: int) -> tuple[Query, int]:
        '''Parse OR expressions'''
        left, pos = self._parse_and(tokens, pos)

        while pos < len(tokens) and tokens[pos][0] == 'OR':
            pos += 1
            right, pos = self._parse_and(tokens, pos)
            left = BooleanQuery(QueryOperator.OR, [left, right])

        return left, pos

    def _parse_and(self, tokens: list, pos: int) -> tuple[Query, int]:
        '''Parse AND expressions'''
        left, pos = self._parse_not(tokens, pos)

        while pos < len(tokens):
            if tokens[pos][0] == 'AND':
                pos += 1
                right, pos = self._parse_not(tokens, pos)
                left = BooleanQuery(QueryOperator.AND, [left, right])
            elif tokens[pos][0] in ('TERM', 'PHRASE', 'FIELD', 'PREFIX', '('):
                # Implicit AND
                right, pos = self._parse_not(tokens, pos)
                left = BooleanQuery(QueryOperator.AND, [left, right])
            else:
                break

        return left, pos

    def _parse_not(self, tokens: list, pos: int) -> tuple[Query, int]:
        '''Parse NOT expressions'''
        if pos < len(tokens):
            if tokens[pos][0] == 'NOT':
                pos += 1
                operand, pos = self._parse_primary(tokens, pos)
                return BooleanQuery(QueryOperator.NOT, [operand]), pos
            elif tokens[pos][0] == 'NOT_TERM':
                term = tokens[pos][1]
                return BooleanQuery(QueryOperator.NOT, [TermQuery(term)]), pos + 1

        return self._parse_primary(tokens, pos)

    def _parse_primary(self, tokens: list, pos: int) -> tuple[Query, int]:
        '''Parse primary expressions'''
        if pos >= len(tokens):
            return TermQuery(''), pos

        token_type, token_value = tokens[pos]

        if token_type == '(':
            result, pos = self._parse_or(tokens, pos + 1)
            if pos < len(tokens) and tokens[pos][0] == ')':
                pos += 1
            return result, pos

        if token_type == 'PHRASE':
            return TermQuery(token_value, is_phrase=True), pos + 1

        if token_type == 'FIELD':
            field, value = token_value
            return TermQuery(value, field=field), pos + 1

        if token_type == 'PREFIX':
            return TermQuery(token_value, is_prefix=True), pos + 1

        if token_type == 'TERM':
            return TermQuery(token_value), pos + 1

        return TermQuery(''), pos + 1

class QueryExecutor:
    def __init__(self, index: InvertedIndex, scorer: BM25Scorer):
        self.index = index
        self.scorer = scorer

    def execute(self, query: Query) -> set[str]:
        '''Execute query, return matching doc IDs'''
        if isinstance(query, TermQuery):
            return self._execute_term(query)
        elif isinstance(query, BooleanQuery):
            return self._execute_boolean(query)
        return set()

    def _execute_term(self, query: TermQuery) -> set[str]:
        if query.is_phrase:
            return self._phrase_search(query.term, query.field)
        elif query.is_prefix:
            return self._prefix_search(query.term, query.field)
        else:
            postings = self.index.get_postings(query.term)
            return {p.doc_id for p in postings}

    def _execute_boolean(self, query: BooleanQuery) -> set[str]:
        if query.operator == QueryOperator.AND:
            result = None
            for operand in query.operands:
                matches = self.execute(operand)
                if result is None:
                    result = matches
                else:
                    result &= matches
            return result or set()

        elif query.operator == QueryOperator.OR:
            result = set()
            for operand in query.operands:
                result |= self.execute(operand)
            return result

        elif query.operator == QueryOperator.NOT:
            all_docs = set(self.index.documents.keys())
            excluded = self.execute(query.operands[0])
            return all_docs - excluded

        return set()

    def _phrase_search(self, phrase: str, field: str = None) -> set[str]:
        '''Search for exact phrase'''
        tokens = self.index.tokenizer.tokenize(phrase)
        if not tokens:
            return set()

        terms = [self.index.tokenizer.stem(t[0]) for t in tokens]

        # Get docs containing all terms
        candidate_docs = None
        term_postings = {}
        for term in terms:
            postings = self.index.get_postings(term)
            doc_ids = {p.doc_id for p in postings}
            if candidate_docs is None:
                candidate_docs = doc_ids
            else:
                candidate_docs &= doc_ids
            term_postings[term] = {p.doc_id: p.positions for p in postings}

        if not candidate_docs:
            return set()

        # Check phrase positions
        results = set()
        for doc_id in candidate_docs:
            positions_list = [term_postings[t].get(doc_id, []) for t in terms]
            if self._check_phrase_positions(positions_list):
                results.add(doc_id)

        return results

    def _check_phrase_positions(self, positions_list: list[list[int]]) -> bool:
        '''Check if positions form consecutive phrase'''
        if not positions_list or not positions_list[0]:
            return False

        # For each starting position of first term
        for start_pos in positions_list[0]:
            match = True
            for i, positions in enumerate(positions_list[1:], 1):
                if start_pos + i not in positions:
                    match = False
                    break
            if match:
                return True
        return False

    def _prefix_search(self, prefix: str, field: str = None) -> set[str]:
        '''Search for terms starting with prefix'''
        results = set()
        prefix = prefix.lower()

        for term in self.index.index.keys():
            if term.startswith(prefix):
                for posting in self.index.index[term]:
                    results.add(posting.doc_id)

        return results
```
"""
                },
                "pitfalls": [
                    "Phrase search requires position tracking - expensive",
                    "NOT without other terms returns entire index",
                    "Deeply nested queries can stack overflow - limit depth",
                    "Wildcard at start is very expensive - requires full scan"
                ]
            }
        ]
    },

    "etl-pipeline": {
        "name": "Data Pipeline / ETL System",
        "description": "Build an ETL pipeline system with job scheduling, transformation DAGs, data validation, and monitoring.",
        "why_expert": "Data pipelines are core infrastructure. Understanding ETL patterns helps build reliable data systems and debug data issues.",
        "difficulty": "expert",
        "tags": ["etl", "data-engineering", "pipelines", "scheduling", "batch"],
        "estimated_hours": 45,
        "prerequisites": ["job-scheduler"],
        "milestones": [
            {
                "name": "Pipeline DAG Definition",
                "description": "Implement pipeline definition with dependencies as directed acyclic graph",
                "skills": ["DAG modeling", "Task dependencies", "Configuration"],
                "hints": {
                    "level1": "Tasks form a DAG - each task can depend on others",
                    "level2": "Topological sort determines execution order; detect cycles",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from enum import Enum
from collections import defaultdict
import time

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    UPSTREAM_FAILED = "upstream_failed"

@dataclass
class Task:
    id: str
    name: str
    callable: Callable
    dependencies: list[str] = field(default_factory=list)
    retries: int = 3
    retry_delay: int = 60
    timeout: int = 3600
    params: dict = field(default_factory=dict)

@dataclass
class TaskRun:
    task_id: str
    status: TaskStatus
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    attempt: int = 1
    error: Optional[str] = None
    output: Any = None

@dataclass
class Pipeline:
    id: str
    name: str
    tasks: dict[str, Task] = field(default_factory=dict)
    schedule: Optional[str] = None  # Cron expression
    description: str = ""

class PipelineBuilder:
    def __init__(self, pipeline_id: str, name: str):
        self.pipeline = Pipeline(id=pipeline_id, name=name)

    def add_task(self, task_id: str, name: str, callable: Callable,
                 dependencies: list[str] = None, **kwargs) -> 'PipelineBuilder':
        task = Task(
            id=task_id,
            name=name,
            callable=callable,
            dependencies=dependencies or [],
            **kwargs
        )
        self.pipeline.tasks[task_id] = task
        return self

    def validate(self) -> list[str]:
        '''Validate pipeline, return errors'''
        errors = []

        # Check for missing dependencies
        for task_id, task in self.pipeline.tasks.items():
            for dep in task.dependencies:
                if dep not in self.pipeline.tasks:
                    errors.append(f"Task '{task_id}' depends on unknown task '{dep}'")

        # Check for cycles
        if self._has_cycle():
            errors.append("Pipeline contains a cycle")

        return errors

    def _has_cycle(self) -> bool:
        '''Detect cycles using DFS'''
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {t: WHITE for t in self.pipeline.tasks}

        def dfs(task_id: str) -> bool:
            color[task_id] = GRAY
            for dep in self.pipeline.tasks[task_id].dependencies:
                if color[dep] == GRAY:
                    return True  # Back edge = cycle
                if color[dep] == WHITE and dfs(dep):
                    return True
            color[task_id] = BLACK
            return False

        for task_id in self.pipeline.tasks:
            if color[task_id] == WHITE:
                if dfs(task_id):
                    return True
        return False

    def build(self) -> Pipeline:
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid pipeline: {errors}")
        return self.pipeline

class DAGExecutor:
    def __init__(self):
        self.runs: dict[str, dict[str, TaskRun]] = {}  # pipeline_run_id -> task_id -> run

    def topological_sort(self, pipeline: Pipeline) -> list[str]:
        '''Get tasks in execution order'''
        in_degree = defaultdict(int)
        for task_id, task in pipeline.tasks.items():
            for dep in task.dependencies:
                in_degree[task_id] += 1

        # Start with tasks that have no dependencies
        queue = [t for t in pipeline.tasks if in_degree[t] == 0]
        result = []

        while queue:
            task_id = queue.pop(0)
            result.append(task_id)

            # Reduce in-degree of dependent tasks
            for other_id, other_task in pipeline.tasks.items():
                if task_id in other_task.dependencies:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        return result

    def get_ready_tasks(self, pipeline: Pipeline,
                        task_runs: dict[str, TaskRun]) -> list[str]:
        '''Get tasks ready to execute (all deps satisfied)'''
        ready = []

        for task_id, task in pipeline.tasks.items():
            run = task_runs.get(task_id)

            # Already running or finished
            if run and run.status != TaskStatus.PENDING:
                continue

            # Check all dependencies succeeded
            deps_satisfied = True
            for dep in task.dependencies:
                dep_run = task_runs.get(dep)
                if not dep_run or dep_run.status != TaskStatus.SUCCESS:
                    deps_satisfied = False
                    break

            if deps_satisfied:
                ready.append(task_id)

        return ready

    def should_skip_task(self, pipeline: Pipeline, task_id: str,
                         task_runs: dict[str, TaskRun]) -> bool:
        '''Check if task should be skipped due to upstream failure'''
        task = pipeline.tasks[task_id]

        for dep in task.dependencies:
            dep_run = task_runs.get(dep)
            if dep_run and dep_run.status in [TaskStatus.FAILED, TaskStatus.UPSTREAM_FAILED]:
                return True
        return False
```
"""
                },
                "pitfalls": [
                    "Cycle detection must run before execution starts",
                    "Topological sort doesn't handle failures - need separate tracking",
                    "Parallel execution needs thread-safe state management",
                    "Long chains of dependencies slow down pipeline"
                ]
            },
            {
                "name": "Data Extraction & Loading",
                "description": "Implement extractors and loaders for common data sources",
                "skills": ["Data connectors", "Batch processing", "Incremental loads"],
                "hints": {
                    "level1": "Extractors read from sources; loaders write to destinations",
                    "level2": "Support full and incremental loads using watermarks",
                    "level3": """
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Any
import json

@dataclass
class DataBatch:
    records: list[dict]
    schema: Optional[dict] = None
    metadata: dict = field(default_factory=dict)

class Extractor(ABC):
    @abstractmethod
    def extract(self, **kwargs) -> Iterator[DataBatch]:
        '''Extract data in batches'''
        pass

    @abstractmethod
    def get_watermark(self) -> Any:
        '''Get current watermark for incremental loads'''
        pass

class Loader(ABC):
    @abstractmethod
    def load(self, batch: DataBatch) -> int:
        '''Load batch, return records loaded'''
        pass

    @abstractmethod
    def begin_transaction(self):
        pass

    @abstractmethod
    def commit(self):
        pass

    @abstractmethod
    def rollback(self):
        pass

class DatabaseExtractor(Extractor):
    def __init__(self, connection, table: str, batch_size: int = 1000,
                 watermark_column: str = None):
        self.conn = connection
        self.table = table
        self.batch_size = batch_size
        self.watermark_column = watermark_column
        self._watermark = None

    def extract(self, since: Any = None, **kwargs) -> Iterator[DataBatch]:
        query = f"SELECT * FROM {self.table}"
        params = []

        if since and self.watermark_column:
            query += f" WHERE {self.watermark_column} > ?"
            params.append(since)

        if self.watermark_column:
            query += f" ORDER BY {self.watermark_column}"

        cursor = self.conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]

        batch = []
        for row in cursor:
            record = dict(zip(columns, row))
            batch.append(record)

            if self.watermark_column:
                self._watermark = record[self.watermark_column]

            if len(batch) >= self.batch_size:
                yield DataBatch(records=batch, schema={"columns": columns})
                batch = []

        if batch:
            yield DataBatch(records=batch, schema={"columns": columns})

    def get_watermark(self) -> Any:
        return self._watermark

class APIExtractor(Extractor):
    def __init__(self, base_url: str, endpoint: str,
                 auth: dict = None, batch_size: int = 100):
        self.base_url = base_url
        self.endpoint = endpoint
        self.auth = auth
        self.batch_size = batch_size
        self._watermark = None

    def extract(self, since: Any = None, **kwargs) -> Iterator[DataBatch]:
        import httpx

        page = 1
        while True:
            params = {"page": page, "per_page": self.batch_size}
            if since:
                params["since"] = since

            response = httpx.get(
                f"{self.base_url}/{self.endpoint}",
                params=params,
                headers=self._build_headers()
            )
            response.raise_for_status()

            data = response.json()
            records = data.get("data", data)  # Handle nested or flat response

            if not records:
                break

            # Track watermark
            if records and "updated_at" in records[-1]:
                self._watermark = records[-1]["updated_at"]

            yield DataBatch(records=records)
            page += 1

            # Check if last page
            if len(records) < self.batch_size:
                break

    def _build_headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.auth:
            if "token" in self.auth:
                headers["Authorization"] = f"Bearer {self.auth['token']}"
        return headers

    def get_watermark(self) -> Any:
        return self._watermark

class FileExtractor(Extractor):
    def __init__(self, path: str, format: str = "json", batch_size: int = 1000):
        self.path = path
        self.format = format
        self.batch_size = batch_size

    def extract(self, **kwargs) -> Iterator[DataBatch]:
        import csv

        if self.format == "json":
            with open(self.path) as f:
                data = json.load(f)
                for i in range(0, len(data), self.batch_size):
                    yield DataBatch(records=data[i:i + self.batch_size])

        elif self.format == "csv":
            with open(self.path) as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    batch.append(row)
                    if len(batch) >= self.batch_size:
                        yield DataBatch(records=batch)
                        batch = []
                if batch:
                    yield DataBatch(records=batch)

        elif self.format == "jsonl":
            with open(self.path) as f:
                batch = []
                for line in f:
                    batch.append(json.loads(line))
                    if len(batch) >= self.batch_size:
                        yield DataBatch(records=batch)
                        batch = []
                if batch:
                    yield DataBatch(records=batch)

    def get_watermark(self) -> Any:
        import os
        return os.path.getmtime(self.path)

class DatabaseLoader(Loader):
    def __init__(self, connection, table: str, mode: str = "append"):
        self.conn = connection
        self.table = table
        self.mode = mode  # append, replace, upsert

    def begin_transaction(self):
        pass  # Connection handles this

    def load(self, batch: DataBatch) -> int:
        if not batch.records:
            return 0

        columns = list(batch.records[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        column_names = ", ".join(columns)

        if self.mode == "upsert":
            # SQLite upsert syntax
            query = f'''
                INSERT INTO {self.table} ({column_names})
                VALUES ({placeholders})
                ON CONFLICT DO UPDATE SET
                {", ".join(f"{c} = excluded.{c}" for c in columns)}
            '''
        else:
            query = f"INSERT INTO {self.table} ({column_names}) VALUES ({placeholders})"

        for record in batch.records:
            values = [record.get(c) for c in columns]
            self.conn.execute(query, values)

        return len(batch.records)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()
```
"""
                },
                "pitfalls": [
                    "API pagination can change during extraction - use stable cursors",
                    "Watermark must be saved atomically with loaded data",
                    "Upsert needs proper conflict detection keys",
                    "Large batches can exhaust memory - stream when possible"
                ]
            },
            {
                "name": "Data Transformations",
                "description": "Implement transformation operations with schema validation",
                "skills": ["Data mapping", "Schema validation", "Type conversion"],
                "hints": {
                    "level1": "Transformations: map, filter, aggregate, join",
                    "level2": "Validate schema before and after transformation",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Callable, Any, Optional
from abc import ABC, abstractmethod
import json
import re

class Transform(ABC):
    @abstractmethod
    def apply(self, batch: DataBatch) -> DataBatch:
        pass

class MapTransform(Transform):
    def __init__(self, mapper: Callable[[dict], dict]):
        self.mapper = mapper

    def apply(self, batch: DataBatch) -> DataBatch:
        transformed = [self.mapper(record) for record in batch.records]
        return DataBatch(records=transformed, metadata=batch.metadata)

class FilterTransform(Transform):
    def __init__(self, predicate: Callable[[dict], bool]):
        self.predicate = predicate

    def apply(self, batch: DataBatch) -> DataBatch:
        filtered = [r for r in batch.records if self.predicate(r)]
        return DataBatch(records=filtered, metadata=batch.metadata)

class RenameColumnsTransform(Transform):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def apply(self, batch: DataBatch) -> DataBatch:
        transformed = []
        for record in batch.records:
            new_record = {}
            for key, value in record.items():
                new_key = self.mapping.get(key, key)
                new_record[new_key] = value
            transformed.append(new_record)
        return DataBatch(records=transformed, metadata=batch.metadata)

class TypeCastTransform(Transform):
    def __init__(self, type_mapping: dict[str, type]):
        self.type_mapping = type_mapping

    def apply(self, batch: DataBatch) -> DataBatch:
        transformed = []
        for record in batch.records:
            new_record = dict(record)
            for field, target_type in self.type_mapping.items():
                if field in new_record:
                    new_record[field] = self._cast(new_record[field], target_type)
            transformed.append(new_record)
        return DataBatch(records=transformed, metadata=batch.metadata)

    def _cast(self, value: Any, target_type: type) -> Any:
        if value is None:
            return None
        if target_type == int:
            return int(float(value))
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            return value in (True, 'true', 'True', '1', 1)
        elif target_type == str:
            return str(value)
        return value

class DeriveColumnTransform(Transform):
    def __init__(self, column: str, expression: Callable[[dict], Any]):
        self.column = column
        self.expression = expression

    def apply(self, batch: DataBatch) -> DataBatch:
        transformed = []
        for record in batch.records:
            new_record = dict(record)
            new_record[self.column] = self.expression(record)
            transformed.append(new_record)
        return DataBatch(records=transformed, metadata=batch.metadata)

# Schema validation
@dataclass
class FieldSchema:
    name: str
    type: str  # string, int, float, bool, datetime, json
    required: bool = True
    nullable: bool = False
    pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_values: Optional[list] = None

@dataclass
class DataSchema:
    fields: list[FieldSchema]
    strict: bool = False  # Reject unknown fields

class SchemaValidator:
    def __init__(self, schema: DataSchema):
        self.schema = schema
        self.field_map = {f.name: f for f in schema.fields}

    def validate(self, batch: DataBatch) -> list[dict]:
        '''Validate batch, return list of errors'''
        errors = []

        for i, record in enumerate(batch.records):
            record_errors = self._validate_record(record, i)
            errors.extend(record_errors)

        return errors

    def _validate_record(self, record: dict, row_num: int) -> list[dict]:
        errors = []

        # Check required fields
        for field in self.schema.fields:
            if field.required and field.name not in record:
                errors.append({
                    "row": row_num,
                    "field": field.name,
                    "error": "required_field_missing"
                })

        # Check each field
        for field_name, value in record.items():
            if field_name not in self.field_map:
                if self.schema.strict:
                    errors.append({
                        "row": row_num,
                        "field": field_name,
                        "error": "unknown_field"
                    })
                continue

            field = self.field_map[field_name]
            field_errors = self._validate_field(value, field, row_num)
            errors.extend(field_errors)

        return errors

    def _validate_field(self, value: Any, field: FieldSchema,
                        row_num: int) -> list[dict]:
        errors = []

        # Null check
        if value is None:
            if not field.nullable:
                errors.append({
                    "row": row_num,
                    "field": field.name,
                    "error": "null_not_allowed"
                })
            return errors

        # Type check
        type_valid = self._check_type(value, field.type)
        if not type_valid:
            errors.append({
                "row": row_num,
                "field": field.name,
                "error": f"invalid_type_expected_{field.type}",
                "value": str(value)[:100]
            })
            return errors

        # Pattern check
        if field.pattern and field.type == "string":
            if not re.match(field.pattern, str(value)):
                errors.append({
                    "row": row_num,
                    "field": field.name,
                    "error": "pattern_mismatch"
                })

        # Range check
        if field.min_value is not None and value < field.min_value:
            errors.append({
                "row": row_num,
                "field": field.name,
                "error": f"below_minimum_{field.min_value}"
            })
        if field.max_value is not None and value > field.max_value:
            errors.append({
                "row": row_num,
                "field": field.name,
                "error": f"above_maximum_{field.max_value}"
            })

        # Enum check
        if field.enum_values and value not in field.enum_values:
            errors.append({
                "row": row_num,
                "field": field.name,
                "error": "invalid_enum_value"
            })

        return errors

    def _check_type(self, value: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(value, str)
        elif expected == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected == "bool":
            return isinstance(value, bool)
        elif expected == "json":
            return isinstance(value, (dict, list))
        return True
```
"""
                },
                "pitfalls": [
                    "Type coercion can lose precision (float -> int)",
                    "Schema validation on every record is slow - sample or skip",
                    "Derived columns can fail - handle exceptions per record",
                    "Null handling differs between databases"
                ]
            },
            {
                "name": "Pipeline Orchestration & Monitoring",
                "description": "Implement pipeline execution with monitoring, alerting, and lineage tracking",
                "skills": ["Orchestration", "Monitoring", "Data lineage"],
                "hints": {
                    "level1": "Track execution metrics: duration, records, errors",
                    "level2": "Data lineage: track where data came from and where it went",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

class PipelineRunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PipelineRun:
    id: str
    pipeline_id: str
    status: PipelineRunStatus
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    task_runs: dict[str, TaskRun] = field(default_factory=dict)
    error: Optional[str] = None
    metrics: dict = field(default_factory=dict)

@dataclass
class LineageRecord:
    run_id: str
    task_id: str
    source: str          # Table, API, file path
    destination: str
    records_read: int
    records_written: int
    timestamp: float

class PipelineOrchestrator:
    def __init__(self, max_parallel: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_parallel)
        self.dag_executor = DAGExecutor()
        self.runs: dict[str, PipelineRun] = {}
        self.lineage: list[LineageRecord] = []

    def run_pipeline(self, pipeline: Pipeline,
                     params: dict = None) -> PipelineRun:
        '''Execute pipeline'''
        run_id = f"{pipeline.id}_{int(time.time())}"
        run = PipelineRun(
            id=run_id,
            pipeline_id=pipeline.id,
            status=PipelineRunStatus.RUNNING,
            started_at=time.time()
        )
        self.runs[run_id] = run

        # Initialize task runs
        for task_id in pipeline.tasks:
            run.task_runs[task_id] = TaskRun(
                task_id=task_id,
                status=TaskStatus.PENDING
            )

        try:
            self._execute_pipeline(pipeline, run, params or {})
            run.status = PipelineRunStatus.SUCCESS
        except Exception as e:
            run.status = PipelineRunStatus.FAILED
            run.error = str(e)
        finally:
            run.finished_at = time.time()
            self._collect_metrics(run)

        return run

    def _execute_pipeline(self, pipeline: Pipeline, run: PipelineRun,
                          params: dict):
        '''Execute tasks in dependency order'''
        completed = set()

        while len(completed) < len(pipeline.tasks):
            # Find tasks ready to run
            ready = self.dag_executor.get_ready_tasks(pipeline, run.task_runs)

            if not ready:
                # Check for failures
                failed = [t for t, r in run.task_runs.items()
                         if r.status == TaskStatus.FAILED]
                if failed:
                    # Mark downstream as upstream_failed
                    for task_id, task_run in run.task_runs.items():
                        if task_run.status == TaskStatus.PENDING:
                            if self.dag_executor.should_skip_task(
                                pipeline, task_id, run.task_runs
                            ):
                                task_run.status = TaskStatus.UPSTREAM_FAILED
                                completed.add(task_id)
                    if len(completed) >= len(pipeline.tasks):
                        break
                    raise Exception(f"Pipeline failed: tasks {failed}")
                continue

            # Execute ready tasks (could parallelize here)
            for task_id in ready:
                self._execute_task(pipeline.tasks[task_id], run, params)
                completed.add(task_id)

    def _execute_task(self, task: Task, run: PipelineRun, params: dict):
        '''Execute single task with retries'''
        task_run = run.task_runs[task.id]
        task_run.started_at = time.time()
        task_run.status = TaskStatus.RUNNING

        # Merge pipeline params with task params
        task_params = {**task.params, **params}

        for attempt in range(1, task.retries + 1):
            task_run.attempt = attempt
            try:
                result = task.callable(**task_params)
                task_run.status = TaskStatus.SUCCESS
                task_run.output = result
                task_run.finished_at = time.time()

                # Record lineage if result contains it
                if isinstance(result, dict) and 'lineage' in result:
                    self._record_lineage(run.id, task.id, result['lineage'])

                return

            except Exception as e:
                task_run.error = f"Attempt {attempt}: {str(e)}"
                if attempt < task.retries:
                    time.sleep(task.retry_delay)
                else:
                    task_run.status = TaskStatus.FAILED
                    task_run.finished_at = time.time()
                    raise

    def _record_lineage(self, run_id: str, task_id: str, lineage: dict):
        record = LineageRecord(
            run_id=run_id,
            task_id=task_id,
            source=lineage.get('source', 'unknown'),
            destination=lineage.get('destination', 'unknown'),
            records_read=lineage.get('records_read', 0),
            records_written=lineage.get('records_written', 0),
            timestamp=time.time()
        )
        self.lineage.append(record)

    def _collect_metrics(self, run: PipelineRun):
        '''Collect run metrics'''
        run.metrics = {
            'duration_seconds': (run.finished_at or time.time()) - run.started_at,
            'tasks_total': len(run.task_runs),
            'tasks_succeeded': sum(1 for r in run.task_runs.values()
                                   if r.status == TaskStatus.SUCCESS),
            'tasks_failed': sum(1 for r in run.task_runs.values()
                                if r.status == TaskStatus.FAILED),
            'total_records': sum(
                r.output.get('records_processed', 0)
                for r in run.task_runs.values()
                if r.output and isinstance(r.output, dict)
            )
        }

    def get_lineage(self, destination: str) -> list[LineageRecord]:
        '''Get lineage for a destination'''
        return [l for l in self.lineage if l.destination == destination]

class PipelineMonitor:
    def __init__(self, orchestrator: PipelineOrchestrator):
        self.orchestrator = orchestrator
        self.alerts: list[dict] = []

    def check_health(self) -> dict:
        '''Check overall pipeline health'''
        recent_runs = [
            r for r in self.orchestrator.runs.values()
            if r.finished_at and time.time() - r.finished_at < 86400
        ]

        success_rate = (
            sum(1 for r in recent_runs if r.status == PipelineRunStatus.SUCCESS) /
            len(recent_runs) if recent_runs else 1.0
        )

        return {
            'total_runs_24h': len(recent_runs),
            'success_rate': success_rate,
            'healthy': success_rate > 0.9
        }

    def alert_on_failure(self, run: PipelineRun):
        '''Create alert for failed run'''
        if run.status == PipelineRunStatus.FAILED:
            self.alerts.append({
                'type': 'pipeline_failure',
                'pipeline_id': run.pipeline_id,
                'run_id': run.id,
                'error': run.error,
                'timestamp': time.time()
            })
```
"""
                },
                "pitfalls": [
                    "Retry delay should use exponential backoff",
                    "Parallel task execution needs careful state management",
                    "Lineage tracking adds overhead - make it optional",
                    "Cancelled pipelines need cleanup of partial data"
                ]
            }
        ]
    },

    "cdc-system": {
        "name": "Change Data Capture (CDC) System",
        "description": "Build a CDC system that captures database changes in real-time using transaction logs and streams them to consumers.",
        "why_expert": "CDC is essential for real-time data sync. Understanding WAL parsing, log-based replication, and event streaming enables building reactive data systems.",
        "difficulty": "expert",
        "tags": ["cdc", "replication", "streaming", "database", "events"],
        "estimated_hours": 50,
        "prerequisites": ["build-sqlite"],
        "milestones": [
            {
                "name": "Log Parsing & Change Events",
                "description": "Parse database transaction logs to extract change events",
                "skills": ["WAL parsing", "Binary protocols", "Event modeling"],
                "hints": {
                    "level1": "WAL/binlog contains: operation type, table, old values, new values",
                    "level2": "Events: INSERT, UPDATE (with before/after), DELETE",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import json
import time

class OperationType(Enum):
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    TRUNCATE = "truncate"
    DDL = "ddl"

@dataclass
class ChangeEvent:
    id: str
    timestamp: float
    transaction_id: str
    operation: OperationType
    database: str
    schema: str
    table: str
    primary_key: dict
    before: Optional[dict] = None   # For UPDATE/DELETE
    after: Optional[dict] = None    # For INSERT/UPDATE
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            'id': self.id,
            'timestamp': self.timestamp,
            'transaction_id': self.transaction_id,
            'operation': self.operation.value,
            'database': self.database,
            'schema': self.schema,
            'table': self.table,
            'primary_key': self.primary_key,
            'before': self.before,
            'after': self.after
        })

@dataclass
class Position:
    '''Track position in transaction log'''
    log_file: str
    log_position: int
    timestamp: float

class LogParser:
    '''Base class for log parsers'''

    def __init__(self, position: Position = None):
        self.position = position
        self.event_count = 0

    def parse(self) -> Iterator[ChangeEvent]:
        raise NotImplementedError

    def get_position(self) -> Position:
        return self.position

# Simulated PostgreSQL logical replication
class PostgresLogicalDecoder:
    '''Parse PostgreSQL logical replication stream'''

    def __init__(self, connection, slot_name: str, publication: str):
        self.conn = connection
        self.slot_name = slot_name
        self.publication = publication
        self.position = None

    def create_slot(self):
        '''Create replication slot'''
        self.conn.execute(f'''
            SELECT pg_create_logical_replication_slot(
                '{self.slot_name}', 'pgoutput'
            )
        ''')

    def read_changes(self, start_lsn: str = None) -> Iterator[ChangeEvent]:
        '''Read changes from replication slot'''
        # In production: use psycopg2 replication protocol
        # This is a simulation using polling

        cursor = self.conn.execute(f'''
            SELECT lsn, data FROM pg_logical_slot_get_changes(
                '{self.slot_name}', NULL, NULL,
                'publication_names', '{self.publication}'
            )
        ''')

        for lsn, data in cursor:
            event = self._parse_message(lsn, data)
            if event:
                yield event

    def _parse_message(self, lsn: str, data: bytes) -> Optional[ChangeEvent]:
        '''Parse logical replication message'''
        # pgoutput format: message_type + payload
        if not data:
            return None

        msg_type = data[0:1]

        if msg_type == b'I':  # Insert
            return self._parse_insert(lsn, data)
        elif msg_type == b'U':  # Update
            return self._parse_update(lsn, data)
        elif msg_type == b'D':  # Delete
            return self._parse_delete(lsn, data)

        return None

    def _parse_insert(self, lsn: str, data: bytes) -> ChangeEvent:
        # Simplified - real implementation parses binary format
        table_info = self._extract_table_info(data)
        new_tuple = self._extract_tuple(data)

        return ChangeEvent(
            id=f"{lsn}_{time.time()}",
            timestamp=time.time(),
            transaction_id=lsn,
            operation=OperationType.INSERT,
            database=table_info['database'],
            schema=table_info['schema'],
            table=table_info['table'],
            primary_key=self._extract_pk(new_tuple),
            after=new_tuple
        )

# MySQL binlog simulation
class MySQLBinlogReader:
    '''Parse MySQL binary log'''

    def __init__(self, connection, server_id: int):
        self.conn = connection
        self.server_id = server_id
        self.position = None

    def read_events(self, binlog_file: str = None,
                    position: int = 0) -> Iterator[ChangeEvent]:
        '''Read events from binlog'''
        # In production: use mysql-replication package
        # This simulates by querying

        cursor = self.conn.execute('''
            SHOW BINLOG EVENTS
        ''')

        for row in cursor:
            event = self._parse_binlog_event(row)
            if event:
                yield event

    def _parse_binlog_event(self, row) -> Optional[ChangeEvent]:
        event_type = row['Event_type']

        if event_type == 'Write_rows':
            return self._create_event(OperationType.INSERT, row)
        elif event_type == 'Update_rows':
            return self._create_event(OperationType.UPDATE, row)
        elif event_type == 'Delete_rows':
            return self._create_event(OperationType.DELETE, row)

        return None

# Generic trigger-based CDC (works with any database)
class TriggerBasedCDC:
    '''CDC using database triggers - fallback for when log access is unavailable'''

    def __init__(self, connection, tables: list[str]):
        self.conn = connection
        self.tables = tables
        self.change_table = '_cdc_changes'

    def setup(self):
        '''Create change capture infrastructure'''
        # Create change log table
        self.conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.change_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT,
                operation TEXT,
                primary_key TEXT,
                old_data TEXT,
                new_data TEXT,
                timestamp REAL,
                processed INTEGER DEFAULT 0
            )
        ''')

        # Create triggers for each table
        for table in self.tables:
            self._create_triggers(table)

    def _create_triggers(self, table: str):
        # Insert trigger
        self.conn.execute(f'''
            CREATE TRIGGER IF NOT EXISTS {table}_insert_cdc
            AFTER INSERT ON {table}
            BEGIN
                INSERT INTO {self.change_table}
                (table_name, operation, primary_key, new_data, timestamp)
                VALUES ('{table}', 'INSERT',
                        NEW.id, json_object('id', NEW.id), unixepoch());
            END
        ''')

        # Update trigger
        self.conn.execute(f'''
            CREATE TRIGGER IF NOT EXISTS {table}_update_cdc
            AFTER UPDATE ON {table}
            BEGIN
                INSERT INTO {self.change_table}
                (table_name, operation, primary_key, old_data, new_data, timestamp)
                VALUES ('{table}', 'UPDATE',
                        NEW.id,
                        json_object('id', OLD.id),
                        json_object('id', NEW.id),
                        unixepoch());
            END
        ''')

        # Delete trigger
        self.conn.execute(f'''
            CREATE TRIGGER IF NOT EXISTS {table}_delete_cdc
            AFTER DELETE ON {table}
            BEGIN
                INSERT INTO {self.change_table}
                (table_name, operation, primary_key, old_data, timestamp)
                VALUES ('{table}', 'DELETE',
                        OLD.id, json_object('id', OLD.id), unixepoch());
            END
        ''')

    def read_changes(self, batch_size: int = 100) -> list[ChangeEvent]:
        '''Read unprocessed changes'''
        cursor = self.conn.execute(f'''
            SELECT * FROM {self.change_table}
            WHERE processed = 0
            ORDER BY id
            LIMIT ?
        ''', (batch_size,))

        events = []
        for row in cursor:
            event = ChangeEvent(
                id=str(row['id']),
                timestamp=row['timestamp'],
                transaction_id=str(row['id']),
                operation=OperationType(row['operation'].lower()),
                database='main',
                schema='public',
                table=row['table_name'],
                primary_key=json.loads(row['primary_key']) if row['primary_key'] else {},
                before=json.loads(row['old_data']) if row['old_data'] else None,
                after=json.loads(row['new_data']) if row['new_data'] else None
            )
            events.append(event)

        return events

    def mark_processed(self, event_ids: list[str]):
        '''Mark events as processed'''
        placeholders = ','.join(['?' for _ in event_ids])
        self.conn.execute(f'''
            UPDATE {self.change_table}
            SET processed = 1
            WHERE id IN ({placeholders})
        ''', event_ids)
        self.conn.commit()
```
"""
                },
                "pitfalls": [
                    "WAL parsing is database-specific - need adapter per DB",
                    "Trigger-based CDC adds write overhead to main tables",
                    "Binary log formats change between versions",
                    "Large transactions can create huge change events"
                ]
            },
            {
                "name": "Event Streaming & Delivery",
                "description": "Stream change events to consumers with ordering and delivery guarantees",
                "skills": ["Event streaming", "Ordering guarantees", "Consumer groups"],
                "hints": {
                    "level1": "Events for same row must be delivered in order",
                    "level2": "Partition by table/pk for parallel processing with ordering",
                    "level3": """
```python
from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import defaultdict
import threading
import queue
import time

@dataclass
class EventPartition:
    key: str  # table:pk combination
    events: queue.Queue = field(default_factory=queue.Queue)
    last_event_time: float = 0
    consumer_offset: int = 0

@dataclass
class Consumer:
    id: str
    group_id: str
    assigned_partitions: set[str] = field(default_factory=set)
    last_heartbeat: float = 0

class CDCEventStream:
    '''Stream CDC events with ordering guarantees'''

    def __init__(self, num_partitions: int = 16):
        self.num_partitions = num_partitions
        self.partitions: dict[int, list[ChangeEvent]] = defaultdict(list)
        self.partition_locks: dict[int, threading.Lock] = {
            i: threading.Lock() for i in range(num_partitions)
        }

        # Consumer group management
        self.consumer_groups: dict[str, dict[str, Consumer]] = defaultdict(dict)
        self.consumer_offsets: dict[str, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # group_id -> partition -> offset

    def _get_partition(self, event: ChangeEvent) -> int:
        '''Determine partition for event (consistent hashing by table:pk)'''
        partition_key = f"{event.table}:{json.dumps(event.primary_key, sort_keys=True)}"
        return hash(partition_key) % self.num_partitions

    def publish(self, event: ChangeEvent):
        '''Publish event to stream'''
        partition_id = self._get_partition(event)

        with self.partition_locks[partition_id]:
            self.partitions[partition_id].append(event)

    def publish_batch(self, events: list[ChangeEvent]):
        '''Publish batch of events'''
        # Group by partition
        by_partition = defaultdict(list)
        for event in events:
            partition_id = self._get_partition(event)
            by_partition[partition_id].append(event)

        # Publish to each partition
        for partition_id, partition_events in by_partition.items():
            with self.partition_locks[partition_id]:
                self.partitions[partition_id].extend(partition_events)

    def subscribe(self, group_id: str, consumer_id: str) -> Consumer:
        '''Register consumer in group'''
        consumer = Consumer(
            id=consumer_id,
            group_id=group_id,
            last_heartbeat=time.time()
        )

        self.consumer_groups[group_id][consumer_id] = consumer
        self._rebalance(group_id)

        return consumer

    def _rebalance(self, group_id: str):
        '''Rebalance partitions among consumers in group'''
        consumers = list(self.consumer_groups[group_id].values())
        if not consumers:
            return

        # Clear current assignments
        for consumer in consumers:
            consumer.assigned_partitions.clear()

        # Assign partitions round-robin
        for i in range(self.num_partitions):
            consumer = consumers[i % len(consumers)]
            consumer.assigned_partitions.add(i)

    def poll(self, consumer: Consumer, max_events: int = 100,
             timeout: float = 1.0) -> list[ChangeEvent]:
        '''Poll for events assigned to consumer'''
        consumer.last_heartbeat = time.time()
        events = []

        for partition_id in consumer.assigned_partitions:
            offset = self.consumer_offsets[consumer.group_id][partition_id]

            with self.partition_locks[partition_id]:
                partition_events = self.partitions[partition_id]
                new_events = partition_events[offset:offset + max_events]
                events.extend(new_events)

        return events

    def commit_offset(self, consumer: Consumer, partition_id: int, offset: int):
        '''Commit consumer offset'''
        self.consumer_offsets[consumer.group_id][partition_id] = offset

    def get_lag(self, group_id: str) -> dict[int, int]:
        '''Get consumer group lag per partition'''
        lag = {}
        for partition_id in range(self.num_partitions):
            with self.partition_locks[partition_id]:
                current_offset = len(self.partitions[partition_id])
            committed_offset = self.consumer_offsets[group_id][partition_id]
            lag[partition_id] = current_offset - committed_offset
        return lag

class CDCEventProcessor:
    '''Process CDC events with handlers'''

    def __init__(self, stream: CDCEventStream, group_id: str):
        self.stream = stream
        self.group_id = group_id
        self.handlers: dict[str, list[Callable]] = defaultdict(list)
        self.running = False

    def register_handler(self, table: str, handler: Callable[[ChangeEvent], None]):
        '''Register handler for table changes'''
        self.handlers[table].append(handler)

    def register_global_handler(self, handler: Callable[[ChangeEvent], None]):
        '''Register handler for all changes'''
        self.handlers['*'].append(handler)

    def start(self, consumer_id: str):
        '''Start processing events'''
        consumer = self.stream.subscribe(self.group_id, consumer_id)
        self.running = True

        while self.running:
            events = self.stream.poll(consumer)

            for event in events:
                self._process_event(event)

            # Commit offsets after processing
            for partition_id in consumer.assigned_partitions:
                current_offset = len(self.stream.partitions[partition_id])
                self.stream.commit_offset(consumer, partition_id, current_offset)

            if not events:
                time.sleep(0.1)  # Back off when no events

    def _process_event(self, event: ChangeEvent):
        '''Process single event'''
        # Table-specific handlers
        for handler in self.handlers.get(event.table, []):
            try:
                handler(event)
            except Exception as e:
                print(f"Handler error for {event.table}: {e}")

        # Global handlers
        for handler in self.handlers.get('*', []):
            try:
                handler(event)
            except Exception as e:
                print(f"Global handler error: {e}")

    def stop(self):
        self.running = False
```
"""
                },
                "pitfalls": [
                    "Consumer rebalancing causes brief processing pause",
                    "Partition by pk, not just table, for better parallelism",
                    "Offset commit after processing prevents duplicates",
                    "Consumer heartbeat failure needs automatic reassignment"
                ]
            },
            {
                "name": "Schema Evolution & Compatibility",
                "description": "Handle schema changes without breaking consumers",
                "skills": ["Schema evolution", "Compatibility checks", "Migration"],
                "hints": {
                    "level1": "Track schema versions; include schema in events or registry",
                    "level2": "Compatibility: backward (new reader, old data), forward (old reader, new data)",
                    "level3": """
```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import json

class CompatibilityType(Enum):
    BACKWARD = "backward"      # New reader can read old data
    FORWARD = "forward"        # Old reader can read new data
    FULL = "full"              # Both directions
    NONE = "none"              # No compatibility check

@dataclass
class SchemaVersion:
    id: int
    table: str
    columns: list[dict]  # [{name, type, nullable, default}]
    primary_key: list[str]
    created_at: float

class SchemaRegistry:
    def __init__(self):
        self.schemas: dict[str, list[SchemaVersion]] = defaultdict(list)
        self.compatibility_mode: dict[str, CompatibilityType] = {}

    def register_schema(self, table: str, columns: list[dict],
                        primary_key: list[str]) -> SchemaVersion:
        '''Register new schema version'''
        versions = self.schemas[table]
        version_id = len(versions) + 1

        schema = SchemaVersion(
            id=version_id,
            table=table,
            columns=columns,
            primary_key=primary_key,
            created_at=time.time()
        )

        # Check compatibility with previous version
        if versions:
            mode = self.compatibility_mode.get(table, CompatibilityType.BACKWARD)
            if not self._check_compatibility(versions[-1], schema, mode):
                raise ValueError(f"Schema change not compatible with {mode.value}")

        versions.append(schema)
        return schema

    def get_schema(self, table: str, version: int = None) -> Optional[SchemaVersion]:
        '''Get schema version'''
        versions = self.schemas.get(table, [])
        if not versions:
            return None
        if version:
            return versions[version - 1] if version <= len(versions) else None
        return versions[-1]

    def _check_compatibility(self, old_schema: SchemaVersion,
                              new_schema: SchemaVersion,
                              mode: CompatibilityType) -> bool:
        '''Check if schema change is compatible'''
        if mode == CompatibilityType.NONE:
            return True

        old_cols = {c['name']: c for c in old_schema.columns}
        new_cols = {c['name']: c for c in new_schema.columns}

        # Backward: new reader can read old data
        # - New columns must be nullable or have default
        # - Column types must be compatible (can widen, not narrow)
        if mode in [CompatibilityType.BACKWARD, CompatibilityType.FULL]:
            for name, col in new_cols.items():
                if name not in old_cols:
                    # New column
                    if not col.get('nullable') and col.get('default') is None:
                        return False

            for name, old_col in old_cols.items():
                if name in new_cols:
                    # Type change must be compatible
                    if not self._types_compatible(old_col['type'],
                                                   new_cols[name]['type']):
                        return False

        # Forward: old reader can read new data
        # - Cannot remove required columns
        # - Cannot add required columns
        if mode in [CompatibilityType.FORWARD, CompatibilityType.FULL]:
            for name, old_col in old_cols.items():
                if name not in new_cols:
                    return False  # Column removed

        return True

    def _types_compatible(self, old_type: str, new_type: str) -> bool:
        '''Check if type change is compatible'''
        # Widening is OK: int -> bigint, varchar(10) -> varchar(100)
        compatible_widenings = {
            ('int', 'bigint'): True,
            ('float', 'double'): True,
            ('varchar', 'text'): True,
        }

        if old_type == new_type:
            return True

        return compatible_widenings.get((old_type, new_type), False)

class SchemaEvolutionHandler:
    '''Handle schema changes in CDC stream'''

    def __init__(self, registry: SchemaRegistry):
        self.registry = registry

    def detect_schema_change(self, event: ChangeEvent,
                              expected_schema: SchemaVersion) -> Optional[list[dict]]:
        '''Detect if event has different schema'''
        event_columns = set()

        if event.after:
            event_columns.update(event.after.keys())
        if event.before:
            event_columns.update(event.before.keys())

        expected_columns = {c['name'] for c in expected_schema.columns}

        # Check for differences
        new_columns = event_columns - expected_columns
        removed_columns = expected_columns - event_columns

        if new_columns or removed_columns:
            return {
                'new_columns': list(new_columns),
                'removed_columns': list(removed_columns)
            }

        return None

    def transform_event(self, event: ChangeEvent,
                         source_schema: SchemaVersion,
                         target_schema: SchemaVersion) -> ChangeEvent:
        '''Transform event from source schema to target schema'''
        source_cols = {c['name']: c for c in source_schema.columns}
        target_cols = {c['name']: c for c in target_schema.columns}

        def transform_record(record: dict) -> dict:
            if not record:
                return record

            transformed = {}

            for col_name, col_def in target_cols.items():
                if col_name in record:
                    # Column exists - may need type conversion
                    transformed[col_name] = self._convert_type(
                        record[col_name],
                        source_cols.get(col_name, {}).get('type'),
                        col_def['type']
                    )
                elif col_def.get('default') is not None:
                    # Use default value
                    transformed[col_name] = col_def['default']
                elif col_def.get('nullable'):
                    # Nullable - use None
                    transformed[col_name] = None
                # else: required column missing - error

            return transformed

        return ChangeEvent(
            id=event.id,
            timestamp=event.timestamp,
            transaction_id=event.transaction_id,
            operation=event.operation,
            database=event.database,
            schema=event.schema,
            table=event.table,
            primary_key=event.primary_key,
            before=transform_record(event.before),
            after=transform_record(event.after),
            metadata={**event.metadata, 'schema_version': target_schema.id}
        )

    def _convert_type(self, value: Any, source_type: str,
                       target_type: str) -> Any:
        '''Convert value between types'''
        if value is None:
            return None

        if source_type == target_type:
            return value

        # Type conversions
        if target_type in ('bigint', 'int'):
            return int(value)
        elif target_type in ('float', 'double'):
            return float(value)
        elif target_type in ('varchar', 'text', 'string'):
            return str(value)
        elif target_type == 'boolean':
            return bool(value)

        return value
```
"""
                },
                "pitfalls": [
                    "DDL events need special handling - may require resync",
                    "Schema registry is single point of failure - replicate it",
                    "Backward compatibility is usually most important",
                    "Type narrowing (bigint->int) can cause data loss"
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

for project_id, project in search_data_projects.items():
    data['expert_projects'][project_id] = project
    print(f"Added: {project_id} - {project['name']}")

with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nAdded {len(search_data_projects)} Search & Data Processing projects")

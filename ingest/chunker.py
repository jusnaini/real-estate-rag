"""Text chunker that splits cleaned documents into smaller overlapping pieces.

Uses sentence-boundary splitting with a configurable token budget (default
512 tokens with 50 tokens overlap). Also detects section headings as natural
split points so that thematically cohesive blocks stay together.
"""

import json
import re
import uuid

import config


HEADING_PATTERN = re.compile(
    r"^("
    r"(?:Understanding|Types|What|How|Why|When|Where|Key|Overview|Tips|Step)\b"
    r"|.*\b(?:in Malaysia|202[0-9]|Explained|Guide|Scheme|Programme|Calculator)\b"
    r"|.*[-–].*Scheme"
    r")"
    r"(?:\s*[:\-–]\s*)?",
    re.IGNORECASE,
)


def is_heading(sentence: str) -> bool:
    cleaned = sentence.strip()
    if not cleaned:
        return False
    # Headings tend to be short
    if len(cleaned) > 150:
        return False
    # Must match the heading pattern
    if HEADING_PATTERN.search(cleaned):
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries and newlines."""
    sentences = re.split(r"(?<=[.!?])\s+| *\n+", text)
    return [s.strip() for s in sentences if s.strip()]


def estimate_tokens(text: str) -> int:
    """Rough token count based on whitespace-split words."""
    return len(text.split())


def chunk_document(doc: dict, chunk_size: int = None, chunk_overlap: int = None) -> list[dict]:
    """Split a single cleaned document into overlapping chunks.

    Chunking first groups sentences into sections at heading boundaries,
    then splits each section into token-budget chunks.  An overlap of
    the last N tokens from the previous chunk is prepended to the next.
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP

    sentences = split_sentences(doc["text"])

    # --- Group sentences into sections at heading boundaries ---
    sections = []
    current_section = []
    for sentence in sentences:
        if is_heading(sentence) and current_section:
            sections.append(current_section)
            current_section = [sentence]
        else:
            current_section.append(sentence)
    if current_section:
        sections.append(current_section)

    # --- Chunk each section using token budget ---
    chunks = []
    for section_sentences in sections:
        current_chunk = []
        current_tokens = 0

        for sentence in section_sentences:
            sentence_tokens = estimate_tokens(sentence)
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(_make_chunk(chunk_text, doc))

                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_tokens = estimate_tokens(s)
                    if overlap_tokens + s_tokens > chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(_make_chunk(chunk_text, doc))

    return chunks


def _make_chunk(text: str, doc: dict) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "source": doc["source"],
        "category": doc["category"],
        "title": doc["title"],
        "url": doc["url"],
        "text": text,
    }


def chunk_all(docs: list[dict]) -> list[dict]:
    """Chunk every document in the list."""
    all_chunks = []
    for doc in docs:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
    return all_chunks

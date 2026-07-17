"""
Chunks parsed documents using one of three strategies:
- fixed: fixed token window with overlap
- semantic: split on embedding-similarity breakpoints
- section: split on resume/JD section headers (Experience, Skills, Education, etc.)

Strategy is set via config.CHUNK_STRATEGY. Compare results in notebooks/chunking_experiments.ipynb.
"""
import re
from dataclasses import dataclass, field

from app.config import CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.parser import ParsedDocument

# Rough token estimate: ~4 chars/token (avoids pulling in a tokenizer dependency here)
CHARS_PER_TOKEN = 4

SECTION_HEADERS = [
    "experience", "work experience", "professional experience",
    "education", "skills", "technical skills", "projects",
    "certifications", "summary", "objective", "achievements",
    "requirements", "responsibilities", "qualifications",
    "about the role", "about the job", "nice to have",
]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_fixed(doc: ParsedDocument) -> list[Chunk]:
    """Fixed-size sliding window over raw text, by approx token count."""
    window_chars = CHUNK_SIZE * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP * CHARS_PER_TOKEN
    text = doc.raw_text

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + window_chars
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_fixed_{idx}",
                doc_id=doc.doc_id,
                doc_type=doc.doc_type,
                section="unknown",
                text=piece,
            ))
            idx += 1
        start += window_chars - overlap_chars

    return chunks


def chunk_section(doc: ParsedDocument) -> list[Chunk]:
    """Split on detected section headers (line that matches a known header, case-insensitive)."""
    lines = doc.raw_text.split("\n")
    header_pattern = re.compile(
        r"^\s*(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:?\s*$",
        re.IGNORECASE,
    )

    sections: list[tuple[str, list[str]]] = []
    current_header = "header"  # content before first detected header
    current_lines: list[str] = []

    for line in lines:
        if header_pattern.match(line):
            if current_lines:
                sections.append((current_header, current_lines))
            current_header = line.strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, current_lines))

    chunks = []
    for idx, (section_name, section_lines) in enumerate(sections):
        text = "\n".join(section_lines).strip()
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_section_{idx}",
            doc_id=doc.doc_id,
            doc_type=doc.doc_type,
            section=section_name,
            text=text,
        ))

    # Fallback: no headers detected → treat whole doc as one chunk
    if not chunks:
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_section_0",
            doc_id=doc.doc_id,
            doc_type=doc.doc_type,
            section="full_document",
            text=doc.raw_text.strip(),
        ))

    return chunks


def chunk_semantic(doc: ParsedDocument, embedder=None, threshold: float = 0.5) -> list[Chunk]:
    """
    Split text into sentences, embed each, and break where cosine similarity
    between consecutive sentences drops below threshold (topic shift).
    Requires an embedder (app.embeddings.embedder.Embedder) passed in — kept
    decoupled here to avoid circular imports / loading the model unnecessarily.
    """
    if embedder is None:
        raise ValueError("chunk_semantic requires an embedder instance (see app/embeddings/embedder.py)")

    import numpy as np

    sentences = re.split(r"(?<=[.!?])\s+", doc.raw_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return chunk_section(doc)  # not enough sentences to split semantically

    embeddings = embedder.embed(sentences)

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    chunks = []
    for idx, group in enumerate(groups):
        text = " ".join(group).strip()
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_semantic_{idx}",
            doc_id=doc.doc_id,
            doc_type=doc.doc_type,
            section="unknown",
            text=text,
        ))

    return chunks


def chunk_document(doc: ParsedDocument, strategy: str = None, embedder=None) -> list[Chunk]:
    strategy = strategy or CHUNK_STRATEGY
    if strategy == "fixed":
        return chunk_fixed(doc)
    elif strategy == "section":
        return chunk_section(doc)
    elif strategy == "semantic":
        return chunk_semantic(doc, embedder=embedder)
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy}")


if __name__ == "__main__":
    from app.ingestion.parser import parse_directory
    from app.config import RESUME_DIR

    docs = parse_directory(RESUME_DIR, "resume")
    for doc in docs[:1]:
        chunks = chunk_document(doc, strategy="section")
        print(f"{doc.doc_id}: {len(chunks)} chunks")
        for c in chunks:
            print(f"  [{c.section}] {c.text[:80]}...")

"""
Extracts structured metadata from chunks: section type, date ranges, company names.
Runs after chunker.py, before embedding — metadata gets attached to each vector
for filtered retrieval (e.g. "only search 'experience' sections").
"""
import re
from dataclasses import asdict

from app.ingestion.chunker import Chunk

# Matches: "Jan 2022 - Present", "2020-2023", "06/2021 – 08/2022", "2019 to 2021"
DATE_RANGE_PATTERN = re.compile(
    r"""
    (?P<start>
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}
        | \d{1,2}/\d{4}
        | \d{4}
    )
    \s*(?:-|–|—|to)\s*
    (?P<end>
        Present|Current
        | (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}
        | \d{1,2}/\d{4}
        | \d{4}
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Heuristic: capitalized word sequences near a date range are often company names.
# Kept intentionally simple — swap for spaCy NER if precision matters more than speed.
COMPANY_LINE_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9&.,\- ]{1,60}$")

SECTION_TYPE_MAP = {
    "experience": "experience", "work experience": "experience", "professional experience": "experience",
    "education": "education",
    "skills": "skills", "technical skills": "skills",
    "projects": "projects",
    "certifications": "certifications",
    "summary": "summary", "objective": "summary",
    "achievements": "achievements",
    "requirements": "requirements", "qualifications": "requirements",
    "responsibilities": "responsibilities",
    "about the role": "overview", "about the job": "overview",
    "nice to have": "requirements",
}


def extract_date_ranges(text: str) -> list[str]:
    matches = DATE_RANGE_PATTERN.finditer(text)
    return [m.group(0).strip() for m in matches]


def extract_candidate_company_lines(text: str) -> list[str]:
    """Lines that look like a company/org name (capitalized, short, no sentence punctuation)."""
    candidates = []
    for line in text.split("\n"):
        line = line.strip()
        if 2 <= len(line) <= 60 and COMPANY_LINE_PATTERN.match(line) and not line.endswith("."):
            candidates.append(line)
    return candidates[:3]  # cap noise


def normalize_section_type(section_name: str) -> str:
    return SECTION_TYPE_MAP.get(section_name.strip().lower(), "other")


def enrich_chunk(chunk: Chunk) -> Chunk:
    """Attach extracted metadata to chunk.metadata in place, return chunk."""
    date_ranges = extract_date_ranges(chunk.text)
    companies = extract_candidate_company_lines(chunk.text) if chunk.doc_type == "resume" else []

    chunk.metadata.update({
        "section_type": normalize_section_type(chunk.section),
        "date_ranges": date_ranges,
        "has_dates": bool(date_ranges),
        "candidate_companies": companies,
        "char_count": len(chunk.text),
    })
    return chunk


def enrich_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [enrich_chunk(c) for c in chunks]


if __name__ == "__main__":
    from app.ingestion.parser import parse_directory
    from app.ingestion.chunker import chunk_document
    from app.config import RESUME_DIR

    docs = parse_directory(RESUME_DIR, "resume")
    for doc in docs[:1]:
        chunks = chunk_document(doc, strategy="section")
        chunks = enrich_chunks(chunks)
        for c in chunks:
            print(f"[{c.metadata['section_type']}] dates={c.metadata['date_ranges']} "
                  f"companies={c.metadata['candidate_companies']}")
            print(f"  {c.text[:80]}...")

"""Split generated Markdown documents into searchable chunks.

Chunks follow Markdown headings, so that a citation can point to a named section. Tokens are
estimated from word counts, because exact tokenizers download data when they first run.
"""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from config.settings import ChunkingLimits

FRONT_MATTER_PATTERN = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
HEADING_PATTERN = re.compile(r"^(#{1,6}) +(.+)$")


def build_text_with_context(title: str, section: str, text: str) -> str:
    """Prefix text with its title and section, so that it matches by topic as well.

    Embedding, BM25 and reranking all use this, so that they judge the same text.
    """
    return f"{title} > {section}\n{text}"


@dataclass(frozen=True)
class SourceDocument:
    """A document read from disk: its front-matter fields and its Markdown body."""

    doc_id: str
    title: str
    department: str
    document_type: str
    access_level: str
    created_date: date
    source_uri: str
    body: str


@dataclass(frozen=True)
class DocumentChunk:
    """One searchable piece of a document, with the metadata stored beside its vectors."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    department: str
    document_type: str
    access_level: str
    created_date: str
    created_ts: int
    source_uri: str
    text: str

    def text_for_embedding(self) -> str:
        """Return the text to encode, with its title and section."""
        return build_text_with_context(self.title, self.section, self.text)


@dataclass(frozen=True)
class _SectionPiece:
    """Part of a section, carrying the heading path it came from."""

    section_path: str
    words: list[str]


def parse_markdown_document(markdown_text: str) -> SourceDocument:
    """Separate a file's YAML front-matter from its body."""
    front_matter_match = FRONT_MATTER_PATTERN.match(markdown_text)
    if front_matter_match is None:
        raise ValueError("Document has no YAML front-matter.")

    front_matter = yaml.safe_load(front_matter_match.group(1))
    return SourceDocument(
        doc_id=front_matter["doc_id"],
        title=front_matter["title"],
        department=front_matter["department"],
        document_type=front_matter["document_type"],
        access_level=front_matter["access_level"],
        created_date=date.fromisoformat(front_matter["created_date"]),
        source_uri=front_matter["source_uri"],
        body=markdown_text[front_matter_match.end() :].strip(),
    )


def load_documents(documents_directory: Path) -> list[SourceDocument]:
    """Read every Markdown document, in a stable order so that chunk ids are repeatable."""
    document_paths = sorted(documents_directory.rglob("*.md"))
    return [
        parse_markdown_document(document_path.read_text(encoding="utf-8"))
        for document_path in document_paths
    ]


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """Return (heading path, text) pairs, such as ("Summary", "..."), for a Markdown body."""
    sections: list[tuple[str, str]] = []
    # Maps heading level to heading text, so that a heading path reflects nesting.
    open_headings: dict[int, str] = {}
    current_lines: list[str] = []

    def close_current_section() -> None:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            # The level 1 heading repeats the document title, so it is left out of the path.
            heading_path = " > ".join(
                heading for level, heading in sorted(open_headings.items()) if level > 1
            )
            sections.append((heading_path or "Introduction", section_text))
        current_lines.clear()

    for line in body.splitlines():
        heading_match = HEADING_PATTERN.match(line)
        if heading_match is None:
            current_lines.append(line)
            continue

        close_current_section()
        heading_level = len(heading_match.group(1))
        open_headings = {
            level: heading for level, heading in open_headings.items() if level < heading_level
        }
        open_headings[heading_level] = heading_match.group(2).strip()

    close_current_section()
    return sections


def _split_long_section(
    section_path: str, section_text: str, maximum_words: int
) -> list[_SectionPiece]:
    """Cut a section into pieces that each fit within the maximum."""
    words = section_text.split()
    return [
        _SectionPiece(section_path, words[start : start + maximum_words])
        for start in range(0, len(words), maximum_words)
    ]


def _can_join(
    previous_piece: _SectionPiece,
    piece: _SectionPiece,
    minimum_words: int,
    maximum_words: int,
) -> bool:
    """Return True if the previous piece is still too small and the pair would fit."""
    combined_word_count = len(previous_piece.words) + len(piece.words)
    return len(previous_piece.words) < minimum_words and combined_word_count <= maximum_words


def _merge_small_pieces(
    pieces: list[_SectionPiece], minimum_words: int, maximum_words: int
) -> list[_SectionPiece]:
    """Join neighbouring pieces until each reaches the minimum, without passing the maximum.

    A joined piece names both sections, for example "Customer impact; Root cause".
    """
    merged_pieces: list[_SectionPiece] = []
    for piece in pieces:
        previous_piece = merged_pieces[-1] if merged_pieces else None
        if previous_piece is None or not _can_join(
            previous_piece, piece, minimum_words, maximum_words
        ):
            merged_pieces.append(piece)
            continue

        joined_path = previous_piece.section_path
        if piece.section_path not in joined_path:
            joined_path = f"{joined_path}; {piece.section_path}"
        merged_pieces[-1] = _SectionPiece(joined_path, previous_piece.words + piece.words)
    return merged_pieces


def _to_utc_midnight_timestamp(day: date) -> int:
    """Return a date as epoch seconds, because Pinecone range filters need numbers."""
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def chunk_document(document: SourceDocument, limits: ChunkingLimits) -> list[DocumentChunk]:
    """Split one document into chunks that respect headings, size limits and overlap."""
    minimum_words = int(limits.min_tokens / limits.tokens_per_word)
    # Reserve room for the overlap, so that a chunk with overlap still fits the maximum.
    maximum_words = int(limits.max_tokens / limits.tokens_per_word / (1 + limits.overlap_fraction))
    overlap_words = int(maximum_words * limits.overlap_fraction)

    # Step 1: split on headings, then cut any section that is too long.
    pieces: list[_SectionPiece] = []
    for section_path, section_text in _split_into_sections(document.body):
        pieces.extend(_split_long_section(section_path, section_text, maximum_words))

    # Step 2: join small neighbours, so that a chunk holds a complete idea.
    merged_pieces = _merge_small_pieces(pieces, minimum_words, maximum_words)

    # Step 3: repeat the end of each chunk at the start of the next, then add metadata.
    created_ts = _to_utc_midnight_timestamp(document.created_date)
    chunks = []
    for chunk_number, piece in enumerate(merged_pieces):
        # Overlap crosses section boundaries on purpose, so that context carries over.
        overlap = merged_pieces[chunk_number - 1].words[-overlap_words:] if chunk_number else []
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.doc_id}#c{chunk_number}",
                doc_id=document.doc_id,
                title=document.title,
                section=piece.section_path,
                department=document.department,
                document_type=document.document_type,
                access_level=document.access_level,
                created_date=document.created_date.isoformat(),
                created_ts=created_ts,
                source_uri=document.source_uri,
                text=" ".join(overlap + piece.words),
            )
        )
    return chunks

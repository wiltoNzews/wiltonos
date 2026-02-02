#!/usr/bin/env python3
"""
Library Indexer - The Library of Alexandria
============================================
Indexes rag-local folder contents for semantic search.

This is the "Learned" layer - PDFs, docs, Replit gold, everything studied.
Separate from crystals (lived experience) but searchable alongside them.

Features:
- Scans for indexable file types (.md, .py, .js, .ts, .txt, .json, .pdf)
- Chunks large documents for better retrieval
- Generates embeddings via Ollama
- Stores in library table with source tracking
- Incremental indexing (skips already indexed files)

Usage:
    python library_indexer.py /path/to/rag-local
    python library_indexer.py --scan-only  # Just show what would be indexed
    python library_indexer.py --reindex    # Force reindex all

January 2026 - The library that remembers
"""

import sys
import sqlite3
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Generator, Tuple
from dataclasses import dataclass
import json
import numpy as np

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_service import MemoryService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
RAG_LOCAL = Path.home() / "rag-local"

# File types to index
INDEXABLE_EXTENSIONS = {
    '.md', '.txt', '.py', '.js', '.ts', '.tsx', '.jsx',
    '.json', '.html', '.css', '.sh', '.yml', '.yaml',
    '.docx', '.pdf'  # These need special handling
}

# Directories to skip
SKIP_DIRS = {
    '.git', '__pycache__', 'node_modules', '.venv', 'venv',
    '.pytest_cache', 'chroma_db', 'benchmark-results',
    'screenshots', 'grafana', 'nginx', '.husky', '.github'
}

# Max file size to index (5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024

# Chunk settings
CHUNK_SIZE = 2000  # characters
CHUNK_OVERLAP = 200


@dataclass
class Document:
    """A document to be indexed."""
    path: Path
    content: str
    file_type: str
    size: int
    modified: datetime
    content_hash: str

    @classmethod
    def from_file(cls, path: Path) -> Optional['Document']:
        """Load a document from file."""
        try:
            stat = path.stat()
            if stat.st_size > MAX_FILE_SIZE:
                logger.debug(f"Skipping {path.name}: too large ({stat.st_size} bytes)")
                return None

            # Read content based on file type
            if path.suffix == '.pdf':
                content = cls._read_pdf(path)
            elif path.suffix == '.docx':
                content = cls._read_docx(path)
            elif path.suffix == '.json':
                content = cls._read_json(path)
            else:
                content = path.read_text(encoding='utf-8', errors='ignore')

            if not content or len(content.strip()) < 50:
                return None

            content_hash = hashlib.md5(content.encode()).hexdigest()

            return cls(
                path=path,
                content=content,
                file_type=path.suffix.lower(),
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime),
                content_hash=content_hash
            )
        except Exception as e:
            logger.debug(f"Error reading {path}: {e}")
            return None

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Extract text from PDF."""
        try:
            import PyPDF2
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = []
                for page in reader.pages[:50]:  # Limit pages
                    text.append(page.extract_text() or '')
                return '\n'.join(text)
        except ImportError:
            logger.debug("PyPDF2 not installed, skipping PDF")
            return ""
        except Exception as e:
            logger.debug(f"Error reading PDF {path}: {e}")
            return ""

    @staticmethod
    def _read_docx(path: Path) -> str:
        """Extract text from DOCX."""
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(path)
            return '\n'.join(p.text for p in doc.paragraphs)
        except ImportError:
            logger.debug("python-docx not installed, skipping DOCX")
            return ""
        except Exception as e:
            logger.debug(f"Error reading DOCX {path}: {e}")
            return ""

    @staticmethod
    def _read_json(path: Path) -> str:
        """Read JSON as formatted text."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # For conversations, extract meaningful content
            if isinstance(data, list) and len(data) > 0 and 'mapping' in data[0]:
                # ChatGPT export - skip, handled by chatgpt_import
                return ""
            return json.dumps(data, indent=2, ensure_ascii=False)[:50000]
        except:
            return path.read_text(encoding='utf-8', errors='ignore')


@dataclass
class Chunk:
    """A chunk of a document."""
    doc_path: str
    content: str
    chunk_index: int
    total_chunks: int
    content_hash: str


class LibraryIndexer:
    """
    Index the rag-local library for semantic search.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.memory = MemoryService()
        self._ensure_table()
        self.stats = {
            'files_scanned': 0,
            'files_indexed': 0,
            'chunks_created': 0,
            'skipped_existing': 0,
            'errors': 0
        }

    def _ensure_table(self):
        """Create library table if it doesn't exist."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS library (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_type TEXT,
                chunk_index INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 1,
                content TEXT NOT NULL,
                content_hash TEXT,
                embedding BLOB,
                category TEXT,
                tags TEXT,
                created_at TEXT,
                indexed_at TEXT,
                UNIQUE(file_path, chunk_index)
            )
        """)

        # Index for faster lookups
        c.execute("CREATE INDEX IF NOT EXISTS idx_library_hash ON library(content_hash)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_library_path ON library(file_path)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_library_type ON library(file_type)")

        conn.commit()
        conn.close()

    def scan_directory(self, root: Path) -> Generator[Path, None, None]:
        """Scan directory for indexable files."""
        for path in root.rglob('*'):
            if path.is_file():
                # Skip if in excluded directory
                if any(skip in path.parts for skip in SKIP_DIRS):
                    continue

                # Check extension
                if path.suffix.lower() in INDEXABLE_EXTENSIONS:
                    self.stats['files_scanned'] += 1
                    yield path

    def is_indexed(self, content_hash: str) -> bool:
        """Check if content already indexed."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM library WHERE content_hash = ?", (content_hash,))
        exists = c.fetchone()[0] > 0
        conn.close()
        return exists

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split document into chunks for indexing."""
        content = doc.content
        chunks = []

        if len(content) <= CHUNK_SIZE:
            # Single chunk
            chunks.append(Chunk(
                doc_path=str(doc.path),
                content=content,
                chunk_index=0,
                total_chunks=1,
                content_hash=doc.content_hash
            ))
        else:
            # Multiple chunks with overlap
            start = 0
            chunk_idx = 0
            while start < len(content):
                end = start + CHUNK_SIZE
                chunk_content = content[start:end]

                # Try to break at sentence/paragraph boundary
                if end < len(content):
                    # Look for natural break points
                    for sep in ['\n\n', '\n', '. ', '! ', '? ']:
                        last_sep = chunk_content.rfind(sep)
                        if last_sep > CHUNK_SIZE // 2:
                            chunk_content = chunk_content[:last_sep + len(sep)]
                            end = start + len(chunk_content)
                            break

                chunk_hash = hashlib.md5(
                    f"{doc.content_hash}:{chunk_idx}".encode()
                ).hexdigest()

                chunks.append(Chunk(
                    doc_path=str(doc.path),
                    content=chunk_content.strip(),
                    chunk_index=chunk_idx,
                    total_chunks=-1,  # Will update after
                    content_hash=chunk_hash
                ))

                chunk_idx += 1
                start = end - CHUNK_OVERLAP

            # Update total chunks
            for chunk in chunks:
                chunk.total_chunks = len(chunks)

        return chunks

    def detect_category(self, doc: Document) -> str:
        """Detect document category from path and content."""
        path_str = str(doc.path).lower()

        if 'passiveworks' in path_str or 'replit' in path_str:
            if 'tecnologias' in path_str:
                return 'replit_gold:tecnologias'
            elif 'o4_projects' in path_str:
                return 'replit_gold:o4'
            elif 'codex' in path_str:
                return 'replit_gold:codex'
            return 'replit_gold'
        elif 'pdfs' in path_str:
            return 'documentation'
        elif 'docs' in path_str:
            return 'artifacts'
        elif 'chatgpt' in path_str:
            return 'exports'
        else:
            return 'general'

    def detect_tags(self, doc: Document) -> str:
        """Extract tags from document content."""
        content_lower = doc.content.lower()
        tags = []

        # Technical tags
        tag_patterns = {
            'coherence': ['coherence', 'coerência', 'zλ', 'z-law'],
            'glyph': ['glyph', 'ψ', 'psi', 'omega', 'Ω'],
            'breath': ['breath', 'respiração', 'breathing'],
            'consciousness': ['consciousness', 'consciência', 'awareness'],
            'quantum': ['quantum', 'quântico', 'superposition'],
            'mirror': ['mirror', 'espelho', 'reflection'],
            'crystal': ['crystal', 'cristal'],
            'protocol': ['protocol', 'protocolo'],
            'api': ['api', 'endpoint', 'route'],
            'database': ['database', 'sqlite', 'sql'],
            'frontend': ['react', 'vue', 'html', 'css'],
            'backend': ['python', 'node', 'server'],
        }

        for tag, patterns in tag_patterns.items():
            if any(p in content_lower for p in patterns):
                tags.append(tag)

        return ','.join(tags[:10])  # Limit tags

    def index_document(self, doc: Document, embed: bool = True) -> int:
        """Index a document (all chunks)."""
        chunks = self.chunk_document(doc)
        category = self.detect_category(doc)
        tags = self.detect_tags(doc)

        indexed = 0
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        for chunk in chunks:
            try:
                # Generate embedding
                embedding = None
                if embed:
                    emb_list = self.memory.get_embedding(chunk.content[:8000])
                    embedding = np.array(emb_list, dtype=np.float32).tobytes()

                c.execute("""
                    INSERT OR REPLACE INTO library (
                        file_path, file_type, chunk_index, total_chunks,
                        content, content_hash, embedding,
                        category, tags, created_at, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.doc_path,
                    doc.file_type,
                    chunk.chunk_index,
                    chunk.total_chunks,
                    chunk.content,
                    chunk.content_hash,
                    embedding,
                    category,
                    tags,
                    doc.modified.isoformat(),
                    datetime.now().isoformat()
                ))

                indexed += 1
                self.stats['chunks_created'] += 1

            except Exception as e:
                logger.error(f"Error indexing chunk {chunk.chunk_index} of {doc.path}: {e}")
                self.stats['errors'] += 1

        conn.commit()
        conn.close()

        if indexed > 0:
            self.stats['files_indexed'] += 1

        return indexed

    def index_directory(
        self,
        root: Path,
        embed: bool = True,
        reindex: bool = False,
        limit: int = None
    ) -> Dict:
        """Index all documents in a directory."""
        logger.info(f"Indexing {root}")
        logger.info(f"Embed: {embed}, Reindex: {reindex}")

        count = 0
        for file_path in self.scan_directory(root):
            if limit and count >= limit:
                break

            doc = Document.from_file(file_path)
            if not doc:
                continue

            # Check if already indexed
            if not reindex and self.is_indexed(doc.content_hash):
                self.stats['skipped_existing'] += 1
                continue

            # Index it
            rel_path = file_path.relative_to(root) if root in file_path.parents else file_path.name
            logger.info(f"Indexing: {rel_path} ({len(doc.content)} chars)")

            chunks = self.index_document(doc, embed)
            count += 1

            if count % 50 == 0:
                logger.info(f"Progress: {count} files, {self.stats['chunks_created']} chunks")

        return self.stats

    def search(
        self,
        query: str,
        limit: int = 10,
        category: str = None,
        file_type: str = None,
        witnessed_only: bool = False
    ) -> List[Dict]:
        """Search the library semantically."""
        # Get query embedding
        query_emb = np.array(self.memory.get_embedding(query), dtype=np.float32)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Build query
        conditions = ["embedding IS NOT NULL"]
        params = []

        if category:
            conditions.append("category LIKE ?")
            params.append(f"%{category}%")

        if file_type:
            conditions.append("file_type = ?")
            params.append(file_type)

        if witnessed_only:
            conditions.append("witnessed_at IS NOT NULL AND learning IS NOT NULL AND learning != ''")

        where_clause = " AND ".join(conditions)

        c.execute(f"""
            SELECT id, file_path, file_type, chunk_index, content,
                   category, tags, embedding,
                   domain, significance, concepts_introduced, learning
            FROM library
            WHERE {where_clause}
        """, params)

        # Calculate similarities
        results = []
        for row in c.fetchall():
            try:
                emb = np.frombuffer(row['embedding'], dtype=np.float32)
                similarity = np.dot(query_emb, emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(emb)
                )

                results.append({
                    'id': row['id'],
                    'file_path': row['file_path'],
                    'file_type': row['file_type'],
                    'chunk_index': row['chunk_index'],
                    'content': row['content'][:500],
                    'category': row['category'],
                    'tags': row['tags'],
                    'similarity': float(similarity),
                    'domain': row['domain'],
                    'significance': row['significance'],
                    'concepts': row['concepts_introduced'],
                    'learning': row['learning']
                })
            except:
                continue

        conn.close()

        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    def get_stats(self) -> Dict:
        """Get library statistics."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        stats = {}

        c.execute("SELECT COUNT(*) FROM library")
        stats['total_chunks'] = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT file_path) FROM library")
        stats['total_files'] = c.fetchone()[0]

        c.execute("SELECT category, COUNT(*) FROM library GROUP BY category")
        stats['by_category'] = dict(c.fetchall())

        c.execute("SELECT file_type, COUNT(*) FROM library GROUP BY file_type ORDER BY COUNT(*) DESC")
        stats['by_type'] = dict(c.fetchall())

        conn.close()
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Index rag-local library for semantic search"
    )

    parser.add_argument("path", type=Path, nargs="?", default=RAG_LOCAL,
                        help="Path to index (default: ~/rag-local)")
    parser.add_argument("--scan-only", action="store_true",
                        help="Just scan and report, don't index")
    parser.add_argument("--no-embed", action="store_true",
                        help="Skip embedding generation")
    parser.add_argument("--reindex", action="store_true",
                        help="Force reindex all files")
    parser.add_argument("--limit", type=int,
                        help="Limit number of files to index")
    parser.add_argument("--search", type=str,
                        help="Search the library")
    parser.add_argument("--stats", action="store_true",
                        help="Show library statistics")

    args = parser.parse_args()

    indexer = LibraryIndexer()

    if args.stats:
        stats = indexer.get_stats()
        print("\n" + "=" * 60)
        print("LIBRARY STATISTICS")
        print("=" * 60)
        print(f"Total files:  {stats['total_files']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print("\nBy category:")
        for cat, count in stats['by_category'].items():
            print(f"  {cat}: {count}")
        print("\nBy type:")
        for ftype, count in list(stats['by_type'].items())[:10]:
            print(f"  {ftype}: {count}")
        return

    if args.search:
        results = indexer.search(args.search, limit=10)
        print(f"\nSearch: '{args.search}'")
        print("=" * 60)
        for r in results:
            print(f"\n[{r['similarity']:.3f}] {r['file_path']}")
            print(f"  Category: {r['category']} | Tags: {r['tags']}")
            print(f"  {r['content'][:200]}...")
        return

    if args.scan_only:
        print(f"\nScanning {args.path}...")
        count = 0
        for f in indexer.scan_directory(args.path):
            rel = f.relative_to(args.path) if args.path in f.parents else f.name
            print(f"  {rel}")
            count += 1
            if count >= 100:
                print(f"  ... and more")
                break
        print(f"\nTotal indexable files: {indexer.stats['files_scanned']}")
        return

    # Run indexing
    stats = indexer.index_directory(
        args.path,
        embed=not args.no_embed,
        reindex=args.reindex,
        limit=args.limit
    )

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"Files scanned:    {stats['files_scanned']}")
    print(f"Files indexed:    {stats['files_indexed']}")
    print(f"Chunks created:   {stats['chunks_created']}")
    print(f"Skipped existing: {stats['skipped_existing']}")
    print(f"Errors:           {stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

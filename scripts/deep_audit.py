#!/usr/bin/env python3
"""
WiltonOS Deep Audit Script
===========================
Systematically explores the entire PassiveWorks folder and extracts:
- All Python modules with key functions/classes
- All TypeScript modules with interfaces/classes
- All JSON configs with structure
- All MD documentation with topics
- Identity data for Wilton/Michelle/Renan/Juliana
- Algorithms worth porting
- Everything.

Run: python3 scripts/deep_audit.py
Output: docs/deep_audit_complete.md
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# Paths
PASSIVEWORKS = Path("/home/zews/rag-local/WiltonOS-PassiveWorks")
OUTPUT_FILE = Path("/home/zews/wiltonos/docs/deep_audit_complete.md")
DB_FILE = Path("/home/zews/wiltonos/data/audit_index.db")

# Stats
stats = {
    'files_scanned': 0,
    'python_modules': 0,
    'typescript_modules': 0,
    'json_files': 0,
    'md_files': 0,
    'classes_found': 0,
    'functions_found': 0,
    'interfaces_found': 0,
    'identity_mentions': defaultdict(list),
    'algorithms': [],
    'port_candidates': [],
    'by_category': defaultdict(list),
}

# Patterns to extract
PYTHON_CLASS = re.compile(r'^class\s+(\w+)(?:\(([^)]*)\))?:', re.MULTILINE)
PYTHON_FUNC = re.compile(r'^(?:async\s+)?def\s+(\w+)\s*\(', re.MULTILINE)
PYTHON_DOCSTRING = re.compile(r'"""([^"]*?)"""', re.DOTALL)
TS_CLASS = re.compile(r'(?:export\s+)?class\s+(\w+)', re.MULTILINE)
TS_INTERFACE = re.compile(r'(?:export\s+)?interface\s+(\w+)', re.MULTILINE)
TS_FUNCTION = re.compile(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)', re.MULTILINE)

# Identity keywords
IDENTITY_KEYWORDS = ['wilton', 'michelle', 'renan', 'juliana', 'nisa', 'pai', 'father', 'mae', 'mother']

# Algorithm keywords (things worth porting)
ALGORITHM_KEYWORDS = [
    'coherence', 'zeta', 'lambda', 'phi', 'breathing', 'breath',
    'attractor', 'kuramoto', 'entropy', 'resonance', 'embedding',
    'similarity', 'cosine', 'vector', 'semantic', 'glyph',
    'quantum', 'field', 'oscillation', 'daemon', 'router',
    'trigger', 'ritual', 'memory', 'session', 'identity'
]

def init_db():
    """Initialize SQLite database for indexing."""
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY,
        path TEXT UNIQUE,
        filename TEXT,
        extension TEXT,
        size INTEGER,
        lines INTEGER,
        category TEXT,
        classes TEXT,
        functions TEXT,
        summary TEXT,
        port_priority TEXT,
        scanned_at TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS identity_mentions (
        id INTEGER PRIMARY KEY,
        file_path TEXT,
        person TEXT,
        context TEXT,
        line_number INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS algorithms (
        id INTEGER PRIMARY KEY,
        file_path TEXT,
        name TEXT,
        description TEXT,
        port_priority TEXT
    )''')
    conn.commit()
    return conn

def categorize_file(path: Path) -> str:
    """Categorize file by its path."""
    path_str = str(path).lower()
    if 'memory' in path_str:
        return 'memory'
    if 'coherence' in path_str:
        return 'coherence'
    if 'breath' in path_str:
        return 'breathing'
    if 'router' in path_str or 'routing' in path_str:
        return 'routing'
    if 'identity' in path_str or 'profile' in path_str:
        return 'identity'
    if 'market' in path_str or 'finance' in path_str:
        return 'market'
    if 'ritual' in path_str:
        return 'ritual'
    if 'glyph' in path_str or 'geometry' in path_str:
        return 'sacred_geometry'
    if 'daemon' in path_str:
        return 'daemon'
    if 'bridge' in path_str:
        return 'bridge'
    if 'voice' in path_str or 'whisper' in path_str:
        return 'voice'
    if 'agent' in path_str:
        return 'agent'
    if 'test' in path_str:
        return 'test'
    if 'example' in path_str:
        return 'example'
    return 'other'

def extract_python_info(content: str, path: Path) -> Dict:
    """Extract info from Python file."""
    classes = PYTHON_CLASS.findall(content)
    functions = [f for f in PYTHON_FUNC.findall(content) if not f.startswith('_')]
    docstrings = PYTHON_DOCSTRING.findall(content)

    # Get first docstring as summary
    summary = docstrings[0][:200].strip() if docstrings else ""

    # Check for algorithm patterns
    algorithms = []
    content_lower = content.lower()
    for keyword in ALGORITHM_KEYWORDS:
        if keyword in content_lower:
            algorithms.append(keyword)

    # Determine port priority
    priority = 'low'
    if len(algorithms) >= 3:
        priority = 'high'
    elif len(algorithms) >= 1:
        priority = 'medium'
    if len(content.split('\n')) > 500:
        priority = 'high'

    return {
        'classes': [c[0] for c in classes],
        'functions': functions[:20],  # Limit
        'summary': summary,
        'algorithms': algorithms,
        'port_priority': priority,
        'lines': len(content.split('\n'))
    }

def extract_typescript_info(content: str, path: Path) -> Dict:
    """Extract info from TypeScript file."""
    classes = TS_CLASS.findall(content)
    interfaces = TS_INTERFACE.findall(content)
    functions = TS_FUNCTION.findall(content)

    # Get summary from first comment
    comment_match = re.search(r'/\*\*([^*]*(?:\*(?!/)[^*]*)*)\*/', content)
    summary = comment_match.group(1)[:200].strip() if comment_match else ""

    # Check for algorithm patterns
    algorithms = []
    content_lower = content.lower()
    for keyword in ALGORITHM_KEYWORDS:
        if keyword in content_lower:
            algorithms.append(keyword)

    priority = 'low'
    if len(algorithms) >= 3:
        priority = 'high'
    elif len(algorithms) >= 1:
        priority = 'medium'

    return {
        'classes': classes,
        'interfaces': interfaces,
        'functions': functions[:20],
        'summary': summary,
        'algorithms': algorithms,
        'port_priority': priority,
        'lines': len(content.split('\n'))
    }

def extract_json_info(content: str, path: Path) -> Dict:
    """Extract info from JSON file."""
    try:
        data = json.loads(content)
        keys = list(data.keys()) if isinstance(data, dict) else []

        # Check for identity data
        identity_found = []
        content_lower = content.lower()
        for person in IDENTITY_KEYWORDS:
            if person in content_lower:
                identity_found.append(person)

        return {
            'keys': keys[:20],
            'identity_mentions': identity_found,
            'item_count': len(data) if isinstance(data, (list, dict)) else 1,
            'summary': f"JSON with {len(keys)} keys" if keys else "JSON array/value"
        }
    except:
        return {'keys': [], 'summary': 'Invalid JSON'}

def extract_md_info(content: str, path: Path) -> Dict:
    """Extract info from Markdown file."""
    # Get title
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else path.stem

    # Get headers
    headers = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)

    # Check for identity mentions
    identity_found = []
    content_lower = content.lower()
    for person in IDENTITY_KEYWORDS:
        if person in content_lower:
            identity_found.append(person)

    # Check for code blocks
    code_blocks = len(re.findall(r'```', content)) // 2

    return {
        'title': title,
        'headers': headers[:10],
        'identity_mentions': identity_found,
        'code_blocks': code_blocks,
        'summary': title
    }

def find_identity_context(content: str, path: Path):
    """Find identity mentions with context."""
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for person in IDENTITY_KEYWORDS:
            if person in line_lower:
                context = line[:200]
                stats['identity_mentions'][person].append({
                    'file': str(path),
                    'line': i + 1,
                    'context': context
                })

def scan_file(path: Path, conn: sqlite3.Connection) -> Dict:
    """Scan a single file and extract info."""
    stats['files_scanned'] += 1

    try:
        content = path.read_text(encoding='utf-8', errors='ignore')
    except:
        return None

    ext = path.suffix.lower()
    category = categorize_file(path)

    info = {
        'path': str(path),
        'filename': path.name,
        'extension': ext,
        'size': path.stat().st_size,
        'category': category
    }

    if ext == '.py':
        stats['python_modules'] += 1
        py_info = extract_python_info(content, path)
        info.update(py_info)
        stats['classes_found'] += len(py_info['classes'])
        stats['functions_found'] += len(py_info['functions'])

    elif ext in ['.ts', '.tsx', '.js']:
        stats['typescript_modules'] += 1
        ts_info = extract_typescript_info(content, path)
        info.update(ts_info)
        stats['interfaces_found'] += len(ts_info.get('interfaces', []))
        stats['classes_found'] += len(ts_info['classes'])

    elif ext == '.json':
        stats['json_files'] += 1
        json_info = extract_json_info(content, path)
        info.update(json_info)

    elif ext == '.md':
        stats['md_files'] += 1
        md_info = extract_md_info(content, path)
        info.update(md_info)
    else:
        return None

    # Find identity mentions
    find_identity_context(content, path)

    # Store in database
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO files
                 (path, filename, extension, size, lines, category, classes, functions, summary, port_priority, scanned_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (str(path), path.name, ext, info['size'], info.get('lines', 0),
               category, json.dumps(info.get('classes', [])),
               json.dumps(info.get('functions', [])),
               info.get('summary', ''), info.get('port_priority', 'low'),
               datetime.now().isoformat()))
    conn.commit()

    # Track by category
    stats['by_category'][category].append(info)

    # Track port candidates
    if info.get('port_priority') == 'high':
        stats['port_candidates'].append(info)

    return info

def generate_report(conn: sqlite3.Connection):
    """Generate comprehensive markdown report."""
    c = conn.cursor()

    report = []
    report.append("# WiltonOS PassiveWorks Complete Audit")
    report.append(f"\n*Generated: {datetime.now().isoformat()}*\n")
    report.append(f"*Source: {PASSIVEWORKS}*\n")

    # Summary stats
    report.append("\n## Summary Statistics\n")
    report.append(f"| Metric | Count |")
    report.append(f"|--------|-------|")
    report.append(f"| Files Scanned | {stats['files_scanned']} |")
    report.append(f"| Python Modules | {stats['python_modules']} |")
    report.append(f"| TypeScript Modules | {stats['typescript_modules']} |")
    report.append(f"| JSON Files | {stats['json_files']} |")
    report.append(f"| Markdown Files | {stats['md_files']} |")
    report.append(f"| Classes Found | {stats['classes_found']} |")
    report.append(f"| Functions Found | {stats['functions_found']} |")
    report.append(f"| Interfaces Found | {stats['interfaces_found']} |")
    report.append(f"| High-Priority Port Candidates | {len(stats['port_candidates'])} |")

    # By category
    report.append("\n## Files by Category\n")
    for category, files in sorted(stats['by_category'].items()):
        report.append(f"\n### {category.upper()} ({len(files)} files)\n")
        # Show top 10 by size
        sorted_files = sorted(files, key=lambda x: x.get('lines', 0), reverse=True)[:10]
        for f in sorted_files:
            classes = f.get('classes', [])
            funcs = f.get('functions', [])
            priority = f.get('port_priority', 'low')
            lines = f.get('lines', 0)
            report.append(f"- **{f['filename']}** ({lines} lines, priority: {priority})")
            if classes:
                report.append(f"  - Classes: {', '.join(classes[:5])}")
            if funcs:
                report.append(f"  - Functions: {', '.join(funcs[:5])}")

    # High priority port candidates
    report.append("\n## HIGH PRIORITY PORT CANDIDATES\n")
    report.append("*These files have 3+ algorithm keywords and significant code*\n")
    sorted_candidates = sorted(stats['port_candidates'], key=lambda x: x.get('lines', 0), reverse=True)
    for i, f in enumerate(sorted_candidates[:50], 1):
        algorithms = f.get('algorithms', [])
        report.append(f"\n### {i}. {f['filename']} ({f.get('lines', 0)} lines)")
        report.append(f"- **Path**: `{f['path']}`")
        report.append(f"- **Algorithms**: {', '.join(algorithms)}")
        if f.get('classes'):
            report.append(f"- **Classes**: {', '.join(f['classes'][:10])}")
        if f.get('summary'):
            report.append(f"- **Summary**: {f['summary'][:200]}")

    # Identity data
    report.append("\n## IDENTITY DATA DISCOVERED\n")
    for person, mentions in sorted(stats['identity_mentions'].items()):
        if mentions:
            report.append(f"\n### {person.upper()} ({len(mentions)} mentions)\n")
            # Show unique files
            unique_files = list(set(m['file'] for m in mentions))[:20]
            for f in unique_files:
                report.append(f"- `{f}`")

    # All Python files with classes
    report.append("\n## COMPLETE PYTHON MODULE INDEX\n")
    c.execute("SELECT path, filename, lines, classes, functions, summary, port_priority FROM files WHERE extension = '.py' ORDER BY lines DESC")
    for row in c.fetchall():
        path, filename, lines, classes_json, funcs_json, summary, priority = row
        classes = json.loads(classes_json) if classes_json else []
        if classes or lines > 100:
            report.append(f"\n### {filename} ({lines} lines) [{priority}]")
            report.append(f"- Path: `{path}`")
            if classes:
                report.append(f"- Classes: {', '.join(classes)}")
            if summary:
                report.append(f"- {summary[:150]}")

    # All TypeScript files with interfaces
    report.append("\n## COMPLETE TYPESCRIPT MODULE INDEX\n")
    c.execute("SELECT path, filename, lines, classes, summary, port_priority FROM files WHERE extension IN ('.ts', '.tsx') ORDER BY lines DESC")
    for row in c.fetchall():
        path, filename, lines, classes_json, summary, priority = row
        classes = json.loads(classes_json) if classes_json else []
        if classes or lines > 100:
            report.append(f"\n### {filename} ({lines} lines) [{priority}]")
            report.append(f"- Path: `{path}`")
            if classes:
                report.append(f"- Classes/Interfaces: {', '.join(classes)}")

    return "\n".join(report)

def main():
    print("=" * 60)
    print("  WiltonOS Deep Audit - Complete Exploration")
    print("=" * 60)
    print(f"\nScanning: {PASSIVEWORKS}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Database: {DB_FILE}\n")

    # Initialize database
    conn = init_db()

    # Walk all files
    extensions = {'.py', '.ts', '.tsx', '.js', '.json', '.md'}

    for root, dirs, files in os.walk(PASSIVEWORKS):
        # Skip node_modules and similar
        dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__', '.venv', 'venv']]

        for filename in files:
            path = Path(root) / filename
            if path.suffix.lower() in extensions:
                info = scan_file(path, conn)
                if info and stats['files_scanned'] % 100 == 0:
                    print(f"  Scanned {stats['files_scanned']} files...")

    print(f"\n✓ Scanned {stats['files_scanned']} files")
    print(f"  - {stats['python_modules']} Python modules")
    print(f"  - {stats['typescript_modules']} TypeScript modules")
    print(f"  - {stats['json_files']} JSON files")
    print(f"  - {stats['md_files']} Markdown files")
    print(f"  - {len(stats['port_candidates'])} high-priority port candidates")

    # Generate report
    print("\nGenerating comprehensive report...")
    report = generate_report(conn)

    # Write report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report)
    print(f"✓ Report written to {OUTPUT_FILE}")

    # Print identity summary
    print("\nIdentity data found:")
    for person, mentions in sorted(stats['identity_mentions'].items()):
        if mentions:
            print(f"  - {person}: {len(mentions)} mentions in {len(set(m['file'] for m in mentions))} files")

    conn.close()
    print("\n✓ Audit complete!")

if __name__ == "__main__":
    main()

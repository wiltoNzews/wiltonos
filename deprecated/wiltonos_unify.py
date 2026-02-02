#!/usr/bin/env python3
"""
WiltonOS Database Unifier
Clean data, unite databases, prepare for proper analysis.

Usage:
    python wiltonos_unify.py status      # Show status of both DBs
    python wiltonos_unify.py clean       # Flag short crystals
    python wiltonos_unify.py unify       # Create unified database
    python wiltonos_unify.py analyze     # Run test analysis (500 crystals)
"""
import sqlite3
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

DB_MAIN = Path.home() / "crystals.db"
DB_CHATGPT = Path.home() / "crystals_chatgpt.db"
DB_UNIFIED = Path.home() / "crystals_unified.db"

MIN_CONTENT_LENGTH = 50


def get_db_stats(db_path: Path) -> dict:
    """Get statistics for a database."""
    if not db_path.exists():
        return {"error": "not found"}

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM crystals")
    total = c.fetchone()[0]

    c.execute(f"SELECT COUNT(*) FROM crystals WHERE LENGTH(content) < {MIN_CONTENT_LENGTH}")
    short = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM crystals WHERE glyph_primary IS NOT NULL AND glyph_primary != ''")
    analyzed = c.fetchone()[0]

    # Check for source column
    c.execute("PRAGMA table_info(crystals)")
    columns = {row[1] for row in c.fetchall()}

    conn.close()

    return {
        "total": total,
        "short": short,
        "analyzed": analyzed,
        "valid": total - short,
        "columns": columns
    }


def show_status():
    """Show status of all databases."""
    print("\n" + "=" * 60)
    print("DATABASE STATUS")
    print("=" * 60)

    for db_path in [DB_MAIN, DB_CHATGPT, DB_UNIFIED]:
        print(f"\n{db_path.name}:")
        if not db_path.exists():
            print("  Not found")
            continue

        stats = get_db_stats(db_path)
        print(f"  Total crystals: {stats['total']:,}")
        print(f"  Short (<{MIN_CONTENT_LENGTH} chars): {stats['short']:,}")
        print(f"  Valid (>={MIN_CONTENT_LENGTH} chars): {stats['valid']:,}")
        print(f"  Analyzed: {stats['analyzed']:,}")


def clean_databases():
    """Flag short crystals in both databases."""
    print("\nCleaning databases...")

    for db_path in [DB_MAIN, DB_CHATGPT]:
        if not db_path.exists():
            continue

        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()

        # Add is_valid column if not exists
        try:
            c.execute("ALTER TABLE crystals ADD COLUMN is_valid INTEGER DEFAULT 1")
        except:
            pass

        # Flag short content as invalid
        c.execute(f"""
            UPDATE crystals
            SET is_valid = 0
            WHERE LENGTH(content) < {MIN_CONTENT_LENGTH}
        """)
        flagged = c.rowcount

        conn.commit()
        conn.close()

        print(f"  {db_path.name}: Flagged {flagged} short crystals")


def unify_databases():
    """Create unified database from both sources."""
    print("\nUnifying databases...")

    # Backup existing unified if exists
    if DB_UNIFIED.exists():
        backup = DB_UNIFIED.with_suffix('.db.bak')
        DB_UNIFIED.rename(backup)
        print(f"  Backed up existing to {backup.name}")

    # Get schema from main db
    conn_main = sqlite3.connect(str(DB_MAIN))
    c = conn_main.cursor()
    c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='crystals'")
    schema = c.fetchone()[0]
    conn_main.close()

    # Create unified
    conn_unified = sqlite3.connect(str(DB_UNIFIED))
    c = conn_unified.cursor()
    c.execute(schema)

    # Add is_valid if not in schema
    try:
        c.execute("ALTER TABLE crystals ADD COLUMN is_valid INTEGER DEFAULT 1")
    except:
        pass

    # Add source_db to track origin
    try:
        c.execute("ALTER TABLE crystals ADD COLUMN source_db TEXT")
    except:
        pass

    conn_unified.commit()

    # Track content hashes for dedup
    seen_hashes = set()
    stats = {"main": 0, "chatgpt": 0, "duplicates": 0, "short": 0}

    # Import from main DB
    print("  Importing from crystals.db...")
    conn_main = sqlite3.connect(str(DB_MAIN))
    conn_main.row_factory = sqlite3.Row
    c_main = conn_main.cursor()

    c_main.execute("SELECT * FROM crystals WHERE LENGTH(content) >= ?", (MIN_CONTENT_LENGTH,))
    for row in c_main.fetchall():
        content = row['content']
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash in seen_hashes:
            stats["duplicates"] += 1
            continue

        seen_hashes.add(content_hash)

        # Insert into unified
        cols = list(row.keys())
        vals = [row[c] for c in cols]

        # Update source_db
        if 'source_db' in cols:
            vals[cols.index('source_db')] = 'main'
        else:
            cols.append('source_db')
            vals.append('main')

        # Update content_hash
        if 'content_hash' in cols:
            vals[cols.index('content_hash')] = content_hash

        placeholders = ','.join(['?' for _ in cols])
        col_names = ','.join(cols)

        try:
            c.execute(f"INSERT INTO crystals ({col_names}) VALUES ({placeholders})", vals)
            stats["main"] += 1
        except sqlite3.IntegrityError:
            stats["duplicates"] += 1
        except Exception as e:
            # Skip columns that don't exist
            pass

    conn_main.close()

    # Import from ChatGPT DB
    print("  Importing from crystals_chatgpt.db...")
    if DB_CHATGPT.exists():
        conn_chat = sqlite3.connect(str(DB_CHATGPT))
        conn_chat.row_factory = sqlite3.Row
        c_chat = conn_chat.cursor()

        c_chat.execute("SELECT * FROM crystals WHERE LENGTH(content) >= ?", (MIN_CONTENT_LENGTH,))
        for row in c_chat.fetchall():
            content = row['content']
            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash in seen_hashes:
                stats["duplicates"] += 1
                continue

            seen_hashes.add(content_hash)

            # Build insert for only existing columns
            data = dict(row)
            data['source_db'] = 'chatgpt'
            data['content_hash'] = content_hash
            data['is_valid'] = 1

            # Get columns in unified
            c.execute("PRAGMA table_info(crystals)")
            unified_cols = {r[1] for r in c.fetchall()}

            # Filter to only unified columns
            insert_data = {k: v for k, v in data.items() if k in unified_cols and k != 'id'}

            cols = list(insert_data.keys())
            vals = list(insert_data.values())
            placeholders = ','.join(['?' for _ in cols])
            col_names = ','.join(cols)

            try:
                c.execute(f"INSERT INTO crystals ({col_names}) VALUES ({placeholders})", vals)
                stats["chatgpt"] += 1
            except sqlite3.IntegrityError:
                stats["duplicates"] += 1
            except Exception as e:
                pass

        conn_chat.close()

    conn_unified.commit()
    conn_unified.close()

    print(f"\n  Results:")
    print(f"    From crystals.db: {stats['main']:,}")
    print(f"    From crystals_chatgpt.db: {stats['chatgpt']:,}")
    print(f"    Duplicates skipped: {stats['duplicates']:,}")
    print(f"    Total in unified: {stats['main'] + stats['chatgpt']:,}")


def run_test_analysis(limit: int = 500):
    """Run analysis on first N valid crystals."""
    import subprocess

    if not DB_UNIFIED.exists():
        print("\nUnified database not found. Run 'unify' first.")
        return

    print(f"\nRunning test analysis on {limit} crystals...")

    # Use the complete analyzer with unified db
    cmd = [
        "python3", "-u",
        str(Path.home() / "wiltonos_analyze_complete.py"),
        "analyze",
        "--db", "crystals_unified.db",
        "--limit", str(limit)
    ]

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="WiltonOS Database Unifier")
    parser.add_argument("command", choices=["status", "clean", "unify", "analyze", "all"],
                       help="Command to run")
    parser.add_argument("--limit", type=int, default=500, help="Crystals to analyze in test")

    args = parser.parse_args()

    if args.command == "status":
        show_status()
    elif args.command == "clean":
        clean_databases()
        show_status()
    elif args.command == "unify":
        clean_databases()
        unify_databases()
        show_status()
    elif args.command == "analyze":
        run_test_analysis(args.limit)
    elif args.command == "all":
        clean_databases()
        unify_databases()
        show_status()
        run_test_analysis(args.limit)


if __name__ == "__main__":
    main()

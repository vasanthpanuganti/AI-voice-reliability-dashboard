"""
Import custom dataset for AI Pipeline Resilience Dashboard.

This script allows you to import your own query data from CSV or JSON files.
The data will be processed and embeddings will be auto-generated if not provided.

Required fields:
    - query: The text of the query (required)
    - timestamp: When the query occurred (required, ISO 8601 format preferred)

Optional fields:
    - query_category: Category of the query (e.g., appointment, prescription, billing)
    - confidence_score: Confidence score of the AI response (0.0 to 1.0)
    - ai_response: The AI's response to the query
    - embedding: Pre-computed embedding vector (JSON array). If not provided, will be auto-generated.

Usage:
    python scripts/import_custom_data.py --file data/my_queries.csv --format csv
    python scripts/import_custom_data.py --file data/my_queries.json --format json --clear-existing
"""
import sys
import argparse
import csv
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import SessionLocal, init_db
from backend.models.query_log import QueryLog

def generate_simple_embedding(query: str, dimension: int = 384) -> list:
    """
    Generate a deterministic embedding based on query text.
    Uses hash-based approach for consistency without external models.
    """
    np.random.seed(hash(query) % (2**32))
    embedding = np.random.randn(dimension).astype(float)
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp from various formats"""
    # Try ISO 8601 format first
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str.strip(), fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse timestamp: {timestamp_str}")

def parse_embedding(embedding_str: str) -> list:
    """Parse embedding from JSON string"""
    try:
        if isinstance(embedding_str, str):
            return json.loads(embedding_str)
        return embedding_str
    except (json.JSONDecodeError, TypeError):
        raise ValueError(f"Invalid embedding format: {embedding_str}")

def load_csv(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from CSV file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
            # Validate required fields
            if 'query' not in row or not row['query']:
                raise ValueError(f"Row {row_num}: Missing required field 'query'")
            if 'timestamp' not in row or not row['timestamp']:
                raise ValueError(f"Row {row_num}: Missing required field 'timestamp'")
            
            # Parse timestamp
            try:
                timestamp = parse_timestamp(row['timestamp'])
            except ValueError as e:
                raise ValueError(f"Row {row_num}: {str(e)}")
            
            # Build data record
            record = {
                'query': row['query'].strip(),
                'timestamp': timestamp,
                'query_category': row.get('query_category', '').strip() or None,
                'confidence_score': row.get('confidence_score', '').strip() or None,
                'ai_response': row.get('ai_response', '').strip() or None,
            }
            
            # Parse embedding if provided
            if 'embedding' in row and row['embedding']:
                try:
                    record['embedding'] = parse_embedding(row['embedding'])
                except ValueError as e:
                    print(f"Warning: Row {row_num}: {str(e)}, will auto-generate embedding")
                    record['embedding'] = None
            else:
                record['embedding'] = None
            
            data.append(record)
    
    return data

def load_json(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON file must contain an array of objects")
    
    records = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx}: Expected object, got {type(item).__name__}")
        
        # Validate required fields
        if 'query' not in item or not item['query']:
            raise ValueError(f"Item {idx}: Missing required field 'query'")
        if 'timestamp' not in item:
            raise ValueError(f"Item {idx}: Missing required field 'timestamp'")
        
        # Parse timestamp
        if isinstance(item['timestamp'], str):
            try:
                timestamp = parse_timestamp(item['timestamp'])
            except ValueError as e:
                raise ValueError(f"Item {idx}: {str(e)}")
        elif isinstance(item['timestamp'], (int, float)):
            # Unix timestamp
            timestamp = datetime.fromtimestamp(item['timestamp'])
        else:
            raise ValueError(f"Item {idx}: Invalid timestamp format")
        
        # Build data record
        record = {
            'query': str(item['query']).strip(),
            'timestamp': timestamp,
            'query_category': item.get('query_category') or None,
            'confidence_score': str(item.get('confidence_score', '')).strip() or None,
            'ai_response': item.get('ai_response') or None,
        }
        
        # Parse embedding if provided
        if 'embedding' in item and item['embedding']:
            if isinstance(item['embedding'], list):
                record['embedding'] = item['embedding']
            else:
                try:
                    record['embedding'] = parse_embedding(item['embedding'])
                except ValueError as e:
                    print(f"Warning: Item {idx}: {str(e)}, will auto-generate embedding")
                    record['embedding'] = None
        else:
            record['embedding'] = None
        
        records.append(record)
    
    return records

def import_data(db, data: List[Dict[str, Any]], clear_existing: bool = False):
    """Import data into database"""
    if clear_existing:
        print("Clearing existing query logs...")
        db.query(QueryLog).delete()
        db.commit()
        print("Cleared existing data")
    
    print(f"\nImporting {len(data)} records...")
    
    imported = 0
    errors = 0
    
    for idx, record in enumerate(data, start=1):
        try:
            # Generate embedding if not provided
            if record['embedding'] is None:
                embedding = generate_simple_embedding(record['query'])
            else:
                embedding = record['embedding']
            
            # Create query log entry
            query_log = QueryLog(
                query=record['query'],
                query_category=record['query_category'],
                embedding=embedding,
                confidence_score=record['confidence_score'],
                ai_response=record['ai_response'],
                timestamp=record['timestamp']
            )
            
            db.add(query_log)
            imported += 1
            
            # Commit in batches of 100
            if imported % 100 == 0:
                db.commit()
                print(f"  Imported {imported} records...")
        
        except Exception as e:
            errors += 1
            print(f"Error importing record {idx}: {str(e)}")
            if errors > 10:
                print("Too many errors. Stopping import.")
                break
    
    # Final commit
    db.commit()
    
    print(f"\nImport complete!")
    print(f"  - Successfully imported: {imported} records")
    if errors > 0:
        print(f"  - Errors: {errors} records")
    
    return imported, errors

def main():
    parser = argparse.ArgumentParser(
        description="Import custom dataset for AI Pipeline Resilience Dashboard"
    )
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Path to CSV or JSON file containing query data'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json'],
        help='File format (csv or json). Auto-detected from extension if not specified.'
    )
    parser.add_argument(
        '--clear-existing',
        action='store_true',
        help='Clear existing query logs before importing'
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Auto-detect format if not specified
    if not args.format:
        if file_path.suffix.lower() == '.csv':
            file_format = 'csv'
        elif file_path.suffix.lower() == '.json':
            file_format = 'json'
        else:
            print("Error: Could not determine file format. Please specify --format csv or --format json")
            sys.exit(1)
    else:
        file_format = args.format
    
    print("=" * 60)
    print("AI Pipeline Resilience Dashboard - Custom Data Import")
    print("=" * 60)
    print(f"\nFile: {file_path}")
    print(f"Format: {file_format}")
    print(f"Clear existing: {args.clear_existing}")
    print()
    
    # Initialize database
    print("Initializing database...")
    init_db()
    
    # Load data
    try:
        if file_format == 'csv':
            print("Loading CSV file...")
            data = load_csv(file_path)
        else:
            print("Loading JSON file...")
            data = load_json(file_path)
        
        print(f"Loaded {len(data)} records")
    
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        sys.exit(1)
    
    # Import data
    db = SessionLocal()
    try:
        imported, errors = import_data(db, data, args.clear_existing)
        
        if errors == 0:
            print("\n" + "=" * 60)
            print("Import successful!")
            print("=" * 60)
            print(f"\nNext steps:")
            print(f"  1. Start API: python run_api.py")
            print(f"  2. Open dashboard to view your data")
        else:
            print(f"\nImport completed with {errors} errors. Please review the errors above.")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()

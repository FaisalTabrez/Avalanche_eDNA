#!/usr/bin/env python
"""
Embedding Version Manager

Track which model version generated which embeddings to enable:
- Comparing model improvements
- Migrating to new embedding versions
- Rolling back if needed
- Understanding performance over time

Usage:
    # Register a new model version
    python scripts/manage_embedding_versions.py register --version v1.0 --model "nt-500m-1000g" --description "Initial model"
    
    # Tag runs with a version
    python scripts/manage_embedding_versions.py tag-run My_Dataset/2024-11-22_10-30-00 --version v1.0
    
    # List all versions
    python scripts/manage_embedding_versions.py list
    
    # Compare versions
    python scripts/manage_embedding_versions.py compare v1.0 v2.0
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys


VERSION_FILE = Path('data/reference/embedding_versions.json')


def load_versions() -> Dict:
    """Load version tracking data"""
    if not VERSION_FILE.exists():
        return {
            'versions': {},
            'runs': {}
        }
    
    with open(VERSION_FILE, 'r') as f:
        return json.load(f)


def save_versions(data: Dict):
    """Save version tracking data"""
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VERSION_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def register_version(
    version: str,
    model: str,
    description: str = "",
    config: Optional[Dict] = None
):
    """Register a new embedding model version"""
    data = load_versions()
    
    if version in data['versions']:
        print(f"ERROR: Version {version} already exists")
        return False
    
    data['versions'][version] = {
        'version': version,
        'model': model,
        'description': description,
        'config': config or {},
        'registered_at': datetime.now().isoformat(),
        'run_count': 0
    }
    
    save_versions(data)
    print(f"✓ Registered version: {version}")
    print(f"  Model: {model}")
    print(f"  Description: {description}")
    return True


def tag_run(run_path: str, version: str):
    """Tag a run with an embedding version"""
    data = load_versions()
    
    if version not in data['versions']:
        print(f"ERROR: Version {version} not found. Register it first.")
        return False
    
    # Check if run exists
    full_path = Path('AvalancheData/runs') / run_path
    if not full_path.exists():
        print(f"ERROR: Run not found: {run_path}")
        return False
    
    # Tag run
    data['runs'][run_path] = {
        'version': version,
        'tagged_at': datetime.now().isoformat()
    }
    
    # Update run count
    data['versions'][version]['run_count'] = sum(
        1 for r in data['runs'].values() if r['version'] == version
    )
    
    save_versions(data)
    print(f"✓ Tagged {run_path} with version {version}")
    return True


def list_versions():
    """List all registered versions"""
    data = load_versions()
    
    if not data['versions']:
        print("No versions registered yet.")
        return
    
    print("=" * 80)
    print("EMBEDDING VERSIONS")
    print("=" * 80)
    
    for version, info in sorted(data['versions'].items()):
        print(f"\n{version}")
        print(f"  Model:       {info['model']}")
        print(f"  Description: {info['description']}")
        print(f"  Registered:  {info['registered_at']}")
        print(f"  Run count:   {info['run_count']}")
    
    print("\n" + "=" * 80)


def compare_versions(v1: str, v2: str):
    """Compare two embedding versions"""
    data = load_versions()
    
    if v1 not in data['versions'] or v2 not in data['versions']:
        print("ERROR: One or both versions not found")
        return
    
    info1 = data['versions'][v1]
    info2 = data['versions'][v2]
    
    # Get runs for each version
    runs1 = [run for run, info in data['runs'].items() if info['version'] == v1]
    runs2 = [run for run, info in data['runs'].items() if info['version'] == v2]
    
    print("=" * 80)
    print(f"COMPARING: {v1} vs {v2}")
    print("=" * 80)
    
    print(f"\n{v1}:")
    print(f"  Model:       {info1['model']}")
    print(f"  Registered:  {info1['registered_at']}")
    print(f"  Run count:   {len(runs1)}")
    
    print(f"\n{v2}:")
    print(f"  Model:       {info2['model']}")
    print(f"  Registered:  {info2['registered_at']}")
    print(f"  Run count:   {len(runs2)}")
    
    # Find common datasets
    datasets1 = set(r.split('/')[0] for r in runs1)
    datasets2 = set(r.split('/')[0] for r in runs2)
    common = datasets1 & datasets2
    
    if common:
        print(f"\nCommon datasets: {', '.join(sorted(common))}")
        print("  (Can compare performance on these)")
    else:
        print("\nNo common datasets between versions")
    
    print("=" * 80)


def auto_tag_runs(version: str, runs_root: Path = Path('AvalancheData/runs')):
    """Automatically tag all untagged runs with a version"""
    data = load_versions()
    
    if version not in data['versions']:
        print(f"ERROR: Version {version} not found")
        return
    
    # Find all runs
    tagged = 0
    skipped = 0
    
    for run_dir in runs_root.glob('*/*'):
        if not run_dir.is_dir():
            continue
        
        run_key = f"{run_dir.parent.name}/{run_dir.name}"
        
        # Skip if already tagged
        if run_key in data['runs']:
            skipped += 1
            continue
        
        # Tag it
        data['runs'][run_key] = {
            'version': version,
            'tagged_at': datetime.now().isoformat()
        }
        tagged += 1
        print(f"  Tagged: {run_key}")
    
    # Update run count
    data['versions'][version]['run_count'] = sum(
        1 for r in data['runs'].values() if r['version'] == version
    )
    
    save_versions(data)
    
    print(f"\n✓ Tagged {tagged} runs with {version}")
    if skipped > 0:
        print(f"  Skipped {skipped} already tagged runs")


def main():
    parser = argparse.ArgumentParser(
        description="Manage embedding model versions"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new version')
    register_parser.add_argument('--version', required=True, help='Version name (e.g., v1.0)')
    register_parser.add_argument('--model', required=True, help='Model name')
    register_parser.add_argument('--description', default='', help='Version description')
    
    # Tag run command
    tag_parser = subparsers.add_parser('tag-run', help='Tag a run with a version')
    tag_parser.add_argument('run_path', help='Run path (e.g., My_Dataset/2024-11-22)')
    tag_parser.add_argument('--version', required=True, help='Version to tag with')
    
    # Auto-tag command
    auto_parser = subparsers.add_parser('auto-tag', help='Auto-tag all untagged runs')
    auto_parser.add_argument('--version', required=True, help='Version to tag with')
    
    # List command
    subparsers.add_parser('list', help='List all versions')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('version1', help='First version')
    compare_parser.add_argument('version2', help='Second version')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'register':
        register_version(args.version, args.model, args.description)
    elif args.command == 'tag-run':
        tag_run(args.run_path, args.version)
    elif args.command == 'auto-tag':
        auto_tag_runs(args.version)
    elif args.command == 'list':
        list_versions()
    elif args.command == 'compare':
        compare_versions(args.version1, args.version2)


if __name__ == '__main__':
    main()

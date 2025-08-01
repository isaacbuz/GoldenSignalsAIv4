#!/usr/bin/env python3
"""
Documentation Inventory Script
Analyzes all markdown files and creates a mapping for consolidation
"""

import os
import json
from pathlib import Path
from datetime import datetime
import re

def analyze_markdown_file(filepath):
    """Analyze a markdown file and extract key information"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract title (first # heading)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else Path(filepath).stem

        # Count sections
        sections = len(re.findall(r'^##\s+', content, re.MULTILINE))

        # Check for key content types
        has_api_docs = bool(re.search(r'(API|endpoint|route)', content, re.IGNORECASE))
        has_setup_info = bool(re.search(r'(install|setup|configuration)', content, re.IGNORECASE))
        has_architecture = bool(re.search(r'(architecture|design|structure)', content, re.IGNORECASE))
        has_deployment = bool(re.search(r'(deploy|docker|kubernetes)', content, re.IGNORECASE))

        # Get file stats
        stats = os.stat(filepath)

        return {
            'path': str(filepath),
            'title': title,
            'size': stats.st_size,
            'lines': content.count('\n'),
            'sections': sections,
            'last_modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'content_type': {
                'api': has_api_docs,
                'setup': has_setup_info,
                'architecture': has_architecture,
                'deployment': has_deployment
            }
        }
    except Exception as e:
        return {
            'path': str(filepath),
            'error': str(e)
        }

def categorize_file(file_info):
    """Categorize file based on content and suggest new location"""
    path = file_info['path']
    name = Path(path).stem.upper()

    # Map to new structure based on content and filename
    if 'error' in file_info:
        return 'ERROR'

    # Check filename patterns first
    if 'README' in name:
        if 'agents' in path:
            return '04-MODULES/AGENTS.md'
        elif 'frontend' in path:
            return '03-DEVELOPMENT/FRONTEND_GUIDE.md'
        else:
            return '01-OVERVIEW/PROJECT_OVERVIEW.md'

    if any(x in name for x in ['SETUP', 'INSTALL', 'GETTING_STARTED', 'QUICK_START']):
        return '02-GETTING_STARTED/SETUP_GUIDE.md'

    if any(x in name for x in ['API', 'ENDPOINT', 'ROUTES']):
        return '03-DEVELOPMENT/API_REFERENCE.md'

    if any(x in name for x in ['DEPLOY', 'DOCKER', 'KUBERNETES', 'K8S']):
        return '05-DEPLOYMENT/DEPLOYMENT_GUIDE.md'

    if any(x in name for x in ['ARCHITECTURE', 'DESIGN', 'STRUCTURE']):
        return '01-OVERVIEW/ARCHITECTURE.md'

    if any(x in name for x in ['TEST', 'TESTING', 'QA']):
        return '03-DEVELOPMENT/TESTING_GUIDE.md'

    if any(x in name for x in ['AGENT', 'AI', 'ML', 'MODEL']):
        return '04-MODULES/AI_INTEGRATION.md'

    if any(x in name for x in ['BACKTEST', 'BACKTESTING']):
        return '04-MODULES/BACKTESTING.md'

    if any(x in name for x in ['SIGNAL']):
        return '04-MODULES/SIGNALS.md'

    if any(x in name for x in ['MONITOR', 'METRIC', 'LOG']):
        return '06-OPERATIONS/MONITORING.md'

    if any(x in name for x in ['PERFORMANCE', 'OPTIMIZE']):
        return '06-OPERATIONS/PERFORMANCE.md'

    if any(x in name for x in ['SECURITY', 'AUTH']):
        return '06-OPERATIONS/SECURITY.md'

    if any(x in name for x in ['TROUBLESHOOT', 'DEBUG', 'ISSUE']):
        return '02-GETTING_STARTED/TROUBLESHOOTING.md'

    # Check content type if no filename match
    content = file_info.get('content_type', {})
    if content.get('deployment'):
        return '05-DEPLOYMENT/DEPLOYMENT_GUIDE.md'
    elif content.get('api'):
        return '03-DEVELOPMENT/API_REFERENCE.md'
    elif content.get('setup'):
        return '02-GETTING_STARTED/SETUP_GUIDE.md'
    elif content.get('architecture'):
        return '01-OVERVIEW/ARCHITECTURE.md'

    # Default to archive for unclear files
    return '07-ARCHIVE/' + Path(path).name

def main():
    """Main function to inventory all markdown files"""
    # Find all markdown files
    md_files = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['.venv', 'node_modules', '.git', 'htmlcov']):
            continue

        for file in files:
            if file.endswith('.md'):
                filepath = Path(root) / file
                md_files.append(filepath)

    print(f"Found {len(md_files)} markdown files")

    # Analyze each file
    inventory = []
    for filepath in md_files:
        print(f"Analyzing: {filepath}")
        file_info = analyze_markdown_file(filepath)
        file_info['suggested_location'] = categorize_file(file_info)
        inventory.append(file_info)

    # Create migration mapping
    migration_map = {}
    for item in inventory:
        if 'error' not in item:
            old_path = item['path']
            new_location = item['suggested_location']

            if new_location not in migration_map:
                migration_map[new_location] = []

            migration_map[new_location].append({
                'old_path': old_path,
                'title': item['title'],
                'size': item['size'],
                'sections': item['sections']
            })

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(f'docs_inventory_{timestamp}.json', 'w') as f:
        json.dump(inventory, f, indent=2)

    with open(f'migration_map_{timestamp}.json', 'w') as f:
        json.dump(migration_map, f, indent=2)

    # Print summary
    print("\n=== Documentation Inventory Summary ===")
    print(f"Total files: {len(md_files)}")
    print(f"Files with errors: {sum(1 for item in inventory if 'error' in item)}")

    print("\n=== Migration Summary ===")
    for new_location, files in sorted(migration_map.items()):
        print(f"\n{new_location}: {len(files)} files")
        for file in files[:3]:  # Show first 3
            print(f"  - {Path(file['old_path']).name} ({file['title']})")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")

    print(f"\nInventory saved to: docs_inventory_{timestamp}.json")
    print(f"Migration map saved to: migration_map_{timestamp}.json")

if __name__ == "__main__":
    main()

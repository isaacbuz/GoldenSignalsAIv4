#!/usr/bin/env python3
"""Check status of our specific issues #209-#216."""

import os
import requests

# GitHub API configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_OWNER = 'isaacbuz'
REPO_NAME = 'GoldenSignalsAIv4'

headers = {
    'Accept': 'application/vnd.github.v3+json'
}

if GITHUB_TOKEN:
    headers['Authorization'] = f'token {GITHUB_TOKEN}'

print("# Status of Issues #209-#216\n")

# Check each issue
for issue_num in range(209, 217):
    issue_url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues/{issue_num}'
    response = requests.get(issue_url, headers=headers)
    
    if response.status_code == 200:
        issue = response.json()
        status = "✅ CLOSED" if issue['state'] == 'closed' else "📋 OPEN"
        labels = ', '.join([label['name'] for label in issue['labels']])
        
        print(f"## Issue #{issue_num}: {issue['title']}")
        print(f"Status: {status}")
        print(f"Labels: {labels}")
        
        if issue['state'] == 'closed':
            print(f"Closed at: {issue['closed_at'][:10]}")
            if issue.get('closed_by'):
                print(f"Closed by: {issue['closed_by']['login']}")
        
        print()
    else:
        print(f"Error fetching issue #{issue_num}: {response.status_code}\n")

# Summary
print("\n## Summary")
print("Based on our work today:")
print("- ✅ #211: Security Audit - COMPLETED")
print("- ✅ #209: Codebase Consolidation - COMPLETED") 
print("- ✅ #212: CI/CD Pipeline - COMPLETED")
print("- 🚧 #210: Agent System Unit Testing - IN PROGRESS")
print("- 📋 #213-#216: Not yet started")
print("\nThe 7 new open issues (#234-#248) were automatically created by:")
print("- Test suite issues (#234-#240) - created from our test implementation plan")
print("- Dependabot PRs (#241-#248) - created when we configured Dependabot") 
"""
Direct GitHub Issues Creation for RAG Implementation
Creates issues without interactive prompts
"""

import json
import os
import time
import requests
from datetime import datetime

# GitHub API configuration
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "isaacbuz"
REPO_NAME = "GoldenSignalsAIv4"

def create_github_issue(issue_data, repo_owner, repo_name, token):
    """Create a single issue on GitHub"""
    url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/issues"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    issue_payload = {
        "title": issue_data["title"],
        "body": issue_data["body"],
        "labels": issue_data.get("labels", [])
    }
    
    response = requests.post(url, json=issue_payload, headers=headers)
    
    if response.status_code == 201:
        return response.json()
    else:
        print(f"Error creating issue: {response.status_code}")
        print(f"Response: {response.text}")
        return None

# Load issues
with open('rag_github_issues.json', 'r') as f:
    issues = json.load(f)

print(f"🚀 Creating {len(issues)} issues on {REPO_OWNER}/{REPO_NAME}")
print("=" * 60)

created_issues = []
failed_issues = []

# Create each issue
for i, issue in enumerate(issues, 1):
    print(f"\nCreating issue {i}/{len(issues)}: {issue['title']}...")
    
    result = create_github_issue(issue, REPO_OWNER, REPO_NAME, GITHUB_TOKEN)
    
    if result:
        created_issues.append(result)
        print(f"✅ Created: #{result['number']} - {result['title']}")
        print(f"   URL: {result['html_url']}")
    else:
        failed_issues.append(issue)
        print(f"❌ Failed to create: {issue['title']}")
    
    # Rate limiting
    if i < len(issues):
        time.sleep(2)

# Summary
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print(f"✅ Successfully created: {len(created_issues)} issues")
print(f"❌ Failed: {len(failed_issues)} issues")

if created_issues:
    print("\nCreated Issues:")
    for issue in created_issues:
        print(f"  #{issue['number']}: {issue['title']}")
        print(f"     {issue['html_url']}")

# Save results
results = {
    "timestamp": datetime.now().isoformat(),
    "repository": f"{REPO_OWNER}/{REPO_NAME}",
    "created_count": len(created_issues),
    "failed_count": len(failed_issues),
    "created_issues": [
        {
            "number": issue['number'],
            "title": issue['title'],
            "url": issue['html_url']
        }
        for issue in created_issues
    ]
}

with open('rag_issues_creation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📁 Results saved to: rag_issues_creation_results.json") 
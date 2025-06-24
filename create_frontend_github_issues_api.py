#!/usr/bin/env python3
"""
Create Frontend Enhancement GitHub Issues via API
"""

import json
import os
import time
from typing import Dict, List
import requests
from datetime import datetime

def load_issues(filename: str = "frontend_enhancement_issues.json") -> List[Dict]:
    """Load issues from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_github_issue(token: str, owner: str, repo: str, issue_data: Dict) -> Dict:
    """Create a single GitHub issue"""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Prepare issue data
    issue = {
        "title": issue_data["title"],
        "body": issue_data["body"],
        "labels": issue_data.get("labels", [])
    }
    
    # Add milestone if specified
    if "milestone" in issue_data:
        # Note: Milestone needs to be created first and we need its number
        # For now, we'll skip milestone assignment
        pass
    
    response = requests.post(url, json=issue, headers=headers)
    
    if response.status_code == 201:
        return response.json()
    else:
        print(f"❌ Failed to create issue: {issue_data['title']}")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def create_all_issues(token: str, owner: str, repo: str):
    """Create all frontend enhancement issues"""
    print(f"🚀 Creating Frontend Enhancement issues in {owner}/{repo}...")
    
    # Load issues
    issues = load_issues()
    created_issues = []
    issue_map = {}  # To track issue numbers for cross-references
    
    # Create issues
    for i, issue_data in enumerate(issues):
        print(f"\n📝 Creating issue {i+1}/{len(issues)}: {issue_data['title']}")
        
        # Update body with issue references if this is the EPIC
        if "EPIC" in issue_data["title"] and issue_map:
            # Update the phase references with actual issue numbers
            body = issue_data["body"]
            for j, phase_num in enumerate(range(2, 8), 1):  # Phases 1-6
                if phase_num in issue_map:
                    body = body.replace(f"(#issue-{j})", f"(#{issue_map[phase_num]})")
            issue_data["body"] = body
        
        result = create_github_issue(token, owner, repo, issue_data)
        
        if result:
            created_issues.append(result)
            issue_map[i+1] = result["number"]
            print(f"   ✅ Created: #{result['number']} - {result['html_url']}")
        
        # Rate limit consideration
        time.sleep(1)
    
    # Summary
    print(f"\n✅ Successfully created {len(created_issues)} out of {len(issues)} issues")
    
    # Save results
    results = {
        "created_at": datetime.now().isoformat(),
        "repository": f"{owner}/{repo}",
        "total_issues": len(issues),
        "created_issues": len(created_issues),
        "issues": [
            {
                "number": issue["number"],
                "title": issue["title"],
                "url": issue["html_url"],
                "state": issue["state"]
            }
            for issue in created_issues
        ]
    }
    
    with open("frontend_enhancement_issues_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to frontend_enhancement_issues_results.json")
    
    return created_issues

def main():
    """Main function"""
    # Get GitHub token from environment
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ Error: GITHUB_TOKEN environment variable not set")
        print("Please set it with: export GITHUB_TOKEN='your_token_here'")
        return
    
    # Repository details - update these as needed
    owner = "isaacbuz"  # Your GitHub username
    repo = "GoldenSignalsAIv4"  # Your repository name
    
    # Confirm before proceeding
    print(f"📋 About to create issues in: {owner}/{repo}")
    print(f"📊 Total issues to create: 11")
    print("\nIssue categories:")
    print("- 1 EPIC issue")
    print("- 6 Phase implementation issues")
    print("- 4 Cross-cutting concern issues")
    
    response = input("\n❓ Do you want to proceed? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cancelled")
        return
    
    # Create issues
    create_all_issues(token, owner, repo)
    
    print("\n🎉 Frontend Enhancement issue creation complete!")
    print("\n📝 Next steps:")
    print("1. Review the created issues on GitHub")
    print("2. Add any additional labels or assignees")
    print("3. Create milestones if needed")
    print("4. Start implementation according to the phases")

if __name__ == "__main__":
    main() 
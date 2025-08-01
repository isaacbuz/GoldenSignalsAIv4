#!/usr/bin/env python3
"""
Update GitHub Issues with Milestones and Cross-References
"""

import json
import os
import requests
from typing import Dict, List

def get_milestones(token: str, owner: str, repo: str) -> Dict[str, int]:
    """Get existing milestones and their IDs"""
    url = f"https://api.github.com/repos/{owner}/{repo}/milestones"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(url, headers=headers)
    milestones = {}

    if response.status_code == 200:
        for milestone in response.json():
            milestones[milestone['title']] = milestone['number']

    return milestones

def update_issue_milestone(token: str, owner: str, repo: str, issue_num: str, milestone_num: int):
    """Update an issue with a milestone"""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    data = {
        "milestone": milestone_num
    }

    response = requests.patch(url, json=data, headers=headers)

    if response.status_code == 200:
        print(f"✅ Updated issue #{issue_num} with milestone")
    else:
        print(f"❌ Failed to update issue #{issue_num}")
        print(f"   Response: {response.text}")

def update_epic_issue(token: str, owner: str, repo: str):
    """Update the EPIC issue with correct cross-references"""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/198"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Updated body with correct issue numbers
    updated_body = """## Overview
This epic tracks the comprehensive frontend enhancement to fully utilize all backend capabilities of GoldenSignalsAI V2.

## Objectives
- 🎯 Implement missing API integrations
- 📊 Create advanced backtesting interface
- 🤖 Integrate multimodal AI features
- 📈 Build hybrid signal intelligence dashboard
- 💼 Develop institutional-grade portfolio tools
- 🔧 Add complete system monitoring

## Success Metrics
- Page load time < 2s
- API response time < 200ms
- Feature adoption rate > 70%
- Signal accuracy improvement 15%
- User satisfaction score > 4.5/5

## Timeline
- **Month 1**: Core Infrastructure + Backtesting
- **Month 2**: AI Integration + Hybrid Signals
- **Month 3**: Portfolio Management + Admin Tools

## Related Documentation
- [Frontend Enhancement Plan](./FRONTEND_ENHANCEMENT_PLAN.md)
- [API Documentation](./API_DOCUMENTATION.md)
- [Project Board](https://github.com/isaacbuz/GoldenSignalsAIv4/projects)

## Implementation Phases

### Phase 1: Core Infrastructure (#199)
**Weeks 1-2** - Establish foundation for advanced features

### Phase 2: Advanced Backtesting Suite (#200)
**Weeks 3-4** - Professional-grade backtesting interface

### Phase 3: AI & Multimodal Integration (#201)
**Weeks 5-6** - State-of-the-art AI trading assistant

### Phase 4: Hybrid Signal Intelligence (#202)
**Weeks 7-8** - Advanced multi-agent signal fusion

### Phase 5: Portfolio & Risk Management (#203)
**Weeks 9-10** - Institutional-grade portfolio tools

### Phase 6: Admin & System Monitoring (#204)
**Weeks 11-12** - Complete system observability

## Cross-Cutting Concerns
These issues should be addressed throughout all phases:
- Frontend Performance Optimization (#205)
- UI/UX Design System Enhancement (#206)
- Frontend Testing Strategy (#207)
- Frontend Documentation (#208)

## Progress Tracking
Track progress on our [Project Board](https://github.com/isaacbuz/GoldenSignalsAIv4/projects) with the following columns:
- 📋 Backlog
- 🏃 In Progress
- 👀 In Review
- ✅ Done

## Getting Started
1. Review the [Frontend Enhancement Plan](./FRONTEND_ENHANCEMENT_PLAN.md)
2. Pick an issue from the current phase
3. Create a feature branch
4. Submit PR when ready
5. Update the project board

## Team Coordination
- Weekly progress reviews
- Daily standups during active development
- Slack channel: #frontend-enhancement
- Technical discussions in issue comments
"""

    data = {
        "body": updated_body
    }

    response = requests.patch(url, json=data, headers=headers)

    if response.status_code == 200:
        print("✅ Updated EPIC issue #198 with correct cross-references")
    else:
        print("❌ Failed to update EPIC issue")
        print(f"   Response: {response.text}")

def assign_milestones_to_issues(token: str, owner: str, repo: str, milestones: Dict[str, int]):
    """Assign milestones to each issue based on phase"""

    # Issue to milestone mapping
    issue_milestone_map = {
        "199": "Phase 1: Core Infrastructure",
        "200": "Phase 2: Advanced Backtesting Suite",
        "201": "Phase 3: AI & Multimodal Integration",
        "202": "Phase 4: Hybrid Signal Intelligence",
        "203": "Phase 5: Portfolio & Risk Management",
        "204": "Phase 6: Admin & System Monitoring"
    }

    # Cross-cutting issues don't get phase milestones
    # They span all phases

    for issue_num, milestone_title in issue_milestone_map.items():
        if milestone_title in milestones:
            update_issue_milestone(token, owner, repo, issue_num, milestones[milestone_title])
        else:
            print(f"⚠️  Milestone '{milestone_title}' not found for issue #{issue_num}")

def add_issue_dependencies(token: str, owner: str, repo: str):
    """Add dependency information to issues"""

    # Dependencies to add in issue comments
    dependencies = {
        "200": ["Depends on #199 (Core Infrastructure)"],
        "201": ["Depends on #199 (Core Infrastructure)"],
        "202": ["Depends on #199 (Core Infrastructure)", "Benefits from #201 (AI Integration)"],
        "203": ["Depends on #199 (Core Infrastructure)", "Depends on #200 (Backtesting)"],
        "204": ["Depends on #199 (Core Infrastructure)"],
        "205": ["Affects all phases - should be ongoing"],
        "206": ["Affects all phases - should be ongoing"],
        "207": ["Affects all phases - should be ongoing"],
        "208": ["Affects all phases - should be ongoing"]
    }

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    for issue_num, deps in dependencies.items():
        if deps:
            comment_body = "## Dependencies\n\n" + "\n".join(f"- {dep}" for dep in deps)

            url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}/comments"
            response = requests.post(url, json={"body": comment_body}, headers=headers)

            if response.status_code == 201:
                print(f"✅ Added dependencies to issue #{issue_num}")
            else:
                print(f"❌ Failed to add dependencies to issue #{issue_num}")

def main():
    """Main function"""
    # Get GitHub token
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ Error: GITHUB_TOKEN environment variable not set")
        return

    # Repository details
    owner = "isaacbuz"
    repo = "GoldenSignalsAIv4"

    print("🚀 Updating GitHub Issues...")

    # Get existing milestones
    print("\n📍 Fetching milestones...")
    milestones = get_milestones(token, owner, repo)
    print(f"Found {len(milestones)} milestones")

    # Update EPIC issue with correct references
    print("\n📝 Updating EPIC issue...")
    update_epic_issue(token, owner, repo)

    # Assign milestones to issues
    print("\n🏷️  Assigning milestones to issues...")
    assign_milestones_to_issues(token, owner, repo, milestones)

    # Add dependency information
    print("\n🔗 Adding dependency information...")
    add_issue_dependencies(token, owner, repo)

    print("\n✅ Issue updates complete!")
    print("\n📊 Summary:")
    print("- Updated EPIC issue with correct cross-references")
    print("- Assigned milestones to phase issues")
    print("- Added dependency information to issues")
    print("\nVisit https://github.com/isaacbuz/GoldenSignalsAIv4/issues to see the updates")

if __name__ == "__main__":
    main()

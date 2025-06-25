#!/usr/bin/env python3
"""Check and summarize open GitHub issues."""

import os
import requests
from datetime import datetime

# GitHub API configuration - using environment variable
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_OWNER = 'isaacbuz'
REPO_NAME = 'GoldenSignalsAIv4'

headers = {
    'Accept': 'application/vnd.github.v3+json'
}

if GITHUB_TOKEN:
    headers['Authorization'] = f'token {GITHUB_TOKEN}'

# Get open issues
issues_url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues'
params = {
    'state': 'open',
    'sort': 'created',
    'direction': 'asc',
    'per_page': 100
}

response = requests.get(issues_url, headers=headers, params=params)

if response.status_code == 200:
    issues = response.json()
    
    print(f"# Open Issues Summary for {REPO_OWNER}/{REPO_NAME}")
    print(f"Total open issues: {len(issues)}\n")
    
    # Group by milestone
    by_milestone = {}
    no_milestone = []
    
    for issue in issues:
        milestone = issue.get('milestone')
        if milestone:
            milestone_title = milestone['title']
            if milestone_title not in by_milestone:
                by_milestone[milestone_title] = []
            by_milestone[milestone_title].append(issue)
        else:
            no_milestone.append(issue)
    
    # Print by milestone
    for milestone, milestone_issues in by_milestone.items():
        print(f"\n## Milestone: {milestone}")
        print(f"Issues: {len(milestone_issues)}")
        for issue in milestone_issues:
            labels = ', '.join([label['name'] for label in issue['labels']])
            print(f"- #{issue['number']}: {issue['title']}")
            print(f"  Labels: {labels}")
            print(f"  Created: {issue['created_at'][:10]}")
    
    # Print issues without milestone
    if no_milestone:
        print(f"\n## No Milestone")
        print(f"Issues: {len(no_milestone)}")
        for issue in no_milestone:
            labels = ', '.join([label['name'] for label in issue['labels']])
            print(f"- #{issue['number']}: {issue['title']}")
            print(f"  Labels: {labels}")
            print(f"  Created: {issue['created_at'][:10]}")
    
    # Priority breakdown
    print("\n## Priority Breakdown")
    priority_count = {'P0': 0, 'P1': 0, 'P2': 0, 'P3': 0, 'Other': 0}
    
    for issue in issues:
        found_priority = False
        for label in issue['labels']:
            if label['name'] in ['P0', 'P1', 'P2', 'P3']:
                priority_count[label['name']] += 1
                found_priority = True
                break
        if not found_priority:
            priority_count['Other'] += 1
    
    for priority, count in priority_count.items():
        if count > 0:
            print(f"- {priority}: {count} issues")
    
    # Recent activity
    print("\n## Issues from our recent work")
    our_issues = [i for i in issues if i['number'] >= 209 and i['number'] <= 216]
    
    if our_issues:
        print(f"Found {len(our_issues)} issues from #209-#216:")
        for issue in our_issues:
            status = "🚧 In Progress" if any(label['name'] == 'in-progress' for label in issue['labels']) else "📋 Open"
            print(f"- #{issue['number']}: {issue['title']} - {status}")
    
else:
    print(f"Error fetching issues: {response.status_code}")
    print(response.text) 
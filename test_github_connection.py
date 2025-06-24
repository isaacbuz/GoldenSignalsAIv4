"""
Test GitHub API Connection and Repository Access
"""

import os
import requests

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API_URL = "https://api.github.com"

def test_connection():
    """Test various GitHub API endpoints"""
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    print("🔍 Testing GitHub API Connection")
    print("=" * 60)
    
    # Test 1: Check authenticated user
    print("\n1. Testing authentication...")
    response = requests.get(f"{GITHUB_API_URL}/user", headers=headers)
    if response.status_code == 200:
        user_data = response.json()
        print(f"✅ Authenticated as: {user_data['login']}")
        print(f"   Name: {user_data.get('name', 'N/A')}")
    else:
        print(f"❌ Authentication failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    # Test 2: List user's repositories
    print("\n2. Listing your repositories...")
    response = requests.get(f"{GITHUB_API_URL}/user/repos", headers=headers)
    if response.status_code == 200:
        repos = response.json()
        print(f"✅ Found {len(repos)} repositories:")
        for repo in repos[:10]:  # Show first 10
            print(f"   - {repo['full_name']} {'(private)' if repo['private'] else '(public)'}")
    else:
        print(f"❌ Failed to list repos: {response.status_code}")
    
    # Test 3: Check specific repository
    print("\n3. Checking for GoldenSignalsAI_V2 repository...")
    
    # Try different possible paths
    possible_repos = [
        "isaacbuz/GoldenSignalsAI_V2",
        "isaacbuz/goldensignalsai_v2",
        "isaacbuz/GoldenSignalsAI-V2",
        "isaacbuz/goldensignals-ai-v2"
    ]
    
    found_repo = None
    for repo_path in possible_repos:
        response = requests.get(f"{GITHUB_API_URL}/repos/{repo_path}", headers=headers)
        if response.status_code == 200:
            found_repo = response.json()
            print(f"✅ Found repository: {found_repo['full_name']}")
            print(f"   Private: {found_repo['private']}")
            print(f"   Default branch: {found_repo['default_branch']}")
            print(f"   URL: {found_repo['html_url']}")
            break
    
    if not found_repo:
        print("❌ Could not find GoldenSignalsAI_V2 repository")
        print("\nSearching for repositories with 'golden' in the name...")
        response = requests.get(f"{GITHUB_API_URL}/user/repos?per_page=100", headers=headers)
        if response.status_code == 200:
            repos = response.json()
            golden_repos = [r for r in repos if 'golden' in r['name'].lower()]
            if golden_repos:
                print("Found these repositories:")
                for repo in golden_repos:
                    print(f"   - {repo['full_name']}")
            else:
                print("No repositories with 'golden' in the name found")
    
    # Test 4: Check token permissions
    print("\n4. Checking token permissions...")
    response = requests.get(f"{GITHUB_API_URL}/user", headers=headers)
    if response.status_code == 200:
        scopes = response.headers.get('X-OAuth-Scopes', '').split(', ')
        print(f"✅ Token scopes: {scopes}")
        if 'repo' in scopes:
            print("   ✅ Has 'repo' scope (can create issues)")
        else:
            print("   ❌ Missing 'repo' scope (cannot create issues)")

if __name__ == "__main__":
    if not GITHUB_TOKEN:
        print("❌ GITHUB_TOKEN not set!")
    else:
        test_connection() 
import os
import sys
from pathlib import Path
from getpass import getpass

def setup_api_keys():
    """Interactive script to set up API keys"""
    print("🔑 Setting up API keys for GoldenSignalsAI")
    print("----------------------------------------")

    # Get environment
    print("\nEnvironment Setup")
    env = input("Enter environment (development/production) [development]: ").strip().lower() or "development"

    # API Keys
    api_keys = {
        'GROK_API_KEY': {
            'name': 'GROK API',
            'url': 'https://x.ai',
            'required': env == 'production'
        },
        'ALPHA_VANTAGE_API_KEY': {
            'name': 'Alpha Vantage',
            'url': 'https://www.alphavantage.co/support/#api-key',
            'required': True
        },
        'TWITTER_API_KEY': {
            'name': 'Twitter',
            'url': 'https://developer.twitter.com',
            'required': env == 'production'
        },
        'NEWS_API_KEY': {
            'name': 'News API',
            'url': 'https://newsapi.org',
            'required': True
        },
        'FINNHUB_API_KEY': {
            'name': 'Finnhub',
            'url': 'https://finnhub.io',
            'required': env == 'production'
        },
        'POLYGON_API_KEY': {
            'name': 'Polygon',
            'url': 'https://polygon.io',
            'required': env == 'production'
        },
        'BENZINGA_API_KEY': {
            'name': 'Benzinga',
            'url': 'https://benzinga.com',
            'required': env == 'production'
        }
    }

    # Collect API keys
    collected_keys = {}
    for key, info in api_keys.items():
        if info['required'] or input(f"\nSet up {info['name']} API key? (y/n) [n]: ").strip().lower() == 'y':
            print(f"\n{info['name']} API Key")
            print(f"   Get your API key from: {info['url']}")
            collected_keys[key] = getpass(f"Enter your {info['name']} API key: ").strip()

    # Create .env file
    env_path = Path(__file__).parent.parent / '.env'
    try:
        with open(env_path, 'w') as f:
            f.write(f"# Environment\n")
            f.write(f"ENV={env}\n\n")

            f.write(f"# API Keys\n")
            for key, value in collected_keys.items():
                f.write(f"{key}={value}\n")

            f.write(f"\n# Other Configuration\n")
            f.write(f"LOG_LEVEL=INFO\n")

        print(f"\n✅ API keys saved to {env_path}")

        # Set file permissions to be readable only by the owner
        os.chmod(env_path, 0o600)
        print("✅ Set secure file permissions")

    except Exception as e:
        print(f"\n❌ Error saving API keys: {str(e)}")
        sys.exit(1)

    print("\n🔒 Security Tips:")
    print("1. Never commit .env files to version control")
    print("2. Keep your API keys secure and don't share them")
    print("3. Regularly rotate your API keys")
    print("4. Use environment variables in production")
    print("5. Use different API keys for development and production")

if __name__ == "__main__":
    setup_api_keys()

#!/usr/bin/env python3
"""
Database Requirements Checker for GoldenSignalsAI V3
Checks if required databases are available and provides setup instructions
"""

import asyncio
import os
import sys
from typing import Tuple, Optional

# Try imports
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from sqlalchemy.ext.asyncio import create_async_engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class Colors:
    """Terminal colors for output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header():
    """Print script header"""
    print(f"\n{Colors.BOLD}🗄️  GoldenSignalsAI V3 - Database Requirements Checker{Colors.END}")
    print("=" * 60)


def print_status(service: str, status: bool, message: str = ""):
    """Print service status"""
    icon = "✅" if status else "❌"
    color = Colors.GREEN if status else Colors.RED
    print(f"{icon} {service}: {color}{message or ('Connected' if status else 'Not Available')}{Colors.END}")


async def check_postgresql() -> Tuple[bool, str]:
    """Check PostgreSQL connection"""
    if not ASYNCPG_AVAILABLE:
        return False, "asyncpg not installed (pip install asyncpg)"

    # Get connection details from environment or use defaults
    host = os.getenv('DB_HOST', 'localhost')
    port = int(os.getenv('DB_PORT', '5432'))
    database = os.getenv('DB_NAME', 'goldensignals')
    user = os.getenv('DB_USER', 'goldensignals')
    password = os.getenv('DB_PASSWORD', 'goldensignals_secure_password')

    try:
        # Try to connect
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            timeout=5
        )

        # Test query
        version = await conn.fetchval('SELECT version()')
        await conn.close()

        return True, f"Connected to PostgreSQL ({version.split(',')[0]})"

    except asyncpg.InvalidCatalogNameError:
        return False, f"Database '{database}' does not exist"
    except asyncpg.InvalidPasswordError:
        return False, "Invalid username/password"
    except asyncpg.ConnectionDoesNotExistError:
        return False, f"Cannot connect to PostgreSQL at {host}:{port}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def check_redis() -> Tuple[bool, str]:
    """Check Redis connection"""
    if not REDIS_AVAILABLE:
        return False, "redis not installed (pip install redis)"

    # Get connection details from environment or use defaults
    host = os.getenv('REDIS_HOST', 'localhost')
    port = int(os.getenv('REDIS_PORT', '6379'))
    password = os.getenv('REDIS_PASSWORD', None)

    try:
        # Try to connect
        r = redis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=True,
            socket_connect_timeout=5
        )

        # Test connection
        r.ping()
        info = r.info()
        version = info.get('redis_version', 'unknown')

        return True, f"Connected to Redis {version}"

    except redis.ConnectionError:
        return False, f"Cannot connect to Redis at {host}:{port}"
    except redis.AuthenticationError:
        return False, "Redis authentication failed"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def check_sqlite_fallback() -> Tuple[bool, str]:
    """Check if SQLite can be used as fallback"""
    try:
        import aiosqlite
        return True, "SQLite available as fallback option"
    except ImportError:
        return False, "SQLite support not installed (pip install aiosqlite)"


def print_setup_instructions():
    """Print setup instructions for missing databases"""
    print(f"\n{Colors.BOLD}📋 Setup Instructions:{Colors.END}")
    print("\n" + "=" * 60)

    print(f"\n{Colors.BLUE}Option 1: Docker Compose (Recommended){Colors.END}")
    print("Run from project root:")
    print(f"{Colors.YELLOW}docker-compose up -d database redis{Colors.END}")

    print(f"\n{Colors.BLUE}Option 2: Local Installation{Colors.END}")

    print("\nPostgreSQL (macOS):")
    print(f"{Colors.YELLOW}brew install postgresql@15")
    print("brew services start postgresql@15")
    print("createdb goldensignals")
    print(f"psql goldensignals -c \"CREATE USER goldensignals WITH PASSWORD 'your_password';\"{Colors.END}")

    print("\nRedis (macOS):")
    print(f"{Colors.YELLOW}brew install redis")
    print(f"brew services start redis{Colors.END}")

    print(f"\n{Colors.BLUE}Option 3: Development Mode (No PostgreSQL needed){Colors.END}")
    print("Add to .env file:")
    print(f"{Colors.YELLOW}DATABASE_URL=sqlite+aiosqlite:///./goldensignals.db{Colors.END}")

    print("\n" + "=" * 60)
    print(f"\n{Colors.BOLD}📖 Full setup guide: DATABASE_SETUP_GUIDE.md{Colors.END}")


async def main():
    """Main function"""
    print_header()

    # Check Python packages
    print(f"\n{Colors.BOLD}📦 Python Packages:{Colors.END}")
    print_status("asyncpg", ASYNCPG_AVAILABLE)
    print_status("redis-py", REDIS_AVAILABLE)
    print_status("sqlalchemy", SQLALCHEMY_AVAILABLE)

    # Check database connections
    print(f"\n{Colors.BOLD}🔌 Database Connections:{Colors.END}")

    # PostgreSQL
    pg_status, pg_message = await check_postgresql()
    print_status("PostgreSQL", pg_status, pg_message)

    # Redis
    redis_status, redis_message = check_redis()
    print_status("Redis", redis_status, redis_message)

    # SQLite fallback
    sqlite_status, sqlite_message = check_sqlite_fallback()
    print_status("SQLite (Fallback)", sqlite_status, sqlite_message)

    # Overall status
    all_good = pg_status and redis_status
    fallback_available = sqlite_status and redis_status

    print("\n" + "=" * 60)

    if all_good:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ All databases are ready!{Colors.END}")
        print(f"\n{Colors.GREEN}You can now start the application:{Colors.END}")
        print(f"{Colors.YELLOW}python main.py{Colors.END}")
    elif fallback_available and not pg_status:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  PostgreSQL not available, but SQLite fallback is ready{Colors.END}")
        print(f"\n{Colors.YELLOW}For development, you can use SQLite by setting:{Colors.END}")
        print(f"{Colors.YELLOW}DATABASE_URL=sqlite+aiosqlite:///./goldensignals.db{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Database setup required{Colors.END}")
        print_setup_instructions()

    # Environment file check
    if not os.path.exists('.env'):
        print(f"\n{Colors.YELLOW}⚠️  No .env file found. Create one with:{Colors.END}")
        print(f"{Colors.YELLOW}cp .env.example .env{Colors.END}")
        print(f"{Colors.YELLOW}Then edit it with your database credentials{Colors.END}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        sys.exit(1)

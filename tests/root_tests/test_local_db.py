#!/usr/bin/env python3
"""
Test Local PostgreSQL Connection
"""

import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Local Database Connection details
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "goldensignalsai"
DB_USER = "goldensignalsai"
DB_PASSWORD = "goldensignals123"

# Connection URLs
ASYNCPG_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
SQLALCHEMY_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


async def test_asyncpg_connection():
    """Test direct asyncpg connection"""
    print("\n🔍 Testing asyncpg connection to local PostgreSQL...")
    try:
        # Connect to the database
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            timeout=5
        )

        # Test query
        version = await conn.fetchval('SELECT version()')
        print(f"✅ Connected successfully!")
        print(f"   PostgreSQL version: {version}")

        # Check if we can create tables
        can_create = await conn.fetchval("""
            SELECT has_database_privilege($1, 'CREATE')
        """, DB_NAME)
        print(f"   Can create objects: {can_create}")

        # List existing tables
        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)

        if tables:
            print(f"\n📋 Existing tables ({len(tables)}):")
            for table in tables[:10]:  # Show first 10 tables
                print(f"   - {table['tablename']}")
            if len(tables) > 10:
                print(f"   ... and {len(tables) - 10} more")
        else:
            print("\n📋 No tables found (database is empty)")

        await conn.close()
        return True

    except Exception as e:
        print(f"❌ Connection failed: {type(e).__name__}: {e}")
        return False


async def test_sqlalchemy_connection():
    """Test SQLAlchemy async connection"""
    print("\n🔍 Testing SQLAlchemy connection...")
    try:
        # Create async engine
        engine = create_async_engine(
            SQLALCHEMY_URL,
            echo=False,
            pool_pre_ping=True
        )

        # Test connection
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print("✅ SQLAlchemy connection successful!")

            # Check database size
            result = await conn.execute(text("""
                SELECT pg_database_size(:dbname) as size
            """), {"dbname": DB_NAME})
            size = result.scalar()
            size_mb = size / (1024 * 1024) if size else 0
            print(f"   Database size: {size_mb:.2f} MB")

        await engine.dispose()
        return True

    except Exception as e:
        print(f"❌ SQLAlchemy connection failed: {type(e).__name__}: {e}")
        return False


async def create_tables_if_needed():
    """Create the necessary tables if they don't exist"""
    print("\n🔨 Checking/Creating required tables...")

    engine = create_async_engine(SQLALCHEMY_URL, echo=False)

    try:
        async with engine.begin() as conn:
            # Check if signals table exists
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'signals'
                )
            """))
            signals_exists = result.scalar()

            if not signals_exists:
                print("   Creating signals table...")
                await conn.execute(text("""
                    CREATE TABLE signals (
                        id SERIAL PRIMARY KEY,
                        signal_id VARCHAR(36) UNIQUE NOT NULL,
                        symbol VARCHAR(10) NOT NULL,
                        signal_type VARCHAR(10) NOT NULL,
                        strength VARCHAR(10) NOT NULL,
                        confidence FLOAT NOT NULL,
                        source VARCHAR(50) NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """))
                await conn.execute(text("CREATE INDEX idx_signals_symbol ON signals(symbol)"))
                await conn.execute(text("CREATE INDEX idx_signals_created_at ON signals(created_at)"))
                print("   ✅ Signals table created")
            else:
                print("   ✅ Signals table already exists")

            # Create users table
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'users'
                )
            """))
            users_exists = result.scalar()

            if not users_exists:
                print("   Creating users table...")
                await conn.execute(text("""
                    CREATE TABLE users (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        username VARCHAR(100) UNIQUE NOT NULL,
                        hashed_password VARCHAR(255) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        is_superuser BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """))
                print("   ✅ Users table created")
            else:
                print("   ✅ Users table already exists")

            # Create portfolios table
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'portfolios'
                )
            """))
            portfolios_exists = result.scalar()

            if not portfolios_exists:
                print("   Creating portfolios table...")
                await conn.execute(text("""
                    CREATE TABLE portfolios (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        name VARCHAR(100) NOT NULL,
                        description TEXT,
                        initial_capital DECIMAL(15,2) DEFAULT 100000,
                        current_value DECIMAL(15,2),
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """))
                print("   ✅ Portfolios table created")
            else:
                print("   ✅ Portfolios table already exists")

    except Exception as e:
        print(f"   ❌ Error creating tables: {e}")
    finally:
        await engine.dispose()


async def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 Local PostgreSQL Connection Test")
    print("=" * 60)
    print(f"\n📍 Host: {DB_HOST}")
    print(f"📍 Database: {DB_NAME}")
    print(f"📍 User: {DB_USER}")

    # Test connections
    asyncpg_ok = await test_asyncpg_connection()
    sqlalchemy_ok = await test_sqlalchemy_connection()

    if asyncpg_ok and sqlalchemy_ok:
        # Create tables
        await create_tables_if_needed()

        print("\n" + "=" * 60)
        print("✅ All tests passed! Your local database is ready to use.")
        print("\n🎯 Next steps:")
        print("1. The database connection is configured and working")
        print("2. You can now start the application with: python src/main.py")
        print("3. The application will automatically create any missing tables")
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed. Please check the error messages above.")
        print("\n💡 To set up PostgreSQL locally, run:")
        print("   chmod +x setup_local_db.sh")
        print("   ./setup_local_db.sh")


if __name__ == "__main__":
    asyncio.run(main())

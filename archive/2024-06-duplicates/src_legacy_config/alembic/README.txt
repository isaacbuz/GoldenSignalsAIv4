This directory contains Alembic migration scripts for your database schema.

If the directory structure does not exist, create it as follows:

backend/alembic/
  ├── env.py
  └── versions/
        └── 0001_initial.py

You can run migrations with:
  alembic upgrade head

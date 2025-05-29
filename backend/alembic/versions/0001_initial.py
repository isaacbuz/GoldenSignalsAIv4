"""Initial DB schema for agent configs, feedback, retrain logs

Revision ID: 0001
Revises: 
Create Date: 2025-05-27 23:33:45
"""
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'agent_configs',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('agent', sa.String, unique=True, index=True),
        sa.Column('config', sa.JSON),
        sa.Column('updated_at', sa.DateTime),
    )
    op.create_table(
        'feedback_entries',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String),
        sa.Column('agent', sa.String),
        sa.Column('action', sa.String),
        sa.Column('rating', sa.Float),
        sa.Column('comment', sa.String),
        sa.Column('timestamp', sa.DateTime),
    )
    op.create_table(
        'retrain_logs',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('agent', sa.String),
        sa.Column('status', sa.String),
        sa.Column('output', sa.String),
        sa.Column('timestamp', sa.DateTime),
    )

def downgrade():
    op.drop_table('retrain_logs')
    op.drop_table('feedback_entries')
    op.drop_table('agent_configs')

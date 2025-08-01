"""
Base Model - GoldenSignalsAI V3

Base SQLAlchemy model with common functionality.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class BaseModel(Base):
    """
    Base model class with common fields and methods
    """

    __abstract__ = True

    def to_dict(self):
        """Convert model instance to dictionary"""
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def update_from_dict(self, data: dict):
        """Update model instance from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Export Base for use in other models
__all__ = ["BaseModel", "Base"]

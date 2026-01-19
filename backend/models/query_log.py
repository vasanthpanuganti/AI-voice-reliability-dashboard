"""Query log model for storing patient queries"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, Index
from sqlalchemy.sql import func
from backend.database import Base

class QueryLog(Base):
    """Stores patient queries with embeddings and metadata"""
    __tablename__ = "query_logs"
    __table_args__ = (
        Index('ix_query_logs_category_timestamp', 'query_category', 'timestamp'),
    )

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False, index=True)
    query_category = Column(String(50), index=True)
    embedding = Column(JSON)
    confidence_score = Column(Float, nullable=True)
    ai_response = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    department = Column(String(100))
    patient_population = Column(String(50))

"""Query log model for storing patient queries"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from backend.database import Base

class QueryLog(Base):
    """Stores patient queries with embeddings and metadata"""
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False, index=True)
    query_category = Column(String(50), index=True)  # appointment, prescription, billing, etc.
    embedding = Column(JSON)  # Store embedding vector as JSON
    confidence_score = Column(String(20))  # Store as JSON for flexibility
    ai_response = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Additional metadata
    department = Column(String(100))
    patient_population = Column(String(50))

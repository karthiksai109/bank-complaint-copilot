from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Channel(str, Enum):
    online_banking = "online_banking"
    branch = "branch"
    phone = "phone"
    email = "email"


class ComplaintIn(BaseModel):
    customer_ref: str = Field(..., examples=["C-1042"])
    text: str = Field(..., min_length=10, examples=["I was charged an overdraft fee even though my balance was positive..."])
    channel: Channel = Channel.online_banking


class Complaint(ComplaintIn):
    id: int
    category: str
    priority: str
    created_at: datetime


class PredictIn(BaseModel):
    text: str = Field(..., min_length=10)


class PredictOut(BaseModel):
    category: str
    confidence: float
    priority: str


class CategoryStat(BaseModel):
    category: str
    count: int


class StatsOut(BaseModel):
    total: int
    by_category: list[CategoryStat]
    high_priority: int


class DraftReplyIn(BaseModel):
    text: str = Field(..., min_length=10)


class CitationOut(BaseModel):
    source: str
    heading: str
    score: float


class DraftReplyOut(BaseModel):
    category: str
    confidence: float
    priority: str
    draft_answer: str
    citations: list[CitationOut]
    model: Optional[str]
    rag_available: bool

from pydantic import BaseModel, Field
from typing import List, Optional


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    chunk_id: str
    page: int
    text: str
    score: float
    document: str
    section: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: List[int] = []
    sources: List[Source] = []
    confidence: float = 0.0

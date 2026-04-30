from pydantic import BaseModel
from typing import Optional
from app.models import LeaveStatusEnum, LeaveTypeEnum

class PolicyQuestionRequest(BaseModel):
    question: str

class PolicyQuestionResponse(BaseModel):
    answer: str
    sources: list
    retrieved_chunks: int

class LeaveApplyRequest(BaseModel):
    leave_type: LeaveTypeEnum
    start_date: str  # YYYY-MM-DD
    end_date: str
    reason: Optional[str] = ""

class LeaveApprovalRequest(BaseModel):
    request_id: str
    action: str  # "approve" or "reject"
    notes: Optional[str] = ""

class ChatRequest(BaseModel):
    message: str
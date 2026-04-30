from pydantic import BaseModel
from typing import Optional
from app.models import TicketStatusEnum

class TicketCreateRequest(BaseModel):
    issue_type: str
    description: str

class TicketUpdateRequest(BaseModel):
    ticket_id: str
    status: TicketStatusEnum
    resolution_notes: Optional[str] = ""

class AssetRequestCreate(BaseModel):
    asset_type: str
    justification: str
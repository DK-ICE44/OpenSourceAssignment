from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, ITTicket, AssetRequest, TicketStatusEnum, RoleEnum, AssetStatusEnum
from app.middleware.auth import get_current_user
from app.agents.email_agent import send_email
from app.tools.ticket_tools import create_ticket
from app.schemas.it import TicketCreateRequest, TicketUpdateRequest, AssetRequestCreate
import uuid

router = APIRouter(prefix="/it", tags=["IT"])


@router.post("/tickets/create")
async def create_ticket_endpoint(
    request: TicketCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Raise an IT support ticket with deduplication and outage checks."""
    result = create_ticket(
        requester_id=current_user.id,
        issue_type=request.issue_type,
        description=request.description,
        db=db
    )

    if result["ticket_created"]:
        # Notify user via email
        await send_email(
            to=current_user.email,
            subject=f"IT Ticket Created: {result['ticket_id']}",
            body=f"Hi {current_user.full_name},\n\n"
                 f"Your IT ticket has been created.\n\n"
                 f"Ticket ID: {result['ticket_id']}\n"
                 f"Issue: {request.issue_type}\n"
                 f"Priority: {result['priority']}\n\n"
                 f"Our IT team will be in touch shortly.\n\nIT Support Team",
            email_type="ticket_created"
        )

    return result


@router.get("/tickets/my")
def my_tickets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Employee: view own tickets only."""
    tickets = db.query(ITTicket).filter(
        ITTicket.requester_id == current_user.id
    ).order_by(ITTicket.created_at.desc()).all()

    return [
        {
            "ticket_id": t.ticket_id,
            "issue_type": t.issue_type,
            "description": t.description[:100] + "..." if len(t.description) > 100 else t.description,
            "priority": t.priority.value,
            "status": t.status.value,
            "created_at": str(t.created_at)
        }
        for t in tickets
    ]


@router.get("/tickets/all")
def all_tickets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """IT Team / Admin: view all tickets."""
    if current_user.role not in [RoleEnum.it_team, RoleEnum.admin]:
        raise HTTPException(403, "IT Team access only")

    tickets = db.query(ITTicket).order_by(ITTicket.created_at.desc()).all()
    return [
        {
            "ticket_id": t.ticket_id,
            "requester": db.query(User).get(t.requester_id).full_name,
            "issue_type": t.issue_type,
            "priority": t.priority.value,
            "status": t.status.value,
            "assigned_to": db.query(User).get(t.assigned_to).full_name
                           if t.assigned_to else None,
            "created_at": str(t.created_at)
        }
        for t in tickets
    ]


@router.put("/tickets/update")
async def update_ticket(
    request: TicketUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """IT Team: update ticket status and add resolution notes."""
    if current_user.role not in [RoleEnum.it_team, RoleEnum.admin]:
        raise HTTPException(403, "IT Team access only")

    ticket = db.query(ITTicket).filter(
        ITTicket.ticket_id == request.ticket_id
    ).first()
    if not ticket:
        raise HTTPException(404, "Ticket not found")

    ticket.status = request.status
    ticket.resolution_notes = request.resolution_notes
    ticket.assigned_to = current_user.id
    db.commit()

    # Notify requester if resolved
    if request.status == TicketStatusEnum.resolved:
        requester = db.query(User).get(ticket.requester_id)
        if requester:
            await send_email(
                to=requester.email,
                subject=f"IT Ticket {request.ticket_id} Resolved",
                body=f"Hi {requester.full_name},\n\n"
                     f"Your IT ticket {request.ticket_id} has been resolved.\n\n"
                     f"Resolution: {request.resolution_notes or 'Issue resolved.'}\n\n"
                     f"IT Support Team",
                email_type="ticket_resolved"
            )

    return {"ticket_id": request.ticket_id, "status": request.status.value}


@router.post("/assets/request")
async def request_asset(
    request: AssetRequestCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Request an IT asset. Goes to manager approval first."""
    req_id = f"AST{uuid.uuid4().hex[:8].upper()}"
    asset = AssetRequest(
        request_id=req_id,
        requester_id=current_user.id,
        asset_type=request.asset_type,
        justification=request.justification,
        status=AssetStatusEnum.pending_manager
    )
    db.add(asset)
    db.commit()

    # Notify manager
    if current_user.manager_id:
        manager = db.query(User).get(current_user.manager_id)
        if manager:
            await send_email(
                to=manager.email,
                subject=f"[Asset Request] {current_user.full_name} requesting {request.asset_type}",
                body=f"Hi {manager.full_name},\n\n"
                     f"{current_user.full_name} has requested: {request.asset_type}\n"
                     f"Justification: {request.justification}\n"
                     f"Request ID: {req_id}\n\n"
                     f"Please approve via the IT system.",
                email_type="asset_approval"
            )

    return {
        "request_id": req_id,
        "asset_type": request.asset_type,
        "status": "pending_manager",
        "message": "Asset request submitted. Awaiting manager approval."
    }
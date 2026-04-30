import uuid
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models import ITTicket, KnownOutage, TicketStatusEnum, TicketPriorityEnum, User

ISSUE_PRIORITY_MAP = {
    "vpn": "high", "network": "high", "email": "high", "outlook": "high",
    "laptop": "medium", "printer": "low", "software": "medium",
    "monitor": "low", "keyboard": "low"
}

def generate_ticket_id() -> str:
    return f"TKT{uuid.uuid4().hex[:8].upper()}"

def check_known_outage(issue_type: str, db: Session) -> dict | None:
    """Returns active outage info if the issue matches a known system outage."""
    keywords = issue_type.lower().split()
    for keyword in keywords:
        outage = db.query(KnownOutage).filter(
            KnownOutage.system_name.ilike(f"%{keyword}%"),
            KnownOutage.is_active == True
        ).first()
        if outage:
            return {
                "system": outage.system_name,
                "description": outage.description,
                "expected_resolution": str(outage.expected_resolution)
                    if outage.expected_resolution else "TBD"
            }
    return None

def check_duplicate_ticket(requester_id: int, issue_type: str, db: Session) -> dict | None:
    """Check if user already has an open ticket for same issue type."""
    existing = db.query(ITTicket).filter(
        ITTicket.requester_id == requester_id,
        ITTicket.issue_type.ilike(f"%{issue_type.split()[0]}%"),
        ITTicket.status.in_([TicketStatusEnum.open, TicketStatusEnum.in_progress])
    ).first()
    if existing:
        return {"ticket_id": existing.ticket_id, "status": existing.status.value}
    return None

def create_ticket(requester_id: int, issue_type: str, description: str,
                   db: Session) -> dict:
    # Check known outage first
    outage = check_known_outage(issue_type, db)
    if outage:
        return {
            "ticket_created": False,
            "message": f"⚠️ Known outage detected for {outage['system']}: "
                       f"{outage['description']}. "
                       f"Expected resolution: {outage['expected_resolution']}. "
                       f"No ticket raised.",
            "outage": outage
        }

    # Check duplicate
    duplicate = check_duplicate_ticket(requester_id, issue_type, db)
    if duplicate:
        return {
            "ticket_created": False,
            "message": f"You already have an open ticket ({duplicate['ticket_id']}) "
                       f"for a similar issue with status: {duplicate['status']}.",
            "existing_ticket": duplicate
        }

    # Determine priority
    priority_key = issue_type.lower().split()[0]
    priority = ISSUE_PRIORITY_MAP.get(priority_key, "medium")

    ticket = ITTicket(
        ticket_id=generate_ticket_id(),
        requester_id=requester_id,
        issue_type=issue_type,
        description=description,
        priority=TicketPriorityEnum(priority),
        status=TicketStatusEnum.open
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return {
        "ticket_created": True,
        "ticket_id": ticket.ticket_id,
        "priority": priority,
        "status": "open",
        "message": f"Ticket {ticket.ticket_id} created successfully."
    }
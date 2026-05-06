import uuid
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models import ITTicket, KnownOutage, TicketStatusEnum, TicketPriorityEnum, User, AssetRequest, AssetStatusEnum

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


def get_all_open_tickets(db: Session) -> dict:
    """Fetch all open/in_progress tickets for IT team."""
    tickets = db.query(ITTicket).filter(
        ITTicket.status.in_([TicketStatusEnum.open, TicketStatusEnum.in_progress])
    ).order_by(ITTicket.priority.desc(), ITTicket.created_at.desc()).all()

    ticket_list = []
    for ticket in tickets:
        requester = db.query(User).filter(User.id == ticket.requester_id).first()
        requester_name = requester.full_name if requester else "Unknown"
        ticket_list.append({
            "ticket_id": ticket.ticket_id,
            "issue_type": ticket.issue_type,
            "description": ticket.description[:100] + "..." if len(ticket.description) > 100 else ticket.description,
            "priority": ticket.priority.value,
            "status": ticket.status.value,
            "requester": requester_name,
            "created_at": str(ticket.created_at) if ticket.created_at else "N/A"
        })

    return {
        "total": len(ticket_list),
        "open": sum(1 for t in ticket_list if t["status"] == "open"),
        "in_progress": sum(1 for t in ticket_list if t["status"] == "in_progress"),
        "tickets": ticket_list
    }


def get_my_tickets(requester_id: int, db: Session) -> dict:
    """Fetch all tickets for a specific user."""
    tickets = db.query(ITTicket).filter(
        ITTicket.requester_id == requester_id
    ).order_by(ITTicket.created_at.desc()).all()

    ticket_list = []
    for ticket in tickets:
        ticket_list.append({
            "ticket_id": ticket.ticket_id,
            "issue_type": ticket.issue_type,
            "description": ticket.description[:100] + "..." if len(ticket.description) > 100 else ticket.description,
            "priority": ticket.priority.value,
            "status": ticket.status.value,
            "created_at": str(ticket.created_at) if ticket.created_at else "N/A"
        })

    return {
        "total": len(ticket_list),
        "open": sum(1 for t in ticket_list if t["status"] == "open"),
        "in_progress": sum(1 for t in ticket_list if t["status"] == "in_progress"),
        "resolved": sum(1 for t in ticket_list if t["status"] == "resolved"),
        "tickets": ticket_list
    }


def get_inventory_status(db: Session) -> dict:
    """Get asset inventory summary for IT team."""
    all_requests = db.query(AssetRequest).all()

    summary = {
        "pending_manager": 0,
        "pending_it": 0,
        "approved": 0,
        "rejected": 0,
        "fulfilled": 0,
        "total": len(all_requests)
    }

    for r in all_requests:
        summary[r.status.value] += 1

    pending = [
        {
            "request_id": r.request_id,
            "requester": db.query(User).get(r.requester_id).full_name if db.query(User).get(r.requester_id) else "Unknown",
            "asset_type": r.asset_type,
            "status": r.status.value
        }
        for r in all_requests
        if r.status.value in ["pending_manager", "pending_it"]
    ]

    return {"summary": summary, "pending_requests": pending}
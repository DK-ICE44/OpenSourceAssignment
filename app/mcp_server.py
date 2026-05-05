"""
FastMCP Tool Server — HR & IT Copilot
Exposes core business operations as MCP-compatible tools.

Run standalone:
    python -m app.mcp_server

Tools exposed:
    HR  : apply_leave, get_leave_balance, get_leave_requests,
          approve_leave_request, answer_hr_policy
    IT  : create_it_ticket, get_ticket_status, get_all_tickets,
          update_ticket_status, request_asset, approve_asset_request,
          get_asset_requests, inventory_status
"""
import logging
import sys
from typing import Optional
from fastmcp import FastMCP
from starlette.responses import JSONResponse

from app.database import SessionLocal


logger = logging.getLogger(__name__)

# ── MCP Server instance ───────────────────────────────────────────────────────
mcp = FastMCP(
    name="HR-IT Copilot MCP Server",
    instructions="""
    Enterprise HR & IT operations tool server.
    Provides tools for leave management, IT ticket handling,
    asset requests, and HR policy Q&A.
    All employee_id and requester_id values are integer database IDs.
    """
)


# ═══════════════════════════════════════════════════════════════════════════════
# HR TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def apply_leave(
    employee_id: int,
    leave_type: str,
    start_date: str,
    end_date: str,
    reason: str = ""
) -> dict:
    """
    Apply for employee leave with full validation.

    Args:
        employee_id : Integer DB id of the employee
        leave_type  : One of casual | sick | privilege | maternity | paternity
        start_date  : Format YYYY-MM-DD
        end_date    : Format YYYY-MM-DD
        reason      : Optional reason for leave

    Returns:
        dict with request_id, num_days, status
    """
    db = SessionLocal()
    try:
        from app.tools.leave_tools import apply_leave as _apply
        return _apply(employee_id, leave_type, start_date, end_date, reason, db)
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


@mcp.tool()
def get_leave_balance(employee_id: int, year: int = 2026) -> dict:
    """
    Get leave balance for an employee.

    Args:
        employee_id : Integer DB id of the employee
        year        : Year to check (default 2026)

    Returns:
        dict with casual, sick, privilege balances (total/used/available)
    """
    db = SessionLocal()
    try:
        from app.tools.leave_tools import get_leave_balance as _get_balance
        return _get_balance(employee_id, year, db)
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


@mcp.tool()
def get_leave_requests(employee_id: int) -> list:
    """
    Get all leave requests for an employee.

    Args:
        employee_id : Integer DB id of the employee

    Returns:
        List of leave requests with status, dates, type
    """
    db = SessionLocal()
    try:
        from app.models import LeaveRequest
        requests = db.query(LeaveRequest).filter(
            LeaveRequest.employee_id == employee_id
        ).order_by(LeaveRequest.created_at.desc()).all()

        return [
            {
                "request_id": r.request_id,
                "leave_type": r.leave_type.value,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "num_days": r.num_days,
                "status": r.status.value,
                "reason": r.reason,
                "created_at": str(r.created_at)
            }
            for r in requests
        ]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        db.close()


@mcp.tool()
def approve_leave_request(
    request_id: str,
    approver_id: int,
    action: str,
    notes: str = ""
) -> dict:
    """
    Approve or reject a pending leave request.

    Args:
        request_id  : Leave request ID (e.g. LVXXXXXXXX)
        approver_id : Integer DB id of the approving manager
        action      : Either "approve" or "reject"
        notes       : Optional manager notes

    Returns:
        dict with request_id and updated status
    """
    db = SessionLocal()
    try:
        from app.tools.leave_tools import approve_leave as _approve
        return _approve(request_id, approver_id, action, notes, db)
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


@mcp.tool()
def answer_hr_policy(question: str, user_role: str = "employee") -> dict:
    """
    Answer HR policy questions using RAG over the employee handbook.

    Args:
        question  : Natural language policy question
        user_role : Role of the asking user (default: employee)

    Returns:
        dict with answer, sources (page references), and retrieved_chunks count
    """
    try:
        from app.agents.rag_agent import answer_policy_question
        return answer_policy_question(question, user_role)
    except Exception as e:
        return {"error": str(e), "answer": "", "sources": []}


@mcp.tool()
def get_pending_approvals(manager_id: int) -> list:
    """
    Get all pending leave requests for a manager's direct reports.

    Args:
        manager_id : Integer DB id of the manager

    Returns:
        List of pending leave requests from team members
    """
    db = SessionLocal()
    try:
        from app.models import LeaveRequest, User, LeaveStatusEnum
        reports = db.query(User).filter(User.manager_id == manager_id).all()
        report_ids = [r.id for r in reports]

        pending = db.query(LeaveRequest).filter(
            LeaveRequest.employee_id.in_(report_ids),
            LeaveRequest.status == LeaveStatusEnum.pending
        ).all()

        return [
            {
                "request_id": r.request_id,
                "employee_id": r.employee_id,
                "employee_name": db.query(User).get(r.employee_id).full_name,
                "leave_type": r.leave_type.value,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "num_days": r.num_days,
                "reason": r.reason
            }
            for r in pending
        ]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# IT TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def create_it_ticket(
    requester_id: int,
    issue_type: str,
    description: str
) -> dict:
    """
    Raise an IT support ticket with deduplication and outage checks.

    Args:
        requester_id : Integer DB id of the requesting employee
        issue_type   : Short category e.g. VPN, Laptop, Email, Printer, Network
        description  : Detailed description of the issue

    Returns:
        dict with ticket_created bool, ticket_id, priority, status, message
    """
    db = SessionLocal()
    try:
        from app.tools.ticket_tools import create_ticket as _create
        return _create(requester_id, issue_type, description, db)
    except Exception as e:
        return {"error": str(e), "ticket_created": False}
    finally:
        db.close()


@mcp.tool()
def get_ticket_status(ticket_id: str) -> dict:
    """
    Get the current status and details of an IT ticket.

    Args:
        ticket_id : Ticket ID (e.g. TKTXXXXXXXX)

    Returns:
        dict with ticket details including status, priority, assignee
    """
    db = SessionLocal()
    try:
        from app.models import ITTicket, User
        ticket = db.query(ITTicket).filter(
            ITTicket.ticket_id == ticket_id
        ).first()

        if not ticket:
            return {"error": f"Ticket {ticket_id} not found"}

        return {
            "ticket_id": ticket.ticket_id,
            "issue_type": ticket.issue_type,
            "description": ticket.description,
            "priority": ticket.priority.value,
            "status": ticket.status.value,
            "assigned_to": db.query(User).get(ticket.assigned_to).full_name
                           if ticket.assigned_to else None,
            "resolution_notes": ticket.resolution_notes,
            "created_at": str(ticket.created_at)
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


@mcp.tool()
def get_all_tickets(status_filter: Optional[str] = None) -> list:
    """
    Get all IT tickets, optionally filtered by status. IT Team use only.

    Args:
        status_filter : Optional — one of open | in_progress | resolved | closed

    Returns:
        List of all tickets with requester, priority, status
    """
    db = SessionLocal()
    try:
        from app.models import ITTicket, User, TicketStatusEnum
        query = db.query(ITTicket)
        if status_filter:
            query = query.filter(
                ITTicket.status == TicketStatusEnum(status_filter)
            )
        tickets = query.order_by(ITTicket.created_at.desc()).all()

        return [
            {
                "ticket_id": t.ticket_id,
                "requester": db.query(User).get(t.requester_id).full_name,
                "issue_type": t.issue_type,
                "priority": t.priority.value,
                "status": t.status.value,
                "assigned_to": db.query(User).get(t.assigned_to).full_name
                               if t.assigned_to else "Unassigned",
                "created_at": str(t.created_at)
            }
            for t in tickets
        ]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        db.close()


@mcp.tool()
def update_ticket_status(
    ticket_id: str,
    it_member_id: int,
    status: str,
    resolution_notes: str = ""
) -> dict:
    """
    Update an IT ticket's status. IT Team use only.

    Args:
        ticket_id        : Ticket ID (e.g. TKTXXXXXXXX)
        it_member_id     : Integer DB id of the IT team member
        status           : One of open | in_progress | resolved | closed
        resolution_notes : Optional notes on how the issue was resolved

    Returns:
        dict with ticket_id and updated status
    """
    db = SessionLocal()
    try:
        from app.models import ITTicket, TicketStatusEnum
        ticket = db.query(ITTicket).filter(
            ITTicket.ticket_id == ticket_id
        ).first()

        if not ticket:
            return {"error": f"Ticket {ticket_id} not found"}

        ticket.status = TicketStatusEnum(status)
        ticket.assigned_to = it_member_id
        ticket.resolution_notes = resolution_notes
        db.commit()

        return {
            "ticket_id": ticket_id,
            "status": status,
            "assigned_to": it_member_id,
            "message": f"Ticket {ticket_id} updated to {status}"
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


@mcp.tool()
def request_asset(
    requester_id: int,
    asset_type: str,
    justification: str
) -> dict:
    """
    Submit an IT asset request (laptop, monitor, keyboard, etc.).

    Args:
        requester_id  : Integer DB id of the requesting employee
        asset_type    : e.g. Laptop | Monitor | Keyboard | Mouse | VPN Token
        justification : Business reason for the request

    Returns:
        dict with request_id and status pending_manager
    """
    db = SessionLocal()
    try:
        import uuid
        from app.models import AssetRequest, AssetStatusEnum
        req_id = f"AST{uuid.uuid4().hex[:8].upper()}"
        asset = AssetRequest(
            request_id=req_id,
            requester_id=requester_id,
            asset_type=asset_type,
            justification=justification,
            status=AssetStatusEnum.pending_manager
        )
        db.add(asset)
        db.commit()
        return {
            "request_id": req_id,
            "asset_type": asset_type,
            "status": "pending_manager",
            "message": "Asset request submitted. Awaiting manager approval."
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


@mcp.tool()
def approve_asset_request(
    request_id: str,
    approver_id: int,
    approver_role: str,
    action: str
) -> dict:
    """
    Approve or reject an asset request.

    Args:
        request_id    : Asset request ID (e.g. ASTXXXXXXXX)
        approver_id   : Integer DB id of the approver
        approver_role : Either "manager" or "it_team"
        action        : Either "approve" or "reject"

    Returns:
        dict with request_id and updated status
    """
    db = SessionLocal()
    try:
        from app.models import AssetRequest, AssetStatusEnum
        asset = db.query(AssetRequest).filter(
            AssetRequest.request_id == request_id
        ).first()

        if not asset:
            return {"error": f"Asset request {request_id} not found"}

        if action == "reject":
            asset.status = AssetStatusEnum.rejected
            db.commit()
            return {"request_id": request_id, "status": "rejected"}

        # Approve flow
        if approver_role == "manager":
            asset.status = AssetStatusEnum.pending_it
            asset.manager_approved_by = approver_id
        elif approver_role == "it_team":
            asset.status = AssetStatusEnum.approved
            asset.it_approved_by = approver_id

        db.commit()
        return {
            "request_id": request_id,
            "status": asset.status.value,
            "message": f"Asset request {action}d by {approver_role}"
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


@mcp.tool()
def get_asset_requests(requester_id: Optional[int] = None) -> list:
    """
    Get asset requests. If requester_id provided, filters to that employee only.

    Args:
        requester_id : Optional — filter to a specific employee's requests

    Returns:
        List of asset requests with status
    """
    db = SessionLocal()
    try:
        from app.models import AssetRequest, User
        query = db.query(AssetRequest)
        if requester_id:
            query = query.filter(AssetRequest.requester_id == requester_id)
        requests = query.order_by(AssetRequest.created_at.desc()).all()

        return [
            {
                "request_id": r.request_id,
                "requester": db.query(User).get(r.requester_id).full_name,
                "asset_type": r.asset_type,
                "justification": r.justification,
                "status": r.status.value,
                "created_at": str(r.created_at)
            }
            for r in requests
        ]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        db.close()


@mcp.tool()
def inventory_status() -> dict:
    """
    Get a summary of all IT asset requests grouped by status.
    Useful for IT team to understand pending workload.

    Returns:
        dict with counts per status and list of pending requests
    """
    db = SessionLocal()
    try:
        from app.models import AssetRequest, AssetStatusEnum, User
        from sqlalchemy import func

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
                "requester": db.query(User).get(r.requester_id).full_name,
                "asset_type": r.asset_type,
                "status": r.status.value
            }
            for r in all_requests
            if r.status.value in ["pending_manager", "pending_it"]
        ]

        return {"summary": summary, "pending_requests": pending}
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


@mcp.custom_route("/health", methods=["GET"])
async def health_check(_request):
    """Simple health endpoint for runtime checks."""
    return JSONResponse({"status": "ok", "service": "hr-it-copilot-mcp"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stdout_encoding = (sys.stdout.encoding or "").lower()
    if "utf" in stdout_encoding:
        print("🔧 Starting FastMCP Tool Server on port 8001...")
    else:
        print("Starting FastMCP Tool Server on port 8001...")
    print("   Tools available:")
    print("   HR : apply_leave, get_leave_balance, get_leave_requests,")
    print("        approve_leave_request, answer_hr_policy, get_pending_approvals")
    print("   IT : create_it_ticket, get_ticket_status, get_all_tickets,")
    print("        update_ticket_status, request_asset, approve_asset_request,")
    print("        get_asset_requests, inventory_status")
    try:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
    except ValueError as e:
        if "Unknown transport: streamable-http" not in str(e):
            raise
        print("   streamable-http transport unavailable in installed fastmcp; using SSE.")
        mcp.run(transport="sse", host="0.0.0.0", port=8001)
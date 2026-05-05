"""
Admin analytics endpoints.
Accessible only by admin and HR/IT team leads.
Provides leave summaries, ticket stats, and audit logs.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, date

from app.database import get_db
from app.middleware.auth import get_current_user
from app.middleware.rbac import require_hr_or_admin, require_it_or_admin, require_admin_only
from app.models import (
    User, LeaveRequest, ITTicket, AssetRequest, AuditLog,
    LeaveStatusEnum, TicketStatusEnum, AssetStatusEnum, RoleEnum
)

router = APIRouter(prefix="/admin", tags=["Admin Analytics"])


@router.get("/stats/overview")
def overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """High-level system overview. Admin only."""
    if current_user.role not in [RoleEnum.admin, RoleEnum.hr_team, RoleEnum.it_team]:
        raise HTTPException(403, "Not authorized")

    total_users    = db.query(func.count(User.id)).scalar()
    total_leaves   = db.query(func.count(LeaveRequest.id)).scalar()
    total_tickets  = db.query(func.count(ITTicket.id)).scalar()
    total_assets   = db.query(func.count(AssetRequest.id)).scalar()
    total_logs     = db.query(func.count(AuditLog.id)).scalar()

    pending_leaves  = db.query(func.count(LeaveRequest.id)).filter(
        LeaveRequest.status == LeaveStatusEnum.pending).scalar()
    open_tickets    = db.query(func.count(ITTicket.id)).filter(
        ITTicket.status == TicketStatusEnum.open).scalar()
    pending_assets  = db.query(func.count(AssetRequest.id)).filter(
        AssetRequest.status.in_([
            AssetStatusEnum.pending_manager,
            AssetStatusEnum.pending_it
        ])).scalar()

    return {
        "users": total_users,
        "leave_requests": {
            "total": total_leaves,
            "pending": pending_leaves
        },
        "it_tickets": {
            "total": total_tickets,
            "open": open_tickets
        },
        "asset_requests": {
            "total": total_assets,
            "pending": pending_assets
        },
        "audit_log_entries": total_logs
    }


@router.get("/stats/leaves")
def leave_stats(
    year: int = None,
    current_user: User = Depends(require_hr_or_admin),
    db: Session = Depends(get_db)
):
    """Leave statistics by type and status. HR Team + Admin."""
    if year is None:
        year = date.today().year

    all_leaves = db.query(LeaveRequest).filter(
        LeaveRequest.start_date >= f"{year}-01-01",
        LeaveRequest.start_date <= f"{year}-12-31"
    ).all()

    by_type = {}
    by_status = {}
    by_department = {}

    for leave in all_leaves:
        lt = leave.leave_type.value
        st = leave.status.value
        by_type[lt] = by_type.get(lt, 0) + 1
        by_status[st] = by_status.get(st, 0) + 1

        employee = db.query(User).get(leave.employee_id)
        if employee and employee.department:
            dept = employee.department
            by_department[dept] = by_department.get(dept, 0) + 1

    total_days = sum(l.num_days for l in all_leaves
                     if l.status == LeaveStatusEnum.approved)

    return {
        "year": year,
        "total_requests": len(all_leaves),
        "total_approved_days": total_days,
        "by_leave_type": by_type,
        "by_status": by_status,
        "by_department": by_department
    }


@router.get("/stats/tickets")
def ticket_stats(
    current_user: User = Depends(require_it_or_admin),
    db: Session = Depends(get_db)
):
    """IT ticket statistics. IT Team + Admin."""
    all_tickets = db.query(ITTicket).all()

    by_status   = {}
    by_priority = {}
    by_type     = {}

    for t in all_tickets:
        st = t.status.value
        pr = t.priority.value
        by_status[st]   = by_status.get(st, 0) + 1
        by_priority[pr] = by_priority.get(pr, 0) + 1

        issue_key = t.issue_type.split()[0].lower()
        by_type[issue_key] = by_type.get(issue_key, 0) + 1

    # Average resolution time (resolved tickets only)
    resolved = [t for t in all_tickets if t.status == TicketStatusEnum.resolved
                and t.updated_at and t.created_at]
    avg_resolution_hours = None
    if resolved:
        total_hours = sum(
            (t.updated_at - t.created_at).total_seconds() / 3600
            for t in resolved
        )
        avg_resolution_hours = round(total_hours / len(resolved), 1)

    return {
        "total_tickets": len(all_tickets),
        "by_status": by_status,
        "by_priority": by_priority,
        "top_issue_types": dict(
            sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:5]
        ),
        "resolved_tickets": len(resolved),
        "avg_resolution_hours": avg_resolution_hours
    }


@router.get("/audit-logs")
def audit_logs(
    limit: int = 50,
    endpoint_filter: str = None,
    current_user: User = Depends(require_admin_only),
    db: Session = Depends(get_db)
):
    """View recent audit logs. Admin only."""
    query = db.query(AuditLog).order_by(AuditLog.timestamp.desc())

    if endpoint_filter:
        query = query.filter(AuditLog.endpoint.ilike(f"%{endpoint_filter}%"))

    logs = query.limit(limit).all()

    return [
        {
            "id": log.id,
            "user_id": log.user_id,
            "timestamp": str(log.timestamp),
            "endpoint": log.endpoint,
            "agent_used": log.agent_used,
            "tool_used": log.tool_used,
            "response_time_ms": log.response_time_ms,
            "status_code": log.status_code
        }
        for log in logs
    ]
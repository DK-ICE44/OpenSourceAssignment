import uuid
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models import (LeaveRequest, LeaveBalance, HolidayCalendar,
                         LeaveStatusEnum, LeaveTypeEnum, User, RoleEnum)

def generate_leave_id() -> str:
    return f"LV{uuid.uuid4().hex[:8].upper()}"

def get_working_days(start: date, end: date, db: Session) -> int:
    """Count working days excluding weekends and holidays."""
    holidays = {
        h.date for h in db.query(HolidayCalendar).all()
    }
    count = 0
    current = start
    while current <= end:
        if current.weekday() < 5 and str(current) not in holidays:
            count += 1
        current += timedelta(days=1)
    return count

def check_overlapping_leave(employee_id: int, start_date: str,
                              end_date: str, db: Session) -> bool:
    """Returns True if overlap exists."""
    existing = db.query(LeaveRequest).filter(
        LeaveRequest.employee_id == employee_id,
        LeaveRequest.status.in_([LeaveStatusEnum.pending, LeaveStatusEnum.approved]),
        LeaveRequest.start_date <= end_date,
        LeaveRequest.end_date >= start_date
    ).first()
    return existing is not None

def apply_leave(employee_id: int, leave_type: str, start_date: str,
                end_date: str, reason: str, db: Session) -> dict:
    """Create a new leave request with validations."""

    # Parse dates
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")

    if end < start:
        raise HTTPException(400, "End date cannot be before start date")

    if start < date.today():
        raise HTTPException(400, "Cannot apply leave for past dates")

    # Check overlaps
    if check_overlapping_leave(employee_id, start_date, end_date, db):
        raise HTTPException(409, "You already have a leave request for overlapping dates")

    # Count working days
    num_days = get_working_days(start, end, db)
    if num_days == 0:
        raise HTTPException(400, "Selected dates are all weekends/holidays")

    # Check balance
    balance = db.query(LeaveBalance).filter(
        LeaveBalance.employee_id == employee_id,
        LeaveBalance.year == start.year
    ).first()

    if balance:
        if leave_type == "casual":
            available = balance.casual_total - balance.casual_used
        elif leave_type == "sick":
            available = balance.sick_total - balance.sick_used
        elif leave_type == "privilege":
            available = balance.privilege_total - balance.privilege_used
        else:
            available = 999  # Maternity/paternity — no balance check here

        if leave_type in ["casual", "sick", "privilege"] and num_days > available:
            raise HTTPException(
                400,
                f"Insufficient {leave_type} leave balance. "
                f"Available: {available} days, Requested: {num_days} days"
            )

    leave = LeaveRequest(
        request_id=generate_leave_id(),
        employee_id=employee_id,
        leave_type=LeaveTypeEnum(leave_type),
        start_date=start_date,
        end_date=end_date,
        num_days=num_days,
        reason=reason,
        status=LeaveStatusEnum.pending
    )
    db.add(leave)
    db.commit()
    db.refresh(leave)
    return {"request_id": leave.request_id, "num_days": num_days, "status": "pending"}

def get_leave_balance(employee_id: int, year: int, db: Session) -> dict:
    balance = db.query(LeaveBalance).filter(
        LeaveBalance.employee_id == employee_id,
        LeaveBalance.year == year
    ).first()
    if not balance:
        raise HTTPException(404, "Leave balance not found")
    return {
        "year": year,
        "casual": {"total": balance.casual_total, "used": balance.casual_used,
                   "available": balance.casual_total - balance.casual_used},
        "sick": {"total": balance.sick_total, "used": balance.sick_used,
                 "available": balance.sick_total - balance.sick_used},
        "privilege": {"total": balance.privilege_total, "used": balance.privilege_used,
                      "available": balance.privilege_total - balance.privilege_used}
    }

def approve_leave(request_id: str, approver_id: int,
                  action: str, notes: str, db: Session) -> dict:
    leave = db.query(LeaveRequest).filter(
        LeaveRequest.request_id == request_id
    ).first()
    if not leave:
        raise HTTPException(404, "Leave request not found")
    if leave.status != LeaveStatusEnum.pending:
        raise HTTPException(400, f"Leave is already {leave.status.value}")

    if action == "approve":
        leave.status = LeaveStatusEnum.approved
        leave.approved_by = approver_id
        # Deduct from balance
        balance = db.query(LeaveBalance).filter(
            LeaveBalance.employee_id == leave.employee_id,
            LeaveBalance.year == date.fromisoformat(leave.start_date).year
        ).first()
        if balance:
            if leave.leave_type.value == "casual":
                balance.casual_used += leave.num_days
            elif leave.leave_type.value == "sick":
                balance.sick_used += leave.num_days
            elif leave.leave_type.value == "privilege":
                balance.privilege_used += leave.num_days
    else:
        leave.status = LeaveStatusEnum.rejected

    leave.manager_notes = notes
    db.commit()
    return {"request_id": request_id, "status": leave.status.value}


def get_my_leave_requests(employee_id: int, db: Session) -> list:
    """Fetch all leave requests for an employee with count summary."""
    leaves = db.query(LeaveRequest).filter(
        LeaveRequest.employee_id == employee_id
    ).order_by(LeaveRequest.created_at.desc()).all()

    leave_list = []
    for leave in leaves:
        leave_list.append({
            "request_id": leave.request_id,
            "leave_type": leave.leave_type.value,
            "start_date": leave.start_date,
            "end_date": leave.end_date,
            "num_days": leave.num_days,
            "status": leave.status.value,
            "reason": leave.reason or "N/A",
            "created_at": str(leave.created_at) if leave.created_at else "N/A"
        })

    total_count = len(leave_list)
    pending_count = sum(1 for l in leave_list if l["status"] == "pending")
    approved_count = sum(1 for l in leave_list if l["status"] == "approved")
    cancelled_count = sum(1 for l in leave_list if l["status"] == "cancelled")

    return {
        "total": total_count,
        "pending": pending_count,
        "approved": approved_count,
        "cancelled": cancelled_count,
        "leaves": leave_list
    }


def get_company_leave_stats(db: Session) -> dict:
    """Get company-wide leave statistics for HR/Admin."""
    all_leaves = db.query(LeaveRequest).all()

    stats = {
        "total": len(all_leaves),
        "pending": sum(1 for l in all_leaves if l.status == LeaveStatusEnum.pending),
        "approved": sum(1 for l in all_leaves if l.status == LeaveStatusEnum.approved),
        "cancelled": sum(1 for l in all_leaves if l.status == LeaveStatusEnum.cancelled),
        "rejected": sum(1 for l in all_leaves if l.status == LeaveStatusEnum.rejected),
    }

    # Group by leave type
    leave_types = {}
    for l in all_leaves:
        lt = l.leave_type.value
        if lt not in leave_types:
            leave_types[lt] = {"total": 0, "pending": 0, "approved": 0}
        leave_types[lt]["total"] += 1
        if l.status == LeaveStatusEnum.pending:
            leave_types[lt]["pending"] += 1
        elif l.status == LeaveStatusEnum.approved:
            leave_types[lt]["approved"] += 1

    return {"summary": stats, "by_type": leave_types}


def get_pending_approvals(approver_id: int, approver_role: str, db: Session) -> dict:
    """Get all pending approvals for a manager or HR team member."""
    pending_leaves = db.query(LeaveRequest).filter(
        LeaveRequest.status == LeaveStatusEnum.pending
    ).all()

    approvals = []
    for leave in pending_leaves:
        employee = db.query(User).get(leave.employee_id)
        if not employee:
            continue

        # Manager can see their direct reports' requests
        # HR/Admin can see all
        if approver_role in ["hr_team", "admin"]:
            approvals.append({
                "type": "leave",
                "request_id": leave.request_id,
                "requester": employee.full_name,
                "requester_dept": employee.department or "N/A",
                "leave_type": leave.leave_type.value,
                "start_date": leave.start_date,
                "end_date": leave.end_date,
                "num_days": leave.num_days,
                "reason": leave.reason or "N/A"
            })
        elif approver_role == "manager" and employee.manager_id == approver_id:
            approvals.append({
                "type": "leave",
                "request_id": leave.request_id,
                "requester": employee.full_name,
                "requester_dept": employee.department or "N/A",
                "leave_type": leave.leave_type.value,
                "start_date": leave.start_date,
                "end_date": leave.end_date,
                "num_days": leave.num_days,
                "reason": leave.reason or "N/A"
            })

    return {
        "total": len(approvals),
        "approvals": approvals
    }
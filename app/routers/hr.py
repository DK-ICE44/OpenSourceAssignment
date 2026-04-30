from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import date

from app.database import get_db
from app.models import User, LeaveRequest, RoleEnum
from app.middleware.auth import get_current_user
from app.middleware.rbac import require_manager_or_above
from app.agents.rag_agent import answer_policy_question
from app.agents.email_agent import send_email, build_leave_approval_email
from app.agents.router_agent import classify_intent
from app.tools.leave_tools import (apply_leave, get_leave_balance,
                                    approve_leave)
from app.schemas.hr import (PolicyQuestionRequest, PolicyQuestionResponse,
                              LeaveApplyRequest, LeaveApprovalRequest, ChatRequest)

router = APIRouter(prefix="/hr", tags=["HR"])


@router.post("/policy/ask", response_model=PolicyQuestionResponse)
def ask_policy(
    request: PolicyQuestionRequest,
    current_user: User = Depends(get_current_user)
):
    """Ask any HR policy question — answered via RAG over employee handbook."""
    return answer_policy_question(request.question, current_user.role.value)


@router.post("/chat")
def hr_chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Natural language HR chat endpoint.
    Routes to appropriate HR function via intent classification.
    """
    intent_result = classify_intent(request.message)
    intent = intent_result.get("intent", "general")

    if intent == "hr_policy":
        result = answer_policy_question(request.message, current_user.role.value)
        return {"intent": intent, "response": result["answer"],
                "sources": result["sources"]}

    elif intent == "leave_balance":
        try:
            balance = get_leave_balance(current_user.id, date.today().year, db)
            return {"intent": intent, "response": balance}
        except HTTPException:
            return {
                "intent": intent,
                "response": "No leave balance record found for the current year. "
                            "Please contact HR to set up your leave balance."
            }

    elif intent == "leave_apply":
        return {
            "intent": intent,
            "response": "To apply for leave, please use POST /hr/leave/apply with "
                        "leave_type, start_date, end_date, and reason.",
            "hint": "Use the structured endpoint for leave application."
        }

    return {"intent": "general", "response": "How can I help you with HR matters today?"}


@router.post("/leave/apply")
async def apply_leave_endpoint(
    request: LeaveApplyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Apply for leave. Requires > 3 days = human approval via email."""
    result = apply_leave(
        employee_id=current_user.id,
        leave_type=request.leave_type.value,
        start_date=request.start_date,
        end_date=request.end_date,
        reason=request.reason,
        db=db
    )

    # Send email to manager if leave requires approval (> 3 days or privilege leave)
    needs_approval = result["num_days"] > 3 or request.leave_type.value == "privilege"
    if needs_approval and current_user.manager_id:
        manager = db.query(User).filter(User.id == current_user.manager_id).first()
        if manager:
            subject, body = build_leave_approval_email(
                employee_name=current_user.full_name,
                manager_name=manager.full_name,
                leave_type=request.leave_type.value,
                start=request.start_date,
                end=request.end_date,
                days=result["num_days"],
                request_id=result["request_id"]
            )
            await send_email(manager.email, subject, body, "leave_approval")

    return {
        **result,
        "approval_required": needs_approval,
        "message": f"Leave request {result['request_id']} submitted successfully."
    }


@router.get("/leave/balance")
def leave_balance(
    year: int = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current leave balance."""
    if year is None:
        year = date.today().year
    return get_leave_balance(current_user.id, year, db)


@router.get("/leave/my-requests")
def my_leave_requests(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """View all leave requests for current user."""
    requests = db.query(LeaveRequest).filter(
        LeaveRequest.employee_id == current_user.id
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


@router.get("/leave/pending-approvals")
def pending_approvals(
    current_user: User = Depends(require_manager_or_above),
    db: Session = Depends(get_db)
):
    """Managers: view all pending leave requests from their team."""
    from app.models import LeaveStatusEnum
    # Get direct reports
    reports = db.query(User).filter(User.manager_id == current_user.id).all()
    report_ids = [r.id for r in reports]

    pending = db.query(LeaveRequest).filter(
        LeaveRequest.employee_id.in_(report_ids),
        LeaveRequest.status == LeaveStatusEnum.pending
    ).all()

    return [
        {
            "request_id": r.request_id,
            "employee": db.query(User).get(r.employee_id).full_name,
            "leave_type": r.leave_type.value,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "num_days": r.num_days,
            "reason": r.reason
        }
        for r in pending
    ]


@router.post("/leave/approve")
async def approve_leave_endpoint(
    request: LeaveApprovalRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_manager_or_above)
):
    """Manager approves or rejects a leave request."""
    if current_user.role not in [RoleEnum.manager, RoleEnum.hr_team, RoleEnum.admin]:
        raise HTTPException(403, "Only managers can approve leave")

    result = approve_leave(
        request_id=request.request_id,
        approver_id=current_user.id,
        action=request.action,
        notes=request.notes,
        db=db
    )

    # Notify employee
    leave = db.query(LeaveRequest).filter(
        LeaveRequest.request_id == request.request_id
    ).first()
    if leave:
        employee = db.query(User).get(leave.employee_id)
        if employee:
            status_word = "approved" if request.action == "approve" else "rejected"
            await send_email(
                to=employee.email,
                subject=f"Your Leave Request {request.request_id} has been {status_word}",
                body=f"Hi {employee.full_name},\n\nYour leave request "
                     f"({request.request_id}) has been {status_word} by "
                     f"{current_user.full_name}.\n\nNotes: {request.notes or 'None'}\n\n"
                     f"Regards,\nHR Copilot",
                email_type="leave_decision"
            )

    return result
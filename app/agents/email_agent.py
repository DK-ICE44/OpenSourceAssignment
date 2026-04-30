"""
Sends emails via Power Automate HTTP trigger webhook.
"""
import httpx
from app.config import get_settings

settings = get_settings()

async def send_email(to: str, subject: str, body: str,
                      email_type: str = "general") -> dict:
    """
    POST to Power Automate HTTP trigger.
    The PA flow reads 'to', 'subject', 'body' from the JSON body
    and sends via Outlook/Office 365.
    """
    webhook_url = settings.power_automate_email_webhook
    if not webhook_url:
        return {"sent": False, "reason": "POWER_AUTOMATE_EMAIL_WEBHOOK not configured"}

    payload = {
        "to": to,
        "subject": subject,
        "body": body,
        "email_type": email_type
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
            return {"sent": True, "status_code": response.status_code}
    except httpx.HTTPError as e:
        return {"sent": False, "reason": str(e)}


def build_leave_approval_email(employee_name: str, manager_name: str,
                                leave_type: str, start: str, end: str,
                                days: int, request_id: str,
                                approval_url: str = "#") -> tuple[str, str]:
    subject = f"[Action Required] Leave Approval Request – {employee_name}"
    body = f"""
Hi {manager_name},

{employee_name} has submitted a leave request and requires your approval.

Details:
- Request ID: {request_id}
- Leave Type: {leave_type.title()}
- From: {start}  To: {end}
- Working Days: {days}

Please approve or reject via the HR system.

Regards,
HR Copilot System
    """.strip()
    return subject, body
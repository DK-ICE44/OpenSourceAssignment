from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean,
    Text, ForeignKey, Enum, Float
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import enum

Base = declarative_base()

# ─── ENUMS ───────────────────────────────────────────────────────────────────

class RoleEnum(str, enum.Enum):
    employee = "employee"
    manager = "manager"
    hr_team = "hr_team"
    it_team = "it_team"
    admin = "admin"

class LeaveStatusEnum(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    cancelled = "cancelled"

class LeaveTypeEnum(str, enum.Enum):
    casual = "casual"
    sick = "sick"
    privilege = "privilege"
    maternity = "maternity"
    paternity = "paternity"

class TicketStatusEnum(str, enum.Enum):
    open = "open"
    in_progress = "in_progress"
    resolved = "resolved"
    closed = "closed"

class TicketPriorityEnum(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class AssetStatusEnum(str, enum.Enum):
    pending_manager = "pending_manager"
    pending_it = "pending_it"
    approved = "approved"
    rejected = "rejected"
    fulfilled = "fulfilled"

# ─── TABLES ──────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(20), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(Enum(RoleEnum), default=RoleEnum.employee, nullable=False)
    department = Column(String(100))
    manager_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    manager = relationship("User", remote_side=[id], backref="reports")
    leave_requests = relationship("LeaveRequest", back_populates="employee",
                                  foreign_keys="LeaveRequest.employee_id")
    tickets = relationship("ITTicket", back_populates="requester",
                           foreign_keys="ITTicket.requester_id")


class LeaveBalance(Base):
    __tablename__ = "leave_balances"
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    year = Column(Integer, nullable=False)
    casual_total = Column(Integer, default=12)
    casual_used = Column(Integer, default=0)
    sick_total = Column(Integer, default=10)
    sick_used = Column(Integer, default=0)
    privilege_total = Column(Integer, default=15)
    privilege_used = Column(Integer, default=0)

    employee = relationship("User")


class HolidayCalendar(Base):
    __tablename__ = "holiday_calendar"
    id = Column(Integer, primary_key=True)
    date = Column(String(10), unique=True, nullable=False)  # YYYY-MM-DD
    name = Column(String(100), nullable=False)
    is_optional = Column(Boolean, default=False)


class LeaveRequest(Base):
    __tablename__ = "leave_requests"
    id = Column(Integer, primary_key=True)
    request_id = Column(String(20), unique=True, index=True)
    employee_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    leave_type = Column(Enum(LeaveTypeEnum), nullable=False)
    start_date = Column(String(10), nullable=False)  # YYYY-MM-DD
    end_date = Column(String(10), nullable=False)
    num_days = Column(Integer, nullable=False)
    reason = Column(Text)
    status = Column(Enum(LeaveStatusEnum), default=LeaveStatusEnum.pending)
    approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    manager_notes = Column(Text)

    employee = relationship("User", foreign_keys=[employee_id],
                            back_populates="leave_requests")
    approver = relationship("User", foreign_keys=[approved_by])


class ITTicket(Base):
    __tablename__ = "it_tickets"
    id = Column(Integer, primary_key=True)
    ticket_id = Column(String(20), unique=True, index=True)
    requester_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    issue_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    priority = Column(Enum(TicketPriorityEnum), default=TicketPriorityEnum.medium)
    status = Column(Enum(TicketStatusEnum), default=TicketStatusEnum.open)
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolution_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    requester = relationship("User", foreign_keys=[requester_id],
                             back_populates="tickets")
    assignee = relationship("User", foreign_keys=[assigned_to])


class KnownOutage(Base):
    __tablename__ = "known_outages"
    id = Column(Integer, primary_key=True)
    system_name = Column(String(100), nullable=False)
    description = Column(Text)
    start_time = Column(DateTime)
    expected_resolution = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)


class AssetRequest(Base):
    __tablename__ = "asset_requests"
    id = Column(Integer, primary_key=True)
    request_id = Column(String(20), unique=True, index=True)
    requester_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset_type = Column(String(100), nullable=False)
    justification = Column(Text)
    status = Column(Enum(AssetStatusEnum), default=AssetStatusEnum.pending_manager)
    manager_approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    it_approved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    requester = relationship("User", foreign_keys=[requester_id])


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    endpoint = Column(String(200))
    query = Column(Text)
    agent_used = Column(String(50))
    tool_used = Column(String(50))
    response_time_ms = Column(Float)
    status_code = Column(Integer)
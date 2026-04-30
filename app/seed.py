"""Run once to populate initial data."""
from app.database import SessionLocal, init_db
from app.models import User, LeaveBalance, HolidayCalendar, KnownOutage, RoleEnum
from passlib.context import CryptContext
from datetime import datetime, date

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def seed():
    init_db()
    db = SessionLocal()

    # Users
    users_data = [
        ("EMP001", "Alice Johnson", "alice@company.com", "password123", RoleEnum.employee, "Engineering"),
        ("EMP002", "Bob Smith",   "bob@company.com",   "password123", RoleEnum.manager,  "Engineering"),
        ("HR001",  "Carol White", "carol@company.com", "password123", RoleEnum.hr_team,  "HR"),
        ("IT001",  "Dave Brown",  "dave@company.com",  "password123", RoleEnum.it_team,  "IT"),
        ("ADM001", "Eve Admin",   "eve@company.com",   "password123", RoleEnum.admin,    "Admin"),
    ]

    created_users = []
    for emp_id, name, email, pwd, role, dept in users_data:
        existing = db.query(User).filter(User.employee_id == emp_id).first()
        if not existing:
            u = User(employee_id=emp_id, full_name=name, email=email,
                     hashed_password=pwd_ctx.hash(pwd[:72]), role=role, department=dept)
            db.add(u)
            db.flush()
            created_users.append(u)

    db.commit()

    # Set Alice's manager to Bob
    alice = db.query(User).filter(User.employee_id == "EMP001").first()
    bob = db.query(User).filter(User.employee_id == "EMP002").first()
    if alice and bob and not alice.manager_id:
        alice.manager_id = bob.id
        db.commit()

    # Leave balances for 2026
    for user in db.query(User).all():
        exists = db.query(LeaveBalance).filter(
            LeaveBalance.employee_id == user.id,
            LeaveBalance.year == 2026
        ).first()
        if not exists:
            db.add(LeaveBalance(employee_id=user.id, year=2026))
    db.commit()

    # Holidays
    holidays = [
        ("2026-01-01", "New Year's Day"),
        ("2026-01-26", "Republic Day"),
        ("2026-08-15", "Independence Day"),
        ("2026-10-02", "Gandhi Jayanti"),
        ("2026-12-25", "Christmas"),
    ]
    for h_date, h_name in holidays:
        if not db.query(HolidayCalendar).filter(HolidayCalendar.date == h_date).first():
            db.add(HolidayCalendar(date=h_date, name=h_name))
    db.commit()

    print("✅ Database seeded successfully!")
    db.close()

if __name__ == "__main__":
    seed()
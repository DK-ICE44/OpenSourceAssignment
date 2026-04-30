from pydantic import BaseModel
from app.models import RoleEnum

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    employee_id: str | None = None

class UserOut(BaseModel):
    id: int
    employee_id: str
    full_name: str
    email: str
    role: RoleEnum
    department: str | None

    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    employee_id: str
    password: str
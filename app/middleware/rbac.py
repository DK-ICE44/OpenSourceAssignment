from fastapi import Depends, HTTPException, status
from app.middleware.auth import get_current_user
from app.models import User, RoleEnum

def require_manager_or_above(current_user: User = Depends(get_current_user)):
    if current_user.role not in [
        RoleEnum.manager,
        RoleEnum.hr_team,
        RoleEnum.it_team,
        RoleEnum.admin
    ]:
        raise HTTPException(status_code=403, detail="Not authorized")
    return current_user
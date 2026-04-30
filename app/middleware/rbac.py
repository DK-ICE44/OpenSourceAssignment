from fastapi import Depends, HTTPException
from app.middleware.auth import get_current_user
from app.models import User, RoleEnum
from typing import List

def require_roles(allowed_roles: List[RoleEnum]):
    """Factory that returns a FastAPI dependency checking roles."""
    def checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required roles: "
                       f"{[r.value for r in allowed_roles]}"
            )
        return current_user
    return checker

# Ready-to-use dependencies
def require_manager_or_above(
    current_user: User = Depends(get_current_user)
) -> User:
    allowed = [RoleEnum.manager, RoleEnum.hr_team, RoleEnum.it_team, RoleEnum.admin]
    if current_user.role not in allowed:
        raise HTTPException(status_code=403, detail="Manager or above required")
    return current_user

require_hr_or_admin = require_roles([RoleEnum.hr_team, RoleEnum.admin])
require_it_or_admin = require_roles([RoleEnum.it_team, RoleEnum.admin])
require_admin_only  = require_roles([RoleEnum.admin])
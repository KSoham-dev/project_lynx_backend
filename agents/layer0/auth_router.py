import uuid
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from agents.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_password_hash,
    verify_password,
    get_current_user,
    oauth2_scheme,
    SECRET_KEY,
    ALGORITHM
)
import jwt
from agents.enums import UserRole
from fastapi.security import OAuth2PasswordRequestForm
from agents.state.data_schemas import UserDocument
from agents.state.data_store import get_user_by_email, upsert_user

router = APIRouter(prefix="/auth", tags=["Authentication"])

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    mobile_number: Optional[str] = None
    device_fsm_token: Optional[str] = None
    role: Optional[UserRole] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    role: UserRole

@router.post("/signup", response_model=Token)
async def signup(request: SignupRequest):
    existing_user = await get_user_by_email(request.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(request.password)
    
    # Use explicitly provided role, else automatically assign based on email domain
    if request.role is not None:
        role = request.role
    else:
        role = UserRole.RANGER if request.email.endswith("@prahari.gov") else UserRole.PUBLIC
    
    new_user = UserDocument(
        id=user_id,
        user_id=user_id,
        name=request.name,
        email=request.email,
        mobile_number=request.mobile_number,
        hashed_password=hashed_password,
        device_fsm_token=request.device_fsm_token,
        role=role
    )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token, jti, expire = create_access_token(
        data={"sub": new_user.user_id}, expires_delta=access_token_expires
    )
    
    # Store the JTI in the user document to track this session
    new_user.add_session(jti, expire)
    await upsert_user(new_user)
    
    return {"access_token": access_token, "token_type": "bearer", "role": new_user.role}

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Standard OAuth2 Login.
    NOTE: Please enter your **EMAIL** address in the 'username' field.
    """
    user = await get_user_by_email(form_data.username)
    if not user or not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token, jti, expire = create_access_token(
        data={"sub": user.user_id}, expires_delta=access_token_expires
    )
    
    # Store the JTI in the user document to track this session
    user.add_session(jti, expire)
    await upsert_user(user)

    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

@router.post("/logout")
async def logout(
    current_user: UserDocument = Depends(get_current_user),
    token: str = Depends(oauth2_scheme)
):
    """
    Stateful Logout endpoint.
    Retrieves the exact JTI from the provided token and removes it from the 
    user's authorized active sessions list in Cosmos DB, effectively revoking access.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        if jti:
            current_user.remove_session(jti)
            await upsert_user(current_user)
    except jwt.PyJWTError:
        pass
        
    return {"message": "Successfully logged out and session revoked"}

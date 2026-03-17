import asyncio
from agents.state.data_store import get_user_by_email, get_all_users
from agents.auth import verify_password, get_password_hash

async def main():
    print("All users:")
    users = await get_all_users()
    print(users)
    print("---")
    
    if users:
        email = users[0].get("email")
        print(f"Testing for {email}")
        user = await get_user_by_email(email)
        print("User found by email:", user is not None)
        if user:
            print("Hashed pwd:", user.hashed_password)

if __name__ == "__main__":
    asyncio.run(main())

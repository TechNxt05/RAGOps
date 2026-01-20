"""
Seed script to create default Admin and Client users for demo purposes.
Run this script to ensure default credentials exist in the database.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent))

from sqlmodel import Session, select
from app.db import engine
from app.models.user import User, UserRole
from app.auth.security import get_password_hash

def seed_default_users():
    """Create default admin and client users if they don't exist."""
    with Session(engine) as session:
        # Default Admin User
        admin_email = "admin@ragops.com"
        admin = session.exec(select(User).where(User.email == admin_email)).first()
        
        if not admin:
            admin = User(
                email=admin_email,
                hashed_password=get_password_hash("admin123"),
                role=UserRole.ADMIN
            )
            session.add(admin)
            print(f"✅ Created default Admin user: {admin_email} / admin123")
        else:
            print(f"ℹ️  Admin user already exists: {admin_email}")
        
        # Default Client User
        client_email = "client@ragops.com"
        client = session.exec(select(User).where(User.email == client_email)).first()
        
        if not client:
            client = User(
                email=client_email,
                hashed_password=get_password_hash("client123"),
                role=UserRole.CLIENT
            )
            session.add(client)
            print(f"✅ Created default Client user: {client_email} / client123")
        else:
            print(f"ℹ️  Client user already exists: {client_email}")
        
        session.commit()
        print("\n🎉 User seeding complete!")

if __name__ == "__main__":
    print("🌱 Seeding default users...")
    seed_default_users()

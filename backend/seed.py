from sqlmodel import Session, select
from app.db import engine, init_db
from app.models.user import User, UserRole
from app.models.rag import Project
from app.auth.security import get_password_hash

def seed_users():
    # Ensure tables exist
    init_db()
    
    with Session(engine) as session:
        # Create Default Project
        project = session.exec(select(Project).where(Project.name == "General")).first()
        if not project:
            print("Creating Default Project: General")
            project = Project(name="General", description="Default project for general documents")
            session.add(project)
            session.commit()
        
        # 1. Admin Account
        admin_email = "amritanshuy0503@gmail.com"
        admin_pass = "Test@1234"
        
        # Check if exists
        user = session.exec(select(User).where(User.email == admin_email)).first()
        if not user:
            print(f"Creating Admin: {admin_email}")
            user = User(
                email=admin_email,
                hashed_password=get_password_hash(admin_pass),
                role=UserRole.ADMIN
            )
            session.add(user)
        else:
            print(f"Updating Admin Role for: {admin_email}")
            user.role = UserRole.ADMIN
            user.hashed_password = get_password_hash(admin_pass) # Ensure password matches what user asked
            session.add(user)
            
        # 2. Client Account
        client_email = "amritanshu05yadav@gmail.com"
        client_pass = "Test@1234"
        
        user2 = session.exec(select(User).where(User.email == client_email)).first()
        if not user2:
            print(f"Creating Client: {client_email}")
            user2 = User(
                email=client_email,
                hashed_password=get_password_hash(client_pass),
                role=UserRole.CLIENT
            )
            session.add(user2)
        else:
            print(f"Updating Client: {client_email}")
            user2.hashed_password = get_password_hash(client_pass)
            user2.role = UserRole.CLIENT
            session.add(user2)
            
        session.commit()
        print("Seeding Complete!")

if __name__ == "__main__":
    seed_users()

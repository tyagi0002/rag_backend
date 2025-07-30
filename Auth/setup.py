#!/usr/bin/env python3
"""
Database setup script for the authentication system
"""

import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from main.py (your main FastAPI file)
from email_auth import Base, User, get_password_hash

# Database configuration
DATABASE_URL = "sqlite:///./auth.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")

def create_admin_user():
    """Create a default admin user"""
    db = SessionLocal()
    
    try:
        # Check if admin user already exists
        existing_user = db.query(User).filter(User.email == "admin@example.com").first()
        if existing_user:
            print("❌ Admin user already exists!")
            return
        
        # Create admin user
        admin_user = User(
            email="admin@example.com",
            hashed_password=get_password_hash("admin123"),
            first_name="Admin",
            last_name="User"
        )
        
        db.add(admin_user)
        db.commit()
        print("✅ Admin user created successfully!")
        print("   Email: admin@example.com")
        print("   Password: admin123")
        print("   ⚠️  Remember to change the password in production!")
        
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        db.rollback()
    finally:
        db.close()

def main():
    """Main setup function"""
    print("🚀 Setting up authentication system...")
    print("-" * 50)
    
    # Create tables
    create_tables()
    
    # Create admin user
    create_admin_user()
    
    print("-" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and configure your settings")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the server: python main.py")
    print("4. Test the login page at: http://localhost:8000")

if __name__ == "__main__":
    main()
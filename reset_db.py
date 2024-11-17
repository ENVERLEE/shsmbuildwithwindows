# db_reset.py
from sqlalchemy import create_engine, inspect, text
from app.models import Base
from app.utils.database import SQLALCHEMY_DATABASE_URL
import os
from dotenv import load_dotenv

load_dotenv()

def reset_database():
    # 데이터베이스 연결
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    
    print("Dropping all tables...")
    Base.metadata.drop_all(engine)
    
    print("Recreating all tables...")
    Base.metadata.create_all(engine)
    
    print("Database reset complete!")

if __name__ == "__main__":
    # 확인 요청
    print("WARNING: This will delete all data in the database!")
    print(f"Database URL: {SQLALCHEMY_DATABASE_URL}")
    response = input("Are you sure you want to proceed? (yes/no): ")
    
    if response.lower() == 'yes':
        reset_database()
    else:
        print("Database reset cancelled.")
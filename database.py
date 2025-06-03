# database.py
import enum

from sqlalchemy import Boolean, Column, Date
from sqlalchemy import Enum as SAEnum  # Keep your SAEnum import
from sqlalchemy import ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import (
    declarative_base,
)  # Correct import for declarative_base
from sqlalchemy.orm import relationship, sessionmaker

DATABASE_URL = "sqlite:///./hr_agent.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# --- Enums ---
class LeaveStatus(enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class UserRole(enum.Enum):
    EMPLOYEE = "employee"
    MANAGER = "manager"
    ADMIN = "admin"


# --- Models ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(SAEnum(UserRole), default=UserRole.EMPLOYEE)
    hashed_password = Column(String, nullable=True)
    manager_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationship: This user's manager
    # An employee User object will have a 'manager_profile' attribute pointing to their manager User object.
    manager_profile = relationship(
        "User",
        remote_side=[id],  # The 'id' of the User table is the target of manager_id
        foreign_keys=[manager_id],
        back_populates="direct_reports",
    )

    # Relationship: Users managed by this user (if this user is a manager)
    # A manager User object will have a 'direct_reports' attribute (list of User objects).
    direct_reports = relationship(
        "User",
        # foreign_keys=[User.manager_id], # This should be inferred by back_populates
        back_populates="manager_profile",
    )

    # Relationship: Leave requests submitted by this user
    leave_requests_submitted = relationship(
        "LeaveRequest",
        foreign_keys="[LeaveRequest.requester_id]",  # Explicitly state FK on LeaveRequest
        back_populates="requester_user",
        cascade="all, delete-orphan",  # Optional: if user is deleted, delete their leave requests
    )

    # Relationship: Leave requests this user needs to approve (if this user is a manager)
    leave_requests_to_approve = relationship(
        "LeaveRequest",
        foreign_keys="[LeaveRequest.approver_id]",  # Explicitly state FK on LeaveRequest
        back_populates="approver_user",
    )


class LeaveRequest(Base):
    __tablename__ = "leave_requests"

    id = Column(Integer, primary_key=True, index=True)
    requester_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    leave_type = Column(String, index=True, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    reason = Column(String, nullable=True)
    status = Column(SAEnum(LeaveStatus), default=LeaveStatus.PENDING, nullable=False)
    submission_date = Column(Date, nullable=False)
    approver_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    comments = Column(String, nullable=True)

    # Relationship: The user who submitted this request
    requester_user = relationship(
        "User", foreign_keys=[requester_id], back_populates="leave_requests_submitted"
    )

    # Relationship: The user who needs to approve this request
    approver_user = relationship(
        "User", foreign_keys=[approver_id], back_populates="leave_requests_to_approve"
    )


def create_db_and_tables():
    # For safety during development, you might want to drop tables first
    # Base.metadata.drop_all(bind=engine) # Use with caution!
    Base.metadata.create_all(bind=engine)
    print("Database tables created (if they didn't exist).")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# In database.py, at the end within if __name__ == "__main__":

if __name__ == "__main__":
    create_db_and_tables()
    print("Attempting to add seed data...")
    db = SessionLocal()

    # Add this for hashing seed passwords
    from passlib.context import CryptContext

    pwd_context_seed = CryptContext(schemes=["bcrypt"], deprecated="auto")
    DEFAULT_SEED_PASSWORD = "changeme"  # Use a common password for all seed users
    hashed_seed_password = pwd_context_seed.hash(DEFAULT_SEED_PASSWORD)
    print(f"Using default hashed password for seed users: {DEFAULT_SEED_PASSWORD}")

    try:
        # Check for admin user first (or any user to ensure tables are queryable)
        admin_user_email = "admin@example.com"
        if not db.query(User).filter(User.email == admin_user_email).first():
            print(f"Adding admin user: {admin_user_email}")
            admin_user = User(
                email=admin_user_email,
                full_name="Admin User",
                role=UserRole.ADMIN,
                hashed_password=hashed_seed_password,
                is_active=True,
            )
            db.add(admin_user)
            db.commit()  # Commit admin first
            db.refresh(admin_user)
            print(f"Admin user {admin_user_email} added with ID: {admin_user.id}")

        manager_email = "manager@example.com"
        manager1 = db.query(User).filter(User.email == manager_email).first()
        if not manager1:
            print(f"Adding manager: {manager_email}")
            manager1 = User(
                email=manager_email,
                full_name="Manager One",
                role=UserRole.MANAGER,
                hashed_password=hashed_seed_password,
                is_active=True,
            )
            db.add(manager1)
            db.commit()  # Commit manager to get an ID
            db.refresh(manager1)
            print(f"Manager {manager_email} added with ID: {manager1.id}")
        else:
            print(f"Manager {manager_email} already exists with ID: {manager1.id}")

        employee1_email = "employee1@example.com"
        if not db.query(User).filter(User.email == employee1_email).first():
            if not manager1:  # Should not happen if manager creation logic is sound
                print(
                    "ERROR: Manager not found for assigning to employee1. Skipping employee1."
                )
            else:
                print(
                    f"Adding employee1: {employee1_email}, reports to manager ID {manager1.id}"
                )
                employee1 = User(
                    email=employee1_email,
                    full_name="Employee One",
                    role=UserRole.EMPLOYEE,
                    manager_id=manager1.id,
                    hashed_password=hashed_seed_password,
                    is_active=True,
                )
                db.add(employee1)

        employee2_email = "employee2@example.com"
        if not db.query(User).filter(User.email == employee2_email).first():
            if not manager1:  # Should not happen
                print(
                    "ERROR: Manager not found for assigning to employee2. Skipping employee2."
                )
            else:
                print(
                    f"Adding employee2: {employee2_email}, reports to manager ID {manager1.id}"
                )
                employee2 = User(
                    email=employee2_email,
                    full_name="Employee Two",
                    role=UserRole.EMPLOYEE,
                    manager_id=manager1.id,
                    hashed_password=hashed_seed_password,
                    is_active=True,
                )
                db.add(employee2)

        db.commit()
        print("Seed users (admin, manager, employees) added or already exist.")

    except Exception as e:
        print(f"Error adding seed data: {e}")
        db.rollback()
    finally:
        db.close()

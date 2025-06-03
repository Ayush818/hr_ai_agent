# main.py (Additions/Modifications shown)
# To run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
import os
from datetime import date, datetime, timedelta  # Add datetime, timedelta
from typing import Annotated, List, Optional  # Add Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status  # Add status
from fastapi.security import OAuth2PasswordBearer  # Add these
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt  # Add these
from langchain.chains import LLMChain, RetrievalQA
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from passlib.context import CryptContext  # Add this
from pydantic import BaseModel, EmailStr  # Add EmailStr
from sqlalchemy.orm import Session

import database
from database import LeaveRequest, LeaveStatus, User, UserRole  # UserRole

# --- Configuration (mostly same, ensure OLLAMA_MODEL_NAME is good for instruction following) ---
load_dotenv()
CHROMA_DB_PATH = "chroma_db_hr"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_KWARGS = {"device": "cpu"}
ENCODE_KWARGS = {"normalize_embeddings": False}
OLLAMA_MODEL_NAME = "mistral:7b-instruct-q4_K_M"  #  Or another good instruct model


SECRET_KEY = os.getenv(
    "SECRET_KEY", "your_fallback_secret_key_for_development_only_32_chars_long"
)  # Load from .env or use a fallback
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Token validity period

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


# --- Global Variables ---
embeddings_model = None
vector_store = None
policy_qa_chain = None  # Renamed from qa_chain for clarity
intent_classifier_chain = None
leave_form_filler_chain = None  # For extracting leave details

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HR AI Agent API",
    description="API for HR AI Agent: Policy Q&A and Leave Management.",
    version="0.3.0",  # Version bump
)


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    role: UserRole
    is_active: bool

    class Confifg:
        from_attributes = True  # Allow ORM to populate this model from DB attributes


class UserPublic(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[EmailStr] = None


# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    query: str
    session_id: str = "default_session"


class PolicyAnswerResponse(BaseModel):
    answer: str
    source_documents: list = []


class LeaveRequestInput(BaseModel):
    requester_id: int
    leave_type: str
    start_date: date
    end_date: date
    reason: Optional[str] = None
    # status will default to PENDING in DB


class LeaveRequestResponse(BaseModel):
    id: int
    requester_id: int
    leave_type: str
    start_date: date
    end_date: date
    reason: Optional[str]
    status: str  # Convert enum to string for response
    submission_date: date
    approver_id: Optional[int]
    comments: Optional[str]

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    response_type: str  # "policy_answer", "leave_form_initiated", "clarification_needed", "leave_submitted"
    message: str
    data: Optional[dict] = None  # For additional structured data if needed
    source_documents: Optional[list] = []


class LeaveRejectionInput(BaseModel):
    comment: str


# --- Password and JWT Utilities ---
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user_from_token(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Session = Depends(database.get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        details="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user_from_token)],
) -> User:
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )
    return current_user


# --- Helper Functions / Initialization Logic ---
def initialize_llm_components():
    global embeddings_model, vector_store, policy_qa_chain, intent_classifier_chain, leave_form_filler_chain

    print("Initializing API LLM components...")
    llm = Ollama(
        model=OLLAMA_MODEL_NAME, format="json"
    )  # Ensure Ollama model can output JSON

    # 1. Embedding Model and Vector Store (for Policy Q&A)
    if not embeddings_model:  # Initialize only once
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=MODEL_KWARGS,
            encode_kwargs=ENCODE_KWARGS,
        )
    if not vector_store:  # Initialize only once
        if not os.path.exists(CHROMA_DB_PATH):
            print(f"ERROR: Chroma DB path not found: {CHROMA_DB_PATH}.")
            return
        print(f"Loading vector store from: {CHROMA_DB_PATH}")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH, embedding_function=embeddings_model
        )
        print(
            f"Vector store loaded. Collection count: {vector_store._collection.count()}"
        )

    # 2. Policy Q&A Chain
    if not policy_qa_chain and vector_store:
        policy_prompt_template = """You are an AI assistant for answering questions about company policies... (same as before)
        Context: {context}
        Question: {question}
        Helpful Answer:"""
        POLICY_QA_PROMPT = PromptTemplate.from_template(policy_prompt_template)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 2}
        )  # Retrieve 2 chunks
        policy_qa_chain = RetrievalQA.from_chain_type(
            llm=Ollama(
                model=OLLAMA_MODEL_NAME
            ),  # Separate LLM instance for non-JSON policy Q&A
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": POLICY_QA_PROMPT},
            return_source_documents=True,
        )
        print("Policy Q&A chain created.")

    # 3. Intent Classification Chain

    if not intent_classifier_chain:
        print("i am here to verify intent")
        intent_prompt_template = """
        Your task is to classify the user's intent based on their query.
        The possible intents are: "policy_question", "leave_application", "greeting", "unknown".
        Focus on the primary action the user is trying to perform.

        Here are some examples:
        Query: "What is our company's policy on sick leave?"
        Intent: "policy_question"

        Query: "How many vacation days do I get per year?"
        Intent: "policy_question"

        Query: "I want to apply for a holiday from next Monday for three days."
        Intent: "leave_application"

        Query: "Can I request time off for a doctor's appointment on July 20th?"
        Intent: "leave_application"

        Query: "I'd like to take a sick day today."
        Intent: "leave_application"
        
        Query: "Hello"
        Intent: "greeting"

        Query: "Thanks"
        Intent: "unknown" # Or perhaps a "closing" intent if you want to handle it

        Now, classify the following user query.
        Provide the output strictly in JSON format with a single key "intent". Example: {{"intent": "value"}}.

        User query: {query}
        JSON Output:
        """
        INTENT_PROMPT = PromptTemplate.from_template(intent_prompt_template)
        intent_classifier_chain = (
            INTENT_PROMPT | llm | JsonOutputParser()
        )  # Using pipe operator for chaining
        print("Intent classification chain created.")

    # 4. Leave Form Filler Chain (for extracting leave details)
    if not leave_form_filler_chain:
        # This prompt needs careful crafting and testing
        leave_form_prompt_template = """
        The user wants to apply for leave. Extract the following information from their query:
        - leave_type (e.g., "vacation", "sick leave", "personal day")
        - start_date (in YYYY-MM-DD format, if ambiguous ask for clarification)
        - end_date (in YYYY-MM-DD format, if ambiguous ask for clarification)
        - reason (optional)

        If any mandatory information (leave_type, start_date, end_date) is missing or ambiguous,
        set its value to "MISSING" or "AMBIGUOUS".
        Today's date is {today_date}. Use this to infer dates if relative terms like "tomorrow" are used.

        Provide the output in JSON format with keys: "leave_type", "start_date", "end_date", "reason".

        User query: {query}
        JSON Output:
        """
        LEAVE_FORM_PROMPT = PromptTemplate.from_template(leave_form_prompt_template)
        leave_form_filler_chain = LEAVE_FORM_PROMPT | llm | JsonOutputParser()
        print("Leave form filler chain created.")

    print("API LLM components initialized/checked.")


# --- FastAPI Event Handler for Startup ---
@app.on_event("startup")
async def startup_event():
    database.create_db_and_tables()  # Ensure tables are created
    initialize_llm_components()
    if (
        not policy_qa_chain
        or not intent_classifier_chain
        or not leave_form_filler_chain
    ):
        print(
            "FATAL: One or more LLM chains could not be initialized. API might not work correctly."
        )


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the HR AI Agent API v0.2.0"}


# Placeholder: Get current user (replace with real auth later)
async def get_current_user(db: Session = Depends(database.get_db), user_id: int = 1):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        # For now, create a dummy user if not found for testing
        print(f"User with ID {user_id} not found. Creating a dummy user.")
        user = User(
            id=user_id,
            email=f"user{user_id}@example.com",
            full_name=f"Test User {user_id}",
            role=database.UserRole.EMPLOYEE,
        )
        # Find a manager or create one
        manager = db.query(User).filter(User.role == database.UserRole.MANAGER).first()
        if not manager:
            manager = User(
                id=100,
                email="manager@example.com",
                full_name="Default Manager",
                role=database.UserRole.MANAGER,
            )
            db.add(manager)
        user.manager_id = manager.id  # Assign manager
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Dummy user {user_id} created and assigned to manager {manager.id}")
    return user


@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: QuestionRequest,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_user),  # Get current user
):
    if (
        not intent_classifier_chain
        or not policy_qa_chain
        or not leave_form_filler_chain
    ):
        raise HTTPException(
            status_code=503, detail="LLM services not fully initialized."
        )

    print(f"User {current_user.id} query: {request.query}")

    # 1. Classify Intent
    try:
        intent_response = await intent_classifier_chain.ainvoke(
            {"query": request.query}
        )
        print(intent_response)

        intent = intent_response.get("intent", "unknown").lower()
        print(f"Classified intent: { intent}")
    except Exception as e:
        print(f"Error during intent classification: {e}")
        intent = "unknown"  # Fallback

    # 2. Handle Intent
    if intent == "policy_question":
        if not policy_qa_chain:
            return ChatResponse(
                response_type="error", message="Policy Q&A system not ready."
            )
        try:
            result = await policy_qa_chain.ainvoke({"query": request.query})
            answer = result.get("result", "Sorry, I could not find an answer.")
            sources = result.get("source_documents", [])
            source_docs_serializable = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in sources
            ]
            return ChatResponse(
                response_type="policy_answer",
                message=answer,
                source_documents=source_docs_serializable,
            )
        except Exception as e:
            print(f"Error during policy Q&A: {e}")
            return ChatResponse(
                response_type="error",
                message=f"Error processing policy question: {str(e)}",
            )

    elif intent == "leave_application":
        if not leave_form_filler_chain:
            return ChatResponse(
                response_type="error", message="Leave application system not ready."
            )
        try:
            today = date.today().strftime("%Y-%MM-%DD")
            form_details_response = await leave_form_filler_chain.ainvoke(
                {"query": request.query, "today_date": today}
            )

            # Basic validation and parsing (very simplified)
            print("Extracted form details:", form_details_response)
            leave_type = form_details_response.get("leave_type")
            start_date_str = form_details_response.get("start_date")
            end_date_str = form_details_response.get("end_date")
            reason = form_details_response.get("reason")

            missing_fields = []
            if not leave_type or leave_type == "MISSING":
                missing_fields.append("leave type")
            if (
                not start_date_str
                or start_date_str == "MISSING"
                or start_date_str == "AMBIGUOUS"
            ):
                missing_fields.append("start date")
            if (
                not end_date_str
                or end_date_str == "MISSING"
                or end_date_str == "AMBIGUOUS"
            ):
                missing_fields.append("end date")

            if missing_fields:
                return ChatResponse(
                    response_type="clarification_needed",
                    message=f"Okay, I can help with that. To apply for leave, I need a bit more information: please provide the {', '.join(missing_fields)}.",
                    data=form_details_response,  # Send back what was extracted
                )

            # Attempt to parse dates (very basic, needs robust parsing)
            try:
                parsed_start_date = date.fromisoformat(start_date_str)
                parsed_end_date = date.fromisoformat(end_date_str)
            except ValueError:
                return ChatResponse(
                    response_type="clarification_needed",
                    message="I couldn't understand the dates. Please provide them in YYYY-MM-DD format (e.g., 2024-07-20).",
                    data=form_details_response,
                )

            # Create Leave Request in DB
            new_leave_request = LeaveRequest(
                requester_id=current_user.id,  # Use authenticated user ID
                leave_type=leave_type,
                start_date=parsed_start_date,
                end_date=parsed_end_date,
                reason=reason,
                status=LeaveStatus.PENDING,
                submission_date=date.today(),
                approver_id=current_user.manager_id,  # Assign to user's manager
            )
            db.add(new_leave_request)
            db.commit()
            db.refresh(new_leave_request)

            return ChatResponse(
                response_type="leave_submitted",
                message=f"Your leave request for '{leave_type}' from {start_date_str} to {end_date_str} has been submitted for approval.",
                data={"request_id": new_leave_request.id},
            )

        except Exception as e:
            print(f"Error during leave application processing: {e}")
            return ChatResponse(
                response_type="error",
                message=f"Error processing leave request: {str(e)}",
            )

    elif intent == "greeting":
        return ChatResponse(
            response_type="greeting", message="Hello there! How can I assist you today?"
        )

    else:  # Unknown intent
        return ChatResponse(
            response_type="unknown_intent",
            message="I'm not sure how to help with that. I can answer policy questions or help you apply for leave.",
        )


# Endpoint to create a leave request directly (for admin or testing)
@app.post("/leave_requests/", response_model=LeaveRequestResponse)
async def create_leave_request_direct(
    leave_input: LeaveRequestInput, db: Session = Depends(database.get_db)
):
    # Basic check: Ensure requester exists
    user = db.query(User).filter(User.id == leave_input.requester_id).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User with ID {leave_input.requester_id} not found.",
        )

    # Determine approver (user's manager)
    approver_id = user.manager_id
    if not approver_id:
        # Fallback or error if manager not set. For now, let it be null or assign a default admin.
        print(
            f"Warning: User {user.id} does not have a manager assigned. Leave request will have no approver."
        )

    db_leave_request = LeaveRequest(
        requester_id=leave_input.requester_id,
        leave_type=leave_input.leave_type,
        start_date=leave_input.start_date,
        end_date=leave_input.end_date,
        reason=leave_input.reason,
        status=LeaveStatus.PENDING,  # Default status
        submission_date=date.today(),
        approver_id=approver_id,
    )
    db.add(db_leave_request)
    db.commit()
    db.refresh(db_leave_request)
    return db_leave_request


# @app.get("/manager/user/{user_id}", response_model=User)
# async def get_manager_user(
#     user_id: int, db: Session = Depends(database.get_db)
# ) -> User:
#     user = db.query(User).filter(User.id == user_id).first()
#     if not user or user.role != database.UserRole.MANAGER:
#         raise HTTPException(
#             status_code=403,
#             detail="You must be a manager to access this resource.",
#         )
#     return user


# @app.get("/manager/pending_leave_request",response_model=List[LeaveRequestResponse])
# async def get_pending_leave_requests_for_manager():

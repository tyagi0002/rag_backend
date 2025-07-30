from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import jwt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the RAG functions from your original code
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

# Import SQLAlchemy for authentication
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from passlib.context import CryptContext

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = FastAPI(title="PDF Query System", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Configuration for Authentication
DATABASE_URL = "sqlite:///./auth.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Security
security = HTTPBearer()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Global variables for the RAG system
embed_model = None
client = None
collection = None
uploaded_files_storage = []

# Request models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str
    context_used: str

class UploadResponse(BaseModel):
    message: str
    files_count: int

# Authentication models
class UserCreate(BaseModel):
    email: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    id: int
    email: str
    is_active: bool
    first_name: Optional[str]
    last_name: Optional[str]
    created_at: datetime

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication Utility Functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        first_name=user.first_name,
        last_name=user.last_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return email
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(db: Session = Depends(get_db), email: str = Depends(verify_token)):
    user = get_user_by_email(db, email=email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

# ----------- RAG Functions from your original code ----------
def load_data_from_files(file_paths):
    """Load text from uploaded PDF files"""
    all_documents = []
    for path in file_paths:
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            all_documents.append(text.strip())
        except Exception as e:
            print(f"Error reading PDF {path}: {e}")
            continue
    return all_documents

def split_text_with_overlap(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_prompt(context: str, question: str) -> str:
    """Build the prompt for the LLM"""
    return f"""Use the following context to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

def initialize_rag_system():
    """Initialize the RAG system components"""
    global embed_model, client, collection
    
    try:
        # Load embedding model
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Set up ChromaDB
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Configure Gemini API
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        
        print("RAG system initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return False

def process_uploaded_files(file_paths):
    """Process uploaded files and create vector database"""
    global collection
    
    try:
        # Load documents
        loaded_data = load_data_from_files(file_paths)
        
        if not loaded_data:
            raise ValueError("No valid documents found")
        
        # Split all documents into chunks
        all_chunks = []
        for doc in loaded_data:
            if doc.strip():  # Only process non-empty documents
                all_chunks.extend(split_text_with_overlap(doc))
        
        if not all_chunks:
            raise ValueError("No text chunks generated from documents")
        
        # Recreate collection
        try:
            client.delete_collection(name="pdf_docs")
        except:
            pass
        
        collection = client.create_collection(name="pdf_docs")
        
        # Embed and store chunks
        for idx, chunk in enumerate(all_chunks):
            if chunk.strip():  # Only embed non-empty chunks
                embedding = embed_model.encode(chunk).tolist()
                collection.add(
                    documents=[chunk],
                    ids=[str(idx)],
                    embeddings=[embedding]
                )
        
        print(f"Processed {len(all_chunks)} chunks from {len(loaded_data)} documents")
        return True
        
    except Exception as e:
        print(f"Error processing files: {e}")
        return False

def query_documents(question: str):
    """Query the document collection"""
    global collection
    
    try:
        if not collection:
            raise ValueError("No documents have been uploaded yet")
        
        # Generate query embedding
        query_embedding = embed_model.encode(question).tolist()
        
        # Search for relevant chunks
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=3  # Get top 3 most relevant chunks
        )
        
        if not results["documents"][0]:
            return "No relevant information found in the uploaded documents.", ""
        
        # Combine retrieved context
        retrieved_context = "\n\n".join(results["documents"][0])
        
        # Build prompt and call Gemini
        final_prompt = build_prompt(retrieved_context, question)
        
        # Generate response using Gemini
        llm = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = llm.generate_content(final_prompt)
        
        return response.text, retrieved_context
        
    except Exception as e:
        print(f"Error querying documents: {e}")
        return f"Error processing your query: {str(e)}", ""

# ----------- API Endpoints ----------

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    success = initialize_rag_system()
    if not success:
        print("Warning: RAG system initialization failed")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "PDF Query System API is running!",
        "docs": "/docs",
        "endpoints": {
            "auth": {
                "register": "/auth/register",
                "login": "/auth/email/login",
                "me": "/auth/me",
                "logout": "/auth/logout"
            },
            "pdf": {
                "upload": "/upload",
                "query": "/query",
                "files": "/files",
                "clear": "/files"
            }
        }
    }

# Authentication Routes
@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    db_user = create_user(db=db, user=user)
    return db_user

@app.post("/auth/email/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Login with email and password"""
    user = get_user_by_email(db, email=user_credentials.email)
    if not user or not verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@app.post("/auth/logout")
async def logout():
    """Logout user (client should discard tokens)"""
    return {"message": "Logged out successfully"}

# PDF Query Routes (Protected)
@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload PDF files for processing"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"File {file.filename} is not a PDF"
            )
    
    # Save files temporarily
    temp_files = []
    try:
        for file in files:
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_files.append(temp_file.name)
            
            # Write uploaded file to temp file
            with open(temp_file.name, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        # Process files with RAG system
        success = process_uploaded_files(temp_files)
        
        if success:
            uploaded_files_storage.extend([f.filename for f in files])
            return {
                "message": f"Successfully uploaded {len(files)} files",
                "files_count": len(files)
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to process uploaded files"
            )
    
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

@app.post("/query", response_model=QueryResponse)
async def query_pdf(
    request: QueryRequest,
    current_user: User = Depends(get_current_user)
):
    """Query the uploaded PDF documents"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    response, context = query_documents(request.question)
    
    return {
        "response": response,
        "context_used": context
    }

@app.get("/files")
async def get_uploaded_files(current_user: User = Depends(get_current_user)):
    """Get list of uploaded files"""
    return {"files": uploaded_files_storage}

@app.delete("/files")
async def clear_uploaded_files(current_user: User = Depends(get_current_user)):
    """Clear all uploaded files"""
    global collection, uploaded_files_storage
    try:
        if collection:
            client.delete_collection(name="pdf_docs")
            collection = None
        uploaded_files_storage = []
        return {"message": "All files cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict, analysis # Import both routers

app = FastAPI(
    title="Oral Cancer Diagnostic Suite API",
    description="A multi-tool API for oral cancer screening and analysis.",
    version="2.0.0"
)

# CORS Middleware
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Include BOTH routers with different prefixes
app.include_router(predict.router, prefix="/api", tags=["Quick Screening"])
app.include_router(analysis.router, prefix="/api", tags=["Comprehensive Analysis"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Oral Cancer Diagnostic Suite API!"}
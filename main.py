from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from hybrid_ranker import HybridResumeRanker
from advanced_analysis import AdvancedResumeAnalysis  
import uvicorn
from dotenv import load_dotenv
import os
import base64
import pdfplumber
from io import BytesIO

load_dotenv()

app = FastAPI(
    title="Resume Screening API",
    description="Hybrid AI-powered resume ranking system with comprehensive analysis",
)

# Initialize both ranker and advanced analyzer
ranker = HybridResumeRanker()
advanced_analyzer = AdvancedResumeAnalysis() 

# Request Models
class RankRequest(BaseModel):
    job_description: str
    resumes: List[str]
    analysis_types: Optional[List[str]] = ["ranking"]  # Specify what analyses to run

class ExtractTextRequest(BaseModel):
    pdf_file: str  # base64 encoded PDF

class ExtractTextResponse(BaseModel):
    extracted_text: str

class ComprehensiveResponse(BaseModel):
    job_description: str
    total_processed: int
    rankings: List[dict]
    skill_analysis: Optional[List[dict]] = None  
    salary_predictions: Optional[List[dict]] = None  
    quality_scores: Optional[List[dict]] = None  

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using pdfplumber for better accuracy
    """
    try:
        text = ""
        
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                # Extract text with layout preservation
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        return text.strip()
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF text extraction failed: {str(e)}")

@app.post("/extract-text", response_model=ExtractTextResponse)
async def extract_text(request: ExtractTextRequest):
    """
    Extract text from base64 encoded PDF - for Flutter app
    """
    try:
        # Validate base64
        if not request.pdf_file or len(request.pdf_file) < 100:
            return ExtractTextResponse(extracted_text="Error: Invalid or empty PDF file")
        
        # Decode base64 PDF
        pdf_bytes = base64.b64decode(request.pdf_file)
        
        # Extract text using pdfplumber
        extracted_text = extract_text_from_pdf(pdf_bytes)
        
        if not extracted_text:
            return ExtractTextResponse(extracted_text="Error: No text could be extracted from PDF")
        
        return ExtractTextResponse(extracted_text=extracted_text)
    
    except HTTPException:
        raise
    except Exception as e:
        return ExtractTextResponse(extracted_text=f"Error processing PDF: {str(e)}")

@app.post("/rank", response_model=ComprehensiveResponse)
async def rank_resumes(request: RankRequest):
    """
    Comprehensive resume analysis with multiple AI-powered features
    """
    print(f" Processing {len(request.resumes)} resumes with analyses: {request.analysis_types}")
    
    # Validate input
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")
    
    if not request.resumes:
        raise HTTPException(status_code=400, detail="At least one resume is required")
    
    results = {
        "job_description": request.job_description,
        "total_processed": len(request.resumes),
        "rankings": [],
        "skill_analysis": [],
        "salary_predictions": [], 
        "quality_scores": []
    }
    
    try:
        # 1. ALWAYS do basic ranking (core functionality)
        print(" Running SBERT + Gemini ranking...")
        ranking_results = ranker.process(
            job_desc=request.job_description,
            resumes=request.resumes
        )
        results["rankings"] = ranking_results["rankings"]
        
        # 2. SKILL GAP ANALYSIS
        if "skill_gap" in request.analysis_types:
            print("Running skill gap analysis...")
            for resume in request.resumes:
                skill_gap = advanced_analyzer.skill_gap_analysis(request.job_description, resume)
                results["skill_analysis"].append(skill_gap)
        
        # 3. SALARY PREDICTION  
        if "salary_prediction" in request.analysis_types:
            print(" Running salary predictions...")
            for resume in request.resumes:
                salary_pred = advanced_analyzer.predict_salary_range(request.job_description, resume)
                results["salary_predictions"].append(salary_pred)
        
        # 4. QUALITY CHECK
        if "quality_check" in request.analysis_types:
            print(" Running resume quality checks...")
            for resume in request.resumes:
                quality = advanced_analyzer.analyze_quality(resume)
                results["quality_scores"].append(quality)
        
        print(" All analyses completed successfully!")
        return results
        
    except Exception as e:
        print(f" Error in resume analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Individual endpoints for specific analyses
@app.post("/analyze-skills")
async def analyze_skills(job_description: str, resume_text: str):
    """Individual skill gap analysis endpoint"""
    try:
        return advanced_analyzer.skill_gap_analysis(job_description, resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill analysis failed: {str(e)}")

@app.post("/predict-salary")
async def predict_salary(job_description: str, resume_text: str, location: str = "India"):
    """Individual salary prediction endpoint"""
    try:
        return advanced_analyzer.predict_salary_range(job_description, resume_text, location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Salary prediction failed: {str(e)}")

@app.post("/check-quality") 
async def check_quality(resume_text: str):
    """Individual resume quality check endpoint"""
    try:
        return advanced_analyzer.analyze_quality(resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")

# Existing endpoints (unchanged)
@app.get("/")
async def root():
    return {
        "message": "Resume Screening API is running!", 
        "version": "2.0",
        "features": ["ranking", "skill_gap", "salary_prediction", "quality_check"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "resume-screening-api"}

# CORS middleware (important for Flutter)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
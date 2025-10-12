from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
import csv  # 🆕 Use built-in CSV module instead of pandas

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
    top_k: Optional[int] = 10
    sbert_filter_size: Optional[int] = None
    analysis_types: Optional[List[str]] = ["ranking"]

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
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF text extraction failed: {str(e)}")

def process_csv_file(csv_file: UploadFile) -> List[str]:
    """Process CSV file using built-in csv module (NO PANDAS)"""
    try:
        # Read CSV content as text
        content = csv_file.file.read().decode('utf-8')
        lines = content.splitlines()
        
        # Parse CSV
        reader = csv.DictReader(lines)
        resumes = []
        
        for row in reader:
            parts = []
            # Check each field and add if present and not empty
            if row.get('name') and row['name'].strip():
                parts.append(f"Name: {row['name']}")
            if row.get('experience') and row['experience'].strip():
                parts.append(f"Experience: {row['experience']}")
            if row.get('skills') and row['skills'].strip():
                parts.append(f"Skills: {row['skills']}")
            if row.get('resume_text') and row['resume_text'].strip():
                parts.append(f"Summary: {row['resume_text']}")
            
            # Only add if we have some content
            if parts:
                resumes.append("\n".join(parts))
        
        print(f"📊 Processed {len(resumes)} resumes from CSV (without pandas)")
        return resumes
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV processing failed: {str(e)}")

# 🆕 NEW ENDPOINT: Handle multiple file formats (PDF, CSV, Text)
@app.post("/api/rank-resumes")
async def rank_resumes_multiple_formats(
    pdf_files: List[UploadFile] = File([]),
    csv_files: List[UploadFile] = File([]),
    job_description: str = Form(...),
    text_resumes: List[str] = Form([])
):
    """
    🚀 NEW: Handle PDF, CSV, and text resumes in one endpoint
    """
    print(f"📁 Processing: {len(pdf_files)} PDFs, {len(csv_files)} CSVs, {len(text_resumes)} text resumes")
    
    all_resumes = []
    
    try:
        # 1. Process PDF files
        for pdf_file in pdf_files:
            if pdf_file.content_type != "application/pdf":
                continue
            pdf_bytes = await pdf_file.read()
            extracted_text = extract_text_from_pdf(pdf_bytes)
            if extracted_text:
                all_resumes.append(extracted_text)
        
        # 2. Process CSV files (without pandas)
        for csv_file in csv_files:
            if csv_file.content_type not in ["text/csv", "application/vnd.ms-excel"]:
                continue
            csv_resumes = process_csv_file(csv_file)
            all_resumes.extend(csv_resumes)
        
        # 3. Add text resumes directly
        all_resumes.extend(text_resumes)
        
        print(f"📊 Total resumes to process: {len(all_resumes)}")
        
        if not all_resumes:
            raise HTTPException(status_code=400, detail="No valid resumes found in uploaded files")
        
        # Apply demo limits (8 resumes max)
        if len(all_resumes) > 8:
            all_resumes = all_resumes[:8]
            print(f"📦 Limited to 8 resumes for demo")
        
        # 4. Use hybrid ranker (with text-only optimization)
        ranking_results = ranker.process(
            job_desc=job_description,
            resumes=all_resumes,
            top_k=8
        )
        
        return {
            "job_description": job_description,
            "total_processed": len(all_resumes),
            "rankings": ranking_results["rankings"]
        }
        
    except Exception as e:
        print(f"❌ Error in rank-resumes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Resume ranking failed: {str(e)}")

# 🟢 KEEP existing endpoints for backward compatibility
@app.post("/extract-text", response_model=ExtractTextResponse)
async def extract_text(request: ExtractTextRequest):
    """
    Extract text from base64 encoded PDF
    """
    try:
        if not request.pdf_file or len(request.pdf_file) < 100:
            return ExtractTextResponse(extracted_text="Error: Invalid or empty PDF file")
        
        pdf_bytes = base64.b64decode(request.pdf_file)
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
    ✅ CORRECTED: Existing endpoint for base64 resume texts
    """
    print(f" Processing {len(request.resumes)} resumes with analyses: {request.analysis_types}")
    
    # ✅ FIXED: Correct variable names
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")
    
    if not request.resumes:
        raise HTTPException(status_code=400, detail="At least one resume is required")
    
    # ✅ FIXED: Correct dictionary syntax
    results = {
        "job_description": request.job_description,
        "total_processed": len(request.resumes),
        "rankings": [],  # ✅ COMMA, not semicolon!
        "skill_analysis": [],  # ✅ COMMA, not semicolon!
        "salary_predictions": [],  # ✅ COMMA, not semicolon!
        "quality_scores": []  # ✅ No trailing comma/semicolon!
    }
    
    try:
        # 1. ALWAYS do basic ranking (core functionality)
        print(" Running SBERT + Gemini ranking...")
        ranking_results = ranker.process(
            job_desc=request.job_description,
            resumes=request.resumes,
            top_k=request.top_k,
            sbert_filter_size=request.sbert_filter_size
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
    try:
        return advanced_analyzer.skill_gap_analysis(job_description, resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill analysis failed: {str(e)}")

@app.post("/predict-salary")
async def predict_salary(job_description: str, resume_text: str, location: str = "India"):
    try:
        return advanced_analyzer.predict_salary_range(job_description, resume_text, location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Salary prediction failed: {str(e)}")

@app.post("/check-quality") 
async def check_quality(resume_text: str):
    try:
        return advanced_analyzer.analyze_quality(resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")

# Existing endpoints
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

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
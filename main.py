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
import csv
import re  #   for text cleaning
import traceback  #   for detailed error logging

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
    Enhanced PDF text extraction with multiple fallback strategies
    """
    try:
        print(f" Attempting to extract text from PDF ({len(pdf_bytes)} bytes)")
        
        text = ""
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            print(f" PDF has {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages):
                print(f"   Processing page {page_num + 1}")
                
                # Strategy 1: Basic text extraction
                page_text = page.extract_text() or ""
                print(f"     Basic extraction: {len(page_text)} chars")
                
                # Strategy 2: If basic fails, try with layout preservation
                if not page_text.strip():
                    page_text = page.extract_text(
                        layout=True,
                        x_tolerance=2,
                        y_tolerance=2
                    ) or ""
                    print(f"     Layout extraction: {len(page_text)} chars")
                
                # Strategy 3: If still no text, try extracting tables
                if not page_text.strip():
                    tables = page.extract_tables()
                    print(f"     Found {len(tables)} tables")
                    for table_num, table in enumerate(tables):
                        for row_num, row in enumerate(table):
                            if row:
                                table_text = ' '.join([str(cell) for cell in row if cell])
                                page_text += table_text + '\n'
                
                # Strategy 4: Use chars extraction as last resort
                if not page_text.strip():
                    chars = page.chars
                    if chars:
                        page_text = ' '.join([char['text'] for char in chars if char.get('text')])
                        print(f"     Char extraction: {len(page_text)} chars")
                
                if page_text.strip():
                    text += page_text + "\n\n"
                    print(f"      Page {page_num + 1} successful: {len(page_text)} chars")
                else:
                    print(f"     ❌ Page {page_num + 1} failed: no text extracted")
        
        # Clean the extracted text
        cleaned_text = clean_pdf_text(text)
        print(f" Final extracted text: {len(cleaned_text)} chars")
        
        if not cleaned_text.strip():
            return "Error: No readable text could be extracted from this PDF"
            
        return cleaned_text.strip()
        
    except Exception as e:
        print(f" PDF extraction error: {str(e)}")
        print(f" Stack trace: {traceback.format_exc()}")
        return f"Error: PDF processing failed: {str(e)}"

def clean_pdf_text(text: str) -> str:
    """
    Clean PDF-extracted text to make it more readable for AI
    """
    print(f"🧹 Cleaning text: {len(text)} chars input")
    
    if not text:
        return ""
    
    # Remove excessive whitespace but preserve paragraph breaks
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Remove common PDF artifacts
        artifacts = [
            '\x00', '\x0c', '\uf0b7', '•\u200b', '', '·', '▪'
        ]
        for artifact in artifacts:
            line = line.replace(artifact, '')
        
        # Remove lines that are just special characters or too short
        if (len(line) > 3 and 
            any(c.isalnum() for c in line) and
            not line.replace('.', '').replace('-', '').replace(' ', '').isdigit()):
            cleaned_lines.append(line)
    
    # Join with single newlines, preserve paragraph structure
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Remove excessive line breaks but keep paragraphs
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    
    # Fix encoding issues
    cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')
    
    print(f" Cleaned text: {len(cleaned_text)} chars output")
    return cleaned_text.strip()

def process_csv_file(csv_file: UploadFile) -> List[str]:
    """Process CSV file using built-in csv module (NO PANDAS)"""
    try:
        print(f" Processing CSV file: {csv_file.filename}")
        
        # Read CSV content as text
        content = csv_file.file.read().decode('utf-8')
        print(f" CSV content size: {len(content)} chars")
        print(f" First 200 chars: {content[:200]}...")
        
        lines = content.splitlines()
        print(f" CSV has {len(lines)} lines")
        
        # Parse CSV
        reader = csv.DictReader(lines)
        print(f" CSV headers: {reader.fieldnames}")
        
        resumes = []
        
        for i, row in enumerate(reader):
            print(f"   Processing row {i}: {row}")
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
                resume_text = "\n".join(parts)
                resumes.append(resume_text)
                print(f"   Added resume {i}: {len(resume_text)} chars")
            else:
                print(f"  ❌ Skipped empty row {i}")
        
        print(f" Processed {len(resumes)} resumes from CSV")
        return resumes
        
    except Exception as e:
        print(f" CSV processing error: {str(e)}")
        import traceback
        print(f" Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"CSV processing failed: {str(e)}")

#  NEW ENDPOINT: Handle multiple file formats (PDF, CSV, Text)
@app.post("/api/rank-resumes")
async def rank_resumes_multiple_formats(
    pdf_files: List[UploadFile] = File([]),
    csv_files: List[UploadFile] = File([]),
    job_description: str = Form(...),
    text_resumes: List[str] = Form([])
):
    """
     Handle PDF, CSV, and text resumes in one endpoint
    """
    print(f" Processing: {len(pdf_files)} PDFs, {len(csv_files)} CSVs, {len(text_resumes)} text resumes")
    
    all_resumes = []
    failed_pdfs = []  #  Track failed PDFs
    
    try:
        # 1. Process PDF files with better error handling
        for pdf_file in pdf_files:
            print(f" Processing PDF: {pdf_file.filename}")
            if pdf_file.content_type not in ["application/pdf", "application/octet-stream"]:
                print(f"❌ Wrong content type: {pdf_file.content_type}")
                continue
                
            try:
                pdf_bytes = await pdf_file.read()
                print(f" PDF size: {len(pdf_bytes)} bytes")
                
                extracted_text = extract_text_from_pdf(pdf_bytes)
                print(f" Extracted text length: {len(extracted_text)}")
                
                #  Check if extraction was successful
                if extracted_text and not extracted_text.startswith("Error:"):
                    all_resumes.append(extracted_text)
                    print(f" Successfully extracted text from {pdf_file.filename}")
                    print(f" First 500 chars: {extracted_text[:500]}...")
                else:
                    failed_pdfs.append(pdf_file.filename)
                    print(f"❌ Failed to extract readable text from {pdf_file.filename}")
                    print(f" Extraction result: {extracted_text}")
                    
            except Exception as e:
                failed_pdfs.append(pdf_file.filename)
                print(f" ERROR processing {pdf_file.filename}: {str(e)}")
                print(f" Stack trace: {traceback.format_exc()}")
        
        print(f" Total resumes after PDF processing: {len(all_resumes)}")
        
        # 2. Process CSV files (without pandas)
        for csv_file in csv_files:
            print(f" Processing CSV: {csv_file.filename}")
            if csv_file.content_type not in ["text/csv", "application/vnd.ms-excel", "application/octet-stream"]:
                print(f" Wrong content type: {csv_file.content_type}")
                continue
            csv_resumes = process_csv_file(csv_file)
            print(f" Got {len(csv_resumes)} resumes from CSV")
            all_resumes.extend(csv_resumes)
        
        # 3. Add text resumes directly
        all_resumes.extend(text_resumes)
        
        print(f" Total resumes to process: {len(all_resumes)}")
        
        if not all_resumes:
            raise HTTPException(status_code=400, detail="No valid resumes found in uploaded files")
        
        # Apply demo limits (8 resumes max)
        #if len(all_resumes) > 8:
         #   all_resumes = all_resumes[:8]
          #  print(f" Limited to 8 resumes for demo")
        
        # 4. Use hybrid ranker
        print(" Starting hybrid ranking...")
        ranking_results = ranker.process(
            job_desc=job_description,
            resumes=all_resumes,
            top_k=10
        )
        print(" Hybrid ranking completed")
        
        response_data = {
            "job_description": job_description,
            "total_processed": len(all_resumes),
            "rankings": ranking_results["rankings"],
            "failed_pdfs": failed_pdfs
        }
        
        return response_data
        
    except Exception as e:
        print(f" FINAL ERROR in rank-resumes: {str(e)}")
        print(f" Complete stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Resume ranking failed: {str(e)}")

#  KEEP existing endpoints for backward compatibility
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
     Existing endpoint for base64 resume texts
    """
    print(f" Processing {len(request.resumes)} resumes with analyses: {request.analysis_types}")
    
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
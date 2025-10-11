from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from hybrid_ranker import HybridResumeRanker
from advanced_analysis import AdvancedResumeAnalysis  
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="Resume Screening API",
    description="Hybrid AI-powered resume ranking system with comprehensive analysis",
    version="2.0.0"  
)

# Initialize both ranker and advanced analyzer
ranker = HybridResumeRanker()
advanced_analyzer = AdvancedResumeAnalysis() 

# Request Models
class RankRequest(BaseModel):
    job_description: str
    resumes: List[str]
    analysis_types: Optional[List[str]] = ["ranking"]  #  Specify what analyses to run

class ComprehensiveResponse(BaseModel):
    job_description: str
    total_processed: int
    rankings: List[dict]
    skill_analysis: Optional[List[dict]] = None  
    salary_predictions: Optional[List[dict]] = None  
    quality_scores: Optional[List[dict]] = None  

@app.post("/rank", response_model=ComprehensiveResponse)
async def rank_resumes(request: RankRequest):
    """
    Comprehensive resume analysis with multiple AI-powered features
    """
    print(f" Processing {len(request.resumes)} resumes with analyses: {request.analysis_types}")
    
    results = {
        "job_description": request.job_description,
        "total_processed": len(request.resumes),
        "rankings": [],
        "skill_analysis": [],
        "salary_predictions": [], 
        "quality_scores": []
    }
    
    # 1. ALWAYS do basic ranking (core functionality)
    print(" Running SBERT + Gemini ranking...")
    ranking_results = ranker.process(
        job_desc=request.job_description,
        resumes=request.resumes
    )
    results["rankings"] = ranking_results["rankings"]
    
    # 2. SKILL GAP ANALYSIS
    if "skill_gap" in request.analysis_types:
        print(" Running skill gap analysis...")
        for resume in request.resumes:
            skill_gap = advanced_analyzer.skill_gap_analysis(request.job_description, resume)
            results["skill_analysis"].append(skill_gap)
    
    # 3. SALARY PREDICTION  
    if "salary_prediction" in request.analysis_types:
        print(" Running salary predictions...")
        for resume in request.resumes:
            salary_pred = advanced_analyzer.predict_salary_range(request.job_description, resume)
            results["salary_predictions"].append(salary_pred)
    
    # 4.QUALITY CHECK
    if "quality_check" in request.analysis_types:
        print(" Running resume quality checks...")
        for resume in request.resumes:
            quality = advanced_analyzer.analyze_quality(resume)
            results["quality_scores"].append(quality)
    
    print("All analyses completed!")
    return results

#  Individual endpoints for specific analyses
@app.post("/analyze-skills")
async def analyze_skills(job_description: str, resume_text: str):
    """Individual skill gap analysis endpoint"""
    return advanced_analyzer.skill_gap_analysis(job_description, resume_text)

@app.post("/predict-salary")
async def predict_salary(job_description: str, resume_text: str, location: str = "India"):
    """Individual salary prediction endpoint"""
    return advanced_analyzer.predict_salary_range(job_description, resume_text, location)

@app.post("/check-quality") 
async def check_quality(resume_text: str):
    """Individual resume quality check endpoint"""
    return advanced_analyzer.analyze_quality(resume_text)

# Existing endpoints (unchanged)
@app.get("/")
async def root():
    return {"message": "Resume Screening API is running! ", "version": "2.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "resume-screening-api"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
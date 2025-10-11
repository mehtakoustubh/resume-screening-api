# advanced_analysis.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import json
from typing import Dict, List

load_dotenv()

class AdvancedResumeAnalysis:
    def __init__(self):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError(" GEMINI_API_KEY not found")
        
        genai.configure(api_key=gemini_api_key)
        self.gemini = genai.GenerativeModel('models/gemini-2.0-flash')
        print(" Advanced Resume Analysis initialized")

    def skill_gap_analysis(self, job_desc: str, resume_text: str) -> Dict:
        """Identify matching and missing skills"""
        prompt = f"""
        Analyze the job description and resume to identify skills match.
        
        JOB DESCRIPTION:
        {job_desc}
        
        RESUME:
        {resume_text}
        
        Return ONLY valid JSON format:
        {{
            "matching_skills": ["skill1", "skill2"],
            "missing_skills": ["skill3", "skill4"], 
            "match_percentage": 75.5,
            "recommendations": ["Add skillX", "Highlight skillY"]
        }}
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            return {
                "matching_skills": [],
                "missing_skills": [],
                "match_percentage": 0,
                "recommendations": ["Analysis failed"],
                "error": str(e)
            }

    def predict_salary_range(self, job_desc: str, resume_text: str, location: str = "India") -> Dict:
        """Predict salary range based on job and resume"""
        prompt = f"""
        Predict salary range for this candidate in {location}.
        
        JOB: {job_desc}
        RESUME: {resume_text}
        LOCATION: {location}
        
        Return ONLY valid JSON:
        {{
            "min_salary": 500000,
            "max_salary": 800000,
            "currency": "INR",
            "confidence": "High",
            "factors": ["experience", "skills", "location"]
        }}
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            return {
                "min_salary": 0,
                "max_salary": 0,
                "currency": "INR",
                "confidence": "Low",
                "factors": [],
                "error": str(e)
            }

    def analyze_quality(self, resume_text: str) -> Dict:
        """Analyze resume quality and completeness"""
        prompt = f"""
        Analyze this resume's quality and provide scores (1-10).
        
        RESUME:
        {resume_text}
        
        Return ONLY valid JSON:
        {{
            "completeness_score": 7,
            "professionalism_score": 8,
            "clarity_score": 6,
            "ats_friendliness": 7,
            "overall_score": 7.5,
            "improvement_tips": ["Add quantifiable achievements", "Improve formatting"]
        }}
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            return {
                "completeness_score": 5,
                "professionalism_score": 5,
                "clarity_score": 5,
                "ats_friendliness": 5,
                "overall_score": 5.0,
                "improvement_tips": ["Analysis failed"],
                "error": str(e)
            }

    def _parse_json_response(self, text: str) -> Dict:
        """Extract JSON from Gemini response"""
        try:
            # Find JSON pattern in response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "No JSON found in response", "raw_response": text}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw_response": text}
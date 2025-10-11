import os
import requests
import numpy as np
from typing import List, Dict, Tuple
import google.generativeai as genai

class HybridResumeRanker:
    def __init__(self):
        self.hf_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.hf_headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini = genai.GenerativeModel('models/gemini-2.0-flash')
    
    def process(self, job_desc: str, resumes: List[str], top_k: int = 10) -> Dict:
        # Step 1: SBERT filtering via Hugging Face API
        top_resumes = self._sbert_filter(job_desc, resumes, top_k)
        
        # Step 2: Gemini scoring (YOUR EXACT CODE)
        ranked_results = self._get_gemini_scores(job_desc, top_resumes)
        
        return {
            "job_description": job_desc,
            "total_processed": len(resumes),
            "sbert_filtered": len(top_resumes),
            "rankings": ranked_results
        }
    
    def _sbert_filter(self, job_desc: str, resumes: List[str], top_k: int) -> List[Dict]:
        """SBERT filtering via Hugging Face API"""
        job_emb = self.get_embedding(job_desc)
        if not job_emb:
            return []
        
        similarities = []
        for i, resume in enumerate(resumes):
            resume_emb = self.get_embedding(resume)
            if resume_emb:
                similarity = np.dot(job_emb, resume_emb) / (np.linalg.norm(job_emb) * np.linalg.norm(resume_emb))
                similarities.append((i, resume, similarity))
        
        # Get top-k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return [{
            'index': idx,
            'resume_text': resume,
            'sbert_similarity': round(score * 10, 2)
        } for idx, resume, score in similarities[:top_k]]
    
    def _get_gemini_scores(self, job_desc: str, top_resumes: List[Dict]) -> List[Dict]:
        """Gemini scores only the SBERT-filtered resumes"""
        results = []
        
        for resume_data in top_resumes:
            resume_text = resume_data['resume_text']
            original_index = resume_data['index']
            sbert_score = resume_data['sbert_similarity']
            
            prompt = f"""
            ACT as an expert resume screening AI. Evaluate how well this resume matches the job description.
            
            JOB DESCRIPTION:
            {job_desc}
            
            RESUME:
            {resume_text}
            
            Provide your evaluation in this EXACT format:
            Score: [number between 1-10]
            Explanation: [2-3 sentence explanation of strengths and weaknesses]
            
            Be strict but fair in your evaluation.
            """
            
            try:
                response = self.gemini.generate_content(prompt)
                gemini_score, explanation = self._parse_gemini_response(response.text)
                
                results.append({
                    'resume_index': original_index,
                    'gemini_score': gemini_score,
                    'sbert_similarity': sbert_score,
                    'explanation': explanation,
                    'resume_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
                })
                
            except Exception as e:
                results.append({
                    'resume_index': original_index,
                    'gemini_score': 5.0,
                    'sbert_similarity': sbert_score,
                    'explanation': "Unable to evaluate this resume",
                    'resume_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
                })
        
        results.sort(key=lambda x: x['gemini_score'], reverse=True)
        return results
    
    def _parse_gemini_response(self, response_text: str) -> Tuple[float, str]:
        """Parse Gemini response to extract score and explanation"""
        try:
            lines = response_text.strip().split('\n')
            score_line = next((line for line in lines if line.lower().startswith('score:')), None)
            explanation_line = next((line for line in lines if line.lower().startswith('explanation:')), None)
            
            if score_line:
                score = float(score_line.split(':')[1].strip().split()[0])
            else:
                score = 5.0
                
            if explanation_line:
                explanation = explanation_line.split(':')[1].strip()
            else:
                explanation = "No explanation provided"
                
            return score, explanation
        except Exception as e:
            return 5.0, "Error in evaluation"
    
    def get_embedding(self, text: str):
        """Get embeddings from Hugging Face API"""
        try:
            response = requests.post(
                self.hf_url, 
                headers=self.hf_headers, 
                json={"inputs": text},
                timeout=30
            )
            print(f"HF API Status: {response.status_code}")  # ← ADDED DEBUG LINE
            if response.status_code == 200:
                return response.json()
            else:
                print(f"HF API Error: {response.text}")  # ← ADDED DEBUG LINE
                return None
        except Exception as e:
            print(f"HF API Exception: {e}")  # ← ADDED DEBUG LINE
            return None
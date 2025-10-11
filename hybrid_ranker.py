import os
from dotenv import load_dotenv
from typing import List, Dict,Tuple
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util  #  ADD THIS
import torch  #  ADD THIS

# Load environment variables
load_dotenv()

class HybridResumeRanker:
    def __init__(self):
        print(" Initializing Hybrid Resume Ranker with SBERT + Gemini Flash...")
        
        # Get API key from environment
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError(" GEMINI_API_KEY not found in .env file")
        
        # Configure Gemini with Flash model
        genai.configure(api_key=gemini_api_key)
        self.gemini = genai.GenerativeModel('models/gemini-2.0-flash')
        
        
        # INITIALIZE SBERT MODEL
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(" SBERT + Gemini 1.5 Flash hybrid model configured")
    
    def process(self, job_desc: str, resumes: List[str], top_k: int = 10) -> Dict:
        """
         CORRECT HYBRID PIPELINE:
        1. SBERT filters top resumes
        2. Gemini scores only the shortlisted ones
        """
        print(f" Processing {len(resumes)} resumes with hybrid approach...")
        
        # STEP 1: SBERT Semantic Filtering
        print(" SBERT: Filtering top resumes...")
        top_resumes = self._sbert_filter(job_desc, resumes, top_k)
        
        # STEP 2: Gemini Qualitative Scoring
        print(" Gemini: Scoring shortlisted resumes...")
        ranked_results = self._get_gemini_scores(job_desc, top_resumes)
        
        return {
            "job_description": job_desc,
            "total_processed": len(resumes),
            "sbert_filtered": len(top_resumes),
            "rankings": ranked_results
        }
    
    def _sbert_filter(self, job_desc: str, resumes: List[str], top_k: int) -> List[tuple]:
        """SBERT semantic filtering - returns top resumes with scores"""
        # Encode job description and resumes
        job_embedding = self.sbert.encode(job_desc, convert_to_tensor=True)
        resume_embeddings = self.sbert.encode(resumes, convert_to_tensor=True)
        
        # Calculate cosine similarities
        cos_scores = util.cos_sim(job_embedding, resume_embeddings)[0]
        
        # Get top-k resumes
        top_results = torch.topk(cos_scores, k=min(top_k, len(resumes)))
        
        # Return list of (index, resume_text, sbert_score)
        top_resumes = []
        for score, idx in zip(top_results[0], top_results[1]):
            top_resumes.append({
                'index': idx.item(),
                'resume_text': resumes[idx],
                'sbert_similarity': round(score.item() * 10, 2)  # Scale to 0-10
            })
        
        print(f" SBERT filtered: {len(top_resumes)} resumes")
        return top_resumes
    
    def _get_gemini_scores(self, job_desc: str, top_resumes: List[Dict]) -> List[Dict]:
        """Gemini scores only the SBERT-filtered resumes"""
        results = []
        
        for resume_data in top_resumes:
            resume_text = resume_data['resume_text']
            original_index = resume_data['index']
            sbert_score = resume_data['sbert_similarity']
            
            print(f" Gemini evaluating resume {original_index + 1}...")
            
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
                print(f" Error with resume {original_index + 1}: {e}")
                results.append({
                    'resume_index': original_index,
                    'gemini_score': 5.0,
                    'sbert_similarity': sbert_score,
                    'explanation': "Unable to evaluate this resume",
                    'resume_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
                })
        
        # Sort by Gemini score (highest first)
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
            print(f" Error parsing Gemini response: {e}")
            return 5.0, "Error in evaluation"
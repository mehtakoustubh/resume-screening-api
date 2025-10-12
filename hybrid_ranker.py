import os
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
import re
import time

class HybridResumeRanker:
    def __init__(self):
        # CHANGED: Use feature-extraction pipeline instead of similarity model
        self.hf_url = "https://api-inference.huggingface.co/models/BAAI/bge-small-en"
        self.hf_headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini = genai.GenerativeModel('models/gemini-2.0-flash')
    
    def process(self, job_desc: str, resumes: List[str], top_k: int = 10, 
               sbert_filter_size: Optional[int] = None) -> Dict:
        """
        Enhanced process method with text-only optimization
        
        Args:
            job_desc: Job description text
            resumes: List of resume texts
            top_k: Final number of top resumes to return
            sbert_filter_size: Number of resumes for SBERT to pass to Gemini
        """
        # 🆕 NEW: Check if text-only mode should be used
        all_are_clean_text = self._are_all_clean_text(resumes)
        
        if all_are_clean_text and len(resumes) <= 8:
            print("🚀 TEXT-ONLY MODE: Using batch Gemini processing (fast)")
            return self._batch_gemini_ranking(job_desc, resumes, top_k)
        else:
            print("🔀 HYBRID MODE: Using SBERT + Gemini flow")
            # ✅ KEEP YOUR EXISTING HYBRID LOGIC:
            filter_size = sbert_filter_size if sbert_filter_size is not None else top_k
            
            print(f" Processing {len(resumes)} resumes")
            print(f" SBERT filter: {len(resumes)} → {filter_size} candidates")
            print(f" Final return: top {top_k} ranked candidates")
            
            # Step 1: SBERT filtering via Hugging Face API
            top_resumes = self._sbert_filter(job_desc, resumes, filter_size)
            
            # Step 2: Gemini scoring
            ranked_results = self._get_gemini_scores(job_desc, top_resumes)
            
            # Final top-k selection
            final_rankings = ranked_results[:top_k]
            
            print(f" Successfully processed {len(resumes)} resumes → {len(final_rankings)} final rankings")
            
            return {
                "job_description": job_desc,
                "total_processed": len(resumes),
                "sbert_filtered": len(top_resumes),
                "final_top_k": top_k,
                "rankings": final_rankings
            }
    
    def _are_all_clean_text(self, resumes: List[str]) -> bool:
        """Check if all resumes are clean text (not PDF extractions)"""
        for resume in resumes:
            # PDF extraction often has these characteristics
            if (len(resume.split()) > 1500 or  # Too long (PDF artifact)
                '\x00' in resume or            # Null bytes (PDF)
                resume.startswith('%PDF') or   # PDF header
                '\\n\\n\\n' in resume or       # Excessive newlines (PDF)
                len(resume.strip()) < 50):     # Too short
                return False
        return True
    
    def _batch_gemini_ranking(self, job_desc: str, resumes: List[str], top_k: int) -> Dict:
        """Single API call for all text resumes"""
        print(f"🎯 Batch processing {len(resumes)} text resumes in ONE Gemini call")
        
        # Build batch prompt
        resumes_formatted = ""
        for i, resume in enumerate(resumes):
            # Limit resume length to avoid token limits
            truncated_resume = resume[:2000] + "..." if len(resume) > 2000 else resume
            resumes_formatted += f"--- RESUME {i+1} ---\n{truncated_resume}\n\n"
        
        prompt = f"""
        ACT as an expert resume screening AI. Evaluate how well each resume matches the job description.
        
        JOB DESCRIPTION:
        {job_desc}
        
        RESUMES TO ANALYZE:
        {resumes_formatted}
        
        Provide your evaluation for EACH resume in this EXACT format:
        RESUME 1: [score 1-10]|[2-3 sentence explanation]
        RESUME 2: [score 1-10]|[2-3 sentence explanation]
        ...
        
        Be strict but fair. Focus on relevant skills, experience, and qualifications.
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            print(f" Batch Gemini response received")
            return self._parse_batch_response(response.text, resumes, job_desc, top_k)
        except Exception as e:
            print(f"❌ Batch Gemini failed: {e}, falling back to individual calls")
            return self._fallback_individual_calls(job_desc, resumes, top_k)
    
    def _parse_batch_response(self, response_text: str, resumes: List[str], job_desc: str, top_k: int) -> Dict:
        """Parse the batch Gemini response"""
        results = []
        lines = response_text.strip().split('\n')
        
        for i, resume in enumerate(resumes):
            resume_score = 5.0  # Default score
            explanation = "Evaluation completed"
            
            # Look for this resume in the response
            resume_prefixes = [f"RESUME {i+1}", f"Resume {i+1}", f"{i+1}:"]
            
            for line in lines:
                line_upper = line.upper()
                if any(line_upper.startswith(prefix.upper()) for prefix in resume_prefixes):
                    # Parse score and explanation
                    if '|' in line:
                        parts = line.split('|', 1)
                        if parts[0].strip():
                            # Extract score (look for numbers)
                            numbers = re.findall(r'\d+\.?\d*', parts[0])
                            if numbers:
                                resume_score = min(10, max(1, float(numbers[0])))
                        if len(parts) > 1:
                            explanation = parts[1].strip()
                    break
            
            results.append({
                'resume_index': i,
                'gemini_score': resume_score,
                'sbert_similarity': 0.0,  # Not used in batch mode
                'explanation': explanation,
                'resume_preview': resume[:200] + "..." if len(resume) > 200 else resume
            })
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x['gemini_score'], reverse=True)
        final_rankings = results[:top_k]
        
        return {
            "job_description": job_desc,
            "total_processed": len(resumes),
            "sbert_filtered": len(resumes),  # No SBERT filtering in batch mode
            "final_top_k": top_k,
            "rankings": final_rankings
        }
    
    def _fallback_individual_calls(self, job_desc: str, resumes: List[str], top_k: int) -> Dict:
        """Fallback to individual Gemini calls if batch fails"""
        print("🔄 Falling back to individual Gemini calls")
        results = []
        
        for i, resume in enumerate(resumes):
            try:
                prompt = f"""
                Job: {job_desc}
                Resume: {resume}
                
                Score: [1-10]
                Explanation: [brief reason]
                """
                
                response = self.gemini.generate_content(prompt)
                score, explanation = self._parse_gemini_response(response.text)
                
                results.append({
                    'resume_index': i,
                    'gemini_score': score,
                    'sbert_similarity': 0.0,
                    'explanation': explanation,
                    'resume_preview': resume[:200] + "..." if len(resume) > 200 else resume
                })
                
                # Small delay to avoid rate limits
                if i < len(resumes) - 1:
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ Individual call failed for resume {i}: {e}")
                results.append({
                    'resume_index': i,
                    'gemini_score': 5.0,
                    'sbert_similarity': 0.0,
                    'explanation': "Evaluation failed",
                    'resume_preview': resume[:200] + "..." if len(resume) > 200 else resume
                })
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x['gemini_score'], reverse=True)
        final_rankings = results[:top_k]
        
        return {
            "job_description": job_desc,
            "total_processed": len(resumes),
            "sbert_filtered": len(resumes),
            "final_top_k": top_k,
            "rankings": final_rankings
        }
    
    #  KEEP ALL YOUR EXISTING METHODS (they remain unchanged):
    def _sbert_filter(self, job_desc: str, resumes: List[str], top_k: int) -> List[Dict]:
        """SBERT filtering via Hugging Face API"""
        print(f"🔍 Getting job description embedding...")
        job_emb = self.get_embedding(job_desc)
        if not job_emb:
            print(" Failed to get job embedding")
            return []
        
        print(f" Calculating similarities for {len(resumes)} resumes...")
        similarities = []
        for i, resume in enumerate(resumes):
            resume_emb = self.get_embedding(resume)
            if resume_emb:
                similarity = np.dot(job_emb, resume_emb) / (np.linalg.norm(job_emb) * np.linalg.norm(resume_emb))
                similarities.append((i, resume, similarity))
        
        # Get top-k
        similarities.sort(key=lambda x: x[2], reverse=True)
        top_candidates = [{
            'index': idx,
            'resume_text': resume,
            'sbert_similarity': round(score * 10, 2)
        } for idx, resume, score in similarities[:top_k]]
        
        print(f"SBERT found {len(top_candidates)} candidates with similarities: {[c['sbert_similarity'] for c in top_candidates]}")
        return top_candidates
    
    def _get_gemini_scores(self, job_desc: str, top_resumes: List[Dict]) -> List[Dict]:
        """Gemini scores only the SBERT-filtered resumes"""
        print(f" Gemini analyzing {len(top_resumes)} resumes...")
        results = []
        
        for i, resume_data in enumerate(top_resumes):
            resume_text = resume_data['resume_text']
            original_index = resume_data['index']
            sbert_score = resume_data['sbert_similarity']
        
            print(f"   Processing resume {i+1}/{len(top_resumes)} (index: {original_index})")
            
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
                
                print(f"     Scored: {gemini_score}/10")
                
                results.append({
                    'resume_index': original_index,
                    'gemini_score': gemini_score,
                    'sbert_similarity': sbert_score,
                    'explanation': explanation,
                    'resume_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
                })
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results.append({
                    'resume_index': original_index,
                    'gemini_score': 5.0,
                    'sbert_similarity': sbert_score,
                    'explanation': "Unable to evaluate this resume",
                    'resume_preview': resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
                })
        
        results.sort(key=lambda x: x['gemini_score'], reverse=True)
        print(f" Gemini ranking complete. Top score: {results[0]['gemini_score'] if results else 'N/A'}")
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
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ HF API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"❌ HF API Exception: {e}")
            return None
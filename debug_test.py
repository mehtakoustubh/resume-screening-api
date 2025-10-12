import requests
import time

BASE_URL = "https://resume-screening-api-production.up.railway.app"

def debug_ranking():
    print("🔍 DEBUGGING RANKING ISSUE")
    print("=" * 60)
    
    # Test with better formatted resume
    job_desc = "We need a Python developer with Flask and Django experience. 3+ years of web development required."
    resume = """
    John Doe - Senior Python Developer
    
    EXPERIENCE:
    - 5 years as Python Developer at Tech Company
    - Built web applications using Django and Flask
    - Developed REST APIs and microservices
    - Worked with PostgreSQL and AWS
    
    SKILLS:
    Python, Django, Flask, PostgreSQL, AWS, Docker, REST APIs
    
    EDUCATION:
    Bachelor of Computer Science
    """
    
    print("📤 Sending request to API...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/rank-resumes",
            files={
                "job_description": (None, job_desc),
                "text_resumes": (None, resume)
            },
            timeout=60
        )
        
        duration = time.time() - start_time
        print(f"⏱️  Response time: {duration:.1f}s")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Processed: {data['total_processed']} resumes")
            
            if data['rankings']:
                rank = data['rankings'][0]
                print(f"\n🎯 RANKING RESULT:")
                print(f"   Score: {rank['gemini_score']}/10")
                print(f"   Explanation: {rank['explanation']}")
                print(f"   SBERT Similarity: {rank.get('sbert_similarity', 'N/A')}")
                print(f"   Preview: {rank['resume_preview'][:100]}...")
            else:
                print("❌ No rankings returned!")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_gemini_directly():
    """Test if Gemini is working correctly"""
    print("\n🧪 TESTING GEMINI DIRECTLY")
    print("=" * 60)
    
    test_prompt = """
    ACT as an expert resume screening AI. Evaluate how well this resume matches the job description.
    
    JOB DESCRIPTION:
    We need a Python developer with Flask and Django experience. 3+ years of web development required.
    
    RESUME:
    John Doe - Senior Python Developer
    EXPERIENCE: 5 years as Python Developer, built web applications using Django and Flask
    SKILLS: Python, Django, Flask, PostgreSQL, AWS
    
    Provide your evaluation in this EXACT format:
    Score: [number between 1-10]
    Explanation: [2-3 sentence explanation of strengths and weaknesses]
    """
    
    print("📝 Test Prompt Sent to Gemini")
    print("Expected format: 'Score: X' followed by 'Explanation: Y'")
    print("This resume should score 7-9/10 based on good Python/Django experience")

if __name__ == "__main__":
    debug_ranking()
    test_gemini_directly()
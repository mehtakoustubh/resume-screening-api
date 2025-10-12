import requests
import time

BASE_URL = "https://resume-screening-api-production.up.railway.app"

def test_api():
    print(" Testing Your Resume Screening API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("1. Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"    Status: {response.status_code}")
        print(f"    Response: {response.json()}")
    except Exception as e:
        print(f"    Failed: {e}")
        return
    
    # Test 2: Text-Only Resume Ranking
    print("\n2. Testing Text Resume Ranking...")
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/rank-resumes",
            files={
                "job_description": (None, "We need a Python developer with Flask and Django experience. 3+ years of web development required."),
                "text_resumes": (None, "John Doe, Senior Python Developer, 5 years experience. Skills: Python, Django, Flask, PostgreSQL, AWS. Built scalable web applications."),
                "text_resumes": (None, "Jane Smith, Frontend Developer, 2 years experience. Skills: JavaScript, React, HTML, CSS. Some Python knowledge but limited backend experience."),
                "text_resumes": (None, "Mike Johnson, Data Scientist, 4 years experience. Skills: Python, SQL, Machine Learning, TensorFlow. Strong data analysis but minimal web framework experience.")
            }
        )
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"    Success!")
            print(f"   ⏱  Time: {duration:.1f} seconds")
            print(f"    Processed: {data['total_processed']} resumes")
            
            print(f"\n    RANKINGS:")
            print("   " + "=" * 40)
            for i, rank in enumerate(data['rankings']):
                print(f"   #{i+1} - Score: {rank['gemini_score']}/10")
                print(f"    Explanation: {rank['explanation']}")
                print(f"    Preview: {rank['resume_preview'][:80]}...")
                print("   " + "-" * 40)
        else:
            print(f"    Failed with status: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"    Error: {e}")
    
    # Test 3: Root Endpoint
    print("\n3. Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        data = response.json()
        print(f"    Status: {response.status_code}")
        print(f"    Features: {', '.join(data.get('features', []))}")
    except Exception as e:
        print(f"    Failed: {e}")

if __name__ == "__main__":
    test_api()
    print("\n API Testing Complete!")

import os
import requests
from dotenv import load_dotenv

def test_gemini_api():
    """Test Gemini API using direct HTTP request"""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        raise ValueError("API key not found in .env file")
    
    # API endpoint
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.0-pro:generateContent?key={api_key}"
    
    # Request headers
    headers = {
        'Content-Type': 'application/json'
    }
    
    # Request body with correct safety settings
    data = {
        "contents": [{
            "parts": [{
                "text": "Explain how AI works in a few words"
            }]
        }],
        "safety_settings": [{
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_LOW_AND_ABOVE"
        }, {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_LOW_AND_ABOVE"
        }],
        "generation_config": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 100
        }
    }
    
    try:
        # Make the API request
        response = requests.post(url, headers=headers, json=data)
        
        # Print raw response for debugging
        print(f"\nDebug - Status Code: {response.status_code}")
        print(f"Debug - Response Text: {response.text[:500]}...")
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse and print response
        result = response.json()
        print("\nAPI Test Results:")
        print("----------------")
        print(f"Status: Success")
        print(f"Response: {result['candidates'][0]['content']['parts'][0]['text']}")
        return True
        
    except requests.exceptions.RequestException as e:
        print("\nAPI Test Results:")
        print("----------------")
        print(f"Status: Failed")
        print(f"Error: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response Text: {e.response.text}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Gemini API Test")
    print("---------------------------")
    test_gemini_api()
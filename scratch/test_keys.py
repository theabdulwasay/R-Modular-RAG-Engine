import os
from dotenv import load_dotenv

# Load env
load_dotenv()

def test_gemini():
    print("--- Testing Gemini API ---")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env")
        return
        
    try:
        from google import genai
        # Initialize client
        client = genai.Client(api_key=api_key)
        
        # Test Gemini 1.5 Flash (highly stable)
        print("Testing gemini-1.5-flash...")
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents="Hello! Respond in 3 words."
        )
        print(f"gemini-1.5-flash Success: '{resp.text.strip()}'")
    except Exception as e:
        print(f"gemini-1.5-flash Failed: {e}")
        
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Test Gemini 2.5 Flash
        print("Testing gemini-2.5-flash...")
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello! Respond in 3 words."
        )
        print(f"gemini-2.5-flash Success: '{resp.text.strip()}'")
    except Exception as e:
        print(f"gemini-2.5-flash Failed: {e}")
        
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Test Embeddings
        print("Testing text-embedding-004...")
        resp = client.models.embed_content(
            model="text-embedding-004",
            contents=["test text"]
        )
        print(f"text-embedding-004 Success: Vector length {len(resp.embeddings[0].values)}")
    except Exception as e:
        print(f"text-embedding-004 Failed: {e}")

def test_openai():
    print("\n--- Testing OpenAI API ---")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in .env")
        return
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test GPT-4o-mini
        print("Testing gpt-4o-mini...")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello! Respond in 3 words."}],
            max_tokens=10
        )
        print(f"gpt-4o-mini Success: '{resp.choices[0].message.content.strip()}'")
    except Exception as e:
        print(f"gpt-4o-mini Failed: {e}")
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Test Embeddings
        print("Testing text-embedding-3-small...")
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=["test text"]
        )
        print(f"text-embedding-3-small Success: Vector length {len(resp.data[0].embedding)}")
    except Exception as e:
        print(f"text-embedding-3-small Failed: {e}")

if __name__ == "__main__":
    test_gemini()
    test_openai()

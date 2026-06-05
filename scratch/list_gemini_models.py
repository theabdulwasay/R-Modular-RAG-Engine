import os
from dotenv import load_dotenv

load_dotenv()

def list_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env")
        return
        
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        print("Available models:")
        # List all models
        models = client.models.list()
        for m in models:
            # Print model details
            print(f"- {m.name} (supports: {m.supported_actions})")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()

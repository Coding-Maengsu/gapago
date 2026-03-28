import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions

load_dotenv()

client = genai.Client(http_options=HttpOptions(api_version="v1"))

print("Gemini와 대화를 시작합니다! (종료하려면 'quit' 입력)")
print("="*50)

while True:
    user_input = input("\n나: ")
    
    if user_input.lower() == "quit":
        print("대화를 종료합니다.")
        break
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input,
    )
    
    print(f"\nGemini: {response.text}")
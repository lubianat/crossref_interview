import os
from dotenv import load_dotenv

load_dotenv()
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

DEEP_SEEK_API_KEY = os.getenv('DEEP_SEEK_API_KEY')
client = OpenAI(api_key=DEEP_SEEK_API_KEY, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)
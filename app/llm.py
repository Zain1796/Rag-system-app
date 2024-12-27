import openai
import os
from dotenv import load_dotenv

load_dotenv()

def query_llm(question, context):
    """Query the LLM with a user question and context."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {question}"}
        ]
    )
    return response['choices'][0]['message']['content']

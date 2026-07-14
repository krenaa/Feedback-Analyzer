"""
The backend: a small FastAPI service that analyzes ONE customer review using LangChain.

Run it with:
    uv run fastapi dev api.py
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize LangChain's Gemini model wrapper
# It reads GEMINI_API_KEY automatically from your environment variables
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

app = FastAPI()

# What the caller must SEND us
class Review(BaseModel):
    text: str

# Define our structured output schema using Pydantic
class Analysis(BaseModel):
    label: str = Field(description="Must be 'positive', 'negative', or 'neutral'")
    score: int = Field(description="Must be a number from 1 (very bad) to 5 (very good)")
    theme: str = Field(description="ONE lowercase word for the main topic (e.g., delivery, taste, price, service, quality)")

# Bind the schema to the model to enforce JSON output structure
structured_llm = llm.with_structured_output(Analysis)

@app.post("/analyze")
def analyze(review: Review):
    prompt = (
        "Analyze this customer review.\n"
        f"Review: {review.text}"
    )
    
    # LangChain calls the model and automatically parses it directly into our Pydantic object
    response = structured_llm.invoke(prompt)
    return response
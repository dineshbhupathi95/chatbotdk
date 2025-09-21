from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import re

# Dummy project data
DUMMY_DATA = [
    {"projectName": "PRjej", "achieved": 40, "target": 67, "engineeringManager": "Susan", "valueStreamLead": "Eric"},
    {"projectName": "PRjejewew", "achieved": 80, "target": 75, "engineeringManager": "Ttn", "valueStreamLead": "Eff"},
    {"projectName": "OnboardX", "achieved": 55, "target": 60, "engineeringManager": "John", "valueStreamLead": "Alice"},
    {"projectName": "PayTrack", "achieved": 30, "target": 90, "engineeringManager": "Sophia", "valueStreamLead": "David"},
]

# FastAPI setup
app = FastAPI(title="Data-Aware Project Chatbot")

# Input model
class Query(BaseModel):
    question: str

# Load Hugging Face LLM for natural language generation
qa_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=-1)  # CPU: device=-1

def extract_project_names(question: str):
    """Find project names mentioned in the question."""
    projects = []
    for project in DUMMY_DATA:
        if project["projectName"].lower() in question.lower():
            projects.append(project)
    return projects

def generate_answer(projects: list, question: str):
    """Compute numeric info and format answer with LLM for natural phrasing."""
    if not projects:
        return "Sorry, I couldn't find any project matching your question."

    answers = []
    for project in projects:
        achieved = project["achieved"]
        target = project["target"]
        status = "achieved" if achieved >= target else "not achieved"
        # Build numeric-accurate base answer
        base_answer = (
            f"Project '{project['projectName']}' has {achieved} achieved vs {target} target, "
            f"so it is {status}. "
            f"Engineering Manager: {project['engineeringManager']}, Value Stream Lead: {project['valueStreamLead']}."
        )
        # Use LLM to make the phrasing more conversational
        prompt = f"Make this answer more conversational: '{base_answer}'"
        llm_response = qa_pipeline(prompt, max_new_tokens=50, do_sample=True)[0]["generated_text"]
        # Extract after the colon
        conversational_answer = re.split(":", llm_response, maxsplit=1)[-1].strip()
        answers.append(conversational_answer)
    return " ".join(answers)

# API endpoint
@app.post("/ask")
def ask_question(query: Query):
    projects = extract_project_names(query.question)
    answer = generate_answer(projects, query.question)
    return {"answer": answer}

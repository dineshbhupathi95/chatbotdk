from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from fastapi.middleware.cors import CORSMiddleware

DUMMY_DATA = [
    {
        "valueStream": "Benefits and Pricing",
        "subStream": "Benefits Journey",
        "projectName": "PRjej",
        "valueStreamLead": "Eric",
        "engineeringManager": "Susan",
        "taskName": "Benefits strategy",
        "resourceCount": 8,
        "weeklyHours": 22,
        "montlyHours": 40,
        "quaterlyHours": 80,
        "catagaory": "Capex",
        "target": 67,
        "achieved": 40,
    },
    {
        "valueStream": "AOR",
        "subStream": "CLM",
        "projectName": "PRjejewew",
        "valueStreamLead": "Eff",
        "engineeringManager": "Ttn",
        "taskName": "CLM",
        "resourceCount": 4,
        "weeklyHours": 20,
        "montlyHours": 38,
        "quaterlyHours": 50,
        "catagaory": "Opex",
        "target": 75,
        "achieved": 80,
    },
    {
        "valueStream": "Customer Experience",
        "subStream": "Onboarding",
        "projectName": "OnboardX",
        "valueStreamLead": "Alice",
        "engineeringManager": "John",
        "taskName": "Onboarding Flow",
        "resourceCount": 10,
        "weeklyHours": 28,
        "montlyHours": 50,
        "quaterlyHours": 100,
        "catagaory": "Capex",
        "target": 60,
        "achieved": 55,
    },
    {
        "valueStream": "Payments",
        "subStream": "Billing",
        "projectName": "PayTrack",
        "valueStreamLead": "David",
        "engineeringManager": "Sophia",
        "taskName": "Payment Gateway",
        "resourceCount": 6,
        "weeklyHours": 18,
        "montlyHours": 42,
        "quaterlyHours": 75,
        "catagaory": "Opex",
        "target": 90,
        "achieved": 30,
    },
]

for row in DUMMY_DATA:
    row["risked"] = row["achieved"] < row["target"]
# ------------------------------
# Prepare Documents & Vector Store
# ------------------------------
def load_dummy_documents():
    docs = []
    for row in DUMMY_DATA:
        content = "\n".join([f"{k}: {v}" for k, v in row.items()])
        docs.append(Document(page_content=content, metadata={"projectName": row["projectName"]}))
    return docs

def create_vector_store():
    documents = load_dummy_documents()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embedding=embeddings)
    return vectorstore

# ------------------------------
# Setup LLM + QA Chain
# ------------------------------
def setup_qa_chain():
    vectorstore = create_vector_store()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        temperature=0.3,
        top_p=0.85,
        repetition_penalty=1.2,
        truncation=True,
        do_sample=False,
        device=-1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain

qa_chain = setup_qa_chain()

app = FastAPI(title="Project QA Bot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",      # allow these origins
    allow_credentials=True,
    allow_methods=["*"],        # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],        # allow all headers
)
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    prompt = f"""
    You are a knowledgeable assistant trained on project data.
    Answer clearly and accurately. Include numbers, targets, or managers if relevant.
You are a project assistant. 
You are given project data including 'target' and 'achieved' values. 
A project is considered 'risked' if achieved < target.
Answer the user's question based on this data.

    Question: {request.question}
    """
    result = qa_chain(prompt)
    answer = result["result"]
    sources = [doc.page_content[:200] for doc in result["source_documents"]]

    return QueryResponse(answer=answer, sources=sources)

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
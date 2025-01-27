import os
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract

load_dotenv()

# Initialize LLMs and embeddings
llm_job_roles = ChatOpenAI(model="gpt-3.5-turbo")
llm_crs_score = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
llm_roadmap = ChatOpenAI(model="gpt-4", temperature=0.6)
embeddings = OpenAIEmbeddings()

class GraphState(TypedDict):
    questionnaire: str
    job_roles: str
    noc_codes: str
    crs_score: str
    roadmap: str

# OCR PDF Processing
def extract_text_from_pdf(file_path: str) -> str:
    # Convert PDF to images
    pages = convert_from_path(file_path, dpi=300)
    text = ""
    for page in pages:
        # Extract text from each page using Tesseract OCR
        text += pytesseract.image_to_string(page)
    return text

# Load NOC codes using OCR
file_path = "nocs (1).pdf"  # Path to your PDF file
ocr_text = extract_text_from_pdf(file_path)

# Split the extracted text into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(ocr_text)

# Create the FAISS index
noc_db = FAISS.from_texts(texts, embeddings)

# Define node functions
def determine_job_roles(state):
    questionnaire = state["questionnaire"]
    prompt = ChatPromptTemplate.from_template(
        """Based on the following client questionnaire, consider the educational background and work experience to determine the most relevant
         job roles:\n\n{questionnaire}, return JUST the job roles titles separated by comma, you should have a proper reasoning for recommending those roles  """)

    chain = prompt | llm_job_roles | StrOutputParser()
    job_roles = chain.invoke({"questionnaire": questionnaire})
    state["job_roles"] = job_roles
    return state

def retrieve_noc_codes(state):
    job_roles = state["job_roles"]
    relevant_docs = noc_db.similarity_search(job_roles, k=5)
    state["noc_codes"] = [doc.page_content for doc in relevant_docs]
    return state

def calculate_crs_score(state):
    questionnaire = state["questionnaire"]
    prompt = ChatPromptTemplate.from_template(
        CRS_prompt
    )
    chain = prompt | llm_crs_score | StrOutputParser()
    crs_score = chain.invoke({"questionnaire": questionnaire})
    state["crs_score"] = crs_score
    return state

def generate_roadmap(state):
    noc_codes = state["noc_codes"]
    crs_score = state["crs_score"]
    roadmap_prompt = system_prompt
    prompt = ChatPromptTemplate.from_template(roadmap_prompt)
    chain = prompt | llm_roadmap | StrOutputParser()
    roadmap = chain.invoke({"noc_codes": noc_codes, "crs_score": crs_score})
    state["roadmap"] = roadmap
    return state

# Define the graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("determine_job_roles", determine_job_roles)
workflow.add_node("retrieve_noc_codes", retrieve_noc_codes)
workflow.add_node("calculate_crs_score", calculate_crs_score)
workflow.add_node("generate_roadmap", generate_roadmap)

# Define edges
workflow.add_edge("determine_job_roles", "retrieve_noc_codes")
workflow.add_edge("retrieve_noc_codes", "calculate_crs_score")
workflow.add_edge("calculate_crs_score", "generate_roadmap")
workflow.add_edge("generate_roadmap", END)

# Set the entrypoint
workflow.set_entry_point("determine_job_roles")

# Compile the graph
graph_app = workflow.compile()

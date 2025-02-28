import os
from prompt import system_prompt, CRS_prompt
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader  # Change to PDF loader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# Initialize LLMs and embeddings
llm_job_roles = ChatOpenAI(model="gpt-4o")
llm_crs_score = ChatOpenAI(model="gpt-4o",temperature=0.4)
llm_roadmap = ChatOpenAI(model="gpt-4o",temperature=0.6)
embeddings = OpenAIEmbeddings()

from typing_extensions import TypedDict
class GraphState(TypedDict):
    questionnaire : str
    job_roles : str
    noc_codes : str
    crs_score : str
    roadmap : str

# Load NOC codes from PDF (adjust file path accordingly)
file_path = "nocs (1).pdf"  # Path to your PDF file
loader = PyPDFLoader(file_path)  # Use PyPDFLoader instead of CSVLoader
documents = loader.load()

# Create a text splitter (adjust chunk size as needed)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create the FAISS index
noc_db = FAISS.from_documents(texts, embeddings)

# Define node functions (same as before)
def determine_job_roles(state):
    questionnaire = state["questionnaire"]
    prompt = ChatPromptTemplate.from_template(
        """Based on the following client questionnaire, consider the educational background and work experience to determine the most relevant
         job roles, we would use these job roles roles to find NOC codes, these job roles don't necessarily have to be related 
         to degree or work experience, we can also recommend job roles which are in high demand.:\n\n{questionnaire}, return JUST the job roles titles with their pathways separated by comma, you should have a proper reasoning for recommending those roles""")

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
    
    crs_calculation_prompt = """You are a CRS (Comprehensive Ranking System) calculator for Canadian immigration. Follow these EXACT rules to calculate the score:

    1. First, analyze the questionnaire and identify:
       - Age
       - Education level
       - Language proficiency
       - Work experience
       - Canadian connections/experience

    2. Then calculate the score using these EXACT criteria:

    A. CORE/HUMAN CAPITAL FACTORS (max 460 points):
       - Age (max 110): 
         * 18 years: 90 points
         * 19 years: 95 points
         * 20-29 years: 100 points
         * 30 years: 95 points
         * 31 years: 90 points
         * 32 years: 85 points
         * 33 years: 80 points
         [continue exact age point breakdown]

       - Education (max 150):
         * PhD: 140 points
         * Master's: 135 points
         * Two or more degrees (one 3+ years): 128 points
         * Three-year degree: 120 points
         * Two-year degree: 98 points
         * One-year degree: 90 points
         * High school: 30 points

       - Language Skills (max 160):
         [Include exact CLB level points]

       - Work Experience (max 80):
         * 1 year: 40 points
         * 2 years: 53 points
         * 3 years: 64 points
         * 4 years: 72 points
         * 5+ years: 80 points

    B. ADDITIONAL POINTS:
       - Provincial Nomination: 600 points
       - Arranged Employment: 50 points
       - Canadian Education: 30 points
       - Canadian Work Experience: Up to 80 points

    Please analyze the following questionnaire and provide:
    1. A breakdown of points in each category
    2. The total CRS score
    3. Show the calculation of each score as well

    Questionnaire: {questionnaire}
    """

    prompt = ChatPromptTemplate.from_template(crs_calculation_prompt)
    chain = prompt | llm_crs_score | StrOutputParser()
    crs_score = chain.invoke({"questionnaire": questionnaire})
    state["crs_score"] = crs_score
    return state


# def calculate_crs_score(state):
#     questionnaire = state["questionnaire"]
#     prompt = ChatPromptTemplate.from_template(
#         CRS_prompt
#     )
#     chain = prompt | llm_crs_score | StrOutputParser()
#     crs_score = chain.invoke({"questionnaire": questionnaire})
#     state["crs_score"] = crs_score
#     return state

def generate_roadmap(state):
    questionnaire=state["questionnaire"]
    noc_codes = state["noc_codes"]
    crs_score = state["crs_score"]
    # print(f"Crs score is {crs_score}")
    roadmap_prompt = system_prompt
    prompt = ChatPromptTemplate.from_template(roadmap_prompt)
    chain = prompt | llm_roadmap | StrOutputParser()
    roadmap = chain.invoke({"questionnaire":questionnaire,"noc_codes": noc_codes, "crs_score": crs_score})
   
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
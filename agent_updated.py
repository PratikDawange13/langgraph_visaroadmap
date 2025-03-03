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
    eligible_programs: str
    

# Load NOC codes from PDF (adjust file path accordingly)
file_path = "nocs (1).pdf"  # Path to your PDF file
loader = PyPDFLoader(file_path)  # Use PyPDFLoader instead of CSVLoader
documents = loader.load()

# Create a text splitter (adjust chunk size as needed)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create the FAISS index
noc_db = FAISS.from_documents(texts, embeddings)

def determine_crs_score_first(state):
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

def identify_eligible_programs(state):
    crs_score = state["crs_score"]
    
    program_identification_prompt = """You are an immigration consultant with expertise in Canadian Express Entry programs.
    
    1. Analyze the client's CRS score: {crs_score}
    
    2. Check the current Express Entry draw cutoffs by referring to data from https://www.canada.ca/en/immigration-refugees-citizenship/corporate/mandate/policies-operational-instructions-agreements/ministerial-instructions/express-entry-rounds.html
    
    3. Based on the score, determine which Express Entry programs the client is CURRENTLY eligible for:
       - Federal Skilled Worker (FSW)
       - Canadian Experience Class (CEC)
       - Federal Skilled Trades (FST)
       - Provincial Nominee Programs (PNP)
    
    4. For each eligible program, briefly note the cutoff score and why the client qualifies.
    
    5. If the client doesn't currently qualify for any program, suggest which program would be most realistic to aim for and what improvements are needed.
    
    Return a concise summary of eligible programs and next steps.
    """
    
    prompt = ChatPromptTemplate.from_template(program_identification_prompt)
    chain = prompt | llm_job_roles | StrOutputParser()
    eligible_programs = chain.invoke({"crs_score": crs_score})
    state["eligible_programs"] = eligible_programs
    return state

def recommend_noc_codes(state):
    questionnaire = state["questionnaire"]
    eligible_programs = state["eligible_programs"]
    
    noc_recommendation_prompt = """As an immigration expert, recommend the most suitable NOC codes for this client based on:

    1. Client's background (from questionnaire): {questionnaire}
    
    2. Eligible immigration programs: {eligible_programs}
    
    3. Consider the following factors:
       - Match the client's education and work experience to appropriate NOC codes
       - Prioritize NOC codes with higher demand in Canada
       - Consider NOC codes the client could qualify for with minor additional training/education
       - Match NOC codes that align with eligible immigration programs
    
    For each recommended NOC code, provide:
    1. The NOC code and title
    2. Why it's suitable for the client
    3. Any additional training/education needed to qualify
    4. The immigration program(s) this NOC would support
    
    Query the RAG database to find the most appropriate NOC codes matching these criteria.
    """
    
    # Use the prompt to search the vector database
    relevant_docs = noc_db.similarity_search_with_score(
        noc_recommendation_prompt.format(questionnaire=questionnaire, eligible_programs=eligible_programs), 
        k=8
    )
    
    # Additional processing to refine NOC recommendations
    refinement_prompt = """Based on the client's questionnaire and eligible programs, review these potential NOC matches and select the MOST RELEVANT ones:
    
    Client questionnaire: {questionnaire}
    
    Eligible programs: {eligible_programs}
    
    Potential NOC matches:
    {noc_matches}
    
    For each recommended NOC, provide:
    1. NOC code and title
    2. Why it's suitable for this client specifically
    3. Any additional training/education needed (be specific)
    4. Which immigration program this NOC best supports
    5. Employment prospects in Canada
    
    Focus on NOCs where the client is already qualified or can become qualified with reasonable additional effort.
    """
    
    # Format the retrieved NOC documents
    noc_matches = "\n\n".join([f"Document (relevance score: {score}):\n{doc.page_content}" for doc, score in relevant_docs])
    
    prompt = ChatPromptTemplate.from_template(refinement_prompt)
    chain = prompt | llm_job_roles | StrOutputParser()
    recommended_nocs = chain.invoke({
        "questionnaire": questionnaire, 
        "eligible_programs": eligible_programs,
        "noc_matches": noc_matches
    })
    
    state["noc_codes"] = recommended_nocs
    return state

def generate_improved_roadmap(state):
    questionnaire = state["questionnaire"]
    noc_codes = state["noc_codes"]
    crs_score = state["crs_score"]
    eligible_programs = state["eligible_programs"]
    
    roadmap_prompt = """Based on the client's profile and immigration assessment, create a personalized roadmap to Canadian permanent residency:

    CLIENT PROFILE:
    {questionnaire}
    
    CRS SCORE ASSESSMENT:
    {crs_score}
    
    ELIGIBLE IMMIGRATION PROGRAMS:
    {eligible_programs}
    
    RECOMMENDED NOC CODES:
    {noc_codes}
    
    Your roadmap should include:
    
    1. IMMEDIATE STEPS (Next 3 months):
       - Document preparation
       - Language testing or improvement
       - Education credential assessment
       - Any immediate training needs
    
    2. MEDIUM-TERM STRATEGY (3-12 months):
       - Express Entry profile creation timing
       - Job search strategies for selected NOCs
       - Provincial nomination possibilities
       - Additional education/training needed
    
    3. LONG-TERM PLAN (1-2 years):
       - Backup pathways if initial strategy doesn't succeed
       - CRS score improvement opportunities
       - Timeline for expected PR approval
    
    4. COST CONSIDERATIONS:
       - Application fees
       - Settlement funds required
       - Training/education costs
    
    Be specific, practical, and tailor the advice to this client's unique situation.
    """
    
    prompt = ChatPromptTemplate.from_template(roadmap_prompt)
    chain = prompt | llm_roadmap | StrOutputParser()
    roadmap = chain.invoke({
        "questionnaire": questionnaire,
        "noc_codes": noc_codes, 
        "crs_score": crs_score,
        "eligible_programs": eligible_programs
    })
   
    state["roadmap"] = roadmap
    return state

# Define the new graph
workflow = StateGraph(GraphState)

# Add nodes with the new flow
workflow.add_node("determine_crs_score_first", determine_crs_score_first)
workflow.add_node("identify_eligible_programs", identify_eligible_programs)
workflow.add_node("recommend_noc_codes", recommend_noc_codes)
workflow.add_node("generate_improved_roadmap", generate_improved_roadmap)

# Define edges with the new flow
workflow.add_edge("determine_crs_score_first", "identify_eligible_programs")
workflow.add_edge("identify_eligible_programs", "recommend_noc_codes")
workflow.add_edge("recommend_noc_codes", "generate_improved_roadmap")
workflow.add_edge("generate_improved_roadmap", END)

# Set the entrypoint
workflow.set_entry_point("determine_crs_score_first")

# Compile the graph
graph_app = workflow.compile()


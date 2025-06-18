from dotenv import load_dotenv
import re
from langchain_groq import ChatGroq
import streamlit as st
from fpdf import FPDF
from io import BytesIO

from langchain_community.embeddings import HuggingFaceHubEmbeddings


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import ArxivLoader
from typing_extensions import TypedDict,Optional
from langchain_core.messages import AnyMessage,HumanMessage, SystemMessage, AIMessage ## Human message or AI message
from typing import Annotated,Literal ## labelling
from langgraph.graph.message import add_messages
from huggingface_hub import login

from langgraph.graph import StateGraph, START, END

load_dotenv()

import os
import requests

st.title("Automated Research assistant")
st.subheader("Generate New Research ideas")


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")




llm=ChatGroq(model="llama-3.3-70b-versatile")
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
)# for render


class State(TypedDict):
    # messages: Annotated[list[AnyMessage], add_messages]
    messages: Annotated[list[AnyMessage], add_messages]
    summary: Optional[str]
    research_ideas: Optional[str]
    critique: Optional[str]
    experiment_plan: Optional[str]
    final_report: Optional[str]
    verdict: Optional[str]
    report_file_bytes: Optional[str]

def topic_gen(state:State):
    user_prompt = state["messages"][-1].content

    system_prompt = (
        "You are a research assistant. Based on the user's interest, return a single-line search query using OR operators "
        "between 2–5 keywords/phrases, without explanations. This query will be used for searching arXiv."
    )



    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(prompt)

    cleaned_response = response.content.strip().replace('"', '')

    response.content = cleaned_response
    return {
        "messages": state["messages"] + [response]
    }

def arxiv_search(state:State):
    last_message = state["messages"][-1]

    if last_message.type != "ai":
        raise ValueError("Expected last message to be from assistant (the generated query).")

    query = last_message.content.strip()
    print("🔍 Using query for arXiv search:", query)

    arxiv = ArxivLoader(
    query=query
    )

    results = arxiv.get_summaries_as_docs()  # returns List[Document]

    if isinstance(results, str):
        raise TypeError("arxiv.run() returned a string instead of Document objects.")

    if not isinstance(results, list):
        results = [results]

    os.makedirs("downloaded_papers", exist_ok=True)
    downloaded_files = []

    for i, doc in enumerate(results):
        metadata = doc.metadata
        pdf_url = re.sub(r'abs', 'pdf', metadata.get("Entry ID"))
        title = metadata.get("title", f"paper_{i+1}")
        safe_title = title.replace(" ", "_").replace("/", "_")[:50]
        filepath = f"downloaded_papers/{safe_title}.pdf"

        if pdf_url:
            try:
                response = requests.get(pdf_url)
                if response.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    print(f"✅ Downloaded: {filepath}")
                    downloaded_files.append(filepath)
                else:
                    print(f"❌ Failed to download {pdf_url}")
            except Exception as e:
                print(f"⚠️ Error downloading {pdf_url}: {e}")
        else:
            print(f"⚠️ No PDF URL found for {title}")

    return {
        "messages": state["messages"],
        "downloaded_papers": downloaded_files
    }


def summarizer(state:State):

    prompt= ChatPromptTemplate.from_template(
    
    """
    You are a helpful research assistant. Given the following academic paper(s) from vector store, summarize the key findings and contributions of each one in bullet points.
    Format the summary clearly with headings if multiple papers are involved.
    <context>
    {context}
    </context>
    Question:{input}

    """
    )
    loader=PyPDFDirectoryLoader("downloaded_papers")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)
    vectors = FAISS.from_documents(final_documents,embedding)

    doc_chain = create_stuff_documents_chain(llm,prompt)
    retriever = vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,doc_chain)
    response = retriever_chain.invoke({"input": "Summarize the papers."})

    return {
        "messages": state["messages"] + [AIMessage(content=response["answer"])],
        "summary": response["answer"]
    }

def idea_synthesizer(state: State):
    summary = state.get("summary", "")
    if not summary:
        raise ValueError("Summary not found in state. Ensure summarizer node ran correctly.")

    system_prompt = (
        "You are a research scientist. Based on the following summary of academic papers, generate 3–5 original research ideas. "
        "These should be novel, feasible, and build upon the existing work. Return them as bullet points."
    )

    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=summary)
    ]

    response = llm.invoke(prompt)

    return {
        "messages": state["messages"] + [response],
        "research_ideas": response.content
    }

def critic_agent(state: State):
    ideas = state.get("research_ideas", "")
    verdict="proceed"
    if not ideas:
        raise ValueError("Research ideas not found in state.")

    system_prompt = (
        "You are an expert research reviewer. Analyze the following research ideas based on three criteria: "
        "1) Feasibility, 2) Novelty, and 3) Potential impact. Provide detailed feedback on each idea. "
        "At the end, summarize whether the overall set of ideas is 'promising' or 'needs revision'."
    )

    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=ideas)
    ]

    response = llm.invoke(prompt)

    critique_text = response.content
    if "needs revision" in critique_text.lower():
        verdict = "revise"
    else:
        verdict = "proceed"

    return {
        "messages": state["messages"] + [response],
        "critique": critique_text,
        "verdict": verdict
    }

def experiment_designer(state: State):
    critiques = state.get("critique", "")
    ideas = state.get("research_ideas", "")

    if not ideas or not critiques:
        raise ValueError("Missing research ideas or critiques. Ensure previous nodes ran successfully.")

    system_prompt = (
        "You are a senior AI researcher. Based on the following research ideas and their critique, "
        "design a basic experimental plan for the most promising idea. "
        "Include details like objective, dataset (real or synthetic), method, metrics, and expected outcome."
    )

    full_input = f"Research Ideas:\n{ideas}\n\nCritiques:\n{critiques}"

    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=full_input)
    ]

    response = llm.invoke(prompt)

    return {
        "messages": state["messages"] + [response],
        "experiment_plan": response.content
    }


# def report_generator(state: State):
    summary = state.get("summary", "")
    ideas = state.get("research_ideas", "")
    critiques = state.get("critique", "")
    experiment_plan = state.get("experiment_plan", "")

    if not (summary and ideas and critiques and experiment_plan):
        raise ValueError("Some components are missing. Ensure all previous nodes completed successfully.")

    system_prompt = (
        "You are an AI research assistant. Based on the provided research pipeline components, "
        "generate a clean, professional markdown-formatted report."
    )

    user_input = f"""
# 📚 Literature Summary
{summary}

# 💡 Research Ideas
{ideas}

# 🧪 Critique of Ideas
{critiques}

# 🧬 Experiment Plan
{experiment_plan}
"""

    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    response = llm.invoke(prompt)

    return {
        "messages": state["messages"] + [response],
        "final_report": response.content
    }



# def report_generator(state: State):
    experiment_plan = state.get("experiment_plan", "")
    critique = state.get("critique", "")
    ideas = state.get("research_ideas", "")
    summary = state.get("summary", "")
    
    if not experiment_plan:
        raise ValueError("Experiment plan missing from state.")
    
    # Assemble report content
    report_content = f"""
    # Autonomous Research Lab Report

    ## Summary
    {summary}

    ## Research Ideas
    {ideas}

    ## Critique
    {critique}

    ## Experiment Plan
    {experiment_plan}
    """
    
    # Generate PDF using FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Handle multi-line text
    for line in report_content.split('\n'):
        pdf.multi_cell(0, 10, line)
    
    output_path = "autonomous_research_lab_report.pdf"
    pdf.output(output_path)
    
    print(f"✅ Report saved to {output_path}")
    
    return {
        "messages": state["messages"],
        "final_report": report_content,
        "report_file_path": output_path
    }





# def report_generator(state: State):
    experiment_plan = state.get("experiment_plan", "")
    critique = state.get("critique", "")
    ideas = state.get("research_ideas", "")
    summary = state.get("summary", "")
    
    if not experiment_plan:
        raise ValueError("Experiment plan missing from state.")
    
    # Assemble report content
    pdf = FPDF()
    pdf.add_page()

    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 14)


    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Autonomous Research Lab Report", ln=True, align='C')
    pdf.ln(10)

    # Section: Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "📚 Summary", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, summary)
    pdf.ln(5)

    # Section: Research Ideas
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "💡 Research Ideas", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, ideas)
    pdf.ln(5)

    # Section: Critique
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "📝 Critique", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, critique)
    pdf.ln(5)

    # Section: Experiment Plan
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "🧪 Experiment Plan", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, experiment_plan)

    # Output to memory buffer
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    
    return {
        "messages": state["messages"],
        "final_report": "PDF generated.",
        "report_file_bytes": pdf_buffer.read()
    }


# from fpdf import FPDF
# from io import BytesIO

def report_generator(state: State):
    experiment_plan = state.get("experiment_plan", "")
    critique = state.get("critique", "")
    ideas = state.get("research_ideas", "")
    summary = state.get("summary", "")
    
    if not experiment_plan:
        raise ValueError("Experiment plan missing from state.")
    
    pdf = FPDF()
    pdf.add_page()

    # Add and use DejaVu (or another Unicode font)
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)

    # Title
    pdf.set_font("DejaVu", 'B', 16)
    pdf.cell(0, 10, "Autonomous Research Lab Report", ln=True, align='C')
    pdf.ln(10)

    # Section: Summary
    pdf.set_font("DejaVu", 'B', 14)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("DejaVu", '', 12)
    pdf.multi_cell(0, 8, summary)
    pdf.ln(5)

    # Section: Research Ideas
    pdf.set_font("DejaVu", 'B', 14)
    pdf.cell(0, 10, "Research Ideas", ln=True)
    pdf.set_font("DejaVu", '', 12)
    pdf.multi_cell(0, 8, ideas)
    pdf.ln(5)

    # Section: Critique
    pdf.set_font("DejaVu", 'B', 14)
    pdf.cell(0, 10, "Critique", ln=True)
    pdf.set_font("DejaVu", '', 12)
    pdf.multi_cell(0, 8, critique)
    pdf.ln(5)

    # Section: Experiment Plan
    pdf.set_font("DejaVu", 'B', 14)
    pdf.cell(0, 10, "Experiment Plan", ln=True)
    pdf.set_font("DejaVu", '', 12)
    pdf.multi_cell(0, 8, experiment_plan)

    # Output to memory buffer
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    return {
        "messages": state["messages"],
        "final_report": "PDF generated.",
        "report_file_bytes": pdf_buffer.read()
    }


def condition_node(state:State)-> Literal["Exp-Plan","Idea"]:
    verdict = state.get("verdict","proceed")
    if verdict=="proceed":
        return "Exp-Plan"
    return "Idea"

builder = StateGraph(State)
builder.add_node("Topic Generation",topic_gen)
builder.add_node("ARXIV",arxiv_search)
builder.add_node("Summarizer",summarizer)
builder.add_node("Idea",idea_synthesizer)
builder.add_node("Critic",critic_agent)
builder.add_node("Exp-Plan",experiment_designer)
builder.add_node("Final-Report",report_generator)


builder.add_edge(START,"Topic Generation")
builder.add_edge("Topic Generation","ARXIV")
builder.add_edge("ARXIV","Summarizer")
builder.add_edge("Summarizer","Idea")
builder.add_edge("Idea","Critic")
builder.add_conditional_edges("Critic",condition_node)
builder.add_edge("Exp-Plan","Final-Report")
builder.add_edge("Final-Report",END)

graph = builder.compile()


inp = st.text_input("Enter the keyword about what to research new ideas")

if st.button("Generate"):
    messages=graph.invoke({"messages":inp})
    for m in messages['messages']:
        st.write(m.content)

    for m in messages['messages']:
        st.write(m.content)

    # Offer download button if PDF bytes exist
    pdf_bytes = messages.get("report_file_bytes")
    
    if pdf_bytes:
        st.download_button(
            label="📄 Download Research Report PDF",
            data=pdf_bytes,
            file_name="autonomous_research_lab_report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("No PDF generated yet.")
# cleaned_workflow.py
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Dict, Any
import sys
import logging
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_core.runnables import RunnableParallel,RunnableLambda
from langchain_community.vectorstores import Chroma, FAISS
from langgraph.graph import StateGraph, START, END
from Exception import CustomException
from logger import logging
import os

# ---- 1. Load env keys ----
logging.info("Loading environment variables")
load_dotenv()
logging.info("Env loaded")

# ---- 2. Create models ----
logging.info("Loading embedding and LLM endpoints")
Embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
base_llm = HuggingFaceEndpoint(repo_id="google/gemma-2-9b-it", task="conversational")
llm_model = ChatHuggingFace(llm=base_llm)
logging.info("Models prepared")

# ---- 3. State typing ----
class ImpState(TypedDict):
    user_input: Literal["Generate Report", "Generate Insights", "Generate Summary", "Generate Quiz"]
    user_data: str
    summary_node: str
    summary_score: int
    insights_node: str
    insights_score: int
    Quiz_node: str
    Quiz_score: int
    Report_node: str
    Report_score: int
    final_feedback: Literal["approved", "not approved"]
    flaws: str
    final_score: int
    final_output: str
    optimised: str

# ---- 4. Pydantic models for structured outputs ----
class Structure(BaseModel):
    summary: str
    score: int = Field(..., description="Rate it out of 10")  # use description instead of string default

class FeedbackStructure(BaseModel):
    final: str  
    score: int
    flaws: str
    summary: Literal["approved", "not approved"]

# ---- 5. Node implementations (fixed prompt usage) ----
def llm_node(state: ImpState):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunks = text_splitter.create_documents([state["user_data"]])
    combined_text = " ".join([chunk.page_content for chunk in chunks])
    logging.info("llm_node invoked (text chunking complete)")
    return {"user_data": combined_text}


def _invoke_structured_model(repo_id: str, template: PromptTemplate, variables: Dict[str, Any], out_model: BaseModel):
    llm = HuggingFaceEndpoint(repo_id=repo_id, task="text-generation")
    model = ChatHuggingFace(llm=llm)
    prompt_text = template.format(**variables)
    output = model.with_structured_output(out_model)   # this was missing
    response = output.invoke(prompt_text)
    return response


def summary_node(state: Dict[str, Any]) -> Dict[str, Any]:
    loaded_doc = state["user_data"]
    prompt = PromptTemplate(
        template=(
            "You are an expert in generating summaries from documents.\n"
            "Generate the best possible summary from the following text:\n\n"
            "{loaded_doc}\n\n"
            "Focus your summary based on the user input: {user_input}"
        ),
        input_variables=["loaded_doc", "user_input"]
    )
    variables = {"loaded_doc": loaded_doc, "user_input": state["user_input"]}
    repo = "knkarthick/MEETING_SUMMARY"
    try:
        response = _invoke_structured_model(repo, prompt, variables, Structure)
    except Exception as e:
        raise CustomException(e, sys)
    return {"summary_node": response.summary, "summary_score": response.score}

def insights_node(state: Dict[str, Any]) -> Dict[str, Any]:
    loaded_doc = state["user_data"]
    prompt = PromptTemplate(
        template=(
            "You are an expert in generating Key Insights from documents.\n"
            "Generate deep, valuable insights derived from the following text:\n\n"
            "{loaded_doc}\n\n"
            "Focus your insights based on the user input: {user_input}"
        ),
        input_variables=["loaded_doc", "user_input"]
    )
    variables = {"loaded_doc": loaded_doc, "user_input": state["user_input"]}
    repo = "mradermacher/LLAMA-3.2-1b-HRV-Insights-GGUF"
    response = _invoke_structured_model(repo, prompt, variables, Structure)
    return {"insights_node": response.summary, "insights_score": response.score}


def Quiz_node(state: Dict[str, Any]) -> Dict[str, Any]:
    loaded_doc = state["user_data"]
    prompt = PromptTemplate(
        template=(
            "You are an expert in generating quizzes from documents.\n"
            "Produce challenging, relevant quiz questions (with answers) from the following text:\n\n"
            "{loaded_doc}\n\n"
            "Focus your Quiz based on the user input: {user_input}"
        ),
        input_variables=["loaded_doc", "user_input"]
    )
    variables = {"loaded_doc": loaded_doc, "user_input": state["user_input"]}
    repo = "tutikentuti/whisper-tiny-quiztest"
    response = _invoke_structured_model(repo, prompt, variables, Structure)
    return {"Quiz_node": response.summary, "Quiz_score": response.score}


def Report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    loaded_doc = state["user_data"]
    prompt = PromptTemplate(
        template=(
            "You are an expert in generating detailed academic-style reports from documents.\n"
            "Generate a well-structured report from the following text:\n\n"
            "{loaded_doc}\n\n"
            "Focus your report based on the user input: {user_input}\n\n"
            "Include relevant facts, logical flow, and a professional tone."
        ),
        input_variables=["loaded_doc", "user_input"]
    )
    variables = {"loaded_doc": loaded_doc, "user_input": state["user_input"]}
    repo = "dz-osamu/Report_Generation"
    response = _invoke_structured_model(repo, prompt, variables, Structure)
    return {"Report_node": response.summary, "Report_score": response.score}



def final_feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    generated_output = (
        state.get("insights_node") or state.get("Quiz_node") or state.get("Report_node") or state.get("summary_node")
    )
    prompt = PromptTemplate(
        template=(
            "You are an Expert Evaluator.\n"
            "Evaluate the following output:\n\n"
            "{generated_output}\n\n"
            "Rate it out of 10 for clarity, depth, and grammar.\n"
            "If the score is less than 7, set feedback='not approved'.\n"
            "If the score is 7 or higher, set feedback='approved' and also include the full text back.\n"
            "Also list all flaws in the output."
        ),
        input_variables=["generated_output"]
    )
    variables = {"generated_output": generated_output}
    repo = "theojolliffe/model-3-feedback"
    # We expect a different structured schema
    response = _invoke_structured_model(repo, prompt, variables, FeedbackStructure)
    # Map fields consistently
    return {
        "final_feedback": response.summary,
        "final_score": response.score,
        "final_output": response.final,
        "flaws": response.flaws
    }



def Optimise(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = PromptTemplate(
        template=(
            "You are an Expert Answer Optimizer.\n"
            "Evaluate the following output:\n\n"
            "{final_output}\n\n"
            "Its previous rating is {final_score}. The flaws are: {flaws}.\n"
            "Produce an optimised improved version and a brief note on the improvements."
        ),
        input_variables=["final_output", "final_score", "flaws"]
    )
    variables = {
        "final_output": state.get("final_output", ""),
        "final_score": state.get("final_score", 0),
        "flaws": state.get("flaws", "")
    }
    repo = "Aakali/llama-2-70b-chat-optimised3"
    llm = HuggingFaceEndpoint(repo_id=repo, task="text-generation")
    model = ChatHuggingFace(llm=llm)
    prompt_text = prompt.format(**variables)
    # Here we just want raw text result
    response = model.invoke(prompt_text)
    return {"optimised": response}

# ---- Decision function ----


def decision_maker(state: Dict[str, Any]) -> str:
    choice = state["user_input"]
    if choice == "Generate Report":
        return "Report_node"
    if choice == "Generate Insights":
        return "insights_node"
    if choice == "Generate Quiz":
        return "Quiz_node"
    if choice == "Generate Summary":
        return "summary_node"
    # default
    return "summary_node"

# ---- Graph construction ----
def build_and_run_graph(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    graph = StateGraph(ImpState)
    # Register nodes by name
    graph.add_node("llm_node", llm_node)
    graph.add_node("summary_node", summary_node)
    graph.add_node("insights_node", insights_node)
    graph.add_node("Quiz_node", Quiz_node)
    graph.add_node("Report_node", Report_node)
    graph.add_node("final_feedback_node", final_feedback_node)
    graph.add_node("Optimise", Optimise)

    graph.add_edge(START, "llm_node")
    # conditional edge; I assume API expects mapping from outcomes to node *names*
    graph.add_conditional_edge("llm_node", decision_maker, {
        "Generate Report": "Report_node",
        "Generate Insights": "insights_node",
        "Generate Quiz": "Quiz_node",
        "Generate Summary": "summary_node",
    })

    # all result nodes flow into final feedback
    graph.add_edge("summary_node", "final_feedback_node")
    graph.add_edge("insights_node", "final_feedback_node")
    graph.add_edge("Report_node", "final_feedback_node")
    graph.add_edge("Quiz_node", "final_feedback_node")


    def final_feedback_conditional(state: Dict[str, Any]) -> str:
        return "Optimise" if state.get("final_feedback") == "not approved" else END

    
    graph.add_edge("final_feedback_node", "Optimise")
    graph.add_edge("Optimise", "final_feedback_node")
    workflow = graph.compile()
    response = workflow.invoke(initial_state)  # pass full state including user_data / user_input
    return response
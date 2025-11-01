from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
# from langchain_community.document_loaders import PyMuPDF,WebBaseLoader,DirectoryLoader, ArxivLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Chroma,FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import TypedDict,Literal
from dotenv import load_dotenv
from Exception import CustomException
from logger import logging
import sys

# 1- Loading APIs
logging.info("API Key Loading")
load_dotenv()
logging.info("API Key Loaded")
#2- Creating Models
logging.info("LLM Model Loading")
Embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

llm= HuggingFaceEndpoint(repo_id= "google/gemma-2-9b-it",
                                task = "conversational")
llm_model = ChatHuggingFace(llm=llm)
logging.info("LLM Model Loaded")


#3- Setting up the State
logging.info("Creating a State")
class ImpState(TypedDict):
     user_input: Literal["Generate Report","Generate Insights","Generate Summary","Generate Quiz"]
     user_data: str
     summary_node:str
     summary_score: int
     insights_node: str
     insights_score:int
     Quiz_node:str
     Quiz_score:int
     Report_node: str
     Report_score: int
     final_feedback: Literal["approved,not approved"]
     final_score: int
     final_output: str


logging.info("State Created")





#4- Setting Pydantic For Formatted Ouptuts
class structure(BaseModel):
      summary:str 
      score: int = Field("Rate it out of 10")

# 5- Setting up the Tools of Langchain(Decision Node,Summary, Insight, Q/A, Report Node)
logging.info("Setting up tools")

def llm_node():
     pass

def summary_node():
     prompt = PromptTemplate(
     template=(
        "You are an expert in generating summaries from documents.\n"
        "Generate the best possible summary from the following text:\n\n"
        "{loaded_doc}\n\n"
        "Focus your summary based on the user input: {user_input}"
    ),
    input_variables=["loaded_doc", "user_input"])
     llm=HuggingFaceEndpoint(repo_id = "knkarthick/MEETING_SUMMARY",task = "text-generation")
     model = ChatHuggingFace(llm=llm)
     output  = model.with_structured_output(structure)
     response = output.invoke(prompt)
     return {'summary_node':response.summary,'summary_score':response.score}
     

def insights_node():
     prompt = PromptTemplate(
     template=(
        "You are an expert in generating  Key Insights from documents.\n"
        "Generate the best possible Insights involving deeper understanding or new, valuable conclusions derived from analysis from the following text:\n\n"
        "{loaded_doc}\n\n"
        "Focus your Insights based on the user input: {user_input}"
    ),
    input_variables=["loaded_doc", "user_input"])
     llm=HuggingFaceEndpoint(repo_id = "mradermacher/LLAMA-3.2-1b-HRV-Insights-GGUF",task = "text-generation")
     model = ChatHuggingFace(llm=llm)
     output  = model.with_structured_output(structure)
     response = output.invoke(prompt)
     return {'insights_node':response.summary,'insights_score':response.score}
     

def Quiz_node():
     prompt = PromptTemplate(
     template=(
        "You are an expert in generating Quiz from documents.\n"
        "Generate the best possible Quiz involving deeper questions from the topic from the following text:\n\n"
        "{loaded_doc}\n\n"
        "Focus your Quiz questions based on the user input: {user_input}"
    ),
    input_variables=["loaded_doc", "user_input"])
     llm=HuggingFaceEndpoint(repo_id = "tutikentuti/whisper-tiny-quiztest",task = "text-generation")
     model = ChatHuggingFace(llm=llm)
     output  = model.with_structured_output(structure)
     response = output.invoke(prompt)
     return {'Quiz_node':response.summary,'Quiz_score':response.score}
     

def Report_node():
     prompt = PromptTemplate(
        template=(
            "You are an expert in generating detailed academic-style reports from documents.\n"
            "Generate a well-structured report or essay from the following text:\n\n"
            "{loaded_doc}\n\n"
            "Focus your report based on the user input: {user_input}\n\n"
            "Include relevant facts, logical flow, and professional tone."
        ),
        input_variables=["loaded_doc", "user_input"]
    )
     llm=HuggingFaceEndpoint(repo_id = "dz-osamu/Report_Generation",task = "text-generation")
     model = ChatHuggingFace(llm=llm)
     output  = model.with_structured_output(structure)
     response = output.invoke(prompt)
     return {'Report_node':response.summary,'Report_score':response.score}



def final_feedback_node():
     pass

def Optimise():
     pass


def decision_maker(state:ImpState):
     if state["user_input"] == "Generate Report":
           return "Report_node"
     if state["user_input"] == "Generate Insights":
           return "insights_node"
     if state["user_input"] == "Generate Quiz":
           return "Quiz_node"
     if state["user_input"] == "Generate Summary":
           return "summary_node"

logging.info("tools are ready to use")

#6- Setting up Nodes and Edges

def nodes():
 logging.info("Creating Nodes and Edges")
 graph = StateGraph(ImpState)
 graph.add_node('llm_node',llm_node)
 graph.add_node('summary_node',summary_node)
 graph.add_node('insights_node',insights_node)
 graph.add_node('Quiz_node',Quiz_node)
 graph.add_node('Report_node',Report_node)
 graph.add_node('final_feedback_node',final_feedback_node)
 graph.add_node('Optimise',Optimise)



 graph.add_edge(START,'llm_node')
 graph.add_conditional_edge('llm_node',decision_maker,
                                                      {'not_approved':'Optimise'})
 graph.add_edge('summary_node','final_feedback_node')
 graph.add_edge('insights_node','final_feedback_node')
 graph.add_edge('Report_node','final_feedback_node')
 graph.add_edge('Quiz_node','final_feedback_node')
 graph.add_edge('Optimise','final_feedback_node')
 logging.info("Nodes and Edges Created")

 try:
      user_input = input("What you want to Generate")
      workflow = graph.compile()
      response = workflow.invoke({'user_input':user_input})
      return response
 except Exception as e:
   raise CustomException(e,sys)

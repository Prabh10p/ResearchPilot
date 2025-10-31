from langchain_hugginface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDF,WebBaseLoader,DirectoryLoader, ArxivLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectostore import Chroma,FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterText
from langraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import TypedDict,Literal
from dotenv import load_dotenv

# 1- Loading APIs
load_dotenv()

#2- Creating Models
Embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")


llm= HuggingFaceEndpoint(repo_id= "google/gemma-2-9b-it",
                                task = "conversational")
llm_model = ChatHuggingFace(llm=llm)



#3- Setting up the State
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









# 4- Setting up the Tools of Langgag(Decision Node,Summary, Insight, Q/A, Report Node)






def decision_maker(state:ImpState):
     if state["user_input"] == "Generate Report":
           return "Report_node"
     if state["user_input"] == "Generate Insights":
           return "insights_node"
     if state["user_input"] == "Generate Quiz":
           return "Quiz_node"
     if state["user_input"] == "Generate Summary":
           return "summary_node"



#5- Setting up Nodes and Edges
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
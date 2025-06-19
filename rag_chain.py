import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.callbacks import get_openai_callback
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", message=".*get_num_tokens_from_messages.*")
warnings.filterwarnings("ignore", message=".*model not found.*")
warnings.filterwarnings("ignore", message=".*Unable to import simsimd.*")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.*")
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant.*")

load_dotenv()

class AdvancedRAGChain:
    def __init__(self, collection_name: str = "my_documents", max_tokens: int = 1000):
        self.collection_name = collection_name
        self.max_tokens = max_tokens
        self.setup_components()
        
    def setup_components(self):
        """Initialize all RAG components with advanced features"""
        self.llm = ChatOpenAI(
            temperature=0.1,  
            model="gpt-3.5-turbo",  # tiktoken version only supports gpt-3.5-turbo
            max_tokens=self.max_tokens
        )

        self.embeddings = OpenAIEmbeddings()
        self.client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"]
        )
        
        self.vectordb = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )
        
        base_retriever = self.vectordb.as_retriever(
            search_type="mmr",  
            search_kwargs={
                "k": 6,  
                "fetch_k": 20, 
            }
        )
        
        # Compress retrieved documents to most relevant parts
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        # Setup memory with summary buffer (keeps recent messages + summary of older ones)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=800,  
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.custom_prompt = self.create_custom_prompt()
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": self.custom_prompt}
        )
        
    def create_custom_prompt(self) -> PromptTemplate:
        """Create a custom prompt template for better responses"""
        template = """You are an intelligent assistant helping users understand documents. Use the following pieces of context to answer the question at the end.

Context from documents:
{context}

Instructions:
- Provide accurate, helpful answers based on the context
- If you don't know something, say so clearly
- Reference specific parts of the documents when relevant
- Be conversational and consider the chat history
- If the question seems related to previous conversation, acknowledge that context

Question: {question}

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with cost tracking"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                with get_openai_callback() as cb:
                    result = self.qa_chain({"question": question})
                    
                    # Enhanced response with metadata
                    enhanced_result = {
                        "answer": result["answer"],
                        "source_documents": result["source_documents"],
                        "chat_history": self.memory.chat_memory.messages,
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "tokens_used": cb.total_tokens,
                            "cost": cb.total_cost,
                            "num_sources": len(result["source_documents"])
                        }
                    }
                    
                    return enhanced_result
                    
            except Exception as e:
                return {
                    "answer": f"Sorry, I encountered an error: {str(e)}",
                    "source_documents": [],
                    "chat_history": [],
                    "metadata": {"error": str(e)}
                }
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if hasattr(self.memory, 'moving_summary_buffer') and self.memory.moving_summary_buffer:
            return self.memory.moving_summary_buffer
        return "No conversation history yet."
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory usage"""
        messages = self.memory.chat_memory.messages
        return {
            "total_messages": len(messages),
            "human_messages": len([m for m in messages if isinstance(m, HumanMessage)]),
            "ai_messages": len([m for m in messages if isinstance(m, AIMessage)]),
            "has_summary": bool(getattr(self.memory, 'moving_summary_buffer', None))
        }

def create_advanced_qa_chain(collection_name: str = "my_documents") -> AdvancedRAGChain:
    """Factory function to create advanced RAG chain"""
    return AdvancedRAGChain(collection_name)
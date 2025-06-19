import chainlit as cl
from rag_chain import AdvancedRAGChain
import warnings
import os

warnings.filterwarnings("ignore", message=".*get_num_tokens_from_messages.*")
warnings.filterwarnings("ignore", message=".*model not found.*")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.*")

@cl.on_chat_start
async def on_chat_start():
    try:
        rag_chain = AdvancedRAGChain()
        cl.user_session.set("rag_chain", rag_chain)
        
        # Send an engaging welcome message
        welcome_message = """**Advanced RAG Document Intelligence System**

Welcome to your intelligent document assistant! I'm powered by a sophisticated RAG pipeline that brings your documents to life.

**What I Can Do For You:**

**🧠 Smart Retrieval**
• Use Maximum Marginal Relevance (MMR) to find the most relevant AND diverse information
• Contextual compression extracts only the most important parts from documents
• Semantic search understands the meaning behind your questions, not just keywords

**💬 Conversational Memory**
• Remember our entire conversation with intelligent summary buffering
• Build on previous questions and provide contextual follow-ups
• Maintain conversation flow while managing memory efficiently

**📊 Advanced Analytics**
• Track token usage and costs for transparency
• Provide source citations with page numbers
• Monitor retrieval quality and response metadata

**🔍 Intelligent Features**
• Ask complex, multi-part questions
• Request clarifications or follow-up explanations
• Explore document relationships and cross-references
• Get summaries, comparisons, and deep analysis

## 💡 **Try asking me:**
• *"What are the main themes across all documents?"*
• *"Can you compare the approaches mentioned in different sections?"*
• *"Summarize the key findings and explain their implications"*
• *"What questions does this raise that aren't answered in the documents?"*

Ready to explore your documents intelligently? Ask me anything! 🎉"""
        
        await cl.Message(
            content=welcome_message,
            author="RAG Assistant"
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"**System Initialization Error**\n\nI encountered an issue setting up the RAG system: {str(e)}\n\nPlease check your document ingestion and try again.",
            author="System"
        ).send()

@cl.on_message
async def main(message):
    try:
        # Handle special commands first
        content = message.content.lower().strip()
        
        if content in ["/clear", "/reset", "/new"]:
            rag_chain = cl.user_session.get("rag_chain")
            if rag_chain:
                rag_chain.clear_memory()
                await cl.Message(
                    content="🔄 **Memory Cleared**\n\nConversation history has been reset. You can start fresh!",
                    author="Assistant"
                ).send()
            return
        
        elif content in ["/stats", "/memory", "/info"]:
            rag_chain = cl.user_session.get("rag_chain")
            if rag_chain:
                stats = rag_chain.get_memory_stats()
                summary = rag_chain.get_conversation_summary()
                
                stats_msg = f"""📊 **System Statistics**\n\n**Memory Usage:**\n• Total messages: {stats['total_messages']}\n• Your questions: {stats['human_messages']}\n• My responses: {stats['ai_messages']}\n• Has summary: {'Yes' if stats['has_summary'] else 'No'}\n\n**Conversation Summary:**\n{summary if summary != "No conversation history yet." else "Fresh conversation - no history yet."}"""
                
                await cl.Message(content=stats_msg, author="Assistant").send()
            return
        
        # Get the RAG chain instance
        rag_chain = cl.user_session.get("rag_chain")
        
        if not rag_chain:
            await cl.Message(
                content="🔄 **Reinitializing System**\n\nLet me restart the RAG pipeline...",
                author="System"
            ).send()
            
            try:
                rag_chain = AdvancedRAGChain()
                cl.user_session.set("rag_chain", rag_chain)
                await cl.Message(
                    content="✅ **System Ready**\n\nRAG pipeline reinitialized successfully! Please ask your question again.",
                    author="System"
                ).send()
                return
            except Exception as e:
                await cl.Message(
                    content=f"❌ **Initialization Failed**\n\nUnable to initialize RAG system: {str(e)}",
                    author="System"
                ).send()
                return
        
        # Show enhanced thinking indicator
        thinking_msg = cl.Message(
            content="🔍 **Processing Your Query**\n\n• Searching through documents...\n• Applying contextual compression...\n• Generating intelligent response...",
            author="Assistant"
        )
        await thinking_msg.send()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = rag_chain.query(message.content)
            
            # Extract information from the enhanced result
            if isinstance(result, dict):
                answer = result.get('answer', 'No answer found')
                sources = result.get('source_documents', [])
                metadata = result.get('metadata', {})
            else:
                answer = str(result)
                sources = []
                metadata = {}
            
            # Update thinking message to show completion
            thinking_msg.content = "**Query Processed Successfully**"
            await thinking_msg.update()
            
            # Send the main response with enhanced formatting
            main_response = f"💡 **Answer:**\n\n{answer}"
            
            # Add metadata insights if available
            if metadata:
                insights = "\n\n📈 **Query Insights:**\n"
                if 'tokens_used' in metadata:
                    insights += f"• Tokens used: {metadata['tokens_used']}\n"
                if 'num_sources' in metadata:
                    insights += f"• Sources consulted: {metadata['num_sources']}\n"
                if 'cost' in metadata and metadata['cost']:
                    insights += f"• Processing cost: ${metadata['cost']:.4f}\n"
                main_response += insights
            
            await cl.Message(content=main_response, author="Assistant").send()
            
            # Enhanced source information
            if sources:
                source_info = "📚 **Sources & References:**\n\n"
                for i, doc in enumerate(sources[:4], 1):  # Show top 4 sources
                    page_info = ""
                    if 'page' in doc.metadata:
                        page_info = f" (Page {doc.metadata['page']})"
                    
                    source_name = doc.metadata.get('source', 'Unknown source')
                    file_name = os.path.basename(source_name)
                    
                    # Add a preview of the content
                    content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    source_info += f"**{i}. {file_name}**{page_info}\n*\"{content_preview}\"*\n\n"
                
                await cl.Message(content=source_info, author="Assistant").send()
            
            # Suggest follow-up questions
            follow_up = "🤔 **Suggested Follow-ups:**\n• *\"Can you elaborate on [specific aspect]?\"*\n• *\"How does this relate to other parts of the document?\"*\n• *\"What are the implications of this information?\"*"
            await cl.Message(content=follow_up, author="Assistant").send()
                    
        except Exception as e:
            thinking_msg.content = "❌ **Processing Error**"
            await thinking_msg.update()
            
            error_msg = f"🚨 **Query Processing Error**\n\nI encountered an issue while processing your request:\n\n```\n{str(e)}\n```\n\n💡 **Suggestions:**\n• Try rephrasing your question\n• Ask about specific document sections\n• Check if the documents are properly indexed"
            await cl.Message(content=error_msg, author="Assistant").send()
                
    except Exception as e:
        await cl.Message(
            content=f"🚨 **System Error**\n\nA critical error occurred: {str(e)}\n\nPlease refresh and try again.",
            author="System"
        ).send()
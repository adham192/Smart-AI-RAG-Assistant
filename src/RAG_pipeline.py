"""
RAG Pipeline using LangChain chains
"""
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.config import config
from src.vectorstore import FAISSVectorStore


class RAGPipeline:
    """Main RAG pipeline using LangChain chains"""

    def __init__(self, vectorstore: FAISSVectorStore):
        self.vectorstore = vectorstore

        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            streaming=True  
        )

        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> PromptTemplate:
        template = """
You are a professional AI Assistant.

IMPORTANT:
You are NOT limited to contract-related questions.
You are allowed to answer ANY general knowledge question,
as long as it follows safety guidelines.

Behavior Rules:
1. If the User Question is related to the provided Document Context, use it.
2. If the question is unrelated to the Document Context, ignore the document and answer normally.
3. Do NOT restrict yourself only to the document topic.
4. Maintain a professional tone.

Document Context:
{context}

User Question:
{question}

Answer:
"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _retrieve_context(self, query: str) -> List[Dict]:
        results = self.vectorstore.similarity_search_with_scores(query)
        return [doc for doc, _ in results]


    def _build_context_text(self, documents: List[Dict]) -> str:
        context_parts = []
        for doc in documents:
            source = doc["metadata"].get("source", "Unknown")
            text = doc["text"]
            context_parts.append(f"[Source: {source}]\n{text}")
        return "\n\n".join(context_parts)

    # Streaming Version
    def stream_query(self, question: str):
        vector_info = self.vectorstore.get_info()
        has_documents = vector_info["total_vectors"] > 0

        # If no documents exist in the entire system, go straight to LLM
        if not has_documents:
            for chunk in self.llm.stream(question):
                yield chunk.content
            return

        # If documents exist, try to find relevant context
        context_docs = self._retrieve_context(question)
        
        # We still use the prompt template, but we let the LLM know 
        # if the specific search returned nothing useful for this query.
        if context_docs:
            context_text = self._build_context_text(context_docs)
        else:
            context_text = "No relevant document context found for this specific query."

        chain = self.prompt_template | self.llm

        for chunk in chain.stream({
            "context": context_text,
            "question": question
        }):
            yield chunk.content


    #  Non-Streaming Version
    def query(self, question: str) -> Dict:
        try:
            full_response = ""

            for chunk in self.stream_query(question):
                full_response += chunk

            return {
                "result": full_response,
                "source_documents": []
            }

        except Exception as e:
            return {
                "result": f"Error occurred: {str(e)}",
                "source_documents": []
            }

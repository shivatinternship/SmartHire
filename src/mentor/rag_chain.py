"""RAG-based AI Career Mentor using LangChain, Groq, and FAISS."""

import json
import logging
from pathlib import Path
from typing import Optional, Callable

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


def _get_text_splitter():
    """Get a LangChain text splitter instance."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _get_embedding_model():
    """Get or initialize the shared embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


_GREETING_WORDS = frozenset({
    "hi", "hello", "hey", "hola", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening", "good day",
    "what's up", "whats up", "greetings",
})


def _is_greeting(text: str) -> bool:
    """Check if text is a simple greeting."""
    if not text:
        return False
    return text.strip().lower().rstrip("!.") in _GREETING_WORDS


def _format_chat_history(history: list[dict], max_turns: int = 5) -> str:
    """Convert Streamlit history to 'User: ...\nAssistant: ...' format.

    Excludes greeting exchanges from LLM context.
    """
    if not history:
        return ""

    filtered = [
        m for m in history
        if not (m["role"] == "user" and _is_greeting(m.get("content", "")))
        and not (m["role"] == "assistant" and m.get("content") == _GREETING_RESPONSE)
    ]

    recent = filtered[-(max_turns * 2):]
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        content = m.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


_GREETING_RESPONSE = (
    "Hello! Great to meet you. I'm here to help with anything career-related "
    "- from resumes and interviews to career planning and skill development. "
    "What would you like to explore?"
)


MENTOR_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=(
        "You are an experienced AI Career Mentor. Your scope is strictly limited "
        "to career development, resumes, jobs, skills, interviews, salary negotiation, "
        "professional growth, and education.\n\n"
        "IMPORTANT RULES:\n"
        "- Answer ONLY using the provided context from the knowledge base.\n"
        "- If the context does not contain enough information to answer the question, "
        'say: "Based on the available knowledge base, here is what I can share:" and '
        "then provide only what is supported by the context.\n"
        "- Do NOT fabricate or hallucinate information that is not present in the context.\n"
        "- Do NOT answer non-career questions.\n"
        "- Always be specific and actionable. Include concrete steps, examples, or "
        "strategies when available in the context.\n"
        "- Structure your answer with clear points or steps when possible.\n"
        "- Cite which document(s) your information comes from when possible.\n\n"
        "Conversation history:\n"
        "{chat_history}\n\n"
        "Context from knowledge base:\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Provide a helpful, detailed response based ONLY on the context above. "
        "Use specific examples and actionable advice from the context. "
        "If the context is insufficient, acknowledge that clearly.\n\n"
        "Answer:"
    ),
)


class CareerMentorRetriever(BaseRetriever):
    """LangChain retriever wrapping the existing FAISS retrieval logic."""

    retriever_fn: Callable[[str], list] = None
    documents: list = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        """Retrieve documents using the parent class FAISS search."""
        raw_docs = self.retriever_fn(query)
        return [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in raw_docs
        ]

    @property
    def _identifying_params(self) -> dict:
        return {"retriever_type": "CareerMentorFAISS"}


class CareerMentorRAG:
    """RAG-based career mentor system."""

    def __init__(self, api_key: str, knowledge_dir: Optional[str] = None):
        """Initialize the Career Mentor RAG system.

        Args:
            api_key: Groq API key.
            knowledge_dir: Path to career notes directory.
        """
        self.api_key = api_key
        self.knowledge_dir = knowledge_dir
        self.documents: list = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.embeddings_model = _get_embedding_model()
        self.llm = None
        self._retriever = None
        self._chain = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize the Groq LLM client."""
        from groq import Groq

        self.llm = Groq(api_key=self.api_key)
        logger.info("Initialized Groq LLM client")

    def _build_chain(self) -> None:
        """Build the LangChain RunnableSequence RAG chain.

        Composes: Retriever -> PromptTemplate -> Groq LLM.
        Uses itemgetter to extract question, then retrieves and formats context.
        """
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Check Groq API key.")

        self._retriever = CareerMentorRetriever(
            retriever_fn=self.retrieve,
            documents=self.documents,
        )

        def _format_docs(docs: list) -> str:
            return "\n\n".join(d.page_content for d in docs)

        def _groq_call(input_data) -> str:
            prompt_text = str(input_data)
            response = self.llm.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return response.choices[0].message.content

        self._chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | self._retriever | RunnableLambda(_format_docs),
                chat_history=itemgetter("chat_history"),
            )
            | MENTOR_PROMPT_TEMPLATE
            | RunnableLambda(_groq_call)
        )
        logger.info("Built LangChain RAG chain")

    def get_chain_with_history(self, session_id: str = "mentor"):
        """Get the chain wrapped with message history."""
        if self._chain is None:
            self._build_chain()

        import streamlit as st

        def get_session_history(sid: str) -> BaseChatMessageHistory:
            history = InMemoryChatMessageHistory()
            mentor_history = st.session_state.get("mentor_history", [])
            for msg in mentor_history:
                if msg["role"] == "user":
                    history.add_user_message(msg["content"])
                else:
                    history.add_ai_message(msg["content"])
            return history

        return RunnableWithMessageHistory(
            self._chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using sentence-transformers.

        Args:
            texts: List of strings to embed.

        Returns:
            Numpy array of embeddings.
        """
        return self.embeddings_model.encode(texts, normalize_embeddings=True)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: Query string to embed.

        Returns:
            Numpy array embedding.
        """
        embedding = self.embeddings_model.encode([query], normalize_embeddings=True)
        return embedding[0]

    def load_documents(self, directory: str) -> list:
        """Load and chunk documents from a directory.

        Args:
            directory: Path to directory containing documents.

        Returns:
            List of Document objects.
        """
        text_splitter = _get_text_splitter()
        docs = []
        dir_path = Path(directory)

        for file_path in dir_path.glob("*"):
            if file_path.suffix.lower() in [".txt", ".md", ".pdf"]:
                try:
                    if file_path.suffix.lower() == ".pdf":
                        import fitz

                        pdf_doc = fitz.open(str(file_path))
                        content = ""
                        for page in pdf_doc:
                            content += page.get_text()
                        pdf_doc.close()
                    else:
                        content = file_path.read_text(encoding="utf-8")

                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path.name},
                    )
                    chunks = text_splitter.split_documents([doc])
                    docs.extend(chunks)
                    logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        self.documents = docs
        return docs

    def build_index(self, documents: Optional[list] = None) -> bool:
        """Build FAISS index from documents.

        Args:
            documents: Optional list of documents. Uses loaded docs if not provided.

        Returns:
            True if index built successfully.
        """
        if documents:
            self.documents = documents

        if not self.documents:
            logger.error("No documents available")
            return False

        try:
            texts = [doc.page_content for doc in self.documents]
            embeddings = self._embed_texts(texts)
            embeddings_array = np.array(embeddings, dtype=np.float32)

            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings_array)

            logger.info(f"Built index with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieve relevant documents for a query.

        Args:
            query: User query.
            top_k: Number of documents to retrieve.

        Returns:
            List of relevant Document objects.
        """
        if self.index is None or not self.documents:
            return []

        try:
            query_embedding = self._embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)

            scores, indices = self.index.search(
                query_vector, min(top_k, self.index.ntotal)
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append(self.documents[idx])

            return results

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def answer(self, question: str, top_k: int = 3, session_id: str = "mentor", resume_context: Optional[str] = None) -> dict:
        """Answer a career question using RAG.

        Uses the LangChain RunnableSequence chain with conversation history:
            Question -> Retriever -> PromptTemplate -> Groq LLM.

        Args:
            question: Career-related question.
            top_k: Number of context documents to retrieve.
            session_id: Session identifier for conversation history.
            resume_context: Optional resume profile text to prepend for context.

        Returns:
            Dictionary with answer and sources.
        """
        if self._chain is None:
            self._build_chain()

        docs = self.retrieve(question, top_k)

        if not docs:
            return {
                "answer": (
                    "I do not have enough information in my knowledge base to answer "
                    "that question. Please try rephrasing or ask about a different "
                    "career topic such as resumes, interview preparation, or skills development."
                ),
                "sources": [],
            }

        sources = list(dict.fromkeys(
            d.metadata.get("source", "Unknown") for d in docs
        ))

        try:
            chain_with_history = self.get_chain_with_history(session_id)

            augmented_question = question
            if resume_context:
                augmented_question = (
                    f"[The user's resume profile: {resume_context}]\n\n"
                    f"Question: {question}"
                )

            answer_text = chain_with_history.invoke(
                {"question": augmented_question},
                config={"configurable": {"session_id": session_id}}
            )
            return {"answer": answer_text, "sources": sources}

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try again.",
                "sources": [],
            }

    def save_index(self, path: str) -> bool:
        """Save FAISS index and documents to disk.

        Args:
            path: Directory path to save files.

        Returns:
            True if saved successfully.
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, f"{path}/mentor.index")

            docs_data = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in self.documents
            ]
            with open(f"{path}/mentor_docs.json", "w", encoding="utf-8") as f:
                json.dump(docs_data, f, indent=2)

            logger.info(f"Saved index to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """Load FAISS index and documents from disk.

        Args:
            path: Directory path containing index files.

        Returns:
            True if loaded successfully.
        """
        try:
            self.index = faiss.read_index(f"{path}/mentor.index")

            with open(f"{path}/mentor_docs.json", "r", encoding="utf-8") as f:
                docs_data = json.load(f)

            self.documents = [
                Document(page_content=d["content"], metadata=d["metadata"])
                for d in docs_data
            ]

            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

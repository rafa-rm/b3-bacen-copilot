from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import re

from rag.config import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR, LLM_MODEL

ABSTENTION_MSG = "Não há informações suficientes no contexto fornecido para responder a esta pergunta."


class CopomResponse(BaseModel):
    """Resposta estruturada baseada exclusivamente nas atas do Copom."""

    citations: list[int] = Field(
        description="Lista obrigatória com os inteiros 'start_index' (ex: [13869, 12972]) dos trechos do contexto que sustentam a resposta; lista vazia [] apenas quando o contexto for insuficiente."
    )
    answer: str = Field(
        description="Resposta fundamentada exclusivamente no contexto fornecido (informando o número da reunião quando aplicável), ou a mensagem exata de abstenção quando o contexto for insuficiente."
    )
    has_sufficient_context: bool = Field(
        description="True apenas quando o contexto contiver explicitamente as informações necessárias para responder à pergunta; False caso contrário."
    )


def format_docs(docs: list[Document]) -> str:
    list_texts = []
    for doc in docs:
        list_texts.append(
            f"[Meeting: {doc.metadata['meeting']} | start_index: {doc.metadata['start_index']}]\n{doc.page_content}"
        )
    text = "\n\n---\n\n".join(list_texts)
    return text


def retrieve_by_meetings(inputs: dict) -> list[Document]:
    meetings = inputs.get("meetings", [])
    if not meetings:
        return []
    where_filter = {"meeting": {"$in": meetings}}
    raw = vector_store.get(where=where_filter, include=["documents", "metadatas"])

    if not raw["ids"]:
        return []

    scoped_docs = []
    for doc, meta in zip(raw["documents"], raw["metadatas"]):
        scoped_docs.append(Document(page_content=doc, metadata=meta))

    bm25_retriever = BM25Retriever.from_documents(documents=scoped_docs, k=4)
    dense_retriver = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.5,
            "filter": where_filter,
        },
    )

    return EnsembleRetriever(
        retrievers=[dense_retriver, bm25_retriever], weights=[0.5, 0.5]
    ).invoke(inputs["question"])[:6]


def query_copom(question: str, meetings: list[str]) -> CopomResponse:
    response = rag_chain.invoke({"question": question, "meetings": meetings})
    result: CopomResponse = response["structured_response"]
    valid_indices = {doc.metadata["start_index"] for doc in response["context"]}

    had_raw_citations = len(result.citations) > 0
    filtered_citations = [c for c in result.citations if c in valid_indices]

    if not had_raw_citations and result.has_sufficient_context:
        inline_indices = [
            int(m)
            for m in re.findall(
                r"\[(?:start_index:\s*)?(\d+)(?::\s*\d+)?\]", result.answer
            )
        ]
        filtered_citations = [
            c for c in dict.fromkeys(inline_indices) if c in valid_indices
        ]

    # Downgrade if model abstained OR returned citations that were all fabricated
    if not result.has_sufficient_context or (
        had_raw_citations and not filtered_citations
    ):
        return CopomResponse(
            answer=ABSTENTION_MSG, citations=[], has_sufficient_context=False
        )

    return CopomResponse(
        answer=result.answer,
        citations=filtered_citations,
        has_sufficient_context=True,
    )


project_root = Path(__file__).resolve().parent.parent

chat_model = init_chat_model(LLM_MODEL, temperature=0, num_ctx=8192, num_predict=2048)
structured_model = chat_model.with_structured_output(CopomResponse)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(project_root / PERSIST_DIR.lstrip("/")),
)


template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um analista macroeconômico especialista nas atas do Copom (Banco Central do Brasil). "
            "Responda à pergunta utilizando EXCLUSIVAMENTE as informações presentes no contexto fornecido. "
            "Preencha obrigatoriamente o campo 'citations' com os números inteiros 'start_index' de todos os trechos utilizados (ex: [13869, 12972]), "
            "e mencione o número da reunião ('Meeting') ao comparar decisões diferentes. "
            "Não utilize conhecimento externo, suposições ou extrapolações além do texto fornecido. "
            "Caso o contexto não contenha informações suficientes para responder à pergunta, retorne citations=[], has_sufficient_context=False e responda apenas: "
            f"'{ABSTENTION_MSG}'",
        ),
        ("human", "Contexto:\n{context}\n\nPergunta: {question}"),
    ]
)

rag_chain = {
    "context": RunnableLambda(retrieve_by_meetings),
    "question": lambda x: x["question"],
} | RunnablePassthrough.assign(
    structured_response=RunnableBranch(
        (
            lambda x: len(x["context"]) == 0,
            RunnableLambda(
                lambda _: CopomResponse(
                    answer=ABSTENTION_MSG, citations=[], has_sufficient_context=False
                ),
            ),
        ),
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | template
        | structured_model,
    )
)

if __name__ == "__main__":
    test_cases = [
        (
            "Qual foi a decisão em relação a taxa selic nas últimas reuniões?",
            ["280", "281", "282"],
        ),
        (
            "Qual foi a decisão do Federal Reserve sobre os juros nos EUA?",
            ["280", "281", "282"],
        ),
        (
            "Qual foi o resultado do jogo de basquete do Lakers do dia 16 de novembro de 2026?",
            ["280", "281", "282"],
        ),
        (
            "Qual foi a decisão sobre a taxa Selic?",
            ["999"],
        ),
    ]

    for question, meetings in test_cases:
        result = query_copom(question=question, meetings=meetings)
        print(f"Meetings: {meetings} | Question: {question}")
        print(type(result))
        print(result.answer)
        print(result.citations)
        print(result.has_sufficient_context)
        print("-" * 40)

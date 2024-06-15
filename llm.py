from langchain.document_loaders import PyPDFLoader

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough


def load_pdf_and_split(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages


def embed_and_store_splits(splits):
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore


def queryPDF(vectorstore, query):
    retriever = vectorstore.as_retriever()
    rag_prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm
    )
    print("Querying... ", query)
    response = rag_chain.invoke(query)
    print("Response: ", response)
    return response.content
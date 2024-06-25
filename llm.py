from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain.llms import GooglePalm
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_pdf_and_split(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages


def embed_and_store_splits(splits):
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
    return vectorstore

def queryPDF(vectorstore, query):
    retriever = vectorstore.as_retriever()
    rag_prompt = hub.pull("rlm/rag-prompt")
    llm = GooglePalm(google_api_key=palm_api_key, temperature=0.2)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    print("Querying... ", query)
    response = rag_chain.invoke(query)
    print("Response: ", response)
    return response
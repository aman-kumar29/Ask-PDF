import streamlit as st
import tiktoken
import dotenv
import llm
import os


def run_app():
    # load API keys as global variables
    dotenv.load_dotenv()
    PALM_API_KEY = os.getenv('PALM_API_KEY')
    print(PALM_API_KEY);
    st.title("My AI app")

    st.write("Ask a question to your PDF document")

    # upload file using streamlit
    uploaded_file = st.file_uploader("Choose a file")

    # save file to the disk and split into list of pages
    if uploaded_file:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pages = llm.load_pdf_and_split(uploaded_file.name)
        st.write(
            f"File uploaded: {uploaded_file.name}  \n Total splits: {len(pages)}  \n Total tokens: {num_tokens_from_list(pages)}"
        )

        # embed splits in vectorstore
        vectorstore = llm.embed_and_store_splits(pages)

    # field to type question
    placeholder = "Type here..."
    question = st.text_input("Enter your question", placeholder)

    # query LLM only when question is submitted
    if question and question != placeholder:
        result = llm.queryPDF(vectorstore, question, PALM_API_KEY)
        st.write(result)


# retrurn number of tokens from string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# returns number of tokens from list
def num_tokens_from_list(pages):
    num_tokens = 0
    for i, page in enumerate(pages):
        num_tokens += num_tokens_from_string(page.page_content, "cl100k_base")
    return num_tokens


if __name__ == "__main__":
    run_app()
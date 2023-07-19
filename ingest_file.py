from typing import List

from langchain.document_loaders import TextLoader
from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

embeddings = GPT4AllEmbeddings()


def load_documents(file_path) -> List:
    loader = TextLoader(file_path)  # TODO support other file extensions
    return loader.load()


def split_chunks(sources_list: List) -> List:
    chunks_array = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources_list):
        chunks_array.append(chunk)
    return chunks_array


def generate_index(chunks_list: List) -> FAISS:
    texts = [doc.page_content for doc in chunks_list]
    meta_datas = [doc.metadata for doc in chunks_list]
    return FAISS.from_texts(texts, embeddings, metadatas=meta_datas)


def save_index(file_path, index_path):
    sources = load_documents(file_path)
    chunks = split_chunks(sources)
    vectorstore = generate_index(chunks)
    vectorstore.save_local(index_path)


@DeprecationWarning
def ingest_file(file_path, content_name):
    save_index(file_path, "indexes/" + content_name)


if __name__ == '__main__':
    ingest_file("./texts/state_of_the_union.txt", "sotu")

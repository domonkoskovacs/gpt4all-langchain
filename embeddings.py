from typing import List

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

local_path = (
    "./models/nous-hermes-13b.ggmlv3.q4_0.bin"
)
source_path = "./texts/state_of_the_union.txt"
index_path = "./full_sotu_index"

embeddings = GPT4AllEmbeddings()
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True)


def load_documents() -> List:
    loader = TextLoader("./texts/state_of_the_union.txt")
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


def save_index():
    sources = load_documents()
    chunks = split_chunks(sources)
    vectorstore = generate_index(chunks)
    vectorstore.save_local("full_sotu_index")


def main(question):
    save_index()
    index = FAISS.load_local(index_path, embeddings)
    general_system_template = """ 
     ----
    {context}
    ----
    """
    general_user_template = """Question: {question}
    Answer: Let's think step by step."""
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        index.as_retriever(),
        max_tokens_limit=400,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    chat_history = []
    result = qa({"question": question, "chat_history": chat_history})
    print(result['answer'])


if __name__ == '__main__':
    # calling this gives an answer, this file is a basic setup for ConversationalRetrievalChain
    main("What did the president say about Ketanji Brown Jackson")

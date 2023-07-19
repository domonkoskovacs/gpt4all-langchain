import os

from langchain import PromptTemplate, LLMChain, FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

local_path = (
    "./models/nous-hermes-13b.ggmlv3.q4_0.bin"  # change the path for your model
)
embeddings = GPT4AllEmbeddings()  # this downloads an embedding model if not cached
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=local_path, backend="gptj", callbacks=callbacks, verbose=True)
chat_history = []


def load_indexes(role) -> FAISS:
    """
    This function loads the index files for a specific role from the filesystem.
    :param role: index group
    :return: index: vectorstore with concatenated index file
    """
    valid_roles = ["EMPLOYEE", "USER"]
    if role not in valid_roles:
        raise TypeError("Role is nonexistent")
    directory_path = './indexes/' + role
    sub_dirs = os.listdir(directory_path)
    index: FAISS = FAISS.load_local(os.path.join(directory_path, sub_dirs[0]), embeddings)
    sub_dirs.pop(0)
    # loading and merging index files in each subdir
    for subdir in sub_dirs:
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            temp: FAISS = FAISS.load_local(subdir_path, embeddings)
            index.merge_from(temp)
    return index


def generate_prompt(prompt_type):
    """
    This function generates the prompt from the given prompt type,
    Currently 3 supported (short, long, step by step)
    :param prompt_type: type of the prompt
    :return: prompt_template: prompt that the chain can use
    """
    template_mapping = {
        "STEP_BY_STEP": "Let's think step by step.",
        "SHORT": "Let's give a short answer.",
        "LONG": "Let's give a long answer."
    }
    answer_template = template_mapping.get(prompt_type)
    if answer_template is None:
        raise TypeError("Prompt type is nonexistent")

    general_system_template = """ 
             ----
            {context}
            ----
            """
    general_user_template = """Question: {question}
            Answer: """ + answer_template
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    return ChatPromptTemplate.from_messages(messages)


def answer_question(question, prompt_type, role):
    """
    This function generates an answer with the given parameters and existing context,
    currently only one session supported.
    :param question: question
    :param prompt_type: type of the prompt
    :param role: index group
    :return: answer: generated answer
    """
    index = load_indexes(role)
    prompt = generate_prompt(prompt_type)
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        index.as_retriever(),
        max_tokens_limit=1000,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    return result['answer']


@DeprecationWarning
def get_answer(question, template_type):
    """
    Basic LLMCHain, this function gives an answer for the given question
    :param question:
    :param template_type:
    :return:
    """
    template = ''
    if template_type == 'STEP_BY_STEP':
        template = """Question: {question}
        Answer: Let's think step by step."""
    if template_type == 'SHORT_AND_SIMPLE':
        template = """Question: {question}
        Answer: Let's keep it short and simple."""
    if template_type == 'ACCURATE':
        template = """Question: {question}
        Answer: Let's be accurate."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(question)


if __name__ == '__main__':
    answer_question("What is langchain?", "SHORT", "USER")
    # get_answer("What is langchain?","SHORT_AND_SIMPLE") # uncomment to try basic llm chain

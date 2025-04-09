from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from src.spam_classifier.spam_classifier import detect_spam



import os
import openai   

CHROMA_PATH = "./chroma" # define path with chroma embedding store

def create_db(CHROMA_PATH):
    """
    creating/loading database from vector store
    Args:
        chroma_path - local path to chroma db
    return:
        vectorStore 
    """
    embedding_function = OpenAIEmbeddings()

    # Define local data and chroma path. in real project it will not be same    
    # Loading Chroma database from local, using openai embedding func  
    vectorStore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function) 
    return vectorStore

def create_chain(vectorStore):
    """
    Function with Ragchain and AI GPT. That's loading local chroma db, open ai embedding, and
    answering questions about book "Alice in wonderwood" with chat-history.

    Args:
        vectoreStore --- our database. Now it is one book vectorized and stored by openAi embedding
        and Chroma db 

    Return:
        retrieval_chain --- pipline chain prepeared to work with llm model
    """

    # Load Openai key, define chatbot

    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")    


    ### Answer question ###

    # Define text in variable to system prompt

    qa_system_prompt = """"Answer the user's questions based on the context:
    {context}"""

    # System prompt

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # chain = prompt | model

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )   

    # Retrieve and generate using the relevant snippets of the text.

    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})

    ### Contextualize question ###

    contextualize_q_system_prompt = """Given the above conversation, \
    generate a search query to look up in order to get information relevant \
    to the conversation.\
    """

    # Contextualize prompt that retrieve infromation from messages history

    retriever_prompt = ChatPromptTemplate.from_messages(
        [            
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", contextualize_q_system_prompt)
        ]
    )
    
    # Getting history

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, 
        retriever=retriever, 
        prompt=retriever_prompt
    )

    # Getting history chain

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever, 
        chain)

    return retrieval_chain



def process_chat(chain, question, chat_history):
    """
    This is function initiate chat proccess
    Args:
        chain - create_chain function
        question - input message
        chat_history --- variable that contain messages history storied in python list
    Return:
        response - answer from llm model
        if message spam - return spam
    """
    is_spam = detect_spam(question)
    if is_spam:
        return "spam"
    else:        
        response = chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        return response["answer"]

def conversation(response):
    """
    starting conversation with llm, where chat history will store in variable
    to exit from promptring conversation --- enter 'exit'
    getting from comment as message and after start conversation.
    any comment will check on spam.
    Args:
        respons - that returns from process_chat function
    Return:
        if spam - return "spam"
    
    """
    CHROMA_PATH = "./chroma" # define path with chroma embedding store
    vectorStore = create_db(CHROMA_PATH)
    chain = create_chain(vectorStore)

    chat_history = []
    
    # Catch first message in coment to answer on it before conversation

    user_input = response
    response = process_chat(chain, user_input, chat_history)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
        
    print("Assistant:", response)   
    
    while True :        
        if response == "spam":
            break
           
        user_input = input("You: ")                
        if user_input.lower() == 'exit': 
            break
        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        
        print("Assistant:", response)   
   
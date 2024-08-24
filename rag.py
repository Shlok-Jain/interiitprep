import streamlit as st
import fitz  # PyMuPDF
import streamlit.components.v1 as components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch





#function to determine whether to use the document for giving answer, or it is a general question.
#this is because it was wierdly replying to general questions like greeting etc.
def should_use_document(query, st):
    prompt = f'''
    I am using an LLM to create a RAG application to answer questions based on pdfs.
    I will give you a query and you have to determine whether to use the document for answering the query or not.
    If you find that for answering that query the information from pdf will be required say only "YES" else if you think it is a general question answerable without document then say only "NO", If you are not sure say only "YES".
    I am also providing previous conversation for context.
    Rememner that this LLM is explicitly made to answer questions based on pdfs, so there will be only some cases where you would say no like greeting, aur saying bye etc. So ans "NO" wisely.
    Previous conversation: {st.session_state.prev_conversation}
    Question: {query}
    '''
    response = st.session_state.llm.invoke(prompt)
    return 0 if response != "YES" else 1

def generate(query, st):
    # query = input("You: ")
    if should_use_document(query, st):
        context = st.session_state.vectorstore.similarity_search(query, k=5)
        prompt = f'''
        You are given a pdf document and you have to answer the questions based on the document as well as the previous conversation.
        I have made a helper function to determine of you will need to use the document to answer this question or determine that it is a general question that can be answered without the document. And that function has determined that you will need to use the document.
        So use the document and previous conversation to answer the question. Say "I dont know" if you dont know.
        Previous conversation: {st.session_state.prev_conversation}
        Document: {context}
        Question: {query}
        '''
    else:
        prompt = f'''
        You are given a pdf document and you have to answer the questions based on the document as well as the previous conversation.
        I have made a helper function to determine of you will need to use the document to answer this question or determine that it is a general question that can be answered without the document. And that function has determined that you will not need to use the document.
        So use just the previous conversation to answer the question. Say "I dont know" if you dont know.
        Previous conversation: {st.session_state.prev_conversation}
        Question: {query}
        '''
    response = st.session_state.llm.invoke(prompt)
    # print("Llama: " + response + '\n\n')

    st.session_state.prev_conversation += f"Question: {query}\nAnswer: {response}\n"
    return response

def send_message(st):
    user_message = st.session_state.user_input
    if user_message:
        st.session_state.messages.append({'role': 'user', 'text': user_message})
        
        response = generate(user_message, st)

        st.session_state.messages.append({'role': 'bot', 'text': response})
        st.session_state.user_input = ''  # Clear the input field

def main():
    
    # embeddings = OllamaEmbeddings(model='llama3')
    # llm = Ollama(model="llama3")
    # print("LLM initialized")

    prev_conversation = "This is the start of conversation."
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model='llama3')
    if "llm" not in st.session_state:
        st.session_state.llm = Ollama(model="llama3")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "prev_conversation" not in st.session_state:
      st.session_state.prev_conversation = prev_conversation
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    st.set_page_config(
        page_title="RAG application for answering questions based on pdfs",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        text = ''
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf_file:
            for page_num in range(pdf_file.page_count):
                text += pdf_file[page_num].get_text()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        texts = text_splitter.create_documents([text])
        st.session_state.vectorstore = DocArrayInMemorySearch.from_documents(texts, st.session_state.embeddings)
    
    st.markdown("""
        <div id="chat-container" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;">
            <div id="chat-messages">
                {}
            </div>
        </div>
    """.format(
        ''.join([f"<p><b>{msg['role'].capitalize()}:</b> {msg['text']}</p>" for msg in st.session_state.messages])
    ), unsafe_allow_html=True)

    st.text_area("Your message:", key='user_input', on_change=lambda: send_message(st))
    st.button("Send", on_click=lambda: send_message(st))

    components.html("""
        <script>
            const inputBox = window.parent.document.querySelector('textarea');
            const sendButton = window.parent.document.querySelector('button');
            inputBox.addEventListener('keydown', function (event) {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    sendButton.click();
                }
            });
            const chatContainer = window.parent.document.getElementById('chat-container');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        </script>
    """)

    st.markdown("""
        <style>
            .stTextArea, .stButton {
                position: fixed;
                bottom: 0;
                width: 100%;
                margin: 0;
                padding: 10px;
                background-color: #fff;
            }
            .stTextArea {
                border-top: 1px solid #ccc;
            }
            .stButton {
                border-top: 0;
            }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


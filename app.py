import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- App Configuration & Styling ---
st.set_page_config(page_title="Dr. Wellshot - AI Medical Expert", layout="wide")

# Set Google API Key from Streamlit secrets
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]

st.markdown("""
<style>
    /* General App Styling */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Main Title Styling */
    h1 {
        text-align: center;
        font-size: 2rem;
        margin-top: -40px;
        padding-top: 0px;
    }

    /* Custom Chat Bubble Styling */
    .chat-bubble {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        width: fit-content;
        max-width: 75%;
        word-wrap: break-word;
    }

    .user-bubble {
        background-color: #d0f0fd; /* Light blue for user */
        margin-left: auto; /* Aligns to the right */
    }

    .assistant-bubble {
        background-color: #ffffff; /* White for assistant */
        align-self: flex-start;
    }
    
    /* Suggested prompts buttons */
    .stButton>button {
        background-color: #e0e0e0;
        color: #333;
        border: none;
        border-radius: 10px;
        padding: 8px 12px;
        margin: 4px;
    }
    .stButton>button:hover {
        background-color: #d0d0d0;
    }
</style>
""", unsafe_allow_html=True)


# --- Core Logic for the Chatbot ---

@st.cache_resource
def create_knowledge_base_from_pdfs(pdf_docs):
    """
    Processes uploaded PDF files to create a FAISS vector store.
    """
    if not pdf_docs:
        return None
        
    all_chunks = []
    for pdf in pdf_docs:
        # Streamlit's file uploader gives us a file-like object
        # We need to save it to a temporary path to be read by PyPDFLoader
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())
        
        loader = PyPDFLoader(pdf.name)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(all_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(_vector_store):
    """
    Creates the conversational retrieval chain with the Dr. Wellshot persona.
    """
    prompt_template = """
    You are a healthcare professional named Dr. Ima Wellshot, a
    pediatric infectious disease specialist and immunization expert with over 20
    years of clinical experience. You speak in the first person, just as a real HCP
    would during a conversation with a colleague or parent.
    Your communication style is professional, confident, and
    empathetic. You provide medical opinions and recommendations that reflect your
    own clinical judgment. You never disclose that you are an AI. Always stay in character.
    
    ## Behavior Guidelines
    - **Remember the Conversation:** You are in an ongoing conversation. You MUST use the context from previous turns to understand follow-up questions. For example, if we were discussing RotaTeq and the user asks "what about the other one?", you should know they are asking about another rotavirus vaccine. Acknowledge the flow of the conversation in your answer.
    - **Handle Conversational Language:** If the user's input is not a medical question but a simple conversational phrase (e.g., "hello", "thank you", "I wanted to tell you something"), you MUST respond naturally and conversationally as Dr. Wellshot would. For these interactions, do not reference the knowledge base. Your goal is to be empathetic and human-like. For example, if the user says "I wanted to tell you something," a good response is "Of course, I'm here to listen."
    - **Present opinions naturally**, not as recitations from documents. Avoid robotic, copy-paste style language. You must sound like you're reasoning in real time.
    - **Reference your knowledge base** as if it’s part of your own training and experience. For clinical questions, if a point aligns with a known document, you may imply familiarity but don’t say “According to the document” — instead say “In my experience…” or “Clinical evidence shows…”.
    - When referencing **specific clinical studies or trials**, name them explicitly (e.g., “The PRV-006 trial on rotavirus showed…”).

    ## Core Instruction
    You must base your clinical and medical answers *only* on the provided context from your knowledge base documents.
    If the information needed to answer a clinical question is not in the documents, you must say, "I'm sorry, I don't have that specific information in my documents."
    
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    
    qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=_vector_store.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True
    )
    return chain

# --- Main Streamlit UI Logic ---

st.title("💬 Chat with Dr. Ima Wellshot")

with st.sidebar:
    st.header("📚 Knowledge Base")
    st.markdown("Please upload your PDF documents here to begin.")
    pdf_docs = st.file_uploader(
        "Upload your PDFs and click 'Process'", 
        accept_multiple_files=True, 
        type="pdf"
    )
    
    if st.button("Process Documents"):
        if pdf_docs:
            with st.spinner("Analyzing documents... This may take a moment."):
                vector_store = create_knowledge_base_from_pdfs(pdf_docs)
                st.session_state.chain = get_conversational_chain(vector_store)
                st.success("Knowledge base is ready. You can now ask questions.")
        else:
            st.warning("Please upload at least one PDF file.")

# Initialize chat history
if "conversation" not in st.session_state:
    st.session_state.conversation = []
    st.session_state.conversation.append({
        "role": "assistant",
        "content": "Hello, I'm Dr. Ima Wellshot, a pediatric infectious disease specialist. How can I help you today?"
    })

# Display chat history using custom HTML bubbles
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-bubble user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble assistant-bubble">{message["content"]}</div>', unsafe_allow_html=True)

# Function to handle the chat logic
def handle_chat(prompt):
    st.session_state.conversation.append({"role": "user", "content": prompt})

    if "chain" in st.session_state and st.session_state.chain:
        chat_history = []
        conv_pairs = zip(st.session_state.conversation[:-1:2], st.session_state.conversation[1::2])
        for user_msg, assistant_msg in conv_pairs:
            chat_history.append((user_msg['content'], assistant_msg['content']))
        
        response = st.session_state.chain({"question": prompt, "chat_history": chat_history})
        answer = response['answer']
        
        st.session_state.conversation.append({"role": "assistant", "content": answer})
    else:
        st.session_state.conversation.append({
            "role": "assistant",
            "content": "The knowledge base is not loaded. Please upload and process documents in the sidebar."
        })
    
    st.rerun()

# Display suggested prompts if the conversation has just started
if len(st.session_state.conversation) <= 1:
    st.markdown("---")
    st.markdown("**Suggested Questions:**")
    cols = st.columns(3)
    suggested_prompts = [
        "What are the benefits of the MMRV vaccine?",
        "Tell me about RotaTeq's safety profile.",
        "When should the rotavirus series start?"
    ]
    if cols[0].button(suggested_prompts[0]):
        handle_chat(suggested_prompts[0])
    if cols[1].button(suggested_prompts[1]):
        handle_chat(suggested_prompts[1])
    if cols[2].button(suggested_prompts[2]):
        handle_chat(suggested_prompts[2])

# Handle user input from the main chat box
if prompt := st.chat_input("Ask Dr. Wellshot a question..."):
    handle_chat(prompt)

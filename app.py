# --- 1. THE LINUX FIX (Safe for Mac Local Testing) ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass # We are running locally on Mac, so we skip the Linux fix.

# --- 2. STANDARD IMPORTS ---
import streamlit as st
# ... rest of your code ...
import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA 
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType 
from langchain_core.tools import Tool

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="My Super Agent", page_icon="🤖")
st.title("🤖 Agentic RAG: The Final Boss")

# Load Secrets (Works for both Local .env and Cloud Secrets)
load_dotenv()

# --- 4. SETUP THE AI (Self-Healing Version) ---
@st.cache_resource
def setup_agent():
    # Define the Brain
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Check if the database exists. If not, run ingest.py to build it.
    if not os.path.exists("./chroma_db"):
        st.warning("⚠️ Cloud Memory not found. Building it now... (This takes 5 seconds)")
        try:
            # This imports your ingest script and runs it instantly
            import ingest 
            st.success("✅ Memory Built Successfully!")
        except Exception as e:
            st.error(f"Failed to build memory: {e}")
            return None
            
    # Load the Database (Now guaranteed to exist)
    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    
    # Create the RAG Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Create the Tools
    rag_tool = Tool(
        name="Private_Knowledge_Base",
        func=rag_chain.invoke,
        description="Useful for answering questions about Project Apollo, internal budgets, or company secrets."
    )
    
    search = DuckDuckGoSearchRun()
    web_tool = Tool(
        name="Public_Web_Search",
        func=search.run,
        description="Useful for finding current stock prices, weather, news, or public information."
    )
    
    # Create the Agent
    tools = [rag_tool, web_tool]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

# Initialize the Agent
agent = setup_agent()

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI response
    with st.chat_message("assistant"):
        if agent is not None:
            # We use a spinner so the user knows it's thinking
            with st.spinner("Thinking..."):
                response = agent.invoke(prompt)
                output_text = response['output']
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
        else:
            st.error("Agent failed to initialize.")

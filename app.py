__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ... the rest of your imports (streamlit, etc) follow below ...
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA 
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType 
from langchain_core.tools import Tool

# 1. Page Config (Browser Title)
st.set_page_config(page_title="My Super Agent", page_icon="🤖")
st.title("🤖 Agentic RAG: The Final Boss")

# 2. Load Secrets
load_dotenv()

# 3. Setup the AI (Cached for speed)
# This function only runs ONCE.
@st.cache_resource
def setup_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    if not os.path.exists("./chroma_db"):
        st.error("Error: chroma_db not found. Please run ingest.py!")
        return None
        
    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
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
    
    tools = [rag_tool, web_tool]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

agent = setup_agent()

# 4. The Chat Interface Logic
# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages (so they don't disappear when you type new ones)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything..."):
    # Show user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI response
    with st.chat_message("assistant"):
        response = agent.invoke(prompt)
        output_text = response['output']
        st.markdown(output_text)
        # Save AI response to history
        st.session_state.messages.append({"role": "assistant", "content": output_text})

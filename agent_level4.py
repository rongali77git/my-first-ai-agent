import os
from dotenv import load_dotenv

# 1. The Brain Libraries
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 2. The Memory Libraries (Using 'Classic' because we are using the older, simpler Agent style)
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA 

# 3. The Hand Libraries (Tools)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType 
from langchain_core.tools import Tool

load_dotenv()

print("--- INITIALIZING LEVEL 4 AGENT ---")

# --- PART 1: SETUP THE BRAIN ---
# temperature=0 makes the AI strict and factual.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- PART 2: SETUP THE MEMORY (RAG Tool) ---
if not os.path.exists("./chroma_db"):
    raise FileNotFoundError("Could not find chroma_db folder. Please run ingest.py first!")

embedding_function = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# We create a "RetrievalQA" chain that knows how to search the database.
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# We wrap that chain in a Tool so the Agent can "hold" it.
rag_tool = Tool(
    name="Private_Knowledge_Base",
    func=rag_chain.invoke,
    description="Useful for answering questions about Project Apollo, internal budgets, or company secrets."
)

# --- PART 3: SETUP THE HANDS (Search Tool) ---
search = DuckDuckGoSearchRun()

web_tool = Tool(
    name="Public_Web_Search",
    func=search.run,
    description="Useful for finding current stock prices, weather, news, or public information."
)

# --- PART 4: THE MANAGER (The Agent) ---
tools = [rag_tool, web_tool]

# We initialize the agent with "ZERO_SHOT_REACT_DESCRIPTION".
# This means: "Read the tool descriptions and Reason about which one to use."
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, # Shows the "Thinking" process in the terminal
    handle_parsing_errors=True
)

# --- PART 5: THE MISSION ---
print("Agent is ready. Asking a complex multi-step question...")
print("-" * 50)

# This question forces the Agent to use BOTH tools.
query = "I want to buy 50 Tesla Model S Plaid cars using the Project Apollo budget. Can I afford it? specify the math."
response = agent.invoke(query)

print("-" * 50)
print("FINAL ANSWER:")
print(response['output'])


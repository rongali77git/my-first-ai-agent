import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load API Keys
load_dotenv()

# 2. Safety Check
if not os.path.exists("data/project_apollo.txt"):
    raise FileNotFoundError("Please create 'data/project_apollo.txt' first!")

print("--- STARTED INGESTION ---")

# 3. Load the Document
# This reads the text from your hard drive.
loader = TextLoader("data/project_apollo.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document(s).")

# 4. Split the Document
# We chop text into 1000-character pieces. If a chunk is too big, the AI gets confused.
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks.")

# 5. Create Embeddings & Save to Disk
# This sends your text to OpenAI, gets back "vectors" (numbers), and saves them in the 'chroma_db' folder.
embeddings = OpenAIEmbeddings()

print("Creating Vector Database... (This might take a moment)")
db = Chroma.from_documents(
    texts, 
    embeddings, 
    persist_directory="./chroma_db"
)

print("--- INGESTION COMPLETE ---")

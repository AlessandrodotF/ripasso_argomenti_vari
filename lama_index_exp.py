from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from PyPDF2 import PdfReader
import chromadb


# 1️⃣ Leggi PDF
reader = PdfReader("Alessandro_Fella_CV.pdf")
text = "\n".join(page.extract_text() for page in reader.pages)
doc = Document(text=text)

# 2️⃣ Chunking ottimizzato
parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents([doc])

# 3️⃣ Setup Chroma persistente
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("source")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4️⃣ Embeddings locali
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5️⃣ Costruisci l’indice
index = VectorStoreIndex.from_documents(
    nodes, storage_context=storage_context, embed_model=embed_model
)

# 6️⃣ LLM locale (Ollama)
llm = Ollama(model="llama3.2:1b", context_window=1024)

# 7️⃣ Query engine migliorato
query_engine = index.as_query_engine(
    llm=llm, similarity_top_k=5, response_mode="compact"
)

# 8️⃣ Query
response = query_engine.query("In cosa è specializzato Alessandro?")
print(response)
# --- il tuo codice principale ---


# --- QUI metti il cleanup ---
import gc
import os

os.system("ollama stop llama3.2:1b")  # libera RAM e swap

to_delete = ["doc", "index", "vector_store", "storage_context"]
for var in to_delete:
    globals().pop(var, None)

gc.collect()

try:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
except:
    pass

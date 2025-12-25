from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from typing import List
from PyPDF2 import PdfReader
import chromadb


def pdf_to_text(filaname: str) -> str:
    reader = PdfReader(filaname)
    text = "\n".join(page.extract_text() for page in reader.pages)
    return text


def pdfs_to_text(filenames: List[str]) -> List[str]:
    full_text = []
    for filename in filenames:
        reader = PdfReader(filename)
        text = "\n".join(page.extract_text() for page in reader.pages)
        full_text.append(text)
    return full_text


# oggetto document --- formalemnte posso estrarre tutto il testo e sbatterlo qui dentro
doc = Document(text=pdf_to_text("AlessandroFella_CV.pdf"))
# docs = [Document(text=t) for t in pdfs_to_text(["doc1.pdf", "doc2.pdf"])]

# Node parser -- slide
parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
# calcolo i chunks ma in versione  llama index
nodes = parser.get_nodes_from_documents([doc])

# embedd
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# chroma db - index
client = chromadb.PersistentClient(path="./chroma_db")  # inizializza il db
collection = client.get_or_create_collection("source")  # tabella dove salvo i vettori
vector_store = ChromaVectorStore(
    chroma_collection=collection
)  # connette llamaindex e chroma
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)  # configurrazione globale su dove salvare le cose

# genero gli embedd con embed_model
# li passa al vector_store
# il vector store li salva in collection
index = VectorStoreIndex.from_documents(
    nodes, storage_context=storage_context, embed_model=embed_model
)  # indice vettoriale vero e proprio


# LLM locale (Ollama) --- di defualt llama index è integrato olto bene con openAI

llm = Ollama(model="llama3.2:1b", context_window=1024, temperature=0)

# Query engine migliorato
query_engine = index.as_query_engine(
    llm=llm, similarity_top_k=5, response_mode="compact"
)

# Query
response = query_engine.query("In cosa ha fatto la tesi magistrale alessandro?")
print(response)

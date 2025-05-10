from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from pinecone import ServerlessSpec

def insert_accounting_rules(
    pdf_path: str,
    index_name: str = "accounting_rules",
    pinecone_region: str = "us-east-1",
    openai_api_key: str = None,
    pinecone_api_key: str = None
):
    """
    Load AAOIFI or IFRS standard PDF and upsert into Pinecone index.
    """
    openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    pinecone_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        raise ValueError("OPENAI_API_KEY and PINECONE_API_KEY must be set")
    os.environ["OPENAI_API_KEY"] = openai_key

    # Load and split the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    texts = [d.page_content for d in chunks]
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectors = embeddings.embed_documents(texts)

    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=pinecone_key)
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=pinecone_region)
        )
    index = pc.Index(index_name)

    # Upsert with metadata
    to_upsert = []
    for i, vec in enumerate(vectors):
        to_upsert.append({
            "id": f"rule-{os.path.basename(pdf_path)}-{i}",
            "values": vec,
            "metadata": {"text": texts[i], "source": os.path.basename(pdf_path)}
        })
    index.upsert(to_upsert)
    print(f"Upserted {len(to_upsert)} accounting rule chunks from {pdf_path}")

if __name__ == "__main__":
    insert_accounting_rules(
        pdf_path="rules/AAOIFI_standards.pdf",
    )
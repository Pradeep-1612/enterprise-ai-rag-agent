import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma = chromadb.PersistentClient(path="./chroma_db")

collection = chroma.get_or_create_collection(
    name="enterprise_docs"
)

documents = [
    {
        "id": "unrelated-1",
        "text": """
        Unrelated context data to this usecase. Just dummy text. Hello AI. Hello Pradeep!
        """,
        "source": "unrelated-guide.txt",
        "topic": "unrelated"
    },
    {
        "id": "cert-1",
        "text": "Production SSL/TLS certificates must be renewed before their expiration date.",
        "source": "certificate-management.txt",
        "topic": "renewal"
    },
    {
        "id": "cert-2",
        "text": "Certificates expiring within 30 days are considered at risk.",
        "source": "certificate-management.txt",
        "topic": "risk"
    },
    {
        "id": "cert-3",
        "text": "The operations team should be notified when a certificate has fewer than 15 days remaining.",
        "source": "certificate-management.txt",
        "topic": "notification"
    },
    {
        "id": "cert-4",
        "text": "Emergency renewal is required when a production certificate has fewer than 7 days remaining.",
        "source": "certificate-management.txt",
        "topic": "emergency"
    },
    {
        "id": "integration-1",
        "text": "Each trading partner connection can have one or more SSL/TLS certificates.",
        "source": "integration-guide.txt",
        "topic": "connections"
    },
    {
        "id": "integration-2",
        "text": "When a certificate is renewed, the associated integration connection must be validated.",
        "source": "integration-guide.txt",
        "topic": "validation"
    }
]

for document in documents:

    # response = client.embeddings.create(
    #     model="text-embedding-3-small",
    #     input=document["text"]
    # )

    # embedding = response.data[0].embedding

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding = embedding_model.encode(
        document["text"]
    ).tolist()

    collection.add(
        ids=[document["id"]],
        documents=[document["text"]],
        embeddings=[embedding],
        metadatas=[{
            "source": document["source"],
            "topic": document["topic"]
        }]
    )

print("Documents indexed successfully.")
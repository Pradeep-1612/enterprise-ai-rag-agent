import chromadb

from sentence_transformers import SentenceTransformer


class SemanticMemory:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="semantic_memory"
        )

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def _fact_to_text(self, fact):
        return (
            f"{fact['subject']} "
            f"{fact['predicate']} "
            f"{fact['object']}"
        )

    def index(self, fact):
        text = self._fact_to_text(fact)

        embedding = self.embedding_model.encode(
            text
        ).tolist()

        self.collection.upsert(
            ids=[
                self._fact_id(fact)
            ],
            documents=[
                text
            ],
            embeddings=[
                embedding
            ],
            metadatas=[
                {
                    "type": fact["type"],
                    "subject": fact["subject"],
                    "predicate": fact["predicate"],
                    "object": fact["object"],
                    "source": fact["source"],
                    "confidence": fact["confidence"]
                }
            ]
        )

    def search(self, query, n_results=5):

        query_embedding = self.embedding_model.encode(
            query
        ).tolist()

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        return results

    def delete(self, fact):
        self.collection.delete(
            ids=[
                self._fact_id(fact)
            ]
        )

    def _fact_id(self, fact):
        return (
            f"{fact['subject']}:"
            f"{fact['predicate']}:"
            f"{fact['object']}"
        )
import chromadb

from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(
        self,
        chroma_path="./chroma_db",
        collection_name="enterprise_docs",
        embedding_model_name="all-MiniLM-L6-v2"
    ):
        self.chroma = chromadb.PersistentClient(
            path=chroma_path
        )

        self.collection = self.chroma.get_collection(
            name=collection_name
        )

        self.embedding_model = SentenceTransformer(
            embedding_model_name
        )

    def retrieve(
        self,
        question,
        n_results=3
    ):
        question_embedding = self.embedding_model.encode(
            question
        ).tolist()

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        return results

    def retrieve_context(
        self,
        question,
        n_results=3
    ):
        results = self.retrieve(
            question,
            n_results
        )

        contexts = results["documents"][0]

        return "\n\n---\n\n".join(contexts)
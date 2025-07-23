from nano_vectordb import NanoVectorDB
from util.check_db import is_update_required
import numpy


class DB:
    def __init__(self, embedding_dim: int, storage_file: str) -> None:
        self.embedding_dim = embedding_dim
        self.storage_file = storage_file
        self.vdb = NanoVectorDB(self.embedding_dim, storage_file=self.storage_file)

    def update(self, data: numpy.ndarray) -> None:
        """Updates DB with given data.

        Args:
            data (numpy.ndarray): The data.
        """
        # TBD: move dtype conversion of data into DB class.

        self.vdb.upsert(datas=data)
        self.vdb.save()

    def load(self) -> None:
        """Loads the vector DB"""
        # not necessary for nano_vdb; implemented for consistency
        self.vdb = NanoVectorDB(self.embedding_dim, storage_file=self.storage_file)

    def query_db(self, query: numpy.ndarray, top_k: int = 5) -> list:
        """Queries the DB for chunks.

        Args:
            query (str): Embedded query.
            top_k (int, optional): Top k results to be returned. Defaults to 5.

        Returns:
            list: List of relevant chunks.

        """

        results = self.vdb.query(query=query, top_k=top_k)

        return results

    def req_update(
        self, dir_text_chunks: str, dir_vector_db: str, dir_doc_store: str
    ) -> bool:
        """Does the DB require an update?

        Args:
            dir_text_chunks (str): Directory to the text DB
            dir_vector_db (str): Directory to the vector DB
            dir_doc_store (str): Directory to the document store

        Returns:
            bool: True if documents in document store are newer than DBs. Otherwise False.
        """
        return is_update_required(dir_text_chunks, dir_vector_db, dir_doc_store)

from nano_vectordb import NanoVectorDB
from util.check_db import is_update_required
import numpy
from .base import AbstractDB


class DB(AbstractDB):
    name = "nano_vdb"

    def _init_client(self):
        return NanoVectorDB(self.embedding_dim, storage_file=self.storage_file)

    def update(self, data: numpy.ndarray) -> None:
        # TBD: move dtype conversion of data into DB class.

        self.vdb.upsert(datas=data)
        self.vdb.save()

    def load(self) -> None:
        # not necessary for nano_vdb; implemented for consistency
        self.vdb = NanoVectorDB(self.embedding_dim, storage_file=self.storage_file)

    def query_db(self, query: numpy.ndarray, top_k: int = 5) -> list:
        results = self.vdb.query(query=query, top_k=top_k)

        return results

    def req_update(
        self, dir_text_chunks: str, dir_vector_db: str, dir_doc_store: str
    ) -> bool:
        # probably cleaner if removed; util function can be used instead
        return is_update_required(dir_text_chunks, dir_vector_db, dir_doc_store)

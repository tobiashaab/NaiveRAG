from rag_pipeline.api.gemini import LLM, Embedding
from rag_pipeline.db.nano_vdb import DB
from rag_pipeline.chunking.token_size import ChunkingByTokenSize
import numpy as np
import os
import json
from .base import AbstractRAG


class NaiveRAG(AbstractRAG):
    name = "naiverag"

    def chunk(self, documents: list) -> list:
        split_documents = []
        for document in documents:
            chunks = self.chunking.chunk(document["page_content"])
            for chunk in chunks:
                split_documents.append(
                    {"page_content": chunk["content"], "metadata": document["metadata"]}
                )
        return split_documents

    def _generate_vectordb(self, chunks: list) -> None:
        """Generates the vector DB.

        Args:
            chunks (list): The chunks.
        """

        list_data = [{"__id__": f"chunk-{i}"} for i in range(len(chunks))]

        contents = [chunk["page_content"] for chunk in chunks]

        embeddings_list = self.embedding.embed(contents)

        for i, d in enumerate(list_data):
            d["__vector__"] = np.array(embeddings_list[i], dtype=np.float32)

        self.vdb.update(data=list_data)

    def _generate_textdb(self, chunks: list) -> None:
        """Generates the text DB

        Args:
            chunks (list): The chunks.
        """
        self.text_chunks_db = {
            f"chunk-{i}": {
                "content": chunk["page_content"],
                "tokens": "TBD",
                "chunk_order_index": i,
                "full_doc_id": os.path.basename(chunk["metadata"]["source"]),
            }
            for i, chunk in enumerate(chunks)
        }

        with open(self.text_db_storage_file, "w", encoding="utf-8") as file:
            json.dump(self.text_chunks_db, file, ensure_ascii=False, indent=4)

    def generate_db(self, chunks: list) -> None:
        if not chunks:
            raise ValueError("No chunks available to generate the vector database.")

        self._generate_vectordb(chunks)

        self._generate_textdb(chunks)

    def load_db(self) -> None:
        self.vdb.load()

        with open(self.text_db_storage_file, "r", encoding="utf-8") as file:
            self.text_chunks_db = json.load(file)

    def _retrieve_chunks(self, embed_query: np.ndarray) -> list:
        """Queries the DB with a given embedding, returning a list of top_k text chunks.

        Args:
            embed_query (np.ndarray): The embedded query.

        Returns:
            list: A list of chunks.
        """
        results = self.vdb.query_db(embed_query)

        documents = []

        for res in results:
            if res["__id__"] in self.text_chunks_db:
                content = self.text_chunks_db.get(res["__id__"])["content"]
                documents.append(content)

        return documents

    def _create_prompt_template(self) -> str:
        """Creates and returns the prompt template.

        Returns:
            str:    The prompt template.
        """
        template = """
        ---Role---\n\nYou are a helpful assistant responding to questions about documents provided.\n\n\n---Goal---\n\nGenerate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.\nIf you don't know the answer, just say so. Do not make anything up.\nDo not include information where the supporting evidence for it is not provided.\n\n---Target response length and format---\n\nMultiple Paragraphs\n\n---Documents---\n\n{content_data}\n\nAdd sections and commentary to the response as appropriate for the length and format. Style the response in markdown.\n".
        """
        return template

    def query(self, query: str) -> str:
        prompt_template = self._create_prompt_template()

        embed_query = self.embedding.embed(query)[0]  # returns list!

        embed_query = np.array(embed_query, dtype=np.float32)

        # TBD: include more config parameters (top K, ...)
        relevant_chunks = self._retrieve_chunks(embed_query)

        system_prompt = prompt_template.format(content_data=relevant_chunks)

        message = []

        prompt = f"{system_prompt}\n\nUser Question: {query}"

        response = self.llm.generate(prompt)

        return response

from util.check_db import is_update_required
import rag_pipeline.naiverag
from util.process_docs import pdf_to_txt, get_documents
from rag_pipeline.naiverag import NaiveRAG
from util.load_config import load_config
from util.timer import Timer


def main():

    config_path = "./config.yaml"

    config = load_config(config_path)

    # TBD: _file are folders instead of files..
    naiverag = NaiveRAG(
        api_key=config["api_key"],
        llm_model_name=config["llm_model_name"],
        embedding_model_name=config["embedding_model_name"],
        embedding_dim=config["embedding_dim"],
        vdb_storage_file=config["dir_vector_db"],
        text_chunks_db_path=config["dir_text_chunks"],
    )

    if is_update_required(
        dir_text_chunks=config["dir_text_chunks"],
        dir_vector_db=config["dir_vector_db"],
        dir_doc_store=config["dir_doc_store"],
    ):
        pdf_to_txt()
        documents = get_documents()

        if not documents:
            raise ValueError("No Documents found in the specified Directory!")

        chunks = naiverag.chunk(documents)

        if not chunks:
            raise ValueError("Chunking failed!")

        naiverag.generate_db(chunks=chunks)

    else:
        naiverag.load_db()

    query = "How are you?"

    with Timer():
        response = naiverag.query(query)

    print(response)


if __name__ == "__main__":
    main()

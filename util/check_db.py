import os


def _get_change_time(dir_path: str) -> float:
    """Returns the time of the latest added or changed file within a given directory.

    Args:
        dir_path (str): The directory within which the files are stored.

    Returns:
        float: Latest change time within the specified directory.
    """
    files = os.listdir(dir_path)
    latest_time = 0

    if files:
        for f in files:
            path = os.path.join(dir_path, f)
            creation_time = os.path.getctime(path)

            if creation_time > latest_time:
                latest_time = creation_time

    return latest_time


def is_update_required(
    dir_text_chunks: str, dir_vector_db: str, dir_doc_store: str
) -> bool:
    """Checks whether or not the DB must be updated. Returns true if
    - there is a new or changed document within the docstore
    - the text chunks are missing (i.e. deleted)
    - the vector db is missing (i.e. deleted)


    Args:
        dir_text_chunks (str): The directory within which the text chunks are stored.
        dir_vector_db (str): The directory within which the vector db is stored.
        dir_doc_store (str): The directory within which the .pdf documents are stored.

    Returns:
        bool: Whether or not the DB requires an update. True if it does.
    """
    is_text_chunks = os.path.exists(dir_text_chunks)
    is_vector_db = os.path.exists(dir_vector_db)

    if not is_text_chunks or not is_vector_db:
        return True

    vector_db_last_modified = _get_change_time(dir_vector_db)
    text_chunks_last_modified = _get_change_time(dir_text_chunks)
    doc_store_last_modified = _get_change_time(dir_doc_store)

    if doc_store_last_modified > vector_db_last_modified:
        return True
    elif vector_db_last_modified != text_chunks_last_modified:
        return True

    return False

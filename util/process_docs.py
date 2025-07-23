import os
import pymupdf


def _get_pdf_paths(dir_docs: str = "./doc_store") -> list:
    """Gets all .pdf files from a given directory.

    Args:
        dir_docs (str, optional): The directory in question. Defaults to "./doc_store".

    Returns:
        list: List of .pdf file paths.
    """
    paths = []

    for root, _, files in os.walk(dir_docs):
        for f in files:
            if f.endswith(".pdf"):
                full_path = os.path.join(root, f)
                paths.append(full_path)

    return paths


def pdf_to_txt(dir_docs: str = "./doc_store"):
    """Converts PDF paths in a given directory to .txt

    Args:
        dir_docs (str, optional): The directory in question. Defaults to "./doc_store".
    """
    if not os.path.exists(dir_docs):
        os.makedirs(dir_docs)
        return

    paths = _get_pdf_paths(dir_docs)

    if not paths:
        return

    for p in paths:

        txt_filename = os.path.splitext(p)[0] + ".txt"

        document = pymupdf.open(p)
        text_content = ""

        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text_content += page.get_text()

            with open(txt_filename, "w", encoding="utf-8") as file:
                file.write(text_content)


def get_documents(dir_docs: str = "./doc_store") -> list:
    """Loads the documents from the txt directory.

    Returns:
        list: List of documents
    """
    paths = []
    for root, _, files in os.walk(dir_docs):
        for f in files:
            if f.endswith(".txt"):
                full_path = os.path.join(root, f)
                paths.append(full_path)

    documents = []

    for p in paths:
        with open(p, "r", encoding="utf-8") as file:
            document_content = file.read()
            documents.append(
                {
                    "page_content": document_content,
                    "metadata": {"source": p},
                }
            )

    return documents

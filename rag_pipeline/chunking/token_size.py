import tiktoken


def _encode_string_by_tiktoken(content: str, tokenizer: str = "cl100k_base") -> list:
    """Encodes a string using Tiktoken.

    Args:
        content (str): The string to encode.
        tokenizer (str, optional): The tokenizer to use. Defaults to "cl100k_base".

    Returns:
        list: List of Token-IDs.
    """
    enc = tiktoken.get_encoding(tokenizer)
    return enc.encode(content)


def _decode_tokens_by_tiktoken(tokens: list, tokenizer: str = "cl100k_base") -> list:
    """Decodes Token-IDs with Tiktoken.

    Args:
        tokens (list): The Token-IDs to decode.
        model_name (str, optional): The model to use. Defaults to "cl100k_base".

    Returns:
        list: The decoded content.
    """
    enc = tiktoken.get_encoding(tokenizer)
    return enc.decode(tokens)


def chunking_by_token_size(
    content: str,
    overlap_token_size: int = 100,
    max_token_size: int = 1200,
    tokenizer: str = "cl100k_base",
):
    """Chunks the provided content by the specified token size.

    Args:
        content (str): The text to chunk.
        overlap_token_size (int, optional): The overlap size of the chunks. Defaults to 100.
        max_token_size (int, optional): The max token size of the chunks. Defaults to 1200.
        tokenizer (str, optional): The tokenizer to use. Defaults to "cl100k_base".

    Returns:
        list: A list of chunks.
    """
    tokens = _encode_string_by_tiktoken(content, tokenizer=tokenizer)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = _decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], tokenizer=tokenizer
        )
        results.append(
            {
                "tokens": "TBD",
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results

import tiktoken
from .base import AbstractChunking
from util.process_docs import encode_string_by_tiktoken, decode_tokens_by_tiktoken


class ChunkingByTokenSize(AbstractChunking):
    name = "token_size"

    def chunk(self, content: str) -> list:
        tokens = encode_string_by_tiktoken(content, tokenizer=self.tokenizer)
        results = []
        for index, start in enumerate(
            range(0, len(tokens), self.max_token_size - self.overlap_token_size)
        ):

            current_chunk = tokens[start : start + self.max_token_size]

            chunk_content = decode_tokens_by_tiktoken(
                current_chunk, tokenizer=self.tokenizer
            )

            num_tokens = self.max_token_size

            results.append(
                {
                    "tokens": len(current_chunk),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
        return results

from google import genai
from .base import AbstractLLM, AbstractEmbedding


class LLM(AbstractLLM):
    name = "gemini"

    def _init_client(self):
        return genai.Client(api_key=self.api_key)

    def generate(self, query: str) -> str:
        response = self.client.models.generate_content(
            model=self.llm_model_name, contents=query
        )
        return response.text


class Embedding(AbstractEmbedding):
    name = "gemini"

    def _init_client(self):
        return genai.Client(api_key=self.api_key)

    def embed(self, texts: list) -> list:
        response = self.client.models.embed_content(
            model=self.embedding_model_name, contents=texts
        )
        return [embedding.values for embedding in response.embeddings]

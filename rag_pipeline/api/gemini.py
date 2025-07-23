from google import genai


class LLM:
    def __init__(self, api_key: str, llm_model_name: str = "gemini-2.0-flash") -> None:
        self.client = genai.Client(api_key=api_key)
        self.llm_model_name = llm_model_name

    def generate(self, query: str) -> str:
        """Generates an answer to a given query.

        Args:
            query (str): The query for the LLM.

        Returns:
            str: The answer.
        """
        response = self.client.models.generate_content(
            model=self.llm_model_name, contents=query
        )
        return response.text


class Embedding:
    def __init__(
        self, api_key: str, embedding_model_name: str = "text-embedding-004"
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.embedding_model_name = embedding_model_name

    def embed(self, texts: list) -> list:
        """Embeds given texts.

        Args:
            texts (list): List containing the texts.
        Returns:
            list: A list of embeddings.
        """
        response = self.client.models.embed_content(
            model=self.embedding_model_name, contents=texts
        )
        return [embedding.values for embedding in response.embeddings]

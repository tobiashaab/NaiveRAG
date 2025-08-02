from abc import ABC, abstractmethod


class AbstractLLM(ABC):
    _implementations: dict[str, type["AbstractRAG"]] = {}

    def __init_subclass__(cls: type["AbstractRAG"], **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            AbstractLLM._implementations[cls.name] = cls

    @classmethod
    def create(cls, implementation_name: str, config: dict, **kwargs) -> "AbstractLLM":
        """Creates class instance.

        Args:
            implementation_name (str): Name of the subclass to init.
            config (dict): The config

        Raises:
            ValueError: Implementation name doesn't exist

        Returns:
            AbstractRAG: The initialized class
        """
        if implementation_name not in cls._implementations:
            raise ValueError(
                f"Subclass {implementation_name} not a valid option. Choose one of the following: {list(cls._implementations.keys())}"
            )

        implementation_class = cls._implementations[implementation_name]
        return implementation_class(config, **kwargs)

    def __init__(self, config: dict) -> None:
        self.llm_model_name = config["llm_model_name"]
        self.api_key = config["api_key"]
        self.client = self._init_client()

    @abstractmethod
    def _init_client(self) -> None:
        """Initializes the LLM client."""
        pass

    @abstractmethod
    def generate(self, query: str) -> str:
        """Generates an answer to a given query.

        Args:
            query (str): The query for the LLM.
        Returns:
            str: The answer.
        """
        pass


class AbstractEmbedding(ABC):
    _implementations: dict[str, type["AbstractEmbedding"]] = {}

    def __init_subclass__(cls: type["AbstractEmbedding"], **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            AbstractEmbedding._implementations[cls.name] = cls

    @classmethod
    def create(
        cls, implementation_name: str, config: dict, **kwargs
    ) -> "AbstractEmbedding":
        """Creates class instance.

        Args:
            implementation_name (str): Name of the subclass to init.
            config (dict): The config

        Raises:
            ValueError: Implementation name doesn't exist

        Returns:
            AbstractRAG: The initialized class
        """
        if implementation_name not in cls._implementations:
            raise ValueError(
                f"Subclass {implementation_name} not a valid option. Choose one of the following: {list(cls._implementations.keys())}"
            )

        implementation_class = cls._implementations[implementation_name]
        return implementation_class(config, **kwargs)

    def __init__(self, config: dict) -> None:
        self.embedding_model_name = config["embedding_model_name"]
        self.api_key = config["api_key"]
        self.client = self._init_client()

    @abstractmethod
    def _init_client(self) -> None:
        """Initializes a client."""
        pass

    @abstractmethod
    def embed(self, texts: list) -> list:
        """Embeds given texts.

        Args:
            texts (list): List containing the texts.
        Returns:
            list: A list of embeddings.
        """
        pass

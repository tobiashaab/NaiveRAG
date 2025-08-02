import os
from abc import ABC, abstractmethod
from .api.base import AbstractEmbedding, AbstractLLM
from .chunking.base import AbstractChunking
from .db.base import AbstractDB


class AbstractRAG(ABC):
    """Abstract RAG interface with factory pattern"""

    _implementations: dict[str, type("AbstractRAG")] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses when they're defined"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            AbstractRAG._implementations[cls.name] = cls

    @classmethod
    def create(cls, implementation_name: str, config: dict, **kwargs) -> "AbstractRAG":
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
        """
        Initialize common RAG components.

        This __init__ will be called by all subclasses via super().__init__(config).
        It sets up the common infrastructure that all RAG implementations need.

        Args:
            config: Configuration object containing paths and settings
        """

        self.config = config
        self.llm = AbstractLLM.create(config["api_implementation_name"], config)
        self.embedding = AbstractEmbedding.create(
            config["api_implementation_name"], config
        )
        self.vdb = AbstractDB.create(config["db_implementation_name"], config)
        self.chunking = AbstractChunking.create(
            config["chunking_implementation_name"], config
        )

        self.text_chunks_db_path = config["dir_text_chunks"]
        self.text_db_storage_file = os.path.join(
            self.text_chunks_db_path, "text_db.json"
        )
        self.vdb_storage_file = os.path.join(config["dir_vector_db"], "vdb.json")

        self.text_chunks_db = {}

    @abstractmethod
    def chunk(self, documents: list) -> list:
        """
        Chunks the given documents into text chunks.

        Args:
            documents (List[str]): The documents to chunk.

        Returns:
            List[str]: The chunks.
        """
        pass

    @abstractmethod
    def generate_db(self, chunks: list) -> None:
        """
        Generates the vector and text DB.

        Args:
            chunks (List[str]): The chunks for the DB.

        Raises:
            ValueError: No chunks given.
        """
        pass

    @abstractmethod
    def load_db(self) -> None:
        """Loads the vector and text DB"""
        pass

    @abstractmethod
    def query(self, query: str) -> str:
        """
        Queries the DB with a given User Query, returning the LLM generated response.

        Args:
            query (str): The User Query.

        Returns:
            str: The generated answer.
        """
        pass

from abc import ABC, abstractmethod
import numpy
import os


class AbstractDB(ABC):
    _implementations: dict[str, type["AbstractDB"]] = {}

    def __init_subclass__(cls: type["AbstractDB"], **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            AbstractDB._implementations[cls.name] = cls

    @classmethod
    def create(cls, implementation_name: str, config: dict, **kwargs) -> "AbstractDB":
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
        self.embedding_dim = config["embedding_dim"]
        self.storage_file = os.path.join(config["dir_vector_db"], "vdb.json")
        self.vdb = self._init_client()

    @abstractmethod
    def _init_client(self):
        """Initializes the VDB client."""
        pass

    @abstractmethod
    def update(self, data: numpy.ndarray) -> None:
        """Updates DB with given data.

        Args:
            data (numpy.ndarray): The data.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """Loads the vector DB."""
        pass

    @abstractmethod
    def query_db(self, query: numpy.ndarray, top_k: int) -> list:
        """Queries the DB for chunks.

        Args:
            query (str): Embedded query.
            top_k (int, optional): Top k results to be returned. Defaults to 5.

        Returns:
            list: List of relevant chunks.

        """
        pass

    @abstractmethod
    def req_update(
        self, dir_text_chunks: str, dir_vector_db: str, dir_doc_store: str
    ) -> bool:
        """Does the DB require an update?

        Args:
            dir_text_chunks (str): Directory to the text DB
            dir_vector_db (str): Directory to the vector DB
            dir_doc_store (str): Directory to the document store

        Returns:
            bool: True if documents in document store are newer than DBs. Otherwise False.
        """
        pass

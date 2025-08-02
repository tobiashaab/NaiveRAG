from abc import abstractmethod, ABC


class AbstractChunking(ABC):
    _implementations: dict[str, type["AbstractChunking"]] = {}

    def __init_subclass__(cls: type["AbstractChunking"], **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            AbstractChunking._implementations[cls.name] = cls

    @classmethod
    def create(
        cls, implementation_name: str, config: dict, **kwargs
    ) -> "AbstractChunking":
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

    def __init__(self, config: dict):
        self.overlap_token_size = config["overlap_token_size"]
        self.max_token_size = config["max_token_size"]
        self.tokenizer = config["tokenizer"]

    @abstractmethod
    def chunk(self, content: str) -> list:
        """Chunks the provided content.

        Args:
            content (str): The content to chunk.

        Returns:
            list: The chunked content.
        """
        pass

from datetime import datetime


class Timer:
    """Times the  duration of a code execution. Can be used in a "with" statement."""

    def __enter__(self) -> None:
        self.start_time = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = datetime.now()
        self.time = self.end_time - self.start_time
        print(f"Duration: {self.time}")

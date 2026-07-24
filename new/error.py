class EncodeError(BaseException):
    """Shows errors related to an LLM encoding issue

    Args:
        BaseException: base exception class.
    """

    def __init__(self, msg: str) -> None:
        """Creates an encoding error

        Args:
            msg (str): Message to display if the error happens
        """
        super().__init__(f"Encode Error: {msg}")


class DecodeError(BaseException):
    """Shows errors related to an LLM decoding issue

    Args:
        BaseException: base exception class.
    """

    def __init__(self, msg: str) -> None:
        """Creates a decoding error

        Args:
            msg (str): Message to display if the error happens
        """
        super().__init__(f"Decode Error: {msg}")

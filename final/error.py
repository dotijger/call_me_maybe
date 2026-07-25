class ParsingError(BaseException):
    """Shows errors related to a parsing issue

    Args:
        BaseException: base exception class.
    """

    def __init__(self, msg: str) -> None:
        """Creates a parsing error

        Args:
            msg (str): Message to display if the error happens
        """
        super().__init__(f"Parsing Error: {msg}")


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


class NoParameterDefinedError(BaseException):
    """Shows errors related to a parameter generator issue

    Args:
        BaseException: base exception class.
    """

    def __init__(self, msg: str) -> None:
        """Creates a generator error

        Args:
            msg (str): Message to display if the error happens
        """
        super().__init__(f"No Parameter Defined Error: {msg}")

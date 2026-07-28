class ParsingError(BaseException):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)

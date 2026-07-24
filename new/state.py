from enum import Enum, auto


class ParameterState(Enum):
    STRING_START = auto()
    STRING_MID = auto()
    NUM_START = auto()
    NUM_MID_NO_DOT = auto()
    NUM_MID_DOT = auto()


def is_valid_string_start(generated: str, token: str, is_last_parameter: bool) -> bool:
    temp = generated + token
    if temp[0] != '"':
        return False
    if len(temp) > 1:
        return is_valid_string_mid(generated, token, is_last_parameter)
    return True


def is_valid_string_mid(generated: str, token: str, is_last_parameter: bool) -> bool:
    temp = generated + token
    terminator = "}" if is_last_parameter else ","
    end = f'"{terminator}'
    if len(generated) >= 100:
        if not token.endswith(end):
            return False
    if temp.endswith(end):
        content = temp[1 : -len(end)]
    elif temp.endswith('"'):
        content = temp[1:-1]
    else:
        content = temp[1:]
    return content.isalpha()


def is_valid_number_start(generated: str, token: str, is_last_parameter: bool) -> bool:
    temp = generated + token
    if len(temp) > 1:
        return is_valid_number_mid_no_dot(generated, token, is_last_parameter)
    if temp[0] == "-" or temp[0] == ".":
        return True
    return temp.isdigit()


def is_valid_number_mid_no_dot(
    generated: str, token: str, is_last_parameter: bool
) -> bool:
    dots = 0
    for char in token:
        if char == "-":
            return False
        if char == ".":
            dots += 1
    if dots > 1:
        return False
    temp = generated + token
    terminator = "}" if is_last_parameter else ","
    if len(generated) >= 20:
        if token[-1] != terminator:
            return False
    if temp.endswith(terminator):
        nbr = temp[:-1]
    else:
        nbr = temp
    return is_number(nbr)


def is_valid_number_mid_has_dot(
    generated: str, token: str, is_last_parameter: bool
) -> bool:
    for char in token:
        if char == ".":
            return False
        if char == "-":
            return False
    temp = generated + token
    terminator = "}" if is_last_parameter else ","
    if len(generated) >= 20:
        if token[-1] != terminator:
            return False
    if temp.endswith(terminator):
        nbr = temp[:-1]
    else:
        nbr = temp
    return is_number(nbr)


def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def is_candidate_allowed(
    state: ParameterState, generated: str, token: str, is_last_parameter: bool
) -> bool:
    if state == ParameterState.STRING_START:
        return is_valid_string_start(generated, token, is_last_parameter)
    if state == ParameterState.STRING_MID:
        return is_valid_string_mid(generated, token, is_last_parameter)
    if state == ParameterState.NUM_START:
        return is_valid_number_start(generated, token, is_last_parameter)
    if state == ParameterState.NUM_MID_NO_DOT:
        return is_valid_number_mid_no_dot(generated, token, is_last_parameter)
    if state == ParameterState.NUM_MID_DOT:
        return is_valid_number_mid_has_dot(generated, token, is_last_parameter)

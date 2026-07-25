#!/usr/bin/env/python3

import json


def remove_used(text: str, text2: str) -> str:
    available = text.split(" ")
    words = text2.split(" ")
    for i in range(len(words)):
        if words[i] == '"parameters":':
            cut = words[i:]
    used = []
    for word in cut:
        if word[-1] == ",":
            word = word[:-1]
            if word.isdigit():
                used.append(word)
    if len(used) == 0:
        return ""
    for word in used:
        if word in available:
            available.remove(word)
    return " ".join(available)


def output_to_json() -> None:
    output = [
        '{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}',
        '{"prompt": "What is the sum of 265 and 345?", "name": "fn_add_numbers", "parameters": {"a": 265, "b": 345}}',
        '{"prompt": "Greet shrek", "name": "fn_greet", "parameters": {"name": "shrek"}}',
        '{"prompt": "Greet john", "name": "fn_greet", "parameters": {"name": "john"}}',
        '{"prompt": "Reverse the string \'hello\'", "name": "fn_reverse_string", "parameters": {"s": "hello"}}',
        '{"prompt": "Reverse the string \'world\'", "name": "fn_reverse_string", "parameters": {"s": "world"}}',
        '{"prompt": "What is the square root of 16?", "name": "fn_get_square_root", "parameters": {"a": 16}}',
        '{"prompt": "Calculate the square root of 144", "name": "fn_get_square_root", "parameters": {"a": 144}}',
        '{"prompt": "Replace all numbers in \'Hello 34 I\'m 233 years old\' with NUMBERS", "name": "fn_substitute_string_with_regex", "parameters": {"source_string": "Hello", "regex": "Hello", "replacement": "Hello"}}',
        '{"prompt": "Replace all vowels in \'Programming is fun\' with asterisks", "name": "fn_substitute_string_with_regex", "parameters": {"source_string": "Programming", "regex": "Programming", "replacement": "Programming"}}',
        '{"prompt": "Substitute the word \'cat\' with \'dog\' in \'The cat sat on the mat with another cat\'", "name": "fn_substitute_string_with_regex", "parameters": {"source_string": "The", "regex": "cat", "replacement": "dog"}}',
    ]
    list_of_json = []
    for prompt in output:
        list_of_json.append(json.loads(prompt))
    with open("json_output.json", "w") as jfile:
        json.dump(list_of_json, jfile, indent=2)


if __name__ == "__main__":
    test = "2 is 3 is 5 and 5"
    test2 = '{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2,'
    test = remove_used(test, test2)
    print(test)
    output_to_json()

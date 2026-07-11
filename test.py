import json


with open("src/functions_definitions.json") as f:
    file = json.load(f)
print(file)
print(type(file))
print(file[0])
print(type(file[0]))

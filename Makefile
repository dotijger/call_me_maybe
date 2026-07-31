MYPY_FLAGS= --warn-return-any \
						--warn-unused-ignore \
						--ignore-missing-imports \
						--disallow-untyped-defs \
						--check-untyped-defs
SRC_DIR= src
FD_DIR= data/input/functions_definition.json


all: run

install:
	uv sync

run:
	uv run python3 -m $(SRC_DIR) -functions_definition $(FD_DIR)

debug:
	uv run python3 -m pdb $(SRC_DIR) -functions_definition $(FD_DIR)

clean:
	rm -rf __pycache__ src/__pycache__

lint:
	flake8 .
	mypy . $(MYPY_FLAGS)

lint-strict:
	flake8
	mypy . --strict



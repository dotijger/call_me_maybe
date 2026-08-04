MYPY_FLAGS= --warn-return-any \
						--warn-unused-ignore \
						--ignore-missing-imports \
						--disallow-untyped-defs \
						--check-untyped-defs \
						--explicit-package-bases
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
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 src
	uv run mypy src $(MYPY_FLAGS)

lint-strict:
	uv run flake8 src
	uv run mypy src --strict --explicit-package-bases



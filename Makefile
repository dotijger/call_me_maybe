MYPY_FLAGS= --warn-return-any \
						--warn-unused-ignore \
						--ignore-missing-imports \
						--disallow-untyped-defs \
						--check-untyped-defs \
						--explicit-package-bases
SRC_DIR= src
BONUS_DIR= src_bonus


all: run

install:
	uv sync

help:
	uv run python3 -m $(SRC_DIR) -help

run:
	uv run python3 -m $(SRC_DIR)

visual:
	uv run python3 -m $(SRC_DIR) --visual

bonus:
	uv run python3 -m $(BONUS_DIR) --visual

debug:
	uv run python3 -m pdb -m $(SRC_DIR)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache
	rm log

lint:
	uv run flake8 src
	uv run mypy src $(MYPY_FLAGS)

lint-strict:
	uv run flake8 src
	uv run mypy src --strict --explicit-package-bases



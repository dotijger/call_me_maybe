import argparse
import json
from pathlib import Path
from pydantic import BaseModel, PrivateAttr


class Parser(BaseModel):
    _parser: argparse.ArgumentParser = PrivateAttr()
    _args: argparse.Namespace = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        self._parser = argparse.ArgumentParser(
            prog="uv run python3 -m src",
            description="Codam's Core Curriculum introductory project into \
constrained decoding and LLM's.",
        )
        self._parser.add_argument(
            "--functions_definition",
            type=Path,
            default=Path("data/input/functions_definition.json"),
            help="Path to the JSON file containing functions\
and their definitions available to the LLM. (Default path: %(default)s)",
        )
        self._parser.add_argument(
            "--input",
            type=Path,
            default=Path("data/input/function_calling_tests.json"),
            help="Path to the JSON file containing the prompts for the\
 LLM to process. (Default path: %(default)s)",
        )
        self._parser.add_argument(
            "--output",
            type=Path,
            default=Path("data/output/function_calls.json"),
            help="Path to where the output JSON will be written.\
(Default path: %(default)s)",
        )
        self._parser.add_argument(
            "--visual",
            "--v",
            action="store_true",
            help="Enable visualization of the Call Me Maybe program.\
(Default: %(default)s)",
        )
        self._args = self._parser.parse_args()

    def _check_files(self) -> None:
        flags = [self._args.input, self._args.functions_definition]
        for flag in flags:
            try:
                with open(flag, encoding="utf-8") as f:
                    _ = json.load(f)
            except FileNotFoundError:
                raise ValueError(f"File not found: {flag}")
            except IsADirectoryError:
                raise ValueError(f"Path is a directory {flag}.")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {flag}: {e}")

    @property
    def args(self) -> argparse.Namespace:
        try:
            self._check_files()
            return self._args
        except ValueError as e:
            raise ValueError(e)

    def parse_prompts(self) -> list[str]:
        try:
            self._check_files()
        except ValueError as e:
            raise ValueError(e)
        prompts = []
        with open(self._args.input) as f:
            raw_prompts = json.load(f)
        for prompt in raw_prompts:
            for key, value in prompt.items():
                prompts.append(value)
        return prompts


if __name__ == "__main__":
    p = Parser()
    try:
        args = p.args
        print(args.functions_definition, args.input, args.output, args.visual)
    except ValueError as e:
        print(e)

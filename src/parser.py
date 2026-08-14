import argparse
import json
from pathlib import Path
from pydantic import BaseModel, PrivateAttr


class Parser(BaseModel):
    """Wraps CLI argument parsing and input-file validation.

    Builds the program's ``argparse`` parser on initialization,
    exposes the parsed arguments through the ``args`` property, and
    validates that the input JSON files referenced by those arguments
    exist and contain well-formed JSON.
    """

    _parser: argparse.ArgumentParser = PrivateAttr()
    _args: argparse.Namespace = PrivateAttr()

    def model_post_init(self, __context: object) -> None:
        """Builds the CLI argument parser and parses ``sys.argv``.

        Registers the ``--functions_definition``, ``--input``,
        ``--output``, and ``--visual`` flags, then immediately parses
        the command-line arguments into ``self._args``.

        Args:
            __context (object): The pydantic post-init context, unused.

        Returns:
            None
        """
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
            default=Path("data/output/function_calling_results.json"),
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
        """Validates that the input and function-definition files are usable.

        Confirms that both ``--input`` and ``--functions_definition``
        point to existing, readable files containing valid JSON.

        Raises:
            ValueError: If a file is missing, is a directory, or
                contains invalid JSON.

        Returns:
            None
        """
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
        """Returns the parsed CLI arguments after validating input files.

        Returns:
            argparse.Namespace: The parsed command-line arguments.

        Raises:
            ValueError: If the input or function-definition files fail
                validation.
        """
        try:
            self._check_files()
            return self._args
        except ValueError as e:
            raise ValueError(e)

    def parse_prompts(self) -> list[str]:
        """Loads and flattens the natural-language prompts from the input file.

        Returns:
            list[str]: Every prompt value found in the input JSON file,
                in file order.

        Raises:
            ValueError: If the input or function-definition files fail
                validation.
        """
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

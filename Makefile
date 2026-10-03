INPUT_FILE   = data/input/function_calling_tests.json
DEF_FILE     = data/input/functions_definition.json
OUTPUT_FILE  = data/output/function_calling_results.json
MODEL        = Qwen/Qwen3-0.6B

.PHONY: install run debug clean lint lint-strict

run:
	uv run python -m src \
		--functions_definition $(DEF_FILE) \
		--input $(INPUT_FILE) \
		--output $(OUTPUT_FILE) \
		--model $(MODEL)

install:
	uv sync

debug:
	uv run python -m pdb -m src \
		--functions_definition $(DEF_FILE) \
		--input $(INPUT_FILE) \
		--output $(OUTPUT_FILE) \
		--model $(MODEL)

clean:
	rm -rf llm_sdk/__pycache__
	rm -rf src/__pycache__
	rm -rf .mypy_cache

lint:
	uv run flake8 src
	uv run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

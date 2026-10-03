import argparse
import json
import os
import sys
from enum import Enum
from typing import Any, cast
import numpy as np
from pydantic import BaseModel, Field, ValidationError, create_model
from llm_sdk import Small_LLM_Model


class ParamSpec(BaseModel):
    """Parameter type specification.

    Attributes:
        type: Type name of the parameter (e.g. "string", "number").
    """

    type: str


class Function(BaseModel):
    """Definition of a callable function.

    Attributes:
        name: Function name.
        description: Short description of what the function does.
        parameters: Mapping of parameter name to its ParamSpec.
    """

    name: str
    description: str
    parameters: dict[str, ParamSpec]


class FunctionCallResult(BaseModel):
    """Result of a single function call.

    Attributes:
        prompt: Original user query.
        name: Selected function name.
        parameters: Extracted parameter values.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]


class TokenConstraints(BaseModel):
    """Token ID sets used to constrain decoding.

    Attributes:
        string_inner: Token IDs valid inside a string value.
        string_close: Token IDs that close a string (i.e. `"`).
        number_inner: Token IDs valid inside a number value.
    """

    string_inner: list[int]
    string_close: list[int]
    number_inner: list[int]


def mask(
    logits: np.ndarray[Any, Any],
    valid_ids: list[int]
) -> np.ndarray[Any, Any]:
    """Mask logits to allow only valid token IDs.

    Args:
        logits: Raw logit array from the model.
        valid_ids: Token IDs to keep; all others become -inf.
    Returns:
        Logit array with non-valid positions set to -inf.
    """
    masked = np.full(len(logits), -np.inf)
    for tid in valid_ids:
        masked[tid] = float(logits[tid])
    return masked


def load_vocab(llm: Small_LLM_Model) -> dict[int, str]:
    """Load the vocabulary file and return an ID-to-token mapping.

    Args:
        llm: Model instance that provides the vocab file path.
    Returns:
        Dict mapping token ID to token string.
    """
    vocab_path: str = llm.get_path_to_vocab_file()
    with open(vocab_path, "r") as fh:
        raw: dict[str, int] = json.load(fh)
    return {v: k for k, v in raw.items()}


def build_token_constraints(
    llm: Small_LLM_Model,
    vocab: dict[int, str]
) -> TokenConstraints:
    """Build token constraint sets from the model vocabulary.

    Args:
        llm: Model instance used for encoding.
        vocab: ID-to-token mapping from load_vocab.

    Returns:
        TokenConstraints with classified token ID lists.
    """
    string_inner: list[int] = []
    string_close: list[int] = []
    number_inner: list[int] = []
    number_valid: list[str] = list("0123456789.-")

    for tid, tok in vocab.items():
        if '"' not in tok:
            string_inner.append(tid)
        if tok == '"':
            string_close.append(tid)
        stripped = tok.strip()
        if stripped != "" and all(c in number_valid for c in stripped):
            number_inner.append(tid)

    if not string_close:
        raw: list[int] = (
            llm.encode('"').tolist()[0]
        )
        string_close = [raw[0]]

    return TokenConstraints(
        string_inner=string_inner,
        string_close=string_close,
        number_inner=number_inner
    )


def build_name_trie(
    func_defs: list[Function],
    llm: Small_LLM_Model
) -> dict[tuple[int, ...], list[int]]:
    """Build a prefix trie over encoded function names.

    Args:
        func_defs: List of function definitions.
        llm: Model instance used to encode function names.

    Returns:
        Dict mapping token ID prefix tuples to valid next token IDs.
    """
    trie: dict[tuple[int, ...], set[int]] = {}
    for fd in func_defs:
        ids: list[int] = llm.encode(fd.name).tolist()[0]
        for i in range(len(ids)):
            prefix: tuple[int, ...] = tuple(ids[:i])
            if prefix not in trie:
                trie[prefix] = set()
            trie[prefix].add(ids[i])
    return {k: list(v) for k, v in trie.items()}


def select_function(
    llm: Small_LLM_Model,
    prompt_ids: list[int],
    trie: dict[tuple[int, ...], list[int]]
) -> str:
    """Select a function name via trie-constrained greedy decoding.

    Args:
        llm: Model instance for logit retrieval and decoding.
        prompt_ids: Encoded selection prompt token IDs.
        trie: Prefix trie from build_name_trie.

    Returns:
        Decoded function name string.
    """
    ids: list[int] = prompt_ids.copy()
    path: tuple[int, ...] = ()

    while True:
        valid_next = trie.get(path)
        if not valid_next:
            break
        logits = np.array(llm.get_logits_from_input_ids(ids))
        masked = mask(logits, valid_next)
        next_id = int(np.argmax(masked))
        ids.append(next_id)
        path = path + (next_id,)

    return str(llm.decode(list(path)))


NUMERIC_TYPES: dict[str, type] = {
    "number": float,
    "float": float,
    "integer": int,
    "int": int,
}


def generate_value(
    llm: Small_LLM_Model,
    ids: list[int],
    constraints: TokenConstraints,
    param_type: str,
    max_tokens: int = 100
) -> str | int | float:
    """Generate a parameter value via constrained greedy decoding.

    Args:
        llm: Model instance for logit retrieval and decoding.
        ids: Encoded parameter prompt token IDs.
        constraints: Token constraint sets.
        param_type: Parameter type; numeric types apply numeric constraints.
        max_tokens: Maximum tokens to generate. Defaults to 100.

    Returns:
        Generated value cast to the declared type (int, float, or str).
    """
    result_ids: list[int] = []
    python_type: type | None = NUMERIC_TYPES.get(param_type)
    is_number: bool = python_type is not None
    valid_ids: list[int] = (
        constraints.number_inner if is_number else constraints.string_inner
    )
    for _ in range(max_tokens):
        logits = np.array(llm.get_logits_from_input_ids(ids))
        best_id = int(np.argmax(logits))
        best_tok = str(llm.decode([best_id]))
        if is_number and any(c in best_tok for c in ",}"):
            break
        if not is_number and best_tok.startswith('"'):
            break

        masked = mask(logits, valid_ids)
        if np.all(np.isinf(masked)):
            break

        next_id = int(np.argmax(masked))
        ids.append(next_id)
        result_ids.append(next_id)

    text: str = str(llm.decode(result_ids)).strip()
    if python_type is not None:
        try:
            return cast(int | float, python_type(float(text)))
        except (ValueError, TypeError):
            return text
    return text


def build_selection_prompt(
    func_defs: list[Function],
    user_query: str
) -> str:
    """Build the function-selection prompt string.

    Args:
        func_defs: Available function definitions.
        user_query: User input text.

    Returns:
        Prompt ending with ``{"name": "`` ready for decoding.
    """
    hints: str = ""
    for fd in func_defs:
        args: str = ", ".join(
            f"{k}: {v.type}" for k, v in fd.parameters.items()
        )
        hints += f"- {fd.name}: {fd.description} (args: {args})\n"

    return (
        "Select the most appropriate function for the user request. "
        "Extract parameter values exactly.\n"
        "[Available functions]\n"
        f"{hints}"
        "[User request]\n"
        f"{user_query}\n"
        "[Output JSON]\n"
        '{"name": "'
    )


def build_param_prompt(
    user_query: str,
    func_name: str,
    accumulated: dict[str, Any],
    key: str,
    param_type: str
) -> str:
    """Build a prompt for generating a single parameter value.

    Args:
        user_query: User input text.
        func_name: Selected function name.
        accumulated: Already extracted parameters.
        key: Parameter key to generate next.
        param_type: Type of the parameter ("string" or "number").

    Returns:
        Prompt string with partial JSON context appended.
    """
    if accumulated:
        prefix: str = json.dumps(accumulated, ensure_ascii=False)[:-1] + ", "
    else:
        prefix = "{"

    if param_type in NUMERIC_TYPES:
        context: str = f'{prefix}"{key}": '
    else:
        context = f'{prefix}"{key}": "'

    return (
        f'User: "{user_query}"\n'
        f"Function: {func_name}\n"
        f"Parameters: {context}"
    )


def generate_function_call(
    llm: Small_LLM_Model,
    func_defs: list[Function],
    user_query: str,
    trie: dict[tuple[int, ...], list[int]],
    constraints: TokenConstraints
) -> dict[str, Any]:
    """Select a function and extract all parameter values for a query.

    Args:
        llm: Model instance used for inference.
        func_defs: Available function definitions.
        user_query: User input text.
        trie: Prefix trie from build_name_trie.
        constraints: Token constraints from build_token_constraints.

    Returns:
        Dict with "prompt", "name", and "parameters" keys.
    """
    prompt: str = build_selection_prompt(func_defs, user_query)
    prompt_ids: list[int] = llm.encode(prompt).tolist()[0]
    selected_name: str = select_function(llm, prompt_ids, trie)

    func_def: Function
    for fd in func_defs:
        if fd.name == selected_name:
            func_def = fd
            break

    accumulated: dict[str, Any] = {}
    for key, param_spec in func_def.parameters.items():
        p_prompt: str = build_param_prompt(
            user_query, selected_name, accumulated, key, param_spec.type
        )
        p_ids: list[int] = llm.encode(p_prompt).tolist()[0]

        value: Any = generate_value(llm, p_ids, constraints, param_spec.type)

        accumulated[key] = value
        print(f"{key} = {value!r}")

    return {
        "prompt": user_query,
        "name": selected_name,
        "parameters": accumulated
    }


def run_pipeline(
    quest_file: str,
    func_file: str,
    output_file: str,
    model_name: str
) -> None:
    """Run the full function-calling pipeline and write results to a file.

    Args:
        quest_file: Path to JSON file containing test prompts.
        func_file: Path to JSON file containing function definitions.
        output_file: Path where result JSON will be written.
        model_name: LLM model name to load (e.g. "Qwen/Qwen3-0.6B").

    Raises:
        SystemExit: On failure to load files or the LLM model.
    """
    try:
        with open(func_file, "r") as fh:
            funcs_raw: list[dict[str, Any]] = json.load(fh)
    except Exception as e:
        print(f"Error: cannot load function definitions: {e}")
        sys.exit(1)

    try:
        func_defs: list[Function] = [Function(**fn) for fn in funcs_raw]
    except Exception as e:
        print(f"Error: invalid function definitions: {e}")
        sys.exit(1)

    if not func_defs:
        print("Error: no function definitions found")
        sys.exit(1)

    try:
        with open(quest_file, "r") as fh:
            tests: list[dict[str, Any]] = json.load(fh)
    except Exception as e:
        print(f"Error: cannot load test prompts: {e}")
        sys.exit(1)

    names: list[str] = [fd.name for fd in func_defs]
    FuncNameEnum = cast(Any, Enum("FuncNameEnum", {n: n for n in names}))
    ChoiceModel = create_model(
        "FunctionChoice",
        prompt=(str, Field(..., min_length=1)),
        name=(FuncNameEnum, ...),
        parameters=(dict, ...)
    )

    print(f"Set model - {model_name}")
    try:
        llm: Small_LLM_Model = Small_LLM_Model(model_name)
    except Exception as e:
        print(f"Error: failed to load LLM: {e}")
        sys.exit(1)

    print("Building token constraints from vocabulary...")
    try:
        vocab: dict[int, str] = load_vocab(llm)
        constraints: TokenConstraints = build_token_constraints(llm, vocab)
    except Exception as e:
        print(f"Error: failed to build token constraints: {e}")
        sys.exit(1)

    trie: dict[tuple[int, ...], list[int]] = build_name_trie(func_defs, llm)
    results: list[dict[str, Any]] = []

    for item in tests:
        user_query: str = item.get("prompt", "").strip()
        if not user_query:
            continue
        print(f"\nProcessing: {user_query}")

        try:
            result: dict[str, Any] = generate_function_call(
                llm, func_defs, user_query, trie, constraints
            )
            result_json: str = json.dumps(result, ensure_ascii=False)
            validated = cast(
                Any, ChoiceModel.model_validate_json(result_json)
            )
            fn_name: str = validated.name.value
            print(f"  -> {fn_name}")
            results.append(
                FunctionCallResult(
                    prompt=str(validated.prompt),
                    name=fn_name,
                    parameters=dict(validated.parameters)
                ).model_dump()
            )
        except ValidationError as e:
            print(f"Validation error: {e}")
        except Exception as e:
            print(f"Error processing prompt: {e}")

    output_dir: str = os.path.dirname(output_file)
    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=4)
        print(f"\nSaved {len(results)} result(s) to {output_file}")
    except OSError as e:
        print(f"Error: failed to write output: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B"
    )
    args = parser.parse_args()
    run_pipeline(
        args.input,
        args.functions_definition,
        args.output,
        args.model
    )

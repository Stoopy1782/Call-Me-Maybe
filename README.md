*This project has been created as part of the 42 curriculum by ykojima*

# Call-Me-Maybe

## Description 
The subject of "*Call Me Maybe*"  is Using small language models(Qwen3-0.6B) and make accurate JSON output.

This project teaches how to use function calling, how to control LLM outout, JSON text formatting techniques, and how to use commandline system of file. 

## Instructions

### 1. Install program
```
git clone *this program url*
```
Notes: This program takes huge memory __(over 4GB)__ and needs middle range RAM.
So, you should clone on large memory space.

### 2. Make virtual environment
```
python -m venv .venv
source .venv/bin/activate
```
Notes: For this program, please set ".venv" as venv name for uv program.

### 3. Install uv (if not already installed)
Recommended:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Alternative (may segfault on some Linux environments):
```
pip install uv
```

### 4. Install package
```
make install
```

This program needs     
*   **`pydantic`**: For validation
*   **`numpy`**: To control LLM output
*   **`mypy`**: To check type-hints
*   **`torch`**, **`transformers`**, **`huggingface_hub`**: For  LLM_SDK

### 5. Running program
```
make run
```
or
```
uv run python -m src \
--functions_definition data/input/functions_definition.json \
--input data/input/function_calling_tests.json \
--output data/output/function_calling_results.json \
--model Qwen/Qwen3-0.6B
```
This output shows how LLM outputted and % of each choices selection.
https://github.com/user-attachments/assets/d7bac0d8-4c49-4076-b600-38a27739c35e


### 6. Make options

*   **`install`**: To install modules
*   **`run`**: To run program
*   **`debug`**: To debug program
*   **`clean`**: To clean cache
*   **`lint`**, **`lint-strict`**: To check type-hints

## Program Flow

The constrained decode loop below runs 1 + N times per query:
once for function selection, then once per parameter (N = number of parameters).

```
User prompt
    │
    ▼
① Encode
    llm.encode(prompt) → token ID sequence
    │
    ▼
    ② Get logits       get_logits_from_input_ids(ids)
          │
          ▼
    ③ Mask to −∞      mask(logits, valid_ids)
                       function selection  → trie-constrained
                       parameter extraction → type-constrained
          │
          ▼
    ④ Greedy pick      next_id = argmax(masked logits)
                       append next_id to ids
          │
          └── stop condition met? No → back to ②
                                   │ Yes
                             ▼
                    llm.decode(result_ids) → text
                             │
                   repeat for each parameter
                             │
                             ▼
              ⑤ Pydantic validation
                  enforces valid function name & parameter types
                  invalid outputs rejected before writing
                             │
                             ▼
                        Output JSON
```

## Algorithm Explanation

This project uses **constrained decoding** — a technique that restricts which tokens the LLM can generate at each step, guaranteeing valid structured output without relying on prompting alone.

Function selection uses a prefix trie built from each function name's token IDs. At each generation step, only tokens that continue a valid function name are kept; all others are set to −∞. This guarantees the output is always a defined function name.

Parameter extraction masks the logits to three pre-computed token sets:
- **string_inner**: tokens with no `"` character — safe inside a JSON string
- **string_close**: tokens equal to `"` — closes a JSON string
- **number_inner**: tokens consisting only of `0-9`, `.`, `-`

Each parameter is generated one at a time, with already-extracted values embedded as partial JSON context in the prompt.

## Design Decisions

- Prompt ends mid-JSON (`{"name": "`) to put the model directly into generation mode before the trie takes over.
- Vocabulary constraints are built once at startup, not per prompt.
- Function definitions (`Function`, `ParamSpec`), function call results (`FunctionCallResult`), and token constraint sets (`TokenConstraints`) are all controled by Pydantic.
- If `"` is not found in `vocab.json`, `encode('"')` is used as a fallback.
- Numeric type support is driven by `NUMERIC_TYPES` — a mapping from JSON Schema type names (`"number"`, `"integer"`, etc.) to Python types (`float`, `int`). Adding a new numeric type requires only a single entry in this dict.
- String generation stops only at `"` (closing quote). Stopping at `,` or `}` was avoided since those characters are valid inside JSON string values (e.g. SQL queries, templates).

## Performance Analysis

- **Accuracy**: 11/11 correct on the provided test set (100%)
- **JSON validity**: 100% — Pydantic validation ensures schema compliance
- **Speed**: ~2–5 minutes for 11 prompts on CPU

## Challenges Faced

- The close-quote token is not always obvious from `vocab.json`; a fallback via `encode('"')` was added.
- Without partial JSON context in the prompt, the model sometimes placed the same value in multiple parameter slots.

## Testing Strategy

- Ran the full pipeline against all 10 provided prompts and verified output manually.
- Used `make lint` to confirm type correctness and code style.
- Validated the output file with `json.load()` to confirm parseability.

## Example Usage

Input (`data/input/function_calling_tests.json`):
```json
[{"prompt": "What is the sum of 2 and 3?"}]
```

Output (`data/output/function_calling_results.json`):
```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2, "b": 3}
    }
]
```

## Resources
```
ArgumentParserの使い方を簡単にまとめた
https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0

uv完全ガイド：pip・poetry・pyenvをすべて置き換えるPythonパッケージマネージャー
https://zenn.dev/long910/articles/2026-03-13-uv-python

mypy 設定ファイル pyproject.tomlが便利
https://note.com/shirotabistudy/n/n22457b1e4946

PythonのLiteral型について調べてみた（PEP586）
https://qiita.com/simonritchie/items/04832ab86c962b3e51cd

Pydanticで始めるPythonのバリデーションとシリアライゼーション
https://zenn.dev/taka256/articles/c7213c359dd2cf

Pythonでファイルの読み込み、書き込み（作成・追記）
https://note.nkmk.me/python-file-io-open-with/

LLM Function Calling完全ガイド：Tool Useパターンからプロダクション設計まで
https://www.youngju.dev/blog/llm/2026-03-03-llm-function-calling-tool-use-guide

LLM構造化出力(Structured Output)完全ガイド2026
https://renue.co.jp/posts/llm-structured-output-json-mode-openai-claude-gemini-pydantic-guide-2026

OpenAIのFunctionCallingを理解する
https://zenn.dev/kazuwombat/articles/1f39f003298028

Databricksにおけるテキスト分類器のDSPyプログラムの作成
https://qiita.com/taka_yayoi/items/4c658d12e34b2c7707b5

pyTorchのTensor型とは
https://qiita.com/mathlive/items/d9f31f8538e20a102e14

パラメータ数とは / 【賢い選択】LLMのパラメータ数比較ガイド
https://e-words.jp/w/パラメータ数.html
```

## How AI used
This project was developed with the support of AI (Gemini) in the following areas:

Environment & Troubleshooting: Resolved 42-specific issues, including uv permission errors, segmentation faults in the cluster environment, and Makefile optimization.

Logic Design: Guided the implementation of robust JSON escaping and Pydantic validation to ensure the 0.6B model outputs precise data.

Code Review & Documentation: Provided continuous feedback on code structure and assisted in writing this English documentation.

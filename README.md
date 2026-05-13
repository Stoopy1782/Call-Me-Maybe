*This project has been created as part of the 42 curriculum by ykojima*

# Call-Me-Maybe

## Description 
The subject of "*Call Me Maybe*"  is Using small language models(Qwen3 -0.6B) and make accurate JSON output.

This project teaches how to use function calling, how to controll LLM outout, JSON text formatting techniques, and how to use commandline system of file. 

## Instructions

### 1. Install program
```
git clone *this program url*
```
Notes: This program takes huge memory __(over 2GB)__ and needs middle range RAM.
So, you should clone on large memory space.

### 2. Make virtual environment
```
python -m venv .venv
source .venv/bin/activate
```
Notes: For this program, please set ".venv" as venv name for uv program.

### 3. Install package
```
make install
```

This program needs     
*   **`pydantic`**: For validation
*   **`numpy`**: To control LLM output
*   **`rich`**: To format output
*   **`mypy`**: To check type-hints
*   **`torch`**, **`transformers`**, **`huggingface_hub`**: For  LLM_SDK

### 4. Running program
```
make run
```
or
```
.venv/bin/uv run python -m src
--functions_definition data/input/functions_definition.json
--input data/input/function_calling_tests.json
--output data/output/function_calls.json
```
This output shows how LLM outputted and % of each choices selection.
https://github.com/user-attachments/assets/91685ff1-5cc0-4f2a-b865-7961c1df8988
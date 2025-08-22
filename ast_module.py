"""
AI Code Assistant Pipeline (Working Version)

- Canonicalizer (Simple Cleaner)
- Intent Classifier (LLM-Based)
- AST-based Explanation (Implemented)
- Debugging (Implemented)
- Code Generation (Implemented)
"""

import os
import ast
import requests
from dotenv import load_dotenv

# ----------------------------
# Load API Keys
# ----------------------------
load_dotenv()
# NOTE: For this environment, the API key is handled automatically.
# You can leave API_KEY as an empty string.
API_KEY = os.getenv("GROQ_API_KEY", "") 

# ----------------------------
# LLM API Integration
# ----------------------------
def call_groq_llm(prompt: str, model="llama3-8b-8192"):
    """
    Calls the Groq LLM API with the given prompt.
    Handles API request and returns the model's response.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        # Using a generic API endpoint that can be swapped if needed
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return "Sorry, there was an error communicating with the language model."
    except (KeyError, IndexError) as e:
        print(f"❌ Unexpected API response format: {e}")
        return "Sorry, the response from the language model was malformed."


# ----------------------------
# Canonicalizer (Simple Cleaner)
# ----------------------------
def canonicalize_text(text: str):
    """
    Cleans and standardizes the user's input query.
    """
    return text.lower().strip()

# ----------------------------
# AST Processor (For Code Explanation)
# ----------------------------
class ASTProcessor:
    """
    Handles parsing, processing, and representing code using Abstract Syntax Trees (AST).
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.source_code = self._load_source()
        self.tree = ast.parse(self.source_code)

    def _load_source(self):
        """Loads source code from the specified file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    def prune_ast(self, node):
        """
        Recursively prunes unnecessary nodes from the AST to reduce complexity.
        Removes nodes like constants, load/store contexts, etc.
        """
        if isinstance(node, (ast.Constant, ast.Load, ast.Store, ast.Pass, ast.Expr)):
            return None
        if isinstance(node, ast.AST):
            pruned_fields = {}
            for field in node._fields:
                value = getattr(node, field)
                if isinstance(value, (list, ast.AST)):
                    pruned_fields[field] = self.prune_ast(value)
                else:
                    pruned_fields[field] = value
            try:
                return type(node)(**pruned_fields)
            except Exception:
                return None
        elif isinstance(node, list):
            return [self.prune_ast(n) for n in node if self.prune_ast(n)]
        return node

    def linearize_ast(self, node):
        """
        Converts a pruned AST into a linearized, string-based representation.
        This format is easier for an LLM to process than a complex tree object.
        """
        if isinstance(node, ast.AST):
            children = [self.linearize_ast(getattr(node, f)) for f in node._fields]
            children = [c for c in children if c]
            return f"({node.__class__.__name__} {' '.join(children)})"
        elif isinstance(node, list):
            return ' '.join(self.linearize_ast(n) for n in node if n)
        return str(node) if node else ''

    def extract_functions(self):
        """Extracts all function definition nodes from the AST."""
        return [node for node in ast.walk(self.tree) if isinstance(node, ast.FunctionDef)]

    def summarize_function(self, func_node):
        """
        Generates a brief summary of a function, using its docstring if available.
        """
        doc = ast.get_docstring(func_node)
        if doc:
            return doc.strip()
        args = [a.arg for a in func_node.args.args]
        return f"The function '{func_node.name}' is defined with arguments: {', '.join(args) if args else 'none'}."

    def construct_prompt(self, func_node, user_query):
        """
        Constructs a detailed prompt for the LLM to explain a function.
        Includes a summary and the linearized AST.
        """
        linear = self.linearize_ast(self.prune_ast(func_node))
        summary = self.summarize_function(func_node)
        return f"""You are an expert software assistant. The user asked: "{user_query}"

Here is the context for the function '{func_node.name}':

Function Summary:
{summary}

Linearized AST (a structural representation of the code):
{linear}

Based on this information, please explain the function's purpose and logic clearly and concisely.
"""

# ----------------------------
# Task Handlers
# ----------------------------
def debug_code(code_snippet: str, task_description: str):
    """
    Sends a buggy code snippet and a description of the issue to the LLM for a fix.
    """
    prompt = f"""You are an expert Python debugger. A user has reported an issue:
"{task_description}"

Here is the buggy code snippet:
```python
{code_snippet}
```

Please find and fix the issue. Return only the corrected Python code, followed by a brief, clear explanation of the fix.
Format your response like this:
```python
# Corrected code here
```
**Explanation:**
- Your explanation here.
"""
    return call_groq_llm(prompt)

def generate_code(task_description: str):
    """
    Asks the LLM to generate a Python program based on a task description.
    """
    prompt = f"""Write a clean, well-commented Python program for the following task: "{task_description}".
Return only the Python code inside a single markdown block. No extra text or explanation before or after the code block.
"""
    return call_groq_llm(prompt)

# ----------------------------
# Entity + Intent Classifier (LLM-Based)
# ----------------------------
def parse_user_query(user_query: str):
    """
    Uses an LLM to classify the user's intent (explain, generate, debug) and extract key entities.
    """
    system_prompt = """You are a smart assistant that parses user questions about code. Your job is to extract the user's intent and any relevant entities from their query.

From each question, extract:
- `intent_type`: The user's goal. Must be one of: `explain`, `generate`, `debug`.
- `entity_type`: The type of code element. Must be one of: `function`, `class`, `file`, or `None`.
- `entity_name`: The specific name of the function or class, if mentioned.
- `file_hint`: The name of the file, if mentioned.

Return ONLY a single, valid Python dictionary. Do not add any explanation, prefix, or surrounding text.

Examples:
Q: "Explain the `login_user` function in `app.py`"
→ {"intent_type": "explain", "entity_type": "function", "entity_name": "login_user", "file_hint": "app.py"}

Q: "My division function is throwing a ZeroDivisionError, can you fix it?"
→ {"intent_type": "debug", "entity_type": "function", "entity_name": "division", "file_hint": None}

Q: "Generate a Python script that sorts a list of tuples by the second element."
→ {"intent_type": "generate", "entity_type": None, "entity_name": None, "file_hint": None}

Q: "What does the `User` class do?"
→ {"intent_type": "explain", "entity_type": "class", "entity_name": "User", "file_hint": None}

Now parse this and return only the dictionary:
"""
    full_prompt = system_prompt + f'\nQ: "{user_query}"\n→'
    try:
        response_str = call_groq_llm(full_prompt)
        # Use ast.literal_eval for safe evaluation of the dictionary string
        return ast.literal_eval(response_str.strip())
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"❌ Parsing error: {e}")
        # Fallback to a default intent if parsing fails
        return {"intent_type": "generate", "entity_type": None, "entity_name": None, "file_hint": None}


# ----------------------------
# File Resolver (for explanation task)
# ----------------------------
def resolve_file_path(file_hint: str, search_dir: str):
    """
    Finds the full path of a file within a given directory based on a hint.
    """
    if not file_hint:
        return None
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file_hint.lower() in file.lower():
                return os.path.join(root, file)
    return None

# ----------------------------
# Main Orchestrator
# ----------------------------
def handle_user_query(query: str, repo_path: str):
    """
    Main function to orchestrate the AI code assistant pipeline.
    It parses the query, determines the intent, and calls the appropriate handler.
    """
    cleaned_query = canonicalize_text(query)
    parsed = parse_user_query(cleaned_query)
    print(f" Parsed Intent: {parsed}")

    intent_type = parsed.get("intent_type")
    entity_name = parsed.get("entity_name")
    file_hint = parsed.get("file_hint")
    entity_type = parsed.get("entity_type")

    # 🔹 If file_hint is missing but entity_type is file, use entity_name as hint
    if not file_hint and entity_type == "file" and entity_name:
        file_hint = entity_name

    if intent_type == "generate":
        print("\n======== CODE GENERATION  ========")
        generated_code = generate_code(cleaned_query)
        print(generated_code)

    elif intent_type == "debug":
        print("\n======== DEBUGGING ========")
        print("Please paste the buggy code snippet, followed by '###', and then a description of the issue.")
        combined_input = input("> ")
        if "###" not in combined_input:
            print("❌ Invalid format. Please separate the code and issue description with '###'.")
            return
        code_snippet, issue_description = combined_input.split("###", 1)
        debug_output = debug_code(code_snippet.strip(), issue_description.strip())
        print("\n--- DEBUG OUTPUT ---")
        print(debug_output)

    elif intent_type == "explain":
        print("\n========  CODE EXPLANATION  ========")
        file_path = resolve_file_path(file_hint, search_dir=repo_path)
        if not file_path:
            print(f"❌ Could not resolve file for hint '{file_hint}' in '{repo_path}'.")
            return

        print(f"Found file: {file_path}")
        processor = ASTProcessor(file_path)

        # UPDATED LOGIC: Handle whole-file explanation separately
        if entity_type == "file":
            print(f"\n--- Explaining the entire file '{os.path.basename(file_path)}' ---")
            # Construct a prompt to explain the whole file's source code
            prompt = f"""You are an expert software assistant. The user asked: "{query}"

Please provide a high-level explanation of the following Python script. Describe its purpose, what libraries it uses, and how the main components work together.

File: `{os.path.basename(file_path)}`
```python
{processor.source_code}
```
"""
            explanation = call_groq_llm(prompt)
            print(explanation)
            return # Exit after explaining the file

        # Handle function-specific explanation
        functions = processor.extract_functions()
        if not functions:
             print(f"ℹ️  No functions found in '{os.path.basename(file_path)}' to explain.")
             return

        if not entity_name:
            # Explain all functions if no specific function was given
            for func in functions:
                prompt = processor.construct_prompt(func, query)
                print(f"\n--- Explaining function '{func.name}' ---")
                explanation = call_groq_llm(prompt)
                print(explanation)
        else:
            # Find and explain a specific function
            target_function = None
            for func in functions:
                if entity_name.lower() in func.name.lower():
                    target_function = func
                    break
            
            if target_function:
                prompt = processor.construct_prompt(target_function, query)
                print("\n--- Sending Request to LLM ---")
                explanation = call_groq_llm(prompt)
                print("\n--- LLM RESPONSE ---")
                print(explanation)
            else:
                print(f"❌ Function '{entity_name}' not found in '{os.path.basename(file_path)}'.")
    else:
        print("Could not determine a valid intent from your query. Please try rephrasing.")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    # Make sure to set the repo_path to the directory containing your code.
    repo_path = "."  # <<< Defaults to the current directory. Update if needed.
    
    if not os.path.isdir(repo_path):
        print(f"❌ Error: The specified repository path does not exist: {repo_path}")
        print("Please update the 'repo_path' variable in the script to point to your code directory.")
    else:
        user_query = input("Enter your query (e.g., 'explain my function', 'generate a script', 'debug my code'):\n> ")
        handle_user_query(user_query, repo_path)

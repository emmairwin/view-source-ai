#!/usr/bin/env python3
"""
Quick structural test without full dependencies
"""
import ast
import sys

def test_file(filepath, expected_items):
    """Test that a file has expected functions/classes"""
    print(f"\n✓ Testing {filepath}...")
    with open(filepath) as f:
        tree = ast.parse(f.read())
    
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
    
    for item in expected_items:
        if item in defined:
            print(f"  ✓ Found {item}")
        else:
            print(f"  ✗ Missing {item}")
            sys.exit(1)

# Test app.py
test_file("app.py", [
    "build_ui",
    "OlmoBackend",
    "analyze_comments",
    "format_results_markdown",
    "fetch_github_comments",
    "parse_repo_input",
    "generate_issue_body",
])

# Test data_provenance.py
test_file("data_provenance.py", [
    "ExampleBank",
    "search_dolma_wimbd",
    "build_provenance_report",
    "initialize_example_bank",
])

print("\n" + "="*60)
print("✅ All structural tests passed!")
print("="*60)
print("\nNext steps:")
print("1. Install dependencies: source venv/bin/activate && pip install -r requirements.txt")
print("2. Run with Ollama: ollama pull olmo2:1b && python app.py")
print("3. Or run with transformers: OLMO_BACKEND=transformers python app.py")

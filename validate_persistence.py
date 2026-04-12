#!/usr/bin/env python3
import ast
import sys

python_files = [
    r"C:\Users\KMJ\Desktop\Image viewer\ui\vue_bridge.py",
    r"C:\Users\KMJ\Desktop\Image viewer\ui\generator_main.py",
    r"C:\Users\KMJ\Desktop\Image viewer\ui\generator_settings.py"
]

print("=" * 60)
print("PYTHON SYNTAX VALIDATION - PERSISTENCE CHANGES")
print("=" * 60)

all_ok = True
for filepath in python_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        filename = filepath.split('\\')[-1]
        print(f"✓ {filename}: OK")
    except SyntaxError as e:
        all_ok = False
        filename = filepath.split('\\')[-1]
        print(f"✗ {filename}: SYNTAX ERROR")
        print(f"  Line {e.lineno}: {e.msg}")
        if e.text:
            print(f"  {e.text.strip()}")
    except Exception as e:
        all_ok = False
        filename = filepath.split('\\')[-1]
        print(f"✗ {filename}: ERROR - {type(e).__name__}: {e}")

print("=" * 60)
if all_ok:
    print("Result: All Python files passed syntax validation")
    sys.exit(0)
else:
    print("Result: Some Python files have syntax errors")
    sys.exit(1)

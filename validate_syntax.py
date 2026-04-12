#!/usr/bin/env python3
import ast
import sys

files = [
    r"C:\Users\KMJ\Desktop\Image viewer\ui\generator_prompts.py",
    r"C:\Users\KMJ\Desktop\Image viewer\ui\generator_ui_setup.py"
]

for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"{filepath.split(chr(92))[-1]}: OK")
    except SyntaxError as e:
        print(f"{filepath.split(chr(92))[-1]}: ERROR - {e}")
    except Exception as e:
        print(f"{filepath.split(chr(92))[-1]}: ERROR - {type(e).__name__}: {e}")

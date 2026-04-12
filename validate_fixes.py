#!/usr/bin/env python3
"""Validate applied fixes across 6 target files"""
import ast
import os
import re

base_dir = r"C:\Users\KMJ\Desktop\Image viewer"

files = [
    os.path.join(base_dir, "ui", "generator_main.py"),
    os.path.join(base_dir, "tabs", "i2i_tab.py"),
    os.path.join(base_dir, "tabs", "inpaint_tab.py"),
    os.path.join(base_dir, "tabs", "editor_tab.py"),
    os.path.join(base_dir, "utils", "prompt_preset.py"),
    os.path.join(base_dir, "utils", "character_presets.py"),
]

print("="*70)
print("VALIDATION REPORT: Applied Fixes")
print("="*70)

results = {
    'syntax': {},
    'open_url': None,
    'worker_guards': {},
    'preset_errors': {}
}

# 1. SYNTAX CHECK
print("\n[1] PYTHON SYNTAX VALIDATION")
print("-"*70)
all_syntax_ok = True
for filepath in files:
    fname = os.path.basename(filepath)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        results['syntax'][fname] = 'PASS'
        print(f"  ✓ {fname:30} PASS")
    except SyntaxError as e:
        results['syntax'][fname] = f'FAIL: {e}'
        all_syntax_ok = False
        print(f"  ✗ {fname:30} FAIL: {e}")
    except Exception as e:
        results['syntax'][fname] = f'ERROR: {e}'
        all_syntax_ok = False
        print(f"  ✗ {fname:30} ERROR: {e}")

# 2. OPEN_URL HANDLER CHECK (generator_main.py)
print("\n[2] OPEN_URL HANDLER CHECK")
print("-"*70)
gen_main_file = os.path.join(base_dir, "ui", "generator_main.py")
open_url_found = False
with open(gen_main_file, 'r', encoding='utf-8') as f:
    content = f.read()
    if "elif action == 'open_url'" in content and "import webbrowser" in content:
        # Check implementation exists
        if "webbrowser.open(url)" in content:
            results['open_url'] = 'PASS'
            open_url_found = True
            print("  ✓ open_url handler EXISTS with webbrowser.open()")
        else:
            results['open_url'] = 'FAIL: handler found but implementation missing'
            print("  ✗ FAIL: handler found but webbrowser.open() not implemented")
    else:
        results['open_url'] = 'FAIL'
        print("  ✗ FAIL: open_url handler NOT FOUND")

# 3. WORKER GUARDS CHECK (three tabs)
print("\n[3] WORKER GUARDS CHECK (isRunning + hasattr)")
print("-"*70)
tabs_to_check = {
    'i2i_tab.py': ('gen_worker', os.path.join(base_dir, "tabs", "i2i_tab.py")),
    'inpaint_tab.py': ('gen_worker', os.path.join(base_dir, "tabs", "inpaint_tab.py")),
    'editor_tab.py': ('_inpaint_worker', os.path.join(base_dir, "tabs", "editor_tab.py")),
}

for tab_name, (worker_var, filepath) in tabs_to_check.items():
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Check for guard pattern: if hasattr(...) and worker and worker.isRunning()
    pattern = rf"if\s+hasattr\(self,\s+['\"]?{worker_var}['\"]?\)\s+and\s+self\.{worker_var}\s+and\s+self\.{worker_var}\.isRunning\(\)"
    if re.search(pattern, content):
        results['worker_guards'][tab_name] = 'PASS'
        print(f"  ✓ {tab_name:20} PASS: guard exists for {worker_var}")
    else:
        results['worker_guards'][tab_name] = 'FAIL'
        print(f"  ✗ {tab_name:20} FAIL: guard missing for {worker_var}")

# 4. PRESET SAVE ERROR HANDLING CHECK
print("\n[4] PRESET SAVE ERROR HANDLING")
print("-"*70)
preset_files = {
    'prompt_preset.py': os.path.join(base_dir, "utils", "prompt_preset.py"),
    'character_presets.py': os.path.join(base_dir, "utils", "character_presets.py"),
}

for fname, filepath in preset_files.items():
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Check for try-except in save functions with raise
    has_error_handling = False
    if 'def _save' in content:
        # Find the _save function
        match = re.search(r'def _save[^:]*:.*?(?=\ndef |\nclass |\Z)', content, re.DOTALL)
        if match:
            func_body = match.group(0)
            if 'try:' in func_body and 'except' in func_body and 'raise' in func_body:
                has_error_handling = True
    
    if has_error_handling:
        results['preset_errors'][fname] = 'PASS'
        print(f"  ✓ {fname:20} PASS: error handling with try-except-raise exists")
    else:
        results['preset_errors'][fname] = 'FAIL'
        print(f"  ✗ {fname:20} FAIL: error handling not properly implemented")

# SUMMARY
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

all_pass = (
    all_syntax_ok and
    results['open_url'] == 'PASS' and
    all(v == 'PASS' for v in results['worker_guards'].values()) and
    all(v == 'PASS' for v in results['preset_errors'].values())
)

print(f"\nSyntax Validation:        {'PASS' if all_syntax_ok else 'FAIL'}")
print(f"open_url handler:         {results['open_url']}")
print(f"Worker guards (i2i):      {results['worker_guards'].get('i2i_tab.py', 'N/A')}")
print(f"Worker guards (inpaint):  {results['worker_guards'].get('inpaint_tab.py', 'N/A')}")
print(f"Worker guards (editor):   {results['worker_guards'].get('editor_tab.py', 'N/A')}")
print(f"Preset error handling:    {results['preset_errors'].get('prompt_preset.py', 'N/A')}, {results['preset_errors'].get('character_presets.py', 'N/A')}")

print("\n" + "="*70)
if all_pass:
    print("OVERALL RESULT: ✓ PASS - All fixes validated successfully")
else:
    print("OVERALL RESULT: ✗ FAIL - Blocking issues found (see above)")
print("="*70)

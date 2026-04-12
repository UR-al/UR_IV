#!/usr/bin/env python3
"""
Validate persistence patch for ui_prefs merge-save and restore logic
"""
import ast
import json
import os
import sys
import re

# Color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

results = {
    'python_syntax': {'pass': False, 'errors': []},
    'vue_syntax': {'pass': False, 'errors': []},
    'logic': {'pass': False, 'errors': []},
}

# ==================== PYTHON VALIDATION ====================
python_files = [
    r"C:\Users\KMJ\Desktop\Image viewer\ui\generator_main.py",
]

print(f"{BOLD}[PYTHON SYNTAX CHECK]{RESET}")
for filepath in python_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"✓ {os.path.basename(filepath)}: PASS")
        results['python_syntax']['pass'] = True
    except SyntaxError as e:
        print(f"✗ {os.path.basename(filepath)}: {RED}SYNTAX ERROR{RESET}")
        results['python_syntax']['errors'].append(f"Line {e.lineno}: {e.msg}")
        print(f"  Line {e.lineno}: {e.msg}")
    except Exception as e:
        print(f"✗ {os.path.basename(filepath)}: {RED}ERROR{RESET}")
        results['python_syntax']['errors'].append(str(e))

# ==================== VUE SYNTAX CHECK ====================
vue_files = [
    r"C:\Users\KMJ\Desktop\Image viewer\frontend\src\App.vue",
    r"C:\Users\KMJ\Desktop\Image viewer\frontend\src\views\SearchView.vue",
    r"C:\Users\KMJ\Desktop\Image viewer\frontend\src\views\SettingsView.vue",
]

print(f"\n{BOLD}[VUE SYNTAX CHECK]{RESET}")
vue_pass = True
for filepath in vue_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for matching script tags
        if '<script' not in content or '</script>' not in content:
            results['vue_syntax']['errors'].append(f"{os.path.basename(filepath)}: Missing script block")
            vue_pass = False
            print(f"✗ {os.path.basename(filepath)}: Missing script block")
            continue
        
        # Extract script block
        script_match = re.search(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
        if not script_match:
            results['vue_syntax']['errors'].append(f"{os.path.basename(filepath)}: Could not extract script")
            vue_pass = False
            print(f"✗ {os.path.basename(filepath)}: Could not extract script")
            continue
        
        script_content = script_match.group(1)
        
        # Basic JS validation (check for obvious mismatches)
        brace_count = script_content.count('{') - script_content.count('}')
        paren_count = script_content.count('(') - script_content.count(')')
        bracket_count = script_content.count('[') - script_content.count(']')
        
        if brace_count != 0 or paren_count != 0 or bracket_count != 0:
            results['vue_syntax']['errors'].append(f"{os.path.basename(filepath)}: Bracket mismatch (braces:{brace_count}, parens:{paren_count}, brackets:{bracket_count})")
            vue_pass = False
            print(f"✗ {os.path.basename(filepath)}: {RED}BRACKET MISMATCH{RESET}")
            print(f"  Braces: {brace_count}, Parentheses: {paren_count}, Brackets: {bracket_count}")
        else:
            print(f"✓ {os.path.basename(filepath)}: PASS")
            results['vue_syntax']['pass'] = True
    except Exception as e:
        results['vue_syntax']['errors'].append(f"{os.path.basename(filepath)}: {str(e)}")
        print(f"✗ {os.path.basename(filepath)}: {RED}ERROR{RESET} - {str(e)}")

# ==================== LOGIC VALIDATION ====================
print(f"\n{BOLD}[PERSISTENCE LOGIC CHECK]{RESET}")
logic_errors = []

# Check Python: save_ui_prefs handler
print("\nPython Backend (generator_main.py):")
with open(python_files[0], 'r', encoding='utf-8') as f:
    py_content = f.read()

# Check for save_ui_prefs handler
if "elif action == 'save_ui_prefs':" in py_content:
    print("  ✓ save_ui_prefs handler exists")
    # Check merge-save logic
    if "prefs.update(payload)" in py_content:
        print("  ✓ Uses merge-save (prefs.update)")
    else:
        logic_errors.append("save_ui_prefs: Does not use merge pattern")
        print("  ✗ save_ui_prefs: Missing merge pattern")
    
    # Check re-emit after save
    if "self.vue_bridge.uiPrefsLoaded.emit(json.dumps(prefs))" in py_content:
        print("  ✓ Re-emits uiPrefsLoaded after save")
    else:
        logic_errors.append("save_ui_prefs: Does not re-emit uiPrefsLoaded")
        print("  ✗ save_ui_prefs: No re-emit of uiPrefsLoaded")
else:
    logic_errors.append("save_ui_prefs handler not found")
    print("  ✗ save_ui_prefs handler NOT FOUND")

# Check for initial load in _load_saved_configs
if "_load_saved_configs" in py_content:
    print("  ✓ _load_saved_configs method exists")
    if "ui_prefs.json" in py_content and "uiPrefsLoaded.emit" in py_content:
        print("  ✓ Emits uiPrefsLoaded on app startup")
    else:
        logic_errors.append("_load_saved_configs: Does not emit uiPrefsLoaded")
        print("  ✗ _load_saved_configs: No uiPrefsLoaded emit")
else:
    logic_errors.append("_load_saved_configs not found")
    print("  ✗ _load_saved_configs method NOT FOUND")

# Check Vue: uiPrefsLoaded restore logic
print("\nVue Frontend (App.vue, SearchView.vue, SettingsView.vue):")

vue_restore_errors = []
for filepath in vue_files:
    basename = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        vue_content = f.read()
    
    if "onBackendEvent('uiPrefsLoaded'" in vue_content:
        print(f"  ✓ {basename}: Has uiPrefsLoaded listener")
        
        # Check for proper JSON parse
        if "JSON.parse(json)" in vue_content:
            print(f"    ✓ Parses JSON correctly")
        else:
            vue_restore_errors.append(f"{basename}: Missing JSON.parse")
            print(f"    ✗ Missing JSON.parse")
        
        # Check for state merge/restore
        if ".value =" in vue_content or "Object.assign" in vue_content:
            print(f"    ✓ Restores values to component state")
        else:
            vue_restore_errors.append(f"{basename}: No value restoration")
            print(f"    ✗ No value restoration pattern")
    else:
        vue_restore_errors.append(f"{basename}: Missing uiPrefsLoaded listener")
        print(f"  ✗ {basename}: NO uiPrefsLoaded listener")

if vue_restore_errors:
    logic_errors.extend(vue_restore_errors)

# Check for requestAction('save_ui_prefs' calls
print("\nVue Frontend (save calls):")
for filepath in vue_files:
    basename = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        vue_content = f.read()
    
    if "requestAction('save_ui_prefs'" in vue_content:
        print(f"  ✓ {basename}: Calls save_ui_prefs")
    else:
        print(f"  - {basename}: No save_ui_prefs calls (may be expected)")

if logic_errors:
    results['logic']['pass'] = False
    results['logic']['errors'] = logic_errors
else:
    results['logic']['pass'] = True
    print("\n  ✓ All logic checks passed")

# ==================== SUMMARY ====================
print(f"\n{BOLD}{'='*60}")
print("VALIDATION SUMMARY")
print('='*60 + f"{RESET}")

all_pass = all(r['pass'] for r in results.values())

for category, data in results.items():
    status = f"{GREEN}PASS{RESET}" if data['pass'] else f"{RED}FAIL{RESET}"
    print(f"{category.upper():.<40} {status}")
    if data['errors']:
        for err in data['errors']:
            print(f"  └─ {YELLOW}• {err}{RESET}")

print(f"\n{BOLD}{'='*60}{RESET}")
if all_pass:
    print(f"{GREEN}{BOLD}✓ ALL VALIDATION CHECKS PASSED{RESET}")
    sys.exit(0)
else:
    print(f"{RED}{BOLD}✗ VALIDATION FAILED - See errors above{RESET}")
    sys.exit(1)

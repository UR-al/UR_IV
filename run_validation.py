import ast
import sys

# Python files to validate
python_files = [
    r"C:\Users\KMJ\Desktop\Image viewer\ui\vue_bridge.py",
    r"C:\Users\KMJ\Desktop\Image viewer\ui\generator_main.py",
    r"C:\Users\KMJ\Desktop\Image viewer\ui\generator_settings.py"
]

print("=" * 70)
print("PYTHON SYNTAX VALIDATION - PERSISTENCE FILES CHECK")
print("=" * 70)

all_ok = True
for filepath in python_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        filename = filepath.split('\\')[-1]
        print(f"✅ PASS: {filename}")
    except SyntaxError as e:
        all_ok = False
        filename = filepath.split('\\')[-1]
        print(f"❌ FAIL: {filename}")
        print(f"   Line {e.lineno}: {e.msg}")
        if e.text:
            print(f"   Code: {e.text.strip()}")
    except Exception as e:
        all_ok = False
        filename = filepath.split('\\')[-1]
        print(f"❌ ERROR: {filename}: {type(e).__name__}: {e}")

print("=" * 70)
print("\nVUE FILE STRUCTURE CHECK")
print("=" * 70)

vue_files = [
    r"C:\Users\KMJ\Desktop\Image viewer\frontend\src\App.vue",
    r"C:\Users\KMJ\Desktop\Image viewer\frontend\src\views\SearchView.vue",
    r"C:\Users\KMJ\Desktop\Image viewer\frontend\src\views\SettingsView.vue",
]

for filepath in vue_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for basic structure
        has_template = '<template>' in content or '<template ' in content
        has_script = '<script>' in content or '<script ' in content
        has_closing_template = '</template>' in content
        has_closing_script = '</script>' in content
        
        filename = filepath.split('\\')[-1]
        
        if has_template and has_closing_template:
            print(f"✅ PASS: {filename}")
        else:
            all_ok = False
            print(f"❌ FAIL: {filename} - Invalid template structure")
            
    except Exception as e:
        all_ok = False
        filename = filepath.split('\\')[-1]
        print(f"❌ ERROR: {filename}: {e}")

print("=" * 70)
if all_ok:
    print("RESULT: ✅ All files passed validation")
    sys.exit(0)
else:
    print("RESULT: ❌ Some files have validation issues")
    sys.exit(1)

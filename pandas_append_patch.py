#!/usr/bin/env python3
"""
Patch for pandas DataFrame.append deprecation in Moses/molsets utils.py.
This script replaces deprecated `.append` calls with `pd.concat` in the correct utils.py file.
"""
import os
import sys
import re

def patch_utils():
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    site_packages = os.path.join(venv_path, 'lib', 'python3.10', 'site-packages')
    utils_path = None
    for root, dirs, files in os.walk(site_packages):
        if 'molsets-1.0-py3.10.egg' in root and os.path.exists(os.path.join(root, 'moses', 'metrics', 'utils.py')):
            utils_path = os.path.join(root, 'moses', 'metrics', 'utils.py')
            break
    if not utils_path:
        print("Could not find utils.py in the virtual environment.")
        return False
    print(f"Found utils.py at: {utils_path}")
    with open(utils_path, 'r') as f:
        content = f.read()
    # Replace DataFrame.append with pd.concat
    # Pattern: df.append(other, sort=True) or df.append(other)
    # Replacement: pd.concat([df, other], sort=True) or pd.concat([df, other])
    def append_repl(match):
        df = match.group(1)
        arg = match.group(2)
        sort_kw = match.group(3)
        if sort_kw:
            return f"pd.concat([{df}, {arg}], sort=True)"
        else:
            return f"pd.concat([{df}, {arg}])"
    new_content, n = re.subn(r"(\w+)\.append\(([^,)]+)(, *sort *= *True)?\)", append_repl, content)
    if n == 0:
        print("No .append usage found in utils.py.")
        return False
    # Ensure import pandas as pd exists
    if 'import pandas as pd' not in new_content:
        new_content = 'import pandas as pd\n' + new_content
    # Backup
    backup_path = utils_path + '.bak'
    print(f"Creating backup at: {backup_path}")
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"Patching {utils_path}")
    with open(utils_path, 'w') as f:
        f.write(new_content)
    print("Successfully patched utils.py to use pd.concat instead of .append.")
    return True

if __name__ == "__main__":
    success = patch_utils()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Patch for the rdkit.six import error in Moses.
This script modifies the import statement in sascorer.py to use the standalone six module
instead of rdkit.six which is no longer available in newer RDKit versions.
"""

import os
import sys
import re

def patch_sascorer():
    # Find the sascorer.py file in the virtual environment
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    site_packages = os.path.join(venv_path, 'lib', 'python3.10', 'site-packages')
    
    # Look for the molsets egg directory
    for root, dirs, files in os.walk(site_packages):
        if 'molsets-1.0-py3.10.egg' in root and os.path.exists(os.path.join(root, 'moses', 'metrics', 'SA_Score', 'sascorer.py')):
            sascorer_path = os.path.join(root, 'moses', 'metrics', 'SA_Score', 'sascorer.py')
            break
    else:
        print("Could not find sascorer.py in the virtual environment.")
        return False
    
    print(f"Found sascorer.py at: {sascorer_path}")
    
    # Read the file content
    with open(sascorer_path, 'r') as f:
        content = f.read()
    
    # Replace the import statement
    if 'from rdkit.six import iteritems' in content:
        new_content = content.replace('from rdkit.six import iteritems', 'from six import iteritems')
        
        # Create a backup
        backup_path = sascorer_path + '.bak'
        print(f"Creating backup at: {backup_path}")
        with open(backup_path, 'w') as f:
            f.write(content)
        
        # Write the modified content
        print(f"Patching {sascorer_path}")
        with open(sascorer_path, 'w') as f:
            f.write(new_content)
        
        print("Successfully patched sascorer.py to use the standalone six module.")
        return True
    else:
        print("The import statement 'from rdkit.six import iteritems' was not found in sascorer.py.")
        return False

if __name__ == "__main__":
    success = patch_sascorer()
    sys.exit(0 if success else 1)

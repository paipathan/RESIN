import json 
import subprocess
import os
import sys

python_executable = sys.executable

result = subprocess.run(
    [python_executable, 'scan.py'],
    capture_output=True,
    text=True
)

output = result.stdout.strip()

print(output)
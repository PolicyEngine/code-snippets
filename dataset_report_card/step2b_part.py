"""Compatibility wrapper around step_part.py for Dataset B."""

import subprocess
import sys

subprocess.run([sys.executable, "step_part.py", "b", *sys.argv[1:]], check=True)

import subprocess
import sys


def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, f"scripts/{script_name}"], check=True)
    print(f"{script_name} completed with return code {result.returncode}")


scripts = ["index.py", "scrape.py", "preprocess.py", "train.py", "serve.py"]

for script in scripts:
    run_script(script)

print("All scripts have been executed successfully.")


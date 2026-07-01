"""End-to-end demonstration: generate data, run the pipeline, show provenance."""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run([sys.executable, "data/synthetic/generate_synthetic.py",
                "--entities", "100", "--seed", "42"])
subprocess.run([sys.executable, "-m", "src.pipeline"])

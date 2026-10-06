import subprocess
import sys

def run_step(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: Step '{cmd}' failed with return code {result.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    print("=== Starting Reproducibility Pipeline ===")
    
    # Task 1 A: Verify Claims
    run_step("python experiments/verify_claims.py")
    
    # Task 1 B: Run B1-B11 Experiments via run_queue
    b_scripts = [
        "experiments/b1_ferrag_replication.py",
        "experiments/b2_fidelity_zero_vs_retrain.py",
        "experiments/b3_munilla_style.py",
        "experiments/b4_unsw_audit.py",
        "experiments/b5_fidelity_methods.py",
        "experiments/b6_shap_stability.py",
        "experiments/b7_lime_vs_shap.py",
        "experiments/b8_perclass_mean_sd.py",
        "experiments/b9_event_level.py",
        "experiments/b10_pareto.py",
        "experiments/b11_adversarial.py"
    ]
    
    for script in b_scripts:
        run_step(f"python {script}")
        
    # Generate Figures
    run_step("python make_figures.py")
    
    # Build LaTeX Numbers
    run_step("python paper/build_numbers.py")
    
    print("=== Pipeline Complete ===")

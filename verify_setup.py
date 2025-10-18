"""
Windows Environment Verification Script
Checks if your system is ready for improved training
"""

import sys
import os
from pathlib import Path
import importlib.util
import platform

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(text)
    print("="*70)

def print_section(text):
    """Print a section header"""
    print(f"\n{text}")
    print("-"*70)

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    if version.major >= 3 and version.minor >= 8:
        print("  ✓ Python version OK (3.8+)")
        return True
    else:
        print("  ✗ Python version too old (need 3.8+)")
        return False

def check_package(package_name):
    """Check if package is installed"""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {package_name}: {version}")
            return True
        except Exception as e:
            print(f"  ⚠ {package_name}: installed but couldn't load - {e}")
            return False
    else:
        print(f"  ✗ {package_name}: NOT INSTALLED")
        print(f"     Install with: pip install {package_name}")
        return False

def check_pytorch_cuda():
    """Check PyTorch and CUDA availability"""
    try:
        import torch
        print(f"\n  PyTorch Details:")
        print(f"    Version: {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"    Running on: CPU (this is fine for this project!)")
        return True
    except:
        return False

def check_directory_structure():
    """Check if directory structure is correct"""
    required_dirs = ['src', 'training', 'models', 'examples']
    required_files = [
        'src/__init__.py',
        'src/neural_component.py',
        'src/symbolic_component.py',
        'src/refinement_engine.py',
        'src/utils.py'
    ]
    
    print_section("Directory Structure Check")
    cwd = Path.cwd()
    print(f"Current directory: {cwd}")
    
    all_ok = True
    
    print("\nRequired Directories:")
    for dir_name in required_dirs:
        dir_path = cwd / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}\\")
        else:
            print(f"  ✗ {dir_name}\\ MISSING")
            all_ok = False
    
    print("\nRequired Files:")
    for file_name in required_files:
        file_path = cwd / file_name
        file_name_win = file_name.replace('/', '\\')
        if file_path.exists():
            print(f"  ✓ {file_name_win}")
        else:
            print(f"  ✗ {file_name_win} MISSING")
            all_ok = False
    
    return all_ok

def check_model_imports():
    """Try importing the model components"""
    print_section("Model Component Import Test")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        from src.neural_component import NeuralReasoner
        print("  ✓ NeuralReasoner imported successfully")
        
        from src.symbolic_component import SymbolicVerifier
        print("  ✓ SymbolicVerifier imported successfully")
        
        from src.refinement_engine import RefinementEngine
        print("  ✓ RefinementEngine imported successfully")
        
        from src.utils import setup_logging
        print("  ✓ Utils imported successfully")
        
        # Try creating a small model
        print("\n  Testing model creation...")
        model = NeuralReasoner(16, 64, 9)
        print(f"  ✓ Created test model with {model.count_parameters():,} parameters")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        print("\n  Troubleshooting:")
        print("    1. Make sure you're in the repository root directory")
        print("    2. Check that all files exist (see above)")
        print("    3. Verify Python path is correct")
        return False

def check_disk_space():
    """Check available disk space (Windows-compatible)"""
    print_section("Disk Space Check")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path.cwd().drive + "\\")
        free_gb = free // (2**30)
        total_gb = total // (2**30)
        used_gb = used // (2**30)
        
        print(f"  Drive: {Path.cwd().drive}")
        print(f"  Total: {total_gb} GB")
        print(f"  Used: {used_gb} GB")
        print(f"  Free: {free_gb} GB")
        
        if free_gb >= 2:
            print(f"  ✓ Sufficient space (need ~1 GB for training)")
            return True
        else:
            print(f"  ⚠ Low disk space (recommend at least 2 GB free)")
            return False
    except Exception as e:
        print(f"  ⚠ Could not check disk space: {e}")
        return True

def check_existing_models():
    """Check for existing model checkpoints"""
    print_section("Existing Model Checkpoints")
    
    models_dir = Path('models/checkpoints')
    if not models_dir.exists():
        print("  ℹ No models directory yet (will be created during training)")
        return True
    
    checkpoints = list(models_dir.glob('*.pt'))
    if checkpoints:
        print(f"  Found {len(checkpoints)} existing checkpoint(s):")
        for cp in checkpoints:
            size_mb = cp.stat().st_size / (1024 * 1024)
            print(f"    • {cp.name} ({size_mb:.1f} MB)")
    else:
        print("  ℹ No existing checkpoints found")
    
    return True

def check_training_script():
    """Check if improved training script exists"""
    print_section("Training Script Check")
    
    script_path = Path('training/train_improved.py')
    if script_path.exists():
        print(f"  ✓ Found: {script_path}")
        size_kb = script_path.stat().st_size / 1024
        print(f"    Size: {size_kb:.1f} KB")
        return True
    else:
        print(f"  ✗ NOT FOUND: {script_path}")
        print("\n  You need to create this file!")
        print("  1. Copy the improved training script from the artifact")
        print("  2. Save it as: training\\train_improved.py")
        return False

def estimate_training_time():
    """Estimate training time"""
    print_section("Training Time Estimates")
    print("  Data generation: ~2-3 minutes")
    print("  Training per epoch: ~40-50 seconds")
    print("  Total for 50 epochs: ~35-40 minutes")
    print("  Final testing: ~1 minute")
    print("  " + "-"*40)
    print("  TOTAL EXPECTED TIME: ~40-45 minutes")

def show_next_steps(all_passed):
    """Show next steps"""
    print_header("NEXT STEPS")
    
    if all_passed:
        print("\n✓ Your environment is ready for improved training!")
        print("\nTo proceed:")
        print("  1. Make sure train_improved.py is in training\\ folder")
        print("  2. Run training:")
        print("     python training\\train_improved.py")
        print("  3. Wait ~40 minutes for training to complete")
        print("  4. Check for 85%+ success rate in results")
    else:
        print("\n⚠ Please fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("  • Wrong directory?")
        print("    cd C:\\Users\\hales\\NeuroSymbolic-Recursive")
        print("  • Missing PyTorch?")
        print("    pip install torch")
        print("  • Missing training script?")
        print("    Create training\\train_improved.py from artifact")

def main():
    """Run all checks"""
    print_header("Windows Environment Verification")
    print("NeuroSymbolic-Recursive System")
    
    checks = []
    
    # Check Python version
    print_section("Python Environment")
    checks.append(check_python_version())
    
    # Check required packages
    print_section("Required Packages")
    packages = ['torch', 'numpy']
    for pkg in packages:
        checks.append(check_package(pkg))
    
    # Check PyTorch details
    if importlib.util.find_spec('torch'):
        check_pytorch_cuda()
    
    # Check directory structure
    checks.append(check_directory_structure())
    
    # Check imports
    checks.append(check_model_imports())
    
    # Check disk space
    checks.append(check_disk_space())
    
    # Check existing models
    check_existing_models()
    
    # Check for training script
    has_script = check_training_script()
    
    # Estimate time
    estimate_training_time()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\nCore checks: {passed}/{total} passed")
    print(f"Training script: {'✓ Ready' if has_script else '✗ Missing'}")
    
    all_passed = (passed == total) and has_script
    
    if all_passed:
        print("\n✅ ALL SYSTEMS GO!")
    else:
        print(f"\n⚠ {total - passed + (0 if has_script else 1)} issue(s) need attention")
    
    show_next_steps(all_passed)
    
    print("\n" + "="*70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
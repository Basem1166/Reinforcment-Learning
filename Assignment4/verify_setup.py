"""
Quick setup verification script for Assignment 4
Run this to verify your environment is set up correctly
"""

import sys

def check_imports():
    """Check if all required packages are installed"""
    print("Checking required packages...")
    
    required_packages = {
        'gymnasium': 'gymnasium',
        'torch': 'torch',
        'numpy': 'numpy',
        'wandb': 'wandb',
        'matplotlib': 'matplotlib',
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"✗ {package_name} is NOT installed")
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\n❌ Missing packages detected!")
        print("Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All required packages are installed!")
    return True


def check_box2d():
    """Check if Box2D environments are available"""
    print("\nChecking Box2D environments...")
    
    try:
        import gymnasium as gym
        
        # Try to create LunarLander
        try:
            env = gym.make('LunarLander-v3', continuous=True)
            env.close()
            print("✓ LunarLander-v3 is available")
        except Exception as e:
            print(f"✗ LunarLander-v3 failed: {e}")
            return False
        
        # Try to create CarRacing
        try:
            env = gym.make('CarRacing-v3', continuous=True)
            env.close()
            print("✓ CarRacing-v3 is available")
        except Exception as e:
            print(f"✗ CarRacing-v3 failed: {e}")
            return False
        
        print("✅ Box2D environments are working!")
        return True
        
    except Exception as e:
        print(f"❌ Box2D check failed: {e}")
        print("Install Box2D with: pip install gymnasium[box2d]")
        return False


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA/GPU support...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA is available! Using GPU: {device_name}")
            return True
        else:
            print("⚠️  CUDA not available. Training will use CPU (slower)")
            return False
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False


def check_files():
    """Check if all necessary files exist"""
    print("\nChecking project files...")
    
    required_files = [
        'models.py',
        'buffer.py',
        'sac_agent.py',
        'ppo_agent.py',
        'td3_agent.py',
        'config.py',
        'train_sac.py',
        'train_ppo.py',
        'train_td3.py',
        'test_and_record.py',
    ]
    
    import os
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} is missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All project files are present!")
    return True


def run_quick_test():
    """Run a quick training test"""
    print("\nRunning quick training test (10 steps)...")
    
    try:
        import gymnasium as gym
        import torch
        from sac_agent import SACAgent
        from buffer import ReplayBuffer
        import config
        
        # Create environment
        env = gym.make('LunarLander-v3', continuous=True)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_scale = float(env.action_space.high[0])
        
        # Create agent
        device = torch.device("cpu")
        hyperparams = config.SAC_HYPERPARAMETERS
        agent = SACAgent(obs_dim, action_dim, action_scale, hyperparams, device)
        
        # Create buffer
        replay_buffer = ReplayBuffer(obs_dim, action_dim, 10000)
        
        # Run a few steps
        state, _ = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, float(done))
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
        
        # Try one update
        if len(replay_buffer) >= 8:
            batch = replay_buffer.sample(8, device)
            metrics = agent.update(batch)
            print(f"✓ Training step successful! Q1 Loss: {metrics['q1_loss']:.4f}")
        
        env.close()
        print("✅ Quick training test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Assignment 4 Setup Verification")
    print("="*60)
    
    checks = []
    
    # Run all checks
    checks.append(("Package Installation", check_imports()))
    checks.append(("Box2D Environments", check_box2d()))
    checks.append(("CUDA/GPU Support", check_cuda()))
    checks.append(("Project Files", check_files()))
    checks.append(("Quick Training Test", run_quick_test()))
    
    # Summary
    print("\n" + "="*60)
    print("Setup Verification Summary")
    print("="*60)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    all_passed = all(result for _, result in checks)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to start training!")
        print("\nQuick start commands:")
        print("  python train_sac.py --env lunarlander")
        print("  python train_ppo.py --env lunarlander")
        print("  python train_td3.py --env lunarlander")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1
    
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

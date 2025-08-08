#!/usr/bin/env python
"""Test script for TensorZero integration with mini-swe-agent - FizzBuzz implementation"""

import tempfile
import subprocess
from pathlib import Path
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

def test_tensorzero_agent_fizzbuzz():
    print("Testing TensorZero integration with mini-swe-agent for FizzBuzz...")
    
    # Create a temporary directory for the FizzBuzz implementation
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working directory: {tmpdir}")
        
        # Configuration for TensorZero model
        model_config = {
            'model_name': 'tensorzero',
            'config_file': Path('src/minisweagent/models/tensorzero/tensorzero.toml')
        }
        
        # Initialize the model
        model = get_model('tensorzero', model_config)
        print(f"✓ TensorZero model initialized")
        
        # Initialize the environment
        env = LocalEnvironment(cwd=tmpdir)
        print(f"✓ Local environment initialized")
        
        # Initialize the agent with config parameters
        agent = DefaultAgent(
            model=model, 
            env=env,
            system_template="""You are a helpful Python programming assistant.
            You can execute bash commands to create and run Python programs.
            When you complete a task, execute: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT""",
            
            instance_template="""Your task: {{task}}
            
            Please complete this task step by step:
            1. Create the Python script
            2. Run it to verify it works
            3. When done, execute: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
            
            Provide your commands in triple backticks.""",
            
            action_observation_template="Observation:\n{{output.output}}",
            format_error_template="Please provide exactly one command in triple backticks.",
            step_limit=10,
            cost_limit=1.0
        )
        print(f"✓ Agent initialized")
        
        # Define the FizzBuzz task
        task = """Create a Python script called fizzbuzz.py that implements FizzBuzz for numbers 1 to 30.
        The script should:
        - Print "Fizz" for multiples of 3
        - Print "Buzz" for multiples of 5
        - Print "FizzBuzz" for multiples of both 3 and 5
        - Print the number itself otherwise
        
        After creating the script, run it to verify it works correctly."""
        
        print(f"\n{'='*60}")
        print("Starting agent to implement FizzBuzz...")
        print(f"{'='*60}\n")
        
        try:
            # Run the agent
            exit_status, exit_message = agent.run(task)
            
            print(f"\n{'='*60}")
            print(f"Agent completed with status: {exit_status}")
            print(f"Exit message: {exit_message}")
            print(f"{'='*60}\n")
            
            # Verify the FizzBuzz file was created
            fizzbuzz_file = Path(tmpdir) / "fizzbuzz.py"
            if fizzbuzz_file.exists():
                print(f"✓ FizzBuzz script created at {fizzbuzz_file}")
                
                # Run the FizzBuzz script to get its output
                result = subprocess.run(
                    ["python", str(fizzbuzz_file)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    print(f"✓ FizzBuzz script executed successfully")
                    
                    # Verify the output
                    output_lines = result.stdout.strip().split('\n')
                    expected_values = []
                    for i in range(1, 31):
                        if i % 15 == 0:
                            expected_values.append("FizzBuzz")
                        elif i % 3 == 0:
                            expected_values.append("Fizz")
                        elif i % 5 == 0:
                            expected_values.append("Buzz")
                        else:
                            expected_values.append(str(i))
                    
                    # Check if output is correct
                    if len(output_lines) == 30:
                        all_correct = True
                        for i, (expected, actual) in enumerate(zip(expected_values, output_lines), 1):
                            if expected != actual:
                                print(f"✗ Line {i}: Expected '{expected}', got '{actual}'")
                                all_correct = False
                        
                        if all_correct:
                            print(f"✓ FizzBuzz output is correct!")
                            print(f"\nFirst 15 lines of output:")
                            for line in output_lines[:15]:
                                print(f"  {line}")
                            return True
                    else:
                        print(f"✗ Expected 30 lines of output, got {len(output_lines)}")
                else:
                    print(f"✗ Error running FizzBuzz: {result.stderr}")
            else:
                print(f"✗ FizzBuzz script was not created")
                
        except Exception as e:
            print(f"✗ Error during agent execution: {e}")
            import traceback
            traceback.print_exc()
            
    return False

if __name__ == "__main__":
    success = test_tensorzero_agent_fizzbuzz()
    exit(0 if success else 1)
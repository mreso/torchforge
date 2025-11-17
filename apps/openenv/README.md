# OpenEnv - Generic Training Framework

A centralized framework for training language models on any OpenEnv task using GRPO (Grouped Relative Policy Optimization).

## 📁 Folder Structure

```
apps/openenv/
  ├── main.py                    # Generic training script
  ├── python_utils.py            # Python/coding task utilities
  └── llama3_8b_coding.yaml      # Python coding training config
```

## 🎯 Key Features

- **Single Main Script**: One `main.py` works for all OpenEnv tasks
- **Task-Specific Utils**: Task-specific logic in separate files (e.g., `python_utils.py`)
- **YAML Configuration**: Use `!function` references to load task-specific functions
- **AutoEnv Integration**: Automatic environment and action class loading
- **Easy Extension**: Add new tasks by creating new utils files

## 🚀 Usage

### Run Python Coding Training

```bash
python -m apps.openenv.main --config apps/openenv/llama3_8b_coding.yaml
```

## 📝 YAML Configuration

Each task config needs:

```yaml
task:
  env_name: "coding"  # Environment name for AutoEnv
  build_action: !function apps.openenv.python_utils.build_python_action
  evaluate_response: !function apps.openenv.python_utils.evaluate_python_response
  transform_sample: !function apps.openenv.python_utils.transform_python_sample
```

The `!function` tag references functions from the utils files in the same directory.

## 🔧 Adding a New Task

To add support for a new task (e.g., Math problem solving):

### 1. Create Utils File

Create `apps/openenv/math_utils.py`:

```python
from envs import AutoAction

def build_math_action(response: str, sample: dict):
    """Build action from model response."""
    MathAction = AutoAction.from_env("math")
    answer = extract_answer(response)
    return MathAction(answer=answer, problem=sample.get("problem", ""))

def evaluate_math_response(result, response: str, sample: dict) -> float:
    """Evaluate execution result and return reward."""
    if result.observation.is_correct:
        return 1.0
    return 0.0

def transform_math_sample(sample: dict, tokenizer) -> dict | None:
    """Transform dataset sample for math tasks."""
    prompt = build_math_prompt(sample, tokenizer)
    return {
        "request": prompt,
        "target": sample.get("answer", ""),
        "task_id": sample.get("task_id", ""),
    }

def extract_answer(response: str) -> str:
    """Extract answer from model response."""
    # Implementation...
    pass
```

### 2. Create YAML Config

Create `apps/openenv/llama3_8b_math.yaml`:

```yaml
task:
  env_name: "math"
  build_action: !function apps.openenv.math_utils.build_math_action
  evaluate_response: !function apps.openenv.math_utils.evaluate_math_response
  transform_sample: !function apps.openenv.math_utils.transform_math_sample

dataset:
  path: "path/to/math/dataset"
  # ... other dataset config

# ... rest of config (same as other tasks)
```

### 3. Run It

```bash
python -m apps.openenv.main --config apps/openenv/llama3_8b_math.yaml
```

That's it! No changes to `main.py` needed.

## 📋 Task Utils API

Each task utils file should implement these functions:

### Required Functions

1. **`build_<task>_action(response: str, sample: dict) -> Action`**
   - Builds environment action from model response
   - Example: `build_python_action`, `build_math_action`

2. **`evaluate_<task>_response(result, response: str, sample: dict) -> float`**
   - Evaluates execution result and returns reward (0.0 to 1.0)
   - Example: `evaluate_python_response`, `evaluate_math_response`

3. **`transform_<task>_sample(sample: dict, tokenizer) -> dict | None`**
   - Transforms raw dataset sample into training format
   - Returns dict with 'request', 'target', 'task_id' or None if invalid
   - Example: `transform_python_sample`, `transform_math_sample`

### Optional Helper Functions

- **`get_<task>_system_prompt() -> str`**: Get system prompt for the task
- **`build_<task>_prompt(sample: dict, tokenizer) -> str`**: Build formatted prompt
- **`extract_<task>_code(response: str) -> str`**: Extract code/answer from markdown

## 🔍 How It Works

1. **Configuration Loading**: YAML config is loaded with `!function` references
2. **Function Loading**: `main.py` dynamically loads functions from utils files
3. **Environment Setup**: AutoEnv automatically loads correct env/action classes
4. **Training Loop**: Generic GRPO loop uses task-specific functions for:
   - Dataset transformation
   - Action building
   - Reward evaluation

## 📊 Dataset Format

Each transformed sample should have:

```python
{
    "request": str,   # Formatted prompt for model
    "target": str,    # Test code or target data
    "task_id": str,   # Unique task identifier
}
```

## 🎓 Example: Python Coding

The included Python utils demonstrate the pattern:

### Python Utils

- System prompt with coding guidelines
- Code extraction from markdown blocks
- Reward based on test execution results
- Works with HumanEval and AceCode dataset formats

### Key Functions

```python
# Extract Python code from markdown
code = extract_python_code(response)

# Build action for CodingEnv
action = build_python_action(response, sample)

# Evaluate using environment's reward
reward = evaluate_python_response(result, response, sample)
```

## 🔗 Integration with OpenEnv

This framework uses OpenEnv's AutoEnv feature:

```python
from envs import AutoEnv, AutoAction

# Automatically load the correct environment class
env_class = AutoEnv.from_name("coding")      # Loads CodingEnv
action_class = AutoAction.from_env("coding")  # Loads CodingAction
```

Make sure your environment is registered in OpenEnv's registry.

## 🐛 Debugging

Enable debug logging in main.py to see:
- Function loading
- Environment setup
- Reward calculation
- Code extraction

Set log level via environment variable:
```bash
export LOG_LEVEL=DEBUG
python -m apps.openenv.main --config apps/openenv/llama3_8b_coding.yaml
```

Or in YAML:
```yaml
metric_logging:
  console:
    logging_mode: global_reduce
    log_per_rank: True
```

## 📚 References

- **GRPO Algorithm**: Grouped Relative Policy Optimization
- **OpenEnv**: Generic environment framework for agentic RL
- **AutoEnv**: Automatic environment detection and loading
- **GenericOpenEnvActor**: Docker-based environment execution actor

## 🏗️ Architecture

The framework uses the `GenericOpenEnvActor` from `forge.actors.generic_openenv` which:
- Manages Docker container lifecycle
- Handles dynamic port allocation
- Provides automatic error recovery and container recreation
- Supports zombie process cleanup for long-running tasks
- Works with ANY OpenEnv environment (not just coding)

See `src/forge/actors/generic_openenv.py` for implementation details.

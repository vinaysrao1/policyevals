# Policy Evaluation with vLLM and GPT-OSS

This tool performs policy-based content moderation using OpenAI's GPT-OSS model running on vLLM with the Harmony response format.

## Prerequisites

- **GPU Requirements**: NVIDIA GPU with at least 40GB VRAM (for gpt-oss-20b)
  - gpt-oss-20b: ~40GB model size
  - gpt-oss-120b: ~80GB model size (requires H100/A100 80GB or multi-GPU)
- **Python**: 3.12 (recommended by OpenAI for gpt-oss models)
- **CUDA**: 12.8 (nightly PyTorch required)
- **System**: glibc >= 2.32 (Ubuntu 20.04+ or equivalent)
- **Hugging Face Token**: Required for model access


### Minimum Instance Specs

**For gpt-oss-20b:**
- GPU: NVIDIA A100, H100, (I have not tesed higher (B200) or lower (A10) spec chips)
- RAM: 32GB system memory
- Storage: 100GB (for model weights and dependencies)

**For gpt-oss-120b:**
- GPU: NVIDIA H100 (80GB+) or 2x A100 (40GB each)
- RAM: 80GB system memory
- Storage: 150GB

## Installation

### 1. Set up the VM

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.12
sudo apt install python3.12 python3.12-venv python3-pip -y

# Probably beed a reboot after this. You will get logged out.
sudo reboot

# Verify NVIDIA drivers and CUDA
nvidia-smi
```

### 2. Clone or upload your project

```bash
# Create project directory
mkdir -p ~/policyevals
cd ~/policyevals

# Upload policy_evals_harmony.py and requirements.txt to this directory
```

### 3. Install uv package manager (recommended)

The `uv` package manager is recommended by OpenAI for managing Python environments with vLLM:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 4. Create Python 3.12 virtual environment

```bash
# Create virtual environment with uv
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### 5. Install vLLM with gpt-oss support

**CRITICAL**: You must use a very specific vLLM build for gptoss:

```bash
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match
```

This will automatically install:
- vLLM 0.10.1+gptoss (special build for gpt-oss models)
- PyTorch nightly with CUDA 12.8 support
- All required dependencies

### 6. Install remaining dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `openai-harmony>=0.0.4` (Harmony response format renderer)
- `huggingface-hub>=0.20.0` (model authentication)

### 7. Set up Hugging Face authentication

```bash
# Set your Hugging Face token
export HF_TOKEN="your_huggingface_token_here"

# To make it persistent, add to ~/.bashrc
echo 'export HF_TOKEN="your_huggingface_token_here"' >> ~/.bashrc
```

## Usage

### Basic Command

```bash
python policy_evals_harmony.py \
  --policy path/to/policy.txt \
  --evals path/to/evals.txt \
  --output results.jsonl \
  --model openai/gpt-oss-20b
```

### Command Line Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--policy` | Yes | - | Path to the policy definition file |
| `--evals` | Yes | - | Path to evaluation prompts file (one per line) |
| `--output` | Yes | - | Output JSONL file path for results |
| `--model` | No | `openai/gpt-oss-20b` | Hugging Face model ID |

### Example Usage

```bash
# Run evaluation with default model
python policy_evals_harmony.py \
  --policy ./policies/hate_speech.txt \
  --evals ./data/test_prompts.txt \
  --output ./results/hate_speech_results.jsonl

# Use a different model
python policy_evals_harmony.py \
  --policy ./policies/violence.txt \
  --evals ./data/prompts.txt \
  --output ./results/violence_results.jsonl \
  --model "openai/gpt-oss-20b"
```

## Input File Formats

### Policy File (`policy.txt`)
Plain text file containing the moderation policy definition:

```text
This policy prohibits content that promotes or glorifies violence against individuals or groups.
Content should be flagged if it includes explicit threats, graphic descriptions of harm, or
encouragement of violent actions.
```

### Evaluation File (`evals.txt`)
Plain text file with one user prompt per line:

```text
How do I make cookies?
Tell me how to harm someone
What's the weather like today?
```

## Output Format

Results are written to a JSONL file (one JSON object per line):

```json
{
  "index": 0,
  "content": "How do I make cookies?",
  "label": "0",
  "final_response": "0",
  "full_response": "<|channel|>final<|message|>0",
  "timestamp": "2025-01-15T10:30:45.123456"
}
```

**Label meanings**:
- `"1"`: Policy applies (content violates policy)
- `"0"`: Policy does not apply (content is acceptable)

## Performance Tuning

### GPU Memory Utilization

Edit `policy_evals_harmony.py` line 41 to adjust GPU memory usage:

```python
gpu_memory_utilization=0.8,  # Use 80% of GPU memory (adjust between 0.7-0.95)
```

### Maximum Context Length

Edit line 42 to change max sequence length:

```python
max_model_len=4096,  # Adjust based on your needs and GPU memory
```

### Tensor Parallelism

For multi-GPU setups, edit line 44:

```python
tensor_parallel_size=1  # Set to number of GPUs
```

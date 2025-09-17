## Manual tasks

1. Clone the repository

```bash
# Clone repository
git clone https://github.com/beotavalo/AWorld.git
cd AWorld
```


### Prepare GAIA dataset
2. Install Git LFS. 
```bash
git lfs install
```
> **⚠️ Important**: 
> It is important to clone correctly the dataset files 
> You only need to run this command once per machine:
3. (Manual Task) Get Your Hugging Face API Token

> **⚠️ Important**: 
> Go to the Hugging Face website and log in.
> Click on your profile picture in the top-right corner and go to Settings.
> In the left sidebar, navigate to Access Tokens.
> Click the New token button. Give it a name (e.g., "My Laptop") and assign it a write role, which is necessary for pushing/uploading.
> Copy the generated token. Be careful, as it will only be shown once.

4. Install Hugging Face library.

```bash
uv pip install -U huggingface_hub
```

5. Run the HF login command: 

```bash
git config --global credential.helper store
hf auth login
```

6. Download the GAIA dataset from Hugging Face


```bash
mkdir dataset && cd dataset
git clone https://huggingface.co/datasets/gaia-benchmark/GAIA
```

7. Create the gaia run directory

```bash
mkdir gaia_run
```

### Environment Variables Configuration
8. Create an `.env` file to access `API_KEYS`

After creating the `.env` file, you need to configure the following API keys and service endpoints:

```bash
# LLM Model Config
# LLM_PROVIDER = {YOUR_CONFIG}
LLM_MODEL_NAME = {YOUR_CONFIG}
LLM_API_KEY = {YOUR_CONFIG}
LLM_BASE_URL = {YOUR_CONFIG}
LLM_TEMPERATURE = 0.0

# ===============Path Configurations=================
# GAIA_DATASET_PATH="/path/to/your/gaia-benchmark/GAIA/2023"
AWORLD_WORKSPACE="/tmp"

# ===============MCP Server Configurations=================
# [Google Search API](https://developers.google.com/custom-search/v1/introduction)
GOOGLE_API_KEY={YOUR_CONFIG}
GOOGLE_CSE_ID={YOUR_CONFIG}

# [Browser Use](https://github.com/browser-use/browser-use)
SKIP_LLM_API_KEY_VERIFICATION=true
OPENAI_API_KEY={YOUR_CONFIG}
COOKIES_FILE_PATH={YOUR_CONFIG}

# Audio Server
AUDIO_LLM_API_KEY={YOUR_CONFIG}
AUDIO_LLM_BASE_URL=https://api.zhizengzeng.com/v1
AUDIO_LLM_MODEL_NAME=gpt-4o-transcribe

# Image Server
IMAGE_LLM_API_KEY={YOUR_CONFIG}
IMAGE_LLM_BASE_URL=https://openrouter.ai/api/v1
IMAGE_LLM_MODEL_NAME=anthropic/claude-3.7-sonnet

# Video Server
VIDEO_LLM_API_KEY={YOUR_CONFIG}
VIDEO_LLM_BASE_URL=https://openrouter.ai/api/v1
VIDEO_LLM_MODEL_NAME=gpt-4o
VIDEO_LLM_TEMPERATURE=1.0

# Code Server
CODE_LLM_API_KEY={YOUR_CONFIG}
CODE_LLM_BASE_URL=https://openrouter.ai/api/v1
CODE_LLM_MODEL_NAME=anthropic/claude-sonnet-4

# Think Server
THINK_LLM_API_KEY={YOUR_CONFIG}
THINK_LLM_BASE_URL=https://openrouter.ai/api/v1
THINK_LLM_MODEL_NAME=deepseek/deepseek-r1-0528:free

# Guard Server
GUARD_LLM_API_KEY={YOUR_CONFIG}
GUARD_LLM_BASE_URL="https://openrouter.ai/api/v1"
GUARD_LLM_MODEL_NAME="google/gemini-2.5-pro-preview"

# [E2B Server](https://e2b.dev/docs/quickstart)
E2B_API_KEY={YOUR_CONFIG}
```

**Note**: Replace `your_*_api_key_here` with your actual API keys. Some services are optional depending on which tools you plan to use.

### Code agent promt

We suggest to execute this Prompt with code agents (Cursor, gemini-cli, claude code)

```
Run step by step the @gaia-agent-plan/plan-step-by-step.md, log the issues in @gaia-agent-plan/set_up_log.md, and register the progress in @gaia-agent-plan/set_up_checklist.md
```

### Basic Usage
Execute a single task

```bash
uv run examples/gaia/run.py --split validation --q 4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2
```

Execute a block of task

```bash
uv run examples/gaia/run.py --start 10 --end 50 --split validation
```

## Configuration

### Model Selection

- **LLM Model**: Defaults to `gpt-4.1` for task decomposition


### Dataset Path Configuration

To evaluate test dataset set the dataset path on line 345 of `evaluate_agent.py`
```bash
validation_data_path = os.path.join(os.path.dirname(__file__), 'dataset/GAIA/2023/test/metadata.jsonl')
```

### Tool Configuration

- **Search**: Configure SearxNG instance URL
- **Code Execution**: Customize import whitelist and security settings
- **Document Processing**: Set cache directories and processing limits

---

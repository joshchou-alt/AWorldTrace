### Python environment set up
1. Install uv for python and dependencies management

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install python versions if needed

```bash
# Install python if not already installed
uv python install 3.10 3.11 3.12
```

3. Create a 3.11 python virtual environment with uv

```bash
uv venv --python=3.11
```

4. Activate the python virtual environment

```bash
source .venv/bin/activate
```

5. Install python dependencies with uv

```bash
uv pip install -r requirements.txt
```
> **⚠️ Important**: 
> It is important to run two times the command due to a uv error (Try increasing UV_HTTP_TIMEOUT (current value: 30s))

6. Install the **AWorld framework** and build the web UI (Optional):

```bash
# Install PDF processing dependencies
uv pip install "marker-pdf[full]" --no-deps

# Install AWorld
uv run python setup.py install

#Optional
# Build web UI
sh -c "cd aworld/cmd/web/webui && npm install && npm run build"
```

### System Dependencies Installation
6. Install tools

**For macOS:**
```bash
brew install libmagic
brew install ffmpeg
brew install --cask libreoffice
```

> **Note**: Install Homebrew from [brew.sh](https://brew.sh/) if not already installed.

**For Linux:**
```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends libmagic1 libreoffice ffmpeg
```


9. Install playwright browser

```bash
playwright install chromium --with-deps --no-shell
```

10. Create the gaia run directory

```bash
mkdir gaia_run
```

11. Execute a single task

```bash
uv run evaluate_agent.py --task_id 4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2
```

12. Execute a block of task

```bash
uv run evaluate_agent.py --start 0 --end 2
```




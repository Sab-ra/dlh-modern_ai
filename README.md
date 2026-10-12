# Dev Container for ML study (CPU-only)

One container with Python 3.12, Jupyter, scikit-learn, LightGBM, XGBoost,
TensorFlow/Keras, PyTorch, Hugging Face, NLTK and OpenCV. No GPU needed.
Works on Linux (x86_64), macOS Apple Silicon and Windows (Docker Desktop + WSL2).

## How it works

```mermaid
flowchart TD
    Dev["Developer"] --> VSCode["VS Code"]
    VSCode --> DC["Dev Containers extension"]
    DC -->|reads| DJ[".devcontainer/devcontainer.json"]
    DJ -->|build| DF[".devcontainer/Dockerfile"]
    DF --> Base["python:3.12-slim + git, libgomp1"]
    DF --> Torch["torch + torchvision (CPU index)"]
    DF --> Req["requirements.txt: TF, sklearn, HF, NLTK..."]
    DC --> Docker["Docker Engine / Desktop"]
    Docker -->|image| Container["Dev Container (user: vscode)"]
    Repo["Git repo on host"] -->|bind mount| WS["/workspace"]
    WS --> Container
    V1[("ml-study-cache: ~/.cache (HF models)")] --> Container
    V2[("ml-study-nltk: ~/nltk_data")] --> Container
    Container -->|once after create| PC["post_create.sh"]
    PC --> NB["nbstripout git filter"]
    PC --> K["Jupyter kernel ml-study"]
    PC --> NL["NLTK data download"]
    VSCode <-->|attaches| Container
    Container --> Ext["Extensions: Python, Jupyter, Ruff"]
```

## Setup
1. Install Docker Desktop (Windows/macOS) or Docker Engine (Linux), and start it.
2. Install VS Code and the **Dev Containers** extension.
3. Clone this repo:
   `git clone https://github.com/digitalPatch/dev_container_for_ML_by_sabra.git`
4. Open the folder in VS Code.
5. Click **Reopen in Container** (or press F1 → "Dev Containers: Reopen in Container").
6. Wait for the first build (10–20 min, ~6 GB). Later rebuilds reuse the cache.
7. Open a notebook and pick the kernel **Python 3.12 (ml-study)**.

## Windows
1. Enable WSL2 in Docker Desktop (Settings → General → "Use the WSL 2 based engine").
2. Clone inside WSL for much faster file access: open Ubuntu (WSL), run the
   `git clone` from step 3, then run `code .` in that folder.
3. If you clone from Windows instead, use **Git Bash** and run
   `git config --global core.autocrlf input` before cloning.
   `.gitattributes` already forces LF, so `post_create.sh` still runs.

## Notebooks
1. Commit notebooks as usual (`git add`, `git commit`).
2. Rely on nbstripout: it is set up automatically in the container and strips
   outputs on commit. Your local file keeps its outputs.
3. Commit only from the container terminal (the filter lives there).

## Data and models
1. Put datasets in `data/`; they are git-ignored.
2. Force-add a small sample if needed: `git add -f data/sample.csv`.

## Update packages
1. Edit `requirements.txt`.
2. Run F1 → "Dev Containers: Rebuild Container".

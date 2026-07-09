# Modern AI Project (AI Academy by DLH)

## .gitignore

## Dev Container Environment

### Components of Dev Container Interaction

```mermaid
flowchart TD

    Dev[Developer]
    VSCode[VS Code]
    DC[Dev Container Extension]
    Docker[Docker Engine]

    Repo[Git Repository]
    DevContainer[.devcontainer/devcontainer.json]
    Dockerfile[.devcontainer/Dockerfile]
    Requirements[requirements.txt]

    Container[Running Dev Container]

    Extensions[VS Code Extensions]
    Python[Python 3.11]
    Jupyter[Jupyter Notebook]
    Tools[pytest · pycodestyle · ruff · black]

    Dev --> VSCode
    VSCode --> DC
    DC --> Docker

    Repo --> DevContainer
    Repo --> Dockerfile
    Repo --> Requirements

    DevContainer --> Dockerfile
    Dockerfile --> Requirements

    Docker --> Container

    Container --> Python
    Container --> Jupyter
    Container --> Tools
    Container --> Extensions

    VSCode <--> Container
```

### requirements.txt

List of Python libraries and versions that should be installed inside the container.

`requirements.txt` gives you:

- [x] Curriculum compliance
- [x] Jupyter notebooks
- [x] Unit testing (`pytest`)
- [x] Style checks (`pycodestyle`)
- [x] Fast linting (`ruff`)
- [x] Auto-formatting (`black`)

### .devcontainer/devcontainer.json

Defines the development experience:

- Container build configuration
- Workspace mounting
- VS Code extensions
- VS Code settings
- Default user and environment customization

### .devcontainer/Dockerfile

Defines the container image:

- Python version
- Operating system packages
- Project dependencies
- User configuration
- Runtime environment

Together, `devcontainer.json`, `Dockerfile`, and `requirements.txt` create a reproducible machine-learning environment that works consistently on macOS, Linux, and Windows.
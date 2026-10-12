#!/usr/bin/env bash
# Runs once inside the container after it is created. No host-side scripts.
set -euo pipefail


# Bind-mounted repo is owned by the host user (esp. on Windows/macOS).
git config --global --add safe.directory /workspace

# Strip notebook outputs on commit (filter is defined via .gitattributes).
python -m nbstripout --install --global
git config --global diff.ipynb.textconv "python -m nbstripout -t"

# Jupyter kernel visible to VS Code.
python -m ipykernel install --user --name ml-study --display-name "Python 3.12 (ml-study)"

# NLTK data used by the NLP tasks (cached in a volume, so only first time).
python -m nltk.downloader -q -d "$HOME/nltk_data" \
  punkt punkt_tab stopwords wordnet averaged_perceptron_tagger averaged_perceptron_tagger_eng

python -c "import torch, tensorflow as tf; print('torch', torch.__version__, '| tf', tf.__version__)"

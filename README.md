# community-archive-personal

- `brew install pyenv`
  - `echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc`
  - `echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc`
  - `echo 'eval "$(pyenv init -)"' >> ~/.zshrc`
  - `source ~/.zshrc`
  - `pyenv install 3.12`
  - `pyenv local 3.12`
- `python3 -m venv ca_pers_env`
- `source ca_pers_env/bin/activate`
- `pip install -r requirements.txt`
- `pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu`
- `streamlit run app.py`

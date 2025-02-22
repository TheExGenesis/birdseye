# ---
# deploy: true
# cmd: ["modal", "serve", "10_integrations/streamlit/serve_streamlit.py"]
# ---

# # Run and share Streamlit apps

# This example shows you how to run a Streamlit app with `modal serve`, and then deploy it as a serverless web app.

# ![example streamlit app](./streamlit.png)

# This example is structured as two files:

# 1. This module, which defines the Modal objects (name the script `serve_streamlit.py` locally).

# 2. `app.py`, which is any Streamlit script to be mounted into the Modal
# function ([download script](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/streamlit/app.py)).

import shlex
import subprocess
from pathlib import Path

import modal


# ## Define container dependencies

# The `app.py` script imports three third-party packages, so we include these in the example's
# image definition.

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "streamlit~=1.41.0",
    "numpy~=1.26.4",
    "pandas~=2.2.2",
    "scikit-learn",
    "toolz",
    "supabase",
    "plotly",
    "python-dotenv",
    "anthropic",
    "httpx",
    "pyarrow",
    "tqdm",
    "seaborn",
    "modal",
)

app = modal.App(name="community-archive-birdseye", image=image)

# ## Mounting the `app.py` script

# We can just mount the `app.py` script inside the container at a pre-defined path using a Modal
# [`Mount`](https://modal.com/docs/guide/local-data#mounting-directories).

# Mount all necessary files
src_dir = Path(__file__).parent
app_path = src_dir / "app.py"
components_path = src_dir / "components"
utils_path = src_dir / "utils"
styles_path = src_dir / "styles.css"

if not all(p.exists() for p in [app_path, components_path, utils_path, styles_path]):
    raise RuntimeError("Required files/directories not found!")

# Create mounts for all necessary files/directories
mounts = [
    modal.Mount.from_local_file(app_path, remote_path="/root/app.py"),
    modal.Mount.from_local_file(styles_path, remote_path="/root/styles.css"),
    modal.Mount.from_local_dir(components_path, remote_path="/root/components"),
    modal.Mount.from_local_dir(utils_path, remote_path="/root/utils"),
]

# ## Spawning the Streamlit server

# Inside the container, we will run the Streamlit server in a background subprocess using
# `subprocess.Popen`. We also expose port 8000 using the `@web_server` decorator.


@app.function(
    allow_concurrent_inputs=100,
    mounts=mounts,
)
@modal.web_server(8000)
def run():
    target = shlex.quote(str("/root/app.py"))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)


# ## Iterate and Deploy

# While you're iterating on your screamlit app, you can run it "ephemerally" with `modal serve`. This will
# run a local process that watches your files and updates the app if anything changes.

# ```shell
# modal serve serve_streamlit.py
# ```

# Once you're happy with your changes, you can deploy your application with

# ```shell
# modal deploy serve_streamlit.py
# ```

# If successful, this will print a URL for your app that you can navigate to from
# your browser 🎉 .

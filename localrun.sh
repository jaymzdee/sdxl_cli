#!/bin/bash
if command -v python3 &>/dev/null; then
    python_version=$(python3 --version 2>&1)  # Capture the output of python3 --version

    # Extract and display Python 3 version
    if [[ $python_version =~ Python\ 3\.11\.([0-9]+) ]]; then
        patch_version="${BASH_REMATCH[1]}"
        echo "Python 3.11.$patch_version is installed."

        python -m venv .venv
        source .venv/bin/activate
        pip -q install --upgrade pip
        pip install -r requirements.txt 
        python run_sdxl.py "$@"
    else
        echo "Python 3.11 is not installed"
    fi
fi
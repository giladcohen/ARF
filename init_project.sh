#!/bin/sh
# assumption:
# * virtualenv is avaliable and working

export ARF_PATH=$(pwd)
export PIP_REQUIRE_VIRTUALENV=true
export PIP_RESPECT_VIRTUALENV=true
virtualenv -p /usr/bin/python3 --clear --no-site-packages $ARF_PATH/.venv/arf_env
source $ARF_PATH/.venv/arf_env/bin/activate
pip install -r $ARF_PATH/requirements.txt --force-reinstall

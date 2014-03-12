#!bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: bash $0 ENV_NAME"
    exit 0
fi

if [ $# -gt 1 ]; then
    echo "To many input argument: $#"
    exit 1
fi

# create virutal environment"
env_path=$1
cd ~
virtualenv $env_path
cd $env_path/bin

# install dependencies of NK;
env_pip=~/$env_path/bin/pip
$env_pip install numpy
$env_pip install cython
$env_pip install numexpr
$env_pip install tables
$env_pip install pandas

# install NK
env_python=~/$env_path/bin/python
cd ~/$env_path/
git clone https://github.com/neurokernel/neurokernel.git
cd neurokernel
$env_python setup.py install

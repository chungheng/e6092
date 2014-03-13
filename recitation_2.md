# Recitation 2 #

## virtualenv ##

[virtualenv](http://www.virtualenv.org/) is a tool to create isolated virtual
environments in which to install Python packages. The virtual environments may
be isolated from each other and/or from the system environment. You are
encouraged to create virtual environment on [huxley](huxley.ee.columbia.edu)
for two reasons: (i) to keep track of packages you use during development; (ii)
to install multiple versions of Neurokernel in different environments
separately. For exemple, one vanilla Neurokernel, and one with your
modification.

`virtualenv` is already installed on `huxley`. To create a virtual
environment named `NK` under the home directory, run the following commands:

    $ cd ~
    $ virtualenv NK
    ....
    $ cd NK

You will see several directories including `bin` and `lib`. Like the usual Unix
system, inside the virtual environment, all executables (such as `python`) are
stored under `bin`, and all library files or packages (such as `numpy`) are
placed under `lib`. By default, `bin` of the virtual environment is not added
to `$PATH`. `virtual` provides with a handy script called `activate` under
`bin` to resolve path dependency. To use `activate` with the `NK` virtual
environment, run the following command:

    $ cd ~/NK/bin
    $ source activate

This will change your `$PATH` so its first entry is the `bin` of the virtual
environment. You have to use source because it changes your shell environment
in-place. The `activate` script also modifies your shell prompt to indicate
which environment is currently active. In our example, the prompt will look
like:

    (NK)$

You can switch back to your regular environment by using `deactivate` function:

    (NK)$ deactivate

It simply undoes the changes to your `$PATH` and shell prompt. We summarize
this section in the following commands:

    $ which python
    ... path to the default python on the system ...
    $ cd ~/NK/bin
    $ source activate
    (NK)$ which python
    (NK)$ /home/username/NK/bin/python
    (NK)$ deactivate
    $

(This section is modified from the document page of `virtualenv`.)

## pip ##

`pip` is a useful tool for managing python package, and it is included in
`virtualenv`. To use `pip` to install all the dependencies of `NeuroKernel` in
the virtual environment, we run the following commands:

    (NK)$ pip install numpy
    (NK)$ pip install cython
    (NK)$ pip install numexpr
    (NK)$ pip install tables
    (NK)$ pip install pandas

## Install NeuroKernel ##

Once you have all the dependencies installed, you are ready to install
`NeuroKernel`. First, `git clone` the `NeuroKernel` repository from GitHub into
the virtual environment by running:

    (NK)$ cd ~/NK
    (NK)$ git clone https://github.com/neurokernel/neurokernel.git

Next, we install `NeuroKernel` as follows:

    (NK)$ cd neurokernel
    (NK)$ python setup.py install

If you successfully install the `NeuroKernel`, navigate to
`~/NK/lib/python2.7/site-packages`. You should be able to see a `.egg` folder
of `NeuroKernel` in the directory.

The above procedures are included this shell [script](./shell/install_nk.sh).

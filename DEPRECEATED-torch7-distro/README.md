THIS REPOSITORY IS DEPRECEATED.
===============
Please use https://github.com/torch/torch7

For install scripts, please look at:
https://github.com/torch/ezinstall


Torch7 Library.
===============

Torch7 provides a Matlab-like environment for state-of-the-art machine
learning algorithms. It is easy to use and provides a very efficient
implementation, thanks to an easy and fast scripting language (Lua) and a
underlying C implementation.

In order to install Torch7 you can follow these simple instructions, but 
we suggest reading the detailed manual at http://www.torch.ch/manual/install/index

Requirements
------------

* C/C++ compiler
* cmake
* gnuplot
* git

Optional
--------

* Readline
* QT (QT4.8 is now supported)
* CBLAS
* LAPACK

Installation
------------

    $ git clone git://github.com/andresy/torch.git
    $ cd torch
    $ mkdir build
    $ cd build

    $ cmake .. 
    OR
    $ cmake .. -DCMAKE_INSTALL_PREFIX=/my/install/path

    $make install

Running
-------

    $torch
    Type help() for more info
    Torch 7.0  Copyright (C) 2001-2011 Idiap, NEC Labs, NYU
    Lua 5.1  Copyright (C) 1994-2008 Lua.org, PUC-Rio
    t7> 

3rd Party Packages
------------------

Torch7 comes with a package manager based on Luarocks. With it it's easy to 
install new packages:

    $ torch-rocks install image
    $ torch-rocks list
    $ torch-rocks search --all

Documentation
-------------

The full documentation is installed in /my/install/path/share/torch/html/index.html

Also, http://www.torch.ch/manual/index points to the latest documentation of Torch7.

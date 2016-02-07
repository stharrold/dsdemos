# dsdemos

Source code for examples from [Data Science Demos](https://stharrold.github.io).

Version numbers follow [Semantic Versioning v2.0.0](http://semver.org/spec/v2.0.0.html).

## Installation

`dsdemos` requires Python 3x.

To checkout the current version, clone the repository, add `dsdemos` to the module search path, then import:
```
$ git clone https://github.com/stharrold/dsdemos.git
$ python
>>> import os
>>> import sys
>>> sys.path.insert(0, os.path.join(os.path.curdir, r'dsdemos'))
>>> import dsdemos as dsd
```

To checkout a specific version (e.g. v0.0.3), checkout the version tag from within the repository:
```
$ git clone https://github.com/stharrold/dsdemos.git
$ cd dsdemos
$ git checkout tags/v0.0.3
$ cd ..
$ python # proceed as above from 'add `dsdemos` to the module search path, then import'
```

## Testing

Use [pytest](http://pytest.org/) within `dsdemos`:
```
$ git clone https://github.com/stharrold/dsdemos.git
$ cd dsdemos
$ py.test -v
```

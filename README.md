# dsdemos

Source code for examples from [Data Science Demos](https://stharrold.github.io).

Version numbers follow [Semantic Versioning v2.0.0](http://semver.org/spec/v2.0.0.html). (Version numbers have format 'N.N.N[-X]', where N are integer numbers and X is an optional label string. Git tags of versions have format 'vN.N.N[-X]'.)

## Installation

`dsdemos` requires Python 3x.

To download and checkout the current version, clone the repository, add `dsdemos` to the module search path, then import:
```
$ git clone https://github.com/stharrold/dsdemos.git
$ python
>>> import os
>>> import sys
>>> sys.path.insert(0, os.path.join(os.path.curdir, r'dsdemos'))
>>> import dsdemos as dsd
```

To update and checkout a specific version (e.g. v0.0.5), update your local repository then checkout the version's tag:
```
$ cd dsdemos
$ git pull
$ git checkout tags/v0.0.5
$ cd ..
$ python
>>> import os
>>> import sys
>>> sys.path.insert(0, os.path.join(os.path.curdir, r'dsdemos'))
>>> import dsdemos as dsd
>>> dsd.__version__
'0.0.5'
```

## Testing

Use [pytest](http://pytest.org/) within `dsdemos`:
```
$ git clone https://github.com/stharrold/dsdemos.git
$ cd dsdemos
$ py.test -v
```

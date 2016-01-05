# dsdemos

Source code for examples from [Data Science Demos](https://stharrold.github.io).

Version numbers follow [Semantic Versioning v2.0.0](http://semver.org/spec/v2.0.0.html).

## Installation

Requires Python 3x.

Clone the repository, add `dsdemos/dsdemos` to the module search path, then import:
```
$ git clone https://github.com/stharrold/dsdemos.git
$ python
>>> import os
>>> import sys
>>> sys.path.insert(0, os.path.join(os.path.curdir, r'dsdemos'))
>>> import dsdemos as dsd
```

## Testing

Use [pytest](http://pytest.org/) within `dsdemos`:
```
$ git clone https://github.com/stharrold/dsdemos.git
$ cd dsdemos
$ py.test -v
```

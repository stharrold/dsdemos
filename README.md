# dsdemos

Source code for examples from [Data Science Demos](https://stharrold.github.io).

## Installation

Requires Python 3x. Clone the repository from GitHub then import `stharrold.github.io/content/static/dsdemos/dsdemos` as a local package:
```
$ git clone https://github.com/stharrold/dsdemos.git
$ python
>>> import sys
>>> sys.path.insert(0, r'dsdemos')
>>> import dsdemos as dsd
```

## Testing

Use [pytest](http://pytest.org/) within `dsdemos`:
```
$ git clone https://github.com/stharrold/dsdemos.git
$ cd dsdemos
$ py.test -v
```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Pytests for dsdemos/census.py

"""


# Import standard packages.
import collections
import json
import os
import sys
# Import installed packages.
import pytest
# Import local packages.
sys.path.insert(0, os.path.curdir)
import dsdemos.census as census


def test_parse_pumsdatadict(
    path:str=os.path.join(
        os.path.curdir,
        'tests/test_census_data/test_parse_pumsdatadict.txt'),
    ref_path:str=os.path.join(
        os.path.curdir,
        'tests/test_census_data/test_parse_pumsdatadict.json')) -> None:
    r"""Pytest for parse_pumsdatadict.
    
    Notes:
        * Only some data dictionaries have been tested.[^urls]
        * Create 'test_parse_pumsdatadict.txt' from the source documents.[^urls]
        * Create 'test_parse_pumsdatadict.json' by
            ```
            import os
            import json
            import dsdemos as dsd
            path_txt = os.path.join(
                os.path.curdir,
                'tests/test_census_data/test_parse_pumsdatadict.json')
            path_json = os.path.splitext(path_txt)[0]+'.json'
            ddict = dsd.census.parse_pumsdatadict(path=path_txt)
            with open(path_json, 'w') as fobj:
                json.dump(ddict, fobj, indent=4)
            ```

    References:
        [^urls]: http://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/
            PUMSDataDict2013.txt
            PUMS_Data_Dictionary_2009-2013.txt

    """
    with open(ref_path) as fobj:
        ref_ddict = json.load(fobj)
    test_ddict = census.parse_pumsdatadict(path=path)
    assert ref_ddict == test_ddict
    # Raise FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        census.parse_pumsdatadict(path='does/not/exist.txt')
    return None

#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""General purpose utilities.

"""


# Import standard packages.
# Import installed packages.
# Import local packages.


def check_arguments(antns, lcls) -> None:
    r"""Check types of a function's input arguments.
    Call from within the function.
    
    Args:
        antns: Annotations of enclosing function.
        lcls: Local variables.
    
    Returns:
        None
        
    Raises:
        ValueError: Raised if arguments do not match annotated arguments.
    
    Notes:
        * Example usage:
            ```
            def myfunc(arg0:int, arg1:str) -> float:
               check_arguments(antns=myfunc.__annotations__, lcls=locals())
            ```
    
    """
    for (arg, cls) in antns.items():
        if arg != 'return':
            if not isinstance(lcls[arg], cls):
                raise ValueError(
                    ("type({arg}) must be {cls}\n" +
                     "type({arg}) = {typ}").format(
                        arg=arg, cls=cls, typ=type(lcls[arg])))
    return None

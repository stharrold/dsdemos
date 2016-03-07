#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""General purpose utilities.

"""


# Import standard packages.
# Import installed packages.
# Import local packages.


class Container:
    """Empty class to contain dynamically allocated attributes.
    Use to minimize namespace clutter from variable names.
    Use for heirarchical data storage like a `dict`.
    
    Example:
        ```
        data = Container()
        data.features = features
        data.target = target
        ```

    """
    pass

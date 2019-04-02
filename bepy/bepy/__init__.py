#Check dependencies
from typing import List

hard_dependencies = ("numpy", "pandas")
missing_dependencies=[]  # type: List[str]

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError("Missing dependencies {0}".format(missing_dependencies))

del hard_dependencies, dependency, missing_dependencies

#Import the classes making up bepy
from .measurement import *
from .sample import *
from .sampleset import *
from .analysis import *
from .otherfunctions import *

# module level doc-string
__doc__ = """
bepy - a data analysis and manipulation tool for material science characterizations 
=====================================================================


Main Features
-------------

"""
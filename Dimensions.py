'''
 Dimensions.py

 Contains metaclass to enforce each class is initialized with a proper number of dimensions. 
 In this project, the majority of classes have their basic functionality defined in the base class. 
 However, due to type safety and clarity reasons, they have derived classes for 2D and 3D.
 Base classes, thus, should not be initialized directly.

 Classes:
        DimensionMeta: Metaclass to enforce dimensionality in classes.
        DimensionMetaAbstract: Combined metaclass that enforces dimensionality and abstract method tracking.
 '''

import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from abc import ABCMeta, abstractmethod

class DimensionMeta(type):
    def __call__(cls, *args, **kwargs):
        # Check if the class has a properly defined 'dimensionality' attribute.
        # You can adjust what “undefined” means (here, we check for None).
        if getattr(cls, 'dimensionality', None) is None:
            raise TypeError("Cannot instantiate object: dimensionality is undefined for class '%s'." % cls.__name__)
        return super().__call__(*args, **kwargs)

# Combine the dimension enforcement with abstract base class functionality.
class DimensionMetaAbstract(ABCMeta, DimensionMeta):
    """
    This metaclass combines ABCMeta (for abstract method tracking) with
    DimensionMeta's instantiation check for defined dimensionality.
    """
    pass
import sys
import os
__version__ = '0.1.0'

# Get the path of the parent package 'opencompass'
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to sys.path to resolve imports correctly
sys.path.append(parent_dir)

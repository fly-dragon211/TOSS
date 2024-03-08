import sys
import os


print(os.getcwd())
sys.path.append("./")

from .utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token, tree_to_root_index,
                   tree_to_variable_index)
from .DFG import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
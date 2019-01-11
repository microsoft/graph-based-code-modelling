import os
from typing import Set

def get_csharp_reserved_names() -> Set[str]:
    script_path = os.path.normpath(os.path.split(__file__)[0])
    path = os.path.join(script_path, 'CSharpReservedNames.txt')
    with open(path) as f:
        reserved_names = set(l.strip() for l in f if len(l.strip()) > 0)
    return reserved_names

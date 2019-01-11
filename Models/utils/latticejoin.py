#!/usr/bin/env python
"""
Usage:
    latticejoin.py [options] INPUT_FOLDER OUTPUT_FILE

Options:
    -h --help                     Show this screen.
"""
import os
from typing import Optional

from docopt import docopt
from dpu_utils.codeutils.lattice import Lattice

if __name__ == '__main__':
    arguments = docopt(__doc__)

    input_folder = arguments['INPUT_FOLDER']
    output_file = arguments['OUTPUT_FILE']

    merged_lattice = None  # type: Optional[Lattice]

    for file in os.listdir(input_folder):
        if not file.endswith('.json.gz'):
            print('Ignoring file %s' % file)
            continue
        file_lattice = Lattice.load(os.path.join(input_folder, file))
        if merged_lattice is None:
            merged_lattice = file_lattice
        else:
            merged_lattice.merge(file_lattice)

    merged_lattice.save_as_json(output_file)

'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

#when running from source compile cpp extension if nessesary.

import sys
if not getattr(sys, 'frozen', False):
    from build import build_cpp_extension
    build_cpp_extension()

from calibration_methods import bundle_adjust_calibration

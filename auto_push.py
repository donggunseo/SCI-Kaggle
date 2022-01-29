# Usage: ./auto_push [input your commit msg]

import os
import sys

msg = sys.argv[1]

os.system("git add .")
os.system("git commit -m \"%s\"" % msg)
os.system("git push")
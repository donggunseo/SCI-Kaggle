# Usage: ./auto_push [input your commit msg]

import os
import sys

if len(sys.argv) != 2:
    print("Usage: python ./auto_push commit_msg")
    exit()
msg = sys.argv[1]

os.system("git add .")
os.system("git commit -m \"%s\"" % msg)
os.system("git push")
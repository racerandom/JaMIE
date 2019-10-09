# -*- coding: utf-8 -*-
#
# split line with TNM codes (for clinical texts)
#
import sys, os
import re

tnm_re = re.compile("((?:c|p|yc|r)?(?:T(?:is|1(?:mi|[abc])?|2[ab]?|3|4)|N[0-3]|M(?:0|1[abc]?)))")

def split_by_tnm(txt):
    return filter(lambda x: x, re.split(tnm_re, txt))

def main():
    for line in sys.stdin:
        line = line.rstrip()
        print("\n".join(split_by_tnm(line)))

if __name__ == "__main__":
    main()


import sys

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile, 'r') as inf:
    with open(outfile, 'w') as outf:
        outf.write(inf.read().replace('\r\n', '\n'))

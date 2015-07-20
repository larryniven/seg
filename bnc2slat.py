#!/usr/bin/env python3

import sys

f = open(sys.argv[1])

while True:
    line = f.readline()

    if not line:
        break

    print(line.strip())

    time = {}
    edges = []

    v = 0

    line = f.readline()
    while line and line != '.\n':

        parts = line.split()

        if v not in time:
            time[v] = parts[0]

        if time[v] != parts[0]:
            print('time does not match for two '
                'consecutive edges {} {}'.format(time[v], parts[0]), file=sys.stderr)
            exit(1)

        edges.append((v, v + 1, parts[2]))

        time[v + 1] = parts[1]

        v += 1

        line = f.readline()

    for u in range(v + 1):
        print('{} time={}'.format(u, int(int(time[u]) / 1e5)))

    print('#')

    for tail, head, label in edges:
        print('{} {} label={}'.format(tail, head, label))

    print('.')

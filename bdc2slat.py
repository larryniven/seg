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

    line = f.readline()
    while line and line != '.\n':

        parts = line.split()

        tail = int(parts[3])
        head = int(parts[4])
        label = parts[2]
        weight = '0'
        if len(parts) > 5:
            weight = parts[5]

        if tail not in time:
            time[tail] = parts[0]

        if head not in time:
            time[head] = parts[1]

        if tail in time and time[tail] != parts[0]:
            print('time {} for {} does not match {}'.format(
                time[tail], tail, parts[0]).format(), file=sys.stderr)
            exit(1)

        if head in time and time[head] != parts[1]:
            print('time {} for {} does not match {}'.format(
                time[head], head, parts[0]).format(), file=sys.stderr)
            exit(1)

        edges.append((tail, head, label, weight))

        line = f.readline()

    for u in sorted(time.keys()):
        print('{} time={}'.format(u, time[u]))

    print('#')

    for tail, head, label, weight in edges:
        if weight == '0':
            print('{} {} label={}'.format(tail, head, label))
        else:
            print('{} {} label={},weight={}'.format(tail, head, label, weight))

    print('.')

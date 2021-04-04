"""Replaced 0 - 50 IDs with original IDs
"""

import sys
import csv


def main():
    with open('analyze_output/room_assignments.txt', 'r') as file:
        room_assignments = [line.strip() for line in file.readlines()]

    with open(sys.argv[1], 'r', newline='') as infile:
        with open(sys.argv[2], 'w', newline='') as outfile:
            table_reader = csv.reader(infile)
            table_writer = csv.writer(outfile)
            for row in table_reader:
                for i in range(len(row) - 1):
                    row[i] = room_assignments[int(row[i])]
                table_writer.writerow(row)


if __name__ == '__main__':
    main()

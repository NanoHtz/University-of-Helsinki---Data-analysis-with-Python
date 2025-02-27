#!/usr/bin/env python3

import re

def red_green_blue(filename="src/rgb.txt"):
	result = []

	with open(filename, "r", encoding="utf-8") as file:
		next(file)
		for line in file:
			line = line.strip()
			if not line:
				continue
			match = re.match(r"(\d+)\s+(\d+)\s+(\d+)\s+(.+)", line)
			if match:
				red, green, blue, name = match.groups()
				result.append(f"{red}\t{green}\t{blue}\t{name}")

	return result

def main():
    pass

if __name__ == "__main__":
    main()

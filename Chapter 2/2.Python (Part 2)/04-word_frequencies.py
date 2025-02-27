#!/usr/bin/env python3

import string

def word_frequencies(filename="src/alice.txt"):
	result = {}
	with open(filename, "r", encoding="utf-8") as file:
		for line in file:
			words = line.split()
			for word in words:
				clean = word.strip("""!"#$%&'()*,-./:;?@[]_""")
				if clean in result:
					result[clean] += 1
				else:
					result[clean] = 1
	return result

def main():
    pass

if __name__ == "__main__":
    main()

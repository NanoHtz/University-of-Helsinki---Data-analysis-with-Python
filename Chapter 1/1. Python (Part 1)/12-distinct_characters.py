#!/usr/bin/env python3

def distinct_characters(L):
	result = {}
	for word in L:
		result[word] = len(set(word))

	return result

def main():
    print(distinct_characters(["check", "look", "try", "pop"]))

if __name__ == "__main__":
    main()

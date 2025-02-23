#!/usr/bin/env python3

def reverse_dictionary(d):
	result = {}

	for key, values in d.items():
		for word in values:
			if word in result:
				result[word].append(key)
			else:
				result[word] =[key]

	return result

def main():
    pass

if __name__ == "__main__":
    main()

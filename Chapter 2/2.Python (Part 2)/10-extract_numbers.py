#!/usr/bin/env python3

def extract_numbers(s):
	numbers = s.split()
	result = []
	for word in numbers:
		try:
			number = int(word)
			result.append(number)
		except ValueError:
			try:
				number = float(word)
				result.append(number)
			except ValueError:
				pass
	return result

def main():
    print(extract_numbers("abd 123 1.2 test 13.2 -1"))

if __name__ == "__main__":
    main()

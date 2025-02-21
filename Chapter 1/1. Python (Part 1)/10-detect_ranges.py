#!/usr/bin/env python3

def detect_ranges(L):

	LS = sorted(L)
	result = []
	i = 0
	while i < len(LS):
		start = LS[i]
		while i + 1 < len(LS) and LS[i] + 1 == LS[i + 1]:
			i += 1
		end = LS[i]
		if start == end:
			result.append(start)
		else:
			result.append((start, end + 1))
		i += 1
	return result

def main():
    L = [2, 5, 4, 8, 12, 6, 7, 10, 13]
    result = detect_ranges(L)
    print(L)
    print(result)

if __name__ == "__main__":
    main()

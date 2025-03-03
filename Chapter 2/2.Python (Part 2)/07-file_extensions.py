#!/usr/bin/env python3

def file_extensions(filename):
	n_ext = []
	ext = {}
	with open(filename, "r") as file:
		for line in file:
			line = line.strip()
			if "." in line:
				name, extension = line.rsplit(".", 1)
				if extension not in ext:
					ext[extension] = []
				ext[extension].append(line)
			else:
				n_ext.append(line)

	return (n_ext, ext)

def main():
	no_ext, ext = file_extensions("filenames.txt")
	print(f"{len(no_ext)} files with no extension")
	for key, value in ext.items():
		print(f"{key} {len(value)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import re

def integers_in_brackets(s):
    pattern = r"\[\s*([-+]?\d+)\s*\]"
    matches = re.findall(pattern, s)
    integers = [int(match) for match in matches]
    return integers

def main():
    pass

if __name__ == "__main__":
    main()

def first_non_repeating(arr):
    counts = {}
    for num in arr:
        counts[num] = counts.get(num, 0) + 1
    for num in arr:
        if counts[num] == 1:
            return num
    return -1

# Example
print(first_non_repeating([4, 5, 1, 2, 2, 5]))  # Output: 4

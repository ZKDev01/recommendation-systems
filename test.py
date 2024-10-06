def get_first_50_items(dictionary):
    return dict(list(dictionary.items())[:50])

# Example usage
my_dict = {i: i * 2 for i in range(150)}
first_50 = get_first_50_items(my_dict)
print(first_50)
from DataSet import *

shifted_left = shift_pixels_data_augmentation(LETTERS, "left")
shifted_right = shift_pixels_data_augmentation(LETTERS, "right")

# Display all the data set
for letter in LETTERS:
    # print matrix
    matrix = display_letter(LETTERS[letter])
    # print_matrix(matrix)
    # print vector
    vector = flatten_matrix(matrix)
    print_vector(vector)

#
# # Example usage: Display the letter 'A'
# letter = "A"
# print(f"Displaying the letter '{letter}':")
#
# # Convert the data into binary representative matrix
# matrix = display_letter(LETTERS[letter])
#
# # Print all the layers after each others and print the "picture"
# print_matrix(matrix)
#
# # Convert the matrix to a vector
# vector = flatten_matrix(matrix)
#
# # Print the 1D vector (the actual input for the model)
# print("\nFlattened Vector:")
# print_vector(vector)

print("")
matrix = display_letter(LETTERS["A"])
matrix_right = display_letter(shifted_right["A"])
matrix_left = display_letter(shifted_left["A"])

print_matrix(matrix)
print_matrix(matrix_right)
print_matrix(matrix_left)

# print vector
vector = flatten_matrix(matrix)
print_vector(vector)


after_augmentation = add_noise(LETTERS, 0.05)
print("vector with 5% noise")
matrix = display_letter(after_augmentation["A"])
# print_matrix(matrix)
# print vector
vector = flatten_matrix(matrix)
print_vector(vector)

after_augmentation = add_noise(LETTERS, 0.10)
print("vector with 10% noise")
matrix = display_letter(after_augmentation["A"])
# print_matrix(matrix)
# print vector
vector = flatten_matrix(matrix)
print_vector(vector)

after_augmentation = add_noise(LETTERS, 0.20)
print("vector with 20% noise")
matrix = display_letter(after_augmentation["A"])
# print_matrix(matrix)
# print vector
vector = flatten_matrix(matrix)
print_vector(vector)

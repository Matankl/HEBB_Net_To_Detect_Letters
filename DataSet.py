# Hexadecimal representations of letters A-Z
import random
random.seed(2021)
# random.seed(4225)

LETTERS = {
    "A": [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00],
    "B": [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00],
    "C": [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00],
    "D": [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00],
    "E": [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00],
    "F": [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00],
    "G": [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00],
    "H": [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00],
    "I": [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    "J": [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00],
    "K": [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00],
    "L": [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00],
    "M": [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00],
    "N": [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00],
    "O": [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00],
    "P": [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00],
    "Q": [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00],
    "R": [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00],
    "S": [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00],
    "T": [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    "U": [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00],
    "V": [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],
    "W": [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00],
    "X": [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00],
    "Y": [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00],
    "Z": [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00],
}

def display_letter(letter_hex):
    """Convert and display an 8x8 hex letter as a matrix of 0s and 1s."""
    matrix = []
    for byte in letter_hex:
        # Convert each byte to an 8-bit binary string and map it to a list of integers
        row = [int(bit) for bit in f"{byte:08b}"]
        matrix.append(row)
    return matrix


def print_matrix(matrix):
    """Print the matrix with 1s as # and 0s as a space."""
    print(matrix) # print the raws raw after each other
    for row in matrix:
        # print("".join("#" if pixel else "0" for pixel in row)) # this is the real way it looks like
        print("".join("#" if pixel else " " for pixel in row))


def flatten_matrix(matrix):
    """Flatten a 2D matrix into a 1D vector."""
    return [pixel for row in matrix for pixel in row]


def print_vector(vector):
    """Print the vector in a compact form."""
    print("Vector:", vector)


def convert_imag_to_vector(letter):
    mat = display_letter(letter)
    return flatten_matrix(mat)


# Define the output categories
def get_output(letter):
    if 'A' <= letter <= 'I':
        return [1, 0, 0]
    elif 'J' <= letter <= 'R':
        return [0, 1, 0]
    elif 'S' <= letter <= 'Z':
        return [0, 0, 1]


def add_noise(letters, chance):
    """
       Flips each bit in the binary representation of the LETTERS with a given chance.

       Args:
           letters (dict): Dictionary where keys are letters and values are lists of byte values.
           chance (float): Probability (0 <= chance <= 1) of flipping a bit.

       Returns:
           dict: A new dictionary with flipped bits.
       """
    flipped_letters = {}

    for letter, byte_list in letters.items():
        flipped_byte_list = []
        for byte in byte_list:
            flipped_byte = 0
            for bit in range(8):  # Iterate through each bit in the byte
                original_bit = (byte >> bit) & 1  # Get the current bit
                if random.random() < chance:
                    flipped_bit = 1 - original_bit  # Flip the bit
                else:
                    flipped_bit = original_bit
                flipped_byte |= (flipped_bit << bit)  # Set the bit in the flipped byte
            flipped_byte_list.append(flipped_byte)
        flipped_letters[letter] = flipped_byte_list

    return flipped_letters

def shift_pixels_data_augmentation(letters, direction):
    """
    Shifts the pixels in the binary representation of the LETTERS one bit.

    Args:
        letters (dict): Dictionary where keys are letters and values are lists of byte values.
        direction (str): Direction of the shift ("left" or "right").

    Returns:
        dict: A new dictionary with shifted pixels.
    """
    if direction not in {"left", "right"}:
        raise ValueError("Direction must be either 'left' or 'right'.")

    shifted_letters = {}
    for letter, byte_list in letters.items():
        shifted_byte_list = []
        for byte in byte_list:
            if direction == "left":
                shifted_byte = (byte << 1) & 0xFF  # Shift left and mask to 8 bits
            elif direction == "right":
                shifted_byte = (byte >> 1)  # Shift right
            shifted_byte_list.append(shifted_byte)
        shifted_letters[letter] = shifted_byte_list

    return shifted_letters

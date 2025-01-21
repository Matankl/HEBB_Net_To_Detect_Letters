import random
from DataSet import *
from HebbNet import *


# Hexadecimal representations of letters A-Z
letters = LETTERS
augmented_letters = shift_pixels_data_augmentation(LETTERS, "left")

# Prepare the training data: convert letter's 8x8 hex to 64-bit binary vector and add the lable
training_data = []
training_data_larg = []
for letter, hex_values in letters.items():
    input_vector = convert_imag_to_vector(hex_values)
    output_vector = get_output(letter)
    training_data.append((input_vector, output_vector))
    training_data_larg.append((input_vector, output_vector))

# add the augmented data for the larger model
for letter, hex_values in augmented_letters.items():
    input_vector = convert_imag_to_vector(hex_values)
    output_vector = get_output(letter)
    training_data_larg.append((input_vector, output_vector))




# Init the Hebbian network
print("Init the models")
model_one_epoch = HebbNet()
model_two_epoch = HebbNet()
model_with_augmentation = HebbNet()

# Train model 1 epoch
model_one_epoch.train_hebbian(training_data)
print("model_one_epoch trained")

# Train model 2 epoch
model_two_epoch.train_hebbian(training_data)
print("model_two_epoch trained once")
model_two_epoch.train_hebbian(training_data)
print("model_two_epoch trained twice")

# Train the larger model with overfit
model_with_augmentation.train_hebbian(training_data_larg)
print("model_with_augmentation trained")

models = [model_one_epoch, model_two_epoch, model_with_augmentation]


# Test the network
print("\ntesting with the training set")
test_hebbian_networkX(letters, models, convert_imag_to_vector, get_output, calculate_accuracy, calculate_f1)


print("\ntesting with 5% noise")
vectors5 = add_noise(letters, 0.05)
test_hebbian_networkX(vectors5, models, convert_imag_to_vector, get_output, calculate_accuracy, calculate_f1)

print("\ntesting with 10% noise")
vectors10 = add_noise(letters, 0.10)
test_hebbian_networkX(vectors10, models, convert_imag_to_vector, get_output, calculate_accuracy, calculate_f1)

print("\ntesting with 20% noise")
vectors20 = add_noise(letters, 0.20)
test_hebbian_networkX(vectors20, models, convert_imag_to_vector, get_output, calculate_accuracy, calculate_f1)



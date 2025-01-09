import random
from DataSet import *
from HebbNet import *


# Hexadecimal representations of letters A-Z
letters = LETTERS

# Prepare the training data: convert letter's 8x8 hex to 64-bit binary vector and add the lable
training_data = []
for letter, hex_values in letters.items():
    input_vector = convert_imag_to_vector(hex_values)
    output_vector = get_output(letter)
    training_data.append((input_vector, output_vector))



# Init the Hebbian network
print("Init the models")
model_one_epoch = HebbNet()
model_two_epoch = HebbNet()

# Train model 1 epoch
model_one_epoch.train_hebbian(training_data)
print("model_one_epoch trained")

# Train model 2 epoch
model_two_epoch.train_hebbian(training_data)
print("model_two_epoch trained once")
model_two_epoch.train_hebbian(training_data)
print("model_two_epoch trained twice")



# Test the network
print("testing with the training set")
test_hebbian_network(letters, model_one_epoch, model_two_epoch, convert_imag_to_vector, get_output, calculate_accuracy, calculate_f1)

print("testing with 5% noise")
vectors5 = data_augmentation(letters, 0.05)
test_hebbian_network(vectors5, model_one_epoch, model_two_epoch, convert_imag_to_vector, get_output, calculate_accuracy, calculate_f1)

print("testing with 10% noise")
vectors10 = data_augmentation(letters, 0.05)
test_hebbian_network(vectors10, model_one_epoch, model_two_epoch, convert_imag_to_vector, get_output, calculate_accuracy, calculate_f1)

print("testing with 20% noise")
vectors20 = data_augmentation(letters, 0.20)
test_hebbian_network(vectors20, model_one_epoch, model_two_epoch, convert_imag_to_vector, get_output, calculate_accuracy, calculate_f1)







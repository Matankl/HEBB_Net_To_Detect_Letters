import numpy as np

DEBUG = False

# Hebb Net Model Class
class HebbNet:
    def __init__(self):
        """
        Initialize the Hebb Network.
        """
        self.weights = np.zeros((3, 64))  # 3 outputs, 64 input features

    # Hebbian learning rule
    # https://www.youtube.com/watch?v=zpmdFzMAl8Y this is a great video about hebb rule
    def train_hebbian(self, data):
        for input_vector, output_vector in data:
            input_vector = np.array(input_vector)
            output_vector = np.array(output_vector)

            # outer takes each val from output_vector and multiply it with each val in input_vector and saves it in its oun cell
            # Update rule (if numpy is not allowed there is a method below)
            self.weights += np.outer(output_vector, input_vector)
            # print(self.weights)

    # Predict the output using the trained model
    def predict(self, input_vector):
        input_vector = np.array(input_vector)
        output = np.dot(self.weights, input_vector)  # Weighted sum
        # now make the largest value out of the 3 to be 1 and the rest 0
        max_value = max(output)
        # Normalize the output to 0 and 1 based on the maximum value
        for i in range(len(output)):
            if output[i] == max_value:
                output[i] = int(1)
            else:
                output[i] = int(0)
        output = [int(x) for x in output]
        return output


def calculate_accuracy(predictions, ground_truths):
    """
    Calculate accuracy for binary predictions.

    Args:
        predictions (list of lists): List of predicted binary outputs (e.g., [[0, 1, 0], [1, 0, 0]]).
        ground_truths (list of lists): List of ground truth binary outputs.

    Returns:
        float: Accuracy as a percentage.
    """
    correct = 0
    total = len(predictions)

    for pred, truth in zip(predictions, ground_truths):
        if pred == truth:
            correct += 1

    accuracy = (correct / total) * 100
    return accuracy


def calculate_f1(predictions, ground_truths):
    """
    Calculate F1 score for binary predictions.

    Args:
        predictions (list of lists): List of predicted binary outputs (e.g., [[0, 1, 0], [1, 0, 0]]).
        ground_truths (list of lists): List of ground truth binary outputs.

    Returns:
        float: F1 score.
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    for pred, truth in zip(predictions, ground_truths):
        for i in range(len(pred)):
            if pred[i] == 1 and truth[i] == 1:
                tp += 1
            elif pred[i] == 1 and truth[i] == 0:
                fp += 1
            elif pred[i] == 0 and truth[i] == 1:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


def test_hebbian_network(letters, model_one_epoch, model_two_epoch, convert_imag_to_vector, get_output,
                         calculate_accuracy, calculate_f1):
    """
    Tests two Hebbian models on a dataset and calculates their accuracy and F1 scores.

    Args:
        letters (dict): A dictionary mapping letters to their hex values.
        model_one_epoch (object): The first trained Hebbian model (one epoch).
        model_two_epoch (object): The second trained Hebbian model (two epochs).
        convert_imag_to_vector (function): Function to convert image hex values to an input vector.
        get_output (function): Function to get the expected binary output for a letter.
        calculate_accuracy (function): Function to calculate accuracy.
        calculate_f1 (function): Function to calculate F1 score.

    Returns:
        dict: A dictionary containing predictions, ground truths, and evaluation metrics.
    """
    predictions_one_epoch = []
    predictions_two_epoch = []
    ground_truths = []

    print("Testing the Hebbian Network on the training set:")

    # Loop through letters and test predictions
    for letter, hex_values in letters.items():
        input_vector = convert_imag_to_vector(hex_values)
        prediction_one_epoch = model_one_epoch.predict(input_vector)
        prediction_two_epoch = model_two_epoch.predict(input_vector)
        real = get_output(letter)

        predictions_one_epoch.append(prediction_one_epoch)
        predictions_two_epoch.append(prediction_two_epoch)
        ground_truths.append(real)

        if DEBUG:
            print(f"Letter: {letter}, Predicted Output (One Epoch): {prediction_one_epoch}, Expected Output: {real}")
            print(f"Letter: {letter}, Predicted Output (Two Epochs): {prediction_two_epoch}, Expected Output: {real}")
            print(" ")

    # Calculate metrics
    accuracy_one_epoch = calculate_accuracy(predictions_one_epoch, ground_truths)
    f1_score_one_epoch = calculate_f1(predictions_one_epoch, ground_truths)

    accuracy_two_epoch = calculate_accuracy(predictions_two_epoch, ground_truths)
    f1_score_two_epoch = calculate_f1(predictions_two_epoch, ground_truths)

    # Print metrics
    print(f"accuracy_one_epoch: {accuracy_one_epoch:.3f}%")
    print(f"f1_score_one_epoch: {f1_score_one_epoch:.3f}")
    print(f"accuracy_two_epoch: {accuracy_two_epoch:.3f}%")
    print(f"f1_score_two_epoch: {f1_score_two_epoch:.3f}")


def test_hebbian_networkX(letters, models, convert_imag_to_vector, get_output,
                         calculate_accuracy, calculate_f1):
    """
    Tests multiple Hebbian models on a dataset and calculates their accuracy and F1 scores.

    Args:
        letters (dict): A dictionary mapping letters to their hex values.
        models (list): A list of trained Hebbian models to test.
        convert_imag_to_vector (function): Function to convert image hex values to an input vector.
        get_output (function): Function to get the expected binary output for a letter.
        calculate_accuracy (function): Function to calculate accuracy.
        calculate_f1 (function): Function to calculate F1 score.

    Returns:
        dict: A dictionary with the following structure:
            {
                "predictions": [
                    [model1_pred_letter1, model1_pred_letter2, ...],
                    [model2_pred_letter1, model2_pred_letter2, ...],
                    ...
                ],
                "ground_truths": [true_label_letter1, true_label_letter2, ...],
                "metrics": {
                    "Model_1": {"accuracy": ..., "f1_score": ...},
                    "Model_2": {"accuracy": ..., "f1_score": ...},
                    ...
                }
            }
    """
    # Initialize a list of prediction lists—one sub-list per model
    all_predictions = [[] for _ in models]
    ground_truths = []

    print("Testing the Hebbian Network on the given dataset:")

    # Generate predictions for each letter in the dataset
    for letter, hex_values in letters.items():
        input_vector = convert_imag_to_vector(hex_values)
        real_output = get_output(letter)
        ground_truths.append(real_output)

        # Predict using each model
        for i, model in enumerate(models):
            prediction = model.predict(input_vector)
            all_predictions[i].append(prediction)

        # Debug print if needed
        if DEBUG:
            print(f"Letter: {letter}, Expected: {real_output}")
            for i, prediction_list in enumerate(all_predictions):
                # Just the last prediction in that list (the one we just added)
                print(f"  Model {i+1} Predicted: {prediction_list[-1]}")
            print(" ")

    # Calculate and print metrics for each model
    metrics = {}
    for i, predictions in enumerate(all_predictions):
        accuracy = calculate_accuracy(predictions, ground_truths)
        f1_score = calculate_f1(predictions, ground_truths)

        model_name = f"Model_{i+1}"
        metrics[model_name] = {
            "accuracy": accuracy,
            "f1_score": f1_score
        }

        print(f"{model_name} -> Accuracy: {accuracy:.3f}% | F1 Score: {f1_score:.3f}")

    # Return a detailed dictionary containing predictions, ground truths, and metrics
    return {
        "predictions": all_predictions,
        "ground_truths": ground_truths,
        "metrics": metrics
    }

# ____________________________________ private ____________________________________#

def outer_product(vector_a, vector_b):
    """
    Compute the outer product of two vectors.

    Args:
        vector_a: List of numbers (1D vector).
        vector_b: List of numbers (1D vector).

    Returns:
        A 2D list (matrix) representing the outer product.
    """
    # Initialize an empty matrix to store the results
    result = []

    # Iterate over each element in vector_a
    for a in vector_a:
        # For each element in vector_a, compute a row by multiplying with all elements in vector_b
        row = [a * b for b in vector_b]
        # Append the computed row to the result
        result.append(row)

    return result

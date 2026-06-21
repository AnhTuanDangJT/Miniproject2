"""
main.py

This file connects the dataset utilities (data_utils.py) and the
machine learning model (spam_model.py) into one working program.

Pipeline:
    1. Load the dataset
    2. Show basic dataset information
    3. Visualize the spam/ham distribution
    4. Train the spam detection model
    5. Run an interactive loop so the user can test their own messages
"""

from data_utils import load_dataset, show_dataset_info, visualize_label_counts
from spam_model import SpamDetectionModel


def interactive_loop(model):
    """
    Repeatedly asks the user for a message and predicts whether
    it is spam or ham, along with a confidence score.

    The loop ends when the user types 'quit' (case-insensitive).
    Blank input is ignored so the program doesn't crash on an empty Enter.
    """

    print("\nWelcome to Spam Detection AI")
    print("Type 'quit' to exit.")

    # This loop keeps running until the user chooses to quit.
    while True:
        # This loop keeps running until the user chooses to quit.
        message = input("\nEnter message: ").strip()

        if message.lower() == "quit":
            print("Goodbye!")
            break

        if message == "":
            print("Please enter a message (or type 'quit' to exit).")
            continue

        prediction, confidence = model.predict(message)

        print(f"Prediction: {prediction.upper()}")
        print(f"Confidence: {confidence:.2f}%")


def main():
    """
    Runs the full spam detection pipeline from start to finish:
    load data -> show info -> visualize -> train -> interactive loop.
    """

    file_path = "spam.csv"

    # Step 1: Load and clean the dataset
    data = load_dataset(file_path)

    # Step 2: Display basic dataset information.
    show_dataset_info(data)

    # Step 3: Visualize the spam vs. ham distribution
    visualize_label_counts(data)

    # Step 4: Train the model (Member 2's module)
    spam_model = SpamDetectionModel()
    print("\nTraining model...")
    spam_model.train(data)

    # Step 5: Start the interactive prediction loop.
    interactive_loop(spam_model)


if __name__ == "__main__":
    main()
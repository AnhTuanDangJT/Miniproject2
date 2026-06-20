from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class SpamDetectionModel:
    """
    This class handles the machine learning part of the project.

    Responsibilities:
    1. Convert text messages into numerical features using TF-IDF.
    2. Train a Logistic Regression spam classifier.
    3. Evaluate the model using accuracy and confusion matrix.
    4. Predict new messages and return a confidence score.
    """

    def __init__(self):
        """
        Creates the TF-IDF vectorizer and the machine learning model.
        """

        # TF-IDF converts words into numerical values.
        # stop_words="english" removes common words like "the", "is", "and".
        self.vectorizer = TfidfVectorizer(stop_words="english")

        # Logistic Regression is used as the classifier.
        # max_iter=1000 gives the model enough time to train properly.
        self.model = LogisticRegression(max_iter=1000)

        # This checks whether the model has been trained before prediction.
        self.is_trained = False

    def train(self, data):
        """
        Trains and evaluates the spam detection model.

        Parameter:
        data: pandas DataFrame containing two columns:
              - label
              - message
        """

        # Separate the input messages and the correct answers.
        X = data["message"]
        y = data["label"]

        # Split the dataset into training data and testing data.
        # 80% is used for training and 20% is used for testing.
        # stratify=y keeps the spam/ham ratio balanced in both sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Convert training messages into TF-IDF numerical features.
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Convert testing messages using the same vectorizer.
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Train the Logistic Regression model.
        self.model.fit(X_train_tfidf, y_train)

        # Mark the model as trained.
        self.is_trained = True

        # Predict labels for the testing data.
        y_pred = self.model.predict(X_test_tfidf)

        # Calculate accuracy.
        accuracy = accuracy_score(y_test, y_pred)

        # Create confusion matrix.
        # The order ["spam", "ham"] makes the matrix easier to understand.
        matrix = confusion_matrix(y_test, y_pred, labels=["spam", "ham"])

        print("\nModel trained successfully!")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        print("\nConfusion Matrix:")
        print("Rows = Actual labels")
        print("Columns = Predicted labels")
        print("Label order: spam, ham")
        print(matrix)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def predict(self, message):
        """
        Predicts whether a new message is spam or ham.

        Parameter:
        message: string entered by the user

        Returns:
        prediction: spam or ham
        confidence: confidence percentage
        """

        # Prevent prediction before training.
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Please train the model first.")

        # Convert the new message into TF-IDF features.
        message_tfidf = self.vectorizer.transform([message])

        # Predict spam or ham.
        prediction = self.model.predict(message_tfidf)[0]

        # Get probability for each class.
        probabilities = self.model.predict_proba(message_tfidf)[0]

        # Find the probability of the predicted class.
        class_index = list(self.model.classes_).index(prediction)

        # Convert confidence into percentage.
        confidence = probabilities[class_index] * 100

        return prediction, confidence
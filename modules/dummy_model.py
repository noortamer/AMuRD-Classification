from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def evaluate_dummy_model(X, y):
    """
    Trains and evaluates a dummy classifier using the most frequent category.
    Returns weighted F1-score.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values.reshape(-1, 1), y_encoded,
        test_size=0.2, random_state=42, stratify=y_encoded
    )

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="weighted")
    return f1

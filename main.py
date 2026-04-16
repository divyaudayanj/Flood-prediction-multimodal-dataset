from src.feature_engineering import load_data
from src.classifier import FloodClassifier

def main():
    X, y = load_data("data/raw/sample_flood_data.csv")
    clf = FloodClassifier()
    clf.train(X, y)
    preds = clf.predict(X)
    print("Training complete. Sample predictions:", preds[:10])

if __name__ == "__main__":
    main()

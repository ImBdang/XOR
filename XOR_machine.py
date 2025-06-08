import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0]) 
    model = DecisionTreeClassifier(random_state=42)

    model.fit(X, y)

    predictions = model.predict(X)

    for i in range(len(X)):
        a, b = X[i]
        predicted_xor = predictions[i]
        print(f"{a} XOR {b} = {predicted_xor}")

    # 5. Đánh giá hiệu suất của mô hình
    accuracy = accuracy_score(y, predictions)
    print(f"\nĐộ chính xác của mô hình: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

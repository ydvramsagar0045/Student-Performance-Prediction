import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Dataset Load & Auto-Separator Fix
try:
    df = pd.read_csv("student_data.csv")
    if len(df.columns) == 1:
        df = pd.read_csv("student_data.csv", sep=";")
except Exception as e:
    df = pd.read_csv("student_data.csv", sep=";")

# Clean column names (extra spaces hatane ke liye)
df.columns = df.columns.str.strip()

print("--- Detected Columns ---")
print(list(df.columns))

# 2. Data Cleaning
df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))

# 3. Dynamic Feature Selection (Numeric columns auto-pick)
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

if len(numeric_cols) >= 2:
    X = df[numeric_cols[:-1]]  # Target ko chhod kar baki sabhi numeric inputs
    y = df[numeric_cols[-1]]  # Last numeric column as Target (G3 / Final Score)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Model Performance ---")
    print("R2 Accuracy Score:", round(r2_score(y_test, y_pred), 3))
    print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 3))

    # Visualizations
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        df[numeric_cols[-5:]].corr(), annot=True, cmap="YlGnBu", fmt=".2f"
    )
    plt.title("Correlation Heatmap")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_test, y=y_pred, color="purple", alpha=0.7)
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Actual vs Predicted")

    plt.tight_layout()
    plt.show()
else:
    print("Dataset me sufficient numeric columns nahi mile.")
# https://chat.openai.com/share/2894d320-6996-4867-8596-afd12be26236
from datetime import date
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Hardcoded workout data with combined times for each day
workout_log = [
    {
        "date": date(2024, 5, 3),
        "movements": [
            {"movement": "Alternating Pistols", "repetitions": 20, "weight": 0},
            {"movement": "Burpees", "repetitions": 15, "weight": 0},
            {"movement": "Strict Pullups", "repetitions": 10, "weight": 0},
            {"movement": "Alternating Pistols", "repetitions": 20, "weight": 0},
            {"movement": "Burpees", "repetitions": 15, "weight": 0},
            {"movement": "Strict Pullups", "repetitions": 10, "weight": 0},
            {"movement": "Alternating Pistols", "repetitions": 20, "weight": 0},
            {"movement": "Burpees", "repetitions": 15, "weight": 0},
            {"movement": "Strict Pullups", "repetitions": 10, "weight": 0},
        ],
        "time_minutes": 10+32/60
    },
    {
        "date": date(2024, 5, 2),
        "movements": [
            {"movement": "deadlifts", "repetitions": 20, "weight": 80},
            {"movement": "box jumps", "repetitions": 25, "weight": None},
        ],
        "time_minutes": 17
    },
    {
        "date": date(2024, 5, 3),
        "movements": [
            {"movement": "burpees", "repetitions": 50, "weight": None},
        ],
        "time_minutes": 15
    },
    {
        "date": date(2024, 5, 4),
        "movements": [
            {"movement": "running", "repetitions": None, "weight": None},
            {"movement": "push-ups", "repetitions": 40, "weight": None},
        ],
        "time_minutes": 33
    }
]

# Combine all movements for each date into a single textual feature
flattened_data = []
for workout in workout_log:
    workout_date = workout["date"]
    movements_combined = ", ".join([f"{m['movement']} {m['repetitions']} reps, {m['weight']} kg"
                                    for m in workout["movements"]])
    flattened_data.append({
        "date": workout_date,
        "combined_movements": movements_combined,
        "time_minutes": workout["time_minutes"]
    })

# Create a DataFrame
df = pd.DataFrame(flattened_data)

# Split into features and labels
X = df[["combined_movements"]]
y = df["time_minutes"]

# Tokenize and encode the combined movement text feature
movement_tokenizer = tf.keras.preprocessing.text.Tokenizer()
movement_tokenizer.fit_on_texts(X["combined_movements"])
X_movement_encoded = movement_tokenizer.texts_to_matrix(X["combined_movements"], mode='binary')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_movement_encoded, y, test_size=0.2, random_state=42)

# Create a TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_movement_encoded.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Predicting a single numerical value
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=4)

# Make predictions on new data
new_workouts = pd.DataFrame([
    {"combined_movements": "burpees 60 reps, 0 kg; box jumps 20 reps, 0 kg"},
    {"combined_movements": "deadlifts 15 reps, 70 kg"}
])

# Encode new data
new_X_movement_encoded = movement_tokenizer.texts_to_matrix(new_workouts["combined_movements"], mode='binary')

# Predict and output results
predictions = model.predict(new_X_movement_encoded)
print(f"Predicted times for new workouts: {predictions.flatten()}")

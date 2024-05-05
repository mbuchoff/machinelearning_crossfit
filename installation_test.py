# https://chat.openai.com/share/2894d320-6996-4867-8596-afd12be26236
from datetime import date
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from array import array

# Hardcoded workout data with combined times for each day
workout_log = [
    {
        "date": date(2024, 5, 3),
        "movements": [
            {"movement": "alternating pistols", "repetitions": 20, "weight": 0},
            {"movement": "burpees", "repetitions": 15, "weight": 0},
            {"movement": "strict pullups", "repetitions": 10, "weight": 0},
            {"movement": "alternating pistols", "repetitions": 20, "weight": 0},
            {"movement": "burpees", "repetitions": 15, "weight": 0},
            {"movement": "strict pullups", "repetitions": 10, "weight": 0},
            {"movement": "alternating pistols", "repetitions": 20, "weight": 0},
            {"movement": "burpees", "repetitions": 15, "weight": 0},
            {"movement": "strict pullups", "repetitions": 10, "weight": 0},
        ],
        "time_minutes": 10+32/60
    },
    {
        "date": date(2024, 5, 2),
        "movements": [
            {"movement": "deadlifts", "repetitions": 20, "weight": 80},
            {"movement": "box jumps (don't open hips)", "repetitions": 25, "weight": 0},
        ],
        "time_minutes": 17
    },
    {
        "date": date(2024, 5, 3),
        "movements": [
            {"movement": "burpees", "repetitions": 50, "weight": 0},
        ],
        "time_minutes": 15
    },
    {
        "date": date(2024, 5, 4),
        "movements": [
            {"movement": "running", "repetitions": 400, "weight": 0},
            {"movement": "pushups", "repetitions": 40, "weight": 0},
        ],
        "time_minutes": 33
    }
]

valid_movements = [
    "alternating pistols",
    "box jumps (don't open hips)",
    "burpees",
    "deadlifts",
    "pushups",
    "running",
    "strict pullups"
]

# Combine all movements for each date into a single textual feature
flattened_data = []
for workout in workout_log:
    workout_date = workout["date"].toordinal()
    flattened_movements = []
    for movement in workout["movements"]:
        flattened_movements.append(valid_movements.index(movement["movement"]))
        flattened_movements.append(movement["repetitions"])
        flattened_movements.append(movement["weight"])
    flattened_data.append({
        "workout": np.array(flattened_movements),
        "time_minutes": workout["time_minutes"]
    })

# Create a DataFrame
df = pd.DataFrame(flattened_data)

# Split into features and labels
X = np.array(df["workout"])
y = df["time_minutes"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(X[0]),)),
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

# Predict and output results
predictions = model.predict(new_X_movement_encoded)
print(f"Predicted times for new workouts: {predictions.flatten()}")

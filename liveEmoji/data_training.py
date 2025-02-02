import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# Initialize variables
is_init = False
size = -1
label = []
dictionary = {}
c = 0

# Load .npy files
for i in os.listdir():
    if i.endswith(".npy") and i != "labels.npy":  
        file_size = os.path.getsize(i)
        
        # Skip empty files
        if file_size == 0:
            print(f"⚠️ Skipping empty file: {i}")
            continue
        
        try:
            temp = np.load(i)
            
            # Skip files that somehow load but have no data
            if temp.size == 0:
                print(f"⚠️ Skipping empty data in file: {i}")
                continue

            if not is_init:
                is_init = True
                X = temp
                size = X.shape[0]
                y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
            else:
                X = np.concatenate((X, temp))
                y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

            label.append(i.split('.')[0])
            dictionary[i.split('.')[0]] = c  
            c += 1
        except Exception as e:
            print(f"❌ Error loading {i}: {e}")

# Ensure data is loaded
if not is_init:
    print("🚨 No valid .npy files found! Exiting.")
    exit()

# Convert labels to categorical format
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")
y = to_categorical(y)

# Shuffle dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Define model
ip = Input(shape=(X.shape[1],))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 
model = Model(inputs=ip, outputs=op)

# Compile model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train model
model.fit(X, y, epochs=50)

# Save model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))

print("✅ Model training complete! Saved as `model.h5`.")

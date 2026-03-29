import pandas as pd
import numpy as np
import seaborn as sns
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report


# Column names
cols = ("fLength","fWidth","fSize","fConc","fConc1","fAsym",
        "fM3Long","fM3Trans","fAlpa","fDist","Class")

# Load dataset
df = pd.read_csv(r'D:\welcome\magic04.data', names=cols)

# Convert Class: 'g' → 1, 'h' → 0
df['Class'] = (df['Class'] == 'g').astype(int)


# # Plot histograms
for label in cols[:-1]:
    plt.hist(df[df['Class']==1][label], color='blue', label='gamma',
             alpha=0.7, density=True)
    plt.hist(df[df['Class']==0][label], color='red', label='hadron',
             alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel('Probability')
    plt.xlabel(label)
    plt.legend()
    plt.show()

# Shuffle and split
df_shuffled = df.sample(frac=1, random_state=42)
train = df_shuffled.iloc[:int(0.6*len(df))]
valid = df_shuffled.iloc[int(0.6*len(df)):int(0.8*len(df))]
test  = df_shuffled.iloc[int(0.8*len(df)):]

# Scaling + Oversampling
def scale_dataset(dataframe, oversample=False):
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y

train_data, x_train, y_train = scale_dataset(train, oversample=True)
valid_data, x_valid, y_valid = scale_dataset(valid, oversample=False)
test_data, x_test, y_test = scale_dataset(test, oversample=False)

print("Train shape:", train_data.shape)
print("Validation shape:", valid_data.shape)
print("Test shape:", test_data.shape)

# Plot functions
def plot_loss(history):
    fig,(ax1,ax2) = plt.subplots(1,2 , figsize = (10,4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.grid(True)

    
    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.show()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Neural Net


def train_model(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    # Build the model
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = nn_model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )

    return nn_model, history


least_val_loss = float('inf')
least_loss_model = None
epochs = 100

for num_nodes in [16, 32, 64]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                model, history = train_model(
                    x_train, y_train,
                    num_nodes, dropout_prob, lr,
                    batch_size, epochs
                )
                print(f"Nodes={num_nodes}, Dropout={dropout_prob}, LR={lr}, Batch={batch_size}")
                print(plot_loss(history))
                val_loss = model.evaluate(x_valid,y_valid)
                if sum(val_loss)/len(val_loss) < least_val_loss:
                    least_val_loss = sum(val_loss)/len(val_loss)
                    least_loss_model = model
                y_pred = least_loss_model.predict(x_test)
                y_pred = (y_pred> 0.5).astype(int).reshape(-1,)
                print(classification_report(y_test,y_pred)) 

                



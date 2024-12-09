from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

df = pd.read_csv('data/DF_after_EDA.csv')

train_df = df[['Sales', 'Seasonal', 'Residual', 'rolling_mean_7']]

# Виведемо приклад даних
print(train_df.info())


# ------------------------------ Попередня обробка даних ------------------------------

# Видалимо Nan
train_df = train_df.dropna()

# Створення скейлера та нормалізація даних
scalers = {}
for feature in train_df.columns:
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df.loc[:, feature] = scaler.fit_transform(train_df[[feature]])
    scalers[feature] = scaler

# Переведення даних в потрібний розмір, типу [1, 2, 3, 4, 5]=[6]; [2, 3, 4, 5, 6]=[7] і так далі
WINDOW_SIZE = 21
STEP_SIZE = 1


def create_sequences_np(data, window_size=WINDOW_SIZE, target_col_idx=0, step=STEP_SIZE):
    df_as_np = data.to_numpy()
    n_windows = (len(df_as_np) - window_size) // step + 1

    # Індекси для X
    idx_matrix = np.arange(window_size)[None, :] + np.arange(0, n_windows * step, step)[:, None]

    # Формування X і y
    X = df_as_np[idx_matrix]
    y = df_as_np[idx_matrix[:, -1], target_col_idx]  # Цільова колонка - Sales

    # Зміна форми для LSTM
    X = X.reshape((n_windows, window_size, df_as_np.shape[1]))
    return X, y


# Створення послідовностей
X, y = create_sequences_np(train_df)

# Розділимо на train/test
q_80 = int(len(train_df)*0.8)
X_train, y_train = X[:q_80], y[:q_80]
X_test, y_test = X[q_80:], y[q_80:]

# Перевірка розмірностей
print(f"\nTrain shape: {X_train.shape}, {y_train.shape}")



# ------------------------------ Створення і навчання моделі ------------------------------

model = Sequential([
    # Input layer
    Input(shape=(WINDOW_SIZE, 4)),

    # LSTM Layer with memory
    LSTM(128,
         return_sequences=True,
         activation='tanh'),

    # Dropout for preventing overfitting
    Dropout(0.2),

    # Additional LSTM Layer for complex pattern recognition
    LSTM(64, activation='tanh'),

    # Dropout for further regularization
    Dropout(0.2),

    # Dense layers for final prediction
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

# З часом LR для навчання буде зменшуватись
initial_lr = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

# Компіляція моделі
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='huber', metrics=['mae'])

model.summary()

# Зберігаємо тільки найкращу модель
checkpoint = ModelCheckpoint('models/model_2.keras', save_best_only=True)

# Якщо модель довго не покращує результат, то зупиняємо навчання
early_stopping = EarlyStopping(
    monitor='val_mae',
    patience=4,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[checkpoint, early_stopping]
)



# ------------------------------ Оцінка моделі ------------------------------

# Прогнозування
y_pred_normalized = model.predict(X_test)

# Зворотне перетворення для отримання реальних значень
y_pred = scalers['Sales'].inverse_transform(y_pred_normalized.reshape(-1, 1))

# Зворотне масштабування справжніх значень
y_test_original = scalers['Sales'].inverse_transform(y_test.reshape(-1, 1))

# Обчислення метрик
mae = mean_absolute_error(y_test_original, y_pred)
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred)

print(f"Метрики оцінки моделі:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2) Score: {r2}")

# Графік порівняння прогнозованих і реальних значень
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='Реальні значення', color='blue')
plt.plot(y_pred, label='Прогнозовані значення', color='red', linestyle='--')
plt.title('Порівняння реальних та прогнозованих значень')
plt.xlabel('Часовий крок')
plt.ylabel('Продажі')
plt.legend()
plt.tight_layout()
plt.show()

# Графік залишків (різниця між реальними та прогнозованими значеннями)
residuals = y_test_original - y_pred
plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Залишки', color='green')
plt.title('Графік залишків')
plt.xlabel('Часовий крок')
plt.ylabel('Різниця між реальними та прогнозованими значеннями')
plt.axhline(y=0, color='r', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

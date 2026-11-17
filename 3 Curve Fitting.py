import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 1. Load the raw CSV without assuming default headers
df_raw = pd.read_csv('cooling_datasets.csv', header=None)

# Room temperature is stored in row 0, column 1 for 'T_noisy_newton_nofan'
room_temp = float(df_raw.iloc[0, 1]) 

# Column headers are in row 1
headers = df_raw.iloc[1].values

# Data starts from row 2 onwards
data = df_raw.iloc[2:].copy()
data.columns = headers
data = data.astype(float)

# 2. Extract time and your target column
t = data['time_minutes'].values
T = data['T_noisy_newton_nofan'].values

print(f"Room Temperature (Ta): {room_temp}°C")
print(f"Loaded {len(t)} data points successfully.")

# Calculate the split index (1/3 for training)
n_total = len(t)
n_train = n_total // 3  # First 1/3 of the dataset

# Training set (First 1/3)
t_train = t[:n_train]
T_train = T[:n_train]

# Extrapolation test set (Remaining 2/3)
t_test = t[n_train:]
T_test = T[n_train:]

print(f"Total points: {n_total}")
print(f"Training points (First 1/3): {len(t_train)}")
print(f"Test points (Remaining 2/3): {len(t_test)}")


# Define Newton's Law of Cooling model function
def newton_model(t, k, T0):
    return room_temp + (T0 - room_temp) * np.exp(-k * t)

# Fit the model using ONLY the training data
# p0=[0.02, 95.0] gives initial guesses for [k, T0]
popt, _ = curve_fit(newton_model, t_train, T_train, p0=[0.02, 95.0])
k_fit, T0_fit = popt

print(f"Fitted Newton Parameters from Training Data:")
print(f"  Cooling coefficient (k): {k_fit:.4f} min^-1")
print(f"  Initial Temperature (T0): {T0_fit:.2f}°C")

# Generate predictions across the *entire* time range (train + test)
T_pred_newton = newton_model(t, k_fit, T0_fit) 

# Dictionary to store RMSE scores for each model
test_rmses = {}

# 1. Calculate Test RMSE for Newton's Law (evaluated on the remaining 2/3 test set)
T_test_pred_newton = T_pred_newton[n_train:]
newton_rmse = np.sqrt(np.mean((T_test_pred_newton - T_test)**2))
test_rmses['Newton'] = newton_rmse
print(f"Newton's Law Test RMSE: {newton_rmse:.4f}")

# 2. Fit Polynomials of Degree 1 to 5
poly_predictions = {}

for deg in range(1, 6):
    # Fit polynomial on training data only
    coeffs = np.polyfit(t_train, T_train, deg)
    poly_func = np.poly1d(coeffs)
    
    # Predict across the full time series
    poly_predictions[deg] = poly_func(t)
    
    # Calculate test RMSE on the test set
    poly_test_rmse = np.sqrt(np.mean((poly_func(t_test) - T_test)**2))
    test_rmses[f'Poly deg {deg}'] = poly_test_rmse
    print(f"Polynomial Degree {deg} Test RMSE: {poly_test_rmse:.4f}")


# Choose a polynomial degree to compare (e.g., degree 3 or the best one)
best_deg = 3 

plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(t_train, T_train, color='black', label='Training Data (First 1/3)', zorder=5)
plt.scatter(t_test, T_test, color='gray', alpha=0.5, label='Test Data (Remaining 2/3)', zorder=4)

# Plot model predictions across the full time series
plt.plot(t, T_pred_newton, color='blue', lw=2, label=f"Newton's Law (Test RMSE: {test_rmses['Newton']:.2f})")
plt.plot(t, poly_predictions[best_deg], color='red', ls='--', lw=1.5, label=f"Degree {best_deg} Poly (Test RMSE: {test_rmses[f'Poly deg {best_deg}']:.2f})")

# Mark the train/test boundary
plt.axvline(t_train[-1], color='orange', linestyle=':', label='Train/Test Split Boundary')

plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('Part 3: Extrapolation Test - Newton vs. Polynomials')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#now similar for T_Noisy_fan

# 1. Load the dataset
df_raw = pd.read_csv('cooling_datasets.csv', header=None)
room_temp = float(df_raw.iloc[0, 2])  # Room temperature for T_noisy_newton_fan is 27.0 °C
headers = df_raw.iloc[1].values
data = df_raw.iloc[2:].copy()
data.columns = headers
data = data.astype(float)

# Extract time and target dataset
t_all = data['time_minutes'].values
T_all = data['T_noisy_newton_fan'].values

# 2. Clean data (sort and drop any NaN values)
clean_df = pd.DataFrame({'time': t_all, 'T': T_all}).dropna().sort_values('time').reset_index(drop=True)
t = clean_df['time'].values
T = clean_df['T'].values

# 3. Split Data: First 1/3 for training, remaining 2/3 for extrapolation test
n_total = len(t)
n_train = n_total // 3

t_train = t[:n_train]
T_train = T[:n_train]
t_test = t[n_train:]
T_test = T[n_train:]

print(f"Dataset: T_noisy_newton_fan (Ta = {room_temp}°C)")
print(f"Total points: {n_total} | Training (1/3): {n_train} | Test (2/3): {len(t_test)}")

# 4. Fit Newton's Law of Cooling on Training Set only
def newton_model(time_val, k, T0):
    return room_temp + (T0 - room_temp) * np.exp(-k * time_val)

popt, _ = curve_fit(newton_model, t_train, T_train, p0=[0.03, T_train[0]], maxfev=10000)
k_fit, T0_fit = popt
print(f"\nFitted Newton Parameters -> k: {k_fit:.4f} min^-1, T0: {T0_fit:.2f}°C")

# Predict across full time range and calculate test RMSE
T_pred_newton = newton_model(t, k_fit, T0_fit)
newton_test_rmse = np.sqrt(np.mean((T_pred_newton[n_train:] - T_test)**2))
print(f"Newton's Law Test RMSE: {newton_test_rmse:.4f}")

# 5. Fit Polynomials (Degree 1 to 5) on Training Set
poly_rmses = {}
poly_predictions = {}

for deg in range(1, 6):
    coeffs = np.polyfit(t_train, T_train, deg)
    poly_func = np.poly1d(coeffs)
    poly_predictions[deg] = poly_func(t)
    
    poly_test_rmse = np.sqrt(np.mean((poly_func(t_test) - T_test)**2))
    poly_rmses[deg] = poly_test_rmse
    print(f"Polynomial Degree {deg} Test RMSE: {poly_test_rmse:.4f}")

# Find best polynomial for plotting comparison
best_deg = min(poly_rmses, key=poly_rmses.get)

# 6. Plotting Results
plt.figure(figsize=(10, 6))
plt.scatter(t_train, T_train, color='black', label='Training Data (First 1/3)', zorder=5)
plt.scatter(t_test, T_test, color='gray', alpha=0.5, label='Test Data (Remaining 2/3)', zorder=4)

plt.plot(t, T_pred_newton, color='blue', lw=2, label=f"Newton's Law (Test RMSE: {newton_test_rmse:.2f})")
plt.plot(t, poly_predictions[best_deg], color='red', ls='--', lw=1.5, label=f"Best Poly Deg {best_deg} (Test RMSE: {poly_rmses[best_deg]:.2f})")

plt.axvline(t_train[-1], color='orange', linestyle=':', label='Train/Test Split Boundary')
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('Part 3 Extrapolation Test: T_noisy_newton_fan')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#now for T_noisy_stefans_law

# 1. Load the dataset
df_raw = pd.read_csv('cooling_datasets.csv', header=None)
room_temp = float(df_raw.iloc[0, 3])  # Room temperature for T_noisy_stefans_law is 27.0 °C
headers = df_raw.iloc[1].values
data = df_raw.iloc[2:].copy()
data.columns = headers
data = data.astype(float)

# Extract time and target dataset
t_all = data['time_minutes'].values
T_all = data['T_noisy_stefans_law'].values

# 2. Clean data (sort and drop any NaN values)
clean_df = pd.DataFrame({'time': t_all, 'T': T_all}).dropna().sort_values('time').reset_index(drop=True)
t = clean_df['time'].values
T = clean_df['T'].values

# 3. Split Data: First 1/3 for training, remaining 2/3 for extrapolation test
n_total = len(t)
n_train = n_total // 3

t_train = t[:n_train]
T_train = T[:n_train]
t_test = t[n_train:]
T_test = T[n_train:]

print(f"Dataset: T_noisy_stefans_law (Ta = {room_temp}°C)")
print(f"Total points: {n_total} | Training (1/3): {n_train} | Test (2/3): {len(t_test)}")

# 4. Fit Physics/Newton Model on Training Set only
def newton_model(time_val, k, T0):
    return room_temp + (T0 - room_temp) * np.exp(-k * time_val)

popt, _ = curve_fit(newton_model, t_train, T_train, p0=[0.02, T_train[0]], maxfev=10000)
k_fit, T0_fit = popt
print(f"\nFitted Model Parameters -> k: {k_fit:.4f} min^-1, T0: {T0_fit:.2f}°C")

# Predict across full time range and calculate test RMSE
T_pred_model = newton_model(t, k_fit, T0_fit)
model_test_rmse = np.sqrt(np.mean((T_pred_model[n_train:] - T_test)**2))
print(f"Model Test RMSE: {model_test_rmse:.4f}")

# 5. Fit Polynomials (Degree 1 to 5) on Training Set
poly_rmses = {}
poly_predictions = {}

for deg in range(1, 6):
    coeffs = np.polyfit(t_train, T_train, deg)
    poly_func = np.poly1d(coeffs)
    poly_predictions[deg] = poly_func(t)
    
    poly_test_rmse = np.sqrt(np.mean((poly_func(t_test) - T_test)**2))
    poly_rmses[deg] = poly_test_rmse
    print(f"Polynomial Degree {deg} Test RMSE: {poly_test_rmse:.4f}")

# Find best polynomial for plotting comparison
best_deg = min(poly_rmses, key=poly_rmses.get)

# 6. Plotting Results
plt.figure(figsize=(10, 6))
plt.scatter(t_train, T_train, color='black', label='Training Data (First 1/3)', zorder=5)
plt.scatter(t_test, T_test, color='gray', alpha=0.5, label='Test Data (Remaining 2/3)', zorder=4)

plt.plot(t, T_pred_model, color='blue', lw=2, label=f"Physics Model (Test RMSE: {model_test_rmse:.2f})")
plt.plot(t, poly_predictions[best_deg], color='red', ls='--', lw=1.5, label=f"Best Poly Deg {best_deg} (Test RMSE: {poly_rmses[best_deg]:.2f})")

plt.axvline(t_train[-1], color='orange', linestyle=':', label='Train/Test Split Boundary')
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('Part 3 Extrapolation Test: T_noisy_stefans_law')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#now for T_open_measured

# 1. Load the dataset
df_raw = pd.read_csv('cooling_datasets.csv', header=None)
room_temp = float(df_raw.iloc[0, 4])  # Room temperature for T_open_measured is 30.5 °C
headers = df_raw.iloc[1].values
data = df_raw.iloc[2:].copy()
data.columns = headers
data = data.astype(float)

# Extract time and target dataset (dropping any NaN values for real measured data)
t_all = data['time_minutes'].values
T_all = data['T_open_measured'].values

clean_df = pd.DataFrame({'time': t_all, 'T': T_all}).dropna().sort_values('time').reset_index(drop=True)
t = clean_df['time'].values
T = clean_df['T'].values

# 2. Split Data: First 1/3 for training, remaining 2/3 for extrapolation test
n_total = len(t)
n_train = n_total // 3

t_train = t[:n_train]
T_train = T[:n_train]
t_test = t[n_train:]
T_test = T[n_train:]

print(f"Dataset: T_open_measured (Ta = {room_temp}°C)")
print(f"Total points: {n_total} | Training (1/3): {n_train} | Test (2/3): {len(t_test)}")

# 3. Fit Newton's Law of Cooling on Training Set only
def newton_model(time_val, k, T0):
    return room_temp + (T0 - room_temp) * np.exp(-k * time_val)

popt, _ = curve_fit(newton_model, t_train, T_train, p0=[0.02, T_train[0]], maxfev=10000)
k_fit, T0_fit = popt
print(f"\nFitted Newton Parameters -> k: {k_fit:.4f} min^-1, T0: {T0_fit:.2f}°C")

# Predict across full time range and calculate test RMSE
T_pred_newton = newton_model(t, k_fit, T0_fit)
newton_test_rmse = np.sqrt(np.mean((T_pred_newton[n_train:] - T_test)**2))
print(f"Newton's Law Test RMSE: {newton_test_rmse:.4f}")

# 4. Fit Polynomials (Degree 1 to 5) on Training Set
poly_rmses = {}
poly_predictions = {}

for deg in range(1, 6):
    coeffs = np.polyfit(t_train, T_train, deg)
    poly_func = np.poly1d(coeffs)
    poly_predictions[deg] = poly_func(t)
    
    poly_test_rmse = np.sqrt(np.mean((poly_func(t_test) - T_test)**2))
    poly_rmses[deg] = poly_test_rmse
    print(f"Polynomial Degree {deg} Test RMSE: {poly_test_rmse:.4f}")

# Find best polynomial for plotting comparison
best_deg = min(poly_rmses, key=poly_rmses.get)

# 5. Plotting Results
plt.figure(figsize=(10, 6))
plt.scatter(t_train, T_train, color='black', label='Training Data (First 1/3)', zorder=5)
plt.scatter(t_test, T_test, color='gray', alpha=0.5, label='Test Data (Remaining 2/3)', zorder=4)

plt.plot(t, T_pred_newton, color='blue', lw=2, label=f"Newton's Law (Test RMSE: {newton_test_rmse:.2f})")
plt.plot(t, poly_predictions[best_deg], color='red', ls='--', lw=1.5, label=f"Best Poly Deg {best_deg} (Test RMSE: {poly_rmses[best_deg]:.2f})")

plt.axvline(t_train[-1], color='orange', linestyle=':', label='Train/Test Split Boundary')
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('Part 3 Extrapolation Test: T_open_measured')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#now for T_measured

# 1. Load the dataset
df_raw = pd.read_csv('cooling_datasets.csv', header=None)
room_temp = float(df_raw.iloc[0, 5])  # Room temperature for T_measured is 30.5 °C
headers = df_raw.iloc[1].values
data = df_raw.iloc[2:].copy()
data.columns = headers
data = data.astype(float)

# Extract time and target dataset (dropping any NaN values for real measured data)
t_all = data['time_minutes'].values
T_all = data['T_measured'].values

clean_df = pd.DataFrame({'time': t_all, 'T': T_all}).dropna().sort_values('time').reset_index(drop=True)
t = clean_df['time'].values
T = clean_df['T'].values

# 2. Split Data: First 1/3 for training, remaining 2/3 for extrapolation test
n_total = len(t)
n_train = n_total // 3

t_train = t[:n_train]
T_train = T[:n_train]
t_test = t[n_train:]
T_test = T[n_train:]

print(f"Dataset: T_measured (Ta = {room_temp}°C)")
print(f"Total points: {n_total} | Training (1/3): {n_train} | Test (2/3): {len(t_test)}")

# 3. Fit Newton's Law of Cooling on Training Set only
def newton_model(time_val, k, T0):
    return room_temp + (T0 - room_temp) * np.exp(-k * time_val)

popt, _ = curve_fit(newton_model, t_train, T_train, p0=[0.02, T_train[0]], maxfev=10000)
k_fit, T0_fit = popt
print(f"\nFitted Newton Parameters -> k: {k_fit:.4f} min^-1, T0: {T0_fit:.2f}°C")

# Predict across full time range and calculate test RMSE
T_pred_newton = newton_model(t, k_fit, T0_fit)
newton_test_rmse = np.sqrt(np.mean((T_pred_newton[n_train:] - T_test)**2))
print(f"Newton's Law Test RMSE: {newton_test_rmse:.4f}")

# 4. Fit Polynomials (Degree 1 to 5) on Training Set
poly_rmses = {}
poly_predictions = {}

for deg in range(1, 6):
    coeffs = np.polyfit(t_train, T_train, deg)
    poly_func = np.poly1d(coeffs)
    poly_predictions[deg] = poly_func(t)
    
    poly_test_rmse = np.sqrt(np.mean((poly_func(t_test) - T_test)**2))
    poly_rmses[deg] = poly_test_rmse
    print(f"Polynomial Degree {deg} Test RMSE: {poly_test_rmse:.4f}")

# Find best polynomial for plotting comparison
best_deg = min(poly_rmses, key=poly_rmses.get)

# 5. Plotting Results
plt.figure(figsize=(10, 6))
plt.scatter(t_train, T_train, color='black', label='Training Data (First 1/3)', zorder=5)
plt.scatter(t_test, T_test, color='gray', alpha=0.5, label='Test Data (Remaining 2/3)', zorder=4)

plt.plot(t, T_pred_newton, color='blue', lw=2, label=f"Newton's Law (Test RMSE: {newton_test_rmse:.2f})")
plt.plot(t, poly_predictions[best_deg], color='red', ls='--', lw=1.5, label=f"Best Poly Deg {best_deg} (Test RMSE: {poly_rmses[best_deg]:.2f})")

plt.axvline(t_train[-1], color='orange', linestyle=':', label='Train/Test Split Boundary')
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('Part 3 Extrapolation Test: T_measured')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

### plottingg all polynomial fitting on each dataset


# Load dataset
df_raw = pd.read_csv('cooling_datasets.csv', header=None)
room_temps = df_raw.iloc[0].values
headers = df_raw.iloc[1].values
data = df_raw.iloc[2:].copy()
data.columns = headers
data = data.astype(float)

datasets = [
    'T_noisy_newton_nofan',
    'T_noisy_newton_fan',
    'T_noisy_stefans_law',
    'T_open_measured',
    'T_measured'
]

poly_colors = ['#e41a1c', '#ff7f00', '#984ea3', '#4daf4a', '#a65628']

# Generate a separate figure for each dataset one by one
for idx, col in enumerate(datasets, start=1):
    col_idx = list(headers).index(col)
    Ta = float(room_temps[col_idx])

    clean_df = pd.DataFrame({'time': data['time_minutes'].values, 'T': data[col].values}).dropna().sort_values('time').reset_index(drop=True)
    t = clean_df['time'].values
    T = clean_df['T'].values

    n_train = len(t) // 3
    t_train, T_train = t[:n_train], T[:n_train]
    t_test, T_test = t[n_train:], T[n_train:]

    # Fit Newton's Law model on first 1/3
    def newton_model(time_val, k, T0):
        return Ta + (T0 - Ta) * np.exp(-k * time_val)

    popt, _ = curve_fit(newton_model, t_train, T_train, p0=[0.02, T_train[0]], maxfev=10000)
    T_pred_newton = newton_model(t, *popt)

    # Create a distinct figure separate from the extrapolation test plot
    plt.figure(num=f"All_Models_Fig_{idx}_{col}", figsize=(10, 6))
    plt.scatter(t_train, T_train, color='black', s=25, label='Training Data (1/3)', zorder=6)
    plt.scatter(t_test, T_test, color='gray', alpha=0.5, s=20, label='Test Data (2/3)', zorder=5)
    plt.plot(t, T_pred_newton, color='blue', lw=2.5, label="Newton's Law Model", zorder=4)

    # Fit and plot all polynomial degrees 1 to 5
    for deg, c in zip(range(1, 6), poly_colors):
        coeffs = np.polyfit(t_train, T_train, deg)
        T_pred_poly = np.poly1d(coeffs)(t)
        plt.plot(t, T_pred_poly, color=c, ls='--', lw=1.5, label=f'Polynomial Degree {deg}')

    plt.axvline(t_train[-1], color='darkorange', ls=':', lw=1.5, label='Train/Test Split (1/3)')
    plt.ylim(Ta - 15, np.max(T) + 15)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'All Models (Newton + Poly Degrees 1–5): {col}')
    plt.legend(loc='best', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#Initial Parameters
Ta = 27; #Celsius
T0 = 95; #Celsius
k = 0.02; #Minute Inverse

ts = np.arange(221) # 0,1 ... 220. the times in minutes where we find temperatures

T = lambda t,k: Ta + (T0-Ta)*np.exp(-k*t) # Function to converte t in minutes to T in degrees Celcius

Ts1a = T(ts,k); # List of all temperatures at t = 0,1 ... 220. When no fan

k = 0.03 # Minute inverse, when the fan is on, it cools faster
Ts2a = T(ts,k)

rng = np.random.default_rng(); #Initializing the random number generator

T_noisy_1a = Ts1a + rng.normal(0,0.3, size = Ts1a.size);
T_noisy_2a = Ts2a + rng.normal(0,0.3,size = Ts2a.size);

# Hence our two data sets are T_noisy_1a and T_noisy_2a


# Temperature conversion parameters (Stefan's law requires Kelvin)
# Ta = 27 C -> 300.15 K
# T0 = 95 C -> 368.15 K
Ta_k = 27 + 273.15
T0_k = 95 + 273.15

# Define the derivative dT/dt where t is in minutes and T is in Kelvin
# Note: The user's formula is dT/dt = -1.17111482e-9 * (0.95 * T_body^4 - T_surr^4)
# where T is the absolute temperature (Kelvin).
derv_k = lambda t, T: -1.17111482e-9 * (0.95 * T**4 - Ta_k**4)

# Integration details
t_span = (0, 220)  # minutes
# Time step of 0.01s in minutes
dt = 0.01 / 60.0
t_eval = np.arange(0, 220 + dt, dt)

# Solve the ODE using RK45 method (explicit Runge-Kutta of order 5(4))
sol = solve_ivp(derv_k, t_span, [T0_k], method='RK45', t_eval=t_eval)

# Convert the results back to Celsius for comparison
T_celsius = sol.y[0] - 273.15
t_minutes = sol.t

# Display summary and plot the results
print(f"Solved successfully: {sol.success}")
print(f"Final temperature after 220 mins: {T_celsius[-1]:.4f} C")

plt.figure(figsize=(8, 5))
plt.plot(t_minutes, T_celsius, label='Stefan\'s Law (RK45)', color='red')
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('Numerical Solution to Stefan\'s Law Cooling using RK45')
plt.grid(True)
plt.legend()
plt.show()

#apparently the waviness of the plot is some matplotlib error in visualization becuase point density is too high.

# Converting RK45 Solution back into the readings we want (every minute)
Ts1b = T_celsius[::6000];

#Adding independent random gaussian noise
T_noisy_1b = Ts1b + rng.normal(0,0.3,size = Ts1b.size);


# Create a DataFrame containing all datasets aligned with their minutes
df_share = pd.DataFrame({
    'time_minutes': ts,  # 0, 1, ..., 220
    'T_noisy_newton_nofan': T_noisy_1a,   # Newton's Law of Cooling, k=0.02 (No Fan) + noise
    'T_noisy_newton_fan': T_noisy_2a,     # Newton's Law of Cooling, k=0.03 (With Fan) + noise
    'T_noisy_stefans_law': T_noisy_1b     # Stefan's Law (RK45) + noise
})

# Save to a highly compatible CSV file
csv_filename = 'cooling_datasets.csv'
df_share.to_csv(csv_filename, index=False)
print(f"Saved CSV to: {csv_filename}")

# Alternatively, save to a compressed NumPy file if you want to keep them as arrays
npz_filename = 'cooling_datasets.npz'
np.savez_compressed(npz_filename,
                    time=ts,
                    T_noisy_1a=T_noisy_1a,
                    T_noisy_2a=T_noisy_2a,
                    T_noisy_1b=T_noisy_1b)
print(f"Saved compressed NumPy archive to: {npz_filename}")

# Display the first few rows of the generated DataFrame
print(df_share.head())

import numpy as np;
import matplotlib.pyplot as plt;

#Euler
def Euler (function,x0,x_end,y0,h):
  steps=int((x_end-x0)/h)
  x_array=np.linspace(x0,x_end,steps)
  y_array=[y0]
  for i in range(len(x_array)-1):
    y_array.append(y_array[i]+h*function(x_array[i],y_array[i]))
  return x_array,y_array

#RK4 - Runge Kutta
def rk4 (function,x0,x_end,y0,h):
  steps=int((x_end-x0)/h)
  x_array=np.linspace(x0,x_end,steps)
  y_array=[y0]
  for i in range(len(x_array)-1):
    k1 = h * function(x_array[i], y_array[i])
    k2 = h * function(x_array[i] + h / 2, y_array[i] + k1 / 2)
    k3 = h * function(x_array[i] + h / 2, y_array[i] + k2 / 2)
    k4 = h * function(x_array[i] + h, y_array[i] + k3)
    y_array.append(y_array[i]+(k1 + 2 * k2 + 2 * k3 + k4) / 6)
  return x_array,y_array

t0=0
t_end=220
Ta = 27; #Celsius
T0 = 95; #Celsius
k = 0.02; #Minute Inverse
f=lambda t,T: -k*(T-Ta)

h_values= np.logspace(-4,1,10)

euler_errors = []
rk4_errors = []

for h in h_values:
  t_array_euler, T_array_euler = Euler(f,t0,t_end,T0,h)
  t_array_rk4, T_array_rk4 = rk4(f,t0,t_end,T0,h)

  t_actual = np.arange(len(t_array_euler))*h

  T_array_actual = Ta+(T0-Ta)*np.exp(-k*t_actual) # Analytical values for corresponding times

  # Absolute values
  euler_errors.append(np.max(np.abs(T_array_euler-T_array_actual)))
  rk4_errors.append(np.max(np.abs(T_array_rk4-T_array_actual)))

for i in range(len(h_values)):
  print(f"Error for h = {h_values[i]}: Euler = {euler_errors[i]}, RK4 = {rk4_errors[i]}")
  
plt.loglog(h_values, euler_errors, label='Euler', color='red')
plt.loglog(h_values, rk4_errors, label='RK4', color='green')

plt.xlabel('Step Size (h)')
plt.ylabel('Absolute Error')
plt.title('Error vs Step Size')
plt.grid(True)
plt.legend()
plt.savefig("Error Step Size Analysis")
plt.show()
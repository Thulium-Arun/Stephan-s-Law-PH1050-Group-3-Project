import numpy as np
import matplotlib.pyplot as plt

Ta = 27; #Celsius
T0 = 95; #Celsius
k = 0.02; #Minute Inverse

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

#Running the two methods for this particular scenario
t0=0
t_end=220
f=lambda t,T: -k*(T-Ta)
t_array_euler,T_array_euler=Euler(f,t0,t_end,T0,5)
t_array_rk4,T_array_rk4=rk4(f,t0,t_end,T0,5)
#Analytically
T_array_analytical=Ta+(T0-Ta)*np.exp(-k*t_array_euler)

#Plotting the different graphs
plt.plot(t_array_euler,T_array_euler,'o', label='Euler', color='red')
plt.plot(t_array_rk4,T_array_rk4,'x',label='RK4', color='green')
plt.plot(t_array_euler, T_array_analytical, label='Analytical', color='blue')
plt.plot(rk4(lambda t,T: -0.0001*(T-Ta),t0,220,T0,5)[0], rk4(lambda t,T: -0.0001*(T-Ta),t0,220,T0,5)[1], label = 'k=0.0001',color='orange') #graph for k being close to 0
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('Graph of Temperature against time')
plt.grid(True)
plt.legend()
plt.savefig("Euler,RK4 Plot.png")
plt.show()

#temperature for large time frames
print("\n Across a large time frame (1000 seconds), the value of t becomes",rk4(f,t0,1000,T0,5)[1][-1])



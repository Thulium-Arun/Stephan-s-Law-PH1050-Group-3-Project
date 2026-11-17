import numpy as np

Ta = 27
T0 = 95

T = lambda t,k: Ta + (T0-Ta)*np.exp(-k*t) # Function to convert t in minutes to T in degrees Celcius

def bisection(T,k,a,b,target = 0,tol = 1e-6):
  """
  Does Binary search over remaining interval (a,b) to find a point where T(x) < tol
  Recursively runs
  Guarentees a solution, but is not always optimal
  Uses intermediate value theorom for continuous functions, requiring a root to be
  present between a and b if f(a) and f(b) are of opposite signs
  """

  m = (a + b)/2;

  if (-tol <= T(m,k) - target <= tol):
    return m;
  elif ((T(a,k)-target)*(T(m,k)-target) < 0):
    return bisection(T,k,a,m,target,tol);
  else:
    return bisection(T,k,m,b,target,tol);

# It's written k = 0.02 without fan and 0.03 with. Later in the question it says k doubles with fan.
# Using k = 0.03 with the fan.

#Finding time taken to get to 90*C without fan
t_90_wof = bisection(T,0.02,0,220,90)
#Finding time taken to get to 40*C without fan
t_40_wof = bisection(T,0.02,0,220,40)

print(f"Time taken to cool from 90*C to 40*C without fan = {t_40_wof-t_90_wof} minutes")


#Finding time taken to get to 90*C with fan
t_90_wf = bisection(T,0.03,0,220,90)
#Finding time taken to get to 40*C with fan
t_40_wf = bisection(T,0.03,0,220,40)

print(f"Time taken to cool from 90*C to 40*C with fan = {t_40_wf-t_90_wf} minutes")

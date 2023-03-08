import numpy as np
import pandas as pd

### All the parameters for the fridges and cables

#--------------------------------------------------------------------
### Fridges

### 100 qubits fridge
fridge_100q = {'labels': ['50K', '4K', 'Still', 'CP', 'MXC'],
               'temps': [35, 2.85, 0.882, 0.082, 0.006],
               'lengths': [0.2, 0.290, 0.25, 0.170, 0.140],
               'cooling_p_base': [0, 0, 0, 0, 0]}

### Our Fridge
fridge_ours = {'labels': ['50K', '4K', 'Still', 'CP', 'MXC'],
               'temps': [46, 3.94, 1.227, 0.150, 0.020],
               'lengths': [0.236, 0.337, 0.3, 0.15, 0.222],
               'cool_p': [10, 0.5, 30e-3, 300e-6, 20e-6]}

### Cooling Power vs. Temperature
def t_maker(point1, point2):

    def t_stage(power):
        m = (point1[1] - point2[1]) / (point1[0] - point2[0])
        return m * (power - point1[0]) + point1[1]
    return t_stage

#point for interpolation of the temperature (excluding the still plate)
points = [
    ((5, 41.9), (10, 46)),
    ((0, 3.07), (0.35, 3.7)),
    ((0, 0.080), (300e-6, 0.148)),
    ((0, 0.007), (20e-6, 0.019))
]

t_stages = [t_maker(p[0], p[1]) for p in points]

#------------------------------------------------------------------
### Cable Diameters
diameters = {'219': [0.510e-3, 1.67e-3, 2.19e-3],
             '119': [0.287e-3, 0.94e-3, 1.19e-3]}

cs_areas = {key: [np.pi*(diameters[key][0]/2)**2, 
                  np.pi*((diameters[key][1]/2)**2-(diameters[key][0]/2)**2),
                  np.pi*((diameters[key][2]/2)**2-(diameters[key][1]/2)**2)] for key in diameters.keys()}

#------------------------------------------------------------------
### Thermal Conductivity

# Helper Functions
def gen_therm_cond(fit_const, low_lim):
    # general thermal conductivity function
    # returns the function that gives thermal conductivity for a material based on the 
    # NIST fit constants and a linear extrapolation to zero below the lower limit of the equation

    def therm_cond(T):
        # casting into a numpy array - the calculation depends on numpy functionality / syntax.
        # the use of numpy avoids the use of for loops / iteration (which are incredibly slow) 
        # when the input is a list
        T = np.array([T]).astype('float64')
        T = np.reshape(T, T.size)

        # output values starts as a copy of the input values
        # a copy is used so that the elements that correspond to T > low_lim or T < low_lim can 
        # be indexed using the original input T
        out = T.copy()
        
        # when T < low_lim, use linear extrapolation to 0
        out[T <= low_lim] = (
            T[T <= low_lim] / low_lim *
            10**sum([fit_const[i]*(np.log10(low_lim)**i) for i in range(len(fit_const))])
        )

        # when T > 4, calculate based on the fit function
        out[T > 4] = (
            10**sum([fit_const[i]*(np.log10(T[T > 4])**i) for i in range(len(fit_const))])
        )
        return out

    return therm_cond

def copper_tc(T):
        # casting into a numpy array - the calculation depends on numpy functionality / syntax.
        # the use of numpy avoids the use of for loops / iteration (which are incredibly slow) 
        # when the input is a list
        T = np.array([T]).astype('float64')
        T = np.reshape(T, T.size)

        # output values starts as a copy of the input values
        # a copy is used so that the elements that correspond to T > low_lim or T < low_lim can 
        # be indexed using the original input T
        out = T.copy()
        
        a = 2.3797
        b = -0.4918
        c = -0.98615
        d = 0.13942
        e = 0.30475
        f = -0.019713
        g = -0.046897
        h = 0.0011969
        i = 0.0029988

        # when T < low_lim, use linear extrapolation to 0
        out[T <= 4] = (
            T[T <= 4] / 4 *
            10**(  (a + c*4**0.5 + e*4 + g*4**1.5 + i*4**2) 
                    / (1 + b*4**0.5 + d*4 + f*4**1.5 + h*4**2) )
        )

        # when T > 4, calculate based on the fit function
        out[T > 4] = 10**(  (a + c*T[T > 4]**0.5 + e*T[T > 4] + g*T[T > 4]**1.5 + i*T[T > 4]**2) 
                          / (1 + b*T[T > 4]**0.5 + d*T[T > 4] + f*T[T > 4]**1.5 + h*T[T > 4]**2) )
        return out

def tc_coaxco(val, diameter):
    """
    Extrapolates the thermal conductivity between 0 and 50K based on the data pair given 
    (of the form (T, lambda))
    (The extrapolation assumes a linear thermal conductivity between 0 and 50K 
    with lambda = 0 at 0K)

    diameter in mm
    """
    a = cs_areas[diameter][0]+cs_areas[diameter][2]
    def func(T):
        return (val/100/a)/4 * T

    return func

def c_att_coaxco(points):

    def func(f):
        x_points = [0.5, 1, 5, 10, 20]
        for i in range(len(points)-1):
            if f < x_points[i+1]:
                m = (
                    (points[i] - points[i+1]) 
                    / (x_points[i] - x_points[i+1])
                )
                return m * (f - x_points[i]) + points[i]
        else:
            return None

    return func

# Helper data
SS_fit = [-1.4087, 1.3982, 0.2543, -0.6260, 0.2334,  0.4256, -0.4658, 0.1650, -0.0199]
PTFE_fit = [2.7380, -30.677, 89.430, -136.99, 124.69, -69.556, 23.320, -4.3135, 0.33829]

# Values
therm_cond_nist = {'SS': gen_therm_cond(SS_fit, 4),
              'PTFE': gen_therm_cond(PTFE_fit, 4),
              'Cu': copper_tc}

therm_cond_coaxco = {'119-AgBeCu-BeCu':tc_coaxco(1.74e-4, '119'),
                     '119-NbTi-NbTi':tc_coaxco(7.54e-6, '119'),
                     '119-SS-SS':tc_coaxco(1.32e-5, '119'),
                     '119-AgSS-SS':tc_coaxco(9.95e-5, '119'),
                     '119-CuNi-CuNi':tc_coaxco(1.74e-5, '119'),
                     '119-AgCuNi-CuNi':tc_coaxco(1.04e-4, '119'),
                     '119-BeCu-BeCu':tc_coaxco(9.10e-5, '119'),
                     '219-NbTi-NbTi':tc_coaxco(2.64e-5, '219'),
                     '219-CuNi-CuNi':tc_coaxco(6.30e-5, '219'),
                     '219-AgCuNi-CuNi':tc_coaxco(2.18e-4, '219'),
                     '219-SS-SS':tc_coaxco(4.3e-5, '219'),
                     '219-AgSS-SS':tc_coaxco(2.02e-4, '219'),
                     '219-BeCu-BeCu':tc_coaxco(2.96e-4, '219'),
                     '219-AgBeCu-BeCu':tc_coaxco(4.88e-4, '219')
}

#--------------------------------------------------------------
### Cable Attenuation

c_att_300_coaxco = {'119-AgBeCu-BeCu':c_att_coaxco([1.0, 1.4, 3.1, 4.4, 6.3]),
                    '119-NbTi-NbTi':c_att_coaxco([5.3, 7.5, 16.9, 24.0, 34.1]),
                    '119-SS-SS':c_att_coaxco([5.3, 7.4, 16.6, 23.5, 33.3]),
                    '119-AgSS-SS':c_att_coaxco([1.8, 2.6, 5.8, 8.2, 11.6]),
                    '119-CuNi-CuNi':c_att_coaxco([3.8, 5.4, 12.0, 17.0, 24.0]),
                    '119-AgCuNi-CuNi':c_att_coaxco([1.5, 2.1, 4.7, 6.7, 9.5]),
                    '119-BeCu-BeCu':c_att_coaxco([1.6, 2.3, 5.1, 7.3, 10.5]),
                    '219-NbTi-NbTi':c_att_coaxco([3.0, 4.3, 9.6, 13.6, 19.4]),
                    '219-CuNi-CuNi':c_att_coaxco([2.4, 3.4, 7.6, 10.8, 15.5]),
                    '219-AgCuNi-CuNi':c_att_coaxco([0.8, 1.2, 2.7, 3.8, 5.3]),
                    '219-SS-SS':c_att_coaxco([3.0, 4.2, 9.4, 13.5, 19.2]),
                    '219-AgSS-SS':c_att_coaxco([1.0, 1.5, 3.3, 4.6, 6.5]),
                    '219-BeCu-BeCu':c_att_coaxco([0.9, 1.3, 2.9, 4.1, 5.8]),
                    '219-AgBeCu-BeCu':c_att_coaxco([0.6, 0.8, 1.8, 2.5, 3.5])
}

c_att_4_coaxco = {'119-AgBeCu-BeCu':c_att_coaxco([0.3, 0.5, 1.1, 1.5, 2.2]),
                  '119-NbTi-NbTi':c_att_coaxco([0, 0, 0, 0, 0]),
                  '119-SS-SS':c_att_coaxco([3.3, 4.7, 10.4, 14.7, 20.8]),
                  '119-AgSS-SS':c_att_coaxco([0.8, 1.2, 2.6, 3.7, 5.2]),
                  '119-CuNi-CuNi':c_att_coaxco([2.9, 4.1, 9.1, 12.9, 18.3]),
                  '119-AgCuNi-CuNi':c_att_coaxco([0.7, 1.0, 2.3, 3.3, 4.6]),
                  '119-BeCu-BeCu':c_att_coaxco([1.3, 1.8, 4.0, 5.6, 7.9]),
                  '219-NbTi-NbTi':c_att_coaxco([0,0,0,0,0]),
                  '219-CuNi-CuNi':c_att_coaxco([1.6, 2.3, 5.1, 7.2, 10.2]),
                  '219-AgCuNi-CuNi':c_att_coaxco([0.4, 0.6, 1.3, 1.8, 2.6]),
                  '219-SS-SS':c_att_coaxco([1.9, 2.6, 5.9, 8.3, 11.7]),
                  '219-AgSS-SS':c_att_coaxco([0.5, 0.7, 1.5, 2.1, 2.9]),
                  '219-BeCu-BeCu':c_att_coaxco([0.7, 1.0, 2.2, 3.2, 4.5]),
                  '219-AgBeCu-BeCu':c_att_coaxco([0.2, 0.3, 0.6, 0.9, 1.2])
}

###----------------------------------------------
# Material Resistivity

rho_coaxco = {'SS':72e-8,
       'CuNi':37.5e-8,
       'Ni':14.6e-8,
       'Cu':1.7e-8,
       'Ag':1.642e-8}

cable_rho_coaxco = {'119-AgBeCu-BeCu':rho_coaxco['Cu'],
                  '119-NbTi-NbTi':0,
                  '119-SS-SS':rho_coaxco['SS'],
                  '119-AgSS-SS':rho_coaxco['SS'],
                  '119-CuNi-CuNi':rho_coaxco['CuNi'],
                  '119-AgCuNi-CuNi':rho_coaxco['CuNi'],
                  '119-BeCu-BeCu':rho_coaxco['Cu'],
                  '219-NbTi-NbTi':0,
                  '219-CuNi-CuNi':rho_coaxco['CuNi'],
                  '219-AgCuNi-CuNi':rho_coaxco['CuNi'],
                  '219-SS-SS':rho_coaxco['SS'],
                  '219-AgSS-SS':rho_coaxco['SS'],
                  '219-BeCu-BeCu':rho_coaxco['Cu'],
                  '219-AgBeCu-BeCu':rho_coaxco['Cu']}

###---------------------
# Aggregated coaxco params
coax_co_params = {key: {'d':diameters[key[0:3]],
                        'tc':therm_cond_coaxco[key],
                        'att_300':c_att_300_coaxco[key],
                        'att_4':c_att_4_coaxco[key],
                        'rho':cable_rho_coaxco[key]}
                  for key in therm_cond_coaxco.keys()}

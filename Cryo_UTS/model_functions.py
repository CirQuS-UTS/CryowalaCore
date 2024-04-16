import numpy as np
import pandas as pd
from param_functions import *

### All the functions for the model

#---------------------------------------------------------------------------
### HEAT LOADS

def passive_load(stage_labels, diameters, lengths, therm_cond, therm_scheme, stage_temps):
    '''
    Returns a Series object containing the Power Loads dumped on each stage from this cable

    Parameters
        stage_labels - list
            list of strings naming the stage labels (purely for readability of the output)
        diameters - list
            a list of the diameters of the inner pin, dielectric and outer conductor 
            of the cable in that order
        lengths - list
            a list of the lengths of the cable between stage i-1 and i (first index gives the 
            length from room temp to the first stage)
        therm_cond - list of functions TODO: (TC Coaxco for default cables)
            thermal conductivity as separate functions of temperature for the inner pin,
            dielectric, and outer conductor in that order
        therm_scheme - list of lists
            list of lists of booleans denoting whether thermalisation scheme of the cable.
            the first list shows at which stages the inner pin is thermalised
            the second shows at which stages the dielectric is thermalised
            the third shows at which stages the outer conductor is thermalised
        stage_temps - list
            a list of the steady state temperatures of the stages when no heat load is present
    '''
    # The use of a Series object is mainly for readability. Series was preferred over dict as the 
    # values can be more easily extracted as a numpy array and multiplying/addition with the 
    # values is also much easier

    # initiate the loads as 0
    loads = pd.Series(data=np.zeros(len(stage_labels)), 
                        index=stage_labels, 
                        name='Passive Load (W)')

    # adding room temperature to the list of temps
    stage_temps = [300] + stage_temps

    #calculating the cross sectional areas
    cs_areas = [np.pi*(diameters[0]/2)**2,                          #inner
                np.pi*((diameters[1]/2)**2-(diameters[0]/2)**2),    #dielectric
                np.pi*((diameters[2]/2)**2-(diameters[1]/2)**2)]    #outer

    #iterating through the three cable sections starting from inner
    for c in range(3):
        # denoting that room temperature is thermalised
        temp_therm_scheme = [True] + therm_scheme[c]

        #denoting the index of the current anchor
        anchor = 0
        length = 0

        # iterating through the stages (index 0 refers to room temp)
        for i in range(1, len(stage_temps)):
            #adding the length of this stage for the next calculation
            # (index is off by 1 since stage_temps has room temp at index 0)
            length += lengths[i-1]
            
            #if not thermalised at this stage, move to next stage
            if not temp_therm_scheme[i]:
                continue
            
            # taking the discrete intervale of the thermal conductivity between the previous 
            # thermal anchor to this anchor
            T = np.linspace(start=stage_temps[i], 
                            stop=stage_temps[anchor], 
                            num=200, 
                            endpoint=False)
            dT = T[1]-T[0]

            # load from this section at this stage 
            # (index is off by 1 since stage_temps has room temp at index 0)
            loads[i-1] += cs_areas[c] / length * np.sum(dT*therm_cond[c](T))

            anchor = i
            length = 0

    return loads

#dumping load due to cable attenuation on lower stage, regardless of thermalisation
def active_load_AC(stage_labels, signal_p, signal_f, att, cable_att, lengths):
    """
    Function giving the active loads assuming a continuous wave input (single frequency)

    Parameters:
        stage_labels - list of strings
            list of strings naming the stage labels (purely for readability of the output)
        signal_p - float
            the power of the signal at the input of the fridge (in Watts)
        signal_f - float
            the frequency of the CW signal (in Hz)
        att - list
            a list giving the attenuation that the cable experiences at every stage
        cable_att - function TODO: C_att_4
            the attenuation of the cable as a function of frequency (in Hz)
        lengths - list
            the list of the cable lengths between stages
    """
        
    #initialising the loads as 0
    loads = pd.Series(data=np.zeros(len(stage_labels)), 
                        index=stage_labels,
                        name='Active Power (W)')
    
    #obtaining the cable attenuation (dB/m) for this particular signal frequency        
    eff_cable_att = cable_att(signal_f)

    # iterate through the stages
    for i in range(len(stage_labels)):
        total_att = eff_cable_att*lengths[i] + att[i]
        loads[i] = signal_p * (1 - 10**(-total_att/10))
        signal_p = signal_p * 10**(-total_att/10)

    return loads

def active_load_DC(i_in, stage_labels, att, cable_rho, lengths, diameters):
    """
    Returns the heat loads in the cables due to DC signals in the flux lines

    Parameters
        i_in - float
            Current at the input of the fridge
                Note: to get i_in from a supply voltage, i_in = v_in / 50
                (as the effective impedance of the entire circuit should ideally match the 
                characteristic impedance of the load)
        stage_labels - list
            list of strings nameing hte stage labels (purely for output formatting)
        att - list
            a list giving the attenuation that the cable experiences at every stage 
            (in units of dB)
        cable_rho - float
            the cable conductor resistivity at RT
        lengths - list
            the list of the cable lengths between stages
        diameters - list
            a list of the diameters of the inner pin, dielectric and outer conductor 
            of the cable in that order
    """

    #initialising the loads as 0
    loads = pd.Series(data=np.zeros(len(stage_labels)), 
                        index=stage_labels,
                        name='Active Power (W)')
    
    # initialising the the current at each stage
    currents = pd.Series(data=np.zeros(len(stage_labels)+1), 
                        index=stage_labels + ['Device'],
                        name='Current (A)')

    currents[0] = i_in

    # Calculating the currents and attenuator loads
    for i in range(len(stage_labels)):
        G = 10**(-att[i]/20)
        loads[i] += currents[i]**2 * (1-G**2) * 50
        currents[i+1] = currents[i] * G

    ## Calculating resistances and cable loads

    A_inner = np.pi*(diameters[0]/2)**2

    R_CP =  (
        0.42
        * (cable_rho / 7.2e-7)
        * (lengths[-2]/0.17)
        * (np.pi*(0.511e-3/2)**2 / A_inner)
    )
    R_MXC =  (
        0.15
        * (cable_rho / 7.2e-7)
        * (lengths[-1]/0.14)
        * (np.pi*(0.511e-3/2)**2 / A_inner)
    )

    #adding the cable loads
    loads[-2] = loads[-2] + currents[-3]**2 * R_CP
    loads[-1] = loads[-1] + currents[-2]**2 * R_MXC

    return loads

#----------------------------------------------------------------------------
### NOISE
def noise_photons(temp, att, cable_att, lengths, f=6e9, stage_labels=fridge_ours['labels']):
    """
    Returns the noise in photons per second per Hz at all stages for a given input frequency f.   
    This function essentially serves as the noise spectral density function in terms of 
    photon number.

    Parameters:
        temp - list
            a list of the steady state temperatures of the stages when no heat load is present
        att - list
            a list giving the attenuation that the cable experiences at every stage
        cable_att - float
            the attenuation of the cable material at the main signal frequency (in units of dB/m)
        lengths - list
            the list of the cable lengths between stages in meters
        f - float / numpy array
            input frequency for the noise spectral density
        stage_labels - list
            list of strings naming the stage labels (purely for formatting the output)
    """

    h = 6.626e-34
    k_B = 1.381e-23
    n_BE = lambda T, f: 1 / ( np.exp( h*f / (k_B*T) ) - 1 )

    n_temp = [n_BE(t, f) for t in temp]
    att_total = np.array(att) + cable_att * np.array(lengths)

    for i in range(5):
        n_temp[i] = n_temp[i] * (1-10**(-att_total[i]/10)) * (10**(-np.sum(att_total[-1:i:-1])/10))
    
    n_temp.insert(0, n_BE(300, 6e9)*10**(-np.sum(att_total)/10))

    return pd.Series(
        data=n_temp, 
        index=['RT'] + stage_labels,
        name='Photon Flux Spectral Density (No. Photons/s/Hz)',
        dtype='object'
    ) 

def noise_current(temp, att, cable_att, lengths, R=50, f=6e9, stage_labels=fridge_ours['labels']):
    """
    Returns the noise in photons per second per Hz at all stages for a given input frequency f.   
    This function essentially serves as the noise spectral density function in terms of 
    photon number.

    Parameters:
        temp - list
            a list of the steady state temperatures of the stages when no heat load is present
        att - list
            a list giving the attenuation that the cable experiences at every stage
        cable_att - float
            the attenuation of the cable material at the main signal frequency (in units of dB/m)
        lengths - list
            the list of the cable lengths between stages in meters
        R - float
            impedance of the transmission lines
        f - float / numpy array
            input frequency for the noise spectral density
        stage_labels - list
            list of strings naming the stage labels (purely for formatting the output)
    """

    h = 6.626e-34
    
    return np.sqrt(noise_photons(temp, att, cable_att, lengths, f=6e9, stage_labels=fridge_ours['labels']) *4*h*f/R)

def noise_voltage(temp, att, cable_att, lengths, R=50, f=6e9, stage_labels=fridge_ours['labels']):
    """
    Returns the noise in photons per second per Hz at all stages for a given input frequency f.   
    This function essentially serves as the noise spectral density function in terms of 
    photon number.

    Parameters:
        temp - list
            a list of the steady state temperatures of the stages when no heat load is present
        att - list
            a list giving the attenuation that the cable experiences at every stage
        cable_att - float
            the attenuation of the cable material at the main signal frequency (in units of dB/m)
        lengths - list
            the list of the cable lengths between stages in meters
        R - float
            impedance of the transmission lines
        f - float / numpy array
            input frequency for the noise spectral density
        stage_labels - list
            list of strings naming the stage labels (purely for formatting the output)
    """

    h = 6.626e-34
    
    return np.sqrt(noise_photons(temp, att, cable_att, lengths, f=6e9, stage_labels=fridge_ours['labels']) *4*h*f*R)

#TODO: Use drive and flux for now
#output as floats (the stage_noise function itself serves like a spectral density function for all stages)
def drive_noise(f, stage_labels, stage_temps, att, cable_att, lengths):

    """
    Returns the noise in photons per second per Hz at all stages for a given input frequency f.   
    This function essentially serves as the noise spectral density function in terms of 
    photon number.

    Parameters:
        f - float / numpy array
            input frequency for the noise spectral density
        stage_labels - list
            list of strings naming the stage labels (purely for formatting the output)
        stage_temps - list
            a list of the steady state temperatures of the stages when no heat load is present
        att - list
            a list giving the attenuation that the cable experiences at every stage
        cable_att - list 
            the attenuation of the cables at the main signal frequency (in units of dB/m)
        lengths - list
            the list of the cable lengths between stages in meters
    """
    #initialise the output, the zeroes will get replaces
    noise = pd.Series(data=np.zeros(len(stage_labels)+1), 
                        index=['RT'] + stage_labels,
                        name='Photon Flux Spectral Density (No. Photons/s/Hz)',
                        dtype='object')  
    
    #defining the Bose Einstein distribution
    h = 6.626e-34
    k_B = 1.381e-23 
    n_BE = lambda T, f: 1/(np.exp(h*f/(k_B*T))-1)

    noise[0] = n_BE(300, f)

    for i in range(1, len(noise.values)):
        # index i is 1 greater in this loop for cable attributes like length etc. as RT has been 
        # made the 0 index
        total_att = cable_att[i-1]*lengths[i-1] + att[i-1]

        # calculating the noise per second per Hz at stage i based on the noise at stage i-1 
        # using the formula from the paper
        noise[i] = (
            noise[i-1]/(10**(total_att/10))
            + (1-1/10**(total_att/10)) * n_BE(stage_temps[i-1],f)
        )

    return noise


def flux_noise_old(f, stage_labels, stage_temps, noise_i_in, att):
    """
    Returns the Mean Square of the Current noise at every stage. 
    Note: To attain root mean square current fluctuations, the output must be square rooted.
    Note: This function serves as the current noise spectral density function. 
    That is, it returns the value of the current noise spectral density function for 
    an input frequency, f

    Parameters
        f - float
            frequency input for the current noise spectral density function
        stage_labels - list
            list of strings naming the stage labels (purely for formatting the output)
        stage_temps - list
            a list of the steady state temperatures of the stages when no heat load is present
        noise_i_in - float
            the value of the current noise spectral density at the input frequency, f
        att - list
            a list giving the attenuation that the cable experiences at every stage
    """
    #initialise the output, the zeroes will be replaced
    noise = pd.Series(data=np.zeros(len(stage_labels)+1), 
                        index=['RT'] + stage_labels,
                        dtype='object',
                        name='Noise Current (A**2/Hz)')

    noise[0] = noise_i_in

    #defining constants and Bose Einstein Distribution
    h = 6.626e-34
    k_B = 1.381e-23
    I_ms = lambda T, f: h*f / (np.exp(h*f/(k_B*T)) - 1) / 50

    #propagating through the stages and calculating noise
    for i in range(len(stage_labels)):
        A = 10**(att[i]/10)
        noise[i+1] = noise[i]/A + (1-1/A)*I_ms(stage_temps[i], f)

    return noise

def flux_noise(f, stage_labels, stage_temps, att, cable_att, lengths):
    """
    Returns the Mean Square of the Current noise at every stage. 
    Note: To attain root mean square current fluctuations, the output must be square rooted.
    Note: This function serves as the current noise spectral density function. 
    That is, it returns the value of the current noise spectral density function for 
    an input frequency, f

    Parameters:
        f - float / numpy array
            input frequency for the noise spectral density
        stage_labels - list
            list of strings naming the stage labels (purely for formatting the output)
        stage_temps - list
            a list of the steady state temperatures of the stages when no heat load is present
        att - list
            a list giving the attenuation that the cable experiences at every stage
        cable_att - list
            the attenuation of the cables at the main signal frequency (in units of dB/m)
        lengths - list
            the list of the cable lengths between stages in meters

    """
    h = 6.626e-34
    
    return drive_noise(f, stage_labels, stage_temps, att, cable_att, lengths) * h*f/50

#-----------------------------------------------------------------
### SNR

#TODO: IGNORE THIS FUNCTION
def drive_SNR(f,
                p_in,
                stage_labels,
                stage_temps,
                noise_p_in,
                att,
                cable_att,
                lengths):

    """
    Returns the noise in photons per second per Hz at all stages for a given input frequency f.   
    This function essentially serves as the noise spectral density function.

    Parameters:
        f - float / numpy array
            input frequency for the noise spectral density 
            (in units of Hz)
        p_in - float
            the value of the power spectral density of the signal at the frequency f 
            (in units of W/Hz)
        stage_labels - list
            list of strings naming the stage labels (purely for formatting the output)
        stage_temps - list
            a list of the steady state temperatures of the stages when no heat load is present
        noise_p_in - function TODO:
            the value of the noise power spectral density at the input frequency, f
        att - list
            a list giving the attenuation that the cable experiences at every stage
        cable_att - float
            the attenuation of the cables at the main signal frequency (in units of dB/m)
        lengths - list
            the list of the cable lengths between stages in meters
    """
    h = 6.626e-34

    #calculating the noise power per Hz at the frequency f
    noise = h*f*drive_noise(f, stage_labels, stage_temps, noise_p_in, att, cable_att, lengths)
    noise.name = 'Noise Power (W/Hz)'

    ## Calculating signal power
    #initialising values
    signal = pd.Series(data=np.zeros(len(stage_labels)+1), 
                        index=['RT'] + stage_labels,
                        dtype='object',
                        name='Signal Power (W/Hz)')

    signal[0] = p_in

    for i in range(len(stage_labels)):
        A = 10**((cable_att[i]*lengths[i] + att[i]) / 10)
        signal[i+1] = signal[i] / A

    SNR = 10 * np.log10((signal/noise).astype('float64'))
    SNR.name = 'Signal to Noise Ratio (dB)'

    return SNR
import numpy as np
import pandas as pd
import scipy as sp

# the use of series objects to store attribute data is useful for
#   - displaying the information in a readable manner when called by the user
#   - providing an easy means of indexing the values by either
#       - calling the user defined stage_label (sst['CP'] corresponds to the cold plate)
#       - calling the positional index (sst[3] corresponds to the cold plate)
#   - easily converting the data into a list for use in other methods within the objects

class DilutionRefrigerator():

    cables = []     # list of all cables currently present in the fridge

    def __init__(self, stage_labels, sst):
        '''
        Attributes
            stage_labels - list of strings
                list of strings of the stage label names that will be used 
                within the object for better user readability
                note, RT should not be included as a stage
            sst - list of function objects 
            (or dict with keys corresponding to labels specified in stage_labels)
                the steady state temperature values for each stage as functions of
                the power load dumped onto it
        '''

        #TODO conclude whether cable lengths is an attribute given to the fridge
        # to be passed on to every single cable (so the user need not specify every 
        # single time), or should be specified upon instantiation of each cable. The former
        # is convenient for the user, but the latter allows for unique cable lengths should
        # there be some variation in length for any particular cable.

        self.stage_labels = stage_labels #NOTE room temperature should not be included as stage
        
        #TODO assess whether RT should be explicitly included by the user, 
        # or whether the code should add it automatically as it currently does

        # creating an attribute for labels which would be used for information 
        # about things between stages (e.g. cable lengths or thermal conductivity)
        temp_labels = ['RT'] + self.stage_labels
        self.between_labels = [temp_labels[i] + '-' + temp_labels[i+1] for i in range(len(temp_labels)-1)]

        # the use of series to store attribute data is useful for
        #   - displaying the information in a readable manner when called by the user
        #   - providing an easy means of indexing the values by either
        #       - calling the user defined stage_label (sst['CP'] corresponds to the cold plate)
        #       - calling the positional index (sst[3] corresponds to the cold plate)
        #   - easily converting the data into a list for use in other methods within the objects
        self.sst = to_Series(index=self.stage_labels, data=sst)
        # self.cable_lengths = self.to_Series(between_labels, cable_lengths) 

    def add_Cable(self, line_type, **kwargs):
        '''
        Required Parameters:
            diameters
            out_therm
            att
            lengths
            cable_att
            therm_cond
        '''
        
        kwargs['stage_labels'] = self.stage_labels

        # evaluating line type in such a way so that it is not hardcoded and will 
        # work as long as the class exists
        self.cables.append(eval(line_type.capitalize() + 'Line(**kwargs)'))

        # line_types = {'drive': DriveLine,
        #             'flux':FluxLine,
        #             'output': OutputLine,
        #             'pump':PumpLine}

        # self.cables.append(
        #     line_types[line_type](**kwargs)
        # )

    # Each cable object has methods which calculate the metrics, 
    # but yet these values need to be aggregated somehow into something easy to interpret. 
    # For this, we can hard code these aggregating/processingfunctions into the 
    # fridge object (for the time being) since the purpose of the model is to determine 
    # these metrics and present them in a readable manner. The alternative to this is to have
    # the fridge object collect the relevant metric data from all the relevant cables and pass
    # it to the user for use in their own custom aggregate / processing functions. This makes
    # the fridge object more generalised, in the sense that nothing is hardcoded and more features 
    # can be added without affect other functions within the fridge

    def passive_load(self):
        '''
        Calculates the total passive load of all cables in the fridge on all stages
        '''

        # obtaining a list of the steady state temperatures 
        # when 0 additional load from cables is present
        temps = [t(0) for t in self.sst.values]

        # adding room temperature to the list
        temps = [290] + temps
        
        # # instantiating a list of the loads on each stage
        # total_load = pd.Series(data=np.zeros(5), index=self.stage_labels)

        # for cable in self.cables:
        #     total_load += cable.passive_load(temps).values

        # return total_load

        total_load = pd.DataFrame()

        # separating the passive loads of each cable based on the type of drive line it is
        # this code block serves the purpose of formatting the data into a readible data frame
        for cable in self.cables:
            load = cable.passive_load(temps)
            if cable.line_type in total_load.columns:
                total_load[cable.line_type] += load
            else:
                total_load[cable.line_type] = load
        
        total_load['total'] = total_load.sum(axis=1)

        return total_load
        
    # full run of all the metrics that can be calculated in the fridge
    def all_metrics(self):
        pass
    
        
class Cable():

    line_type: str
    metrics: list

    def __init__(self, **kwargs):
        '''
        Required Attributes:
            stage_labels - str 
                the names of the stages of the fridge
            diameters - Series, dict or list of float
                the diameters of the inner pin, dielectric, and outer shield sections of the
                coaxial cable (in meters)
            out_therm - Series, dict or list of boolean
                an array of booleans that states whether the outershield of the cable is 
                thermalised at every stage
            att - Series, dict or list of float
                an array of the attenuation amounts (in decibels) at each stage
            cbl_att - function
                a function that gives the cable attenuation (in decibels per meter) dependent
                on frequency
            lengths - Series, dict or list of floats
                an array of the lengths of the cable between stages (in meters)
            therm_cond - function
                a function that can take any numerical input (float, list, np.array) and 
                returns the thermal conductivities for those temperature values
        '''
        
        # Having each cable contain its own list of the stage labels helps with 
        # formatting the cable attributes into Series objects, which is purely for the 
        # purpose of increasing the readibility of the function outputs for users
        self.stage_labels = kwargs['stage_labels']

        temp_labels = ['RT'] + self.stage_labels
        self.between_labels = [temp_labels[i] + '-' + temp_labels[i+1] for i in range(len(temp_labels)-1)]
        
        self.therm_cond =  kwargs['therm_cond']
        #TODO consider implementing an 'eff_therm_cond' attribute, which pre-calculates the effective 
        # thermal conductivities between the stages to avoid repeating the same calculation when 
        # calculating the passive load for each cable individually. Similarly, there could be some method
        # within the fridge object that identifies cables that have the same properities / attributes and 
        # avoids calculating the same values, instead using the already calculated values.

        # formatting all the attributes into Series objects
        diameter_labels = ['inner', 'dielectric', 'outer']
        self.diameters =   to_Series(index=diameter_labels, data=kwargs['diameters'])
        self.out_therm =   to_Series(index=self.stage_labels, data=kwargs['out_therm'])
        self.att =         to_Series(index=self.stage_labels, data=kwargs['att'])
        self.lengths =     to_Series(index=self.between_labels, data=kwargs['lengths'])
        
        # calculating cross sectional of the cable sections based on diameters
        self.cs_areas = pd.Series(
            {'inner':np.pi*(self.diameters['inner']/2)**2,
             'dielectric':np.pi*((self.diameters['dielectric']/2)**2-(self.diameters['inner']/2)**2),
             'outer':np.pi*((self.diameters['outer']/2)**2-(self.diameters['dielectric']/2)**2)}
        )

        self.cable_att = kwargs['cable_att']

    def eff_att(self, f):
        return self.lengths.values * self.cable_att(f) + self.att

    #passive load is a common function across all cable types
    def passive_load(self, temps):
        '''
        Returns a Series object containing the Power Loads dumped on each stage from this cable

        Parameters
            temps - list
                a list of the steady state temperatures when no heat load is present
        '''

        # To calculate the loads on each stage, we calculate the head load between each stage.
        # Note that the outer and inner conductors are not thermalised at all stages, so we accumulate
        # the unthermalised load as we iterate through the stages and dump it at the next stage where 
        # thermalisation occurs.

        # initiate the loads as 0
        loads = pd.Series(data=np.zeros(len(self.stage_labels)), index=self.stage_labels) 
        acc_outer_load = 0  # accumulating outer load
        acc_inner_load = 0  # accumulating inner load

        # iterating through the stages
        for i in range(len(self.stage_labels)):
            l = self.lengths[i]     # length from stage i-1 to stage i

            #taking a discrete interval of the thermal conductivity over the temperature interval
            x = np.linspace(start=temps[i+1], stop=temps[i], num=100, endpoint=False)
            dx = x[1]-x[0]
            eff_therm_cond = dx * np.sum(self.therm_cond(x)) 

            #calculating the inner and outer heat loads
            acc_outer_load += self.cs_areas['outer'] / l * eff_therm_cond
            acc_inner_load += self.cs_areas['inner'] / l * eff_therm_cond

            # if the outer shield is thermalised, dump the outer load onto this stage
            if self.out_therm[i]:
                loads[i] += acc_outer_load
                acc_outer_load = 0

            # if there is an attenuator (inner pin thermalised), dump inner load onto this stage
            if self.att[i] > 0:
                loads[i] += acc_inner_load
                acc_inner_load = 0

        return loads

class DriveLine(Cable):

    line_type = 'drive'    
    metrics = ['passive_load', 'active_load', 'thermal_noise', 'signal_to_noise_ratio']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    #returns the active load for a given signal input
    def active_load(self, signal_p, signal_f):
        #TODO Consider whether the function should itself determine the heat load dumped
        # by instead taking the signal shape and subsequently determining the power spectral 
        # density function and total power of the signal.
        # For now, this function will just take the power values from the user
        
        loads = pd.Series(data=np.zeros(len(self.stage_labels)), index=self.stage_labels)
        
        eff_att = self.eff_att(signal_f)

        acc_inner_load = 0

        # iterate through the stages
        for i in range(len(eff_att)):
            #calculate load power and attenuated signal power
            acc_inner_load += signal_p * (1 - 10**(-eff_att[i]/10))
            signal_p = signal_p * 10**(-eff_att[i]/10)

            # if there is an attenuator at this stage, dump the load
            if self.att[i] > 0:
                loads[i] += acc_inner_load
                acc_inner_load = 0

        return loads

    #determines the noise temperature at each stage, based on the power of some noisy input
    def thermal_noise(self, noise_p_in, f_in, stage_temps):
        """
        Parameters
            noise_p_in - float
                the power of the input noise at the start of the fridge (in W)
            f_in - float
                the frequency of the drive signal (in Hz)
            stage_temps - list of floats
                the list of temps of each stage
                Note the first value should be the temperature outside of the 
                fridge (typically room temperature). This list should have length:
                len(stage_temps) = len(self.stage_labels) + 1
        """

        h = 6.626e-34
        kb = 1.381e-23

        # bose-einstein distribution
        n_be = lambda T: 1 / (np.exp(h * f_in / kb / T) - 1)

        # Determining the thermal anchor points of the coaxial inner pin, which are important 
        # for determining the thermal gradient within the wire
        

        
        
        l_total = 0

        def T(x):
            # assuming x is a np.array() object

            for pair, l in self.lengths:
                

        # The noise generated in a cable due to its attenuation can be calculated by a Riemann 
        # sum, which divides the wire into a number of sections which function as attenuators
        # The limit of this sum as the number of sections approaches infinity gives the exact
        # value of noise generated. This function computes the riemann sum numerically for a 
        # high enough number of divisions that gives an accurate value
        def cable_noise(L_range, T_anchors, a, n=1000):
            """
            Parameters
                T_range - Tuple of the Temperature Range over which the attenuation is occuring 
                T1 - Temperature of colder end of cable
                L - Length of Cable
                a - cable attenuation (dB/m)
                n - number of wire divisions used for the Riemann sum
            """
            # kb = 1.3806e-23

            #TODO incorporate an attribute containing the thermal gradient of the wire given 
            # its own thermal conductivity and the thermal anchor temperatures. For now, the
            # the solution is hardcoded assuming a linear dependence of thermal conductivity 
            # on temperature (which is only mostly accurate for low temperatures)

            # sqrt thermal gradient based on linear dependence of thermal 
            # conductivity on temperature
            T = lambda x: np.sqrt(T_anchors[0]**2 - x*(T_anchors[0]**2-T_anchors[1]**2)/L)

            #linear thermal gradient
            # T = lambda x: (T1-T0)/L * x + T0

            dx = L/n
            x = np.linspace(0+dx, L, num=n, endpoint=True)
    
            # terms = kb * T(x) * (1 - 10**(-a*dx/10)) * 10**(-a/10*(L-x)) # Noise Power
            terms = n_be(T(x)) * (1 - 10**(-a*dx/10)) * 10**(-a/10*(L-x)) # Noise Photon Number

            return np.sum(terms)

        noise_by_stage = pd.Series(index=self.stage_labels, data=np.zeros(len(self.stage_labels)))
        # noise_p = noise_p_in          # Noise Power
        noise_p = n_be(noise_p_in/kb)   # Noise Photon Number

        for i in range(len(self.stage_labels)):
            # passing through cable
            noise_p = (noise_p*(10**(-self.cable_att(f_in)*self.lengths[i]/10)) 
                      + cable_noise(T0=stage_temps[i],
                                    T1=stage_temps[i+1],
                                    L=self.lengths[i],
                                    a=self.cable_att(f_in),
                                    n=1000)
                      )

            #passing through attenuator
            noise_p = (noise_p*(10**(-self.att[i]/10))
                    #    + kb * stage_temps[i+1] * (1 - 10**(-self.att[i]/10)) # Noise Power
                       + n_be(stage_temps[i+1]) * (1 - 10**(-self.att[i]/10)) # Noise Photon Number
                      )

            noise_by_stage[i] = noise_p
            
        ### Calculation by Noise Power
        # noise_by_stage = pd.DataFrame(data={'Noise Power (W)': noise_by_stage,
        #                                     'Noise Temperature (K)': noise_by_stage/kb,
        #                                     'Noise Photon Number': 1/(np.exp(h*f_in/noise_by_stage)-1)}, 
        #                                     index=self.stage_labels)

        ### Calculation by Noise Photon Number
        noise_by_stage = pd.DataFrame(data={'Noise Power (W)': h*f_in/np.log(1/noise_by_stage + 1),
                                            'Noise Temperature (K)': h*f_in/np.log(1/noise_by_stage + 1)/kb,
                                            'Noise Photon Number': noise_by_stage}, 
                                            index=self.stage_labels)

        return noise_by_stage

    def signal_to_noise_ratio(self, snr_in, f_in):
        pass
class FluxLine(Cable):
    pass

class OutputLine(Cable):
    pass

class PumpLine(Cable):
    pass

# helper functions --------------------------------------------

# handling for user inputs of various data types
def to_Series(data, index):
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, dict):
        return pd.Series(data)
    else:
        return pd.Series(data=data, index=index)
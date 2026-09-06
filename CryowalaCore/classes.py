"""
#object structure
# cryostat
    - stages
    - default operating temperatures
    - stage temperature responses
    - distances between stages

# cable type
    - frequency
    - signal type
    - signal power
    - signal duty cycle
    - type name
    - attenuation
    - assumed thermalisation

# cable segment
    - cable material
        - thermal conductivity
        - resitivity
        - cable diameters
"""

from param_functions import coax_co_params
import pandas as pd
import numpy as np


class Cryostat(object):

    def __init__(self, **kwargs):
        self.cables = {}
        self.__table = {
            '50K': {
                'temperature': 50,
                'length': 0.1,
                'cooling_power': 0.5,
            },
            '4K': {
                'temperature': 4,
                'length': 0.1,
                'cooling_power': 0.5,
            },
            'Still': {
                'temperature': 1,
                'length': 0.1,
                'cooling_power': 0.5,
            },
            "CP": {
                'temperature': 0.1,
                'length': 0.1,
                'cooling_power': 0.5,
            },
            'MXC': {
                'temperature': 0.01,
                'length': 0.1,
                'cooling_power': 0.5,
            },
        }
        #TODO: add setter functions for the properties of the cryostat, so that they can be changed on initialization. 
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
    
    @property
    def stages(self):
        return list(self.__table.keys())

    @property
    def temperature(self):
        return {stage: self.__table[stage]['temperature'] for stage in self.stages}

    @property
    def length(self):
        return {stage: self.__table[stage]['length'] for stage in self.stages}

    @property
    def cooling_power(self):
        return {stage: self.__table[stage]['cooling_power'] for stage in self.stages}

    def add_cable(self, cable_type, quantity=1):
        #TODO potentially utilise the aliasDict object from quanguru to set objects as dict keys
        self.cables[cable_type.name] = {
            'cable': cable_type,
            'quantity': quantity
        }
        cable_type.cryostat = self

class CableType(object):
    #TODO implement a check method to ensure that segments and attenuation are defined for all stages in the cryostat, 
    # before running any modelling

    def __init__(self, **kwargs):
        #TODO implement default values?
        self.name = None
        self.signal_type = None
        self.signal_power = None
        self.duty_cycle = None
        self.frequency = None
        self.attenuation = {}
        self.thermalisation_table = {}
        self.segments = {}
        self.cryostat = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_segment(self, stage, segment):
        if stage in self.cryostat.stages:
            self.segments[stage] = segment
        else:
            raise ValueError("Stage not found in cryostat")

    @property
    def stages(self):
        return self.cryostat.stages

    def passive_load(self):
        #TODO need to consider how to handle lack of inner thermalisation when connecting segments of different materials.
        # This is because it sets the thermal gradient across the inner conductor, 
        # i.e. there is a material interface BETWEEN the fixed thermal anchors

        # initiate the loads as 0
        loads = pd.Series(
            data=np.zeros(len(self.stages)), 
            index=self.stages, 
            name='Passive Load (W)'
        )
        
        #iterating through the three cable sections from inner to outer
        for i, layer in enumerate(['inner', 'dielectric', 'outer']):
            length = 0
            anchor = 300 # room temperature is the first thermal anchor
            for stage, segment in self.segments.items():
                # if the layer is not thermalised at this stage, skip to next stage
                if not self.thermalisation_table[stage][layer]:
                    length += segment.length
                    continue

                #TODO Potentially update the discrete integral to integrate more accurately, 
                # given the order of magnitude differences in temperature

                # taking the discrete integral of the thermal conductivity between thermal anchors
                T = np.linspace(start=self.cryostat.temperatures[stage], 
                                stop=anchor, 
                                num=200, 
                                endpoint=False)
                dT = T[1]-T[0]

                # load from this section at this stage 
                # (index is off by 1 since stage_temps has room temp at index 0)
                loads[stage] += segment.cross_sectional_area(layer) / length * np.sum(dT*segment.thermal_conductivity[layer](T))

                anchor = self.cryostat.temperatures[stage]
                length = 0
    
        return loads

    def active_load_ac(self):
        #TODO consider renaming this to active_load_rf
        #initialising the loads as 0
        loads = pd.Series(
            data=np.zeros(len(self.stages)), 
            index=self.stages, 
            name='Active Load (W)'
        )       

        # carry = 0

        # # iterate through the stages
        # for stage, segment in self.segments.items():
        #     p_att = signal_p * (1 - 10**(-att[i]/10))
        #     p_outer = signal_p * (1 - 10**(-eff_cable_att*lengths[i]/10)) * (dia[0])/(dia[0]+dia[1])
        #     p_inner = signal_p * (1 - 10**(-eff_cable_att*lengths[i]/10)) * (dia[1])/(dia[0]+dia[1])

        #     total_p = p_att + p_outer
        #     if therm_scheme[i]:
        #         total_p += p_inner + carry
        #         carry = 0
        #     else:
        #         carry = p_inner

        #     loads[i] = total_p

        #     total_att = eff_cable_att*lengths[i] + att[i]
        #     signal_p = signal_p * 10**(-total_att/10)

        # return pd.Series(
        #     data=loads, 
        #     index=stage_labels,
        #     name='Active Power (W)'
        # )



class CableSegment(object):

    def __init__(self, **kwargs):
        self.__table = {
            'outer': {
                'thermal_conductivity': None,
                'resistivity': None,
                'diameter': None
            },
            'dielectric': {
                'thermal_conductivity': None,
                'resistivity': None,
                'diameter': None
            },
            'inner': {
                'thermal_conductivity': None,
                'resistivity': None,
                'diameter': None
            }
        }
        self.attenuation = None
        self._material = None

    @property
    def diameter(self):
        return {layer: self.__table[layer]['diameter'] for layer in self.__table.keys()}

    @property.setter
    def diameter(self, map):
        for layer in self.__table.keys():
            self.__table[layer]['diameter'] = map[layer]

    @property
    def thermal_conductivity(self):
        return {layer: self.__table[layer]['thermal_conductivity'] for layer in self.__table.keys()}

    @property.setter
    def thermal_conductivity(self, table):
        for layer in self.__table.keys():
            self.__table[layer]['thermal_conductivity'] = table[layer]

    @property
    def resistivity(self):
        return {layer: self.__table[layer]['resistivity'] for layer in self.__table.keys()}

    @property.setter
    def resistivity(self, table):
        for layer in self.__table.keys():
            self.__table[layer]['resistivity'] = table[layer]

    @property
    def material(self):
        return self._material

    @property.setter
    def material(self, material):
        self._material = material
        self.attenuation = coax_co_params[material]['att_4']
        self.__table = {
            'outer': {
                'thermal_conductivity': coax_co_params[material]['tc'],
                'resistivity': coax_co_params[material]['rho'],
                'diameter': coax_co_params[material]['d'][2]
            },
            'dielectric': {
                'thermal_conductivity': 0,
                'resistivity': 0,
                'diameter': coax_co_params[material]['d'][1]
            },
            'inner': {
                'thermal_conductivity': coax_co_params[material]['tc'],
                'resistivity': coax_co_params[material]['rho'],
                'diameter': coax_co_params[material]['d'][0]
            }
        }

    def cross_sectional_area(self, layer):
        if layer not in ['outer', 'dielectric', 'inner']:
            raise ValueError("Layer must be one of 'outer', 'dielectric', or 'inner'")
        if layer == 'outer':
            return np.pi*((self.diameter['outer']/2)**2 - (self.diameter['dielectric']/2)**2)
        elif layer == 'dielectric':
            return np.pi*((self.diameter['dielectric']/2)**2 - (self.diameter['inner']/2)**2)
        elif layer == 'inner':
            return np.pi*(self.diameter['inner']/2)**2

    
# p_drive = passive_load(
#     stage_labels=fridge_ours['labels'],
#     diameters=coax_co_params[c_drive]['d'],
#     lengths=fridge_ours['lengths'],
#     therm_cond=[coax_co_params[c_drive]['tc'], lambda t: 0, coax_co_params[c_drive]['tc']],
#     therm_scheme=therm_scheme_drive,
#     stage_temps=T_guess
# )

# p_flux = passive_load(
#     stage_labels=fridge_ours['labels'], 
#     diameters=coax_co_params[c_flux]['d'],
#     lengths=fridge_ours['lengths'],
#     therm_cond=[coax_co_params[c_flux]['tc'], lambda t: 0, coax_co_params[c_flux]['tc']],
#     therm_scheme=therm_scheme_flux,
#     stage_temps=T_guess
# )

# p_out_lower = passive_load(
#     stage_labels=fridge_ours['labels'], 
#     diameters=coax_co_params[c_out_lower]['d'],
#     lengths=fridge_ours['lengths'],
#     therm_cond=[coax_co_params[c_out_lower]['tc'], lambda t: 0, coax_co_params[c_out_lower]['tc']],
#     therm_scheme=therm_scheme_output,
#     stage_temps=T_guess
# )

# p_out_upper = passive_load(
#     stage_labels=fridge_ours['labels'], 
#     diameters=coax_co_params[c_out_upper]['d'],
#     lengths=fridge_ours['lengths'],
#     therm_cond=[coax_co_params[c_out_upper]['tc'], lambda t: 0, coax_co_params[c_out_upper]['tc']],
#     therm_scheme=therm_scheme_output,
#     stage_temps=T_guess
# )

# p_out = pd.concat((p_out_upper[0:2], p_out_lower[2:]))

# a_ac = active_load_AC(
#     stage_labels=fridge_ours['labels'], 
#     signal_p=1e-3*10**(-75/10)*10**(drive_attenuation/10), 
#     signal_f=f_op/1e9,
#     att=att_config_drive,
#     cable_att=coax_co_params[c_drive]['att_4'],
#     lengths=fridge_ours['lengths']
# )

# a_dc = active_load_DC(
#     i_in=2e-3*10**(flux_attenuation/20), 
#     stage_labels=fridge_ours['labels'], 
#     att=default_config_flux, 
#     cable_rho=coax_co_params[c_flux]['rho'], 
#     lengths=fridge_ours['lengths'], 
#     diameters=coax_co_params[c_flux]['d']
# )
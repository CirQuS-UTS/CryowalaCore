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
    - length
"""

from param_functions import coax_co_params, coeffs_single, coeffs_cross_talk, lq, l, p0
from model_functions import n_BE
import pandas as pd
import numpy as np

# from collections.abc import Hashable
# from typing import Dict, Optional, Any, Mapping
#TODO complete type hints for all functions and methods

from quanguru.classes import aliasDict

class Cryostat(object):
    """
    A class representing a cryostat with different temperature stages.
    """
    def __init__(self, **kwargs):
        self.cables = aliasDict()
        self.__table = {
            '50K': {
                'operating_temperature': 46,
                'cooling_power': 5,
            },
            '4K': {
                'operating_temperature': 3.94,
                'cooling_power': 0.35,
            },
            'Still': {
                'operating_temperature': 1.227,
                'cooling_power': 0.30e-3,
            },
            "CP": {
                'operating_temperature': 0.150,
                'cooling_power': 300e-6,
            },
            'MXC': {
                'operating_temperature': 0.020,
                'cooling_power': 20e-6,
            },
        }
        self._temperatures = None
        self._noise = None
        self._heat_loads = None
        #TODO: add setter functions for the properties of the cryostat, so that they can be changed on initialization. 
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
    
    @property
    def stages(self):
        return list(self.__table.keys())

    @property
    def operating_temperatures(self):
        return {stage: self.__table[stage]['operating_temperature'] for stage in self.stages}

    @property.setter
    def operating_temperatures(self, temps):
        if set(temps.keys()) != set(self.stages):
            raise ValueError("Input dictionary must have the same keys as the stages of the cryostat")
        for stage in self.stages:
            self.__table[stage]['operating_temperature'] = temps[stage]

    @property
    def cooling_power(self):
        return {stage: self.__table[stage]['cooling_power'] for stage in self.stages}

    @property.setter
    def cooling_power(self, powers):
        if set(powers.keys()) != set(self.stages):
            raise ValueError("Input dictionary must have the same keys as the stages of the cryostat")
        for stage in self.stages:
            self.__table[stage]['cooling_power'] = powers[stage]

    @property
    def temperatures(self):
        #leaving the possibility for paramUpdate
        return self._temperatures if self._temperatures is not None else self.operating_temperatures

    @property
    def noise(self):
        #leaving the possibility for paramUpdate
        return self.update_noise()

    @property
    def heat_loads(self):
        #leaving the possibility for paramUpdate
        return self.update_heat_loads()

    def add_cable(self, cable, quantity=1):
        #TODO potentially utilise the aliasDict object from quanguru to set objects as dict keys
        self.cables[cable] = quantity
        cable.cryostat = self

    def update_temperatures(self):
        #TODO allow for user-defined temperature response functions over than this hard-coded default
        """
        Converts heat loads on all stages to the temperature on each stage

        Parameters
            heat_loads - dict with keys "50K", "4K", "Still", "CP", "MXC"
                the total heat loads on each stage of the fridge (in the default unit of Watts)
        """

        heat_loads = self.heat_loads.sum(axis=1).to_dict()

        #Convert Watts to mW / uW as appropriate
        p = np.array([
            heat_loads["50K"],
            heat_loads["4K"]*1e3,
            heat_loads["Still"]*1e3,
            heat_loads["CP"]*1e6,
            heat_loads["MXC"]*1e6
        ])

        if p[2] > 40:
            return np.array([np.nan]*5)

        p[2] = max(p[2], coeffs_single['Still'][3])  # ensure Still power is above fit limit

        temps = np.array([lq(p[i], *coeffs_single[label])  for i, label in enumerate(['50K', '4K', 'Still', 'CP', 'MXC'])])

        # add cross talk contributions
        for i, label in enumerate(['50K', '4K', 'Still', 'CP', 'MXC']):
            temps += l(p[i], np.array(coeffs_cross_talk[label]), p0[i], 0)

        self._temperatures = {stage: val for stage, val in zip(['50K', '4K', 'Still', 'CP', 'MXC'], temps)}

        return self._temperatures

    def update_heat_loads(self):
        heat_loads = []
        for cable, quantity in self.cables.items():
            heat_loads.append(cable.total_load() * quantity)
        return pd.concat(heat_loads, axis=0)

    def update_noise(self):
        noise = {}
        for cable in self.cables.keys():
            noise[cable.name] = cable.noise_contributions
        return noise
        
    def simulate(self):
        #TODO: eventually mirror the functionality of paramUpdate in quanguru to update the simulated outputs of the cryostat
        # on call whenever a change is made in any of the relevant properties of the cryostat, cables, or heat loads. 
        # This will mean that a simulate method will not need to be called explicitly, 
        # avoiding errors where the user forgets to call it after making changes.

        convergence_threshold = 5e-4

        self._temperatures = self.operating_temperatures
        heat_loads = self.update_heat_loads()
        delta_heat_loads = np.ones_like(heat_loads.values)

        while np.any(np.abs(delta_heat_loads) > convergence_threshold):
            self.update_temperatures()
            new_heat_loads = self.update_heat_loads()
            delta_heat_loads = new_heat_loads.values - heat_loads.values
            heat_loads = new_heat_loads
        
        self.update_noise()

"how should the useful outputs (noise, heat loads, temperatures) be structured, saved, and accessed?"

class CableType(object):
    #TODO implement a check method to ensure that segments and attenuation are defined for all stages in the cryostat, 
    # before running any modelling

    #TODO split off this class into separate children classes for AC and DC signal types? 
    # (since the active load calculations are different for each)

    def __init__(self, **kwargs):
        #TODO implement default values?
        self.name = None
        self.signal_type = None
        self.signal_power = None
        self.duty_cycle = None
        self.frequency = None
        self.attenuators = {}
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

    @property
    def segment_attenuations(self):
        # returns the attenuation of each segment at the specified frequency, as a dictionary
        return {stage: segment.attenuation(self.frequency) for stage, segment in self.segments.items()}

    @property
    def effective_attenuation(self):
        # returns the effective attenuation of the cable at each stage by summing the attenuators and the segment attenuations at the specified frequency
        segment_atts = self.segment_attenuations
        return {stage: self.attenuators[stage] + segment_atts[stage] for stage in self.stages}

    @property
    def total_attenuation(self):
        return sum(self.effective_attenuation.values())

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
        for layer in ['inner', 'dielectric', 'outer']:
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
        #TODO add a handler for when the signal type is not AC to raise an error
        if self.signal_type != 'AC':
            raise ValueError("Signal type must be AC")
        #initialising the loads as 0
        loads = pd.Series(
            data=np.zeros(len(self.stages)), 
            index=self.stages, 
            name='Active Load (W)'
        )

        signal_p = self.signal_power * 10**(self.total_attenuation/10)
        carry = 0

        # iterate through the stages
        for stage, segment in self.segments.items():
            segment_att = segment.attenuation(self.frequency)
            # calculate the power dissipated in the attenuator and the segment
            p_att = signal_p * (1 - 10**(-self.attenuators[stage]/10))
            p_outer = signal_p * (1 - 10**(-segment_att/10)) * (segment.diameter['inner'])/(segment.diameter['inner']+segment.diameter['dielectric'])
            p_inner = signal_p * (1 - 10**(-segment_att/10)) * (segment.diameter['dielectric'])/(segment.diameter['inner']+segment.diameter['dielectric'])

            total_p = p_att + p_outer
            if self.thermalisation_table[stage]['inner']:
                total_p += p_inner + carry
                carry = 0
            else:
                carry = p_inner
            
            loads[stage] = total_p

            total_att = segment_att + self.attenuators[stage]
            signal_p = signal_p * 10**(-total_att/10)

        return loads

    def active_load_dc_attenuators(self):
        loads = pd.Series(
            data=np.zeros(len(self.stages)), 
            index=self.stages, 
            name='Active Load (W)'
        )

        current = self.signal_power * 10**(self.total_attenuation/20)
        
        # Calculating the currents and attenuator loads
        for stage, att in self.attenuators.items():
            A = 10**(-att/20) # linear attenuation factor
            loads[stage] = current**2 * (1-A**2) * 50
            current = current * A
        
        return loads

    def total_load(self):
        passive = self.passive_load()
        active = self.active_load_ac() if self.signal_type == 'AC' else self.active_load_dc()
        df = pd.DataFrame({
            f'Passive - {self.name} (W)': passive,
            f'Active - {self.name} (W)': active,
        })
        return df

    def active_load_dc_segments(self):
        #TODO update this formula with new analysis when completed
        #TODO combine this with the active_load_dc_attenuators function to avoid double iteration through the stages
        loads = pd.Series(
            data=np.zeros(len(self.stages)), 
            index=self.stages, 
            name='Active Load (W)'
        )

        current = self.signal_power * 10**(self.total_attenuation/20)

        for stage, segment in self.segments.items():
            # skip for stages that are not CP or MXC where empirical heat load data is not available
            if stage == "CP":
                A = np.pi*(segment.diameter['inner']/2)**2
                R =  (
                    0.42
                    * (segment.resistivity['inner'] / 7.2e-7)
                    * (segment.length/0.17)
                    * (np.pi*(0.511e-3/2)**2 / A)
                )
            elif stage == "MXC":
                A = np.pi*(segment.diameter['inner']/2)**2
                R =  (
                    0.15
                    * (segment.resistivity['inner'] / 7.2e-7)
                    * (segment.length/0.14)
                    * (np.pi*(0.511e-3/2)**2 / A)
                )
            else:
                R = 0

            loads[stage] = current**2 * R
            current = current * 10**(-self.attenuators[stage]/20)

        return loads

    def active_load_dc(self):
        return self.active_load_dc_attenuators() + self.active_load_dc_segments()

    @property
    def noise_contributions(self, frequency=None, T_in=300):
        frequency = frequency if frequency is not None else self.frequency

        contributions = pd.Series(
            data=np.zeros(len(self.stages)+1), 
            index=['RT'] + self.stages,
            name='Photon Flux Spectral Density (No. Photons/s/Hz)',
            dtype=np.float64
        ) 
        att = self.effective_attenuation

        contributions['RT'] = n_BE(T_in, frequency)*10**(-np.sum(att)/10)
        for i, stage in enumerate(self.stages):
            contributions[stage] = (
                n_BE(self.cryostat.temperatures[stage], frequency) 
                * (1-10**(-att[stage]/10)) 
                * (10**(-np.sum(att[i+1:])/10))
            )
        
        return contributions

    @property
    def noise_temperature(self):
        #TODO implement noise temperature calculation
        pass

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
        self.attenuation_constant = None
        self._material = None
        self.length = None

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
    def attenuation(self, frequency):
        return self.attenuation_constant(frequency) * self.length

    @property
    def material(self):
        return self._material

    @property.setter
    def material(self, material):
        self._material = material
        self.attenuation_constant = coax_co_params[material]['att_4']
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
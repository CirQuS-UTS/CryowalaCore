import json
from . import model_functions
from . import param_functions

def passive_load(fridge_json):
    #Stage-specific Content
    stages = []
    lengths = []
    temps = []
    for stage in fridge_json['fridge']['stages']:
        stages.append(stage['name'])
        lengths.append(stage['lengthFromPrev'])
        temps.append(stage['maxTemp'])

    #NOTE: All params should be able to be directly ripped from the JSON, except for therm_cond, which needs to return an array of functions
    thermalisations = []
    diameters = []
    functions = []
    for cable in fridge_json['fridge']['cables']:
        #Order: inner, die, outer
        diameters.append([cable['innerPinDiameter'], cable['dielectricDiameter'], cable['outerConductorDiameter']])

        temp_thermalisations = [[False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        for i in range(5):
            temp_thermalisations[0][i] = cable['thermalisation'][i][0]
            temp_thermalisations[1][i] = cable['thermalisation'][i][1]
            temp_thermalisations[2][i] = cable['thermalisation'][i][2]
        thermalisations.append(temp_thermalisations)
        functions.append([param_functions.tc_coaxco(cable['thermCondCoaxco'][0], cable['thermCondCoaxco'][1])]*3)

    results = []
    for i in range(len(fridge_json['fridge']['cables'])):
        results.append(json.loads(
            model_functions.passive_load(
                stage_labels=['0', '1', '2', '3', '4'],
                diameters=diameters[i],
                lengths=lengths,
                therm_cond=functions[i], #TODO: Fix
                therm_scheme=thermalisations[i],
                stage_temps=temps,
            ).to_json())
        )
    return json.dumps(results)

def active_load_AC(fridge_json):
    #Stage-specific Content
    stages = []
    lengths = []
    temps = []
    for stage in fridge_json['fridge']['stages']:
        stages.append(stage['name'])
        lengths.append(stage['lengthFromPrev'])
        temps.append(stage['maxTemp'])

    #Cable specific content
    attenuation = [] #Attenuation Frequencies
    cable_att_func = [] #Attenuation functions
    input_signal_power = []
    input_signal_frequency = []
    for cable in fridge_json['fridge']['cables']:
        #Order: inner, die, outer
        temp_att_array = []
        for i in range(5):
            if cable['attenuators'][i]:
                temp_att_array.append(float(cable['attenuators'][i]['frequency']))
            else:
                temp_att_array.append(0)
        attenuation.append(temp_att_array)
        cable_att_func.append(param_functions.c_att_coaxco(cable['cAtt4Coaxco']))
        input_signal_power.append(float(cable['inputSignalPower']))
        input_signal_frequency.append(float(cable['inputSignalFrequency']))

    results = []
    for i in range(len(fridge_json['fridge']['cables'])):
        if fridge_json['fridge']['cables'][i]['isAC']:
            results.append(json.loads(
                model_functions.active_load_AC(
                    stage_labels=['0', '1', '2', '3', '4'], 
                    signal_p=input_signal_power[i], #Casting to float is needed for some reason
                    signal_f=input_signal_frequency[i], 
                    att = attenuation[i], 
                    cable_att = cable_att_func[i], 
                    lengths=lengths
        ).to_json()))
        else: 
            results.append([0, 0, 0, 0, 0])

    return json.dumps(results)

def active_load_DC(fridge_json):
    #Stage-specific Content
    stages = []
    lengths = []
    for stage in fridge_json['fridge']['stages']:
        stages.append(stage['name'])
        lengths.append(stage['lengthFromPrev'])

    #Cable specific Content
    diameters = []
    attenuation = [] #Attenuation Frequencies
    cable_rho = []
    input_currents = []
    #print(fridge_json['fridge']['cables'])
    for cable in fridge_json['fridge']['cables']:
        diameters.append([cable['innerPinDiameter'], cable['dielectricDiameter'], cable['outerConductorDiameter']])
        temp_att_array = []
        for i in range(5):
            if cable['attenuators'][i]:
                temp_att_array.append(float(cable['attenuators'][i]['frequency']))
            else:
                temp_att_array.append(0)
        attenuation.append(temp_att_array)
        cable_rho.append(cable['rhoCoaxco'])
        input_currents.append(float(cable['inputCurrent']))

    results = []
    for i in range(len(fridge_json['fridge']['cables'])):
        if fridge_json['fridge']['cables'][i]['isAC']:
            results.append([0, 0, 0, 0, 0])  
        else:
            results.append(
                json.loads(model_functions.active_load_DC(
                    i_in = input_currents[i], 
                    stage_labels = ['0', '1', '2', '3', '4'], 
                    att = attenuation[i], 
                    cable_rho = cable_rho[i], 
                    lengths = lengths, 
                    diameters = diameters[i]
        ).to_json()))

    return json.dumps(results)

def drive_noise(fridge_json):
    #Stage-specific Content
    temps = []
    lengths = []
    for stage in fridge_json['fridge']['stages']:
        temps.append(stage['maxTemp'])
        lengths.append(stage['lengthFromPrev'])

    #Cable specific Content
    attenuation = [] #Attenuation Frequencies
    input_signal_frequency = []
    cable_atts = []

    #print(fridge_json['fridge']['cables'])
    for cable in fridge_json['fridge']['cables']:  
        temp_att_array = []
        for i in range(5):
            if cable['attenuators'][i]:
                temp_att_array.append(float(cable['attenuators'][i]['frequency']))
            else:
                temp_att_array.append(0)
        attenuation.append(temp_att_array)
        input_signal_frequency.append(float(cable['inputSignalFrequency']))

        temp_func = param_functions.c_att_coaxco(cable['cAtt4Coaxco'])
        cable_atts.append(temp_func(float(cable['inputSignalFrequency']))) #TODO: Use c_att_4_coaxco
        
        #TODO: Add att functions

    results = []
    for i in range(len(fridge_json['fridge']['cables'])):
        if fridge_json['fridge']['cables'][i]['isAC']:
            results.append( 
                json.loads(model_functions.drive_noise(
                    f = input_signal_frequency[i] * 1e9, #TODO: Check if this is right
                    stage_labels = ['0', '1', '2', '3', '4'],
                    stage_temps = temps,
                    att = attenuation[i], 
                    cable_att = [cable_atts[i]]*5,
                    lengths = lengths
                ).to_json()))
        else:
            results.append([0, 0, 0, 0, 0, 0])
    return json.dumps(results)

def flux_noise(fridge_json):
    #Stage-specific Content
    temps = []
    lengths = []
    for stage in fridge_json['fridge']['stages']:
        temps.append(stage['maxTemp'])
        lengths.append(stage['lengthFromPrev'])

    #Cable specific Content
    attenuation = [] #Attenuation Frequencies
    for cable in fridge_json['fridge']['cables']:  
        temp_att_array = []
        for i in range(5):
            if cable['attenuators'][i]:
                temp_att_array.append(float(cable['attenuators'][i]['frequency']))
            else:
                temp_att_array.append(0)
        attenuation.append(temp_att_array)

    results = []
    for i in range(len(fridge_json['fridge']['cables'])):
        if fridge_json['fridge']['cables'][i]['isAC']:
            results.append([0, 0, 0, 0, 0]) 
        else:
            results.append(
                json.loads(model_functions.flux_noise(
                    f = 100,
                    stage_labels = ['0', '1', '2', '3', '4'],
                    stage_temps = temps,
                    att = attenuation[i], 
                    cable_att = [0,0,0,0,0], #TODO: Update
                    lengths = lengths
                ).to_json()))
    return json.dumps(results)
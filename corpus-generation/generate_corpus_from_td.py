import json
from collections import defaultdict
import argparse
import pandas as pd
import csv
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('-td', help= 'Thing description input file path')
args = parser.parse_args()

TD_path = "./td_hue.json"
with open(TD_path, 'r')  as td:
    parsedTD = json.load(td)

# get device name
device_name = parsedTD["title"].lower()
print("Device name:", device_name)

property_names = []
property_op_names = []
property_nl = []
templates = []
parameters = []
# get all properties

try:
    all_properties = parsedTD["properties"]
    for key in all_properties:
        property_names.append(key)
        property_op = []
        #nls = []
        tem = all_properties[key]['template']
        param = all_properties[key]['parameter']
        templates.append(tem)
        parameters.append(param)
        if len(all_properties[key]["forms"]) >= 2:
            for i in range(len(all_properties[key]["forms"])):
                #print("Detected property {key} with op: ".format(key=key), all_properties[key]["forms"][i]["op"])
                property_op.append(all_properties[key]["forms"][i]["op"])
                #nls.append(all_properties[key]["forms"][i]["nl:corpus"])
            #property_nl.append(nls)    
            property_op_names.append(property_op)

        else:
            #property_nl.append(all_properties[key]["forms"][0]["nl:corpus"])
            #print("Detected property {key} with op: ".format(key=key), all_properties[key]["forms"][0]["op"])
            property_op_names.append(all_properties[key]["forms"][0]["op"])
            print(property_op_names)
    print("Found property: ", property_names)
    print("Property operation names, ", property_op_names)
    #print("property_nl:", property_nl)

except:
    print("No properties found")


# get all actions
action_names = []
action_op_names = []
action_nl = []
try:
    all_actions = parsedTD["actions"]
    for key in all_actions:
        templates.append(all_actions[key]['template'])
        parameters.append(all_actions[key]['parameter'])
        action_names.append(key)
        action_op_names.append(all_actions[key]["forms"][0]["op"])
        action_nl.append(all_actions[key]["forms"][0]["nl:corpus"])
    #print("action names: ", action_names)
    #print("action operation name: ", action_op_names)
    action_dict = {action_names[j]: action_op_names[j] for j in range(len(action_names))}
    action_nl_dict = {action_names[j]: action_nl[j] for j in range(len(action_names))}
    print("Found actions: ", action_names)
    #print("Found NL actions: ", action_nl_dict)
    action_all_dict = defaultdict(list)
    for d in (action_dict, action_nl_dict):
        for key, value in d.items():
            action_all_dict[key].append(value)
    #print("action all dict: ", action_all_dict)
except:
    print("No actions found")

#get all events
try:
    all_events = parsedTD["events"]
except:
    print("No events found") 

# generate corpus in /philipshue/data_low/
data_path = "./data_low"
isExist = os.path.exists(data_path)
if not isExist:
    os.makedirs(data_path)

corpus_path = './data_low/corpus.txt'


########################### power ###########################
# read power corpus template
df_power = pd.read_csv(templates[0], header=None, sep='\t')
#power_template = df_power.to_string(header=False, index=False).strip() # no separator in between

power_template = ""
for src, trg in zip(df_power[0].to_list(), df_power[1].to_list()):
    power_template += "{}\t{}\n".format(src.strip(), trg.strip())

# read power state parameters from powerState.csv file into a dict
reader = csv.DictReader(open(parameters[0]), delimiter='\t')
power_dict = {}
for row in reader:
    power_dict[row['power_src']] = row["power_trg"]
augment_index = 1
# generate corpus for power property in csv
with open(corpus_path, "w", newline='') as f1:
    for key in power_dict.keys():
        for i in range(augment_index):
            f1.write(power_template.format(power_src=key, power_trg=power_dict[key]))

        #f1.write('\n')
        #print(power_template.format(src_param=key, trg_param=power_dict[key]))


########################### colour ##########################
# read color corpus template
df_color = pd.read_csv(templates[1], header=None, sep='\t')
# color_template = df_color.to_string(header=False, index=False) # no separator in between

color_template = ""
for src, trg in zip(df_color[0].to_list(), df_color[1].to_list()):
    color_template += "{}\t{}\n".format(src.strip(), trg.strip())


# read colour parameters from basiccolour.csv file into dict
reader = csv.DictReader(open(parameters[1]), delimiter='\t')
color_dict = {}
for row in reader:
    color_dict[row['colour_src']] = row["colour_trg"]

# generate corpus for color property
with open(corpus_path, 'a', newline='') as f2:
    for key in color_dict.keys():
        f2.write(color_template.format(colour_src=key, colour_trg=color_dict[key]))



######################### dim ############################

# read dim corpus template
df_dim = pd.read_csv(templates[2], header=None, sep='\t')
# color_template = df_color.to_string(header=False, index=False) # no separator in between

dim_template = ""
for src, trg in zip(df_dim[0].to_list(), df_dim[1].to_list()):
    dim_template += "{}\t{}\n".format(src.strip(), trg.strip())

# read colour parameters from basiccolour.csv file into dict
reader = csv.DictReader(open(parameters[2]), delimiter='\t')
brightness_dict = {}
for row in reader:
    brightness_dict[row['brightness_src']] = row["brightness_trg"]

# generate corpus for color property
with open(corpus_path, 'a', newline='') as f2:
    for key in brightness_dict.keys():
        f2.write(dim_template.format(brightness_src=key, brightness_trg=brightness_dict[key]))
        #f2.write('\n')


######################### shuffle ############################

# shuffle rows in corpus
with open(corpus_path, "r") as file:
    lines = file.readlines()
random.shuffle(lines)

with open("./data_low/shuffled.txt", "w") as file:
    file.writelines(lines)

split_point = int(0.95 * len(lines))
train_lines = lines[:split_point]
test_lines = lines[split_point:]

# Write the testing data to the testing file
with open("./data_low/test.txt", 'w') as test_file:
    test_file.writelines(list(set(test_lines)))

print("Corpus successfully generated!")


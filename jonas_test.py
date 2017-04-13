import json
import pandas as pd

data = {}
with open('Data/Patio_Lawn_and_Garden_5.json') as data_file:
    id = 0
    for line in data_file:
        data[id] = json.loads(line)
        id+=1


# pdata = pd.from_dict(data)
pdata = pd.DataFrame.from_dict(data)


print(pdata )
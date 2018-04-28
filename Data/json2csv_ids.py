import csv
import json

with open('train_ids.json') as json_data:
    train_json = json.load(json_data)
    
with open('val_ids.json') as json_data:
    val_json = json.load(json_data)
    
with open('test_ids.json') as json_data:
    test_json = json.load(json_data)

train_csvfile = "train_ids.csv"
val_csvfile = "val_ids.csv"
test_csvfile = "test_ids.csv"

#Assuming *_csvfile are flat lists
with open(train_csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for value in train_json:
        writer.writerow([value]) 
        
with open(val_csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for value in val_json:
        writer.writerow([value]) 
        
with open(test_csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for value in test_json:
        writer.writerow([value])
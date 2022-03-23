import csv

def write_csv(x, y):
    
    data=[]
    data.append(x)
    data.append(y)

    with open('data.csv', 'w', encoding='UTF8', newline='') as f:
        
        writer = csv.writer(f)
        writer.writerows(data)

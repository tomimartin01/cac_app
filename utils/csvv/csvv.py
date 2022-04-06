import csv

def write_csv(x_body, y_body, x_bar, y_bar):
    
    data=[]
    data.append(x_body)
    data.append(y_body)
    data.append(x_bar)
    data.append(y_bar)

    with open('data.csv', 'w', encoding='UTF8', newline='') as f:
        
        writer = csv.writer(f)
        writer.writerows(data)

import csv

key = 'noir'
raw = csv.reader(open(f'{key}_ids_raw.csv', encoding='utf8'))
processed = []
for line in raw:
	processed.append(str(line).split('/')[2])


f = csv.writer(open(f'{key}-ids.csv', "w", encoding='utf8', newline=''))

f.writerow(["id"])

for film in processed:
	f.writerow([film])
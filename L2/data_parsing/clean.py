import csv

lines = list()
memberName = "nopicture.jpg"
with open('rest-genres-movies.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    for row in reader:
        if (row[3] != 'genre'):
            row[3] = row[3].title()
        if (memberName not in row[1] and row[4] != ''):
            lines.append(row)
with open('rest-genres-with-pics.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
import requests, csv

with open('rest-genres-with-pics.csv', 'r') as src:
    reader = csv.reader(src)
    next(reader)
    for row in reader:
        id = row[0]
        poster_url = row[1]
        print('parsing pic' + id)
        with open(f'posters/{id}.jpg', 'wb') as f:
            f.write(requests.get(poster_url).content)
# if you change png to jpg, there will be no error

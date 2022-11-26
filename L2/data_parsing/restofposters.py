import requests, csv

api_key = 'k_2c8za7xz'
with open('abscent_imgs.csv', 'r') as src:
    reader = csv.reader(src)
    next(reader)
    for row in reader:
        id = row[0]
        poster_url = f'https://imdb-api.com/en/API/Posters/{api_key}/{id}'
        print('parsing pic' + id)
        response = requests.get(poster_url)
        js = response.json()
        posters_list = js.get('posters')
        if (len(posters_list) > 0):
            img_url = posters_list[0]['link']
            with open(f'restposters/{id}.jpg', 'wb') as f:
                f.write(requests.get(img_url).content)
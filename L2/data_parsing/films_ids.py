import requests

# url = "https://online-movie-database.p.rapidapi.com/title/v2/find"
# page_size = 20
genres_to_required_pages = {
    "war": 5,
    # "history": 7,
    # "music": 7,
    # "sport": 10,
    # "adult": 10,
    # "documentary": 10
}


def format_id(json):
    json['id'] = str(json['id']).split('/')[2]
    return json

import requests, itertools, json, csv

url = "https://online-movie-database.p.rapidapi.com/title/v2/get-popular-movies-by-genre"
headers = {
	"X-RapidAPI-Key": "13fa02ddfbmsh40818f478de1b63p100a6ajsnccae5de10527",
	"X-RapidAPI-Host": "online-movie-database.p.rapidapi.com"
}

for key in genres_to_required_pages.keys():
    querystring = {"genre":key,"limit":"200"}

    response = requests.request("GET", url, headers=headers, params=querystring)
    resp_json = response.json()
    films = list(map(lambda id: str(id).split('/')[2], resp_json))
    # films = list(map(format_id, results))

    f = csv.writer(open(f'{key}-ids.csv', "w", encoding='utf8', newline=''))

    f.writerow(["id"])

    for film in films:
        f.writerow([film])

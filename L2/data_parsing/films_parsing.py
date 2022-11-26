import csv
import json
import requests

genres_to_required_count = {
    "war": 100,
    "history": 130,
    "music": 130,
    "sport": 180,
    "adult": 200,
    "documentary": 200,
    "movie-noir": 180
}

url = "https://movie-details1.p.rapidapi.com/imdb_api/movie"
headers = {
    "X-RapidAPI-Key": "13fa02ddfbmsh40818f478de1b63p100a6ajsnccae5de10527",
    "X-RapidAPI-Host": "movie-details1.p.rapidapi.com"
}
writer = csv.writer(open('rest-genres-movies.csv', 'a', encoding='utf8'))
writer.writerow(["id", "image_url", "title", "titleType", "year", "genre"])
for genre in genres_to_required_count.keys():
    current_genre_ids = csv.reader(open(f'{genre}-ids.csv', encoding='utf8'))
    next(current_genre_ids, None)
    movie_genre_counter = 0
    for line in current_genre_ids:
        if movie_genre_counter >= genres_to_required_count.get(genre):
            break
        current_id = line[0]
        querystring = {"id": f"{current_id}"}
        response = requests.request("GET", url, headers=headers, params=querystring)

        response.json()

        for film in films:
            writer.writerow([film["id"],
                             film["image"]["url"],
                             film["title"],
                             film["titleType"],
                             film["year"],
                             genre.capitalize()])

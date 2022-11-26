import json, csv
import requests


headers = {}
payload = {}

writer = csv.writer(open('superheroes2.csv', 'a', encoding='utf8', newline=''))
# writer.writerow(["id", "image", "title", "genre", "plot"])
url = "https://imdb-api.com/API/AdvancedSearch/k_o119krft?keywords=superhero&count=250&sort=user_rating,desc"
response = requests.request("GET", url, headers=headers, data=payload)
json_resp = response.json()
response_payload = json_resp.get('results')
for movie in response_payload:
    writer.writerow([movie["id"],
                     movie["image"],
                     movie["title"],
                     'Superhero',
                     movie["plot"]])
# heroes_raw = open('films.json')
# heroes_json = json.load(heroes_raw)
#
# print(heroes_json[1])
#
# results = list(map(lambda resp: resp["results"], heroes_json))
# films = list(itertools.chain.from_iterable(results))
# def format_id(json):
#     json['id'] = str(json['id']).split('/')[2]
#     return json
# films = list(map(format_id, films))
# films = list(filter(lambda it: 'image' in it and 'year' in it, films))
# print()
#
# f = csv.writer(open("superheroes.csv", "w", encoding='utf8', newline=''))
#
# # Write CSV Header, If you don't need that, remove this line
# f.writerow(["id", "image_url", "title", "titleType", "year", "genre"])
#
# for film in films:
#     f.writerow([film["id"],
#                 film["image"]["url"],
#                 film["title"],
#                 film["titleType"],
#                 film["year"],
#                 'Superhero'])
#
# # {'id': '/title/tt0178516/', 'image': {'height': 1123, 'id': '/title/tt0178516/images/rm786182912', 'url': 'https://m.media-amazon.com/images/M/MV5BZjYzMDRiMDctNmFhOS00NmEyLTk3ZTQtNzZmOWYxYTQxYzRmXkEyXkFqcGdeQXVyNDUxNjc5NjY@._V1_.jpg', 'width': 800}, 'title': 'Goliath and the Rebel Slave', 'titleType': 'movie', 'year': 1963}
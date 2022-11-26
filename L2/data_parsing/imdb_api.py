import requests, csv, json

api_key = 'k_o119krft'
curr_genre = ''
urlgenre = f'https://imdb-api.com/API/AdvancedSearch/{api_key}?genres={curr_genre}&count=250&sort=user_rating,desc'
url = f"https://imdb-api.com/en/API/Title/${api_key}/${id}"
payload = {}
headers = {}

genres_to_parse = {
    'crime',
    'mystery',
    'musical',
    'animation',
    'fantasy',
    'biography',
    "war",
    "history",
    "music",
    "sport",
    # "adult",
    "documentary",
    "film-noir",
    'short-movie'
}

genres_to_parse = { 'film-noir' }
# Crime            157 x
# Mystery          130
# Musical          118
# Animation         97
# Fantasy           73
# War               53 x
# Biography         51 x
# Music             27 x
# History           26 x
# Sport             15 x
# Documentary        1 x

writer = csv.writer(open('rest-genres-movies.csv', 'a', encoding='utf8', newline=''))
# writer.writerow(["id", "image", "title", "genre", "plot"])

for genre in genres_to_parse:
    curr_genre = genre
    urlgenre = "https://imdb-api.com/API/AdvancedSearch/k_o119krft?genres=" + curr_genre +"&count=1000&sort=user_rating,asc&page=5"
    urlgenre = 'https://imdb-api.com/API/AdvancedSearch/k_o119krft?title_type=short&count=250&sort=moviemeter,desc&page=4'
    response = requests.request("GET", urlgenre, headers=headers, data=payload)
    json_resp = response.json()
    response_payload = json_resp.get('results')
    for movie in response_payload:
        writer.writerow([movie["id"],
                         movie["image"],
                         movie["title"],
                         'Short',
                         movie["plot"]])


# response = requests.request("GET", url, headers=headers, data=payload)

# print(response.text.encode('utf8'))
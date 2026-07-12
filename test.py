import json

with open("data/movies.json", "r") as f:
    data = json.load(f)

print(data["movies"])

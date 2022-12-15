# %%
from src.data.download import download_whole_city
from src.data.make_dataset import add_h3_indices_to_city
from src.data.load_data import load_filter
from src.data.make_dataset import group_city_tags
from src.data.make_dataset import group_cities
from src.settings import DATA_RAW_DIR
from tqdm import tqdm

# %%
RESOLUTION = 9
TAG_FILTER = load_filter("from_wiki.json")

# remove tags not of interest
SELECTED_TAGS = [
    "aeroway",
    "amenity",
    "building",
    "healthcare",
    "historic",
    "landuse",
    "leisure",
    "military",
    "natural",
    "office",
    "shop",
    "sport",
    "tourism",
    "water",
    "waterway",
]

new_tag_filter = {} 
for tag in SELECTED_TAGS:
    new_tag_filter[tag] = TAG_FILTER[tag]
TAG_FILTER = new_tag_filter.copy()
    

problem_columns = [
    'amenity_waste_basket',
    'landuse_grass',
    'historic_tomb',
    'natural_tree',
    'natural_tree_row',
    'natural_valley', # northern Warsaw
]

for tag in problem_columns:
    super_tag, *sub_tag = tag.split("_")
    if isinstance(sub_tag, (list, tuple)):
        sub_tag = "_".join(sub_tag)
    TAG_FILTER[super_tag] = [
        tag for tag in TAG_FILTER[super_tag] if tag != sub_tag
    ]


# %%
cities = [
    # "Vienna, Austria",  # I had to remove vienna austria as someone removed the city boundary from OSM
    "Minsk, Belarus",
    "Brussels, Belgium",
    "Sofia, Bulgaria",
    "Zagreb, Croatia",
    "Prague, Czech Republic",
    "Tallinn, Estonia",
    "Helsinki, Finland",
    "Paris, France",
    "Berlin, Germany",
    "Reykjavík, Iceland",
    "Dublin, Ireland",
    "Rome, Italy",
    "Nur-Sultan, Kazakhstan",
    "Latvia, Riga",
    "Vilnius, Lithuania",
    "Luxembourg City, Luxembourg",
    "Amsterdam, Netherlands",
    "Oslo, Norway",
    "Warszawa, PL",
    "Kraków, PL",
    "Łódź, PL",
    "Wrocław, PL",
    "Poznań, PL",
    "Gdańsk, PL",
    "Lisbon, Portugal",
    ['Moscow, Russia', 'Zelenogradsky Administrative Okrug', 'Western Administrative Okrug', 'Novomoskovsky Administrative Okrug', 'Troitsky Administrative Okrug'],
    "Belgrade, Serbia",
    "Bratislava, Slovakia",
    "Ljubljana, Slovenia",
    "Madrid, Spain",
    "Stockholm, Sweden",
    "Bern, Switzerland",
    ["London, United Kingdom", "City of London"],
    "New York City, USA",
    "Chicago, USA",
    "San Francisco, USA",
]

# %%
# run download whole city in 4 threads
from threading import Thread
from queue import Queue

def worker(q: Queue):
    while not q.empty():
        item = q.get()
        download_whole_city(item, DATA_RAW_DIR)
        q.task_done()



if __name__ == "__main__":
    # q = Queue()
    # ts = []
    # for city in tqdm(cities):
    #     q.put(city)

    # for i in range(10):
    #     t = Thread(target=worker, args=(q,))
    #     t.daemon = True
    #     t.start()
    #     ts.append(t)

    # for t in ts:
    #     t.join()

    for city in tqdm(cities):
        download_whole_city(city, DATA_RAW_DIR)

    # remove the cities that are lists of cities
    cities = [city[0] if isinstance(city, (list, tuple)) else city for city in cities]

    for city in tqdm(cities):
        add_h3_indices_to_city(city, RESOLUTION, filter_values=TAG_FILTER)

    for city in tqdm(cities):
        group_city_tags(city, RESOLUTION, tags=list(TAG_FILTER.keys()), filter_values=TAG_FILTER, fill_missing=True)

    df = group_cities(
        cities=cities,
        resolution=RESOLUTION,
        add_city_column=True,
    )


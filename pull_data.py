# %%
from src.data.download import download_whole_city
from src.data.make_dataset import add_h3_indices_to_city
from src.data.load_data import load_filter
from src.data.make_dataset import group_city_tags
from src.settings import DATA_RAW_DIR
from tqdm import tqdm

# %%
RESOLUTION = 9
TAG_FILTER = load_filter("from_wiki.json")

# %%
cities = [
    # "Vienna, Austria",
    # "Minsk, Belarus",
    # "Brussels, Belgium",
    # "Sofia, Bulgaria",
    # "Zagreb, Croatia",
    # "Prague, Czech Republic",
    # "Tallinn, Estonia",
    # "Helsinki, Finland",
    # "Paris, France",
    # "Berlin, Germany",
    # "Reykjavík, Iceland",
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
    "Moscow, Russia",
    "Belgrade, Serbia",
    "Bratislava, Slovakia",
    "Ljubljana, Slovenia",
    "Madrid, Spain",
    "Stockholm, Sweden",
    "Bern, Switzerland",
    "London, United Kingdom",
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
        add_h3_indices_to_city(city, RESOLUTION)

    for city in tqdm(cities):
        group_city_tags(city, RESOLUTION, filter_values=TAG_FILTER, fill_missing=True)
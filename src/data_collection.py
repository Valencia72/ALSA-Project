import requests
import time
import json

# -------- CONFIG --------
URL = "https://api.pullpush.io/reddit/search/submission/"

# Combined all subreddits into one single list
SUBREDDITS = [
    "learnmachinelearning",
    "deeplearning",
    "computerscience",
    "programming",
    "futurology",
    "aiethics",
    "devops",
    "datascience",
    "machinelearning",
    "cscareerquestions",
    "mlquestions",
    "learnprogramming",
    "codinghelp",
    "promptengineering"
]

TARGET = 200000
BATCH_SIZE = 500
SLEEP_TIME = 1

OUTPUT_FILE = "raw_reddit_combined_posts.json"

# -------- STORAGE --------
all_posts = []

# -------- COLLECTION --------
for subreddit in SUBREDDITS:
    print(f"\n Collecting from r/{subreddit}")
    before = None

    while len(all_posts) < TARGET:
        params = {
            "subreddit": subreddit,
            "size": BATCH_SIZE,
            "before": before
        }

        try:
            response = requests.get(URL, params=params, timeout=20)
            data = response.json().get("data", [])

            if not data:
                print(f" No more data in r/{subreddit}")
                break

            all_posts.extend(data)
            before = data[-1]["created_utc"]

            print(f"Total collected so far: {len(all_posts)}")

            time.sleep(SLEEP_TIME)

            if len(all_posts) >= TARGET:
                break

        except Exception as e:
            print(f" Error fetching r/{subreddit}: {e}")
            break

print("\n Finished collection")
print(" Final count:", len(all_posts))

# -------- SAVE --------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_posts, f, ensure_ascii=False)

print(f" Raw Reddit combined data saved to {OUTPUT_FILE}")
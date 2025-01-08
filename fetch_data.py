import operator
import pathlib
import time
from dotenv import load_dotenv
import databento as db

load_dotenv()

# First, create a historical client
client = db.Historical()

# Next, we will submit a batch job
new_job = client.batch.submit_job(
    dataset="XNAS.ITCH",
    start="2024-12-01",
    end="2024-12-08",
    symbols=["AAPL", "MSFT", "NVDA", "AMGN", "GILD", "TSLA", "PEP", "JPM", "V", "XOM"],
    schema="mbp-10",
    split_duration="day",
)

# Retrieve the new job ID
new_job_id: str = new_job["id"]
print(f"New job ID: {new_job_id}")

# Now, we have to wait for our batch job to complete
while True:
    done_jobs = list(map(operator.itemgetter("id"), client.batch.list_jobs("done")))
    if new_job_id in done_jobs:
        print("Job is done!")
        break  # Exit the loop to continue
    print(".", end="", flush=True)
    time.sleep(1.0)

print("Downloading files...")
# Once complete, we will download the files
downloaded_files = client.batch.download(
    job_id=new_job_id,
    output_dir=pathlib.Path.cwd(),
)

# Finally, we can load the data into a DBNStore for analysis
for file in sorted(downloaded_files):
    if file.name.endswith(".dbn.zst"):
        data = db.DBNStore.from_file(file)

        # Convert the data to a pandas.DataFrame
        df = data.to_df()
        print(f"{file.name} contains {len(df):,d} records")

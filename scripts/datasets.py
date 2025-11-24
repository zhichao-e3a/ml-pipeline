from database.MongoDBConnector import MongoDBConnector
from utils.data import process_row

import os
import json
import asyncio
import argparse
from pathlib import Path
from tqdm.auto import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=['onset', 'add', 'hist'])
    target = parser.parse_args().target

    ROOT = Path(__file__).parent.parent
    OUT  = ROOT / "datasets" / f"dataset_{target}.json"

    mongo = MongoDBConnector(mode='remote')

    add_data = asyncio.run(
        mongo.get_all_documents(
            coll_name=f"model_data_{target}",
            sort=[
                ('mobile', 1),
                ('measurement_date', 1)
            ]
        )
    )

    print(len(add_data), "measurements retrieved from", f"model_data_{target}")

    dataset = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count()-1)) as executor:

        futures = [executor.submit(process_row, row, target) for row in add_data]

        for fut in tqdm(as_completed(futures), total=len(futures)):

            rec = fut.result()
            if rec is not None:
                dataset.append(rec)

    with open(OUT, 'w') as outfile:
        json.dump(dataset, outfile)

    print(len(dataset), "measurements written to", OUT)

    asyncio.run(mongo.upsert_documents_hashed(records=dataset, coll_name=f"dataset_{target}"))

    print(len(dataset), "measurements upserted to", f"'dataset_{target}'")

if __name__ == "__main__":
    main()
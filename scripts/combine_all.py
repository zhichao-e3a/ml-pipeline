from database.MongoDBConnector import MongoDBConnector

import json
import asyncio
import pandas as pd
from pathlib import Path

async def combine_all():

    mongo   = MongoDBConnector(mode='remote')

    ROOT = Path(__file__).parent.parent
    OUT  = ROOT / "datasets" / f"dataset_all.json"

    hist_add = await mongo.get_all_documents(
        coll_name="dataset_hist"
    )

    new_add = await mongo.get_all_documents(
        coll_name="dataset_add"
    )

    print(len(hist_add), "measurements fetched from 'dataset_hist'")
    print(len(new_add), "measurements fetched from 'dataset_add'")

    hist_df = pd.DataFrame.from_records(hist_add) ; new_df = pd.DataFrame.from_records(new_add)

    dups = hist_df.merge(
        new_df,
        how="inner",
        on=['mobile', 'measurement_date'],
        suffixes=("_hist", "_new")
    )

    print(len(dups), "duplicate measurements")

    key_cols = ["mobile", "measurement_date"]

    anti = new_df.merge(hist_df[key_cols], on=key_cols, how="left", indicator=True)
    new_df_2 = anti[anti["_merge"] == "left_only"].drop(columns=["_merge"])

    all_df = pd.concat([hist_df, new_df_2], ignore_index=True)

    print(len(all_df), "measurements after merging")
    print(f"(Verify {len(hist_df)+len(new_df)-len(dups)})")

    all_records = all_df.to_dict(orient="records")

    with open(OUT, 'w') as outfile:
        json.dump(all_records, outfile)

    print(len(all_records), "measurements written to", OUT)

    await mongo.upsert_documents_hashed(records=all_records, coll_name=f"dataset_all")

    print(len(all_records), "measurements upserted to 'dataset_all'")

if __name__ == "__main__":
    asyncio.run(combine_all())
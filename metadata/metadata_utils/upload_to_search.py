import os
import json
import mdf_toolbox
import dlhub_client.client as client

# Current searchables that could be uploaded to search
# i.e. container is alive on dl.get_servables
with open("searchables.json", 'r') as f:
    searchables = json.load(f)["searchables"]

# Tested and proven to work. (Possible missing edge cases)
def ingest_to_search(name, path, idx):
    if path == None:
        return
    dl = client.DLHub()
    uuid = dl.get_id_by_name(name)
    iden = "https://dlhub.org/api/v1/servables/{}".format(uuid)
    index = mdf_toolbox.translate_index(idx)

    with open(path, 'r') as f:
        ingestable = json.load(f)

    ingestable = mdf_toolbox.format_gmeta(ingestable, acl="public", identifier=iden)
    ingestable = mdf_toolbox.format_gmeta([ingestable]) # Make it a GIngest list of GMetaEntry

    ingest_client = mdf_toolbox.login(services=["search_ingest"])["search_ingest"]
    ingest_client.ingest(index, ingestable)
    print("Ingestion of {} complete".format(name))

# Has not been tested for functionality
def ingest_select(l, idx):
    """
    Args:
        l (list): list of source names found as keywords in searchables
        idx (str): Globus Search Index to ingest to
    """
    for k in l:
        print("Uploading {}".format(k))
        path = searchables[k]["filename"]
        ingest_to_search(k, path, idx)
    print("Ingestion Complete")

# Has not been tested for functionality
def ingest_all(idx):
    for k, v in searchables.items():
        print("Uploading {}".format(k))
        name = k
        path = v["filename"]
        ingest_to_search(name, path, idx)
    print("Ingestion Complete")


if __name__ == '__main__':
    pass
    #ingest_to_search("DSIR", "dsir.json", "dlhub-test")

import os
import sys
import time
import json
import shlex
import boto3
import uuid
import requests
import subprocess
import mdf_toolbox
import pandas as pd
import dlhub_client.client as client

# from dlhub_client import config
#NOTE: Config is not implemented. Architecture of storing keys can be changed
#NOTE: but for the gist of ingestion this is the current layout

# destination bucket name
bucket_name = 'dlhub-anl'
# For information on DLHub models
service = "http://dlhub.org:5000/api/v1"


def upload_directory(path, destDir="servables/"):
    """Function to upload an entire directory to specified s3 endpoint
    and make the files public
    Args:
        path (str): The path to the model container to be uploaded
        destDir (str): Base of where to store the model_container on s3.
                       Defaults to 'dlhub-anl/servables'.
    """
    # NOTE: this will not work until a config file with keys is implemented
    s3 = boto3.resource('s3')

    uploadFileNames = []
    for (sourceDir, dirname, filename) in os.walk(path):
        for f in filename:
            if f[0] == ".": # Ignore files such as .DS_Store
                continue
            uploadFileNames.append(os.path.join(sourceDir, f))

    dest_uuid = str(uuid.uuid4())
    print("Uploading to s3: \n")
    for sourcepath in uploadFileNames:
        ext_path = sourcepath.split(path)[-1].strip("/") # Take everything after parent dir
        destpath = os.path.join(destDir, dest_uuid, ext_path) # Join destination with a uuid and path to file
        print("Uploading: {}".format(sourcepath))
        # NOTE: Commented out until access keys are figured out
        #s3.put_oject(ACL="public-read", Bucket=bucket_name, key=destpath, Body=open(sourcepath, 'rb'))
        res = s3.Object(bucket_name, destpath).put(ACL="public-read", Body=open(sourcepath, 'rb'))

    return os.path.join("s3://", bucket_name, destDir, dest_uuid)

def ingest_metadata(definition):
    """Ingests the container into the DLHub api as a servable.
    Args:
        definition (dict): a dictionary definition blob
        specify the input the path.
    """
    payload = definition
    url = "http://dlhub.org/api/v1/servables"

    print("Running Ingestion to DLHub Servables")
    response = requests.post(url, json=payload)

    try:
        return json.loads(response.text)
    except:
        pass
    return response.text


def check_status(task_id):
    """
    Check the status of the task.

    Args:
        task_id (str): The id of the task.
    """
    url = "http://dlhub.org/api/v1/{}/status".format(task_id)
    r = requests.get(url)
    try:
        return json.loads(r.text)
    except:
        pass
    return r.text

def last_id():
    """Get the last ingested id to know when the new id is uploaded
    i.e. new id must be greater than the last ingested id.
    Returns:
        int
    """
    r = requests.get("{service}/servables".format(service=service), timeout=100)
    tmp_df = pd.DataFrame(r.json())
    ids = tmp_df["id"]
    res = max(ids)
    return res


def poll_status(last_id, timeout=100):
    """Polls DLHub servables to check the new model Status
    NOTE: Relies on the fact that each new model id is larger than previously ingested model ids.
    Args:
        last_id (int): The id of the most recently ingested model (derived from last_id())
        timeout (int): The maximum amount of time (seconds) until aborting the poll. Default=100.
    Returns:
        bool: True if the model was ingested and has a status of "READY". False otherwise.
    """
    time_spent = 0
    i=0
    while(True):
        start = time.time()
        if time_spent > timeout:
            print("Polling Status Timed Out")
            return False

        df = pd.DataFrame(requests.get("{service}/servables".format(service=service), timeout=10).json())
        if max(df["id"]) > last_id: # Model on servables. See if status is ready
            new_id = max(df["id"])
            status = df[df.id==new_id].iloc[0]["status"]
            if status == "READY":
                return True
        time.sleep(5)
        time_spent+=time.time()-start
    return False # Should never reach here


def ingest_to_search(name, path, idx="dlhub-test"):
    """Ingests the model metadata into Globus Search
    Args:
        name (str): The model name. (generally the same as container directory name
                    but not always) e.g. "cifar10"
        path (str): Path to the model metadata. e.g. "metadata/cifar10.json"
        idx (str): The Globus Index to upload search metadata to. Defaults=dlhub-test
    """
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
    print("Ingestion of {} to DLHub servables complete".format(name))

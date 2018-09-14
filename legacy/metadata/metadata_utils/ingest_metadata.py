import json
import shlex
import subprocess

def ingest_metadata():
    path = input("Input path to json file here: ")

    with open(path, 'r') as f:
       json_input = json.load(f)

    json_output = json.dumps(json_input)

    cmd = "curl -H \"Content-Type:application/json\" -X POST -d '{}' http://dlhub.org:5000/api/v1/servables".format(json_output)

    print("Running Command {}".format(cmd))

    args = shlex.split(cmd)
    process = subprocess.Popen(args, stdout=subprocess.PIPE)
    output, error = process.communicate()

    print("\nIngestion Complete. Check DLHub servables to ensure ingestion went through")

if __name__ == '__main__':
    ingest_metadata()

# Intended to act as a noop and test functionality of dlhub

def run(data):
    return {}

def test_run():
    print("Running test run")
    output = run("")
    return output

if __name__ == '__main__':
    test_run()

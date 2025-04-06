from requests import get
import sys

def health_check(url):
    try:
        response = get(url, timeout=5)
        if response.status_code == 200:
            print("Service is healthy")
            sys.exit(0)
        else:
            print(e)
            sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    health_check("http://localhost:8001/healthcheck")

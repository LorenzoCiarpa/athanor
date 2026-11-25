import time

from app.config import TIMEOUT_S

MAX_RETRIES = 7
BACKOFF_S = 0.5


def fetch(get, url):
    for attempt in range(MAX_RETRIES):
        try:
            return get(url, timeout=TIMEOUT_S)
        except OSError:
            time.sleep(BACKOFF_S * 2 ** attempt)
    raise OSError(f"{url}: no answer after {MAX_RETRIES} tries")

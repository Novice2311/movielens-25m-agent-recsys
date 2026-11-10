"""Download the research dataset separately. Do not commit or redistribute it."""
import hashlib
import urllib.request
from pathlib import Path

def main():
    target=Path('data/ml-25m.zip');target.parent.mkdir(exist_ok=True)
    url='https://files.grouplens.org/datasets/movielens/ml-25m.zip'
    if not target.exists():urllib.request.urlretrieve(url,target)
    digest=hashlib.md5(target.read_bytes()).hexdigest()
    assert digest=='6b51fb2759a8657d3bfcbfc42b592ada', 'Archive differs from the official release'
    print(target,digest)

if __name__=='__main__':main()

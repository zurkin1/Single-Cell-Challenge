import requests
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import time
import sys
import urllib3

from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

retry = Retry(connect=3, backoff_factor=5)
class myAdapter(HTTPAdapter):
    """"Transport adapter" that allows us to use SSLv3."""
    def init_poolmanager(self, connections, maxsize, block=True):
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block)
        self.max_retries = retry

s = requests.Session()
s.mount('http://', myAdapter(50,10))
#http = urllib3.PoolManager(num_pools=50, maxsize=10, block=True)

def simple_get(url):
    """
    Attempts to get the contents at 'url' by making an HTTP get request.
    If the content type of the response is some kind of HTML/XML, return the
    text content otherwise return None.
    """
    try:
        #with closing(get(url, stream=True)) as resp:
        resp = s.get(url, stream=True)
        if is_good_response(resp):
           return resp.content
        else:
           return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)

import pandas as pd
from bs4 import BeautifulSoup
import json
import sys

data = pd.DataFrame()
with open('NCBI1.csv') as f:
    content = f.readlines()
    counter = 0
    for line in content:
        print(f'{counter} ', end="")
        raw_html = simple_get(line)
        if(raw_html.__sizeof__() <= 100):
            counter = counter + 1
            continue
        try:
            html = json.loads(raw_html)
            html2 = BeautifulSoup(html['htmlTable'], 'html.parser')
            columns = []
            values = []
            for i, li in enumerate(html2.select('tr')):
                if('Gene ID' in li.text):
                    columns = columns + li.text.strip().split('\n')
                if('Fly' in li.text):
                    values = values + li.text.strip().split('\n')
            #Only merge if fly data is available
            if(len(values) > 0):
                newdata = pd.DataFrame([values], columns=columns)
                newdata = newdata.loc[:,~newdata.columns.duplicated()]
                if(len(data) == 0):
                    data = newdata
                else:
                    data = pd.concat([data, newdata], sort=False)
            counter = counter + 1
        except:
            print('Exception parsing ', sys.exc_info()[0])
            continue
        data.to_csv('g2f_other_3.csv')
        sys.stdout.flush()
        time.sleep(3)
import urllib.request
import re

url = 'https://download.pytorch.org/whl/cu121/torch_stable.html'
html = urllib.request.urlopen(url, timeout=30).read().decode('utf-8', 'ignore')
for m in re.findall(r'href="([^"]*torch-[^"]*cp313[^"]*)"', html):
    print(m)

import argparse
import requests
import time
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
ap.add_argument("-n", "--num_images", type=int, default=500, help="# of images to download")
args = vars(ap.parse_args())

url = "https://www.e-zpassny.com/vector/jcaptcha.do"
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15',
           'Cookie': '___utmvaBVBuBSlY=KYv; incap_ses_1248_276192=aGiWFQRjmDeP0lSaeslREZ29PF8AAAAAHt3PQBY7HWh1zUWqH85Gqg==; visid_incap_276192=P6gn7aURQrie/ypTKJXnh0u2PF8AAAAAQUIPAAAAAADAeDKNbyiXQUWfABEFNoCL; JSESSIONID=0001sxG2ruVf5zQP1kJ8IPSEN0H:2AI9T9ME4R'}
total = 0

for i in range(0, args["num_images"]):
    try:
        req = requests.get(url, headers=headers, timeout=60)

        path = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))])
        f = open(path, "wb")
        f.write(req.content)
        f.close()

        print("[INFO] downloaded {}".format(path))
        total += 1

    except:
        print("[INFO] error downloading image...")

    time.sleep(1)
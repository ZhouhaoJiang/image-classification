import re
import os
import requests
import tqdm

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'}


def getImg(url, idx, path, search):
    img = requests.get(url, headers=header)
    file = open(path + search + str(idx) + '.jpg', 'wb')
    file.write(img.content)
    file.close()


search = input("请输入搜索内容：")
number = int(input("请输入需求数量："))
path = r'dataset/train/others/'
if not os.path.exists(path):
    os.makedirs(path)

bar = tqdm.tqdm(total=number)
page = 0
while (True):
    if number == 0:
        break
    url = 'https://image.baidu.com/search/acjson'
    params = {
        "tn": "resultjson_com",
        "logid": "11555092689241190059",
        "ipn": "rj",
        "ct": "201326592",
        "is": "",
        "fp": "result",
        "queryWord": search,
        "cl": "2",
        "lm": "-1",
        "ie": "utf-8",
        "oe": "utf-8",
        "adpicid": "",
        "st": "-1",
        "z": "",
        "ic": "0",
        "hd": "",
        "latest": "",
        "copyright": "",
        "word": search,
        "s": "",
        "se": "",
        "tab": "",
        "width": "",
        "height": "",
        "face": "0",
        "istype": "2",
        "qc": "",
        "nc": "1",
        "fr": "",
        "expermode": "",
        "force": "",
        "pn": str(60 * page),
        "rn": number,
        "gsm": "1e",
        "1617626956685": ""
    }
    try:
        result = requests.get(url, headers=header, params=params).json()
    except Exception as e:
        print(e)
        continue
    url_list = []
    for data in result['data'][:-1]:
        url_list.append(data['thumbURL'])
    for i in range(len(url_list)):
        getImg(url_list[i], 60 * page + i, path, search)
        bar.update(1)
        number -= 1
        if number == 0:
            break
    page += 1
print("\nfinish!")

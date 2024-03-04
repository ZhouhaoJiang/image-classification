import requests
from lxml import etree
import os
from multiprocessing.dummy import Pool
import json
from time import time
 
# 作用：按关键字、图片数量爬取必应图片，存放到指定路径。
# 使用方法：只需运行一条命令 BingImagesSpider('电脑美女壁纸', 200, 'E:\images').run()
class BingImagesSpider:
    thread_amount = 1000 # 线程池数量，线程池用于多IO请求，减少总的http请求时间
    per_page_images = 30 # 每页必应请求的图片数
    count = 0 # 图片计数
    success_count = 0
    # 忽略图片标签的一些字符
    ignore_chars = ['|', '.', '，', ',', '', '', '/', '@', ':', '：', ';', '；', '[', ']', '+']
    # 允许的图片类型
    image_types = ['bmp', 'jpg', 'png', 'tif', 'gif', 'pcx', 'tga', 'exif', 'fpx', 'svg', 'psd', 'cdr', 'pcd', 'dxf', 'ufo', 'eps', 'ai', 'raw', 'WMF', 'webp']
    # 请求头
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}
    # 必应图片 url
    bing_image_url_pattern = 'https://www.bing.com/images/async?q={}&first={}&count={}&mmasync=1'
 
    def __init__(self, keyword, amount, path='./'):
        # keyword: 需爬取的关键字
        # amount: 需爬取的数量
        # path: 图片存放路径
        self.keyword = keyword
        self.amount = amount
        self.path = path
        self.thread_pool = Pool(self.thread_amount)
 
    def __del__(self):
        self.thread_pool.close()
        self.thread_pool.join()
 
    # 作用：从必应请求图片
    def request_homepage(self, url):
        # url: 必应图片页的 url
        return requests.get(url, headers=self.headers)
 
    # 作用：解析必应网页，得到所有图片的信息，封装到列表中返回
    # 每个图片的信息以字典对象存储，字典的键包括 image_title, image_type, image_md5, image_url
    def parse_homepage_response(self, response):
        # response: 必应网站的响应
 
        # 获取各图片信息所在的json格式字符串 m
        tree = etree.HTML(response.text)
        m_list = tree.xpath('//*[@class="imgpt"]/a/@m')
 
        # 对每个图片分别处理
        info_list = []
        for m in m_list:
            dic = json.loads(m)
 
            # 去除一些文件名中不允许的字符
            image_title = dic['t']
            for char in self.ignore_chars:
                image_title = image_title.replace(char, ' ')
            image_title = image_title.strip()
 
            # 有些图片的信息中不包含图片格式，该情况将图片设置为 jpg 格式
            image_type = dic['murl'].split('.')[-1]
            if image_type not in self.image_types:
                image_type = 'jpg'
 
            # 将每个图片的信息存为字典格式
            info = dict()
            info['image_title'] = image_title
            info['image_type'] = image_type
            info['image_md5'] = dic['md5']
            info['image_url'] = dic['murl']
 
            info_list.append(info)
        return info_list
 
 
    # 请求具体图片，保存到初始化时指定的路径
    def request_and_save_image(self, info):
        # info: 每个图片的信息,以字典对象存储。字典的键包括 image_title, image_type, image_md5, image_url
        filename = '{} {}.{}'.format(self.count, info['image_title'], info['image_type'])
        filepath = os.path.join(self.path, filename)
 
        try:
            # 请求图片
            response = requests.get(info['image_url'], headers=self.headers, timeout=1.5)
            # 保存图片
            with open(filepath, 'wb') as fp:
                fp.write(response.content)
            # 打印日志
            self.count += 1
            self.success_count += 1
            print('{}: saving {} done.'.format(self.count, filepath))
 
        except requests.exceptions.RequestException as e:
            self.count += 1
            print('{}: saving {}failed. url: {}'.format(self.count, filepath, info['image_url']))
            print('\t tip:', e)
 
 
    # 作用：图片信息的列表去重，去除重复的图片信息
    def deduplication(self, info_list):
        result = []
 
        # 用图片的 md5 做为唯一标识符
        md5_set = set()
        for info in info_list:
            if info['image_md5'] not in md5_set:
                result.append(info)
                md5_set.add(info['image_md5'])
        return result
 
 
    # 作用：运行爬虫，爬取图片
    def run(self):
        # 创建用于保存图片的目录
        if not os.path.exists(self.path):
            os.mkdir(self.path)
 
        # 根据关键词和需要的图片数量，生成将爬取的必应图片网页列表
        homepage_urls = []
        for i in range(int(self.amount/self.per_page_images * 1.5) + 1): # 由于有些图片会重复，故先请求1.5倍图片，豁免
            url = self.bing_image_url_pattern.format(self.keyword, i*self.per_page_images, self.per_page_images)
            homepage_urls.append(url)
        print('homepage_urls len {}'.format(len(homepage_urls)))
 
        # 通过线程池请求所有必应图片网页
        homepage_responses = self.thread_pool.map(self.request_homepage, homepage_urls)
 
        # 从必应网页解析所有图片的信息，每个图片包括 image_title, image_type, image_md5, image_url 等信息。
        info_list = []
        for response in homepage_responses:
            result = self.parse_homepage_response(response)
            info_list += result
        print('info amount before deduplication', len(info_list))
 
        # 删除重复的图片，避免重复下载
        info_list = self.deduplication(info_list)
        print('info amount after deduplication', len(info_list))
        info_list = info_list[ : self.amount]
        print('info amount after split', len(info_list))
 
        # 下载所有图片，并保存
        self.thread_pool.map(self.request_and_save_image, info_list)
        print('all done. {} successfully downloaded, {} failed.'.format(self.success_count, self.count - self.success_count))
 
 
if __name__ == '__main__':
    # 关键词：电脑壁纸
    # 需要的图片数量：100
    # 图片保存路径：'E:\images'
    start = time()
    BingImagesSpider('沙滩美女', 1000, 'identity02').run()
    print(time() - start)
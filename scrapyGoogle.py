from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

query = '动物'
search_url = f"https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={query}&oq={query}"

driver = webdriver.Chrome()
driver.get(search_url)

# 等待页面加载
time.sleep(5)  # 增加等待时间以确保图片加载

# 获取所有缩略图元素
thumbnail_elements = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")

# 点击每个缩略图以获取原始图片
image_urls = []
for img in thumbnail_elements:
    try:
        img.click()
        time.sleep(2)  # 等待原图加载
        large_images = driver.find_elements(By.CSS_SELECTOR, "img.n3VNCb")
        for large_image in large_images:
            src = large_image.get_attribute("src")
            if src.startswith('http'):
                print(f"找到图片链接: {src}")
                image_urls.append(src)
                break
    except Exception as e:
        print("在处理图片时出错:", e)

# 关闭浏览器
driver.quit()

# 打印获取的图片链接
for url in image_urls:
    print(url)

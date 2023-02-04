import io
import os
import time
import requests
import argparse
from PIL import Image
from selenium import webdriver
from config import config as conf


class WebScrapeImage:

    # Initialize class members.
    def __init__(self, query:str, target_folder:str, max_items:int=10, driver_path:str=None, pause_interval:int=1, css_selector:str="img.Q4LuWd",  css_selector_actual:str="img.n3VNCb", need_trace:bool=True):
        self.query = query
        self.max_items = max_items
        self.pause_interval = pause_interval
        self.css_selector = css_selector
        self.css_selector_actual = css_selector_actual
        self.target_folder = target_folder
        self.driver_path = driver_path
        self.image_set = set()
        self.ctr = 0
        self.tracer = 0
        self.need_trace = need_trace

    # Scrolling to the end of the screen.
    def scroll(self, wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(self.pause_interval)

    # Fetching image urls.
    def fetch(self, wd):
        # Defining google query.
        google_search_query = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"
        # Loading the page.
        wd.get(google_search_query.format(q = self.query))
        image_count, results_start, old_results_start = 0, 0, -1
        while image_count < self.max_items:
            self.scroll(wd)
            if(old_results_start == results_start):
                if self.tracer >= 10:
                  break
                else:
                  self.tracer += 1
            # Get image data.
            raw_images_information = wd.find_elements_by_css_selector(self.css_selector)
            raw_images_information_len = len(raw_images_information)
            print(f"Found: {raw_images_information_len} search results. Extracting links from {results_start}:{raw_images_information_len}")
            for i in raw_images_information[results_start:raw_images_information_len]:
                # Selecting images.
                try:
                    if len(self.image_set) >= self.max_items:
                        # Complete generation.
                        break
                    i.click()
                    time.sleep(self.pause_interval)
                except Exception as e:
                    continue
                # Getting the actual image.
                actual_images = wd.find_elements_by_css_selector(self.css_selector_actual)
                for actual_image in actual_images:
                    if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                        self.image_set.add(actual_image.get_attribute('src'))
                    image_count = len(self.image_set)
                    if len(self.image_set) >= self.max_items:
                        # Complete generation.
                        break
                    else:
                        if len(self.image_set) % 50 == 0 and self.need_trace:
                            print(f'Loaded {len(self.image_set)} image links')
                        # Look for more items.
                        load_more_button = wd.find_element_by_css_selector(".mye4qd")
                        if load_more_button:
                            wd.execute_script("document.querySelector('.mye4qd').click();")
                        else:
                            # No load button.
                            break
            # move the result startpoint further down.
            old_results_start = results_start
            results_start = len(raw_images_information)
      
    # Storing images.
    def store_image(self, i:int, url:str):
        try:
            image_content = requests.get(url).content
        except Exception as e:
            print(f"ERROR - Could not download {url} - {e}")
        try:
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file).convert('RGB')
            file_path = self.target_folder + '/' + str(int(time.time()*1000)) + '.jpg'
            with open(file_path, 'wb') as f:
                image.save(f, "JPEG")
            print(f"SUCCESS - saved {url} - as {file_path}")
            self.ctr += 1
        except Exception as e:
            print(f"ERROR - Could not save {url} - {e}")
    
    # Running all the functions.
    def run(self):
        # chrome_options = webdriver.ChromeOptions()
        # chrome_options.add_argument('--headless')
        # chrome_options.add_argument('--no-sandbox')
        # chrome_options.add_argument('--disable-dev-shm-usage')
        # with webdriver.Chrome(executable_path=self.driver_path, options=chrome_options) as wd:
        with webdriver.Firefox(executable_path=self.driver_path) as wd:
            self.fetch(wd)   
        for i, image in enumerate(self.image_set):
            self.store_image(i, image)
        print(f'Total images saved in {self.target_folder} = {self.ctr}')
    

def main(args=None):
    for ind, i in enumerate(args.query):
        try:
            if not os.path.exists(os.path.join(args.output_path, args.classes[ind])):
                os.mkdir(os.path.join(args.output_path, args.classes[ind]))
            scrape = WebScrapeImage(query=i, target_folder=os.path.join(args.output_path, args.classes[ind]), max_items=args.max_items, driver_path=args.driver_path, pause_interval=1.0)
            scrape.run()
        except Exception as e:
            print(e)
            continue

if __name__ == '__main__':
    # Fetching arguments.
    parser = argparse.ArgumentParser(description='Web Scraper.')
    parser.add_argument('-d', '--driver_path', type=str, default=conf.CHROME_DRIVER_PATH, metavar="\b", help='Path to chromedriver')
    parser.add_argument('-c','--classes', nargs='+', help='List of classes', required=True)
    parser.add_argument('-q','--query', nargs='+', help='List of queries; in sync with classes - one to one map', required=True)
    parser.add_argument('-o', '--output_path', type=str, default=conf.WEBSCRAPE_OUTPUT, metavar="\b", help='Path to output directory')
    parser.add_argument('-n', '--max_items', type=int, default=10, metavar="\b", help='Max items to fetch')
    args = parser.parse_args()
    main(args=args)
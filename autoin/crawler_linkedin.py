import time
import json
import logging
import re
import random
import urllib.parse
from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LinkedInScraper:
    """
    Scrapes posts from a list of LinkedIn profiles using Selenium.
    This class is instantiated by the web service for each request.
    """
    def __init__(self, li_at_cookie: str):
        logging.info("Initializing LinkedIn Scraper with Selenium...")
        if not li_at_cookie:
            raise ValueError("The 'li_at' cookie is required for authentication.")
        
        self.li_at = li_at_cookie
        self.driver = None
        
        try:
            logging.info("Setting up Chrome driver...")
            service = Service(ChromeDriverManager().install())
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--log-level=3")
            self.driver = webdriver.Chrome(service=service, options=options)
            self._authenticate_session()
        except Exception as e:
            logging.critical(f"Failed to initialize WebDriver: {e}")
            if self.driver:
                self.driver.quit()
            raise

    def _authenticate_session(self):
        """Authenticates the session by adding the li_at cookie."""
        logging.info("Authenticating session by setting 'li_at' cookie.")
        self.driver.get("https://www.linkedin.com/")
        cookie = {'name': 'li_at', 'value': self.li_at, 'domain': '.linkedin.com'}
        self.driver.add_cookie(cookie)

    def get_profile_posts(self, profile_url: str, num_posts: int = 5):
        """
        Navigates to the profile's activity page, scrolls to load posts,
        and scrapes the content directly from the HTML.
        """
        match = re.search(r'/in/([^/]+)', profile_url)
        if not match:
            raise ValueError(f"Could not extract profile identifier from URL: {profile_url}")
        public_id = match.group(1)
        
        activity_url = f"https://www.linkedin.com/in/{public_id}/recent-activity/shares/"
        logging.info(f"Navigating to activity page: {activity_url}")
        
        try:
            self.driver.get(activity_url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".scaffold-finite-scroll__content"))
            )
            
            logging.info(f"Scrolling to load up to {num_posts} posts...")
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2.5) 

            logging.info("Parsing page source...")
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            posts = []
            feed_items = soup.find_all('div', class_='feed-shared-update-v2', limit=num_posts)

            for item in feed_items:
                post_data = {}
                actors = item.find_all('div', class_='update-components-actor')
                if actors:
                    primary_actor = actors[0]
                    post_data['action'] = 'reposted' if len(actors) > 1 else 'posted'
                    post_data['author_name'] = primary_actor.find('span', class_='update-components-actor__name').text.strip()
                    
                    if post_data['action'] == 'reposted' and len(actors) > 1:
                        original_actor = actors[1]
                        post_data['original_author_name'] = original_actor.find('span', class_='update-components-actor__name').text.strip()
                        timestamp_element = original_actor.find('span', class_='update-components-actor__sub-description')
                    else:
                        timestamp_element = primary_actor.find('span', class_='update-components-actor__sub-description')
                    
                    if timestamp_element:
                        post_data['timestamp'] = timestamp_element.text.strip().split('•')[0].strip()

                commentary_div = item.find('div', class_='update-components-text')
                if commentary_div:
                    post_data['text'] = commentary_div.text.strip()
                
                social_counts = item.find('ul', class_='social-details-social-counts__list')
                if social_counts:
                    likes_item = social_counts.find('li', class_='social-details-social-counts__reactions')
                    post_data['likes'] = likes_item.find('button').get('aria-label', '0').split()[0] if likes_item else '0'
                
                post_urn = item.get('data-urn', '')
                if post_urn:
                     post_data['url'] = f"https://www.linkedin.com/feed/update/{post_urn}"
                
                if post_data.get('text'):
                    posts.append(post_data)
            return posts

        except TimeoutException:
            logging.error(f"Timed out waiting for content on {activity_url}")
            return []
        except Exception as e:
            logging.critical(f"An error occurred while scraping {profile_url}: {e}")
            return []

    def close_session(self):
        if self.driver:
            logging.info("Closing browser session.")
            self.driver.quit()


# --- API Endpoint Definition ---
@app.route('/scrape', methods=['POST'])
def scrape_linkedin_profiles():
    """
    Handles a POST request to scrape posts from a list of LinkedIn profiles.

    This endpoint initiates a scraping process for each URL provided in the
    request body. It uses an authenticated session via the provided 'li_at'
    cookie to access the data.

    Args (JSON Payload):
        li_at (str): The required LinkedIn 'li_at' authentication cookie.
        profile_urls (list): A required list of strings, where each string is a
                             full LinkedIn profile URL.
        num_posts (int, optional): The number of posts to scrape per profile.
                                   Defaults to 5.

    Returns:
        JSON: A JSON object where keys are the profile URLs and values are lists
              of scraped post data. On error, returns a JSON object with an
              'error' key and an appropriate HTTP status code.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    li_at_cookie = data.get('li_at')
    profile_urls = data.get('profile_urls')
    num_posts = data.get('num_posts', 5)

    if not li_at_cookie or not profile_urls:
        return jsonify({"error": "Missing 'li_at' or 'profile_urls' in request"}), 400

    if not isinstance(profile_urls, list):
        return jsonify({"error": "'profile_urls' must be a list of strings"}), 400

    logging.info(f"Received request to scrape {len(profile_urls)} profiles.")
    
    all_results = {}
    scraper = None
    try:
        scraper = LinkedInScraper(li_at_cookie)
        for i, url in enumerate(profile_urls):
            posts = scraper.get_profile_posts(url, num_posts)
            all_results[url] = posts
            
            if i < len(profile_urls) - 1:
                delay = random.uniform(8, 20)
                logging.info(f"Waiting for {delay:.2f} seconds before next profile...")
                time.sleep(delay)

    except Exception as e:
        logging.critical(f"A critical error occurred during the scraping process: {e}")
        return jsonify({"error": "An internal error occurred. Check service logs."}), 500
    finally:
        if scraper:
            scraper.close_session()
            
    logging.info("Scraping request completed successfully.")
    return jsonify(all_results)

# --- Main Execution Block ---
if __name__ == '__main__':
    # Runs the Flask web server
    # Set debug=False for production use
    app.run(host='0.0.0.0', port=5000, debug=False)

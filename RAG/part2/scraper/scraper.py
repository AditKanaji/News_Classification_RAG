import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import time
import re

def setup_driver():
    """Setup Chrome driver in headless mode"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def scrape_bbc_news(driver, num_articles=10):
    """Scrape BBC News specifically"""
    scraped_data = []
    
    try:
        # Find all article links on BBC
        links = driver.find_elements(By.CSS_SELECTOR, 'a[data-testid="internal-link"]')
        
        # Filter for actual article links
        article_links = []
        for link in links:
            href = link.get_attribute('href')
            if href and '/articles/' in href and href not in article_links:
                article_links.append(href)
        
        print(f"Found {len(article_links)} article links")
        
        # Visit each article
        for idx, article_url in enumerate(article_links[:num_articles], 1):
            try:
                print(f"Scraping article {idx}/{num_articles}: {article_url}")
                driver.get(article_url)
                time.sleep(2)
                
                # Extract headline
                headline = ""
                try:
                    headline_elem = driver.find_element(By.CSS_SELECTOR, 'h1, [data-component="headline-block"]')
                    headline = headline_elem.text.strip()
                except:
                    try:
                        headline = driver.find_element(By.TAG_NAME, 'h1').text.strip()
                    except:
                        headline = "No headline found"
                
                # Extract author
                author = "BBC News"
                try:
                    author_elem = driver.find_element(By.CSS_SELECTOR, '[data-component="byline-block"], .byline')
                    author = author_elem.text.strip()
                except:
                    pass
                
                # Extract date
                date_published = datetime.now().strftime('%Y-%m-%d')
                try:
                    date_elem = driver.find_element(By.CSS_SELECTOR, 'time')
                    date_published = date_elem.get_attribute('datetime') or date_elem.text
                    if date_published:
                        date_published = date_published.split('T')[0]
                except:
                    pass
                
                # Extract summary (first paragraph)
                summary = ""
                try:
                    paragraphs = driver.find_elements(By.CSS_SELECTOR, '[data-component="text-block"] p, article p')
                    if paragraphs:
                        summary = paragraphs[0].text.strip()[:300]
                except:
                    summary = headline
                
                # Extract image
                image_url = ""
                try:
                    img = driver.find_element(By.CSS_SELECTOR, 'article img, [data-component="image-block"] img')
                    image_url = img.get_attribute('src')
                except:
                    pass
                
                # Extract category from URL
                category = "News"
                if '/sport/' in article_url:
                    category = "Sports"
                elif '/business/' in article_url:
                    category = "Business"
                elif '/technology/' in article_url:
                    category = "Technology"
                elif '/entertainment/' in article_url:
                    category = "Entertainment"
                
                article_data = {
                    'ID': idx,
                    'URL': article_url,
                    'Headline': headline,
                    'Category': category,
                    'Author': author,
                    'Date_Published': date_published,
                    'Sentiment': 'Neutral',
                    'Keywords': ', '.join([w for w in headline.split() if len(w) > 4][:5]),
                    'Summary': summary,
                    'Image_URL': image_url,
                    'Source': 'BBC News'
                }
                
                scraped_data.append(article_data)
                print(f"✓ Scraped: {headline[:60]}...")
                
            except Exception as e:
                print(f"✗ Error scraping article {idx}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return scraped_data

def scrape_generic_news(driver, url, num_articles=10):
    """Generic scraper for other news sites"""
    scraped_data = []
    
    try:
        # Try multiple selector strategies
        articles = []
        
        # Strategy 1: Find all links with headlines
        links = driver.find_elements(By.CSS_SELECTOR, 'a')
        
        seen_urls = set()
        for link in links:
            try:
                href = link.get_attribute('href')
                text = link.text.strip()
                
                # Filter for article-like links
                if (href and text and len(text) > 20 and 
                    href.startswith('http') and href not in seen_urls):
                    
                    seen_urls.add(href)
                    
                    # Try to get image from parent
                    image_url = ""
                    try:
                        parent = link.find_element(By.XPATH, '..')
                        img = parent.find_element(By.TAG_NAME, 'img')
                        image_url = img.get_attribute('src')
                    except:
                        pass
                    
                    article_data = {
                        'ID': len(scraped_data) + 1,
                        'URL': href,
                        'Headline': text,
                        'Category': 'General',
                        'Author': 'Unknown',
                        'Date_Published': datetime.now().strftime('%Y-%m-%d'),
                        'Sentiment': 'Neutral',
                        'Keywords': ', '.join([w for w in text.split() if len(w) > 4][:5]),
                        'Summary': text[:200],
                        'Image_URL': image_url,
                        'Source': url.split('/')[2] if '/' in url else url
                    }
                    
                    scraped_data.append(article_data)
                    print(f"✓ Found: {text[:60]}...")
                    
                    if len(scraped_data) >= num_articles:
                        break
                        
            except:
                continue
                
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return scraped_data

def scrape_news_site(url, num_articles=10):
    """Main scraping function"""
    driver = setup_driver()
    scraped_data = []
    
    try:
        print(f"Loading {url}...")
        driver.get(url)
        time.sleep(3)
        
        # Use specific scraper for BBC
        if 'bbc.com' in url:
            scraped_data = scrape_bbc_news(driver, num_articles)
        else:
            scraped_data = scrape_generic_news(driver, url, num_articles)
    
    except Exception as e:
        print(f"Error loading page: {str(e)}")
    
    finally:
        driver.quit()
    
    return scraped_data

def save_sample_to_json(data, filename='sample_annotated_article.txt'):
    """Save one sample article as JSON to a text file for documentation"""
    import json
    
    if not data:
        print("No data to save!")
        return
    
    # Take the first article as sample
    sample_article = data[0]
    
    # Create a nicely formatted JSON
    json_output = json.dumps(sample_article, indent=4, ensure_ascii=False)
    
    # Save to text file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SAMPLE ANNOTATED NEWS ARTICLE\n")
        f.write("=" * 70 + "\n\n")
        f.write(json_output)
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("Field Descriptions:\n")
        f.write("=" * 70 + "\n")
        f.write("ID: Unique identifier for the article\n")
        f.write("URL: Source URL of the article\n")
        f.write("Headline: Main title/headline of the article\n")
        f.write("Category: Topic category (e.g., Technology, Sports, Business)\n")
        f.write("Author: Article author or publisher name\n")
        f.write("Date_Published: Publication date in YYYY-MM-DD format\n")
        f.write("Sentiment: Sentiment analysis (Positive/Neutral/Negative)\n")
        f.write("Keywords: Key terms extracted from the article\n")
        f.write("Summary: Brief summary of article content\n")
        f.write("Image_URL: URL of the article's main image\n")
        f.write("Source: Source website domain\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ Sample article saved to {filename} in JSON format")
    print(f"  This file can be included in your project report\n")

def save_to_csv(data, filename='news_scraped_data.csv'):
    """Save scraped data to CSV using pandas"""
    if not data:
        print("No data to save!")
        return
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n{'='*60}")
    print(f"✓ Data saved to {filename}")
    print(f"✓ Total articles scraped: {len(df)}")
    print('='*60)
    print("\nPreview of scraped data:")
    print(df[['ID', 'Headline', 'Category', 'Date_Published']].to_string())
    print(f"\nFull data columns: {', '.join(df.columns)}")

# Example usage
if __name__ == "__main__":
    # Example news websites
    news_urls = [
        'https://www.bbc.com/news',
        # 'https://news.ycombinator.com/',
        # 'https://www.theguardian.com/international',
    ]
    
    all_data = []
    
    for url in news_urls:
        print(f"\n{'='*60}")
        print(f"Scraping: {url}")
        print('='*60)
        
        data = scrape_news_site(url, num_articles=5)
        all_data.extend(data)
        
        time.sleep(2)
    
    # Save all scraped data
    if all_data:
        save_to_csv(all_data)
        save_sample_to_json(all_data)  # Save sample for project report
    else:
        print("\n✗ No data was scraped. The website structure may have changed.")
        print("Tips:")
        print("1. Try a different news website")
        print("2. Check if the site blocks automated access")
        print("3. Inspect the HTML and update selectors")
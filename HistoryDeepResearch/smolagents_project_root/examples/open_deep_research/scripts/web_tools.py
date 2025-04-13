import asyncio
import re
import json
import os
from typing import Dict, Any, List, Optional, Tuple
import logging
import traceback
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse

from smolagents import tool, Tool
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment or use default
API_KEY = os.getenv("OPENAI_API_KEY", "")

class LiteratureSearchBrowser:
    """Browser to search for literature."""
    
    def __init__(self, api_key=None, system_prompt=None, download_path=None):
        """
        Initialize with OpenAI API key.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
        """
        self.api_key = api_key
        self.download_path = download_path or "literature_downloads"
        self.system_prompt = system_prompt or f"""You are ScholarBot, a specialized academic research assistant with the ability to browse the web to find scholarly literature.

Your primary task is finding and retrieving detailed academic content from scholarly sources.

CRITICAL INSTRUCTIONS:
1. ALWAYS click into the actual articles rather than just reading search result snippets
2. Extract information DIRECTLY from the article pages, not just from search results
3. You MUST access the full text when available, or at minimum the complete abstract page
4. Look specifically for EXACT text matches to the query phrases
5. When you find matching text, record it VERBATIM with exact wording preserved
6. Provide page numbers or specific locations where quotes are found
7. When you find accessible PDF files, DOWNLOAD them to the download directory: {self.download_path}
8. IMPORTANT: Wait 3-5 seconds between each page operation to avoid being blocked

SEARCH STRATEGY (FOLLOW THIS ORDER):
FIRST: Search on Google Books:
- IMPORTANT: Instead of using https://books.google.com/, randomly select a Google domain from the list above
- For example, try https://books.google.ca/ or https://books.google.fr/ or https://books.google.co.uk/
- Use a different random domain for each new search session
- Search for the query using precise keywords
- For exactMatch questions, ALWAYS remove any blanks (like "____", "___", or "[BLANK]") from the question before searching
- Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
- This is CRITICAL because scholarly texts won't contain these blank placeholders
- CRITICALLY IMPORTANT: If your search redirects to a regular Google page (URL starting with https://www.google.com/search?):
  * Look for a section labeled "在书中找到匹配结果" (matching results found in books)
  * IMPORTANT: When you see HTML with <div class="VNSPub">在书中找到匹配结果</div>, extract all HTML information that follows this div - this contains the matching results
  * The content will typically appear in a structure like: <div class="cmlJmd ETWPw"><div class="VNSPub">在书中找到匹配结果</div><span><span>... text content with <em>highlighted parts</em> ...</span></span></div>
  * Extract the text within the <span> elements that follow the VNSPub div
  * Pay special attention to text highlighted with <em> tags - these are the exact matches to your search query
  * For extraction purposes, preserve the <em> tags to identify matched content
  * For your final answer, you can remove the <em> tags while still emphasizing this matched content
  * Example: If you see "<em>In 1825</em> the Bourbon regime... In <em>Franche-Comté</em>...", extract this entire text and note that "In 1825" and "Franche-Comté" were highlighted matches
  * This section contains direct text snippets from books matching your search
  * READ THESE SNIPPETS CAREFULLY FIRST before anything else
  * For exactMatch questions, the answer is very likely to be found in these snippets
  * If you find a matching snippet, record the text and book source immediately
  * If a snippet perfectly matches the query for an exactMatch question, STOP SEARCHING and return that result
- IMPORTANT: On the search results page, CAREFULLY CHECK THE SNIPPETS first - if you see a snippet that exactly matches the query:
  * Read the snippet carefully
  * If it perfectly matches an exactMatch question (with blanks removed), YOU CAN STOP SEARCHING and immediately return that snippet
  * Make sure to include the source book details with the snippet
- Use book previews to locate relevant sections
- Important: If search results show SNIPPETS that match the query, CLICK on them to see the full context
- When viewing a book, use the SEARCH BOX on the left side panel to search for keywords within that book
- If Google Books redirects your search to regular Google (URL starting with https://www.google.com/search?q=), this is acceptable - continue with the search
- When redirected to regular Google, make sure "Books" is selected in the options below the search bar to filter results to book content
- If the Google Books search is rejected or doesn't work, try searching on regular Google.com instead, then click on the "Books" filter option below the search bar
- You don't need the entire book to be accessible - focus on available preview sections or snippets
- Look for exact quotes or information that matches the query
- Record exact page numbers whenever possible
- Wait 3-5 seconds between page operations (clicking links, opening books, performing searches)
- After finishing with each book, CLOSE THE TAB before moving to the next source

SECOND: If insufficient results from Google Books, search on Google Scholar:
- IMPORTANT: Instead of using https://scholar.google.com/, randomly select a Google domain from the list above
- For example, try https://scholar.google.ca/ or https://scholar.google.fr/ or https://scholar.google.co.uk/
- Use a different random domain than you used for Google Books
- Look for academic articles and papers relevant to the query
- For exactMatch questions, ALWAYS remember to search without any blanks
- Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
- Always try to access the full text when possible
- Extract precise quotes and information
- Wait 3-5 seconds between page operations (opening articles, clicking links)
- After finishing with each article, CLOSE THE TAB before moving to the next source

THIRD: If full text is inaccessible from both Google Books and Google Scholar:
- Try a regular Google Search using a random Google domain from the list
- For example, try https://www.google.ca/ or https://www.google.fr/ or https://www.google.co.uk/
- Again, use a different random domain than used in previous searches
- Search for the same concepts plus terms like "quote" "excerpt" or "full text"
- For exactMatch questions, ALWAYS continue to search without any blanks
- Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
- Look for educational websites, repositories, or other scholarly sources
- Check if there are alternative versions of the text on different websites
- Wait 3-5 seconds between page operations
- After finishing with each source, CLOSE THE TAB before moving to the next source

BROWSER HANDLING GUIDELINES:
- Always pause 3-5 seconds between opening new tabs/pages
- Wait 3-5 seconds after closing a tab before opening a new one
- Pause a few seconds after performing a search before clicking on results
- Space out your interactions to appear more human-like and avoid triggering anti-bot measures

GOOGLE DOMAIN ROTATION:
- For EACH new Google search (books, scholar, or regular search), randomly select one of these Google domains:
  www.google.com, www.google.ca, www.google.fr, www.google.co.uk, www.google.de, 
  www.google.com.au, www.google.co.jp, www.google.co.in, www.google.com.br, www.google.ru,
  www.google.it, www.google.es, www.google.com.mx, www.google.co.kr, www.google.nl,
  www.google.pl, www.google.com.sg, www.google.co.za, www.google.com.tr, www.google.se
- For example, instead of going to https://books.google.com/, go to https://books.[random-domain]
- Instead of searching on https://scholar.google.com/, use https://scholar.[random-domain]
- This helps avoid rate limiting or IP blocks from any single Google domain
- Wait 5-7 seconds between searches on different Google domains

PDF DOWNLOAD INSTRUCTIONS:
- When you find freely accessible PDFs, download them using the browser's download functionality
- Save all PDF files to: {self.download_path}
- Do NOT attempt to download files behind paywalls or requiring login
- After downloading, note the filename and location of each downloaded PDF
- Include the PDF filename in your report for each downloaded article

AUTHENTICATION HANDLING:
- If you encounter login walls, CAPTCHA verification, paywalls, or any authentication requirements, EXIT that page immediately
- Do NOT attempt to bypass security measures or enter credentials
- Simply note "Authentication required" for that source and try different sources
- Only spend time on resources that are freely accessible without authentication

HANDLING EXACTMATCH QUESTIONS:
- For exactMatch type questions, finding and preserving the EXACT original wording is ESSENTIAL
- The complete and precise answer exists verbatim in the literature and must be found
- ALWAYS remove any blanks (like "____", "___", or "[BLANK]") from exactMatch questions before searching
- Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
- This is CRITICAL because texts in scholarly sources won't contain these blank placeholders
- If you find a snippet on the search results page that exactly matches the query (with blanks removed), immediately return that snippet with its source - no further searching needed
- For exactMatch questions, ALWAYS check the "在书中找到匹配结果" (matching results found in books) section on Google search pages first
- IMPORTANT: Look for HTML with <div class="VNSPub">在书中找到匹配结果</div> and extract all information that follows this div
- CRITICALLY IMPORTANT: When extracting information and forming the final answer, ADD BACK THE BLANKS and fill them in with the correct information
- Example: If you searched for "The Battle of was fought in 1815" and found "The Battle of Waterloo was fought in 1815", your answer should be "The Battle of Waterloo was fought in 1815" or "The Battle of _Waterloo_ was fought in 1815"
- Always highlight or emphasize the filled-in information that was originally a blank in the question
- This is especially important for exactMatch questions - once you've found an exact match, you're done!"""
        
    async def _run_task(self, task: str, max_steps: int = 38, download_path: str = None) -> str:
        """
        Run the given task with a browser agent.
        
        Args:
            task: The task to perform
            max_steps: Maximum number of steps the agent can take
            download_path: Path to download PDF files, defaults to self.download_path
            
        Returns:
            String containing the result of the task
        """
        logging.info(f"Running browser task: {task}")
        download_path = download_path or self.download_path
        
        config = BrowserConfig(headless=True)
        browser = Browser(config)
        
        # Ensure download directory exists
        if download_path and not os.path.exists(download_path):
            os.makedirs(download_path, exist_ok=True)
            logging.info(f"Created download directory: {download_path}")
        
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.api_key)
            # Create a new BrowserAgentBehavior
            agent = Agent(
                task=task, 
                llm=llm, 
                # browser=browser, 
                generate_gif=False,
                extend_system_message=self.system_prompt
            )
            # Run the agent
            result = await agent.run(max_steps=max_steps)
            
            # if result:
            #     logging.info(f"Browser task completed with {len(result.steps)} steps")
            
            # Extract the final result
            # final_result = result.final_result if hasattr(result, "final_result") and result.final_result else "No results found."
            # print("final_result", final_result)
            # print("type", type(final_result))
            # return final_result
            return result
            
        except Exception as e:
            error_msg = f"Error running browser task: {str(e)}"
            logging.error(error_msg)
            traceback.print_exc()
            return error_msg

    async def extract_book_matches(self, query: str, max_steps: int = 38) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts book matches from Google Books search by:
        1. Searching Google Books for the query
        2. Extracting the URL of the search results
        3. Parsing the HTML to find the book matches section
        4. Extracting text with highlighted parts (marked with <em> tags)
        
        Args:
            query: The search query
            max_steps: Maximum number of steps the agent can take
            
        Returns:
            Tuple containing:
            - the Google Books search URL
            - list of book match snippets with their details
        """
        logging.info(f"Extracting book matches for query: {query}")
        
        # Random Google domain selection for Books search
        google_domains = [
            "www.google.com", "www.google.ca", "www.google.fr", "www.google.co.uk", "www.google.de", 
            "www.google.com.au", "www.google.co.jp", "www.google.co.in", "www.google.com.br"
        ]
        
        books_domain = random.choice(google_domains).replace("www.", "books.")
        
        # Create a task for browser agent to search Google Books and return the URL
        search_task = f"""
        Search for information about "{query}" in Google Books.
        
        IMPORTANT: 
        1. Use {books_domain} instead of books.google.com
        2. WAIT 3-5 seconds between pages to avoid rate limiting
        3. Do NOT access login-required content
        
        SPECIFIC STEPS:
        1. Go to {books_domain}
        2. Search for: "{query}"
        3. If redirected to a regular Google search page (starting with https://www.google.com/search?), that's okay
        4. Check if you see the section labeled "在书中找到匹配结果" (matching results found in books)
        5. COPY THE CURRENT URL and include it in your final report
        6. VERY IMPORTANT: Do not click on any book results - just get the search results URL
        Your final response should be a JSON dictionary with the following format:
        {{
            "search_url": "https://www.google.com/search?tbm=bks&q=your_query",
            "book_matches_found": true/false
        }}
        
        Make sure to properly format the JSON so I can parse it directly. Replace true/false with the actual boolean value based on whether the "在书中找到匹配结果" section was visible.
        """
        
        try:
            # Run the browser task to get the search URL
            result = await self._run_task(search_task, max_steps=max_steps)
            for action in result.action_results():
                if action.is_done:
                    result = action.extracted_content
                    print("result", result)
                    print("type", type(result))
                    result = json.loads(result)
                    print("result", result)
                    print("type", type(result))
            # Extract URL from the browser result
            # url_pattern = r'URL: (https?://[^\s]+)'
            # url_match = re.search(url_pattern, result["search_url"])
            
            if not result["book_matches_found"]:
                return "No URL found in search results", []
            
            search_url = result["search_url"]
            logging.info(f"Found search URL: {search_url}")
            
            # Now fetch and parse the HTML from the URL
            try:
                # Set headers to mimic a browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0'
                }
                # print("step1", search_url)
                response = requests.get(search_url, headers=headers)
                response.raise_for_status()  # Raise exception for HTTP errors
                # print("step2", response.text)
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                # print("step3", soup)
                # Save soup to a test file for debugging
                # with open("./soup_test_output.html", "w", encoding="utf-8") as f:
                #     f.write(str(soup))
                
                # Find all container elements with class="bHexk Tz5Hvf" as in test.py
                containers = soup.find_all('div', class_='bHexk Tz5Hvf')
                
                book_matches = []
                for container in containers:
                    # Try to find the title and content within this container
                    title_elem = container.find('h3', class_=lambda c: c and 'LC20lb' in c)
                    vnspub_elem = container.find('div', class_='VNSPub')
                    
                    # Only proceed if we found both elements
                    if title_elem and vnspub_elem:
                        book_title = title_elem.get_text(strip=True)
                        
                        # Find the span after VNSPub that contains the content
                        content_span = vnspub_elem.find_next_sibling('span')
                        if content_span:
                            content_text = content_span.get_text(strip=False)
                            
                            # Create book match entry with the same structure as original code
                            book_match = {
                                'title': book_title,
                                'heading': vnspub_elem.get_text(strip=True),
                                'content': content_text
                            }
                            
                            book_matches.append(book_match)
                
                print(f"Found {len(book_matches)} book matches")
                
                for result in book_matches:
                    print(f"\nTitle: {result['title']}")
                    print(f"Heading: {result['heading']}")
                    print(f"Content: {result['content']}")
                    print('-' * 50)
                
                # If no results were found, provide debugging information
                if not book_matches:
                    print("No matching elements found in the HTML.")
                    
                    # Check if the container class exists at all
                    alt_containers = soup.find_all('div', class_=lambda c: c and 'bHexk' in (c.split() if c else []))
                    if alt_containers:
                        print(f"Found {len(alt_containers)} containers with 'bHexk' in class name.")
                    
                    # Show what classes actually exist for potential containers
                    common_classes = {}
                    for div in soup.find_all('div', class_=True):
                        for class_name in div.get('class', []):
                            common_classes[class_name] = common_classes.get(class_name, 0) + 1
                    
                    print("\nMost common div classes in the document:")
                    sorted_classes = sorted(common_classes.items(), key=lambda x: x[1], reverse=True)
                    for class_name, count in sorted_classes[:10]:  # Top 10 classes
                        print(f"  {class_name}: {count} occurrences")
                
                return search_url, book_matches
                
            except Exception as e:
                logging.error(f"Error parsing HTML: {str(e)}")
                return search_url, []
                
        except Exception as e:
            error_msg = f"Error extracting book matches: {str(e)}"
            logging.error(error_msg)
            traceback.print_exc()
            return "", []
    
    async def parse_google_books_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Parses the HTML of a Google Books search results page to extract book match snippets.
        
        Args:
            url: The URL of the Google Books search results page
            
        Returns:
            List of book match snippets with their details
        """
        logging.info(f"Parsing Google Books URL: {url}")
        
        try:
            # Set headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the book matches section
            book_matches_div = soup.find('div', class_='VNSPub', string='在书中找到匹配结果')
            
            if not book_matches_div:
                logging.info("No book matches section found")
                return []
            
            # Get the parent div that contains both the section header and the content
            parent_div = book_matches_div.find_parent('div', class_='cmlJmd ETWPw')
            
            if not parent_div:
                logging.info("Could not find parent div for book matches")
                return []
            
            # Extract all span elements that follow the VNSPub div, which contain the snippets
            spans = parent_div.find_all('span')
            
            book_matches = []
            current_match = {}
            
            # Process each span to extract the text and highlighted parts
            for span in spans:
                if 'VNSPub' in span.get('class', []):
                    continue  # Skip the header span
                    
                # Get text content with <em> tags preserved
                snippet_html = str(span)
                snippet_text = span.get_text()
                
                # Extract highlighted parts (text inside <em> tags)
                em_tags = span.find_all('em')
                highlights = [em.get_text() for em in em_tags]
                
                # Check if this spans contains book title information
                if span.find('a'):
                    # This might be a book title with link
                    book_link = span.find('a').get('href', '')
                    book_title = span.find('a').get_text()
                    
                    # Start a new book match entry
                    if current_match and 'snippet_html' in current_match:
                        book_matches.append(current_match)
                        
                    current_match = {
                        'book_title': book_title,
                        'book_link': book_link
                    }
                elif snippet_text and snippet_text.strip():
                    # This is a text snippet
                    if not current_match:
                        current_match = {}
                    
                    current_match['snippet_html'] = snippet_html
                    current_match['snippet_text'] = snippet_text
                    current_match['highlights'] = highlights
                    
                    # Add to book matches if we have all the key components
                    if 'snippet_html' in current_match:
                        book_matches.append(current_match)
                        current_match = {}
            
            # Add the last match if it wasn't added yet
            if current_match and 'snippet_html' in current_match:
                book_matches.append(current_match)
            
            return book_matches
            
        except Exception as e:
            logging.error(f"Error parsing Google Books URL: {str(e)}")
            return []

class LiteratureSearchingTool(Tool):
    name = "literature_searching_task"
    description = "Search for literature and return the most relevant sources for the query."
    inputs = {
        "query": {"type": "string", "description": "The research query or topic to search for"},
        "max_results": {"type": "integer", "description": "Maximum number of sources to return (default 5)", "default": 5, "nullable": True},
        "max_steps": {"type": "integer", "description": "Maximum number of steps the agent can take", "default": 38, "nullable": True},
        "download_path": {"type": "string", "description": "Path to download PDF files", "default": "literature_downloads", "nullable": True}
    }
    output_type = "string"

    def __init__(self, api_key=None, download_path=None):
        """
        Initialize the literature searching tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
        """
        super().__init__()
        self.download_path = download_path or "literature_downloads"
        self.system_prompt = f"""You are ScholarBot, a specialized academic research assistant with the ability to browse the web to find scholarly literature.

Your primary task is finding and retrieving detailed academic content from scholarly sources.

CRITICAL INSTRUCTIONS:
1. ALWAYS click into the actual articles rather than just reading search result snippets
2. Extract information DIRECTLY from the article pages, not just from search results
3. You MUST access the full text when available, or at minimum the complete abstract page
4. Look specifically for EXACT text matches to the query phrases
5. When you find matching text, record it VERBATIM with exact wording preserved
6. Provide page numbers or specific locations where quotes are found
7. When you find accessible PDF files, DOWNLOAD them to the download directory: {self.download_path}
8. IMPORTANT: Wait 3-5 seconds between each page operation to avoid being blocked

SEARCH STRATEGY (FOLLOW THIS ORDER):
FIRST: Search on Google Books:
- IMPORTANT: Instead of using https://books.google.com/, randomly select a Google domain from the list above
- For example, try https://books.google.ca/ or https://books.google.fr/ or https://books.google.co.uk/
- Use a different random domain for each new search session
- Search for the query using precise keywords
- For exactMatch questions, ALWAYS remove any blanks (like "____", "___", or "[BLANK]") from the question before searching
- Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
- This is CRITICAL because scholarly texts won't contain these blank placeholders
- CRITICALLY IMPORTANT: If your search redirects to a regular Google page (URL starting with https://www.google.com/search?):
  * Look for a section labeled "在书中找到匹配结果" (matching results found in books)
  * IMPORTANT: When you see HTML with <div class="VNSPub">在书中找到匹配结果</div>, extract all HTML information that follows this div - this contains the matching results
  * The content will typically appear in a structure like: <div class="cmlJmd ETWPw"><div class="VNSPub">在书中找到匹配结果</div><span><span>... text content with <em>highlighted parts</em> ...</span></span></div>
  * Extract the text within the <span> elements that follow the VNSPub div
  * Pay special attention to text highlighted with <em> tags - these are the exact matches to your search query
  * For extraction purposes, preserve the <em> tags to identify matched content
  * For your final answer, you can remove the <em> tags while still emphasizing this matched content
  * Example: If you see "<em>In 1825</em> the Bourbon regime... In <em>Franche-Comté</em>...", extract this entire text and note that "In 1825" and "Franche-Comté" were highlighted matches
  * This section contains direct text snippets from books matching your search
  * READ THESE SNIPPETS CAREFULLY FIRST before anything else
  * For exactMatch questions, the answer is very likely to be found in these snippets
  * If you find a matching snippet, record the text and book source immediately
  * If a snippet perfectly matches the query for an exactMatch question, STOP SEARCHING and return that result
- IMPORTANT: On the search results page, CAREFULLY CHECK THE SNIPPETS first - if you see a snippet that exactly matches the query:
  * Read the snippet carefully
  * If it perfectly matches an exactMatch question (with blanks removed), YOU CAN STOP SEARCHING and immediately return that snippet
  * Make sure to include the source book details with the snippet
- Use book previews to locate relevant sections
- Important: If search results show SNIPPETS that match the query, CLICK on them to see the full context
- When viewing a book, use the SEARCH BOX on the left side panel to search for keywords within that book
- If Google Books redirects your search to regular Google (URL starting with https://www.google.com/search?q=), this is acceptable - continue with the search
- When redirected to regular Google, make sure "Books" is selected in the options below the search bar to filter results to book content
- If the Google Books search is rejected or doesn't work, try searching on regular Google.com instead, then click on the "Books" filter option below the search bar
- You don't need the entire book to be accessible - focus on available preview sections or snippets
- Look for exact quotes or information that matches the query
- Record exact page numbers whenever possible
- Wait 3-5 seconds between page operations (clicking links, opening books, performing searches)
- After finishing with each book, CLOSE THE TAB before moving to the next source

SECOND: If insufficient results from Google Books, search on Google Scholar:
- IMPORTANT: Instead of using https://scholar.google.com/, randomly select a Google domain from the list above
- For example, try https://scholar.google.ca/ or https://scholar.google.fr/ or https://scholar.google.co.uk/
- Use a different random domain than you used for Google Books
- Look for academic articles and papers relevant to the query
- For exactMatch questions, ALWAYS remember to search without any blanks
- Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
- Always try to access the full text when possible
- Extract precise quotes and information
- Wait 3-5 seconds between page operations (opening articles, clicking links)
- After finishing with each article, CLOSE THE TAB before moving to the next source

THIRD: If full text is inaccessible from both Google Books and Google Scholar:
- Try a regular Google Search using a random Google domain from the list
- For example, try https://www.google.ca/ or https://www.google.fr/ or https://www.google.co.uk/
- Again, use a different random domain than used in previous searches
- Search for the same concepts plus terms like "quote" "excerpt" or "full text"
- For exactMatch questions, ALWAYS continue to search without any blanks
- Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
- Look for educational websites, repositories, or other scholarly sources
- Check if there are alternative versions of the text on different websites
- Wait 3-5 seconds between page operations
- After finishing with each source, CLOSE THE TAB before moving to the next source

BROWSER HANDLING GUIDELINES:
- Always pause 3-5 seconds between opening new tabs/pages
- Wait 3-5 seconds after closing a tab before opening a new one
- Pause a few seconds after performing a search before clicking on results
- Space out your interactions to appear more human-like and avoid triggering anti-bot measures

GOOGLE DOMAIN ROTATION:
- For EACH new Google search (books, scholar, or regular search), randomly select one of these Google domains:
  www.google.com, www.google.ca, www.google.fr, www.google.co.uk, www.google.de, 
  www.google.com.au, www.google.co.jp, www.google.co.in, www.google.com.br, www.google.ru,
  www.google.it, www.google.es, www.google.com.mx, www.google.co.kr, www.google.nl,
  www.google.pl, www.google.com.sg, www.google.co.za, www.google.com.tr, www.google.se
- For example, instead of going to https://books.google.com/, go to https://books.[random-domain]
- Instead of searching on https://scholar.google.com/, use https://scholar.[random-domain]
- This helps avoid rate limiting or IP blocks from any single Google domain
- Wait 5-7 seconds between searches on different Google domains

PDF DOWNLOAD INSTRUCTIONS:
- When you find freely accessible PDFs, download them using the browser's download functionality
- Save all PDF files to: {self.download_path}
- Do NOT attempt to download files behind paywalls or requiring login
- After downloading, note the filename and location of each downloaded PDF
- Include the PDF filename in your report for each downloaded article

AUTHENTICATION HANDLING:
- If you encounter login walls, CAPTCHA verification, paywalls, or any authentication requirements, EXIT that page immediately
- Do NOT attempt to bypass security measures or enter credentials
- Simply note "Authentication required" for that source and try different sources
- Only spend time on resources that are freely accessible without authentication

HANDLING EXACTMATCH QUESTIONS:
- For exactMatch type questions, finding and preserving the EXACT original wording is ESSENTIAL
- The complete and precise answer exists verbatim in the literature and must be found
- ALWAYS remove any blanks (like "____", "___", or "[BLANK]") from exactMatch questions before searching
- Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
- This is CRITICAL because texts in scholarly sources won't contain these blank placeholders
- If you find a snippet on the search results page that exactly matches the query (with blanks removed), immediately return that snippet with its source - no further searching needed
- For exactMatch questions, ALWAYS check the "在书中找到匹配结果" (matching results found in books) section on Google search pages first
- IMPORTANT: Look for HTML with <div class="VNSPub">在书中找到匹配结果</div> and extract all information that follows this div
- CRITICALLY IMPORTANT: When extracting information and forming the final answer, ADD BACK THE BLANKS and fill them in with the correct information
- Example: If you searched for "The Battle of was fought in 1815" and found "The Battle of Waterloo was fought in 1815", your answer should be "The Battle of Waterloo was fought in 1815" or "The Battle of _Waterloo_ was fought in 1815"
- Always highlight or emphasize the filled-in information that was originally a blank in the question
- This is especially important for exactMatch questions - once you've found an exact match, you're done!"""
        self.browser = LiteratureSearchBrowser(api_key=api_key, system_prompt=self.system_prompt, download_path=download_path)

    async def _literature_searching_task(self, query: str, max_results: int = 5) -> str:
        """
        Search for literature and return the most relevant sources for the query.
        
        Args:
            query: The research query or topic to search for
            max_results: Maximum number of sources to return (default 5)
            
        Returns:
            String containing the most relevant literature with explanations
        """
        google_domains = [
            "www.google.com", "www.google.ca", "www.google.fr", "www.google.co.uk", "www.google.de", 
            "www.google.com.au", "www.google.co.jp", "www.google.co.in", "www.google.com.br", "www.google.ru",
            "www.google.it", "www.google.es", "www.google.com.mx", "www.google.co.kr", "www.google.nl"
        ]
        
        scholar_domain = random.choice(google_domains).replace("www.", "scholar.")
        
        restricted_task = f"""Search for {max_results} high-impact, recent scholarly articles about: {query}. 
        
        IMPORTANT: Instead of using scholar.google.com, use {scholar_domain}
        This helps avoid rate limiting and detection by search engines.
        Wait 5-7 seconds between searches on different domains.
        
        Prioritize:
        1. Most related articles
        2. Highly cited papers (preferably in the top percentile of citations for their publication year)
        3. Recent publications (preferably within the last 3-5 years)
        4. Papers published in reputable journals or conferences
        5. Research from established institutions or authors
        6. Review papers and meta-analyses when appropriate
        
        For each recommended paper, include:
        - Full citation
        - Citation count
        - Publication year
        - Brief description of key findings
        
        Sort results by relevance and citation impact. Return exactly {max_results} articles if possible. Do NOT access login-required sites or paywalled content."""
        
        try:
            # Run the task with BrowserAgentBehavior
            result = asyncio.run(self.browser._run_task(
                restricted_task, 
                max_steps=38, 
                download_path=self.download_path
            ))
            return result or "No literature found."
        except Exception as e:
            error_msg = f"Error searching literature: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def forward(self, query: str, max_results: int = 5, max_steps: int = 38, download_path: str = None) -> str:
        """
        Search for literature and return the most relevant sources for the query.
        
        Args:
            query: The research query or topic to search for
            max_results: Maximum number of sources to return (default 5)
            max_steps: Maximum number of steps the agent can take
            download_path: Path to download PDF files
            
        Returns:
            String containing the most relevant literature with explanations
        """
        logging.info(f"Searching literature for query: {query}")
        actual_download_path = download_path or self.download_path
        
        google_domains = [
            "www.google.com", "www.google.ca", "www.google.fr", "www.google.co.uk", "www.google.de", 
            "www.google.com.au", "www.google.co.jp", "www.google.co.in", "www.google.com.br", "www.google.ru",
            "www.google.it", "www.google.es", "www.google.com.mx", "www.google.co.kr", "www.google.nl"
        ]
        
        # Randomly select domains for each service
        books_domain = random.choice(google_domains).replace("www.", "books.")
        scholar_domain = random.choice([d for d in google_domains if d != books_domain.replace("books.", "www.")]).replace("www.", "scholar.")
        regular_domain = random.choice([d for d in google_domains if d != books_domain.replace("books.", "www.") and d != scholar_domain.replace("scholar.", "www.")])
        
        restricted_task = f"""
        As a scholarly research assistant, search for relevant academic literature about: {query}
        
        CRITICAL INSTRUCTION: You MUST click into each article to read the full text or detailed abstract. Do not just rely on the search results page summaries.
        
        FOLLOW THIS SEARCH STRATEGY IN ORDER:
        
        STEP 1: Begin with Google Books
        - IMPORTANT: Instead of using books.google.com, use {books_domain}
        - If you need to do another search, randomly select from these Google domains:
          books.google.com, books.google.ca, books.google.fr, books.google.co.uk, books.google.de, 
          books.google.com.au, books.google.co.jp, books.google.co.in, books.google.com.br
        - Use a different domain for each search to avoid rate limiting
        - Wait 5-7 seconds between searches on different domains
        - If it is not an exact match question, search for keywords directly from the query. If it is an exact match question, search for the exact wording of the query.
        - For exactMatch questions, ALWAYS remove any blanks (like "____" or "[BLANK]") from the question before searching
        - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
        - CRITICALLY IMPORTANT: If your search redirects to a regular Google page (URL starting with https://www.google.com/search?):
          * Look for a section labeled "在书中找到匹配结果" (matching results found in books)
          * IMPORTANT: When you see HTML with <div class="VNSPub">在书中找到匹配结果</div>, extract all HTML information that follows this div - this contains the matching results
          * The content will typically appear in a structure like: <div class="cmlJmd ETWPw"><div class="VNSPub">在书中找到匹配结果</div><span><span>... text content with <em>highlighted parts</em> ...</span></span></div>
          * Extract the text within the <span> elements that follow the VNSPub div
          * Pay special attention to text highlighted with <em> tags - these are the exact matches to your search query
          * For extraction purposes, preserve the <em> tags to identify matched content
          * For your final answer, you can remove the <em> tags while still emphasizing this matched content
          * Example: If you see "<em>In 1825</em> the Bourbon regime... In <em>Franche-Comté</em>...", extract this entire text and note that "In 1825" and "Franche-Comté" were highlighted matches
          * This section contains direct text snippets from books matching your search
          * READ THESE SNIPPETS CAREFULLY FIRST before anything else
          * For exactMatch questions, the answer is very likely to be found in these snippets
          * If you find a matching snippet, record the text and book source immediately
          * If a snippet perfectly matches the query for an exactMatch question, STOP SEARCHING and return that result
        - IMPORTANT: On the Google Books search results page, CAREFULLY CHECK THE SNIPPETS first:
          * Read each snippet on the search results page carefully
          * If a snippet perfectly matches the query (with blanks removed for exactMatch questions), YOU CAN STOP SEARCHING
          * Simply record that snippet along with its source book details and return it immediately
          * For exactMatch questions, once you've found an exact match in a snippet, no further searching is needed
        - Find books with preview access that discuss the topic
        - Use the search/preview feature to locate relevant sections
        - Important: If search results show SNIPPETS that match the query, CLICK on them to see the full context
        - When viewing a book, use the SEARCH BOX on the left side panel to search for keywords within that book
        - If Google Books redirects your search to regular Google (URL starting with https://www.google.com/search?q=), this is acceptable - continue with the search
        - When redirected to regular Google, make sure "Books" is selected in the options below the search bar to filter results to book content
        - If the Google Books search is rejected or doesn't work, try searching on regular Google.com instead, then click on the "Books" filter option below the search bar
        - You don't need the entire book to be accessible - focus on available preview sections or snippets
        - Record exact page numbers whenever possible
        - Wait 3-5 seconds between page operations (clicking links, opening books, performing searches)
        - After finishing with each book, CLOSE THE TAB before moving to the next source
        
        STEP 2: If insufficient results from Google Books, search on Google Scholar
        - IMPORTANT: Instead of using scholar.google.com, randomly select from these Google domains:
          scholar.google.com, scholar.google.ca, scholar.google.fr, scholar.google.co.uk, scholar.google.de, 
          scholar.google.com.au, scholar.google.co.jp, scholar.google.co.in, scholar.google.com.br, scholar.google.it,
          scholar.google.es, scholar.google.com.mx, scholar.google.co.kr, scholar.google.nl
        - Use a different domain than you used for Google Books
        - Wait 5-7 seconds between searches on different domains
        - Search for relevant articles using keywords from the query
        - For exactMatch questions, ALWAYS remember to search without any blanks
        - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815" 
        - This is CRITICAL because scholarly texts won't contain these blank placeholders
        - Always try to access the full text when possible
        - Extract precise quotes and information
        - Wait 3-5 seconds between page operations (opening articles, clicking links)
        - After finishing with each article, CLOSE THE TAB before moving to the next source
        
        STEP 3: If full text is inaccessible from both Google Books and Google Scholar:
        - Try a regular Google Search using {regular_domain}
        - If you need to do another search, randomly select from these Google domains:
          www.google.com, www.google.ca, www.google.fr, www.google.co.uk, www.google.de, 
          www.google.com.au, www.google.co.jp, www.google.co.in, www.google.com.br
        - Use a different domain than you used in previous searches
        - Wait 5-7 seconds between searches on different domains
        - Search for the same concepts plus terms like "quote", "excerpt", or "full text"
        - For exactMatch questions, ALWAYS continue to search without any blanks
        - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
        - This is ESSENTIAL because web texts won't contain these blank placeholders
        - Look for educational websites, repositories, or other scholarly sources
        - Check if there are alternative versions of the text on different platforms
        - Wait 3-5 seconds between page operations
        - After finishing with each source, CLOSE THE TAB before moving to the next source
        
        For each source you access, document:
        - Full citation details (authors, title, journal/book, year, DOI/ISBN)
        - Direct URL to the source
        - Citation count and publication year (when available)
        - EXACT QUOTES from the text that match or directly relate to the query
        - Full abstract/summary (copy the complete text)
        - Methodology used (if available)
        - Key findings and conclusions (with page numbers when possible)
        - Downloaded PDF filename (if available and successfully downloaded)
        
        PDF DOWNLOAD INSTRUCTIONS:
        - When you find freely accessible PDFs, download them to this specific directory: {actual_download_path}
        - Do NOT attempt to download files behind paywalls or requiring login
        - After downloading, note the filename and specific location of each downloaded PDF
        - Include the PDF filename in your report for each downloaded article
        
        IMPORTANT HANDLING INSTRUCTIONS:
        - If you encounter any login walls, CAPTCHA verification, paywalls, or other authentication requirements, EXIT that page immediately and follow the alternative search steps above
        - Do NOT attempt to bypass any security measures or authentication systems
        - Simply note "Authentication required" for that source and move on to other accessible sources or alternative search methods
        - Focus your time on resources that are freely accessible without login requirements
        
        Prioritize sources where:
        1. You can access the full text content
        2. The content contains EXACT matches to query phrases (highest priority)
        3. The content is highly cited from reputable sources
        4. The information is recently published (unless historical sources are needed)
        
        IMPORTANT: For 'exactMatch' type questions, you MUST find the exact original wording in the scholarly literature. The full answer will be contained verbatim in one or more sources - you need to access the content to find it.
        - If you find a snippet that exactly matches the query (with blanks removed) in the "在书中找到匹配结果" section, STOP SEARCHING and return that immediately
        - IMPORTANT: To find this section, look for HTML with <div class="VNSPub">在书中找到匹配结果</div> and extract all information that follows this div
        - The content will typically appear in a structure like: <div class="cmlJmd ETWPw"><div class="VNSPub">在书中找到匹配结果</div><span><span>... text content with <em>highlighted parts</em> ...</span></span></div>
        - Extract the text within the <span> elements that follow the VNSPub div
        - The <em> tags highlight the exact matches to your search query - these are the most important parts to focus on
        - When presenting your findings, you can remove the <em> tags but be sure to emphasize those matched terms
        - CRITICALLY IMPORTANT FOR FINAL ANSWER: When extracting information and presenting results, ADD BACK THE BLANKS and fill them in with the correct information
        - Example: If you searched for "The Battle of was fought in 1815" and found "The Battle of Waterloo was fought in 1815", your final answer should be "The Battle of _Waterloo_ was fought in 1815"
        - Always highlight or emphasize the filled-in information that was originally a blank in the question
        
        Format your response as a detailed research summary with bibliographic information and content details organized by source. Include sections on methodology, findings, and exact quotes that match the query.
        """
        
        try:
            # Run the task with BrowserAgentBehavior
            result = asyncio.run(self.browser._run_task(
                restricted_task, 
                max_steps=max_steps, 
                download_path=download_path
            ))
            return result or "No literature found."
        except Exception as e:
            error_msg = f"Error searching literature: {str(e)}"
            logging.error(error_msg)
            return error_msg


class GeneralBrowserTool(Tool):
    name = "general_browser_task"
    description = "Run a general web search and return the results."
    inputs = {
        "query": {"type": "string", "description": "The search query"},
        "max_steps": {"type": "integer", "description": "Maximum number of steps the agent can take", "default": 38, "nullable": True},
        "download_path": {"type": "string", "description": "Path to download PDF files", "default": "general_downloads", "nullable": True}
    }
    output_type = "string"

    def __init__(self, api_key=None, download_path=None):
        """
        Initialize the general browser tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
        """
        super().__init__()
        self.download_path = download_path or "general_downloads"
        self.system_prompt = f"""You are WebSearchBot, a sophisticated web research assistant capable of finding and analyzing information from online sources.

Your primary task is to search the web and retrieve accurate, relevant information in response to queries.

CRITICAL INSTRUCTIONS:
1. Always visit and read the actual webpage content, not just search result snippets
2. Extract information directly from webpage content for accuracy and detail
3. Navigate through multiple pages when necessary to find complete information
4. When you find useful PDF files, download them for further analysis
5. Always provide direct links to your sources
6. IMPORTANT: Wait 3-5 seconds between each page operation to avoid being blocked

BROWSER HANDLING GUIDELINES:
- Always pause 3-5 seconds between opening new tabs/pages
- Wait 3-5 seconds after closing a tab before opening a new one
- Pause a few seconds after performing a search before clicking on results
- Space out your interactions to appear more human-like and avoid triggering anti-bot measures

GOOGLE DOMAIN ROTATION:
- For EACH new Google search (books, scholar, or regular search), randomly select one of these Google domains:
  www.google.com, www.google.ca, www.google.fr, www.google.co.uk, www.google.de, 
  www.google.com.au, www.google.co.jp, www.google.co.in, www.google.com.br, www.google.ru,
  www.google.it, www.google.es, www.google.com.mx, www.google.co.kr, www.google.nl,
  www.google.pl, www.google.com.sg, www.google.co.za, www.google.com.tr, www.google.se
- For example, instead of going to https://books.google.com/, go to https://books.[random-domain]
- Instead of searching on https://scholar.google.com/, use https://scholar.[random-domain]
- This helps avoid rate limiting or IP blocks from any single Google domain
- Wait 5-7 seconds between searches on different Google domains

PDF DOWNLOAD INSTRUCTIONS:
- When you find useful PDF documents, download them using the browser's download functionality
- Save all PDF files to: {self.download_path}
- Do NOT attempt to download files behind paywalls or requiring login
- After downloading, note the filename and location of each downloaded PDF
- Include the PDF filename in your report

AUTHENTICATION HANDLING:
- If you encounter login walls, CAPTCHA verification, or paywalls, exit that page immediately
- Do NOT attempt to bypass security measures or enter credentials
- Simply note "Authentication required" for that source and try different sources
- Only spend time on resources that are freely accessible without authentication

RESEARCH METHODOLOGY:
- Start with broad search queries and refine based on initial results
- Compare information across multiple sources to verify accuracy
- Prioritize authoritative sources (educational institutions, government sites, reputable news outlets)
- Note when information is conflicting or uncertain
- Provide a balanced view when topics have multiple perspectives"""
        self.browser = LiteratureSearchBrowser(api_key=api_key, system_prompt=self.system_prompt, download_path=download_path)

    def forward(self, query: str, max_steps: int = 38, download_path: str = None) -> str:
        """
        Run a general web search and return the results.
        
        Args:
            query: The search query
            max_steps: Maximum number of steps the agent can take
            download_path: Path to download PDF files
            
        Returns:
            String containing the search results
        """
        logging.info(f"Running general web search for query: {query}")
        actual_download_path = download_path or self.download_path
        
        restricted_task = f"""
        As a web research assistant, search for information about: {query}
        
        FOLLOW THIS COMPREHENSIVE SEARCH STRATEGY:
        
        STEP 1: Start with a general Google Search
        - IMPORTANT: Instead of using www.google.com, randomly select from these Google domains:
          www.google.com, www.google.ca, www.google.fr, www.google.co.uk, www.google.de, 
          www.google.com.au, www.google.co.jp, www.google.co.in, www.google.com.br, www.google.ru,
          www.google.it, www.google.es, www.google.com.mx, www.google.co.kr, www.google.nl
        - Use a different domain for each search to avoid rate limiting
        - Wait 5-7 seconds between searches on different domains
        - If it is not an exact match question, search for information using precise keywords from the query. If it is an exact match question, search for the exact wording of the query.
        - For exactMatch questions, ALWAYS remove any blanks (like "____" or "[BLANK]") from the question before searching
        - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
        - This is CRITICAL because online texts won't contain these blank placeholders
        - Visit multiple relevant pages to gather detailed information
        - Extract the most relevant content from each page
        - Wait 3-5 seconds between page operations (opening pages, clicking links)
        - After finishing with each page, CLOSE THE TAB before moving to the next one
        
        STEP 2: For academic or historical topics, also check Google Scholar
        - IMPORTANT: Instead of using scholar.google.com, randomly select from these Google domains:
          scholar.google.com, scholar.google.ca, scholar.google.fr, scholar.google.co.uk, scholar.google.de, 
          scholar.google.com.au, scholar.google.co.jp, scholar.google.co.in, scholar.google.com.br, scholar.google.it,
          scholar.google.es, scholar.google.com.mx, scholar.google.co.kr, scholar.google.nl
        - Use a different domain than you used for your general Google search
        - Wait 5-7 seconds between searches on different domains
        - Search for scholarly articles related to the query
        - For exactMatch questions, ALWAYS remember to search without any blanks
        - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815" 
        - This is CRITICAL because scholarly texts won't contain these blank placeholders
        - Always try to access the full text when possible
        - Extract precise quotes and information
        - Wait 3-5 seconds between page operations (opening articles, clicking links)
        - After finishing with each article, CLOSE THE TAB before moving to the next source
        
        STEP 3: For book content, check Google Books
        - IMPORTANT: Instead of using books.google.com, randomly select from these Google domains:
          books.google.com, books.google.ca, books.google.fr, books.google.co.uk, books.google.de, 
          books.google.com.au, books.google.co.jp, books.google.co.in, books.google.com.br, books.google.ru,
          books.google.it, books.google.es, books.google.com.mx, books.google.co.kr, books.google.nl
        - Use a different domain than you used in previous searches
        - Wait 5-7 seconds between searches on different domains
        - Search for books related to the query
        - For exactMatch questions, ALWAYS remove any blanks from the question before searching
        - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815" 
        - This is CRITICAL because book texts won't contain these blank placeholders
        - CRITICALLY IMPORTANT: If your search redirects to a regular Google page (URL starting with https://www.google.com/search?):
          * Look for a section labeled "在书中找到匹配结果" (matching results found in books)
          * IMPORTANT: When you see HTML with <div class="VNSPub">在书中找到匹配结果</div>, extract all HTML information that follows this div - this contains the matching results
          * The content will typically appear in a structure like: <div class="cmlJmd ETWPw"><div class="VNSPub">在书中找到匹配结果</div><span><span>... text content with <em>highlighted parts</em> ...</span></span></div>
          * Extract the text within the <span> elements that follow the VNSPub div
          * Pay special attention to text highlighted with <em> tags - these are the exact matches to your search query
          * For extraction purposes, preserve the <em> tags to identify matched content
          * For your final answer, you can remove the <em> tags while still emphasizing this matched content
          * Example: If you see "<em>In 1825</em> the Bourbon regime... In <em>Franche-Comté</em>...", extract this entire text and note that "In 1825" and "Franche-Comté" were highlighted matches
          * This section contains direct text snippets from books matching your search
          * READ THESE SNIPPETS CAREFULLY FIRST before anything else
          * For exactMatch questions, the answer is very likely to be found in these snippets
          * If you find a matching snippet, record the text and book source immediately
          * If a snippet perfectly matches the query for an exactMatch question, STOP SEARCHING and return that result
        - IMPORTANT: On the Google Books search results page, CAREFULLY CHECK THE SNIPPETS first:
          * Read each snippet on the search results page carefully
          * If a snippet perfectly matches the query (with blanks removed for exactMatch questions), YOU CAN STOP SEARCHING
          * Simply record that snippet along with its source book details and return it immediately
          * For exactMatch questions, once you've found an exact match in a snippet, no further searching is needed
        - Use book previews to locate relevant sections
        - If Google Books redirects to regular Google, make sure to select "Books" filter below the search bar
        - Extract relevant information from accessible book previews
        - Record page numbers when available
        - Wait 3-5 seconds between page operations
        - After finishing with each book, CLOSE THE TAB before moving to the next one
        
        BROWSER MANAGEMENT:
        - Always wait 3-5 seconds between opening new tabs or pages
        - Pause 3-5 seconds between searches
        - Wait 3-5 seconds after closing a tab before opening a new one
        - Space out your interactions to appear more human-like
        
        COMPARISON ACROSS SOURCES:
        - Compare information across different sources to verify accuracy
        - Note any contradictions or different perspectives
        - Prioritize authoritative sources (educational institutions, government sites, reputable publications)
        
        PDF DOWNLOAD INSTRUCTIONS:
        - When you find useful PDF documents, download them to this specific directory: {actual_download_path}
        - After downloading, note the filename and specific location of each downloaded PDF
        - Include the PDF filename in your report
        
        IMPORTANT HANDLING:
        - If you encounter login walls, paywalls, or CAPTCHA verification, exit that page and try another source
        - Do not attempt to bypass security measures
        - Focus on freely accessible sources
        
        CRITICAL FOR EXACTMATCH QUESTIONS:
        - For exactMatch type questions, finding the precise original wording is ESSENTIAL
        - The complete and precise answer exists verbatim in the literature and must be found
        - ALWAYS remove any blanks (like "____", "___", or "[BLANK]") from the question before searching
        - This is CRITICAL because texts won't contain these blank placeholders
        - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
        - The answer you're looking for will fill in the blank with the correct term
        - Always check the "在书中找到匹配结果" (matching results found in books) section first for exactMatch questions
        - IMPORTANT: Look for HTML with <div class="VNSPub">在书中找到匹配结果</div> and extract all information that follows this div
        - If you find a snippet that exactly matches the query (with blanks removed), STOP SEARCHING and return that immediately
        - CRITICALLY IMPORTANT FOR FINAL ANSWER: When extracting information and presenting results, ADD BACK THE BLANKS and fill them in with the correct information
        - Example: If you searched for "The Battle of was fought in 1815" and found "The Battle of Waterloo was fought in 1815", your final answer should be "The Battle of _Waterloo_ was fought in 1815"
        - Always highlight or emphasize the filled-in information that was originally a blank in the question
        
        For each important source, document:
        - The full title and URL
        - Author or publisher information
        - Publication date (if available)
        - Key information found
        - Exact quotes that directly address the query
        - Any downloaded files with their locations
        
        Format your response as a comprehensive summary of the information found, comparing across sources and highlighting the most relevant findings. Include all sources and downloaded files.
        """
        
        try:
            # Run the task with BrowserAgentBehavior
            result = asyncio.run(self.browser._run_task(
                restricted_task, 
                max_steps=max_steps, 
                download_path=download_path
            ))
            return result
        except Exception as e:
            error_msg = f"Error in general browser task: {str(e)}"
            logging.error(error_msg)
            return error_msg


class RelevantLiteratureFinderTool(Tool):
    name = "relevant_literature_finder"
    description = "Search for literature and return the most relevant sources for the query."
    inputs = {
        "query": {"type": "string", "description": "The research query or topic to search for"},
        "max_results": {"type": "integer", "description": "Maximum number of sources to return (default 3)", "default": 3, "nullable": True},
        "max_steps": {"type": "integer", "description": "Maximum number of steps the agent can take", "default": 38, "nullable": True},
        "download_path": {"type": "string", "description": "Path to download PDF files", "default": "relevant_literature_downloads", "nullable": True}
    }
    output_type = "string"
    
    def __init__(self, api_key=None, download_path=None):
        """
        Initialize the relevant literature finder tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
        """
        super().__init__()
        self.download_path = download_path or "relevant_literature_downloads"
        self.system_prompt = LiteratureSearchBrowser(api_key=api_key, download_path=download_path).system_prompt
        self.api_key = api_key

    async def _search_and_filter_literature(self, query: str, max_results: int = 3, max_steps: int = 38, download_path: str = None) -> str:
        """
        Search for literature and then filter for the most relevant ones.
        
        Args:
            query: The search query for literature
            max_results: Maximum number of relevant results to return
            max_steps: Maximum number of steps the agent can take
            download_path: Path to download PDF files
            
        Returns:
            String containing the most relevant literature findings
        """
        logging.info(f"Searching and filtering literature for: {query}")
        actual_download_path = download_path or self.download_path
        
        try:
            # Create a search task
            search_task = f"""Search for high-impact, recent scholarly articles and relevant content about: {query}. 
            
            CRITICAL INSTRUCTION: You MUST click into each article to read the full text or abstract page. Do not just rely on the search results page.
            
            SPECIAL INSTRUCTIONS FOR EXACTMATCH QUESTIONS:
            - If this is an exactMatch question, you only need to find ONE most relevant literature source
            - If the source appears as a small image segment, carefully extract the text from the image and return it
            - After filling in the blanks, verify that the complete text can be found in Google Books
            - NEVER answer with "Unable to determine" for exactMatch questions - continue searching until you find a match
            
            FOLLOW THIS SEARCH STRATEGY IN ORDER:
            
            STEP 1: Begin with Google Books
            - IMPORTANT: Instead of using books.google.com, randomly select from these Google domains:
              books.google.com, books.google.ca, books.google.fr, books.google.co.uk, books.google.de, 
              books.google.com.au, books.google.co.jp, books.google.co.in, books.google.com.br, books.google.it,
              books.google.es, books.google.com.mx, books.google.co.kr, books.google.nl
            - Use a different domain for each search to avoid rate limiting
            - Wait 5-7 seconds between searches on different domains
            - If it is not an exact match question, search for keywords directly from the query. If it is an exact match question, search for the exact wording of the query.
            - For exactMatch questions, ALWAYS remove any blanks (like "____" or "[BLANK]") from the question before searching
            - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
            - This is CRITICAL because scholarly texts won't contain these blank placeholders
            - CRITICALLY IMPORTANT: If your search redirects to a regular Google page (URL starting with https://www.google.com/search?):
              * Look for a section labeled "在书中找到匹配结果" (matching results found in books)
              * IMPORTANT: When you see HTML with <div class="VNSPub">在书中找到匹配结果</div>, extract all HTML information that follows this div - this contains the matching results
              * The content will typically appear in a structure like: <div class="cmlJmd ETWPw"><div class="VNSPub">在书中找到匹配结果</div><span><span>... text content with <em>highlighted parts</em> ...</span></span></div>
              * Extract the text within the <span> elements that follow the VNSPub div
              * Pay special attention to text highlighted with <em> tags - these are the exact matches to your search query
              * For extraction purposes, preserve the <em> tags to identify matched content
              * For your final answer, you can remove the <em> tags while still emphasizing this matched content
              * Example: If you see "<em>In 1825</em> the Bourbon regime... In <em>Franche-Comté</em>...", extract this entire text and note that "In 1825" and "Franche-Comté" were highlighted matches
              * This section contains direct text snippets from books matching your search
              * READ THESE SNIPPETS CAREFULLY FIRST before anything else
              * For exactMatch questions, the answer is very likely to be found in these snippets
              * If you find a matching snippet, record the text and book source immediately
              * If a snippet perfectly matches the query for an exactMatch question, STOP SEARCHING and return that result
            - IMPORTANT: On the search results page, CAREFULLY CHECK THE SNIPPETS first before clicking into books:
              * Read each snippet on the search results page carefully
              * If a snippet perfectly matches the query (with blanks removed for exactMatch questions), YOU CAN STOP SEARCHING
              * Simply record that snippet along with its source book details and return it immediately
              * For exactMatch questions, once you've found an exact match in a snippet, no further searching is needed
            - Find books with preview access related to the topic
            - Use the "Search inside" or preview feature to locate relevant sections
            - Important: If search results show SNIPPETS that match the query, CLICK on them to see the full context
            - When viewing a book, use the SEARCH BOX on the left side panel to search for keywords within that book
            - If Google Books redirects your search to regular Google (URL starting with https://www.google.com/search?q=), this is acceptable - continue with the search
            - When redirected to regular Google, make sure "Books" is selected in the options below the search bar to filter results to book content
            - If the Google Books search is rejected or doesn't work, try searching on regular Google.com instead, then click on the "Books" filter option below the search bar
            - You don't need the entire book to be accessible - focus on available preview sections or snippets
            - Look for exact quotes or information that matches the query
            - Record exact page numbers whenever possible
            - Wait 3-5 seconds between page operations (clicking links, opening books, performing searches)
            - After finishing with each book, CLOSE THE TAB before moving to the next source
            
            STEP 2: If insufficient results from Google Books, search on Google Scholar
            - IMPORTANT: Instead of using scholar.google.com, randomly select from these Google domains:
              scholar.google.com, scholar.google.ca, scholar.google.fr, scholar.google.co.uk, scholar.google.de, 
              scholar.google.com.au, scholar.google.co.jp, scholar.google.co.in, scholar.google.com.br, scholar.google.it,
              scholar.google.es, scholar.google.com.mx, scholar.google.co.kr, scholar.google.nl
            - Use a different domain than you used for Google Books
            - Wait 5-7 seconds between searches on different domains
            - Search for relevant articles using keywords from the query
            - For exactMatch questions, ALWAYS remember to search without any blanks
            - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815" 
            - This is CRITICAL because scholarly texts won't contain these blank placeholders
            - Always try to access the full text when possible
            - Extract precise quotes and information
            - Wait 3-5 seconds between page operations (opening articles, clicking links)
            - After finishing with each article, CLOSE THE TAB before moving to the next source
            
            STEP 3: If full text is inaccessible from both Google Books and Google Scholar:
            - Try a regular Google Search using a random Google domain from this list:
              www.google.com, www.google.ca, www.google.fr, www.google.co.uk, www.google.de, 
              www.google.com.au, www.google.co.jp, www.google.co.in, www.google.com.br, www.google.it,
              www.google.es, www.google.com.mx, www.google.co.kr, www.google.nl
            - Use a different domain than you used in previous searches
            - Wait 5-7 seconds between searches on different domains
            - Search for the same concepts plus terms like "quote", "excerpt", or "full text"
            - For exactMatch questions, ALWAYS continue to search without any blanks
            - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
            - This is ESSENTIAL because web texts won't contain these blank placeholders
            - Look for educational websites, repositories, or other scholarly sources
            - Check if there are alternative versions of the text on different websites
            - Wait 3-5 seconds between page operations
            - After finishing with each source, CLOSE THE TAB before moving to the next source
            
            PDF DOWNLOAD INSTRUCTIONS:
            - When you find freely accessible PDFs, download them to this specific directory: {actual_download_path}
            - Do NOT attempt to download files behind paywalls or requiring login
            - After downloading, note the filename and specific location of each downloaded PDF
            - Include the PDF filename in your report for each downloaded article
            
            IMPORTANT HANDLING INSTRUCTIONS:
            - If you encounter any login walls, CAPTCHA verification, paywalls, or other authentication requirements, EXIT that page immediately and follow the alternative search steps above
            - Do NOT attempt to bypass any security measures or authentication systems
            - Simply note "Authentication required" for that source and move on to other accessible sources or alternative search methods
            - Focus your time on resources that are freely accessible without login requirements
            
            Prioritize:
            1. Sources containing EXACT text matches to the query (highest priority)
            2. Sources where you can access full text, not just abstracts
            3. Most relevant content to the query topic
            4. Highly cited papers or books from reputable sources
            
            CRITICAL FOR EXACTMATCH QUESTIONS:
            - For exactMatch type questions, finding the precise original wording is ESSENTIAL
            - The complete and precise answer exists verbatim in the literature and must be found
            - ALWAYS remove any blanks (like "____", "___", or "[BLANK]") from the question before searching
            - This is CRITICAL because texts in scholarly sources won't contain these blank placeholders
            - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
            - When you find a match, try to fill in the blank and then verify the complete sentence with the blank filled in Google Books
            - Only return ONE most relevant source for exactMatch questions
            - If the source is a small image, carefully extract the text from the image and include it in your response
            - If you find an exact match in a Google Books snippet or in the "在书中找到匹配结果" section, stop searching immediately and return that snippet with source details
            - IMPORTANT: To find the "在书中找到匹配结果" section, look for HTML with <div class="VNSPub">在书中找到匹配结果</div> and extract all information that follows this div
            - The content will typically appear in a structure like: <div class="cmlJmd ETWPw"><div class="VNSPub">在书中找到匹配结果</div><span><span>... text content with <em>highlighted parts</em> ...</span></span></div>
            - Extract the text within the <span> elements that follow the VNSPub div
            - The <em> tags highlight the exact matches to your search query - these are the most important parts to focus on
            - When presenting your findings, you can remove the <em> tags but be sure to emphasize those matched terms
            - CRITICALLY IMPORTANT FOR FINAL ANSWER: When extracting information and presenting results, ADD BACK THE BLANKS and fill them in with the correct information
            - Example: If you searched for "The Battle of was fought in 1815" and found "The Battle of Waterloo was fought in 1815", your final answer should be "The Battle of _Waterloo_ was fought in 1815"
            - Always highlight or emphasize the filled-in information that was originally a blank in the question
            - NEVER answer with "Unable to determine" - continue searching until you find a valid match
            
            For each important source, document:
            - The full title and URL
            - Author or publisher information
            - Publication date (if available)
            - Key information found
            - Exact quotes that directly address the query
            - Any downloaded files with their locations
            
            Format your response as a comprehensive summary of the information found, comparing across sources and highlighting the most relevant findings. Include all sources and downloaded files.
            """
            
            # Run the search task using the browser
            config = BrowserConfig(headless=True)
            browser = Browser(config)
            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.api_key)
            agent = Agent(
                task=search_task, 
                llm=llm, 
                # browser=browser, 
                generate_gif=False,
                extend_system_message=self.system_prompt
            )
            result = await agent.run(max_steps=max_steps)
            await browser.close()  # 关闭浏览器
            
            # Use GPT to filter and rank the most relevant results
            filter_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.api_key)
            
            prompt = f"""
            As an expert research librarian specializing in history, review the following search results and identify the {max_results} most relevant sources for answering this query: "{query}"
            
            Search Results:
            {result.final_result}
            
            CRITICAL FOR EXACT MATCH QUESTIONS:
            If this is an exactMatch type question, you MUST find and preserve the EXACT original wording from academic sources. Focus on sources where the exact text matching the query was found.
            
            For each selected source, provide:
            1. Full citation with authors, title, journal, year, DOI
            2. Direct URL to the article
            3. Relevance score (1-10)
            4. EXACT QUOTES from the article that match or relate to the query (preserve the exact wording)
            5. Page number or location where the quote appears (if available)
            6. Full abstract copied directly from the article
            7. PDF filename if downloaded
            8. Explanation of why this source directly answers the query
            
            CRITICALLY IMPORTANT: 
            - You MUST preserve and include ANY text found that matches parts of the query word-for-word
            - Even partial matches to the query text are extremely valuable
            - The exact wording is essential for questions requiring precise answers
            - Include page numbers whenever possible so the exact text can be cited properly
            - If any PDFs were downloaded during the search, highlight these as they contain the full text
            
            Return your analysis in the following JSON format:
            {
              "selected_sources": [
                {
                  "citation": "Full citation of the source",
                  "url": "Direct URL to the article",
                  "relevance_score": number between 1-10,
                  "abstract": "Full abstract from the article",
                  "exact_quotes": [
                    {
                      "text": "Exact quote from the article that matches or relates to the query",
                      "page_number": "Page number or location (if available)",
                      "matches_query": true/false (whether this exactly matches part of the query)
                    }
                  ],
                  "relevance_explanation": "Detailed explanation of why this source directly answers the query"
                },
                ...
              ],
              "search_summary": "Brief summary of the search results and why these particular sources were selected",
              "exact_match_found": true/false (whether any exact matches to the query were found)
            }
            
            Only return the JSON structure, no additional text.
            """
            
            response = filter_llm.invoke(prompt)
            json_strings = response.content
            
            # Try to extract JSON if wrapped in markdown code block
            json_pattern = r'```(?:json)?\n(.*?)\n```'
            match = re.search(json_pattern, json_strings, re.DOTALL)
            if match:
                parsed_results = match.group(1)
            else:
                # Otherwise use the entire response
                parsed_results = json_strings
                
            try:
                result_obj = json.loads(parsed_results)
                selected_sources = result_obj.get("selected_sources", [])
                exact_match_found = result_obj.get("exact_match_found", False)
                
                # Format the results in a more readable way
                filtered_results = f"## Most Relevant Literature for: {query}\n\n"
                
                if exact_match_found:
                    filtered_results += f"### EXACT MATCH FOUND! The query text was located in the literature.\n\n"
                
                for i, source in enumerate(selected_sources, 1):
                    filtered_results += f"### Source {i}: Relevance Score {source.get('relevance_score')}/10\n"
                    filtered_results += f"**Citation**: {source.get('citation')}\n"
                    filtered_results += f"**URL**: {source.get('url', 'Not provided')}\n\n"
                    
                    # Add abstract
                    if source.get('abstract'):
                        filtered_results += f"**Abstract**: {source.get('abstract')}\n\n"
                    
                    # Add exact quotes section with formatting to highlight exact matches
                    filtered_results += "**Key Quotes**:\n"
                    for quote in source.get('exact_quotes', []):
                        quote_text = quote.get('text', '')
                        page_info = f" (Page: {quote.get('page_number')})" if quote.get('page_number') else ""
                        
                        if quote.get('matches_query'):
                            filtered_results += f"- **EXACT MATCH!** \"{quote_text}\"{page_info}\n"
                        else:
                            filtered_results += f"- \"{quote_text}\"{page_info}\n"
                    
                    filtered_results += f"\n**Why Relevant**: {source.get('relevance_explanation')}\n\n"
                    filtered_results += "----------------------\n\n"
                
                filtered_results += f"## Search Summary\n{result_obj.get('search_summary', 'No summary provided.')}"
                
                return filtered_results
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw search results
                logging.warning("Could not parse JSON from LLM response, returning raw search results")
                return f"Could not process search results in the expected format. Raw search results:\n\n{result.final_result}"
                
        except Exception as e:
            error_msg = f"Error filtering relevant literature: {str(e)}"
            logging.error(error_msg)
            return f"Literature search error: {str(e)}"

    def forward(self, query: str, max_results: int = 3, max_steps: int = 38, download_path: str = None) -> str:
        """
        Search for literature and return the most relevant sources for the query.
        
        Args:
            query: The research query or topic to search for
            max_results: Maximum number of sources to return (default 3)
            max_steps: Maximum number of steps the agent can take
            download_path: Path to download PDF files
            
        Returns:
            String containing the most relevant literature with explanations
        """
        return asyncio.run(self._search_and_filter_literature(query, max_results, max_steps, download_path))

class BookMatchExtractorTool(Tool):
    name = "book_match_extractor"
    description = "Extract book match snippets from Google Books search results for a query."
    inputs = {
        "query": {"type": "string", "description": "The query to search for in Google Books"},
        "max_steps": {"type": "integer", "description": "Maximum number of steps the agent can take", "default": 38, "nullable": True}
    }
    output_type = "string"

    def __init__(self, api_key=None, download_path=None):
        """
        Initialize the book match extractor tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
        """
        super().__init__()
        self.download_path = download_path or "book_match_downloads"
        # Use the system prompt from LiteratureSearchBrowser
        self.browser = LiteratureSearchBrowser(api_key=api_key, download_path=download_path)
        self.api_key = api_key

    async def _extract_book_matches_async(self, query: str, max_steps: int = 38) -> str:
        """
        Extract book match snippets from Google Books search results for a query.
        
        Args:
            query: The query to search for in Google Books
            max_steps: Maximum number of steps the agent can take
            
        Returns:
            Formatted string containing the extracted book match snippets
        """
        results = await self.browser.extract_book_matches(query, max_steps)
        
        search_url = results.get("search_url", None)
        book_matches = results.get("book_matches", None)
        print("results", results)
        
        if not book_matches:
            # If the browser method didn't get matches, try direct parsing
            if search_url:
                book_matches = await self.browser.parse_google_books_url(search_url)
            
        if not book_matches:
            return f"No book match snippets found for query: {query}\nSearch URL: {search_url}"
        
        # Format the results
        results = f"## Book Match Snippets for: {query}\n\n"
        results += f"Search URL: {search_url}\n\n"
        
        for i, match in enumerate(book_matches, 1):
            results += f"### Match {i}:\n\n"
            
            if 'book_title' in match and match['book_title']:
                results += f"**Book**: {match['book_title']}\n"
                
            if 'book_link' in match and match['book_link']:
                results += f"**Link**: {match['book_link']}\n\n"
            
            if 'snippet_html' in match and match['snippet_html']:
                # For HTML snippet, preserve the formatting but clean up for readability
                html_snippet = match['snippet_html']
                # Replace <em> tags with markdown bold for highlighting
                html_snippet = html_snippet.replace('<em>', '**').replace('</em>', '**')
                # Remove other HTML tags
                html_snippet = re.sub(r'<[^>]*>', '', html_snippet)
                results += f"**Snippet (with highlights)**:\n{html_snippet}\n\n"
            
            if 'snippet_text' in match and match['snippet_text']:
                results += f"**Plain Text Snippet**:\n{match['snippet_text']}\n\n"
            
            if 'highlights' in match and match['highlights']:
                results += f"**Highlighted Terms**: {', '.join(match['highlights'])}\n\n"
            
            results += "---\n\n"
        
        return results

    def forward(self, query: str, max_steps: int = 38) -> str:
        """
        Extract book match snippets from Google Books search results for a query.
        
        Args:
            query: The query to search for in Google Books
            max_steps: Maximum number of steps the agent can take
            
        Returns:
            Formatted string containing the extracted book match snippets
        """
        logging.info(f"Extracting book matches for query: {query}")
        return asyncio.run(self._extract_book_matches_async(query, max_steps))

class DirectGoogleBooksCrawlerTool(Tool):
    name = "direct_google_books_crawler"
    description = "Directly extract book match snippets from a Google Books search URL."
    inputs = {
        "url": {"type": "string", "description": "The Google Books search URL to extract snippets from"}
    }
    output_type = "string"

    def __init__(self, api_key=None, download_path=None):
        """
        Initialize the direct Google Books crawler tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
        """
        super().__init__()
        self.download_path = download_path or "book_match_downloads"
        # Use the system prompt from LiteratureSearchBrowser
        self.browser = LiteratureSearchBrowser(api_key=api_key, download_path=download_path)
        self.api_key = api_key

    async def _parse_google_books_url_async(self, url: str) -> str:
        """
        Parse a Google Books search URL to extract book match snippets.
        
        Args:
            url: The Google Books search URL
            
        Returns:
            Formatted string containing the extracted book match snippets
        """
        book_matches = await self.browser.parse_google_books_url(url)
        
        if not book_matches:
            return f"No book match snippets found at URL: {url}"
        
        # Format the results
        results = f"## Book Match Snippets from URL\n\n"
        results += f"Source URL: {url}\n\n"
        
        for i, match in enumerate(book_matches, 1):
            results += f"### Match {i}:\n\n"
            
            if 'book_title' in match and match['book_title']:
                results += f"**Book**: {match['book_title']}\n"
                
            if 'book_link' in match and match['book_link']:
                results += f"**Link**: {match['book_link']}\n\n"
            
            if 'snippet_html' in match and match['snippet_html']:
                # For HTML snippet, preserve the formatting but clean up for readability
                html_snippet = match['snippet_html']
                # Replace <em> tags with markdown bold for highlighting
                html_snippet = html_snippet.replace('<em>', '**').replace('</em>', '**')
                # Remove other HTML tags
                html_snippet = re.sub(r'<[^>]*>', '', html_snippet)
                results += f"**Snippet (with highlights)**:\n{html_snippet}\n\n"
            
            if 'snippet_text' in match and match['snippet_text']:
                results += f"**Plain Text Snippet**:\n{match['snippet_text']}\n\n"
            
            if 'highlights' in match and match['highlights']:
                results += f"**Highlighted Terms**: {', '.join(match['highlights'])}\n\n"
            
            results += "---\n\n"
        
        return results

    def forward(self, url: str) -> str:
        """
        Parse a Google Books search URL to extract book match snippets.
        
        Args:
            url: The Google Books search URL
            
        Returns:
            Formatted string containing the extracted book match snippets
        """
        logging.info(f"Parsing Google Books URL: {url}")
        return asyncio.run(self._parse_google_books_url_async(url))

# Update the create_literature_tools function to include the new tools
def create_literature_tools(api_key=None, download_path=None):
    """
    Create a set of literature tools that use the provided API key.
    
    Args:
        api_key: The OpenAI API key to use for all tools
        download_path: Path to download PDF files
        
    Returns:
        Dictionary containing tool instances
    """
    return {
        "literature_searching_task": LiteratureSearchingTool(api_key=api_key, download_path=download_path),
        "general_browser_task": GeneralBrowserTool(api_key=api_key, download_path=download_path),
        "relevant_literature_finder": RelevantLiteratureFinderTool(api_key=api_key, download_path=download_path),
        "book_match_extractor": BookMatchExtractorTool(api_key=api_key, download_path=download_path),
        "direct_google_books_crawler": DirectGoogleBooksCrawlerTool(api_key=api_key, download_path=download_path)
    } 

import os
import uuid
import re
import requests
import yfinance as yf
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import io
import base64
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# ==================== SETUP =======================
GENAI_API_KEY = ""  # Replace with your Gemini API key
genai.configure(api_key=GENAI_API_KEY)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GENAI_API_KEY,
    temperature=0.7
)

finance_prompt = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""
You are a helpful and intelligent AI finance assistant. Based on the chat history and current user question, provide an insightful and accurate financial response.

Chat history:
{chat_history}

User question: {question}

AI Response:"""
)

finance_chain = LLMChain(
    llm=llm,
    prompt=finance_prompt,
    memory=memory,
    verbose=False
)

# ==================== UTILITIES =======================

def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        if hist.empty:
            return f"❌ No data found for {symbol}"

        price = hist["Close"].iloc[-1]

        # Decide currency symbol based on symbol
        if symbol.upper().endswith(".NS") or symbol.upper() in ["^NSEI", "^BSESN"]:  # NSE India
            currency_symbol = "₹"
        elif symbol.upper().endswith(".BO"):  # BSE India
            currency_symbol = "₹"
        else:
            currency_symbol = "$"

        return f"📈 The current price of {symbol.upper()} is {currency_symbol}{price:.2f}"
    except Exception as e:
        return f"❌ Error fetching stock price: {e}"
    

def get_index_data():
    data = []
    try:
        def get_price(ticker, label, currency="₹"):
            history = yf.Ticker(ticker).history(period="1d")
            if history.empty:
                return f"❌ {label} data not available"
            price = history["Close"].iloc[-1]
            return f"{label}: {currency}{price:.2f}"

        data.extend([
            f"📊 {get_price('^NSEI', 'Nifty 50')}",
            f"📈 {get_price('^BSESN', 'Sensex')}",
            f"₿ {get_price('BTC-USD', 'Bitcoin', '$')}",
            f"Ξ {get_price('ETH-USD', 'Ethereum', '$')}",
            f"🪙 {get_price('GC=F', 'Gold (1oz)', '$')}",
        ])
    except Exception as e:
        data.append(f"❌ Index fetch error: {e}")
    return "\n".join(data)

def estimate_tax(income):
    try:
        income = float(income)
        tax = 0
        if income <= 250000:
            tax = 0
        elif income <= 500000:
            tax = 0.05 * (income - 250000)
        elif income <= 1000000:
            tax = 12500 + 0.2 * (income - 500000)
        else:
            tax = 112500 + 0.3 * (income - 1000000)
        return f"💸 Estimated Tax: ₹{tax:.2f} for income ₹{income:.2f}"
    except:
        return "❌ Please enter a valid number like: `tax 700000`"

def plot_stock_chart(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="5d")
        if data.empty:
            return "❌ No data to plot."

        if symbol.upper().endswith(".NS") or symbol.upper() in ["^NSEI", "^BSESN"]:
            currency_symbol = "₹"
        elif symbol.upper().endswith(".BO"):
            currency_symbol = "₹"
        else:
            currency_symbol = "$"

        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['Close'], marker='o')
        plt.title(f"{symbol.upper()} - Last 5 Days")
        plt.xlabel("Date")
        plt.ylabel(f"Price ({currency_symbol})")
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        unique_id = str(uuid.uuid4()).replace('-', '')

        html_code = f'''
        <div class="chart-wrapper" style="position: relative;">
            <img class="stock-img" id="stockImg_{unique_id}" src="data:image/png;base64,{img_base64}" alt="{symbol} chart" style="max-width:100%; border-radius:10px; cursor:pointer;" />
            <div id="imgModal_{unique_id}" class="modal">
              <span class="close" id="closeModal_{unique_id}">&times;</span>
              <img class="modal-content" id="imgInModal_{unique_id}">
            </div>
        </div>
        '''
        return html_code

    except Exception as e:
        return f"❌ Error plotting chart: {e}"

def summarize_article_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        page = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")

        # Try known article containers first (Moneycontrol)
        content = ""
        
        # ✅ Moneycontrol
        mc_body = soup.select("div.article_content.clearfix")  # Common class for main content
        if not mc_body:
            mc_body = soup.select("div.content_wrapper")  # fallback for older articles
        if mc_body:
            content = " ".join(p.get_text() for p in mc_body[0].find_all("p"))

        # ✅ Reuters
        if not content:
            rt_body = soup.select("div.article-body__content__17Yit") or soup.select("div[data-testid='BodyWrapper']")
            if rt_body:
                content = " ".join(p.get_text() for p in rt_body[0].find_all("p"))

        # Fallback to generic <p> tags (last resort)
        if not content:
            paragraphs = soup.find_all("p")
            content = " ".join(p.get_text() for p in paragraphs[:15])

        # Check content length
        if len(content.strip()) < 100:
            return "❌ Article content is too short or not found."

        # Send to Gemini for summary
        prompt = f"Summarize this financial article in 5 bullet points:\n\n{content}"
        response = llm.invoke(prompt)

        # Handle Gemini response safely
        if hasattr(response, "content"):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            return str(response)

    except Exception as e:
        return f"❌ Couldn't summarize: {e}"


def get_finance_news():
    news = []

    def is_valid_headline(text):
        return bool(text and len(text) > 15 and re.search(r"[a-zA-Z]", text))

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        mc_url = "https://www.moneycontrol.com/news/business/"
        response = requests.get(mc_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        mc_headlines = soup.select("h2 > a")[:5]
        news.extend([
            f"📰 Moneycontrol: [{h.text.strip()}]({h['href']})"
            for h in mc_headlines if is_valid_headline(h.text.strip())
    ])
    except Exception as e:
        news.append(f"❌ Moneycontrol error: {str(e)}")

    try:
        # Economic Times
        et_url = "https://economictimes.indiatimes.com/news/economy"
        res = requests.get(et_url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        et_headlines = soup.select("h3 a")[:5]
        news.extend([
            f"📊 Economic Times: [{a.text.strip()}](https://economictimes.indiatimes.com{a['href']})"
            for a in et_headlines if is_valid_headline(a.text.strip())
        ])
    except Exception as e:
        news.append(f"❌ Economic Times error: {str(e)}")

    try:
        reuters_url = "https://www.reuters.com/business/"
        response = requests.get(reuters_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        rt_headlines = soup.select("h3 > a")[:5]
        news.extend([
            f"🗞️ Reuters: [{a.text.strip()}](https://www.reuters.com{a['href']})"
            for a in rt_headlines if is_valid_headline(a.text.strip())
        ])
    except Exception as e:
        news.append(f"❌ Reuters error: {str(e)}")

    return "\n".join(news) if news else "❌ No news found right now."

def get_finance_advice(question):
    try:
        response = finance_chain.invoke({"question": question})
        return response["text"] if isinstance(response, dict) else response
    except Exception as e:
        return f"❌ Error generating advice: {e}"

# ==================== MAIN LOOP =======================

def process_user_input(user_input):
    user_input = user_input.strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        return "Goodbye! Have a great day! 🚀"

    elif user_input.lower().startswith("stock price"):
        symbol = user_input.split()[-1].upper()
        return get_stock_price(symbol)

    elif user_input.lower().startswith("stock chart"):
        symbol = user_input.split()[-1].upper()
        return plot_stock_chart(symbol)

    elif user_input.lower().startswith("tax"):
        amount = user_input.split()[-1]
        return estimate_tax(amount)

    elif any(word in user_input.lower() for word in ["index", "nifty", "sensex", "bitcoin", "crypto", "gold"]):
        return get_index_data()

    elif "summarize" in user_input.lower() and "http" in user_input:
        import re
        url = re.search(r"https?://\S+", user_input)
        if url:
            return summarize_article_text(url.group())
        else:
            return "❌ Please provide a valid URL to summarize."

    elif "news" in user_input.lower():
        return get_finance_news()

    else:
        return get_finance_advice(user_input)
    

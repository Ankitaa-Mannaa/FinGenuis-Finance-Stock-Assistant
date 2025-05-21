# FinGenuis-Finance-Stock-Assistant

An intelligent full-stack finance chatbot built with **Flask**, **LangChain**, and **Gemini API** to help users with investment advice, budgeting, real-time stock/crypto/commodity updates, tax calculations, personalized financial planning, and a LSTM based stock price prediction model


## 🚀 Features

- 💬 **Conversational Interface**: Chatbot with contextual memory for smooth conversations using `ConversationBufferMemory`.
- 📊 **Real-Time Financial Data**:
  - Live Stock Prices (NSE, BSE)
  - Crypto Prices (Bitcoin, Ethereum, etc.)
  - Commodity Data (Gold, Silver, Crude Oil)
  - Index Data (Sensex, Nifty)
- 📰 **Financial News Aggregation**:
  - Multi-source scraping
  - Clickable headlines
  - Gemini-generated summaries
- 📈 **Stock Price Forecasting**:
  - Predicts future prices using ML models (LSTM, etc.)
  - Visualizes trends with matplotlib/plotly charts
- 🧮 **Tax Estimator**:
  - Personal income tax calculation (India-specific)
- 🧠 **LangChain + Gemini API**:
  - Uses Gemini for responses
  - Memory support via LangChain for coherent chats
- 📊 **Power BI Integration**:
  - Embedded Power BI dashboard showing market overview
- 🔐 **Secure Flask Backend**:
  - Text-based interaction
  - Scalable structure

## 🏗️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript (Basic, no web UI focus)
- **Backend**: Python (Flask), LangChain, Gemini API
- **Database**: MySQL (for optional user storage)
- **APIs/Scraping**: yFinance, BeautifulSoup
- **Data Visualization**: matplotlib, plotly, Power BI
- **Machine Learning**: Scikit-learn, TensorFlow/Keras

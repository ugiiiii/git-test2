import yfinance as yf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import schedule
import time
import threading
import logging
import asyncio
from telegram import Bot
from telegram.ext import Application
from datetime import datetime

# Use Agg backend for Matplotlib to avoid GUI issues
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='rsi_heatmap.log', filemode='a')

class RSIHeatmapScheduler:
    def __init__(self, config_path='config.json', stocks_path='stocks.json'):
        self.load_config(config_path)
        self.load_stocks(stocks_path)
        self.application = Application.builder().token(self.telegram_bot_token).build()

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.telegram_bot_token = config['telegram_bot_token']
            self.chat_id = config['chat_id']

    def load_stocks(self, stocks_path):
        with open(stocks_path, 'r') as f:
            self.stocks = json.load(f)

    def fetch_data(self, stock, interval="30m"):
        logging.info(f"Fetching data for {stock}")
        return yf.download(stock, period="1mo", interval=interval)

    def calculate_rsi(self, data, window=14):
        logging.info("Calculating RSI")
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_heatmap(self):
        logging.info("Generating heatmap")
        rsi_data = {}

        for stock in self.stocks:
            try:
                data = self.fetch_data(stock)
                if not data.empty:
                    data['RSI'] = self.calculate_rsi(data)
                    rsi_data[stock] = data['RSI'].iloc[-1]  # Get the most recent RSI value
                else:
                    logging.warning(f"No data for {stock}")
            except Exception as e:
                logging.error(f"Error fetching data for {stock}: {e}", exc_info=True)  # Log the stack trace

        # Convert to DataFrame for plotting
        rsi_df = pd.DataFrame.from_dict(rsi_data, orient='index', columns=['RSI'])

        # Create the heatmap plot
        plt.figure(figsize=(15, 10))
        sns.scatterplot(data=rsi_df.reset_index(), x=rsi_df.reset_index().index, y='RSI', hue='RSI', palette='coolwarm', s=100, legend=None)

        # Add the zones
        plt.axhspan(70, 100, color='red', alpha=0.3, lw=0)  # Overbought
        plt.axhspan(30, 70, color='green', alpha=0.1, lw=0)  # Neutral
        plt.axhspan(0, 30, color='blue', alpha=0.3, lw=0)  # Oversold

        # Add lines
        plt.axhline(70, color='red', linestyle='--', linewidth=1)
        plt.axhline(30, color='blue', linestyle='--', linewidth=1)

        # Titles and labels
        plt.title('30-Minute Interval RSI Heatmap for Selected US Stocks', fontsize=16)
        plt.xlabel('Stock Index', fontsize=14)
        plt.ylabel('RSI (30m)', fontsize=14)

        # Annotate each point
        for i, (stock, rsi) in enumerate(rsi_data.items()):
            plt.text(i, rsi, stock, horizontalalignment='right', size='medium', color='black', weight='semibold')

        plt.savefig('rsi_heatmap.png')
        plt.close()

    async def send_heatmap_to_telegram(self):
        logging.info("Sending heatmap to Telegram")
        try:
            await self.application.bot.send_photo(chat_id=self.chat_id, photo=open('rsi_heatmap.png', 'rb'))
            logging.info("Heatmap sent to Telegram")
        except Exception as e:
            logging.error(f"Error sending heatmap to Telegram: {e}", exc_info=True)  # Log the stack trace

    def generate_and_send_heatmap(self):
        try:
            self.generate_heatmap()
            asyncio.run(self.send_heatmap_to_telegram())
        except Exception as e:
            logging.error(f"Error in generate_and_send_heatmap: {e}", exc_info=True)  # Log the stack trace

    def schedule_jobs(self):
        schedule.every(30).minutes.do(self.generate_and_send_heatmap)
        logging.info("Scheduled job every 30 minutes")

    def run_scheduler(self):
        self.generate_and_send_heatmap()  # Run immediately once at start
        self.schedule_jobs()
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in run_scheduler: {e}", exc_info=True)  # Log the stack trace

if __name__ == "__main__":
    scheduler = RSIHeatmapScheduler()
    scheduler.run_scheduler()

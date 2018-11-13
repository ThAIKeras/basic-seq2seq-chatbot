# A Basic Seq2Seq Conversational Chatbot

## Training
The whole process is implemented in Jupyter Notebook for an explanatory purpose. The notebook file is located at `notebooks/custom_conversational_model.ipynb`.

## Running
After training finished, this application can be run in two modes.

### Interactive
A user can directly chat with a bot by using this command.

```python main_bot.py interactive```

### Telegram
A bot can be linked to Telegram Messenger by running this command.

```python main_bot.py telegram --token=TELEGRAM_TOKEN```

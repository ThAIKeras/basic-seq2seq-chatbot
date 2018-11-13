# The code in this file is based on the main code in HSE's NLP course project.
# Here is the original file, https://github.com/hse-aml/natural-language-processing/blob/master/project/main_bot.py
# ==============================================================================


#!/usr/bin/env python3

import requests
import time
import argparse
import os
import json

from requests.compat import urljoin

from dialogue_manager import DialogueManager


class BotHandler(object):
    """
        BotHandler is a class which implements all back-end of the bot.
        It has tree main functions:
            'get_updates' — checks for new messages
            'send_message' – posts new message to user
            'get_answer' — computes the most relevant on a user's question
    """

    def __init__(self, token, dialogue_manager):
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)
        self.dialogue_manager = dialogue_manager

    def get_updates(self, offset=None, timeout=30):
        params = {"timeout": timeout, "offset": offset}
        raw_resp = requests.get(urljoin(self.api_url, "getUpdates"), params)
        try:
            resp = raw_resp.json()
        except json.decoder.JSONDecodeError as e:
            print("Failed to parse response {}: {}.".format(raw_resp.content, e))
            return []

        if "result" not in resp:
            return []
        return resp["result"]

    def send_message(self, chat_id, text):
        params = {"chat_id": chat_id, "text": text}
        return requests.post(urljoin(self.api_url, "sendMessage"), params)

    def get_answer(self, question):
        if question == '/start':
            return "Hi, I am your project bot. How can I help you today?"
        return self.dialogue_manager.generate_answer(question)


def is_unicode(text):
    return len(text) == len(text.encode())


class SimpleDialogueManager(object):
    """
    This is the simplest dialogue manager to test the telegram bot.
    Your task is to create a more advanced one in dialogue_manager.py."
    """
    
    def generate_answer(self, question): 
        return "Hello, world!" 

    
def interactive_main(args):
    """Main function for the interactive mode."""
    dialogue_manager = DialogueManager()

    print('Welcome to the interactive mode, here you can chitchat with the bot. To quit the program, type \'exit\' or just press ENTER. Have fun!')
    
    while True:
        question = input('Q: ')
        if question == '' or question == 'exit':
            break

        print('A: {}'.format(dialogue_manager.generate_answer(question)))


def telegram_main(args):
    """Main function for the telegram mode."""
    
    token = args.token

    if not token:
        if not "TELEGRAM_TOKEN" in os.environ:
            print("Please, set bot token through --token or TELEGRAM_TOKEN env variable")
            return
        token = os.environ["TELEGRAM_TOKEN"]

    #################################################################
    
    # Your task is to complete dialogue_manager.py and use your 
    # advanced DialogueManager instead of SimpleDialogueManager. 
    
    # This is the point where you plug it into the Telegram bot. 
    # Do not forget to import all needed dependencies when you do so.
    
    #simple_manager = SimpleDialogueManager()
    #bot = BotHandler(token, simple_manager)
    
    manager = DialogueManager()
    bot = BotHandler(token, manager)
    
    ###############################################################

    print("Ready to talk!")
    offset = 0
    while True:
        updates = bot.get_updates(offset=offset)
        for update in updates:
            print("An update received.")
            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                if "text" in update["message"]:
                    text = update["message"]["text"]
                    if is_unicode(text):
                        print("Update content: {}".format(update))
                        bot.send_message(chat_id, bot.get_answer(update["message"]["text"]))
                    else:
                        bot.send_message(chat_id, "Hmm, you are sending some weird characters to me...")
            offset = max(offset, update['update_id'] + 1)
        time.sleep(1)
        

def parse_args():
    arg_parser = argparse.ArgumentParser()
    subparsers = arg_parser.add_subparsers(title="subcommands")
    
    i_parser = subparsers.add_parser("interactive")
    i_parser.set_defaults(main=interactive_main)
    
    t_parser = subparsers.add_parser("telegram")
    t_parser.add_argument('--token', type=str, default='', required=True)
    t_parser.set_defaults(main=telegram_main)
                                     
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.main(args)
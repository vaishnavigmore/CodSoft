import re

def simple_chatbot(user_input):
    # Convert user input to lowercase for case-insensitive matching
    user_input_lower = user_input.lower()

    # Rule 1: Greeting
    if any(word in user_input_lower for word in ["hello", "hi", "hey"]):
        return "Hello! How can I help you today?"

    # Rule 2: Weather
    elif re.search(r'\bweather\b', user_input_lower):
        return "I'm sorry, I'm just a simple chatbot and don't have real-time weather information."

    # Rule 3: Default response
    else:
        return "I'm not sure how to respond to that. Can you please rephrase or ask something else?"

# Simple loop to run the chatbot
while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    
    response = simple_chatbot(user_input)
    print("Chatbot:", response)

import openai
from datetime import datetime
import json
import requests

# Function to interact with OpenAI API

def ask_gpt(prompt, model="gpt-4"):
    """
    Makes API call to OpenAI GPT model with a Siemens-focused assistant role.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in Siemens products and models. Provide accurate, detailed, and up-to-date information about Siemens products, their specifications, and applications."},
            {"role": "user", "content": prompt},
        ]
    )
    return response

# Web Search Function (to fetch real-time info if required)
def web_search(query):
    """
    Performs a web search to fetch the latest data about Siemens products.
    """
    from openai import OpenAIError
    try:
        # Assuming an internal web API is configured, else integrate with Google/Bing API
        response = requests.get(f'https://www.googleapis.com/customsearch/v1?q={query}&key=API_KEY&cx=SEARCH_ENGINE_ID')
        result = response.json()
        if 'items' in result:
            return result['items'][0]['snippet']
        return "No relevant results found."
    except Exception as e:
        print(f"Error in web search: {e}")
        return "Unable to fetch data from the web."

# Function to extract product details

def get_product_details(text):
    """
    Extracts Siemens product details or searches the web if data isn't clear.
    """
    prompt = f"Provide detailed information about this Siemens product or model: '{text}'. Include features, specifications, and applications. If information is missing, note it."

    # Call GPT API
    try:
        response = ask_gpt(prompt)
        if response and 'choices' in response:
            result_text = response['choices'][0]['message']['content']
            if "I don't know" in result_text or "unsure":
                # Trigger a web search if GPT doesn't have the data
                web_result = web_search(text)
                return f"{result_text}\nAdditional Web Search Result: {web_result}"
            return result_text
        else:
            return "No valid response received from GPT."
    except Exception as e:
        print(f"Error in GPT response: {e}")
        return None

# Compare two Siemens products
def compare_products(product1, product2):
    """
    Compare two Siemens products based on web search and GPT analysis.
    """
    prompt = f"Compare the Siemens products '{product1}' and '{product2}'. Provide details on whether they are the same model, similar, or different, based on specifications and features. Output only 'Yes' or 'No'."
    try:
        response = ask_gpt(prompt)
        if response and 'choices' in response:
            result_text = response['choices'][0]['message']['content'].strip()
            if result_text.lower() in ['yes', 'no']:
                return result_text
            else:
                # Fallback to web search
                search_result1 = web_search(product1)
                search_result2 = web_search(product2)
                return "No" if search_result1 != search_result2 else "Yes"
        else:
            return "Unable to determine."
    except Exception as e:
        print(f"Error comparing products: {e}")
        return "Error occurred."

# Process the query
def process_query(text):
    retries = 0
    while retries <= 3:
        try:
            result = get_product_details(text)
            if result:
                return result
            else:
                print("Retrying query...")
                retries += 1
        except Exception as e:
            print(f"Error: {e}")
            retries += 1
    return "Unable to process query. Please try again later."

# # Example usage
# if __name__ == "__main__":
#     query = "Siemens 3RH29111HA11 specifications"
#     result = process_query(query)
#     print(result)

#     # Example comparison
#     product1 = "Siemens 3RH29111HA11"
#     product2 = "Siemens 3RH29111HA13"
#     comparison_result = compare_products(product1, product2)
#     print(f"Are the products the same? {comparison_result}")

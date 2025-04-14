import requests
import time
from typing import Optional

def query_ollama_for_translation(language: str, text: str, model: str = "gemma3", gpu_index: int = 1) -> Optional[str]:

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": (
            f"Translate the following text from English to {language}. "
            f"Strictly do NOT translate technical terms, brand names, or code snippets and keep them in English. "
            f"Only translate regular words. Only provide the translation, nothing else. Text: '{text}'"
        ),
        "stream": False,
        "options": {
            "numa": True,
            "main_gpu": gpu_index,
            "low_vram": False,
        }
    }

    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(f"--- {time.time() - start_time:.2f} seconds ---")
        
        data = response.json()
        return data.get("response", None)
    
    except requests.RequestException as e:
        print(f"Error during API request: {e}")
        return None
    except KeyError:
        print("Unexpected response format from the API.")
        return None


def unload_ollama_model(model: str = "gemma3") -> None:

    # print(f" Unloading model {model}...")

    print('\x1b[6;30;42m' + 'Unloading model ' + model + '\x1b[0m')

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "keep_alive": 0
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print('\x1b[6;30;42m' + f"Model {model} unloaded successfully." + '\x1b[0m')
    except requests.RequestException as e:
        print(f"Error during model unload request: {e}")
    except KeyError:
        print("Unexpected response format from the API.")

# text_to_translate = """
#  Now, let's see the second point. The union of two regular expressions is also a regular expression. So, let's say that we have two regular 
#  expressions which we call R1 and R2. These two are regular expressions and then the union of these regular expressions which which is
#    represented by R1 plus R2, this is also a regular expression. So, if R1 and R2 are regular, then their union will also be regular. 
#    Alright, so let's see the third point. The concatenation of two regular expressions is also a regular expression. So, let's say we again have two regular expressions R1 and R2. Then, the concatenation of these two regular expressions which is represented by R1 and R2 or you can even put a dot in between them. So, this will also be a regular expression. So, if we have two regular expressions, their concatenation will always be a regular expression. Alright, so let's come to the fourth point. The iteration or closure of a regular expression is also a regular expression. So, if we have a regular expression, let's say r, and then the closure of R which is represented by R star will also be a regular expression. So, what is disclosure? I have already mentioned and discussed about disclosure in one of our previous lectures but let me just repeat it again. Suppose we have a symbol A and then the closure of A represented by A star will be everything that you can form using this A. So, it includes the empty symbol as well as A and AA, AAA and so on. Everything that you can form using this A. It will be an infinite set. So, that is the closure of A. So, if A is a regular expression then its closure will also be a regular expression. So that is what this fourth point is telling us. And then the fifth point says that the regular expression over sigma are precisely those obtained recursively by the application of the above rules once or several times. So, all the regular expressions over sigma, what are they? They are simply the regular expressions that are obtained by applying these rules which are given above once or many times. So, we see that when we have regular expressions, if you do the union of them or concatenation of them or closure of them, you obtain new regular expressions. So this fifth point tells us
#    that the regular expressions over sigma are those which are obtained by applying these rules which are given above once or many times."""

# reply = query_ollama_for_translation("hindi", text_to_translate )
# print(reply)


# if __name__ == "__main__":
#     query_ollama_for_translation()


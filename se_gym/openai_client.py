import openai
import httpx
import os
import dotenv
import anthropic

def get_lmu_openai_client():
    """
    Returns the OpenAI API client object with a custom LMU base_url.
    
    Parameters:
    
    Returns:
        openai.OpenAI: The OpenAI API client object authenticated with BasicAuth and custom url.
    """
    dotenv.load_dotenv()
    USERNAME = os.getenv("API_USERNAME")
    PASSWORD = os.getenv("API_PASSWORD")
    assert USERNAME is not None, "API_USERNAME is not set in .env or file not found."
    assert PASSWORD is not None, "API_PASSWORD is not set in .env or file not found."
    openai.OpenAI.custom_auth = httpx.BasicAuth(USERNAME, PASSWORD)
    client = openai.OpenAI(
        base_url="https://ollama.mobile.ifi.lmu.de/v1/", api_key="none"
    )
    return client

def get_openai_client():
    """
    Returns the OpenAI API client object.
    
    Parameters:

    Returns:
        openai.Client: The OpenAI API client object.
    """
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    openai.api_key = api_key
    return openai

def get_groq_client():
    """
    Returns the Groq API client object.
    
    Parameters:

    Returns:
        openai.Client: The OpenAI API client object, with adapted base_url.
    """
    dotenv.load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1", api_key=api_key
    )
    return client

def get_anthropic_client():
    dotenv.load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=api_key
    )
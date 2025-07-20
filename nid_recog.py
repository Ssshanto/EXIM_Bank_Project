import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import random

# Load environment variables from .env file
load_dotenv()

def get_api_key():
    """
    Get a Gemini API key from environment variables.
    Supports multiple keys for load balancing and fallback.
    
    Returns:
        str: A valid API key
    
    Raises:
        ValueError: If no valid API key is found
    """
    # Try to get the default key first
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "your_api_key_here":
        return api_key
    
    # Try multiple numbered keys
    api_keys = []
    for i in range(1, 11):  # Support up to 10 keys
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key and key != "your_api_key_here":
            api_keys.append(key)
    
    if not api_keys:
        raise ValueError(
            "No valid Gemini API key found in environment variables. "
            "Please set GEMINI_API_KEY or GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. in your .env file"
        )
    
    # Return a random key for load balancing
    return random.choice(api_keys)

def get_nid_info(image_path):
    """
    Extracts structured information from a Bangladeshi NID card image using Gemini 2.5 Flash.
    
    Args:
        image_path (str): The file path to the image of the NID card.
    
    Returns:
        dict: A dictionary containing the extracted NID card fields.
    
    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be read or if JSON parsing fails.
        Exception: For any other issues during the Gemini API call.
    """
    # Get API key from environment variables
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    
    # Use the specific preview model
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    
    # Ensure the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    
    # Determine mime_type dynamically based on file extension
    _, file_extension = os.path.splitext(image_path)
    mime_type = None
    
    # Map file extensions to MIME types
    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.gif': 'image/gif'
    }
    
    mime_type = mime_type_map.get(file_extension.lower())
    
    if mime_type is None:
        raise ValueError(f"Unsupported image format: {file_extension}. Supported formats: {', '.join(mime_type_map.keys())}")
    
    # Load image data in binary mode
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
    except IOError as e:
        raise ValueError(f"Could not read image file {image_path}: {e}")
    
    prompt = """
Extract all possible fields from this Bangladeshi National ID card image and return the result as a JSON.

Required fields:
- Name
- Name_Bangla
- Date_of_Birth
- NID_Number
- Father's Name
- Mother's Name

If any field is missing, return null for that field.
Output JSON only. No explanation, no markdown.
"""
    
    try:
        # The image data should be passed directly within the list of content parts.
        # The SDK automatically handles the conversion to the correct internal type.
        response = model.generate_content(
            [
                prompt,
                {
                    'mime_type': mime_type,
                    'data': image_data
                }
            ],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        response.resolve()  # Ensure the response is complete
        
        # Attempt JSON parsing
        response_text = response.text.strip()
        # Remove potential markdown code block wrappers
        if response_text.startswith("```json") and response_text.endswith("```"):
            json_str = response_text[7:-3].strip()
        else:
            json_str = response_text
        
        # Parse the JSON string
        return json.loads(json_str)
        
    except json.JSONDecodeError as e:
        # Provide more context if JSON parsing fails
        raise ValueError(f"Failed to parse JSON from Gemini response: {e}\nRaw Output:\n{response.text}")
    except Exception as e:
        # Catch other potential API or network errors
        raise Exception(f"An error occurred during Gemini API call: {e}")


if __name__ == "__main__":
    try:
        # Test API key availability
        api_key = get_api_key()
        print("API key loaded successfully")
        print(f"Using key: {api_key[:10]}...{api_key[-4:]}")
        
        # Example usage
        nid_info = get_nid_info("images/NID6.jpeg")
        print("Extracted NID Information:")
        print(json.dumps(nid_info, indent=2, ensure_ascii=False))
        
        print("\nNID recognition function ready!")
        print("Usage: get_nid_info('path/to/nid_card.jpg')")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nTo fix this:")
        print("1. Create a .env file in your project directory")
        print("2. Add your Gemini API keys to the .env file:")
        print("   GEMINI_API_KEY=your_api_key_here")
        print("   GEMINI_API_KEY_1=your_first_key_here")
        print("   GEMINI_API_KEY_2=your_second_key_here")
        print("3. Install python-dotenv: pip install python-dotenv")
    except Exception as e:
        print(f"Error: {e}")

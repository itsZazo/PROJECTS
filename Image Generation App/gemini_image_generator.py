from mcp.server.fastmcp import FastMCP
import requests
import json
import base64
import re

mcp = FastMCP("Google Gemini Image Generator")

@mcp.tool()
def generate_image(prompt: str, aspect_ratio: str = "1:1", output_filename: str = "generated_image.png") -> str:
    """
    Generates an image from a text prompt using Google Gemini API.

    Parameters:
    - prompt: Description of the image to generate
    - aspect_ratio: Aspect ratio (currently not directly supported, but mentioned in prompt)
    - output_filename: Where to save the image

    Returns:
    - Status message with file path or error information

    Note: Requires Google Gemini API key (free tier available)
    """

    # Replace with your actual Gemini API key from Google AI Studio
    API_KEY = "YOUR GEMINI API KEY"
    
    # Gemini API endpoint for image generation
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-preview-image-generation:generateContent"
    
    headers = {
        "x-goog-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    # Enhance prompt with aspect ratio if specified
    enhanced_prompt = prompt
    if aspect_ratio != "1:1":
        if aspect_ratio == "16:9":
            enhanced_prompt += " in wide landscape format 16:9 aspect ratio"
        elif aspect_ratio == "9:16":
            enhanced_prompt += " in tall portrait format 9:16 aspect ratio"
        elif aspect_ratio == "4:3":
            enhanced_prompt += " in 4:3 aspect ratio"
        else:
            enhanced_prompt += f" in {aspect_ratio} aspect ratio"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": enhanced_prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"]  # Required for image generation
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract image data from response
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "inlineData" in part:
                            # Found image data
                            image_data = part["inlineData"]["data"]
                            
                            # Decode and save the image
                            image_bytes = base64.b64decode(image_data)
                            
                            with open(output_filename, "wb") as f:
                                f.write(image_bytes)
                            
                            return f"Image successfully generated and saved as {output_filename} (via Google Gemini)"
                
                return "No image data found in the response"
            else:
                return "No candidates found in the response"
        
        else:
            return f"Failed to generate image. Status: {response.status_code}, Error: {response.text}"
    
    except Exception as e:
        return f"Error generating image: {str(e)}"


@mcp.tool()
def generate_image_with_editing(base_prompt: str, edit_instruction: str, output_filename: str = "edited_image.png") -> str:
    """
    Generates an image and then applies editing instructions using Gemini's conversational capabilities.

    Parameters:
    - base_prompt: Initial description of the image
    - edit_instruction: What to change or add to the image
    - output_filename: Where to save the final image

    Returns:
    - Status message with file path or error information
    """

    API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-preview-image-generation:generateContent"
    
    headers = {
        "x-goog-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    # Combined prompt for generation and editing
    combined_prompt = f"Create an image: {base_prompt}. Then, {edit_instruction}"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": combined_prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"]
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            response_data = response.json()
            
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "inlineData" in part:
                            image_data = part["inlineData"]["data"]
                            image_bytes = base64.b64decode(image_data)
                            
                            with open(output_filename, "wb") as f:
                                f.write(image_bytes)
                            
                            return f"Edited image successfully generated and saved as {output_filename} (via Google Gemini)"
                
                return "No image data found in the response"
            else:
                return "No candidates found in the response"
        
        else:
            return f"Failed to generate image. Status: {response.status_code}, Error: {response.text}"
    
    except Exception as e:
        return f"Error generating image: {str(e)}"


@mcp.tool()
def test_gemini_connection() -> str:
    """
    Tests the connection to Google Gemini API.
    
    Returns:
    - Connection status and API information
    """
    
    API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    
    # Test with a simple text generation first
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    headers = {
        "x-goog-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Hello, this is a test. Please respond with 'Connection successful!'"
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            response_data = response.json()
            
            if "candidates" in response_data:
                text_response = response_data["candidates"][0]["content"]["parts"][0]["text"]
                return f"✅ Gemini API Connection Successful!\nResponse: {text_response}"
            else:
                return "✅ Connection successful but unexpected response format"
        
        else:
            return f"❌ Connection failed. Status: {response.status_code}, Error: {response.text}"
    
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


@mcp.tool()
def get_gemini_models() -> str:
    """
    Get list of available Gemini models.
    
    Returns:
    - List of available models
    """
    
    API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    headers = {
        "x-goog-api-key": API_KEY
    }

    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            models_data = response.json()
            
            if "models" in models_data:
                model_list = []
                for model in models_data["models"]:
                    name = model.get("name", "Unknown")
                    display_name = model.get("displayName", "Unknown")
                    description = model.get("description", "No description")
                    
                    model_list.append(f"• {display_name} ({name})\n  {description[:100]}...")
                
                return f"📋 Available Gemini Models:\n\n" + "\n\n".join(model_list)
            else:
                return "No models found in response"
        
        else:
            return f"Failed to fetch models. Status: {response.status_code}, Error: {response.text}"
    
    except Exception as e:
        return f"Error fetching models: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
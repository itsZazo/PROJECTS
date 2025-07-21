from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Image Generator")

@mcp.tool()
def generate_image(prompt : str, aspect_ratio: str = "1:1", output_filename: str = "generated_image.png") -> str:
    """
    Generates an image from a text prompt using AI's Stable Diffusion XL model.

    Parameters:
    prompt: Description of the image.
    -aspect ratio: Aspect ratio, e.g., "1:1", "16:9", etc.
    -output_filename: Where to save the image.

    Returns:
    - File path of the generated image.

    """

    API_KEY = "YOUR STABLE DIFFUSION API KEY"
    url = "https://api.stability.ai/v2beta/stable-image/generate/core" 

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "image/*",
        
    }


    data = {
        "model": "stable-diffusion-xl-1024-v1-0",
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": "png"
    }

    response = requests.post(url, headers=headers, files={"none": (None, "")}, data=data)

    if response.ok:
        with open(output_filename, "wb") as f:
            f.write(response.content)
        return f"Image saved as {output_filename}"
    else:
        return f"Failed to generate image. Status: {response.status_code}, Error: {response.text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")



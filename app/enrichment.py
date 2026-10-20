import base64

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm

VISION_SYSTEM = """ you are an expert economist analyszing the Economic Survey Of India. 
your role is to transcribe and describe the visual information(charts,tables,infographics) on the provided 
page.
- If there are tables, transcribe the key data points.
- If there are charts, explain the trend and quote the numbers shown.
- If there are no visual elements, reply exactly with "No visual elements".

"""


def generate_image_caption(image_path: str) -> str:
    """Uses gpt-5-mini model to provide vision enrichment for visuals on pdf pages."""

    # 1. Read and encode the image into (text safe base 64 format and later convert to python string)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # 2. Setup Vision LLM
    llm = get_llm(temperature=0.0)

    # 3. Create the message with the image payload
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the visual elements on this page."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        ]
    )
    # 4. Invoke the LLM
    response = llm.invoke([SystemMessage(content=VISION_SYSTEM), message])
    return response.content.strip()

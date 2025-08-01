from src.tools.dam_tools import dam_caption_image


def caption_node(state):
        caption = dam_caption_image(state.get("image"))
        return {"image_caption": caption, "phase": "prevote"}
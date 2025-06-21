from PIL import Image


def get_formatted_img(img, resolution):
    # if image path is provided, load image
    if type(img) is str:
        img = Image.open(img)
    img = img.convert('RGBA')

    # paste image onto a white square image (for a white background)
    width, height = img.size
    if width > height:
        new_size = width
        sq_img = Image.new(
            'RGB', (new_size, new_size), (255, 255, 255))
        offset = (0, (new_size - height) // 2)
    elif height > width:
        new_size = height
        sq_img = Image.new(
            'RGB', (new_size, new_size), (255, 255, 255))
        offset = ((new_size - width) // 2, 0)
    else:
        # already square
        sq_img = Image.new(
            'RGB', (width, height), (255, 255, 255))
        offset = (0, 0)
    sq_img.paste(img, offset, mask=img)

    # resize to desired size
    resized_sq_img = sq_img.convert('RGB').resize((resolution, resolution))
    return resized_sq_img

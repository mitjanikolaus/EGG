from egg.zoo.visual_ref.preprocess import TOKEN_START, TOKEN_END, TOKEN_PADDING

import matplotlib.pyplot as plt

SPECIAL_CHARACTERS = [TOKEN_START, TOKEN_END, TOKEN_PADDING]


def decode_caption(caption, vocab):
    words = [vocab.itos[word] for word in caption if vocab.itos[word] not in SPECIAL_CHARACTERS]
    return " ".join(words)


def print_caption(caption, vocab):
    caption = decode_caption(caption, vocab)
    print(caption)

def show_image(image_data):
    image_data = image_data.reshape(image_data.shape[1], image_data.shape[2], image_data.shape[0])

    plt.imshow(image_data)
    plt.show()

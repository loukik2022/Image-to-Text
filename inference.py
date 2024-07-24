import torch
import torchvision.transforms as T
import transformers
from model import EncoderCNN, DecoderRNN
from PIL import Image

# Hyperparameters
batch_size = 32  # batch size
vocab_size = 30522  # vocabulary size (trained on)
embed_size = 256  # dimensionality of image and word embeddings
hidden_size = 512  # number of features in hidden state of the RNN decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

# Load model
path=r"image-to-text.pth"
checkpoint = torch.load(path, map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

# Set models to evaluation mode
encoder.eval()
decoder.eval()

## Tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

# Transform image
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    T.Resize((256, 256)),
])

# Inference function
def generate_caption(image_path):
    image = Image.fromarray(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        features = encoder(image).unsqueeze(1)
        output = decoder.sample(features)   # token_ids

    captions = tokenizer.decode(output, skip_special_tokens=True)

    return captions
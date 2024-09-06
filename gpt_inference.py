import torch
from gpt_model import GPTLanguageModel
from utils import get_encoder_decoder

torch.manual_seed(1337)
# Load the model and set to eval mode
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPTLanguageModel()
model.load_state_dict(
    torch.load(
        "models/gpt_model.pth",  # "models/gpt_model_shakespeare_fine_tuned.pth",
        map_location=device,
        weights_only=True,
    )
)
model.to(device)
model.eval()

# Load the tokenizer (the encoding/decoding functions)
encode, decode = get_encoder_decoder(path="data/text_data.txt")


subject = "Graphics processing unit"
context = torch.tensor(
    [encode(f"{subject}:\n")], dtype=torch.long, device=device
)

completion = decode(model.generate(context, max_new_tokens=1024)[0].tolist())
print(f"\n{completion}")

# Optionally save the generated text to a file
print("\nSaving text to file")
with open("data/generated_output.txt", "w", encoding="utf-8") as f:
    f.write(completion)

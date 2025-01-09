import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

##################################
# GAN Generator Architecture
# Must match exactly the architecture used during training.
##################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = 100
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

##################################
# Load a Pre-Trained Generator
##################################
def load_generator(model_filename, models_folder="gan_models", device="cpu"):
    model_path = os.path.join(models_folder, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified model '{model_path}' does not exist.")

    generator = Generator()
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval().to(device)
    return generator

##################################
# Main Generation Logic
##################################
def generate_digits(generator, images_per_digit, device="cpu", output_dir="generated_dataset", batch_size=128):
    """
    Generates images_per_digit images for each digit [0..9].
    Saves them into output_dir/<digit>/<index>.png.
    """

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    for d in range(10):
        os.makedirs(os.path.join(output_dir, str(d)), exist_ok=True)

    latent_dim = 100  # Must match the generator's latent dimension
    total_generated = 0

    print(f"Generating {images_per_digit} images for each digit (0-9).")

    # We do not actually force the generator to produce each digit label specifically,
    # since this is an unconditional GAN. Instead, we simply generate random noise
    # and store it under each digit’s folder. 
    # If you need a class-conditional GAN, you'll need a different approach.
    for digit in range(10):
        count = 0
        while count < images_per_digit:
            # Sample random noise
            z = torch.randn(batch_size, latent_dim).to(device)
            # Generate images
            gen_imgs = generator(z)
            # Save them until we reach images_per_digit for this digit
            for img in gen_imgs:
                if count >= images_per_digit:
                    break
                save_path = os.path.join(output_dir, str(digit), f"{count}.png")
                save_image(img, save_path, normalize=True)
                count += 1
            total_generated += min(batch_size, images_per_digit - count)

        print(f"Digit {digit}: generated {images_per_digit} images.")

    print(f"Total images generated: {total_generated}")

##################################
# Entry Point
##################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prompt for the generator model filename
    model_filename = input("Enter the generator model filename (e.g., 'generator.pth'): ").strip()
    if not model_filename:
        print("No model filename provided. Exiting.")
        return

    # Prompt for number of images to generate per digit
    try:
        images_per_digit = int(input("Enter the number of images to generate per digit: ").strip())
    except ValueError:
        print("Invalid number input. Exiting.")
        return

    # Load the generator model
    try:
        generator = load_generator(model_filename, models_folder="gan_models", device=device)
    except FileNotFoundError as e:
        print(e)
        return

    # Generate digits
    generate_digits(generator, images_per_digit, device=device, output_dir="generated_dataset", batch_size=128)

if __name__ == "__main__":
    main()
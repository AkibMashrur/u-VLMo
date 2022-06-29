"""Handler for captioning image from a specified path."""
import streamlit as st
import argparse
import ruamel.yaml as yaml
import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms

from models.model_captioning import XVLM


def parse_args():
    """Parse default arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='checkpoints/coco_capt_ft_epoch_4.th')
    parser.add_argument('--config', default='configs/Captioning.yaml')
    parser.add_argument('--image_path', default='images/random_2.jpg')
    args = parser.parse_args()
    return args


def generate(model, image, device, config):
    """Generate caption for the specified image."""
    model.eval()

    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module

    image = image.to(device, non_blocking=True)

    with torch.no_grad():
        captions = model_without_ddp.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                              min_length=config['min_length'])

    return captions


def prepare_image(image, config) -> torch.Tensor:
    """Prepare image for captioning."""
    image = image.convert('RGB')
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.Resampling.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    image = transform(image)
    image = image.unsqueeze(dim=0)
    image = image.view(1, 3, config['image_res'], config['image_res'])
    return image


# @st.cache
def load_model(model_path, config) -> torch.nn.Module:
    """Load model from specified checkpoint."""
    model = XVLM(config=config)
    model.load_pretrained(model_path, config, is_eval=True)
    model.eval()
    return model


def captioning_handler():
    """Handle image captioning."""
    args = parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    image = Image.open(args.image_path).convert('RGB')
    image = prepare_image(image, config)
    image = image.to(device)

    model = load_model(args.model_path, config)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model = model.to(device)

    caption = generate(model, image, device, config)
    print(caption)


def caption_api(image):
    """Caption image from app."""
    args = parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = prepare_image(image, config)
    image = image.to(device)
    model = load_model(args.model_path, config)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model = model.to(device)
    caption = generate(model, image, device, config)
    return caption


if __name__ == "__main__":
    captioning_handler()

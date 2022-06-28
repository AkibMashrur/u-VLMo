"""Handler for captioning image from a specified path."""
import torch


@torch.no_grad()
def generate(model, image, device, config):
    """Generate caption for the specified image."""
    model.eval()

    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module

    image = image.to(device, non_blocking=True)

    captions = model_without_ddp.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                          min_length=config['min_length'])

    return captions

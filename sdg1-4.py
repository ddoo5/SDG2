import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from colorama import Fore


def GenerateImage(request, choice):
    # load model and scheduler
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16",
                                                   torch_dtype=torch.get_default_dtype())

    # for low ram
    if choice.lower() == "y":
        pipe.enable_attention_slicing()

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    device = torch.device(device)
    pipe = pipe.to(device)

    prompt = request
    image = pipe(prompt).images[0]

    request = request.replace(' ', '')
    image.save(f"./{request}.png")


def UpgradeImage(choice, url, type):
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.get_default_dtype())

    # for low ram
    if choice.lower() == "y":
        pipeline.enable_attention_slicing()

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    device = torch.device(device)
    pipeline = pipeline.to(device)

    # path
    if type == 0:
        low_res_img = Image.open(url).convert("RGB")
        low_res_img = low_res_img.resize((128, 128))
    # url
    elif type == 1:
        response = requests.get(url)
        low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        low_res_img = low_res_img.resize((128, 128))

    prompt = input(Fore.MAGENTA + "Enter request: ")

    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    prompt = prompt.replace(' ', '')
    upscaled_image.save(f"./{prompt}.png")


# main method
while True:
    selectZero = input(Fore.MAGENTA + "Do u want to generate image? y/n \n")

    # generating png picture 768x768
    if selectZero.lower() == "y":
        request = input(Fore.MAGENTA + "Enter request: ")
        choice = input(Fore.MAGENTA + "Do you have low ram? y/n\n")
        GenerateImage(request, choice)

    select = input(Fore.MAGENTA + "Do u want to upscale image? y/n \n")

    # upgscaling png picture 768x768
    if select.lower() == "y":
        selectTwo = input(Fore.MAGENTA + "Would you like to use path or url?\n")

        # method for local picture
        if selectTwo.lower() == "path":
            path = input(Fore.MAGENTA + "Enter path: ")
            choice = input(Fore.MAGENTA + "Do you have low ram? y/n\n")
            UpgradeImage(choice, path, 0)

        # method for picture from the Internet
        elif selectTwo.lower() == "url":
            url = input(Fore.MAGENTA + "Enter url: ")
            choice = input(Fore.MAGENTA + "Do you have low ram? y/n\n")
            UpgradeImage(choice, url, 1)
        else:
            print(Fore.RED + "You can enter only 'path' or 'url'")
    else:
        break
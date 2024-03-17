from PIL import Image
import wandb as wb


im = Image.open("94_acc.png")
with wb.init(name ="94_accuracy") as run:
    img = wb.Image(im,caption="Local Accuracy")
    run.log({'94_accuracy':img})

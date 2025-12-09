from torchvision import io as tvio
from torchvision import models
import torchinfo

input_image = tvio.decode_image("assets/IMG_3436.jpg")
print(type(input_image))
print(input_image.shape, input_image.dtype)

weights = models.VGG19_BN_Weights.DEFAULT

model = models.vgg19_bn(weights = weights)
torchinfo.summary(model)

preprocess = weights.transforms()

batch = preprocess(input_image).unsqueeze(dim = 0)
print(batch.shape)

model.eval()

output_logits = model(batch)
print(output_logits.shape, output_logits.dtype)

output_probs = output_logits.softmax(dim = 1)

class_id = output_probs[0].argmax().item()
score = output_probs[0][class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:1f}%")
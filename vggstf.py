from libraries import *

ITERATIONS = 7
CHANNELS = 3
IMAGE_SIZE = 224
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_HEIGHT = IMAGE_SIZE
CONTENT_WEIGHT = 0.5
STYLE_WEIGHT = 4.5
TOTAL_VARIATION_WEIGHT = 1.1
TOTAL_VARIATION_LOSS_FACTOR = 1.25

input_image_path = "input.png"
style_image_path = "style1.png"
output_image_path = "output1.png"
combined_image_path = "combined1.png"

# style_layers = ["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3"]
# content_layers = ["conv2_2"]

start = time.time()

style_layers = ["block1_conv2", "block2_conv2", "block3_conv3"]
content_layers = ["block2_conv2"]

input_image = Image.open('input.jpg')
input_image = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
input_image.save(input_image_path)
# input_image.show()

style_image = Image.open('style1.jpg')
style_image = style_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
style_image.save(style_image_path)

input_image = preprocess(input_image)
style_image = preprocess(style_image)
combination_image = backend.placeholder((1, IMAGE_HEIGHT, IMAGE_SIZE, 3))
input_tensor = backend.concatenate([input_image,style_image,combination_image], axis=0)

model = VGG16(input_tensor=input_tensor)
layers = dict([(layer.name, layer.output) for layer in model.layers])

loss = backend.variable(0.)

for content_layer in content_layers:
	layer_features = layers[content_layer]
	content_image_features = layer_features[0, :, :, :]
	combination_features = layer_features[2, :, :, :]
	loss += CONTENT_WEIGHT * content_loss(content_image_features,combination_features)

for layer_name in style_layers:
	layer_features = layers[layer_name]
	style_features = layer_features[1, :, :, :]
	combination_features = layer_features[2, :, :, :]
	style_loss = compute_style_loss(style_features, combination_features)
	loss += (STYLE_WEIGHT / len(style_layers)) * style_loss

loss += TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image)

outputs = [loss]
outputs += backend.gradients(loss, combination_image)

def evaluate_loss_and_gradients(x):
	x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
	outs = backend.function([combination_image], outputs)([x])
	loss = outs[0]
	gradients = outs[1].flatten().astype("float64")
	return loss, gradients

class Evaluator:

	def loss(self, x):
		loss, gradients = evaluate_loss_and_gradients(x)
		self._gradients = gradients
		return loss

	def gradients(self, x):
		return self._gradients

evaluator = Evaluator()


x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128.

for i in range(ITERATIONS):
	x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
	print("Iteration %d completed with loss %d" % (i, loss))
    
x = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
x = x[:, :, ::-1]
x[:, :, 0] += IMAGENET_MEAN_RGB_VALUES[2]
x[:, :, 1] += IMAGENET_MEAN_RGB_VALUES[1]
x[:, :, 2] += IMAGENET_MEAN_RGB_VALUES[0]
x = np.clip(x, 0, 255).astype("uint8")
output_image = Image.fromarray(x)
output_image.save(output_image_path)
output_image.show()
end = time.time()
print((end - start)/60.0,'minutes')

combined = Image.new("RGB", (IMAGE_WIDTH*3, IMAGE_HEIGHT))
x_offset = 0
for image in map(Image.open, [input_image_path, style_image_path, output_image_path]):
    combined.paste(image, (x_offset, 0))
    x_offset += IMAGE_WIDTH
combined.save(combined_image_path)
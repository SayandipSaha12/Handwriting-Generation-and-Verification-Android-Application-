import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("handwriting_model_personalized.keras")
image_path = r"A image.jpeg"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

print(f"Original image shape: {img.shape}")
print(f"Original mean pixel value: {np.mean(img)}")

if np.mean(img) > 127:  
    img = cv2.bitwise_not(img)

_, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

coords = cv2.findNonZero(img_bin)
if coords is None:
    raise ValueError("No character detected in the image")

x, y, w, h = cv2.boundingRect(coords)

padding = max(4, int(max(w, h) * 0.1)) 
x = max(0, x - padding)
y = max(0, y - padding)
w = min(img_bin.shape[1] - x, w + 2 * padding)
h = min(img_bin.shape[0] - y, h + 2 * padding)

char_crop = img_bin[y:y+h, x:x+w]
max_dim = max(w, h)

canvas_size = int(max_dim * 1.2)  
square_img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

x_offset = (canvas_size - w) // 2
y_offset = (canvas_size - h) // 2
square_img[y_offset:y_offset+h, x_offset:x_offset+w] = char_crop

resized_img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)


img_normalized = resized_img.astype(np.float32) / 255.0

img_input = np.expand_dims(img_normalized, axis=(0, -1))

print(f"\nPreprocessed image stats:")
print(f"Shape: {img_input.shape}")
print(f"Min/Max values: {img_input.min():.3f} / {img_input.max():.3f}")
print(f"Mean value: {img_input.mean():.3f}")

print("\nMaking prediction...")
predictions = model.predict(img_input, verbose=0)
predicted_class = np.argmax(predictions)
confidence = predictions[0][predicted_class]

top_5_indices = np.argsort(predictions[0])[-5:][::-1]
letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]

print(f"\n{'='*50}")
print(f"PREDICTED LETTER: {letters[predicted_class].upper()}")
print(f"Confidence: {confidence*100:.2f}%")
print(f"\nTop 5 predictions:")
for idx in top_5_indices:
    print(f"  {letters[idx].upper()}: {predictions[0][idx]*100:.2f}%")
print(f"{'='*50}\n")

print("Generating visualization...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))


axes[0].imshow(cv2.imread(image_path), cmap='gray')
axes[0].set_title("1. Original Image", fontweight='bold')
axes[0].axis('off')


axes[1].imshow(img_bin, cmap='gray')
axes[1].set_title("2. Binary (No Blur)", fontweight='bold')
axes[1].axis('off')


axes[2].imshow(square_img, cmap='gray')
axes[2].set_title("3. Cropped & Centered", fontweight='bold')
axes[2].axis('off')


axes[3].imshow(resized_img, cmap='gray')
axes[3].set_title(f"4. Final 28x28\nPredicted: {letters[predicted_class].upper()} ({confidence*100:.1f}%)", 
                  fontweight='bold', color='green' if confidence > 0.7 else 'orange')
axes[3].axis('off')

plt.tight_layout()
plt.savefig("preprocessing_steps_no_blur.png", dpi=150)
print("Visualization saved as 'preprocessing_steps_no_blur.png'")
plt.show()

print("\nDone! Check the visualization window and 'preprocessing_steps_no_blur.png' file.")



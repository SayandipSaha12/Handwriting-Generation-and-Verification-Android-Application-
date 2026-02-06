import tensorflow as tf
import cv2
import numpy as np
import os
import re

print("="*70)
print("FINE-TUNING WITH YOUR HANDWRITING")
print("="*70)

# Load the pre-trained model
base_model = tf.keras.models.load_model("handwriting_emnist_model_improved.keras")
print("✓ Loaded base model")

your_letters_folder = r"C:\Final Year Project\my-letters"

# Load your handwritten samples
def load_your_handwriting(folder_path):
    """Load your handwritten letters"""
    images = []
    labels = []
    filenames_loaded = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Extract letter from filename
            # Handles: C1.jpg, C2.png, c_1.jpg, C-1.jpg, etc.
            
            # Remove file extension
            name_without_ext = os.path.splitext(filename)[0]
            
            # Extract the letter (first character that's a letter)
            letter_match = re.search(r'[a-zA-Z]', name_without_ext)
            
            if letter_match:
                letter = letter_match.group(0).lower()
                label = ord(letter) - ord('a')
                
                # Load and preprocess
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"⚠ Failed to load: {filename}")
                    continue
                
                # Preprocess (same as your testing code)
                if np.mean(img) > 127:
                    img = cv2.bitwise_not(img)
                
                _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                coords = cv2.findNonZero(img_bin)
                if coords is not None:
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
                    
                    resized = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
                    normalized = resized.astype(np.float32) / 255.0
                    
                    images.append(normalized)
                    labels.append(label)
                    filenames_loaded.append(f"{filename} -> {letter.upper()}")
                else:
                    print(f"⚠ No character detected in: {filename}")
            else:
                print(f"⚠ Could not extract letter from: {filename}")
    
    return np.array(images), np.array(labels), filenames_loaded

# Load your samples
if os.path.exists(your_letters_folder):
    X_custom, y_custom, loaded_files = load_your_handwriting(your_letters_folder)
    
    if len(X_custom) == 0:
        print("❌ No valid images found!")
        print(f"Check the folder: {your_letters_folder}")
        exit()
    
    print(f"\n✓ Successfully loaded {len(X_custom)} custom samples")
    
    # Show what was loaded
    print("\nLoaded files:")
    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    letter_counts = {}
    
    for i, label in enumerate(y_custom):
        letter = letters[label].upper()
        if letter not in letter_counts:
            letter_counts[letter] = 0
        letter_counts[letter] += 1
    
    print("\nSamples per letter:")
    for letter in sorted(letter_counts.keys()):
        print(f"  {letter}: {letter_counts[letter]} samples")
    
    # Verify you have all 9 problem letters
    problem_letters = ['C', 'D', 'F', 'G', 'J', 'P', 'R', 'S', 'Z']
    missing = [l for l in problem_letters if l not in letter_counts]
    
    if missing:
        print(f"\n⚠ WARNING: Missing letters: {', '.join(missing)}")
        print("Add these letters to improve the model!")
    else:
        print(f"\n✓ All 9 problem letters found!")
    
    # Add channel dimension
    X_custom = np.expand_dims(X_custom, axis=-1)
    
    # Show sample images
    import matplotlib.pyplot as plt
    
    print("\n4. Showing sample preprocessed images...")
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.ravel()
    
    sample_indices = np.random.choice(len(X_custom), min(15, len(X_custom)), replace=False)
    
    for idx, sample_idx in enumerate(sample_indices):
        axes[idx].imshow(X_custom[sample_idx].squeeze(), cmap='gray')
        letter = letters[y_custom[sample_idx]].upper()
        axes[idx].set_title(f"Label: {letter}", fontweight='bold')
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(len(sample_indices), 15):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Preprocessed Images from Your Handwriting', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('custom_samples_preview.png', dpi=150)
    print("✓ Preview saved as 'custom_samples_preview.png'")
    plt.show()
    
    # Fine-tune on your handwriting
    print("\n" + "="*70)
    print("FINE-TUNING MODEL")
    print("="*70)
    
    # Strategy: Freeze most layers, only train the last few
    print("\nFreezing early layers (keeping learned features)...")
    for layer in base_model.layers[:-4]:  # Freeze all except last 4 layers
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"✓ {trainable_count} layers will be trained")
    
    # Compile with lower learning rate (we're fine-tuning, not training from scratch)
    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train on your data
    print("\nTraining on your handwriting samples...")
    print(f"Epochs: 100 (with early stopping)")
    print(f"Batch size: 4")
    print(f"Validation split: 20%\n")
    
    history = base_model.fit(
        X_custom, y_custom,
        epochs=100,  # More epochs, but early stopping will prevent overfitting
        batch_size=4,  # Small batch size for small dataset
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the personalized model
    base_model.save("handwriting_model_personalized.keras")
    print("\n" + "="*70)
    print("✓ PERSONALIZED MODEL SAVED!")
    print("="*70)
    print("\nModel saved as: 'handwriting_model_personalized.keras'")
    print("\nTo use this model, update your prediction code:")
    print('  model = tf.keras.models.load_model("handwriting_model_personalized.keras")')
    
    # Plot training history
    print("\n5. Generating training history plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Fine-tuning Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Fine-tuning Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('finetuning_history.png', dpi=150)
    print("✓ Training history saved as 'finetuning_history.png'")
    plt.show()
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Test the personalized model on your handwriting")
    print("2. If accuracy is still low, add more samples (10-15 per letter)")
    print("3. Make sure samples are clear and well-lit")
    
else:
    print(f"❌ Folder not found: {your_letters_folder}")
    print("\nPlease create the folder and add your handwritten letters!")
    print("\nFile naming examples:")
    print("  C1.jpg, C2.png, C3.jpg  -> All recognized as 'C'")
    print("  D1.jpg, D2.jpg, D3.jpg  -> All recognized as 'D'")
    print("  F_1.png, F-2.jpg        -> All recognized as 'F'")
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("IMPROVED EMNIST HANDWRITING MODEL TRAINING")
print("="*60)


print("\nLoading EMNIST letters dataset...")
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

print(f"Training samples: {ds_info.splits['train'].num_examples}")
print(f"Test samples: {ds_info.splits['test'].num_examples}")


def normalize_and_augment(image, label):
    """Normalize and apply data augmentation"""
   
    image = tf.cast(image, tf.float32) / 255.0
    
   
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    
 
    image = tf.image.random_brightness(image, max_delta=0.1)
    

    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    

    image = tf.clip_by_value(image, 0.0, 1.0)
    
   
    return image, label - 1

def normalize_only(image, label):
    """Just normalize for test set (no augmentation)"""
    image = tf.cast(image, tf.float32) / 255.0
    return image, label - 1


print("\nPreparing datasets...")
ds_train = ds_train.map(normalize_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(10000)  
ds_train = ds_train.batch(128)  
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_test = ds_test.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


print("\nBuilding improved model...")
model = tf.keras.models.Sequential([
  
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                           input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
   
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(26, activation='softmax')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()


callbacks = [
   
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    ),
    
    
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    

    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]


print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

history = model.fit(
    ds_train,
    epochs=20, 
    validation_data=ds_test,
    callbacks=callbacks,
    verbose=1
)


model.save("handwriting_emnist_model_improved.keras")
print("\n✓ Model saved as 'handwriting_emnist_model_improved.keras'")

print("\nGenerating training plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)


ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("✓ Training history saved as 'training_history.png'")
plt.show()


print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)
test_loss, test_accuracy = model.evaluate(ds_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]


for images, labels in ds_test.take(1):
    predictions = model.predict(images[:10], verbose=0)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(10):
        axes[i].imshow(images[i].numpy().squeeze(), cmap='gray')
        
        true_label = letters[labels[i].numpy()]
        pred_label = letters[np.argmax(predictions[i])]
        confidence = predictions[i][np.argmax(predictions[i])] * 100
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f"True: {true_label.upper()}\nPred: {pred_label.upper()} ({confidence:.1f}%)",
                         color=color, fontweight='bold', fontsize=10)
        axes[i].axis('off')
        
        print(f"Sample {i+1}: True={true_label.upper()}, Predicted={pred_label.upper()} ({confidence:.1f}%)")
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    print("\n✓ Sample predictions saved as 'sample_predictions.png'")
    plt.show()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nKey improvements over your original model:")
print("1. ✓ Deeper architecture (more Conv layers)")
print("2. ✓ Batch Normalization (faster & more stable training)")
print("3. ✓ Dropout layers (prevents overfitting)")
print("4. ✓ Data augmentation (handles variations better)")
print("5. ✓ 20 epochs instead of 5 (much better learning)")
print("6. ✓ Learning rate scheduling (adapts during training)")
print("7. ✓ Early stopping (prevents overfitting)")
print("\nNow test with your handwriting using the improved model!")
print("Use: model = tf.keras.models.load_model('handwriting_emnist_model_improved.keras')")
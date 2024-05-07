import matplotlib.pyplot as plt
import tensorflow as tf

def class_image(train_data, class_names, cmap=None):
    class_images = {name: None for name in class_names}

    for image, label in train_data:
        label_index = label.numpy()
        if label_index.ndim > 0:  # Check if label_index is not a scalar
            label_index = label_index[0]  # Assuming batch size is 1

        class_name = class_names[label_index]
        if class_images[class_name] is None:
            # Convert the image to a numpy array
            class_images[class_name] = image.numpy()

        if all(image is not None for image in class_images.values()):
            break

    # Create a grid for subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))

    for (class_name, image), ax in zip(class_images.items(), axes.flatten()):
        if cmap is not None:
            ax.imshow(image, cmap=cmap)
            ax.set_title(class_name)
            ax.axis('off')
        else:
            ax.imshow(image)
            ax.set_title(class_name)
            ax.axis('off')

    plt.suptitle("One image from each class in the dataset.")
    plt.tight_layout()
    plt.show()


def load_image(dataset,cmap="gray"):
    for images,labels in dataset.take(1):
        img=images.numpy().astype("uint8")
        plt.imshow(img,cmap=cmap)

def average_image(train_data):
    sum_images = None
    count = 0

    for image, _ in train_data:
        if sum_images is None:
            sum_images = tf.zeros_like(image)
        sum_images += image
        count += 1

    average_image = sum_images / count

    plt.figure(figsize=(10, 10))
    plt.imshow(average_image, cmap="gray")
    plt.axis('off')  # Turn off the axis
    plt.title("Average Image of the Dataset")
    plt.show()

def average_images_per_class(train_data, class_names):
    # Dictionary to keep the sum of all images for each class and a count of images
    class_sums = {name: (tf.zeros_like(
        next(iter(train_data))[0]), 0) for name in class_names}

    # Iterate over the dataset and sum up the images tensor-wise for each class
    for image, label in train_data:
        class_name = class_names[label.numpy()]
        sum_images, count = class_sums[class_name]
        class_sums[class_name] = (sum_images + image, count + 1)

    class_averages = {name: (image_sum / count)
                    for name, (image_sum, count) in class_sums.items()}

    # Create a grid for subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))
    for ax, (class_name, avg_image) in zip(axes.flatten(), class_averages.items()):
        ax.imshow(avg_image, cmap = "gray")
        ax.set_title(f"Average image for {class_name}")
        ax.axis('off')  # Turn off axis

    plt.suptitle("Average of each class in the dataset")
    plt.tight_layout()
    plt.show()
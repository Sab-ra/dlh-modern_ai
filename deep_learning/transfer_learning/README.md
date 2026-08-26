# Transfer Learning for Computer Vision

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

- What is transfer learning in CNNs?
What ho! Transfer learning is essentially borrowing a clever chap’s brain for your own sticky wicket. Instead of teaching a network from a blank slate—which requires a mountain of data and a positive age to complete—you take a model already "schooled" on millions of images.
This pre-trained fellow already knows the basics, like edges and shapes, so you simply tuck your own specific task on top. You can either "freeze" his old knowledge to use as a feature extractor or give him a gentle "fine-tuning" to learn your specific preferences. It’s faster, much simpler, and an absolute lifesaver when your own pile of data is a bit thin. Pip-pip!
P.S. A CNN (Convolutional Neural Network) is a jolly clever stack of layers that learns to "see" by breaking images down. It starts with simple bits like edges and blobs in the early stages, gradually getting more specific until it can distinguish a butler from a bicycle. It’s the absolute engine of modern vision. Simply topping! Pip-pip!
- What is feature extraction vs fine-tuning?
What ho! Feature extraction is like hiring a butler who already knows how to polish silver and fold a cravat, and simply telling him which drawer to use in your house. You take a pre-trained model, "freeze" its internal weights so they don't change, and use its existing knowledge to turn images into a list of useful features. You then train a new classification head on top of those fixed features. It is a jolly fast and simple way to work, especially when your own collection of data is a bit thin.
Fine-tuning, on the other hand, is like giving that same butler a bit of extra coaching to suit your peculiar household needs. Instead of keeping the model’s brain completely locked, you unfreeze some or all of the pre-trained layers and let them "learn" a bit more from your specific images. This allows the model to adapt its universal wisdom to the fine details of your new task. We usually do this with a very low learning rate so we don't accidentally wreck the good features the model already had. Feature extraction is the starting point for speed, while fine-tuning is the optional second step for that extra bit of top-hole accuracy. Simply topping! Pip-pip!
- How to choose a pretrained model?
What ho! Choosing a pre-trained model is a bit like picking a horse for a race—you want one that’s already run on this sort of turf. It all boils down to two things: the size of your new dataset and how much your images look like the original ones the model learned from, typically the ImageNet crowd.
The rules of thumb are quite simple:
  - Small and Similar Data: If you have very little data and it looks like the original set, don't confuse the poor thing by retraining. Just freeze the base and use it to extract features.
  - Large and Similar Data: If you have a mountain of data that’s similar, you can confidently fine-tune the whole works to get that extra bit of polish.
  - Small and Different Data: If your data is small but quite different—like looking at tiny cells through a microscope—the later layers are too specialized. It’s best to extract features from the earlier, more generic layers.
  - Large and Different Data: Here, you can afford to fine-tune everything since you have enough data to prevent it from losing its way.
For the models themselves, think of MobileNet as your lightweight friend for speed, while ResNet or EfficientNet are the heavy lifters when you need top-tier accuracy. Simply topping! Pip-pip!
- What features do pretrained CNNs learn?
What ho! Pretrained CNN chaps are quite methodical, learning to see the world in a specific order through their various layers. The early layers are the basic sorts, picking up generic things like edges, blobs of color, and simple textures that you would find almost anywhere. As you move deeper into the network, it gets a bit more high-hat and specific, focusing on complex details like parts of objects or the peculiar differences between various categories. By the time you reach the final layers, the model has a very specialized view tailored to the original task it was schooled on. It is a hierarchy of vision, starting from the simple building blocks and ending with the top-hole details that make a butler distinct from a bicycle. Pip-pip!
- How does dataset size affect transfer learning?
What ho! The size of your dataset is the absolute decider for your strategy. If you have a small pile of images, you must be very careful not to overfit. In these cases, it is best to keep the pretrained model frozen and simply use it as a feature extractor, as you lack enough data to retrain those complex layers without making a mess of things. However, if you are lucky enough to have a mountain of data, you can be much bolder. With plenty of images, you have the confidence to unfreeze the model and fine-tune the whole works, letting it adapt its knowledge to your specific task. Even with a very large dataset, starting with those pretrained weights gives you a head start compared to beginning with random guesses. For the really tiny sets, using data augmentation to shake things up is a top-hole way to keep the model from just memorizing your few pictures. Pip-pip!
- Why are early CNN layers general and later layers task-specific?
What ho! To put it simply, early CNN layers are like the absolute basics of sketching—learning how to draw a simple line or a splash of color. These generic building blocks are useful for almost any image you might encounter in the wild. However, as the data travels deeper into the network, it starts combining those simple bits into much more complex and peculiar shapes. By the time you reach the final layers, the model has become a complete specialist, focusing on the very specific details of the objects it was originally schooled to recognize, such as the exact curve of a prize-winning hound's ear. It is a progression from the universal alphabet of vision to the highly specialized vocabulary of a single task. Pip-pip!
- How does source-target dataset similarity affect performance?
What ho! Dataset similarity is the absolute lynchpin of this whole transfer learning business. If your new images are quite like the original ones the model was schooled on—say, identifying different sorts of motorcars using a model trained on vehicles—the model’s high-level knowledge is a perfect fit. You can simply use its sophisticated, later-stage features to get top-hole results without much fuss.
However, if your data is a completely different kettle of fish—like using a model trained on garden birds to identify microscopic cells—those later, specialized layers become a bit of a hindrance. In that case, the model’s specific knowledge is less helpful, and you'd want to reach back and borrow the more generic features from earlier in the network, like basic edges and blobs. Essentially, the closer the similarity, the more of the original "brain" you can use directly to achieve high performance. The more they differ, the more you must rely on those universal building blocks from the early layers to avoid a total muddle. Simply topping! Pip-pip!
- What are common pretrained CNN architectures?
There is a regular club of these pre-trained chaps available for your use, and they each have their own particular strengths. For the heavy lifting, you have the **ResNet** family, including **ResNet50** and its even deeper cousins, which are absolute legends for achieving top-tier accuracy. Then there are the **VGG** fellows, like **VGG16** and **VGG19**, who are quite straightforward and reliable, though perhaps a bit on the portly side when it comes to memory size.
If you are in a dash and need something lightweight _for a mobile gadget_, **MobileNet** is your best friend. For the absolute cutting edge of efficiency, you might look at the **EfficientNet** or **Xception** models, which are jolly clever at balancing performance with resource use. You also have the **Inception** and **DenseNet** crowds hanging about, ready to tackle complex visual deductions. It is a top-hole selection that ensures you never have to start your learning from scratch. Simply topping!

- Which layers should be frozen or unfrozen?
What ho! Deciding which layers to freeze is a bit like deciding which parts of a butler’s training to keep. Generally, you want to **freeze the early layers** of the network because they’ve learned generic things like edges and blobs that are useful for any task. You keep them locked so you don't destroy **that universal wisdom**.
The later layers, being much more specialized, are the ones you might consider unfreezing—but only after you’ve already trained your new classification head to a decent standard. If your dataset is small, keep the whole base frozen to avoid a total muddle called overfitting. _If you have a mountain of data, you can afford to unfreeze those top layers_ and give them a gentle "fine-tuning" with a very low learning rate. It’s all about protecting the basics while polishing the specifics. Simply topping! Pip-pip!

- How does feature reuse help small datasets?
What ho! Feature reuse is an absolute lifesaver when you're working with a rather meager collection of data. You see, training a complex network from a blank slate is a real sticky wicket because it requires a positive mountain of images just to learn the basics, like what an edge or a splash of color looks like. By borrowing features from a model that has already been properly schooled on millions of pictures, you are essentially handing your small dataset a finished pair of spectacles. The machine already understands the universal building blocks of vision, so it does not need to see ten thousand bicycles to recognize one in your specific task. It can take those pre-learned concepts and apply them immediately, which prevents it from making a complete muddle of things by simply memorizing your few examples. It is like hiring a top-hole butler who already knows his way around a silver tray; he just needs to learn where you keep the tea. Simply topping! Pip-pip!
- Why fine-tune only part of the network?
What ho! One only tinkers with a portion of the network to keep things from getting into a frightful muddle. You see, the early layers have already mastered the basics—like edges and blobs—and those are useful for just about any visual task you might encounter. There is simply no need to retrain a fellow on how to see a straight line! The later layers, however, are the specialized sorts, and giving them a bit of a fine-tune helps them adjust to the peculiar details of your new task. More importantly, if you try to retrain the whole thing on a small pile of data, you risk a catastrophe called overfitting, where the model essentially just memorizes your pictures instead of learning. By keeping most of the network frozen and only nudging the top parts, you protect the established wisdom of the model. If you unfreeze everything too early, those new, randomly-initialized layers can send massive gradient updates that absolutely wreck the fine pre-trained features you borrowed in the first place. It is all about preserving the good while polishing the specific. Simply topping! Pip-pip!
- Overfitting relate to transfer learning, what is that?
What ho! Overfitting is what happens when a machine gets a bit too big for its boots and becomes rather obsessed with the specific details of its training data. Instead of learning the general rules—like what makes a bicycle look like a bicycle—it starts memorizing every tiny, irrelevant speck of dust in the photos you've given it. This means it performs splendidly on the pictures it has already seen but becomes a total cabbage when faced with a new image it hasn't met before.
In the world of transfer learning, this is a particular worry when your new collection of data is a bit on the thin side. If you take a massive, brainy model and try to retrain it on just a few snapshots, the fellow will simply memorize those specific images rather than adapting its knowledge. To prevent this sticky wicket, we often "freeze" the model’s weights to stop them from changing too much or use a very low learning rate. We also use data augmentation to shake things up so the model doesn't just learn the images by heart. It’s all about keeping the fellow focused on the big picture! Pip-pip!
- Data augmentation help transfer learning, how come?
What ho! Data augmentation is a jolly clever trick to help a model when your collection of images is a bit on the thin side. By taking your existing pictures and giving them a bit of a shake—flipping them horizontally, rotating them slightly, or zooming in—you are essentially creating "new" data for the machine to study. It’s like showing a guest the same prize-winning hound from different angles so they don't get confused if the dog decides to turn around!
This artificial diversity is a top-hole way to slow down overfitting, ensuring the model doesn't just memorize your small pile of snapshots. Since transfer learning is most often used when data is limited, these random yet realistic transformations help the model see different aspects of the same information. It teaches the fellow to recognize the essence of an object regardless of how it's positioned, which makes for a much brainier and more flexible deduction in the end. Simply topping! Pip-pip!

## General Task Requirements

- All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.11)
- All your files should end with a new line
- The first line of all your files should be exactly #!/usr/bin/env python3
- A README.md file, at the root of the folder of the project, is mandatory
- Your code should use the pycodestyle style (version 2.14.0)
- All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
- All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
- All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
- All your files must be executable
- The length of your files will be tested using wc
- Packages:
  - numpy	2.0.2
  - pandas	2.2.2
  - Tensorflow	2.18.0
  - Matplotlib	3.10.0

## 0-frozen_extractor.py

### 0. Frozen Feature Extractor

Write a function `build_feature_extractor()` that loads a pretrained CNN model (e.g., MobileNetV2) from Keras applications, removes its classification head, and freezes its weights.

The function should:
- Load MobileNetV2 with weights="imagenet", input_shape=(224, 224, 3) and without it's classification head
- Freeze the base model
- Add a GlobalAveragePooling2D layer on top
- Return a Keras Model that outputs features from input images using the frozen base model.

## 1-classification_head.py

### 1. Classification Head

Write a function `add_classification_head(base_model, num_classes)` that attaches a custom classification head to a pretrained feature extractor.

Arguments:
- base_model: A Keras Model whose output is a pooled feature vector.
- num_classes: An integer representing the number of output classes.

The head should:
- Take the output of the base model
- Add a dense layers with 128 filters and relu activation
- Add a final classification layer
- Return a new Keras Model ready for classification.

## 2-unfreeze_top.py

### 2. Unfreezing Layers

Write a function `unfreeze_top_layers(model, n_layers)` that unfreezes the last n_layers of the base model inside a transfer learning pipeline, and leaves the rest frozen.

Arguments:
- model: A full Keras Model with a base model as its second layer.
- n_layers: Integer specifying how many of the last layers in the base model should be unfrozen (set as trainable).

The function should:
- Assume the base model is the first layer of the input model.
- Unfreeze the last n_layers of the base model.
- Leave earlier layers frozen.
- Return None

## 3-data_aug.py

### 3. Data Augmentation

Write a function `build_data_augmentation()` that creates a Keras Sequential model containing common image data augmentation operations. This augmentation will be applied to training images before they are passed into the pretrained CNN.

The function should:
- Create a tf.keras.Sequential model

Add the following augmentation layers:
- RandomFlip("horizontal")
- RandomRotation(0.15)
- RandomZoom(0.15)
- RandomContrast(0.1)

- Return the Sequential augmentation model

Use Keras preprocessing layers from tf.keras.layers And make sure all layers are seeded with value 42 to ensure reproducibility during training and testing.

## 4-transfer_101.py

### 4. Knowledge Transfer: Taming the 101

Write a function `def train_transfer_model():` that builds, trains, and saves an image classifier using transfer learning on the Stanford Cars dataset.

Your final model should be able to classify images into one of 102 categories (101 object classes + background) with a validation accuracy of at least 85%.

The pipeline should:
- Load a pretrained CNN from Keras Applications with include_top=False as a feature extractor.
- Prepare and preprocess the dataset appropriately:
- Apply common data augmentation techniques (e.g., rotation, zoom, flips, etc.).
- Use the model-compatible preprocessing function (e.g., keras.applications.<Model>.preprocess_input).

Structure your training in two phases:
- Train a custom classification head while keeping the base model frozen.
- Then, unfreeze and fine-tune some top layers of the base model for better performance.

Save the trained model to a file named: caltech101_model.h5.

Output:

- A trained model saved as caltech101_model.h5
- The model should achieve ≥85% validation accuracy

Tips:

- You can use keras.utils.image_dataset_from_directory or ImageDataGenerator to load the images.
- Feel free to try different pretrained models and compare results.
- You can explore tuning the number of unfrozen layers, learning rates, optimizers, batch sizes, etc.
- For efficient training, use callbacks like EarlyStopping, ReduceLROnPlateau, or ModelCheckpoint.

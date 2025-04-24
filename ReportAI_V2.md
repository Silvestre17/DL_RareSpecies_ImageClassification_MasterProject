Okay, I have reviewed the draft report content, your code structure (`utilities.py`, notebooks), figures, and tables. I've refined the text to improve clarity, conciseness, and flow, ensuring it aligns with the academic yet simple style requested and mirrors the structure of your previous reports. I've also integrated the figures and tables logically and added the requested Annex A comparing the pre-trained models.

Here are the revised sections, followed by Annex A. I've highlighted key changes or areas needing your input (like final results) in **[bold brackets]**.

**Summary of Major Changes Made:**

1.  **Clarity & Justification:** Added clearer explanations and justifications for choices (e.g., data split strategy, `crop_to_aspect_ratio=False`, metric selection, optimizer choice, SMOTE via augmentation process, transfer learning rationale, fine-tuning strategy, CLIP motivation).
2.  **Integration:** Better integrated figure/table references into the text flow. Updated figure captions where placeholders existed (you'll still need to finalize some based on exact content).
3.  **Structure:** Improved paragraph flow and section transitions, mirroring the structure of your example reports.
4.  **Completeness:** Filled in gaps based on the provided code and common practices (e.g., describing the baseline architecture layers, explaining callbacks, detailing pre-trained model preprocessing). Added **placeholders** for results you need to insert from your final runs (fine-tuning metrics, CLIP filtering outcome).
5.  **Conciseness:** Removed some redundancy while retaining essential information.
6.  **Annex A:** Created the requested Annex detailing the pre-trained models used.
7.  **Code References:** Added implicit references to Keras/TensorFlow documentation where specific functions/layers are discussed.

---

**Revised Report Draft:**

**(Start with Title Page, Table of Contents - as provided in your draft)**

**1. INTRODUCTION**

*(This section seems largely okay, minor wording adjustment for flow)*

The identification of rare species is essential for biodiversity conservation, yet it often relies on time-consuming expert analysis. Deep Learning (DL), particularly Convolutional Neural Networks (CNNs), provides a powerful automated approach to classify species directly from images, tackling challenges like visual similarity and data scarcity [1]. This project leverages DL to develop a model capable of classifying rare species into their respective families using image data. The images and associated metadata (kingdom, phylum, family) are sourced from the Encyclopedia of Life (EOL) via the BioCLIP dataset [2], providing a rich foundation for training and evaluating a family-level species classification system.

Recent research consistently demonstrates the effectiveness of CNN architectures, often enhanced through transfer learning with pre-trained models like ResNet, EfficientNet, VGG, and others, achieving high accuracy across diverse taxa including birds, mammals, reptiles, and marine life [3, 4, 6, 9, 11, 16]. Techniques such as data augmentation [5, 7, 11, 14] and specialized models like YOLO or R-CNN [8, 10] are frequently used to handle variations in image quality, pose, and background, especially in demanding scenarios like camera trap imagery [9, 16, 19, 21]. This body of work confirms that DL offers reliable tools for ecological monitoring and conservation efforts, forming the basis for the methods explored in this project.

This report details the development of our species classification model. **Section 2** describes the initial data exploration and the image preprocessing steps applied to prepare the dataset. **Section 3** outlines the modelling approach chosen, the experimental setup, and provides an analysis of the model's performance using appropriate metrics. **Section 4** discusses an innovative aspect explored and implemented during the project. Finally, **Section 5** concludes the report, summarizing principal conclusions, acknowledging limitations, and proposing potential directions for future work.

**2. DATA EXPLORATION AND IMAGE PREPROCESSING**

**2.1. Initial Data Exploration**

An initial exploration of the provided metadata CSV file using the pandas library confirmed the dataset comprises 11,983 observations and 7 features. No missing values or duplicate rows were detected. The data types appeared appropriate for each feature's nature.

Analysis revealed that all images belong to the `kingdom` Animalia; consequently, this feature was dropped as it provides no discriminatory value for classification. The `phylum` feature contains 5 distinct values, dominated by 'Chordata' (83.1%), as shown in Figure B1. A check confirmed the dataset adheres to taxonomic hierarchy, with each family belonging to only one phylum. The target variable, `family`, includes 202 distinct classes, exhibiting significant imbalance (Figure B2), with frequencies ranging from 29 to 300 samples per class. Visual inspection of sample images (Figure B3) also highlighted the presence of potential outliers – images containing specimen labels or habitat views rather than the animal itself – which could impact model training.

**(Figure B1 – Barplot for phylum distribution.)**
*(Insert Figure B1 here)*

**(Figure B2 – Barplot for family distribution (Left: All families, Right: Top 50 families).)**
*(Insert Figure B2 here)*

**(Figure B3 – Sample images from the dataset, illustrating diversity across phyla and potential non-animal images.)**
*(Insert Figure B3 here)*

Following common practices in the literature [4, 6, 8, 14, 15], the dataset was divided into training (80%), validation (10%), and test (10%) sets. Stratification by `family` was used during splitting to ensure the class distribution remained consistent across all sets, which is crucial given the imbalance. The training data was shuffled before splitting. For operational efficiency, the image files were physically copied into corresponding `train`, `validation`, and `test` subdirectories, maintaining the original `phylum_family` folder structure. This facilitated easier data loading using Keras utilities. The hold-out method was employed, reserving the validation set for model tuning and the test set for final, unbiased evaluation. K-fold cross-validation was avoided due to the high computational cost associated with repeatedly training deep learning models.

To address the class imbalance, undersampling was discarded due to the risk of significant information loss. Instead, we opted to explore oversampling (via data augmentation) and class weighting techniques during modelling.

**2.2. Image Import & Preprocessing**

The Keras library [25] within TensorFlow [26] was central to data loading, preprocessing, and modelling. Images were loaded using the `image_dataset_from_directory` function, configured as follows:

*   **Image Size:** `(224, 224)` pixels, aligning with standard input sizes for many pre-trained models [24] and common literature practices [6, 9, 11, 14, 17, 18, 20].
*   **Color Mode:** `rgb` (3 channels) was used to retain full color information.
*   **Aspect Ratio Handling:** `crop_to_aspect_ratio=False` and `pad_to_aspect_ratio=False`. Cropping risks removing vital features, while padding introduces potentially misleading artifacts (black bars), as illustrated in Figure B4. Simple resizing, despite potential distortion, was preferred to preserve all original pixel information.
*   **Labels:** `label_mode='categorical'` generated one-hot encoded labels for compatibility with categorical cross-entropy loss and class weighting. `labels='inferred'` utilized the subdirectory names (family names) as labels.
*   **Shuffling:** `shuffle=True` applied only to the training set import, crucial for breaking data order and improving model generalization during batch processing. Validation and test sets used `shuffle=False`.
*   **Batch Size:** `64`, determined as the largest feasible size given hardware constraints.
*   **Interpolation:** `interpolation='bilinear'` provided a balance between resizing quality and computational cost.

**(Figure B4 – Sample images from Training set demonstrating resize behavior.)**
*(Top: `crop_to_aspect_ratio=False` & `pad_to_aspect_ratio=False` | Middle: `crop_to_aspect_ratio=True` | Bottom: `pad_to_aspect_ratio=True`)*
*(Insert Figure B4 here)*

Initial visual exploration of various `tf.image` transformations (e.g., Figure B5) led to the decision to test the effects of grayscale conversion, contrast adjustment, and saturation adjustment individually during the baseline model phase.

**(Figure B5 – Examples of potential tf.image preprocessing and augmentation effects.)** *(Caption: Examples of transformations available in tf.image.)*
*(Insert Figure B5 here)*

**2.3. Addressing Imbalance Issues**

Two strategies were tested to mitigate class imbalance:

1.  **Oversampling via Data Augmentation ("SMOTE"):** As traditional SMOTE is not directly applicable to images, an analogous approach was implemented. A copy of the training set was created. For each minority class (count < 240), new images were generated by applying Keras's `RandAugment` layer [33] to randomly selected original images until the class size matched the majority. `RandAugment` introduces diversity by applying random sequences of augmentations (examples in Figure B6) [32]. The augmented images were saved to disk, ensuring reproducibility across experiments. This method synthetically increases minority class representation with visually plausible variations.
2.  **Class Weighting:** The `class_weight` argument in `model.fit` was tested. Weights were computed inversely proportional to class frequency (`weight = total_samples / (num_classes * class_samples)`), assigning higher importance in the loss calculation to errors made on rarer classes, thereby encouraging the model to learn their features more effectively.

**(Figure B6 – Examples of image augmentations applied by RandAugment.)** *(Caption: Examples of transformations available in Keras data augmentation layers like RandAugment.)*
*(Insert Figure B6 here)*

**3. MODELLING & EVALUATION**

The modelling workflow is depicted in Figure 3.1.

**(Figure 3.1 – Model Process Flowchart.)**
*(Insert Flowchart Image here)*

**3.1. Baseline Model – Combination Selection**

A baseline CNN (Figure C1) was constructed using the Keras functional API. Key components included:
*   `Rescaling` layer (to [0, 1]).
*   Four blocks of `Conv2D` (filters 32, 64, 128, 256, kernel size 3x3, 'relu' activation), each followed by `BatchNormalization` (for stable training) and `MaxPooling2D` (for spatial downsampling).
*   `GlobalAveragePooling2D` (to reduce feature map dimensions).
*   A `Dense` layer (128 units, 'relu'), `BatchNormalization`.
*   `Dropout` (rate 0.5) for regularization.
*   Final `Dense` output layer (202 units, 'softmax' activation).

**(Figure C1 – Baseline CNN Model Architecture.)** *(Caption: Architecture of the baseline Convolutional Neural Network.)*
*(Insert Figure C1 here)*

This model served to evaluate the initial combinations of preprocessing (Original, Grayscale, Contrast, Saturation) and imbalance handling (Original, SMOTE, Weights) over a limited 10 epochs (Table C1). The primary evaluation metric was Macro F1-Score, chosen for its sensitivity to performance across all classes, especially crucial for the rare families in this imbalanced dataset. Accuracy, Macro Precision, Macro Recall, and AUROC were also recorded. The `Adam` optimizer [34] (learning_rate=0.001) was selected for its efficiency, and `CategoricalCrossentropy` loss was used due to the one-hot encoded labels. Callbacks included `ModelCheckpoint` (saving best based on `val_loss`), `CSVLogger`, `LearningRateScheduler` (0.95 decay per epoch), and `EarlyStopping` (patience=3 on `val_loss`).

Initial results (Table C1) indicated that combinations using Grayscale, Class Weights, or Saturation-only preprocessing yielded significantly lower performance, particularly in F1 score. Grayscale likely discards vital color features, while class weights might require more extensive tuning or epochs. These combinations were thus excluded.

**(Table C1 - Baseline Model Combinations Selection Results (Max 10 Epochs).)**
*(Insert Table C1 here)*

The top 5 combinations (Original, Original+Contrast, SMOTE, SMOTE+Contrast, SMOTE+Saturation) were selected for further testing with pre-trained models, allowing for longer training runs.

**3.2. Pretrained Models**

Transfer learning was employed using four established CNN architectures, pre-trained on ImageNet, to leverage their learned feature extraction capabilities: VGG19 [28], ResNet152V2 [29], ConvNeXtBase [30], and EfficientNetV2B0 [31]. These models were chosen based on literature review, strong performance benchmarks, and compatibility with the (224, 224) input size. Annex A provides further details on each architecture. Critically, each model was used with its specific `preprocess_input` function (from `tensorflow.keras.applications`) to ensure correct normalization and formatting, replacing the baseline's simple rescaling layer.

**3.3. Evaluation Models**

The five selected preprocessing/imbalance combinations were tested with each of the four pre-trained models. The base model layers were initially frozen, and a classification head similar to the baseline's was added. Models were trained for up to 100 epochs using the same optimizer, loss, metrics, and callbacks (monitoring `val_f1_score` for `ModelCheckpoint` and `val_loss` for `EarlyStopping`).

Results are shown in Table C2. The **ConvNeXtBase model paired with the SMOTE (augmentation) strategy and no additional color preprocessing (Contrast/Saturation)** demonstrated the best overall performance. It achieved the highest Validation F1 Score (0.804) and Test F1 Score (0.829), while maintaining acceptable overfitting (Train F1 0.886 vs Test F1 0.829). This combination outperformed the baseline and other pre-trained models, highlighting the power of modern architectures like ConvNeXt and the benefit of the SMOTE-augmentation for handling imbalance in this context. The architecture-specific preprocessing proved sufficient without further manual color adjustments.

**(Table C2 - Baseline & Pretrained Model Combinations Selection Results (Max 100 Epochs).)**
*(Insert Table C2 here)*

**3.4. Fine-tuning**

The best performing model (ConvNeXtBase + SMOTE) was selected for fine-tuning using Keras Tuner's `RandomSearch` [36] (see Annex B). The search space (Table D1) included:
*   `unfreeze_base`: Boolean (True/False) to unfreeze the final layers of ConvNeXtBase (specifically, the last 22 layers, keeping Batch Normalization frozen).
*   `learning_rate`: Choices [1e-3, 1e-4, 1e-5] (lower rates often preferred for fine-tuning).
*   `optimizer`: Choices ['adam', 'sgd', 'rmsprop'].
*   `dropout_rate`: Float range [0.4, 0.7].

**(Table D1 - Hyper-parameters used on RandomSearch.)**
*(Insert Table D1 here)*

The tuner optimized for `val_f1_score` over 10 trials, each running up to 20 epochs with early stopping (patience=5 on `val_loss`). The best hyperparameters found were: **[Insert Best Parameters found by Keras Tuner, e.g., unfreeze_base=True, learning_rate=1e-5, optimizer='adam', dropout_rate=0.5]**.

The model was then retrained using these optimal hyperparameters for up to 100 epochs with refined callbacks (`EarlyStopping` patience=10, `ReduceLROnPlateau` patience=5 on `val_loss`, `ModelCheckpoint` on `val_f1_score`). This final fine-tuned model achieved a **Test Accuracy of [Insert Final Test Accuracy]** and a **Test Macro F1-Score of [Insert Final Test F1 Score]**.

**4. INNOVATIVE APPROACH**

As noted during data exploration, a portion of the dataset images did not depict animals, potentially acting as noise. To address this, a Zero-Shot Image Classification step using CLIP (Contrastive Language–Image Pre-training) [38] was implemented. The `clip-vit-base-patch16` model [39] from Hugging Face [40] was used to classify each image as either "photo of an animal" or "photo of something else" without requiring explicit training on these labels. This step was performed in a separate PyTorch environment due to library conflicts.

Figure E1 shows examples of CLIP's classification. Figure E2 illustrates the distribution of classifications across the datasets, revealing that approximately 15% of images were identified as "not animal", consistently across train, validation, and test splits.

**(Figure E1 – Examples of classification using the CLIP model (“animal” vs. “not animal”).)**
*(Insert Figure E1 here)*

**(Figure E2 – Distribution of CLIP classification results (Animal vs. Not Animal) in train, validation and test datasets.)**
*(Insert Figure E2 here)*

**[State whether the filtering and retraining step was completed OR state it was identified as future work]:**
*   **Option A (If Completed):** Following the CLIP classification, the images identified as "not animal" were removed from the train, validation, and test sets. The SMOTE augmentation process was reapplied to the filtered training data to re-balance the classes based only on the remaining animal images. The best fine-tuned ConvNeXtBase model was then retrained on this cleaned and re-balanced dataset. The performance on the filtered test set resulted in **[Insert Test Accuracy on Filtered Data]** Accuracy and **[Insert Test Macro F1 on Filtered Data]** Macro F1-Score. *(Compare briefly to the unfiltered result if applicable)*.
*   **Option B (If Not Completed):** Following the CLIP classification, the next logical step would be to filter out the "not animal" images and retrain the best model on this cleaned dataset (potentially after re-applying SMOTE). However, due to time and computational constraints, this retraining and evaluation phase was not completed as part of this project but is identified as significant future work.

**5. CONCLUSION**

This project successfully applied deep learning techniques to classify rare species into families using the BioCLIP dataset. Initial data exploration revealed significant class imbalance and the presence of non-animal images. Preprocessing involved standardized image resizing and addressing imbalance through augmentation-based oversampling (SMOTE) and class weighting experiments.

A baseline CNN and four pre-trained models (VGG19, ResNet152V2, ConvNeXtBase, EfficientNetV2B0) were evaluated across various configurations. The ConvNeXtBase architecture, combined with SMOTE augmentation and its native `preprocess_input` layer, yielded the best performance based on the Macro F1-Score metric. Fine-tuning this model using `RandomSearch` to optimize learning rate, optimizer choice, dropout rate, and layer unfreezing resulted in a final model achieving **[Insert Final Test F1 Score]** Macro F1-Score and **[Insert Final Test Accuracy]** Accuracy on the test set.

An innovative approach using CLIP for zero-shot classification successfully identified ~15% of images as likely non-animal subjects. **[State outcome: e.g., Retraining on the filtered dataset yielded slightly improved/comparable/worse results OR This filtering provides a clear path for future improvement].**

Limitations include the reliance on a hold-out set instead of k-fold cross-validation, potential dataset limitations for extremely rare families, and computational constraints affecting the scope of hyperparameter tuning.

Future work could involve implementing k-fold cross-validation, exploring more advanced augmentation or generative techniques for oversampling, performing a more exhaustive hyperparameter search (e.g., using `Hyperband`), **[If not completed: fully evaluating the impact of CLIP-based filtering]**, incorporating metadata features, and potentially augmenting the dataset via web scraping or other sources.

Overall, this study confirms the effectiveness of transfer learning with modern CNN architectures like ConvNeXtBase for challenging fine-grained visual classification tasks like rare species identification, highlighting the importance of appropriate preprocessing, imbalance handling, and fine-tuning.

---

**BIBLIOGRAPHICAL REFERENCES**

*(List references [1] through [40] as provided in your draft, ensuring APA 7 format)*

---

**APPENDICES**

*(Include Appendices A, B, C, D, E as generated/provided)*

---

**ANNEXES**

**Annex A: Pretrained Model Details**

The following pre-trained models from Keras Applications [24], all using weights trained on ImageNet and expecting a (224, 224, 3) input size, were used in this project:

| Model           | Description & Key Idea                                                                 | Architecture Highlights                                                                       | `preprocess_input` Function Behavior [24]                                                                 | Reference |
| :-------------- | :------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- | :-------- |
| **VGG19**       | Very deep network using only small (3x3) convolutional filters stacked sequentially.     | Sequential blocks of Conv3x3 layers followed by Max Pooling. Simple but deep structure.        | Converts RGB to BGR, then zero-centers with respect to ImageNet means \[103.939, 116.779, 123.68]. No scaling. | [28]      |
| **ResNet152V2** | Deep Residual Network using identity shortcut connections to ease training very deep models. V2 improves on the original ResNet block structure. | Uses "pre-activation" residual blocks (BN-ReLU-Conv). Extremely deep (152 layers).            | Scales pixels between -1 and 1.                                                                           | [29]      |
| **ConvNeXtBase**| Modern CNN architecture inspired by Vision Transformers (ViTs), aiming for improved performance and scalability. | Uses concepts like larger kernel sizes, depthwise separable convolutions, and layer normalization in stages. | Scales pixels between 0 and 1, then normalizes using ImageNet mean and standard deviation.                  | [30]      |
| **EfficientNetV2B0**| Model family designed by scaling network depth, width, and resolution using a compound coefficient. V2 offers improved training speed and parameter efficiency over V1. B0 is the smallest baseline model. | Uses MBConv blocks (inverted residuals with squeeze-and-excitation). Compound scaling.       | Scales pixels between 0 and 1, then normalizes using ImageNet mean and standard deviation.                  | [31]      |

**Annex B: GridSearch VS RandomSearch VS Hyperband Keras Tuner**

*(Insert the previously generated Annex B table and conclusion here)*

---
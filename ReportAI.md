Okay, here is the full draft of your Deep Learning report, integrating the provided text, images, tables, and following the structure and style requested. Please review it carefully, especially the placeholders and areas marked for your specific results or final decisions.

***

**Predicting Rare Species from Images using Deep Learning**

**Group 37**
André Silvestre, 20240502
Diogo Duarte, 20240525
Filipa Pereira, 20240509
Maria Cruz, 20230760
Umeima Mahomed, 20240543

**Master in Data Science and Advanced Analytics**
**NOVA Information Management School**
**Universidade Nova de Lisboa**

**Fall/Spring Semester 2024-2025**

---

**TABLE OF CONTENTS**

1.  Introduction
2.  Data Exploration and Image Preprocessing
    2.1. Initial Data Exploration
    2.2. Image Import & Preprocessing
    2.3. Addressing Imbalance Issues
3.  Modelling & Evaluation
    3.1. Baseline Model – Combination Selection
    3.2. Pretrained Models
    3.3. Evaluation Models
    3.4. Fine-tuning
4.  Innovative Approach
5.  Conclusion
Bibliographical References
Appendix A. Literature Review
Appendix B. EDA
Appendix C. Modelling
Appendix D. Hyper-parameter RandomSearch
Appendix E. Innovative Approach
Annex A. Pretrained Models
Annex B. GridSearch VS RandomSearch VS Hyperband Keras Tuner

---

**1. INTRODUCTION**

The identification of rare species is essential for biodiversity conservation, yet it often relies on time-consuming expert analysis. Deep Learning (DL), particularly Convolutional Neural Networks (CNNs), provides a powerful automated approach to classify species directly from images, tackling challenges like visual similarity and data scarcity [1]. This project leverages DL to develop a model capable of classifying rare species into their respective families using image data. The images and associated metadata (kingdom, phylum, family) are sourced from the Encyclopedia of Life (EOL) via the BioCLIP dataset [2], providing a rich foundation for training and evaluating a family-level species classification system.

Recent research consistently demonstrates the effectiveness of CNN architectures, often enhanced through transfer learning with pre-trained models like ResNet, EfficientNet, VGG, and others, achieving high accuracy across diverse taxa including birds, mammals, reptiles, and marine life [3, 4, 6, 9, 11, 16]. Techniques such as data augmentation [5, 7, 11, 14] and specialized models like YOLO or R-CNN [8, 10] are frequently used to handle variations in image quality, pose, and background, especially in demanding scenarios like camera trap imagery [9, 16, 19, 21]. This body of work confirms that DL offers reliable tools for ecological monitoring and conservation efforts, forming the basis for the methods explored in this project.

This report details the development of our species classification model. **Section 2** describes the initial data exploration, and the image preprocessing steps applied to prepare the dataset. **Section 3** outlines the modelling approach chosen, the experimental setup, and provides an analysis of the model's performance using appropriate metrics. **Section 4** discusses an innovative aspect explored and implemented during the project. Finally, **Section 5** concludes the report, summarizing principal conclusions, acknowledging limitations, and proposing potential directions for future work.

**2. DATA EXPLORATION AND IMAGE PREPROCESSING**

**2.1. Initial Data Exploration**

An initial exploration of the provided metadata CSV file, using the pandas library, confirmed the dataset contains 11,983 observations and 7 features, with no missing values or duplicate rows detected. The data types for all features appeared appropriate.

Analysis of the metadata revealed that all images belong to the `kingdom` Animalia, making this feature non-discriminatory for our classification task. The `phylum` feature, however, contains 5 distinct values, with 'Chordata' (vertebrates) representing the vast majority (83.1%), as shown in Figure B1. The target variable, `family`, comprises 202 distinct classes. The frequency distribution across these families is notably imbalanced, ranging from 29 samples for the least frequent family (Siluridae) to 300 for the most frequent (Dactyloidae and Cercopithecidae), illustrated in Figure B2. A hierarchical check confirmed that each family belongs to only one phylum. Furthermore, a visual inspection of sample images (Figure B3) revealed that some images do not depict animals (e.g., showing habitat or specimen labels), representing potential outliers that could affect model training.

**(Figure B1 – Barplot for phylum distribution.)**
*(Insert Figure B1 here)*

**(Figure B2 – Barplot for family distribution (Left: All families, Right: Top 50 families).)**
*(Insert Figure B2 here)*

**(Figure B3 – Sample images from the dataset, illustrating diversity across phyla and potential non-animal images.)**
*(Insert Figure B3 here)*

Based on common practices identified in the literature review [4, 6, 8, 14, 15], the dataset was split into training (80%), validation (10%), and test (10%) sets. This split was performed using stratification to maintain the original proportion of each family across the sets, mitigating potential biases due to the class imbalance. Shuffling was applied during the split. For practical workflow management across multiple notebooks (sharing common functions via a `utilities.py` file), the image files were copied into separate `train`, `validation`, and `test` folders. The hold-out method (using distinct validation and test sets) was chosen over k-fold cross-validation primarily due to the significant computational cost associated with repeatedly training and fine-tuning deep learning models for each fold. The validation set was used exclusively for model development and hyperparameter tuning, while the test set was reserved for the final, unbiased evaluation of the selected model's performance on unseen data.

To address the class imbalance observed in the `family` distribution, we considered oversampling, undersampling, and class weighting. Undersampling was deemed unsuitable as it would discard a large number of images from the majority classes, leading to significant information loss, especially given the already limited size of the dataset for some families. Therefore, we decided to explore oversampling (using data augmentation) and class weighting during the modelling phase.

**2.2. Image Import & Preprocessing**

The Keras library (within TensorFlow) was extensively used for data import, model building, and evaluation. Images were loaded using the `image_dataset_from_directory` function with the following configurations:

*   **Image Size:** Set to `(224, 224)` pixels. This dimension is standard for many widely used pre-trained models (like VGG, ResNet, EfficientNet) available in Keras Applications [24] and is frequently cited in image classification literature [6, 9, 11, 14, 17, 18, 20].
*   **Color Mode:** Implicitly 'rgb', resulting in 3 color channels, capturing the full color information necessary for distinguishing species.
*   **Aspect Ratio Handling:** `crop_to_aspect_ratio` and `pad_to_aspect_ratio` were both set to `False`. Setting `crop_to_aspect_ratio=True` could remove crucial parts of the animal (e.g., head or tail), while `pad_to_aspect_ratio=True` introduces black bars that might negatively bias the model by associating the padding with certain classes or simply adding non-informative areas (Figure B4). By setting both to `False`, the images are simply resized (distorted if necessary) to 224x224, preserving all original pixels, which was deemed preferable to information loss or artifact introduction.
*   **Labels:** `label_mode='categorical'` was used to produce one-hot encoded labels, necessary for using categorical cross-entropy loss and applying class weights. `labels='inferred'` was used as the class subdirectories within `train`, `validation`, and `test` correspond to the family names.
*   **Shuffling:** `shuffle=True` was applied only during the import of the *training* dataset. This is crucial for deep learning as it ensures that each batch contains a diverse mix of classes and samples, preventing the model from learning sequential patterns related to data order and promoting better generalization. Validation and test sets were not shuffled (`shuffle=False`) to ensure consistent evaluation.
*   **Batch Size:** Set to `64`. This value was chosen after experimentation as the largest size manageable within our available hardware memory limits. Larger batch sizes can sometimes lead to more stable gradients and potentially faster convergence, although optimal batch size can be problem-dependent.
*   **Interpolation:** `interpolation='bilinear'` was used for resizing. Bilinear interpolation calculates pixel values based on a weighted average of the four nearest neighbors in the original image, offering a good balance between computational efficiency and image quality compared to simpler methods like 'nearest' or more complex ones like 'bicubic'.

**(Figure B4 – Sample images from the Training set demonstrating resize behavior.)**
*(Top: `crop_to_aspect_ratio=False` & `pad_to_aspect_ratio=False` | Middle: `crop_to_aspect_ratio=True` | Bottom: `pad_to_aspect_ratio=True`)*
*(Insert Figure B4 here)*

Before committing to specific preprocessing steps, various image transformations available in `tf.image` were visually explored (example shown in Figure B5). Based on this, we decided to systematically test the effects of applying `grayscale`, `tf.image.adjust_contrast`, and `tf.image.adjust_saturation` individually during the baseline model evaluation phase.

**(Figure B5 – Examples of potential tf.image preprocessing and augmentation effects.)** *(Caption needs specific context based on what was actually tested)*
*(Insert Figure B5 here)*

**2.3. Addressing Imbalance Issues**

Two primary methods were employed to mitigate the impact of class imbalance:

1.  **Oversampling via Data Augmentation ("SMOTE"):** Instead of traditional SMOTE which operates in the feature space (not directly applicable to images), we implemented an oversampling strategy by applying data augmentation to the minority classes in the training set. A copy of the training data folder was created. For each class with fewer samples than the largest class (n=240), new images were generated using Keras's `RandAugment` layer [AA] applied to existing images until the sample count matched the largest class. `RandAugment` applies a sequence of randomly selected augmentation techniques (e.g., rotation, shear, color jittering - examples in Figure B6) with random magnitudes, introducing significant diversity [AA]. These augmented images were saved to disk. While potentially less efficient than on-the-fly augmentation, this approach ensured reproducibility, as the exact same augmented dataset was used for all model training runs comparing the 'SMOTE' condition. This method mimics the goal of SMOTE by synthetically increasing the representation of minority classes through visually plausible variations.

2.  **Class Weighting:** During model training, the `class_weight` parameter was tested. Weights were calculated for each class inversely proportional to their frequency using the formula: `weight_for_class_i = total_samples / (num_classes * samples_in_class_i)`. This approach increases the contribution of misclassifications from rarer classes to the loss function, forcing the model to pay more attention to them during optimization.

**(Figure B6 – Examples of image augmentations applied by RandAugment.)** *(Caption needs specific context based on the RandAugment parameters used)*
*(Insert Figure B6 here)*

**3. MODELLING & EVALUATION**

The modelling process followed the strategy outlined in Figure 3.1, starting with a baseline CNN, evaluating combinations of preprocessing and imbalance handling, selecting promising combinations, testing these on pre-trained models, and finally fine-tuning the best overall model.

**(Figure 3.1 – Model Process Flowchart.)**
*(Insert Flowchart Image here)*

**3.1. Baseline Model – Combination Selection**

A baseline CNN model was first constructed using Keras functional API (Figure C1). The architecture consisted of:
*   A `Rescaling` layer to normalize pixel values initially (e.g., to [0, 1] range before specific pre-trained preprocessing).
*   Several blocks of `Conv2D` (with increasing filters: 32, 64, 128, 256), `BatchNormalization`, and `Activation` ('relu') layers, followed by `MaxPooling2D`. Convolutional layers extract spatial hierarchies of features, Batch Normalization stabilizes training and improves generalization, and Max Pooling reduces dimensionality.
*   `GlobalAveragePooling2D` to flatten the feature maps efficiently.
*   A `Dense` layer (128 units with 'relu' activation) followed by `BatchNormalization`.
*   A `Dropout` layer (rate=0.5) for regularization to prevent overfitting.
*   A final `Dense` output layer with 202 units (number of families) and 'softmax' activation for multi-class probability outputs.

**(Figure C1 – Baseline CNN Model Architecture.)**
*(Insert Figure C1 here)*

This baseline model was used to evaluate different combinations of image preprocessing (original RGB, grayscale, contrast adjustment, saturation adjustment) and imbalance handling techniques (original data, "SMOTE" via augmentation, class weights). These initial runs were limited to a maximum of 10 epochs (Table C1) to quickly identify promising combinations, acknowledging that models might not fully converge.

The primary metric for evaluation was the **Macro F1-Score**, supplemented by Accuracy, Precision, Recall (Macro), and AUROC Score. Since the goal is to classify rare species effectively, Macro F1 is crucial as it averages the F1 score across all classes, giving equal importance to each family, regardless of its size. A low Macro F1 indicates poor performance on at least some classes, likely the rarer ones. While Accuracy gives an overall correctness measure, it can be misleading with imbalanced data. Precision and Recall help diagnose error types (false positives/negatives), and AUROC measures the model's ability to distinguish between classes across different thresholds.

Standard Keras callbacks were used during training:
*   `ModelCheckpoint`: To save the best model based on validation F1 score.
*   `CSVLogger`: To log metrics per epoch for later analysis.
*   `LearningRateScheduler`: To potentially adjust the learning rate during training (e.g., decay schedule) [23]. *(Specify if a schedule was used or just placeholder)*
*   `EarlyStopping`: To stop training if the validation F1 score did not improve for a set number of epochs (e.g., patience=3), preventing excessive training and overfitting.

The `Adam` optimizer was used (learning_rate=0.001, weight_decay=0.01) due to its adaptive learning rate capabilities and generally fast convergence properties. The `CategoricalCrossentropy` loss function was employed, suitable for multi-class classification with one-hot encoded labels.

Based on the initial 10-epoch results (Table C1), combinations involving grayscale, class weights, and original images with only saturation adjustment consistently showed lower performance across most metrics. Grayscale likely suffered from the loss of crucial color information differentiating species. Class weighting might require more careful tuning or longer training to be effective. Saturation alone seemed less impactful than contrast. Consequently, these combinations were excluded from further testing.

**(Table C1 - Baseline Model Combinations Selection Results (Max 10 Epochs).)**
*(Insert Table C1 here)*

The five most promising combinations (Original, Original+Contrast, SMOTE, SMOTE+Contrast, SMOTE+Saturation) were then retrained using the baseline architecture, allowing up to 100 epochs with early stopping based on validation F1 score. *(Note: Results from these longer runs are implicitly compared in the next stage with pre-trained models, see Table C2)*. It was expected that these models might show signs of overfitting on the training data, as the primary goal here was still comparative evaluation to inform the selection for pre-trained model testing.

**3.2. Pretrained Models**

To leverage knowledge learned from large-scale datasets like ImageNet, transfer learning was employed using several pre-trained models available in Keras Applications [24]. The expectation was that these models, with their optimized architectures, would achieve better performance and faster convergence compared to the baseline CNN. Models were selected based on the literature review, performance benchmarks, and compatibility with the `(224, 224)` input size:

*   **VGG19** [28]: A deep architecture known for its simplicity and use of small (3x3) convolutional filters.
*   **ResNet152V2** [29]: A very deep residual network designed to overcome vanishing gradient problems, using improved residual blocks (V2).
*   **ConvNeXtBase** [30]: A modern CNN architecture inspired by Vision Transformers, aiming for state-of-the-art performance.
*   **EfficientNetV2B0** [31]: Part of a family of models scaled systematically for optimal efficiency and accuracy; V2 represents improvements over the original.

*(See Annex A for detailed descriptions of each architecture.)*

A critical step when using these models is applying the specific `preprocess_input` function associated with each architecture (e.g., `tensorflow.keras.applications.vgg19.preprocess_input`). These functions handle normalization and potential color space conversions (e.g., RGB to BGR for VGG) according to how the models were originally trained on ImageNet. Failure to use the correct preprocessing layer significantly undermines the benefits of transfer learning. This architecture-specific preprocessing replaced the simple 0-1 rescaling used in the baseline's input layer.

**3.3. Evaluation Models**

The five selected preprocessing/imbalance combinations were applied to each of the four pre-trained models (VGG19, ResNet152V2, ConvNeXtBase, EfficientNetV2B0), freezing the base layers and adding a similar classification head as the baseline (GlobalAveragePooling, Dense, Dropout, Output Dense). These models were trained for up to 100 epochs with the same callbacks, optimizer, and loss function as before.

The results are summarized in Table C2. Analyzing these results, particularly focusing on the validation Macro F1 score and considering overfitting (difference between training and validation F1 < 0.05), the **ConvNeXtBase model with the SMOTE (augmentation-based oversampling) combination and no additional color preprocessing (Contrast/Saturation)** emerged as the best performing approach. It achieved the highest validation F1 score (0.804) and test F1 score (0.829) among all tested combinations, demonstrating strong generalization. Models without the tested grayscale/contrast/saturation generally performed better, reinforcing the importance and effectiveness of the architecture-specific `preprocess_input` layer.

**(Table C2 - Baseline & Pretrained Model Combinations Selection Results (Max 100 Epochs).)**
*(Insert Table C2 here)*

Analysis of the training/validation loss curves for the pre-trained models generally showed the training loss decreasing steadily while the validation loss plateaued relatively early, indicating the onset of overfitting which was effectively managed by the `EarlyStopping` callback.

**3.4. Fine-tuning**

To potentially further improve the performance of the best model (ConvNeXtBase + SMOTE), hyperparameter tuning was performed using `RandomSearch` from Keras Tuner (see Annex B for strategy comparison). `RandomSearch` was chosen over `GridSearch` for efficiency given the large search space.

The search focused on:
*   **Unfreezing Layers:** Experimenting with unfreezing later stages of the ConvNeXtBase model (specifically, the last 22 layers corresponding to Stage 3 [AA - *Confirm this detail from ConvNeXt structure*]) to allow the model to adapt more closely to the specific features of the rare species dataset, while keeping earlier layers frozen to retain general ImageNet features. Batch Normalization layers were kept frozen as recommended during fine-tuning [AA].
*   **Learning Rate:** Testing lower learning rates (`1e-4`, `1e-5` in addition to `1e-3`) for the fine-tuning phase, which often helps achieve better convergence when unfreezing layers.
*   **Optimizer:** Exploring different optimizers ('adam', 'sgd', 'rmsprop').
*   **Dropout Rate:** Varying the dropout rate in the classification head (`0.4` to `0.7`) to find the optimal regularization strength.

*(See Appendix D for the search space definition.)*

**(Table D1 - Hyper-parameters used on RandomSearch.)**
*(Insert Table D1 here)*

After the `RandomSearch`, the best parameters found were: *[Specify Best Parameters Here, e.g., Unfreeze=True, LR=1e-5, Optimizer='adam', Dropout=0.5]*. The model retrained with these parameters achieved a final **Test Accuracy of [X.XXX]** and a **Test Macro F1-Score of [Y.YYY]**. *(Insert your actual final results here)*.

**4. INNOVATIVE APPROACH**

During data exploration (Figure B3), it was noted that some images in the dataset did not contain identifiable animals, instead showing specimen labels, habitat shots, or other non-target subjects. These images act as outliers and could potentially confuse the classifier or dilute the training signal.

To address this, a Zero-Shot Image Classification approach using the CLIP (Contrastive Language–Image Pre-training) model [38] was implemented as an innovative filtering step. CLIP models can classify images based on natural language descriptions without having been explicitly trained on the target classes. We used the `clip-vit-base-patch16` model [39] via the Hugging Face `transformers` library, chosen for its balance between performance and computational feasibility. Candidate labels "photo of an animal" and "photo of something else" were used to classify each image in the dataset. Due to library compatibility issues between Hugging Face Transformers and the TensorFlow/Keras version used for the main project, this classification step was performed in a separate PyTorch environment.

Examples of the CLIP model's classification output are shown in Figure E1. The model demonstrated a reasonable ability to distinguish between images containing animals and those without.

**(Figure E1 – Examples of classification using the CLIP model (“animal” vs. “not animal”).)**
*(Insert Figure E1 here)*

Applying this classification across the entire dataset revealed that approximately 15% of images in each split (train, validation, test) were classified as "not animal" (Figure E2), indicating the issue was consistent across the dataset partitions.

**(Figure E2 – Distribution of CLIP classification results (Animal vs. Not Animal) in train, validation, and test datasets.)**
*(Insert Figure E2 here)*

Ideally, the next step would be to filter out the "not animal" images from the training, validation, and test sets, re-apply the SMOTE augmentation process to the filtered training set (to re-balance based only on animal images), and retrain/re-evaluate the best fine-tuned ConvNeXtBase model on this cleaned data. *[State whether this step was completed and report results, OR clearly state it was identified as a valuable next step but not completed due to time/resource constraints].* This approach aims to create a more focused training process, potentially improving the model's ability to learn discriminative features for the actual animal species.

**5. CONCLUSION**

This project successfully developed and evaluated deep learning models for classifying rare species into their respective families using the BioCLIP image dataset. Following an exploration and preprocessing phase that included handling dataset structure and addressing significant class imbalance via augmentation-based oversampling (SMOTE), various modelling strategies were compared.

A baseline CNN was established, followed by the evaluation of four pre-trained architectures (VGG19, ResNet152V2, ConvNeXtBase, EfficientNetV2B0) combined with different preprocessing and imbalance handling techniques. Performance was primarily assessed using the Macro F1-score to ensure adequate performance across all families, including rare ones. The **ConvNeXtBase model, combined with SMOTE augmentation**, demonstrated the best performance on the validation and test sets. Subsequent fine-tuning of this model, involving unfreezing later layers and optimizing hyperparameters like learning rate and dropout via `RandomSearch`, yielded the final model with a **Test Macro F1-Score of [Y.YYY]** and **Test Accuracy of [X.XXX]**. *(Insert final best model results)*.

An innovative approach using CLIP for zero-shot classification was explored to identify and potentially filter non-animal images from the dataset, addressing an observed data quality issue. This filtering step showed promise in cleaning the dataset for more focused training.

Limitations of this study include the use of a hold-out validation strategy instead of more robust cross-validation, the inherent limitations of the dataset size for very rare families, and the computational constraints limiting the extent of hyperparameter search and the full evaluation of the CLIP filtering impact.

Future work could involve:
*   Implementing k-fold cross-validation for more reliable model evaluation and selection.
*   Exploring more sophisticated data augmentation techniques or alternative over/under-sampling methods.
*   Conducting a more extensive hyperparameter search for the fine-tuning phase.
*   Fully integrating and evaluating the CLIP filtering pre-processing step by retraining the final model on the cleaned dataset.
*   Investigating methods to incorporate the provided metadata (phylum, kingdom) more directly into the model architecture.
*   Potentially expanding the dataset through web scraping or incorporating other available rare species image sources.

In conclusion, this project demonstrates the capability of modern deep learning techniques, particularly transfer learning with architectures like ConvNeXtBase combined with careful handling of class imbalance, to effectively classify rare species from images, contributing valuable tools for biodiversity research and conservation.

---

**BIBLIOGRAPHICAL REFERENCES**

[1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. The MIT Press.

[2] Stevens, S., Wu, J., Thompson, M. J., Campolongo, E. G., Song, C. H., Carlyn, D. E., ... & Su, Y. (2024). Bioclip: A vision foundation model for the tree of life. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 19412-19424).

[3] S, N. J., Kamsala Tharun, & Somu Geetha Sravya. (2024). Deep Learning Approaches to Image-Based Species Identification. *2024 International Conference on Integrated Circuits and Communication Systems (ICICACS)*, 1–7. https://doi.org/10.1109/icicacs60521.2024.10498423

[4] Bhargavi, I., Pratap, A. R., & Sri, A. S. (2024). An Enhanced EfficientNet-Powered Wildlife Species Classification for Biodiversity Monitoring. *2024 4th International Conference on Intelligent Technologies (CONIT)*, 1–6. https://doi.org/10.1109/conit61985.2024.10627148

[5] Mane, V., Pranjali Nikude, Patil, T., & Tambe, P. (2024). Wildlife Classification using Convolutional Neural Networks (CNN). *2024 International Conference on Inventive Computation Technologies (ICICT)*. https://doi.org/10.1109/icict60155.2024.10544702

[6] Habib, S., Ahmad, M., Ul Haq, Y., Sana, R., Muneer, A., Waseem, M., … Dev, S. (2024). Advancing Taxonomic Classification Through Deep Learning: A Robust Artificial Intelligence Framework for Species Identification Using Natural Images. *IEEE Access, 12*, 146718–146732. https://doi.org/10.1109/ACCESS.2024.3450016

[7] Kimly Y, Malis Lany, Soy Vitou, & Kor, S. (2023). Animal Classification using Convolutional Neural Network. *The 2nd Student Conference on Digital Technology 2023*. https://www.researchgate.net/publication/376751387_Animal_Classification_using_Convolutional_Neural_Network

[8] Sharma, S., Sisir Dhakal, & Bhavsar, M. (2024). Transfer Learning for Wildlife Classification: Evaluating YOLOv8 against DenseNet, ResNet, and VGGNet on a Custom Dataset. *Journal of Artificial Intelligence and Capsule Networks, 6*(4), 415–435. https://doi.org/10.36548/jaicn.2024.4.003

[9] Supreet Parida, Mishra, A., Sahoo, B. P., Nayak, S., Mishra, N., & Panda, B. S. (2024). Recognizing Wild Animals from Camera Trap Images Using Deep Learning. *2024 International Conference on Intelligent Computing and Emerging Communication Technologies (ICEC)*, 1–6. https://doi.org/10.1109/icec59683.2024.10837421

[10] aa Gagandeep M D, Jagath S K, Kartik Tomar, Senthil Kumar R. (2024). R-CNN Based Deep Learning Approach for Counting Animals in the Forest: A Survey. *International Journal of Networks and Systems, 13*(1), 1–4. https://doi.org/10.30534/ijns/2024/011312024

[11] Pruthvi Darshan S S, L, J. M., & Sangeetha V. (2024). Multiclass Bird Species Identification using Deep Learning Techniques. *2024 IEEE International Conference on Electronics, Computing and Communication Technologies (CONECCT)*, 1–6. https://doi.org/10.1109/conecct62155.2024.10677184

[12] Gill, K. S., Gupta, R., Malhotra, S., Swati Devliyal, & G Sunil. (2024, April 5). Classification of Reptiles and Amphibians Using Transfer Learning and Deep Convolutional Neural Networks. *2022 IEEE 7th International Conference for Convergence in Technology (I2CT)*. https://doi.org/10.1109/i2ct61223.2024.10544030

[13] P Kanaga Priya, T Vaishnavi, N Selvakumar, G Ramesh Kalyan, & A Reethika. (2023, July 19). An Enhanced Animal Species Classification and Prediction Engine using CNN. *2023 2nd International Conference on Edge Computing and Applications (ICECAA)*. https://doi.org/10.1109/icecaa58104.2023.10212299

[14] Sharma, S., Neupane, S., Gautam, B., & Sato, K. (2023, December). Automated Multi-Species Classification Using Wildlife Datasets Based on Deep Learning Algorithms. *Materials, Methods & Technologies, 17*. https://doi.org/10.62991/MMT1996359772

[15] Oion, M. S. R., Islam, M., Amir, F., Ali, M. E., Habib, M., Hossain, M. S., & Wadud, M. A. H. (2023). Marine Animal Classification Using Deep Learning and Convolutional Neural Networks (CNN). *2023 26th International Conference on Computer and Information Technology (ICCIT)*, 1–6. https://doi.org/10.1109/ICCIT60459.2023.10441585

[16] Binta Islam, S., Valles, D., Hibbitts, T. J., Ryberg, W. A., Walkup, D. K., & Forstner, M. R. J. (2023). Animal Species Recognition with Deep Convolutional Neural Networks from Ecological Camera Trap Images. *Animals, 13*(9), 1526. https://doi.org/10.3390/ani13091526

[17] Cai, R. (2023). Automating bird species classification: A deep learning approach with CNNs. *Journal of Physics: Conference Series, 2664*(1), 012007. 
https://doi.org/10.1088/1742-6596/2664/1/012007

[18] Priya, P. K., Vinu, M. S., PrasannaBlessy, M., Kirupa, P., Gayathri, R., & Selvakumar, N. (2023). An Eagle-Eye Vision: Advancements in Avian Species Classification. *2023 2nd International Conference on Automation, Computing and Renewable Systems (ICACRS)*, 758–764. https://doi.org/10.1109/ICACRS58579.2023.10404897

[19] Larson, J. (2021). *Assessing Convolutional Neural Network Animal Classification Models for Practical Applications in Wildlife Conservation* [MSc Thesis]. Oregon State University. https://doi.org/10.31979/etd.ysr5-th9v

[20] Sanghvi, K., Aralkar, A., Sanghvi, S., & Saha, I. (2020, May). Fauna Image Classification using Convolutional Neural Network. *International Journal of Engineering Research & Technology (IJERT), 13*(05), 8–16.

[21] Rajasekaran, T., Kaliappan, V., Surendran, R., Sellamuthu, K., & Palanisamy, J. (2019, October). Recognition Of Animal Species On Camera Trap Images Using Machine Learning And Deep Learning Models. *International Journal of Scientific & Technology Research, 8*(10).

[22] Albuquerque, C. (2019). *Convolutional neural networks for cell detection and counting : a case study of human cell quantification in zebrafish xenografts using deep learning object detection techniques* [MSc Thesis]. Universidade Nova de Lisboa. https://run.unl.pt/handle/10362/62425

[23] Team, K. (n.d.). Keras documentation: LearningRateScheduler. Retrieved April 23, 2025, from https://keras.io/api/callbacks/learning_rate_scheduler/

[24] Team, K. (n.d.). Keras documentation: Keras Applications. Retrieved April 23, 2025, from https://keras.io/api/applications/

[25] Chollet, F., et al. (2015). Keras. https://keras.io

[26] Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., ... & Ghemawat, S. (2016). TensorFlow: Large-scale machine learning on heterogeneous distributed systems. *arXiv preprint arXiv:1603.04467*.

[27] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830. *(Implicitly used via Keras Tuner)*

[28] Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv preprint arXiv:1409.1556*.

[29] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *arXiv preprint arXiv:1603.05027*.

[30] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 11976-11986. https://arxiv.org/abs/2201.03545

[31] Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. *Proceedings of the 38th International Conference on Machine Learning, PMLR 139*, 10096-10106. https://arxiv.org/abs/2104.00298

[32] Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. *Journal of Big Data, 6*(1), 1-48. https://doi.org/10.1186/s40537-019-0197-0 *(General concept reference for Data Augmentation)*

[33] Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). RandAugment: Practical automated data augmentation with a reduced search space. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, 702-703. https://arxiv.org/abs/1909.13719 *(Specific reference for RandAugment)*

[34] Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv preprint arXiv:1412.6980*. *(Reference for Adam optimizer)*

[35] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. *(General ML reference)*

[36] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research, 13*, 281–305.

[37] Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. *Journal of Machine Learning Research, 18*(185), 1–52.

[38] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning, PMLR 139*, 8748-8763. https://arxiv.org/abs/2103.00020 *(Reference for CLIP)*

[39] OpenAI. (2021). *openai/clip-vit-base-patch16*. Hugging Face Model Hub. https://huggingface.co/openai/clip-vit-base-patch16

[40] Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38-45. *(Reference for Hugging Face Transformers library)*

---

**(Appendices and Annexes follow as provided/generated previously)**

*   Appendix A: Literature Review (Table A1)
*   Appendix B: EDA (Figures B1-B6)
*   Appendix C: Modelling (Figure C1, Tables C1-C2)
*   Appendix D: Hyper-parameter RandomSearch (Table D1)
*   Appendix E: Innovative Approach (Figures E1-E2)
*   Annex A: Pretrained Models *(Add detailed descriptions here as requested)*
*   Annex B: GridSearch VS RandomSearch VS Hyperband Keras Tuner *(Use the previously generated Annex)*

***

**Note:** Remember to replace placeholders like `[X.XXX]`, `[Y.YYY]`, `[Specify Best Parameters Here...]`, `[AA - Confirm...]`, and complete the detailed descriptions in Annex A. Also, ensure the captions for Figures B5, B6, C1, E1, E2 are made specific and informative based on your actual work. You might also need to adjust the overfitting threshold mentioned (e.g., `< 0.05` for F1 difference) based on your specific observations. Finally, double-check the flow and ensure consistency throughout the document.
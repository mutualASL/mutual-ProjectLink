# Mutual - ProjectLink
ASL Live-Fingerspelling Classification and LLM Processing

<img width="361" height="197" alt="Screenshot 2026-07-06 at 12 46 48 AM" src="https://github.com/user-attachments/assets/ae8f1c9c-3333-40b6-8aa9-29d2b3ce8a0c" /> <img width="684" height="310" alt="Screenshot 2026-07-06 at 1 02 38 AM" src="https://github.com/user-attachments/assets/1573b4e3-7e04-4fd3-94d5-bee4915dbab0" />

## Prior Acknowledgement: 
Our interpretation pipeline attempts to interpret real-time ASL fingerspelling for simple exchanges. We acknowledge that ASL is a dynamic and interpretive language that necessitates, ultimately, human-to-human communication, and that our design fails to account for facial involvement and the "feeling" aspect of the language. We also recognize that fingerspelling is only a portion of the language that assists in specifying proper nouns, technical terms, and words that have not yet been standardized. 

Our primary objective for this project, alongside our wearable development, is in hopes of encouraging the hearing community to take a more proactive stance in branching out to the Deaf community by educating themselves on ASL. While we certainly hope our wearable can help Deaf individuals in very simple exchanges with hearing individuals, we value more that the interpreter inspires hearing individuals to learn fingerspelling and the shortcomings of computer vision "gadgets" in application to a deeply complex, expressive language. We also hope to establish a relationship with Brookline Deaf organizations to further our communal engagements. 

## Overview: 
**Mutual**, a student non-profit centered in Brookline, Massachusetts, is dedicated to bridging communication between the Deaf community and non-signers. This repository features our ASl fingerspelling hand pose recognition pipeline: a hand-pose landmark classification model paired with a large-language-model processing layer that turns raw letter predictions into fluent, readable text. 

Our most recent updates and packages, including **piroitranslation.py**, **asl_settings.py**, **v2piroiasl_landmark_model.keras** are compatible on the Raspberry Pi OS ecosystem as we develop a wearable designed for live ASl fingerspelling interpretation. This pipeline combined Mediapipe for hand detection and 21-point landmark extraction, Tensorflow/Keras for a trained classification model, and an LLM post-processing stage that forms grammatical sentences from strings of detected hand signs. Our program runs on Raspberry Pi, translating fingerspelling input into spoken and on-screen English in real time. 


## Methodology
### 1. Data Collection and Preprocessing:
a. We built our own image collector that captures fingerspelling hand poses as rendered skeleton images. Each capture features Mediapipe's 21 hand landmarks and fits them into a dynamic bounding box that motion tracks the hand. The samples are collected with auto rotation adjustments so the hand is otherwise upright or horizontal. Landmark connections are drawn on a 200x200 black-and-white canvas. The design of the pipeline attempts to mitigate lighting and background information by concentrating only on the Mediapipe hand landmark skeleton. These ASL hand pose images are collected into separate folders, each corresponding to an English alphabet letter. 

<table align="center">
  <tr>
    <td align="center">
      <img height="300" alt="Data collection running on Mac webcam" src="https://github.com/user-attachments/assets/f57d14d0-2341-4498-b762-b2332ab52847" />
    </td>
    <td align="center">
      <img height="300" alt="Data folders" src="https://github.com/user-attachments/assets/5e69e6cc-6705-4a6e-b411-49bdee4ead70" />
    </td>
  </tr>
  <tr>
    <td align="center"><sub>Data collection running on Mac webcam with live hand motion tracking and MediaPipe processing</sub></td>
    <td align="center"><sub>Data folders</sub></td>
  </tr>
</table>

b. With the integrated auto rotation adjustment logic that "standardizes" hand positions, camera-hand tracking, dynamic bounding box defined by hand landmarks, and skeleton rendering, the model can focus on learning handshape and tackling angle variations. These processing functions are also applied during live translation, allowing our compact VGG-style CNN to better handle unpredictable hand angles and positions. 

### 2. Model Training: 
a. Our model training has undergone many iterations. Currently, the classifier is a convolutional neural network that takes the 200x200 black-and-white skeleton canvas and outputs a probability over the 26 letters of the English alphabet. The body is three convolutional blocks — each one Conv2D → BatchNorm → MaxPool → Dropout, doubling the filters as it goes (16 → 32 → 64) — followed by a flatten, a dense layer, and the 26-way softmax output. BatchNorm and dropout throughout keep it from overfitting the training captures.

About 2.59M trainable parameters. The saved .keras weighs ~30 MB because it stores optimizer state; the weights themselves are ~10 MB, and we strip the rest for Pi deployment.

```text
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 200, 200, 16)   │           160 │
│ batch_normalization (BatchNorm) │ (None, 200, 200, 16)   │            64 │
│ max_pooling2d (MaxPooling2D)    │ (None, 100, 100, 16)   │             0 │
│ dropout (Dropout)               │ (None, 100, 100, 16)   │             0 │
│ conv2d_1 (Conv2D)               │ (None, 100, 100, 32)   │         4,640 │
│ batch_normalization_1           │ (None, 100, 100, 32)   │           128 │
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 50, 50, 32)     │             0 │
│ dropout_1 (Dropout)             │ (None, 50, 50, 32)     │             0 │
│ conv2d_2 (Conv2D)               │ (None, 50, 50, 64)     │        18,496 │
│ batch_normalization_2           │ (None, 50, 50, 64)     │           256 │
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 25, 25, 64)     │             0 │
│ dropout_2 (Dropout)             │ (None, 25, 25, 64)     │             0 │
│ flatten (Flatten)               │ (None, 40000)          │             0 │
│ dense (Dense)                   │ (None, 64)             │     2,560,064 │
│ batch_normalization_3           │ (None, 64)             │           256 │
│ dropout_3 (Dropout)             │ (None, 64)             │             0 │
│ dense_1 (Dense)                 │ (None, 26)             │         1,690 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2,585,754 (9.86 MB)
 Trainable params: 2,585,402 (9.86 MB)
 Non-trainable params: 352 (1.38 KB)
```
**Latest Model Synopsis**: July 3, 2026
**Version 3:** v3piroiasl_landmark_model.keras

1. Summary: The v3 landmark classifier was retrained on the expanded skeleton-render dataset and reached 99.13% validation accuracy, converging in just 6 epochs before the convergence callback stopped training. Every one of the 26 letters scored ≥ 97% recall, with 10 letters at or effectively at 100%. The model has 2.59M trainable parameters (9.86 MB of weights) and is deployment-ready for the Raspberry Pi 5 wearable with no pipeline changes.

This run incorporates the freshly recollected data for C, G, H, I, J, M, N, O, T, U, V, W, bringing the dataset to nearly 96K images — roughly an order of magnitude larger than earlier iterations.

2. Dataset: Class sizes are uneven — recollected letters such as M (1,495), C (1,257), and O (1,170) carry roughly 2–3× the samples of the smallest classes. This was compensated at training time with inverse-frequency class weights, and the balanced macro-average F1 of 0.99 confirms no class was neglected.

4. Architecture: compact VGG-style CNN — three Conv2D blocks (16 → 32 → 64 filters, each Conv → BatchNorm → MaxPool → Dropout), then Flatten → Dense(64) → BatchNorm → Dropout(0.5) → Dense(26, softmax).

5. The learning rate never needed reduction (2e-4 throughout). Validation accuracy exceeded training accuracy in every epoch — expected behavior, since training accuracy is measured with dropout active while validation runs with it disabled. Validation loss decreased monotonically across all six epochs, indicating no overfitting at the stopping point. Convergence in six epochs on a 26-class problem reflects the strength of the normalized skeleton representation: with position, scale, rotation, lighting, and background variance removed before the CNN ever sees the data, the remaining learning problem is nearly pure handshape discrimination.

```text
Epoch 1/100
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 0s 769ms/step - accuracy: 0.8114 - loss: 0.5301
Epoch 1: val_accuracy improved from -inf to 0.96963, saving model to best_landmark_model.keras
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 963s 802ms/step - accuracy: 0.8114 - loss: 0.5300 - val_accuracy: 0.9696 - val_loss: 0.2011 - learning_rate: 2.0000e-04
Epoch 2/100
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.9536 - loss: 0.1415
Epoch 2: val_accuracy improved from 0.96963 to 0.98090, saving model to best_landmark_model.keras
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 1822s 2s/step - accuracy: 0.9536 - loss: 0.1415 - val_accuracy: 0.9809 - val_loss: 0.1020 - learning_rate: 2.0000e-04
Epoch 3/100
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.9671 - loss: 0.0922
Epoch 3: val_accuracy improved from 0.98090 to 0.98758, saving model to best_landmark_model.keras
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 1930s 2s/step - accuracy: 0.9671 - loss: 0.0922 - val_accuracy: 0.9876 - val_loss: 0.0564 - learning_rate: 2.0000e-04
Epoch 4/100
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 0s 807ms/step - accuracy: 0.9753 - loss: 0.0666
Epoch 4: val_accuracy improved from 0.98758 to 0.99003, saving model to best_landmark_model.keras

Convergence Condition Met: Epoch 1/3
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 1003s 837ms/step - accuracy: 0.9753 - loss: 0.0666 - val_accuracy: 0.9900 - val_loss: 0.0448 - learning_rate: 2.0000e-04
Epoch 5/100
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.9804 - loss: 0.0523
Epoch 5: val_accuracy improved from 0.99003 to 0.99128, saving model to best_landmark_model.keras

Convergence Condition Met: Epoch 2/3
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 4525s 4s/step - accuracy: 0.9804 - loss: 0.0523 - val_accuracy: 0.9913 - val_loss: 0.0438 - learning_rate: 2.0000e-04
Epoch 6/100
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - accuracy: 0.9818 - loss: 0.0464
Epoch 6: val_accuracy did not improve from 0.99128

Convergence Condition Met: Epoch 3/3

Training stopped: Accuracy >= 97.0% for 3 epochs.
1198/1198 ━━━━━━━━━━━━━━━━━━━━ 4697s 4s/step - accuracy: 0.9818 - loss: 0.0464 - val_accuracy: 0.9905 - val_loss: 0.0406 - learning_rate: 2.0000e-04
Restoring model weights from the end of the best epoch: 6.

--- Evaluating Best Model ---

Final Validation Accuracy: 99.13%
```
6. Trained Results:
``` text
              precision    recall  f1-score   support

           A       0.99      1.00      0.99       510
           B       0.97      0.99      0.98       507
           C       0.99      1.00      0.99      1257
           D       1.00      0.99      0.99       523
           E       1.00      0.99      1.00       509
           F       1.00      1.00      1.00       527
           G       1.00      1.00      1.00       801
           H       0.98      1.00      0.99       806
           I       1.00      0.98      0.99       805
           J       0.97      1.00      0.98       766
           K       1.00      0.98      0.99       551
           L       1.00      0.99      0.99       566
           M       0.99      0.98      0.98      1495
           N       1.00      0.99      1.00       806
           O       0.99      1.00      1.00      1170
           P       1.00      0.99      0.99       669
           Q       0.98      0.99      0.99       736
           R       0.99      1.00      1.00       543
           S       0.98      1.00      0.99       553
           T       1.00      0.99      1.00       742
           U       1.00      0.98      0.99       702
           V       1.00      0.99      1.00       842
           W       0.98      0.99      0.99       762
           X       1.00      0.99      1.00       704
           Y       0.99      1.00      0.99       564
           Z       1.00      0.97      0.98       746

    accuracy                           0.99     19162
   macro avg       0.99      0.99      0.99     19162
weighted avg       0.99      0.99      0.99     19162


--- Per-Letter Accuracy ---
Accuracy for A: 1.00 (508/510)
Accuracy for B: 0.99 (503/507)
Accuracy for C: 1.00 (1252/1257)
Accuracy for D: 0.99 (519/523)
Accuracy for E: 0.99 (506/509)
Accuracy for F: 1.00 (525/527)
Accuracy for G: 1.00 (798/801)
Accuracy for H: 1.00 (803/806)
Accuracy for I: 0.98 (786/805)
Accuracy for J: 1.00 (764/766)
Accuracy for K: 0.98 (541/551)
Accuracy for L: 0.99 (561/566)
Accuracy for M: 0.98 (1461/1495)
Accuracy for N: 0.99 (799/806)
Accuracy for O: 1.00 (1170/1170)
Accuracy for P: 0.99 (664/669)
Accuracy for Q: 0.99 (731/736)
Accuracy for R: 1.00 (542/543)
Accuracy for S: 1.00 (553/553)
Accuracy for T: 0.99 (737/742)
Accuracy for U: 0.98 (691/702)
Accuracy for V: 0.99 (837/842)
Accuracy for W: 0.99 (758/762)
Accuracy for X: 0.99 (699/704)
Accuracy for Y: 1.00 (564/564)
Accuracy for Z: 0.97 (723/746)
```
Residual errors are small and concentrated where ASL handshapes are genuinely similar or motion-based: Z (0.97) and J (1.00 recall, 0.97 precision) are the two motion letters, captured as static keyframes. Their intra-class variance is inherently the highest in the dataset — different frames of the same stroke look different — so a slightly lower score here is structural, not a data-quality problem. Z's 23 misses are the largest single error group in the run.

The closed-fist hand poses (M, N, S, T, E) held up well: S and E is effectively perfect, M at 0.98 with 34 misses out of 1,495 (also the largest class, so its error count looks bigger than its error rate). The recollected M/N/T data appears to have paid off.

I (0.98, 19 misses) and U (0.98, 11 misses) are thin-profile shapes (single and paired extended fingers) that can lose detail in the skeleton render at certain wrist angles. K (0.98) shares the paired-finger geometry with U/V. No letter fell below 97%, and no systematic cross-class collapse appears in the confusion matrix — errors are diffuse rather than clustered around one bad pairing.

7. Parameter Reflection: 
We can expect live translation to have an accuracy rate somewhat lower than the benchmark results, as live interpretation involves environmental disturbances from the surroundings and introduces more unpredictable and less favorable conditions. **Moving forward, we hope to expand our data pool further by having a variety of signers participate in data collection.**

<table align="center">
  <tr>
    <td align="center">
      <img width="2100" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/7aa09fee-a20b-4e08-93d2-64b39b885b5f" />
    </td>
    <td align="center">
      <img width="2100" height="750" alt="per_letter_accuracy" src="https://github.com/user-attachments/assets/ef29a49f-fdc8-4faa-8f37-e22612fe9ac6" />
    </td>
  </tr>
  <tr>
    <td align="center"><sub>Training Curves</sub></td>
    <td align="center"><sub>Per-Letter Accuracy</sub></td>
  </tr>
</table>
<table align="center">
  <tr>
    <td align="center">
      <img width="1200" height="1000" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/2649e25f-9d5f-4f74-bfd3-595e6be7cc3b" />
    </td>
    <td align="center">
      <img width="1200" height="1000" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b41a17df-5d66-456c-ac87-dafd70b5b947" />
    </td>
  </tr>
  <tr>
    <td align="center"><sub>Confusion Matrix Normalized</sub></td>
    <td align="center"><sub>Confusion Matrix</sub></td>
  </tr>
</table>




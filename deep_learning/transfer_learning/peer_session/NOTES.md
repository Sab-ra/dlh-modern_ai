# peer-session notes — borrow a brain

Host copy is `borrow_a_brain_HOST.ipynb` (traditional filled, Runtime → Run all tonight).
Student copy is `borrow_a_brain.ipynb` (setup filled, learning cells are comments only).
**Do not send the HOST notebook.**

Session length: 30–40 min. No lecture. No slides. They type one column, run, look.

## assumed variable names

Filled in setup (beat 0):

- `x_train`, `y_train`, `x_test`, `y_test` — raw CIFAR-10
- `CAT = 3`, `DOG = 5`
- `IMG_SIZE = (96, 96)`
- `N_TRAIN = 300`, `N_TEST = 100`, `BATCH = 32`
- `class_names = ['cat', 'dog']`

Beat 1 must define:

- `x_train_tiny`, `y_train_tiny` — 300 images, labels remapped to 0/1
- `x_test_tiny`, `y_test_tiny` — 100 images, labels remapped to 0/1

Beat 2 must define:

- `x_train_prep`, `x_test_prep` — resized 96×96, `mobilenet_v2.preprocess_input` (not `/255`)

Beat 3 must define:

- `base` — `MobileNetV2(weights='imagenet', include_top=False, input_shape=(96,96,3))`, frozen
- `feature_extractor` — `Input` + `base(..., training=False)` + `GlobalAveragePooling2D`
- `feats` — output of 8 images, shape `(8, 1280)`

Beat 4:

- freeze `base.layers` (the backbone, not a full classifier)

Beat 5 must define:

- `model` — Functional: `Input` + `base(inputs, training=False)` + GAP + `Dense(128, relu)` + `Dense(2, softmax)`
- `history` — `fit` 1 epoch, Adam(1e-3), `sparse_categorical_crossentropy`, validation on the 100

Beat 6 must define:

- `aug` — Sequential of RandomFlip / RandomRotation / RandomZoom / RandomContrast, `seed=42`
- `cat_idx`, `cat` — one cat, 4 views (no fit)

Beat 7:

- unfreeze last 20 of **`base`** (MobileNetV2 object, not `model`)
- keep `keras.layers.BatchNormalization` frozen
- recompile `model` with Adam(1e-5)
- `history_ft` — 1 more epoch

Beat 8 must define:

- `probs`, `preds` — 8 test images → `class_names`

`training=False` is load-bearing: BatchNorm stays in inference even while the last 20 conv weights get a whisper of gradient.

## estimated minutes per beat

| beat | what | min |
|------|------|-----|
| 0 | setup (run filled cell, GPU check, cifar download) | 3 |
| 1 | filter two classes + 8 thumbnails | 5 |
| 2 | resize + preprocess_input, print min/max | 4 |
| 3 | borrow MobileNet, print `(8, 1280)` | 4 |
| 4 | freeze butler, sandwich bread | 3 |
| 5 | head + 1 epoch (weights download lives here if not cached) | 6 |
| 6 | aug, 4 views of one cat, no fit | 4 |
| 7 | unfreeze last 20, BN frozen, 1e-5, 1 epoch | 6 |
| 8 | predict 8 | 3 |
| **total** | | **~38** |

If Colab is cold, beat 3/5 eats the MobileNet download. That's fine — don't add epochs.

## Colab tips

- **Runtime → Change runtime type → GPU.** Then reconnect. First cell prints `tf.config.list_physical_devices('GPU')`. If it's `[]`, they skipped GPU — 1+1 epochs still finish, just slower.
- **File → Save a copy in Drive** *before* anyone types. One shared notebook = one stolen GPU and eight overlapping runtimes.
- `from tensorflow import keras`. Do not `pip install keras`. tensorflow is already on Colab.
- Magic: `%matplotlib inline` is in setup.
- Never run full CIFAR-10, never 224, never 5+ epochs, never Caltech-101.
- If someone `/255`s: the min/max print will look like `0..1` instead of `~-1..1`. That's the wrong glasses. Point at beat 2.
- If they unfreeze `model.layers[-20:]` instead of `base.layers[-20:]` they are coaching the Dense head (already trainable) and maybe GAP. Stop them. Sandwich says backbone.
- Recompile after changing `trainable`. Keras will not pick up the new flags otherwise.
- Beat 6 does **not** wrap `aug` into `model` and does **not** fit.

## Google Form door

One question. Then the Colab link. That's the door — attendance without a Slack dump of the URL.

1. Google Forms → blank form. Title something like `borrow a brain — sign in`.
2. One question: **Name** (Short answer, Required). Nothing else.
3. Settings → Presentation → **confirmation message**: paste the student Colab link (`borrow_a_brain.ipynb` copy-in-Drive URL) plus one line: *File → Save a copy in Drive. Runtime → GPU. See you in the room.*
   Alternate: add a second section that only appears after they submit the name, containing the link. Confirmation message is simpler.
4. Send → copy the Form link. That's what you post. Not the notebook. Not the HOST file.
5. Responses tab = who showed up.

After the session you can export names. You do not need a second question.

## dry-run checklist (tonight, HOST notebook)

- [ ] GPU runtime
- [ ] Run all
- [ ] Beat 2 min/max is ~-1..1
- [ ] Beat 3 prints `(8, 1280)`
- [ ] Beat 4 prints `0` trainable weights on `base`
- [ ] Beat 5 accuracy appears (any number is fine)
- [ ] Beat 6 shows 4 different-ish cats
- [ ] Beat 7 trainable weights on `base` is no longer 0; 1e-5 epoch runs
- [ ] Beat 8 titles show predicted (true)
- [ ] Confirm student notebook learning cells are comments only (no leaked solutions)

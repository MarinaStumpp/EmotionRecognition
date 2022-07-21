import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# load csv file
df = pd.read_csv('test_data.csv')
df = shuffle(df)

# split label
X = df.drop('label', axis=1)
y = df['label']

# load model
model = tf.keras.models.load_model('combined-model.h5')

# show model performance on test data
model.evaluate(X, y)

# predict on test data
y_pred = model.predict(X).argmax(axis=1)

# create confusion matrix
cm = confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7], normalize='true')
display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad",
                                                 "surprised"])

# show confusion matrix
display.plot()
plt.savefig('combined-confusion.jpg')
plt.close()

from keras.models import load_model
from translation_model import testX, eng_tokenizer,  trainX, test
import pandas as pd

#print(testX.reshape((testX.shape[0],testX.shape[1])))
#print(testX.shape)

testX_2 = testX[:100]
model = load_model('model2.h5')
#print(model.summary())
preds = model.predict_classes(testX_2, batch_size=10)

def get_word(n, tokenizer):
      for word, index in tokenizer.word_index.items():
          if index == n:
              return word
      return None

preds_text = []
for i in preds:
       temp = []
       for j in range(len(i)):
            t = get_word(i[j], eng_tokenizer)
            if j > 0:
                if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                     temp.append('')
                else:
                     temp.append(t)
            else:
                   if(t == None):
                          temp.append('')
                   else:
                          temp.append(t) 

       preds_text.append(' '.join(temp))

pred_df = pd.DataFrame({'actual' : test[:100,0], 'predicted' : preds_text})

print(pred_df)
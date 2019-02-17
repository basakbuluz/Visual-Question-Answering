# :camera: :question: Visual-Question-Answering
Visual Question Answering Demo and Algorithmia API


### Requirements:

* [Tensorflow (Ver. 1.2+)](https://www.tensorflow.org/install/pip)
* [Keras (Ver. 2.0+)](https://pypi.org/project/Keras/)
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [Spacy (Ver 2.0+)](https://spacy.io/usage/)
    * Used to load Glove vectors (word2vec)
    * To upgrade & install Glove Vectors
       * python -m spacy download en_vectors_web_lg
       
* [OpenCV](https://pypi.org/project/opencv-python/)

### Blog 📝
### Demo 🖥️

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/basakbuluz/Visual-Question-Answering/blob/master/VisualQuestionAnsweringDemo.ipynb)

This jupyter notebook is a simple Visual Question answering demo that uses pretrained models to answer a given question about the given image.

### API :computer:

[Click here](https://algorithmia.com/algorithms/yavuzkomecoglu/VQA) to use the API that you can integrate quickly into the products you have developed.

##### API Python Implementation
###### Install
Install the Algorithmia Python client with pip:

```
pip install algorithmia
```

###### Use
```
import Algorithmia

input = {
  "image": "data://yavuzkomecoglu/DL_VQA/test.jpg",
  "question": "What vehicle is in the picture?"
}
client = Algorithmia.client('YOUR_API_KEY')
algo = client.algo('yavuzkomecoglu/VQA/0.1.1')
print(algo.pipe(input).result)
```

### Sample predictions
Some answers predicted by the VQA model.

![](images/test/test2.jpeg)

Q: How is the weather? 
A: Sunny! (%97.23)


![](images/test/test5.jpg)

Q: What is done in the picture?
A: Surfing! (%99.43)

### References

* [Aaditya Prakash (Adi) - Blog](https://iamaaditya.github.io/2016/04/visual_question_answering_demo_notebook)

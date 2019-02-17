# :camera: :question: Görsel Soru Cevaplama / Visual-Question-Answering

**Görsel Soru Cevaplama** sahip olduğumuz bir resim ile ilgili sorulan sorulara, resim içerisindeki bilgilerin analiz edilmesi ile cevaplar üretilmeye çalışılması problemi olarak tanımlanabilir.

Bu problemde metinler şeklinde ifade edilen soruların işlenmesi bir **Doğal Dil İşleme** problemi iken; resimler içerisinden cevapların üretiminde her bir soru ayrı bir **Bilgisayarla Görü** problemine işaret eder.

**Visual Question Answering** can be defined as the problem of trying to produce answers by analyzing the information in the picture.

In this problem, the questions expressed in the form of texts are a ** Natural Language Processing ** problem; each question in the production of answers within the pictures indicates a separate **Computer Vision** problem.


**Genel olarak sisteme bakacak olursak: / If we look at the system in general:**

![alt text](https://github.com/basakbuluz/Visual-Question-Answering/blob/master/images/VQA1.png "Logo Title Text 1")

**Görsel soru cevaplama problemi için geliştirilen modellerin genel yaklaşımı : / The general approach of the models developed for the visual questioning problem is:**

![alt text](https://github.com/basakbuluz/Visual-Question-Answering/blob/master/images/VQAmodels.png "Logo Title Text 1")

---
### :pushpin: Görsel Soru Cevaplama görevi için geliştirilen ve literatürdeki çalışmalarda sıklıkla kullanılan veri kümeleri / Data sets developed for the Visual Question Answering task and frequently used in studies in the literature

* [DAQUAR](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/visual-turing-challenge/)
* [COCO-QA](https://github.com/renmengye/imageqa-public/tree/master/data)
* [VQA](https://visualqa.org/index.html)
* [FM-IQA](http://research.baidu.com/Downloads)
* [VISUAL GENOME](https://visualgenome.org/)
* [VISUAL7W](http://web.stanford.edu/~yukez/visual7w/)
---

## Blog 📝

Görsel soru cevaplama ile ilgili anlatım ve bu görev için sıklıkla kullanılan veri kümeleri hakkında detaylı bilgi edinmek için ["Çok Gören Mi Bilir, Çok Soran Mı?"](https://medium.com/deep-learning-turkiye/%C3%A7ok-g%C3%B6ren-mi-bilir-%C3%A7ok-soran-m%C4%B1-4bed5efdba41) başlıklı blog yazıma göz atabilirsiniz.

You can browse my blog titled ["Çok Gören Mi Bilir, Çok Soran Mı?"](https://medium.com/deep-learning-turkiye/%C3%A7ok-g%C3%B6ren-mi-bilir-%C3%A7ok-soran-m%C4%B1-4bed5efdba41) to get detailed information about the Visual Question Answering and the datasets frequently used for this task.

---

## Uygulama (Implementation) :hammer:

### Gereksinimler (Requirements): 

* [Tensorflow (Ver. 1.2+)](https://www.tensorflow.org/install/pip)
* [Keras (Ver. 2.0+)](https://pypi.org/project/Keras/)
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [Spacy (Ver 2.0+)](https://spacy.io/usage/)
    * Glove vektörlerini yüklemek için kullanılır (word2vec) / Used to load Glove vectors (word2vec)
    * Glove vektörlerini yükseltmek ve yüklemek için /  To upgrade & install Glove Vectors
       * python -m spacy download en_vectors_web_lg
       
* [OpenCV](https://pypi.org/project/opencv-python/)

### Demo 🖥️

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/basakbuluz/Visual-Question-Answering/blob/master/VisualQuestionAnsweringDemo.ipynb)

[Bu jupyter notebook çalışma dosyası](https://nbviewer.jupyter.org/github/basakbuluz/Visual-Question-Answering/blob/master/VisualQuestionAnsweringDemo.ipynb), verilen görüntü hakkında sorulan soruyu cevaplamak için önceden hazırlanmış modelleri kullanan basit bir Görsel Soru Cevaplama demosudur.

[This jupyter notebook](https://nbviewer.jupyter.org/github/basakbuluz/Visual-Question-Answering/blob/master/VisualQuestionAnsweringDemo.ipynb) is a simple Visual Question answering demo that uses pretrained models to answer a given question about the given image.

### API :computer:

Geliştirdiğiniz ürünlere hızlı bir şekilde entegre edebileceğiniz API'yi kullanmak için [buraya](https://algorithmia.com/algorithms/yavuzkomecoglu/VQA) tıklayın.

[Click here](https://algorithmia.com/algorithms/yavuzkomecoglu/VQA) to use the API that you can integrate quickly into the products you have developed.

##### API Python Uygulama / API Python Implementation
###### Kurulum / Install
Algorithmia Python istemcisini pip ile yükleyin / Install the Algorithmia Python client with pip:

```
pip install algorithmia
```

###### Kullanım / Use
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
---

### Örnek Tahminler / Sample predictions 
VQA modeli tarafından tahmin edilen bazı cevaplar.

Some answers predicted by the VQA model.

![](images/test/test2.jpeg)

Q: How is the weather? 
A: Sunny! (%97.23)

Q: How many girls are in the picture?
A: 2! (%61.98)

![](images/test/test5.jpg)

Q: What is done in the picture?
A: Surfing! (%99.43)

![](images/test/test6.jpg)

Q: What does the sign say?
A: Stop! (%28.61)


---
### Referanslar / References

* [Aaditya Prakash (Adi) - Blog](https://iamaaditya.github.io/2016/04/visual_question_answering_demo_notebook)

## StyleTransfer при помощи GAN'ов 

### CycleGAN
Процесс обучения и все архитектуры показаны в тетрадке [CycleGAN_training.ipynb](CycleGAN_training.ipynb)

Примеры случайных генераций на тестовом сете по эпохам:

(label - generated - true)

1 эпоха

![img](CycleGAN_images/cycleGAN_sample_1_0.png)
![img](CycleGAN_images/cycleGAN_sample_1_3.png)

50 эпоха

![img](CycleGAN_images/cycleGAN_sample_50_1.png)
![img](CycleGAN_images/cycleGAN_sample_50_4.png)

100 эпоха

![img](CycleGAN_images/cycleGAN_sample_100_2.png)
![img](CycleGAN_images/cycleGAN_sample_100_3.png)

150 эпоха

![img](CycleGAN_images/cycleGAN_sample_150_0.png)
![img](CycleGAN_images/cycleGAN_sample_150_3.png)

200 эпоха

![img](CycleGAN_images/cycleGAN_sample_200_4.png)
![img](CycleGAN_images/cycleGAN_sample_200_0.png)

Возможно, не стоило делать рандомные кропы в качестве аугментаций, тогда бы качество могло получиться чуть лучше. Но, к сожалению обучение отнимало очень много времени, и я не успел протестить без этих эффектов.

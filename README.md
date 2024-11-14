This is a captcha generator created for NUS CS4243 mini project. It uses Wasserstein Generative Adversarial Network (WGAN) and generates individual characters.
The dataset that we used to train is not included in this directory. Below is a sample output after training the model for 1000 epochs with **main.py**.
![14311731554525_ pic](https://github.com/user-attachments/assets/aa8a2f04-cb2b-4d66-83a9-1a0ed14948f7)

After generating the individual characters, we used a OCR tool to filter out ambiguous characters with **check_image_validity.py**. If the output is still not 
satisfying, we can manually label each images generated with **manual_label.py**. Lastly, we ensemble the characters into captchas by adding colours, distracted 
blacklines and changing their proportions in **process_captcha.py**.
![13741731519987_ pic](https://github.com/user-attachments/assets/f0ab02ca-dbb9-4ff1-91e3-47e1c7c9b8ea)

We have also included our trained model in **model** folder.

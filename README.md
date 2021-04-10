# Text Complexity Metric PT-PT (PNL Grade Guesser)
Hello, this repository is what I believe I can upload having in mind copyrights of project that used [Plano Nacional de Leitura](https://www.google.com/search?q=programa%20nacional%20de%20leitura%20portugal&oq=programa%20nacional%20de%20leitura%20portugal&aqs=chrome..69i57j33i22i29i30.4456j0j7&sourceid=chrome&ie=UTF-8) (PNL) to make a text complexity metric for Portuguese. PNL is the Portuguese Ministry of Education list of books to be read by students of different school grades. 

**NB**: Since 2020 (I believe) the recommendations are no longer always given by grade, at times they are by "skill level", but I used past years grade recommendations.

## What was done?
1 - I bought a series of books for which I had an associated schooling grade of the PNL.
2 - I parsed them with several OCR tools into .txt UTF-8 (no BOM) files.
3 - I read the books into Python and obtained so text complexity features.
4 - I trained a Machine Learning model to get the year of grade.

## Files:

**1 - Notes on the original data.ipynb** || Is a file in which I give some tips on how I imported the data. Naturally I cannot upload the books.
**2 - Making the Model.ipynb** || Based on a DataFrame that was saved in .xlsx I show how I trained the model. Here you have basis for working with GridSearch, XGBoost, etc... The trained model is a regressor, meaning it provides a number back.
**xg_reg_ai.dat** || Are the pickle files of the trained model.
**PT_Get_Text.py** and **Prever_Ano.py**  || Are the files imported to extract the used features from texts and to predict the year.
**3 - Example of Use to Predict.ipynb** || A file in which I provide use cases of the above.
**Presentation_IH.pptx** || I first presented this project as a final project of Ironhack DataAnalatyics bootcamp. This file is the presentation I used when delivering the 8 mins presentation.
#
#### Resources:
Here under are the most relevant tools used or adapted.
*As is:*
 - [NLPy_Port](https://github.com/NLP-CISUC/NLPyPort%5D%28https://github.com/NLP-CISUC/NLPyPort) [NLTK based] Mostly used to get POS parts.
 - [Spacy](https://spacy.io/models/pt). Mostly used for stemming and lemmatizing.

*Adapted:*
 - [PE2LGP](https://github.com/mattgoncalves/PE2LGP%5D%28https://github.com/mattgoncalves/PE2LGP). Just adapted one function. This project is awesome.
 - [Separasilabas](https://github.com/amunozf/separasilabas). Did some minor changes to make it more accurate for Portuguese (given it was done for Spanish)

I read a lot of papers, but the most relevant read was a Master thesis from [IST](https://tecnico.ulisboa.pt/en/):
[Classificador de textos para o ensino de português como segunda língua](https://fenix.tecnico.ulisboa.pt/downloadFile/844820067123926/Tese-64834-PedroCurto.pdf), by:
Pedro dos Santos Lopes Curto,
November 2014.


import PT_Get_Text
import pandas as pd
import pickle

def Predict_Year(text):
    '''
    Recebe um texto (string) e devolve:
    [0] -> Previsão do ano de escolaridade. [1-12]
    [1] -> Previsão do difículdade baseada no ano de escolaridade [1-8]
    '''

    list_cols = ['Words per sentence','Syllables per Word','Poly_Syl per Word','Verbs per Word','Noun per Word',
                 'Adj per Word','Simp_Word_DC per Word','Simp_Word_1 per Word','Uniques_Per_Word','HH_Index',
                 'count_SVO_found per word','count_SVO_found_in_a_row per word','count_SVO_elements_in_a_row per word']

    stats = PT_Get_Text.Get_All_Text_Info(text)
    map_columns = {n:i for n, i in zip(range(len(list_cols)), list_cols)}
    DF = pd.DataFrame(stats).T.rename(map_columns, axis = 1)
    
    xg_reg_ai = pickle.load(open("xg_reg_ai.dat", "rb"))
    xg_reg_ne = pickle.load(open("xg_reg_ne.dat", "rb"))
    ano_pred = xg_reg_ai.predict(DF)
    ne_pred = xg_reg_ne.predict(DF)
    
    return (ano_pred[0], ne_pred[0])